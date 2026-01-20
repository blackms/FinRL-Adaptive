#!/usr/bin/env python3
"""
Regime Threshold Validation Script

Validates different regime detection threshold configurations across various
market conditions to find optimal settings.

Configurations Tested:
- Baseline (current): volatility_high_percentile=90, strong_trend_threshold=0.6
- Conservative: Higher vol threshold (95), lower trend threshold (0.4)
- Aggressive: Lower vol threshold (80), higher trend threshold (0.7)
- Optimized: Uses optimize script results if available

Metrics Compared:
- Regime accuracy (predicted vs actual market behavior)
- Number of regime transitions (fewer = less whipsaw)
- Strategy return when following regime signals
- Sharpe ratio by configuration

Time Periods Analyzed:
- 2020 COVID crash and recovery
- 2021 bull market
- 2022 bear market
- 2023-2024 recovery

Usage:
    python scripts/validate_regime_thresholds.py

Output:
    - output/threshold_validation.json
    - output/regime_threshold_comparison.png
"""

import sys
sys.path.insert(0, '/opt/FinRL/src')

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Use yfinance for data
try:
    import yfinance as yf
except ImportError:
    print("Installing yfinance...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

from trading.strategies.regime_detector import RegimeDetector, RegimeDetectorConfig, RegimeType


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("/opt/FinRL/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Test symbols - broad market ETFs
SYMBOLS = ["SPY", "QQQ"]

# Time periods for analysis
TIME_PERIODS = {
    "2020_covid": ("2020-01-01", "2020-12-31", "COVID Crash & Recovery"),
    "2021_bull": ("2021-01-01", "2021-12-31", "2021 Bull Market"),
    "2022_bear": ("2022-01-01", "2022-12-31", "2022 Bear Market"),
    "2023_2024_recovery": ("2023-01-01", "2024-12-31", "2023-2024 Recovery"),
}

# Threshold configurations to test
CONFIGURATIONS = {
    "baseline": {
        "name": "Baseline (Current)",
        "volatility_high_percentile": 90.0,
        "strong_trend_threshold": 0.6,
        "min_hold_days": 5,
    },
    "conservative": {
        "name": "Conservative",
        "volatility_high_percentile": 95.0,  # Higher vol threshold
        "strong_trend_threshold": 0.4,        # Lower trend threshold
        "min_hold_days": 7,
    },
    "aggressive": {
        "name": "Aggressive",
        "volatility_high_percentile": 80.0,   # Lower vol threshold
        "strong_trend_threshold": 0.7,        # Higher trend threshold
        "min_hold_days": 3,
    },
    "optimized": {
        "name": "Optimized",
        "volatility_high_percentile": 85.0,
        "strong_trend_threshold": 0.5,
        "min_hold_days": 5,
    },
}


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_data_yfinance(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV data using yfinance."""
    data = {}

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, interval="1d")

            if df.empty:
                print(f"  Warning: No data for {symbol}")
                continue

            # Rename columns to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Reset index to get datetime as column
            df = df.reset_index()
            df = df.rename(columns={'Date': 'datetime', 'date': 'datetime'})

            # Ensure datetime is timezone-naive
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

            data[symbol] = df
            print(f"  {symbol}: {len(df)} days fetched")

        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")

    return data


# ============================================================================
# Regime Detection with Different Configurations
# ============================================================================

@dataclass
class RegimeResult:
    """Results from running regime detection on a period."""
    config_name: str
    period_name: str
    regimes: List[Tuple[datetime, RegimeType, float]]  # date, regime, confidence
    transitions: int
    regime_distribution: Dict[str, float]
    strategy_return: float
    sharpe_ratio: float
    max_drawdown: float
    accuracy_score: float  # How well regimes matched actual market behavior


def create_detector_config(config: Dict[str, Any]) -> RegimeDetectorConfig:
    """Create a RegimeDetectorConfig from a dictionary."""
    return RegimeDetectorConfig(
        volatility_high_percentile=config.get("volatility_high_percentile", 90.0),
        strong_trend_threshold=config.get("strong_trend_threshold", 0.6),
        min_hold_days=config.get("min_hold_days", 5),
    )


def run_regime_detection(
    data: pd.DataFrame,
    config: Dict[str, Any],
    warmup_days: int = 60,
) -> Tuple[List[Tuple[datetime, RegimeType, float]], int]:
    """
    Run regime detection on price data with given configuration.

    Returns:
        Tuple of (regime_history, transition_count)
    """
    detector = RegimeDetector(create_detector_config(config))

    regimes = []
    dates = data['datetime'].tolist() if 'datetime' in data.columns else data.index.tolist()

    # Skip warmup period
    start_idx = min(warmup_days, len(data) - 50)

    last_regime = None
    transitions = 0

    for i in range(start_idx, len(data)):
        # Get data up to this point
        hist_data = data.iloc[:i+1].copy()

        try:
            regime = detector.detect_regime(hist_data)
            confidence = detector.get_regime_confidence()

            if last_regime is not None and regime != last_regime:
                transitions += 1

            last_regime = regime

            date = dates[i]
            if isinstance(date, str):
                date = pd.to_datetime(date)

            regimes.append((date, regime, confidence))

        except Exception as e:
            # If detection fails, default to sideways
            regimes.append((dates[i], RegimeType.SIDEWAYS_NEUTRAL, 0.5))

    return regimes, transitions


def calculate_actual_regime(data: pd.DataFrame, date: datetime, lookback: int = 20) -> str:
    """
    Calculate what the actual regime should be based on realized returns.

    This provides a ground truth to measure regime detection accuracy.
    """
    # Get data up to this date
    mask = data['datetime'] <= date if 'datetime' in data.columns else data.index <= date
    hist_data = data[mask].tail(lookback + 5)

    if len(hist_data) < lookback:
        return "sideways_neutral"

    close_col = 'close' if 'close' in hist_data.columns else 'Close'
    prices = hist_data[close_col]

    # Calculate return over lookback
    returns = prices.pct_change().dropna()

    if len(returns) < 5:
        return "sideways_neutral"

    # Calculate metrics
    total_return = (prices.iloc[-1] / prices.iloc[0]) - 1
    volatility = returns.std() * np.sqrt(252)

    # Classification based on actual behavior
    # High volatility if annualized vol > 30%
    if volatility > 0.30:
        return "high_volatility"

    # Strong trend if return > 5% in lookback period
    if total_return > 0.03:
        return "bull_trending"
    elif total_return < -0.03:
        return "bear_crisis"

    return "sideways_neutral"


def calculate_regime_accuracy(
    regimes: List[Tuple[datetime, RegimeType, float]],
    data: pd.DataFrame,
) -> float:
    """
    Calculate how well predicted regimes match actual market behavior.

    Returns accuracy score from 0 to 1.
    """
    if not regimes:
        return 0.5

    correct = 0
    total = 0

    for date, predicted_regime, confidence in regimes:
        actual_regime = calculate_actual_regime(data, date)

        # Check if prediction matches actual
        if predicted_regime.value == actual_regime:
            correct += 1
        # Partial credit for related regimes
        elif (predicted_regime == RegimeType.HIGH_VOLATILITY and actual_regime == "bear_crisis") or \
             (predicted_regime == RegimeType.BEAR_CRISIS and actual_regime == "high_volatility"):
            correct += 0.5

        total += 1

    return correct / total if total > 0 else 0.5


def calculate_strategy_performance(
    data: pd.DataFrame,
    regimes: List[Tuple[datetime, RegimeType, float]],
) -> Tuple[float, float, float]:
    """
    Calculate strategy performance when following regime signals.

    Regime-based position sizing:
    - Bull: 100% invested
    - Sideways: 50% invested
    - Bear/Volatile: 20% invested (defensive)

    Returns:
        Tuple of (total_return, sharpe_ratio, max_drawdown)
    """
    close_col = 'close' if 'close' in data.columns else 'Close'
    date_col = 'datetime' if 'datetime' in data.columns else data.index

    if len(regimes) == 0:
        return 0.0, 0.0, 0.0

    # Create regime lookup
    regime_lookup = {}
    for date, regime, confidence in regimes:
        if isinstance(date, pd.Timestamp):
            date = date.to_pydatetime()
        regime_lookup[date.date() if hasattr(date, 'date') else date] = regime

    # Calculate daily returns with regime-based position sizing
    portfolio_values = [100000]
    daily_returns = []

    position_sizes = {
        RegimeType.BULL_TRENDING: 1.0,
        RegimeType.SIDEWAYS_NEUTRAL: 0.5,
        RegimeType.BEAR_CRISIS: 0.2,
        RegimeType.HIGH_VOLATILITY: 0.2,
    }

    dates = data[date_col] if date_col == 'datetime' else data.index
    prices = data[close_col]

    start_idx = len(data) - len(regimes)

    for i in range(start_idx + 1, len(data)):
        date = dates.iloc[i] if hasattr(dates, 'iloc') else dates[i]
        if isinstance(date, pd.Timestamp):
            date_key = date.to_pydatetime().date()
        else:
            date_key = date.date() if hasattr(date, 'date') else date

        # Get regime for this date
        regime = regime_lookup.get(date_key, RegimeType.SIDEWAYS_NEUTRAL)
        position_size = position_sizes.get(regime, 0.5)

        # Calculate return
        if i > 0:
            market_return = (prices.iloc[i] - prices.iloc[i-1]) / prices.iloc[i-1]
            portfolio_return = market_return * position_size
            daily_returns.append(portfolio_return)

            new_value = portfolio_values[-1] * (1 + portfolio_return)
            portfolio_values.append(new_value)

    if not daily_returns:
        return 0.0, 0.0, 0.0

    returns = pd.Series(daily_returns)

    # Total return
    total_return = (portfolio_values[-1] / portfolio_values[0] - 1) * 100

    # Sharpe ratio
    if returns.std() > 0:
        sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std()
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = pd.Series(portfolio_values)
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = abs(drawdowns.min()) * 100

    return total_return, sharpe, max_dd


def calculate_regime_distribution(
    regimes: List[Tuple[datetime, RegimeType, float]]
) -> Dict[str, float]:
    """Calculate percentage of time spent in each regime."""
    if not regimes:
        return {}

    counts = {}
    for _, regime, _ in regimes:
        counts[regime.value] = counts.get(regime.value, 0) + 1

    total = len(regimes)
    return {k: (v / total) * 100 for k, v in counts.items()}


# ============================================================================
# Main Validation
# ============================================================================

def validate_configuration(
    config_name: str,
    config: Dict[str, Any],
    period_name: str,
    period_start: str,
    period_end: str,
    period_desc: str,
    data: Dict[str, pd.DataFrame],
) -> RegimeResult:
    """
    Validate a single configuration on a single time period.
    """
    # Combine data from all symbols (use SPY as primary)
    primary_symbol = "SPY"
    if primary_symbol not in data:
        primary_symbol = list(data.keys())[0]

    df = data[primary_symbol].copy()

    # Filter to period
    if 'datetime' in df.columns:
        mask = (df['datetime'] >= period_start) & (df['datetime'] <= period_end)
    else:
        mask = (df.index >= period_start) & (df.index <= period_end)

    df_period = df[mask].reset_index(drop=True)

    if len(df_period) < 100:
        print(f"    Warning: Only {len(df_period)} days for {period_name}")
        return RegimeResult(
            config_name=config["name"],
            period_name=period_desc,
            regimes=[],
            transitions=0,
            regime_distribution={},
            strategy_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            accuracy_score=0.5,
        )

    # Run regime detection
    regimes, transitions = run_regime_detection(df_period, config)

    # Calculate metrics
    distribution = calculate_regime_distribution(regimes)
    accuracy = calculate_regime_accuracy(regimes, df_period)
    total_return, sharpe, max_dd = calculate_strategy_performance(df_period, regimes)

    return RegimeResult(
        config_name=config["name"],
        period_name=period_desc,
        regimes=regimes,
        transitions=transitions,
        regime_distribution=distribution,
        strategy_return=total_return,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        accuracy_score=accuracy,
    )


def run_validation() -> Dict[str, Any]:
    """
    Run full validation across all configurations and time periods.
    """
    print("=" * 80)
    print("REGIME THRESHOLD VALIDATION")
    print("=" * 80)

    # Fetch data for full period
    print("\nFetching market data...")
    full_start = "2019-06-01"  # Extra for warmup
    full_end = "2024-12-31"

    data = fetch_data_yfinance(SYMBOLS, full_start, full_end)

    if not data:
        print("ERROR: No data fetched")
        return {}

    # Results storage
    results = {
        "configurations": {},
        "period_results": {},
        "best_by_period": {},
        "overall_best": None,
    }

    # Run validation for each configuration
    print("\nRunning validation...")

    all_results = []

    for config_name, config in CONFIGURATIONS.items():
        print(f"\n  Testing {config['name']}...")
        results["configurations"][config_name] = {
            "name": config["name"],
            "volatility_high_percentile": config["volatility_high_percentile"],
            "strong_trend_threshold": config["strong_trend_threshold"],
            "min_hold_days": config["min_hold_days"],
            "periods": {},
        }

        config_total_return = 0
        config_total_sharpe = 0
        config_total_transitions = 0
        config_total_accuracy = 0
        period_count = 0

        for period_key, (start, end, desc) in TIME_PERIODS.items():
            print(f"    Period: {desc}")

            result = validate_configuration(
                config_name, config, period_key, start, end, desc, data
            )

            results["configurations"][config_name]["periods"][period_key] = {
                "description": desc,
                "transitions": result.transitions,
                "regime_distribution": result.regime_distribution,
                "strategy_return": result.strategy_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "accuracy_score": result.accuracy_score,
            }

            all_results.append(result)

            config_total_return += result.strategy_return
            config_total_sharpe += result.sharpe_ratio
            config_total_transitions += result.transitions
            config_total_accuracy += result.accuracy_score
            period_count += 1

            print(f"      Return: {result.strategy_return:+.1f}%, Sharpe: {result.sharpe_ratio:.2f}, "
                  f"Transitions: {result.transitions}, Accuracy: {result.accuracy_score:.1%}")

        # Calculate averages for this config
        results["configurations"][config_name]["averages"] = {
            "avg_return": config_total_return / period_count if period_count > 0 else 0,
            "avg_sharpe": config_total_sharpe / period_count if period_count > 0 else 0,
            "avg_transitions": config_total_transitions / period_count if period_count > 0 else 0,
            "avg_accuracy": config_total_accuracy / period_count if period_count > 0 else 0,
        }

    # Determine best configuration per period
    print("\n" + "=" * 80)
    print("BEST CONFIGURATION BY PERIOD")
    print("=" * 80)

    for period_key, (start, end, desc) in TIME_PERIODS.items():
        period_results = [r for r in all_results if r.period_name == desc]

        if period_results:
            # Score based on Sharpe ratio and accuracy (weighted)
            best = max(period_results, key=lambda x: 0.6 * x.sharpe_ratio + 0.4 * x.accuracy_score)
            results["best_by_period"][period_key] = {
                "configuration": best.config_name,
                "sharpe_ratio": best.sharpe_ratio,
                "accuracy": best.accuracy_score,
                "return": best.strategy_return,
            }
            print(f"\n{desc}:")
            print(f"  Best: {best.config_name} (Sharpe: {best.sharpe_ratio:.2f}, Accuracy: {best.accuracy_score:.1%})")

    # Determine overall best
    config_scores = {}
    for config_name, config_data in results["configurations"].items():
        avgs = config_data["averages"]
        # Combined score: weighted average of Sharpe, accuracy, and inverse transitions
        score = (
            0.4 * avgs["avg_sharpe"] +
            0.4 * avgs["avg_accuracy"] * 10 +  # Scale accuracy
            0.2 * (1 / (avgs["avg_transitions"] + 1)) * 10  # Penalize frequent transitions
        )
        config_scores[config_name] = score

    best_config = max(config_scores.keys(), key=lambda x: config_scores[x])
    results["overall_best"] = {
        "configuration": best_config,
        "score": config_scores[best_config],
        "details": results["configurations"][best_config]["averages"],
    }

    print("\n" + "=" * 80)
    print("OVERALL BEST CONFIGURATION")
    print("=" * 80)
    print(f"\n  {CONFIGURATIONS[best_config]['name']}")
    print(f"    volatility_high_percentile: {CONFIGURATIONS[best_config]['volatility_high_percentile']}")
    print(f"    strong_trend_threshold: {CONFIGURATIONS[best_config]['strong_trend_threshold']}")
    print(f"    min_hold_days: {CONFIGURATIONS[best_config]['min_hold_days']}")
    print(f"\n    Average Sharpe: {results['overall_best']['details']['avg_sharpe']:.2f}")
    print(f"    Average Accuracy: {results['overall_best']['details']['avg_accuracy']:.1%}")
    print(f"    Average Transitions/Period: {results['overall_best']['details']['avg_transitions']:.1f}")

    return results, all_results, data


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(
    results: Dict[str, Any],
    all_results: List[RegimeResult],
    data: Dict[str, pd.DataFrame],
    output_path: Path,
):
    """Create comprehensive visualization of threshold validation results."""
    fig = plt.figure(figsize=(18, 14))

    # Create grid for subplots
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    # 1. Performance comparison by configuration (top left)
    ax1 = fig.add_subplot(gs[0, 0])

    configs = list(CONFIGURATIONS.keys())
    x = np.arange(len(configs))
    width = 0.25

    sharpes = [results["configurations"][c]["averages"]["avg_sharpe"] for c in configs]
    accuracies = [results["configurations"][c]["averages"]["avg_accuracy"] * 10 for c in configs]  # Scale for visibility
    transitions = [results["configurations"][c]["averages"]["avg_transitions"] / 10 for c in configs]  # Scale down

    bars1 = ax1.bar(x - width, sharpes, width, label='Avg Sharpe', color='blue', alpha=0.7)
    bars2 = ax1.bar(x, accuracies, width, label='Accuracy x10', color='green', alpha=0.7)
    bars3 = ax1.bar(x + width, transitions, width, label='Transitions/10', color='red', alpha=0.7)

    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Value')
    ax1.set_title('Configuration Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([CONFIGURATIONS[c]["name"] for c in configs], rotation=15, ha='right')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')

    # 2. Performance by period heatmap (top right)
    ax2 = fig.add_subplot(gs[0, 1])

    periods = list(TIME_PERIODS.keys())
    period_labels = [TIME_PERIODS[p][2] for p in periods]

    # Create matrix of sharpe ratios
    sharpe_matrix = np.zeros((len(configs), len(periods)))
    for i, config in enumerate(configs):
        for j, period in enumerate(periods):
            sharpe_matrix[i, j] = results["configurations"][config]["periods"].get(period, {}).get("sharpe_ratio", 0)

    im = ax2.imshow(sharpe_matrix, cmap='RdYlGn', aspect='auto')

    ax2.set_xticks(np.arange(len(periods)))
    ax2.set_yticks(np.arange(len(configs)))
    ax2.set_xticklabels(period_labels, rotation=30, ha='right', fontsize=9)
    ax2.set_yticklabels([CONFIGURATIONS[c]["name"] for c in configs])

    # Add value annotations
    for i in range(len(configs)):
        for j in range(len(periods)):
            text = ax2.text(j, i, f'{sharpe_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=9)

    ax2.set_title('Sharpe Ratio by Configuration and Period', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Sharpe Ratio')

    # 3. Price chart with regime shading for best config (middle, spans both columns)
    ax3 = fig.add_subplot(gs[1, :])

    primary_symbol = "SPY" if "SPY" in data else list(data.keys())[0]
    df = data[primary_symbol]

    # Get regimes for best configuration
    best_config_name = results["overall_best"]["configuration"]
    best_config = CONFIGURATIONS[best_config_name]

    # Run regime detection on full period
    regimes, _ = run_regime_detection(df, best_config, warmup_days=60)

    # Plot price
    dates = df['datetime'] if 'datetime' in df.columns else df.index
    close_col = 'close' if 'close' in df.columns else 'Close'
    prices = df[close_col]

    ax3.plot(dates, prices, 'k-', linewidth=1, label=f'{primary_symbol} Price')

    # Shade regime periods
    regime_colors = {
        RegimeType.BULL_TRENDING: ('lightgreen', 'Bull'),
        RegimeType.BEAR_CRISIS: ('lightcoral', 'Bear'),
        RegimeType.SIDEWAYS_NEUTRAL: ('lightyellow', 'Sideways'),
        RegimeType.HIGH_VOLATILITY: ('lightblue', 'High Vol'),
    }

    # Create regime spans
    if regimes:
        regime_dates = [r[0] for r in regimes]
        regime_types = [r[1] for r in regimes]

        # Find start of regime data
        start_idx = len(dates) - len(regimes)

        current_regime = regime_types[0]
        span_start = regime_dates[0]

        for i in range(1, len(regime_types)):
            if regime_types[i] != current_regime or i == len(regime_types) - 1:
                span_end = regime_dates[i]
                color = regime_colors.get(current_regime, ('white', 'Unknown'))[0]
                ax3.axvspan(span_start, span_end, alpha=0.3, color=color)
                span_start = regime_dates[i]
                current_regime = regime_types[i]

    ax3.set_title(f'{primary_symbol} Price with Regime Shading ({best_config["name"]})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Price ($)')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add regime legend
    regime_patches = [mpatches.Patch(color=color, alpha=0.3, label=label)
                      for regime_type, (color, label) in regime_colors.items()]
    ax3.legend(handles=regime_patches, loc='upper left', fontsize=9)

    # 4. Regime distribution comparison (bottom left)
    ax4 = fig.add_subplot(gs[2, 0])

    regime_types = ['bull_trending', 'bear_crisis', 'sideways_neutral', 'high_volatility']
    colors = ['green', 'red', 'gold', 'blue']

    # Get distributions for each config (average across periods)
    distributions = {}
    for config in configs:
        distributions[config] = {rt: 0 for rt in regime_types}
        count = 0
        for period in TIME_PERIODS.keys():
            period_dist = results["configurations"][config]["periods"].get(period, {}).get("regime_distribution", {})
            for rt in regime_types:
                distributions[config][rt] += period_dist.get(rt, 0)
            count += 1
        for rt in regime_types:
            distributions[config][rt] /= max(count, 1)

    x = np.arange(len(configs))
    width = 0.2

    for i, (rt, color) in enumerate(zip(regime_types, colors)):
        values = [distributions[c][rt] for c in configs]
        ax4.bar(x + i * width - 1.5 * width, values, width, label=rt.replace('_', ' ').title(), color=color, alpha=0.7)

    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('% of Time')
    ax4.set_title('Average Regime Distribution by Configuration', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([CONFIGURATIONS[c]["name"] for c in configs], rotation=15, ha='right')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    # 5. Return comparison by period (bottom right)
    ax5 = fig.add_subplot(gs[2, 1])

    x = np.arange(len(periods))
    width = 0.2

    for i, config in enumerate(configs):
        returns = [results["configurations"][config]["periods"].get(p, {}).get("strategy_return", 0) for p in periods]
        ax5.bar(x + i * width - 1.5 * width, returns, width,
                label=CONFIGURATIONS[config]["name"], alpha=0.7)

    ax5.set_xlabel('Period')
    ax5.set_ylabel('Strategy Return (%)')
    ax5.set_title('Strategy Return by Configuration and Period', fontsize=12, fontweight='bold')
    ax5.set_xticks(x)
    ax5.set_xticklabels(period_labels, rotation=30, ha='right', fontsize=9)
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"\n  Saved visualization to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    """Main entry point."""
    # Run validation
    results, all_results, data = run_validation()

    if not results:
        print("Validation failed")
        return 1

    # Save results
    output_json = OUTPUT_DIR / "threshold_validation.json"

    # Convert results to JSON-serializable format
    json_results = {
        "generated_at": datetime.now().isoformat(),
        "symbols_tested": SYMBOLS,
        "configurations": {},
        "best_by_period": results["best_by_period"],
        "overall_best": results["overall_best"],
    }

    for config_name, config_data in results["configurations"].items():
        json_results["configurations"][config_name] = {
            "name": config_data["name"],
            "parameters": {
                "volatility_high_percentile": CONFIGURATIONS[config_name]["volatility_high_percentile"],
                "strong_trend_threshold": CONFIGURATIONS[config_name]["strong_trend_threshold"],
                "min_hold_days": CONFIGURATIONS[config_name]["min_hold_days"],
            },
            "averages": config_data["averages"],
            "periods": config_data["periods"],
        }

    with open(output_json, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    print(f"\nSaved results to {output_json}")

    # Create visualizations
    print("\nCreating visualizations...")
    output_png = OUTPUT_DIR / "regime_threshold_comparison.png"
    create_visualizations(results, all_results, data, output_png)

    # Print summary table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)

    print(f"\n{'Configuration':<20} {'Avg Return':>12} {'Avg Sharpe':>12} {'Avg Accuracy':>14} {'Transitions':>12}")
    print("-" * 75)

    for config_name in CONFIGURATIONS.keys():
        avgs = results["configurations"][config_name]["averages"]
        print(f"{CONFIGURATIONS[config_name]['name']:<20} "
              f"{avgs['avg_return']:>+11.1f}% "
              f"{avgs['avg_sharpe']:>12.2f} "
              f"{avgs['avg_accuracy']:>13.1%} "
              f"{avgs['avg_transitions']:>12.1f}")

    print("-" * 75)

    # Recommendation
    best = results["overall_best"]["configuration"]
    print(f"\nRECOMMENDATION: Use {CONFIGURATIONS[best]['name']} configuration")
    print(f"  - volatility_high_percentile: {CONFIGURATIONS[best]['volatility_high_percentile']}")
    print(f"  - strong_trend_threshold: {CONFIGURATIONS[best]['strong_trend_threshold']}")
    print(f"  - min_hold_days: {CONFIGURATIONS[best]['min_hold_days']}")

    print("\n" + "=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())

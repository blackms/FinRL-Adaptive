#!/usr/bin/env python3
"""
Cross-Asset Validation for Regime-Blend Strategy

Tests the regime-aware blended strategy across different asset classes:
- Tech Stocks: AAPL, MSFT, NVDA (original)
- Sector ETFs: XLF (Financials), XLE (Energy), XLV (Healthcare), XLK (Tech)
- Broad Market: SPY, QQQ, IWM
- International: EFA (Developed), EEM (Emerging)
- Bonds: TLT (Long-term Treasury), AGG (Aggregate Bond)
- Commodities: GLD (Gold), USO (Oil)

Usage:
    python scripts/cross_asset_validation.py
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
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

from trading.strategies.regime_detector import RegimeDetector, RegimeDetectorConfig, RegimeType


# ============================================================================
# Configuration
# ============================================================================

START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 100000
WARMUP_DAYS = 100

OUTPUT_DIR = Path("/opt/FinRL/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Asset classes to test
ASSET_CLASSES = {
    "Tech Stocks": ["AAPL", "MSFT", "NVDA"],
    "Sector ETFs": ["XLF", "XLE", "XLV", "XLK"],
    "Broad Market": ["SPY", "QQQ", "IWM"],
    "International": ["EFA", "EEM"],
    "Bonds": ["TLT", "AGG"],
    "Commodities": ["GLD", "USO"],
}


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_yfinance_data(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch data for symbols using yfinance directly."""
    data = {}

    # Add warmup period
    start_dt = pd.Timestamp(start) - pd.Timedelta(days=150)
    start_with_warmup = start_dt.strftime('%Y-%m-%d')

    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_with_warmup, end=end, interval="1d")

            if df.empty:
                print(f"   Warning: No data for {symbol}")
                continue

            # Standardize columns
            df = df.reset_index()
            df.columns = [col.replace(' ', '_') for col in df.columns]

            # Rename 'Date' to 'datetime' if present
            if 'Date' in df.columns:
                df = df.rename(columns={'Date': 'datetime'})

            # Ensure datetime is timezone-naive
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
                df = df.set_index('datetime')

            data[symbol] = df

        except Exception as e:
            print(f"   Error fetching {symbol}: {e}")
            continue

    return data


# ============================================================================
# Regime-Aware Strategy (Simplified for single assets)
# ============================================================================

class CrossAssetRegimeStrategy:
    """
    Strategy that adapts to market regimes for any asset class.
    """

    def __init__(self, regime_config: Optional[RegimeDetectorConfig] = None):
        self.regime_detector = RegimeDetector(regime_config or RegimeDetectorConfig())

        # Regime-specific parameters
        self.regime_params = {
            RegimeType.BULL_TRENDING: {"exposure": 1.0, "momentum_weight": 0.7},
            RegimeType.BEAR_CRISIS: {"exposure": 0.3, "momentum_weight": 0.2},
            RegimeType.SIDEWAYS_NEUTRAL: {"exposure": 0.6, "momentum_weight": 0.4},
            RegimeType.HIGH_VOLATILITY: {"exposure": 0.4, "momentum_weight": 0.3},
        }

    def detect_regime(self, df: pd.DataFrame) -> Tuple[RegimeType, float]:
        """Detect regime from price data."""
        if len(df) < 50:
            return RegimeType.SIDEWAYS_NEUTRAL, 0.5

        # Create OHLCV DataFrame for regime detector
        close_col = 'Close' if 'Close' in df.columns else 'close'
        high_col = 'High' if 'High' in df.columns else 'high'
        low_col = 'Low' if 'Low' in df.columns else 'low'
        open_col = 'Open' if 'Open' in df.columns else 'open'

        regime_df = pd.DataFrame({
            'open': df[open_col].values,
            'high': df[high_col].values,
            'low': df[low_col].values,
            'close': df[close_col].values,
            'volume': df.get('Volume', df.get('volume', pd.Series(np.ones(len(df)) * 1000000))).values,
        })

        regime = self.regime_detector.detect_regime(regime_df)
        confidence = self.regime_detector.get_regime_confidence()
        return regime, confidence

    def get_signal(self, df: pd.DataFrame, regime: RegimeType) -> float:
        """
        Generate trading signal based on regime.
        Returns -1 (full sell) to +1 (full buy).
        """
        if len(df) < 60:
            return 0.0

        close_col = 'Close' if 'Close' in df.columns else 'close'
        prices = df[close_col].values

        params = self.regime_params.get(regime, {"exposure": 0.5, "momentum_weight": 0.5})

        # Momentum signal (20-day ROC)
        if len(prices) > 20:
            momentum = (prices[-1] - prices[-20]) / prices[-20]
            momentum_signal = np.clip(momentum * 5, -1, 1)
        else:
            momentum_signal = 0

        # Mean reversion signal (price vs 50-day SMA)
        if len(prices) > 50:
            sma_50 = np.mean(prices[-50:])
            deviation = (prices[-1] - sma_50) / sma_50
            mean_rev_signal = np.clip(-deviation * 3, -1, 1)
        else:
            mean_rev_signal = 0

        # Blend based on regime
        mom_weight = params["momentum_weight"]
        signal = mom_weight * momentum_signal + (1 - mom_weight) * mean_rev_signal

        # Scale by exposure
        signal *= params["exposure"]

        return np.clip(signal, -1, 1)


# ============================================================================
# Backtest Engine
# ============================================================================

def run_regime_strategy_backtest(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    capital: float,
    start_date: str,
    end_date: str,
    rebalance_freq: int = 5,
) -> Dict[str, Any]:
    """
    Run backtest with regime-aware strategy for an asset class.
    """
    strategy = CrossAssetRegimeStrategy()

    # Filter symbols with valid data
    valid_symbols = [s for s in symbols if s in data and len(data[s]) >= WARMUP_DAYS]

    if not valid_symbols:
        return {'error': 'No valid symbols', 'symbols': symbols}

    # Get common trading dates
    try:
        common_dates = sorted(set.intersection(*[set(data[s].index) for s in valid_symbols]))
    except Exception:
        return {'error': 'Cannot find common dates', 'symbols': valid_symbols}

    # Filter to trading period
    trading_dates = [d for d in common_dates
                     if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]

    if len(trading_dates) < WARMUP_DAYS:
        return {'error': 'Insufficient trading dates', 'symbols': valid_symbols}

    # Skip warmup
    trading_dates = trading_dates[WARMUP_DAYS:]

    if len(trading_dates) < 10:
        return {'error': 'Insufficient trading dates after warmup', 'symbols': valid_symbols}

    # Initialize tracking
    cash = capital
    positions = {s: 0.0 for s in valid_symbols}

    portfolio_values = []
    daily_returns = []
    regime_history = []
    regime_counts = {r: 0 for r in RegimeType}

    for i, date in enumerate(trading_dates):
        # Get historical data up to this date
        hist_data = {s: df[df.index <= date].tail(120) for s, df in data.items() if s in valid_symbols}

        # Get current prices
        close_col = 'Close' if 'Close' in data[valid_symbols[0]].columns else 'close'
        prices = {}
        for s in valid_symbols:
            if date in data[s].index:
                prices[s] = data[s].loc[date, close_col]

        if not prices:
            continue

        # Calculate portfolio value
        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in valid_symbols)
        current_value = cash + position_value

        # Detect regime using market proxy (equal-weighted average)
        all_hist = []
        for s in valid_symbols:
            if s in hist_data and len(hist_data[s]) >= 50:
                all_hist.append(hist_data[s])

        if all_hist:
            # Use first symbol as proxy for regime detection
            proxy_df = all_hist[0]
            regime, confidence = strategy.detect_regime(proxy_df)
        else:
            regime = RegimeType.SIDEWAYS_NEUTRAL
            confidence = 0.5

        regime_counts[regime] += 1
        regime_history.append((date, regime.value, confidence))

        # Rebalance at intervals
        should_rebalance = (i == 0 or i % rebalance_freq == 0)

        if should_rebalance:
            signals = {}
            for s in valid_symbols:
                if s in hist_data and len(hist_data[s]) >= 30:
                    signals[s] = strategy.get_signal(hist_data[s], regime)
                else:
                    signals[s] = 0

            # Convert signals to weights
            positive_signals = {k: max(0, v + 0.5) for k, v in signals.items()}
            total = sum(positive_signals.values())
            weights = {k: v / total if total > 0 else 1/len(valid_symbols) for k, v in positive_signals.items()}

            # Execute rebalance
            for symbol in valid_symbols:
                if symbol not in prices:
                    continue

                target_value = weights.get(symbol, 0) * current_value
                current_shares = positions.get(symbol, 0)
                current_pos_value = current_shares * prices[symbol]

                trade_value = target_value - current_pos_value
                trade_shares = trade_value / prices[symbol]

                if abs(trade_shares) > 0.01:
                    positions[symbol] = current_shares + trade_shares
                    cash -= trade_value
                    cash -= abs(trade_value) * 0.001  # Transaction cost

        # Record daily value
        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in valid_symbols)
        daily_value = cash + position_value

        portfolio_values.append({'date': date, 'value': daily_value, 'regime': regime.value})

        if len(portfolio_values) > 1:
            prev_value = portfolio_values[-2]['value']
            if prev_value > 0:
                daily_returns.append((daily_value - prev_value) / prev_value)

    if not portfolio_values:
        return {'error': 'No portfolio values generated', 'symbols': valid_symbols}

    # Calculate metrics
    values = pd.Series([p['value'] for p in portfolio_values])
    returns = pd.Series(daily_returns) if daily_returns else pd.Series([0])

    total_return = (values.iloc[-1] - capital) / capital * 100 if len(values) > 0 else 0

    if len(returns) > 1 and returns.std() > 0:
        sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std()
        volatility = returns.std() * np.sqrt(252) * 100
    else:
        sharpe = 0
        volatility = 0

    # Max drawdown
    if len(returns) > 0:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdowns = (cumulative - running_max) / running_max
        max_dd = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0
    else:
        max_dd = 0

    # Regime distribution
    total_days = sum(regime_counts.values())
    regime_distribution = {
        r.value: {
            'days': regime_counts[r],
            'percentage': regime_counts[r] / total_days * 100 if total_days > 0 else 0
        }
        for r in RegimeType if regime_counts[r] > 0
    }

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility': volatility,
        'final_value': values.iloc[-1] if len(values) > 0 else capital,
        'trading_days': len(trading_dates),
        'regime_distribution': regime_distribution,
        'portfolio_values': portfolio_values,
        'symbols': valid_symbols,
    }


def run_buy_and_hold(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    capital: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Run buy-and-hold benchmark for an asset class."""
    valid_symbols = [s for s in symbols if s in data and len(data[s]) >= WARMUP_DAYS]

    if not valid_symbols:
        return {'error': 'No valid symbols', 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0}

    try:
        common_dates = sorted(set.intersection(*[set(data[s].index) for s in valid_symbols]))
    except Exception:
        return {'error': 'Cannot find common dates', 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0}

    trading_dates = [d for d in common_dates
                     if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]

    if len(trading_dates) < WARMUP_DAYS:
        return {'error': 'Insufficient dates', 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0}

    trading_dates = trading_dates[WARMUP_DAYS:]

    if len(trading_dates) < 10:
        return {'error': 'Insufficient dates after warmup', 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0}

    # Equal weight buy-and-hold
    close_col = 'Close' if 'Close' in data[valid_symbols[0]].columns else 'close'
    capital_per_stock = capital / len(valid_symbols)
    first_date = trading_dates[0]

    shares = {}
    for s in valid_symbols:
        if first_date in data[s].index:
            shares[s] = capital_per_stock / data[s].loc[first_date, close_col]

    portfolio_values = []
    daily_returns = []

    for date in trading_dates:
        prices = {}
        for s in valid_symbols:
            if date in data[s].index:
                prices[s] = data[s].loc[date, close_col]

        value = sum(shares.get(s, 0) * prices.get(s, 0) for s in valid_symbols)
        portfolio_values.append({'date': date, 'value': value})

        if len(portfolio_values) > 1:
            prev = portfolio_values[-2]['value']
            if prev > 0:
                daily_returns.append((value - prev) / prev)

    if not portfolio_values:
        return {'error': 'No portfolio values', 'total_return': 0, 'sharpe_ratio': 0, 'max_drawdown': 0, 'volatility': 0}

    values = pd.Series([p['value'] for p in portfolio_values])
    returns = pd.Series(daily_returns) if daily_returns else pd.Series([0])

    total_return = (values.iloc[-1] - capital) / capital * 100 if len(values) > 0 else 0
    sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std() if len(returns) > 1 and returns.std() > 0 else 0

    if len(returns) > 0:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        max_dd = abs(((cumulative - running_max) / running_max).min()) * 100
    else:
        max_dd = 0

    volatility = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility': volatility,
        'final_value': values.iloc[-1] if len(values) > 0 else capital,
        'portfolio_values': portfolio_values,
    }


# ============================================================================
# Visualization
# ============================================================================

def create_cross_asset_visualization(results: Dict[str, Dict[str, Any]], output_path: Path):
    """Create visualization comparing strategy performance across asset classes."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Filter valid results
    valid_results = {k: v for k, v in results.items() if 'error' not in v['regime_strategy']}

    if not valid_results:
        print("   No valid results to visualize")
        return

    asset_classes = list(valid_results.keys())

    # Color map for asset classes
    colors = plt.cm.tab10(np.linspace(0, 1, len(asset_classes)))
    color_map = dict(zip(asset_classes, colors))

    # 1. Return comparison (Strategy vs Buy & Hold)
    ax1 = axes[0, 0]

    x = np.arange(len(asset_classes))
    width = 0.35

    strategy_returns = [valid_results[ac]['regime_strategy']['total_return'] for ac in asset_classes]
    bh_returns = [valid_results[ac]['buy_and_hold']['total_return'] for ac in asset_classes]

    bars1 = ax1.bar(x - width/2, strategy_returns, width, label='Regime Strategy', color='blue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, bh_returns, width, label='Buy & Hold', color='gray', alpha=0.7)

    ax1.set_ylabel('Total Return (%)')
    ax1.set_title('Total Return by Asset Class', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels([ac.replace(' ', '\n') for ac in asset_classes], fontsize=9)
    ax1.legend()
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars1, strategy_returns):
        ax1.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va='bottom', fontsize=8)

    # 2. Sharpe ratio comparison
    ax2 = axes[0, 1]

    strategy_sharpes = [valid_results[ac]['regime_strategy']['sharpe_ratio'] for ac in asset_classes]
    bh_sharpes = [valid_results[ac]['buy_and_hold']['sharpe_ratio'] for ac in asset_classes]

    bars1 = ax2.bar(x - width/2, strategy_sharpes, width, label='Regime Strategy', color='green', alpha=0.7)
    bars2 = ax2.bar(x + width/2, bh_sharpes, width, label='Buy & Hold', color='gray', alpha=0.7)

    ax2.set_ylabel('Sharpe Ratio')
    ax2.set_title('Risk-Adjusted Return (Sharpe) by Asset Class', fontsize=14)
    ax2.set_xticks(x)
    ax2.set_xticklabels([ac.replace(' ', '\n') for ac in asset_classes], fontsize=9)
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Alpha (Strategy - Buy & Hold)
    ax3 = axes[1, 0]

    alphas = [valid_results[ac]['alpha'] for ac in asset_classes]
    colors_alpha = ['green' if a > 0 else 'red' for a in alphas]

    bars = ax3.bar(asset_classes, alphas, color=colors_alpha, alpha=0.7)
    ax3.set_ylabel('Alpha (%)')
    ax3.set_title('Strategy Alpha vs Buy & Hold', fontsize=14)
    ax3.set_xticklabels([ac.replace(' ', '\n') for ac in asset_classes], fontsize=9, rotation=0)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, alphas):
        va = 'bottom' if val >= 0 else 'top'
        ax3.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     ha='center', va=va, fontsize=9, fontweight='bold')

    # 4. Drawdown comparison
    ax4 = axes[1, 1]

    strategy_dd = [valid_results[ac]['regime_strategy']['max_drawdown'] for ac in asset_classes]
    bh_dd = [valid_results[ac]['buy_and_hold']['max_drawdown'] for ac in asset_classes]

    bars1 = ax4.bar(x - width/2, strategy_dd, width, label='Regime Strategy', color='orange', alpha=0.7)
    bars2 = ax4.bar(x + width/2, bh_dd, width, label='Buy & Hold', color='gray', alpha=0.7)

    ax4.set_ylabel('Max Drawdown (%)')
    ax4.set_title('Maximum Drawdown by Asset Class', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels([ac.replace(' ', '\n') for ac in asset_classes], fontsize=9)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   Saved visualization to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("CROSS-ASSET VALIDATION: REGIME-BLEND STRATEGY")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"   Period:   {START_DATE} to {END_DATE}")
    print(f"   Capital:  ${INITIAL_CAPITAL:,}")
    print(f"   Asset Classes: {len(ASSET_CLASSES)}")

    # Collect all results
    all_results: Dict[str, Dict[str, Any]] = {}

    # Process each asset class
    for asset_class, symbols in ASSET_CLASSES.items():
        print(f"\n{'='*60}")
        print(f"Testing: {asset_class}")
        print(f"Symbols: {', '.join(symbols)}")
        print("=" * 60)

        # Fetch data
        print("\n   Fetching data...")
        data = fetch_yfinance_data(symbols, START_DATE, END_DATE)
        print(f"   Loaded {len(data)}/{len(symbols)} symbols")

        if not data:
            print(f"   ERROR: No data available for {asset_class}")
            all_results[asset_class] = {
                'regime_strategy': {'error': 'No data available'},
                'buy_and_hold': {'error': 'No data available'},
                'alpha': 0,
                'symbols': symbols,
            }
            continue

        # Run regime strategy backtest
        print("   Running regime strategy backtest...")
        regime_result = run_regime_strategy_backtest(
            data, list(data.keys()), INITIAL_CAPITAL, START_DATE, END_DATE
        )

        # Run buy & hold benchmark
        print("   Running buy & hold benchmark...")
        bh_result = run_buy_and_hold(
            data, list(data.keys()), INITIAL_CAPITAL, START_DATE, END_DATE
        )

        # Calculate alpha
        if 'error' not in regime_result and 'error' not in bh_result:
            alpha = regime_result['total_return'] - bh_result['total_return']
        else:
            alpha = 0

        all_results[asset_class] = {
            'regime_strategy': regime_result,
            'buy_and_hold': bh_result,
            'alpha': alpha,
            'symbols': regime_result.get('symbols', symbols),
        }

        # Print results for this asset class
        if 'error' not in regime_result:
            print(f"\n   Results for {asset_class}:")
            print(f"   {'Metric':<20} {'Regime Strategy':>15} {'Buy & Hold':>15} {'Difference':>12}")
            print(f"   {'-'*62}")
            print(f"   {'Total Return':<20} {regime_result['total_return']:>+14.2f}% {bh_result['total_return']:>+14.2f}% {alpha:>+11.2f}%")
            print(f"   {'Sharpe Ratio':<20} {regime_result['sharpe_ratio']:>15.2f} {bh_result['sharpe_ratio']:>15.2f} {regime_result['sharpe_ratio'] - bh_result['sharpe_ratio']:>+12.2f}")
            print(f"   {'Max Drawdown':<20} {regime_result['max_drawdown']:>14.2f}% {bh_result['max_drawdown']:>14.2f}% {bh_result['max_drawdown'] - regime_result['max_drawdown']:>+11.2f}%")
            print(f"   {'Volatility':<20} {regime_result['volatility']:>14.2f}% {bh_result['volatility']:>14.2f}%")

            if regime_result.get('regime_distribution'):
                print(f"\n   Regime Distribution:")
                for regime, dist in regime_result['regime_distribution'].items():
                    print(f"      {regime:<20} {dist['days']:>5} days ({dist['percentage']:>5.1f}%)")
        else:
            print(f"   ERROR: {regime_result.get('error', 'Unknown error')}")

    # ========================================================================
    # Summary Analysis
    # ========================================================================

    print("\n" + "=" * 80)
    print("CROSS-ASSET VALIDATION SUMMARY")
    print("=" * 80)

    # Filter valid results
    valid_results = {k: v for k, v in all_results.items() if 'error' not in v['regime_strategy']}

    if not valid_results:
        print("\nNo valid results to summarize.")
        return

    # Print comparison table
    print(f"\n{'Asset Class':<20} {'Return':>10} {'Sharpe':>10} {'Max DD':>10} {'Alpha':>10} {'Works?':>10}")
    print("-" * 75)

    for asset_class, results in valid_results.items():
        rs = results['regime_strategy']
        alpha = results['alpha']
        works = "YES" if alpha > 0 else "NO"
        works_color = works

        print(f"{asset_class:<20} {rs['total_return']:>+9.2f}% {rs['sharpe_ratio']:>10.2f} {rs['max_drawdown']:>9.2f}% {alpha:>+9.2f}% {works:>10}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    # Best performing asset classes
    sorted_by_alpha = sorted(valid_results.items(), key=lambda x: x[1]['alpha'], reverse=True)

    print("\n1. ASSET CLASSES WHERE STRATEGY ADDS VALUE (Alpha > 0):")
    positive_alpha = [ac for ac, r in sorted_by_alpha if r['alpha'] > 0]
    if positive_alpha:
        for ac in positive_alpha:
            alpha = valid_results[ac]['alpha']
            sharpe_diff = valid_results[ac]['regime_strategy']['sharpe_ratio'] - valid_results[ac]['buy_and_hold']['sharpe_ratio']
            print(f"   - {ac}: +{alpha:.2f}% alpha, Sharpe improvement: {sharpe_diff:+.2f}")
    else:
        print("   None - strategy underperforms Buy & Hold across all asset classes")

    print("\n2. ASSET CLASSES WHERE STRATEGY UNDERPERFORMS (Alpha < 0):")
    negative_alpha = [ac for ac, r in sorted_by_alpha if r['alpha'] < 0]
    if negative_alpha:
        for ac in negative_alpha:
            alpha = valid_results[ac]['alpha']
            print(f"   - {ac}: {alpha:.2f}% alpha")
    else:
        print("   None - strategy outperforms Buy & Hold across all asset classes")

    # Best risk-adjusted returns
    print("\n3. BEST RISK-ADJUSTED PERFORMANCE (Sharpe Ratio):")
    sorted_by_sharpe = sorted(valid_results.items(),
                              key=lambda x: x[1]['regime_strategy']['sharpe_ratio'],
                              reverse=True)
    for i, (ac, r) in enumerate(sorted_by_sharpe[:3], 1):
        print(f"   {i}. {ac}: Sharpe = {r['regime_strategy']['sharpe_ratio']:.2f}")

    # Drawdown protection
    print("\n4. DRAWDOWN PROTECTION ANALYSIS:")
    for ac, r in valid_results.items():
        dd_reduction = r['buy_and_hold']['max_drawdown'] - r['regime_strategy']['max_drawdown']
        if dd_reduction > 0:
            print(f"   - {ac}: Reduced max drawdown by {dd_reduction:.2f}%")
        else:
            print(f"   - {ac}: Increased max drawdown by {abs(dd_reduction):.2f}%")

    # Regime consistency analysis
    print("\n5. REGIME SIGNAL CONSISTENCY:")
    regime_data = {}
    for ac, r in valid_results.items():
        if 'regime_distribution' in r['regime_strategy']:
            regime_data[ac] = r['regime_strategy']['regime_distribution']

    if regime_data:
        print("   Distribution of detected regimes across asset classes:")
        all_regimes = set()
        for rd in regime_data.values():
            all_regimes.update(rd.keys())

        print(f"\n   {'Regime':<25}", end='')
        for ac in regime_data.keys():
            print(f"{ac[:10]:>12}", end='')
        print()

        print(f"   {'-'*25}", end='')
        for _ in regime_data.keys():
            print(f"{'-'*12}", end='')
        print()

        for regime in sorted(all_regimes):
            print(f"   {regime:<25}", end='')
            for ac, rd in regime_data.items():
                pct = rd.get(regime, {}).get('percentage', 0)
                print(f"{pct:>11.1f}%", end='')
            print()

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    avg_alpha = np.mean([r['alpha'] for r in valid_results.values()])
    positive_count = len(positive_alpha)
    total_count = len(valid_results)

    print(f"\n   Average Alpha across asset classes: {avg_alpha:+.2f}%")
    print(f"   Asset classes with positive alpha: {positive_count}/{total_count}")

    if positive_count > total_count / 2:
        print("\n   CONCLUSION: The regime-blend strategy GENERALLY ADDS VALUE")
        print(f"   across multiple asset classes ({positive_count}/{total_count}).")
    elif positive_count > 0:
        print("\n   CONCLUSION: The regime-blend strategy shows MIXED RESULTS.")
        print(f"   Works best on: {', '.join(positive_alpha)}")
    else:
        print("\n   CONCLUSION: The regime-blend strategy UNDERPERFORMS Buy & Hold")
        print("   across all tested asset classes during this period.")

    if positive_alpha:
        best_ac = sorted_by_alpha[0][0]
        best_alpha = sorted_by_alpha[0][1]['alpha']
        print(f"\n   BEST ASSET CLASS: {best_ac} with +{best_alpha:.2f}% alpha")

    # Save results to JSON
    output_json = OUTPUT_DIR / 'cross_asset_validation.json'

    # Prepare JSON-serializable results
    json_results = {
        'configuration': {
            'start_date': START_DATE,
            'end_date': END_DATE,
            'initial_capital': INITIAL_CAPITAL,
            'asset_classes': {k: v for k, v in ASSET_CLASSES.items()},
        },
        'results': {},
        'summary': {
            'average_alpha': avg_alpha,
            'positive_alpha_count': positive_count,
            'total_asset_classes': total_count,
            'best_asset_class': sorted_by_alpha[0][0] if sorted_by_alpha else None,
            'best_alpha': sorted_by_alpha[0][1]['alpha'] if sorted_by_alpha else 0,
        }
    }

    for ac, r in all_results.items():
        json_results['results'][ac] = {
            'symbols': r.get('symbols', []),
            'alpha': r['alpha'],
            'regime_strategy': {
                k: v for k, v in r['regime_strategy'].items()
                if k not in ['portfolio_values']
            },
            'buy_and_hold': {
                k: v for k, v in r['buy_and_hold'].items()
                if k not in ['portfolio_values']
            },
        }

    with open(output_json, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"\n   Saved results to {output_json}")

    # Create visualization
    print("\n   Creating visualization...")
    output_png = OUTPUT_DIR / 'cross_asset_performance.png'
    create_cross_asset_visualization(all_results, output_png)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

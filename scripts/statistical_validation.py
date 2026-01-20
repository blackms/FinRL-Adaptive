#!/usr/bin/env python3
"""
Statistical Validation for Regime-Blend Strategy

Performs comprehensive statistical analysis including:
1. Bootstrap Confidence Intervals for Sharpe and total return
2. Regime Transition Analysis
3. Drawdown Analysis (time to recovery, duration distribution, underwater curve)
4. Risk-Adjusted Metrics (Calmar, Sortino, Omega)
5. Correlation Analysis with SPY benchmark

Usage:
    python scripts/statistical_validation.py
"""

import sys
sys.path.insert(0, '/opt/FinRL/src')

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

from trading.data.fetcher import SP500DataFetcher, CacheConfig
from trading.strategies.regime_detector import RegimeDetector, RegimeDetectorConfig, RegimeType

# ============================================================================
# Configuration
# ============================================================================

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 100000
WARMUP_DAYS = 100
BOOTSTRAP_ITERATIONS = 1000
CONFIDENCE_LEVEL = 0.95

OUTPUT_DIR = Path("/opt/FinRL/output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Data Loading
# ============================================================================

def fetch_strategy_data(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch stock data with caching."""
    cache_config = CacheConfig(enabled=True, directory=Path("/opt/FinRL/.cache"))
    fetcher = SP500DataFetcher(cache_config=cache_config)

    start_dt = pd.Timestamp(start) - pd.Timedelta(days=150)
    start_with_warmup = start_dt.strftime('%Y-%m-%d')

    raw_data = fetcher.fetch_ohlcv(symbols=symbols, start=start_with_warmup, end=end, interval="1d")

    data = {}
    for symbol, stock_data in raw_data.items():
        df = stock_data.data.copy()
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'}
        df = df.rename(columns=col_map)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df = df.set_index('datetime')
        data[symbol] = df

    return data


def fetch_spy_benchmark(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY data using yfinance."""
    print("   Fetching SPY benchmark data from yfinance...")
    spy = yf.download('SPY', start=start, end=end, progress=False)
    spy.index = spy.index.tz_localize(None) if spy.index.tz is not None else spy.index
    return spy


class RegimeAwareStrategy:
    """
    Strategy that adapts to market regimes by blending momentum and hedge fund approaches.
    Simplified version for statistical validation.
    """

    def __init__(self, symbols: List[str], initial_capital: float = 100000):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.regime_detector = RegimeDetector(RegimeDetectorConfig())

        # Regime-specific weights
        self.regime_weights = {
            RegimeType.BULL_TRENDING: {"momentum": 0.65, "adaptive_hf": 0.35},
            RegimeType.BEAR_CRISIS: {"momentum": 0.10, "adaptive_hf": 0.90},
            RegimeType.SIDEWAYS_NEUTRAL: {"momentum": 0.30, "adaptive_hf": 0.70},
            RegimeType.HIGH_VOLATILITY: {"momentum": 0.15, "adaptive_hf": 0.85},
        }

        self.regime_exposure = {
            RegimeType.BULL_TRENDING: 1.0,
            RegimeType.BEAR_CRISIS: 0.50,
            RegimeType.SIDEWAYS_NEUTRAL: 0.75,
            RegimeType.HIGH_VOLATILITY: 0.60,
        }

    def detect_regime(self, data: Dict[str, pd.DataFrame], date: datetime) -> Tuple[RegimeType, float]:
        """Detect current market regime using available data."""
        all_returns = []
        for symbol, df in data.items():
            if len(df) >= 50:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                returns = df[close_col].pct_change()
                all_returns.append(returns)

        if not all_returns:
            return RegimeType.SIDEWAYS_NEUTRAL, 0.5

        avg_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        proxy_prices = (1 + avg_returns).cumprod() * 100
        proxy_prices = proxy_prices.dropna()

        proxy_df = pd.DataFrame({
            'open': proxy_prices.values,
            'high': proxy_prices.values * 1.01,
            'low': proxy_prices.values * 0.99,
            'close': proxy_prices.values,
            'volume': np.ones(len(proxy_prices)) * 1000000,
        }, index=proxy_prices.index)

        regime = self.regime_detector.detect_regime(proxy_df)
        confidence = self.regime_detector.get_regime_confidence()
        return regime, confidence

    def get_momentum_signal(self, df: pd.DataFrame, symbol: str) -> float:
        """Calculate momentum signal for a stock."""
        if len(df) < 60:
            return 0.0

        close_col = 'Close' if 'Close' in df.columns else 'close'
        prices = df[close_col]

        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        fast_ma = prices.rolling(10).mean()
        slow_ma = prices.rolling(30).mean()
        ma_diff = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]

        momentum = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] if len(prices) > 20 else 0

        rsi_signal = 0
        if current_rsi < 30:
            rsi_signal = (30 - current_rsi) / 30
        elif current_rsi > 70:
            rsi_signal = (70 - current_rsi) / 30

        ma_signal = np.clip(ma_diff * 10, -1, 1)
        mom_signal = np.clip(momentum * 5, -1, 1)

        combined = 0.4 * rsi_signal + 0.4 * ma_signal + 0.2 * mom_signal
        return np.clip(combined, -1, 1)

    def get_hf_signal(self, data: Dict[str, pd.DataFrame], symbol: str, regime: RegimeType) -> float:
        """Calculate hedge fund factor signal for a stock."""
        if symbol not in data or len(data[symbol]) < 60:
            return 0.0

        df = data[symbol]
        close_col = 'Close' if 'Close' in df.columns else 'close'
        prices = df[close_col].values

        momentum = (prices[-5] - prices[-60]) / prices[-60] if len(prices) > 60 else 0
        short_mom = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 21 else 0
        value = -short_mom

        returns = pd.Series(prices).pct_change().dropna()
        if len(returns) > 20:
            pos_ratio = (returns > 0).sum() / len(returns)
            vol = returns.std() * np.sqrt(252)
            low_vol = 1 / (vol + 0.01)
        else:
            pos_ratio = 0.5
            low_vol = 1.0

        factors = {
            'momentum': np.clip(momentum * 5, -1, 1),
            'value': np.clip(value * 5, -1, 1),
            'quality': np.clip((pos_ratio - 0.5) * 4, -1, 1),
            'low_vol': np.clip((low_vol - 5) / 10, -1, 1),
        }

        if regime == RegimeType.BULL_TRENDING:
            signal = 0.50 * factors['momentum'] + 0.15 * factors['value'] + 0.20 * factors['quality'] + 0.15 * factors['low_vol']
        elif regime in [RegimeType.BEAR_CRISIS, RegimeType.HIGH_VOLATILITY]:
            signal = 0.20 * factors['momentum'] + 0.20 * factors['value'] + 0.30 * factors['quality'] + 0.30 * factors['low_vol']
        else:
            signal = 0.30 * factors['momentum'] + 0.20 * factors['value'] + 0.25 * factors['quality'] + 0.25 * factors['low_vol']

        return np.clip(signal, -1, 1)

    def generate_blended_signals(self, data: Dict[str, pd.DataFrame], regime: RegimeType, confidence: float) -> Dict[str, float]:
        """Generate blended signals for all symbols based on current regime."""
        weights = self.regime_weights.get(regime, {"momentum": 0.35, "adaptive_hf": 0.65})
        exposure = self.regime_exposure.get(regime, 0.70)

        signals = {}
        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < 60:
                signals[symbol] = 0.0
                continue

            mom_signal = self.get_momentum_signal(data[symbol], symbol)
            hf_signal = self.get_hf_signal(data, symbol, regime)
            blended = weights['momentum'] * mom_signal + weights['adaptive_hf'] * hf_signal
            blended *= exposure
            blended *= (0.5 + 0.5 * confidence)
            signals[symbol] = blended

        return signals

    def signals_to_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Convert signals to portfolio weights."""
        positive_signals = {k: max(0, v) for k, v in signals.items()}
        total_positive = sum(positive_signals.values())

        if total_positive > 0:
            weights = {k: v / total_positive for k, v in positive_signals.items()}
        else:
            weights = {k: 1 / len(signals) for k in signals}

        return weights


def run_regime_backtest_for_validation(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    capital: float,
    start_date: str,
    end_date: str,
) -> Tuple[pd.Series, pd.Series, List[Tuple[datetime, str]]]:
    """
    Run regime-aware backtest and return daily returns, portfolio values, and regime history.
    Simplified version for validation purposes.
    """
    strategy = RegimeAwareStrategy(symbols, capital)

    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))
    trading_dates = [d for d in common_dates if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]

    if len(trading_dates) < WARMUP_DAYS:
        return pd.Series(), pd.Series(), []

    trading_dates = trading_dates[WARMUP_DAYS:]

    cash = capital
    positions = {s: 0.0 for s in symbols}
    weights = {}

    portfolio_values = []
    regime_history = []

    for i, date in enumerate(trading_dates):
        hist_data = {s: df[df.index <= date].tail(120) for s, df in data.items()}
        prices = {s: data[s].loc[date, 'Close'] for s in symbols if date in data[s].index}

        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in symbols)
        current_value = cash + position_value

        regime, confidence = strategy.detect_regime(hist_data, date)
        regime_history.append((date, regime.value))

        if i == 0 or i % 5 == 0:
            signals = strategy.generate_blended_signals(hist_data, regime, confidence)
            new_weights = strategy.signals_to_weights(signals)

            for symbol in symbols:
                if symbol not in prices:
                    continue

                target_value = new_weights.get(symbol, 0) * current_value
                current_shares = positions.get(symbol, 0)
                current_pos_value = current_shares * prices[symbol]

                trade_value = target_value - current_pos_value
                trade_shares = trade_value / prices[symbol]

                if abs(trade_shares) > 0.01:
                    positions[symbol] = current_shares + trade_shares
                    cash -= trade_value
                    cash -= abs(trade_value) * 0.001

            weights = new_weights

        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in symbols)
        daily_value = cash + position_value
        portfolio_values.append({'date': date, 'value': daily_value})

    values = pd.Series([p['value'] for p in portfolio_values],
                       index=[p['date'] for p in portfolio_values])
    returns = values.pct_change().dropna()

    return returns, values, regime_history


# ============================================================================
# 1. Bootstrap Confidence Intervals
# ============================================================================

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * excess_returns.mean() / returns.std()


def calculate_total_return(returns: pd.Series) -> float:
    """Calculate total return from daily returns."""
    if len(returns) == 0:
        return 0.0
    return (np.prod(1 + returns) - 1) * 100


def bootstrap_confidence_intervals(
    returns: pd.Series,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for Sharpe ratio and total return.

    Bootstrap Explanation:
    - Resamples returns WITH REPLACEMENT to create synthetic datasets
    - Calculates statistic for each synthetic dataset
    - Uses percentiles of bootstrap distribution for confidence interval
    - Provides non-parametric estimate of sampling uncertainty
    """
    print("\n" + "=" * 80)
    print("1. BOOTSTRAP CONFIDENCE INTERVALS")
    print("=" * 80)
    print(f"\nMethod: Resample {len(returns)} daily returns with replacement")
    print(f"Iterations: {n_iterations}")
    print(f"Confidence Level: {confidence_level * 100}%")

    sharpe_samples = []
    return_samples = []

    returns_array = returns.values
    n_samples = len(returns_array)

    for i in range(n_iterations):
        # Resample with replacement
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_returns = pd.Series(returns_array[bootstrap_indices])

        sharpe_samples.append(calculate_sharpe_ratio(bootstrap_returns))
        return_samples.append(calculate_total_return(bootstrap_returns))

    sharpe_samples = np.array(sharpe_samples)
    return_samples = np.array(return_samples)

    alpha = 1 - confidence_level

    results = {
        'sharpe_ratio': {
            'point_estimate': calculate_sharpe_ratio(returns),
            'bootstrap_mean': np.mean(sharpe_samples),
            'bootstrap_std': np.std(sharpe_samples),
            'ci_lower': np.percentile(sharpe_samples, alpha / 2 * 100),
            'ci_upper': np.percentile(sharpe_samples, (1 - alpha / 2) * 100),
        },
        'total_return': {
            'point_estimate': calculate_total_return(returns),
            'bootstrap_mean': np.mean(return_samples),
            'bootstrap_std': np.std(return_samples),
            'ci_lower': np.percentile(return_samples, alpha / 2 * 100),
            'ci_upper': np.percentile(return_samples, (1 - alpha / 2) * 100),
        },
    }

    print(f"\nSharpe Ratio:")
    print(f"   Point Estimate:     {results['sharpe_ratio']['point_estimate']:.4f}")
    print(f"   Bootstrap Mean:     {results['sharpe_ratio']['bootstrap_mean']:.4f}")
    print(f"   Bootstrap Std:      {results['sharpe_ratio']['bootstrap_std']:.4f}")
    print(f"   95% CI:             [{results['sharpe_ratio']['ci_lower']:.4f}, {results['sharpe_ratio']['ci_upper']:.4f}]")

    print(f"\nTotal Return:")
    print(f"   Point Estimate:     {results['total_return']['point_estimate']:.2f}%")
    print(f"   Bootstrap Mean:     {results['total_return']['bootstrap_mean']:.2f}%")
    print(f"   Bootstrap Std:      {results['total_return']['bootstrap_std']:.2f}%")
    print(f"   95% CI:             [{results['total_return']['ci_lower']:.2f}%, {results['total_return']['ci_upper']:.2f}%]")

    # Interpretation
    print("\n   INTERPRETATION:")
    if results['sharpe_ratio']['ci_lower'] > 0:
        print("   - Sharpe ratio is STATISTICALLY SIGNIFICANT (CI does not include 0)")
        print("     The strategy has genuine risk-adjusted alpha.")
    else:
        print("   - Sharpe ratio is NOT statistically significant at 95% level")
        print("     The positive Sharpe could be due to chance.")

    ci_width_sharpe = results['sharpe_ratio']['ci_upper'] - results['sharpe_ratio']['ci_lower']
    print(f"   - CI Width (Sharpe): {ci_width_sharpe:.4f}")
    if ci_width_sharpe < 0.5:
        print("     Relatively tight interval indicates stable performance estimate.")
    else:
        print("     Wide interval suggests high uncertainty in performance estimate.")

    return results


# ============================================================================
# 2. Regime Transition Analysis
# ============================================================================

def analyze_regime_transitions(
    regime_history: List[Tuple[datetime, str]],
    returns: pd.Series,
) -> Dict[str, Any]:
    """
    Analyze regime transitions and their impact on performance.

    Regime Transition Analysis helps answer:
    - How often does the strategy switch regimes?
    - What is the average holding period per regime?
    - Do frequent switches (whipsaws) hurt performance?
    """
    print("\n" + "=" * 80)
    print("2. REGIME TRANSITION ANALYSIS")
    print("=" * 80)

    if not regime_history:
        print("No regime history available.")
        return {}

    # Extract regime periods
    regime_periods = []
    current_regime = regime_history[0][1]
    period_start = regime_history[0][0]

    for date, regime in regime_history[1:]:
        if regime != current_regime:
            regime_periods.append({
                'regime': current_regime,
                'start': period_start,
                'end': date,
                'days': (date - period_start).days,
            })
            current_regime = regime
            period_start = date

    # Add final period
    regime_periods.append({
        'regime': current_regime,
        'start': period_start,
        'end': regime_history[-1][0],
        'days': (regime_history[-1][0] - period_start).days,
    })

    # Calculate statistics
    total_days = len(regime_history)
    num_transitions = len(regime_periods) - 1
    transition_frequency = num_transitions / total_days if total_days > 0 else 0

    # Average holding period per regime
    regime_holding_periods = {}
    for regime in set(p['regime'] for p in regime_periods):
        periods = [p for p in regime_periods if p['regime'] == regime]
        durations = [p['days'] for p in periods if p['days'] > 0]
        if durations:
            regime_holding_periods[regime] = {
                'avg_days': np.mean(durations),
                'min_days': np.min(durations),
                'max_days': np.max(durations),
                'count': len(durations),
            }

    print(f"\nTotal trading days: {total_days}")
    print(f"Number of regime transitions: {num_transitions}")
    print(f"Average transitions per month: {transition_frequency * 21:.2f}")
    print(f"Average time between transitions: {total_days / (num_transitions + 1):.1f} days")

    print(f"\nHolding Period by Regime:")
    print(f"{'Regime':<20} {'Count':>8} {'Avg Days':>10} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    for regime, stats in regime_holding_periods.items():
        print(f"{regime:<20} {stats['count']:>8} {stats['avg_days']:>10.1f} {stats['min_days']:>8} {stats['max_days']:>8}")

    # Analyze performance around transitions
    # Calculate returns in windows around transitions
    transition_dates = [regime_periods[i]['end'] for i in range(len(regime_periods) - 1)]

    pre_transition_returns = []
    post_transition_returns = []

    for trans_date in transition_dates:
        # Get returns 5 days before and after transition
        if trans_date in returns.index:
            idx = returns.index.get_loc(trans_date)
            if idx >= 5 and idx < len(returns) - 5:
                pre_ret = returns.iloc[idx-5:idx].sum()
                post_ret = returns.iloc[idx:idx+5].sum()
                pre_transition_returns.append(pre_ret)
                post_transition_returns.append(post_ret)

    # Analyze whipsaw effect
    short_periods = [p for p in regime_periods if p['days'] < 5]
    whipsaw_rate = len(short_periods) / len(regime_periods) if regime_periods else 0

    print(f"\nWhipsaw Analysis:")
    print(f"   Periods < 5 days (potential whipsaws): {len(short_periods)}")
    print(f"   Whipsaw rate: {whipsaw_rate * 100:.1f}%")

    if pre_transition_returns:
        print(f"\nPerformance Around Transitions:")
        print(f"   Avg 5-day return BEFORE transition: {np.mean(pre_transition_returns) * 100:.2f}%")
        print(f"   Avg 5-day return AFTER transition:  {np.mean(post_transition_returns) * 100:.2f}%")

        # Does transition improve performance?
        improvement = np.mean(post_transition_returns) - np.mean(pre_transition_returns)
        print(f"\n   INTERPRETATION:")
        if improvement > 0:
            print(f"   - Transitions tend to IMPROVE performance (+{improvement * 100:.2f}%)")
            print("     The regime detection is adding value by adapting to market conditions.")
        else:
            print(f"   - Transitions tend to HURT performance ({improvement * 100:.2f}%)")
            print("     Consider increasing regime stability thresholds to reduce whipsaws.")

    if whipsaw_rate > 0.3:
        print("   - HIGH whipsaw rate detected. Transaction costs may erode returns.")
        print("     Consider increasing minimum regime duration thresholds.")

    return {
        'total_days': total_days,
        'num_transitions': num_transitions,
        'transition_frequency_per_day': transition_frequency,
        'avg_days_between_transitions': total_days / (num_transitions + 1) if num_transitions > 0 else total_days,
        'regime_holding_periods': regime_holding_periods,
        'whipsaw_rate': whipsaw_rate,
        'short_period_count': len(short_periods),
        'avg_pre_transition_return': np.mean(pre_transition_returns) * 100 if pre_transition_returns else 0,
        'avg_post_transition_return': np.mean(post_transition_returns) * 100 if post_transition_returns else 0,
    }


# ============================================================================
# 3. Drawdown Analysis
# ============================================================================

def analyze_drawdowns(
    returns: pd.Series,
    portfolio_values: pd.Series,
) -> Dict[str, Any]:
    """
    Comprehensive drawdown analysis including:
    - Time to recovery from max drawdown
    - Drawdown duration distribution
    - Underwater curve analysis

    Drawdown Metrics Explanation:
    - Max Drawdown: Largest peak-to-trough decline
    - Time to Recovery: Days to reach new high after drawdown
    - Underwater Period: Consecutive days below previous peak
    """
    print("\n" + "=" * 80)
    print("3. DRAWDOWN ANALYSIS")
    print("=" * 80)

    # Calculate cumulative returns and drawdowns
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    underwater = drawdown < 0

    # Max drawdown
    max_dd = abs(drawdown.min()) * 100
    max_dd_idx = drawdown.idxmin()

    # Find peak before max drawdown
    peak_idx = running_max[:max_dd_idx].idxmax()
    peak_value = cumulative.loc[peak_idx]
    trough_value = cumulative.loc[max_dd_idx]

    # Find recovery point
    post_trough = cumulative[max_dd_idx:]
    recovery_mask = post_trough >= peak_value

    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
        recovery_days = (recovery_idx - max_dd_idx).days
        fully_recovered = True
    else:
        recovery_days = (cumulative.index[-1] - max_dd_idx).days
        fully_recovered = False

    print(f"\nMaximum Drawdown Analysis:")
    print(f"   Max Drawdown:       {max_dd:.2f}%")
    print(f"   Peak Date:          {peak_idx.strftime('%Y-%m-%d')}")
    print(f"   Trough Date:        {max_dd_idx.strftime('%Y-%m-%d')}")
    print(f"   Days to Trough:     {(max_dd_idx - peak_idx).days}")
    if fully_recovered:
        print(f"   Recovery Date:      {recovery_idx.strftime('%Y-%m-%d')}")
        print(f"   Time to Recovery:   {recovery_days} days")
    else:
        print(f"   NOT YET RECOVERED after {recovery_days} days")

    # Identify all drawdown periods
    drawdown_periods = []
    in_drawdown = False
    dd_start = None
    dd_peak_value = None

    for i, (date, dd) in enumerate(drawdown.items()):
        if dd < 0 and not in_drawdown:
            # Start of new drawdown
            in_drawdown = True
            dd_start = date
            dd_peak_value = cumulative.iloc[i-1] if i > 0 else cumulative.iloc[0]
        elif dd >= 0 and in_drawdown:
            # End of drawdown (recovered)
            in_drawdown = False
            dd_end = date
            dd_min = drawdown[dd_start:dd_end].min()
            drawdown_periods.append({
                'start': dd_start,
                'end': dd_end,
                'duration': (dd_end - dd_start).days,
                'depth': abs(dd_min) * 100,
                'recovered': True,
            })

    # Handle ongoing drawdown
    if in_drawdown:
        drawdown_periods.append({
            'start': dd_start,
            'end': cumulative.index[-1],
            'duration': (cumulative.index[-1] - dd_start).days,
            'depth': abs(drawdown[dd_start:].min()) * 100,
            'recovered': False,
        })

    # Drawdown duration distribution
    durations = [p['duration'] for p in drawdown_periods]
    depths = [p['depth'] for p in drawdown_periods]

    print(f"\nDrawdown Duration Distribution:")
    print(f"   Total drawdown periods: {len(drawdown_periods)}")

    if durations:
        print(f"   Average duration:       {np.mean(durations):.1f} days")
        print(f"   Median duration:        {np.median(durations):.1f} days")
        print(f"   Max duration:           {np.max(durations)} days")
        print(f"   Min duration:           {np.min(durations)} days")

        # Percentiles
        print(f"\n   Duration Percentiles:")
        for pct in [25, 50, 75, 90, 95]:
            print(f"      {pct}th percentile: {np.percentile(durations, pct):.0f} days")

    print(f"\nDrawdown Depth Distribution:")
    if depths:
        print(f"   Average depth:          {np.mean(depths):.2f}%")
        print(f"   Median depth:           {np.median(depths):.2f}%")
        print(f"   Worst (non-max):        {sorted(depths, reverse=True)[1]:.2f}%" if len(depths) > 1 else "N/A")

    # Underwater curve analysis
    underwater_days = underwater.sum()
    total_days = len(underwater)
    underwater_pct = underwater_days / total_days * 100

    print(f"\nUnderwater Curve Analysis:")
    print(f"   Days underwater:        {underwater_days} / {total_days}")
    print(f"   Percentage underwater:  {underwater_pct:.1f}%")

    # Average drawdown when underwater
    avg_dd_when_underwater = abs(drawdown[underwater].mean()) * 100
    print(f"   Avg drawdown (underwater): {avg_dd_when_underwater:.2f}%")

    # Interpretation
    print(f"\n   INTERPRETATION:")
    if max_dd < 20:
        print(f"   - Max drawdown of {max_dd:.1f}% is MODERATE (< 20%)")
        print("     Risk management is effective.")
    elif max_dd < 40:
        print(f"   - Max drawdown of {max_dd:.1f}% is SIGNIFICANT (20-40%)")
        print("     Strategy experiences substantial declines during stress periods.")
    else:
        print(f"   - Max drawdown of {max_dd:.1f}% is SEVERE (> 40%)")
        print("     Consider adding drawdown controls or reducing position sizes.")

    if underwater_pct > 50:
        print(f"   - Strategy is underwater {underwater_pct:.0f}% of the time")
        print("     Long recovery periods may test investor patience.")

    return {
        'max_drawdown': max_dd,
        'max_dd_peak_date': str(peak_idx),
        'max_dd_trough_date': str(max_dd_idx),
        'days_to_trough': (max_dd_idx - peak_idx).days,
        'time_to_recovery': recovery_days,
        'fully_recovered': fully_recovered,
        'total_drawdown_periods': len(drawdown_periods),
        'avg_drawdown_duration': np.mean(durations) if durations else 0,
        'median_drawdown_duration': np.median(durations) if durations else 0,
        'max_drawdown_duration': np.max(durations) if durations else 0,
        'avg_drawdown_depth': np.mean(depths) if depths else 0,
        'underwater_days': int(underwater_days),
        'underwater_percentage': underwater_pct,
        'avg_drawdown_when_underwater': avg_dd_when_underwater,
    }


# ============================================================================
# 4. Risk-Adjusted Metrics
# ============================================================================

def calculate_risk_adjusted_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.05,
) -> Dict[str, float]:
    """
    Calculate comprehensive risk-adjusted performance metrics.

    Metrics Explanation:
    - Calmar Ratio: Annual return / Max drawdown. Higher is better. >1 is good.
    - Sortino Ratio: Return / Downside deviation. Like Sharpe but only penalizes downside.
    - Omega Ratio: Probability-weighted gains / losses. >1 means positive expected value.
    """
    print("\n" + "=" * 80)
    print("4. RISK-ADJUSTED METRICS")
    print("=" * 80)

    if len(returns) == 0:
        return {}

    # Basic statistics
    total_return_pct = calculate_total_return(returns)
    annualized_return = ((1 + total_return_pct / 100) ** (252 / len(returns)) - 1) * 100
    volatility = returns.std() * np.sqrt(252) * 100

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_dd = abs(drawdown.min()) * 100

    # Calmar Ratio
    # Interpretation: Return per unit of max drawdown risk
    # > 1.0 is good, > 2.0 is excellent
    calmar = annualized_return / max_dd if max_dd > 0 else 0

    # Sortino Ratio
    # Uses downside deviation instead of total volatility
    # Doesn't penalize upside volatility
    excess_returns = returns - risk_free_rate / 252
    downside_returns = returns[returns < 0]
    downside_std = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 1e-6
    sortino = np.sqrt(252) * excess_returns.mean() / downside_std

    # Omega Ratio
    # Probability-weighted ratio of gains to losses
    # Omega = (integral of returns above threshold) / (integral of returns below threshold)
    threshold = risk_free_rate / 252  # Daily risk-free rate
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]

    sum_gains = gains.sum() if len(gains) > 0 else 0
    sum_losses = losses.sum() if len(losses) > 0 else 1e-6
    omega = (1 + sum_gains / sum_losses) if sum_losses > 0 else float('inf')

    # Additional metrics
    # Sharpe Ratio
    sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0

    # Information Ratio (vs risk-free - simplified)
    tracking_error = returns.std() * np.sqrt(252)
    info_ratio = (annualized_return - risk_free_rate * 100) / tracking_error if tracking_error > 0 else 0

    # Win rate
    win_rate = (returns > 0).sum() / len(returns) * 100

    # Profit factor
    gross_profits = returns[returns > 0].sum()
    gross_losses = abs(returns[returns < 0].sum())
    profit_factor = gross_profits / gross_losses if gross_losses > 0 else float('inf')

    # Average win/loss
    avg_win = returns[returns > 0].mean() * 100 if (returns > 0).sum() > 0 else 0
    avg_loss = returns[returns < 0].mean() * 100 if (returns < 0).sum() > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    print(f"\nPrimary Risk-Adjusted Ratios:")
    print(f"{'Metric':<25} {'Value':>12} {'Interpretation':<35}")
    print("-" * 75)

    # Calmar
    calmar_interp = "Excellent" if calmar > 2 else "Good" if calmar > 1 else "Fair" if calmar > 0.5 else "Poor"
    print(f"{'Calmar Ratio':<25} {calmar:>12.2f} {calmar_interp:<35}")

    # Sortino
    sortino_interp = "Excellent" if sortino > 2 else "Good" if sortino > 1 else "Fair" if sortino > 0.5 else "Poor"
    print(f"{'Sortino Ratio':<25} {sortino:>12.2f} {sortino_interp:<35}")

    # Omega
    omega_interp = "Excellent" if omega > 2 else "Good" if omega > 1.5 else "Positive EV" if omega > 1 else "Negative EV"
    print(f"{'Omega Ratio':<25} {omega:>12.2f} {omega_interp:<35}")

    # Sharpe
    sharpe_interp = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Fair" if sharpe > 0.5 else "Poor"
    print(f"{'Sharpe Ratio':<25} {sharpe:>12.2f} {sharpe_interp:<35}")

    print(f"\nSecondary Metrics:")
    print(f"{'Metric':<25} {'Value':>12}")
    print("-" * 40)
    print(f"{'Annualized Return':<25} {annualized_return:>11.2f}%")
    print(f"{'Annualized Volatility':<25} {volatility:>11.2f}%")
    print(f"{'Max Drawdown':<25} {max_dd:>11.2f}%")
    print(f"{'Win Rate':<25} {win_rate:>11.2f}%")
    print(f"{'Profit Factor':<25} {profit_factor:>12.2f}")
    print(f"{'Avg Win':<25} {avg_win:>11.3f}%")
    print(f"{'Avg Loss':<25} {avg_loss:>11.3f}%")
    print(f"{'Win/Loss Ratio':<25} {win_loss_ratio:>12.2f}")

    # Overall interpretation
    print(f"\n   INTERPRETATION:")
    print(f"   - Calmar Ratio ({calmar:.2f}): ", end="")
    print("Strategy generates good returns relative to max drawdown risk." if calmar > 1 else
          "Returns may not adequately compensate for drawdown risk.")

    print(f"   - Sortino Ratio ({sortino:.2f}): ", end="")
    print("Good downside risk management - not overly penalized for upside volatility." if sortino > sharpe else
          "Downside risk is a concern relative to total volatility.")

    print(f"   - Omega Ratio ({omega:.2f}): ", end="")
    if omega > 1:
        print(f"Positive expected value - gains outweigh losses by {(omega-1)*100:.0f}%.")
    else:
        print("Negative expected value - losses outweigh gains.")

    return {
        'calmar_ratio': calmar,
        'sortino_ratio': sortino,
        'omega_ratio': omega,
        'sharpe_ratio': sharpe,
        'information_ratio': info_ratio,
        'annualized_return': annualized_return,
        'annualized_volatility': volatility,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'win_loss_ratio': win_loss_ratio,
    }


# ============================================================================
# 5. Correlation Analysis
# ============================================================================

def analyze_correlation_with_spy(
    strategy_returns: pd.Series,
    spy_data: pd.DataFrame,
    regime_history: List[Tuple[datetime, str]],
    rolling_window: int = 90,
) -> Dict[str, Any]:
    """
    Analyze correlation between strategy returns and SPY benchmark.

    Correlation Analysis Explanation:
    - Low correlation with SPY indicates diversification benefit
    - Rolling correlation shows if relationship changes over time
    - Regime-specific correlation reveals when strategy decouples from market
    """
    print("\n" + "=" * 80)
    print("5. CORRELATION ANALYSIS WITH SPY")
    print("=" * 80)

    # Calculate SPY returns
    # Handle MultiIndex columns from yfinance
    if isinstance(spy_data.columns, pd.MultiIndex):
        spy_close = spy_data['Close']['SPY'] if ('Close', 'SPY') in spy_data.columns else spy_data.iloc[:, 0]
    else:
        spy_close = spy_data['Close'] if 'Close' in spy_data.columns else spy_data['Adj Close']

    spy_returns = spy_close.pct_change().dropna()
    spy_returns.index = pd.to_datetime(spy_returns.index)

    # Ensure strategy returns index is datetime
    strategy_returns.index = pd.to_datetime(strategy_returns.index)

    # Align dates - normalize to date only (no time component)
    spy_returns.index = spy_returns.index.normalize()
    strategy_returns_aligned = strategy_returns.copy()
    strategy_returns_aligned.index = strategy_returns_aligned.index.normalize()

    common_dates = strategy_returns_aligned.index.intersection(spy_returns.index)
    strat_aligned = strategy_returns_aligned.loc[common_dates]
    spy_aligned = spy_returns.loc[common_dates]

    # Ensure both are 1D series
    strat_aligned = strat_aligned.squeeze()
    spy_aligned = spy_aligned.squeeze()

    if len(common_dates) == 0:
        print("No overlapping dates between strategy and SPY data.")
        return {}

    # Overall correlation
    overall_corr = strat_aligned.corr(spy_aligned)

    print(f"\nOverall Correlation with SPY:")
    print(f"   Correlation coefficient: {overall_corr:.4f}")
    print(f"   R-squared:               {overall_corr ** 2:.4f}")

    # Interpretation of overall correlation
    if abs(overall_corr) < 0.3:
        corr_interp = "LOW - Strong diversification benefit"
    elif abs(overall_corr) < 0.6:
        corr_interp = "MODERATE - Some diversification benefit"
    elif abs(overall_corr) < 0.8:
        corr_interp = "HIGH - Limited diversification"
    else:
        corr_interp = "VERY HIGH - Strategy moves with market"

    print(f"   Interpretation:          {corr_interp}")

    # Rolling correlation (90-day window)
    if len(common_dates) >= rolling_window:
        combined = pd.DataFrame({
            'strategy': strat_aligned,
            'spy': spy_aligned
        })
        rolling_corr = combined['strategy'].rolling(window=rolling_window).corr(combined['spy'])

        print(f"\nRolling {rolling_window}-Day Correlation:")
        print(f"   Mean:      {rolling_corr.mean():.4f}")
        print(f"   Std Dev:   {rolling_corr.std():.4f}")
        print(f"   Min:       {rolling_corr.min():.4f}")
        print(f"   Max:       {rolling_corr.max():.4f}")

        # Periods of low correlation
        low_corr_periods = (rolling_corr < 0.3).sum()
        low_corr_pct = low_corr_periods / len(rolling_corr) * 100
        print(f"\n   Days with correlation < 0.3: {low_corr_periods} ({low_corr_pct:.1f}%)")
    else:
        rolling_corr = pd.Series()
        print(f"\nInsufficient data for {rolling_window}-day rolling correlation")

    # Correlation by regime
    print(f"\nCorrelation by Market Regime:")
    print(f"{'Regime':<20} {'Correlation':>12} {'Days':>8}")
    print("-" * 45)

    regime_correlations = {}
    regime_df = pd.DataFrame(regime_history, columns=['date', 'regime'])
    regime_df['date'] = pd.to_datetime(regime_df['date']).dt.normalize()
    regime_df = regime_df.set_index('date')

    for regime in regime_df['regime'].unique():
        regime_dates = regime_df[regime_df['regime'] == regime].index
        regime_dates_aligned = [d for d in regime_dates if d in common_dates]

        if len(regime_dates_aligned) > 10:
            strat_regime = strat_aligned.loc[regime_dates_aligned]
            spy_regime = spy_aligned.loc[regime_dates_aligned]
            regime_corr = strat_regime.corr(spy_regime)
            regime_correlations[regime] = {
                'correlation': regime_corr,
                'days': len(regime_dates_aligned),
            }
            print(f"{regime:<20} {regime_corr:>12.4f} {len(regime_dates_aligned):>8}")

    # Beta calculation
    covariance = strat_aligned.cov(spy_aligned)
    spy_variance = spy_aligned.var()
    beta = covariance / spy_variance if spy_variance > 0 else 0

    # Alpha (Jensen's alpha)
    rf_daily = 0.05 / 252
    avg_strat_ret = strat_aligned.mean()
    avg_spy_ret = spy_aligned.mean()
    alpha_daily = avg_strat_ret - rf_daily - beta * (avg_spy_ret - rf_daily)
    alpha_annual = alpha_daily * 252 * 100  # Annualized percentage

    print(f"\nBeta and Alpha Analysis:")
    print(f"   Beta:                    {beta:.4f}")
    print(f"   Jensen's Alpha (annual): {alpha_annual:.2f}%")

    # Interpretation
    print(f"\n   INTERPRETATION:")
    if beta < 0.5:
        print(f"   - Low beta ({beta:.2f}): Strategy is defensive, less sensitive to market moves")
    elif beta < 1.0:
        print(f"   - Moderate beta ({beta:.2f}): Less volatile than market")
    elif beta < 1.5:
        print(f"   - Beta near 1.0 ({beta:.2f}): Moves roughly with the market")
    else:
        print(f"   - High beta ({beta:.2f}): More volatile than market")

    if alpha_annual > 0:
        print(f"   - Positive alpha ({alpha_annual:.2f}%): Strategy generates excess returns")
        print("     above what would be expected given its market exposure.")
    else:
        print(f"   - Negative alpha ({alpha_annual:.2f}%): Strategy underperforms")
        print("     its expected return given market exposure.")

    # Downside correlation (during negative SPY days)
    spy_down_days = spy_aligned[spy_aligned < 0].index
    if len(spy_down_days) > 20:
        strat_on_down_days = strat_aligned.loc[spy_down_days]
        spy_on_down_days = spy_aligned.loc[spy_down_days]
        downside_corr = strat_on_down_days.corr(spy_on_down_days)

        print(f"\n   Downside Correlation (SPY down days):")
        print(f"   - Correlation: {downside_corr:.4f}")

        if downside_corr < overall_corr:
            print("   - Lower correlation during market declines - good defensive characteristics")
        else:
            print("   - Similar or higher correlation during declines - limited downside protection")

    return {
        'overall_correlation': overall_corr,
        'r_squared': overall_corr ** 2,
        'rolling_correlation_mean': rolling_corr.mean() if len(rolling_corr) > 0 else None,
        'rolling_correlation_std': rolling_corr.std() if len(rolling_corr) > 0 else None,
        'rolling_correlation_min': rolling_corr.min() if len(rolling_corr) > 0 else None,
        'rolling_correlation_max': rolling_corr.max() if len(rolling_corr) > 0 else None,
        'regime_correlations': regime_correlations,
        'beta': beta,
        'alpha_annual_pct': alpha_annual,
    }


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 80)
    print("STATISTICAL VALIDATION OF REGIME-BLEND STRATEGY")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"   Symbols:              {', '.join(SYMBOLS)}")
    print(f"   Period:               {START_DATE} to {END_DATE}")
    print(f"   Initial Capital:      ${INITIAL_CAPITAL:,}")
    print(f"   Bootstrap Iterations: {BOOTSTRAP_ITERATIONS}")
    print(f"   Confidence Level:     {CONFIDENCE_LEVEL * 100}%")

    # Fetch data
    print("\nLoading data...")

    print("   Fetching strategy portfolio data...")
    data = fetch_strategy_data(SYMBOLS, START_DATE, END_DATE)
    print(f"   Loaded {len(data)} stocks")

    spy_data = fetch_spy_benchmark(START_DATE, END_DATE)
    print(f"   Loaded SPY benchmark ({len(spy_data)} days)")

    # Run backtest to get returns
    print("\nRunning regime-blend backtest for validation...")
    returns, portfolio_values, regime_history = run_regime_backtest_for_validation(
        data, SYMBOLS, INITIAL_CAPITAL, START_DATE, END_DATE
    )

    print(f"   Generated {len(returns)} daily returns")
    print(f"   Regime history: {len(regime_history)} observations")

    # Store all results
    all_results = {
        'configuration': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'initial_capital': INITIAL_CAPITAL,
            'bootstrap_iterations': BOOTSTRAP_ITERATIONS,
            'confidence_level': CONFIDENCE_LEVEL,
            'trading_days': len(returns),
        }
    }

    # 1. Bootstrap Confidence Intervals
    bootstrap_results = bootstrap_confidence_intervals(
        returns,
        n_iterations=BOOTSTRAP_ITERATIONS,
        confidence_level=CONFIDENCE_LEVEL,
    )
    all_results['bootstrap_confidence_intervals'] = bootstrap_results

    # 2. Regime Transition Analysis
    transition_results = analyze_regime_transitions(regime_history, returns)
    all_results['regime_transition_analysis'] = transition_results

    # 3. Drawdown Analysis
    drawdown_results = analyze_drawdowns(returns, portfolio_values)
    all_results['drawdown_analysis'] = drawdown_results

    # 4. Risk-Adjusted Metrics
    risk_adjusted_results = calculate_risk_adjusted_metrics(returns)
    all_results['risk_adjusted_metrics'] = risk_adjusted_results

    # 5. Correlation Analysis
    correlation_results = analyze_correlation_with_spy(
        returns, spy_data, regime_history, rolling_window=90
    )
    all_results['correlation_analysis'] = correlation_results

    # Save results to JSON
    output_path = OUTPUT_DIR / 'statistical_validation.json'

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    all_results = convert_numpy(all_results)

    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"\nOutput file: {output_path}")

    # Summary
    print(f"\n{'=' * 80}")
    print("EXECUTIVE SUMMARY")
    print("=" * 80)

    print(f"\n1. STATISTICAL SIGNIFICANCE:")
    sharpe_ci = bootstrap_results['sharpe_ratio']
    if sharpe_ci['ci_lower'] > 0:
        print(f"   - Sharpe ratio is STATISTICALLY SIGNIFICANT")
        print(f"   - 95% CI: [{sharpe_ci['ci_lower']:.3f}, {sharpe_ci['ci_upper']:.3f}]")
    else:
        print(f"   - Sharpe ratio is NOT statistically significant at 95% level")

    print(f"\n2. REGIME SWITCHING:")
    if transition_results:
        print(f"   - {transition_results['num_transitions']} regime transitions")
        print(f"   - Avg {transition_results['avg_days_between_transitions']:.0f} days between transitions")
        print(f"   - Whipsaw rate: {transition_results['whipsaw_rate']*100:.1f}%")

    print(f"\n3. DRAWDOWN PROFILE:")
    print(f"   - Max drawdown: {drawdown_results['max_drawdown']:.2f}%")
    print(f"   - Time to recovery: {drawdown_results['time_to_recovery']} days")
    print(f"   - Underwater {drawdown_results['underwater_percentage']:.1f}% of time")

    print(f"\n4. RISK-ADJUSTED PERFORMANCE:")
    print(f"   - Calmar Ratio:  {risk_adjusted_results['calmar_ratio']:.2f}")
    print(f"   - Sortino Ratio: {risk_adjusted_results['sortino_ratio']:.2f}")
    print(f"   - Omega Ratio:   {risk_adjusted_results['omega_ratio']:.2f}")

    print(f"\n5. MARKET CORRELATION:")
    if correlation_results:
        print(f"   - SPY correlation: {correlation_results['overall_correlation']:.3f}")
        print(f"   - Beta: {correlation_results['beta']:.3f}")
        print(f"   - Alpha (annual): {correlation_results['alpha_annual_pct']:.2f}%")

    print(f"\n{'=' * 80}")
    print("VALIDATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

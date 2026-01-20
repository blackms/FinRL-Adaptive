#!/usr/bin/env python3
"""
Regime-Blend Strategy Walk-Forward Validation

Implements comprehensive walk-forward validation for the regime-aware blended strategy:
- Train window: 12 months
- Test window: 3 months
- Rolling forward through 2020-2024
- Parameters to optimize: regime thresholds, blend weights

This script validates the robustness of the regime detection and strategy blending
by measuring in-sample vs out-of-sample performance degradation.

Usage:
    python scripts/regime_blend_walkforward.py
"""

import sys
sys.path.insert(0, '/opt/FinRL/src')

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Try to import yfinance
try:
    import yfinance as yf
except ImportError:
    print("ERROR: yfinance not installed. Install with: pip install yfinance")
    sys.exit(1)

from trading.strategies.regime_detector import (
    RegimeDetector,
    RegimeDetectorConfig,
    RegimeType,
)
from trading.strategies.strategy_blender import (
    StrategyBlender,
    StrategyBlenderConfig,
    DEFAULT_REGIME_WEIGHTS,
)


# ============================================================================
# Configuration
# ============================================================================

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INITIAL_CAPITAL = 100000
TRAIN_MONTHS = 12  # 12-month training window
TEST_MONTHS = 3    # 3-month test window
WARMUP_DAYS = 60  # Days needed for indicator calculation (reduced for better windows)

# Walk-forward parameters
START_YEAR = 2020
END_YEAR = 2024

OUTPUT_DIR = Path("/opt/FinRL/output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Parameter Space for Optimization
# ============================================================================

@dataclass
class RegimeParameters:
    """Parameters to optimize for regime detection."""
    volatility_high_percentile: float = 90.0
    strong_trend_threshold: float = 0.6
    min_hold_days: int = 5


@dataclass
class BlendWeights:
    """Strategy blend weights per regime."""
    bull_momentum: float = 0.65
    bull_hf: float = 0.35
    bear_momentum: float = 0.10
    bear_hf: float = 0.90
    sideways_momentum: float = 0.30
    sideways_hf: float = 0.70
    volatile_momentum: float = 0.15
    volatile_hf: float = 0.85


# Parameter grid for optimization (reduced for faster execution)
PARAM_GRID = {
    'volatility_high_percentile': [85.0, 92.0],
    'strong_trend_threshold': [0.55, 0.65],
    'bull_momentum_weight': [0.60, 0.70],
    'bear_momentum_weight': [0.08, 0.12],
}


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_data_yfinance(
    symbols: List[str],
    start_date: str,
    end_date: str
) -> Dict[str, pd.DataFrame]:
    """
    Fetch stock data using yfinance.

    Args:
        symbols: List of ticker symbols.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.

    Returns:
        Dictionary mapping symbols to DataFrames with OHLCV data.
    """
    print(f"   Fetching data from yfinance for {len(symbols)} symbols...")

    # Download all at once for efficiency
    raw_data = yf.download(
        symbols,
        start=start_date,
        end=end_date,
        group_by='ticker',
        progress=False,
        auto_adjust=True,
    )

    data = {}

    for symbol in symbols:
        try:
            if len(symbols) == 1:
                # Single symbol case - different structure
                df = raw_data.copy()
            else:
                # Multiple symbols
                df = raw_data[symbol].copy()

            # Standardize column names
            df.columns = [col.lower() if isinstance(col, str) else col[0].lower() for col in df.columns]

            # Ensure we have required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if all(col in df.columns for col in required):
                # Drop any rows with NaN in key columns
                df = df.dropna(subset=['close'])

                # Ensure index is datetime
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)

                # Remove timezone if present
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)

                data[symbol] = df
                print(f"      {symbol}: {len(df)} days loaded")
            else:
                print(f"      {symbol}: Missing required columns, skipping")

        except Exception as e:
            print(f"      {symbol}: Error - {str(e)}")

    return data


# ============================================================================
# Walk-Forward Engine
# ============================================================================

@dataclass
class WindowResult:
    """Results for a single walk-forward window."""
    window_id: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # In-sample metrics
    is_sharpe: float
    is_return: float
    is_volatility: float
    is_max_drawdown: float

    # Out-of-sample metrics
    oos_sharpe: float
    oos_return: float
    oos_volatility: float
    oos_max_drawdown: float

    # Regime metrics
    regime_accuracy: float  # Did predicted regime match actual market behavior?
    regime_distribution: Dict[str, float]  # % time in each regime
    regime_changes: int

    # Strategy allocation effectiveness
    allocation_effectiveness: float

    # Optimal parameters found
    optimal_params: Dict[str, Any]

    # Degradation
    sharpe_degradation: float
    return_degradation: float


class RegimeAwareWalkForward:
    """
    Walk-forward validation engine for regime-aware strategy.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100000,
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.results: List[WindowResult] = []

    def generate_windows(
        self,
        start_year: int,
        end_year: int,
        train_months: int,
        test_months: int,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Generate walk-forward windows.

        Returns list of (train_start, train_end, test_start, test_end) tuples.
        """
        windows = []

        # Start from the beginning of start_year
        current_date = pd.Timestamp(f"{start_year}-01-01")
        end_date = pd.Timestamp(f"{end_year}-12-31")

        while True:
            train_start = current_date
            train_end = train_start + pd.DateOffset(months=train_months) - pd.DateOffset(days=1)
            test_start = train_end + pd.DateOffset(days=1)
            test_end = test_start + pd.DateOffset(months=test_months) - pd.DateOffset(days=1)

            if test_end > end_date:
                break

            windows.append((train_start, train_end, test_start, test_end))

            # Roll forward by test_months
            current_date = test_start

        return windows

    def run_backtest_period(
        self,
        data: Dict[str, pd.DataFrame],
        start_date: pd.Timestamp,
        end_date: pd.Timestamp,
        regime_config: RegimeDetectorConfig,
        blend_weights: Dict[RegimeType, Dict[str, float]],
    ) -> Dict[str, Any]:
        """
        Run backtest for a specific period with given parameters.
        """
        # Include warmup data BEFORE the start date
        warmup_start = start_date - pd.DateOffset(days=WARMUP_DAYS + 30)

        # Filter data to include warmup + trading period
        period_data = {}
        for symbol, df in data.items():
            # Get data from warmup start through end date
            mask = (df.index >= warmup_start) & (df.index <= end_date)
            period_df = df[mask].copy()
            if len(period_df) > WARMUP_DAYS:
                period_data[symbol] = period_df

        if len(period_data) < 3:  # Need at least 3 stocks
            return {'error': 'Insufficient data'}

        # Initialize regime detector
        detector = RegimeDetector(regime_config)

        # Get common trading dates within the actual trading period (not warmup)
        common_dates = sorted(set.intersection(*[set(df.index) for df in period_data.values()]))

        if len(common_dates) < 20:
            return {'error': 'Insufficient trading dates'}

        # Filter to only dates within the actual trading window
        trading_dates = [d for d in common_dates if d >= start_date and d <= end_date]

        if len(trading_dates) < 20:
            return {'error': 'Insufficient trading dates in window'}

        # Initialize tracking
        cash = self.initial_capital
        positions = {s: 0.0 for s in period_data.keys()}

        portfolio_values = []
        daily_returns = []
        regimes_detected = []
        regime_changes = 0
        last_regime = None

        # Track regime accuracy
        regime_predictions = []
        actual_market_behavior = []

        for i, date in enumerate(trading_dates):
            # Get historical data up to this date
            hist_data = {}
            for s, df in period_data.items():
                hist_df = df[df.index <= date].tail(120)
                if len(hist_df) >= 60:
                    hist_data[s] = hist_df

            if len(hist_data) < 2:
                continue

            # Get current prices
            prices = {}
            for s in hist_data.keys():
                if date in period_data[s].index:
                    prices[s] = period_data[s].loc[date, 'close']

            if not prices:
                continue

            # Calculate portfolio value
            position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)
            current_value = cash + position_value

            # Detect regime using market proxy
            regime, confidence = self._detect_regime_from_data(detector, hist_data)
            regimes_detected.append(regime)

            if last_regime is not None and regime != last_regime:
                regime_changes += 1
            last_regime = regime

            # Track for regime accuracy calculation
            regime_predictions.append(regime)

            # Calculate actual market behavior for accuracy
            if i >= 20:
                market_return = self._calculate_market_return(hist_data, 20)
                market_vol = self._calculate_market_volatility(hist_data, 20)
                actual_market_behavior.append({
                    'return': market_return,
                    'volatility': market_vol,
                })

            # Rebalance every 5 days
            if i == 0 or i % 5 == 0:
                # Generate signals based on regime
                signals = self._generate_regime_signals(hist_data, regime, confidence, blend_weights)

                # Convert to weights
                weights = self._signals_to_weights(signals)

                # Execute rebalance
                for symbol in list(positions.keys()):
                    if symbol not in prices:
                        continue

                    target_value = weights.get(symbol, 0) * current_value
                    current_shares = positions.get(symbol, 0)
                    current_pos_value = current_shares * prices[symbol]

                    trade_value = target_value - current_pos_value
                    if prices[symbol] > 0:
                        trade_shares = trade_value / prices[symbol]

                        if abs(trade_shares) > 0.01:
                            positions[symbol] = current_shares + trade_shares
                            cash -= trade_value
                            # Transaction cost
                            cash -= abs(trade_value) * 0.001

            # Record daily value
            position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in positions)
            daily_value = cash + position_value

            portfolio_values.append({
                'date': date,
                'value': daily_value,
                'regime': regime.value,
            })

            if len(portfolio_values) > 1:
                prev_value = portfolio_values[-2]['value']
                if prev_value > 0:
                    daily_ret = (daily_value - prev_value) / prev_value
                    daily_returns.append(daily_ret)

        if len(portfolio_values) < 10:
            return {'error': 'Insufficient portfolio values'}

        # Calculate metrics
        values = pd.Series([p['value'] for p in portfolio_values])
        returns = pd.Series(daily_returns)

        total_return = (values.iloc[-1] - self.initial_capital) / self.initial_capital * 100

        if len(returns) > 0 and returns.std() > 0:
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
            max_dd = abs(drawdowns.min()) * 100
        else:
            max_dd = 0

        # Regime distribution
        regime_counts = {}
        for r in regimes_detected:
            regime_counts[r.value] = regime_counts.get(r.value, 0) + 1
        total_regimes = len(regimes_detected)
        regime_distribution = {k: v / total_regimes * 100 for k, v in regime_counts.items()}

        # Calculate regime accuracy
        regime_accuracy = self._calculate_regime_accuracy(regime_predictions, actual_market_behavior)

        # Calculate allocation effectiveness
        allocation_effectiveness = self._calculate_allocation_effectiveness(
            portfolio_values, period_data, trading_dates
        )

        return {
            'sharpe': sharpe,
            'total_return': total_return,
            'volatility': volatility,
            'max_drawdown': max_dd,
            'regime_distribution': regime_distribution,
            'regime_changes': regime_changes,
            'regime_accuracy': regime_accuracy,
            'allocation_effectiveness': allocation_effectiveness,
            'trading_days': len(trading_dates),
        }

    def _detect_regime_from_data(
        self,
        detector: RegimeDetector,
        hist_data: Dict[str, pd.DataFrame],
    ) -> Tuple[RegimeType, float]:
        """Detect regime from multi-stock data."""
        # Create market proxy from equal-weighted average
        all_returns = []
        for symbol, df in hist_data.items():
            if len(df) >= 50:
                returns = df['close'].pct_change()
                all_returns.append(returns)

        if not all_returns:
            return RegimeType.SIDEWAYS_NEUTRAL, 0.5

        avg_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        proxy_prices = (1 + avg_returns).cumprod() * 100
        proxy_prices = proxy_prices.dropna()

        if len(proxy_prices) < 50:
            return RegimeType.SIDEWAYS_NEUTRAL, 0.5

        proxy_df = pd.DataFrame({
            'open': proxy_prices.values,
            'high': proxy_prices.values * 1.01,
            'low': proxy_prices.values * 0.99,
            'close': proxy_prices.values,
            'volume': np.ones(len(proxy_prices)) * 1000000,
        }, index=proxy_prices.index)

        try:
            regime = detector.detect_regime(proxy_df)
            confidence = detector.get_regime_confidence()
            return regime, confidence
        except Exception:
            return RegimeType.SIDEWAYS_NEUTRAL, 0.5

    def _calculate_market_return(
        self,
        hist_data: Dict[str, pd.DataFrame],
        lookback: int,
    ) -> float:
        """Calculate market return over lookback period."""
        returns = []
        for symbol, df in hist_data.items():
            if len(df) >= lookback:
                ret = (df['close'].iloc[-1] - df['close'].iloc[-lookback]) / df['close'].iloc[-lookback]
                returns.append(ret)
        return np.mean(returns) if returns else 0

    def _calculate_market_volatility(
        self,
        hist_data: Dict[str, pd.DataFrame],
        lookback: int,
    ) -> float:
        """Calculate market volatility over lookback period."""
        all_vols = []
        for symbol, df in hist_data.items():
            if len(df) >= lookback:
                returns = df['close'].pct_change().tail(lookback)
                vol = returns.std() * np.sqrt(252)
                all_vols.append(vol)
        return np.mean(all_vols) if all_vols else 0.2

    def _calculate_regime_accuracy(
        self,
        predictions: List[RegimeType],
        actual_behavior: List[Dict[str, float]],
    ) -> float:
        """
        Calculate regime detection accuracy.

        Compares predicted regime to actual market behavior:
        - BULL_TRENDING should correspond to positive returns
        - BEAR_CRISIS should correspond to negative returns
        - HIGH_VOLATILITY should correspond to high volatility
        - SIDEWAYS_NEUTRAL should correspond to low volatility and small returns
        """
        if len(actual_behavior) == 0:
            return 0.5

        correct = 0
        total = 0

        # Align predictions with behavior (behavior starts 20 days into predictions)
        offset = len(predictions) - len(actual_behavior)

        for i, behavior in enumerate(actual_behavior):
            if i + offset >= len(predictions):
                break

            pred = predictions[i + offset]
            ret = behavior['return']
            vol = behavior['volatility']

            # Define thresholds
            high_vol_threshold = 0.25  # 25% annualized
            positive_return_threshold = 0.01  # 1% over 20 days
            negative_return_threshold = -0.01

            is_correct = False

            if pred == RegimeType.BULL_TRENDING:
                is_correct = ret > positive_return_threshold
            elif pred == RegimeType.BEAR_CRISIS:
                is_correct = ret < negative_return_threshold
            elif pred == RegimeType.HIGH_VOLATILITY:
                is_correct = vol > high_vol_threshold
            elif pred == RegimeType.SIDEWAYS_NEUTRAL:
                is_correct = abs(ret) < 0.02 and vol < high_vol_threshold
            else:
                is_correct = True  # UNKNOWN or others

            if is_correct:
                correct += 1
            total += 1

        return correct / total if total > 0 else 0.5

    def _calculate_allocation_effectiveness(
        self,
        portfolio_values: List[Dict],
        data: Dict[str, pd.DataFrame],
        trading_dates: List,
    ) -> float:
        """
        Calculate how effective the regime-based allocation was.

        Compare regime-aware strategy return to equal-weight benchmark.
        """
        if len(portfolio_values) < 10 or len(trading_dates) < 10:
            return 0.0

        # Strategy return
        strategy_return = (portfolio_values[-1]['value'] - portfolio_values[0]['value']) / portfolio_values[0]['value']

        # Equal-weight benchmark
        first_date = trading_dates[0]
        last_date = trading_dates[-1]

        benchmark_returns = []
        for symbol, df in data.items():
            if first_date in df.index and last_date in df.index:
                ret = (df.loc[last_date, 'close'] - df.loc[first_date, 'close']) / df.loc[first_date, 'close']
                benchmark_returns.append(ret)

        benchmark_return = np.mean(benchmark_returns) if benchmark_returns else 0

        # Effectiveness = excess return over benchmark
        effectiveness = strategy_return - benchmark_return

        return effectiveness * 100  # As percentage

    def _generate_regime_signals(
        self,
        hist_data: Dict[str, pd.DataFrame],
        regime: RegimeType,
        confidence: float,
        blend_weights: Dict[RegimeType, Dict[str, float]],
    ) -> Dict[str, float]:
        """Generate trading signals based on regime."""
        weights = blend_weights.get(regime, {'momentum': 0.35, 'adaptive_hf': 0.65})

        # Regime-specific exposure
        exposure_map = {
            RegimeType.BULL_TRENDING: 1.0,
            RegimeType.BEAR_CRISIS: 0.50,
            RegimeType.SIDEWAYS_NEUTRAL: 0.75,
            RegimeType.HIGH_VOLATILITY: 0.60,
        }
        exposure = exposure_map.get(regime, 0.70)

        signals = {}
        for symbol, df in hist_data.items():
            if len(df) < 60:
                signals[symbol] = 0.0
                continue

            # Momentum signal
            mom_signal = self._calc_momentum_signal(df)

            # Hedge fund signal
            hf_signal = self._calc_hf_signal(df, regime)

            # Blend
            blended = weights.get('momentum', 0.35) * mom_signal + weights.get('adaptive_hf', 0.65) * hf_signal
            blended *= exposure * (0.5 + 0.5 * confidence)

            signals[symbol] = blended

        return signals

    def _calc_momentum_signal(self, df: pd.DataFrame) -> float:
        """Calculate momentum signal."""
        prices = df['close']

        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # MA crossover
        fast_ma = prices.rolling(10).mean().iloc[-1]
        slow_ma = prices.rolling(30).mean().iloc[-1]
        ma_diff = (fast_ma - slow_ma) / slow_ma if slow_ma > 0 else 0

        # Momentum
        momentum = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] if len(prices) > 20 else 0

        # Combine
        rsi_signal = (30 - current_rsi) / 30 if current_rsi < 30 else ((70 - current_rsi) / 30 if current_rsi > 70 else 0)
        ma_signal = np.clip(ma_diff * 10, -1, 1)
        mom_signal = np.clip(momentum * 5, -1, 1)

        return np.clip(0.4 * rsi_signal + 0.4 * ma_signal + 0.2 * mom_signal, -1, 1)

    def _calc_hf_signal(self, df: pd.DataFrame, regime: RegimeType) -> float:
        """Calculate hedge fund factor signal."""
        prices = df['close'].values

        if len(prices) < 60:
            return 0.0

        # Factors
        momentum = (prices[-5] - prices[-60]) / prices[-60]
        value = -((prices[-1] - prices[-21]) / prices[-21])  # Contrarian

        returns = pd.Series(prices).pct_change().dropna()
        quality = (returns > 0).sum() / len(returns) - 0.5 if len(returns) > 0 else 0
        low_vol = 1 / (returns.std() * np.sqrt(252) + 0.01) if len(returns) > 20 else 1

        # Normalize
        factors = {
            'momentum': np.clip(momentum * 5, -1, 1),
            'value': np.clip(value * 5, -1, 1),
            'quality': np.clip(quality * 4, -1, 1),
            'low_vol': np.clip((low_vol - 5) / 10, -1, 1),
        }

        # Regime-adaptive weighting
        if regime == RegimeType.BULL_TRENDING:
            signal = 0.50 * factors['momentum'] + 0.15 * factors['value'] + 0.20 * factors['quality'] + 0.15 * factors['low_vol']
        elif regime in [RegimeType.BEAR_CRISIS, RegimeType.HIGH_VOLATILITY]:
            signal = 0.20 * factors['momentum'] + 0.20 * factors['value'] + 0.30 * factors['quality'] + 0.30 * factors['low_vol']
        else:
            signal = 0.30 * factors['momentum'] + 0.20 * factors['value'] + 0.25 * factors['quality'] + 0.25 * factors['low_vol']

        return np.clip(signal, -1, 1)

    def _signals_to_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Convert signals to portfolio weights."""
        positive = {k: max(0, v) for k, v in signals.items()}
        total = sum(positive.values())

        if total > 0:
            return {k: v / total for k, v in positive.items()}
        return {k: 1 / len(signals) for k in signals}

    def optimize_parameters(
        self,
        data: Dict[str, pd.DataFrame],
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
    ) -> Tuple[RegimeDetectorConfig, Dict[RegimeType, Dict[str, float]], float]:
        """
        Optimize parameters on training data using grid search.

        Returns best config, weights, and best Sharpe ratio.
        """
        best_sharpe = -999
        best_config = None
        best_weights = None

        # Simplified grid search over key parameters
        for vol_pct in PARAM_GRID['volatility_high_percentile']:
            for trend_thresh in PARAM_GRID['strong_trend_threshold']:
                for bull_mom in PARAM_GRID['bull_momentum_weight']:
                    for bear_mom in PARAM_GRID['bear_momentum_weight']:

                        # Create config
                        config = RegimeDetectorConfig(
                            volatility_high_percentile=vol_pct,
                            strong_trend_threshold=trend_thresh,
                        )

                        # Create blend weights
                        weights = {
                            RegimeType.BULL_TRENDING: {'momentum': bull_mom, 'adaptive_hf': 1 - bull_mom},
                            RegimeType.BEAR_CRISIS: {'momentum': bear_mom, 'adaptive_hf': 1 - bear_mom},
                            RegimeType.SIDEWAYS_NEUTRAL: {'momentum': 0.30, 'adaptive_hf': 0.70},
                            RegimeType.HIGH_VOLATILITY: {'momentum': 0.15, 'adaptive_hf': 0.85},
                        }

                        # Run backtest on training period
                        result = self.run_backtest_period(
                            data, train_start, train_end, config, weights
                        )

                        if 'error' not in result and result['sharpe'] > best_sharpe:
                            best_sharpe = result['sharpe']
                            best_config = config
                            best_weights = weights

        # Fallback to defaults if optimization failed
        if best_config is None:
            best_config = RegimeDetectorConfig()
            best_weights = {
                RegimeType.BULL_TRENDING: {'momentum': 0.65, 'adaptive_hf': 0.35},
                RegimeType.BEAR_CRISIS: {'momentum': 0.10, 'adaptive_hf': 0.90},
                RegimeType.SIDEWAYS_NEUTRAL: {'momentum': 0.30, 'adaptive_hf': 0.70},
                RegimeType.HIGH_VOLATILITY: {'momentum': 0.15, 'adaptive_hf': 0.85},
            }
            best_sharpe = 0

        return best_config, best_weights, best_sharpe

    def run_walk_forward(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> List[WindowResult]:
        """
        Run complete walk-forward validation.
        """
        windows = self.generate_windows(START_YEAR, END_YEAR, TRAIN_MONTHS, TEST_MONTHS)

        print(f"\nGenerated {len(windows)} walk-forward windows")
        print("=" * 80)

        self.results = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}")
            print(f"   Train: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
            print(f"   Test:  {test_start.strftime('%Y-%m-%d')} to {test_end.strftime('%Y-%m-%d')}")

            # Step 1: Optimize on training period
            print("   Optimizing parameters...")
            best_config, best_weights, train_sharpe = self.optimize_parameters(
                data, train_start, train_end
            )

            # Step 2: Run backtest on training period with optimal params
            print("   Running in-sample backtest...")
            is_result = self.run_backtest_period(
                data, train_start, train_end, best_config, best_weights
            )

            if 'error' in is_result:
                print(f"   WARNING: In-sample backtest failed: {is_result['error']}")
                continue

            # Step 3: Run backtest on test period with same params
            print("   Running out-of-sample backtest...")
            oos_result = self.run_backtest_period(
                data, test_start, test_end, best_config, best_weights
            )

            if 'error' in oos_result:
                print(f"   WARNING: Out-of-sample backtest failed: {oos_result['error']}")
                continue

            # Calculate degradation
            sharpe_deg = ((is_result['sharpe'] - oos_result['sharpe']) / abs(is_result['sharpe']) * 100
                          if is_result['sharpe'] != 0 else 0)
            return_deg = ((is_result['total_return'] - oos_result['total_return']) / abs(is_result['total_return']) * 100
                          if is_result['total_return'] != 0 else 0)

            # Create window result
            window_result = WindowResult(
                window_id=i + 1,
                train_start=train_start.strftime('%Y-%m-%d'),
                train_end=train_end.strftime('%Y-%m-%d'),
                test_start=test_start.strftime('%Y-%m-%d'),
                test_end=test_end.strftime('%Y-%m-%d'),

                is_sharpe=is_result['sharpe'],
                is_return=is_result['total_return'],
                is_volatility=is_result['volatility'],
                is_max_drawdown=is_result['max_drawdown'],

                oos_sharpe=oos_result['sharpe'],
                oos_return=oos_result['total_return'],
                oos_volatility=oos_result['volatility'],
                oos_max_drawdown=oos_result['max_drawdown'],

                regime_accuracy=oos_result['regime_accuracy'],
                regime_distribution=oos_result['regime_distribution'],
                regime_changes=oos_result['regime_changes'],

                allocation_effectiveness=oos_result['allocation_effectiveness'],

                optimal_params={
                    'volatility_high_percentile': best_config.volatility_high_percentile,
                    'strong_trend_threshold': best_config.strong_trend_threshold,
                    'bull_momentum_weight': best_weights[RegimeType.BULL_TRENDING]['momentum'],
                    'bear_momentum_weight': best_weights[RegimeType.BEAR_CRISIS]['momentum'],
                },

                sharpe_degradation=sharpe_deg,
                return_degradation=return_deg,
            )

            self.results.append(window_result)

            # Print window summary
            print(f"\n   IN-SAMPLE:      Sharpe: {is_result['sharpe']:.2f}  Return: {is_result['total_return']:+.2f}%")
            print(f"   OUT-OF-SAMPLE:  Sharpe: {oos_result['sharpe']:.2f}  Return: {oos_result['total_return']:+.2f}%")
            print(f"   DEGRADATION:    Sharpe: {sharpe_deg:+.1f}%  Return: {return_deg:+.1f}%")
            print(f"   Regime Accuracy: {oos_result['regime_accuracy']*100:.1f}%")

        return self.results


# ============================================================================
# Main Execution
# ============================================================================

def calculate_overall_metrics(results: List[WindowResult]) -> Dict[str, Any]:
    """Calculate overall walk-forward metrics."""
    if not results:
        return {'error': 'No results'}

    # Average metrics
    avg_is_sharpe = np.mean([r.is_sharpe for r in results])
    avg_oos_sharpe = np.mean([r.oos_sharpe for r in results])
    avg_is_return = np.mean([r.is_return for r in results])
    avg_oos_return = np.mean([r.oos_return for r in results])

    avg_sharpe_deg = np.mean([r.sharpe_degradation for r in results])
    avg_return_deg = np.mean([r.return_degradation for r in results])

    avg_regime_accuracy = np.mean([r.regime_accuracy for r in results])
    avg_allocation_eff = np.mean([r.allocation_effectiveness for r in results])

    # Overall degradation percentage
    overall_sharpe_deg = ((avg_is_sharpe - avg_oos_sharpe) / abs(avg_is_sharpe) * 100
                          if avg_is_sharpe != 0 else 0)
    overall_return_deg = ((avg_is_return - avg_oos_return) / abs(avg_is_return) * 100
                          if avg_is_return != 0 else 0)

    # Std dev of OOS metrics
    std_oos_sharpe = np.std([r.oos_sharpe for r in results])
    std_oos_return = np.std([r.oos_return for r in results])

    # Win rate (positive OOS Sharpe)
    positive_oos_sharpe = sum(1 for r in results if r.oos_sharpe > 0)
    win_rate = positive_oos_sharpe / len(results) * 100

    return {
        'windows_count': len(results),
        'avg_in_sample_sharpe': avg_is_sharpe,
        'avg_out_of_sample_sharpe': avg_oos_sharpe,
        'avg_in_sample_return': avg_is_return,
        'avg_out_of_sample_return': avg_oos_return,
        'avg_sharpe_degradation_pct': avg_sharpe_deg,
        'avg_return_degradation_pct': avg_return_deg,
        'overall_sharpe_degradation_pct': overall_sharpe_deg,
        'overall_return_degradation_pct': overall_return_deg,
        'std_oos_sharpe': std_oos_sharpe,
        'std_oos_return': std_oos_return,
        'win_rate_pct': win_rate,
        'avg_regime_accuracy': avg_regime_accuracy,
        'avg_allocation_effectiveness': avg_allocation_eff,
    }


def main():
    print("=" * 80)
    print("REGIME-BLEND WALK-FORWARD VALIDATION")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"   Symbols:       {', '.join(SYMBOLS)}")
    print(f"   Period:        {START_YEAR} to {END_YEAR}")
    print(f"   Train Window:  {TRAIN_MONTHS} months")
    print(f"   Test Window:   {TEST_MONTHS} months")
    print(f"   Initial Capital: ${INITIAL_CAPITAL:,}")

    # Fetch data with extra buffer for warmup
    print("\nFetching market data...")
    start_with_buffer = f"{START_YEAR - 1}-01-01"
    end_date = f"{END_YEAR}-12-31"

    data = fetch_data_yfinance(SYMBOLS, start_with_buffer, end_date)

    if len(data) < 3:
        print("ERROR: Insufficient data fetched. Exiting.")
        return

    print(f"\n   Loaded data for {len(data)} symbols")

    # Run walk-forward validation
    wf_engine = RegimeAwareWalkForward(
        symbols=list(data.keys()),
        initial_capital=INITIAL_CAPITAL,
    )

    results = wf_engine.run_walk_forward(data)

    if not results:
        print("\nERROR: No valid walk-forward windows completed.")
        return

    # Calculate overall metrics
    overall = calculate_overall_metrics(results)

    # Print results
    print("\n" + "=" * 80)
    print("WALK-FORWARD RESULTS SUMMARY")
    print("=" * 80)

    print(f"\n{'Window':<10} {'Train Period':<25} {'IS Sharpe':>10} {'OOS Sharpe':>11} {'Sharpe Deg':>12} {'Regime Acc':>12}")
    print("-" * 85)

    for r in results:
        print(f"{r.window_id:<10} {r.train_start} - {r.train_end[:7]:<5} {r.is_sharpe:>10.2f} {r.oos_sharpe:>11.2f} {r.sharpe_degradation:>+11.1f}% {r.regime_accuracy*100:>11.1f}%")

    print("\n" + "=" * 80)
    print("OVERALL METRICS")
    print("=" * 80)

    print(f"\n   Total Windows Analyzed:        {overall['windows_count']}")
    print(f"\n   PERFORMANCE METRICS:")
    print(f"      Avg In-Sample Sharpe:       {overall['avg_in_sample_sharpe']:.3f}")
    print(f"      Avg Out-of-Sample Sharpe:   {overall['avg_out_of_sample_sharpe']:.3f}")
    print(f"      OOS Sharpe Std Dev:         {overall['std_oos_sharpe']:.3f}")
    print(f"      Win Rate (OOS Sharpe > 0):  {overall['win_rate_pct']:.1f}%")

    print(f"\n   DEGRADATION ANALYSIS:")
    print(f"      Overall Sharpe Degradation: {overall['overall_sharpe_degradation_pct']:+.1f}%")
    print(f"      Overall Return Degradation: {overall['overall_return_degradation_pct']:+.1f}%")
    print(f"      Avg Window Sharpe Deg:      {overall['avg_sharpe_degradation_pct']:+.1f}%")
    print(f"      Avg Window Return Deg:      {overall['avg_return_degradation_pct']:+.1f}%")

    print(f"\n   REGIME DETECTION:")
    print(f"      Avg Regime Accuracy:        {overall['avg_regime_accuracy']*100:.1f}%")
    print(f"      Avg Allocation Effectiveness: {overall['avg_allocation_effectiveness']:+.2f}%")

    # Robustness assessment
    print("\n" + "=" * 80)
    print("ROBUSTNESS ASSESSMENT")
    print("=" * 80)

    sharpe_deg = overall['overall_sharpe_degradation_pct']
    win_rate = overall['win_rate_pct']
    regime_acc = overall['avg_regime_accuracy'] * 100

    print("\n   VERDICT:")

    if sharpe_deg < 30 and win_rate >= 60 and regime_acc >= 50:
        print("   ROBUST - Strategy shows acceptable out-of-sample performance")
        print(f"      - Sharpe degradation ({sharpe_deg:.1f}%) is within acceptable limits (<30%)")
        print(f"      - Win rate ({win_rate:.1f}%) indicates consistent positive returns")
        print(f"      - Regime accuracy ({regime_acc:.1f}%) shows meaningful predictive power")
    elif sharpe_deg < 50 and win_rate >= 50:
        print("   MODERATE - Strategy shows moderate robustness")
        print(f"      - Sharpe degradation ({sharpe_deg:.1f}%) is elevated but manageable")
        print(f"      - Consider reducing position sizes in live trading")
    else:
        print("   CONCERN - Strategy shows signs of overfitting")
        print(f"      - Sharpe degradation ({sharpe_deg:.1f}%) suggests in-sample overfitting")
        print(f"      - Win rate ({win_rate:.1f}%) is below expectations")
        print("      - Consider simplifying the strategy or using longer training windows")

    # Save results to JSON
    output_json = {
        'configuration': {
            'symbols': SYMBOLS,
            'start_year': START_YEAR,
            'end_year': END_YEAR,
            'train_months': TRAIN_MONTHS,
            'test_months': TEST_MONTHS,
            'initial_capital': INITIAL_CAPITAL,
        },
        'overall_metrics': overall,
        'window_results': [
            {
                'window_id': r.window_id,
                'train_start': r.train_start,
                'train_end': r.train_end,
                'test_start': r.test_start,
                'test_end': r.test_end,
                'in_sample': {
                    'sharpe': r.is_sharpe,
                    'return': r.is_return,
                    'volatility': r.is_volatility,
                    'max_drawdown': r.is_max_drawdown,
                },
                'out_of_sample': {
                    'sharpe': r.oos_sharpe,
                    'return': r.oos_return,
                    'volatility': r.oos_volatility,
                    'max_drawdown': r.oos_max_drawdown,
                },
                'regime_analysis': {
                    'accuracy': r.regime_accuracy,
                    'distribution': r.regime_distribution,
                    'changes': r.regime_changes,
                },
                'allocation_effectiveness': r.allocation_effectiveness,
                'optimal_params': r.optimal_params,
                'degradation': {
                    'sharpe_pct': r.sharpe_degradation,
                    'return_pct': r.return_degradation,
                },
            }
            for r in results
        ],
    }

    output_path = OUTPUT_DIR / 'regime_blend_walkforward.json'
    with open(output_path, 'w') as f:
        json.dump(output_json, f, indent=2)

    print(f"\n   Results saved to: {output_path}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Regime-Aware Blended Strategy Backtest

Backtests a regime-aware strategy that:
- Detects market regimes (bull, bear, sideways, volatile)
- Blends momentum and adaptive hedge fund strategies based on regime
- Tracks regime transitions and performance by regime

Usage:
    python scripts/regime_blend_backtest.py
"""

import sys
sys.path.insert(0, '/opt/FinRL/src')

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from trading.data.fetcher import SP500DataFetcher, CacheConfig
from trading.strategies.regime_detector import RegimeDetector, RegimeDetectorConfig, RegimeType
from trading.strategies.hedge_fund import HedgeFundConfig, run_hedge_fund_backtest
from trading.strategies.momentum import MomentumConfig
from trading.strategies.enhanced_risk_manager import (
    EnhancedRiskManager, EnhancedRiskConfig, RiskLevel, fetch_vix_data
)


# ============================================================================
# Configuration
# ============================================================================

SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
START_DATE = "2004-12-01"  # GOOGL IPO + 70 day warmup for factor calculation
END_DATE = "2024-12-31"
INITIAL_CAPITAL = 100000
WARMUP_DAYS = 100  # Days needed for regime detection and factor calculation

OUTPUT_DIR = Path("/opt/FinRL/output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================================
# Regime-Aware Strategy Class
# ============================================================================

@dataclass
class RegimeTransition:
    """Record of a regime transition."""
    date: datetime
    from_regime: RegimeType
    to_regime: RegimeType
    confidence: float = 0.7


@dataclass
class RegimePerformance:
    """Performance metrics for a specific regime."""
    regime: RegimeType
    days: int
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    trades: int


class RegimeAwareStrategy:
    """
    Strategy that adapts to market regimes by blending momentum and hedge fund approaches.

    In bull markets: Emphasizes momentum for capturing trends
    In bear markets: Emphasizes defensive hedge fund positioning
    In sideways/volatile: Balanced approach with reduced exposure

    ENHANCED: Includes VIX-based leading indicators, volatility spike detection,
    and portfolio-level stop-loss for proactive bear protection.
    """

    def __init__(
        self,
        symbols: List[str],
        initial_capital: float = 100000,
        regime_config: Optional[RegimeDetectorConfig] = None,
        risk_config: Optional[EnhancedRiskConfig] = None,
        vix_data: Optional[pd.DataFrame] = None,
    ):
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Initialize regime detector
        self.regime_detector = RegimeDetector(regime_config or RegimeDetectorConfig())

        # Initialize enhanced risk manager (NEW)
        self.risk_manager = EnhancedRiskManager(risk_config or EnhancedRiskConfig())
        self.vix_data = vix_data

        # Risk tracking
        self.risk_signals: List[Tuple[datetime, RiskLevel, float]] = []  # date, level, exposure

        # Strategy configurations
        self.momentum_config = MomentumConfig(
            rsi_period=14,
            fast_ma_period=10,
            slow_ma_period=30,
            signal_threshold=0.25,
        )

        self.adaptive_hf_config = HedgeFundConfig(
            momentum_weight=0.40,
            value_weight=0.20,
            quality_weight=0.25,
            low_vol_weight=0.15,
            momentum_lookback=40,
            target_volatility=0.18,
            max_position_size=0.15,
            long_percentile=0.40,
            short_percentile=0.10,
            rebalance_frequency=10,
            adaptive_exposure=True,
            base_net_exposure=0.85,
            bear_net_exposure=0.30,
            trend_lookback=40,
        )

        # Regime-specific weights (only use the 4 available RegimeTypes)
        self.regime_weights = {
            RegimeType.BULL_TRENDING: {"momentum": 0.65, "adaptive_hf": 0.35},
            RegimeType.BEAR_CRISIS: {"momentum": 0.10, "adaptive_hf": 0.90},
            RegimeType.SIDEWAYS_NEUTRAL: {"momentum": 0.30, "adaptive_hf": 0.70},
            RegimeType.HIGH_VOLATILITY: {"momentum": 0.15, "adaptive_hf": 0.85},
        }

        # Exposure scaling by regime
        self.regime_exposure = {
            RegimeType.BULL_TRENDING: 1.0,
            RegimeType.BEAR_CRISIS: 0.50,
            RegimeType.SIDEWAYS_NEUTRAL: 0.75,
            RegimeType.HIGH_VOLATILITY: 0.60,
        }

        # State tracking
        self.positions: Dict[str, float] = {}  # symbol -> shares
        self.weights: Dict[str, float] = {}  # symbol -> weight
        self.current_regime: Optional[RegimeType] = None
        self.regime_history: List[Tuple[datetime, RegimeType, float]] = []  # date, regime, confidence
        self.transitions: List[RegimeTransition] = []

    def detect_regime(self, data: Dict[str, pd.DataFrame], date: datetime) -> Tuple[RegimeType, float]:
        """Detect current market regime using available data."""
        # Create market proxy from equal-weighted average
        all_returns = []
        for symbol, df in data.items():
            if len(df) >= 50:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                returns = df[close_col].pct_change()
                all_returns.append(returns)

        if not all_returns:
            return RegimeType.SIDEWAYS_NEUTRAL, 0.5

        # Use average returns to reconstruct market proxy
        avg_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        proxy_prices = (1 + avg_returns).cumprod() * 100
        proxy_prices = proxy_prices.dropna()

        # Create DataFrame for regime detector
        # The detector expects OHLCV columns
        proxy_df = pd.DataFrame({
            'open': proxy_prices.values,
            'high': proxy_prices.values * 1.01,  # Approximate
            'low': proxy_prices.values * 0.99,   # Approximate
            'close': proxy_prices.values,
            'volume': np.ones(len(proxy_prices)) * 1000000,
        }, index=proxy_prices.index)

        regime = self.regime_detector.detect_regime(proxy_df)
        confidence = self.regime_detector.get_regime_confidence()
        return regime, confidence

    def get_momentum_signal(self, df: pd.DataFrame, symbol: str) -> float:
        """
        Calculate momentum signal for a stock.
        Returns signal strength from -1 (strong sell) to +1 (strong buy).
        """
        if len(df) < 60:
            return 0.0

        close_col = 'Close' if 'Close' in df.columns else 'close'
        prices = df[close_col]

        # RSI
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Moving average crossover
        fast_ma = prices.rolling(10).mean()
        slow_ma = prices.rolling(30).mean()
        ma_diff = (fast_ma.iloc[-1] - slow_ma.iloc[-1]) / slow_ma.iloc[-1]

        # Momentum (rate of change)
        momentum = (prices.iloc[-1] - prices.iloc[-20]) / prices.iloc[-20] if len(prices) > 20 else 0

        # Combine signals
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
        """
        Calculate hedge fund factor signal for a stock.
        Returns signal strength from -1 (strong sell) to +1 (strong buy).
        """
        if symbol not in data or len(data[symbol]) < 60:
            return 0.0

        df = data[symbol]
        close_col = 'Close' if 'Close' in df.columns else 'close'
        prices = df[close_col].values

        # Momentum factor
        momentum = (prices[-5] - prices[-60]) / prices[-60] if len(prices) > 60 else 0

        # Value factor (contrarian)
        short_mom = (prices[-1] - prices[-21]) / prices[-21] if len(prices) > 21 else 0
        value = -short_mom

        # Quality factor
        returns = pd.Series(prices).pct_change().dropna()
        if len(returns) > 20:
            pos_ratio = (returns > 0).sum() / len(returns)
        else:
            pos_ratio = 0.5

        # Low volatility factor
        if len(returns) > 20:
            vol = returns.std() * np.sqrt(252)
            low_vol = 1 / (vol + 0.01)
        else:
            low_vol = 1.0

        # Z-score normalize (simplified)
        factors = {
            'momentum': np.clip(momentum * 5, -1, 1),
            'value': np.clip(value * 5, -1, 1),
            'quality': np.clip((pos_ratio - 0.5) * 4, -1, 1),
            'low_vol': np.clip((low_vol - 5) / 10, -1, 1),
        }

        # Regime-adaptive weighting
        if regime == RegimeType.BULL_TRENDING:
            # Emphasize momentum in bull markets
            signal = 0.50 * factors['momentum'] + 0.15 * factors['value'] + 0.20 * factors['quality'] + 0.15 * factors['low_vol']
        elif regime in [RegimeType.BEAR_CRISIS, RegimeType.HIGH_VOLATILITY]:
            # Emphasize quality and low vol in bear/volatile markets
            signal = 0.20 * factors['momentum'] + 0.20 * factors['value'] + 0.30 * factors['quality'] + 0.30 * factors['low_vol']
        else:
            # Balanced
            signal = 0.30 * factors['momentum'] + 0.20 * factors['value'] + 0.25 * factors['quality'] + 0.25 * factors['low_vol']

        return np.clip(signal, -1, 1)

    def generate_blended_signals(
        self,
        data: Dict[str, pd.DataFrame],
        regime: RegimeType,
        confidence: float,
        portfolio_value: float,
        current_date: Optional[datetime] = None,
    ) -> Tuple[Dict[str, float], float]:
        """
        Generate blended signals for all symbols based on current regime.
        Returns tuple of (signals dict, risk_exposure_multiplier).

        ENHANCED: Now includes risk manager's proactive exposure adjustment
        based on VIX, volatility spikes, and stop-losses.
        """
        weights = self.regime_weights.get(regime, {"momentum": 0.35, "adaptive_hf": 0.65})
        base_exposure = self.regime_exposure.get(regime, 0.70)

        # Get risk manager assessment (NEW)
        # Create market proxy for risk assessment
        all_closes = []
        for symbol, df in data.items():
            if len(df) > 0:
                close_col = 'Close' if 'Close' in df.columns else 'close'
                all_closes.append(df[[close_col]].rename(columns={close_col: symbol}))

        if all_closes:
            market_proxy = pd.concat(all_closes, axis=1).mean(axis=1)
            proxy_df = pd.DataFrame({
                'open': market_proxy.values,
                'high': market_proxy.values * 1.01,
                'low': market_proxy.values * 0.99,
                'close': market_proxy.values,
            }, index=market_proxy.index)
        else:
            proxy_df = pd.DataFrame()

        # Get VIX data for this date if available
        vix_slice = None
        if self.vix_data is not None and current_date is not None:
            try:
                vix_slice = self.vix_data[self.vix_data.index <= current_date].tail(30)
            except:
                pass

        # Assess risk using enhanced risk manager
        risk_signal = self.risk_manager.assess_risk(
            portfolio_value=portfolio_value,
            market_data=proxy_df,
            vix_data=vix_slice,
            current_date=current_date,
        )

        # Track risk signal
        self.risk_signals.append((current_date, risk_signal.risk_level, risk_signal.exposure_multiplier))

        # Apply MINIMUM of regime exposure and risk manager exposure
        # This ensures we're defensive when EITHER system says to be
        final_exposure = min(base_exposure, risk_signal.exposure_multiplier)

        signals = {}
        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < 60:
                signals[symbol] = 0.0
                continue

            # Get individual strategy signals
            mom_signal = self.get_momentum_signal(data[symbol], symbol)
            hf_signal = self.get_hf_signal(data, symbol, regime)

            # Blend signals
            blended = weights['momentum'] * mom_signal + weights['adaptive_hf'] * hf_signal

            # Scale by COMBINED exposure (regime + risk manager)
            blended *= final_exposure

            # Adjust by regime confidence
            blended *= (0.5 + 0.5 * confidence)

            signals[symbol] = blended

        return signals, risk_signal.exposure_multiplier

    def signals_to_weights(self, signals: Dict[str, float]) -> Dict[str, float]:
        """Convert signals to portfolio weights."""
        # Normalize positive signals to get long weights
        positive_signals = {k: max(0, v) for k, v in signals.items()}
        total_positive = sum(positive_signals.values())

        if total_positive > 0:
            weights = {k: v / total_positive for k, v in positive_signals.items()}
        else:
            # Equal weight if no positive signals
            weights = {k: 1 / len(signals) for k in signals}

        return weights


# ============================================================================
# Backtest Engine
# ============================================================================

def fetch_data(symbols: List[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Fetch stock data with caching."""
    cache_config = CacheConfig(enabled=True, directory=Path("/opt/FinRL/.cache"))
    fetcher = SP500DataFetcher(cache_config=cache_config)

    # Fetch extra history for warmup
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


def run_regime_aware_backtest(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    capital: float,
    start_date: str,
    end_date: str,
    rebalance_freq: int = 5,  # Rebalance every 5 days
    vix_data: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Run backtest with regime-aware blended strategy.

    ENHANCED: Now includes VIX-based leading indicators, volatility spike
    detection, and portfolio stop-losses for proactive bear protection.
    """
    # Configure enhanced risk management (AGGRESSIVE for better bear protection)
    risk_config = EnhancedRiskConfig(
        # Lower VIX thresholds for earlier detection
        vix_normal=12.0,        # was 15
        vix_elevated=16.0,      # was 20 - start reducing at lower VIX
        vix_high=20.0,          # was 25
        vix_extreme=25.0,       # was 30
        vix_crisis=35.0,        # was 40
        # More sensitive volatility spike detection
        vol_spike_threshold=1.3,  # was 1.5 (30% increase = spike, not 50%)
        vol_spike_lookback=3,     # faster detection (3 days vs default 5)
        # Tighter stop-losses
        stop_loss_portfolio=0.12,  # was 0.15 (12% max drawdown)
        stop_loss_trailing=0.08,   # was 0.10 (8% trailing stop)
        stop_loss_daily=0.04,      # was 0.05 (4% daily limit)
        # Faster drawdown response
        drawdown_threshold_1=0.03,  # 3% DD -> reduce 10%
        drawdown_threshold_2=0.06,  # 6% DD -> reduce 25%
        drawdown_threshold_3=0.10,  # 10% DD -> reduce 50%
        drawdown_threshold_4=0.15,  # 15% DD -> reduce 75%
        # CRASH DETECTION (fastest signal)
        crash_1day_threshold=-0.025,  # 2.5% drop in 1 day
        crash_3day_threshold=-0.04,   # 4% drop in 3 days
        crash_5day_threshold=-0.06,   # 6% drop in 5 days
        crash_enabled=True,
    )

    strategy = RegimeAwareStrategy(symbols, capital, vix_data=vix_data, risk_config=risk_config)

    # Get common trading dates
    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))

    # Filter to trading period
    trading_dates = [d for d in common_dates if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]

    if len(trading_dates) < WARMUP_DAYS:
        return {'error': 'Insufficient trading dates'}

    # Skip warmup
    trading_dates = trading_dates[WARMUP_DAYS:]

    # Initialize tracking
    cash = capital
    positions = {s: 0.0 for s in symbols}  # shares
    weights = {}

    portfolio_values = []
    daily_returns = []
    trade_count = 0
    regime_periods = []  # (start_date, end_date, regime)
    current_regime_start = None
    last_regime = None

    # Performance by regime
    regime_returns: Dict[RegimeType, List[float]] = {r: [] for r in RegimeType}

    # Track emergency de-risk events
    emergency_derisk_count = 0
    last_risk_level = RiskLevel.NORMAL

    for i, date in enumerate(trading_dates):
        # Get historical data up to this date
        hist_data = {s: df[df.index <= date].tail(120) for s, df in data.items()}

        # Get current prices
        prices = {s: data[s].loc[date, 'Close'] for s in symbols if date in data[s].index}

        # Calculate portfolio value
        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in symbols)
        current_value = cash + position_value

        # ========== DAILY RISK CHECK (NEW) ==========
        # Check risk signals EVERY day for emergency de-risking
        # Create market proxy for risk assessment
        all_closes = []
        for symbol_check, df_check in hist_data.items():
            if len(df_check) > 0:
                close_col = 'Close' if 'Close' in df_check.columns else 'close'
                all_closes.append(df_check[[close_col]].rename(columns={close_col: symbol_check}))

        if all_closes:
            market_proxy = pd.concat(all_closes, axis=1).mean(axis=1)
            proxy_df = pd.DataFrame({
                'open': market_proxy.values,
                'high': market_proxy.values * 1.01,
                'low': market_proxy.values * 0.99,
                'close': market_proxy.values,
            }, index=market_proxy.index)
        else:
            proxy_df = pd.DataFrame()

        # Get VIX slice for this date
        vix_slice = None
        if vix_data is not None:
            try:
                vix_slice = vix_data[vix_data.index <= date].tail(30)
            except:
                pass

        # Assess risk using enhanced risk manager
        risk_signal = strategy.risk_manager.assess_risk(
            portfolio_value=current_value,
            market_data=proxy_df,
            vix_data=vix_slice,
            current_date=date,
        )

        # EMERGENCY DE-RISKING: If risk level is HIGH or worse, reduce positions immediately
        if risk_signal.risk_level in [RiskLevel.HIGH, RiskLevel.EXTREME, RiskLevel.CRISIS]:
            if last_risk_level not in [RiskLevel.HIGH, RiskLevel.EXTREME, RiskLevel.CRISIS]:
                # Just entered high risk state - emergency de-risk
                emergency_exposure = risk_signal.exposure_multiplier
                for symbol in symbols:
                    if symbol not in prices or positions.get(symbol, 0) <= 0:
                        continue

                    current_shares = positions[symbol]
                    # Reduce to emergency_exposure level
                    target_shares = current_shares * emergency_exposure
                    sell_shares = current_shares - target_shares

                    if sell_shares > 0.01:
                        sell_value = sell_shares * prices[symbol]
                        positions[symbol] = target_shares
                        cash += sell_value
                        # Transaction cost
                        cost = sell_value * 0.001
                        cash -= cost
                        trade_count += 1
                        emergency_derisk_count += 1

        last_risk_level = risk_signal.risk_level
        # ========== END DAILY RISK CHECK ==========

        # Detect regime
        regime, confidence = strategy.detect_regime(hist_data, date)
        strategy.regime_history.append((date, regime, confidence))

        # Track regime transitions
        if last_regime is not None and regime != last_regime:
            strategy.transitions.append(RegimeTransition(
                date=date,
                from_regime=last_regime,
                to_regime=regime,
                confidence=confidence,
            ))
            if current_regime_start is not None:
                regime_periods.append((current_regime_start, date, last_regime))
            current_regime_start = date

        if current_regime_start is None:
            current_regime_start = date

        last_regime = regime

        # Rebalance at intervals
        should_rebalance = (i == 0 or i % rebalance_freq == 0)

        if should_rebalance:
            # Generate blended signals (with enhanced risk management)
            signals, risk_exposure = strategy.generate_blended_signals(
                hist_data, regime, confidence,
                portfolio_value=current_value,
                current_date=date
            )

            # Convert to weights
            new_weights = strategy.signals_to_weights(signals)

            # Execute rebalance
            for symbol in symbols:
                if symbol not in prices:
                    continue

                old_weight = weights.get(symbol, 0)
                new_weight = new_weights.get(symbol, 0)

                target_value = new_weight * current_value
                current_shares = positions.get(symbol, 0)
                current_pos_value = current_shares * prices[symbol]

                trade_value = target_value - current_pos_value
                trade_shares = trade_value / prices[symbol]

                if abs(trade_shares) > 0.01:
                    positions[symbol] = current_shares + trade_shares
                    cash -= trade_value
                    # Simple transaction cost
                    cost = abs(trade_value) * 0.001
                    cash -= cost
                    trade_count += 1

            weights = new_weights
            strategy.weights = weights

        # Record daily value
        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in symbols)
        daily_value = cash + position_value

        portfolio_values.append({
            'date': date,
            'value': daily_value,
            'regime': regime.value,
            'confidence': confidence,
        })

        if len(portfolio_values) > 1:
            prev_value = portfolio_values[-2]['value']
            daily_ret = (daily_value - prev_value) / prev_value
            daily_returns.append(daily_ret)

            # Track return by regime
            regime_returns[regime].append(daily_ret)

    # Final regime period
    if current_regime_start is not None and last_regime is not None:
        regime_periods.append((current_regime_start, trading_dates[-1], last_regime))

    # Calculate metrics
    values = pd.Series([p['value'] for p in portfolio_values])
    returns = pd.Series(daily_returns)

    total_return = (values.iloc[-1] - capital) / capital * 100

    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std()
        sortino_denom = returns[returns < 0].std()
        sortino = np.sqrt(252) * (returns.mean() - 0.05/252) / sortino_denom if sortino_denom > 0 else 0
        volatility = returns.std() * np.sqrt(252) * 100
    else:
        sharpe = sortino = volatility = 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

    # Calmar ratio
    calmar = total_return / max_dd if max_dd > 0 else 0

    # Regime-specific performance
    regime_performance = {}
    for regime, rets in regime_returns.items():
        if len(rets) > 10:
            ret_series = pd.Series(rets)
            regime_performance[regime.value] = {
                'days': len(rets),
                'total_return': (np.prod(1 + ret_series) - 1) * 100,
                'annualized_return': ((np.prod(1 + ret_series) ** (252 / len(rets))) - 1) * 100,
                'volatility': ret_series.std() * np.sqrt(252) * 100,
                'sharpe': np.sqrt(252) * ret_series.mean() / ret_series.std() if ret_series.std() > 0 else 0,
            }
        else:
            regime_performance[regime.value] = {
                'days': len(rets),
                'total_return': 0,
                'annualized_return': 0,
                'volatility': 0,
                'sharpe': 0,
            }

    # Regime time distribution
    regime_time = {}
    for regime in RegimeType:
        days = len(regime_returns[regime])
        regime_time[regime.value] = {
            'days': days,
            'percentage': days / len(trading_dates) * 100 if trading_dates else 0,
        }

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'volatility': volatility,
        'calmar_ratio': calmar,
        'total_trades': trade_count,
        'final_value': values.iloc[-1],
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
        'regime_performance': regime_performance,
        'regime_time': regime_time,
        'transitions': len(strategy.transitions),
        'regime_periods': [(str(s), str(e), r.value) for s, e, r in regime_periods],
    }


def run_pure_momentum_backtest(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    capital: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Run backtest with pure momentum strategy."""
    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))
    trading_dates = [d for d in common_dates if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]

    if len(trading_dates) < WARMUP_DAYS:
        return {'error': 'Insufficient trading dates'}

    trading_dates = trading_dates[WARMUP_DAYS:]

    cash = capital
    positions = {s: 0.0 for s in symbols}

    portfolio_values = []
    daily_returns = []

    strategy = RegimeAwareStrategy(symbols, capital)

    for i, date in enumerate(trading_dates):
        hist_data = {s: df[df.index <= date].tail(60) for s, df in data.items()}
        prices = {s: data[s].loc[date, 'Close'] for s in symbols if date in data[s].index}

        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in symbols)
        current_value = cash + position_value

        # Rebalance weekly
        if i == 0 or i % 5 == 0:
            signals = {}
            for symbol in symbols:
                if symbol in hist_data and len(hist_data[symbol]) >= 30:
                    signals[symbol] = strategy.get_momentum_signal(hist_data[symbol], symbol)
                else:
                    signals[symbol] = 0

            # Convert to weights
            positive = {k: max(0, v) for k, v in signals.items()}
            total = sum(positive.values())
            weights = {k: v / total if total > 0 else 1/len(symbols) for k, v in positive.items()}

            for symbol in symbols:
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
                    cash -= abs(trade_value) * 0.001

        position_value = sum(positions.get(s, 0) * prices.get(s, 0) for s in symbols)
        daily_value = cash + position_value

        portfolio_values.append({'date': date, 'value': daily_value})

        if len(portfolio_values) > 1:
            prev = portfolio_values[-2]['value']
            daily_returns.append((daily_value - prev) / prev)

    values = pd.Series([p['value'] for p in portfolio_values])
    returns = pd.Series(daily_returns)

    total_return = (values.iloc[-1] - capital) / capital * 100
    sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std() if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    max_dd = abs(((cumulative - running_max) / running_max).min()) * 100 if len(returns) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility': returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0,
        'final_value': values.iloc[-1],
        'portfolio_values': portfolio_values,
    }


def run_buy_and_hold(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    capital: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Run buy-and-hold benchmark."""
    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))
    trading_dates = [d for d in common_dates if d >= pd.Timestamp(start_date) and d <= pd.Timestamp(end_date)]

    if len(trading_dates) < WARMUP_DAYS:
        return {'error': 'Insufficient trading dates'}

    trading_dates = trading_dates[WARMUP_DAYS:]

    # Equal weight buy-and-hold
    capital_per_stock = capital / len(symbols)
    first_date = trading_dates[0]
    shares = {s: capital_per_stock / data[s].loc[first_date, 'Close'] for s in symbols}

    portfolio_values = []
    daily_returns = []

    for date in trading_dates:
        prices = {s: data[s].loc[date, 'Close'] for s in symbols if date in data[s].index}
        value = sum(shares.get(s, 0) * prices.get(s, 0) for s in symbols)
        portfolio_values.append({'date': date, 'value': value})

        if len(portfolio_values) > 1:
            prev = portfolio_values[-2]['value']
            daily_returns.append((value - prev) / prev)

    values = pd.Series([p['value'] for p in portfolio_values])
    returns = pd.Series(daily_returns)

    total_return = (values.iloc[-1] - capital) / capital * 100
    sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std() if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    max_dd = abs(((cumulative - running_max) / running_max).min()) * 100 if len(returns) > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'volatility': returns.std() * np.sqrt(252) * 100 if len(returns) > 0 else 0,
        'final_value': values.iloc[-1],
        'portfolio_values': portfolio_values,
    }


# ============================================================================
# Visualization
# ============================================================================

def create_visualizations(
    regime_result: Dict[str, Any],
    momentum_result: Dict[str, Any],
    adaptive_result: Dict[str, Any],
    bh_result: Dict[str, Any],
    output_path: Path,
):
    """Create performance visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Equity curves with regime shading
    ax1 = axes[0, 0]

    # Get dates and values
    regime_dates = [p['date'] for p in regime_result['portfolio_values']]
    regime_values = [p['value'] for p in regime_result['portfolio_values']]
    regime_regimes = [p.get('regime', 'sideways_neutral') for p in regime_result['portfolio_values']]

    mom_dates = [p['date'] for p in momentum_result['portfolio_values']]
    mom_values = [p['value'] for p in momentum_result['portfolio_values']]

    bh_dates = [p['date'] for p in bh_result['portfolio_values']]
    bh_values = [p['value'] for p in bh_result['portfolio_values']]

    adaptive_dates = [p['date'] for p in adaptive_result['portfolio_values']]
    adaptive_values = [p['value'] for p in adaptive_result['portfolio_values']]

    # Plot equity curves
    ax1.plot(regime_dates, regime_values, label='Regime Blend', linewidth=2, color='blue')
    ax1.plot(mom_dates, mom_values, label='Pure Momentum', linewidth=1.5, color='green', alpha=0.7)
    ax1.plot(adaptive_dates, adaptive_values, label='Adaptive HF', linewidth=1.5, color='orange', alpha=0.7)
    ax1.plot(bh_dates, bh_values, label='Buy & Hold', linewidth=1.5, color='gray', linestyle='--')

    # Shade regime periods
    regime_colors = {
        'bull_trending': 'lightgreen',
        'bear_crisis': 'lightcoral',
        'sideways_neutral': 'lightyellow',
        'high_volatility': 'lightblue',
    }

    # Group consecutive regime periods for shading
    if len(regime_dates) > 0:
        current_regime = regime_regimes[0]
        start_idx = 0

        for i in range(1, len(regime_regimes)):
            if regime_regimes[i] != current_regime or i == len(regime_regimes) - 1:
                end_idx = i if i == len(regime_regimes) - 1 else i - 1
                color = regime_colors.get(current_regime, 'white')
                ax1.axvspan(regime_dates[start_idx], regime_dates[end_idx], alpha=0.2, color=color)
                start_idx = i
                current_regime = regime_regimes[i]

    ax1.set_title('Portfolio Value with Regime Shading', fontsize=14)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Create legend patches for regimes
    regime_patches = [mpatches.Patch(color=color, alpha=0.3, label=regime.replace('_', ' ').title())
                      for regime, color in regime_colors.items()]
    ax1.legend(handles=ax1.get_legend_handles_labels()[0] + regime_patches,
               loc='upper left', fontsize=8, ncol=2)

    # 2. Performance by regime bar chart
    ax2 = axes[0, 1]

    regime_perf = regime_result['regime_performance']
    regimes = [r for r in regime_perf.keys() if regime_perf[r]['days'] > 10]

    if regimes:
        returns = [regime_perf[r]['annualized_return'] for r in regimes]
        sharpes = [regime_perf[r]['sharpe'] for r in regimes]

        x = np.arange(len(regimes))
        width = 0.35

        bars1 = ax2.bar(x - width/2, returns, width, label='Annualized Return (%)', color='blue', alpha=0.7)
        ax2.set_ylabel('Annualized Return (%)', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ax2_twin = ax2.twinx()
        bars2 = ax2_twin.bar(x + width/2, sharpes, width, label='Sharpe Ratio', color='orange', alpha=0.7)
        ax2_twin.set_ylabel('Sharpe Ratio', color='orange')
        ax2_twin.tick_params(axis='y', labelcolor='orange')

        ax2.set_xticks(x)
        ax2.set_xticklabels([r.replace('_', '\n') for r in regimes], fontsize=9)
        ax2.set_title('Regime Blend Performance by Regime', fontsize=14)

        # Add combined legend
        ax2.legend(handles=[bars1, bars2], loc='upper right')

    # 3. Regime time distribution
    ax3 = axes[1, 0]

    regime_time = regime_result['regime_time']
    labels = [r for r in regime_time.keys() if regime_time[r]['days'] > 0]
    sizes = [regime_time[r]['days'] for r in labels]
    colors = [regime_colors.get(r, 'gray') for r in labels]

    if sizes:
        wedges, texts, autotexts = ax3.pie(sizes, labels=None, autopct='%1.1f%%',
                                            colors=colors, startangle=90)
        ax3.legend(wedges, [r.replace('_', ' ').title() for r in labels],
                   title="Regimes", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        ax3.set_title(f'Regime Distribution\n({regime_result["transitions"]} transitions)', fontsize=14)

    # 4. Strategy comparison bar chart
    ax4 = axes[1, 1]

    strategies = ['Regime Blend', 'Pure Momentum', 'Adaptive HF', 'Buy & Hold']
    returns = [regime_result['total_return'], momentum_result['total_return'],
               adaptive_result['total_return'], bh_result['total_return']]
    sharpes = [regime_result['sharpe_ratio'], momentum_result['sharpe_ratio'],
               adaptive_result['sharpe_ratio'], bh_result['sharpe_ratio']]
    drawdowns = [regime_result['max_drawdown'], momentum_result['max_drawdown'],
                 adaptive_result['max_drawdown'], bh_result['max_drawdown']]

    x = np.arange(len(strategies))
    width = 0.25

    ax4.bar(x - width, returns, width, label='Total Return (%)', color='blue', alpha=0.7)
    ax4.bar(x, [s * 20 for s in sharpes], width, label='Sharpe x 20', color='green', alpha=0.7)
    ax4.bar(x + width, drawdowns, width, label='Max Drawdown (%)', color='red', alpha=0.7)

    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Value')
    ax4.set_title('Strategy Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(strategies)
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
    print("REGIME-AWARE BLENDED STRATEGY BACKTEST")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"   Symbols:  {', '.join(SYMBOLS)}")
    print(f"   Period:   {START_DATE} to {END_DATE}")
    print(f"   Capital:  ${INITIAL_CAPITAL:,}")
    print(f"   Strategy: Regime-adaptive momentum + hedge fund blend")
    print(f"   Enhanced: VIX leading indicator + volatility spike detection + stop-losses")

    # Fetch data
    print("\nFetching data...")
    data = fetch_data(SYMBOLS, START_DATE, END_DATE)
    print(f"   Loaded {len(data)} stocks")

    # Fetch VIX data for enhanced risk management
    print("   Fetching VIX data for leading indicator...")
    vix_data = fetch_vix_data(START_DATE, END_DATE)
    if len(vix_data) > 0:
        print(f"   Loaded VIX data: {len(vix_data)} days")
    else:
        print("   Warning: VIX data unavailable, using synthetic volatility")

    # Run backtests
    print("\nRunning backtests...")

    print("   1. Regime-aware blended strategy (ENHANCED)...")
    regime_result = run_regime_aware_backtest(
        data, SYMBOLS, INITIAL_CAPITAL, START_DATE, END_DATE,
        vix_data=vix_data
    )

    print("   2. Pure momentum strategy...")
    momentum_result = run_pure_momentum_backtest(data, SYMBOLS, INITIAL_CAPITAL, START_DATE, END_DATE)

    print("   3. Adaptive hedge fund strategy...")
    # Use the existing hedge fund backtest with adaptive config
    hf_data = {s: df.copy() for s, df in data.items()}
    adaptive_config = HedgeFundConfig(
        momentum_weight=0.40, value_weight=0.20, quality_weight=0.25, low_vol_weight=0.15,
        momentum_lookback=40, target_volatility=0.18, max_position_size=0.15,
        long_percentile=0.40, short_percentile=0.10, rebalance_frequency=10,
        adaptive_exposure=True, base_net_exposure=0.85, bear_net_exposure=0.30,
    )
    adaptive_result_raw = run_hedge_fund_backtest(
        hf_data, adaptive_config, INITIAL_CAPITAL,
        start_date=START_DATE, end_date=END_DATE
    )
    # Convert to our format
    adaptive_result = {
        'total_return': adaptive_result_raw.get('total_return', 0),
        'sharpe_ratio': adaptive_result_raw.get('sharpe_ratio', 0),
        'max_drawdown': adaptive_result_raw.get('max_drawdown', 0),
        'volatility': adaptive_result_raw.get('volatility', 0),
        'final_value': adaptive_result_raw.get('final_value', INITIAL_CAPITAL),
        'portfolio_values': adaptive_result_raw.get('portfolio_values', []),
    }

    print("   4. Buy & hold benchmark...")
    bh_result = run_buy_and_hold(data, SYMBOLS, INITIAL_CAPITAL, START_DATE, END_DATE)

    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    print(f"\n{'Strategy':<25} {'Return':>12} {'Sharpe':>10} {'Max DD':>10} {'Vol':>10}")
    print("-" * 70)
    print(f"{'Regime Blend':<25} {regime_result['total_return']:>+11.2f}% {regime_result['sharpe_ratio']:>10.2f} {regime_result['max_drawdown']:>9.2f}% {regime_result['volatility']:>9.2f}%")
    print(f"{'Pure Momentum':<25} {momentum_result['total_return']:>+11.2f}% {momentum_result['sharpe_ratio']:>10.2f} {momentum_result['max_drawdown']:>9.2f}% {momentum_result['volatility']:>9.2f}%")
    print(f"{'Adaptive HF':<25} {adaptive_result['total_return']:>+11.2f}% {adaptive_result['sharpe_ratio']:>10.2f} {adaptive_result['max_drawdown']:>9.2f}% {adaptive_result['volatility']:>9.2f}%")
    print(f"{'Buy & Hold':<25} {bh_result['total_return']:>+11.2f}% {bh_result['sharpe_ratio']:>10.2f} {bh_result['max_drawdown']:>9.2f}% {bh_result['volatility']:>9.2f}%")

    # Print regime-specific performance
    print("\n" + "=" * 80)
    print("REGIME-SPECIFIC PERFORMANCE (Regime Blend Strategy)")
    print("=" * 80)

    print(f"\n{'Regime':<20} {'Days':>8} {'Return':>12} {'Ann. Return':>14} {'Sharpe':>10}")
    print("-" * 70)

    for regime, perf in regime_result['regime_performance'].items():
        if perf['days'] > 5:
            print(f"{regime:<20} {perf['days']:>8} {perf['total_return']:>+11.2f}% {perf['annualized_return']:>+13.2f}% {perf['sharpe']:>10.2f}")

    # Regime distribution
    print("\n" + "=" * 80)
    print("REGIME TIME DISTRIBUTION")
    print("=" * 80)

    print(f"\n{'Regime':<20} {'Days':>8} {'Percentage':>12}")
    print("-" * 45)

    for regime, dist in regime_result['regime_time'].items():
        if dist['days'] > 0:
            print(f"{regime:<20} {dist['days']:>8} {dist['percentage']:>11.1f}%")

    print(f"\nTotal regime transitions: {regime_result['transitions']}")

    # Alpha analysis
    print("\n" + "=" * 80)
    print("ALPHA ANALYSIS")
    print("=" * 80)

    regime_alpha = regime_result['total_return'] - bh_result['total_return']
    momentum_alpha = momentum_result['total_return'] - bh_result['total_return']
    adaptive_alpha = adaptive_result['total_return'] - bh_result['total_return']

    print(f"\nAlpha vs Buy & Hold:")
    print(f"   Regime Blend:   {regime_alpha:+.2f}%")
    print(f"   Pure Momentum:  {momentum_alpha:+.2f}%")
    print(f"   Adaptive HF:    {adaptive_alpha:+.2f}%")

    regime_vs_momentum = regime_result['total_return'] - momentum_result['total_return']
    regime_vs_adaptive = regime_result['total_return'] - adaptive_result['total_return']

    print(f"\nRegime Blend vs:")
    print(f"   Pure Momentum:  {regime_vs_momentum:+.2f}%")
    print(f"   Adaptive HF:    {regime_vs_adaptive:+.2f}%")

    # Save results to JSON
    results_json = {
        'configuration': {
            'symbols': SYMBOLS,
            'start_date': START_DATE,
            'end_date': END_DATE,
            'initial_capital': INITIAL_CAPITAL,
        },
        'results': {
            'regime_blend': {
                'total_return': regime_result['total_return'],
                'sharpe_ratio': regime_result['sharpe_ratio'],
                'sortino_ratio': regime_result.get('sortino_ratio', 0),
                'max_drawdown': regime_result['max_drawdown'],
                'volatility': regime_result['volatility'],
                'calmar_ratio': regime_result.get('calmar_ratio', 0),
                'total_trades': regime_result['total_trades'],
                'final_value': regime_result['final_value'],
            },
            'pure_momentum': {
                'total_return': momentum_result['total_return'],
                'sharpe_ratio': momentum_result['sharpe_ratio'],
                'max_drawdown': momentum_result['max_drawdown'],
                'volatility': momentum_result['volatility'],
                'final_value': momentum_result['final_value'],
            },
            'adaptive_hf': {
                'total_return': adaptive_result['total_return'],
                'sharpe_ratio': adaptive_result['sharpe_ratio'],
                'max_drawdown': adaptive_result['max_drawdown'],
                'volatility': adaptive_result['volatility'],
                'final_value': adaptive_result['final_value'],
            },
            'buy_and_hold': {
                'total_return': bh_result['total_return'],
                'sharpe_ratio': bh_result['sharpe_ratio'],
                'max_drawdown': bh_result['max_drawdown'],
                'volatility': bh_result['volatility'],
                'final_value': bh_result['final_value'],
            },
        },
        'regime_analysis': {
            'performance_by_regime': regime_result['regime_performance'],
            'time_distribution': regime_result['regime_time'],
            'transition_count': regime_result['transitions'],
            'regime_periods': regime_result['regime_periods'],
        },
        'alpha_analysis': {
            'regime_vs_bh': regime_alpha,
            'momentum_vs_bh': momentum_alpha,
            'adaptive_vs_bh': adaptive_alpha,
            'regime_vs_momentum': regime_vs_momentum,
            'regime_vs_adaptive': regime_vs_adaptive,
        },
    }

    output_json = OUTPUT_DIR / 'regime_blend_results.json'
    with open(output_json, 'w') as f:
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nSaved results to {output_json}")

    # Create visualizations
    print("\nCreating visualizations...")
    output_png = OUTPUT_DIR / 'regime_blend_performance.png'
    create_visualizations(regime_result, momentum_result, adaptive_result, bh_result, output_png)

    # Final verdict
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    best_return = max(regime_result['total_return'], momentum_result['total_return'],
                      adaptive_result['total_return'], bh_result['total_return'])
    best_sharpe = max(regime_result['sharpe_ratio'], momentum_result['sharpe_ratio'],
                      adaptive_result['sharpe_ratio'], bh_result['sharpe_ratio'])

    if regime_result['total_return'] == best_return:
        print("\nRegime Blend achieves BEST TOTAL RETURN!")
    elif regime_result['sharpe_ratio'] == best_sharpe:
        print("\nRegime Blend achieves BEST RISK-ADJUSTED RETURN (Sharpe)!")

    if regime_alpha > 0:
        print(f"Regime Blend OUTPERFORMS Buy & Hold by {regime_alpha:.2f}%")
    else:
        print(f"Regime Blend underperforms Buy & Hold by {abs(regime_alpha):.2f}%")

    if regime_vs_momentum > 0 and regime_vs_adaptive > 0:
        print("Regime Blend OUTPERFORMS both component strategies!")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

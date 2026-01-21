"""
Multi-Regime Strategy System

Each market regime has its own specialized trading system optimized for that condition.
The Regime Orchestrator detects the current regime and activates the appropriate system.

Systems:
- Bull Market System: Aggressive momentum, trend-following, high exposure
- Bear Market System: Defensive, cash-heavy, capital preservation
- Sideways Market System: Mean reversion, range trading
- High Volatility System: Volatility targeting, reduced exposure
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from .regime_detector import RegimeType, RegimeDetector, RegimeDetectorConfig


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class BullSystemConfig:
    """Configuration for Bull Market System."""
    # Exposure settings
    max_exposure: float = 1.0  # Can go up to 100% invested
    min_exposure: float = 0.7  # Stay at least 70% invested

    # Momentum parameters
    momentum_lookback: int = 20  # Days for momentum calculation
    trend_confirmation_days: int = 5  # Days to confirm uptrend

    # Entry/Exit parameters
    buy_dip_threshold: float = -0.03  # Buy on 3% dips
    profit_target: float = 0.15  # Take some profits at 15%
    stop_loss: float = 0.08  # Tight stops in bull market

    # Position sizing
    concentration_limit: float = 0.35  # Max 35% in single position
    pyramid_threshold: float = 0.05  # Add to winners up 5%


@dataclass
class BearSystemConfig:
    """Configuration for Bear Market System - CAPITAL PRESERVATION FOCUS."""
    # Exposure settings - VERY DEFENSIVE
    max_exposure: float = 0.3  # Maximum 30% invested
    min_exposure: float = 0.0  # Can go to 100% cash
    default_cash_allocation: float = 0.7  # Start with 70% cash

    # Exit parameters - QUICK EXITS
    stop_loss: float = 0.03  # Very tight 3% stop loss
    trailing_stop: float = 0.05  # 5% trailing stop
    profit_target: float = 0.05  # Take profits quickly at 5%

    # Entry parameters - VERY SELECTIVE
    require_oversold: bool = True  # Only buy oversold conditions
    oversold_rsi: float = 25.0  # RSI below 25
    min_bounce_confirmation: int = 2  # Days of bounce before entry

    # Risk management
    max_position_size: float = 0.10  # Max 10% per position
    daily_loss_limit: float = 0.02  # Stop trading after 2% daily loss


@dataclass
class SidewaysSystemConfig:
    """Configuration for Sideways Market System - MEAN REVERSION FOCUS."""
    # Exposure settings
    max_exposure: float = 0.6  # Maximum 60% invested
    min_exposure: float = 0.3  # Keep some positions for opportunities

    # Mean reversion parameters
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # Range trading
    support_lookback: int = 20  # Days to identify support
    resistance_lookback: int = 20  # Days to identify resistance
    range_buffer: float = 0.02  # 2% buffer around levels

    # Position management
    profit_target: float = 0.05  # Smaller profit targets
    stop_loss: float = 0.04  # Tighter stops
    max_position_size: float = 0.20  # Max 20% per position


@dataclass
class HighVolSystemConfig:
    """Configuration for High Volatility System - VOLATILITY TARGETING."""
    # Volatility targeting
    target_volatility: float = 0.15  # Target 15% annualized vol
    max_exposure: float = 0.5  # Never exceed 50% in high vol
    min_exposure: float = 0.1  # Keep minimal exposure

    # Volatility calculation
    vol_lookback: int = 20  # Days for vol calculation
    vol_scaling_factor: float = 1.0  # Multiplier for vol targeting

    # Wider stops for volatile markets
    stop_loss: float = 0.10  # 10% stop loss (wider)
    trailing_stop: float = 0.12  # 12% trailing

    # Quick profit taking
    profit_target: float = 0.08  # Take 8% profits
    partial_profit_threshold: float = 0.05  # Take 50% at 5%

    # Position sizing based on vol
    max_position_size: float = 0.15  # Max 15% per position
    vol_adjusted_sizing: bool = True  # Scale positions by inverse vol


# =============================================================================
# Base System Class
# =============================================================================

class BaseRegimeSystem(ABC):
    """Abstract base class for regime-specific trading systems."""

    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.positions: Dict[str, float] = {s: 0.0 for s in symbols}
        self.entry_prices: Dict[str, float] = {}
        self.peak_prices: Dict[str, float] = {}

    @abstractmethod
    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio_value: float,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Generate trading signals. Returns target weights for each symbol."""
        pass

    @abstractmethod
    def get_exposure_multiplier(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> float:
        """Get the overall exposure multiplier for this system."""
        pass

    def update_position_tracking(
        self,
        symbol: str,
        shares: float,
        price: float,
    ) -> None:
        """Track position entry prices and peaks."""
        if shares > 0 and self.positions.get(symbol, 0) == 0:
            # New position
            self.entry_prices[symbol] = price
            self.peak_prices[symbol] = price
        elif shares > 0:
            # Update peak
            self.peak_prices[symbol] = max(self.peak_prices.get(symbol, price), price)
        elif shares == 0:
            # Position closed
            self.entry_prices.pop(symbol, None)
            self.peak_prices.pop(symbol, None)

        self.positions[symbol] = shares

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not np.isnan(rsi.iloc[-1]) else 50.0

    def _calculate_volatility(self, prices: pd.Series, period: int = 20) -> float:
        """Calculate annualized volatility."""
        if len(prices) < period + 1:
            return 0.2  # Default 20%

        returns = prices.pct_change().dropna()
        if len(returns) < period:
            return 0.2

        vol = returns.rolling(window=period).std().iloc[-1]
        return float(vol * np.sqrt(252)) if not np.isnan(vol) else 0.2


# =============================================================================
# Bull Market System
# =============================================================================

class BullMarketSystem(BaseRegimeSystem):
    """
    Bull Market System - AGGRESSIVE GROWTH FOCUS

    Strategy:
    - Stay heavily invested (70-100%)
    - Follow momentum and trends
    - Buy dips aggressively
    - Pyramid into winners
    - Ride trends with trailing stops
    """

    def __init__(self, symbols: List[str], config: Optional[BullSystemConfig] = None):
        super().__init__(symbols)
        self.config = config or BullSystemConfig()

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio_value: float,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Generate bullish signals - favor momentum and trends."""
        signals: Dict[str, float] = {}

        scores: Dict[str, float] = {}

        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < self.config.momentum_lookback:
                continue

            df = data[symbol]
            close = df["Close"]
            current_price = float(close.iloc[-1])

            # 1. Momentum Score (40% weight)
            if len(close) >= self.config.momentum_lookback:
                momentum = (current_price / float(close.iloc[-self.config.momentum_lookback]) - 1)
                momentum_score = min(max(momentum * 5, -1), 1)  # Scale to [-1, 1]
            else:
                momentum_score = 0

            # 2. Trend Score (30% weight) - Price above moving averages
            sma_20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else current_price
            sma_50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else current_price

            trend_score = 0
            if current_price > sma_20:
                trend_score += 0.5
            if current_price > sma_50:
                trend_score += 0.5
            if sma_20 > sma_50:
                trend_score += 0.5
            trend_score = min(trend_score, 1.0)

            # 3. Dip Buying Score (20% weight)
            recent_high = float(close.rolling(10).max().iloc[-1])
            dip_from_high = (current_price - recent_high) / recent_high

            dip_score = 0
            if dip_from_high < self.config.buy_dip_threshold:
                # Good dip buying opportunity
                dip_score = min(abs(dip_from_high) / abs(self.config.buy_dip_threshold), 1.0)

            # 4. RSI Score (10% weight) - Avoid extremely overbought
            rsi = self._calculate_rsi(close)
            rsi_score = 1.0 if rsi < 70 else max(0, 1 - (rsi - 70) / 30)

            # Combined score
            total_score = (
                momentum_score * 0.4 +
                trend_score * 0.3 +
                dip_score * 0.2 +
                rsi_score * 0.1
            )

            scores[symbol] = total_score

            # Check stop loss for existing positions
            if symbol in self.entry_prices:
                entry = self.entry_prices[symbol]
                pnl = (current_price - entry) / entry

                if pnl < -self.config.stop_loss:
                    scores[symbol] = -1  # Force exit

        # Normalize scores to weights
        if scores:
            # Only invest in positive momentum stocks
            positive_scores = {s: max(0, sc) for s, sc in scores.items()}
            total_positive = sum(positive_scores.values())

            if total_positive > 0:
                for symbol, score in positive_scores.items():
                    weight = score / total_positive
                    # Apply concentration limit
                    signals[symbol] = min(weight, self.config.concentration_limit)

            # Handle forced exits
            for symbol, score in scores.items():
                if score == -1:
                    signals[symbol] = 0.0

        return signals

    def get_exposure_multiplier(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> float:
        """In bull market, stay heavily invested."""
        # Base high exposure
        exposure = self.config.max_exposure

        # Slightly reduce if momentum is waning across all stocks
        avg_momentum = 0
        count = 0
        for symbol in self.symbols:
            if symbol in data and len(data[symbol]) >= 20:
                close = data[symbol]["Close"]
                mom = float(close.iloc[-1] / close.iloc[-20] - 1)
                avg_momentum += mom
                count += 1

        if count > 0:
            avg_momentum /= count
            if avg_momentum < 0:
                # Reduce exposure slightly if overall momentum negative
                exposure = max(self.config.min_exposure, exposure - 0.1)

        return exposure


# =============================================================================
# Bear Market System
# =============================================================================

class BearMarketSystem(BaseRegimeSystem):
    """
    Bear Market System - CAPITAL PRESERVATION FOCUS

    Strategy:
    - Stay mostly in cash (70-100%)
    - Very selective entries on oversold bounces
    - Quick profit taking
    - Tight stop losses
    - Never fight the trend
    """

    def __init__(self, symbols: List[str], config: Optional[BearSystemConfig] = None):
        super().__init__(symbols)
        self.config = config or BearSystemConfig()
        self.daily_pnl: float = 0.0
        self.trading_halted: bool = False

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio_value: float,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Generate defensive signals - preserve capital."""
        signals: Dict[str, float] = {s: 0.0 for s in self.symbols}

        # If trading halted due to daily loss, stay in cash
        if self.trading_halted:
            return signals

        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < 30:
                continue

            df = data[symbol]
            close = df["Close"]
            current_price = float(close.iloc[-1])

            # Check stop loss for existing positions - PRIORITY 1
            if symbol in self.entry_prices:
                entry = self.entry_prices[symbol]
                pnl = (current_price - entry) / entry

                # Tight stop loss
                if pnl < -self.config.stop_loss:
                    signals[symbol] = 0.0  # Exit
                    continue

                # Trailing stop from peak
                if symbol in self.peak_prices:
                    peak = self.peak_prices[symbol]
                    drawdown = (current_price - peak) / peak
                    if drawdown < -self.config.trailing_stop:
                        signals[symbol] = 0.0  # Exit
                        continue

                # Take profit
                if pnl >= self.config.profit_target:
                    signals[symbol] = 0.0  # Exit with profit
                    continue

                # Keep existing position if none of above triggered
                signals[symbol] = self.config.max_position_size
                continue

            # New entry criteria - VERY SELECTIVE
            if self.config.require_oversold:
                rsi = self._calculate_rsi(close)

                if rsi > self.config.oversold_rsi:
                    continue  # Not oversold enough

                # Check for bounce confirmation
                recent_returns = close.pct_change().tail(self.config.min_bounce_confirmation)
                if not all(r > 0 for r in recent_returns):
                    continue  # No confirmed bounce

                # Entry signal with small position
                signals[symbol] = self.config.max_position_size

        return signals

    def get_exposure_multiplier(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> float:
        """In bear market, stay defensive."""
        # Start with low exposure
        exposure = 1.0 - self.config.default_cash_allocation

        # Further reduce if downtrend is strong
        down_trend_count = 0
        for symbol in self.symbols:
            if symbol in data and len(data[symbol]) >= 50:
                close = data[symbol]["Close"]
                sma_20 = float(close.rolling(20).mean().iloc[-1])
                sma_50 = float(close.rolling(50).mean().iloc[-1])
                if sma_20 < sma_50:
                    down_trend_count += 1

        if len(self.symbols) > 0:
            down_ratio = down_trend_count / len(self.symbols)
            if down_ratio > 0.5:
                # Strong downtrend - reduce even more
                exposure = max(self.config.min_exposure, exposure * 0.5)

        return min(exposure, self.config.max_exposure)

    def reset_daily_tracking(self) -> None:
        """Reset daily PnL tracking."""
        self.daily_pnl = 0.0
        self.trading_halted = False


# =============================================================================
# Sideways Market System
# =============================================================================

class SidewaysMarketSystem(BaseRegimeSystem):
    """
    Sideways Market System - MEAN REVERSION FOCUS

    Strategy:
    - Trade the range
    - Buy at support (oversold)
    - Sell at resistance (overbought)
    - Smaller profit targets
    - Tighter risk management
    """

    def __init__(self, symbols: List[str], config: Optional[SidewaysSystemConfig] = None):
        super().__init__(symbols)
        self.config = config or SidewaysSystemConfig()

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio_value: float,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Generate mean reversion signals."""
        signals: Dict[str, float] = {}

        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < self.config.bollinger_period + 10:
                continue

            df = data[symbol]
            close = df["Close"]
            current_price = float(close.iloc[-1])

            # Calculate Bollinger Bands
            sma = close.rolling(self.config.bollinger_period).mean()
            std = close.rolling(self.config.bollinger_period).std()
            upper_band = sma + (std * self.config.bollinger_std)
            lower_band = sma - (std * self.config.bollinger_std)

            current_sma = float(sma.iloc[-1])
            current_upper = float(upper_band.iloc[-1])
            current_lower = float(lower_band.iloc[-1])

            # Calculate RSI
            rsi = self._calculate_rsi(close, self.config.rsi_period)

            # Check stop loss for existing positions
            if symbol in self.entry_prices:
                entry = self.entry_prices[symbol]
                pnl = (current_price - entry) / entry

                if pnl < -self.config.stop_loss:
                    signals[symbol] = 0.0  # Stop loss
                    continue

                if pnl >= self.config.profit_target:
                    signals[symbol] = 0.0  # Take profit
                    continue

                # Sell at resistance (upper band) or overbought RSI
                if current_price >= current_upper or rsi >= self.config.rsi_overbought:
                    signals[symbol] = 0.0  # Exit at resistance
                    continue

                # Hold position
                signals[symbol] = self.config.max_position_size
                continue

            # New entry criteria
            # Buy at support (lower band) and oversold RSI
            if current_price <= current_lower and rsi <= self.config.rsi_oversold:
                signals[symbol] = self.config.max_position_size
            # Buy near support with RSI confirmation
            elif current_price < current_sma and rsi < 40:
                signals[symbol] = self.config.max_position_size * 0.5
            else:
                signals[symbol] = 0.0

        return signals

    def get_exposure_multiplier(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> float:
        """In sideways market, moderate exposure."""
        # Count how many stocks are near extremes (trading opportunities)
        opportunity_count = 0

        for symbol in self.symbols:
            if symbol in data and len(data[symbol]) >= 20:
                close = data[symbol]["Close"]
                rsi = self._calculate_rsi(close)

                if rsi < 35 or rsi > 65:
                    opportunity_count += 1

        # More opportunities = slightly higher exposure
        base_exposure = (self.config.min_exposure + self.config.max_exposure) / 2
        if len(self.symbols) > 0:
            opp_ratio = opportunity_count / len(self.symbols)
            exposure = base_exposure + (opp_ratio * 0.1)
        else:
            exposure = base_exposure

        return min(exposure, self.config.max_exposure)


# =============================================================================
# High Volatility System
# =============================================================================

class HighVolatilitySystem(BaseRegimeSystem):
    """
    High Volatility System - VOLATILITY TARGETING

    Strategy:
    - Scale position sizes inversely with volatility
    - Target constant portfolio volatility
    - Wider stops to avoid whipsaws
    - Quick partial profit taking
    - Reduced overall exposure
    """

    def __init__(self, symbols: List[str], config: Optional[HighVolSystemConfig] = None):
        super().__init__(symbols)
        self.config = config or HighVolSystemConfig()

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        portfolio_value: float,
        current_date: datetime,
    ) -> Dict[str, float]:
        """Generate volatility-adjusted signals."""
        signals: Dict[str, float] = {}

        # Calculate volatility for each symbol
        volatilities: Dict[str, float] = {}
        for symbol in self.symbols:
            if symbol in data and len(data[symbol]) >= self.config.vol_lookback:
                vol = self._calculate_volatility(
                    data[symbol]["Close"],
                    self.config.vol_lookback
                )
                volatilities[symbol] = vol

        for symbol in self.symbols:
            if symbol not in data or len(data[symbol]) < 30:
                continue

            df = data[symbol]
            close = df["Close"]
            current_price = float(close.iloc[-1])

            # Check stop loss for existing positions (wider stops)
            if symbol in self.entry_prices:
                entry = self.entry_prices[symbol]
                pnl = (current_price - entry) / entry

                if pnl < -self.config.stop_loss:
                    signals[symbol] = 0.0  # Stop loss (wider than normal)
                    continue

                # Trailing stop
                if symbol in self.peak_prices:
                    peak = self.peak_prices[symbol]
                    drawdown = (current_price - peak) / peak
                    if drawdown < -self.config.trailing_stop:
                        signals[symbol] = 0.0
                        continue

                # Partial profit at threshold
                if pnl >= self.config.profit_target:
                    signals[symbol] = 0.0  # Full exit at target
                    continue

                # Calculate vol-adjusted position size
                if self.config.vol_adjusted_sizing and symbol in volatilities:
                    vol = volatilities[symbol]
                    # Target position = (target_vol / actual_vol) * base_position
                    vol_scalar = self.config.target_volatility / max(vol, 0.01)
                    vol_scalar = min(vol_scalar, 2.0)  # Cap at 2x
                    position_size = min(
                        self.config.max_position_size * vol_scalar,
                        self.config.max_position_size
                    )
                else:
                    position_size = self.config.max_position_size

                signals[symbol] = position_size
                continue

            # New entry - only if not too volatile
            if symbol in volatilities:
                vol = volatilities[symbol]

                # Skip extremely volatile stocks
                if vol > 0.6:  # 60% annualized vol
                    signals[symbol] = 0.0
                    continue

                # RSI for entry timing
                rsi = self._calculate_rsi(close)

                # Enter on oversold with vol-adjusted size
                if rsi < 35:
                    vol_scalar = self.config.target_volatility / max(vol, 0.01)
                    vol_scalar = min(vol_scalar, 1.5)
                    position_size = self.config.max_position_size * vol_scalar * 0.5
                    signals[symbol] = min(position_size, self.config.max_position_size)
                else:
                    signals[symbol] = 0.0
            else:
                signals[symbol] = 0.0

        return signals

    def get_exposure_multiplier(
        self,
        data: Dict[str, pd.DataFrame],
        current_date: datetime,
    ) -> float:
        """In high vol, reduce exposure based on market volatility."""
        # Calculate average market volatility
        total_vol = 0
        count = 0

        for symbol in self.symbols:
            if symbol in data and len(data[symbol]) >= self.config.vol_lookback:
                vol = self._calculate_volatility(
                    data[symbol]["Close"],
                    self.config.vol_lookback
                )
                total_vol += vol
                count += 1

        if count > 0:
            avg_vol = total_vol / count
            # Scale exposure inversely with volatility
            # If avg_vol is 30%, exposure = 15% / 30% * base = 0.5 * base
            vol_scalar = self.config.target_volatility / max(avg_vol, 0.1)
            vol_scalar = min(vol_scalar, 1.0)  # Never increase exposure

            exposure = self.config.max_exposure * vol_scalar
        else:
            exposure = self.config.max_exposure

        return max(self.config.min_exposure, exposure)


# =============================================================================
# Regime Orchestrator
# =============================================================================

@dataclass
class OrchestratorConfig:
    """Configuration for the Regime Orchestrator."""
    # Transition settings
    transition_buffer_days: int = 2  # Days before switching systems
    require_confirmation: bool = True  # Require multiple days of same regime

    # Risk overlay
    enable_vix_override: bool = True  # VIX can force defensive mode
    vix_crisis_threshold: float = 35.0  # VIX above this = force bear mode

    # Emergency settings
    portfolio_stop_loss: float = 0.15  # 15% portfolio stop loss
    enable_emergency_exit: bool = True


class RegimeOrchestrator:
    """
    Orchestrates between different market regime systems.

    Detects the current market regime and activates the appropriate
    specialized trading system.
    """

    def __init__(
        self,
        symbols: List[str],
        config: Optional[OrchestratorConfig] = None,
        regime_config: Optional[RegimeDetectorConfig] = None,
        bull_config: Optional[BullSystemConfig] = None,
        bear_config: Optional[BearSystemConfig] = None,
        sideways_config: Optional[SidewaysSystemConfig] = None,
        high_vol_config: Optional[HighVolSystemConfig] = None,
    ):
        self.symbols = symbols
        self.config = config or OrchestratorConfig()

        # Initialize regime detector
        self.regime_detector = RegimeDetector(regime_config or RegimeDetectorConfig())

        # Initialize specialized systems
        # LOW_VOLATILITY uses Bull system (calm markets favor trends)
        # UNKNOWN uses Sideways system as a safe default
        bull_system = BullMarketSystem(symbols, bull_config)
        bear_system = BearMarketSystem(symbols, bear_config)
        sideways_system = SidewaysMarketSystem(symbols, sideways_config)
        high_vol_system = HighVolatilitySystem(symbols, high_vol_config)

        self.systems: Dict[RegimeType, BaseRegimeSystem] = {
            RegimeType.BULL_TRENDING: bull_system,
            RegimeType.BEAR_CRISIS: bear_system,
            RegimeType.SIDEWAYS_NEUTRAL: sideways_system,
            RegimeType.HIGH_VOLATILITY: high_vol_system,
            RegimeType.LOW_VOLATILITY: bull_system,  # Low vol favors trends
            RegimeType.UNKNOWN: sideways_system,  # Safe default
        }

        # State tracking
        self.current_regime: RegimeType = RegimeType.SIDEWAYS_NEUTRAL
        self.regime_history: List[Tuple[datetime, RegimeType, float]] = []
        self.regime_confirmation_count: int = 0
        self.pending_regime: Optional[RegimeType] = None

        # Portfolio tracking
        self.peak_portfolio_value: float = 0.0
        self.emergency_mode: bool = False

    def detect_and_get_system(
        self,
        market_data: pd.DataFrame,
        vix_value: Optional[float] = None,
        current_date: Optional[datetime] = None,
    ) -> Tuple[BaseRegimeSystem, RegimeType, float]:
        """
        Detect current regime and return the appropriate system.

        Returns:
            Tuple of (active_system, regime_type, confidence)
        """
        # Detect regime - note: detect_regime returns only RegimeType
        regime = self.regime_detector.detect_regime(market_data)
        confidence = 0.7  # Default confidence since detector doesn't return it

        # VIX override - force bear mode in crisis
        if (
            self.config.enable_vix_override
            and vix_value is not None
            and vix_value > self.config.vix_crisis_threshold
        ):
            regime = RegimeType.BEAR_CRISIS
            confidence = 0.9

        # Handle regime transitions with confirmation
        if self.config.require_confirmation:
            if regime != self.current_regime:
                if regime == self.pending_regime:
                    self.regime_confirmation_count += 1
                else:
                    self.pending_regime = regime
                    self.regime_confirmation_count = 1

                if self.regime_confirmation_count >= self.config.transition_buffer_days:
                    # Confirmed regime change
                    self.current_regime = regime
                    self.pending_regime = None
                    self.regime_confirmation_count = 0

                    if current_date:
                        self.regime_history.append((current_date, regime, confidence))
            else:
                # Same regime, reset pending
                self.pending_regime = None
                self.regime_confirmation_count = 0
        else:
            if regime != self.current_regime:
                self.current_regime = regime
                if current_date:
                    self.regime_history.append((current_date, regime, confidence))

        return self.systems[self.current_regime], self.current_regime, confidence

    def generate_signals(
        self,
        data: Dict[str, pd.DataFrame],
        market_data: pd.DataFrame,
        portfolio_value: float,
        current_date: datetime,
        vix_value: Optional[float] = None,
    ) -> Tuple[Dict[str, float], RegimeType, float, float]:
        """
        Generate trading signals using the appropriate regime system.

        Returns:
            Tuple of (signals, regime, confidence, exposure_multiplier)
        """
        # Track portfolio peak
        self.peak_portfolio_value = max(self.peak_portfolio_value, portfolio_value)

        # Check for emergency exit
        if self.config.enable_emergency_exit:
            drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
            if drawdown > self.config.portfolio_stop_loss:
                self.emergency_mode = True

        # In emergency mode, return zero signals
        if self.emergency_mode:
            return {s: 0.0 for s in self.symbols}, RegimeType.BEAR_CRISIS, 1.0, 0.0

        # Get appropriate system
        system, regime, confidence = self.detect_and_get_system(
            market_data, vix_value, current_date
        )

        # Generate signals from the active system
        signals = system.generate_signals(data, portfolio_value, current_date)

        # Get exposure multiplier from the active system
        exposure = system.get_exposure_multiplier(data, current_date)

        # Scale signals by exposure
        scaled_signals = {s: w * exposure for s, w in signals.items()}

        return scaled_signals, regime, confidence, exposure

    def get_regime_statistics(self) -> Dict[str, Any]:
        """Get statistics about regime detection."""
        stats = {
            "current_regime": self.current_regime.value,
            "total_transitions": len(self.regime_history),
            "emergency_mode": self.emergency_mode,
            "regime_counts": {},
        }

        for _, regime, _ in self.regime_history:
            regime_name = regime.value
            stats["regime_counts"][regime_name] = stats["regime_counts"].get(regime_name, 0) + 1

        return stats

    def reset(self) -> None:
        """Reset orchestrator state."""
        self.current_regime = RegimeType.SIDEWAYS_NEUTRAL
        self.regime_history = []
        self.regime_confirmation_count = 0
        self.pending_regime = None
        self.peak_portfolio_value = 0.0
        self.emergency_mode = False

        # Reset all systems
        for system in self.systems.values():
            system.positions = {s: 0.0 for s in self.symbols}
            system.entry_prices = {}
            system.peak_prices = {}


# =============================================================================
# Helper Functions
# =============================================================================

def create_default_orchestrator(
    symbols: List[str],
    aggressive: bool = False,
) -> RegimeOrchestrator:
    """Create an orchestrator with default or aggressive settings."""

    if aggressive:
        # More aggressive settings for higher returns
        bull_config = BullSystemConfig(
            max_exposure=1.0,
            min_exposure=0.8,
            stop_loss=0.10,
            concentration_limit=0.40,
        )
        bear_config = BearSystemConfig(
            max_exposure=0.2,
            default_cash_allocation=0.8,
            stop_loss=0.025,
            profit_target=0.04,
        )
        sideways_config = SidewaysSystemConfig(
            max_exposure=0.5,
            profit_target=0.04,
            stop_loss=0.03,
        )
        high_vol_config = HighVolSystemConfig(
            max_exposure=0.4,
            target_volatility=0.12,
        )
    else:
        # Default conservative settings
        bull_config = None
        bear_config = None
        sideways_config = None
        high_vol_config = None

    return RegimeOrchestrator(
        symbols=symbols,
        bull_config=bull_config,
        bear_config=bear_config,
        sideways_config=sideways_config,
        high_vol_config=high_vol_config,
    )

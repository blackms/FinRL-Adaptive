"""
Enhanced Bear Market System with Inverse ETF Strategies

Strategies for profiting during bear markets:
1. Inverse ETFs (SH, SDS, SPXU) - profit from market declines
2. Volatility ETFs (VXX, UVXY) - profit from volatility spikes
3. Defensive sectors (utilities, consumer staples)
4. Short selling individual stocks
5. Managed futures / trend following

This system aims to generate POSITIVE returns during bear markets,
not just preserve capital.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd


class BearStrategy(Enum):
    """Bear market sub-strategies."""
    CASH = "cash"                    # 100% cash (safest)
    INVERSE_1X = "inverse_1x"        # 1x inverse ETF (SH)
    INVERSE_2X = "inverse_2x"        # 2x inverse ETF (SDS)
    INVERSE_3X = "inverse_3x"        # 3x inverse ETF (SPXU)
    VOLATILITY = "volatility"        # Long volatility (VXX)
    SHORT_STOCKS = "short_stocks"    # Short individual stocks
    DEFENSIVE = "defensive"          # Defensive sectors
    HYBRID = "hybrid"                # Mix of strategies


@dataclass
class EnhancedBearConfig:
    """Configuration for enhanced bear market system."""

    # Strategy allocation (must sum to 1.0)
    inverse_allocation: float = 0.30      # 30% to inverse ETFs
    volatility_allocation: float = 0.10   # 10% to volatility
    defensive_allocation: float = 0.10    # 10% to defensive
    cash_allocation: float = 0.50         # 50% cash buffer

    # Inverse ETF settings
    inverse_leverage: int = 1             # 1x, 2x, or 3x
    max_inverse_exposure: float = 0.40    # Max 40% in inverse
    inverse_stop_loss: float = 0.10       # 10% stop on inverse

    # Volatility settings
    max_vix_exposure: float = 0.15        # Max 15% in VIX products
    vix_entry_threshold: float = 18.0     # Enter VIX above 18
    vix_exit_threshold: float = 35.0      # Exit VIX above 35 (mean revert)

    # Short selling settings
    enable_short_selling: bool = True
    max_short_exposure: float = 0.20      # Max 20% short
    short_stop_loss: float = 0.08         # 8% stop on shorts

    # Regime transition
    bear_confirmation_days: int = 3       # Days to confirm bear
    gradual_entry: bool = True            # Scale into positions
    entry_scale_days: int = 5             # Days to reach full position

    # Risk management
    max_total_exposure: float = 0.50      # Max 50% exposure in bear
    daily_loss_limit: float = 0.02        # Stop after 2% daily loss
    correlation_limit: float = 0.70       # Max correlation between positions


@dataclass
class ShortPosition:
    """Track a short position."""
    symbol: str
    entry_price: float
    shares: float
    entry_date: datetime
    stop_price: float
    target_price: float


class EnhancedBearSystem:
    """
    Enhanced bear market trading system.

    Goals:
    1. Preserve capital (don't lose money)
    2. Generate positive returns from market decline
    3. Profit from volatility spikes
    4. Manage risk tightly
    """

    def __init__(
        self,
        symbols: List[str],
        config: Optional[EnhancedBearConfig] = None,
    ):
        self.symbols = symbols
        self.config = config or EnhancedBearConfig()

        # Inverse ETF mappings
        self.inverse_etfs = {
            1: "SH",    # ProShares Short S&P 500
            2: "SDS",   # ProShares UltraShort S&P 500
            3: "SPXU",  # ProShares UltraPro Short S&P 500
        }

        # Volatility ETFs
        self.volatility_etfs = ["VXX", "UVXY"]

        # Defensive sector ETFs
        self.defensive_etfs = ["XLU", "XLP", "XLV"]  # Utilities, Staples, Healthcare

        # State tracking
        self.positions: Dict[str, float] = {}
        self.short_positions: Dict[str, ShortPosition] = {}
        self.entry_prices: Dict[str, float] = {}
        self.bear_days_confirmed: int = 0
        self.current_exposure: float = 0.0
        self.daily_pnl: float = 0.0

    def calculate_signals(
        self,
        market_data: pd.DataFrame,
        stock_data: Dict[str, pd.DataFrame],
        vix_value: float,
        portfolio_value: float,
        current_date: datetime,
        bear_intensity: float = 0.5,  # 0-1 scale of how bearish
    ) -> Dict[str, float]:
        """
        Generate trading signals for bear market.

        Args:
            market_data: SPY/market data
            stock_data: Individual stock data
            vix_value: Current VIX level
            portfolio_value: Current portfolio value
            current_date: Current date
            bear_intensity: How bearish is the market (0-1)

        Returns:
            Dict of symbol -> target weight
        """
        signals: Dict[str, float] = {}

        # Scale allocations by bear intensity
        intensity_scale = min(1.0, max(0.3, bear_intensity))

        # 1. INVERSE ETF ALLOCATION
        inverse_weight = self._calculate_inverse_allocation(
            market_data, vix_value, intensity_scale
        )
        inverse_etf = self.inverse_etfs[self.config.inverse_leverage]
        signals[inverse_etf] = inverse_weight

        # 2. VOLATILITY ALLOCATION
        vol_weight = self._calculate_volatility_allocation(vix_value, intensity_scale)
        if vol_weight > 0:
            signals["VXX"] = vol_weight

        # 3. DEFENSIVE SECTORS
        defensive_weight = self._calculate_defensive_allocation(intensity_scale)
        if defensive_weight > 0:
            for etf in self.defensive_etfs:
                signals[etf] = defensive_weight / len(self.defensive_etfs)

        # 4. SHORT INDIVIDUAL STOCKS (weakest performers)
        if self.config.enable_short_selling:
            short_signals = self._calculate_short_signals(stock_data, intensity_scale)
            for symbol, weight in short_signals.items():
                signals[f"SHORT_{symbol}"] = weight

        # 5. CASH (remainder)
        total_allocated = sum(abs(w) for w in signals.values())
        signals["CASH"] = max(0, 1.0 - total_allocated)

        return signals

    def _calculate_inverse_allocation(
        self,
        market_data: pd.DataFrame,
        vix_value: float,
        intensity: float,
    ) -> float:
        """Calculate inverse ETF allocation based on market conditions."""
        base_allocation = self.config.inverse_allocation * intensity

        # Increase inverse when VIX is elevated but not extreme
        if 20 <= vix_value <= 35:
            base_allocation *= 1.2
        elif vix_value > 35:
            # Reduce inverse when VIX is extreme (likely to mean revert)
            base_allocation *= 0.7

        # Check market momentum for timing
        if len(market_data) >= 20:
            close = market_data["Close"]
            sma_10 = float(close.rolling(10).mean().iloc[-1])
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            current = float(close.iloc[-1])

            # Stronger inverse when price below both SMAs
            if current < sma_10 < sma_20:
                base_allocation *= 1.3
            # Reduce inverse if showing signs of bottoming
            elif current > sma_10:
                base_allocation *= 0.6

        return min(base_allocation, self.config.max_inverse_exposure)

    def _calculate_volatility_allocation(
        self,
        vix_value: float,
        intensity: float,
    ) -> float:
        """Calculate volatility ETF allocation."""
        # Only enter VIX if above entry threshold
        if vix_value < self.config.vix_entry_threshold:
            return 0.0

        # Exit VIX if too high (mean reversion likely)
        if vix_value > self.config.vix_exit_threshold:
            return 0.0

        # Sweet spot: VIX between 18-35
        base = self.config.volatility_allocation * intensity

        # Scale with VIX level
        vix_scale = (vix_value - 15) / 20  # 0 at 15, 1 at 35
        vix_scale = max(0, min(1, vix_scale))

        return min(base * vix_scale, self.config.max_vix_exposure)

    def _calculate_defensive_allocation(self, intensity: float) -> float:
        """Calculate defensive sector allocation."""
        # Less defensive as bear intensifies (want more inverse/cash)
        return self.config.defensive_allocation * (1 - intensity * 0.5)

    def _calculate_short_signals(
        self,
        stock_data: Dict[str, pd.DataFrame],
        intensity: float,
    ) -> Dict[str, float]:
        """Identify stocks to short based on weakness."""
        if not self.config.enable_short_selling:
            return {}

        short_candidates: List[Tuple[str, float]] = []

        for symbol, df in stock_data.items():
            if len(df) < 50:
                continue

            close = df["Close"]
            current = float(close.iloc[-1])

            # Weakness indicators
            sma_20 = float(close.rolling(20).mean().iloc[-1])
            sma_50 = float(close.rolling(50).mean().iloc[-1])

            # 20-day momentum
            mom_20 = (current / float(close.iloc[-20]) - 1) if len(close) >= 20 else 0

            # RSI
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.inf)
            rsi = float((100 - (100 / (1 + rs))).iloc[-1])

            # Score weakness (higher = weaker = better short)
            weakness_score = 0

            if current < sma_20:
                weakness_score += 1
            if current < sma_50:
                weakness_score += 1
            if sma_20 < sma_50:
                weakness_score += 1
            if mom_20 < -0.05:
                weakness_score += 1
            if rsi > 60:  # Overbought in downtrend = good short
                weakness_score += 0.5

            if weakness_score >= 3:
                short_candidates.append((symbol, weakness_score))

        # Sort by weakness and take top shorts
        short_candidates.sort(key=lambda x: x[1], reverse=True)

        short_signals = {}
        max_per_short = self.config.max_short_exposure / 3

        for symbol, _ in short_candidates[:3]:
            short_signals[symbol] = max_per_short * intensity

        return short_signals

    def calculate_bear_intensity(
        self,
        market_data: pd.DataFrame,
        vix_value: float,
    ) -> float:
        """
        Calculate how bearish the current market is (0-1 scale).

        0.0 = Mild bearish (just entered bear)
        0.5 = Moderate bear
        1.0 = Severe bear / crisis
        """
        intensity = 0.0

        if len(market_data) < 50:
            return 0.5

        close = market_data["Close"]
        current = float(close.iloc[-1])

        # 1. Price vs moving averages
        sma_20 = float(close.rolling(20).mean().iloc[-1])
        sma_50 = float(close.rolling(50).mean().iloc[-1])

        if current < sma_20:
            intensity += 0.15
        if current < sma_50:
            intensity += 0.15
        if sma_20 < sma_50:
            intensity += 0.10

        # 2. Drawdown from recent high
        high_20 = float(close.rolling(20).max().iloc[-1])
        high_50 = float(close.rolling(50).max().iloc[-1])

        dd_20 = (current - high_20) / high_20
        dd_50 = (current - high_50) / high_50

        if dd_20 < -0.05:
            intensity += 0.10
        if dd_20 < -0.10:
            intensity += 0.10
        if dd_50 < -0.15:
            intensity += 0.10
        if dd_50 < -0.25:
            intensity += 0.10

        # 3. VIX level
        if vix_value > 20:
            intensity += 0.05
        if vix_value > 25:
            intensity += 0.05
        if vix_value > 30:
            intensity += 0.05
        if vix_value > 40:
            intensity += 0.05

        return min(1.0, intensity)

    def get_exposure_summary(self) -> Dict[str, float]:
        """Get summary of current exposures."""
        return {
            "total_exposure": self.current_exposure,
            "long_exposure": sum(v for v in self.positions.values() if v > 0),
            "short_exposure": sum(abs(v) for v in self.positions.values() if v < 0),
            "inverse_exposure": self.positions.get("SH", 0) + self.positions.get("SDS", 0),
            "volatility_exposure": self.positions.get("VXX", 0),
        }


def simulate_inverse_etf_returns(
    market_returns: pd.Series,
    leverage: int = 1,
) -> pd.Series:
    """
    Simulate inverse ETF returns from market returns.

    Note: Leveraged inverse ETFs have decay over time due to daily rebalancing.
    This simulation accounts for that effect.
    """
    # Simple inverse: -1 * market return * leverage
    # But with leverage decay for multi-day periods

    inverse_returns = -leverage * market_returns

    # Add tracking error and expense ratio (approximate)
    tracking_error = 0.0003  # 3 bps daily
    expense_ratio = 0.0003   # ~0.9% annual / 252 days

    inverse_returns = inverse_returns - tracking_error - expense_ratio

    return inverse_returns


def simulate_vix_returns(
    vix_levels: pd.Series,
    holding_period: int = 1,
) -> pd.Series:
    """
    Simulate VXX-like returns from VIX levels.

    VXX tracks VIX futures, not spot VIX.
    It has significant contango decay in normal markets.
    """
    vix_changes = vix_levels.pct_change()

    # VXX captures ~50% of spot VIX moves typically
    vxx_returns = vix_changes * 0.5

    # Contango decay (loses ~5% per month in normal markets)
    contango_decay = 0.002  # ~0.2% per day

    # Less decay when VIX is elevated (backwardation)
    decay_factor = vix_levels.apply(
        lambda x: contango_decay if x < 20 else contango_decay * 0.3
    )

    vxx_returns = vxx_returns - decay_factor

    return vxx_returns

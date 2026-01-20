"""
Breakout Strategy Module

Implements a Turtle-style breakout trading strategy using Donchian Channels
with volume confirmation and ATR-based position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig


@dataclass
class BreakoutConfig(StrategyConfig):
    """
    Configuration for Breakout Strategy.

    Attributes:
        entry_period: Period for entry breakout (20-day high).
        exit_period: Period for exit breakdown (10-day low).
        atr_period: Period for ATR calculation.
        atr_position_risk: Fraction of capital to risk per ATR.
        volume_ma_period: Period for volume moving average.
        volume_threshold: Minimum volume ratio for confirmation (1.5x average).
        use_volume_confirmation: Whether to require volume confirmation.
        max_position_size: Maximum position size as fraction of capital.
        trailing_stop_atr: ATR multiplier for trailing stop.
    """

    name: str = "BreakoutStrategy"
    entry_period: int = 20
    exit_period: int = 10
    atr_period: int = 20
    atr_position_risk: float = 0.01
    volume_ma_period: int = 20
    volume_threshold: float = 1.5
    use_volume_confirmation: bool = True
    max_position_size: float = 0.2
    trailing_stop_atr: float = 2.0


class BreakoutStrategy(BaseStrategy):
    """
    Turtle-style breakout trading strategy.

    Generates buy/sell signals based on:
    - 20-day high breakout for entry (Donchian Channel)
    - 10-day low breakdown for exit
    - Volume confirmation (>1.5x average)
    - ATR-based position sizing and stops

    The strategy captures momentum by entering on significant breakouts
    and uses trailing stops to protect profits.

    Example:
        >>> config = BreakoutConfig(
        ...     entry_period=20,
        ...     exit_period=10,
        ...     volume_threshold=1.5,
        ... )
        >>> strategy = BreakoutStrategy(config)
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        """
        Initialize the breakout strategy.

        Args:
            config: Strategy configuration. Defaults to BreakoutConfig.
        """
        super().__init__(config or BreakoutConfig())
        self._config: BreakoutConfig = self.config  # type: ignore
        self._trailing_stops: dict[str, float] = {}

    def calculate_donchian_channel(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channel (highest high and lowest low over period).

        Args:
            high: Series of high prices.
            low: Series of low prices.
            period: Lookback period.

        Returns:
            Tuple of (upper_channel, middle_channel, lower_channel) as Series.
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2

        return upper, middle, lower

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int | None = None,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of close prices.
            period: Period for calculation. Defaults to config value.

        Returns:
            Series with ATR values.
        """
        period = period or self._config.atr_period

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_volume_ratio(
        self,
        volume: pd.Series,
        period: int | None = None,
    ) -> pd.Series:
        """
        Calculate volume relative to moving average.

        Args:
            volume: Series of volume data.
            period: Period for volume MA. Defaults to config value.

        Returns:
            Series with volume ratio (current / average).
        """
        period = period or self._config.volume_ma_period
        volume_ma = volume.rolling(window=period).mean()
        return volume / volume_ma

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all breakout indicators.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with added indicator columns.
        """
        df = data.copy()

        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Entry Donchian Channel (20-day)
        df["entry_high"], df["entry_mid"], df["entry_low"] = self.calculate_donchian_channel(
            df["high"], df["low"], self._config.entry_period
        )

        # Exit Donchian Channel (10-day)
        df["exit_high"], df["exit_mid"], df["exit_low"] = self.calculate_donchian_channel(
            df["high"], df["low"], self._config.exit_period
        )

        # ATR for position sizing and stops
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"])
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Volume analysis
        if self._config.use_volume_confirmation:
            df["volume_ratio"] = self.calculate_volume_ratio(df["volume"])
            df["volume_confirmed"] = df["volume_ratio"] >= self._config.volume_threshold
        else:
            df["volume_ratio"] = 1.0
            df["volume_confirmed"] = True

        # Breakout detection
        df["prev_entry_high"] = df["entry_high"].shift(1)
        df["prev_exit_low"] = df["exit_low"].shift(1)

        # True if breaking out above previous 20-day high
        df["breakout_up"] = df["high"] > df["prev_entry_high"]

        # True if breaking down below previous 10-day low
        df["breakdown"] = df["low"] < df["prev_exit_low"]

        # Breakout strength (how far above the channel)
        df["breakout_strength"] = np.where(
            df["breakout_up"],
            (df["high"] - df["prev_entry_high"]) / df["atr"],
            0.0
        )

        # Momentum confirmation
        df["momentum_5d"] = df["close"].pct_change(periods=5)
        df["momentum_10d"] = df["close"].pct_change(periods=10)

        # Trend filter (simple: price above/below 50-day MA)
        df["ma_50"] = df["close"].rolling(window=50).mean()
        df["above_ma_50"] = df["close"] > df["ma_50"]

        return df

    def _calculate_signal_strength(
        self,
        breakout_strength: float,
        volume_ratio: float,
        momentum_5d: float,
        momentum_10d: float,
        is_breakout: bool,
    ) -> float:
        """
        Calculate combined signal strength for breakout.

        Args:
            breakout_strength: ATR-normalized breakout distance.
            volume_ratio: Current volume / average volume.
            momentum_5d: 5-day price momentum.
            momentum_10d: 10-day price momentum.
            is_breakout: Whether this is a breakout (True) or breakdown (False).

        Returns:
            Signal strength between 0.0 and 1.0 for breakouts,
            -1.0 to 0.0 for breakdowns.
        """
        # Breakout strength component (0 to 1)
        strength_component = min(1.0, breakout_strength / 2.0)

        # Volume confirmation component (0.5 to 1.5 multiplier)
        volume_multiplier = min(1.5, 0.5 + volume_ratio / 3.0)

        # Momentum component (adds conviction)
        momentum_avg = (momentum_5d + momentum_10d) / 2
        if is_breakout:
            momentum_component = min(0.3, max(0, momentum_avg * 3))
        else:
            momentum_component = min(0.3, max(0, -momentum_avg * 3))

        # Combined signal
        raw_signal = (strength_component + momentum_component) * volume_multiplier

        if is_breakout:
            return float(np.clip(raw_signal, 0.0, 1.0))
        else:
            return float(np.clip(-raw_signal, -1.0, 0.0))

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        atr: float,
        is_long: bool,
    ) -> float | None:
        """
        Update trailing stop for a position.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
            atr: Current ATR value.
            is_long: Whether position is long.

        Returns:
            Updated trailing stop price or None if no stop set.
        """
        stop_distance = atr * self._config.trailing_stop_atr

        if symbol not in self._trailing_stops:
            # Initialize trailing stop
            if is_long:
                self._trailing_stops[symbol] = current_price - stop_distance
            else:
                self._trailing_stops[symbol] = current_price + stop_distance
        else:
            # Update trailing stop (only in favorable direction)
            current_stop = self._trailing_stops[symbol]
            if is_long:
                new_stop = current_price - stop_distance
                self._trailing_stops[symbol] = max(current_stop, new_stop)
            else:
                new_stop = current_price + stop_distance
                self._trailing_stops[symbol] = min(current_stop, new_stop)

        return self._trailing_stops.get(symbol)

    def check_trailing_stop(
        self,
        symbol: str,
        current_price: float,
        is_long: bool,
    ) -> bool:
        """
        Check if trailing stop has been triggered.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.
            is_long: Whether position is long.

        Returns:
            True if stop triggered, False otherwise.
        """
        if symbol not in self._trailing_stops:
            return False

        stop = self._trailing_stops[symbol]

        if is_long:
            return current_price <= stop
        else:
            return current_price >= stop

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate breakout trading signals.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            List of Signal objects.
        """
        if not self.validate_data(data):
            return []

        signals: list[Signal] = []
        df = self.calculate_indicators(data)

        # Get symbol from data or use default
        symbol = df.get("symbol", pd.Series(["UNKNOWN"])).iloc[-1]
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[-1]
        symbol = str(symbol)

        # Get the latest row
        latest = df.iloc[-1]

        # Skip if indicators are not ready
        required_fields = ["entry_high", "exit_low", "atr", "volume_ratio"]
        if any(pd.isna(latest.get(f)) for f in required_fields):
            return []

        # Get timestamp
        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Build metadata
        metadata: dict[str, Any] = {
            "entry_high": latest["entry_high"],
            "entry_low": latest["entry_low"],
            "exit_high": latest["exit_high"],
            "exit_low": latest["exit_low"],
            "atr": latest["atr"],
            "atr_pct": latest["atr_pct"],
            "volume_ratio": latest["volume_ratio"],
            "breakout_strength": latest.get("breakout_strength", 0),
        }

        signal_type = SignalType.HOLD
        strength = 0.0

        # Check if we have an existing position
        has_position = self.has_position(symbol)
        position = self.get_position(symbol) if has_position else None

        if has_position and position:
            # Update trailing stop
            trailing_stop = self.update_trailing_stop(
                symbol=symbol,
                current_price=latest["close"],
                atr=latest["atr"],
                is_long=position.is_long,
            )
            metadata["trailing_stop"] = trailing_stop

            # Check exit conditions for existing position
            if position.is_long:
                # Exit on 10-day low breakdown or trailing stop
                if latest["breakdown"] or self.check_trailing_stop(symbol, latest["close"], True):
                    signal_type = SignalType.SELL
                    strength = -0.7
                    metadata["trigger"] = "exit_long_breakdown" if latest["breakdown"] else "trailing_stop"

                    # Clear trailing stop
                    if symbol in self._trailing_stops:
                        del self._trailing_stops[symbol]

            else:  # Short position
                # Exit on 20-day high breakout or trailing stop
                if latest["breakout_up"] or self.check_trailing_stop(symbol, latest["close"], False):
                    signal_type = SignalType.BUY
                    strength = 0.7
                    metadata["trigger"] = "exit_short_breakout" if latest["breakout_up"] else "trailing_stop"

                    # Clear trailing stop
                    if symbol in self._trailing_stops:
                        del self._trailing_stops[symbol]

        else:
            # No position - check entry conditions
            volume_confirmed = (
                not self._config.use_volume_confirmation
                or latest["volume_confirmed"]
            )

            # Buy on 20-day high breakout with volume confirmation
            if latest["breakout_up"] and volume_confirmed:
                strength = self._calculate_signal_strength(
                    breakout_strength=latest.get("breakout_strength", 1.0),
                    volume_ratio=latest["volume_ratio"],
                    momentum_5d=latest.get("momentum_5d", 0),
                    momentum_10d=latest.get("momentum_10d", 0),
                    is_breakout=True,
                )

                if strength > 0.3:
                    signal_type = SignalType.BUY
                    metadata["trigger"] = "breakout_entry"
                    metadata["volume_confirmed"] = volume_confirmed

                    # Set initial trailing stop
                    initial_stop = latest["close"] - (latest["atr"] * self._config.trailing_stop_atr)
                    metadata["initial_stop"] = initial_stop

            # Sell (short) on breakdown below exit channel with volume
            elif latest["breakdown"] and volume_confirmed:
                strength = self._calculate_signal_strength(
                    breakout_strength=latest.get("breakout_strength", 1.0),
                    volume_ratio=latest["volume_ratio"],
                    momentum_5d=latest.get("momentum_5d", 0),
                    momentum_10d=latest.get("momentum_10d", 0),
                    is_breakout=False,
                )

                if strength < -0.3:
                    signal_type = SignalType.SELL
                    metadata["trigger"] = "breakdown_entry"
                    metadata["volume_confirmed"] = volume_confirmed

                    # Set initial trailing stop
                    initial_stop = latest["close"] + (latest["atr"] * self._config.trailing_stop_atr)
                    metadata["initial_stop"] = initial_stop

        # Create signal if not HOLD
        if signal_type != SignalType.HOLD:
            signal = Signal(
                symbol=symbol,
                signal_type=signal_type,
                strength=strength,
                timestamp=timestamp,
                price=float(latest["close"]),
                metadata=metadata,
            )
            signals.append(signal)
            self.add_signal(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """
        Calculate ATR-based position size.

        Uses the Turtle trading system's position sizing:
        Position Size = (Capital * Risk%) / ATR

        This ensures consistent risk per trade regardless of volatility.

        Args:
            signal: Trading signal.
            capital: Available capital.
            risk_per_trade: Maximum risk per trade (defaults to config value).

        Returns:
            Number of shares to trade.
        """
        risk = risk_per_trade or self._config.atr_position_risk

        # Get ATR from signal metadata
        atr = signal.metadata.get("atr", signal.price * 0.02)

        if atr <= 0 or signal.price <= 0:
            return 0.0

        # Dollar risk per trade
        risk_amount = capital * risk

        # Position size based on ATR (risk per unit = 1 ATR)
        # If we lose 1 ATR, we lose risk_amount
        position_value = risk_amount / (atr / signal.price)

        # Apply max position size limit
        max_value = capital * self._config.max_position_size
        position_value = min(position_value, max_value)

        # Adjust by signal strength (0.7 to 1.3 multiplier)
        strength_multiplier = 0.7 + abs(signal.strength) * 0.6
        position_value *= strength_multiplier

        # Calculate shares
        shares = position_value / signal.price

        return shares if signal.is_buy else -shares

    def get_breakout_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Get current breakout metrics for analysis.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            Dictionary with current indicator values.
        """
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]

        return {
            "entry_high": float(latest.get("entry_high", 0)),
            "entry_low": float(latest.get("entry_low", 0)),
            "exit_high": float(latest.get("exit_high", 0)),
            "exit_low": float(latest.get("exit_low", 0)),
            "atr": float(latest.get("atr", 0)),
            "atr_pct": float(latest.get("atr_pct", 0)),
            "volume_ratio": float(latest.get("volume_ratio", 1)),
            "breakout_up": bool(latest.get("breakout_up", False)),
            "breakdown": bool(latest.get("breakdown", False)),
            "breakout_strength": float(latest.get("breakout_strength", 0)),
            "above_ma_50": bool(latest.get("above_ma_50", False)),
        }


__all__ = ["BreakoutStrategy", "BreakoutConfig"]

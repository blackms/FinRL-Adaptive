"""
Trend Following Strategy Module

Implements a trend following strategy based on Turtle Trading rules,
using ADX for trend strength, Donchian channels for breakouts, and
ATR-based position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig


@dataclass
class TrendFollowingConfig(StrategyConfig):
    """
    Configuration for Trend Following Strategy.

    Attributes:
        donchian_entry_period: Period for Donchian channel entry (default: 20).
        donchian_exit_period: Period for Donchian channel exit (default: 10).
        atr_period: Period for ATR calculation (default: 14).
        adx_period: Period for ADX calculation (default: 14).
        adx_threshold: Minimum ADX value for trend confirmation (default: 25).
        macd_fast: MACD fast EMA period (default: 12).
        macd_slow: MACD slow EMA period (default: 26).
        macd_signal: MACD signal line period (default: 9).
        atr_multiplier: ATR multiplier for stop loss (default: 2.0).
        position_risk_pct: Risk per position as fraction of equity (default: 0.02).
        max_units: Maximum pyramid units per position (default: 4).
        unit_size_atr: ATR multiplier for unit sizing (default: 1.0).
        use_macd_filter: Whether to use MACD as trend filter (default: True).
        use_adx_filter: Whether to require ADX confirmation (default: True).
        trailing_stop_atr: ATR multiplier for trailing stop (default: 3.0).
    """

    name: str = "TrendFollowingStrategy"
    donchian_entry_period: int = 20
    donchian_exit_period: int = 10
    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 25.0
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_multiplier: float = 2.0
    position_risk_pct: float = 0.02
    max_units: int = 4
    unit_size_atr: float = 1.0
    use_macd_filter: bool = True
    use_adx_filter: bool = True
    trailing_stop_atr: float = 3.0
    min_data_points: int = 60


class TrendFollowingStrategy(BaseStrategy):
    """
    Trend following strategy based on Turtle Trading principles.

    Combines multiple indicators for trend identification and confirmation:
    - Donchian Channels: Breakout signals (20-day entry, 10-day exit)
    - ADX: Trend strength measurement
    - MACD: Trend direction confirmation
    - ATR: Position sizing and stop loss calculation

    Position sizing follows the Turtle system:
    - Risk-based: Position size = (Account Risk) / (ATR * Multiplier)
    - Pyramiding: Add to winning positions up to max_units

    Example:
        >>> config = TrendFollowingConfig(
        ...     donchian_entry_period=20,
        ...     donchian_exit_period=10,
        ...     adx_threshold=25,
        ... )
        >>> strategy = TrendFollowingStrategy(config)
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, config: TrendFollowingConfig | None = None) -> None:
        """
        Initialize the trend following strategy.

        Args:
            config: Strategy configuration. Defaults to TrendFollowingConfig.
        """
        super().__init__(config or TrendFollowingConfig())
        self._config: TrendFollowingConfig = self.config  # type: ignore
        self._unit_counts: dict[str, int] = {}
        self._entry_prices: dict[str, list[float]] = {}
        self._trailing_stops: dict[str, float] = {}

    def calculate_donchian_channel(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Donchian Channel.

        Upper band: Highest high over period
        Lower band: Lowest low over period
        Middle band: Average of upper and lower

        Args:
            high: High price series.
            low: Low price series.
            period: Lookback period.

        Returns:
            Tuple of (upper_band, lower_band, middle_band).
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2

        return upper, lower, middle

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int | None = None,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        True Range = max(high - low, abs(high - prev_close), abs(low - prev_close))
        ATR = EMA of True Range

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            period: ATR period. Defaults to config value.

        Returns:
            Series with ATR values.
        """
        period = period or self._config.atr_period

        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    def calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Average Directional Index (ADX).

        Measures trend strength regardless of direction.
        ADX > 25 indicates a strong trend.
        ADX < 20 indicates a weak/ranging market.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.
            period: ADX period. Defaults to config value.

        Returns:
            Tuple of (ADX, +DI, -DI).
        """
        period = period or self._config.adx_period

        # Calculate +DM and -DM
        up_move = high.diff()
        down_move = -low.diff()

        plus_dm = np.where(
            (up_move > down_move) & (up_move > 0),
            up_move,
            0,
        )
        minus_dm = np.where(
            (down_move > up_move) & (down_move > 0),
            down_move,
            0,
        )

        plus_dm = pd.Series(plus_dm, index=high.index)
        minus_dm = pd.Series(minus_dm, index=high.index)

        # Calculate ATR
        atr = self.calculate_atr(high, low, close, period)

        # Calculate smoothed +DI and -DI
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx, plus_di, minus_di

    def calculate_macd(
        self,
        close: pd.Series,
        fast: int | None = None,
        slow: int | None = None,
        signal: int | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD Line = Fast EMA - Slow EMA
        Signal Line = EMA of MACD Line
        Histogram = MACD Line - Signal Line

        Args:
            close: Close price series.
            fast: Fast EMA period. Defaults to config value.
            slow: Slow EMA period. Defaults to config value.
            signal: Signal line period. Defaults to config value.

        Returns:
            Tuple of (MACD line, signal line, histogram).
        """
        fast = fast or self._config.macd_fast
        slow = slow or self._config.macd_slow
        signal = signal or self._config.macd_signal

        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all trend following indicators.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with added indicator columns.
        """
        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # Donchian Channels (entry)
        df["dc_upper"], df["dc_lower"], df["dc_middle"] = self.calculate_donchian_channel(
            df["high"],
            df["low"],
            self._config.donchian_entry_period,
        )

        # Donchian Channels (exit)
        df["dc_exit_upper"], df["dc_exit_lower"], _ = self.calculate_donchian_channel(
            df["high"],
            df["low"],
            self._config.donchian_exit_period,
        )

        # ATR
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"])

        # ADX
        df["adx"], df["plus_di"], df["minus_di"] = self.calculate_adx(
            df["high"],
            df["low"],
            df["close"],
        )

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = self.calculate_macd(df["close"])

        # Breakout signals
        df["breakout_high"] = df["high"] >= df["dc_upper"].shift(1)
        df["breakout_low"] = df["low"] <= df["dc_lower"].shift(1)

        # Exit signals
        df["exit_long"] = df["low"] <= df["dc_exit_lower"].shift(1)
        df["exit_short"] = df["high"] >= df["dc_exit_upper"].shift(1)

        # Trend strength
        df["trend_strength"] = df["adx"] / 100  # Normalize to 0-1

        # Trend direction from MACD
        df["trend_bullish"] = df["macd"] > df["macd_signal"]
        df["trend_bearish"] = df["macd"] < df["macd_signal"]

        return df

    def _check_trend_filter(
        self,
        row: pd.Series,
        direction: str,
    ) -> bool:
        """
        Check if trend filters pass for a given direction.

        Args:
            row: DataFrame row with indicator values.
            direction: 'long' or 'short'.

        Returns:
            True if filters pass, False otherwise.
        """
        # ADX filter
        if self._config.use_adx_filter:
            if row["adx"] < self._config.adx_threshold:
                return False

        # MACD filter
        if self._config.use_macd_filter:
            if direction == "long" and not row["trend_bullish"]:
                return False
            if direction == "short" and not row["trend_bearish"]:
                return False

        # DI confirmation
        if direction == "long" and row["plus_di"] <= row["minus_di"]:
            return False
        if direction == "short" and row["minus_di"] <= row["plus_di"]:
            return False

        return True

    def _calculate_signal_strength(
        self,
        adx: float,
        atr_pct: float,
        macd_hist: float,
    ) -> float:
        """
        Calculate signal strength based on indicators.

        Args:
            adx: ADX value.
            atr_pct: ATR as percentage of price.
            macd_hist: MACD histogram value.

        Returns:
            Signal strength between 0 and 1.
        """
        # ADX component (higher ADX = stronger trend)
        adx_strength = min(adx / 50, 1.0)

        # MACD histogram magnitude
        macd_strength = min(abs(macd_hist) * 100, 1.0)

        # Combined strength
        strength = 0.6 * adx_strength + 0.4 * macd_strength

        return float(np.clip(strength, 0.1, 1.0))

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate trend following signals based on Turtle rules.

        Entry Rules (System 1):
        - Long: Price breaks above 20-day high
        - Short: Price breaks below 20-day low
        - Requires ADX > threshold and MACD confirmation

        Exit Rules:
        - Long exit: Price breaks below 10-day low
        - Short exit: Price breaks above 10-day high
        - Or trailing stop hit

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            List of Signal objects.
        """
        if not self.validate_data(data):
            return []

        signals: list[Signal] = []
        df = self.calculate_indicators(data)

        # Get symbol from data
        symbol = df.get("symbol", pd.Series(["UNKNOWN"])).iloc[-1]
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[-1]
        symbol = str(symbol)

        # Get latest row
        latest = df.iloc[-1]

        # Skip if indicators not ready
        if pd.isna(latest.get("atr")) or pd.isna(latest.get("adx")):
            return []

        # Get timestamp
        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        price = float(latest["close"])
        atr = float(latest["atr"])
        atr_pct = atr / price * 100

        # Calculate signal strength
        strength = self._calculate_signal_strength(
            adx=latest["adx"],
            atr_pct=atr_pct,
            macd_hist=latest["macd_hist"],
        )

        metadata: dict[str, Any] = {
            "atr": atr,
            "adx": latest["adx"],
            "plus_di": latest["plus_di"],
            "minus_di": latest["minus_di"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
            "dc_upper": latest["dc_upper"],
            "dc_lower": latest["dc_lower"],
        }

        has_position = self.has_position(symbol)
        position = self.get_position(symbol)

        # Check for exit signals first
        if has_position and position is not None:
            exit_signal = None

            if position.is_long:
                # Long exit: price below 10-day low or trailing stop
                trailing_stop = self._trailing_stops.get(symbol, 0)
                if latest["exit_long"] or (trailing_stop > 0 and price <= trailing_stop):
                    exit_signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.SELL,
                        strength=strength,
                        timestamp=timestamp,
                        price=price,
                        metadata={**metadata, "action": "exit_long"},
                    )

                # Update trailing stop
                new_stop = price - self._config.trailing_stop_atr * atr
                if new_stop > trailing_stop:
                    self._trailing_stops[symbol] = new_stop

            elif position.is_short:
                # Short exit: price above 10-day high or trailing stop
                trailing_stop = self._trailing_stops.get(symbol, float("inf"))
                if latest["exit_short"] or (trailing_stop < float("inf") and price >= trailing_stop):
                    exit_signal = Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=strength,
                        timestamp=timestamp,
                        price=price,
                        metadata={**metadata, "action": "exit_short"},
                    )

                # Update trailing stop
                new_stop = price + self._config.trailing_stop_atr * atr
                if new_stop < trailing_stop:
                    self._trailing_stops[symbol] = new_stop

            if exit_signal:
                signals.append(exit_signal)
                self.add_signal(exit_signal)
                # Clear tracking
                self._unit_counts.pop(symbol, None)
                self._entry_prices.pop(symbol, None)
                self._trailing_stops.pop(symbol, None)
                return signals

            # Check for pyramid (add to position)
            units = self._unit_counts.get(symbol, 0)
            if units < self._config.max_units:
                entry_prices = self._entry_prices.get(symbol, [])
                if entry_prices:
                    last_entry = entry_prices[-1]
                    # Add unit if price moved 1 ATR in favor
                    if position.is_long and price >= last_entry + atr:
                        pyramid_signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            strength=strength * 0.5,  # Reduced strength for pyramid
                            timestamp=timestamp,
                            price=price,
                            metadata={**metadata, "action": "pyramid_long", "unit": units + 1},
                        )
                        signals.append(pyramid_signal)
                        self.add_signal(pyramid_signal)
                        self._unit_counts[symbol] = units + 1
                        self._entry_prices[symbol].append(price)

                    elif position.is_short and price <= last_entry - atr:
                        pyramid_signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.SELL,
                            strength=-strength * 0.5,
                            timestamp=timestamp,
                            price=price,
                            metadata={**metadata, "action": "pyramid_short", "unit": units + 1},
                        )
                        signals.append(pyramid_signal)
                        self.add_signal(pyramid_signal)
                        self._unit_counts[symbol] = units + 1
                        self._entry_prices[symbol].append(price)

        else:
            # Check for entry signals
            if latest["breakout_high"] and self._check_trend_filter(latest, "long"):
                # Long entry
                entry_signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    timestamp=timestamp,
                    price=price,
                    metadata={**metadata, "action": "entry_long"},
                )
                signals.append(entry_signal)
                self.add_signal(entry_signal)
                # Initialize tracking
                self._unit_counts[symbol] = 1
                self._entry_prices[symbol] = [price]
                self._trailing_stops[symbol] = price - self._config.trailing_stop_atr * atr

            elif latest["breakout_low"] and self._check_trend_filter(latest, "short"):
                # Short entry
                entry_signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=-strength,
                    timestamp=timestamp,
                    price=price,
                    metadata={**metadata, "action": "entry_short"},
                )
                signals.append(entry_signal)
                self.add_signal(entry_signal)
                # Initialize tracking
                self._unit_counts[symbol] = 1
                self._entry_prices[symbol] = [price]
                self._trailing_stops[symbol] = price + self._config.trailing_stop_atr * atr

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """
        Calculate position size using ATR-based risk management.

        Turtle Position Sizing:
        Dollar Volatility = ATR * Price Point Value
        Unit Size = (Account * Risk%) / (ATR * Multiplier)

        Args:
            signal: Trading signal.
            capital: Available capital.
            risk_per_trade: Maximum risk per trade (defaults to config value).

        Returns:
            Number of shares to trade.
        """
        risk = risk_per_trade or self._config.position_risk_pct

        # Get ATR from signal metadata
        atr = signal.metadata.get("atr", signal.price * 0.02)  # Default 2% if no ATR

        # Calculate risk amount
        risk_amount = capital * risk

        # Position size based on ATR
        # We want: position_size * ATR * multiplier = risk_amount
        dollar_risk_per_share = atr * self._config.atr_multiplier

        if dollar_risk_per_share <= 0:
            return 0.0

        shares = risk_amount / dollar_risk_per_share

        # Cap at reasonable percentage of capital
        max_position = capital * 0.1 / signal.price  # Max 10% of capital per position
        shares = min(shares, max_position)

        # Return positive for buy, negative for sell
        return shares if signal.is_buy else -shares

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        is_long: bool,
    ) -> float:
        """
        Calculate initial stop loss based on ATR.

        Args:
            entry_price: Entry price.
            atr: Current ATR value.
            is_long: True for long position, False for short.

        Returns:
            Stop loss price.
        """
        atr_distance = atr * self._config.atr_multiplier

        if is_long:
            return entry_price - atr_distance
        else:
            return entry_price + atr_distance

    def get_trend_analysis(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Get current trend analysis for the data.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            Dictionary with trend analysis.
        """
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]

        adx = float(latest.get("adx", 0))
        plus_di = float(latest.get("plus_di", 0))
        minus_di = float(latest.get("minus_di", 0))

        # Determine trend state
        if adx < 20:
            trend_state = "ranging"
        elif adx < self._config.adx_threshold:
            trend_state = "weak_trend"
        else:
            trend_state = "strong_trend"

        # Determine trend direction
        if plus_di > minus_di:
            trend_direction = "bullish"
        elif minus_di > plus_di:
            trend_direction = "bearish"
        else:
            trend_direction = "neutral"

        return {
            "trend_state": trend_state,
            "trend_direction": trend_direction,
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
            "macd": float(latest.get("macd", 0)),
            "macd_signal": float(latest.get("macd_signal", 0)),
            "macd_histogram": float(latest.get("macd_hist", 0)),
            "atr": float(latest.get("atr", 0)),
            "dc_upper": float(latest.get("dc_upper", 0)),
            "dc_lower": float(latest.get("dc_lower", 0)),
            "dc_middle": float(latest.get("dc_middle", 0)),
        }


__all__ = ["TrendFollowingStrategy", "TrendFollowingConfig"]

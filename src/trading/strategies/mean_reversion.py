"""
Mean Reversion Strategy Module

Implements a mean reversion trading strategy using Bollinger Bands and z-score signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig, Position


@dataclass
class MeanReversionConfig(StrategyConfig):
    """
    Configuration for Mean Reversion Strategy.

    Attributes:
        bb_period: Period for Bollinger Bands calculation.
        bb_std_dev: Number of standard deviations for bands.
        zscore_period: Period for z-score calculation.
        zscore_entry_threshold: Z-score threshold for entry signals.
        zscore_exit_threshold: Z-score threshold for exit signals.
        use_atr_filter: Whether to use ATR filter for volatility.
        atr_period: Period for ATR calculation.
        atr_multiplier: ATR multiplier for stop loss.
        mean_type: Type of mean ('sma', 'ema').
        min_band_width: Minimum band width as percentage for trades.
    """

    name: str = "MeanReversionStrategy"
    bb_period: int = 20
    bb_std_dev: float = 2.0
    zscore_period: int = 20
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.5
    use_atr_filter: bool = True
    atr_period: int = 14
    atr_multiplier: float = 2.0
    mean_type: str = "sma"
    min_band_width: float = 2.0


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion trading strategy.

    Generates buy/sell signals based on:
    - Bollinger Bands (price touching or exceeding bands)
    - Z-score (statistical deviation from mean)
    - Optional ATR-based volatility filter

    The strategy assumes prices tend to revert to their mean and
    takes positions when prices deviate significantly.

    Example:
        >>> config = MeanReversionConfig(
        ...     bb_period=20,
        ...     bb_std_dev=2.0,
        ...     zscore_entry_threshold=2.0,
        ... )
        >>> strategy = MeanReversionStrategy(config)
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, config: MeanReversionConfig | None = None) -> None:
        """
        Initialize the mean reversion strategy.

        Args:
            config: Strategy configuration. Defaults to MeanReversionConfig.
        """
        super().__init__(config or MeanReversionConfig())
        self._config: MeanReversionConfig = self.config  # type: ignore

    def calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int | None = None,
        std_dev: float | None = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.

        Args:
            prices: Series of prices (typically close prices).
            period: Period for calculation. Defaults to config value.
            std_dev: Number of standard deviations. Defaults to config value.

        Returns:
            Tuple of (upper_band, middle_band, lower_band) as Series.
        """
        period = period or self._config.bb_period
        std_dev = std_dev or self._config.bb_std_dev

        if self._config.mean_type == "ema":
            middle = prices.ewm(span=period, adjust=False).mean()
        else:
            middle = prices.rolling(window=period).mean()

        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def calculate_zscore(
        self,
        prices: pd.Series,
        period: int | None = None,
    ) -> pd.Series:
        """
        Calculate z-score (number of standard deviations from mean).

        Args:
            prices: Series of prices.
            period: Period for calculation. Defaults to config value.

        Returns:
            Series with z-score values.
        """
        period = period or self._config.zscore_period

        mean = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        zscore = (prices - mean) / std

        return zscore

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

    def calculate_band_width(
        self,
        upper: pd.Series,
        lower: pd.Series,
        middle: pd.Series,
    ) -> pd.Series:
        """
        Calculate Bollinger Band width as percentage of middle band.

        Args:
            upper: Upper Bollinger Band.
            lower: Lower Bollinger Band.
            middle: Middle Bollinger Band.

        Returns:
            Series with band width percentages.
        """
        return (upper - lower) / middle * 100

    def calculate_percent_b(
        self,
        prices: pd.Series,
        upper: pd.Series,
        lower: pd.Series,
    ) -> pd.Series:
        """
        Calculate %B indicator (position within Bollinger Bands).

        %B = (Price - Lower Band) / (Upper Band - Lower Band)
        - %B > 1: Price above upper band
        - %B < 0: Price below lower band
        - %B = 0.5: Price at middle band

        Args:
            prices: Series of prices.
            upper: Upper Bollinger Band.
            lower: Lower Bollinger Band.

        Returns:
            Series with %B values.
        """
        return (prices - lower) / (upper - lower)

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all mean reversion indicators.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with added indicator columns.
        """
        df = data.copy()

        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # Bollinger Bands
        df["bb_upper"], df["bb_middle"], df["bb_lower"] = self.calculate_bollinger_bands(
            df["close"]
        )

        # Band width and %B
        df["band_width"] = self.calculate_band_width(
            df["bb_upper"], df["bb_lower"], df["bb_middle"]
        )
        df["percent_b"] = self.calculate_percent_b(
            df["close"], df["bb_upper"], df["bb_lower"]
        )

        # Z-score
        df["zscore"] = self.calculate_zscore(df["close"])

        # ATR
        if self._config.use_atr_filter:
            df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"])
            df["atr_pct"] = df["atr"] / df["close"] * 100
        else:
            df["atr"] = 0
            df["atr_pct"] = 0

        # Distance from mean (for signal strength)
        df["distance_from_mean"] = (df["close"] - df["bb_middle"]) / df["bb_middle"] * 100

        # Price momentum (for confirmation)
        df["price_momentum"] = df["close"].pct_change(periods=5)

        return df

    def _calculate_signal_strength(
        self,
        zscore: float,
        percent_b: float,
        distance_from_mean: float,
        band_width: float,
    ) -> float:
        """
        Calculate combined signal strength for mean reversion.

        Args:
            zscore: Current z-score.
            percent_b: Current %B value.
            distance_from_mean: Distance from mean as percentage.
            band_width: Current band width percentage.

        Returns:
            Signal strength between -1.0 and 1.0.
            Negative for oversold (buy), positive for overbought (sell).
        """
        # Z-score component (primary signal)
        # Oversold: zscore < -2 => signal = 1 (buy)
        # Overbought: zscore > 2 => signal = -1 (sell)
        zscore_signal = -np.clip(zscore / 3, -1, 1)

        # %B component
        # Below lower band: percent_b < 0 => signal = 1 (buy)
        # Above upper band: percent_b > 1 => signal = -1 (sell)
        if percent_b < 0:
            percent_b_signal = min(1.0, -percent_b)
        elif percent_b > 1:
            percent_b_signal = -min(1.0, percent_b - 1)
        else:
            percent_b_signal = 0.5 - percent_b  # 0 at upper, 0.5 at middle, 1 at lower

        # Band width filter (wider bands = stronger signals)
        width_multiplier = min(2.0, band_width / self._config.min_band_width)

        # Combined signal (weighted average)
        raw_signal = (
            0.6 * zscore_signal
            + 0.4 * percent_b_signal
        ) * width_multiplier

        return float(np.clip(raw_signal, -1.0, 1.0))

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate mean reversion trading signals.

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

        # Get the latest row
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Skip if indicators are not ready
        required_fields = ["zscore", "bb_upper", "bb_lower", "percent_b", "band_width"]
        if any(pd.isna(latest.get(f)) for f in required_fields):
            return []

        # Get timestamp
        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Check band width filter
        if latest["band_width"] < self._config.min_band_width:
            return []  # Low volatility, skip

        # Calculate signal strength
        strength = self._calculate_signal_strength(
            zscore=latest["zscore"],
            percent_b=latest["percent_b"],
            distance_from_mean=latest["distance_from_mean"],
            band_width=latest["band_width"],
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "zscore": latest["zscore"],
            "percent_b": latest["percent_b"],
            "bb_upper": latest["bb_upper"],
            "bb_middle": latest["bb_middle"],
            "bb_lower": latest["bb_lower"],
            "band_width": latest["band_width"],
            "atr": latest.get("atr", 0),
        }

        signal_type = SignalType.HOLD

        # Buy conditions (mean reversion from oversold)
        buy_conditions = [
            latest["zscore"] < -self._config.zscore_entry_threshold,
            latest["percent_b"] < 0,  # Below lower band
            latest["close"] < latest["bb_lower"],  # Price below lower band
        ]

        # Sell conditions (mean reversion from overbought)
        sell_conditions = [
            latest["zscore"] > self._config.zscore_entry_threshold,
            latest["percent_b"] > 1,  # Above upper band
            latest["close"] > latest["bb_upper"],  # Price above upper band
        ]

        # Exit conditions (reversion to mean)
        exit_long_conditions = [
            abs(latest["zscore"]) < self._config.zscore_exit_threshold,
            0.4 < latest["percent_b"] < 0.6,  # Near middle band
        ]

        exit_short_conditions = exit_long_conditions  # Same for shorts

        # Check for entry signals
        if any(buy_conditions) and strength > 0.3:
            signal_type = SignalType.BUY
            metadata["trigger"] = "oversold_reversal"
            metadata["conditions_met"] = sum(buy_conditions)

        elif any(sell_conditions) and strength < -0.3:
            signal_type = SignalType.SELL
            metadata["trigger"] = "overbought_reversal"
            metadata["conditions_met"] = sum(sell_conditions)

        # Check for exit signals on existing positions
        elif self.has_position(str(symbol)):
            position = self.get_position(str(symbol))
            if position:
                if position.is_long and any(exit_long_conditions):
                    signal_type = SignalType.SELL
                    metadata["trigger"] = "exit_long"
                    strength = -0.5  # Moderate exit signal

                elif position.is_short and any(exit_short_conditions):
                    signal_type = SignalType.BUY
                    metadata["trigger"] = "exit_short"
                    strength = 0.5

        # Create signal if not HOLD
        if signal_type != SignalType.HOLD:
            # Calculate stop loss using ATR
            if self._config.use_atr_filter and latest.get("atr", 0) > 0:
                if signal_type == SignalType.BUY:
                    stop_loss = latest["close"] - (latest["atr"] * self._config.atr_multiplier)
                else:
                    stop_loss = latest["close"] + (latest["atr"] * self._config.atr_multiplier)
                metadata["suggested_stop_loss"] = stop_loss

            signal = Signal(
                symbol=str(symbol),
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
        Calculate position size based on ATR-based risk.

        Uses ATR to determine position size such that the stop loss
        represents the maximum risk per trade.

        Args:
            signal: Trading signal.
            capital: Available capital.
            risk_per_trade: Maximum risk per trade (defaults to config value).

        Returns:
            Number of shares to trade.
        """
        risk = risk_per_trade or self._config.risk_per_trade
        risk_amount = capital * risk

        # Get stop loss from signal metadata
        stop_loss = signal.metadata.get("suggested_stop_loss")

        if stop_loss and signal.price > 0:
            # Risk per share
            risk_per_share = abs(signal.price - stop_loss)

            if risk_per_share > 0:
                shares = risk_amount / risk_per_share
            else:
                # Fallback to simple calculation
                shares = risk_amount / signal.price
        else:
            # Simple position sizing
            shares = risk_amount / signal.price if signal.price > 0 else 0

        # Adjust by signal strength
        shares *= 0.5 + abs(signal.strength) * 0.5

        return shares if signal.is_buy else -shares

    def get_reversion_metrics(self, data: pd.DataFrame) -> dict[str, float]:
        """
        Get current mean reversion metrics for analysis.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            Dictionary with current indicator values.
        """
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]

        return {
            "zscore": float(latest.get("zscore", 0)),
            "percent_b": float(latest.get("percent_b", 0.5)),
            "bb_upper": float(latest.get("bb_upper", 0)),
            "bb_middle": float(latest.get("bb_middle", 0)),
            "bb_lower": float(latest.get("bb_lower", 0)),
            "band_width": float(latest.get("band_width", 0)),
            "distance_from_mean": float(latest.get("distance_from_mean", 0)),
            "atr": float(latest.get("atr", 0)),
        }


__all__ = ["MeanReversionStrategy", "MeanReversionConfig"]

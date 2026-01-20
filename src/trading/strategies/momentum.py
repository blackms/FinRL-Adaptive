"""
Momentum Strategy Module

Implements a momentum-based trading strategy using RSI and moving average crossovers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig, Position


@dataclass
class MomentumConfig(StrategyConfig):
    """
    Configuration for Momentum Strategy.

    Attributes:
        rsi_period: Period for RSI calculation.
        rsi_overbought: RSI threshold for overbought condition.
        rsi_oversold: RSI threshold for oversold condition.
        fast_ma_period: Period for fast moving average.
        slow_ma_period: Period for slow moving average.
        ma_type: Type of moving average ('sma', 'ema', 'wma').
        signal_threshold: Minimum signal strength threshold.
        use_volume_confirmation: Whether to confirm signals with volume.
        volume_ma_period: Period for volume moving average.
    """

    name: str = "MomentumStrategy"
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    fast_ma_period: int = 10
    slow_ma_period: int = 30
    ma_type: str = "ema"
    signal_threshold: float = 0.3
    use_volume_confirmation: bool = True
    volume_ma_period: int = 20


class MomentumStrategy(BaseStrategy):
    """
    Momentum-based trading strategy.

    Generates buy/sell signals based on:
    - RSI indicator (overbought/oversold conditions)
    - Moving average crossovers (fast MA crosses above/below slow MA)
    - Optional volume confirmation

    Signal strength is calculated as a combination of RSI extremity
    and MA crossover strength.

    Example:
        >>> config = MomentumConfig(
        ...     rsi_period=14,
        ...     fast_ma_period=10,
        ...     slow_ma_period=30,
        ... )
        >>> strategy = MomentumStrategy(config)
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, config: MomentumConfig | None = None) -> None:
        """
        Initialize the momentum strategy.

        Args:
            config: Strategy configuration. Defaults to MomentumConfig.
        """
        super().__init__(config or MomentumConfig())
        self._config: MomentumConfig = self.config  # type: ignore

    def calculate_rsi(self, prices: pd.Series, period: int | None = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Series of prices (typically close prices).
            period: RSI period. Defaults to config value.

        Returns:
            Series with RSI values.
        """
        period = period or self._config.rsi_period

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def calculate_moving_average(
        self,
        data: pd.Series,
        period: int,
        ma_type: str | None = None,
    ) -> pd.Series:
        """
        Calculate moving average.

        Args:
            data: Series of values.
            period: Moving average period.
            ma_type: Type of MA ('sma', 'ema', 'wma'). Defaults to config value.

        Returns:
            Series with moving average values.
        """
        ma_type = ma_type or self._config.ma_type

        if ma_type == "sma":
            return data.rolling(window=period).mean()
        elif ma_type == "ema":
            return data.ewm(span=period, adjust=False).mean()
        elif ma_type == "wma":
            weights = np.arange(1, period + 1)
            return data.rolling(window=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum indicators.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with added indicator columns.
        """
        df = data.copy()

        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # RSI
        df["rsi"] = self.calculate_rsi(df["close"])

        # Moving averages
        df["fast_ma"] = self.calculate_moving_average(
            df["close"], self._config.fast_ma_period
        )
        df["slow_ma"] = self.calculate_moving_average(
            df["close"], self._config.slow_ma_period
        )

        # MA crossover
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        df["ma_diff_pct"] = df["ma_diff"] / df["slow_ma"] * 100
        df["ma_cross"] = np.sign(df["ma_diff"]).diff()

        # Volume analysis
        if self._config.use_volume_confirmation:
            df["volume_ma"] = df["volume"].rolling(
                window=self._config.volume_ma_period
            ).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]
        else:
            df["volume_ratio"] = 1.0

        # Momentum
        df["momentum"] = df["close"].pct_change(periods=self._config.fast_ma_period)

        return df

    def _calculate_signal_strength(
        self,
        rsi: float,
        ma_diff_pct: float,
        momentum: float,
        volume_ratio: float,
    ) -> float:
        """
        Calculate combined signal strength.

        Args:
            rsi: Current RSI value.
            ma_diff_pct: MA difference as percentage.
            momentum: Price momentum.
            volume_ratio: Volume relative to average.

        Returns:
            Signal strength between -1.0 and 1.0.
        """
        # RSI component (-1 to 1)
        if rsi >= self._config.rsi_overbought:
            rsi_signal = -((rsi - self._config.rsi_overbought) / (100 - self._config.rsi_overbought))
        elif rsi <= self._config.rsi_oversold:
            rsi_signal = (self._config.rsi_oversold - rsi) / self._config.rsi_oversold
        else:
            # Normalize between oversold and overbought
            mid = (self._config.rsi_overbought + self._config.rsi_oversold) / 2
            rsi_signal = (mid - rsi) / (self._config.rsi_overbought - self._config.rsi_oversold)

        # MA component (capped at +/- 1)
        ma_signal = np.clip(ma_diff_pct / 5, -1, 1)

        # Momentum component
        momentum_signal = np.clip(momentum * 10, -1, 1)

        # Volume confirmation (multiplier)
        volume_multiplier = min(volume_ratio, 2.0) if self._config.use_volume_confirmation else 1.0

        # Combined signal (weighted average)
        raw_signal = (
            0.4 * rsi_signal
            + 0.4 * ma_signal
            + 0.2 * momentum_signal
        ) * (0.5 + 0.5 * volume_multiplier)

        return float(np.clip(raw_signal, -1.0, 1.0))

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate momentum-based trading signals.

        Args:
            data: DataFrame with OHLCV data. Must include 'symbol' column
                  or process a single symbol.

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

        # Get the latest row for signal generation
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        # Skip if indicators are not ready
        if pd.isna(latest.get("rsi")) or pd.isna(latest.get("fast_ma")) or pd.isna(latest.get("slow_ma")):
            return []

        # Get timestamp
        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Calculate signal strength
        strength = self._calculate_signal_strength(
            rsi=latest["rsi"],
            ma_diff_pct=latest["ma_diff_pct"],
            momentum=latest.get("momentum", 0),
            volume_ratio=latest.get("volume_ratio", 1.0),
        )

        # Determine signal type based on conditions
        signal_type = SignalType.HOLD
        metadata: dict[str, Any] = {
            "rsi": latest["rsi"],
            "fast_ma": latest["fast_ma"],
            "slow_ma": latest["slow_ma"],
            "ma_diff_pct": latest["ma_diff_pct"],
            "volume_ratio": latest.get("volume_ratio", 1.0),
        }

        # Buy conditions
        buy_conditions = [
            latest["rsi"] < self._config.rsi_oversold,  # Oversold
            latest["ma_cross"] > 0,  # Bullish crossover
            latest["fast_ma"] > latest["slow_ma"] and prev["fast_ma"] <= prev["slow_ma"],  # Crossover
        ]

        # Sell conditions
        sell_conditions = [
            latest["rsi"] > self._config.rsi_overbought,  # Overbought
            latest["ma_cross"] < 0,  # Bearish crossover
            latest["fast_ma"] < latest["slow_ma"] and prev["fast_ma"] >= prev["slow_ma"],  # Crossover
        ]

        # Check for strong buy signal
        if any(buy_conditions) and strength > self._config.signal_threshold:
            signal_type = SignalType.BUY
            metadata["trigger"] = "buy"
            metadata["conditions_met"] = sum(buy_conditions)

        # Check for strong sell signal
        elif any(sell_conditions) and strength < -self._config.signal_threshold:
            signal_type = SignalType.SELL
            metadata["trigger"] = "sell"
            metadata["conditions_met"] = sum(sell_conditions)

        # Create signal if not HOLD
        if signal_type != SignalType.HOLD:
            signal = Signal(
                symbol=str(symbol),
                signal_type=signal_type,
                strength=abs(strength) if signal_type == SignalType.BUY else -abs(strength),
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
        Calculate position size based on signal strength and risk parameters.

        Uses a volatility-adjusted position sizing approach:
        - Stronger signals get larger positions
        - Higher volatility means smaller positions

        Args:
            signal: Trading signal.
            capital: Available capital.
            risk_per_trade: Maximum risk per trade (defaults to config value).

        Returns:
            Number of shares to trade.
        """
        risk = risk_per_trade or self._config.risk_per_trade

        # Base position size from risk
        risk_amount = capital * risk

        # Adjust by signal strength (0.5 to 1.5 multiplier)
        strength_multiplier = 0.5 + abs(signal.strength)

        # Position value
        position_value = risk_amount * strength_multiplier

        # Calculate shares
        if signal.price <= 0:
            return 0.0

        shares = position_value / signal.price

        # Return positive for buy, negative for sell
        return shares if signal.is_buy else -shares

    def get_current_momentum(self, data: pd.DataFrame) -> dict[str, float]:
        """
        Get current momentum indicators for analysis.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            Dictionary with current indicator values.
        """
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]

        return {
            "rsi": float(latest.get("rsi", 50)),
            "fast_ma": float(latest.get("fast_ma", 0)),
            "slow_ma": float(latest.get("slow_ma", 0)),
            "ma_diff_pct": float(latest.get("ma_diff_pct", 0)),
            "momentum": float(latest.get("momentum", 0)),
            "volume_ratio": float(latest.get("volume_ratio", 1)),
        }


__all__ = ["MomentumStrategy", "MomentumConfig"]

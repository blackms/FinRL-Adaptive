"""
Ensemble Strategy Module

Implements an ensemble trading strategy that combines multiple sub-strategies
with weighted voting to generate superior signals that beat buy-and-hold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig


@dataclass
class EnsembleConfig(StrategyConfig):
    """
    Configuration for Ensemble Strategy.

    Attributes:
        momentum_weight: Weight for momentum sub-strategy (0.40).
        breakout_weight: Weight for breakout sub-strategy (0.35).
        dip_buyer_weight: Weight for dip buyer sub-strategy (0.25).
        buy_threshold: Weighted signal threshold for BUY (0.3).
        sell_threshold: Weighted signal threshold for SELL (-0.3).
        min_market_exposure: Minimum market exposure as fraction (0.70).
        max_positions: Maximum number of positions to hold.
        rebalance_frequency: Rebalancing frequency in trading days (21 = monthly).
        trailing_stop_pct: Trailing stop loss percentage (0.12 = 12%).
        rsi_period: Period for RSI calculation.
        rsi_oversold: RSI threshold for oversold (dip buying).
        rsi_overbought: RSI threshold for overbought (momentum exit).
        fast_ma_period: Period for fast moving average.
        slow_ma_period: Period for slow moving average.
        breakout_period: Period for breakout detection (20-day high).
        breakdown_period: Period for breakdown detection (10-day low).
        volume_ma_period: Period for volume moving average.
    """

    name: str = "EnsembleStrategy"
    momentum_weight: float = 0.40
    breakout_weight: float = 0.35
    dip_buyer_weight: float = 0.25
    buy_threshold: float = 0.30
    sell_threshold: float = -0.30
    min_market_exposure: float = 0.70
    max_positions: int = 10
    rebalance_frequency: int = 21
    trailing_stop_pct: float = 0.12
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    fast_ma_period: int = 10
    slow_ma_period: int = 50
    breakout_period: int = 20
    breakdown_period: int = 10
    volume_ma_period: int = 20


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy combining multiple sub-strategies with weighted voting.

    Combines three distinct approaches:
    1. Momentum (weight: 40%) - RSI + MA crossover for trend following
    2. Breakout (weight: 35%) - 20-day high breakout (Turtle style)
    3. Dip Buyer (weight: 25%) - Buy when RSI < 30 in an uptrend

    Uses weighted voting to generate final signals:
    - final_signal = 0.4*momentum + 0.35*breakout + 0.25*dip_buyer
    - BUY when final_signal > 0.3
    - SELL when final_signal < -0.3
    - HOLD otherwise

    Key Features:
    - Minimum 70% market exposure to reduce cash drag
    - Equal-weight positions across selected stocks
    - Monthly rebalancing (21 trading days)
    - 12% trailing stops for risk management

    Example:
        >>> config = EnsembleConfig(
        ...     momentum_weight=0.40,
        ...     breakout_weight=0.35,
        ...     dip_buyer_weight=0.25,
        ... )
        >>> strategy = EnsembleStrategy(config)
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        """
        Initialize the ensemble strategy.

        Args:
            config: Strategy configuration. Defaults to EnsembleConfig.
        """
        super().__init__(config or EnsembleConfig())
        self._config: EnsembleConfig = self.config  # type: ignore
        self._trailing_highs: dict[str, float] = {}
        self._last_rebalance: datetime | None = None
        self._days_since_rebalance: int = 0

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

        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)

    def calculate_moving_averages(
        self,
        prices: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate fast and slow exponential moving averages.

        Args:
            prices: Series of close prices.

        Returns:
            Tuple of (fast_ma, slow_ma) as Series.
        """
        fast_ma = prices.ewm(span=self._config.fast_ma_period, adjust=False).mean()
        slow_ma = prices.ewm(span=self._config.slow_ma_period, adjust=False).mean()

        return fast_ma, slow_ma

    def calculate_donchian_channel(
        self,
        high: pd.Series,
        low: pd.Series,
        period: int,
    ) -> tuple[pd.Series, pd.Series]:
        """
        Calculate Donchian Channel (highest high and lowest low).

        Args:
            high: Series of high prices.
            low: Series of low prices.
            period: Lookback period.

        Returns:
            Tuple of (upper_channel, lower_channel) as Series.
        """
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()

        return upper, lower

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14,
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        Args:
            high: Series of high prices.
            low: Series of low prices.
            close: Series of close prices.
            period: Period for calculation.

        Returns:
            Series with ATR values.
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all indicators needed for the ensemble strategy.

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

        # Moving Averages
        df["fast_ma"], df["slow_ma"] = self.calculate_moving_averages(df["close"])
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        df["ma_diff_pct"] = df["ma_diff"] / df["slow_ma"] * 100
        df["ma_cross"] = np.sign(df["ma_diff"]).diff()

        # Donchian Channels for breakout
        df["breakout_high"], _ = self.calculate_donchian_channel(
            df["high"], df["low"], self._config.breakout_period
        )
        df["breakdown_high"], df["breakdown_low"] = self.calculate_donchian_channel(
            df["high"], df["low"], self._config.breakdown_period
        )

        # Previous period values for breakout detection
        df["prev_breakout_high"] = df["breakout_high"].shift(1)
        df["prev_breakdown_low"] = df["breakdown_low"].shift(1)

        # Breakout detection
        df["is_breakout"] = df["high"] > df["prev_breakout_high"]
        df["is_breakdown"] = df["low"] < df["prev_breakdown_low"]

        # ATR for volatility
        df["atr"] = self.calculate_atr(df["high"], df["low"], df["close"])
        df["atr_pct"] = df["atr"] / df["close"] * 100

        # Volume analysis
        df["volume_ma"] = df["volume"].rolling(window=self._config.volume_ma_period).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Trend determination (for dip buyer)
        df["in_uptrend"] = df["close"] > df["slow_ma"]
        df["trend_strength"] = (df["close"] - df["slow_ma"]) / df["slow_ma"] * 100

        # Price momentum
        df["momentum_5d"] = df["close"].pct_change(periods=5)
        df["momentum_20d"] = df["close"].pct_change(periods=20)

        return df

    def _calculate_momentum_signal(
        self,
        rsi: float,
        ma_diff_pct: float,
        momentum_20d: float,
    ) -> float:
        """
        Calculate momentum sub-strategy signal.

        Uses RSI + MA crossover for trend following.

        Args:
            rsi: Current RSI value.
            ma_diff_pct: MA difference as percentage.
            momentum_20d: 20-day momentum.

        Returns:
            Signal between -1.0 and 1.0.
        """
        # RSI component
        if rsi >= self._config.rsi_overbought:
            rsi_signal = -((rsi - self._config.rsi_overbought) / (100 - self._config.rsi_overbought))
        elif rsi <= self._config.rsi_oversold:
            rsi_signal = (self._config.rsi_oversold - rsi) / self._config.rsi_oversold
        else:
            # Neutral zone - slight bullish/bearish bias
            mid = (self._config.rsi_overbought + self._config.rsi_oversold) / 2
            rsi_signal = (rsi - mid) / (self._config.rsi_overbought - mid) * 0.3

        # MA crossover component (more weight on this)
        ma_signal = np.clip(ma_diff_pct / 3, -1, 1)

        # Momentum component
        momentum_signal = np.clip(momentum_20d * 5, -1, 1)

        # Combined momentum signal
        signal = 0.3 * rsi_signal + 0.5 * ma_signal + 0.2 * momentum_signal

        return float(np.clip(signal, -1.0, 1.0))

    def _calculate_breakout_signal(
        self,
        is_breakout: bool,
        is_breakdown: bool,
        price: float,
        prev_breakout_high: float,
        prev_breakdown_low: float,
        atr: float,
        volume_ratio: float,
    ) -> float:
        """
        Calculate breakout sub-strategy signal.

        Turtle-style 20-day high breakout, 10-day low breakdown.

        Args:
            is_breakout: Whether price broke above 20-day high.
            is_breakdown: Whether price broke below 10-day low.
            price: Current price.
            prev_breakout_high: Previous 20-day high.
            prev_breakdown_low: Previous 10-day low.
            atr: Current ATR.
            volume_ratio: Current volume / average.

        Returns:
            Signal between -1.0 and 1.0.
        """
        if is_breakout and prev_breakout_high > 0 and atr > 0:
            # Bullish breakout
            breakout_strength = (price - prev_breakout_high) / atr
            volume_multiplier = min(1.5, 0.7 + volume_ratio * 0.3)
            signal = min(1.0, breakout_strength * 0.5) * volume_multiplier
            return float(np.clip(signal, 0.0, 1.0))

        elif is_breakdown and prev_breakdown_low > 0 and atr > 0:
            # Bearish breakdown
            breakdown_strength = (prev_breakdown_low - price) / atr
            volume_multiplier = min(1.5, 0.7 + volume_ratio * 0.3)
            signal = -min(1.0, breakdown_strength * 0.5) * volume_multiplier
            return float(np.clip(signal, -1.0, 0.0))

        return 0.0

    def _calculate_dip_buyer_signal(
        self,
        rsi: float,
        in_uptrend: bool,
        trend_strength: float,
        momentum_5d: float,
    ) -> float:
        """
        Calculate dip buyer sub-strategy signal.

        Buy when RSI < 30 in an overall uptrend.

        Args:
            rsi: Current RSI value.
            in_uptrend: Whether price is above slow MA.
            trend_strength: Trend strength as percentage.
            momentum_5d: 5-day momentum.

        Returns:
            Signal between -1.0 and 1.0.
        """
        # Only active in uptrend
        if not in_uptrend:
            return 0.0

        # Strong buy signal when RSI is oversold in uptrend
        if rsi < self._config.rsi_oversold:
            # Calculate signal strength based on how oversold
            oversold_depth = (self._config.rsi_oversold - rsi) / self._config.rsi_oversold

            # Stronger signal if trend is strong
            trend_multiplier = min(1.3, 1.0 + trend_strength / 20)

            # Check if momentum is turning (less negative = better)
            momentum_boost = 0.2 if momentum_5d > -0.05 else 0.0

            signal = (oversold_depth + momentum_boost) * trend_multiplier
            return float(np.clip(signal, 0.0, 1.0))

        # Mild buy signal when RSI recovering from oversold
        elif rsi < 40 and momentum_5d > 0:
            signal = (40 - rsi) / 40 * 0.3  # Weaker signal
            return float(np.clip(signal, 0.0, 0.5))

        return 0.0

    def calculate_ensemble_signal(
        self,
        momentum_signal: float,
        breakout_signal: float,
        dip_buyer_signal: float,
    ) -> float:
        """
        Calculate the weighted ensemble signal.

        Args:
            momentum_signal: Signal from momentum strategy (-1 to 1).
            breakout_signal: Signal from breakout strategy (-1 to 1).
            dip_buyer_signal: Signal from dip buyer strategy (0 to 1).

        Returns:
            Weighted ensemble signal (-1 to 1).
        """
        ensemble_signal = (
            self._config.momentum_weight * momentum_signal
            + self._config.breakout_weight * breakout_signal
            + self._config.dip_buyer_weight * dip_buyer_signal
        )

        return float(np.clip(ensemble_signal, -1.0, 1.0))

    def update_trailing_stop(
        self,
        symbol: str,
        current_price: float,
    ) -> float | None:
        """
        Update trailing stop based on highest price seen.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.

        Returns:
            Trailing stop price or None.
        """
        # Update highest price
        if symbol not in self._trailing_highs:
            self._trailing_highs[symbol] = current_price
        else:
            self._trailing_highs[symbol] = max(self._trailing_highs[symbol], current_price)

        # Calculate trailing stop (12% below high)
        high = self._trailing_highs[symbol]
        stop = high * (1 - self._config.trailing_stop_pct)

        return stop

    def check_trailing_stop(
        self,
        symbol: str,
        current_price: float,
    ) -> bool:
        """
        Check if trailing stop has been triggered.

        Args:
            symbol: Ticker symbol.
            current_price: Current market price.

        Returns:
            True if stop triggered.
        """
        if symbol not in self._trailing_highs:
            return False

        high = self._trailing_highs[symbol]
        stop = high * (1 - self._config.trailing_stop_pct)

        return current_price <= stop

    def should_rebalance(self, timestamp: datetime) -> bool:
        """
        Check if portfolio should be rebalanced.

        Args:
            timestamp: Current timestamp.

        Returns:
            True if rebalance needed.
        """
        if self._last_rebalance is None:
            self._last_rebalance = timestamp
            return True

        # Simple day counting (could be enhanced with actual trading day calendar)
        self._days_since_rebalance += 1

        if self._days_since_rebalance >= self._config.rebalance_frequency:
            self._last_rebalance = timestamp
            self._days_since_rebalance = 0
            return True

        return False

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate ensemble trading signals using weighted voting.

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
        required_fields = ["rsi", "fast_ma", "slow_ma", "breakout_high", "atr"]
        if any(pd.isna(latest.get(f)) for f in required_fields):
            return []

        # Get timestamp
        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Calculate sub-strategy signals
        momentum_signal = self._calculate_momentum_signal(
            rsi=latest["rsi"],
            ma_diff_pct=latest["ma_diff_pct"],
            momentum_20d=latest.get("momentum_20d", 0),
        )

        breakout_signal = self._calculate_breakout_signal(
            is_breakout=bool(latest.get("is_breakout", False)),
            is_breakdown=bool(latest.get("is_breakdown", False)),
            price=latest["close"],
            prev_breakout_high=latest.get("prev_breakout_high", 0),
            prev_breakdown_low=latest.get("prev_breakdown_low", 0),
            atr=latest["atr"],
            volume_ratio=latest.get("volume_ratio", 1.0),
        )

        dip_buyer_signal = self._calculate_dip_buyer_signal(
            rsi=latest["rsi"],
            in_uptrend=bool(latest.get("in_uptrend", False)),
            trend_strength=latest.get("trend_strength", 0),
            momentum_5d=latest.get("momentum_5d", 0),
        )

        # Calculate ensemble signal
        ensemble_signal = self.calculate_ensemble_signal(
            momentum_signal=momentum_signal,
            breakout_signal=breakout_signal,
            dip_buyer_signal=dip_buyer_signal,
        )

        # Build metadata
        metadata: dict[str, Any] = {
            "momentum_signal": momentum_signal,
            "breakout_signal": breakout_signal,
            "dip_buyer_signal": dip_buyer_signal,
            "ensemble_signal": ensemble_signal,
            "rsi": latest["rsi"],
            "ma_diff_pct": latest["ma_diff_pct"],
            "in_uptrend": bool(latest.get("in_uptrend", False)),
            "is_breakout": bool(latest.get("is_breakout", False)),
            "is_breakdown": bool(latest.get("is_breakdown", False)),
            "atr": latest["atr"],
            "volume_ratio": latest.get("volume_ratio", 1.0),
        }

        signal_type = SignalType.HOLD
        strength = ensemble_signal

        # Check if we have an existing position
        has_position = self.has_position(symbol)
        position = self.get_position(symbol) if has_position else None

        if has_position and position:
            # Update trailing stop
            trailing_stop = self.update_trailing_stop(symbol, latest["close"])
            metadata["trailing_stop"] = trailing_stop

            # Check trailing stop
            if self.check_trailing_stop(symbol, latest["close"]):
                signal_type = SignalType.SELL
                strength = -0.8
                metadata["trigger"] = "trailing_stop"

                # Clear trailing high
                if symbol in self._trailing_highs:
                    del self._trailing_highs[symbol]

            # Check strong sell signal
            elif ensemble_signal < self._config.sell_threshold:
                signal_type = SignalType.SELL
                metadata["trigger"] = "ensemble_sell"

        else:
            # No position - check buy conditions
            if ensemble_signal > self._config.buy_threshold:
                signal_type = SignalType.BUY
                metadata["trigger"] = "ensemble_buy"

                # Identify which sub-strategy contributed most
                contributions = [
                    ("momentum", self._config.momentum_weight * momentum_signal),
                    ("breakout", self._config.breakout_weight * breakout_signal),
                    ("dip_buyer", self._config.dip_buyer_weight * dip_buyer_signal),
                ]
                metadata["primary_driver"] = max(contributions, key=lambda x: x[1])[0]

        # Check rebalance trigger
        if self.should_rebalance(timestamp):
            metadata["rebalance_triggered"] = True

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
        Calculate equal-weight position size with minimum exposure.

        Ensures minimum 70% market exposure and equal weights.

        Args:
            signal: Trading signal.
            capital: Available capital.
            risk_per_trade: Maximum risk per trade (defaults to config value).

        Returns:
            Number of shares to trade.
        """
        # Equal weight across max positions
        weight_per_position = 1.0 / self._config.max_positions

        # Apply minimum market exposure
        min_capital_deployed = capital * self._config.min_market_exposure
        position_value = max(
            capital * weight_per_position,
            min_capital_deployed / self._config.max_positions,
        )

        # Adjust by signal strength (0.7 to 1.2 multiplier)
        strength_multiplier = 0.7 + abs(signal.strength) * 0.5
        position_value *= strength_multiplier

        # Cap at max position size
        max_position = capital * 0.15  # 15% max in single position
        position_value = min(position_value, max_position)

        # Calculate shares
        if signal.price <= 0:
            return 0.0

        shares = position_value / signal.price

        return shares if signal.is_buy else -shares

    def get_ensemble_metrics(self, data: pd.DataFrame) -> dict[str, Any]:
        """
        Get current ensemble metrics for analysis.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            Dictionary with current indicator values and signals.
        """
        df = self.calculate_indicators(data)
        latest = df.iloc[-1]

        momentum_signal = self._calculate_momentum_signal(
            rsi=latest.get("rsi", 50),
            ma_diff_pct=latest.get("ma_diff_pct", 0),
            momentum_20d=latest.get("momentum_20d", 0),
        )

        breakout_signal = self._calculate_breakout_signal(
            is_breakout=bool(latest.get("is_breakout", False)),
            is_breakdown=bool(latest.get("is_breakdown", False)),
            price=latest.get("close", 0),
            prev_breakout_high=latest.get("prev_breakout_high", 0),
            prev_breakdown_low=latest.get("prev_breakdown_low", 0),
            atr=latest.get("atr", 0),
            volume_ratio=latest.get("volume_ratio", 1.0),
        )

        dip_buyer_signal = self._calculate_dip_buyer_signal(
            rsi=latest.get("rsi", 50),
            in_uptrend=bool(latest.get("in_uptrend", False)),
            trend_strength=latest.get("trend_strength", 0),
            momentum_5d=latest.get("momentum_5d", 0),
        )

        ensemble_signal = self.calculate_ensemble_signal(
            momentum_signal, breakout_signal, dip_buyer_signal
        )

        return {
            "rsi": float(latest.get("rsi", 50)),
            "fast_ma": float(latest.get("fast_ma", 0)),
            "slow_ma": float(latest.get("slow_ma", 0)),
            "ma_diff_pct": float(latest.get("ma_diff_pct", 0)),
            "in_uptrend": bool(latest.get("in_uptrend", False)),
            "is_breakout": bool(latest.get("is_breakout", False)),
            "is_breakdown": bool(latest.get("is_breakdown", False)),
            "atr": float(latest.get("atr", 0)),
            "atr_pct": float(latest.get("atr_pct", 0)),
            "volume_ratio": float(latest.get("volume_ratio", 1)),
            "momentum_signal": momentum_signal,
            "breakout_signal": breakout_signal,
            "dip_buyer_signal": dip_buyer_signal,
            "ensemble_signal": ensemble_signal,
            "recommendation": (
                "BUY" if ensemble_signal > self._config.buy_threshold
                else "SELL" if ensemble_signal < self._config.sell_threshold
                else "HOLD"
            ),
        }


__all__ = ["EnsembleStrategy", "EnsembleConfig"]

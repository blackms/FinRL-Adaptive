"""
Market Regime Detection Module

Implements a market regime detection system that classifies market conditions into
distinct regimes (Bull Trending, Bear Crisis, Sideways Neutral, High Volatility)
using a combination of trend, volatility, and momentum indicators.

The detector uses multiple technical indicators to provide robust regime classification:
- Trend: 20/50/200 SMA slopes, ADX for trend strength
- Volatility: 20-day realized volatility vs historical percentile, ATR
- Momentum: RSI, MACD histogram
- Market breadth proxy: Price position relative to moving averages

Includes smoothing to prevent rapid regime switches (minimum 5-day hold).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class RegimeType(Enum):
    """
    Market regime classifications.

    Each regime represents a distinct market condition that may require
    different trading strategies and risk management approaches.

    Primary regimes (used in detection logic):
    - BULL_TRENDING: Strong upward trend
    - BEAR_CRISIS: Strong downward trend
    - SIDEWAYS_NEUTRAL: Range-bound market
    - HIGH_VOLATILITY: Elevated volatility (>90th percentile)

    Additional regimes (for compatibility and extended use):
    - LOW_VOLATILITY: Unusually calm market
    - UNKNOWN: Unable to classify
    """

    BULL_TRENDING = "bull_trending"
    """Strong upward trend with positive momentum and moderate volatility."""

    BEAR_CRISIS = "bear_crisis"
    """Strong downward trend, often with elevated volatility and fear."""

    SIDEWAYS_NEUTRAL = "sideways_neutral"
    """Range-bound market with no clear directional bias."""

    HIGH_VOLATILITY = "high_volatility"
    """Extremely elevated volatility (>90th percentile), regardless of direction."""

    LOW_VOLATILITY = "low_volatility"
    """Unusually calm market with volatility below 10th percentile."""

    UNKNOWN = "unknown"
    """Unable to classify regime due to insufficient data or conflicting signals."""


@dataclass
class RegimeChange:
    """
    Records a regime change event.

    Attributes:
        timestamp: When the regime change occurred.
        previous_regime: The regime before the change.
        new_regime: The regime after the change.
        confidence: Confidence level of the new regime classification.
        indicators: Key indicator values at the time of change.
    """

    timestamp: datetime
    previous_regime: RegimeType
    new_regime: RegimeType
    confidence: float
    indicators: dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeDetectorConfig:
    """
    Configuration for the RegimeDetector.

    Attributes:
        volatility_lookback: Lookback period for volatility calculation (default: 20).
        volatility_history: Historical period for volatility percentile (default: 252).
        volatility_high_percentile: Percentile threshold for high volatility (default: 90).
        strong_trend_threshold: Threshold for strong trend classification (default: 0.6).
        sma_short: Short-term SMA period (default: 20).
        sma_medium: Medium-term SMA period (default: 50).
        sma_long: Long-term SMA period (default: 200).
        rsi_period: RSI calculation period (default: 14).
        atr_period: ATR calculation period (default: 14).
        adx_period: ADX calculation period (default: 14).
        macd_fast: MACD fast EMA period (default: 12).
        macd_slow: MACD slow EMA period (default: 26).
        macd_signal: MACD signal line period (default: 9).
        min_hold_days: Minimum days to hold a regime before switching (default: 5).
        smoothing_window: Window for smoothing regime signals (default: 3).
    """

    volatility_lookback: int = 20
    volatility_history: int = 252
    volatility_high_percentile: float = 80.0  # Optimized: was 90.0, detect regimes earlier
    volatility_low_percentile: float = 10.0   # New: for low volatility detection
    strong_trend_threshold: float = 0.5       # Optimized: was 0.6, more sensitive to trends
    adx_trend_threshold: float = 35.0         # New: require stronger ADX confirmation
    sma_short: int = 20
    sma_medium: int = 50
    sma_long: int = 200
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    min_hold_days: int = 3                    # Optimized: was 5, faster regime switching
    smoothing_window: int = 3


class RegimeDetector:
    """
    Market Regime Detection System.

    Classifies market conditions into distinct regimes using a combination
    of trend, volatility, and momentum indicators. Implements smoothing
    to prevent rapid regime switches (minimum 5-day hold).

    Classification Logic:
        1. If volatility > 90th percentile -> HIGH_VOLATILITY
        2. Elif trend_strength > 0.6 and direction > 0 -> BULL_TRENDING
        3. Elif trend_strength > 0.6 and direction < 0 -> BEAR_CRISIS
        4. Else -> SIDEWAYS_NEUTRAL

    Indicators Used:
        - Trend: 20/50/200 SMA slopes, ADX
        - Volatility: 20-day realized vol vs historical percentile, ATR
        - Momentum: RSI, MACD histogram
        - Market breadth proxy: Price vs moving averages

    Example:
        >>> detector = RegimeDetector()
        >>> regime = detector.detect_regime(prices_df, lookback=60)
        >>> print(f"Current regime: {regime.value}")
        >>> confidence = detector.get_regime_confidence()
        >>> print(f"Confidence: {confidence:.2%}")
        >>> history = detector.get_regime_history()
        >>> print(f"Total regime changes: {len(history)}")
    """

    def __init__(self, config: RegimeDetectorConfig | None = None) -> None:
        """
        Initialize the RegimeDetector.

        Args:
            config: Configuration for the detector. Defaults to RegimeDetectorConfig.
        """
        self.config = config or RegimeDetectorConfig()
        self._current_regime: RegimeType = RegimeType.SIDEWAYS_NEUTRAL
        self._regime_confidence: float = 0.5
        self._regime_history: list[RegimeChange] = []
        self._last_regime_change: datetime | None = None
        self._days_in_regime: int = 0
        self._indicator_cache: dict[str, float] = {}

    @property
    def current_regime(self) -> RegimeType:
        """Get the current detected regime."""
        return self._current_regime

    @property
    def days_in_current_regime(self) -> int:
        """Get the number of days in the current regime."""
        return self._days_in_regime

    def detect_regime(
        self,
        prices_df: pd.DataFrame,
        lookback: int = 60,
    ) -> RegimeType:
        """
        Detect the current market regime from price data.

        Analyzes multiple technical indicators to classify the market into
        one of four regimes: BULL_TRENDING, BEAR_CRISIS, SIDEWAYS_NEUTRAL,
        or HIGH_VOLATILITY.

        Args:
            prices_df: DataFrame with OHLCV data. Expected columns:
                - open: Open price
                - high: High price
                - low: Low price
                - close: Close price
                - volume: Trading volume (optional)
            lookback: Number of periods to use for analysis (default: 60).

        Returns:
            RegimeType indicating the current market regime.

        Raises:
            ValueError: If prices_df is empty or missing required columns.
        """
        if prices_df is None or prices_df.empty:
            raise ValueError("prices_df cannot be empty")

        # Normalize column names
        df = prices_df.copy()
        df.columns = [col.lower() for col in df.columns]

        # Validate required columns
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Use the specified lookback or available data
        if len(df) < lookback:
            lookback = len(df)

        # Calculate all indicators
        indicators = self._calculate_all_indicators(df)
        self._indicator_cache = indicators

        # Determine raw regime classification
        raw_regime = self._classify_regime(indicators)

        # Calculate confidence
        confidence = self._calculate_confidence(indicators, raw_regime)

        # Apply smoothing and minimum hold period
        final_regime = self._apply_smoothing(raw_regime, confidence, df)

        # Update state
        self._update_state(final_regime, confidence, indicators, df)

        return final_regime

    def get_regime_confidence(self) -> float:
        """
        Get the confidence level of the current regime classification.

        Returns:
            Float between 0.0 and 1.0 indicating confidence level.
            Higher values indicate stronger regime signals.
        """
        return self._regime_confidence

    def get_regime_history(self) -> list[RegimeChange]:
        """
        Get the history of regime changes.

        Returns:
            List of RegimeChange objects ordered chronologically,
            containing timestamp, previous/new regime, confidence,
            and indicator values at each regime change.
        """
        return self._regime_history.copy()

    def get_indicator_values(self) -> dict[str, float]:
        """
        Get the latest indicator values used in regime detection.

        Returns:
            Dictionary mapping indicator names to their values.
        """
        return self._indicator_cache.copy()

    def _calculate_all_indicators(self, df: pd.DataFrame) -> dict[str, float]:
        """
        Calculate all indicators needed for regime detection.

        Computes:
        - 20/50/200 SMA and their slopes
        - ADX (trend strength)
        - 20-day realized volatility and historical percentile
        - ATR
        - RSI
        - MACD histogram
        - Price position relative to moving averages

        Args:
            df: DataFrame with OHLCV data (lowercase columns).

        Returns:
            Dictionary with all calculated indicator values.
        """
        close = df["close"]
        high = df["high"]
        low = df["low"]

        indicators: dict[str, float] = {}

        # Moving Averages (20/50/200 SMA)
        sma_20 = close.rolling(window=self.config.sma_short).mean()
        sma_50 = close.rolling(window=self.config.sma_medium).mean()
        sma_200 = close.rolling(window=min(self.config.sma_long, len(df))).mean()

        indicators["sma_20"] = float(sma_20.iloc[-1]) if not pd.isna(sma_20.iloc[-1]) else float(close.iloc[-1])
        indicators["sma_50"] = float(sma_50.iloc[-1]) if not pd.isna(sma_50.iloc[-1]) else float(close.iloc[-1])
        indicators["sma_200"] = float(sma_200.iloc[-1]) if not pd.isna(sma_200.iloc[-1]) else float(close.iloc[-1])

        # SMA Slopes (normalized) - trend indicators
        indicators["sma_20_slope"] = self._calculate_slope(sma_20, periods=5)
        indicators["sma_50_slope"] = self._calculate_slope(sma_50, periods=10)
        indicators["sma_200_slope"] = self._calculate_slope(sma_200, periods=20)

        # Price position relative to MAs (market breadth proxy)
        current_price = float(close.iloc[-1])
        indicators["price_vs_sma_20"] = (current_price - indicators["sma_20"]) / indicators["sma_20"] if indicators["sma_20"] > 0 else 0
        indicators["price_vs_sma_50"] = (current_price - indicators["sma_50"]) / indicators["sma_50"] if indicators["sma_50"] > 0 else 0
        indicators["price_vs_sma_200"] = (current_price - indicators["sma_200"]) / indicators["sma_200"] if indicators["sma_200"] > 0 else 0

        # ADX (trend strength indicator)
        adx, plus_di, minus_di = self._calculate_adx(high, low, close)
        indicators["adx"] = adx
        indicators["plus_di"] = plus_di
        indicators["minus_di"] = minus_di

        # Volatility - 20-day realized vs historical percentile
        realized_vol = self._calculate_realized_volatility(close, self.config.volatility_lookback)
        historical_vol_percentile = self._calculate_volatility_percentile(
            close,
            self.config.volatility_lookback,
            self.config.volatility_history
        )
        indicators["realized_volatility"] = realized_vol
        indicators["volatility_percentile"] = historical_vol_percentile

        # ATR (volatility indicator)
        atr = self._calculate_atr(high, low, close)
        indicators["atr"] = atr
        indicators["atr_percent"] = (atr / current_price * 100) if current_price > 0 else 0

        # RSI (momentum indicator)
        rsi = self._calculate_rsi(close)
        indicators["rsi"] = rsi

        # MACD histogram (momentum indicator)
        macd_line, signal_line, histogram = self._calculate_macd(close)
        indicators["macd"] = macd_line
        indicators["macd_signal"] = signal_line
        indicators["macd_histogram"] = histogram

        # Trend Direction Score (-1 to 1)
        indicators["trend_direction"] = self._calculate_trend_direction(indicators)

        # Trend Strength Score (0 to 1)
        indicators["trend_strength"] = self._calculate_trend_strength(indicators)

        return indicators

    def _calculate_slope(self, series: pd.Series, periods: int = 5) -> float:
        """
        Calculate the normalized slope of a series.

        Args:
            series: Price or indicator series.
            periods: Number of periods for slope calculation.

        Returns:
            Normalized slope value between -1 and 1.
        """
        if len(series) < periods + 1:
            return 0.0

        # Get valid values
        valid_series = series.dropna()
        if len(valid_series) < periods + 1:
            return 0.0

        recent = valid_series.iloc[-periods:]
        if len(recent) < 2:
            return 0.0

        # Calculate percentage change over the period
        start_val = float(recent.iloc[0])
        end_val = float(recent.iloc[-1])

        if start_val == 0 or pd.isna(start_val) or pd.isna(end_val):
            return 0.0

        slope = (end_val - start_val) / start_val

        # Normalize to roughly -1 to 1 range
        return float(np.clip(slope * 10, -1, 1))

    def _calculate_adx(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> tuple[float, float, float]:
        """
        Calculate Average Directional Index (ADX).

        ADX measures trend strength regardless of direction.
        ADX > 25 indicates a strong trend.
        ADX < 20 indicates a weak/ranging market.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.

        Returns:
            Tuple of (ADX, +DI, -DI) values.
        """
        period = self.config.adx_period

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

        # Calculate True Range and ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        # Prevent division by zero
        atr_safe = atr.replace(0, np.nan).fillna(1e-10)

        # Calculate smoothed +DI and -DI
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_safe)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_safe)

        # Calculate DX and ADX
        di_sum = plus_di + minus_di
        di_sum_safe = di_sum.replace(0, np.nan).fillna(1e-10)
        dx = 100 * abs(plus_di - minus_di) / di_sum_safe
        adx = dx.ewm(span=period, adjust=False).mean()

        # Get latest values
        latest_adx = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 0.0
        latest_plus_di = float(plus_di.iloc[-1]) if not pd.isna(plus_di.iloc[-1]) else 0.0
        latest_minus_di = float(minus_di.iloc[-1]) if not pd.isna(minus_di.iloc[-1]) else 0.0

        return latest_adx, latest_plus_di, latest_minus_di

    def _calculate_realized_volatility(
        self,
        close: pd.Series,
        lookback: int,
    ) -> float:
        """
        Calculate 20-day realized volatility (annualized).

        Args:
            close: Close price series.
            lookback: Number of periods for calculation.

        Returns:
            Annualized volatility as a percentage.
        """
        returns = close.pct_change().dropna()

        if len(returns) < lookback:
            lookback = len(returns)

        if lookback < 2:
            return 0.0

        recent_returns = returns.iloc[-lookback:]
        daily_vol = float(recent_returns.std())

        # Annualize (assuming 252 trading days)
        annual_vol = daily_vol * np.sqrt(252) * 100

        return annual_vol

    def _calculate_volatility_percentile(
        self,
        close: pd.Series,
        vol_lookback: int,
        history_lookback: int,
    ) -> float:
        """
        Calculate current volatility's percentile rank in historical context.

        This determines if current volatility is elevated relative to history.
        Used for HIGH_VOLATILITY regime detection (>90th percentile).

        Args:
            close: Close price series.
            vol_lookback: Period for calculating current volatility (20 days).
            history_lookback: Historical period for percentile calculation (252 days).

        Returns:
            Percentile rank (0-100) of current volatility.
        """
        returns = close.pct_change().dropna()

        if len(returns) < vol_lookback + 10:
            return 50.0  # Default to median if insufficient data

        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=vol_lookback).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) < 2:
            return 50.0

        # Get historical window
        history_start = max(0, len(rolling_vol) - history_lookback)
        historical_vols = rolling_vol.iloc[history_start:]

        # Current volatility
        current_vol = float(rolling_vol.iloc[-1])

        # Calculate percentile
        percentile = float((historical_vols < current_vol).sum() / len(historical_vols) * 100)

        return percentile

    def _calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
    ) -> float:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by considering the full range
        of price movement including gaps.

        Args:
            high: High price series.
            low: Low price series.
            close: Close price series.

        Returns:
            Current ATR value.
        """
        period = self.config.atr_period

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()

        return float(atr.iloc[-1]) if not pd.isna(atr.iloc[-1]) else 0.0

    def _calculate_rsi(self, close: pd.Series) -> float:
        """
        Calculate Relative Strength Index (RSI).

        RSI measures momentum and overbought/oversold conditions.

        Args:
            close: Close price series.

        Returns:
            Current RSI value (0-100).
        """
        period = self.config.rsi_period

        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Prevent division by zero
        loss_safe = loss.replace(0, np.nan).fillna(1e-10)
        rs = gain / loss_safe
        rsi = 100 - (100 / (1 + rs))

        return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50.0

    def _calculate_macd(
        self,
        close: pd.Series,
    ) -> tuple[float, float, float]:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        MACD histogram is used as a momentum indicator.

        Args:
            close: Close price series.

        Returns:
            Tuple of (MACD line, signal line, histogram) values.
        """
        fast = self.config.macd_fast
        slow = self.config.macd_slow
        signal_period = self.config.macd_signal

        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        latest_macd = float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0.0
        latest_signal = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
        latest_histogram = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0

        return latest_macd, latest_signal, latest_histogram

    def _calculate_trend_direction(self, indicators: dict[str, float]) -> float:
        """
        Calculate overall trend direction score.

        Combines SMA slopes, DI direction, and MACD to determine
        whether the market is trending up or down.

        Args:
            indicators: Dictionary of calculated indicators.

        Returns:
            Score from -1 (strongly bearish) to 1 (strongly bullish).
        """
        scores = []
        weights = []

        # SMA slopes (20/50/200)
        scores.append(indicators.get("sma_20_slope", 0))
        weights.append(0.2)
        scores.append(indicators.get("sma_50_slope", 0))
        weights.append(0.2)
        scores.append(indicators.get("sma_200_slope", 0))
        weights.append(0.15)

        # Price vs MAs (market breadth proxy)
        scores.append(np.clip(indicators.get("price_vs_sma_20", 0) * 10, -1, 1))
        weights.append(0.1)
        scores.append(np.clip(indicators.get("price_vs_sma_50", 0) * 10, -1, 1))
        weights.append(0.1)

        # DI direction from ADX calculation
        plus_di = indicators.get("plus_di", 0)
        minus_di = indicators.get("minus_di", 0)
        di_diff = plus_di - minus_di
        di_sum = plus_di + minus_di
        if di_sum > 0:
            di_score = di_diff / di_sum
        else:
            di_score = 0
        scores.append(di_score)
        weights.append(0.15)

        # MACD histogram direction (momentum)
        macd_hist = indicators.get("macd_histogram", 0)
        # Normalize to roughly -1 to 1
        macd_score = np.clip(macd_hist * 100, -1, 1)
        scores.append(macd_score)
        weights.append(0.1)

        # Weighted average
        total_weight = sum(weights)
        direction = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return float(np.clip(direction, -1, 1))

    def _calculate_trend_strength(self, indicators: dict[str, float]) -> float:
        """
        Calculate overall trend strength score.

        Combines ADX, MA alignment, and price distance from MAs
        to determine how strong the current trend is.

        Args:
            indicators: Dictionary of calculated indicators.

        Returns:
            Score from 0 (no trend/ranging) to 1 (very strong trend).
        """
        # ADX component (normalized to 0-1, with 50 being very strong)
        adx = indicators.get("adx", 0)
        adx_strength = min(adx / 50, 1.0)

        # MA alignment (all slopes pointing same direction = stronger)
        slopes = [
            indicators.get("sma_20_slope", 0),
            indicators.get("sma_50_slope", 0),
            indicators.get("sma_200_slope", 0),
        ]

        # Check alignment (all same sign)
        if all(s > 0 for s in slopes) or all(s < 0 for s in slopes):
            alignment_strength = np.mean([abs(s) for s in slopes])
        else:
            alignment_strength = 0.4  # Optimized: was 0.2, less harsh penalty for misalignment

        # Price distance from MAs (further from MAs in trend direction = stronger)
        price_offsets = [
            abs(indicators.get("price_vs_sma_20", 0)),
            abs(indicators.get("price_vs_sma_50", 0)),
        ]
        offset_strength = min(np.mean(price_offsets) * 5, 1.0)

        # Combined strength
        strength = (
            0.4 * adx_strength +
            0.3 * alignment_strength +
            0.3 * offset_strength
        )

        return float(np.clip(strength, 0, 1))

    def _classify_regime(self, indicators: dict[str, float]) -> RegimeType:
        """
        Classify the market regime based on indicators.

        OPTIMIZED Classification Logic (v2):
            1. If high vol + negative direction + trend -> BEAR_CRISIS (check bear FIRST!)
            2. Elif volatility > 80th percentile -> HIGH_VOLATILITY
            3. Elif trend_strength > 0.5 and direction > 0.1 -> BULL_TRENDING
            4. Elif trend_strength > 0.5 and direction < -0.1 -> BEAR_CRISIS
            5. Else -> SIDEWAYS_NEUTRAL

        Args:
            indicators: Dictionary of calculated indicators.

        Returns:
            RegimeType classification.
        """
        vol_percentile = indicators.get("volatility_percentile", 50)
        trend_strength = indicators.get("trend_strength", 0)
        trend_direction = indicators.get("trend_direction", 0)
        adx = indicators.get("adx", 0)

        # Step 1: Check for BEAR_CRISIS first (high vol + negative direction)
        # This prevents bear markets from being classified as just "high volatility"
        if (vol_percentile > 75 and
            trend_direction < -0.2 and
            trend_strength > 0.35):
            return RegimeType.BEAR_CRISIS

        # Step 2: Check for pure high volatility (no clear trend)
        if vol_percentile > self.config.volatility_high_percentile:
            return RegimeType.HIGH_VOLATILITY

        # Step 3-4: Check for trending regimes with direction threshold
        if trend_strength > self.config.strong_trend_threshold:
            if trend_direction > 0.1:  # Require minimum positive direction
                return RegimeType.BULL_TRENDING
            elif trend_direction < -0.1:  # Require minimum negative direction
                return RegimeType.BEAR_CRISIS

        # Step 5: Default to sideways/neutral
        return RegimeType.SIDEWAYS_NEUTRAL

    def _calculate_confidence(
        self,
        indicators: dict[str, float],
        regime: RegimeType,
    ) -> float:
        """
        Calculate confidence level for the regime classification.

        Args:
            indicators: Dictionary of calculated indicators.
            regime: The classified regime.

        Returns:
            Confidence score from 0.0 to 1.0.
        """
        trend_strength = indicators.get("trend_strength", 0)
        vol_percentile = indicators.get("volatility_percentile", 50)
        adx = indicators.get("adx", 0)

        if regime == RegimeType.HIGH_VOLATILITY:
            # Confidence based on how far above the 90th percentile threshold
            excess = (vol_percentile - self.config.volatility_high_percentile) / 10
            base_confidence = 0.7 + min(excess, 0.3)

        elif regime in (RegimeType.BULL_TRENDING, RegimeType.BEAR_CRISIS):
            # Confidence based on trend strength and ADX
            trend_conf = min(trend_strength / self.config.strong_trend_threshold, 1.0)
            adx_conf = min(adx / 40, 1.0)  # ADX > 40 is very strong
            base_confidence = 0.5 * trend_conf + 0.5 * adx_conf

        else:  # SIDEWAYS_NEUTRAL
            # Higher confidence when truly neutral (indicators don't point anywhere)
            neutrality = 1 - abs(indicators.get("trend_direction", 0))
            low_vol = 1 - (vol_percentile / 100)
            base_confidence = 0.5 * neutrality + 0.3 * low_vol + 0.2

        return float(np.clip(base_confidence, 0.3, 0.99))

    def _apply_smoothing(
        self,
        raw_regime: RegimeType,
        confidence: float,
        df: pd.DataFrame,
    ) -> RegimeType:
        """
        Apply smoothing to prevent rapid regime switches.

        Uses a minimum hold period of 5 days before allowing regime changes,
        unless the new regime has significantly higher confidence (+15%).

        This prevents whipsawing between regimes during transitional periods.

        Args:
            raw_regime: The raw regime classification.
            confidence: Confidence of the raw regime.
            df: The price DataFrame (for timestamp).

        Returns:
            Smoothed regime classification.
        """
        # If no previous regime or first detection
        if self._last_regime_change is None:
            return raw_regime

        # If regime hasn't changed, no smoothing needed
        if raw_regime == self._current_regime:
            return raw_regime

        # Check minimum hold period (5 days)
        if self._days_in_regime < self.config.min_hold_days:
            # Only allow change if new regime has much higher confidence (+15%)
            confidence_threshold = self._regime_confidence + 0.15
            if confidence > confidence_threshold:
                return raw_regime
            else:
                # Keep current regime during hold period
                return self._current_regime

        # After minimum hold period, allow regime change
        return raw_regime

    def _update_state(
        self,
        regime: RegimeType,
        confidence: float,
        indicators: dict[str, float],
        df: pd.DataFrame,
    ) -> None:
        """
        Update internal state after regime detection.

        Records regime changes in history and updates counters.

        Args:
            regime: The detected regime.
            confidence: Confidence of the detection.
            indicators: Indicator values.
            df: The price DataFrame.
        """
        # Get timestamp from data or use current time
        if "datetime" in df.columns:
            timestamp = df["datetime"].iloc[-1]
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()
        else:
            timestamp = datetime.now()

        # Check if regime changed
        if regime != self._current_regime:
            # Record the change in history
            change = RegimeChange(
                timestamp=timestamp,
                previous_regime=self._current_regime,
                new_regime=regime,
                confidence=confidence,
                indicators={
                    "adx": indicators.get("adx", 0),
                    "volatility_percentile": indicators.get("volatility_percentile", 0),
                    "trend_strength": indicators.get("trend_strength", 0),
                    "trend_direction": indicators.get("trend_direction", 0),
                    "rsi": indicators.get("rsi", 0),
                    "macd_histogram": indicators.get("macd_histogram", 0),
                },
            )
            self._regime_history.append(change)

            # Update state
            self._current_regime = regime
            self._last_regime_change = timestamp
            self._days_in_regime = 1
        else:
            # Same regime, increment day counter
            self._days_in_regime += 1

        # Update confidence
        self._regime_confidence = confidence

    def reset(self) -> None:
        """
        Reset the detector to its initial state.

        Clears all history and resets to SIDEWAYS_NEUTRAL regime.
        """
        self._current_regime = RegimeType.SIDEWAYS_NEUTRAL
        self._regime_confidence = 0.5
        self._regime_history = []
        self._last_regime_change = None
        self._days_in_regime = 0
        self._indicator_cache = {}

    def get_regime_summary(self) -> dict[str, Any]:
        """
        Get a comprehensive summary of the current regime state.

        Returns:
            Dictionary with regime information and key indicators.
        """
        return {
            "current_regime": self._current_regime.value,
            "regime_confidence": self._regime_confidence,
            "days_in_regime": self._days_in_regime,
            "total_regime_changes": len(self._regime_history),
            "indicators": {
                "adx": self._indicator_cache.get("adx", 0),
                "volatility_percentile": self._indicator_cache.get("volatility_percentile", 0),
                "trend_strength": self._indicator_cache.get("trend_strength", 0),
                "trend_direction": self._indicator_cache.get("trend_direction", 0),
                "rsi": self._indicator_cache.get("rsi", 0),
                "atr_percent": self._indicator_cache.get("atr_percent", 0),
                "macd_histogram": self._indicator_cache.get("macd_histogram", 0),
                "sma_20_slope": self._indicator_cache.get("sma_20_slope", 0),
                "sma_50_slope": self._indicator_cache.get("sma_50_slope", 0),
                "sma_200_slope": self._indicator_cache.get("sma_200_slope", 0),
            },
        }


__all__ = [
    "RegimeType",
    "RegimeChange",
    "RegimeDetector",
    "RegimeDetectorConfig",
]

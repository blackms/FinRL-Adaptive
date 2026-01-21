"""
Enhanced Risk Management Module

Implements advanced risk management with four key improvements:
1. VIX as a leading indicator for crisis detection
2. Faster regime switching with shorter lookbacks
3. Pre-emptive risk reduction at volatility spikes
4. Portfolio-level stop-loss mechanisms

This module wraps the RegimeDetector to provide proactive risk management
that responds to market stress BEFORE significant losses occur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd


class RiskLevel(Enum):
    """Risk levels for portfolio management."""
    NORMAL = "normal"           # Full exposure allowed
    ELEVATED = "elevated"       # Reduce exposure 25%
    HIGH = "high"               # Reduce exposure 50%
    EXTREME = "extreme"         # Reduce exposure 75%
    CRISIS = "crisis"           # Maximum defensive (90% reduction)


@dataclass
class RiskSignal:
    """Container for risk assessment signals."""
    timestamp: datetime
    risk_level: RiskLevel
    exposure_multiplier: float  # 0.1 to 1.0
    vix_signal: float           # VIX-based fear indicator
    volatility_spike: bool      # Sudden vol increase detected
    stop_loss_triggered: bool   # Portfolio stop-loss hit
    drawdown_current: float     # Current drawdown from peak
    reasons: List[str] = field(default_factory=list)


@dataclass
class EnhancedRiskConfig:
    """Configuration for enhanced risk management."""

    # VIX thresholds (leading indicator)
    vix_normal: float = 15.0        # Below this = calm markets
    vix_elevated: float = 20.0      # Elevated caution
    vix_high: float = 25.0          # High fear
    vix_extreme: float = 30.0       # Extreme fear
    vix_crisis: float = 40.0        # Crisis mode (2008, 2020 peaks)

    # Volatility spike detection (pre-emptive)
    vol_spike_threshold: float = 1.5    # 50% increase = spike
    vol_spike_lookback: int = 5         # Days to measure spike
    vol_spike_baseline: int = 20        # Baseline period

    # Fast regime switching
    fast_lookback: int = 10             # Faster than default 20
    trend_ema_fast: int = 5             # Fast EMA for trend
    trend_ema_slow: int = 15            # Slow EMA for trend

    # Stop-loss settings
    stop_loss_portfolio: float = 0.15   # 15% portfolio stop-loss
    stop_loss_trailing: float = 0.10    # 10% trailing stop
    stop_loss_daily: float = 0.05       # 5% daily loss limit

    # Drawdown-based risk reduction
    drawdown_threshold_1: float = 0.05  # 5% DD -> reduce 10%
    drawdown_threshold_2: float = 0.10  # 10% DD -> reduce 25%
    drawdown_threshold_3: float = 0.15  # 15% DD -> reduce 50%
    drawdown_threshold_4: float = 0.20  # 20% DD -> reduce 75%

    # Recovery settings
    recovery_days: int = 5              # Days to wait after crisis
    recovery_gradual: bool = True       # Gradually increase exposure

    # CRASH DETECTOR (NEW) - triggers on rapid price decline
    crash_1day_threshold: float = -0.03   # 3% drop in 1 day
    crash_3day_threshold: float = -0.05   # 5% drop in 3 days
    crash_5day_threshold: float = -0.08   # 8% drop in 5 days
    crash_enabled: bool = True            # Enable crash detection


class EnhancedRiskManager:
    """
    Enhanced Risk Management System.

    Provides proactive risk management using:
    - VIX as a leading indicator (detects fear before price drops)
    - Volatility spike detection (catches sudden regime changes)
    - Faster regime switching (shorter lookbacks)
    - Portfolio stop-losses (independent of regime)
    """

    def __init__(self, config: EnhancedRiskConfig | None = None) -> None:
        """Initialize the risk manager."""
        self.config = config or EnhancedRiskConfig()

        # State tracking
        self._peak_value: float = 0.0
        self._current_drawdown: float = 0.0
        self._last_risk_level: RiskLevel = RiskLevel.NORMAL
        self._stop_loss_triggered: bool = False
        self._stop_loss_date: Optional[datetime] = None
        self._days_since_crisis: int = 0
        self._vix_history: List[float] = []
        self._vol_history: List[float] = []

    def assess_risk(
        self,
        portfolio_value: float,
        market_data: pd.DataFrame,
        vix_data: Optional[pd.DataFrame] = None,
        current_date: Optional[datetime] = None,
    ) -> RiskSignal:
        """
        Assess current risk level and recommend exposure.

        Args:
            portfolio_value: Current portfolio value.
            market_data: DataFrame with OHLCV data for the portfolio.
            vix_data: Optional VIX data (if not provided, uses synthetic VIX).
            current_date: Current date for the assessment.

        Returns:
            RiskSignal with risk level and exposure recommendation.
        """
        reasons = []

        # Update peak and drawdown
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        self._current_drawdown = (self._peak_value - portfolio_value) / self._peak_value if self._peak_value > 0 else 0

        # Get VIX signal
        vix_value, vix_signal = self._get_vix_signal(vix_data, market_data)

        # Detect volatility spike
        volatility_spike = self._detect_volatility_spike(market_data)

        # Detect crash (NEW - fastest signal)
        crash_detected, crash_reason = self._detect_crash(market_data)

        # Check stop-loss conditions
        stop_loss_triggered = self._check_stop_loss(portfolio_value, market_data)

        # Determine risk level
        risk_level = self._calculate_risk_level(
            vix_value, vix_signal, volatility_spike,
            stop_loss_triggered, reasons, crash_detected, crash_reason
        )

        # Calculate exposure multiplier
        exposure = self._calculate_exposure(risk_level, reasons)

        # Update state
        self._last_risk_level = risk_level
        if risk_level == RiskLevel.CRISIS:
            self._days_since_crisis = 0
        else:
            self._days_since_crisis += 1

        return RiskSignal(
            timestamp=current_date or datetime.now(),
            risk_level=risk_level,
            exposure_multiplier=exposure,
            vix_signal=vix_signal,
            volatility_spike=volatility_spike,
            stop_loss_triggered=stop_loss_triggered,
            drawdown_current=self._current_drawdown,
            reasons=reasons,
        )

    def _get_vix_signal(
        self,
        vix_data: Optional[pd.DataFrame],
        market_data: pd.DataFrame,
    ) -> Tuple[float, float]:
        """
        Get VIX value and normalized signal.

        If VIX data not available, synthesize from market volatility.

        Returns:
            Tuple of (vix_value, normalized_signal 0-1)
        """
        if vix_data is not None and len(vix_data) > 0:
            # Use actual VIX
            vix_col = 'Close' if 'Close' in vix_data.columns else 'close'
            if vix_col in vix_data.columns:
                vix_value = float(vix_data[vix_col].iloc[-1])
            else:
                # Flatten multi-level columns if needed
                if isinstance(vix_data.columns, pd.MultiIndex):
                    vix_data.columns = vix_data.columns.get_level_values(0)
                vix_value = float(vix_data['Close'].iloc[-1]) if 'Close' in vix_data.columns else 20.0
        else:
            # Synthesize VIX from market data
            vix_value = self._synthesize_vix(market_data)

        self._vix_history.append(vix_value)
        if len(self._vix_history) > 252:
            self._vix_history = self._vix_history[-252:]

        # Normalize to 0-1 signal (0 = calm, 1 = extreme fear)
        signal = np.clip((vix_value - self.config.vix_normal) /
                        (self.config.vix_crisis - self.config.vix_normal), 0, 1)

        return vix_value, signal

    def _synthesize_vix(self, market_data: pd.DataFrame) -> float:
        """
        Synthesize VIX-like value from market data.

        Uses realized volatility scaled to approximate VIX levels.
        """
        df = market_data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        if 'close' not in df.columns or len(df) < 20:
            return 20.0  # Default neutral VIX

        # Calculate 20-day realized volatility
        returns = df['close'].pct_change().dropna()
        if len(returns) < 20:
            return 20.0

        realized_vol = returns.iloc[-20:].std() * np.sqrt(252) * 100

        # Scale to VIX-like levels (realized vol * 1.2 + base)
        synthetic_vix = realized_vol * 1.2 + 5

        return float(np.clip(synthetic_vix, 10, 80))

    def _detect_volatility_spike(self, market_data: pd.DataFrame) -> bool:
        """
        Detect sudden volatility spike (pre-emptive risk signal).

        A spike is detected when recent volatility exceeds baseline
        by the spike threshold.
        """
        df = market_data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        if 'close' not in df.columns:
            return False

        returns = df['close'].pct_change().dropna()

        if len(returns) < self.config.vol_spike_baseline + self.config.vol_spike_lookback:
            return False

        # Recent volatility
        recent_vol = returns.iloc[-self.config.vol_spike_lookback:].std()

        # Baseline volatility
        baseline_end = -self.config.vol_spike_lookback
        baseline_start = baseline_end - self.config.vol_spike_baseline
        baseline_vol = returns.iloc[baseline_start:baseline_end].std()

        if baseline_vol == 0:
            return False

        # Check for spike
        vol_ratio = recent_vol / baseline_vol

        self._vol_history.append(vol_ratio)
        if len(self._vol_history) > 100:
            self._vol_history = self._vol_history[-100:]

        return vol_ratio > self.config.vol_spike_threshold

    def _detect_crash(self, market_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Detect rapid price crash (fastest risk signal).

        Triggers on:
        - 3% drop in 1 day
        - 5% drop in 3 days
        - 8% drop in 5 days

        Returns:
            Tuple of (crash_detected, crash_severity)
        """
        if not self.config.crash_enabled:
            return False, ""

        df = market_data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        if 'close' not in df.columns or len(df) < 6:
            return False, ""

        close = df['close']
        current = float(close.iloc[-1])

        # 1-day return
        if len(close) >= 2:
            ret_1d = (current / float(close.iloc[-2])) - 1
            if ret_1d <= self.config.crash_1day_threshold:
                return True, f"1D crash: {ret_1d:.1%}"

        # 3-day return
        if len(close) >= 4:
            ret_3d = (current / float(close.iloc[-4])) - 1
            if ret_3d <= self.config.crash_3day_threshold:
                return True, f"3D crash: {ret_3d:.1%}"

        # 5-day return
        if len(close) >= 6:
            ret_5d = (current / float(close.iloc[-6])) - 1
            if ret_5d <= self.config.crash_5day_threshold:
                return True, f"5D crash: {ret_5d:.1%}"

        return False, ""

    def _check_stop_loss(
        self,
        portfolio_value: float,
        market_data: pd.DataFrame,
    ) -> bool:
        """
        Check if any stop-loss condition is triggered.

        Three types of stop-loss:
        1. Portfolio stop-loss (from initial value)
        2. Trailing stop-loss (from peak)
        3. Daily loss limit
        """
        # Portfolio drawdown stop-loss
        if self._current_drawdown >= self.config.stop_loss_portfolio:
            self._stop_loss_triggered = True
            return True

        # Trailing stop-loss
        if self._current_drawdown >= self.config.stop_loss_trailing:
            self._stop_loss_triggered = True
            return True

        # Daily loss limit
        df = market_data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        if 'close' in df.columns and len(df) >= 2:
            daily_return = (df['close'].iloc[-1] / df['close'].iloc[-2]) - 1
            if daily_return < -self.config.stop_loss_daily:
                return True

        return False

    def _calculate_risk_level(
        self,
        vix_value: float,
        vix_signal: float,
        volatility_spike: bool,
        stop_loss_triggered: bool,
        reasons: List[str],
        crash_detected: bool = False,
        crash_reason: str = "",
    ) -> RiskLevel:
        """
        Calculate overall risk level from all signals.

        Priority order:
        1. Crash detected -> EXTREME/CRISIS (FASTEST SIGNAL)
        2. Stop-loss triggered -> CRISIS
        3. VIX extreme -> CRISIS
        4. Volatility spike + high VIX -> EXTREME
        5. VIX-based levels
        6. Drawdown-based levels
        """
        # CRASH DETECTION is highest priority (fastest signal)
        if crash_detected:
            reasons.append(f"CRASH DETECTED: {crash_reason}")
            # Crash + any other warning = CRISIS
            if volatility_spike or vix_value >= self.config.vix_elevated:
                return RiskLevel.CRISIS
            return RiskLevel.EXTREME

        # Stop-loss is second priority
        if stop_loss_triggered:
            reasons.append(f"Stop-loss triggered (DD: {self._current_drawdown:.1%})")
            return RiskLevel.CRISIS

        # VIX-based classification
        if vix_value >= self.config.vix_crisis:
            reasons.append(f"VIX crisis level ({vix_value:.1f})")
            return RiskLevel.CRISIS

        if vix_value >= self.config.vix_extreme:
            reasons.append(f"VIX extreme ({vix_value:.1f})")
            if volatility_spike:
                reasons.append("Volatility spike detected")
                return RiskLevel.CRISIS
            return RiskLevel.EXTREME

        if vix_value >= self.config.vix_high:
            reasons.append(f"VIX high ({vix_value:.1f})")
            if volatility_spike:
                reasons.append("Volatility spike detected")
                return RiskLevel.EXTREME
            return RiskLevel.HIGH

        if vix_value >= self.config.vix_elevated:
            reasons.append(f"VIX elevated ({vix_value:.1f})")
            if volatility_spike:
                reasons.append("Volatility spike detected")
                return RiskLevel.HIGH
            return RiskLevel.ELEVATED

        # Volatility spike alone
        if volatility_spike:
            reasons.append("Volatility spike detected")
            return RiskLevel.ELEVATED

        # Drawdown-based (even in calm VIX)
        if self._current_drawdown >= self.config.drawdown_threshold_4:
            reasons.append(f"Severe drawdown ({self._current_drawdown:.1%})")
            return RiskLevel.EXTREME
        elif self._current_drawdown >= self.config.drawdown_threshold_3:
            reasons.append(f"Large drawdown ({self._current_drawdown:.1%})")
            return RiskLevel.HIGH
        elif self._current_drawdown >= self.config.drawdown_threshold_2:
            reasons.append(f"Moderate drawdown ({self._current_drawdown:.1%})")
            return RiskLevel.ELEVATED

        reasons.append("Normal conditions")
        return RiskLevel.NORMAL

    def _calculate_exposure(
        self,
        risk_level: RiskLevel,
        reasons: List[str],
    ) -> float:
        """
        Calculate exposure multiplier based on risk level.

        Returns value between 0.1 (minimum) and 1.0 (full exposure).
        """
        # Base exposure by risk level
        exposure_map = {
            RiskLevel.NORMAL: 1.0,
            RiskLevel.ELEVATED: 0.75,
            RiskLevel.HIGH: 0.50,
            RiskLevel.EXTREME: 0.25,
            RiskLevel.CRISIS: 0.10,
        }

        base_exposure = exposure_map.get(risk_level, 1.0)

        # Gradual recovery from crisis
        if self.config.recovery_gradual and self._days_since_crisis > 0:
            if self._days_since_crisis < self.config.recovery_days:
                # Still in recovery period - limit exposure increase
                recovery_factor = self._days_since_crisis / self.config.recovery_days
                max_exposure = 0.25 + (0.75 * recovery_factor)
                if base_exposure > max_exposure:
                    reasons.append(f"Recovery period ({self._days_since_crisis}/{self.config.recovery_days} days)")
                    base_exposure = max_exposure

        return base_exposure

    def get_fast_trend_signal(self, market_data: pd.DataFrame) -> float:
        """
        Get fast trend signal for quick regime detection.

        Uses shorter EMAs than standard regime detector for faster response.

        Returns:
            Signal from -1 (bearish) to 1 (bullish)
        """
        df = market_data.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        if 'close' not in df.columns or len(df) < self.config.trend_ema_slow:
            return 0.0

        close = df['close']

        # Fast and slow EMAs
        fast_ema = close.ewm(span=self.config.trend_ema_fast, adjust=False).mean()
        slow_ema = close.ewm(span=self.config.trend_ema_slow, adjust=False).mean()

        # Signal based on EMA crossover
        current_fast = float(fast_ema.iloc[-1])
        current_slow = float(slow_ema.iloc[-1])

        if current_slow == 0:
            return 0.0

        # Normalized difference
        signal = (current_fast - current_slow) / current_slow

        # Scale to -1 to 1
        return float(np.clip(signal * 20, -1, 1))

    def reset(self) -> None:
        """Reset the risk manager state."""
        self._peak_value = 0.0
        self._current_drawdown = 0.0
        self._last_risk_level = RiskLevel.NORMAL
        self._stop_loss_triggered = False
        self._stop_loss_date = None
        self._days_since_crisis = 0
        self._vix_history = []
        self._vol_history = []


def fetch_vix_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch VIX data from Yahoo Finance.

    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)

    Returns:
        DataFrame with VIX OHLCV data
    """
    try:
        import yfinance as yf
        vix = yf.download("^VIX", start=start_date, end=end_date, progress=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        return vix
    except Exception as e:
        print(f"Warning: Could not fetch VIX data: {e}")
        return pd.DataFrame()

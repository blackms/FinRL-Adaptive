"""
Strategy Blender Module

Implements regime-aware strategy blending for adaptive portfolio management.
Combines multiple trading strategies with dynamic weighting based on market conditions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from .regime_detector import RegimeType, RegimeDetector, RegimeDetectorConfig


@dataclass
class StrategyWeight:
    """
    Strategy weight configuration for a specific regime.

    Attributes:
        strategy_name: Name of the strategy.
        weight: Weight allocation (0-1).
        regime: Market regime this weight applies to.
    """

    strategy_name: str
    weight: float
    regime: RegimeType

    def __post_init__(self) -> None:
        """Validate weight is within bounds."""
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")


@dataclass
class BlendedSignal:
    """
    Aggregated signal from multiple strategies.

    Attributes:
        symbol: Ticker symbol.
        signal_type: Final signal type after blending.
        strength: Weighted signal strength (-1 to 1).
        timestamp: Signal generation time.
        price: Current price.
        regime: Current market regime.
        position_size_modifier: Adjustment factor for position sizing.
        component_signals: Individual strategy signals.
        weights_used: Strategy weights applied.
        conflict_resolution: How conflicts were resolved.
        metadata: Additional signal metadata.
    """

    symbol: str
    signal_type: SignalType
    strength: float
    timestamp: datetime
    price: float
    regime: RegimeType
    position_size_modifier: float = 1.0
    component_signals: dict[str, float] = field(default_factory=dict)
    weights_used: dict[str, float] = field(default_factory=dict)
    conflict_resolution: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_signal(self) -> Signal:
        """Convert to base Signal object."""
        return Signal(
            symbol=self.symbol,
            signal_type=self.signal_type,
            strength=self.strength,
            timestamp=self.timestamp,
            price=self.price,
            metadata={
                "regime": self.regime.value,
                "position_size_modifier": self.position_size_modifier,
                "component_signals": self.component_signals,
                "weights_used": self.weights_used,
                "conflict_resolution": self.conflict_resolution,
                **self.metadata
            }
        )


@dataclass
class StrategyBlenderConfig:
    """
    Configuration for strategy blending.

    Attributes:
        signal_threshold: Minimum weighted signal to generate trade.
        conflict_threshold: Signal disagreement threshold for conflict resolution.
        high_vol_position_reduction: Position size reduction in high volatility.
        enable_conflict_resolution: Enable automatic conflict resolution.
        min_strategy_agreement: Minimum strategies that must agree for signal.
    """

    signal_threshold: float = 0.2
    conflict_threshold: float = 0.5
    high_vol_position_reduction: float = 0.5  # Reduce positions by 50% in high vol
    enable_conflict_resolution: bool = True
    min_strategy_agreement: float = 0.5  # At least 50% of weight must agree


# Default regime-strategy weight mapping
DEFAULT_REGIME_WEIGHTS: dict[RegimeType, dict[str, float]] = {
    RegimeType.BULL_TRENDING: {
        "momentum": 0.6,
        "adaptive_hf": 0.3,
        "market_neutral": 0.1,
    },
    RegimeType.BEAR_CRISIS: {
        "momentum": 0.1,
        "adaptive_hf": 0.7,
        "market_neutral": 0.2,
    },
    RegimeType.SIDEWAYS_NEUTRAL: {
        "momentum": 0.2,
        "adaptive_hf": 0.2,
        "market_neutral": 0.6,
    },
    RegimeType.HIGH_VOLATILITY: {
        "momentum": 0.1,
        "adaptive_hf": 0.5,
        "market_neutral": 0.4,
    },
}

# Fallback weights when regime is not found
_FALLBACK_WEIGHTS: dict[str, float] = {
    "momentum": 0.33,
    "adaptive_hf": 0.33,
    "market_neutral": 0.34,
}


class StrategyBlender:
    """
    Blends multiple trading strategies with regime-adaptive weighting.

    Combines signals from multiple strategies using weights that adjust
    based on current market regime. Implements conflict resolution when
    strategies disagree and position size adjustment for volatility.

    Example:
        >>> strategies = {
        ...     "momentum": MomentumStrategy(),
        ...     "adaptive_hf": AdaptiveHFStrategy(),
        ...     "market_neutral": MarketNeutralStrategy(),
        ... }
        >>> blender = StrategyBlender(strategies)
        >>> signals = blender.get_blended_signals(RegimeType.BULL_TRENDING, data)
    """

    def __init__(
        self,
        strategies: dict[str, BaseStrategy],
        config: StrategyBlenderConfig | None = None,
        regime_weights: dict[RegimeType, dict[str, float]] | None = None,
        regime_detector: RegimeDetector | None = None,
    ) -> None:
        """
        Initialize the strategy blender.

        Args:
            strategies: Dictionary mapping strategy names to strategy instances.
            config: Blender configuration. Defaults to StrategyBlenderConfig.
            regime_weights: Custom regime-strategy weight mapping. Defaults to DEFAULT_REGIME_WEIGHTS.
            regime_detector: Optional regime detector instance.
        """
        self.strategies = strategies
        self.config = config or StrategyBlenderConfig()
        self._regime_weights = regime_weights or DEFAULT_REGIME_WEIGHTS.copy()
        self.regime_detector = regime_detector or RegimeDetector()
        self._signal_history: list[BlendedSignal] = []

        # Validate strategy names match regime weights
        self._validate_strategy_coverage()

    def _validate_strategy_coverage(self) -> None:
        """Validate that strategies are covered in regime weights."""
        for regime, weights in self._regime_weights.items():
            for strategy_name in weights:
                if strategy_name not in self.strategies:
                    # Warn but don't fail - strategy might be added later
                    pass

    @property
    def signal_history(self) -> list[BlendedSignal]:
        """Get blended signal history."""
        return self._signal_history.copy()

    def get_regime_weights(self, regime: RegimeType) -> dict[str, float]:
        """
        Get strategy weights for a given regime.

        Args:
            regime: Market regime type.

        Returns:
            Dictionary of strategy name to weight mappings.
        """
        weights = self._regime_weights.get(regime, _FALLBACK_WEIGHTS)

        # Filter to only include available strategies
        available_weights = {
            name: weight
            for name, weight in weights.items()
            if name in self.strategies
        }

        # Renormalize if needed
        total = sum(available_weights.values())
        if total > 0 and abs(total - 1.0) > 0.01:
            available_weights = {
                name: weight / total
                for name, weight in available_weights.items()
            }

        return available_weights

    def set_regime_weights(self, regime: RegimeType, weights: dict[str, float]) -> None:
        """
        Set custom weights for a regime.

        Args:
            regime: Market regime type.
            weights: Dictionary of strategy name to weight mappings.

        Raises:
            ValueError: If weights don't sum to 1.0 (within tolerance).
        """
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

        for weight in weights.values():
            if not 0.0 <= weight <= 1.0:
                raise ValueError(f"Individual weights must be between 0.0 and 1.0")

        self._regime_weights[regime] = weights.copy()

    def _get_position_size_modifier(self, regime: RegimeType) -> float:
        """
        Get position size modifier based on regime.

        Args:
            regime: Current market regime.

        Returns:
            Position size multiplier (0-1).
        """
        if regime == RegimeType.HIGH_VOLATILITY:
            return 1.0 - self.config.high_vol_position_reduction
        if regime == RegimeType.BEAR_CRISIS:
            return 0.7  # Reduce exposure in crisis
        return 1.0

    def _collect_strategy_signals(
        self,
        market_data: pd.DataFrame,
        weights: dict[str, float]
    ) -> dict[str, list[Signal]]:
        """
        Collect signals from all weighted strategies.

        Args:
            market_data: DataFrame with OHLCV data.
            weights: Strategy weights to use.

        Returns:
            Dictionary mapping strategy names to their signals.
        """
        signals: dict[str, list[Signal]] = {}

        for strategy_name, weight in weights.items():
            if weight <= 0:
                continue

            strategy = self.strategies.get(strategy_name)
            if strategy is None:
                continue

            try:
                strategy_signals = strategy.generate_signals(market_data)
                signals[strategy_name] = strategy_signals
            except Exception as e:
                # Log error but continue with other strategies
                signals[strategy_name] = []

        return signals

    def _extract_signal_strengths(
        self,
        strategy_signals: dict[str, list[Signal]],
        symbol: str
    ) -> dict[str, float]:
        """
        Extract signal strengths for a symbol from strategy signals.

        Args:
            strategy_signals: Collected signals from strategies.
            symbol: Target symbol.

        Returns:
            Dictionary of strategy name to signal strength.
        """
        strengths: dict[str, float] = {}

        for strategy_name, signals in strategy_signals.items():
            # Find signal for this symbol
            symbol_signals = [s for s in signals if s.symbol == symbol]
            if symbol_signals:
                # Use the most recent signal
                strengths[strategy_name] = symbol_signals[-1].strength
            else:
                # No signal = neutral (hold)
                strengths[strategy_name] = 0.0

        return strengths

    def _resolve_conflicts(
        self,
        signal_strengths: dict[str, float],
        weights: dict[str, float],
        weighted_signal: float
    ) -> tuple[float, str | None]:
        """
        Resolve conflicts when strategies disagree.

        Args:
            signal_strengths: Individual strategy signal strengths.
            weights: Strategy weights.
            weighted_signal: Initial weighted average signal.

        Returns:
            Tuple of (adjusted_signal, conflict_resolution_method).
        """
        if not self.config.enable_conflict_resolution:
            return weighted_signal, None

        # Check for significant disagreement
        positive_signals = [
            (name, strength, weights.get(name, 0))
            for name, strength in signal_strengths.items()
            if strength > 0.1
        ]
        negative_signals = [
            (name, strength, weights.get(name, 0))
            for name, strength in signal_strengths.items()
            if strength < -0.1
        ]

        positive_weight = sum(w for _, _, w in positive_signals)
        negative_weight = sum(w for _, _, w in negative_signals)

        # No conflict if clear majority
        if positive_weight == 0 or negative_weight == 0:
            return weighted_signal, None

        # Calculate conflict severity
        conflict_severity = min(positive_weight, negative_weight) / max(positive_weight, negative_weight)

        if conflict_severity < self.config.conflict_threshold:
            # Mild conflict - go with weighted signal
            return weighted_signal, "weighted_majority"

        # Significant conflict - reduce signal strength
        reduction_factor = 1.0 - conflict_severity * 0.5
        adjusted_signal = weighted_signal * reduction_factor

        # Check if minimum agreement threshold is met
        if positive_weight >= self.config.min_strategy_agreement:
            resolution = "bullish_consensus_with_reduction"
        elif negative_weight >= self.config.min_strategy_agreement:
            resolution = "bearish_consensus_with_reduction"
        else:
            # Neither side has clear consensus - be neutral
            adjusted_signal = weighted_signal * 0.3
            resolution = "no_consensus_heavy_reduction"

        return adjusted_signal, resolution

    def _determine_signal_type(self, strength: float) -> SignalType:
        """
        Determine signal type from strength.

        Args:
            strength: Signal strength (-1 to 1).

        Returns:
            SignalType (BUY, SELL, or HOLD).
        """
        if strength > self.config.signal_threshold:
            return SignalType.BUY
        if strength < -self.config.signal_threshold:
            return SignalType.SELL
        return SignalType.HOLD

    def get_blended_signals(
        self,
        regime: RegimeType,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate blended signals weighted by regime.

        Args:
            regime: Current market regime.
            market_data: DataFrame with OHLCV data.

        Returns:
            DataFrame with blended signals including:
            - symbol: Ticker symbol
            - signal_type: BUY/SELL/HOLD
            - strength: Weighted signal strength
            - position_size_modifier: Position adjustment factor
            - component_signals: Individual strategy signals
        """
        weights = self.get_regime_weights(regime)

        if not weights:
            return pd.DataFrame()

        # Collect signals from all strategies
        strategy_signals = self._collect_strategy_signals(market_data, weights)

        # Determine symbols to process
        all_symbols: set[str] = set()
        for signals in strategy_signals.values():
            for signal in signals:
                all_symbols.add(signal.symbol)

        # If no signals, try to infer symbol from data
        if not all_symbols:
            symbol_col = market_data.get("symbol")
            if symbol_col is not None:
                all_symbols = set(symbol_col.unique())
            else:
                all_symbols = {"UNKNOWN"}

        # Get current timestamp and price
        df = market_data.copy()
        df.columns = [col.lower() for col in df.columns]
        timestamp = datetime.now()
        if 'datetime' in df.columns:
            ts = df['datetime'].iloc[-1]
            timestamp = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts

        current_price = float(df['close'].iloc[-1]) if 'close' in df.columns else 0.0
        position_modifier = self._get_position_size_modifier(regime)

        results: list[dict[str, Any]] = []

        for symbol in all_symbols:
            # Extract signal strengths for this symbol
            signal_strengths = self._extract_signal_strengths(strategy_signals, symbol)

            # Calculate weighted average signal
            weighted_signal = sum(
                signal_strengths.get(name, 0.0) * weight
                for name, weight in weights.items()
            )

            # Resolve conflicts
            adjusted_signal, conflict_resolution = self._resolve_conflicts(
                signal_strengths, weights, weighted_signal
            )

            # Apply position size modifier to strength if reducing exposure
            if position_modifier < 1.0:
                adjusted_signal *= position_modifier

            # Determine signal type
            signal_type = self._determine_signal_type(adjusted_signal)

            # Create blended signal
            blended = BlendedSignal(
                symbol=symbol,
                signal_type=signal_type,
                strength=float(np.clip(adjusted_signal, -1.0, 1.0)),
                timestamp=timestamp,
                price=current_price,
                regime=regime,
                position_size_modifier=position_modifier,
                component_signals=signal_strengths,
                weights_used=weights,
                conflict_resolution=conflict_resolution,
            )

            self._signal_history.append(blended)

            results.append({
                "symbol": symbol,
                "signal_type": signal_type.name,
                "strength": blended.strength,
                "position_size_modifier": position_modifier,
                "component_signals": signal_strengths,
                "weights_used": weights,
                "conflict_resolution": conflict_resolution,
                "regime": regime.value,
                "timestamp": timestamp,
                "price": current_price,
            })

        return pd.DataFrame(results)

    def get_blended_signals_auto_regime(
        self,
        market_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate blended signals with automatic regime detection.

        Args:
            market_data: DataFrame with OHLCV data.

        Returns:
            DataFrame with blended signals.
        """
        regime = self.regime_detector.get_regime_for_weights(market_data)
        return self.get_blended_signals(regime, market_data)

    def get_strategy_contribution(
        self,
        regime: RegimeType,
        market_data: pd.DataFrame
    ) -> dict[str, dict[str, Any]]:
        """
        Analyze individual strategy contributions to blended signal.

        Args:
            regime: Market regime.
            market_data: DataFrame with OHLCV data.

        Returns:
            Dictionary with strategy contributions and analysis.
        """
        weights = self.get_regime_weights(regime)
        strategy_signals = self._collect_strategy_signals(market_data, weights)

        contributions: dict[str, dict[str, Any]] = {}

        for strategy_name, signals in strategy_signals.items():
            weight = weights.get(strategy_name, 0)
            avg_strength = np.mean([s.strength for s in signals]) if signals else 0.0

            contributions[strategy_name] = {
                "weight": weight,
                "signal_count": len(signals),
                "average_strength": avg_strength,
                "weighted_contribution": weight * avg_strength,
                "signals": [
                    {
                        "symbol": s.symbol,
                        "type": s.signal_type.name,
                        "strength": s.strength
                    }
                    for s in signals
                ]
            }

        return contributions

    def clear_history(self) -> None:
        """Clear signal history."""
        self._signal_history.clear()

    def __repr__(self) -> str:
        """String representation."""
        strategy_names = list(self.strategies.keys())
        return f"StrategyBlender(strategies={strategy_names})"


__all__ = [
    "StrategyBlender",
    "StrategyBlenderConfig",
    "StrategyWeight",
    "BlendedSignal",
    "DEFAULT_REGIME_WEIGHTS",
]

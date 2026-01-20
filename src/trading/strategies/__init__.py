"""Trading strategies module."""

from .base import (
    BaseStrategy,
    Signal,
    SignalType,
    Position,
    StrategyConfig,
)
from .momentum import MomentumStrategy, MomentumConfig
from .mean_reversion import MeanReversionStrategy, MeanReversionConfig
from .pairs_trading import PairsTradingStrategy, PairsTradingConfig, PairConfig
from .trend_following import TrendFollowingStrategy, TrendFollowingConfig
from .aggressive_momentum import (
    AggressiveMomentumStrategy,
    AggressiveMomentumConfig,
    AggressivePosition,
    PyramidLevel,
)
from .breakout import BreakoutStrategy, BreakoutConfig
from .ensemble import EnsembleStrategy, EnsembleConfig

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalType",
    "Position",
    "StrategyConfig",
    "MomentumStrategy",
    "MomentumConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    "PairsTradingStrategy",
    "PairsTradingConfig",
    "PairConfig",
    "TrendFollowingStrategy",
    "TrendFollowingConfig",
    "AggressiveMomentumStrategy",
    "AggressiveMomentumConfig",
    "AggressivePosition",
    "PyramidLevel",
    "BreakoutStrategy",
    "BreakoutConfig",
    "EnsembleStrategy",
    "EnsembleConfig",
]

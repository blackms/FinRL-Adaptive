"""
Configuration schema for the S&P 500 Top 100 Trading System.

This module defines the complete configuration hierarchy using Pydantic
for validation and type safety. Configuration can be loaded from YAML files
with environment variable overrides.

Example:
    config = TradingConfig.from_yaml("config/paper.yaml")
    print(config.risk.max_position_pct)
"""

from __future__ import annotations

import os
from dataclasses import field
from datetime import time
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from pydantic import BaseModel, Field, field_validator, model_validator
import yaml


class TradingMode(str, Enum):
    """Trading mode determines broker and risk parameters."""
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"


class DataProvider(str, Enum):
    """Supported market data providers."""
    YAHOO = "yahoo"
    ALPHA_VANTAGE = "alpha_vantage"
    POLYGON = "polygon"
    IEX = "iex"


class BrokerType(str, Enum):
    """Supported broker integrations."""
    PAPER = "paper"
    ALPACA = "alpaca"
    INTERACTIVE_BROKERS = "interactive_brokers"


class OrderType(str, Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """Order time-in-force options."""
    DAY = "day"
    GTC = "gtc"  # Good Till Cancelled
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill


class DataFrequency(str, Enum):
    """Data sampling frequency."""
    MINUTE = "1min"
    FIVE_MINUTE = "5min"
    FIFTEEN_MINUTE = "15min"
    HOUR = "1h"
    DAILY = "1d"


class PositionSizingMethod(str, Enum):
    """Position sizing algorithms."""
    FIXED_FRACTION = "fixed_fraction"
    KELLY_CRITERION = "kelly"
    VOLATILITY_SCALED = "volatility_scaled"
    RISK_PARITY = "risk_parity"


class AlertChannel(str, Enum):
    """Alert notification channels."""
    EMAIL = "email"
    SLACK = "slack"
    DISCORD = "discord"
    SMS = "sms"
    WEBHOOK = "webhook"


# ---------------------------------------------------------------------------
# Data Configuration
# ---------------------------------------------------------------------------

class CacheConfig(BaseModel):
    """In-memory cache configuration."""
    enabled: bool = True
    ttl_seconds: int = Field(default=300, ge=0)
    max_size_mb: int = Field(default=512, ge=64)
    redis_url: Optional[str] = None


class DataProviderConfig(BaseModel):
    """Configuration for a single data provider."""
    provider: DataProvider
    api_key_env: Optional[str] = None  # Environment variable name for API key
    rate_limit: int = Field(default=5, ge=1, description="Requests per second")
    timeout_seconds: int = Field(default=30, ge=1)
    retry_count: int = Field(default=3, ge=0)
    retry_delay_seconds: float = Field(default=1.0, ge=0)


class DataConfig(BaseModel):
    """Data layer configuration."""
    primary_provider: DataProvider = DataProvider.YAHOO
    fallback_providers: List[DataProvider] = Field(default_factory=list)
    providers: Dict[DataProvider, DataProviderConfig] = Field(default_factory=dict)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    database_url: Optional[str] = None
    frequency: DataFrequency = DataFrequency.DAILY
    lookback_days: int = Field(default=252, ge=1, description="Trading days of history")

    @model_validator(mode="after")
    def validate_providers(self) -> "DataConfig":
        """Ensure primary provider has configuration."""
        if self.primary_provider not in self.providers:
            self.providers[self.primary_provider] = DataProviderConfig(
                provider=self.primary_provider
            )
        return self


# ---------------------------------------------------------------------------
# Strategy Configuration
# ---------------------------------------------------------------------------

class StrategyParameterConfig(BaseModel):
    """Generic strategy parameter configuration."""
    name: str
    value: Union[int, float, str, bool, List[Any], Dict[str, Any]]
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    description: Optional[str] = None


class StrategyConfig(BaseModel):
    """Configuration for a trading strategy."""
    name: str
    enabled: bool = True
    weight: float = Field(default=1.0, ge=0, le=1.0, description="Signal weight")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    symbols: Optional[List[str]] = None  # None means all symbols

    class Config:
        extra = "allow"  # Allow strategy-specific parameters


class MomentumStrategyConfig(StrategyConfig):
    """Momentum strategy specific configuration."""
    name: str = "momentum"
    lookback_period: int = Field(default=20, ge=1)
    threshold: float = Field(default=0.02, ge=0)
    rebalance_frequency: int = Field(default=5, ge=1, description="Days between rebalance")


class MeanReversionStrategyConfig(StrategyConfig):
    """Mean reversion strategy specific configuration."""
    name: str = "mean_reversion"
    window: int = Field(default=20, ge=5)
    std_multiplier: float = Field(default=2.0, ge=0.5)
    exit_threshold: float = Field(default=0.5, ge=0)


class TrendFollowingStrategyConfig(StrategyConfig):
    """Trend following strategy specific configuration."""
    name: str = "trend_following"
    short_ma_period: int = Field(default=10, ge=1)
    long_ma_period: int = Field(default=50, ge=2)
    atr_period: int = Field(default=14, ge=1)
    atr_multiplier: float = Field(default=2.0, ge=0)

    @model_validator(mode="after")
    def validate_ma_periods(self) -> "TrendFollowingStrategyConfig":
        """Ensure short MA is less than long MA."""
        if self.short_ma_period >= self.long_ma_period:
            raise ValueError("short_ma_period must be less than long_ma_period")
        return self


class StrategiesConfig(BaseModel):
    """Configuration for all strategies."""
    strategies: List[StrategyConfig] = Field(default_factory=list)
    combination_method: str = Field(
        default="weighted_average",
        description="How to combine signals: weighted_average, majority_vote, highest_confidence"
    )
    min_agreement: float = Field(
        default=0.5, ge=0, le=1.0,
        description="Minimum strategy agreement for signal"
    )


# ---------------------------------------------------------------------------
# Risk Management Configuration
# ---------------------------------------------------------------------------

class StopLossConfig(BaseModel):
    """Stop loss configuration."""
    enabled: bool = True
    type: str = Field(default="percentage", description="percentage, atr, or fixed")
    value: float = Field(default=2.0, ge=0)
    trailing: bool = False
    trailing_activation_pct: float = Field(default=1.0, ge=0)


class TakeProfitConfig(BaseModel):
    """Take profit configuration."""
    enabled: bool = True
    type: str = Field(default="percentage", description="percentage, atr, or fixed")
    value: float = Field(default=5.0, ge=0)
    scale_out: bool = False
    scale_out_levels: List[float] = Field(default_factory=lambda: [0.5, 1.0])


class RiskConfig(BaseModel):
    """Risk management configuration."""
    # Position limits
    max_position_pct: float = Field(
        default=5.0, ge=0.1, le=100.0,
        description="Maximum position size as % of portfolio"
    )
    max_positions: int = Field(default=20, ge=1)
    max_sector_exposure_pct: float = Field(default=25.0, ge=0, le=100.0)

    # Portfolio risk
    max_portfolio_risk_pct: float = Field(
        default=10.0, ge=0,
        description="Maximum portfolio VaR as %"
    )
    max_drawdown_pct: float = Field(default=15.0, ge=0)
    max_correlation: float = Field(default=0.7, ge=0, le=1.0)

    # Position sizing
    position_sizing_method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_SCALED
    kelly_fraction: float = Field(default=0.25, ge=0, le=1.0)
    volatility_target: float = Field(default=0.15, ge=0, description="Annual volatility target")

    # Stop loss and take profit
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)

    # Circuit breakers
    max_daily_trades: int = Field(default=50, ge=0)
    max_daily_loss_pct: float = Field(default=3.0, ge=0)
    halt_on_drawdown: bool = True
    cooldown_after_loss_minutes: int = Field(default=15, ge=0)


# ---------------------------------------------------------------------------
# Execution Configuration
# ---------------------------------------------------------------------------

class BrokerConfig(BaseModel):
    """Broker-specific configuration."""
    broker_type: BrokerType
    api_key_env: Optional[str] = None
    api_secret_env: Optional[str] = None
    base_url: Optional[str] = None
    paper_url: Optional[str] = None
    account_id: Optional[str] = None


class SlippageConfig(BaseModel):
    """Slippage model configuration."""
    model: str = Field(default="percentage", description="percentage, volume_based, or fixed")
    base_slippage_bps: float = Field(default=5.0, ge=0, description="Basis points")
    volume_impact_factor: float = Field(default=0.1, ge=0)


class CommissionConfig(BaseModel):
    """Commission model configuration."""
    model: str = Field(default="per_share", description="per_share, per_trade, or percentage")
    per_share: Decimal = Field(default=Decimal("0.005"))
    per_trade_min: Decimal = Field(default=Decimal("1.00"))
    per_trade_max: Decimal = Field(default=Decimal("0"))  # 0 = no max
    percentage_bps: float = Field(default=0)


class ExecutionConfig(BaseModel):
    """Execution layer configuration."""
    broker: BrokerConfig = Field(
        default_factory=lambda: BrokerConfig(broker_type=BrokerType.PAPER)
    )
    allowed_order_types: List[OrderType] = Field(
        default_factory=lambda: [OrderType.MARKET, OrderType.LIMIT]
    )
    default_time_in_force: TimeInForce = TimeInForce.DAY
    slippage: SlippageConfig = Field(default_factory=SlippageConfig)
    commission: CommissionConfig = Field(default_factory=CommissionConfig)

    # Execution parameters
    smart_routing: bool = True
    check_market_hours: bool = True
    allow_premarket: bool = False
    allow_afterhours: bool = False
    max_order_value: Optional[Decimal] = None
    require_confirmation_above: Optional[Decimal] = None


# ---------------------------------------------------------------------------
# Backtest Configuration
# ---------------------------------------------------------------------------

class BacktestConfig(BaseModel):
    """Backtesting engine configuration."""
    initial_capital: Decimal = Field(default=Decimal("100000"))
    benchmark_symbol: str = "SPY"

    # Simulation settings
    use_adjusted_close: bool = True
    fill_on_close: bool = False  # True = fill at close, False = fill at next open
    fractional_shares: bool = False

    # Data handling
    handle_splits: bool = True
    handle_dividends: bool = True
    dividend_reinvest: bool = True


# ---------------------------------------------------------------------------
# Monitoring Configuration
# ---------------------------------------------------------------------------

class AlertConfig(BaseModel):
    """Alert notification configuration."""
    enabled: bool = True
    channels: List[AlertChannel] = Field(default_factory=lambda: [AlertChannel.EMAIL])

    # Alert thresholds
    drawdown_alert_pct: float = Field(default=5.0, ge=0)
    daily_loss_alert_pct: float = Field(default=2.0, ge=0)
    position_size_alert_pct: float = Field(default=4.0, ge=0)

    # Channel-specific configuration
    email_recipients: List[str] = Field(default_factory=list)
    slack_webhook_url: Optional[str] = None
    discord_webhook_url: Optional[str] = None


class MetricsConfig(BaseModel):
    """Metrics collection configuration."""
    enabled: bool = True
    prometheus_port: int = Field(default=9090, ge=1024, le=65535)
    collection_interval_seconds: int = Field(default=60, ge=1)
    retention_days: int = Field(default=30, ge=1)


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field(default="INFO")
    format: str = Field(default="json", description="json or text")
    file_path: Optional[Path] = None
    rotation: str = Field(default="1 day")
    retention: str = Field(default="30 days")
    include_trade_log: bool = True


class MonitoringConfig(BaseModel):
    """Monitoring layer configuration."""
    alerts: AlertConfig = Field(default_factory=AlertConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    dashboard_enabled: bool = True
    grafana_url: Optional[str] = None


# ---------------------------------------------------------------------------
# Universe Configuration
# ---------------------------------------------------------------------------

class UniverseConfig(BaseModel):
    """Trading universe configuration."""
    type: str = Field(default="sp500_top100", description="sp500_top100, sp500, custom")
    custom_symbols: List[str] = Field(default_factory=list)
    exclude_symbols: List[str] = Field(default_factory=list)
    min_market_cap: Optional[Decimal] = None
    min_avg_volume: Optional[int] = None
    sectors: Optional[List[str]] = None  # Filter by sector


# ---------------------------------------------------------------------------
# Schedule Configuration
# ---------------------------------------------------------------------------

class ScheduleConfig(BaseModel):
    """Trading schedule configuration."""
    market_open: time = Field(default=time(9, 30))
    market_close: time = Field(default=time(16, 0))
    timezone: str = "America/New_York"

    # Trading windows
    trading_start_buffer_minutes: int = Field(default=5, ge=0)
    trading_end_buffer_minutes: int = Field(default=5, ge=0)
    avoid_first_minutes: int = Field(default=15, ge=0)
    avoid_last_minutes: int = Field(default=15, ge=0)

    # Rebalance schedule
    rebalance_time: Optional[time] = None
    rebalance_days: List[str] = Field(
        default_factory=lambda: ["monday"],
        description="Days of week for rebalancing"
    )


# ---------------------------------------------------------------------------
# Main Configuration
# ---------------------------------------------------------------------------

class TradingConfig(BaseModel):
    """
    Root configuration for the trading system.

    This is the main configuration object that contains all subsystem
    configurations. It can be loaded from YAML files with environment
    variable substitution.

    Example:
        config = TradingConfig.from_yaml("config/paper.yaml")
    """

    # System mode
    mode: TradingMode = TradingMode.PAPER
    name: str = Field(default="sp500_trading_system")
    version: str = Field(default="0.1.0")

    # Component configurations
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    strategies: StrategiesConfig = Field(default_factory=StrategiesConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    schedule: ScheduleConfig = Field(default_factory=ScheduleConfig)

    class Config:
        use_enum_values = True

    @classmethod
    def from_yaml(cls, path: Union[str, Path], env_override: bool = True) -> "TradingConfig":
        """
        Load configuration from a YAML file with optional environment variable overrides.

        Environment variables are substituted using ${VAR_NAME} syntax in the YAML file.

        Args:
            path: Path to the YAML configuration file
            env_override: Whether to substitute environment variables

        Returns:
            Validated TradingConfig instance

        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValidationError: If the configuration is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with open(path) as f:
            content = f.read()

        # Substitute environment variables
        if env_override:
            content = cls._substitute_env_vars(content)

        data = yaml.safe_load(content)
        return cls.model_validate(data)

    @staticmethod
    def _substitute_env_vars(content: str) -> str:
        """Substitute ${VAR_NAME} patterns with environment variable values."""
        import re
        pattern = r'\$\{(\w+)\}'

        def replace(match: re.Match) -> str:
            var_name = match.group(1)
            return os.environ.get(var_name, match.group(0))

        return re.sub(pattern, replace, content)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Destination path for the YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def is_live(self) -> bool:
        """Check if running in live trading mode."""
        return self.mode == TradingMode.LIVE

    def is_paper(self) -> bool:
        """Check if running in paper trading mode."""
        return self.mode == TradingMode.PAPER

    def is_backtest(self) -> bool:
        """Check if running in backtest mode."""
        return self.mode == TradingMode.BACKTEST

    @model_validator(mode="after")
    def validate_live_mode_requirements(self) -> "TradingConfig":
        """Ensure live mode has stricter risk limits and proper broker config."""
        if self.mode == TradingMode.LIVE:
            # Enforce stricter limits for live trading
            if self.risk.max_position_pct > 10.0:
                raise ValueError("Live mode requires max_position_pct <= 10.0")
            if self.risk.max_drawdown_pct > 20.0:
                raise ValueError("Live mode requires max_drawdown_pct <= 20.0")

            # Ensure broker is configured
            if self.execution.broker.broker_type == BrokerType.PAPER:
                raise ValueError("Live mode requires a real broker, not paper")

        return self


# ---------------------------------------------------------------------------
# Default Configuration Factory
# ---------------------------------------------------------------------------

def create_default_config(mode: TradingMode = TradingMode.PAPER) -> TradingConfig:
    """
    Create a default configuration for the specified trading mode.

    Args:
        mode: Trading mode (paper, live, or backtest)

    Returns:
        TradingConfig with sensible defaults for the mode
    """
    config = TradingConfig(mode=mode)

    # Add default momentum strategy
    config.strategies.strategies.append(
        MomentumStrategyConfig(
            name="momentum",
            enabled=True,
            weight=0.5,
            lookback_period=20,
            threshold=0.02
        )
    )

    # Add default mean reversion strategy
    config.strategies.strategies.append(
        MeanReversionStrategyConfig(
            name="mean_reversion",
            enabled=True,
            weight=0.5,
            window=20,
            std_multiplier=2.0
        )
    )

    return config

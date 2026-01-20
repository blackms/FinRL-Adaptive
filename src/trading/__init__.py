"""
S&P 500 Trading System

A comprehensive trading system for S&P 500 stocks with:
- Data fetching from Yahoo Finance
- Momentum and mean reversion strategies
- Risk management with position sizing and stop losses
- Portfolio tracking and P&L calculation
- Backtesting engine for historical simulations
- Visualization and reporting capabilities
"""

from .data.fetcher import (
    SP500DataFetcher,
    StockData,
    CacheConfig,
    RateLimitConfig,
)
from .strategies.base import (
    BaseStrategy,
    Signal,
    SignalType,
    Position,
    StrategyConfig,
)
from .strategies.momentum import MomentumStrategy, MomentumConfig
from .strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from .risk.manager import (
    RiskManager,
    RiskConfig,
    PositionSizingMethod,
    StopLossType,
    PositionRisk,
    PortfolioRisk,
)
from .portfolio.portfolio import (
    Portfolio,
    PortfolioPosition,
    PortfolioSnapshot,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Trade,
)
from .backtest.engine import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
)
from .visualization import (
    # Static charts
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_monthly_returns,
    plot_trade_analysis,
    create_performance_dashboard,
    # Interactive charts
    create_dashboard,
    plot_candlestick,
    plot_strategy_comparison,
    create_interactive_equity_curve,
    create_trade_markers,
    # Reports
    generate_html_report,
    generate_pdf_report,
    ReportConfig,
)

__version__ = "1.0.0"

__all__ = [
    # Data
    "SP500DataFetcher",
    "StockData",
    "CacheConfig",
    "RateLimitConfig",
    # Strategies
    "BaseStrategy",
    "Signal",
    "SignalType",
    "Position",
    "StrategyConfig",
    "MomentumStrategy",
    "MomentumConfig",
    "MeanReversionStrategy",
    "MeanReversionConfig",
    # Risk
    "RiskManager",
    "RiskConfig",
    "PositionSizingMethod",
    "StopLossType",
    "PositionRisk",
    "PortfolioRisk",
    # Portfolio
    "Portfolio",
    "PortfolioPosition",
    "PortfolioSnapshot",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "Trade",
    # Backtest
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
    # Visualization - Static charts
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
    "plot_monthly_returns",
    "plot_trade_analysis",
    "create_performance_dashboard",
    # Visualization - Interactive charts
    "create_dashboard",
    "plot_candlestick",
    "plot_strategy_comparison",
    "create_interactive_equity_curve",
    "create_trade_markers",
    # Visualization - Reports
    "generate_html_report",
    "generate_pdf_report",
    "ReportConfig",
]

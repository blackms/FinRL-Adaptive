"""
Visualization Module

Provides comprehensive visualization capabilities for trading analysis including:
- Static charts with matplotlib (equity curves, drawdowns, returns distribution)
- Interactive dashboards with Plotly (candlesticks, strategy comparisons)
- HTML and PDF report generation
"""

from .charts import (
    plot_equity_curve,
    plot_drawdown,
    plot_returns_distribution,
    plot_monthly_returns,
    plot_trade_analysis,
    create_performance_dashboard,
)
from .interactive import (
    create_dashboard,
    plot_candlestick,
    plot_strategy_comparison,
    create_interactive_equity_curve,
    create_trade_markers,
)
from .reports import (
    generate_html_report,
    generate_pdf_report,
    ReportConfig,
)

__all__ = [
    # Static charts
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
    "plot_monthly_returns",
    "plot_trade_analysis",
    "create_performance_dashboard",
    # Interactive charts
    "create_dashboard",
    "plot_candlestick",
    "plot_strategy_comparison",
    "create_interactive_equity_curve",
    "create_trade_markers",
    # Reports
    "generate_html_report",
    "generate_pdf_report",
    "ReportConfig",
]

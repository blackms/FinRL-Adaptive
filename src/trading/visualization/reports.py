"""
Report Generation Module

Provides HTML and PDF report generation for trading analysis.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from ..backtest.engine import BacktestResult


@dataclass
class ReportConfig:
    """
    Report configuration settings.

    Attributes:
        title: Report title.
        subtitle: Report subtitle.
        author: Report author name.
        include_equity_curve: Whether to include equity curve chart.
        include_drawdown: Whether to include drawdown chart.
        include_returns_dist: Whether to include returns distribution.
        include_monthly_returns: Whether to include monthly returns heatmap.
        include_trade_analysis: Whether to include trade analysis.
        include_metrics_table: Whether to include metrics table.
        chart_height: Height of charts in pixels.
        theme: Report theme ('light' or 'dark').
    """

    title: str = "Trading Performance Report"
    subtitle: str = ""
    author: str = ""
    include_equity_curve: bool = True
    include_drawdown: bool = True
    include_returns_dist: bool = True
    include_monthly_returns: bool = True
    include_trade_analysis: bool = True
    include_metrics_table: bool = True
    chart_height: int = 400
    theme: str = "light"


# CSS Styles for the report
REPORT_CSS = """
<style>
    :root {
        --primary-color: #2E86AB;
        --secondary-color: #A23B72;
        --positive-color: #28A745;
        --negative-color: #DC3545;
        --neutral-color: #6C757D;
        --background-color: #F8F9FA;
        --card-bg: #FFFFFF;
        --text-color: #333333;
        --border-color: #DEE2E6;
    }

    * {
        box-sizing: border-box;
        margin: 0;
        padding: 0;
    }

    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        line-height: 1.6;
        color: var(--text-color);
        background-color: var(--background-color);
        padding: 20px;
    }

    .container {
        max-width: 1200px;
        margin: 0 auto;
        background: var(--card-bg);
        padding: 40px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }

    .header {
        text-align: center;
        margin-bottom: 40px;
        padding-bottom: 20px;
        border-bottom: 3px solid var(--primary-color);
    }

    .header h1 {
        font-size: 2.5rem;
        color: var(--primary-color);
        margin-bottom: 10px;
    }

    .header .subtitle {
        font-size: 1.2rem;
        color: var(--neutral-color);
    }

    .header .meta {
        font-size: 0.9rem;
        color: var(--neutral-color);
        margin-top: 10px;
    }

    .section {
        margin-bottom: 40px;
    }

    .section-title {
        font-size: 1.5rem;
        color: var(--primary-color);
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--border-color);
    }

    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
    }

    .metric-card {
        background: var(--background-color);
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        border-left: 4px solid var(--primary-color);
    }

    .metric-card.positive {
        border-left-color: var(--positive-color);
    }

    .metric-card.negative {
        border-left-color: var(--negative-color);
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: var(--text-color);
    }

    .metric-value.positive {
        color: var(--positive-color);
    }

    .metric-value.negative {
        color: var(--negative-color);
    }

    .metric-label {
        font-size: 0.9rem;
        color: var(--neutral-color);
        margin-top: 5px;
    }

    .chart-container {
        margin: 20px 0;
        text-align: center;
    }

    .chart-container img {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }

    th, td {
        padding: 12px 15px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }

    th {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
    }

    tr:nth-child(even) {
        background-color: var(--background-color);
    }

    tr:hover {
        background-color: #E9ECEF;
    }

    .positive-cell {
        color: var(--positive-color);
        font-weight: 600;
    }

    .negative-cell {
        color: var(--negative-color);
        font-weight: 600;
    }

    .summary-box {
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        color: white;
        padding: 30px;
        border-radius: 10px;
        margin: 30px 0;
    }

    .summary-box h3 {
        margin-bottom: 15px;
        font-size: 1.3rem;
    }

    .summary-box p {
        margin: 8px 0;
        font-size: 1.1rem;
    }

    .footer {
        text-align: center;
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid var(--border-color);
        color: var(--neutral-color);
        font-size: 0.9rem;
    }

    @media print {
        body {
            padding: 0;
            background: white;
        }

        .container {
            box-shadow: none;
            padding: 20px;
        }

        .chart-container {
            page-break-inside: avoid;
        }
    }
</style>
"""


def _fig_to_base64(fig: Any, format: str = "png") -> str:
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format=format, dpi=150, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return img_base64


def _plotly_to_html(fig: Any) -> str:
    """Convert Plotly figure to embedded HTML."""
    return fig.to_html(full_html=False, include_plotlyjs="cdn")


def _format_value(value: float, format_type: str = "number") -> tuple[str, str]:
    """
    Format a value and return the formatted string with CSS class.

    Returns:
        Tuple of (formatted_string, css_class).
    """
    if format_type == "percent":
        formatted = f"{value:+.2f}%"
        css_class = "positive" if value > 0 else "negative" if value < 0 else ""
    elif format_type == "currency":
        formatted = f"${value:,.2f}"
        css_class = "positive" if value > 0 else "negative" if value < 0 else ""
    elif format_type == "ratio":
        formatted = f"{value:.2f}"
        css_class = "positive" if value > 1 else "negative" if value < 1 else ""
    else:
        formatted = f"{value:,.2f}"
        css_class = ""

    return formatted, css_class


def _create_metrics_section(results: "BacktestResult") -> str:
    """Create HTML for metrics cards section."""
    metrics = [
        ("Total Return", results.total_return, "percent"),
        ("Annual Return", results.annualized_return, "percent"),
        ("Volatility", results.volatility, "number"),
        ("Sharpe Ratio", results.sharpe_ratio, "ratio"),
        ("Sortino Ratio", results.sortino_ratio, "ratio"),
        ("Max Drawdown", -results.max_drawdown, "percent"),
        ("Calmar Ratio", results.calmar_ratio, "ratio"),
        ("Win Rate", results.win_rate, "percent"),
        ("Profit Factor", results.profit_factor, "ratio"),
        ("Total Trades", results.total_trades, "number"),
    ]

    html_parts = ['<div class="metrics-grid">']

    for label, value, fmt_type in metrics:
        formatted, css_class = _format_value(value, fmt_type)
        card_class = css_class if css_class else ""

        html_parts.append(f"""
            <div class="metric-card {card_class}">
                <div class="metric-value {css_class}">{formatted}</div>
                <div class="metric-label">{label}</div>
            </div>
        """)

    html_parts.append("</div>")
    return "\n".join(html_parts)


def _create_summary_box(results: "BacktestResult") -> str:
    """Create HTML for summary box."""
    initial = results.initial_capital
    final = results.final_value
    period_days = (results.end_date - results.start_date).days

    return f"""
    <div class="summary-box">
        <h3>Portfolio Summary</h3>
        <p><strong>Period:</strong> {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')} ({period_days} days)</p>
        <p><strong>Initial Capital:</strong> ${initial:,.2f}</p>
        <p><strong>Final Value:</strong> ${final:,.2f}</p>
        <p><strong>Net Profit:</strong> ${final - initial:,.2f} ({results.total_return:+.2f}%)</p>
    </div>
    """


def _create_trades_table(results: "BacktestResult", max_trades: int = 20) -> str:
    """Create HTML table for recent trades."""
    trades = results.trades[-max_trades:] if len(results.trades) > max_trades else results.trades

    if not trades:
        return "<p>No trades recorded.</p>"

    html_parts = ["""
    <table>
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Entry Date</th>
                <th>Exit Date</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>Quantity</th>
                <th>P&L</th>
                <th>P&L %</th>
            </tr>
        </thead>
        <tbody>
    """]

    for trade in reversed(trades):
        pnl_class = "positive-cell" if trade.pnl > 0 else "negative-cell"
        html_parts.append(f"""
            <tr>
                <td>{trade.symbol}</td>
                <td>{trade.entry_time.strftime('%Y-%m-%d')}</td>
                <td>{trade.exit_time.strftime('%Y-%m-%d')}</td>
                <td>${trade.entry_price:.2f}</td>
                <td>${trade.exit_price:.2f}</td>
                <td>{trade.quantity:.2f}</td>
                <td class="{pnl_class}">${trade.pnl:+,.2f}</td>
                <td class="{pnl_class}">{trade.pnl_pct:+.2f}%</td>
            </tr>
        """)

    html_parts.append("</tbody></table>")

    if len(results.trades) > max_trades:
        html_parts.append(f"<p><em>Showing most recent {max_trades} of {len(results.trades)} trades.</em></p>")

    return "\n".join(html_parts)


def generate_html_report(
    results: "BacktestResult",
    output_path: str | Path,
    config: ReportConfig | None = None,
    use_interactive: bool = True,
) -> Path:
    """
    Generate a comprehensive HTML report.

    Args:
        results: BacktestResult object.
        output_path: Path to save the HTML report.
        config: Report configuration settings.
        use_interactive: Whether to use interactive Plotly charts (if False, uses matplotlib).

    Returns:
        Path to the generated report.
    """
    config = config or ReportConfig()
    output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build HTML content
    html_parts = [
        "<!DOCTYPE html>",
        "<html lang='en'>",
        "<head>",
        "<meta charset='UTF-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1.0'>",
        f"<title>{config.title}</title>",
        REPORT_CSS,
        "</head>",
        "<body>",
        "<div class='container'>",
    ]

    # Header
    html_parts.append(f"""
    <div class="header">
        <h1>{config.title}</h1>
        {"<p class='subtitle'>" + config.subtitle + "</p>" if config.subtitle else ""}
        <p class="meta">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            {" | Author: " + config.author if config.author else ""}
        </p>
    </div>
    """)

    # Summary box
    html_parts.append(_create_summary_box(results))

    # Metrics section
    if config.include_metrics_table:
        html_parts.append("""
        <div class="section">
            <h2 class="section-title">Key Performance Metrics</h2>
        """)
        html_parts.append(_create_metrics_section(results))
        html_parts.append("</div>")

    # Charts section
    if use_interactive:
        # Use Plotly for interactive charts
        from .interactive import create_interactive_equity_curve, plot_strategy_comparison

        if config.include_equity_curve:
            html_parts.append("""
            <div class="section">
                <h2 class="section-title">Portfolio Performance</h2>
                <div class="chart-container">
            """)
            try:
                fig = create_interactive_equity_curve(
                    results,
                    height=config.chart_height,
                    show_drawdown=config.include_drawdown,
                )
                html_parts.append(_plotly_to_html(fig))
            except Exception as e:
                html_parts.append(f"<p>Could not generate equity curve: {e}</p>")
            html_parts.append("</div></div>")

    else:
        # Use matplotlib for static charts
        from .charts import (
            plot_equity_curve,
            plot_drawdown,
            plot_returns_distribution,
            plot_monthly_returns,
            plot_trade_analysis,
        )
        import matplotlib.pyplot as plt

        if config.include_equity_curve:
            html_parts.append("""
            <div class="section">
                <h2 class="section-title">Equity Curve</h2>
                <div class="chart-container">
            """)
            try:
                fig = plot_equity_curve(results, show_drawdown=False)
                img_base64 = _fig_to_base64(fig)
                html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Equity Curve">')
                plt.close(fig)
            except Exception as e:
                html_parts.append(f"<p>Could not generate equity curve: {e}</p>")
            html_parts.append("</div></div>")

        if config.include_drawdown:
            html_parts.append("""
            <div class="section">
                <h2 class="section-title">Drawdown Analysis</h2>
                <div class="chart-container">
            """)
            try:
                fig = plot_drawdown(results)
                img_base64 = _fig_to_base64(fig)
                html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Drawdown">')
                plt.close(fig)
            except Exception as e:
                html_parts.append(f"<p>Could not generate drawdown chart: {e}</p>")
            html_parts.append("</div></div>")

        if config.include_returns_dist:
            html_parts.append("""
            <div class="section">
                <h2 class="section-title">Returns Distribution</h2>
                <div class="chart-container">
            """)
            try:
                fig = plot_returns_distribution(results)
                img_base64 = _fig_to_base64(fig)
                html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Returns Distribution">')
                plt.close(fig)
            except Exception as e:
                html_parts.append(f"<p>Could not generate returns distribution: {e}</p>")
            html_parts.append("</div></div>")

        if config.include_monthly_returns:
            html_parts.append("""
            <div class="section">
                <h2 class="section-title">Monthly Returns</h2>
                <div class="chart-container">
            """)
            try:
                fig = plot_monthly_returns(results)
                img_base64 = _fig_to_base64(fig)
                html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Monthly Returns">')
                plt.close(fig)
            except Exception as e:
                html_parts.append(f"<p>Could not generate monthly returns heatmap: {e}</p>")
            html_parts.append("</div></div>")

        if config.include_trade_analysis and results.trades:
            html_parts.append("""
            <div class="section">
                <h2 class="section-title">Trade Analysis</h2>
                <div class="chart-container">
            """)
            try:
                fig = plot_trade_analysis(results)
                img_base64 = _fig_to_base64(fig)
                html_parts.append(f'<img src="data:image/png;base64,{img_base64}" alt="Trade Analysis">')
                plt.close(fig)
            except Exception as e:
                html_parts.append(f"<p>Could not generate trade analysis: {e}</p>")
            html_parts.append("</div></div>")

    # Trades table
    if results.trades:
        html_parts.append("""
        <div class="section">
            <h2 class="section-title">Recent Trades</h2>
        """)
        html_parts.append(_create_trades_table(results))
        html_parts.append("</div>")

    # Footer
    html_parts.append("""
    <div class="footer">
        <p>Generated by FinRL Trading System</p>
    </div>
    """)

    html_parts.extend(["</div>", "</body>", "</html>"])

    # Write to file
    html_content = "\n".join(html_parts)
    output_path.write_text(html_content, encoding="utf-8")

    return output_path


def generate_pdf_report(
    results: "BacktestResult",
    output_path: str | Path,
    config: ReportConfig | None = None,
) -> Path | None:
    """
    Generate a PDF report.

    Note: This function requires weasyprint or similar PDF generation library.
    Falls back to HTML if PDF generation fails.

    Args:
        results: BacktestResult object.
        output_path: Path to save the PDF report.
        config: Report configuration settings.

    Returns:
        Path to the generated report, or None if PDF generation failed.
    """
    config = config or ReportConfig()
    output_path = Path(output_path)

    # First generate HTML
    html_path = output_path.with_suffix(".html")
    generate_html_report(results, html_path, config, use_interactive=False)

    try:
        # Try to import weasyprint for PDF generation
        from weasyprint import HTML

        HTML(filename=str(html_path)).write_pdf(str(output_path))
        return output_path

    except ImportError:
        # weasyprint not available, try pdfkit as alternative
        try:
            import pdfkit

            pdfkit.from_file(str(html_path), str(output_path))
            return output_path

        except ImportError:
            # No PDF library available
            import warnings
            warnings.warn(
                "PDF generation requires 'weasyprint' or 'pdfkit' package. "
                "Install with: pip install weasyprint or pip install pdfkit\n"
                f"HTML report saved to: {html_path}"
            )
            return None

    except Exception as e:
        import warnings
        warnings.warn(f"PDF generation failed: {e}. HTML report saved to: {html_path}")
        return None


__all__ = [
    "generate_html_report",
    "generate_pdf_report",
    "ReportConfig",
]

"""
Static Charts Module

Provides publication-quality matplotlib charts for trading analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec

if TYPE_CHECKING:
    from ..backtest.engine import BacktestResult

# Publication-quality style settings
STYLE_CONFIG = {
    "figure.figsize": (12, 8),
    "figure.dpi": 100,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "axes.grid.axis": "both",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelsize": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "legend.fontsize": 10,
    "legend.framealpha": 0.9,
    "font.family": "sans-serif",
    "font.size": 10,
    "lines.linewidth": 1.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
}

# Color palette for consistency
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "positive": "#28A745",
    "negative": "#DC3545",
    "neutral": "#6C757D",
    "background": "#F8F9FA",
    "grid": "#DEE2E6",
    "benchmark": "#FFC107",
}


def _apply_style() -> None:
    """Apply publication-quality style settings."""
    plt.rcParams.update(STYLE_CONFIG)


def _format_currency(x: float, _: Any) -> str:
    """Format value as currency."""
    if abs(x) >= 1e6:
        return f"${x/1e6:.1f}M"
    elif abs(x) >= 1e3:
        return f"${x/1e3:.0f}K"
    else:
        return f"${x:.0f}"


def _format_percent(x: float, _: Any) -> str:
    """Format value as percentage."""
    return f"{x:.1f}%"


def plot_equity_curve(
    results: "BacktestResult",
    benchmark: pd.DataFrame | None = None,
    title: str = "Portfolio Equity Curve",
    figsize: tuple[float, float] = (12, 6),
    show_drawdown: bool = True,
    save_path: str | None = None,
) -> Figure:
    """
    Plot portfolio value over time.

    Args:
        results: BacktestResult object containing equity curve data.
        benchmark: Optional benchmark data with 'date' and 'value' columns.
        title: Chart title.
        figsize: Figure size (width, height).
        show_drawdown: Whether to show drawdown in a subplot.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    _apply_style()

    equity_df = results.equity_curve
    if equity_df.empty:
        raise ValueError("Equity curve data is empty")

    # Create figure with subplots
    if show_drawdown:
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=figsize, height_ratios=[3, 1], sharex=True
        )
    else:
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = None

    # Plot equity curve
    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["total_value"]

    ax1.plot(dates, values, color=COLORS["primary"], linewidth=2, label="Portfolio")
    ax1.fill_between(
        dates, values.min() * 0.95, values, alpha=0.1, color=COLORS["primary"]
    )

    # Plot benchmark if provided
    if benchmark is not None and not benchmark.empty:
        bench_dates = pd.to_datetime(benchmark["date"])
        bench_values = benchmark["value"]
        # Normalize to same starting point
        bench_normalized = bench_values / bench_values.iloc[0] * values.iloc[0]
        ax1.plot(
            bench_dates,
            bench_normalized,
            color=COLORS["benchmark"],
            linewidth=1.5,
            linestyle="--",
            label="Benchmark",
        )

    # Mark peak
    peak_idx = values.idxmax()
    ax1.scatter(
        dates.iloc[peak_idx],
        values.iloc[peak_idx],
        color=COLORS["positive"],
        s=100,
        zorder=5,
        marker="^",
        label=f"Peak: ${values.iloc[peak_idx]:,.0f}",
    )

    ax1.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax1.set_ylabel("Portfolio Value ($)", fontsize=11)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_format_currency))
    ax1.legend(loc="upper left", framealpha=0.9)

    # Add performance annotations
    total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
    ann_text = f"Total Return: {total_return:+.2f}%\nFinal Value: ${values.iloc[-1]:,.0f}"
    ax1.annotate(
        ann_text,
        xy=(0.98, 0.02),
        xycoords="axes fraction",
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Plot drawdown
    if show_drawdown and ax2 is not None:
        # Calculate drawdown
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax * 100

        ax2.fill_between(dates, 0, drawdown, color=COLORS["negative"], alpha=0.3)
        ax2.plot(dates, drawdown, color=COLORS["negative"], linewidth=1)

        ax2.set_ylabel("Drawdown (%)", fontsize=11)
        ax2.set_xlabel("Date", fontsize=11)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(_format_percent))

        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        ax2.scatter(
            dates.iloc[max_dd_idx],
            drawdown.iloc[max_dd_idx],
            color=COLORS["negative"],
            s=80,
            zorder=5,
            marker="v",
        )
        ax2.annotate(
            f"Max DD: {drawdown.iloc[max_dd_idx]:.2f}%",
            xy=(dates.iloc[max_dd_idx], drawdown.iloc[max_dd_idx]),
            xytext=(10, -10),
            textcoords="offset points",
            fontsize=9,
        )

    # Format x-axis dates
    ax = ax2 if ax2 is not None else ax1
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 365 * 2)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_drawdown(
    results: "BacktestResult",
    title: str = "Portfolio Drawdown",
    figsize: tuple[float, float] = (12, 5),
    save_path: str | None = None,
) -> Figure:
    """
    Plot drawdown chart.

    Args:
        results: BacktestResult object containing equity curve data.
        title: Chart title.
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    _apply_style()

    equity_df = results.equity_curve
    if equity_df.empty:
        raise ValueError("Equity curve data is empty")

    fig, ax = plt.subplots(figsize=figsize)

    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["total_value"]

    # Calculate drawdown
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100

    # Plot underwater curve
    ax.fill_between(dates, 0, drawdown, color=COLORS["negative"], alpha=0.4, label="Drawdown")
    ax.plot(dates, drawdown, color=COLORS["negative"], linewidth=1.5)

    # Identify drawdown periods
    in_drawdown = drawdown < 0
    drawdown_starts = in_drawdown & ~in_drawdown.shift(1, fill_value=False)
    drawdown_ends = ~in_drawdown & in_drawdown.shift(1, fill_value=False)

    # Calculate recovery periods
    dd_periods = []
    current_start = None
    for i, (start, end) in enumerate(zip(drawdown_starts, drawdown_ends)):
        if start:
            current_start = i
        if end and current_start is not None:
            dd_periods.append((current_start, i))
            current_start = None

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Annotate significant drawdowns
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()
    ax.scatter(
        dates.iloc[max_dd_idx],
        max_dd,
        color=COLORS["negative"],
        s=100,
        zorder=5,
        marker="v",
    )
    ax.annotate(
        f"Max Drawdown: {max_dd:.2f}%",
        xy=(dates.iloc[max_dd_idx], max_dd),
        xytext=(20, -20),
        textcoords="offset points",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color=COLORS["negative"]),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (%)", fontsize=11)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_percent))

    # Add statistics box
    avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0
    stats_text = f"Max Drawdown: {max_dd:.2f}%\nAvg Drawdown: {avg_dd:.2f}%"
    ax.annotate(
        stats_text,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        ha="left",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(dates) // 365 * 2)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_returns_distribution(
    results: "BacktestResult",
    title: str = "Returns Distribution",
    figsize: tuple[float, float] = (12, 5),
    bins: int = 50,
    save_path: str | None = None,
) -> Figure:
    """
    Plot histogram of returns distribution.

    Args:
        results: BacktestResult object containing equity curve data.
        title: Chart title.
        figsize: Figure size (width, height).
        bins: Number of histogram bins.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    _apply_style()

    equity_df = results.equity_curve
    if equity_df.empty or "returns" not in equity_df.columns:
        # Calculate returns if not present
        values = equity_df["total_value"]
        returns = values.pct_change().dropna() * 100
    else:
        returns = equity_df["returns"].dropna() * 100

    if len(returns) == 0:
        raise ValueError("Insufficient data for returns distribution")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram
    n, bins_edges, patches = ax.hist(
        returns,
        bins=bins,
        density=True,
        alpha=0.7,
        color=COLORS["primary"],
        edgecolor="white",
        linewidth=0.5,
    )

    # Color positive/negative returns differently
    for i, (patch, left_edge) in enumerate(zip(patches, bins_edges[:-1])):
        if left_edge < 0:
            patch.set_facecolor(COLORS["negative"])
        else:
            patch.set_facecolor(COLORS["positive"])

    # Add normal distribution fit
    mu = returns.mean()
    sigma = returns.std()
    x = np.linspace(returns.min(), returns.max(), 100)
    from scipy import stats
    y = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, y, color=COLORS["secondary"], linewidth=2, linestyle="--", label="Normal Fit")

    # Add vertical lines for mean and zero
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.5)
    ax.axvline(x=mu, color=COLORS["primary"], linestyle="--", linewidth=1.5, label=f"Mean: {mu:.2f}%")

    # Add VaR and CVaR lines
    var_95 = returns.quantile(0.05)
    ax.axvline(x=var_95, color=COLORS["negative"], linestyle=":", linewidth=1.5, label=f"VaR 95%: {var_95:.2f}%")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Daily Returns (%)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.legend(loc="upper right", framealpha=0.9)

    # Add statistics box
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    positive_days = (returns > 0).sum()
    total_days = len(returns)
    win_pct = positive_days / total_days * 100

    stats_text = (
        f"Mean: {mu:.2f}%\n"
        f"Std Dev: {sigma:.2f}%\n"
        f"Skewness: {skewness:.2f}\n"
        f"Kurtosis: {kurtosis:.2f}\n"
        f"Win Rate: {win_pct:.1f}%"
    )
    ax.annotate(
        stats_text,
        xy=(0.02, 0.98),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_monthly_returns(
    results: "BacktestResult",
    title: str = "Monthly Returns Heatmap",
    figsize: tuple[float, float] = (14, 8),
    cmap: str = "RdYlGn",
    save_path: str | None = None,
) -> Figure:
    """
    Plot heatmap of monthly returns.

    Args:
        results: BacktestResult object containing equity curve data.
        title: Chart title.
        figsize: Figure size (width, height).
        cmap: Colormap name.
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    _apply_style()

    equity_df = results.equity_curve
    if equity_df.empty:
        raise ValueError("Equity curve data is empty")

    # Prepare data
    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)

    # Calculate monthly returns
    monthly_values = df["total_value"].resample("ME").last()
    monthly_returns = monthly_values.pct_change() * 100

    # Create pivot table for heatmap
    monthly_df = pd.DataFrame({
        "year": monthly_returns.index.year,
        "month": monthly_returns.index.month,
        "return": monthly_returns.values,
    }).dropna()

    if monthly_df.empty:
        raise ValueError("Insufficient data for monthly returns heatmap")

    pivot_table = monthly_df.pivot(index="year", columns="month", values="return")

    # Reorder months
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot_table.columns = [month_names[m - 1] for m in pivot_table.columns]

    fig, ax = plt.subplots(figsize=figsize)

    # Create heatmap
    vmax = max(abs(pivot_table.min().min()), abs(pivot_table.max().max()))
    im = ax.imshow(pivot_table.values, cmap=cmap, aspect="auto", vmin=-vmax, vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=0.8)
    cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_percent))

    # Set ticks
    ax.set_xticks(np.arange(len(pivot_table.columns)))
    ax.set_yticks(np.arange(len(pivot_table.index)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticklabels(pivot_table.index.astype(int))

    # Add text annotations
    for i in range(len(pivot_table.index)):
        for j in range(len(pivot_table.columns)):
            value = pivot_table.iloc[i, j]
            if pd.notna(value):
                text_color = "white" if abs(value) > vmax * 0.5 else "black"
                ax.text(j, i, f"{value:.1f}%", ha="center", va="center",
                        color=text_color, fontsize=9, fontweight="bold")

    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Year", fontsize=11)

    # Add yearly totals on the right
    yearly_returns = monthly_df.groupby("year")["return"].sum()
    ax2 = ax.twinx()
    ax2.set_yticks(np.arange(len(pivot_table.index)))
    ax2.set_yticklabels([f"{r:+.1f}%" for r in yearly_returns.values])
    ax2.set_ylabel("Yearly Return", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def plot_trade_analysis(
    results: "BacktestResult",
    title: str = "Trade Analysis",
    figsize: tuple[float, float] = (14, 10),
    save_path: str | None = None,
) -> Figure:
    """
    Plot comprehensive trade analysis.

    Args:
        results: BacktestResult object containing trades data.
        title: Chart title.
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    _apply_style()

    trades = results.trades
    if not trades:
        raise ValueError("No trades to analyze")

    # Extract trade data
    trade_df = pd.DataFrame([
        {
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "pnl": t.pnl,
            "pnl_pct": t.pnl_pct,
            "symbol": t.symbol,
            "duration": (t.exit_time - t.entry_time).days if t.exit_time and t.entry_time else 0,
        }
        for t in trades
    ])

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

    # 1. Trade P&L scatter by date
    ax1 = fig.add_subplot(gs[0, 0])
    colors = [COLORS["positive"] if pnl > 0 else COLORS["negative"] for pnl in trade_df["pnl"]]
    ax1.scatter(trade_df["exit_time"], trade_df["pnl"], c=colors, alpha=0.6, s=50)
    ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_title("Trade P&L Over Time", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Exit Date", fontsize=10)
    ax1.set_ylabel("P&L ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_format_currency))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 2. Trade P&L distribution
    ax2 = fig.add_subplot(gs[0, 1])
    winners = trade_df[trade_df["pnl"] > 0]["pnl"]
    losers = trade_df[trade_df["pnl"] <= 0]["pnl"]

    if len(winners) > 0:
        ax2.hist(winners, bins=20, alpha=0.7, color=COLORS["positive"], label="Winners", edgecolor="white")
    if len(losers) > 0:
        ax2.hist(losers, bins=20, alpha=0.7, color=COLORS["negative"], label="Losers", edgecolor="white")

    ax2.axvline(x=trade_df["pnl"].mean(), color=COLORS["primary"], linestyle="--", linewidth=2, label=f"Mean: ${trade_df['pnl'].mean():.0f}")
    ax2.set_title("P&L Distribution", fontsize=12, fontweight="bold")
    ax2.set_xlabel("P&L ($)", fontsize=10)
    ax2.set_ylabel("Frequency", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    # 3. Cumulative P&L
    ax3 = fig.add_subplot(gs[1, 0])
    cumulative_pnl = trade_df["pnl"].cumsum()
    ax3.plot(range(len(cumulative_pnl)), cumulative_pnl, color=COLORS["primary"], linewidth=2)
    ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                     where=(cumulative_pnl >= 0), alpha=0.3, color=COLORS["positive"])
    ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                     where=(cumulative_pnl < 0), alpha=0.3, color=COLORS["negative"])
    ax3.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_title("Cumulative P&L", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Trade Number", fontsize=10)
    ax3.set_ylabel("Cumulative P&L ($)", fontsize=10)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(_format_currency))

    # 4. Trade statistics summary
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    win_rate = len(winners) / len(trade_df) * 100 if len(trade_df) > 0 else 0
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = losers.mean() if len(losers) > 0 else 0
    profit_factor = abs(winners.sum() / losers.sum()) if len(losers) > 0 and losers.sum() != 0 else float("inf")
    avg_duration = trade_df["duration"].mean()

    stats_text = (
        f"Trade Statistics\n"
        f"{'=' * 40}\n\n"
        f"Total Trades:        {len(trade_df)}\n"
        f"Winning Trades:      {len(winners)}\n"
        f"Losing Trades:       {len(losers)}\n"
        f"Win Rate:            {win_rate:.1f}%\n\n"
        f"Average Win:         ${avg_win:,.0f}\n"
        f"Average Loss:        ${avg_loss:,.0f}\n"
        f"Profit Factor:       {profit_factor:.2f}\n\n"
        f"Total P&L:           ${trade_df['pnl'].sum():,.0f}\n"
        f"Avg Trade Duration:  {avg_duration:.1f} days"
    )

    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["background"], alpha=0.8))

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


def create_performance_dashboard(
    results: "BacktestResult",
    title: str = "Performance Dashboard",
    figsize: tuple[float, float] = (16, 12),
    save_path: str | None = None,
) -> Figure:
    """
    Create a comprehensive performance dashboard with multiple charts.

    Args:
        results: BacktestResult object.
        title: Dashboard title.
        figsize: Figure size (width, height).
        save_path: Optional path to save the figure.

    Returns:
        Matplotlib Figure object.
    """
    _apply_style()

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    equity_df = results.equity_curve
    if equity_df.empty:
        raise ValueError("Equity curve data is empty")

    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["total_value"]

    # 1. Equity Curve (large, top row)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(dates, values, color=COLORS["primary"], linewidth=2, label="Portfolio")
    ax1.fill_between(dates, values.min() * 0.95, values, alpha=0.1, color=COLORS["primary"])
    ax1.set_title("Equity Curve", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Value ($)", fontsize=10)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_format_currency))
    ax1.legend(loc="upper left")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100
    ax2.fill_between(dates, 0, drawdown, color=COLORS["negative"], alpha=0.4)
    ax2.plot(dates, drawdown, color=COLORS["negative"], linewidth=1)
    ax2.set_title("Drawdown", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 3. Returns Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    returns = values.pct_change().dropna() * 100
    ax3.hist(returns, bins=30, color=COLORS["primary"], alpha=0.7, edgecolor="white")
    ax3.axvline(x=returns.mean(), color=COLORS["secondary"], linestyle="--", linewidth=2)
    ax3.set_title("Returns Distribution", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Daily Return (%)", fontsize=10)
    ax3.set_ylabel("Frequency", fontsize=10)

    # 4. Rolling Sharpe Ratio
    ax4 = fig.add_subplot(gs[1, 2])
    window = min(63, len(returns) // 4)  # ~3 months
    if window > 5:
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
        ax4.plot(dates[window:], rolling_sharpe.dropna(), color=COLORS["primary"], linewidth=1.5)
        ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax4.axhline(y=1, color=COLORS["positive"], linestyle="--", linewidth=1, alpha=0.5)
    ax4.set_title(f"Rolling Sharpe ({window}d)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Sharpe Ratio", fontsize=10)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 5. Cumulative Trade P&L
    ax5 = fig.add_subplot(gs[2, 0])
    if results.trades:
        trade_pnls = [t.pnl for t in results.trades]
        cum_pnl = np.cumsum(trade_pnls)
        ax5.plot(range(len(cum_pnl)), cum_pnl, color=COLORS["primary"], linewidth=2)
        ax5.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                        where=(np.array(cum_pnl) >= 0), alpha=0.3, color=COLORS["positive"])
        ax5.fill_between(range(len(cum_pnl)), 0, cum_pnl,
                        where=(np.array(cum_pnl) < 0), alpha=0.3, color=COLORS["negative"])
    ax5.set_title("Cumulative Trade P&L", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Trade #", fontsize=10)
    ax5.set_ylabel("P&L ($)", fontsize=10)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(_format_currency))

    # 6. Win/Loss breakdown
    ax6 = fig.add_subplot(gs[2, 1])
    if results.winning_trades > 0 or results.losing_trades > 0:
        labels = ["Wins", "Losses"]
        sizes = [results.winning_trades, results.losing_trades]
        colors = [COLORS["positive"], COLORS["negative"]]
        explode = (0.05, 0)
        ax6.pie(sizes, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%",
                shadow=True, startangle=90)
    ax6.set_title("Win/Loss Ratio", fontsize=12, fontweight="bold")

    # 7. Key Metrics
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis("off")

    metrics_text = (
        f"Key Metrics\n"
        f"{'=' * 30}\n\n"
        f"Total Return:     {results.total_return:+.2f}%\n"
        f"Annual Return:    {results.annualized_return:+.2f}%\n"
        f"Volatility:       {results.volatility:.2f}%\n"
        f"Sharpe Ratio:     {results.sharpe_ratio:.2f}\n"
        f"Sortino Ratio:    {results.sortino_ratio:.2f}\n"
        f"Max Drawdown:     {results.max_drawdown:.2f}%\n"
        f"Calmar Ratio:     {results.calmar_ratio:.2f}\n\n"
        f"Total Trades:     {results.total_trades}\n"
        f"Win Rate:         {results.win_rate:.1f}%\n"
        f"Profit Factor:    {results.profit_factor:.2f}"
    )

    ax7.text(0.1, 0.9, metrics_text, transform=ax7.transAxes, fontsize=10,
             verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor=COLORS["background"], alpha=0.8))

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")

    return fig


__all__ = [
    "plot_equity_curve",
    "plot_drawdown",
    "plot_returns_distribution",
    "plot_monthly_returns",
    "plot_trade_analysis",
    "create_performance_dashboard",
]

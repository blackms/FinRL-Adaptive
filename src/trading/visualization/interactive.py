"""
Interactive Charts Module

Provides interactive Plotly charts and dashboards for trading analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if TYPE_CHECKING:
    from ..backtest.engine import BacktestResult
    from ..strategies.base import Signal

# Color scheme for consistency
COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "positive": "#28A745",
    "negative": "#DC3545",
    "neutral": "#6C757D",
    "background": "#F8F9FA",
    "grid": "#DEE2E6",
    "candle_up": "#28A745",
    "candle_down": "#DC3545",
    "volume": "#6C757D",
    "buy_marker": "#00D4AA",
    "sell_marker": "#FF6B6B",
}

# Default layout settings for publication quality
DEFAULT_LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Arial, sans-serif", "size": 12},
    "title_font": {"size": 16, "color": "#333333"},
    "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    "hovermode": "x unified",
    "margin": {"l": 60, "r": 40, "t": 80, "b": 60},
}


def create_interactive_equity_curve(
    results: "BacktestResult",
    benchmark: pd.DataFrame | None = None,
    title: str = "Portfolio Equity Curve",
    height: int = 500,
    show_drawdown: bool = True,
) -> go.Figure:
    """
    Create an interactive equity curve chart.

    Args:
        results: BacktestResult object containing equity curve data.
        benchmark: Optional benchmark data with 'date' and 'value' columns.
        title: Chart title.
        height: Chart height in pixels.
        show_drawdown: Whether to show drawdown as secondary y-axis.

    Returns:
        Plotly Figure object.
    """
    equity_df = results.equity_curve
    if equity_df.empty:
        raise ValueError("Equity curve data is empty")

    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["total_value"]

    # Calculate drawdown
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100

    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2 if show_drawdown else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3] if show_drawdown else [1.0],
        subplot_titles=("Portfolio Value", "Drawdown") if show_drawdown else None,
    )

    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Portfolio",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor=f"rgba(46, 134, 171, 0.1)",
            hovertemplate="Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add benchmark if provided
    if benchmark is not None and not benchmark.empty:
        bench_dates = pd.to_datetime(benchmark["date"])
        bench_values = benchmark["value"]
        # Normalize to same starting point
        bench_normalized = bench_values / bench_values.iloc[0] * values.iloc[0]
        fig.add_trace(
            go.Scatter(
                x=bench_dates,
                y=bench_normalized,
                mode="lines",
                name="Benchmark",
                line=dict(color=COLORS["secondary"], width=1.5, dash="dash"),
                hovertemplate="Date: %{x}<br>Benchmark: $%{y:,.0f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # Mark peak and current values
    peak_idx = values.idxmax()
    fig.add_trace(
        go.Scatter(
            x=[dates.iloc[peak_idx]],
            y=[values.iloc[peak_idx]],
            mode="markers",
            name=f"Peak: ${values.iloc[peak_idx]:,.0f}",
            marker=dict(color=COLORS["positive"], size=12, symbol="triangle-up"),
            hovertemplate="Peak<br>Date: %{x}<br>Value: $%{y:,.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add drawdown
    if show_drawdown:
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                mode="lines",
                name="Drawdown",
                line=dict(color=COLORS["negative"], width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba(220, 53, 69, 0.3)",
                hovertemplate="Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        fig.add_trace(
            go.Scatter(
                x=[dates.iloc[max_dd_idx]],
                y=[drawdown.iloc[max_dd_idx]],
                mode="markers",
                name=f"Max DD: {drawdown.iloc[max_dd_idx]:.2f}%",
                marker=dict(color=COLORS["negative"], size=10, symbol="triangle-down"),
                showlegend=False,
            ),
            row=2,
            col=1,
        )

    # Add annotations
    total_return = (values.iloc[-1] / values.iloc[0] - 1) * 100
    fig.add_annotation(
        x=0.02,
        y=0.98,
        xref="paper",
        yref="paper",
        text=f"Total Return: {total_return:+.2f}%<br>Final Value: ${values.iloc[-1]:,.0f}",
        showarrow=False,
        font=dict(size=12),
        bgcolor="white",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        borderpad=6,
        align="left",
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=height if not show_drawdown else height + 200,
        **DEFAULT_LAYOUT,
    )

    fig.update_yaxes(title_text="Portfolio Value ($)", tickformat="$,.0f", row=1, col=1)
    if show_drawdown:
        fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2 if show_drawdown else 1, col=1)

    return fig


def plot_candlestick(
    data: pd.DataFrame,
    signals: list["Signal"] | None = None,
    title: str = "Price Chart",
    height: int = 600,
    show_volume: bool = True,
    show_indicators: bool = True,
) -> go.Figure:
    """
    Create an interactive candlestick chart with buy/sell markers.

    Args:
        data: DataFrame with OHLCV data (columns: datetime/date, open, high, low, close, volume).
        signals: Optional list of Signal objects to mark on the chart.
        title: Chart title.
        height: Chart height in pixels.
        show_volume: Whether to show volume bars.
        show_indicators: Whether to show moving averages.

    Returns:
        Plotly Figure object.
    """
    df = data.copy()

    # Normalize column names
    col_mapping = {
        "Date": "date", "datetime": "date", "Datetime": "date",
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
    }
    df = df.rename(columns=col_mapping)

    if "date" not in df.columns:
        if df.index.name in ["date", "Date", "datetime", "Datetime"]:
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
        else:
            raise ValueError("Data must have a date/datetime column")

    df["date"] = pd.to_datetime(df["date"])

    # Create subplots
    rows = 2 if show_volume else 1
    row_heights = [0.8, 0.2] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
    )

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color=COLORS["candle_up"],
            decreasing_line_color=COLORS["candle_down"],
            increasing_fillcolor=COLORS["candle_up"],
            decreasing_fillcolor=COLORS["candle_down"],
        ),
        row=1,
        col=1,
    )

    # Add moving averages if requested
    if show_indicators and len(df) > 20:
        # 20-day SMA
        df["sma_20"] = df["close"].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["sma_20"],
                mode="lines",
                name="SMA 20",
                line=dict(color=COLORS["primary"], width=1.5),
            ),
            row=1,
            col=1,
        )

        # 50-day SMA if enough data
        if len(df) > 50:
            df["sma_50"] = df["close"].rolling(window=50).mean()
            fig.add_trace(
                go.Scatter(
                    x=df["date"],
                    y=df["sma_50"],
                    mode="lines",
                    name="SMA 50",
                    line=dict(color=COLORS["secondary"], width=1.5),
                ),
                row=1,
                col=1,
            )

    # Add buy/sell markers from signals
    if signals:
        buy_signals = [s for s in signals if s.is_buy]
        sell_signals = [s for s in signals if s.is_sell]

        if buy_signals:
            buy_dates = [s.timestamp for s in buy_signals]
            buy_prices = [s.price for s in buy_signals]
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(
                        color=COLORS["buy_marker"],
                        size=14,
                        symbol="triangle-up",
                        line=dict(color="white", width=1),
                    ),
                    hovertemplate="Buy<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

        if sell_signals:
            sell_dates = [s.timestamp for s in sell_signals]
            sell_prices = [s.price for s in sell_signals]
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(
                        color=COLORS["sell_marker"],
                        size=14,
                        symbol="triangle-down",
                        line=dict(color="white", width=1),
                    ),
                    hovertemplate="Sell<br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    # Add volume bars
    if show_volume and "volume" in df.columns:
        colors = [
            COLORS["candle_up"] if close >= open_price else COLORS["candle_down"]
            for close, open_price in zip(df["close"], df["open"])
        ]

        fig.add_trace(
            go.Bar(
                x=df["date"],
                y=df["volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=height,
        xaxis_rangeslider_visible=False,
        **DEFAULT_LAYOUT,
    )

    fig.update_yaxes(title_text="Price ($)", tickformat="$.2f", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=rows, col=1)

    return fig


def create_trade_markers(
    results: "BacktestResult",
) -> tuple[list[datetime], list[float], list[datetime], list[float]]:
    """
    Extract buy and sell markers from backtest results.

    Args:
        results: BacktestResult object.

    Returns:
        Tuple of (buy_dates, buy_prices, sell_dates, sell_prices).
    """
    buy_dates = []
    buy_prices = []
    sell_dates = []
    sell_prices = []

    for trade in results.trades:
        buy_dates.append(trade.entry_time)
        buy_prices.append(trade.entry_price)
        sell_dates.append(trade.exit_time)
        sell_prices.append(trade.exit_price)

    return buy_dates, buy_prices, sell_dates, sell_prices


def plot_strategy_comparison(
    results_dict: dict[str, "BacktestResult"],
    title: str = "Strategy Comparison",
    height: int = 700,
    normalize: bool = True,
) -> go.Figure:
    """
    Create an interactive chart comparing multiple strategies.

    Args:
        results_dict: Dictionary mapping strategy names to BacktestResult objects.
        title: Chart title.
        height: Chart height in pixels.
        normalize: Whether to normalize values to start at 100.

    Returns:
        Plotly Figure object.
    """
    if not results_dict:
        raise ValueError("No results provided for comparison")

    # Create figure with secondary y-axis for drawdown
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=True,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
        subplot_titles=(
            "Cumulative Returns",
            "Rolling Sharpe Ratio (63d)",
            "Drawdown",
            "Performance Metrics",
        ),
        specs=[[{}, {}], [{}, {"type": "table"}]],
    )

    colors = [
        "#2E86AB", "#A23B72", "#F18F01", "#048A81", "#54C6EB",
        "#8338EC", "#FF006E", "#3A86FF", "#06D6A0", "#EF476F",
    ]

    metrics_data = []

    for i, (name, results) in enumerate(results_dict.items()):
        color = colors[i % len(colors)]
        equity_df = results.equity_curve

        if equity_df.empty:
            continue

        dates = pd.to_datetime(equity_df["date"])
        values = equity_df["total_value"]

        # Normalize if requested
        if normalize:
            normalized_values = values / values.iloc[0] * 100
            y_label = "Value (Indexed to 100)"
        else:
            normalized_values = values
            y_label = "Portfolio Value ($)"

        # 1. Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=normalized_values,
                mode="lines",
                name=name,
                line=dict(color=color, width=2),
                legendgroup=name,
                hovertemplate=f"{name}<br>Date: %{{x}}<br>Value: %{{y:,.2f}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 2. Rolling Sharpe ratio
        returns = values.pct_change()
        window = min(63, len(returns) // 4)
        if window > 5:
            rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=rolling_sharpe,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.5),
                    legendgroup=name,
                    showlegend=False,
                    hovertemplate=f"{name}<br>Date: %{{x}}<br>Sharpe: %{{y:.2f}}<extra></extra>",
                ),
                row=1,
                col=2,
            )

        # 3. Drawdown
        cummax = values.cummax()
        drawdown = (values - cummax) / cummax * 100

        fig.add_trace(
            go.Scatter(
                x=dates,
                y=drawdown,
                mode="lines",
                name=name,
                line=dict(color=color, width=1.5),
                fill="tozeroy",
                fillcolor=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.1)",
                legendgroup=name,
                showlegend=False,
                hovertemplate=f"{name}<br>Date: %{{x}}<br>Drawdown: %{{y:.2f}}%<extra></extra>",
            ),
            row=2,
            col=1,
        )

        # Collect metrics for table
        metrics_data.append([
            name,
            f"{results.total_return:+.2f}%",
            f"{results.annualized_return:+.2f}%",
            f"{results.volatility:.2f}%",
            f"{results.sharpe_ratio:.2f}",
            f"{results.sortino_ratio:.2f}",
            f"{results.max_drawdown:.2f}%",
            f"{results.win_rate:.1f}%",
            str(results.total_trades),
        ])

    # 4. Add metrics table
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Strategy", "Total", "Annual", "Vol", "Sharpe", "Sortino", "Max DD", "Win %", "Trades"],
                fill_color=COLORS["primary"],
                font=dict(color="white", size=11),
                align="left",
            ),
            cells=dict(
                values=list(zip(*metrics_data)) if metrics_data else [[]],
                fill_color=[
                    [COLORS["background"] if i % 2 == 0 else "white" for i in range(len(metrics_data))]
                ],
                font=dict(size=10),
                align="left",
            ),
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=height,
        **DEFAULT_LAYOUT,
    )

    fig.update_yaxes(title_text=y_label if normalize else "Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", ticksuffix="%", row=2, col=1)

    # Add horizontal line at y=0 and y=1 for Sharpe
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_hline(y=1, line_dash="dot", line_color=COLORS["positive"], row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    return fig


def create_dashboard(
    results: "BacktestResult",
    data: pd.DataFrame | None = None,
    title: str = "Trading Dashboard",
    height: int = 1000,
) -> go.Figure:
    """
    Create a comprehensive interactive trading dashboard.

    Args:
        results: BacktestResult object.
        data: Optional OHLCV data for candlestick chart.
        title: Dashboard title.
        height: Dashboard height in pixels.

    Returns:
        Plotly Figure object.
    """
    equity_df = results.equity_curve
    if equity_df.empty:
        raise ValueError("Equity curve data is empty")

    # Create subplot layout
    has_ohlc = data is not None and not data.empty
    rows = 4 if has_ohlc else 3
    row_heights = [0.35, 0.2, 0.25, 0.2] if has_ohlc else [0.4, 0.25, 0.35]

    subplot_titles = []
    if has_ohlc:
        subplot_titles = ["Price Action", "Volume", "Portfolio Equity", "Drawdown"]
    else:
        subplot_titles = ["Portfolio Equity", "Drawdown", "Trade P&L Distribution"]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )

    current_row = 1

    # Add candlestick chart if data provided
    if has_ohlc:
        df = data.copy()
        col_mapping = {
            "Date": "date", "datetime": "date", "Datetime": "date",
            "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        }
        df = df.rename(columns=col_mapping)

        if "date" not in df.columns:
            if df.index.name in ["date", "Date", "datetime", "Datetime"]:
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"])

        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df["date"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                name="OHLC",
                increasing_line_color=COLORS["candle_up"],
                decreasing_line_color=COLORS["candle_down"],
            ),
            row=current_row,
            col=1,
        )

        # Add trade markers
        buy_dates, buy_prices, sell_dates, sell_prices = create_trade_markers(results)

        if buy_dates:
            fig.add_trace(
                go.Scatter(
                    x=buy_dates,
                    y=buy_prices,
                    mode="markers",
                    name="Buy",
                    marker=dict(color=COLORS["buy_marker"], size=10, symbol="triangle-up"),
                ),
                row=current_row,
                col=1,
            )

        if sell_dates:
            fig.add_trace(
                go.Scatter(
                    x=sell_dates,
                    y=sell_prices,
                    mode="markers",
                    name="Sell",
                    marker=dict(color=COLORS["sell_marker"], size=10, symbol="triangle-down"),
                ),
                row=current_row,
                col=1,
            )

        current_row += 1

        # Volume
        if "volume" in df.columns:
            colors = [
                COLORS["candle_up"] if c >= o else COLORS["candle_down"]
                for c, o in zip(df["close"], df["open"])
            ]
            fig.add_trace(
                go.Bar(x=df["date"], y=df["volume"], name="Volume", marker_color=colors, opacity=0.6),
                row=current_row,
                col=1,
            )
            current_row += 1

    # Equity curve
    dates = pd.to_datetime(equity_df["date"])
    values = equity_df["total_value"]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            name="Portfolio",
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy",
            fillcolor="rgba(46, 134, 171, 0.1)",
        ),
        row=current_row,
        col=1,
    )
    current_row += 1

    # Drawdown
    cummax = values.cummax()
    drawdown = (values - cummax) / cummax * 100

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=drawdown,
            mode="lines",
            name="Drawdown",
            line=dict(color=COLORS["negative"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(220, 53, 69, 0.3)",
        ),
        row=current_row,
        col=1,
    )

    # If no OHLC, add trade P&L distribution
    if not has_ohlc and results.trades:
        current_row += 1
        trade_pnls = [t.pnl for t in results.trades]
        colors = [COLORS["positive"] if pnl > 0 else COLORS["negative"] for pnl in trade_pnls]

        fig.add_trace(
            go.Bar(
                x=list(range(len(trade_pnls))),
                y=trade_pnls,
                name="Trade P&L",
                marker_color=colors,
            ),
            row=current_row,
            col=1,
        )

    # Add metrics annotation
    metrics_text = (
        f"<b>Performance Summary</b><br>"
        f"Total Return: {results.total_return:+.2f}%<br>"
        f"Annual Return: {results.annualized_return:+.2f}%<br>"
        f"Sharpe Ratio: {results.sharpe_ratio:.2f}<br>"
        f"Max Drawdown: {results.max_drawdown:.2f}%<br>"
        f"Win Rate: {results.win_rate:.1f}%<br>"
        f"Total Trades: {results.total_trades}"
    )

    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=metrics_text,
        showarrow=False,
        font=dict(size=11),
        bgcolor="white",
        bordercolor=COLORS["neutral"],
        borderwidth=1,
        borderpad=8,
        align="left",
    )

    # Update layout
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center"),
        height=height,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        **DEFAULT_LAYOUT,
    )

    return fig


__all__ = [
    "create_dashboard",
    "plot_candlestick",
    "plot_strategy_comparison",
    "create_interactive_equity_curve",
    "create_trade_markers",
]

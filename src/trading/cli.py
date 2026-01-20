#!/usr/bin/env python3
"""
Trading System Command Line Interface

A user-friendly CLI for the S&P 500 Trading System providing:
- Backtesting capabilities
- Signal generation
- Risk analysis
- Symbol listing

Usage:
    trading_cli backtest --symbols AAPL,MSFT --strategy momentum
    trading_cli signals --symbol AAPL --strategy momentum
    trading_cli risk --symbol AAPL --capital 100000
    trading_cli list-symbols
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from typing import Any

import click
import pandas as pd

# Import trading system modules
from .data.fetcher import SP500DataFetcher, CacheConfig
from .strategies.momentum import MomentumStrategy, MomentumConfig
from .strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig
from .strategies.base import SignalType
from .risk.manager import RiskManager, RiskConfig, PositionSizingMethod, StopLossType
from .backtest.engine import BacktestEngine, BacktestConfig


# Strategy mapping
STRATEGIES = {
    "momentum": (MomentumStrategy, MomentumConfig),
    "mean_reversion": (MeanReversionStrategy, MeanReversionConfig),
    "pairs": None,  # Placeholder for future implementation
    "trend": None,  # Placeholder for future implementation
}


def format_currency(value: float) -> str:
    """Format a number as currency."""
    return f"${value:,.2f}"


def format_percent(value: float) -> str:
    """Format a number as percentage."""
    return f"{value:.2f}%"


class TradingCLI:
    """Trading system CLI helper class."""

    def __init__(self) -> None:
        self.fetcher = SP500DataFetcher(
            cache_config=CacheConfig(enabled=True)
        )

    def get_strategy(self, strategy_name: str) -> tuple[Any, Any] | None:
        """Get strategy class and config class by name."""
        if strategy_name not in STRATEGIES:
            return None
        return STRATEGIES[strategy_name]

    def fetch_data(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for symbols."""
        stock_data = self.fetcher.fetch_ohlcv(symbols, start_date, end_date)
        return {symbol: sd.data for symbol, sd in stock_data.items() if sd.is_valid}


# Create CLI group
@click.group()
@click.version_option(version="1.0.0", prog_name="trading-cli")
@click.pass_context
def cli(ctx: click.Context) -> None:
    """
    S&P 500 Trading System CLI

    A comprehensive trading system for backtesting, signal generation,
    and risk analysis of S&P 500 stocks.

    \b
    Examples:
        # Run a backtest with default settings
        trading_cli backtest

        # Run backtest with specific symbols and strategy
        trading_cli backtest --symbols AAPL,MSFT,GOOGL --strategy momentum

        # Generate trading signals
        trading_cli signals --symbol AAPL --strategy momentum --days 60

        # Calculate risk metrics
        trading_cli risk --symbol AAPL --capital 100000

        # List available S&P 500 symbols
        trading_cli list-symbols
    """
    ctx.ensure_object(dict)
    ctx.obj["cli"] = TradingCLI()


@cli.command()
@click.option(
    "--symbols", "-s",
    default="AAPL,MSFT,GOOGL",
    help="Comma-separated stock symbols (default: AAPL,MSFT,GOOGL)",
    show_default=True,
)
@click.option(
    "--strategy", "-t",
    type=click.Choice(["momentum", "mean_reversion", "pairs", "trend"], case_sensitive=False),
    default="momentum",
    help="Trading strategy to use",
    show_default=True,
)
@click.option(
    "--start", "-f",
    default="2024-01-01",
    help="Start date (YYYY-MM-DD)",
    show_default=True,
)
@click.option(
    "--end", "-e",
    default="2024-12-31",
    help="End date (YYYY-MM-DD)",
    show_default=True,
)
@click.option(
    "--capital", "-c",
    type=float,
    default=100000.0,
    help="Initial capital in USD",
    show_default=True,
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json", "csv"], case_sensitive=False),
    default="text",
    help="Output format",
    show_default=True,
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.pass_context
def backtest(
    ctx: click.Context,
    symbols: str,
    strategy: str,
    start: str,
    end: str,
    capital: float,
    output: str,
    verbose: bool,
) -> None:
    """
    Run a backtest simulation.

    Performs historical backtesting using the specified strategy and symbols
    over the given date range.

    \b
    Examples:
        # Basic backtest with defaults
        trading_cli backtest

        # Backtest with custom symbols
        trading_cli backtest --symbols AAPL,MSFT,NVDA,TSLA

        # Mean reversion strategy backtest
        trading_cli backtest --strategy mean_reversion --capital 50000

        # Export results as JSON
        trading_cli backtest --output json > results.json

        # Verbose backtest with progress
        trading_cli backtest -v --symbols AAPL,GOOGL
    """
    trading_cli: TradingCLI = ctx.obj["cli"]

    # Parse symbols
    symbol_list = [s.strip().upper() for s in symbols.split(",")]

    # Validate strategy
    strategy_info = trading_cli.get_strategy(strategy)
    if strategy_info is None:
        click.secho(f"Error: Strategy '{strategy}' is not yet implemented.", fg="red")
        click.echo("Available strategies: momentum, mean_reversion")
        sys.exit(1)

    strategy_class, config_class = strategy_info

    if verbose:
        click.echo(f"Fetching data for {len(symbol_list)} symbols...")
        click.echo(f"  Symbols: {', '.join(symbol_list)}")
        click.echo(f"  Date range: {start} to {end}")

    # Fetch data
    try:
        data = trading_cli.fetch_data(symbol_list, start, end)
    except Exception as e:
        click.secho(f"Error fetching data: {e}", fg="red")
        sys.exit(1)

    if not data:
        click.secho("Error: No data available for the specified symbols and date range.", fg="red")
        sys.exit(1)

    if verbose:
        click.echo(f"  Fetched data for {len(data)} symbols")

    # Configure and run backtest
    backtest_config = BacktestConfig(
        initial_capital=capital,
        commission_rate=0.001,
        slippage_rate=0.0005,
    )

    engine = BacktestEngine(config=backtest_config)
    engine.add_strategy(strategy_class(config_class()))

    if verbose:
        click.echo(f"\nRunning backtest with {strategy} strategy...")

    # Run backtest with progress callback
    def progress_callback(current: int, total: int) -> None:
        if verbose and current % 50 == 0:
            pct = (current / total) * 100
            click.echo(f"  Progress: {pct:.1f}% ({current}/{total} days)")

    try:
        result = engine.run(
            data=data,
            start_date=start,
            end_date=end,
            progress_callback=progress_callback if verbose else None,
        )
    except Exception as e:
        click.secho(f"Error running backtest: {e}", fg="red")
        sys.exit(1)

    # Output results
    if output == "json":
        result_dict = result.to_dict()
        result_dict["symbols"] = symbol_list
        result_dict["strategy"] = strategy
        click.echo(json.dumps(result_dict, indent=2, default=str))

    elif output == "csv":
        # Output equity curve as CSV
        if not result.equity_curve.empty:
            click.echo(result.equity_curve.to_csv(index=False))
        else:
            click.echo("date,total_value,cash,positions_value")

    else:  # text
        click.echo(result.generate_report())
        click.echo(f"\nSymbols: {', '.join(symbol_list)}")
        click.echo(f"Strategy: {strategy}")


@cli.command()
@click.option(
    "--symbol", "-s",
    required=True,
    help="Stock symbol to analyze",
)
@click.option(
    "--strategy", "-t",
    type=click.Choice(["momentum", "mean_reversion", "pairs", "trend"], case_sensitive=False),
    default="momentum",
    help="Trading strategy to use",
    show_default=True,
)
@click.option(
    "--days", "-d",
    type=int,
    default=60,
    help="Lookback days for analysis",
    show_default=True,
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format",
    show_default=True,
)
@click.pass_context
def signals(
    ctx: click.Context,
    symbol: str,
    strategy: str,
    days: int,
    output: str,
) -> None:
    """
    Generate trading signals for a symbol.

    Analyzes the specified stock using the chosen strategy and
    generates buy/sell/hold signals with strength indicators.

    \b
    Examples:
        # Generate momentum signals for AAPL
        trading_cli signals --symbol AAPL --strategy momentum

        # Mean reversion signals with 90-day lookback
        trading_cli signals --symbol MSFT --strategy mean_reversion --days 90

        # Output as JSON
        trading_cli signals --symbol GOOGL --output json
    """
    trading_cli: TradingCLI = ctx.obj["cli"]

    symbol = symbol.upper()

    # Validate strategy
    strategy_info = trading_cli.get_strategy(strategy)
    if strategy_info is None:
        click.secho(f"Error: Strategy '{strategy}' is not yet implemented.", fg="red")
        sys.exit(1)

    strategy_class, config_class = strategy_info

    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days + 50)  # Extra days for warmup

    # Fetch data
    try:
        data = trading_cli.fetch_data(
            [symbol],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        click.secho(f"Error fetching data: {e}", fg="red")
        sys.exit(1)

    if symbol not in data or data[symbol].empty:
        click.secho(f"Error: No data available for {symbol}", fg="red")
        sys.exit(1)

    df = data[symbol].copy()
    df["symbol"] = symbol

    # Create strategy and generate signals
    strat = strategy_class(config_class())
    signals_list = strat.generate_signals(df)

    # Get current indicators
    if strategy == "momentum":
        indicators = strat.get_current_momentum(df)
    elif strategy == "mean_reversion":
        indicators = strat.get_reversion_metrics(df)
    else:
        indicators = {}

    # Determine current signal
    if signals_list:
        latest_signal = signals_list[-1]
        signal_type = latest_signal.signal_type.value
        signal_strength = latest_signal.strength
        signal_price = latest_signal.price
    else:
        signal_type = "HOLD"
        signal_strength = 0.0
        signal_price = float(df["close"].iloc[-1]) if "close" in df.columns else 0.0

    # Output results
    if output == "json":
        result = {
            "symbol": symbol,
            "strategy": strategy,
            "signal": signal_type,
            "strength": signal_strength,
            "price": signal_price,
            "timestamp": datetime.now().isoformat(),
            "indicators": indicators,
            "lookback_days": days,
        }
        click.echo(json.dumps(result, indent=2, default=str))

    else:  # text
        click.echo("=" * 50)
        click.echo(f"TRADING SIGNALS: {symbol}")
        click.echo("=" * 50)
        click.echo(f"Strategy: {strategy}")
        click.echo(f"Lookback: {days} days")
        click.echo(f"Current Price: {format_currency(signal_price)}")
        click.echo()

        # Signal with color
        if signal_type == "BUY":
            click.secho(f"Signal: {signal_type}", fg="green", bold=True)
        elif signal_type == "SELL":
            click.secho(f"Signal: {signal_type}", fg="red", bold=True)
        else:
            click.secho(f"Signal: {signal_type}", fg="yellow")

        click.echo(f"Strength: {signal_strength:.2f}")
        click.echo()

        # Display indicators
        click.echo("Indicators:")
        click.echo("-" * 30)
        for key, value in indicators.items():
            if isinstance(value, float):
                click.echo(f"  {key}: {value:.4f}")
            else:
                click.echo(f"  {key}: {value}")
        click.echo("=" * 50)


@cli.command()
@click.option(
    "--symbol", "-s",
    required=True,
    help="Stock symbol to analyze",
)
@click.option(
    "--capital", "-c",
    type=float,
    default=100000.0,
    help="Portfolio capital in USD",
    show_default=True,
)
@click.option(
    "--volatility", "-v",
    is_flag=True,
    help="Use historical volatility for calculations",
)
@click.option(
    "--position-size", "-p",
    type=float,
    default=None,
    help="Position size (shares). If not specified, calculates optimal size.",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format",
    show_default=True,
)
@click.pass_context
def risk(
    ctx: click.Context,
    symbol: str,
    capital: float,
    volatility: bool,
    position_size: float | None,
    output: str,
) -> None:
    """
    Calculate risk metrics for a position.

    Analyzes risk metrics including position sizing, stop loss levels,
    and risk-adjusted returns for the specified symbol.

    \b
    Examples:
        # Basic risk analysis
        trading_cli risk --symbol AAPL --capital 100000

        # Risk analysis with historical volatility
        trading_cli risk --symbol MSFT --capital 50000 --volatility

        # Analyze risk for a specific position size
        trading_cli risk --symbol GOOGL --capital 100000 --position-size 50

        # Output as JSON
        trading_cli risk --symbol NVDA --output json
    """
    trading_cli: TradingCLI = ctx.obj["cli"]

    symbol = symbol.upper()

    # Fetch recent data for volatility calculation
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)

    try:
        data = trading_cli.fetch_data(
            [symbol],
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        )
    except Exception as e:
        click.secho(f"Error fetching data: {e}", fg="red")
        sys.exit(1)

    if symbol not in data or data[symbol].empty:
        click.secho(f"Error: No data available for {symbol}", fg="red")
        sys.exit(1)

    df = data[symbol]

    # Get current price and calculate metrics
    close_col = "close" if "close" in df.columns else "Close"
    high_col = "high" if "high" in df.columns else "High"
    low_col = "low" if "low" in df.columns else "Low"

    current_price = float(df[close_col].iloc[-1])

    # Calculate volatility
    returns = df[close_col].pct_change().dropna()
    daily_vol = float(returns.std())
    annual_vol = daily_vol * (252 ** 0.5)

    # Calculate ATR
    high = df[high_col]
    low = df[low_col]
    close = df[close_col]
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1)),
    ], axis=1).max(axis=1)
    atr = float(tr.tail(14).mean())

    # Create risk manager
    risk_config = RiskConfig(
        position_sizing_method=PositionSizingMethod.VOLATILITY if volatility else PositionSizingMethod.PERCENT_RISK,
        stop_loss_type=StopLossType.ATR_BASED,
        max_position_size_pct=0.10,
        stop_loss_pct=0.02,
        atr_multiplier=2.0,
    )
    risk_manager = RiskManager(risk_config)

    # Calculate recommended position size
    if position_size is None:
        calculated_size = risk_manager.calculate_position_size(
            signal_strength=0.7,  # Assume moderately strong signal
            current_price=current_price,
            capital=capital,
            volatility=daily_vol if volatility else None,
            atr=atr,
        )
        position_size = abs(calculated_size)

    # Calculate stop loss
    stop_loss = risk_manager.calculate_stop_loss(
        entry_price=current_price,
        is_long=True,
        atr=atr,
        volatility=daily_vol,
    )

    # Calculate take profit
    take_profit = risk_manager.calculate_take_profit(
        entry_price=current_price,
        stop_loss=stop_loss,
        is_long=True,
        reward_risk_ratio=2.0,
    )

    # Calculate risk metrics
    position_value = position_size * current_price
    position_pct = (position_value / capital) * 100
    risk_per_share = current_price - stop_loss
    total_risk = position_size * risk_per_share
    risk_pct = (total_risk / capital) * 100
    reward = take_profit - current_price if take_profit else 0
    reward_risk_ratio = reward / risk_per_share if risk_per_share > 0 else 0

    # Output results
    if output == "json":
        result = {
            "symbol": symbol,
            "current_price": current_price,
            "capital": capital,
            "position_size": position_size,
            "position_value": position_value,
            "position_pct": position_pct,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "risk_per_share": risk_per_share,
            "total_risk": total_risk,
            "risk_pct": risk_pct,
            "reward_risk_ratio": reward_risk_ratio,
            "daily_volatility": daily_vol,
            "annual_volatility": annual_vol,
            "atr": atr,
            "atr_pct": (atr / current_price) * 100,
        }
        click.echo(json.dumps(result, indent=2))

    else:  # text
        click.echo("=" * 50)
        click.echo(f"RISK ANALYSIS: {symbol}")
        click.echo("=" * 50)
        click.echo()

        click.echo("PRICE & VOLATILITY")
        click.echo("-" * 30)
        click.echo(f"Current Price:      {format_currency(current_price)}")
        click.echo(f"Daily Volatility:   {format_percent(daily_vol * 100)}")
        click.echo(f"Annual Volatility:  {format_percent(annual_vol * 100)}")
        click.echo(f"ATR (14-day):       {format_currency(atr)}")
        click.echo(f"ATR %:              {format_percent((atr / current_price) * 100)}")
        click.echo()

        click.echo("POSITION SIZING")
        click.echo("-" * 30)
        click.echo(f"Portfolio Capital:  {format_currency(capital)}")
        click.echo(f"Position Size:      {position_size:.2f} shares")
        click.echo(f"Position Value:     {format_currency(position_value)}")
        click.echo(f"Position %:         {format_percent(position_pct)}")
        click.echo()

        click.echo("RISK LEVELS")
        click.echo("-" * 30)
        click.echo(f"Stop Loss:          {format_currency(stop_loss)}")
        click.echo(f"Take Profit:        {format_currency(take_profit or 0)}")
        click.echo(f"Risk per Share:     {format_currency(risk_per_share)}")
        click.echo(f"Total Risk:         {format_currency(total_risk)}")
        click.echo(f"Portfolio Risk %:   {format_percent(risk_pct)}")
        click.echo(f"Reward/Risk Ratio:  {reward_risk_ratio:.2f}")
        click.echo("=" * 50)


@cli.command("list-symbols")
@click.option(
    "--top", "-n",
    type=int,
    default=None,
    help="Show only top N stocks by market cap",
)
@click.option(
    "--output", "-o",
    type=click.Choice(["text", "json"], case_sensitive=False),
    default="text",
    help="Output format",
    show_default=True,
)
@click.pass_context
def list_symbols(
    ctx: click.Context,
    top: int | None,
    output: str,
) -> None:
    """
    List available S&P 500 symbols.

    Displays the list of S&P 500 constituent symbols available
    for trading and analysis.

    \b
    Examples:
        # List all available symbols
        trading_cli list-symbols

        # List top 20 symbols by market cap
        trading_cli list-symbols --top 20

        # Output as JSON
        trading_cli list-symbols --output json
    """
    trading_cli: TradingCLI = ctx.obj["cli"]

    symbols = trading_cli.fetcher.get_sp500_symbols()

    if top:
        click.echo(f"Fetching market caps for top {top} stocks (this may take a moment)...")
        try:
            symbols = trading_cli.fetcher.get_top_stocks_by_market_cap(top, symbols)
        except Exception as e:
            click.secho(f"Warning: Could not fetch market caps: {e}", fg="yellow")
            symbols = symbols[:top]

    if output == "json":
        result = {
            "symbols": symbols,
            "count": len(symbols),
        }
        click.echo(json.dumps(result, indent=2))

    else:  # text
        click.echo("=" * 50)
        click.echo("AVAILABLE S&P 500 SYMBOLS")
        click.echo("=" * 50)
        click.echo(f"Total: {len(symbols)} symbols")
        click.echo()

        # Display in columns
        cols = 5
        for i in range(0, len(symbols), cols):
            row = symbols[i:i + cols]
            click.echo("  " + "  ".join(f"{s:<6}" for s in row))

        click.echo()
        click.echo("=" * 50)


def main() -> None:
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()

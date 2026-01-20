#!/usr/bin/env python3
"""
S&P 500 Momentum Strategy Backtest Script

This script fetches live market data for the top 5 S&P 500 stocks and runs
a backtest using the MomentumStrategy for the year 2024.

Usage:
    python scripts/run_backtest.py

Requirements:
    - yfinance
    - pandas
    - numpy

Output:
    - Performance metrics including total return, Sharpe ratio, and max drawdown
    - Detailed trade statistics
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data.fetcher import SP500DataFetcher, CacheConfig
from src.trading.strategies.momentum import MomentumStrategy, MomentumConfig
from src.trading.backtest.engine import BacktestEngine, BacktestConfig
from src.trading.risk.manager import RiskConfig, PositionSizingMethod, StopLossType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def main() -> int:
    """
    Main entry point for the backtest script.

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    print("=" * 70)
    print("S&P 500 MOMENTUM STRATEGY BACKTEST")
    print("=" * 70)
    print()

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    initial_capital = 100000.0

    print(f"Symbols:         {', '.join(symbols)}")
    print(f"Period:          {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print()

    # Step 1: Fetch live market data
    print("-" * 70)
    print("STEP 1: Fetching Live Market Data")
    print("-" * 70)

    cache_config = CacheConfig(
        enabled=True,
        directory=project_root / "cache",
        expiry_hours=1  # Fresh data for backtest
    )

    fetcher = SP500DataFetcher(cache_config=cache_config)

    try:
        logger.info(f"Fetching OHLCV data for {len(symbols)} symbols...")
        stock_data = fetcher.fetch_ohlcv(
            symbols=symbols,
            start=start_date,
            end=end_date,
            interval="1d"
        )

        if not stock_data:
            logger.error("No data fetched. Check your internet connection and symbol validity.")
            return 1

        # Convert StockData to DataFrames for backtest engine
        data_dict: dict[str, any] = {}
        for symbol, sd in stock_data.items():
            if sd.is_valid:
                df = sd.data.copy()
                # Ensure datetime column exists
                if "datetime" not in df.columns and "date" not in df.columns:
                    df = df.reset_index()
                    if df.columns[0] in ["Date", "date", "Datetime", "datetime"]:
                        df = df.rename(columns={df.columns[0]: "datetime"})
                data_dict[symbol] = df
                print(f"  {symbol}: {len(df)} trading days loaded")
            else:
                logger.warning(f"  {symbol}: No valid data")

        print(f"\nSuccessfully loaded data for {len(data_dict)} symbols")

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return 1

    # Step 2: Configure and run backtest
    print()
    print("-" * 70)
    print("STEP 2: Running Backtest with MomentumStrategy")
    print("-" * 70)

    # Configure momentum strategy
    # Using more sensitive parameters to generate signals in bull markets
    momentum_config = MomentumConfig(
        name="MomentumStrategy",
        rsi_period=14,
        rsi_overbought=65.0,        # Lower threshold for overbought (was 70)
        rsi_oversold=35.0,          # Higher threshold for oversold (was 30)
        fast_ma_period=5,           # Faster MA for quicker signals (was 10)
        slow_ma_period=20,          # Shorter slow MA (was 30)
        ma_type="ema",
        signal_threshold=0.15,      # Lower threshold to trigger more signals (was 0.3)
        use_volume_confirmation=False,  # Disable volume filter for more signals
        volume_ma_period=20,
        lookback_period=20,         # Reduced lookback (was 30)
        min_data_points=30,         # Reduced min data points (was 50)
        risk_per_trade=0.02
    )

    strategy = MomentumStrategy(config=momentum_config)

    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.001,  # 0.1% commission
        slippage_rate=0.0005,   # 0.05% slippage
        margin_rate=0.5,
        enable_shorting=False,  # Long only
        enable_fractional=True,
        data_frequency="1d",
        rebalance_frequency=1,
        warmup_period=30,       # Reduced warmup period (was 50)
        max_positions=5,        # One position per symbol max
        stop_on_max_drawdown=False,
        max_drawdown_threshold=0.25
    )

    # Configure risk management
    risk_config = RiskConfig(
        position_sizing_method=PositionSizingMethod.PERCENT_RISK,
        max_position_size_pct=0.20,      # Max 20% per position
        max_portfolio_exposure=1.0,       # Full exposure allowed
        max_sector_exposure=0.40,         # Max 40% per sector
        stop_loss_type=StopLossType.ATR_BASED,
        stop_loss_pct=0.02,
        atr_multiplier=2.0,
        trailing_stop_pct=0.05,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.15
    )

    # Create and run backtest
    engine = BacktestEngine(config=backtest_config, risk_config=risk_config)
    engine.add_strategy(strategy)

    print("\nStrategy Configuration:")
    print(f"  RSI Period:        {momentum_config.rsi_period}")
    print(f"  RSI Overbought:    {momentum_config.rsi_overbought}")
    print(f"  RSI Oversold:      {momentum_config.rsi_oversold}")
    print(f"  Fast MA Period:    {momentum_config.fast_ma_period}")
    print(f"  Slow MA Period:    {momentum_config.slow_ma_period}")
    print(f"  MA Type:           {momentum_config.ma_type.upper()}")
    print(f"  Signal Threshold:  {momentum_config.signal_threshold}")
    print()

    print("Backtest Configuration:")
    print(f"  Commission Rate:   {backtest_config.commission_rate:.2%}")
    print(f"  Slippage Rate:     {backtest_config.slippage_rate:.2%}")
    print(f"  Max Positions:     {backtest_config.max_positions}")
    print(f"  Warmup Period:     {backtest_config.warmup_period} days")
    print()

    try:
        logger.info("Running backtest...")

        # Normalize datetime columns to remove timezone info for consistent comparison
        import pandas as pd
        for symbol in data_dict:
            df = data_dict[symbol]
            datetime_col = "datetime" if "datetime" in df.columns else "date"
            if datetime_col in df.columns:
                df[datetime_col] = pd.to_datetime(df[datetime_col]).dt.tz_localize(None)
                data_dict[symbol] = df

        # Progress callback
        last_pct = 0
        def progress_callback(current: int, total: int) -> None:
            nonlocal last_pct
            pct = int(current / total * 100)
            if pct >= last_pct + 10:
                print(f"  Progress: {pct}% ({current}/{total} days)")
                last_pct = pct

        result = engine.run(
            data=data_dict,
            start_date=start_date,
            end_date=end_date,
            progress_callback=progress_callback
        )

        print("  Progress: 100% - Complete!")

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Display results
    print()
    print("-" * 70)
    print("STEP 3: Performance Results")
    print("-" * 70)

    # Print the formatted report
    print(result.generate_report())

    # Additional analysis
    print()
    print("ADDITIONAL ANALYSIS")
    print("-" * 40)

    # Profit/Loss breakdown
    total_pnl = result.final_value - result.initial_capital
    print(f"Net Profit/Loss:     ${total_pnl:,.2f}")

    # Risk-adjusted metrics interpretation
    print()
    print("Risk-Adjusted Performance Interpretation:")

    if result.sharpe_ratio > 1.0:
        sharpe_rating = "Good"
    elif result.sharpe_ratio > 0.5:
        sharpe_rating = "Acceptable"
    elif result.sharpe_ratio > 0:
        sharpe_rating = "Below Average"
    else:
        sharpe_rating = "Poor"
    print(f"  Sharpe Ratio:      {result.sharpe_ratio:.2f} ({sharpe_rating})")

    if result.sortino_ratio > 1.5:
        sortino_rating = "Good"
    elif result.sortino_ratio > 1.0:
        sortino_rating = "Acceptable"
    elif result.sortino_ratio > 0:
        sortino_rating = "Below Average"
    else:
        sortino_rating = "Poor"
    print(f"  Sortino Ratio:     {result.sortino_ratio:.2f} ({sortino_rating})")

    if result.max_drawdown < 10:
        dd_rating = "Low Risk"
    elif result.max_drawdown < 20:
        dd_rating = "Moderate Risk"
    elif result.max_drawdown < 30:
        dd_rating = "High Risk"
    else:
        dd_rating = "Very High Risk"
    print(f"  Max Drawdown:      {result.max_drawdown:.2f}% ({dd_rating})")

    # Trade analysis
    if result.total_trades > 0:
        print()
        print("Trade Analysis:")
        print(f"  Total Trades:      {result.total_trades}")
        print(f"  Win Rate:          {result.win_rate:.1f}%")
        print(f"  Profit Factor:     {result.profit_factor:.2f}")

        if result.trades:
            # Analyze by symbol
            trades_by_symbol: dict[str, list] = {}
            for trade in result.trades:
                if trade.symbol not in trades_by_symbol:
                    trades_by_symbol[trade.symbol] = []
                trades_by_symbol[trade.symbol].append(trade)

            print()
            print("Performance by Symbol:")
            for symbol in sorted(trades_by_symbol.keys()):
                trades = trades_by_symbol[symbol]
                symbol_pnl = sum(t.pnl for t in trades)
                symbol_wins = sum(1 for t in trades if t.pnl > 0)
                symbol_win_rate = symbol_wins / len(trades) * 100 if trades else 0
                print(f"  {symbol}: {len(trades)} trades, ${symbol_pnl:,.2f} P&L, {symbol_win_rate:.1f}% win rate")

    # Show open positions at end of backtest
    if result.positions_history and result.positions_history[-1]["positions"]:
        print()
        print("Open Positions at End of Backtest:")
        final_positions = result.positions_history[-1]["positions"]
        for symbol, pos_data in final_positions.items():
            unrealized = pos_data.get("unrealized_pnl", 0)
            market_val = pos_data.get("market_value", 0)
            qty = pos_data.get("quantity", 0)
            print(f"  {symbol}: {qty:.2f} shares, ${market_val:,.2f} value, ${unrealized:,.2f} unrealized P&L")
    else:
        print()
        print("No trades executed during the backtest period.")
        print("This could indicate:")
        print("  - Signal threshold is too high")
        print("  - Market conditions didn't trigger signals")
        print("  - Warmup period consumed most of the data")

    print()
    print("=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Multi-Regime Strategy Backtest

Tests the specialized multi-regime trading systems where each market condition
has its own dedicated trading system:
- Bull Market System: Aggressive momentum, high exposure
- Bear Market System: Defensive, capital preservation
- Sideways Market System: Mean reversion, range trading
- High Volatility System: Volatility targeting, reduced exposure

The Regime Orchestrator automatically switches between systems based on
detected market conditions.
"""

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.strategies import (
    RegimeType,
    RegimeDetectorConfig,
    RegimeOrchestrator,
    OrchestratorConfig,
    BullSystemConfig,
    BearSystemConfig,
    SidewaysSystemConfig,
    HighVolSystemConfig,
    fetch_vix_data,
)


# =============================================================================
# Configuration
# =============================================================================

# Date range for backtest - 20 years
START_DATE = "2004-12-01"  # After GOOGL IPO + warmup
END_DATE = "2024-12-31"

# Symbols to trade
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN"]

# Market proxy for regime detection
MARKET_SYMBOL = "SPY"

# Initial capital
INITIAL_CAPITAL = 100000

# Rebalance frequency (trading days)
REBALANCE_DAYS = 5

# Transaction costs
COMMISSION_RATE = 0.001  # 0.1% per trade


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_data(
    symbols: List[str],
    start_date: str,
    end_date: str,
) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for symbols."""
    data = {}

    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
            )

            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if len(df) > 0:
                data[symbol] = df
                print(f"    {len(df)} days of data")
            else:
                print(f"    WARNING: No data for {symbol}")

        except Exception as e:
            print(f"    ERROR fetching {symbol}: {e}")

    return data


# =============================================================================
# Backtest Engine
# =============================================================================

def run_multi_regime_backtest(
    data: Dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    symbols: List[str],
    initial_capital: float,
    start_date: str,
    end_date: str,
    vix_data: Optional[pd.DataFrame] = None,
    orchestrator_config: Optional[OrchestratorConfig] = None,
    regime_config: Optional[RegimeDetectorConfig] = None,
    bull_config: Optional[BullSystemConfig] = None,
    bear_config: Optional[BearSystemConfig] = None,
    sideways_config: Optional[SidewaysSystemConfig] = None,
    high_vol_config: Optional[HighVolSystemConfig] = None,
) -> Dict[str, Any]:
    """
    Run backtest with the multi-regime orchestrator.
    """
    # Initialize orchestrator
    orchestrator = RegimeOrchestrator(
        symbols=symbols,
        config=orchestrator_config,
        regime_config=regime_config,
        bull_config=bull_config,
        bear_config=bear_config,
        sideways_config=sideways_config,
        high_vol_config=high_vol_config,
    )

    # Get common trading dates
    all_dates = None
    for symbol, df in data.items():
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    # Filter to date range
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    if len(trading_dates) == 0:
        return {"error": "No trading dates in range"}

    # Find first valid date with enough warmup
    warmup_days = 100
    first_valid_idx = warmup_days
    if first_valid_idx >= len(trading_dates):
        return {"error": f"Insufficient data. Need {warmup_days} warmup days."}

    # Initialize portfolio
    cash = initial_capital
    positions: Dict[str, float] = {s: 0.0 for s in symbols}
    portfolio_values: List[Tuple[datetime, float]] = []
    regime_log: List[Tuple[datetime, str, float, float]] = []
    trades: List[Dict[str, Any]] = []

    # Track by regime
    regime_returns: Dict[str, List[float]] = {r.value: [] for r in RegimeType}
    current_regime_start_value: float = initial_capital

    prev_regime: Optional[RegimeType] = None
    prev_portfolio_value: float = initial_capital

    print(f"\nRunning multi-regime backtest from {trading_dates[first_valid_idx].date()} to {trading_dates[-1].date()}")
    print(f"Total trading days: {len(trading_dates) - first_valid_idx}")

    # Main backtest loop
    for i, date in enumerate(trading_dates[first_valid_idx:], start=first_valid_idx):
        # Get current prices
        prices: Dict[str, float] = {}
        for symbol in symbols:
            if symbol in data and date in data[symbol].index:
                prices[symbol] = float(data[symbol].loc[date, "Close"])

        # Calculate portfolio value
        portfolio_value = cash
        for symbol, shares in positions.items():
            if symbol in prices and shares > 0:
                portfolio_value += shares * prices[symbol]

        portfolio_values.append((date, portfolio_value))

        # Get market data for regime detection (use SPY or first symbol)
        if date in market_data.index:
            market_slice = market_data.loc[:date].tail(100)
        else:
            continue

        # Get VIX value if available
        vix_value = None
        if vix_data is not None and date in vix_data.index:
            vix_value = float(vix_data.loc[date, "Close"])

        # Prepare data slices for signal generation
        data_slices: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            if symbol in data:
                data_slices[symbol] = data[symbol].loc[:date].tail(100)

        # Generate signals from orchestrator
        signals, regime, confidence, exposure = orchestrator.generate_signals(
            data=data_slices,
            market_data=market_slice,
            portfolio_value=portfolio_value,
            current_date=date,
            vix_value=vix_value,
        )

        # Track regime changes
        if prev_regime is not None and regime != prev_regime:
            # Record return for previous regime
            regime_return = (portfolio_value - current_regime_start_value) / current_regime_start_value
            regime_returns[prev_regime.value].append(regime_return)
            current_regime_start_value = portfolio_value

        prev_regime = regime

        # Log regime
        regime_log.append((date, regime.value, confidence, exposure))

        # Rebalance on schedule
        day_idx = i - first_valid_idx
        if day_idx % REBALANCE_DAYS == 0:
            # Calculate target positions
            target_values: Dict[str, float] = {}
            for symbol, weight in signals.items():
                target_values[symbol] = portfolio_value * weight

            # Execute trades
            for symbol in symbols:
                if symbol not in prices:
                    continue

                price = prices[symbol]
                current_shares = positions.get(symbol, 0)
                current_value = current_shares * price
                target_value = target_values.get(symbol, 0)

                value_diff = target_value - current_value

                if abs(value_diff) > 100:  # Minimum trade threshold
                    shares_to_trade = value_diff / price

                    if shares_to_trade > 0:
                        # Buy
                        cost = shares_to_trade * price * (1 + COMMISSION_RATE)
                        if cost <= cash:
                            cash -= cost
                            positions[symbol] = current_shares + shares_to_trade
                            trades.append({
                                "date": date,
                                "symbol": symbol,
                                "action": "BUY",
                                "shares": shares_to_trade,
                                "price": price,
                                "regime": regime.value,
                            })
                    else:
                        # Sell
                        shares_to_sell = min(abs(shares_to_trade), current_shares)
                        if shares_to_sell > 0:
                            proceeds = shares_to_sell * price * (1 - COMMISSION_RATE)
                            cash += proceeds
                            positions[symbol] = current_shares - shares_to_sell
                            trades.append({
                                "date": date,
                                "symbol": symbol,
                                "action": "SELL",
                                "shares": shares_to_sell,
                                "price": price,
                                "regime": regime.value,
                            })

                    # Update orchestrator's system position tracking
                    orchestrator.systems[regime].update_position_tracking(
                        symbol, positions[symbol], price
                    )

        prev_portfolio_value = portfolio_value

    # Final regime return
    if prev_regime is not None:
        final_value = portfolio_values[-1][1]
        regime_return = (final_value - current_regime_start_value) / current_regime_start_value
        regime_returns[prev_regime.value].append(regime_return)

    # Calculate metrics
    values = [v[1] for v in portfolio_values]
    dates = [v[0] for v in portfolio_values]

    total_return = (values[-1] - initial_capital) / initial_capital
    daily_returns = pd.Series(values).pct_change().dropna()

    # Annualized metrics
    years = len(values) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

    # Max drawdown
    peak = pd.Series(values).expanding().max()
    drawdown = (pd.Series(values) - peak) / peak
    max_drawdown = drawdown.min()

    # Regime statistics
    regime_stats = {}
    for regime_name, returns in regime_returns.items():
        if returns:
            regime_stats[regime_name] = {
                "periods": len(returns),
                "total_return": sum(returns),
                "avg_return": np.mean(returns),
                "min_return": min(returns),
                "max_return": max(returns),
            }

    # Regime time allocation
    regime_counts = {}
    for _, r, _, _ in regime_log:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    total_days = len(regime_log)
    regime_allocation = {r: c / total_days for r, c in regime_counts.items()}

    # Trade statistics
    buy_trades = [t for t in trades if t["action"] == "BUY"]
    sell_trades = [t for t in trades if t["action"] == "SELL"]

    return {
        "initial_capital": initial_capital,
        "final_value": values[-1],
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "annualized_return": annualized_return,
        "annualized_return_pct": annualized_return * 100,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
        "years": years,
        "total_trading_days": len(values),
        "total_trades": len(trades),
        "buy_trades": len(buy_trades),
        "sell_trades": len(sell_trades),
        "regime_stats": regime_stats,
        "regime_allocation": regime_allocation,
        "portfolio_values": [(str(d.date()), v) for d, v in portfolio_values[::5]],  # Sample every 5 days
        "orchestrator_stats": orchestrator.get_regime_statistics(),
    }


def run_buy_and_hold_benchmark(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    initial_capital: float,
    start_date: str,
    end_date: str,
) -> Dict[str, Any]:
    """Run buy and hold benchmark."""
    # Get common dates
    all_dates = None
    for symbol, df in data.items():
        dates = set(df.index)
        if all_dates is None:
            all_dates = dates
        else:
            all_dates = all_dates.intersection(dates)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup_days = 100
    if warmup_days >= len(trading_dates):
        return {"error": "Insufficient data"}

    first_date = trading_dates[warmup_days]

    # Equal weight allocation
    allocation_per_symbol = initial_capital / len(symbols)
    positions: Dict[str, float] = {}

    for symbol in symbols:
        if symbol in data and first_date in data[symbol].index:
            price = float(data[symbol].loc[first_date, "Close"])
            shares = allocation_per_symbol / price
            positions[symbol] = shares

    # Track portfolio value
    portfolio_values = []
    for date in trading_dates[warmup_days:]:
        value = 0
        for symbol, shares in positions.items():
            if symbol in data and date in data[symbol].index:
                price = float(data[symbol].loc[date, "Close"])
                value += shares * price
        portfolio_values.append((date, value))

    values = [v[1] for v in portfolio_values]
    total_return = (values[-1] - initial_capital) / initial_capital

    years = len(values) / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1
    daily_returns = pd.Series(values).pct_change().dropna()
    annualized_vol = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

    peak = pd.Series(values).expanding().max()
    drawdown = (pd.Series(values) - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "initial_capital": initial_capital,
        "final_value": values[-1],
        "total_return": total_return,
        "total_return_pct": total_return * 100,
        "annualized_return": annualized_return,
        "annualized_return_pct": annualized_return * 100,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown * 100,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    """Run the multi-regime backtest."""
    print("=" * 70)
    print("MULTI-REGIME STRATEGY BACKTEST")
    print("Specialized Systems for Each Market Condition")
    print("=" * 70)

    # Fetch data
    print("\n1. Fetching historical data...")
    all_symbols = SYMBOLS + [MARKET_SYMBOL]
    data = fetch_data(all_symbols, START_DATE, END_DATE)

    if MARKET_SYMBOL not in data:
        print(f"ERROR: Could not fetch market data ({MARKET_SYMBOL})")
        return

    market_data = data.pop(MARKET_SYMBOL)

    # Fetch VIX data
    print("\n2. Fetching VIX data...")
    vix_data = fetch_vix_data(START_DATE, END_DATE)
    if vix_data is not None:
        print(f"  VIX data: {len(vix_data)} days")
    else:
        print("  WARNING: No VIX data available")

    # Configure systems
    print("\n3. Configuring specialized systems...")

    # BALANCED configurations - maximize upside while protecting downside
    # Goal: Better risk-adjusted returns than buy-and-hold

    bull_config = BullSystemConfig(
        max_exposure=1.0,     # Full investment in confirmed bull
        min_exposure=0.85,    # Stay heavily invested
        momentum_lookback=20,
        buy_dip_threshold=-0.03,
        profit_target=0.40,   # Let winners run long
        stop_loss=0.15,       # Wider stops
        concentration_limit=0.40,
        pyramid_threshold=0.05,
    )
    print("  Bull System: 85-100% exposure, let winners run")

    bear_config = BearSystemConfig(
        max_exposure=0.10,  # Max 10% invested in bear - CAPITAL PRESERVATION
        min_exposure=0.0,
        default_cash_allocation=0.95,  # 95% cash in bear
        stop_loss=0.015,    # 1.5% stop - very tight
        trailing_stop=0.02,
        profit_target=0.02,
        require_oversold=True,
        oversold_rsi=20.0,
        min_bounce_confirmation=3,
        max_position_size=0.03,
        daily_loss_limit=0.005,
    )
    print("  Bear System: Max 10% exposure, 95% cash - CAPITAL PRESERVATION")

    sideways_config = SidewaysSystemConfig(
        max_exposure=0.90,   # High exposure since sideways is most common
        min_exposure=0.70,   # Stay heavily invested
        bollinger_period=20,
        bollinger_std=2.0,
        rsi_oversold=30.0,
        rsi_overbought=70.0,
        profit_target=0.08,
        stop_loss=0.05,
        max_position_size=0.30,
    )
    print("  Sideways System: 70-90% exposure, active in range")

    high_vol_config = HighVolSystemConfig(
        target_volatility=0.20,
        max_exposure=0.60,
        min_exposure=0.30,
        stop_loss=0.12,
        trailing_stop=0.15,
        profit_target=0.10,
        partial_profit_threshold=0.06,
        max_position_size=0.20,
        vol_adjusted_sizing=True,
    )
    print("  High Vol System: 30-60% exposure, volatility targeting")

    orchestrator_config = OrchestratorConfig(
        transition_buffer_days=3,  # Require confirmation to avoid whipsaws
        require_confirmation=True,
        enable_vix_override=True,
        vix_crisis_threshold=28.0,  # React early to VIX spikes
        portfolio_stop_loss=0.20,
        enable_emergency_exit=True,
    )
    print("  Orchestrator: 3-day confirmation, VIX override at 28")

    # Regime detection balanced for trend and protection
    regime_config = RegimeDetectorConfig(
        volatility_lookback=20,
        volatility_high_percentile=80.0,
        strong_trend_threshold=0.5,
        adx_trend_threshold=30.0,
        sma_short=20,
        sma_medium=50,
        sma_long=150,
        adx_period=14,
        min_hold_days=5,
        smoothing_window=3,
    )

    # Run multi-regime backtest
    print("\n4. Running multi-regime backtest...")
    multi_regime_result = run_multi_regime_backtest(
        data=data,
        market_data=market_data,
        symbols=SYMBOLS,
        initial_capital=INITIAL_CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE,
        vix_data=vix_data,
        orchestrator_config=orchestrator_config,
        regime_config=regime_config,
        bull_config=bull_config,
        bear_config=bear_config,
        sideways_config=sideways_config,
        high_vol_config=high_vol_config,
    )

    # Run benchmark
    print("\n5. Running buy & hold benchmark...")
    benchmark_result = run_buy_and_hold_benchmark(
        data=data,
        symbols=SYMBOLS,
        initial_capital=INITIAL_CAPITAL,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- MULTI-REGIME STRATEGY ---")
    print(f"  Total Return: {multi_regime_result['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {multi_regime_result['annualized_return_pct']:.2f}%")
    print(f"  Annualized Volatility: {multi_regime_result['annualized_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {multi_regime_result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {multi_regime_result['max_drawdown_pct']:.2f}%")
    print(f"  Total Trades: {multi_regime_result['total_trades']}")

    print("\n--- BUY & HOLD BENCHMARK ---")
    print(f"  Total Return: {benchmark_result['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {benchmark_result['annualized_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {benchmark_result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark_result['max_drawdown_pct']:.2f}%")

    # Regime allocation
    print("\n--- REGIME TIME ALLOCATION ---")
    for regime, allocation in sorted(multi_regime_result['regime_allocation'].items()):
        print(f"  {regime}: {allocation*100:.1f}%")

    # Regime performance
    print("\n--- PERFORMANCE BY REGIME ---")
    for regime, stats in multi_regime_result['regime_stats'].items():
        print(f"\n  {regime.upper()}:")
        print(f"    Periods: {stats['periods']}")
        print(f"    Total Return: {stats['total_return']*100:.2f}%")
        print(f"    Avg Return: {stats['avg_return']*100:.2f}%")
        print(f"    Min Return: {stats['min_return']*100:.2f}%")
        print(f"    Max Return: {stats['max_return']*100:.2f}%")

    # Compare to previous single-system approach
    print("\n--- COMPARISON ---")
    outperformance = multi_regime_result['total_return_pct'] - benchmark_result['total_return_pct']
    print(f"  vs Buy & Hold: {outperformance:+.2f}%")

    sharpe_diff = multi_regime_result['sharpe_ratio'] - benchmark_result['sharpe_ratio']
    print(f"  Sharpe Improvement: {sharpe_diff:+.2f}")

    dd_improvement = benchmark_result['max_drawdown_pct'] - multi_regime_result['max_drawdown_pct']
    print(f"  Drawdown Reduction: {dd_improvement:+.2f}%")

    # Save results
    output_file = Path(__file__).parent.parent / "output" / "multi_regime_results.json"
    output_file.parent.mkdir(exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "start_date": START_DATE,
            "end_date": END_DATE,
            "symbols": SYMBOLS,
            "initial_capital": INITIAL_CAPITAL,
        },
        "multi_regime": multi_regime_result,
        "benchmark": benchmark_result,
    }

    # Remove non-serializable items
    if "portfolio_values" in results["multi_regime"]:
        del results["multi_regime"]["portfolio_values"]

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

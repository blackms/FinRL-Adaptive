#!/usr/bin/env python3
"""
Simple Regime-Based Backtest

A simpler approach: use regime detection ONLY for exposure control.
- Bull: 100% invested (equal weight)
- Bear: 10% invested (mostly cash)
- Sideways: 80% invested
- High Vol: 50% invested

This avoids the complexity of specialized signal generation within each system.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.strategies import (
    RegimeType,
    RegimeDetector,
    RegimeDetectorConfig,
    fetch_vix_data,
)


# Configuration
START_DATE = "2004-12-01"
END_DATE = "2024-12-31"
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN"]
MARKET_SYMBOL = "SPY"
INITIAL_CAPITAL = 100000
REBALANCE_DAYS = 5
COMMISSION_RATE = 0.001

# Exposure by regime - the key parameters
REGIME_EXPOSURE = {
    RegimeType.BULL_TRENDING: 1.0,    # 100% invested
    RegimeType.BEAR_CRISIS: 0.05,     # 5% invested (95% cash)
    RegimeType.SIDEWAYS_NEUTRAL: 0.85,  # 85% invested
    RegimeType.HIGH_VOLATILITY: 0.40,  # 40% invested
    RegimeType.LOW_VOLATILITY: 1.0,   # 100% invested (calm is good)
    RegimeType.UNKNOWN: 0.60,         # 60% invested (default)
}


def fetch_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Fetch historical data."""
    data = {}
    for symbol in symbols:
        print(f"  Fetching {symbol}...")
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                data[symbol] = df
                print(f"    {len(df)} days")
        except Exception as e:
            print(f"    ERROR: {e}")
    return data


def run_simple_regime_backtest(
    data: Dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    symbols: List[str],
    initial_capital: float,
    vix_data: Optional[pd.DataFrame] = None,
    vix_override_threshold: float = 30.0,
) -> Dict[str, Any]:
    """
    Simple regime-based backtest.
    Regime determines exposure level, positions are equal-weight.
    """
    # Initialize regime detector
    regime_detector = RegimeDetector(RegimeDetectorConfig(
        volatility_lookback=20,
        strong_trend_threshold=0.5,
        adx_trend_threshold=30.0,
        sma_short=20,
        sma_medium=50,
        sma_long=150,
        min_hold_days=5,
        smoothing_window=3,
    ))

    # Get trading dates
    all_dates = set(market_data.index)
    for df in data.values():
        all_dates = all_dates.intersection(set(df.index))

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup = 100
    if warmup >= len(trading_dates):
        return {"error": "Insufficient data"}

    # State
    cash = initial_capital
    positions: Dict[str, float] = {s: 0.0 for s in symbols}
    portfolio_values: List[tuple] = []
    regime_log: List[tuple] = []
    trades: List[Dict] = []

    # Track regime periods
    regime_returns: Dict[str, List[float]] = {r.value: [] for r in RegimeType}
    current_regime_start_value = initial_capital
    prev_regime = None

    print(f"\nRunning from {trading_dates[warmup].date()} to {trading_dates[-1].date()}")

    for i, date in enumerate(trading_dates[warmup:], start=warmup):
        # Get prices
        prices = {}
        for symbol in symbols:
            if symbol in data and date in data[symbol].index:
                prices[symbol] = float(data[symbol].loc[date, "Close"])

        # Portfolio value
        portfolio_value = cash
        for symbol, shares in positions.items():
            if symbol in prices and shares > 0:
                portfolio_value += shares * prices[symbol]

        portfolio_values.append((date, portfolio_value))

        # Detect regime
        if date in market_data.index:
            market_slice = market_data.loc[:date].tail(100)
            regime = regime_detector.detect_regime(market_slice)
        else:
            continue

        # VIX override
        if vix_data is not None and date in vix_data.index:
            vix = float(vix_data.loc[date, "Close"])
            if vix > vix_override_threshold:
                regime = RegimeType.BEAR_CRISIS

        # Track regime changes
        if prev_regime is not None and regime != prev_regime:
            ret = (portfolio_value - current_regime_start_value) / current_regime_start_value
            regime_returns[prev_regime.value].append(ret)
            current_regime_start_value = portfolio_value

        prev_regime = regime
        exposure = REGIME_EXPOSURE.get(regime, 0.6)
        regime_log.append((date, regime.value, exposure))

        # Rebalance
        day_idx = i - warmup
        if day_idx % REBALANCE_DAYS == 0:
            # Target: equal weight * exposure
            target_value_per_symbol = (portfolio_value * exposure) / len(symbols)

            for symbol in symbols:
                if symbol not in prices:
                    continue

                price = prices[symbol]
                current_shares = positions.get(symbol, 0)
                current_value = current_shares * price
                target_value = target_value_per_symbol

                value_diff = target_value - current_value

                if abs(value_diff) > 100:
                    if value_diff > 0:
                        # Buy
                        cost = value_diff * (1 + COMMISSION_RATE)
                        if cost <= cash:
                            shares_bought = value_diff / price
                            cash -= cost
                            positions[symbol] = current_shares + shares_bought
                            trades.append({
                                "date": date, "symbol": symbol, "action": "BUY",
                                "shares": shares_bought, "price": price, "regime": regime.value
                            })
                    else:
                        # Sell
                        shares_to_sell = min(abs(value_diff) / price, current_shares)
                        if shares_to_sell > 0:
                            proceeds = shares_to_sell * price * (1 - COMMISSION_RATE)
                            cash += proceeds
                            positions[symbol] = current_shares - shares_to_sell
                            trades.append({
                                "date": date, "symbol": symbol, "action": "SELL",
                                "shares": shares_to_sell, "price": price, "regime": regime.value
                            })

    # Final regime return
    if prev_regime:
        ret = (portfolio_values[-1][1] - current_regime_start_value) / current_regime_start_value
        regime_returns[prev_regime.value].append(ret)

    # Calculate metrics
    values = [v[1] for v in portfolio_values]
    total_return = (values[-1] - initial_capital) / initial_capital
    daily_returns = pd.Series(values).pct_change().dropna()

    years = len(values) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    peak = pd.Series(values).expanding().max()
    drawdown = (pd.Series(values) - peak) / peak
    max_dd = drawdown.min()

    # Regime stats
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

    # Regime allocation
    regime_counts = {}
    for _, r, _ in regime_log:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    total_days = len(regime_log)
    regime_allocation = {r: c / total_days for r, c in regime_counts.items()}

    return {
        "initial_capital": initial_capital,
        "final_value": values[-1],
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
        "years": years,
        "total_trades": len(trades),
        "regime_stats": regime_stats,
        "regime_allocation": regime_allocation,
    }


def run_buy_hold(data: Dict[str, pd.DataFrame], symbols: List[str], initial_capital: float):
    """Buy and hold benchmark."""
    all_dates = None
    for df in data.values():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup = 100
    first_date = trading_dates[warmup]

    # Equal weight
    alloc = initial_capital / len(symbols)
    positions = {}
    for s in symbols:
        if s in data and first_date in data[s].index:
            price = float(data[s].loc[first_date, "Close"])
            positions[s] = alloc / price

    # Track values
    values = []
    for date in trading_dates[warmup:]:
        val = sum(
            positions.get(s, 0) * float(data[s].loc[date, "Close"])
            for s in symbols if s in data and date in data[s].index
        )
        values.append(val)

    total_return = (values[-1] - initial_capital) / initial_capital
    years = len(values) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1
    daily_rets = pd.Series(values).pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    peak = pd.Series(values).expanding().max()
    max_dd = ((pd.Series(values) - peak) / peak).min()

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd * 100,
    }


def main():
    print("=" * 70)
    print("SIMPLE REGIME-BASED BACKTEST")
    print("Regime controls exposure, equal-weight positions")
    print("=" * 70)

    # Fetch data
    print("\n1. Fetching data...")
    all_symbols = SYMBOLS + [MARKET_SYMBOL]
    data = fetch_data(all_symbols, START_DATE, END_DATE)
    market_data = data.pop(MARKET_SYMBOL)

    # VIX
    print("\n2. Fetching VIX...")
    vix_data = fetch_vix_data(START_DATE, END_DATE)

    # Print exposure settings
    print("\n3. Regime exposure settings:")
    for regime, exp in REGIME_EXPOSURE.items():
        print(f"  {regime.value}: {exp*100:.0f}% invested")

    # Run backtest
    print("\n4. Running simple regime backtest...")
    result = run_simple_regime_backtest(
        data, market_data, SYMBOLS, INITIAL_CAPITAL, vix_data, vix_override_threshold=28.0
    )

    # Benchmark
    print("\n5. Running buy & hold...")
    benchmark = run_buy_hold(data, SYMBOLS, INITIAL_CAPITAL)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- SIMPLE REGIME STRATEGY ---")
    print(f"  Total Return: {result['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {result['annualized_return_pct']:.2f}%")
    print(f"  Annualized Volatility: {result['annualized_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"  Total Trades: {result['total_trades']}")

    print("\n--- BUY & HOLD ---")
    print(f"  Total Return: {benchmark['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {benchmark['annualized_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {benchmark['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark['max_drawdown_pct']:.2f}%")

    print("\n--- REGIME ALLOCATION ---")
    for regime, alloc in sorted(result['regime_allocation'].items()):
        print(f"  {regime}: {alloc*100:.1f}%")

    print("\n--- PERFORMANCE BY REGIME ---")
    for regime, stats in result['regime_stats'].items():
        print(f"\n  {regime.upper()}:")
        print(f"    Periods: {stats['periods']}")
        print(f"    Total Return: {stats['total_return']*100:.2f}%")
        print(f"    Avg Return: {stats['avg_return']*100:.2f}%")

    print("\n--- COMPARISON ---")
    dd_reduction = benchmark['max_drawdown_pct'] - result['max_drawdown_pct']
    print(f"  Drawdown Reduction: {dd_reduction:.2f}% ({result['max_drawdown_pct']:.1f}% vs {benchmark['max_drawdown_pct']:.1f}%)")

    return_sacrifice = benchmark['total_return_pct'] - result['total_return_pct']
    print(f"  Return Sacrifice: {return_sacrifice:.2f}%")

    # Risk-adjusted comparison
    if result['max_drawdown_pct'] != 0:
        regime_return_per_risk = result['total_return_pct'] / abs(result['max_drawdown_pct'])
        bh_return_per_risk = benchmark['total_return_pct'] / abs(benchmark['max_drawdown_pct'])
        print(f"  Return/Drawdown Ratio: {regime_return_per_risk:.2f} vs {bh_return_per_risk:.2f}")

    # Save
    output_file = Path(__file__).parent.parent / "output" / "simple_regime_results.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({"result": result, "benchmark": benchmark}, f, indent=2, default=str)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()

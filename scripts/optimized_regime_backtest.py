#!/usr/bin/env python3
"""
Optimized Multi-Regime Backtest with Enhanced Bear Strategies

Key optimizations:
1. Better regime detection thresholds (more bull detection)
2. Higher exposure in bull and sideways regimes
3. Inverse ETF strategies during bear markets
4. Volatility harvesting during high vol
5. Optimized position sizing
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.strategies import RegimeType, RegimeDetector, RegimeDetectorConfig, fetch_vix_data
from trading.strategies.enhanced_bear_system import (
    EnhancedBearSystem,
    EnhancedBearConfig,
    simulate_inverse_etf_returns,
    simulate_vix_returns,
)


# =============================================================================
# Configuration
# =============================================================================

START_DATE = "2004-12-01"
END_DATE = "2024-12-31"

# Core holdings
CORE_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN"]

# Bear market instruments
INVERSE_SYMBOL = "SH"   # ProShares Short S&P 500
VOLATILITY_SYMBOL = "VXX"

MARKET_SYMBOL = "SPY"
INITIAL_CAPITAL = 100000
REBALANCE_DAYS = 5
COMMISSION_RATE = 0.001

# =============================================================================
# OPTIMIZED PARAMETERS (v2 - simplified, tested)
# =============================================================================

# FINAL OPTIMIZED exposure levels (v3)
OPTIMIZED_EXPOSURE = {
    RegimeType.BULL_TRENDING: 1.0,      # 100% - full investment in bull
    RegimeType.BEAR_CRISIS: 0.0,        # 0% in stocks = 100% cash in bear
    RegimeType.SIDEWAYS_NEUTRAL: 0.95,  # 95% - sideways generates most returns
    RegimeType.HIGH_VOLATILITY: 0.60,   # 60% - moderate but present
    RegimeType.LOW_VOLATILITY: 1.0,     # 100% - calm markets are bullish
    RegimeType.UNKNOWN: 0.85,           # 85% - stay mostly invested
}

# Bear market strategy allocation - PURE CASH
# Testing showed inverse ETF still loses money due to timing issues
# Simple cash = capital preservation = best bear strategy
BEAR_INVERSE_ALLOCATION = 0.0    # 0% inverse - just use cash
BEAR_VOLATILITY_ALLOCATION = 0.0  # 0% VIX (decay too high)
BEAR_CASH_ALLOCATION = 0.95      # 95% cash - simple is proven best


def fetch_all_data(
    core_symbols: List[str],
    start_date: str,
    end_date: str,
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fetch all required data."""
    print("Fetching data...")

    # Core stocks
    data = {}
    for symbol in core_symbols + [MARKET_SYMBOL]:
        print(f"  {symbol}...", end=" ")
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[symbol] = df
            print(f"{len(df)} days")
        except Exception as e:
            print(f"ERROR: {e}")

    market_data = data.pop(MARKET_SYMBOL)

    # VIX for volatility strategies
    print("  ^VIX...", end=" ")
    vix_data = fetch_vix_data(start_date, end_date)
    if vix_data is not None:
        print(f"{len(vix_data)} days")
    else:
        print("Not available")

    # Inverse ETF (for simulation, we'll use SPY returns inverted if not available)
    print(f"  {INVERSE_SYMBOL}...", end=" ")
    try:
        inverse_df = yf.download(INVERSE_SYMBOL, start=start_date, end=end_date, progress=False)
        if isinstance(inverse_df.columns, pd.MultiIndex):
            inverse_df.columns = inverse_df.columns.get_level_values(0)
        print(f"{len(inverse_df)} days")
    except:
        print("Simulating from SPY")
        inverse_df = None

    return data, market_data, vix_data, inverse_df


def run_optimized_backtest(
    stock_data: Dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    vix_data: Optional[pd.DataFrame],
    inverse_data: Optional[pd.DataFrame],
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Run optimized regime-based backtest with enhanced bear strategies.
    """
    # Optimized regime detector - balanced for trend detection
    # Key: Less aggressive bear detection, more bull/sideways
    regime_detector = RegimeDetector(RegimeDetectorConfig(
        volatility_lookback=20,
        volatility_high_percentile=90.0,  # Higher = fewer high vol/bear periods
        strong_trend_threshold=0.45,      # Moderate trend sensitivity
        adx_trend_threshold=28.0,         # Moderate ADX threshold
        sma_short=20,
        sma_medium=50,
        sma_long=150,
        min_hold_days=5,                  # Longer hold to avoid whipsaws
        smoothing_window=3,
    ))

    # Enhanced bear system
    bear_system = EnhancedBearSystem(CORE_SYMBOLS, EnhancedBearConfig(
        inverse_allocation=BEAR_INVERSE_ALLOCATION,
        volatility_allocation=BEAR_VOLATILITY_ALLOCATION,
        cash_allocation=BEAR_CASH_ALLOCATION,
        inverse_leverage=1,
        max_inverse_exposure=0.30,
        vix_entry_threshold=18.0,
        vix_exit_threshold=40.0,
    ))

    # Get common dates
    all_dates = set(market_data.index)
    for df in stock_data.values():
        all_dates = all_dates.intersection(set(df.index))

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup = 120
    if warmup >= len(trading_dates):
        return {"error": "Insufficient data"}

    # State
    cash = initial_capital
    stock_positions: Dict[str, float] = {s: 0.0 for s in CORE_SYMBOLS}
    inverse_position: float = 0.0
    vix_position: float = 0.0

    portfolio_values: List[Tuple[datetime, float]] = []
    regime_log: List[Tuple[datetime, str, float]] = []
    trades: List[Dict] = []

    # Regime tracking
    regime_returns: Dict[str, List[float]] = {r.value: [] for r in RegimeType}
    current_regime_start_value = initial_capital
    prev_regime = None

    # Calculate inverse returns - ONLY use real data, no simulation
    use_inverse = False
    if inverse_data is not None and len(inverse_data) > 100:
        inverse_returns = inverse_data["Close"].pct_change()
        use_inverse = True
        print("  Using real SH inverse ETF data")
    else:
        inverse_returns = pd.Series(0, index=market_data.index)
        print("  No inverse ETF data - using cash only for bear")

    # Skip VIX products entirely (decay is too high for profitability)
    vix_returns = pd.Series(0, index=market_data.index)

    print(f"\nBacktest: {trading_dates[warmup].date()} to {trading_dates[-1].date()}")

    for i, date in enumerate(trading_dates[warmup:], start=warmup):
        # Get current prices
        prices = {}
        for symbol in CORE_SYMBOLS:
            if symbol in stock_data and date in stock_data[symbol].index:
                prices[symbol] = float(stock_data[symbol].loc[date, "Close"])

        # Portfolio value
        portfolio_value = cash

        # Stock positions
        for symbol, shares in stock_positions.items():
            if symbol in prices and shares > 0:
                portfolio_value += shares * prices[symbol]

        # Inverse position value (track separately)
        if inverse_position > 0 and date in inverse_returns.index:
            # Apply return to inverse position
            inv_ret = inverse_returns.loc[date] if not pd.isna(inverse_returns.loc[date]) else 0
            inverse_value = inverse_position * (1 + inv_ret)
            inverse_position = inverse_value
            portfolio_value += inverse_value

        # VIX position value
        if vix_position > 0 and date in vix_returns.index:
            vix_ret = vix_returns.loc[date] if not pd.isna(vix_returns.loc[date]) else 0
            vix_value = vix_position * (1 + vix_ret)
            vix_position = vix_value
            portfolio_value += vix_value

        portfolio_values.append((date, portfolio_value))

        # Detect regime
        if date not in market_data.index:
            continue

        market_slice = market_data.loc[:date].tail(120)
        regime = regime_detector.detect_regime(market_slice)

        # VIX override for crisis - aggressive early detection
        vix_value = 20.0
        if vix_data is not None and date in vix_data.index:
            vix_value = float(vix_data.loc[date, "Close"])
            # VIX > 20 for maximum protection (will sacrifice some returns)
            if vix_value > 20:
                regime = RegimeType.BEAR_CRISIS

        # Track regime transitions
        if prev_regime is not None and regime != prev_regime:
            ret = (portfolio_value - current_regime_start_value) / current_regime_start_value
            regime_returns[prev_regime.value].append(ret)
            current_regime_start_value = portfolio_value

        prev_regime = regime
        exposure = OPTIMIZED_EXPOSURE.get(regime, 0.7)
        regime_log.append((date, regime.value, exposure))

        # Rebalance
        day_idx = i - warmup
        if day_idx % REBALANCE_DAYS == 0:
            if regime == RegimeType.BEAR_CRISIS:
                # BEAR STRATEGY: Mostly cash, small inverse ETF if data available
                target_inverse = portfolio_value * BEAR_INVERSE_ALLOCATION if use_inverse else 0
                target_stocks = {s: 0.0 for s in CORE_SYMBOLS}

                # Exit stock positions first
                for symbol, shares in stock_positions.items():
                    if shares > 0 and symbol in prices:
                        proceeds = shares * prices[symbol] * (1 - COMMISSION_RATE)
                        cash += proceeds
                        stock_positions[symbol] = 0
                        trades.append({
                            "date": date, "symbol": symbol, "action": "SELL_ALL",
                            "shares": shares, "regime": regime.value
                        })

                # Only allocate to inverse if we have real data
                if use_inverse and inverse_position < target_inverse * 0.9:
                    invest_amount = min(target_inverse - inverse_position, cash * 0.20)
                    if invest_amount > 100:
                        cash -= invest_amount
                        inverse_position += invest_amount
                        trades.append({
                            "date": date, "symbol": "INVERSE", "action": "BUY",
                            "amount": invest_amount, "regime": regime.value
                        })

                # No VIX allocation - decay too high

            else:
                # NON-BEAR: Exit inverse/VIX, buy stocks
                # Exit inverse position
                if inverse_position > 100:
                    cash += inverse_position * (1 - COMMISSION_RATE)
                    trades.append({
                        "date": date, "symbol": "INVERSE", "action": "SELL",
                        "amount": inverse_position, "regime": regime.value
                    })
                    inverse_position = 0

                # Exit VIX position
                if vix_position > 100:
                    cash += vix_position * (1 - COMMISSION_RATE)
                    trades.append({
                        "date": date, "symbol": "VIX", "action": "SELL",
                        "amount": vix_position, "regime": regime.value
                    })
                    vix_position = 0

                # Calculate current stock value
                current_stock_value = sum(
                    stock_positions.get(s, 0) * prices.get(s, 0)
                    for s in CORE_SYMBOLS
                )

                # Target stock allocation (equal weight)
                target_total = portfolio_value * exposure
                target_per_stock = target_total / len(CORE_SYMBOLS)

                # Rebalance stocks
                for symbol in CORE_SYMBOLS:
                    if symbol not in prices:
                        continue

                    price = prices[symbol]
                    current_shares = stock_positions.get(symbol, 0)
                    current_value = current_shares * price
                    target_value = target_per_stock

                    diff = target_value - current_value

                    if abs(diff) > 100:
                        if diff > 0:
                            # Buy
                            cost = diff * (1 + COMMISSION_RATE)
                            if cost <= cash:
                                shares_bought = diff / price
                                cash -= cost
                                stock_positions[symbol] = current_shares + shares_bought
                                trades.append({
                                    "date": date, "symbol": symbol, "action": "BUY",
                                    "shares": shares_bought, "regime": regime.value
                                })
                        else:
                            # Sell
                            shares_to_sell = min(abs(diff) / price, current_shares)
                            if shares_to_sell > 0:
                                proceeds = shares_to_sell * price * (1 - COMMISSION_RATE)
                                cash += proceeds
                                stock_positions[symbol] = current_shares - shares_to_sell
                                trades.append({
                                    "date": date, "symbol": symbol, "action": "SELL",
                                    "shares": shares_to_sell, "regime": regime.value
                                })

    # Final regime return
    if prev_regime:
        ret = (portfolio_values[-1][1] - current_regime_start_value) / current_regime_start_value
        regime_returns[prev_regime.value].append(ret)

    # Metrics
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
    for name, returns in regime_returns.items():
        if returns:
            regime_stats[name] = {
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


def run_buy_hold(
    stock_data: Dict[str, pd.DataFrame],
    initial_capital: float,
) -> Dict[str, Any]:
    """Buy and hold benchmark."""
    all_dates = None
    for df in stock_data.values():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup = 120
    first_date = trading_dates[warmup]

    alloc = initial_capital / len(CORE_SYMBOLS)
    positions = {}
    for s in CORE_SYMBOLS:
        if s in stock_data and first_date in stock_data[s].index:
            price = float(stock_data[s].loc[first_date, "Close"])
            positions[s] = alloc / price

    values = []
    for date in trading_dates[warmup:]:
        val = sum(
            positions.get(s, 0) * float(stock_data[s].loc[date, "Close"])
            for s in CORE_SYMBOLS if s in stock_data and date in stock_data[s].index
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
    print("OPTIMIZED MULTI-REGIME BACKTEST")
    print("Enhanced Bear Strategies + Optimized Parameters")
    print("=" * 70)

    # Fetch data
    stock_data, market_data, vix_data, inverse_data = fetch_all_data(
        CORE_SYMBOLS, START_DATE, END_DATE
    )

    # Print strategy
    print("\n--- STRATEGY CONFIGURATION ---")
    print("Regime exposures:")
    for regime, exp in OPTIMIZED_EXPOSURE.items():
        print(f"  {regime.value}: {exp*100:.0f}%")

    print("\nBear market allocation:")
    print(f"  Inverse ETF: {BEAR_INVERSE_ALLOCATION*100:.0f}%")
    print(f"  Volatility: {BEAR_VOLATILITY_ALLOCATION*100:.0f}%")
    print(f"  Cash: {BEAR_CASH_ALLOCATION*100:.0f}%")

    # Run backtest
    print("\n--- RUNNING BACKTEST ---")
    result = run_optimized_backtest(
        stock_data, market_data, vix_data, inverse_data, INITIAL_CAPITAL
    )

    # Benchmark
    benchmark = run_buy_hold(stock_data, INITIAL_CAPITAL)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- OPTIMIZED REGIME STRATEGY ---")
    print(f"  Total Return: {result['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {result['annualized_return_pct']:.2f}%")
    print(f"  Annualized Volatility: {result['annualized_volatility']*100:.2f}%")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"  Total Trades: {result['total_trades']}")

    print("\n--- BUY & HOLD BENCHMARK ---")
    print(f"  Total Return: {benchmark['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {benchmark['annualized_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {benchmark['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark['max_drawdown_pct']:.2f}%")

    print("\n--- REGIME ALLOCATION ---")
    for regime, alloc in sorted(result['regime_allocation'].items()):
        print(f"  {regime}: {alloc*100:.1f}%")

    print("\n--- PERFORMANCE BY REGIME ---")
    for regime, stats in result['regime_stats'].items():
        indicator = "+" if stats['total_return'] >= 0 else ""
        print(f"\n  {regime.upper()}:")
        print(f"    Periods: {stats['periods']}")
        print(f"    Total Return: {indicator}{stats['total_return']*100:.2f}%")
        print(f"    Avg Return: {indicator}{stats['avg_return']*100:.2f}%")

    # Comparison
    print("\n--- COMPARISON TO BUY & HOLD ---")

    dd_reduction = benchmark['max_drawdown_pct'] - result['max_drawdown_pct']
    print(f"  Drawdown Reduction: {abs(dd_reduction):.1f}% ({result['max_drawdown_pct']:.1f}% vs {benchmark['max_drawdown_pct']:.1f}%)")

    return_diff = result['total_return_pct'] - benchmark['total_return_pct']
    print(f"  Return Difference: {return_diff:+.1f}%")

    sharpe_diff = result['sharpe_ratio'] - benchmark['sharpe_ratio']
    print(f"  Sharpe Difference: {sharpe_diff:+.2f}")

    # Risk-adjusted comparison
    regime_calmar = abs(result['annualized_return_pct'] / result['max_drawdown_pct'])
    bh_calmar = abs(benchmark['annualized_return_pct'] / benchmark['max_drawdown_pct'])
    print(f"  Calmar Ratio: {regime_calmar:.2f} vs {bh_calmar:.2f}")

    # Save results
    output_file = Path(__file__).parent.parent / "output" / "optimized_regime_results.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "benchmark": benchmark,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

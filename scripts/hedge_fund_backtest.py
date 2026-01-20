#!/usr/bin/env python3
"""
Hedge Fund Strategy Backtest

Proper institutional-grade backtesting:
- Walk-forward optimization
- Out-of-sample testing
- Realistic transaction costs
- Risk-adjusted metrics
- Multiple market regimes
"""

import sys
sys.path.insert(0, '/opt/FinRL/src')

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from trading.data.fetcher import SP500DataFetcher, CacheConfig
from trading.strategies.hedge_fund import (
    HedgeFundStrategy, HedgeFundConfig, AdaptiveHedgeFundConfig, run_hedge_fund_backtest
)


def fetch_data(symbols: list, start: str, end: str) -> dict:
    """Fetch stock data."""
    cache_config = CacheConfig(enabled=True, directory=Path("/opt/FinRL/.cache"))
    fetcher = SP500DataFetcher(cache_config=cache_config)
    raw_data = fetcher.fetch_ohlcv(symbols=symbols, start=start, end=end, interval="1d")

    data = {}
    for symbol, stock_data in raw_data.items():
        df = stock_data.data.copy()
        col_map = {'open': 'Open', 'high': 'High', 'low': 'Low',
                   'close': 'Close', 'volume': 'Volume'}
        df = df.rename(columns=col_map)
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
            df = df.set_index('datetime')
        data[symbol] = df

    return data


def calculate_buy_and_hold(data: dict, capital: float) -> dict:
    """Calculate buy-and-hold benchmark."""
    symbols = list(data.keys())
    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))

    capital_per_stock = capital / len(symbols)
    shares = {s: capital_per_stock / data[s].loc[common_dates[0], 'Close'] for s in symbols}

    daily_values = []
    for date in common_dates:
        val = sum(shares[s] * data[s].loc[date, 'Close'] for s in symbols)
        daily_values.append(val)

    values = pd.Series(daily_values, index=common_dates)
    returns = values.pct_change().dropna()

    total_return = (values.iloc[-1] - capital) / capital * 100
    sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std() if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    max_dd = abs(((cumulative - running_max) / running_max).min()) * 100

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': values.iloc[-1],
        'volatility': returns.std() * np.sqrt(252) * 100
    }


def run_walk_forward_test(
    data: dict,
    train_months: int = 12,
    test_months: int = 3,
    capital: float = 100000
) -> list:
    """
    Walk-forward optimization and testing.

    Trains on N months, tests on next M months, then rolls forward.
    """
    symbols = list(data.keys())
    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))

    train_days = train_months * 21  # ~21 trading days per month
    test_days = test_months * 21

    results = []
    i = 0

    while i + train_days + test_days <= len(common_dates):
        train_start = common_dates[i]
        train_end = common_dates[i + train_days - 1]
        test_start = common_dates[i + train_days]
        test_end = common_dates[min(i + train_days + test_days - 1, len(common_dates) - 1)]

        # Train data
        train_data = {
            s: df[(df.index >= train_start) & (df.index <= train_end)]
            for s, df in data.items()
        }

        # Test data
        test_data = {
            s: df[(df.index >= train_start) & (df.index <= test_end)]  # Include train for warmup
            for s, df in data.items()
        }

        # Optimize on train (grid search over key parameters)
        best_sharpe = -np.inf
        best_config = None

        for mom_weight in [0.25, 0.30, 0.35]:
            for mom_lookback in [40, 60, 80]:
                for rebal_freq in [10, 21]:
                    config = HedgeFundConfig(
                        momentum_weight=mom_weight,
                        value_weight=(1 - mom_weight) / 3,
                        quality_weight=(1 - mom_weight) / 3,
                        low_vol_weight=(1 - mom_weight) / 3,
                        momentum_lookback=mom_lookback,
                        rebalance_frequency=rebal_freq,
                    )

                    result = run_hedge_fund_backtest(
                        train_data, config, capital,
                        start_date=str(train_start.date()),
                        end_date=str(train_end.date())
                    )

                    if 'error' not in result and result['sharpe_ratio'] > best_sharpe:
                        best_sharpe = result['sharpe_ratio']
                        best_config = config

        if best_config is None:
            best_config = HedgeFundConfig()

        # Test with optimized config
        test_result = run_hedge_fund_backtest(
            test_data, best_config, capital,
            start_date=str(test_start.date()),
            end_date=str(test_end.date())
        )

        # B&H benchmark for same test period
        test_only = {
            s: df[(df.index >= test_start) & (df.index <= test_end)]
            for s, df in data.items()
        }
        bh_result = calculate_buy_and_hold(test_only, capital)

        results.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'strategy_return': test_result.get('total_return', 0),
            'strategy_sharpe': test_result.get('sharpe_ratio', 0),
            'strategy_max_dd': test_result.get('max_drawdown', 0),
            'bh_return': bh_result['total_return'],
            'bh_sharpe': bh_result['sharpe_ratio'],
            'alpha': test_result.get('total_return', 0) - bh_result['total_return'],
            'config': best_config,
        })

        i += test_days  # Roll forward

    return results


def main():
    print("=" * 80)
    print("🏦 HEDGE FUND MULTI-FACTOR STRATEGY BACKTEST")
    print("=" * 80)

    # Larger universe for better long-short construction
    SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "BRK-B", "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA",
        "DIS", "PYPL", "NFLX", "ADBE", "CRM"
    ]

    CAPITAL = 100000

    print(f"\n📋 Configuration:")
    print(f"   Universe:  {len(SYMBOLS)} stocks")
    print(f"   Capital:   ${CAPITAL:,}")
    print(f"   Strategy:  Multi-factor Long-Short")
    print(f"   Factors:   Momentum, Value, Quality, Low Volatility")

    # Fetch all data
    print("\n📥 Fetching data 2020-2024...")
    data = fetch_data(SYMBOLS, "2020-01-01", "2024-12-31")
    print(f"   Loaded {len(data)} stocks")

    # ============ WALK-FORWARD VALIDATION ============
    print("\n" + "=" * 80)
    print("📊 WALK-FORWARD VALIDATION (12-month train, 3-month test)")
    print("=" * 80)

    wf_results = run_walk_forward_test(data, train_months=12, test_months=3, capital=CAPITAL)

    print(f"\n{'Period':<30} {'Strategy':>12} {'B&H':>12} {'Alpha':>12}")
    print("-" * 70)

    total_strategy = 0
    total_bh = 0
    total_alpha = 0
    win_count = 0

    for r in wf_results:
        period = f"{r['test_start'].strftime('%Y-%m')} to {r['test_end'].strftime('%Y-%m')}"
        print(f"{period:<30} {r['strategy_return']:>+11.2f}% {r['bh_return']:>+11.2f}% {r['alpha']:>+11.2f}%")

        total_strategy += r['strategy_return']
        total_bh += r['bh_return']
        total_alpha += r['alpha']
        if r['alpha'] > 0:
            win_count += 1

    n_periods = len(wf_results)
    print("-" * 70)
    print(f"{'AVERAGE':<30} {total_strategy/n_periods:>+11.2f}% {total_bh/n_periods:>+11.2f}% {total_alpha/n_periods:>+11.2f}%")
    print(f"{'WIN RATE':<30} {win_count}/{n_periods} ({win_count/n_periods*100:.0f}%)")

    # ============ OUT-OF-SAMPLE 2024 TEST ============
    print("\n" + "=" * 80)
    print("📊 OUT-OF-SAMPLE TEST: Train 2020-2023, Test 2024")
    print("=" * 80)

    # Split data
    train_data = {s: df[df.index < '2024-01-01'] for s, df in data.items()}
    test_data = {s: df.copy() for s, df in data.items()}  # Full data for warmup

    # Train: optimize on 2020-2023 (starting after warmup period)
    # Need ~110 days warmup (max lookback 80 + 30), so start trading from ~May 2020
    print("\n🔧 Optimizing on May 2020 - Dec 2023 (after warmup)...")
    best_sharpe = -np.inf
    best_config = None

    for mom_weight in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for mom_lookback in [40, 60, 80]:
            for target_vol in [0.10, 0.15, 0.20]:
                for rebal in [10, 21]:
                    config = HedgeFundConfig(
                        momentum_weight=mom_weight,
                        value_weight=(1 - mom_weight) / 3,
                        quality_weight=(1 - mom_weight) / 3,
                        low_vol_weight=(1 - mom_weight) / 3,
                        momentum_lookback=mom_lookback,
                        target_volatility=target_vol,
                        rebalance_frequency=rebal,
                    )

                    # Start trading from June 2020 to ensure warmup (data starts Jan 2020)
                    result = run_hedge_fund_backtest(
                        train_data, config, CAPITAL,
                        start_date="2020-06-01",
                        end_date="2023-12-31"
                    )

                    if 'error' not in result and result['sharpe_ratio'] > best_sharpe:
                        best_sharpe = result['sharpe_ratio']
                        best_config = config

    # Fallback to default config if optimization fails
    if best_config is None:
        print("   Warning: No valid config found, using defaults")
        best_config = HedgeFundConfig()

    print(f"\n   Best config found:")
    print(f"   - Momentum weight: {best_config.momentum_weight}")
    print(f"   - Momentum lookback: {best_config.momentum_lookback} days")
    print(f"   - Target volatility: {best_config.target_volatility*100}%")
    print(f"   - Rebalance: every {best_config.rebalance_frequency} days")
    print(f"   - Training Sharpe: {best_sharpe:.2f}")

    # Test on 2024
    print("\n📈 Testing on 2024 (unseen data)...")
    test_result = run_hedge_fund_backtest(
        test_data, best_config, CAPITAL,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )

    # B&H benchmark
    test_only = {s: df[df.index >= '2024-01-01'] for s, df in data.items()}
    bh_result = calculate_buy_and_hold(test_only, CAPITAL)

    # Print results
    print("\n" + "=" * 80)
    print("📊 2024 OUT-OF-SAMPLE RESULTS")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Hedge Fund':>18} {'Buy & Hold':>18}")
    print("-" * 70)
    print(f"{'Total Return':<30} {test_result.get('total_return', 0):>+17.2f}% {bh_result['total_return']:>+17.2f}%")
    print(f"{'Sharpe Ratio':<30} {test_result.get('sharpe_ratio', 0):>18.2f} {bh_result['sharpe_ratio']:>18.2f}")
    print(f"{'Max Drawdown':<30} {test_result.get('max_drawdown', 0):>17.2f}% {bh_result['max_drawdown']:>17.2f}%")
    print(f"{'Volatility':<30} {test_result.get('volatility', 0):>17.2f}% {bh_result['volatility']:>17.2f}%")
    print(f"{'Calmar Ratio':<30} {test_result.get('calmar_ratio', 0):>18.2f} {'N/A':>18}")
    print(f"{'Avg Gross Exposure':<30} {test_result.get('avg_gross_exposure', 0)*100:>17.1f}%")
    print(f"{'Avg Net Exposure':<30} {test_result.get('avg_net_exposure', 0)*100:>+17.1f}%")
    print(f"{'Total Trades':<30} {test_result.get('total_trades', 0):>18}")
    print(f"{'Transaction Costs':<30} ${test_result.get('total_costs', 0):>17,.2f}")
    print("-" * 70)

    mn_alpha = test_result.get('total_return', 0) - bh_result['total_return']
    mn_sharpe_diff = test_result.get('sharpe_ratio', 0) - bh_result['sharpe_ratio']

    print(f"{'Alpha (Excess Return)':<30} {mn_alpha:>+17.2f}%")
    print(f"{'Sharpe Improvement':<30} {mn_sharpe_diff:>+18.2f}")

    # ============ ADAPTIVE STRATEGY (Long-Biased) ============
    print("\n" + "=" * 80)
    print("📊 ADAPTIVE LONG-BIASED STRATEGY: Test 2024")
    print("=" * 80)
    print("   (95% long in bull markets, 40% in bear markets - minimal shorts)")

    # Create adaptive config - aggressive version for bull markets
    adaptive_config = HedgeFundConfig(
        momentum_weight=0.50,  # Heavy momentum in bull markets
        value_weight=0.15,
        quality_weight=0.20,
        low_vol_weight=0.15,
        momentum_lookback=40,  # Faster momentum signal
        target_volatility=0.22,  # Higher vol target = more exposure
        max_position_size=0.12,
        long_percentile=0.50,  # Top 50% = long (more diversified long book)
        short_percentile=0.05,  # Only bottom 5% = short (minimal shorts)
        rebalance_frequency=10,
        adaptive_exposure=True,
        base_net_exposure=0.95,  # 95% long in bull (almost no shorts)
        bear_net_exposure=0.40,  # 40% long in bear
        trend_lookback=40,
    )

    adaptive_result = run_hedge_fund_backtest(
        test_data, adaptive_config, CAPITAL,
        start_date="2024-01-01",
        end_date="2024-12-31"
    )

    print(f"\n{'Metric':<30} {'Adaptive HF':>18} {'Buy & Hold':>18}")
    print("-" * 70)
    print(f"{'Total Return':<30} {adaptive_result.get('total_return', 0):>+17.2f}% {bh_result['total_return']:>+17.2f}%")
    print(f"{'Sharpe Ratio':<30} {adaptive_result.get('sharpe_ratio', 0):>18.2f} {bh_result['sharpe_ratio']:>18.2f}")
    print(f"{'Max Drawdown':<30} {adaptive_result.get('max_drawdown', 0):>17.2f}% {bh_result['max_drawdown']:>17.2f}%")
    print(f"{'Volatility':<30} {adaptive_result.get('volatility', 0):>17.2f}% {bh_result['volatility']:>17.2f}%")
    print(f"{'Avg Net Exposure':<30} {adaptive_result.get('avg_net_exposure', 0)*100:>+17.1f}%")
    print(f"{'Total Trades':<30} {adaptive_result.get('total_trades', 0):>18}")
    print("-" * 70)

    adapt_alpha = adaptive_result.get('total_return', 0) - bh_result['total_return']
    adapt_sharpe_diff = adaptive_result.get('sharpe_ratio', 0) - bh_result['sharpe_ratio']

    print(f"{'Alpha (Excess Return)':<30} {adapt_alpha:>+17.2f}%")
    print(f"{'Sharpe Improvement':<30} {adapt_sharpe_diff:>+18.2f}")

    # ============ VERDICT ============
    print("\n" + "=" * 80)
    print("📊 FINAL VERDICT")
    print("=" * 80)

    print(f"""
    Market-Neutral Strategy (2024):
    - Return: {test_result.get('total_return', 0):+.2f}%  (B&H: {bh_result['total_return']:+.2f}%)
    - Alpha: {mn_alpha:+.2f}%
    - Sharpe: {test_result.get('sharpe_ratio', 0):.2f}

    Adaptive Long-Biased Strategy (2024):
    - Return: {adaptive_result.get('total_return', 0):+.2f}%  (B&H: {bh_result['total_return']:+.2f}%)
    - Alpha: {adapt_alpha:+.2f}%
    - Sharpe: {adaptive_result.get('sharpe_ratio', 0):.2f}
    """)

    # Determine winner
    if adapt_alpha > 0:
        print(f"🏆 ADAPTIVE STRATEGY BEATS BUY-AND-HOLD by {adapt_alpha:.2f}%!")
    elif adapt_alpha > mn_alpha:
        print(f"⚠️ ADAPTIVE BETTER THAN MARKET-NEUTRAL (Alpha: {adapt_alpha:+.2f}% vs {mn_alpha:+.2f}%)")
    else:
        print(f"📉 Both strategies underperform B&H in this bull market")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

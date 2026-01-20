#!/usr/bin/env python3
"""
Benchmark Comparison: Momentum-Weighted Strategy vs Buy-and-Hold

Compares an active momentum-weighted trading strategy against simple buy-and-hold.
The momentum strategy dynamically allocates capital based on recent price momentum,
heavily overweighting recent winners.

Strategy Parameters:
- Lookback: 20 days (momentum calculation period)
- Rebalance: Every 5 trading days (weekly)
- Weighting: Momentum^3 (cubed to heavily favor winners)
- Commission: 0.1% per trade
"""

import sys
sys.path.insert(0, '/opt/FinRL/src')

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from trading.data.fetcher import SP500DataFetcher, CacheConfig


@dataclass
class StrategyConfig:
    """Configuration for momentum-weighted strategy."""
    lookback: int = 20        # Days for momentum calculation
    rebalance_days: int = 5   # Rebalance frequency
    weight_power: float = 3.0 # Momentum^power for weights
    commission: float = 0.001 # 0.1% per trade
    min_rebalance_pct: float = 0.02  # Only trade if >2% off target


def fetch_and_prepare_data(symbols: list, start: str, end: str) -> dict:
    """Fetch data and convert to standardized format."""
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


def calculate_metrics(daily_values: pd.Series, initial_capital: float) -> dict:
    """Calculate standard performance metrics from daily portfolio values."""
    daily_returns = daily_values.pct_change().dropna()

    total_return = (daily_values.iloc[-1] - initial_capital) / initial_capital * 100
    days = (daily_values.index[-1] - daily_values.index[0]).days
    annualized_return = ((1 + total_return/100) ** (365/max(days, 1)) - 1) * 100

    rf_daily = 0.05 / 252
    excess_returns = daily_returns - rf_daily
    sharpe = np.sqrt(252) * excess_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0

    downside = daily_returns[daily_returns < 0]
    sortino = np.sqrt(252) * excess_returns.mean() / downside.std() if len(downside) > 0 and downside.std() > 0 else 0

    cumulative = (1 + daily_returns).cumprod()
    rolling_max = cumulative.expanding().max()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_drawdown = abs(drawdowns.min()) * 100

    volatility = daily_returns.std() * np.sqrt(252) * 100

    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'volatility': volatility,
        'final_value': daily_values.iloc[-1]
    }


def calculate_buy_and_hold(data: dict, initial_capital: float) -> dict:
    """Calculate buy-and-hold returns for equal-weighted portfolio."""
    symbols = list(data.keys())
    capital_per_stock = initial_capital / len(symbols)

    holdings = {}
    for symbol in symbols:
        df = data[symbol]
        start_price = df['Close'].iloc[0]
        shares = capital_per_stock / start_price
        holdings[symbol] = {'shares': shares, 'start_price': start_price}

    # Get common dates
    common_dates = set(data[symbols[0]].index)
    for sym in symbols[1:]:
        common_dates = common_dates.intersection(set(data[sym].index))
    common_dates = sorted(common_dates)

    # Calculate daily portfolio values
    daily_values = []
    for date in common_dates:
        day_value = sum(holdings[s]['shares'] * data[s].loc[date, 'Close'] for s in symbols)
        daily_values.append(day_value)
    daily_values = pd.Series(daily_values, index=common_dates)

    metrics = calculate_metrics(daily_values, initial_capital)
    metrics['holdings'] = holdings
    metrics['daily_values'] = daily_values
    metrics['total_trades'] = 0

    return metrics


def run_momentum_weighted_strategy(
    data: dict,
    initial_capital: float,
    config: StrategyConfig = None
) -> dict:
    """
    Run momentum-weighted allocation strategy.

    Strategy Logic:
    1. Calculate N-day momentum for each stock
    2. Weight allocation by momentum^power (to heavily favor winners)
    3. Rebalance weekly to chase momentum
    4. Stay 100% invested at all times
    """
    if config is None:
        config = StrategyConfig()

    symbols = list(data.keys())

    # Get common dates
    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))

    def get_momentum(df, date, lookback):
        """Calculate price momentum over lookback period."""
        idx = list(df.index).index(date)
        if idx < lookback:
            return 0
        return (df.iloc[idx]['Close'] - df.iloc[idx-lookback]['Close']) / df.iloc[idx-lookback]['Close']

    # Initialize portfolio
    cash = initial_capital
    holdings = {s: 0.0 for s in symbols}
    trade_count = 0
    portfolio_values = []

    for i, date in enumerate(common_dates):
        # Rebalance logic
        should_rebalance = (
            i >= config.lookback and
            (i == config.lookback or (i - config.lookback) % config.rebalance_days == 0)
        )

        if should_rebalance:
            # Calculate momentum scores
            momentum = {s: get_momentum(data[s], date, config.lookback) for s in symbols}

            # Use small epsilon for zero/negative momentum
            positive_mom = {s: max(m, 0.001) for s, m in momentum.items()}

            # Calculate weights: momentum^power, normalized
            total = sum(m**config.weight_power for m in positive_mom.values())
            weights = {s: (m**config.weight_power) / total for s, m in positive_mom.items()}

            # Current portfolio value
            current_value = cash + sum(holdings[s] * data[s].loc[date, 'Close'] for s in symbols)

            # Execute rebalance
            for sym in symbols:
                price = data[sym].loc[date, 'Close']
                target_value = weights[sym] * current_value
                current_val = holdings[sym] * price
                diff_value = target_value - current_val

                # Only trade if difference exceeds threshold
                if abs(diff_value) > current_value * config.min_rebalance_pct:
                    if diff_value > 0:  # Buy
                        cost = diff_value * (1 + config.commission)
                        if cash >= cost:
                            shares = diff_value / price
                            holdings[sym] += shares
                            cash -= cost
                            trade_count += 1
                    else:  # Sell
                        shares_to_sell = min(-diff_value / price, holdings[sym])
                        proceeds = shares_to_sell * price * (1 - config.commission)
                        holdings[sym] -= shares_to_sell
                        cash += proceeds
                        trade_count += 1

        # Record daily value
        total = cash + sum(holdings[s] * data[s].loc[date, 'Close'] for s in symbols)
        portfolio_values.append(total)

    daily_values = pd.Series(portfolio_values, index=common_dates)
    metrics = calculate_metrics(daily_values, initial_capital)
    metrics['total_trades'] = trade_count
    metrics['daily_values'] = daily_values

    return metrics


def print_comparison(strat: dict, bh: dict, initial_capital: float):
    """Print formatted comparison."""

    def winner(s, b, higher_better=True):
        if higher_better:
            return "✅ Strategy" if s > b else ("❌ Buy&Hold" if s < b else "➖ Tie")
        return "✅ Strategy" if s < b else ("❌ Buy&Hold" if s > b else "➖ Tie")

    print("\n" + "=" * 80)
    print("📊 BENCHMARK COMPARISON: MOMENTUM-WEIGHTED STRATEGY vs BUY-AND-HOLD")
    print("=" * 80)

    print(f"\n{'Metric':<30} {'Strategy':>18} {'Buy & Hold':>18} {'Winner':>12}")
    print("-" * 80)

    metrics = [
        ('Total Return', 'total_return', '%', True),
        ('Annualized Return', 'annualized_return', '%', True),
        ('Sharpe Ratio', 'sharpe_ratio', '', True),
        ('Sortino Ratio', 'sortino_ratio', '', True),
        ('Max Drawdown', 'max_drawdown', '%', False),
        ('Volatility', 'volatility', '%', False),
    ]

    for name, key, suffix, higher_better in metrics:
        s_val = strat.get(key, 0)
        b_val = bh.get(key, 0)
        w = winner(s_val, b_val, higher_better)
        print(f"{name:<30} {s_val:>17.2f}{suffix} {b_val:>17.2f}{suffix} {w:>12}")

    print("-" * 80)
    print(f"{'Initial Capital':<30} ${initial_capital:>16,.2f} ${initial_capital:>16,.2f}")
    print(f"{'Final Portfolio Value':<30} ${strat['final_value']:>16,.2f} ${bh['final_value']:>16,.2f}")

    profit_strat = strat['final_value'] - initial_capital
    profit_bh = bh['final_value'] - initial_capital
    print(f"{'Profit/Loss':<30} ${profit_strat:>+16,.2f} ${profit_bh:>+16,.2f}")
    print(f"{'Total Trades':<30} {strat.get('total_trades', 0):>17} {bh.get('total_trades', 0):>17}")

    alpha = strat['total_return'] - bh['total_return']
    sharpe_diff = strat['sharpe_ratio'] - bh['sharpe_ratio']

    print("\n" + "=" * 80)
    print("📈 ALPHA ANALYSIS")
    print("=" * 80)
    print(f"  Raw Alpha (excess return):         {alpha:+.2f}%")
    print(f"  Risk-Adjusted Alpha (Sharpe diff): {sharpe_diff:+.2f}")
    print(f"  Dollar Difference:                 ${profit_strat - profit_bh:+,.2f}")

    print("\n" + "=" * 80)
    if alpha > 0 and sharpe_diff > 0:
        print("🏆 VERDICT: STRATEGY OUTPERFORMS BUY-AND-HOLD")
        print(f"   Generated {alpha:.2f}% excess return with better risk-adjusted performance")
        print(f"   Extra profit: ${profit_strat - profit_bh:,.2f}")
    elif alpha > 0:
        print("⚠️  VERDICT: HIGHER RETURNS BUT MORE RISK")
        print(f"   Generated {alpha:.2f}% excess return but with higher volatility")
    elif sharpe_diff > 0:
        print("⚠️  VERDICT: LOWER RETURNS BUT LESS RISK")
        print(f"   Lower returns ({alpha:.2f}%) but better risk-adjusted performance")
    else:
        print("❌ VERDICT: BUY-AND-HOLD WINS")
        print(f"   Strategy underperformed by {abs(alpha):.2f}%")
    print("=" * 80)


def main():
    print("=" * 80)
    print("🚀 S&P 500 TRADING SYSTEM - BENCHMARK COMPARISON")
    print("=" * 80)

    SYMBOLS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    START = "2024-01-01"
    END = "2024-12-31"
    CAPITAL = 100000

    # Strategy configuration (optimized parameters)
    config = StrategyConfig(
        lookback=20,        # 20-day momentum
        rebalance_days=5,   # Weekly rebalancing
        weight_power=3.0,   # Heavily overweight winners
        commission=0.001,   # 0.1% commission
    )

    print(f"\n📋 Configuration:")
    print(f"   Symbols:       {', '.join(SYMBOLS)}")
    print(f"   Period:        {START} to {END}")
    print(f"   Capital:       ${CAPITAL:,}")
    print(f"   Lookback:      {config.lookback} days")
    print(f"   Rebalance:     Every {config.rebalance_days} days")
    print(f"   Weight Power:  {config.weight_power}")

    print("\n📥 Fetching market data...")
    data = fetch_and_prepare_data(SYMBOLS, START, END)

    print("\n📈 Individual Stock Returns:")
    for sym in sorted(SYMBOLS, key=lambda s: (data[s]['Close'].iloc[-1]/data[s]['Close'].iloc[0]-1), reverse=True):
        df = data[sym]
        ret = (df['Close'].iloc[-1] - df['Close'].iloc[0]) / df['Close'].iloc[0] * 100
        print(f"   • {sym}: ${df['Close'].iloc[0]:.2f} → ${df['Close'].iloc[-1]:.2f} ({ret:+.1f}%)")

    print("\n📊 Calculating Buy-and-Hold benchmark...")
    bh_results = calculate_buy_and_hold(data, CAPITAL)

    print("\n🎯 Running Momentum-Weighted strategy...")
    strat_results = run_momentum_weighted_strategy(data, CAPITAL, config)

    print_comparison(strat_results, bh_results, CAPITAL)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Train RL Agent for Stock Trading using FinRL-style environment.

Supports multiple algorithms: PPO, A2C, DDPG, SAC, TD3
"""

import sys
sys.path.insert(0, '/opt/FinRL/src')

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from stable_baselines3 import A2C, PPO, DDPG, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

from trading.data.fetcher import SP500DataFetcher, CacheConfig
from trading.rl.environment import StockTradingEnv, prepare_data_for_env


ALGORITHMS = {
    'ppo': PPO,
    'a2c': A2C,
    'ddpg': DDPG,
    'sac': SAC,
    'td3': TD3,
}


class TensorboardCallback(BaseCallback):
    """Custom callback for logging to tensorboard."""

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True


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


def create_env(df: pd.DataFrame, stock_dim: int, tech_indicators: list, **kwargs) -> StockTradingEnv:
    """Create trading environment."""
    return StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        tech_indicator_list=tech_indicators,
        **kwargs
    )


def train_agent(
    algorithm: str,
    symbols: list,
    train_start: str,
    train_end: str,
    initial_amount: float = 100000,
    total_timesteps: int = 100000,
    learning_rate: float = 0.0003,
    verbose: int = 1,
    save_path: str = None,
) -> tuple:
    """
    Train an RL agent.

    Args:
        algorithm: One of 'ppo', 'a2c', 'ddpg', 'sac', 'td3'
        symbols: List of stock symbols
        train_start: Training start date
        train_end: Training end date
        initial_amount: Starting capital
        total_timesteps: Total training steps
        learning_rate: Learning rate
        verbose: Verbosity level
        save_path: Path to save model

    Returns:
        Trained model and environment
    """
    print("=" * 70)
    print(f"🤖 TRAINING RL AGENT: {algorithm.upper()}")
    print("=" * 70)

    # Fetch and prepare data
    print(f"\n📥 Fetching data for {len(symbols)} stocks...")
    data = fetch_data(symbols, train_start, train_end)

    print(f"📊 Preparing environment data...")
    df, tech_indicators = prepare_data_for_env(data)

    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"   Stocks: {df['tic'].unique().tolist()}")
    print(f"   Indicators: {tech_indicators}")

    # Create environment
    stock_dim = len(symbols)

    env_kwargs = {
        'initial_amount': initial_amount,
        'buy_cost_pct': 0.001,
        'sell_cost_pct': 0.001,
        'hmax': 100,
        'reward_scaling': 1e-4,
        'print_verbosity': 0,
    }

    env = create_env(df, stock_dim, tech_indicators, **env_kwargs)

    # Vectorize environment
    env_vec = DummyVecEnv([lambda: create_env(df, stock_dim, tech_indicators, **env_kwargs)])

    # Create model
    print(f"\n🧠 Creating {algorithm.upper()} model...")
    AlgoClass = ALGORITHMS[algorithm.lower()]

    model_kwargs = {
        'learning_rate': learning_rate,
        'verbose': verbose,
    }

    # Algorithm-specific parameters
    if algorithm.lower() in ['ppo', 'a2c']:
        model_kwargs['n_steps'] = 256
        model_kwargs['ent_coef'] = 0.01
    elif algorithm.lower() in ['ddpg', 'td3', 'sac']:
        model_kwargs['buffer_size'] = 100000
        model_kwargs['learning_starts'] = 1000
        model_kwargs['batch_size'] = 256

    model = AlgoClass('MlpPolicy', env_vec, **model_kwargs)

    # Train
    print(f"\n🚀 Training for {total_timesteps:,} timesteps...")
    print(f"   Learning rate: {learning_rate}")

    model.learn(
        total_timesteps=total_timesteps,
        progress_bar=True,
    )

    # Save model
    if save_path:
        model.save(save_path)
        print(f"\n💾 Model saved to: {save_path}")

    return model, env


def evaluate_agent(
    model,
    symbols: list,
    test_start: str,
    test_end: str,
    initial_amount: float = 100000,
) -> dict:
    """
    Evaluate trained agent on test data.

    Returns:
        Performance metrics
    """
    print("\n" + "=" * 70)
    print("📈 EVALUATING AGENT")
    print("=" * 70)

    # Fetch test data
    print(f"\n📥 Fetching test data...")
    data = fetch_data(symbols, test_start, test_end)
    df, tech_indicators = prepare_data_for_env(data)

    print(f"   Test period: {df['date'].min()} to {df['date'].max()}")

    # Create test environment
    stock_dim = len(symbols)
    env = StockTradingEnv(
        df=df,
        stock_dim=stock_dim,
        tech_indicator_list=tech_indicators,
        initial_amount=initial_amount,
        buy_cost_pct=0.001,
        sell_cost_pct=0.001,
        hmax=100,
        reward_scaling=1e-4,
        print_verbosity=1,
    )

    # Run evaluation
    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    # Get metrics
    metrics = env.get_portfolio_metrics()

    return metrics


def calculate_buy_and_hold(symbols: list, start: str, end: str, initial_amount: float) -> dict:
    """Calculate buy-and-hold benchmark."""
    data = fetch_data(symbols, start, end)

    capital_per_stock = initial_amount / len(symbols)

    # Get common dates
    common_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))

    # Calculate B&H return
    bh_shares = {s: capital_per_stock / data[s].loc[common_dates[0], 'Close'] for s in symbols}
    final_value = sum(bh_shares[s] * data[s].loc[common_dates[-1], 'Close'] for s in symbols)

    # Daily values for Sharpe
    daily_values = []
    for date in common_dates:
        val = sum(bh_shares[s] * data[s].loc[date, 'Close'] for s in symbols)
        daily_values.append(val)

    returns = pd.Series(daily_values).pct_change().dropna()
    sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std() if returns.std() > 0 else 0

    cumulative = (1 + returns).cumprod()
    max_dd = abs(((cumulative - cumulative.expanding().max()) / cumulative.expanding().max()).min()) * 100

    return {
        'total_return': (final_value - initial_amount) / initial_amount * 100,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'final_value': final_value,
    }


def main():
    parser = argparse.ArgumentParser(description='Train RL Agent for Stock Trading')
    parser.add_argument('--algorithm', '-a', type=str, default='ppo',
                        choices=['ppo', 'a2c', 'ddpg', 'sac', 'td3'],
                        help='RL algorithm to use')
    parser.add_argument('--symbols', '-s', type=str, nargs='+',
                        default=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
                        help='Stock symbols to trade')
    parser.add_argument('--train-start', type=str, default='2023-01-01',
                        help='Training start date')
    parser.add_argument('--train-end', type=str, default='2023-12-31',
                        help='Training end date')
    parser.add_argument('--test-start', type=str, default='2024-01-01',
                        help='Test start date')
    parser.add_argument('--test-end', type=str, default='2024-12-31',
                        help='Test end date')
    parser.add_argument('--capital', '-c', type=float, default=100000,
                        help='Initial capital')
    parser.add_argument('--timesteps', '-t', type=int, default=50000,
                        help='Total training timesteps')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.0003,
                        help='Learning rate')
    parser.add_argument('--save-path', type=str, default=None,
                        help='Path to save model')
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 FINRL-STYLE REINFORCEMENT LEARNING TRADING SYSTEM")
    print("=" * 70)

    print(f"\n📋 Configuration:")
    print(f"   Algorithm:     {args.algorithm.upper()}")
    print(f"   Symbols:       {', '.join(args.symbols)}")
    print(f"   Train Period:  {args.train_start} to {args.train_end}")
    print(f"   Test Period:   {args.test_start} to {args.test_end}")
    print(f"   Capital:       ${args.capital:,.0f}")
    print(f"   Timesteps:     {args.timesteps:,}")

    # Set save path
    if args.save_path is None:
        args.save_path = f"/opt/FinRL/models/{args.algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Train agent
    model, env = train_agent(
        algorithm=args.algorithm,
        symbols=args.symbols,
        train_start=args.train_start,
        train_end=args.train_end,
        initial_amount=args.capital,
        total_timesteps=args.timesteps,
        learning_rate=args.learning_rate,
        save_path=args.save_path,
    )

    # Evaluate on test data
    rl_metrics = evaluate_agent(
        model=model,
        symbols=args.symbols,
        test_start=args.test_start,
        test_end=args.test_end,
        initial_amount=args.capital,
    )

    # Calculate buy-and-hold benchmark
    print("\n📊 Calculating Buy-and-Hold benchmark...")
    bh_metrics = calculate_buy_and_hold(
        args.symbols, args.test_start, args.test_end, args.capital
    )

    # Print comparison
    print("\n" + "=" * 70)
    print("📊 FINAL COMPARISON: RL AGENT vs BUY-AND-HOLD")
    print("=" * 70)

    print(f"\n{'Metric':<25} {'RL Agent':>18} {'Buy & Hold':>18}")
    print("-" * 65)
    print(f"{'Total Return':<25} {rl_metrics.get('total_return', 0):>17.2f}% {bh_metrics['total_return']:>17.2f}%")
    print(f"{'Sharpe Ratio':<25} {rl_metrics.get('sharpe_ratio', 0):>18.2f} {bh_metrics['sharpe_ratio']:>18.2f}")
    print(f"{'Max Drawdown':<25} {rl_metrics.get('max_drawdown', 0):>17.2f}% {bh_metrics['max_drawdown']:>17.2f}%")
    print(f"{'Final Value':<25} ${rl_metrics.get('final_value', args.capital):>16,.2f} ${bh_metrics['final_value']:>16,.2f}")
    print(f"{'Total Trades':<25} {rl_metrics.get('total_trades', 0):>18}")

    alpha = rl_metrics.get('total_return', 0) - bh_metrics['total_return']
    print("-" * 65)
    print(f"{'Alpha (Excess Return)':<25} {alpha:>+17.2f}%")

    print("\n" + "=" * 70)
    if alpha > 0:
        print(f"🏆 RL AGENT BEATS BUY-AND-HOLD BY {alpha:.2f}%!")
    else:
        print(f"📉 RL agent underperformed by {abs(alpha):.2f}%")
        print("   Try: more training, different algorithm, or hyperparameter tuning")
    print("=" * 70)


if __name__ == "__main__":
    main()

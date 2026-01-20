"""
FinRL-Compatible Stock Trading Environment

A Gymnasium-compatible environment for training RL agents on stock trading.
Based on FinRL's StockTradingEnv but simplified and compatible with our data.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from typing import Any


class StockTradingEnv(gym.Env):
    """
    A stock trading environment compatible with Stable-Baselines3.

    Observation Space:
        - Current cash balance (normalized)
        - Current stock holdings for each stock
        - Current prices for each stock
        - Technical indicators (MACD, RSI, etc.) for each stock

    Action Space:
        - Continuous actions in [-1, 1] for each stock
        - Positive = buy, Negative = sell
        - Magnitude = fraction of available cash/shares

    Reward:
        - Change in portfolio value (with optional Sharpe ratio shaping)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int = 100,
        initial_amount: float = 100000,
        buy_cost_pct: float = 0.001,
        sell_cost_pct: float = 0.001,
        reward_scaling: float = 1e-4,
        state_space: int = None,
        action_space_dim: int = None,
        tech_indicator_list: list = None,
        turbulence_threshold: float = None,
        make_plots: bool = False,
        print_verbosity: int = 1,
        day: int = 0,
        initial: bool = True,
        previous_state: list = None,
        model_name: str = "",
        mode: str = "train",
        iteration: str = "",
    ):
        """
        Initialize the trading environment.

        Args:
            df: DataFrame with columns ['date', 'tic', 'open', 'high', 'low', 'close', 'volume', ...]
                Plus any technical indicators.
            stock_dim: Number of stocks being traded.
            hmax: Maximum number of shares to trade per action.
            initial_amount: Starting cash.
            buy_cost_pct: Transaction cost for buying.
            sell_cost_pct: Transaction cost for selling.
            reward_scaling: Scale factor for rewards.
            tech_indicator_list: List of technical indicator column names.
        """
        super().__init__()

        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.tech_indicator_list = tech_indicator_list or []
        self.turbulence_threshold = turbulence_threshold
        self.print_verbosity = print_verbosity
        self.mode = mode

        # State space dimension: cash + holdings + prices + indicators
        self.state_space = (
            1 +  # cash
            stock_dim +  # holdings
            stock_dim +  # prices
            stock_dim * len(self.tech_indicator_list)  # technical indicators
        )

        # Action space: one action per stock, continuous in [-1, 1]
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(stock_dim,), dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,), dtype=np.float32
        )

        # Get unique dates
        self.dates = df['date'].unique()
        self.terminal = False

        # Initialize state
        self.day = day
        self.data = self._get_data_for_day(self.day)

        # Portfolio state
        self.cash = initial_amount
        self.holdings = np.zeros(stock_dim)
        self.cost = 0
        self.trades = 0

        # Memory for tracking
        self.asset_memory = [initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.dates[self.day]]

        # For Sharpe calculation
        self.portfolio_returns = []

    def _get_data_for_day(self, day: int) -> pd.DataFrame:
        """Get data for a specific day."""
        return self.df[self.df['date'] == self.dates[day]]

    def _get_state(self) -> np.ndarray:
        """Construct the state vector."""
        state = []

        # Cash (normalized by initial amount)
        state.append(self.cash / self.initial_amount)

        # Holdings (normalized)
        for i in range(self.stock_dim):
            state.append(self.holdings[i] / self.hmax)

        # Current prices
        prices = self.data['close'].values
        for price in prices:
            state.append(price / 1000)  # Normalize prices

        # Technical indicators
        for indicator in self.tech_indicator_list:
            if indicator in self.data.columns:
                values = self.data[indicator].values
                for val in values:
                    state.append(val / 100 if not np.isnan(val) else 0)

        return np.array(state, dtype=np.float32)

    def _get_portfolio_value(self) -> float:
        """Calculate current portfolio value."""
        prices = self.data['close'].values
        stock_value = np.sum(self.holdings * prices)
        return self.cash + stock_value

    def step(self, actions: np.ndarray) -> tuple:
        """
        Execute one time step.

        Args:
            actions: Array of actions in [-1, 1] for each stock.

        Returns:
            observation, reward, terminated, truncated, info
        """
        self.terminal = self.day >= len(self.dates) - 1

        if self.terminal:
            # Episode done
            final_value = self._get_portfolio_value()
            total_return = (final_value - self.initial_amount) / self.initial_amount

            if self.print_verbosity > 0:
                print(f"Episode ended. Final value: ${final_value:,.2f}, Return: {total_return*100:.2f}%")

            return self._get_state(), 0.0, True, False, {
                "final_value": final_value,
                "total_return": total_return,
                "total_trades": self.trades
            }

        # Record previous value
        prev_value = self._get_portfolio_value()

        # Execute trades
        prices = self.data['close'].values
        actions = np.clip(actions, -1, 1)

        for i, action in enumerate(actions):
            price = prices[i]

            if action > 0:  # Buy
                # Calculate shares to buy
                available_cash = self.cash
                max_shares = int(available_cash / (price * (1 + self.buy_cost_pct)))
                shares_to_buy = min(int(action * self.hmax), max_shares)

                if shares_to_buy > 0:
                    cost = shares_to_buy * price * (1 + self.buy_cost_pct)
                    self.cash -= cost
                    self.holdings[i] += shares_to_buy
                    self.cost += shares_to_buy * price * self.buy_cost_pct
                    self.trades += 1

            elif action < 0:  # Sell
                # Calculate shares to sell
                shares_to_sell = min(int(-action * self.hmax), int(self.holdings[i]))

                if shares_to_sell > 0:
                    proceeds = shares_to_sell * price * (1 - self.sell_cost_pct)
                    self.cash += proceeds
                    self.holdings[i] -= shares_to_sell
                    self.cost += shares_to_sell * price * self.sell_cost_pct
                    self.trades += 1

        # Move to next day
        self.day += 1
        self.data = self._get_data_for_day(self.day)

        # Calculate reward (change in portfolio value)
        current_value = self._get_portfolio_value()
        reward = (current_value - prev_value) * self.reward_scaling

        # Track returns for Sharpe
        daily_return = (current_value - prev_value) / prev_value if prev_value > 0 else 0
        self.portfolio_returns.append(daily_return)

        # Memory
        self.asset_memory.append(current_value)
        self.rewards_memory.append(reward)
        self.actions_memory.append(actions)
        self.date_memory.append(self.dates[self.day])

        return self._get_state(), reward, False, False, {}

    def reset(self, seed=None, options=None) -> tuple:
        """Reset the environment."""
        super().reset(seed=seed)

        self.day = 0
        self.data = self._get_data_for_day(self.day)
        self.terminal = False

        self.cash = self.initial_amount
        self.holdings = np.zeros(self.stock_dim)
        self.cost = 0
        self.trades = 0

        self.asset_memory = [self.initial_amount]
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self.dates[self.day]]
        self.portfolio_returns = []

        return self._get_state(), {}

    def render(self, mode="human"):
        """Render the environment."""
        current_value = self._get_portfolio_value()
        print(f"Day {self.day}: Value=${current_value:,.2f}, Cash=${self.cash:,.2f}, Trades={self.trades}")

    def get_portfolio_metrics(self) -> dict:
        """Calculate portfolio performance metrics."""
        returns = np.array(self.portfolio_returns)

        if len(returns) < 2:
            return {}

        total_return = (self.asset_memory[-1] - self.initial_amount) / self.initial_amount

        # Sharpe ratio (annualized, assuming 252 trading days)
        if returns.std() > 0:
            sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std()
        else:
            sharpe = 0

        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        max_dd = abs(drawdowns.min()) if len(drawdowns) > 0 else 0

        return {
            "total_return": total_return * 100,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd * 100,
            "total_trades": self.trades,
            "final_value": self.asset_memory[-1]
        }


def prepare_data_for_env(
    data: dict,
    tech_indicators: list = None
) -> tuple[pd.DataFrame, list]:
    """
    Convert our data format to FinRL-compatible DataFrame.

    Args:
        data: Dict of {symbol: DataFrame} with OHLCV data.
        tech_indicators: List of technical indicators to calculate.

    Returns:
        DataFrame in FinRL format and list of indicator names.
    """
    if tech_indicators is None:
        tech_indicators = ['macd', 'rsi', 'cci', 'adx']

    dfs = []
    for symbol, df in data.items():
        stock_df = df.copy()
        stock_df['tic'] = symbol

        # Ensure column names are lowercase
        stock_df.columns = [c.lower() for c in stock_df.columns]

        # Add date column from index if needed
        if 'date' not in stock_df.columns:
            stock_df = stock_df.reset_index()
            stock_df = stock_df.rename(columns={stock_df.columns[0]: 'date'})

        # Calculate technical indicators
        # MACD
        exp1 = stock_df['close'].ewm(span=12, adjust=False).mean()
        exp2 = stock_df['close'].ewm(span=26, adjust=False).mean()
        stock_df['macd'] = exp1 - exp2

        # RSI
        delta = stock_df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        stock_df['rsi'] = 100 - (100 / (1 + rs))
        stock_df['rsi'] = stock_df['rsi'].fillna(50)

        # CCI (Commodity Channel Index)
        typical_price = (stock_df['high'] + stock_df['low'] + stock_df['close']) / 3
        sma = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        stock_df['cci'] = (typical_price - sma) / (0.015 * mad)
        stock_df['cci'] = stock_df['cci'].fillna(0)

        # ADX (simplified)
        high_diff = stock_df['high'].diff()
        low_diff = stock_df['low'].diff().abs()
        tr = pd.concat([
            stock_df['high'] - stock_df['low'],
            (stock_df['high'] - stock_df['close'].shift()).abs(),
            (stock_df['low'] - stock_df['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean()
        stock_df['adx'] = (atr / stock_df['close'] * 100).fillna(0)

        dfs.append(stock_df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sort_values(['date', 'tic']).reset_index(drop=True)

    # Drop rows with NaN in indicators
    combined = combined.dropna(subset=tech_indicators)

    return combined, tech_indicators

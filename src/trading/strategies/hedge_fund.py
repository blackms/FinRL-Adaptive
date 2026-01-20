"""
Hedge Fund Multi-Factor Strategy

Implements institutional-grade quantitative trading:
- Multi-factor alpha model (momentum, value, quality, low volatility)
- Market-neutral long-short construction
- Risk parity position sizing
- Volatility targeting
- Regime detection
- Transaction cost modeling
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from datetime import datetime


@dataclass
class HedgeFundConfig:
    """Configuration for hedge fund strategy."""
    # Factor weights (must sum to 1.0)
    momentum_weight: float = 0.30
    value_weight: float = 0.25
    quality_weight: float = 0.25
    low_vol_weight: float = 0.20

    # Momentum parameters
    momentum_lookback: int = 60  # 3 months
    momentum_skip: int = 5  # Skip most recent week (mean reversion)

    # Risk management
    target_volatility: float = 0.15  # 15% annual vol target
    max_position_size: float = 0.20  # Max 20% per position
    max_sector_exposure: float = 0.40  # Max 40% in one sector
    max_gross_exposure: float = 2.0  # 200% gross (100% long + 100% short)
    max_net_exposure: float = 0.30  # +/- 30% net market exposure

    # Long-short construction
    long_percentile: float = 0.20  # Top 20% = long
    short_percentile: float = 0.20  # Bottom 20% = short

    # Rebalancing
    rebalance_frequency: int = 21  # Monthly
    turnover_penalty: float = 0.002  # 20bps per trade

    # Transaction costs
    commission: float = 0.001  # 10bps
    slippage: float = 0.0005  # 5bps
    borrow_cost: float = 0.02  # 2% annual for shorts

    # Adaptive exposure (key for beating B&H!)
    adaptive_exposure: bool = False  # Enable regime-adaptive net exposure
    base_net_exposure: float = 0.70  # Default 70% long bias (captures bull markets)
    bear_net_exposure: float = 0.20  # Defensive in bear regime
    trend_lookback: int = 50  # Days to detect regime (200-day MA equivalent)


@dataclass
class AdaptiveHedgeFundConfig:
    """Configuration for adaptive long-biased hedge fund strategy."""
    # Factor weights
    momentum_weight: float = 0.40  # Emphasize momentum
    value_weight: float = 0.20
    quality_weight: float = 0.25
    low_vol_weight: float = 0.15

    # Momentum parameters
    momentum_lookback: int = 60
    momentum_skip: int = 5

    # Risk management
    target_volatility: float = 0.18  # Slightly higher for more exposure
    max_position_size: float = 0.15
    max_gross_exposure: float = 1.5

    # Long-biased construction (NOT market neutral)
    long_percentile: float = 0.40  # Top 40% = long
    short_percentile: float = 0.10  # Only bottom 10% = short

    # Rebalancing
    rebalance_frequency: int = 10  # Bi-weekly for responsiveness

    # Transaction costs
    commission: float = 0.001
    slippage: float = 0.0005
    borrow_cost: float = 0.02

    # Adaptive exposure settings
    adaptive_exposure: bool = True
    base_net_exposure: float = 0.80  # 80% long in normal/bull markets
    bear_net_exposure: float = 0.30  # 30% long in bear markets
    trend_lookback: int = 50  # Regime detection window


class HedgeFundStrategy:
    """
    Institutional-grade multi-factor long-short equity strategy.

    Alpha Model:
    - Momentum: 60-day price momentum, skip last 5 days
    - Value: Earnings yield proxy (inverse of price momentum)
    - Quality: Low volatility + positive trend
    - Low Vol: Inverse of realized volatility

    Portfolio Construction:
    - Long top quintile, short bottom quintile by composite score
    - Risk parity weighting within long/short books
    - Volatility targeting at portfolio level
    - Market neutral (low net exposure)
    """

    def __init__(self, config: HedgeFundConfig = None):
        self.config = config or HedgeFundConfig()
        self.positions: Dict[str, float] = {}
        self.last_rebalance: Optional[datetime] = None

    def calculate_factors(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate factor scores for each stock.

        Returns DataFrame with columns: symbol, momentum, value, quality, low_vol, composite
        """
        factors = []

        for symbol, df in data.items():
            if len(df) < self.config.momentum_lookback + 10:
                continue

            close = df['Close'].values

            # Momentum: price return over lookback, skipping recent days
            start_idx = -(self.config.momentum_lookback + self.config.momentum_skip)
            end_idx = -self.config.momentum_skip if self.config.momentum_skip > 0 else None

            if end_idx:
                momentum = (close[end_idx] - close[start_idx]) / close[start_idx]
            else:
                momentum = (close[-1] - close[start_idx]) / close[start_idx]

            # Value: Contrarian signal (negative of short-term momentum)
            # In real HF, would use P/E, P/B, EV/EBITDA
            short_mom = (close[-1] - close[-21]) / close[-21] if len(close) > 21 else 0
            value = -short_mom  # Buy recent losers

            # Quality: Trend strength + consistency
            returns = pd.Series(close).pct_change().dropna()
            if len(returns) > 20:
                # Positive return ratio
                pos_ratio = (returns > 0).sum() / len(returns)
                # Trend consistency (R-squared of price vs time)
                x = np.arange(len(close[-60:]))
                y = close[-60:]
                correlation = np.corrcoef(x, y)[0, 1] if len(y) > 1 else 0
                quality = pos_ratio * abs(correlation)
            else:
                quality = 0.5

            # Low Volatility: Inverse of realized vol
            if len(returns) > 20:
                vol = returns.std() * np.sqrt(252)
                low_vol = 1 / (vol + 0.01)  # Avoid division by zero
            else:
                low_vol = 1.0

            factors.append({
                'symbol': symbol,
                'momentum': momentum,
                'value': value,
                'quality': quality,
                'low_vol': low_vol,
                'volatility': vol if len(returns) > 20 else 0.3
            })

        factor_df = pd.DataFrame(factors)

        if len(factor_df) == 0:
            return factor_df

        # Z-score normalize each factor
        for col in ['momentum', 'value', 'quality', 'low_vol']:
            mean = factor_df[col].mean()
            std = factor_df[col].std()
            if std > 0:
                factor_df[f'{col}_z'] = (factor_df[col] - mean) / std
            else:
                factor_df[f'{col}_z'] = 0

        # Composite score
        factor_df['composite'] = (
            self.config.momentum_weight * factor_df['momentum_z'] +
            self.config.value_weight * factor_df['value_z'] +
            self.config.quality_weight * factor_df['quality_z'] +
            self.config.low_vol_weight * factor_df['low_vol_z']
        )

        return factor_df

    def detect_regime(self, data: Dict[str, pd.DataFrame]) -> str:
        """
        Detect market regime based on trend signals.

        Returns: 'bull', 'bear', or 'neutral'
        """
        if not hasattr(self.config, 'adaptive_exposure') or not self.config.adaptive_exposure:
            return 'neutral'

        lookback = getattr(self.config, 'trend_lookback', 50)

        # Use equal-weighted average of all stocks as market proxy
        returns_list = []
        for symbol, df in data.items():
            if len(df) >= lookback:
                ret = (df['Close'].iloc[-1] - df['Close'].iloc[-lookback]) / df['Close'].iloc[-lookback]
                returns_list.append(ret)

        if not returns_list:
            return 'neutral'

        avg_return = np.mean(returns_list)

        # Simple regime detection: >5% = bull, <-5% = bear
        if avg_return > 0.05:
            return 'bull'
        elif avg_return < -0.05:
            return 'bear'
        return 'neutral'

    def construct_portfolio(
        self,
        factors: pd.DataFrame,
        current_prices: Dict[str, float],
        regime: str = 'neutral'
    ) -> Dict[str, float]:
        """
        Construct long-short portfolio with risk parity weighting.

        Returns: Dict of {symbol: weight} where weight can be negative (short)
        """
        if len(factors) < 5:
            return {}

        # Rank by composite score
        factors = factors.sort_values('composite', ascending=False)
        n_stocks = len(factors)

        # Adaptive exposure: adjust long/short allocation based on regime
        if hasattr(self.config, 'adaptive_exposure') and self.config.adaptive_exposure:
            if regime == 'bull':
                # Bull market: heavy long bias
                target_net = getattr(self.config, 'base_net_exposure', 0.80)
                long_alloc = (1 + target_net) / 2  # e.g., 0.90 long
                short_alloc = (1 - target_net) / 2  # e.g., 0.10 short
            elif regime == 'bear':
                # Bear market: defensive
                target_net = getattr(self.config, 'bear_net_exposure', 0.30)
                long_alloc = (1 + target_net) / 2  # e.g., 0.65 long
                short_alloc = (1 - target_net) / 2  # e.g., 0.35 short
            else:
                # Neutral: moderate long bias
                long_alloc = 0.70
                short_alloc = 0.30
        else:
            # Market neutral (original behavior)
            long_alloc = 0.50
            short_alloc = 0.50

        # Select long and short candidates
        n_long = max(1, int(n_stocks * self.config.long_percentile))
        n_short = max(1, int(n_stocks * self.config.short_percentile))

        long_stocks = factors.head(n_long)
        short_stocks = factors.tail(n_short)

        # Risk parity weighting: inverse volatility
        def risk_parity_weights(stocks_df: pd.DataFrame) -> Dict[str, float]:
            inv_vol = 1 / (stocks_df['volatility'] + 0.01)
            total_inv_vol = inv_vol.sum()
            weights = {}
            for _, row in stocks_df.iterrows():
                w = (1 / (row['volatility'] + 0.01)) / total_inv_vol
                # Cap individual position
                w = min(w, self.config.max_position_size)
                weights[row['symbol']] = w
            # Renormalize
            total = sum(weights.values())
            if total > 0:
                weights = {k: v/total for k, v in weights.items()}
            return weights

        long_weights = risk_parity_weights(long_stocks)
        short_weights = risk_parity_weights(short_stocks)

        # Combine with adaptive allocation
        portfolio = {}
        for symbol, weight in long_weights.items():
            portfolio[symbol] = weight * long_alloc
        for symbol, weight in short_weights.items():
            portfolio[symbol] = -weight * short_alloc

        # Volatility targeting
        portfolio = self._apply_vol_target(portfolio, factors)

        return portfolio

    def _apply_vol_target(
        self,
        portfolio: Dict[str, float],
        factors: pd.DataFrame
    ) -> Dict[str, float]:
        """Scale portfolio to target volatility."""
        # Estimate portfolio volatility (simplified: weighted sum of individual vols)
        factor_dict = factors.set_index('symbol')['volatility'].to_dict()

        port_var = 0
        for symbol, weight in portfolio.items():
            if symbol in factor_dict:
                vol = factor_dict[symbol]
                port_var += (weight ** 2) * (vol ** 2)

        port_vol = np.sqrt(port_var)

        if port_vol > 0:
            scale = self.config.target_volatility / port_vol
            scale = min(scale, 2.0)  # Cap leverage
            portfolio = {k: v * scale for k, v in portfolio.items()}

        return portfolio

    def calculate_transaction_costs(
        self,
        old_positions: Dict[str, float],
        new_positions: Dict[str, float],
        prices: Dict[str, float],
        capital: float
    ) -> float:
        """Calculate realistic transaction costs."""
        total_cost = 0

        all_symbols = set(old_positions.keys()) | set(new_positions.keys())

        for symbol in all_symbols:
            old_weight = old_positions.get(symbol, 0)
            new_weight = new_positions.get(symbol, 0)
            trade_size = abs(new_weight - old_weight) * capital

            # Commission + slippage
            cost = trade_size * (self.config.commission + self.config.slippage)

            # Borrow cost for shorts (prorated)
            if new_weight < 0:
                cost += abs(new_weight) * capital * self.config.borrow_cost / 252

            total_cost += cost

        return total_cost


def run_hedge_fund_backtest(
    data: Dict[str, pd.DataFrame],
    config: HedgeFundConfig = None,
    initial_capital: float = 100000,
    start_date: str = None,
    end_date: str = None,
) -> Dict:
    """
    Run hedge fund strategy backtest with proper institutional methodology.

    Note: data should include sufficient history BEFORE start_date for factor calculation.
    The strategy will only trade within [start_date, end_date] but uses historical data
    for factor calculation (no look-ahead bias).
    """
    if config is None:
        config = HedgeFundConfig()

    strategy = HedgeFundStrategy(config)
    symbols = list(data.keys())

    # Get ALL common dates (including history for factor calculation)
    all_dates = sorted(set.intersection(*[set(data[s].index) for s in symbols]))

    # Determine trading period (when we actually execute trades)
    trading_dates = all_dates.copy()
    if start_date:
        trading_dates = [d for d in trading_dates if d >= pd.Timestamp(start_date)]
    if end_date:
        trading_dates = [d for d in trading_dates if d <= pd.Timestamp(end_date)]

    if len(trading_dates) == 0:
        return {'error': 'No trading dates in specified range'}

    # Verify we have enough historical data BEFORE the trading period for factor calculation
    warmup_needed = config.momentum_lookback + 30
    first_trade_date = trading_dates[0]
    dates_before_start = [d for d in all_dates if d < first_trade_date]

    if len(dates_before_start) < warmup_needed:
        return {'error': f'Insufficient historical data. Need {warmup_needed} days before {first_trade_date}, have {len(dates_before_start)}'}

    # Initialize
    cash = initial_capital
    positions = {}  # {symbol: shares}
    weights = {}  # {symbol: weight}

    portfolio_values = []
    daily_returns = []
    trade_count = 0
    total_costs = 0

    for i, date in enumerate(trading_dates):
        # Get current prices
        prices = {s: data[s].loc[date, 'Close'] for s in symbols if date in data[s].index}

        # Calculate current portfolio value
        position_value = sum(
            positions.get(s, 0) * prices.get(s, 0) for s in symbols
        )
        current_value = cash + position_value

        # Rebalance check
        should_rebalance = (
            i == 0 or
            i % config.rebalance_frequency == 0
        )

        if should_rebalance:
            # Get historical data for factor calculation
            hist_data = {}
            for s in symbols:
                df = data[s]
                mask = df.index <= date
                hist_data[s] = df[mask].tail(config.momentum_lookback + 30)

            # Calculate factors
            factors = strategy.calculate_factors(hist_data)

            if len(factors) > 0:
                # Detect market regime for adaptive exposure
                regime = strategy.detect_regime(hist_data)

                # Get new target weights
                new_weights = strategy.construct_portfolio(factors, prices, regime)

                # Calculate transaction costs
                costs = strategy.calculate_transaction_costs(
                    weights, new_weights, prices, current_value
                )
                total_costs += costs
                cash -= costs

                # Execute rebalance
                for symbol in set(list(weights.keys()) + list(new_weights.keys())):
                    old_weight = weights.get(symbol, 0)
                    new_weight = new_weights.get(symbol, 0)

                    if symbol not in prices:
                        continue

                    price = prices[symbol]

                    # Calculate target shares
                    target_value = new_weight * current_value
                    current_shares = positions.get(symbol, 0)
                    current_pos_value = current_shares * price

                    trade_value = target_value - current_pos_value
                    trade_shares = trade_value / price

                    if abs(trade_shares) > 0.01:  # Minimum trade
                        positions[symbol] = current_shares + trade_shares
                        cash -= trade_value
                        trade_count += 1

                weights = new_weights

        # Record daily value
        position_value = sum(
            positions.get(s, 0) * prices.get(s, 0) for s in symbols
        )
        daily_value = cash + position_value
        portfolio_values.append({
            'date': date,
            'value': daily_value,
            'cash': cash,
            'gross_exposure': sum(abs(w) for w in weights.values()),
            'net_exposure': sum(weights.values()),
            'n_long': sum(1 for w in weights.values() if w > 0),
            'n_short': sum(1 for w in weights.values() if w < 0),
        })

        if len(portfolio_values) > 1:
            prev_value = portfolio_values[-2]['value']
            daily_ret = (daily_value - prev_value) / prev_value
            daily_returns.append(daily_ret)

    # Calculate metrics
    values = pd.Series([p['value'] for p in portfolio_values])
    returns = pd.Series(daily_returns)

    total_return = (values.iloc[-1] - initial_capital) / initial_capital * 100

    if len(returns) > 0 and returns.std() > 0:
        sharpe = np.sqrt(252) * (returns.mean() - 0.05/252) / returns.std()
        sortino_denom = returns[returns < 0].std()
        sortino = np.sqrt(252) * (returns.mean() - 0.05/252) / sortino_denom if sortino_denom > 0 else 0
        volatility = returns.std() * np.sqrt(252) * 100
    else:
        sharpe = sortino = volatility = 0

    # Max drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdowns = (cumulative - running_max) / running_max
    max_dd = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

    # Calmar ratio
    calmar = total_return / max_dd if max_dd > 0 else 0

    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_dd,
        'volatility': volatility,
        'calmar_ratio': calmar,
        'total_trades': trade_count,
        'total_costs': total_costs,
        'final_value': values.iloc[-1],
        'avg_gross_exposure': np.mean([p['gross_exposure'] for p in portfolio_values]),
        'avg_net_exposure': np.mean([p['net_exposure'] for p in portfolio_values]),
        'portfolio_values': portfolio_values,
        'daily_returns': daily_returns,
    }

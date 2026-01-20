"""
Backtest Engine Module

Provides backtesting functionality for running historical simulations
and generating performance reports.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from ..portfolio.portfolio import Portfolio, Order, OrderSide, OrderType, Trade
from ..risk.manager import RiskManager, RiskConfig
from ..strategies.base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """
    Backtest configuration.

    Attributes:
        initial_capital: Starting capital.
        commission_rate: Commission rate per trade.
        slippage_rate: Slippage rate per trade.
        margin_rate: Margin requirement for short positions.
        enable_shorting: Whether short selling is allowed.
        enable_fractional: Whether fractional shares are allowed.
        data_frequency: Data frequency ('1d', '1h', '5m', etc.).
        rebalance_frequency: Rebalancing frequency in days.
        warmup_period: Number of periods to warm up indicators.
        max_positions: Maximum number of concurrent positions.
        stop_on_max_drawdown: Stop backtest on max drawdown.
        max_drawdown_threshold: Maximum drawdown threshold.
    """

    initial_capital: float = 100000.0
    commission_rate: float = 0.001
    slippage_rate: float = 0.0005
    margin_rate: float = 0.5
    enable_shorting: bool = False
    enable_fractional: bool = True
    data_frequency: str = "1d"
    rebalance_frequency: int = 1
    warmup_period: int = 50
    max_positions: int = 20
    stop_on_max_drawdown: bool = False
    max_drawdown_threshold: float = 0.25


@dataclass
class BacktestResult:
    """
    Backtest results.

    Attributes:
        config: Backtest configuration used.
        start_date: Backtest start date.
        end_date: Backtest end date.
        initial_capital: Starting capital.
        final_value: Final portfolio value.
        total_return: Total return percentage.
        annualized_return: Annualized return percentage.
        volatility: Annualized volatility percentage.
        sharpe_ratio: Sharpe ratio.
        sortino_ratio: Sortino ratio.
        max_drawdown: Maximum drawdown percentage.
        calmar_ratio: Calmar ratio (return / max drawdown).
        total_trades: Total number of trades.
        winning_trades: Number of winning trades.
        losing_trades: Number of losing trades.
        win_rate: Win rate percentage.
        profit_factor: Profit factor.
        avg_trade_return: Average trade return.
        avg_win: Average winning trade.
        avg_loss: Average losing trade.
        max_consecutive_wins: Maximum consecutive wins.
        max_consecutive_losses: Maximum consecutive losses.
        equity_curve: DataFrame with equity curve data.
        trades: List of all trades.
        positions_history: History of positions over time.
        signals_history: History of all signals generated.
    """

    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_value: float
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    trades: list[Trade] = field(default_factory=list)
    positions_history: list[dict[str, Any]] = field(default_factory=list)
    signals_history: list[Signal] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_value": self.final_value,
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "total_trades": self.total_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
        }

    def generate_report(self) -> str:
        """Generate a text report of backtest results."""
        lines = [
            "=" * 60,
            "BACKTEST RESULTS",
            "=" * 60,
            f"Period: {self.start_date.strftime('%Y-%m-%d')} to {self.end_date.strftime('%Y-%m-%d')}",
            "",
            "PERFORMANCE METRICS",
            "-" * 40,
            f"Initial Capital:     ${self.initial_capital:,.2f}",
            f"Final Value:         ${self.final_value:,.2f}",
            f"Total Return:        {self.total_return:.2f}%",
            f"Annualized Return:   {self.annualized_return:.2f}%",
            f"Volatility:          {self.volatility:.2f}%",
            f"Sharpe Ratio:        {self.sharpe_ratio:.2f}",
            f"Sortino Ratio:       {self.sortino_ratio:.2f}",
            f"Max Drawdown:        {self.max_drawdown:.2f}%",
            f"Calmar Ratio:        {self.calmar_ratio:.2f}",
            "",
            "TRADE STATISTICS",
            "-" * 40,
            f"Total Trades:        {self.total_trades}",
            f"Winning Trades:      {self.winning_trades}",
            f"Losing Trades:       {self.losing_trades}",
            f"Win Rate:            {self.win_rate:.2f}%",
            f"Profit Factor:       {self.profit_factor:.2f}",
            f"Avg Trade Return:    {self.avg_trade_return:.2f}%",
            f"Avg Win:             ${self.avg_win:,.2f}",
            f"Avg Loss:            ${self.avg_loss:,.2f}",
            f"Max Consecutive Wins:   {self.max_consecutive_wins}",
            f"Max Consecutive Losses: {self.max_consecutive_losses}",
            "=" * 60,
        ]
        return "\n".join(lines)


class BacktestEngine:
    """
    Backtesting engine for historical simulations.

    Runs strategies against historical data and calculates performance metrics.

    Example:
        >>> engine = BacktestEngine(config)
        >>> engine.add_strategy(MomentumStrategy())
        >>> result = engine.run(data)
        >>> print(result.generate_report())
    """

    def __init__(
        self,
        config: BacktestConfig | None = None,
        risk_config: RiskConfig | None = None,
    ) -> None:
        """
        Initialize the backtest engine.

        Args:
            config: Backtest configuration.
            risk_config: Risk management configuration.
        """
        self.config = config or BacktestConfig()
        self.risk_config = risk_config or RiskConfig()

        self._portfolio: Portfolio | None = None
        self._risk_manager: RiskManager | None = None
        self._strategies: list[BaseStrategy] = []
        self._data: dict[str, pd.DataFrame] = {}
        self._current_date: datetime | None = None
        self._equity_curve: list[dict[str, Any]] = []
        self._signals: list[Signal] = []
        self._positions_history: list[dict[str, Any]] = []

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """
        Add a strategy to the backtest.

        Args:
            strategy: Trading strategy to add.
        """
        self._strategies.append(strategy)

    def set_data(self, data: dict[str, pd.DataFrame]) -> None:
        """
        Set the historical data for backtesting.

        Args:
            data: Dictionary mapping symbols to DataFrames with OHLCV data.
                  Each DataFrame should have columns: datetime, open, high, low, close, volume.
        """
        self._data = data

    def _initialize(self) -> None:
        """Initialize portfolio and risk manager."""
        self._portfolio = Portfolio(
            initial_cash=self.config.initial_capital,
            commission_rate=self.config.commission_rate,
        )
        self._risk_manager = RiskManager(self.risk_config)
        self._equity_curve.clear()
        self._signals.clear()
        self._positions_history.clear()

    def _get_price_with_slippage(
        self,
        price: float,
        is_buy: bool,
    ) -> float:
        """
        Apply slippage to a price.

        Args:
            price: Original price.
            is_buy: Whether this is a buy order.

        Returns:
            Price with slippage applied.
        """
        slippage = price * self.config.slippage_rate
        if is_buy:
            return price + slippage
        else:
            return price - slippage

    def _get_data_for_date(
        self,
        symbol: str,
        end_date: datetime,
        lookback: int | None = None,
    ) -> pd.DataFrame:
        """
        Get historical data up to a specific date.

        Args:
            symbol: Ticker symbol.
            end_date: End date (exclusive).
            lookback: Number of periods to look back.

        Returns:
            DataFrame with historical data.
        """
        if symbol not in self._data:
            return pd.DataFrame()

        df = self._data[symbol].copy()

        # Normalize datetime column
        if "datetime" not in df.columns:
            if "date" in df.columns:
                df = df.rename(columns={"date": "datetime"})
            elif df.index.name in ["date", "datetime", "Date", "Datetime"]:
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "datetime"})

        # Filter to dates before end_date
        df["datetime"] = pd.to_datetime(df["datetime"])
        mask = df["datetime"] < pd.Timestamp(end_date)
        df = df[mask]

        if lookback:
            df = df.tail(lookback)

        return df

    def _process_signals(
        self,
        date: datetime,
        prices: dict[str, float],
    ) -> list[Signal]:
        """
        Generate signals from all strategies for the current date.

        Args:
            date: Current date.
            prices: Current prices for all symbols.

        Returns:
            List of generated signals.
        """
        all_signals: list[Signal] = []

        for symbol in self._data.keys():
            if symbol not in prices:
                continue

            # Get historical data for the symbol
            lookback = max(
                self.config.warmup_period,
                max(s.config.lookback_period for s in self._strategies) * 2,
            )
            data = self._get_data_for_date(symbol, date, lookback)

            if data.empty or len(data) < self.config.warmup_period:
                continue

            # Add symbol column
            data["symbol"] = symbol

            # Generate signals from each strategy
            for strategy in self._strategies:
                try:
                    signals = strategy.generate_signals(data)
                    all_signals.extend(signals)
                except Exception as e:
                    logger.warning(f"Error generating signals for {symbol}: {e}")

        return all_signals

    def _execute_signals(
        self,
        signals: list[Signal],
        prices: dict[str, float],
        date: datetime,
    ) -> None:
        """
        Execute trading signals.

        Args:
            signals: List of signals to execute.
            prices: Current prices.
            date: Current date.
        """
        if not self._portfolio or not self._risk_manager:
            return

        for signal in signals:
            symbol = signal.symbol

            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Get ATR for position sizing (simplified)
            data = self._get_data_for_date(symbol, date, 20)
            atr = None
            if len(data) >= 14:
                high = data["high"] if "high" in data.columns else data["High"]
                low = data["low"] if "low" in data.columns else data["Low"]
                close = data["close"] if "close" in data.columns else data["Close"]
                tr = pd.concat([
                    high - low,
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1)),
                ], axis=1).max(axis=1)
                atr = tr.tail(14).mean()

            # Calculate position size
            position_size = self._risk_manager.calculate_position_size(
                signal_strength=signal.strength,
                current_price=current_price,
                capital=self._portfolio.cash,
                atr=atr,
            )

            # Apply position limits
            if not self.config.enable_fractional:
                position_size = int(position_size)

            if position_size == 0:
                continue

            # Check if we already have a position
            existing_position = self._portfolio.get_position(symbol)

            # Handle signal based on type
            if signal.is_buy:
                if existing_position and existing_position.is_short:
                    # Close short position first
                    close_price = self._get_price_with_slippage(current_price, True)
                    self._portfolio.close_position(symbol, close_price, date)
                    existing_position = None

                if not existing_position:
                    # Check position limits
                    if len(self._portfolio.positions) >= self.config.max_positions:
                        continue

                    # Check exposure limits
                    position_value = abs(position_size) * current_price
                    allowed, reason = self._risk_manager.check_exposure_limits(
                        proposed_position_value=position_value,
                        current_exposure=self._portfolio.positions_value,
                        portfolio_value=self._portfolio.total_value,
                    )

                    if not allowed:
                        logger.debug(f"Position rejected: {reason}")
                        continue

                    # Execute buy order
                    fill_price = self._get_price_with_slippage(current_price, True)
                    order = self._portfolio.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=abs(position_size),
                        order_type=OrderType.MARKET,
                    )
                    self._portfolio.execute_order(order, fill_price, timestamp=date)

            elif signal.is_sell:
                if existing_position and existing_position.is_long:
                    # Close long position
                    close_price = self._get_price_with_slippage(current_price, False)
                    self._portfolio.close_position(symbol, close_price, date)

                elif self.config.enable_shorting and not existing_position:
                    # Open short position
                    if len(self._portfolio.positions) >= self.config.max_positions:
                        continue

                    fill_price = self._get_price_with_slippage(current_price, False)
                    order = self._portfolio.create_order(
                        symbol=symbol,
                        side=OrderSide.SELL,
                        quantity=abs(position_size),
                        order_type=OrderType.MARKET,
                    )
                    self._portfolio.execute_order(order, fill_price, timestamp=date)

    def _update_stop_losses(
        self,
        prices: dict[str, float],
        date: datetime,
    ) -> None:
        """
        Check and execute stop losses.

        Args:
            prices: Current prices.
            date: Current date.
        """
        if not self._portfolio or not self._risk_manager:
            return

        positions_to_close: list[str] = []

        for symbol, position in self._portfolio.positions.items():
            if symbol not in prices:
                continue

            current_price = prices[symbol]

            # Calculate stop loss
            stop_loss = self._risk_manager.calculate_stop_loss(
                entry_price=position.avg_entry_price,
                is_long=position.is_long,
            )

            # Check if stop loss triggered
            if position.is_long and current_price <= stop_loss:
                positions_to_close.append(symbol)
            elif position.is_short and current_price >= stop_loss:
                positions_to_close.append(symbol)

        # Close positions that hit stop loss
        for symbol in positions_to_close:
            if symbol in prices:
                is_long = self._portfolio.positions[symbol].is_long
                close_price = self._get_price_with_slippage(prices[symbol], not is_long)
                self._portfolio.close_position(symbol, close_price, date)
                logger.debug(f"Stop loss triggered for {symbol} at {close_price}")

    def _record_state(self, date: datetime, prices: dict[str, float]) -> None:
        """Record current state for analysis."""
        if not self._portfolio:
            return

        # Update prices
        self._portfolio.update_prices(prices)

        # Take snapshot
        snapshot = self._portfolio.take_snapshot(date)

        self._equity_curve.append({
            "date": date,
            "total_value": snapshot.total_value,
            "cash": snapshot.cash,
            "positions_value": snapshot.positions_value,
        })

        self._positions_history.append({
            "date": date,
            "positions": snapshot.positions.copy(),
        })

    def run(
        self,
        data: dict[str, pd.DataFrame] | None = None,
        start_date: datetime | str | None = None,
        end_date: datetime | str | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            data: Historical data (overrides previously set data).
            start_date: Start date for backtest.
            end_date: End date for backtest.
            progress_callback: Optional callback for progress updates.

        Returns:
            BacktestResult with performance metrics.
        """
        if data:
            self.set_data(data)

        if not self._data:
            raise ValueError("No data provided for backtest")

        if not self._strategies:
            raise ValueError("No strategies added to backtest")

        # Initialize
        self._initialize()

        # Get date range
        all_dates: set[datetime] = set()
        for symbol, df in self._data.items():
            if "datetime" in df.columns:
                dates = pd.to_datetime(df["datetime"])
            elif "date" in df.columns:
                dates = pd.to_datetime(df["date"])
            else:
                dates = pd.to_datetime(df.index)
            all_dates.update(dates.tolist())

        sorted_dates = sorted(all_dates)

        # Apply date filters
        if start_date:
            start = pd.Timestamp(start_date)
            sorted_dates = [d for d in sorted_dates if pd.Timestamp(d) >= start]

        if end_date:
            end = pd.Timestamp(end_date)
            sorted_dates = [d for d in sorted_dates if pd.Timestamp(d) <= end]

        # Skip warmup period
        sorted_dates = sorted_dates[self.config.warmup_period:]

        if len(sorted_dates) < 2:
            raise ValueError("Insufficient data for backtest")

        total_days = len(sorted_dates)
        logger.info(f"Running backtest from {sorted_dates[0]} to {sorted_dates[-1]} ({total_days} days)")

        # Main backtest loop
        for i, date in enumerate(sorted_dates):
            self._current_date = date

            # Get current prices
            prices: dict[str, float] = {}
            for symbol, df in self._data.items():
                data_df = self._get_data_for_date(symbol, date + timedelta(days=1), 1)
                if not data_df.empty:
                    close_col = "close" if "close" in data_df.columns else "Close"
                    prices[symbol] = float(data_df[close_col].iloc[-1])

            # Update stop losses
            self._update_stop_losses(prices, date)

            # Generate and execute signals
            signals = self._process_signals(date, prices)
            self._signals.extend(signals)
            self._execute_signals(signals, prices, date)

            # Record state
            self._record_state(date, prices)

            # Check max drawdown
            if self.config.stop_on_max_drawdown and self._portfolio:
                peak = max(e["total_value"] for e in self._equity_curve)
                current = self._portfolio.total_value
                drawdown = (peak - current) / peak
                if drawdown > self.config.max_drawdown_threshold:
                    logger.warning(f"Max drawdown threshold reached: {drawdown:.2%}")
                    break

            # Progress callback
            if progress_callback:
                progress_callback(i + 1, total_days)

        # Calculate results
        return self._calculate_results(sorted_dates[0], sorted_dates[-1])

    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> BacktestResult:
        """Calculate backtest results and metrics."""
        if not self._portfolio:
            raise RuntimeError("Portfolio not initialized")

        # Get portfolio metrics
        metrics = self._portfolio.calculate_metrics()
        trades = self._portfolio.get_trades()

        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self._equity_curve)
        if not equity_df.empty:
            equity_df["returns"] = equity_df["total_value"].pct_change()

        # Calculate additional metrics
        total_return = (self._portfolio.total_value / self.config.initial_capital - 1) * 100

        days = (end_date - start_date).days or 1
        annualized_return = ((1 + total_return / 100) ** (365 / days) - 1) * 100

        # Calculate Calmar ratio
        max_drawdown = metrics.get("max_drawdown", 0)
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Calculate consecutive wins/losses
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, current_losses)

        return BacktestResult(
            config=self.config,
            start_date=start_date,
            end_date=end_date,
            initial_capital=self.config.initial_capital,
            final_value=self._portfolio.total_value,
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=metrics.get("volatility", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            sortino_ratio=metrics.get("sortino_ratio", 0),
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            total_trades=len(trades),
            winning_trades=metrics.get("winning_trades", len([t for t in trades if t.pnl > 0])),
            losing_trades=metrics.get("losing_trades", len([t for t in trades if t.pnl <= 0])),
            win_rate=metrics.get("win_rate", 0),
            profit_factor=metrics.get("profit_factor", 0),
            avg_trade_return=np.mean([t.pnl_pct for t in trades]) if trades else 0,
            avg_win=metrics.get("avg_win", 0),
            avg_loss=metrics.get("avg_loss", 0),
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            equity_curve=equity_df,
            trades=trades,
            positions_history=self._positions_history,
            signals_history=self._signals,
        )

    def optimize(
        self,
        param_grid: dict[str, list[Any]],
        metric: str = "sharpe_ratio",
        n_jobs: int = 1,
    ) -> tuple[dict[str, Any], BacktestResult]:
        """
        Optimize strategy parameters using grid search.

        Args:
            param_grid: Dictionary of parameter names to lists of values.
            metric: Metric to optimize (sharpe_ratio, total_return, etc.).
            n_jobs: Number of parallel jobs (not implemented, placeholder).

        Returns:
            Tuple of (best_params, best_result).
        """
        import itertools

        best_params: dict[str, Any] = {}
        best_result: BacktestResult | None = None
        best_metric_value = float("-inf")

        # Generate all combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(itertools.product(*param_values))

        logger.info(f"Running {len(combinations)} parameter combinations")

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Apply parameters to strategies
            for strategy in self._strategies:
                for name, value in params.items():
                    if hasattr(strategy.config, name):
                        setattr(strategy.config, name, value)

            try:
                result = self.run()
                metric_value = getattr(result, metric, 0)

                if metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_params = params.copy()
                    best_result = result

                logger.debug(f"Params {params}: {metric}={metric_value:.4f}")

            except Exception as e:
                logger.warning(f"Failed with params {params}: {e}")

        if best_result is None:
            raise RuntimeError("Optimization failed - no valid results")

        logger.info(f"Best params: {best_params}, {metric}={best_metric_value:.4f}")

        return best_params, best_result


__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
]

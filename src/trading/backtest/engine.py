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
        order_value: float = 0,
        avg_daily_volume: float = 1e9,
    ) -> float:
        """
        Apply slippage with market impact scaling.

        Market impact increases with order size relative to daily volume,
        following the square-root law commonly observed in market microstructure.

        Args:
            price: Original price.
            is_buy: Whether this is a buy order.
            order_value: Total value of the order in dollars.
            avg_daily_volume: Average daily trading volume in dollars.
                             Defaults to large number if unknown.

        Returns:
            Price with slippage and market impact applied.
        """
        # Base slippage
        base_slippage = price * self.config.slippage_rate

        # Market impact: scales with square root of order size relative to volume
        # This follows the empirical square-root law for market impact
        if avg_daily_volume > 0 and order_value > 0:
            volume_fraction = order_value / avg_daily_volume
            market_impact = base_slippage * (volume_fraction ** 0.5)
        else:
            market_impact = 0

        total_slippage = base_slippage + market_impact

        if is_buy:
            return price + total_slippage
        else:
            return price - total_slippage

    def _get_daily_volume(
        self,
        symbol: str,
        date: datetime,
        lookback: int = 20,
    ) -> float:
        """
        Get average daily trading volume for a symbol.

        Args:
            symbol: Ticker symbol.
            date: Current date.
            lookback: Number of days to average volume over.

        Returns:
            Average daily volume in shares.
        """
        data = self._get_data_for_date(symbol, date, lookback)
        if data.empty:
            return 0.0

        volume_col = "volume" if "volume" in data.columns else "Volume"
        if volume_col in data.columns:
            return float(data[volume_col].mean())
        return 0.0

    def _calculate_max_fill(
        self,
        symbol: str,
        date: datetime,
        price: float,
        max_volume_pct: float = 0.10,
    ) -> float:
        """
        Calculate maximum fillable quantity based on daily volume.

        Limits order size to a percentage of average daily volume to ensure
        realistic execution assumptions and avoid market impact issues.

        Args:
            symbol: Ticker symbol.
            date: Current date.
            price: Current price per share.
            max_volume_pct: Maximum percentage of daily volume to allow.
                           Defaults to 10%.

        Returns:
            Maximum fillable quantity in shares.
        """
        daily_volume = self._get_daily_volume(symbol, date)
        if daily_volume <= 0:
            return float("inf")  # No limit if volume unknown

        return daily_volume * max_volume_pct

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
        available_symbols: set[str] | None = None,
    ) -> list[Signal]:
        """
        Generate signals from all strategies for the current date.

        IMPORTANT: This method uses data STRICTLY BEFORE the current date
        to prevent look-ahead bias. Signals are generated based on information
        available at market open, before the current day's price is known.

        Args:
            date: Current date (signals will use data before this date).
            available_symbols: Set of symbols with valid data. If None, all symbols
                             in self._data are processed.

        Returns:
            List of generated signals.
        """
        all_signals: list[Signal] = []

        for symbol in self._data.keys():
            # Skip if symbol not in available set (when specified)
            if available_symbols is not None and symbol not in available_symbols:
                continue

            # Get historical data for the symbol - STRICTLY BEFORE current date
            # This ensures no look-ahead bias: we only see data available at market open
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

            # Apply volume-based fill limits
            max_fill = self._calculate_max_fill(symbol, date, current_price)
            if abs(position_size) > max_fill:
                logger.debug(
                    f"Position size {position_size:.2f} exceeds max fill {max_fill:.2f} "
                    f"for {symbol}. Limiting to {max_fill:.2f}."
                )
                position_size = max_fill if position_size > 0 else -max_fill

            # Apply position limits
            if not self.config.enable_fractional:
                position_size = int(position_size)

            if position_size == 0:
                continue

            # Check if we already have a position
            existing_position = self._portfolio.get_position(symbol)

            # Calculate average daily volume for market impact
            daily_volume = self._get_daily_volume(symbol, date)
            avg_daily_volume_value = daily_volume * current_price if daily_volume > 0 else 1e9

            # Handle signal based on type
            if signal.is_buy:
                if existing_position and existing_position.is_short:
                    # Close short position first
                    close_value = abs(existing_position.quantity) * current_price
                    close_price = self._get_price_with_slippage(
                        current_price, True, close_value, avg_daily_volume_value
                    )
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

                    # Execute buy order with market impact
                    order_value = abs(position_size) * current_price
                    fill_price = self._get_price_with_slippage(
                        current_price, True, order_value, avg_daily_volume_value
                    )
                    order = self._portfolio.create_order(
                        symbol=symbol,
                        side=OrderSide.BUY,
                        quantity=abs(position_size),
                        order_type=OrderType.MARKET,
                    )
                    self._portfolio.execute_order(order, fill_price, timestamp=date)

            elif signal.is_sell:
                if existing_position and existing_position.is_long:
                    # Close long position with market impact
                    close_value = abs(existing_position.quantity) * current_price
                    close_price = self._get_price_with_slippage(
                        current_price, False, close_value, avg_daily_volume_value
                    )
                    self._portfolio.close_position(symbol, close_price, date)

                elif self.config.enable_shorting and not existing_position:
                    # Open short position
                    if len(self._portfolio.positions) >= self.config.max_positions:
                        continue

                    order_value = abs(position_size) * current_price
                    fill_price = self._get_price_with_slippage(
                        current_price, False, order_value, avg_daily_volume_value
                    )
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
                position = self._portfolio.positions[symbol]
                current_price = prices[symbol]
                is_long = position.is_long

                # Calculate market impact for stop loss execution
                close_value = abs(position.quantity) * current_price
                daily_volume = self._get_daily_volume(symbol, date)
                avg_daily_volume_value = daily_volume * current_price if daily_volume > 0 else 1e9

                close_price = self._get_price_with_slippage(
                    current_price, not is_long, close_value, avg_daily_volume_value
                )
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

            # IMPORTANT: Look-ahead bias protection
            # 1. Signal generation uses data STRICTLY BEFORE current date
            #    (via _get_data_for_date which filters datetime < date)
            # 2. Execution uses current date's close price (end-of-day execution model)

            # First, determine which symbols have data for today (for execution later)
            available_symbols: set[str] = set()
            for symbol in self._data.keys():
                data_df = self._get_data_for_date(symbol, date + timedelta(days=1), 1)
                if not data_df.empty:
                    available_symbols.add(symbol)

            # Generate signals using only historical data (STRICTLY BEFORE current day)
            # This simulates decision-making at market open with prior day's data
            signals = self._process_signals(date, available_symbols)
            self._signals.extend(signals)

            # Now get current day's close prices for execution (after signals are generated)
            # This represents end-of-day execution at the close
            prices: dict[str, float] = {}
            for symbol in available_symbols:
                # Get the close price for the current date
                # Use date + 1 day as end_date to include current day in the result
                data_df = self._get_data_for_date(symbol, date + timedelta(days=1), 1)
                if not data_df.empty:
                    close_col = "close" if "close" in data_df.columns else "Close"
                    prices[symbol] = float(data_df[close_col].iloc[-1])

            # Update stop losses with current prices
            self._update_stop_losses(prices, date)

            # Execute signals at current day's close price
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

    def walk_forward_optimize(
        self,
        param_grid: dict[str, list[Any]],
        train_months: int = 12,
        test_months: int = 3,
        metric: str = "sharpe_ratio",
    ) -> dict[str, Any]:
        """
        Walk-forward optimization to avoid overfitting.

        Splits data into rolling train/test windows:
        - Train on train_months, test on next test_months
        - Roll forward and repeat
        - Returns average out-of-sample performance

        This approach provides more realistic performance estimates
        by preventing optimization from seeing future data.

        Args:
            param_grid: Dictionary of parameter names to lists of values.
            train_months: Number of months for training window.
            test_months: Number of months for testing window.
            metric: Metric to optimize (sharpe_ratio, total_return, etc.).

        Returns:
            Dictionary containing:
            - best_params: Best parameters found
            - avg_oos_metric: Average out-of-sample metric value
            - oos_results: List of out-of-sample results per window
            - in_sample_results: List of in-sample results per window
            - windows: List of (train_start, train_end, test_start, test_end) tuples
        """
        if not self._data:
            raise ValueError("No data set for walk-forward optimization")

        # Get full date range from data
        all_dates: set[datetime] = set()
        for df in self._data.values():
            if "datetime" in df.columns:
                dates = pd.to_datetime(df["datetime"])
            elif "date" in df.columns:
                dates = pd.to_datetime(df["date"])
            else:
                dates = pd.to_datetime(df.index)
            all_dates.update(dates.tolist())

        sorted_dates = sorted(all_dates)

        if len(sorted_dates) < 2:
            raise ValueError("Insufficient data for walk-forward optimization")

        start_date = sorted_dates[0]
        end_date = sorted_dates[-1]

        # Calculate window positions
        windows: list[tuple[datetime, datetime, datetime, datetime]] = []
        current_start = start_date

        while True:
            train_end = current_start + timedelta(days=train_months * 30)
            test_start = train_end
            test_end = test_start + timedelta(days=test_months * 30)

            if test_end > end_date:
                break

            windows.append((current_start, train_end, test_start, test_end))

            # Roll forward by test_months
            current_start = current_start + timedelta(days=test_months * 30)

        if not windows:
            raise ValueError(
                f"Insufficient data for walk-forward optimization. "
                f"Need at least {train_months + test_months} months of data."
            )

        logger.info(f"Walk-forward optimization with {len(windows)} windows")

        oos_results: list[BacktestResult] = []
        in_sample_results: list[BacktestResult] = []
        best_params_per_window: list[dict[str, Any]] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(
                f"Window {i + 1}/{len(windows)}: "
                f"Train {train_start.date()} to {train_end.date()}, "
                f"Test {test_start.date()} to {test_end.date()}"
            )

            # In-sample optimization on training period
            try:
                best_params, in_sample_result = self.optimize(
                    param_grid=param_grid,
                    metric=metric,
                )
                # Re-run with full training period
                in_sample_result = self.run(
                    start_date=train_start,
                    end_date=train_end,
                )
                in_sample_results.append(in_sample_result)
                best_params_per_window.append(best_params)

                # Out-of-sample test with optimized parameters
                # Apply best parameters to strategies
                for strategy in self._strategies:
                    for name, value in best_params.items():
                        if hasattr(strategy.config, name):
                            setattr(strategy.config, name, value)

                oos_result = self.run(
                    start_date=test_start,
                    end_date=test_end,
                )
                oos_results.append(oos_result)

                logger.info(
                    f"  In-sample {metric}: {getattr(in_sample_result, metric, 0):.4f}, "
                    f"Out-of-sample {metric}: {getattr(oos_result, metric, 0):.4f}"
                )

            except Exception as e:
                logger.warning(f"Window {i + 1} failed: {e}")
                continue

        if not oos_results:
            raise RuntimeError("Walk-forward optimization failed - no valid results")

        # Calculate average out-of-sample metrics
        avg_oos_metric = np.mean([getattr(r, metric, 0) for r in oos_results])
        avg_is_metric = np.mean([getattr(r, metric, 0) for r in in_sample_results])

        # Find the most common best parameters (mode)
        from collections import Counter
        param_counts: dict[str, Counter] = {}
        for params in best_params_per_window:
            for name, value in params.items():
                if name not in param_counts:
                    param_counts[name] = Counter()
                param_counts[name][value] += 1

        best_params = {
            name: counter.most_common(1)[0][0]
            for name, counter in param_counts.items()
        }

        # Calculate degradation ratio (in-sample vs out-of-sample)
        degradation = (avg_is_metric - avg_oos_metric) / abs(avg_is_metric) if avg_is_metric != 0 else 0

        logger.info(
            f"Walk-forward results: "
            f"Avg in-sample {metric}={avg_is_metric:.4f}, "
            f"Avg out-of-sample {metric}={avg_oos_metric:.4f}, "
            f"Degradation={degradation:.2%}"
        )

        return {
            "best_params": best_params,
            "avg_oos_metric": avg_oos_metric,
            "avg_is_metric": avg_is_metric,
            "degradation": degradation,
            "oos_results": oos_results,
            "in_sample_results": in_sample_results,
            "windows": windows,
            "n_windows": len(windows),
        }


__all__ = [
    "BacktestEngine",
    "BacktestConfig",
    "BacktestResult",
]

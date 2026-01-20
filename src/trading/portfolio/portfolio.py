"""
Portfolio Module

Provides portfolio management functionality including position tracking,
P&L calculation, and performance metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterator

import numpy as np
import pandas as pd


class OrderType(Enum):
    """Order types."""

    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""

    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Order:
    """
    Trading order.

    Attributes:
        order_id: Unique order identifier.
        symbol: Ticker symbol.
        side: Order side (buy/sell).
        order_type: Order type.
        quantity: Number of shares.
        price: Order price (limit/stop price).
        timestamp: Order creation time.
        status: Order status.
        filled_quantity: Quantity filled.
        filled_price: Average fill price.
        filled_timestamp: Time of fill.
        commission: Commission paid.
    """

    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float | None = None
    filled_timestamp: datetime | None = None
    commission: float = 0.0

    @property
    def is_buy(self) -> bool:
        """Check if order is a buy."""
        return self.side == OrderSide.BUY

    @property
    def is_filled(self) -> bool:
        """Check if order is fully filled."""
        return self.status == OrderStatus.FILLED

    @property
    def remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity


@dataclass
class PortfolioPosition:
    """
    Portfolio position.

    Attributes:
        symbol: Ticker symbol.
        quantity: Number of shares (positive=long, negative=short).
        avg_entry_price: Average entry price.
        current_price: Current market price.
        entry_time: Time of first entry.
        last_update: Time of last update.
        sector: Stock sector.
        realized_pnl: Realized P&L from partial closes.
    """

    symbol: str
    quantity: float
    avg_entry_price: float
    current_price: float
    entry_time: datetime
    last_update: datetime = field(default_factory=datetime.now)
    sector: str | None = None
    realized_pnl: float = 0.0

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def market_value(self) -> float:
        """Calculate current market value."""
        return abs(self.quantity) * self.current_price

    @property
    def cost_basis(self) -> float:
        """Calculate cost basis."""
        return abs(self.quantity) * self.avg_entry_price

    @property
    def unrealized_pnl(self) -> float:
        """Calculate unrealized P&L."""
        price_diff = self.current_price - self.avg_entry_price
        return self.quantity * price_diff

    @property
    def unrealized_pnl_pct(self) -> float:
        """Calculate unrealized P&L percentage."""
        if self.avg_entry_price == 0:
            return 0.0
        return (self.current_price / self.avg_entry_price - 1) * (1 if self.is_long else -1) * 100

    @property
    def total_pnl(self) -> float:
        """Calculate total P&L (realized + unrealized)."""
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class Trade:
    """
    Completed trade record.

    Attributes:
        trade_id: Unique trade identifier.
        symbol: Ticker symbol.
        entry_price: Entry price.
        exit_price: Exit price.
        quantity: Number of shares.
        entry_time: Entry time.
        exit_time: Exit time.
        pnl: Profit/loss.
        pnl_pct: P&L percentage.
        commission: Total commission.
    """

    trade_id: str
    symbol: str
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0

    def __post_init__(self) -> None:
        """Calculate P&L if not provided."""
        if self.pnl == 0.0:
            is_long = self.quantity > 0
            price_diff = self.exit_price - self.entry_price
            self.pnl = price_diff * self.quantity - self.commission

            if self.entry_price > 0:
                self.pnl_pct = (price_diff / self.entry_price) * (1 if is_long else -1) * 100


@dataclass
class PortfolioSnapshot:
    """
    Portfolio state at a point in time.

    Attributes:
        timestamp: Snapshot time.
        cash: Cash balance.
        positions_value: Total value of positions.
        total_value: Total portfolio value.
        positions: Position details.
        daily_pnl: Daily P&L.
        daily_pnl_pct: Daily P&L percentage.
    """

    timestamp: datetime
    cash: float
    positions_value: float
    total_value: float
    positions: dict[str, dict[str, Any]] = field(default_factory=dict)
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0


class Portfolio:
    """
    Portfolio management system.

    Tracks positions, calculates P&L, and provides performance metrics.

    Example:
        >>> portfolio = Portfolio(initial_cash=100000)
        >>> portfolio.execute_order(Order(
        ...     order_id="1",
        ...     symbol="AAPL",
        ...     side=OrderSide.BUY,
        ...     order_type=OrderType.MARKET,
        ...     quantity=100,
        ... ), fill_price=150.0)
        >>> print(portfolio.total_value)
    """

    def __init__(
        self,
        initial_cash: float = 100000.0,
        commission_rate: float = 0.0,
        min_commission: float = 0.0,
    ) -> None:
        """
        Initialize the portfolio.

        Args:
            initial_cash: Initial cash balance.
            commission_rate: Commission rate per trade (e.g., 0.001 = 0.1%).
            min_commission: Minimum commission per trade.
        """
        self._cash = initial_cash
        self._initial_cash = initial_cash
        self._commission_rate = commission_rate
        self._min_commission = min_commission

        self._positions: dict[str, PortfolioPosition] = {}
        self._orders: dict[str, Order] = {}
        self._trades: list[Trade] = []
        self._snapshots: list[PortfolioSnapshot] = []

        self._next_order_id = 1
        self._next_trade_id = 1

        # Track peak value for drawdown
        self._peak_value = initial_cash
        self._daily_start_value = initial_cash
        self._last_snapshot_date: datetime | None = None

    @property
    def cash(self) -> float:
        """Get current cash balance."""
        return self._cash

    @property
    def positions(self) -> dict[str, PortfolioPosition]:
        """Get current positions."""
        return self._positions.copy()

    @property
    def positions_value(self) -> float:
        """Get total market value of all positions."""
        return sum(pos.market_value for pos in self._positions.values())

    @property
    def total_value(self) -> float:
        """Get total portfolio value (cash + positions)."""
        return self._cash + self.positions_value

    @property
    def unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self._positions.values())

    @property
    def realized_pnl(self) -> float:
        """Get total realized P&L."""
        return sum(trade.pnl for trade in self._trades)

    @property
    def total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)."""
        return self.unrealized_pnl + self.realized_pnl

    @property
    def total_return(self) -> float:
        """Get total return as percentage."""
        if self._initial_cash == 0:
            return 0.0
        return (self.total_value / self._initial_cash - 1) * 100

    def _generate_order_id(self) -> str:
        """Generate unique order ID."""
        order_id = f"ORD-{self._next_order_id:06d}"
        self._next_order_id += 1
        return order_id

    def _generate_trade_id(self) -> str:
        """Generate unique trade ID."""
        trade_id = f"TRD-{self._next_trade_id:06d}"
        self._next_trade_id += 1
        return trade_id

    def _calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        trade_value = abs(quantity) * price
        commission = trade_value * self._commission_rate
        return max(commission, self._min_commission)

    def create_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
    ) -> Order:
        """
        Create a new order.

        Args:
            symbol: Ticker symbol.
            side: Order side (buy/sell).
            quantity: Number of shares.
            order_type: Order type.
            price: Limit/stop price.

        Returns:
            Created Order object.
        """
        order = Order(
            order_id=self._generate_order_id(),
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
        )
        self._orders[order.order_id] = order
        return order

    def execute_order(
        self,
        order: Order,
        fill_price: float,
        fill_quantity: float | None = None,
        timestamp: datetime | None = None,
    ) -> bool:
        """
        Execute (fill) an order.

        Args:
            order: Order to execute.
            fill_price: Execution price.
            fill_quantity: Quantity to fill (defaults to full order).
            timestamp: Execution time.

        Returns:
            True if order was successfully executed.
        """
        fill_qty = fill_quantity or order.remaining_quantity
        timestamp = timestamp or datetime.now()

        # Calculate commission
        commission = self._calculate_commission(fill_qty, fill_price)

        # Check if we have enough cash for buy orders
        if order.is_buy:
            cost = fill_qty * fill_price + commission
            if cost > self._cash:
                order.status = OrderStatus.REJECTED
                return False
            self._cash -= cost
        else:
            # For sell orders, we receive cash minus commission
            proceeds = fill_qty * fill_price - commission
            self._cash += proceeds

        # Update order
        order.filled_quantity += fill_qty
        order.filled_price = fill_price
        order.filled_timestamp = timestamp
        order.commission += commission

        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
        else:
            order.status = OrderStatus.PARTIALLY_FILLED

        # Update position
        self._update_position(order.symbol, fill_qty if order.is_buy else -fill_qty, fill_price, timestamp)

        return True

    def _update_position(
        self,
        symbol: str,
        quantity_change: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """
        Update position after a trade.

        Args:
            symbol: Ticker symbol.
            quantity_change: Change in quantity (positive=buy, negative=sell).
            price: Trade price.
            timestamp: Trade time.
        """
        if symbol in self._positions:
            position = self._positions[symbol]
            old_quantity = position.quantity
            new_quantity = old_quantity + quantity_change

            if new_quantity == 0:
                # Position closed - record trade
                self._record_closed_trade(position, price, timestamp)
                del self._positions[symbol]
                return

            # Check if we're adding to position or reducing
            if (old_quantity > 0 and quantity_change > 0) or (old_quantity < 0 and quantity_change < 0):
                # Adding to position - update average price
                total_cost = abs(old_quantity) * position.avg_entry_price + abs(quantity_change) * price
                position.avg_entry_price = total_cost / abs(new_quantity)
            elif abs(new_quantity) < abs(old_quantity):
                # Reducing position - realize partial P&L
                closed_qty = abs(quantity_change)
                if position.is_long:
                    realized = (price - position.avg_entry_price) * closed_qty
                else:
                    realized = (position.avg_entry_price - price) * closed_qty
                position.realized_pnl += realized

            position.quantity = new_quantity
            position.current_price = price
            position.last_update = timestamp
        else:
            # New position
            self._positions[symbol] = PortfolioPosition(
                symbol=symbol,
                quantity=quantity_change,
                avg_entry_price=price,
                current_price=price,
                entry_time=timestamp,
                last_update=timestamp,
            )

    def _record_closed_trade(
        self,
        position: PortfolioPosition,
        exit_price: float,
        exit_time: datetime,
    ) -> None:
        """Record a closed trade."""
        trade = Trade(
            trade_id=self._generate_trade_id(),
            symbol=position.symbol,
            entry_price=position.avg_entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=exit_time,
        )
        self._trades.append(trade)

    def update_prices(self, prices: dict[str, float]) -> None:
        """
        Update current prices for all positions.

        Args:
            prices: Dictionary mapping symbols to current prices.
        """
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].current_price = price
                self._positions[symbol].last_update = datetime.now()

        # Update peak value for drawdown calculation
        current_value = self.total_value
        if current_value > self._peak_value:
            self._peak_value = current_value

    def close_position(
        self,
        symbol: str,
        price: float,
        timestamp: datetime | None = None,
    ) -> Trade | None:
        """
        Close an entire position.

        Args:
            symbol: Ticker symbol.
            price: Closing price.
            timestamp: Closing time.

        Returns:
            Trade record or None if no position exists.
        """
        if symbol not in self._positions:
            return None

        position = self._positions[symbol]
        timestamp = timestamp or datetime.now()

        # Calculate commission
        commission = self._calculate_commission(abs(position.quantity), price)

        # Calculate proceeds/cost
        if position.is_long:
            proceeds = abs(position.quantity) * price - commission
            self._cash += proceeds
        else:
            cost = abs(position.quantity) * price + commission
            self._cash -= cost

        # Record trade
        trade = Trade(
            trade_id=self._generate_trade_id(),
            symbol=symbol,
            entry_price=position.avg_entry_price,
            exit_price=price,
            quantity=position.quantity,
            entry_time=position.entry_time,
            exit_time=timestamp,
            commission=commission,
        )
        self._trades.append(trade)

        del self._positions[symbol]
        return trade

    def get_position(self, symbol: str) -> PortfolioPosition | None:
        """Get position for a symbol."""
        return self._positions.get(symbol)

    def has_position(self, symbol: str) -> bool:
        """Check if a position exists."""
        return symbol in self._positions

    def take_snapshot(self, timestamp: datetime | None = None) -> PortfolioSnapshot:
        """
        Take a snapshot of current portfolio state.

        Args:
            timestamp: Snapshot time.

        Returns:
            PortfolioSnapshot object.
        """
        timestamp = timestamp or datetime.now()

        # Calculate daily P&L
        daily_pnl = self.total_value - self._daily_start_value
        daily_pnl_pct = (daily_pnl / self._daily_start_value * 100) if self._daily_start_value > 0 else 0

        # Check if new day
        if self._last_snapshot_date and timestamp.date() != self._last_snapshot_date.date():
            self._daily_start_value = self.total_value
            daily_pnl = 0.0
            daily_pnl_pct = 0.0

        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            cash=self._cash,
            positions_value=self.positions_value,
            total_value=self.total_value,
            positions={
                sym: {
                    "quantity": pos.quantity,
                    "avg_entry_price": pos.avg_entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "market_value": pos.market_value,
                }
                for sym, pos in self._positions.items()
            },
            daily_pnl=daily_pnl,
            daily_pnl_pct=daily_pnl_pct,
        )

        self._snapshots.append(snapshot)
        self._last_snapshot_date = timestamp

        return snapshot

    def calculate_metrics(self) -> dict[str, Any]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary with performance metrics.
        """
        if not self._snapshots:
            return {}

        # Convert snapshots to DataFrame
        df = pd.DataFrame([
            {
                "timestamp": s.timestamp,
                "total_value": s.total_value,
                "daily_pnl_pct": s.daily_pnl_pct,
            }
            for s in self._snapshots
        ])

        if len(df) < 2:
            return {
                "total_return": self.total_return,
                "total_pnl": self.total_pnl,
            }

        # Calculate returns
        df["returns"] = df["total_value"].pct_change()

        # Basic metrics
        total_return = (df["total_value"].iloc[-1] / df["total_value"].iloc[0] - 1) * 100
        total_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days or 1
        annualized_return = ((1 + total_return / 100) ** (365 / total_days) - 1) * 100

        # Risk metrics
        daily_returns = df["returns"].dropna()
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_return = annualized_return / 100 - risk_free_rate
        sharpe_ratio = excess_return / (volatility / 100) if volatility > 0 else 0

        # Sortino ratio (downside volatility)
        downside_returns = daily_returns[daily_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = excess_return / downside_vol if downside_vol > 0 else 0

        # Maximum drawdown
        cumulative = (1 + daily_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min()) * 100 if len(drawdowns) > 0 else 0

        # Current drawdown
        current_drawdown = ((self._peak_value - self.total_value) / self._peak_value * 100) if self._peak_value > 0 else 0

        # Trade statistics
        if self._trades:
            winners = [t for t in self._trades if t.pnl > 0]
            losers = [t for t in self._trades if t.pnl <= 0]
            win_rate = len(winners) / len(self._trades) * 100
            avg_win = np.mean([t.pnl for t in winners]) if winners else 0
            avg_loss = abs(np.mean([t.pnl for t in losers])) if losers else 0
            profit_factor = sum(t.pnl for t in winners) / abs(sum(t.pnl for t in losers)) if losers else float("inf")
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0

        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "current_drawdown": current_drawdown,
            "total_trades": len(self._trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "total_pnl": self.total_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": self.unrealized_pnl,
        }

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Get equity curve from snapshots.

        Returns:
            DataFrame with timestamp and total_value columns.
        """
        if not self._snapshots:
            return pd.DataFrame(columns=["timestamp", "total_value"])

        return pd.DataFrame([
            {"timestamp": s.timestamp, "total_value": s.total_value}
            for s in self._snapshots
        ])

    def get_trades(self) -> list[Trade]:
        """Get list of completed trades."""
        return self._trades.copy()

    def get_orders(self) -> dict[str, Order]:
        """Get all orders."""
        return self._orders.copy()

    def reset(self, initial_cash: float | None = None) -> None:
        """
        Reset portfolio to initial state.

        Args:
            initial_cash: New initial cash (defaults to original).
        """
        self._cash = initial_cash or self._initial_cash
        self._initial_cash = self._cash
        self._positions.clear()
        self._orders.clear()
        self._trades.clear()
        self._snapshots.clear()
        self._peak_value = self._cash
        self._daily_start_value = self._cash
        self._next_order_id = 1
        self._next_trade_id = 1

    def __repr__(self) -> str:
        """String representation."""
        return f"Portfolio(cash={self._cash:.2f}, positions={len(self._positions)}, total_value={self.total_value:.2f})"


__all__ = [
    "Portfolio",
    "PortfolioPosition",
    "PortfolioSnapshot",
    "Order",
    "OrderType",
    "OrderSide",
    "OrderStatus",
    "Trade",
]

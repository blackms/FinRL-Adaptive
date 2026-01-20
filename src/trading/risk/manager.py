"""
Risk Manager Module

Provides risk management functionality including position sizing,
stop loss management, and portfolio exposure limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class PositionSizingMethod(Enum):
    """Position sizing methods."""

    FIXED = "fixed"
    KELLY = "kelly"
    VOLATILITY = "volatility"
    ATR = "atr"
    PERCENT_RISK = "percent_risk"


class StopLossType(Enum):
    """Stop loss types."""

    FIXED_PERCENT = "fixed_percent"
    ATR_BASED = "atr_based"
    VOLATILITY = "volatility"
    TRAILING = "trailing"
    CHANDELIER = "chandelier"


@dataclass
class RiskConfig:
    """
    Risk management configuration.

    Attributes:
        position_sizing_method: Method for calculating position sizes.
        max_position_size_pct: Maximum single position size as % of portfolio.
        max_portfolio_exposure: Maximum total exposure as % of portfolio.
        max_sector_exposure: Maximum exposure per sector as % of portfolio.
        max_correlation_exposure: Maximum correlated exposure.
        stop_loss_type: Type of stop loss to use.
        stop_loss_pct: Fixed stop loss percentage.
        atr_multiplier: ATR multiplier for ATR-based stops.
        trailing_stop_pct: Trailing stop percentage.
        take_profit_pct: Take profit percentage (optional).
        max_daily_loss_pct: Maximum daily loss before stopping trading.
        max_drawdown_pct: Maximum drawdown before reducing exposure.
        kelly_fraction: Fraction of Kelly criterion to use (typically 0.25-0.5).
        min_win_rate: Minimum win rate for Kelly calculation.
    """

    position_sizing_method: PositionSizingMethod = PositionSizingMethod.PERCENT_RISK
    max_position_size_pct: float = 0.10
    max_portfolio_exposure: float = 1.0
    max_sector_exposure: float = 0.25
    max_correlation_exposure: float = 0.50
    stop_loss_type: StopLossType = StopLossType.ATR_BASED
    stop_loss_pct: float = 0.02
    atr_multiplier: float = 2.0
    trailing_stop_pct: float = 0.05
    take_profit_pct: float | None = None
    max_daily_loss_pct: float = 0.03
    max_drawdown_pct: float = 0.15
    kelly_fraction: float = 0.25
    min_win_rate: float = 0.4


@dataclass
class PositionRisk:
    """
    Risk metrics for a position.

    Attributes:
        symbol: Ticker symbol.
        current_price: Current market price.
        entry_price: Position entry price.
        quantity: Number of shares.
        stop_loss: Stop loss price.
        take_profit: Take profit price (optional).
        risk_amount: Dollar amount at risk.
        risk_pct: Risk as percentage of position value.
        reward_risk_ratio: Reward to risk ratio.
        days_held: Number of days position has been held.
    """

    symbol: str
    current_price: float
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float | None = None
    risk_amount: float = 0.0
    risk_pct: float = 0.0
    reward_risk_ratio: float | None = None
    days_held: int = 0

    def __post_init__(self) -> None:
        """Calculate derived metrics."""
        position_value = abs(self.quantity) * self.current_price
        is_long = self.quantity > 0

        if is_long:
            self.risk_amount = abs(self.quantity) * (self.current_price - self.stop_loss)
        else:
            self.risk_amount = abs(self.quantity) * (self.stop_loss - self.current_price)

        if position_value > 0:
            self.risk_pct = self.risk_amount / position_value

        if self.take_profit is not None and self.risk_amount > 0:
            if is_long:
                reward = abs(self.quantity) * (self.take_profit - self.current_price)
            else:
                reward = abs(self.quantity) * (self.current_price - self.take_profit)
            self.reward_risk_ratio = reward / self.risk_amount if self.risk_amount > 0 else None


@dataclass
class PortfolioRisk:
    """
    Portfolio-level risk metrics.

    Attributes:
        total_exposure: Total portfolio exposure (sum of position values).
        exposure_pct: Exposure as percentage of portfolio value.
        total_risk: Total dollar amount at risk across all positions.
        risk_pct: Risk as percentage of portfolio value.
        sector_exposures: Exposure by sector.
        largest_position_pct: Size of largest position as %.
        position_count: Number of open positions.
        daily_pnl: Daily profit/loss.
        daily_pnl_pct: Daily P&L as percentage.
        drawdown: Current drawdown from peak.
        drawdown_pct: Drawdown as percentage.
    """

    total_exposure: float = 0.0
    exposure_pct: float = 0.0
    total_risk: float = 0.0
    risk_pct: float = 0.0
    sector_exposures: dict[str, float] = field(default_factory=dict)
    largest_position_pct: float = 0.0
    position_count: int = 0
    daily_pnl: float = 0.0
    daily_pnl_pct: float = 0.0
    drawdown: float = 0.0
    drawdown_pct: float = 0.0


class RiskManager:
    """
    Risk management system for trading.

    Handles:
    - Position sizing (Kelly criterion, fixed, volatility-based)
    - Stop loss management (fixed, ATR-based, trailing)
    - Portfolio exposure limits
    - Risk monitoring and alerts

    Example:
        >>> config = RiskConfig(
        ...     position_sizing_method=PositionSizingMethod.KELLY,
        ...     max_position_size_pct=0.10,
        ...     stop_loss_type=StopLossType.ATR_BASED,
        ... )
        >>> risk_manager = RiskManager(config)
        >>> size = risk_manager.calculate_position_size(
        ...     signal_strength=0.7,
        ...     current_price=100,
        ...     capital=100000,
        ...     volatility=0.02,
        ... )
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """
        Initialize the risk manager.

        Args:
            config: Risk configuration. Defaults to RiskConfig.
        """
        self.config = config or RiskConfig()
        self._trade_history: list[dict[str, Any]] = []
        self._peak_value: float = 0.0
        self._daily_starting_value: float = 0.0
        self._daily_pnl: float = 0.0

    def calculate_position_size(
        self,
        signal_strength: float,
        current_price: float,
        capital: float,
        volatility: float | None = None,
        atr: float | None = None,
        win_rate: float | None = None,
        avg_win: float | None = None,
        avg_loss: float | None = None,
    ) -> float:
        """
        Calculate position size based on configured method.

        Args:
            signal_strength: Signal strength (-1 to 1).
            current_price: Current asset price.
            capital: Available capital.
            volatility: Asset volatility (for volatility sizing).
            atr: Average True Range (for ATR sizing).
            win_rate: Historical win rate (for Kelly).
            avg_win: Average winning trade (for Kelly).
            avg_loss: Average losing trade (for Kelly).

        Returns:
            Number of shares to trade (positive for buy, negative for sell).
        """
        method = self.config.position_sizing_method

        if method == PositionSizingMethod.FIXED:
            position_value = capital * self.config.max_position_size_pct
        elif method == PositionSizingMethod.KELLY:
            position_value = self._kelly_position_size(
                capital, win_rate, avg_win, avg_loss
            )
        elif method == PositionSizingMethod.VOLATILITY:
            position_value = self._volatility_position_size(
                capital, volatility or 0.02
            )
        elif method == PositionSizingMethod.ATR:
            position_value = self._atr_position_size(capital, atr or 0, current_price)
        elif method == PositionSizingMethod.PERCENT_RISK:
            position_value = self._percent_risk_position_size(
                capital, current_price, atr or (current_price * 0.02)
            )
        else:
            position_value = capital * self.config.max_position_size_pct

        # Apply signal strength modifier
        position_value *= 0.5 + abs(signal_strength) * 0.5

        # Apply maximum position size limit
        max_value = capital * self.config.max_position_size_pct
        position_value = min(position_value, max_value)

        # Convert to shares
        if current_price <= 0:
            return 0.0

        shares = position_value / current_price

        return shares if signal_strength > 0 else -shares

    def _kelly_position_size(
        self,
        capital: float,
        win_rate: float | None,
        avg_win: float | None,
        avg_loss: float | None,
    ) -> float:
        """
        Calculate position size using Kelly Criterion.

        Kelly% = W - [(1-W) / R]
        where W = win rate, R = win/loss ratio

        Args:
            capital: Available capital.
            win_rate: Historical win rate.
            avg_win: Average winning trade.
            avg_loss: Average losing trade.

        Returns:
            Optimal position value.
        """
        # Use defaults if not provided
        win_rate = win_rate or 0.5
        avg_win = avg_win or 1.0
        avg_loss = avg_loss or 1.0

        # Validate win rate
        if win_rate < self.config.min_win_rate:
            return 0.0

        # Calculate win/loss ratio
        if avg_loss == 0:
            return 0.0
        win_loss_ratio = avg_win / avg_loss

        # Kelly formula
        kelly_pct = win_rate - ((1 - win_rate) / win_loss_ratio)

        # Apply fraction (conservative Kelly)
        kelly_pct *= self.config.kelly_fraction

        # Ensure non-negative
        kelly_pct = max(0, kelly_pct)

        return capital * kelly_pct

    def _volatility_position_size(
        self,
        capital: float,
        volatility: float,
    ) -> float:
        """
        Calculate position size based on volatility.

        Higher volatility = smaller position.

        Args:
            capital: Available capital.
            volatility: Asset volatility (daily).

        Returns:
            Position value.
        """
        if volatility <= 0:
            return capital * self.config.max_position_size_pct

        # Target volatility contribution per position
        target_vol = 0.02  # 2% daily volatility contribution

        position_pct = target_vol / volatility
        position_pct = min(position_pct, self.config.max_position_size_pct)

        return capital * position_pct

    def _atr_position_size(
        self,
        capital: float,
        atr: float,
        current_price: float,
    ) -> float:
        """
        Calculate position size based on ATR.

        Args:
            capital: Available capital.
            atr: Average True Range.
            current_price: Current price.

        Returns:
            Position value.
        """
        if atr <= 0 or current_price <= 0:
            return capital * self.config.max_position_size_pct

        # Risk amount
        risk_amount = capital * self.config.stop_loss_pct

        # Risk per share (using ATR multiplier)
        risk_per_share = atr * self.config.atr_multiplier

        # Position size in shares
        shares = risk_amount / risk_per_share

        return shares * current_price

    def _percent_risk_position_size(
        self,
        capital: float,
        current_price: float,
        risk_per_share: float,
    ) -> float:
        """
        Calculate position size based on fixed percentage risk.

        Args:
            capital: Available capital.
            current_price: Current price.
            risk_per_share: Dollar risk per share (stop distance).

        Returns:
            Position value.
        """
        if risk_per_share <= 0:
            return capital * self.config.max_position_size_pct

        risk_amount = capital * self.config.stop_loss_pct
        shares = risk_amount / risk_per_share

        return shares * current_price

    def calculate_stop_loss(
        self,
        entry_price: float,
        is_long: bool,
        atr: float | None = None,
        volatility: float | None = None,
    ) -> float:
        """
        Calculate stop loss price based on configured method.

        Args:
            entry_price: Position entry price.
            is_long: Whether position is long.
            atr: Average True Range (for ATR-based stops).
            volatility: Asset volatility (for volatility stops).

        Returns:
            Stop loss price.
        """
        stop_type = self.config.stop_loss_type

        if stop_type == StopLossType.FIXED_PERCENT:
            stop_distance = entry_price * self.config.stop_loss_pct
        elif stop_type == StopLossType.ATR_BASED:
            stop_distance = (atr or entry_price * 0.02) * self.config.atr_multiplier
        elif stop_type == StopLossType.VOLATILITY:
            stop_distance = entry_price * (volatility or 0.02) * 2
        elif stop_type == StopLossType.TRAILING:
            stop_distance = entry_price * self.config.trailing_stop_pct
        elif stop_type == StopLossType.CHANDELIER:
            stop_distance = (atr or entry_price * 0.02) * 3
        else:
            stop_distance = entry_price * self.config.stop_loss_pct

        if is_long:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: float,
        is_long: bool,
        reward_risk_ratio: float = 2.0,
    ) -> float | None:
        """
        Calculate take profit price.

        Args:
            entry_price: Position entry price.
            stop_loss: Stop loss price.
            is_long: Whether position is long.
            reward_risk_ratio: Target reward to risk ratio.

        Returns:
            Take profit price or None if not configured.
        """
        if self.config.take_profit_pct is not None:
            # Use fixed percentage
            if is_long:
                return entry_price * (1 + self.config.take_profit_pct)
            else:
                return entry_price * (1 - self.config.take_profit_pct)

        # Calculate based on reward/risk ratio
        risk = abs(entry_price - stop_loss)
        reward = risk * reward_risk_ratio

        if is_long:
            return entry_price + reward
        else:
            return entry_price - reward

    def update_trailing_stop(
        self,
        current_price: float,
        current_stop: float,
        is_long: bool,
        atr: float | None = None,
    ) -> float:
        """
        Update trailing stop loss.

        Args:
            current_price: Current market price.
            current_stop: Current stop loss price.
            is_long: Whether position is long.
            atr: Current ATR (for ATR-based trailing stops).

        Returns:
            Updated stop loss price (never worse than current).
        """
        if self.config.stop_loss_type == StopLossType.TRAILING:
            stop_distance = current_price * self.config.trailing_stop_pct
        elif self.config.stop_loss_type == StopLossType.CHANDELIER:
            stop_distance = (atr or current_price * 0.02) * 3
        else:
            stop_distance = current_price * self.config.stop_loss_pct

        if is_long:
            new_stop = current_price - stop_distance
            return max(current_stop, new_stop)
        else:
            new_stop = current_price + stop_distance
            return min(current_stop, new_stop)

    def check_position_risk(
        self,
        symbol: str,
        entry_price: float,
        current_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float | None = None,
        entry_time: datetime | None = None,
    ) -> PositionRisk:
        """
        Calculate risk metrics for a position.

        Args:
            symbol: Ticker symbol.
            entry_price: Position entry price.
            current_price: Current market price.
            quantity: Number of shares.
            stop_loss: Stop loss price.
            take_profit: Take profit price (optional).
            entry_time: Position entry time.

        Returns:
            PositionRisk object with calculated metrics.
        """
        days_held = 0
        if entry_time:
            days_held = (datetime.now() - entry_time).days

        return PositionRisk(
            symbol=symbol,
            current_price=current_price,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            days_held=days_held,
        )

    def check_portfolio_risk(
        self,
        positions: list[dict[str, Any]],
        portfolio_value: float,
    ) -> PortfolioRisk:
        """
        Calculate portfolio-level risk metrics.

        Args:
            positions: List of position dictionaries with keys:
                - symbol, quantity, current_price, entry_price, stop_loss, sector
            portfolio_value: Total portfolio value.

        Returns:
            PortfolioRisk object with calculated metrics.
        """
        risk = PortfolioRisk()

        if portfolio_value <= 0:
            return risk

        total_exposure = 0.0
        total_risk = 0.0
        sector_exposures: dict[str, float] = {}
        largest_position = 0.0

        for pos in positions:
            position_value = abs(pos["quantity"]) * pos["current_price"]
            total_exposure += position_value

            # Track largest position
            if position_value > largest_position:
                largest_position = position_value

            # Calculate position risk
            is_long = pos["quantity"] > 0
            if is_long:
                pos_risk = abs(pos["quantity"]) * (pos["current_price"] - pos["stop_loss"])
            else:
                pos_risk = abs(pos["quantity"]) * (pos["stop_loss"] - pos["current_price"])
            total_risk += max(0, pos_risk)

            # Track sector exposure
            sector = pos.get("sector", "Unknown")
            sector_exposures[sector] = sector_exposures.get(sector, 0) + position_value

        risk.total_exposure = total_exposure
        risk.exposure_pct = total_exposure / portfolio_value
        risk.total_risk = total_risk
        risk.risk_pct = total_risk / portfolio_value
        risk.sector_exposures = {
            sector: exp / portfolio_value for sector, exp in sector_exposures.items()
        }
        risk.largest_position_pct = largest_position / portfolio_value
        risk.position_count = len(positions)

        # Update peak and drawdown
        if portfolio_value > self._peak_value:
            self._peak_value = portfolio_value

        if self._peak_value > 0:
            risk.drawdown = self._peak_value - portfolio_value
            risk.drawdown_pct = risk.drawdown / self._peak_value

        return risk

    def check_exposure_limits(
        self,
        proposed_position_value: float,
        current_exposure: float,
        portfolio_value: float,
        sector: str | None = None,
        sector_exposures: dict[str, float] | None = None,
    ) -> tuple[bool, str]:
        """
        Check if a proposed trade would exceed exposure limits.

        Args:
            proposed_position_value: Value of proposed position.
            current_exposure: Current total exposure.
            portfolio_value: Total portfolio value.
            sector: Sector of proposed position.
            sector_exposures: Current sector exposures.

        Returns:
            Tuple of (is_allowed, reason).
        """
        if portfolio_value <= 0:
            return False, "Invalid portfolio value"

        # Check position size limit
        position_pct = proposed_position_value / portfolio_value
        if position_pct > self.config.max_position_size_pct:
            return False, f"Position size {position_pct:.1%} exceeds limit {self.config.max_position_size_pct:.1%}"

        # Check total exposure limit
        new_exposure = current_exposure + proposed_position_value
        exposure_pct = new_exposure / portfolio_value
        if exposure_pct > self.config.max_portfolio_exposure:
            return False, f"Total exposure {exposure_pct:.1%} would exceed limit {self.config.max_portfolio_exposure:.1%}"

        # Check sector exposure limit
        if sector and sector_exposures:
            current_sector_exp = sector_exposures.get(sector, 0)
            new_sector_exp = current_sector_exp + proposed_position_value
            sector_pct = new_sector_exp / portfolio_value
            if sector_pct > self.config.max_sector_exposure:
                return False, f"Sector {sector} exposure {sector_pct:.1%} would exceed limit {self.config.max_sector_exposure:.1%}"

        return True, "OK"

    def should_stop_trading(
        self,
        portfolio_value: float,
        daily_starting_value: float | None = None,
    ) -> tuple[bool, str]:
        """
        Check if trading should be stopped due to risk limits.

        Args:
            portfolio_value: Current portfolio value.
            daily_starting_value: Portfolio value at start of day.

        Returns:
            Tuple of (should_stop, reason).
        """
        daily_start = daily_starting_value or self._daily_starting_value
        if daily_start <= 0:
            return False, "OK"

        # Check daily loss
        daily_pnl_pct = (portfolio_value - daily_start) / daily_start
        if daily_pnl_pct < -self.config.max_daily_loss_pct:
            return True, f"Daily loss {-daily_pnl_pct:.1%} exceeds limit {self.config.max_daily_loss_pct:.1%}"

        # Check drawdown
        if self._peak_value > 0:
            drawdown_pct = (self._peak_value - portfolio_value) / self._peak_value
            if drawdown_pct > self.config.max_drawdown_pct:
                return True, f"Drawdown {drawdown_pct:.1%} exceeds limit {self.config.max_drawdown_pct:.1%}"

        return False, "OK"

    def record_trade(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_time: datetime,
        exit_time: datetime,
    ) -> None:
        """
        Record a completed trade for statistics.

        Args:
            symbol: Ticker symbol.
            entry_price: Entry price.
            exit_price: Exit price.
            quantity: Number of shares.
            entry_time: Entry time.
            exit_time: Exit time.
        """
        is_long = quantity > 0
        if is_long:
            pnl = (exit_price - entry_price) * abs(quantity)
        else:
            pnl = (entry_price - exit_price) * abs(quantity)

        pnl_pct = (exit_price / entry_price - 1) * (1 if is_long else -1)

        self._trade_history.append({
            "symbol": symbol,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": quantity,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "is_winner": pnl > 0,
        })

    def get_trade_statistics(self) -> dict[str, Any]:
        """
        Calculate trading statistics from history.

        Returns:
            Dictionary with trading statistics.
        """
        if not self._trade_history:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
            }

        trades = self._trade_history
        winners = [t for t in trades if t["is_winner"]]
        losers = [t for t in trades if not t["is_winner"]]

        total_trades = len(trades)
        win_rate = len(winners) / total_trades if total_trades > 0 else 0

        avg_win = np.mean([t["pnl"] for t in winners]) if winners else 0
        avg_loss = abs(np.mean([t["pnl"] for t in losers])) if losers else 0

        gross_profit = sum(t["pnl"] for t in winners)
        gross_loss = abs(sum(t["pnl"] for t in losers))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            "total_trades": total_trades,
            "winners": len(winners),
            "losers": len(losers),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": gross_profit - gross_loss,
        }

    def set_daily_starting_value(self, value: float) -> None:
        """Set the portfolio value at start of day."""
        self._daily_starting_value = value

    def reset_peak_value(self, value: float) -> None:
        """Reset peak value (e.g., after capital withdrawal)."""
        self._peak_value = value


__all__ = [
    "RiskManager",
    "RiskConfig",
    "PositionSizingMethod",
    "StopLossType",
    "PositionRisk",
    "PortfolioRisk",
]

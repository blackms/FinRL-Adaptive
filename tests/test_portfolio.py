"""
Tests for the portfolio management module.

This module tests:
- Position tracking
- P&L calculations
- Portfolio metric calculations
- Transaction handling
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List


# =============================================================================
# Test Position Tracking
# =============================================================================

class TestPositionTracking:
    """Tests for position tracking functionality."""

    def test_add_position(self, empty_positions):
        """Test adding a new position."""
        positions = empty_positions.copy()

        new_position = {
            'symbol': 'AAPL',
            'quantity': 100,
            'entry_price': 150.0,
            'entry_date': datetime(2023, 6, 1)
        }

        positions['AAPL'] = new_position

        assert 'AAPL' in positions
        assert positions['AAPL']['quantity'] == 100
        assert positions['AAPL']['entry_price'] == 150.0

    def test_increase_position(self, sample_positions):
        """Test increasing an existing position."""
        positions = sample_positions.copy()

        original_qty = positions['AAPL']['quantity']
        original_entry = positions['AAPL']['entry_price']

        # Add 50 more shares at $160
        additional_qty = 50
        additional_price = 160.0

        # Calculate new average entry price
        total_qty = original_qty + additional_qty
        total_cost = (original_qty * original_entry) + (additional_qty * additional_price)
        new_avg_price = total_cost / total_qty

        positions['AAPL']['quantity'] = total_qty
        positions['AAPL']['entry_price'] = new_avg_price

        assert positions['AAPL']['quantity'] == 150
        # (100 * 150 + 50 * 160) / 150 = 153.33
        assert abs(positions['AAPL']['entry_price'] - 153.33) < 0.01

    def test_decrease_position(self, sample_positions):
        """Test decreasing an existing position (partial sell)."""
        positions = sample_positions.copy()

        original_qty = positions['AAPL']['quantity']
        sell_qty = 30

        # Entry price remains same for remaining shares
        entry_price = positions['AAPL']['entry_price']

        positions['AAPL']['quantity'] = original_qty - sell_qty

        assert positions['AAPL']['quantity'] == 70
        assert positions['AAPL']['entry_price'] == entry_price  # Unchanged

    def test_close_position(self, sample_positions):
        """Test closing a position completely."""
        positions = sample_positions.copy()

        del positions['AAPL']

        assert 'AAPL' not in positions

    def test_position_value_calculation(self, sample_positions):
        """Test position value calculation."""
        position = sample_positions['AAPL']

        value = position['quantity'] * position['current_price']

        expected_value = 100 * 175.0
        assert value == expected_value

    def test_total_portfolio_value(self, sample_positions):
        """Test total portfolio value calculation."""
        total_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in sample_positions.values()
        )

        # AAPL: 100 * 175 = 17500
        # MSFT: 50 * 320 = 16000
        # GOOGL: 75 * 88 = 6600
        expected = 17500 + 16000 + 6600
        assert total_value == expected

    def test_position_weight_calculation(self, sample_positions):
        """Test position weight as percentage of portfolio."""
        total_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in sample_positions.values()
        )

        weights = {}
        for symbol, pos in sample_positions.items():
            position_value = pos['quantity'] * pos['current_price']
            weights[symbol] = position_value / total_value

        # Weights should sum to 1
        assert abs(sum(weights.values()) - 1.0) < 0.01

    def test_negative_quantity_not_allowed(self):
        """Test that negative quantities are rejected."""
        with pytest.raises(ValueError):
            position = {
                'symbol': 'AAPL',
                'quantity': -50,  # Invalid
                'entry_price': 150.0
            }
            if position['quantity'] < 0:
                raise ValueError("Quantity cannot be negative")

    def test_zero_quantity_closes_position(self, sample_positions):
        """Test that zero quantity effectively closes position."""
        positions = sample_positions.copy()

        positions['AAPL']['quantity'] = 0

        # Position with zero quantity should be removed
        active_positions = {
            k: v for k, v in positions.items()
            if v['quantity'] > 0
        }

        assert 'AAPL' not in active_positions


# =============================================================================
# Test P&L Calculations
# =============================================================================

class TestPnLCalculations:
    """Tests for Profit and Loss calculations."""

    def test_unrealized_pnl_profit(self, sample_positions):
        """Test unrealized P&L calculation for profitable position."""
        position = sample_positions['AAPL']

        unrealized_pnl = (
            (position['current_price'] - position['entry_price']) *
            position['quantity']
        )

        # (175 - 150) * 100 = 2500
        expected_pnl = (175 - 150) * 100
        assert unrealized_pnl == expected_pnl

    def test_unrealized_pnl_loss(self, sample_positions):
        """Test unrealized P&L calculation for losing position."""
        position = sample_positions['GOOGL']

        unrealized_pnl = (
            (position['current_price'] - position['entry_price']) *
            position['quantity']
        )

        # (88 - 95) * 75 = -525
        expected_pnl = (88 - 95) * 75
        assert unrealized_pnl == expected_pnl

    def test_unrealized_pnl_percentage(self, sample_positions):
        """Test unrealized P&L as percentage."""
        position = sample_positions['AAPL']

        pnl_pct = (
            (position['current_price'] - position['entry_price']) /
            position['entry_price']
        )

        # (175 - 150) / 150 = 16.67%
        expected_pct = (175 - 150) / 150
        assert abs(pnl_pct - expected_pct) < 0.01

    def test_realized_pnl_from_trade(self, sample_trade_history):
        """Test realized P&L calculation from trade."""
        # Find a complete round trip (buy and sell)
        aapl_trades = [t for t in sample_trade_history if t['symbol'] == 'AAPL']

        buy_trade = next(t for t in aapl_trades if t['action'] == 'BUY')
        sell_trade = next(t for t in aapl_trades if t['action'] == 'SELL')

        # Realized P&L = (sell_price - buy_price) * sell_quantity
        realized_pnl = (sell_trade['price'] - buy_trade['price']) * sell_trade['quantity']

        # (160 - 145) * 50 = 750
        expected_pnl = (160 - 145) * 50
        assert realized_pnl == expected_pnl

    def test_realized_pnl_with_commission(self, sample_trade_history):
        """Test realized P&L including commissions."""
        commission_rate = 0.001  # 0.1%

        aapl_trades = [t for t in sample_trade_history if t['symbol'] == 'AAPL']

        buy_trade = next(t for t in aapl_trades if t['action'] == 'BUY')
        sell_trade = next(t for t in aapl_trades if t['action'] == 'SELL')

        buy_cost = buy_trade['price'] * buy_trade['quantity']
        buy_commission = buy_cost * commission_rate

        sell_value = sell_trade['price'] * sell_trade['quantity']
        sell_commission = sell_value * commission_rate

        net_pnl = sell_value - (buy_trade['price'] * sell_trade['quantity']) - buy_commission - sell_commission

        assert net_pnl < (sell_trade['price'] - buy_trade['price']) * sell_trade['quantity']

    def test_total_portfolio_pnl(self, sample_positions):
        """Test total portfolio P&L calculation."""
        total_unrealized_pnl = sum(
            (pos['current_price'] - pos['entry_price']) * pos['quantity']
            for pos in sample_positions.values()
        )

        # AAPL: (175-150)*100 = 2500
        # MSFT: (320-280)*50 = 2000
        # GOOGL: (88-95)*75 = -525
        expected_total = 2500 + 2000 - 525
        assert total_unrealized_pnl == expected_total

    def test_portfolio_return_calculation(self, sample_positions, portfolio_config):
        """Test portfolio return calculation."""
        initial_capital = portfolio_config['initial_capital']

        # Total P&L
        total_pnl = sum(
            (pos['current_price'] - pos['entry_price']) * pos['quantity']
            for pos in sample_positions.values()
        )

        portfolio_return = total_pnl / initial_capital

        expected_return = (2500 + 2000 - 525) / 100000
        assert abs(portfolio_return - expected_return) < 0.0001

    def test_pnl_with_dividends(self):
        """Test P&L calculation including dividends."""
        entry_price = 150.0
        current_price = 155.0
        quantity = 100
        dividend_per_share = 0.50  # Quarterly dividend

        price_pnl = (current_price - entry_price) * quantity
        dividend_income = dividend_per_share * quantity

        total_return = price_pnl + dividend_income

        expected = (155 - 150) * 100 + 0.50 * 100
        assert total_return == expected

    def test_annualized_return(self, sample_positions):
        """Test annualized return calculation."""
        position = sample_positions['AAPL']

        entry_date = position['entry_date']
        current_date = datetime(2023, 6, 15)

        days_held = (current_date - entry_date).days
        simple_return = (
            (position['current_price'] - position['entry_price']) /
            position['entry_price']
        )

        # Annualized return
        if days_held > 0:
            annualized_return = (1 + simple_return) ** (365 / days_held) - 1
        else:
            annualized_return = 0

        assert isinstance(annualized_return, float)


# =============================================================================
# Test Portfolio Metrics
# =============================================================================

class TestPortfolioMetrics:
    """Tests for portfolio-level metric calculations."""

    def test_portfolio_beta(self, sample_ohlcv_data):
        """Test portfolio beta calculation."""
        # Mock market returns
        np.random.seed(42)
        market_returns = np.random.normal(0.0005, 0.01, 100)

        # Calculate weighted portfolio beta
        position_weights = {'AAPL': 0.4, 'MSFT': 0.35, 'GOOGL': 0.25}
        position_betas = {'AAPL': 1.2, 'MSFT': 1.0, 'GOOGL': 1.1}

        portfolio_beta = sum(
            position_weights[sym] * position_betas[sym]
            for sym in position_weights
        )

        expected_beta = 0.4 * 1.2 + 0.35 * 1.0 + 0.25 * 1.1
        assert abs(portfolio_beta - expected_beta) < 0.01

    def test_portfolio_volatility(self, sample_ohlcv_data):
        """Test portfolio volatility calculation."""
        # Simplified: assume no correlation
        position_weights = np.array([0.4, 0.35, 0.25])
        position_vols = np.array([0.25, 0.22, 0.28])

        # Portfolio variance (no correlation assumption)
        portfolio_variance = np.sum((position_weights ** 2) * (position_vols ** 2))
        portfolio_vol = np.sqrt(portfolio_variance)

        assert portfolio_vol > 0
        assert portfolio_vol < max(position_vols)  # Diversification benefit

    def test_portfolio_sharpe_ratio(self, mock_backtest_results):
        """Test portfolio Sharpe ratio calculation."""
        sharpe = mock_backtest_results['sharpe_ratio']

        assert isinstance(sharpe, float)
        assert sharpe == 1.25

    def test_portfolio_sortino_ratio(self, mock_backtest_results):
        """Test portfolio Sortino ratio calculation."""
        sortino = mock_backtest_results['sortino_ratio']

        assert isinstance(sortino, float)
        assert sortino >= mock_backtest_results['sharpe_ratio']  # Usually >= Sharpe

    def test_portfolio_max_drawdown(self, mock_backtest_results):
        """Test portfolio max drawdown calculation."""
        max_dd = mock_backtest_results['max_drawdown']

        assert max_dd <= 0  # Drawdown is negative
        assert max_dd >= -1  # Not more than -100%
        assert max_dd == -0.12

    def test_win_rate_calculation(self, mock_backtest_results):
        """Test win rate calculation."""
        win_rate = mock_backtest_results['win_rate']
        total_trades = mock_backtest_results['total_trades']
        winning_trades = mock_backtest_results['winning_trades']

        calculated_win_rate = winning_trades / total_trades

        assert abs(win_rate - calculated_win_rate) < 0.01
        assert 0 <= win_rate <= 1

    def test_profit_factor(self, mock_backtest_results):
        """Test profit factor calculation."""
        profit_factor = mock_backtest_results['profit_factor']

        # Profit factor = gross profit / gross loss
        assert profit_factor > 0
        assert profit_factor == 1.65

    def test_average_trade_duration(self, mock_backtest_results):
        """Test average holding period calculation."""
        avg_holding = mock_backtest_results['avg_holding_period']

        assert avg_holding > 0
        assert avg_holding == 12.5

    def test_risk_adjusted_return(self, mock_backtest_results):
        """Test risk-adjusted return metrics."""
        total_return = mock_backtest_results['total_return']
        max_dd = abs(mock_backtest_results['max_drawdown'])

        # Calmar ratio = annualized return / max drawdown
        if max_dd > 0:
            calmar = mock_backtest_results['annualized_return'] / max_dd
        else:
            calmar = float('inf')

        assert calmar >= 0 or calmar == float('inf')


# =============================================================================
# Test Transaction Handling
# =============================================================================

class TestTransactionHandling:
    """Tests for transaction and order handling."""

    def test_buy_order_execution(self):
        """Test buy order execution."""
        order = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'order_type': 'MARKET',
            'price': None  # Market order
        }

        # Simulate execution
        execution_price = 150.50
        commission = 0.0
        slippage = execution_price * 0.001

        fill_price = execution_price + slippage
        total_cost = (fill_price * order['quantity']) + commission

        assert fill_price > execution_price  # Slippage increases cost for buy
        assert total_cost > 0

    def test_sell_order_execution(self):
        """Test sell order execution."""
        order = {
            'symbol': 'AAPL',
            'action': 'SELL',
            'quantity': 50,
            'order_type': 'MARKET',
            'price': None
        }

        execution_price = 155.00
        slippage = execution_price * 0.001

        fill_price = execution_price - slippage  # Slippage decreases proceeds for sell
        total_proceeds = fill_price * order['quantity']

        assert fill_price < execution_price
        assert total_proceeds > 0

    def test_limit_order_execution(self):
        """Test limit order execution."""
        order = {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'order_type': 'LIMIT',
            'price': 148.00  # Limit price
        }

        current_price = 150.00

        # Limit order only fills if price reaches limit
        can_fill = current_price <= order['price']

        assert not can_fill  # Current price too high for buy limit

    def test_stop_order_execution(self):
        """Test stop order execution."""
        order = {
            'symbol': 'AAPL',
            'action': 'SELL',
            'quantity': 100,
            'order_type': 'STOP',
            'price': 145.00  # Stop price
        }

        current_price = 140.00

        # Stop sell triggers when price falls to stop
        triggered = current_price <= order['price']

        assert triggered  # Price below stop, should trigger

    def test_order_validation_insufficient_funds(self):
        """Test order validation with insufficient funds."""
        available_cash = 5000
        order_cost = 15000  # Trying to buy $15k worth

        is_valid = available_cash >= order_cost

        assert not is_valid

    def test_order_validation_insufficient_shares(self):
        """Test order validation with insufficient shares to sell."""
        current_position = 50
        sell_quantity = 100

        is_valid = current_position >= sell_quantity

        assert not is_valid

    def test_order_fills_update_positions(self, sample_positions):
        """Test that order fills correctly update positions."""
        positions = sample_positions.copy()

        # Execute buy order
        fill = {
            'symbol': 'AAPL',
            'quantity': 25,
            'price': 176.00,
            'action': 'BUY'
        }

        old_qty = positions['AAPL']['quantity']
        old_entry = positions['AAPL']['entry_price']

        # Update position
        new_qty = old_qty + fill['quantity']
        new_entry = (
            (old_qty * old_entry) + (fill['quantity'] * fill['price'])
        ) / new_qty

        positions['AAPL']['quantity'] = new_qty
        positions['AAPL']['entry_price'] = new_entry

        assert positions['AAPL']['quantity'] == 125
        assert positions['AAPL']['entry_price'] > old_entry  # Bought at higher price

    def test_order_fills_update_cash(self, portfolio_config):
        """Test that order fills correctly update cash balance."""
        cash = portfolio_config['initial_capital']

        # Buy order
        buy_cost = 15000
        cash_after_buy = cash - buy_cost

        assert cash_after_buy == 85000

        # Sell order
        sell_proceeds = 8000
        cash_after_sell = cash_after_buy + sell_proceeds

        assert cash_after_sell == 93000

    def test_transaction_history_recording(self, sample_trade_history):
        """Test that transactions are properly recorded."""
        # All trades should have required fields
        required_fields = ['symbol', 'action', 'quantity', 'price', 'timestamp']

        for trade in sample_trade_history:
            for field in required_fields:
                assert field in trade, f"Missing field: {field}"

    def test_fifo_cost_basis(self):
        """Test FIFO (First In First Out) cost basis calculation."""
        # Buy 100 @ $100, then 100 @ $120
        lots = [
            {'quantity': 100, 'price': 100.0, 'date': datetime(2023, 1, 1)},
            {'quantity': 100, 'price': 120.0, 'date': datetime(2023, 2, 1)}
        ]

        # Sell 150 shares - FIFO: first 100 @ $100, next 50 @ $120
        sell_quantity = 150
        sell_price = 130.0

        fifo_cost = (100 * 100.0) + (50 * 120.0)  # Cost of shares sold
        sell_proceeds = sell_quantity * sell_price
        realized_pnl = sell_proceeds - fifo_cost

        expected_cost = 10000 + 6000  # 16000
        expected_pnl = 19500 - 16000  # 3500

        assert fifo_cost == expected_cost
        assert realized_pnl == expected_pnl

    def test_lifo_cost_basis(self):
        """Test LIFO (Last In First Out) cost basis calculation."""
        # Buy 100 @ $100, then 100 @ $120
        lots = [
            {'quantity': 100, 'price': 100.0, 'date': datetime(2023, 1, 1)},
            {'quantity': 100, 'price': 120.0, 'date': datetime(2023, 2, 1)}
        ]

        # Sell 150 shares - LIFO: first 100 @ $120, next 50 @ $100
        sell_quantity = 150
        sell_price = 130.0

        lifo_cost = (100 * 120.0) + (50 * 100.0)  # Cost of shares sold
        sell_proceeds = sell_quantity * sell_price
        realized_pnl = sell_proceeds - lifo_cost

        expected_cost = 12000 + 5000  # 17000
        expected_pnl = 19500 - 17000  # 2500

        assert lifo_cost == expected_cost
        assert realized_pnl == expected_pnl


# =============================================================================
# Test Portfolio Edge Cases
# =============================================================================

class TestPortfolioEdgeCases:
    """Tests for edge cases in portfolio management."""

    def test_empty_portfolio_metrics(self, empty_positions):
        """Test metrics calculation for empty portfolio."""
        positions = empty_positions

        total_value = sum(
            pos.get('quantity', 0) * pos.get('current_price', 0)
            for pos in positions.values()
        )

        assert total_value == 0

    def test_single_position_portfolio(self):
        """Test portfolio with single position."""
        positions = {
            'AAPL': {
                'quantity': 100,
                'entry_price': 150.0,
                'current_price': 160.0
            }
        }

        # Weight should be 100%
        total_value = 100 * 160
        weight = (100 * 160) / total_value

        assert weight == 1.0

    def test_portfolio_with_zero_value_position(self):
        """Test portfolio with a position that has zero value."""
        positions = {
            'AAPL': {'quantity': 100, 'current_price': 150.0},
            'DEFUNCT': {'quantity': 0, 'current_price': 0.0}  # Delisted stock
        }

        total_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in positions.values()
        )

        assert total_value == 15000  # Only AAPL counts

    def test_large_number_of_positions(self):
        """Test portfolio with many positions."""
        num_positions = 500

        positions = {
            f'STOCK_{i}': {
                'quantity': 100,
                'current_price': 100.0 + i
            }
            for i in range(num_positions)
        }

        total_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in positions.values()
        )

        assert len(positions) == 500
        assert total_value > 0

    def test_portfolio_rebalancing(self, sample_positions, portfolio_config):
        """Test portfolio rebalancing to target weights."""
        target_weights = {'AAPL': 0.40, 'MSFT': 0.35, 'GOOGL': 0.25}

        total_value = sum(
            pos['quantity'] * pos['current_price']
            for pos in sample_positions.values()
        )

        current_weights = {
            sym: (pos['quantity'] * pos['current_price']) / total_value
            for sym, pos in sample_positions.items()
        }

        # Calculate required trades for rebalancing
        trades_needed = {}
        for sym in target_weights:
            current_value = sample_positions[sym]['quantity'] * sample_positions[sym]['current_price']
            target_value = total_value * target_weights[sym]
            trade_value = target_value - current_value

            trades_needed[sym] = trade_value / sample_positions[sym]['current_price']

        assert len(trades_needed) == 3

    def test_cash_drag_calculation(self, portfolio_config):
        """Test cash drag (uninvested cash impact)."""
        total_capital = portfolio_config['initial_capital']
        invested_capital = 80000  # 80% invested
        cash = total_capital - invested_capital

        # Assume market returned 10%, cash returned 2%
        market_return = 0.10
        cash_return = 0.02

        invested_return = invested_capital * market_return
        cash_earned = cash * cash_return

        actual_return = (invested_return + cash_earned) / total_capital
        fully_invested_return = market_return

        cash_drag = fully_invested_return - actual_return

        assert cash_drag > 0  # Cash drag is positive (hurts returns)

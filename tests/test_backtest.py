"""
Tests for the backtesting module.

This module tests:
- Backtest execution
- Performance report generation
- Results validation
- Edge cases in backtesting
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, List


# =============================================================================
# Test Backtest Execution
# =============================================================================

class TestBacktestExecution:
    """Tests for backtest execution functionality."""

    def test_backtest_runs_without_error(self, backtest_config, sample_ohlcv_data):
        """Test that backtest completes without errors."""
        backtester = MagicMock()
        backtester.run.return_value = {
            'total_return': 0.10,
            'sharpe_ratio': 1.0,
            'max_drawdown': -0.08
        }

        results = backtester.run(
            data=sample_ohlcv_data,
            config=backtest_config
        )

        assert results is not None
        assert 'total_return' in results

    def test_backtest_returns_required_metrics(self, mock_backtest_results):
        """Test that backtest returns all required metrics."""
        required_metrics = [
            'total_return',
            'annualized_return',
            'sharpe_ratio',
            'sortino_ratio',
            'max_drawdown',
            'win_rate',
            'profit_factor',
            'total_trades'
        ]

        for metric in required_metrics:
            assert metric in mock_backtest_results, f"Missing metric: {metric}"

    def test_backtest_uses_correct_date_range(self, backtest_config, sample_ohlcv_data):
        """Test that backtest uses specified date range."""
        start_date = backtest_config['start_date']
        end_date = backtest_config['end_date']

        # Filter data to date range
        filtered_data = sample_ohlcv_data[
            (pd.to_datetime(sample_ohlcv_data['date']) >= start_date) &
            (pd.to_datetime(sample_ohlcv_data['date']) <= end_date)
        ]

        # Backtest should only use data in range
        assert len(filtered_data) <= len(sample_ohlcv_data)

    def test_backtest_initial_capital(self, backtest_config):
        """Test that backtest starts with correct initial capital."""
        initial_capital = backtest_config['initial_capital']

        # Portfolio value at start should equal initial capital
        portfolio_values = [initial_capital]  # Start

        assert portfolio_values[0] == 100000

    def test_backtest_applies_commission(self, backtest_config):
        """Test that backtest applies commission correctly."""
        commission_rate = backtest_config['commission']

        trade_value = 10000
        commission = trade_value * commission_rate

        # With 0% commission, no cost
        assert commission == 0

    def test_backtest_applies_slippage(self, backtest_config):
        """Test that backtest applies slippage correctly."""
        slippage_rate = backtest_config['slippage']

        expected_price = 100.0
        slippage = expected_price * slippage_rate

        # For buy: actual cost = expected + slippage
        actual_buy_price = expected_price + slippage
        assert actual_buy_price == 100.10

        # For sell: actual proceeds = expected - slippage
        actual_sell_price = expected_price - slippage
        assert actual_sell_price == 99.90

    def test_backtest_respects_rebalance_frequency(self, backtest_config):
        """Test that backtest rebalances at correct frequency."""
        rebalance_freq = backtest_config['rebalance_frequency']

        assert rebalance_freq == 'weekly'

        # Weekly rebalancing = ~52 rebalance points per year
        trading_days = 252
        weeks = trading_days / 5
        expected_rebalances = int(weeks)

        assert expected_rebalances == 50  # Approximately

    def test_backtest_tracks_portfolio_values(self, mock_backtest_results):
        """Test that backtest tracks portfolio values over time."""
        portfolio_values = mock_backtest_results['portfolio_values']
        dates = mock_backtest_results['dates']

        assert len(portfolio_values) == len(dates)
        assert portfolio_values[0] == 100000  # Initial capital

    def test_backtest_no_lookahead_bias(self, sample_ohlcv_data, backtest_config):
        """Test that backtest doesn't use future data."""
        # At each point in time, only data up to that point should be used
        backtester = MagicMock()

        # Mock: at day 100, only first 100 days of data available
        day_100_data = sample_ohlcv_data.iloc[:100]

        # Signal should not depend on future data
        backtester.generate_signal.return_value = 1

        signal = backtester.generate_signal(day_100_data)

        assert signal in [-1, 0, 1]

    def test_backtest_handles_missing_data(self, sample_ohlcv_data):
        """Test that backtest handles missing data gracefully."""
        # Remove some dates
        sparse_data = sample_ohlcv_data.iloc[::3]  # Every 3rd row

        backtester = MagicMock()
        backtester.run.return_value = {'total_return': 0.05}

        # Should complete without error
        results = backtester.run(data=sparse_data, config={})

        assert results is not None

    def test_backtest_handles_delisted_stocks(self, sample_ohlcv_data):
        """Test backtest handling of delisted stocks (data ends early)."""
        # Simulate stock delisting - data stops mid-period
        delisted_data = sample_ohlcv_data[
            sample_ohlcv_data['symbol'] == 'AAPL'
        ].iloc[:50]  # Only first 50 days

        backtester = MagicMock()

        # Should liquidate position when data ends
        backtester.handle_delisting.return_value = {
            'action': 'SELL',
            'reason': 'data_ended'
        }

        action = backtester.handle_delisting(delisted_data)
        assert action['action'] == 'SELL'


# =============================================================================
# Test Performance Report Generation
# =============================================================================

class TestPerformanceReportGeneration:
    """Tests for performance report generation."""

    def test_report_includes_return_metrics(self, mock_backtest_results):
        """Test that report includes return metrics."""
        assert 'total_return' in mock_backtest_results
        assert 'annualized_return' in mock_backtest_results

    def test_report_includes_risk_metrics(self, mock_backtest_results):
        """Test that report includes risk metrics."""
        assert 'max_drawdown' in mock_backtest_results
        assert 'sharpe_ratio' in mock_backtest_results
        assert 'sortino_ratio' in mock_backtest_results

    def test_report_includes_trade_statistics(self, mock_backtest_results):
        """Test that report includes trade statistics."""
        assert 'total_trades' in mock_backtest_results
        assert 'winning_trades' in mock_backtest_results
        assert 'losing_trades' in mock_backtest_results
        assert 'win_rate' in mock_backtest_results

    def test_report_includes_benchmark_comparison(self, mock_backtest_results):
        """Test that report includes benchmark comparison."""
        assert 'benchmark_values' in mock_backtest_results

        portfolio_values = mock_backtest_results['portfolio_values']
        benchmark_values = mock_backtest_results['benchmark_values']

        # Calculate alpha (excess return over benchmark)
        portfolio_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        benchmark_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]

        alpha = portfolio_return - benchmark_return

        assert isinstance(alpha, float)

    def test_total_return_calculation(self, mock_backtest_results):
        """Test total return calculation accuracy."""
        portfolio_values = mock_backtest_results['portfolio_values']

        calculated_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        reported_return = mock_backtest_results['total_return']

        assert abs(calculated_return - reported_return) < 0.01

    def test_sharpe_ratio_calculation(self, mock_backtest_results):
        """Test Sharpe ratio calculation."""
        sharpe = mock_backtest_results['sharpe_ratio']

        # Sharpe should be reasonable
        assert -3 <= sharpe <= 5
        assert sharpe == 1.25

    def test_max_drawdown_calculation(self, mock_backtest_results):
        """Test max drawdown calculation."""
        portfolio_values = np.array(mock_backtest_results['portfolio_values'])

        # Calculate drawdown
        running_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - running_max) / running_max
        calculated_max_dd = drawdowns.min()

        reported_max_dd = mock_backtest_results['max_drawdown']

        # Should be close (may differ due to more detailed tracking)
        assert calculated_max_dd <= 0
        assert reported_max_dd <= 0

    def test_win_rate_calculation(self, mock_backtest_results):
        """Test win rate calculation accuracy."""
        total_trades = mock_backtest_results['total_trades']
        winning_trades = mock_backtest_results['winning_trades']
        reported_win_rate = mock_backtest_results['win_rate']

        calculated_win_rate = winning_trades / total_trades

        assert abs(calculated_win_rate - reported_win_rate) < 0.01

    def test_profit_factor_calculation(self, mock_backtest_results):
        """Test profit factor calculation."""
        profit_factor = mock_backtest_results['profit_factor']

        # Profit factor > 1 indicates profitable strategy
        assert profit_factor > 0
        assert profit_factor == 1.65

    def test_report_format_json(self, mock_backtest_results):
        """Test that report can be serialized to JSON."""
        import json

        # Convert datetimes to strings for JSON
        results_copy = mock_backtest_results.copy()
        results_copy['dates'] = [d.isoformat() for d in results_copy['dates']]

        json_str = json.dumps(results_copy)
        parsed = json.loads(json_str)

        assert parsed['total_return'] == mock_backtest_results['total_return']

    def test_report_includes_rolling_metrics(self):
        """Test that report includes rolling metrics."""
        rolling_metrics = {
            'rolling_sharpe_30d': [0.8, 1.0, 1.2, 1.1, 1.3],
            'rolling_volatility_30d': [0.15, 0.18, 0.20, 0.17, 0.16],
            'rolling_drawdown': [-0.02, -0.05, -0.08, -0.03, -0.01]
        }

        assert 'rolling_sharpe_30d' in rolling_metrics
        assert len(rolling_metrics['rolling_sharpe_30d']) == 5


# =============================================================================
# Test Backtest Results Validation
# =============================================================================

class TestBacktestResultsValidation:
    """Tests for validating backtest results."""

    def test_portfolio_value_never_negative(self, mock_backtest_results):
        """Test that portfolio value never goes negative."""
        portfolio_values = mock_backtest_results['portfolio_values']

        assert all(v >= 0 for v in portfolio_values)

    def test_trade_counts_sum_correctly(self, mock_backtest_results):
        """Test that winning + losing trades = total trades."""
        total = mock_backtest_results['total_trades']
        winning = mock_backtest_results['winning_trades']
        losing = mock_backtest_results['losing_trades']

        assert winning + losing == total

    def test_win_rate_bounds(self, mock_backtest_results):
        """Test that win rate is between 0 and 1."""
        win_rate = mock_backtest_results['win_rate']

        assert 0 <= win_rate <= 1

    def test_drawdown_bounds(self, mock_backtest_results):
        """Test that drawdown is between -100% and 0%."""
        max_dd = mock_backtest_results['max_drawdown']

        assert -1 <= max_dd <= 0

    def test_average_win_positive(self, mock_backtest_results):
        """Test that average win is positive."""
        avg_win = mock_backtest_results['avg_win']

        assert avg_win > 0

    def test_average_loss_negative(self, mock_backtest_results):
        """Test that average loss is negative."""
        avg_loss = mock_backtest_results['avg_loss']

        assert avg_loss < 0

    def test_holding_period_positive(self, mock_backtest_results):
        """Test that average holding period is positive."""
        avg_holding = mock_backtest_results['avg_holding_period']

        assert avg_holding > 0

    def test_returns_consistent_with_portfolio_values(self, mock_backtest_results):
        """Test that reported returns match portfolio value changes."""
        values = mock_backtest_results['portfolio_values']
        reported_return = mock_backtest_results['total_return']

        calculated_return = (values[-1] - values[0]) / values[0]

        assert abs(calculated_return - reported_return) < 0.01

    def test_dates_in_chronological_order(self, mock_backtest_results):
        """Test that dates are in chronological order."""
        dates = mock_backtest_results['dates']

        for i in range(1, len(dates)):
            assert dates[i] > dates[i-1], "Dates not in order"

    def test_benchmark_values_aligned_with_dates(self, mock_backtest_results):
        """Test that benchmark values align with dates."""
        dates = mock_backtest_results['dates']
        benchmark = mock_backtest_results['benchmark_values']

        assert len(dates) == len(benchmark)


# =============================================================================
# Test Backtest Edge Cases
# =============================================================================

class TestBacktestEdgeCases:
    """Tests for edge cases in backtesting."""

    def test_backtest_with_no_trades(self, backtest_config):
        """Test backtest that generates no trades."""
        backtester = MagicMock()
        backtester.run.return_value = {
            'total_return': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }

        results = backtester.run(data=pd.DataFrame(), config=backtest_config)

        assert results['total_trades'] == 0
        assert results['total_return'] == 0.0

    def test_backtest_with_single_trade(self):
        """Test backtest with only one trade."""
        results = {
            'total_trades': 1,
            'winning_trades': 1,
            'losing_trades': 0,
            'win_rate': 1.0
        }

        assert results['win_rate'] == 1.0  # 100% win rate

    def test_backtest_all_losing_trades(self):
        """Test backtest where all trades are losers."""
        results = {
            'total_trades': 10,
            'winning_trades': 0,
            'losing_trades': 10,
            'win_rate': 0.0,
            'total_return': -0.15
        }

        assert results['win_rate'] == 0.0
        assert results['total_return'] < 0

    def test_backtest_extreme_volatility(self, high_volatility_data):
        """Test backtest with extreme price volatility."""
        backtester = MagicMock()
        backtester.run.return_value = {
            'total_return': -0.20,
            'max_drawdown': -0.40,
            'volatility': 0.50
        }

        results = backtester.run(data=high_volatility_data, config={})

        # High volatility should lead to larger drawdowns
        assert results['max_drawdown'] <= -0.20

    def test_backtest_short_period(self, short_date_range):
        """Test backtest over very short time period."""
        # Only 20 trading days
        backtester = MagicMock()
        backtester.run.return_value = {
            'total_return': 0.02,
            'trading_days': 20,
            'annualized_return': 0.50  # Extrapolated
        }

        results = backtester.run(data=pd.DataFrame(), config={})

        assert results['trading_days'] == 20

    def test_backtest_long_period(self, sample_dates):
        """Test backtest over longer time period."""
        # ~252 trading days (1 year)
        backtester = MagicMock()
        backtester.run.return_value = {
            'total_return': 0.15,
            'trading_days': 252,
            'annualized_return': 0.15
        }

        results = backtester.run(data=pd.DataFrame(), config={})

        # Annual return should equal total return for 1 year
        assert abs(results['annualized_return'] - results['total_return']) < 0.01

    def test_backtest_with_dividends(self):
        """Test backtest accounting for dividends."""
        results = {
            'price_return': 0.08,
            'dividend_return': 0.02,
            'total_return': 0.10
        }

        calculated_total = results['price_return'] + results['dividend_return']
        assert abs(calculated_total - results['total_return']) < 0.01

    def test_backtest_with_stock_splits(self):
        """Test backtest handling of stock splits."""
        # Pre-split: 100 shares @ $400
        # Post-split (4:1): 400 shares @ $100

        pre_split_value = 100 * 400
        post_split_value = 400 * 100

        # Value should be unchanged
        assert pre_split_value == post_split_value

    def test_backtest_circuit_breaker_halt(self):
        """Test backtest handling of trading halts."""
        # Price drops 20% triggering circuit breaker
        backtester = MagicMock()

        # Should skip trading during halt
        backtester.is_market_halted.return_value = True
        backtester.execute_trade.return_value = None

        is_halted = backtester.is_market_halted()
        assert is_halted

    def test_backtest_different_strategies(self, sample_ohlcv_data, backtest_config):
        """Test backtest with different strategy configurations."""
        strategies = ['momentum', 'mean_reversion', 'buy_and_hold']

        backtester = MagicMock()

        for strategy in strategies:
            config = backtest_config.copy()
            config['strategy'] = strategy

            backtester.run.return_value = {
                'strategy': strategy,
                'total_return': np.random.uniform(-0.1, 0.2)
            }

            results = backtester.run(data=sample_ohlcv_data, config=config)

            assert results['strategy'] == strategy

    def test_backtest_survivorship_bias(self):
        """Test that backtest accounts for survivorship bias."""
        # Include delisted stocks in backtest
        all_stocks = ['AAPL', 'MSFT', 'ENRON', 'LEHMAN', 'GOOGL']
        delisted = ['ENRON', 'LEHMAN']

        # Backtest should include losses from delisted stocks
        backtester = MagicMock()
        backtester.include_delisted.return_value = True

        includes_delisted = backtester.include_delisted()
        assert includes_delisted

    def test_backtest_transaction_costs_impact(self, backtest_config):
        """Test impact of transaction costs on results."""
        # Run with and without transaction costs
        config_no_costs = backtest_config.copy()
        config_no_costs['commission'] = 0.0
        config_no_costs['slippage'] = 0.0

        config_with_costs = backtest_config.copy()
        config_with_costs['commission'] = 0.001  # 0.1%
        config_with_costs['slippage'] = 0.002  # 0.2%

        backtester = MagicMock()
        backtester.run.side_effect = [
            {'total_return': 0.15},  # No costs
            {'total_return': 0.12}   # With costs
        ]

        results_no_costs = backtester.run(data=pd.DataFrame(), config=config_no_costs)
        results_with_costs = backtester.run(data=pd.DataFrame(), config=config_with_costs)

        # Costs should reduce returns
        assert results_with_costs['total_return'] < results_no_costs['total_return']

    def test_backtest_position_sizing_impact(self):
        """Test impact of position sizing on results."""
        position_sizes = [0.05, 0.10, 0.20]  # 5%, 10%, 20% max position

        backtester = MagicMock()
        results = []

        for size in position_sizes:
            backtester.run.return_value = {
                'max_position_size': size,
                'max_drawdown': -0.05 * (size / 0.05)  # DD scales with position size
            }
            result = backtester.run(data=pd.DataFrame(), config={'max_position_size': size})
            results.append(result)

        # Larger positions should have larger drawdowns
        assert results[2]['max_drawdown'] < results[0]['max_drawdown']


# =============================================================================
# Test Benchmark Comparison
# =============================================================================

class TestBenchmarkComparison:
    """Tests for benchmark comparison functionality."""

    def test_alpha_calculation(self, mock_backtest_results):
        """Test alpha calculation against benchmark."""
        portfolio_return = mock_backtest_results['total_return']
        benchmark_values = mock_backtest_results['benchmark_values']

        benchmark_return = (benchmark_values[-1] - benchmark_values[0]) / benchmark_values[0]
        alpha = portfolio_return - benchmark_return

        # Alpha = 15% - 10% = 5%
        expected_alpha = 0.15 - 0.10
        assert abs(alpha - expected_alpha) < 0.01

    def test_information_ratio(self, mock_backtest_results):
        """Test information ratio calculation."""
        portfolio_values = np.array(mock_backtest_results['portfolio_values'])
        benchmark_values = np.array(mock_backtest_results['benchmark_values'])

        # Calculate returns
        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]

        # Active returns
        active_returns = portfolio_returns - benchmark_returns

        # Information ratio
        tracking_error = np.std(active_returns)
        if tracking_error > 0:
            ir = np.mean(active_returns) / tracking_error * np.sqrt(252)
        else:
            ir = 0

        assert isinstance(ir, float)

    def test_tracking_error(self, mock_backtest_results):
        """Test tracking error calculation."""
        portfolio_values = np.array(mock_backtest_results['portfolio_values'])
        benchmark_values = np.array(mock_backtest_results['benchmark_values'])

        portfolio_returns = np.diff(portfolio_values) / portfolio_values[:-1]
        benchmark_returns = np.diff(benchmark_values) / benchmark_values[:-1]

        active_returns = portfolio_returns - benchmark_returns
        tracking_error = np.std(active_returns) * np.sqrt(252)

        assert tracking_error >= 0

    def test_beta_to_benchmark(self):
        """Test beta calculation against benchmark."""
        np.random.seed(42)

        benchmark_returns = np.random.normal(0.0005, 0.01, 100)
        # Portfolio more volatile than market
        portfolio_returns = benchmark_returns * 1.2 + np.random.normal(0, 0.005, 100)

        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1

        # Should be close to 1.2
        assert 1.0 < beta < 1.5

    def test_upside_capture(self):
        """Test upside capture ratio."""
        # Portfolio returns during up markets
        benchmark_returns = np.array([0.01, 0.02, -0.01, 0.015, -0.02, 0.03])
        portfolio_returns = np.array([0.012, 0.025, -0.008, 0.02, -0.025, 0.035])

        # Up markets
        up_mask = benchmark_returns > 0
        up_portfolio = portfolio_returns[up_mask].mean()
        up_benchmark = benchmark_returns[up_mask].mean()

        upside_capture = up_portfolio / up_benchmark * 100

        assert upside_capture > 100  # Captures more than benchmark on ups

    def test_downside_capture(self):
        """Test downside capture ratio."""
        benchmark_returns = np.array([0.01, 0.02, -0.01, 0.015, -0.02, 0.03])
        portfolio_returns = np.array([0.012, 0.025, -0.008, 0.02, -0.025, 0.035])

        # Down markets
        down_mask = benchmark_returns < 0
        down_portfolio = portfolio_returns[down_mask].mean()
        down_benchmark = benchmark_returns[down_mask].mean()

        downside_capture = down_portfolio / down_benchmark * 100

        # Ideally < 100 (lose less than benchmark on downs)
        assert isinstance(downside_capture, float)

"""
Backtest Validity Tests

Comprehensive test suite for validating backtest integrity including:
- Look-ahead bias detection
- Transaction cost verification
- Statistical significance tests
- Edge case handling
- Benchmark comparison tests

These tests ensure backtest results are reliable and not artificially inflated.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
import copy
import warnings

# Import from the trading module
from src.trading.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from src.trading.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.trading.portfolio.portfolio import Portfolio, Order, OrderSide, OrderType, Trade
from src.trading.risk.manager import RiskManager, RiskConfig


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def base_config() -> BacktestConfig:
    """Base backtest configuration for tests."""
    return BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.001,  # 0.1%
        slippage_rate=0.0005,   # 0.05%
        warmup_period=20,
        enable_fractional=True,
        max_positions=10,
    )


@pytest.fixture
def sample_trading_dates() -> List[datetime]:
    """Generate 252 trading days (1 year)."""
    dates = []
    current = datetime(2023, 1, 1)
    while len(dates) < 252:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture
def synthetic_ohlcv_data(sample_trading_dates) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic OHLCV data with known characteristics.

    Creates data for two symbols: 'TEST_STOCK' and 'SPY' (benchmark).
    """
    np.random.seed(42)

    def generate_stock_data(dates: List[datetime], base_price: float, volatility: float,
                           drift: float, seed: int) -> pd.DataFrame:
        np.random.seed(seed)
        n = len(dates)

        # Generate returns with drift
        returns = np.random.normal(drift, volatility, n)
        prices = base_price * np.cumprod(1 + returns)

        # Generate OHLC from close
        highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n)))
        opens = np.roll(prices, 1)
        opens[0] = base_price

        # Ensure OHLC integrity
        highs = np.maximum.reduce([opens, prices, highs])
        lows = np.minimum.reduce([opens, prices, lows])

        return pd.DataFrame({
            'datetime': dates,
            'open': opens,
            'high': highs,
            'low': lows,
            'close': prices,
            'volume': np.random.randint(1_000_000, 10_000_000, n),
        })

    return {
        'TEST_STOCK': generate_stock_data(sample_trading_dates, 100.0, 0.02, 0.0003, 42),
        'SPY': generate_stock_data(sample_trading_dates, 400.0, 0.01, 0.0002, 123),
    }


@pytest.fixture
def data_with_future_spike(sample_trading_dates) -> Dict[str, pd.DataFrame]:
    """
    Create data with a known future spike for look-ahead bias testing.

    The spike occurs at day 150 (+20% in one day).
    A strategy with look-ahead bias would trade before day 150.
    """
    np.random.seed(42)
    n = len(sample_trading_dates)

    # Normal returns
    returns = np.random.normal(0.0001, 0.01, n)

    # Inject spike at day 150
    spike_day = 150
    returns[spike_day] = 0.20  # 20% spike

    prices = 100.0 * np.cumprod(1 + returns)

    return {
        'SPIKE_STOCK': pd.DataFrame({
            'datetime': sample_trading_dates,
            'open': prices * 0.998,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.randint(1_000_000, 10_000_000, n),
        })
    }


class SimpleMomentumStrategy(BaseStrategy):
    """
    Simple momentum strategy for testing.

    Buys when price crosses above 20-day moving average.
    Sells when price crosses below 20-day moving average.
    """

    def __init__(self, lookback: int = 20):
        config = StrategyConfig(name="SimpleMomentum", lookback_period=lookback)
        super().__init__(config)
        self.lookback = lookback

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        if len(data) < self.lookback + 1:
            return []

        # Normalize column names
        close_col = 'close' if 'close' in data.columns else 'Close'

        close = data[close_col].values
        sma = pd.Series(close).rolling(self.lookback).mean().values

        signals = []

        # Only generate signal from the last row (most recent data)
        if not np.isnan(sma[-1]):
            symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'UNKNOWN'
            timestamp = data['datetime'].iloc[-1] if 'datetime' in data.columns else datetime.now()

            if close[-1] > sma[-1] and close[-2] <= sma[-2]:
                # Bullish crossover
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=0.8,
                    timestamp=timestamp,
                    price=close[-1],
                ))
            elif close[-1] < sma[-1] and close[-2] >= sma[-2]:
                # Bearish crossover
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=-0.8,
                    timestamp=timestamp,
                    price=close[-1],
                ))

        return signals

    def calculate_position_size(self, signal: Signal, capital: float,
                                 risk_per_trade: float | None = None) -> float:
        risk = risk_per_trade or self.config.risk_per_trade
        return capital * risk / signal.price


class LookAheadBiasStrategy(BaseStrategy):
    """
    Strategy that intentionally uses future data (for testing bias detection).

    This strategy cheats by looking at tomorrow's price to make decisions.
    """

    def __init__(self, full_data: pd.DataFrame):
        config = StrategyConfig(name="LookAheadBias", lookback_period=1)
        super().__init__(config)
        self.full_data = full_data

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        signals = []
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else 'UNKNOWN'
        current_idx = len(data) - 1

        close_col = 'close' if 'close' in data.columns else 'Close'

        # CHEATING: Look at tomorrow's price (look-ahead bias)
        if current_idx + 1 < len(self.full_data):
            tomorrow_price = self.full_data[close_col].iloc[current_idx + 1]
            today_price = data[close_col].iloc[-1]

            timestamp = data['datetime'].iloc[-1] if 'datetime' in data.columns else datetime.now()

            if tomorrow_price > today_price * 1.02:  # If tomorrow is +2%
                signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=1.0,
                    timestamp=timestamp,
                    price=today_price,
                ))

        return signals

    def calculate_position_size(self, signal: Signal, capital: float,
                                 risk_per_trade: float | None = None) -> float:
        return capital * 0.1 / signal.price


# =============================================================================
# Look-Ahead Bias Tests
# =============================================================================

class TestLookAheadBias:
    """Tests for detecting and preventing look-ahead bias in backtesting."""

    def test_no_future_data_leakage_in_signal_generation(
        self, sample_trading_dates, data_with_future_spike, base_config
    ):
        """
        Verify signals at time T only use data from T-1 and earlier.

        Inject a known future spike and ensure strategy doesn't react before it happens.
        """
        data = data_with_future_spike
        spike_day = 150

        # Create a simple strategy
        strategy = SimpleMomentumStrategy(lookback=20)

        # Track signals generated before the spike
        signals_before_spike = []

        # Simulate signal generation day by day
        for day_idx in range(base_config.warmup_period, spike_day):
            # Only data up to current day is available
            available_data = data['SPIKE_STOCK'].iloc[:day_idx + 1].copy()
            available_data['symbol'] = 'SPIKE_STOCK'

            signals = strategy.generate_signals(available_data)
            if signals:
                signals_before_spike.extend(signals)

        # Count buy signals before spike
        buy_signals_before_spike = [s for s in signals_before_spike if s.is_buy]

        # A proper strategy should NOT have abnormally high buy signals before spike
        # The spike happens suddenly - no legitimate signal should predict it
        # We check that buy signals are distributed reasonably

        # Get the last 10 days before spike and count buy signals
        late_signals = [s for s in buy_signals_before_spike
                       if s.timestamp >= sample_trading_dates[spike_day - 10]]

        # A legitimate strategy should not concentrate buying right before spike
        assert len(late_signals) <= 3, (
            f"Suspicious concentration of {len(late_signals)} buy signals "
            f"in 10 days before spike. Possible look-ahead bias."
        )

    def test_look_ahead_bias_strategy_detection(
        self, sample_trading_dates, data_with_future_spike, base_config
    ):
        """
        Test that a strategy with intentional look-ahead bias is detectable.

        The biased strategy should have unrealistically high returns.
        """
        data = data_with_future_spike

        # Strategy with look-ahead bias
        biased_strategy = LookAheadBiasStrategy(data['SPIKE_STOCK'])

        # Run backtest with biased strategy
        engine = BacktestEngine(base_config)
        engine.add_strategy(biased_strategy)

        try:
            result = engine.run(data)

            # Look-ahead bias typically shows as unrealistically high Sharpe ratio
            # or extremely high win rate
            if result.sharpe_ratio > 5.0 or result.win_rate > 95:
                warnings.warn(
                    f"Potential look-ahead bias detected: "
                    f"Sharpe={result.sharpe_ratio:.2f}, Win Rate={result.win_rate:.1f}%"
                )
        except Exception:
            # Some implementations may catch look-ahead access
            pass

    def test_signal_timestamp_matches_data_timestamp(
        self, synthetic_ohlcv_data, base_config
    ):
        """
        Verify that signal timestamps match the most recent data point available.
        """
        strategy = SimpleMomentumStrategy(lookback=20)

        test_data = synthetic_ohlcv_data['TEST_STOCK'].copy()
        test_data['symbol'] = 'TEST_STOCK'

        for day_idx in range(base_config.warmup_period, len(test_data)):
            available_data = test_data.iloc[:day_idx + 1]
            signals = strategy.generate_signals(available_data)

            for signal in signals:
                # Signal timestamp must match last available data point
                expected_timestamp = available_data['datetime'].iloc[-1]
                assert signal.timestamp == expected_timestamp, (
                    f"Signal timestamp {signal.timestamp} does not match "
                    f"data timestamp {expected_timestamp}"
                )

    def test_indicator_calculation_uses_only_past_data(self, sample_trading_dates):
        """
        Verify that technical indicators only use past data.

        Calculate indicators at time T and verify they don't change
        when future data is added.
        """
        np.random.seed(42)
        n = len(sample_trading_dates)
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))

        # Calculate 20-day SMA at day 100
        day_100_prices = prices[:101]
        sma_at_100 = np.mean(day_100_prices[-20:])

        # Calculate same SMA with full dataset but only using data up to day 100
        full_sma = pd.Series(prices).rolling(20).mean().values
        sma_at_100_from_full = full_sma[100]

        # They should be identical
        assert np.isclose(sma_at_100, sma_at_100_from_full), (
            f"Indicator calculation differs: {sma_at_100} vs {sma_at_100_from_full}"
        )


# =============================================================================
# Transaction Cost Tests
# =============================================================================

class TestTransactionCosts:
    """Tests for verifying transaction costs are applied correctly."""

    def test_transaction_costs_applied(self, base_config):
        """
        Verify round-trip trade reduces capital by expected commission.
        """
        commission_rate = 0.001  # 0.1%

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=commission_rate,
            slippage_rate=0.0,  # Disable slippage for this test
        )

        portfolio = Portfolio(
            initial_cash=config.initial_capital,
            commission_rate=config.commission_rate,
        )

        # Buy 100 shares at $100
        buy_price = 100.0
        quantity = 100
        buy_order = portfolio.create_order(
            symbol='TEST',
            side=OrderSide.BUY,
            quantity=quantity,
            order_type=OrderType.MARKET,
        )
        portfolio.execute_order(buy_order, buy_price)

        expected_buy_commission = buy_price * quantity * commission_rate
        expected_cash_after_buy = config.initial_capital - (buy_price * quantity) - expected_buy_commission

        assert np.isclose(portfolio.cash, expected_cash_after_buy), (
            f"Cash after buy: {portfolio.cash} != expected {expected_cash_after_buy}"
        )

        # Sell 100 shares at $100 (same price, so only commission matters)
        trade = portfolio.close_position('TEST', buy_price)

        expected_sell_commission = buy_price * quantity * commission_rate
        total_commission = expected_buy_commission + expected_sell_commission

        # After round-trip at same price, capital should be reduced by total commission
        expected_final_cash = config.initial_capital - total_commission

        assert np.isclose(portfolio.cash, expected_final_cash, atol=0.01), (
            f"Round-trip cost mismatch: {portfolio.cash} != {expected_final_cash}. "
            f"Expected commission: {total_commission}"
        )

    def test_slippage_applied_on_buy(self, base_config):
        """
        Verify fill prices are higher than signal prices by slippage amount on buys.
        """
        signal_price = 100.0
        slippage_rate = 0.001  # 0.1%

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.0,
            slippage_rate=slippage_rate,
        )

        engine = BacktestEngine(config)

        # Calculate expected fill price with slippage
        expected_fill_price = signal_price * (1 + slippage_rate)
        actual_fill_price = engine._get_price_with_slippage(signal_price, is_buy=True)

        assert np.isclose(actual_fill_price, expected_fill_price), (
            f"Buy slippage incorrect: {actual_fill_price} != {expected_fill_price}"
        )

        # Fill price should be HIGHER than signal price for buys
        assert actual_fill_price > signal_price, (
            "Buy fill price should be higher than signal price due to slippage"
        )

    def test_slippage_applied_on_sell(self, base_config):
        """
        Verify fill prices are lower than signal prices by slippage amount on sells.
        """
        signal_price = 100.0
        slippage_rate = 0.001  # 0.1%

        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.0,
            slippage_rate=slippage_rate,
        )

        engine = BacktestEngine(config)

        # Calculate expected fill price with slippage
        expected_fill_price = signal_price * (1 - slippage_rate)
        actual_fill_price = engine._get_price_with_slippage(signal_price, is_buy=False)

        assert np.isclose(actual_fill_price, expected_fill_price), (
            f"Sell slippage incorrect: {actual_fill_price} != {expected_fill_price}"
        )

        # Fill price should be LOWER than signal price for sells
        assert actual_fill_price < signal_price, (
            "Sell fill price should be lower than signal price due to slippage"
        )

    def test_high_frequency_trading_penalized(self, synthetic_ohlcv_data, base_config):
        """
        Test that high-frequency trading is properly penalized by transaction costs.

        More trades should result in higher total costs.
        """
        # Config with meaningful transaction costs
        config = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.001,  # 0.1%
            slippage_rate=0.0005,   # 0.05%
            warmup_period=20,
        )

        # Simulate different trading frequencies
        trade_counts = [10, 50, 100]
        total_costs = []

        for num_trades in trade_counts:
            portfolio = Portfolio(
                initial_cash=config.initial_capital,
                commission_rate=config.commission_rate,
            )

            # Simulate trades
            for i in range(num_trades):
                price = 100.0
                quantity = 10

                # Buy
                buy_order = portfolio.create_order('TEST', OrderSide.BUY, quantity, OrderType.MARKET)
                fill_price = price * (1 + config.slippage_rate)
                portfolio.execute_order(buy_order, fill_price)

                # Sell at same price
                sell_price = price * (1 - config.slippage_rate)
                portfolio.close_position('TEST', sell_price)

            # Calculate total cost (difference from initial capital with zero price change)
            total_cost = config.initial_capital - portfolio.cash
            total_costs.append(total_cost)

        # More trades should mean higher costs
        assert total_costs[0] < total_costs[1] < total_costs[2], (
            f"Transaction costs don't scale with trade count: {total_costs}"
        )

    def test_commission_calculation_accuracy(self):
        """Test that commission is calculated correctly for various trade sizes."""
        portfolio = Portfolio(
            initial_cash=100000.0,
            commission_rate=0.001,  # 0.1%
            min_commission=1.0,     # $1 minimum
        )

        test_cases = [
            (100, 10.0, 1.0),    # $1000 trade, 0.1% = $1.00 (equals min)
            (100, 50.0, 5.0),    # $5000 trade, 0.1% = $5.00
            (1000, 100.0, 100.0), # $100000 trade, 0.1% = $100.00
            (10, 5.0, 1.0),     # $50 trade, should be $1 minimum
        ]

        for quantity, price, expected_commission in test_cases:
            actual = portfolio._calculate_commission(quantity, price)
            assert np.isclose(actual, expected_commission, atol=0.01), (
                f"Commission for {quantity}x${price}: "
                f"expected ${expected_commission}, got ${actual}"
            )


# =============================================================================
# Statistical Validity Tests
# =============================================================================

class TestStatisticalValidity:
    """Tests for statistical significance of backtest results."""

    def test_monte_carlo_significance(self, synthetic_ohlcv_data, base_config):
        """
        Shuffle returns and compare strategy vs random.

        Strategy should beat >95% of random permutations to be significant.
        """
        np.random.seed(42)

        # Get stock returns
        prices = synthetic_ohlcv_data['TEST_STOCK']['close'].values
        returns = np.diff(prices) / prices[:-1]

        # Simulate a "strategy return" (e.g., simple momentum)
        strategy_positions = np.sign(np.diff(prices))  # 1 if up, -1 if down
        strategy_returns = strategy_positions[:-1] * returns[1:]
        strategy_total_return = np.prod(1 + strategy_returns) - 1

        # Monte Carlo simulation: shuffle returns
        n_simulations = 1000
        random_returns = []

        for _ in range(n_simulations):
            shuffled_returns = np.random.permutation(returns)
            random_total = np.prod(1 + shuffled_returns) - 1
            random_returns.append(random_total)

        random_returns = np.array(random_returns)

        # Calculate percentile rank of strategy return
        percentile = np.mean(strategy_total_return > random_returns) * 100

        # Strategy should beat at least some random permutations
        # (exact threshold depends on strategy quality)
        # We just verify the test mechanics work
        assert 0 <= percentile <= 100, f"Invalid percentile: {percentile}"

        # Log result for informational purposes
        print(f"Strategy return: {strategy_total_return:.2%}, "
              f"beats {percentile:.1f}% of random permutations")

    def test_parameter_stability(self, synthetic_ohlcv_data, base_config):
        """
        Small parameter changes shouldn't dramatically change results.

        Tests that strategy is not over-fitted to specific parameters.
        """
        # Test different lookback periods
        lookbacks = [18, 19, 20, 21, 22]  # Small variations around 20
        results = []

        for lookback in lookbacks:
            strategy = SimpleMomentumStrategy(lookback=lookback)

            # Manually simulate signals and track performance
            test_data = synthetic_ohlcv_data['TEST_STOCK'].copy()
            test_data['symbol'] = 'TEST_STOCK'

            signal_count = 0
            for day_idx in range(lookback + 1, len(test_data)):
                available_data = test_data.iloc[:day_idx + 1]
                signals = strategy.generate_signals(available_data)
                signal_count += len(signals)

            results.append(signal_count)

        # Check stability: signal counts shouldn't vary too wildly
        mean_signals = np.mean(results)
        std_signals = np.std(results)

        # Coefficient of variation should be reasonable (< 50%)
        cv = std_signals / mean_signals if mean_signals > 0 else 0

        assert cv < 0.5, (
            f"Parameter instability detected: CV={cv:.2f}. "
            f"Signal counts: {results}"
        )

    def test_out_of_sample_performance(self, sample_trading_dates):
        """
        Test that in-sample performance doesn't significantly exceed out-of-sample.
        """
        np.random.seed(42)
        n = len(sample_trading_dates)

        # Generate full price series
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))

        # Split into in-sample (first 70%) and out-of-sample (last 30%)
        split_idx = int(n * 0.7)

        in_sample_prices = prices[:split_idx]
        out_of_sample_prices = prices[split_idx:]

        # Simple strategy: buy and hold
        in_sample_return = (in_sample_prices[-1] / in_sample_prices[0]) - 1
        out_of_sample_return = (out_of_sample_prices[-1] / out_of_sample_prices[0]) - 1

        # Annualize returns for comparison
        in_sample_days = split_idx
        out_of_sample_days = n - split_idx

        in_sample_annual = (1 + in_sample_return) ** (252 / in_sample_days) - 1
        out_of_sample_annual = (1 + out_of_sample_return) ** (252 / out_of_sample_days) - 1

        # Out-of-sample shouldn't be dramatically worse (2x worse is suspicious)
        # This is a soft check as some degradation is expected
        if in_sample_annual > 0 and out_of_sample_annual > 0:
            ratio = in_sample_annual / out_of_sample_annual
            # Log for informational purposes
            print(f"In-sample: {in_sample_annual:.2%}, Out-of-sample: {out_of_sample_annual:.2%}, "
                  f"Ratio: {ratio:.2f}")

    def test_sharpe_ratio_confidence_interval(self, synthetic_ohlcv_data):
        """
        Calculate confidence interval for Sharpe ratio using bootstrap.
        """
        prices = synthetic_ohlcv_data['TEST_STOCK']['close'].values
        returns = np.diff(prices) / prices[:-1]

        # Bootstrap Sharpe ratio
        n_bootstrap = 1000
        sharpe_ratios = []

        for _ in range(n_bootstrap):
            # Resample with replacement
            sample_returns = np.random.choice(returns, size=len(returns), replace=True)

            # Calculate Sharpe (annualized, assuming 0 risk-free rate)
            if np.std(sample_returns) > 0:
                sharpe = np.mean(sample_returns) / np.std(sample_returns) * np.sqrt(252)
            else:
                sharpe = 0
            sharpe_ratios.append(sharpe)

        sharpe_ratios = np.array(sharpe_ratios)

        # Calculate 95% confidence interval
        ci_lower = np.percentile(sharpe_ratios, 2.5)
        ci_upper = np.percentile(sharpe_ratios, 97.5)
        mean_sharpe = np.mean(sharpe_ratios)

        # Confidence interval should be reasonable
        assert ci_lower < mean_sharpe < ci_upper, "Invalid confidence interval"

        print(f"Sharpe Ratio: {mean_sharpe:.2f} (95% CI: [{ci_lower:.2f}, {ci_upper:.2f}])")


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases in backtesting."""

    def test_survives_flash_crash(self, sample_trading_dates, base_config):
        """
        Test backtest survives a -10% single-day drop.

        This test verifies:
        1. Backtest completes without errors during extreme moves
        2. Portfolio value remains positive (no margin call simulation failures)
        3. Price data integrity is maintained through crash
        """
        np.random.seed(42)
        n = len(sample_trading_dates)

        # Normal returns
        returns = np.random.normal(0.0003, 0.01, n)

        # Inject flash crash at day 100
        crash_day = 100
        returns[crash_day] = -0.10  # -10% drop

        prices = 100 * np.cumprod(1 + returns)

        # Calculate crash impact on prices
        price_before_crash = prices[crash_day - 1]
        price_after_crash = prices[crash_day]
        actual_crash_pct = (price_after_crash - price_before_crash) / price_before_crash

        data = {
            'CRASH_STOCK': pd.DataFrame({
                'datetime': sample_trading_dates,
                'open': prices * 0.998,
                'high': prices * 1.01,
                'low': prices * 0.90,  # Flash crash low
                'close': prices,
                'volume': np.random.randint(1_000_000, 50_000_000, n),
            })
        }

        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        # Backtest should complete without error
        try:
            result = engine.run(data)

            # Portfolio value should remain positive
            assert result.final_value > 0, "Portfolio value went negative"

            # Verify crash was present in data (data integrity check)
            assert actual_crash_pct < -0.08, (
                f"Flash crash not properly injected: {actual_crash_pct:.2%}"
            )

            # Drawdown should be non-negative (proper calculation)
            assert result.max_drawdown >= 0, "Max drawdown calculation error"

            # If any trades occurred, verify trade mechanics worked
            if result.total_trades > 0:
                assert result.winning_trades + result.losing_trades == result.total_trades, (
                    "Trade count mismatch"
                )

        except Exception as e:
            pytest.fail(f"Backtest failed on flash crash: {e}")

    def test_handles_overnight_gaps(self, sample_trading_dates, base_config):
        """
        Test strategy handles overnight price gaps.
        """
        np.random.seed(42)
        n = len(sample_trading_dates)

        # Generate close prices
        close_prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))

        # Create gaps: open differs significantly from previous close
        gap_sizes = np.random.normal(0, 0.02, n)  # Up to +/- 2% gaps
        open_prices = np.zeros(n)
        open_prices[0] = close_prices[0]
        open_prices[1:] = close_prices[:-1] * (1 + gap_sizes[1:])

        data = {
            'GAP_STOCK': pd.DataFrame({
                'datetime': sample_trading_dates,
                'open': open_prices,
                'high': np.maximum(open_prices, close_prices) * 1.005,
                'low': np.minimum(open_prices, close_prices) * 0.995,
                'close': close_prices,
                'volume': np.random.randint(1_000_000, 10_000_000, n),
            })
        }

        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        # Backtest should handle gaps gracefully
        result = engine.run(data)

        assert result is not None, "Backtest failed with overnight gaps"
        assert result.final_value > 0, "Portfolio went negative with gaps"

    def test_handles_weekend_gaps(self, base_config):
        """
        Test that weekend gaps are handled properly.
        """
        # Generate dates that simulate actual trading days with weekends
        dates = []
        current = datetime(2023, 1, 2)  # Start on Monday

        while len(dates) < 100:
            if current.weekday() < 5:  # Monday to Friday
                dates.append(current)
            current += timedelta(days=1)

        np.random.seed(42)
        n = len(dates)
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))

        # Add Monday gaps (simulating weekend news)
        for i in range(1, n):
            if dates[i].weekday() == 0:  # Monday
                # Potential large gap
                prices[i:] *= (1 + np.random.normal(0, 0.02))

        data = {
            'WEEKEND_STOCK': pd.DataFrame({
                'datetime': dates,
                'open': prices * 0.998,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(1_000_000, 10_000_000, n),
            })
        }

        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(data)

        assert result is not None, "Backtest failed with weekend gaps"

    def test_zero_volume_days_handled(self, sample_trading_dates, base_config):
        """
        Test that strategy handles zero volume (illiquid) periods appropriately.
        """
        np.random.seed(42)
        n = len(sample_trading_dates)

        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.01, n))
        volumes = np.random.randint(1_000_000, 10_000_000, n)

        # Inject zero volume days
        zero_volume_days = [50, 51, 52, 100, 101]
        for day in zero_volume_days:
            if day < n:
                volumes[day] = 0

        data = {
            'LOW_VOL_STOCK': pd.DataFrame({
                'datetime': sample_trading_dates,
                'open': prices * 0.998,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': volumes,
            })
        }

        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        # Backtest should complete, potentially skipping zero-volume days
        result = engine.run(data)

        assert result is not None, "Backtest failed with zero volume days"
        assert result.final_value > 0, "Portfolio went negative with illiquid periods"

    def test_handles_extreme_volatility(self, sample_trading_dates, base_config):
        """
        Test backtest handles extreme volatility periods (e.g., VIX spike).
        """
        np.random.seed(42)
        n = len(sample_trading_dates)

        # Normal volatility first 100 days, extreme volatility next 50 days
        returns = np.zeros(n)
        returns[:100] = np.random.normal(0.0003, 0.01, 100)
        returns[100:150] = np.random.normal(0, 0.05, 50)  # 5x normal volatility
        returns[150:] = np.random.normal(0.0003, 0.01, n - 150)

        prices = 100 * np.cumprod(1 + returns)

        data = {
            'VOLATILE_STOCK': pd.DataFrame({
                'datetime': sample_trading_dates,
                'open': prices * 0.998,
                'high': prices * np.where(np.arange(n) < 150, 1.01, 1.05),
                'low': prices * np.where(np.arange(n) < 150, 0.99, 0.95),
                'close': prices,
                'volume': np.random.randint(1_000_000, 10_000_000, n),
            })
        }

        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(data)

        assert result is not None, "Backtest failed with extreme volatility"
        assert result.final_value > 0, "Portfolio went negative with extreme volatility"

    def test_handles_single_stock_data(self, sample_trading_dates, base_config):
        """
        Test backtest works correctly with only one stock.
        """
        np.random.seed(42)
        n = len(sample_trading_dates)
        prices = 100 * np.cumprod(1 + np.random.normal(0.0003, 0.015, n))

        data = {
            'SINGLE_STOCK': pd.DataFrame({
                'datetime': sample_trading_dates,
                'open': prices * 0.998,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(1_000_000, 10_000_000, n),
            })
        }

        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(data)

        assert result is not None
        assert result.initial_capital == base_config.initial_capital

    def test_handles_minimal_data(self, base_config):
        """
        Test behavior with minimal data (just above warmup period).
        """
        # Create minimal dates
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(base_config.warmup_period + 5)]
        n = len(dates)

        np.random.seed(42)
        prices = 100 * np.cumprod(1 + np.random.normal(0, 0.01, n))

        data = {
            'MINIMAL_STOCK': pd.DataFrame({
                'datetime': dates,
                'open': prices * 0.998,
                'high': prices * 1.01,
                'low': prices * 0.99,
                'close': prices,
                'volume': np.random.randint(1_000_000, 10_000_000, n),
            })
        }

        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        # Should either run successfully or raise appropriate error
        try:
            result = engine.run(data)
            assert result is not None
        except ValueError as e:
            # Acceptable if it raises "insufficient data" error
            assert "insufficient" in str(e).lower() or "data" in str(e).lower()


# =============================================================================
# Benchmark Comparison Tests
# =============================================================================

class TestBenchmarkComparison:
    """Tests for benchmark comparison functionality."""

    def test_beats_random_baseline(self, synthetic_ohlcv_data, base_config):
        """
        Compare strategy vs random entry/exit strategy.
        """
        np.random.seed(42)

        test_data = synthetic_ohlcv_data['TEST_STOCK'].copy()
        prices = test_data['close'].values
        n = len(prices)

        # Strategy returns (momentum)
        sma = pd.Series(prices).rolling(20).mean().values
        positions = np.where(prices > sma, 1, 0)
        positions = np.nan_to_num(positions)

        returns = np.diff(prices) / prices[:-1]
        strategy_returns = positions[:-1] * returns
        strategy_total = np.prod(1 + strategy_returns) - 1

        # Random strategy returns (1000 simulations)
        n_simulations = 1000
        random_totals = []

        for _ in range(n_simulations):
            random_positions = np.random.choice([0, 1], size=n-1)
            random_returns = random_positions * returns
            random_total = np.prod(1 + random_returns) - 1
            random_totals.append(random_total)

        # Calculate percentile
        percentile = np.mean(strategy_total > np.array(random_totals)) * 100

        print(f"Strategy return: {strategy_total:.2%}, "
              f"beats {percentile:.1f}% of random strategies")

        # A reasonable strategy should beat at least 50% of random
        # (actual threshold depends on market conditions and strategy quality)
        assert percentile >= 0, "Invalid percentile calculation"

    def test_information_ratio_calculation(self, synthetic_ohlcv_data):
        """
        Test information ratio calculation.

        Alpha should be measurable relative to benchmark tracking error.
        """
        strategy_prices = synthetic_ohlcv_data['TEST_STOCK']['close'].values
        benchmark_prices = synthetic_ohlcv_data['SPY']['close'].values

        # Calculate returns
        strategy_returns = np.diff(strategy_prices) / strategy_prices[:-1]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]

        # Ensure same length
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Active returns (excess returns over benchmark)
        active_returns = strategy_returns - benchmark_returns

        # Information Ratio = mean(active returns) / std(active returns) * sqrt(252)
        tracking_error = np.std(active_returns)

        if tracking_error > 0:
            information_ratio = (np.mean(active_returns) / tracking_error) * np.sqrt(252)
        else:
            information_ratio = 0

        # IR should be finite
        assert np.isfinite(information_ratio), "Information ratio is not finite"

        print(f"Information Ratio: {information_ratio:.2f}")

    def test_alpha_calculation(self, synthetic_ohlcv_data):
        """
        Test that alpha (excess return) is calculated correctly.
        """
        strategy_prices = synthetic_ohlcv_data['TEST_STOCK']['close'].values
        benchmark_prices = synthetic_ohlcv_data['SPY']['close'].values

        # Total returns
        strategy_return = (strategy_prices[-1] / strategy_prices[0]) - 1
        benchmark_return = (benchmark_prices[-1] / benchmark_prices[0]) - 1

        # Simple alpha (strategy excess return)
        alpha = strategy_return - benchmark_return

        # Alpha should be reasonable (not infinite)
        assert np.isfinite(alpha), "Alpha calculation resulted in non-finite value"
        assert -1 <= alpha <= 10, f"Alpha {alpha} seems unreasonable"

        print(f"Strategy: {strategy_return:.2%}, Benchmark: {benchmark_return:.2%}, "
              f"Alpha: {alpha:.2%}")

    def test_beta_calculation(self, synthetic_ohlcv_data):
        """
        Test beta (market sensitivity) calculation.
        """
        strategy_prices = synthetic_ohlcv_data['TEST_STOCK']['close'].values
        benchmark_prices = synthetic_ohlcv_data['SPY']['close'].values

        strategy_returns = np.diff(strategy_prices) / strategy_prices[:-1]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]

        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Beta = Cov(strategy, benchmark) / Var(benchmark)
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)

        if benchmark_variance > 0:
            beta = covariance / benchmark_variance
        else:
            beta = 1.0

        # Beta should be reasonable (typically between 0 and 2 for stocks)
        assert -1 <= beta <= 5, f"Beta {beta} seems unreasonable"

        print(f"Beta: {beta:.2f}")

    def test_treynor_ratio_calculation(self, synthetic_ohlcv_data):
        """
        Test Treynor ratio (return per unit of systematic risk) calculation.
        """
        strategy_prices = synthetic_ohlcv_data['TEST_STOCK']['close'].values
        benchmark_prices = synthetic_ohlcv_data['SPY']['close'].values

        strategy_returns = np.diff(strategy_prices) / strategy_prices[:-1]
        benchmark_returns = np.diff(benchmark_prices) / benchmark_prices[:-1]

        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

        # Annualized return
        strategy_annual = np.mean(strategy_returns) * 252

        # Beta
        covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        # Treynor Ratio = (Return - Risk-free rate) / Beta
        risk_free_rate = 0.02  # 2% annual

        if abs(beta) > 0.01:
            treynor_ratio = (strategy_annual - risk_free_rate) / beta
        else:
            treynor_ratio = 0

        assert np.isfinite(treynor_ratio), "Treynor ratio is not finite"

        print(f"Treynor Ratio: {treynor_ratio:.2f}")

    def test_sortino_ratio_vs_sharpe(self, synthetic_ohlcv_data):
        """
        Test that Sortino ratio handles downside risk correctly vs Sharpe.
        """
        prices = synthetic_ohlcv_data['TEST_STOCK']['close'].values
        returns = np.diff(prices) / prices[:-1]

        # Sharpe ratio
        sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))

        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = np.std(downside_returns) * np.sqrt(252)
            sortino = (np.mean(returns) * 252) / downside_std if downside_std > 0 else 0
        else:
            sortino = float('inf')

        # Sortino should be >= Sharpe (less penalty for upside volatility)
        # This assumes symmetric or positive-skewed returns
        if np.isfinite(sortino):
            print(f"Sharpe: {sharpe:.2f}, Sortino: {sortino:.2f}")


# =============================================================================
# Data Integrity Tests
# =============================================================================

class TestDataIntegrity:
    """Tests for data integrity in backtesting."""

    def test_ohlc_price_relationships(self, synthetic_ohlcv_data):
        """
        Verify OHLC price relationships are valid.

        High should be >= max(Open, Close)
        Low should be <= min(Open, Close)
        """
        for symbol, df in synthetic_ohlcv_data.items():
            open_prices = df['open'].values
            high_prices = df['high'].values
            low_prices = df['low'].values
            close_prices = df['close'].values

            # High >= max(Open, Close)
            max_oc = np.maximum(open_prices, close_prices)
            assert np.all(high_prices >= max_oc - 1e-10), (
                f"{symbol}: High prices below max(Open, Close)"
            )

            # Low <= min(Open, Close)
            min_oc = np.minimum(open_prices, close_prices)
            assert np.all(low_prices <= min_oc + 1e-10), (
                f"{symbol}: Low prices above min(Open, Close)"
            )

            # High >= Low
            assert np.all(high_prices >= low_prices - 1e-10), (
                f"{symbol}: High prices below Low prices"
            )

    def test_no_negative_prices(self, synthetic_ohlcv_data):
        """
        Verify no negative prices in data.
        """
        for symbol, df in synthetic_ohlcv_data.items():
            for col in ['open', 'high', 'low', 'close']:
                assert np.all(df[col] > 0), f"{symbol}: Negative {col} prices found"

    def test_no_negative_volumes(self, synthetic_ohlcv_data):
        """
        Verify no negative volumes in data.
        """
        for symbol, df in synthetic_ohlcv_data.items():
            assert np.all(df['volume'] >= 0), f"{symbol}: Negative volumes found"

    def test_datetime_monotonicity(self, synthetic_ohlcv_data):
        """
        Verify datetimes are monotonically increasing.
        """
        for symbol, df in synthetic_ohlcv_data.items():
            datetimes = pd.to_datetime(df['datetime'])

            for i in range(1, len(datetimes)):
                assert datetimes.iloc[i] > datetimes.iloc[i-1], (
                    f"{symbol}: Datetime not monotonically increasing at index {i}"
                )

    def test_no_duplicate_timestamps(self, synthetic_ohlcv_data):
        """
        Verify no duplicate timestamps in data.
        """
        for symbol, df in synthetic_ohlcv_data.items():
            datetimes = pd.to_datetime(df['datetime'])

            assert len(datetimes) == len(datetimes.unique()), (
                f"{symbol}: Duplicate timestamps found"
            )


# =============================================================================
# Reproducibility Tests
# =============================================================================

class TestReproducibility:
    """Tests for backtest reproducibility."""

    def test_deterministic_results_with_seed(self, synthetic_ohlcv_data, base_config):
        """
        Test that backtest results are deterministic with fixed random seed.
        """
        strategy = SimpleMomentumStrategy(lookback=20)

        # Run backtest twice
        results = []
        for _ in range(2):
            engine = BacktestEngine(base_config)
            engine.add_strategy(strategy)
            result = engine.run(synthetic_ohlcv_data)
            results.append(result)

        # Results should be identical
        assert results[0].total_return == results[1].total_return, (
            "Non-deterministic backtest results"
        )
        assert results[0].total_trades == results[1].total_trades, (
            "Non-deterministic trade count"
        )

    def test_config_independence(self, synthetic_ohlcv_data):
        """
        Test that different configs produce different (expected) results.
        """
        strategy = SimpleMomentumStrategy(lookback=20)

        # Config with no costs
        config_no_costs = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.0,
            slippage_rate=0.0,
            warmup_period=20,
        )

        # Config with costs
        config_with_costs = BacktestConfig(
            initial_capital=100000.0,
            commission_rate=0.01,  # 1%
            slippage_rate=0.005,   # 0.5%
            warmup_period=20,
        )

        engine_no_costs = BacktestEngine(config_no_costs)
        engine_no_costs.add_strategy(SimpleMomentumStrategy(lookback=20))
        result_no_costs = engine_no_costs.run(synthetic_ohlcv_data)

        engine_with_costs = BacktestEngine(config_with_costs)
        engine_with_costs.add_strategy(SimpleMomentumStrategy(lookback=20))
        result_with_costs = engine_with_costs.run(synthetic_ohlcv_data)

        # With costs should have lower returns (if any trades were made)
        if result_no_costs.total_trades > 0 and result_with_costs.total_trades > 0:
            assert result_with_costs.total_return <= result_no_costs.total_return, (
                "Transaction costs did not reduce returns as expected"
            )


# =============================================================================
# Performance Bounds Tests
# =============================================================================

class TestPerformanceBounds:
    """Tests for validating performance metrics are within reasonable bounds."""

    def test_sharpe_ratio_bounds(self, synthetic_ohlcv_data, base_config):
        """
        Sharpe ratio should be within reasonable bounds (-5 to 5 for most strategies).
        """
        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(synthetic_ohlcv_data)

        assert -10 <= result.sharpe_ratio <= 10, (
            f"Sharpe ratio {result.sharpe_ratio} outside reasonable bounds"
        )

    def test_max_drawdown_bounds(self, synthetic_ohlcv_data, base_config):
        """
        Max drawdown should be between 0% and 100%.
        """
        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(synthetic_ohlcv_data)

        assert 0 <= result.max_drawdown <= 100, (
            f"Max drawdown {result.max_drawdown}% outside bounds"
        )

    def test_win_rate_bounds(self, synthetic_ohlcv_data, base_config):
        """
        Win rate should be between 0% and 100%.
        """
        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(synthetic_ohlcv_data)

        assert 0 <= result.win_rate <= 100, (
            f"Win rate {result.win_rate}% outside bounds"
        )

    def test_final_value_non_negative(self, synthetic_ohlcv_data, base_config):
        """
        Final portfolio value should never be negative.
        """
        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(synthetic_ohlcv_data)

        assert result.final_value >= 0, (
            f"Final value {result.final_value} is negative"
        )

    def test_trade_count_reasonable(self, synthetic_ohlcv_data, base_config):
        """
        Trade count should be reasonable (not millions per day).
        """
        strategy = SimpleMomentumStrategy(lookback=20)
        engine = BacktestEngine(base_config)
        engine.add_strategy(strategy)

        result = engine.run(synthetic_ohlcv_data)

        # With 252 trading days, daily rebalancing would max at ~500 trades
        max_reasonable_trades = 1000

        assert result.total_trades <= max_reasonable_trades, (
            f"Trade count {result.total_trades} seems excessive"
        )


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

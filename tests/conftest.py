"""
Pytest configuration and fixtures for S&P 500 trading system tests.

This module provides reusable fixtures for:
- Mock stock data generation
- Sample OHLCV data
- Mock API responses
- Test portfolio and position data
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, MagicMock, patch
import json


# =============================================================================
# Date Fixtures
# =============================================================================

@pytest.fixture
def sample_dates() -> List[datetime]:
    """Generate a list of sample trading dates (weekdays only)."""
    start_date = datetime(2023, 1, 1)
    dates = []
    current = start_date
    while len(dates) < 252:  # Approximately 1 trading year
        if current.weekday() < 5:  # Monday = 0, Friday = 4
            dates.append(current)
        current += timedelta(days=1)
    return dates


@pytest.fixture
def short_date_range() -> List[datetime]:
    """Generate a short date range for quick tests."""
    start_date = datetime(2023, 6, 1)
    dates = []
    current = start_date
    while len(dates) < 20:
        if current.weekday() < 5:
            dates.append(current)
        current += timedelta(days=1)
    return dates


# =============================================================================
# Stock Data Fixtures
# =============================================================================

@pytest.fixture
def sp500_symbols() -> List[str]:
    """Return a subset of S&P 500 symbols for testing."""
    return [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
        'NVDA', 'TSLA', 'BRK.B', 'JPM', 'JNJ',
        'V', 'UNH', 'HD', 'PG', 'MA',
        'DIS', 'PYPL', 'ADBE', 'NFLX', 'CRM'
    ]


@pytest.fixture
def full_sp500_symbols() -> List[str]:
    """Return a larger set of S&P 500 symbols for comprehensive tests."""
    # Representative sample from different sectors
    return [
        # Technology
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Financial
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT',
        # Consumer
        'PG', 'KO', 'PEP', 'WMT', 'COST', 'MCD', 'NKE',
        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',
        # Industrial
        'CAT', 'DE', 'BA', 'UNP', 'HON', 'GE',
    ]


def generate_ohlcv_data(
    symbol: str,
    dates: List[datetime],
    base_price: float = 100.0,
    volatility: float = 0.02,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data for a given symbol.

    Args:
        symbol: Stock ticker symbol
        dates: List of trading dates
        base_price: Starting price for the stock
        volatility: Daily volatility (standard deviation of returns)
        seed: Random seed for reproducibility

    Returns:
        DataFrame with columns: date, open, high, low, close, volume, symbol
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(dates)

    # Generate random returns
    returns = np.random.normal(0.0005, volatility, n)  # Small positive drift

    # Calculate prices
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV
    data = {
        'date': dates,
        'symbol': [symbol] * n,
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0.001, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0.001, 0.02, n)),
        'close': prices,
        'volume': np.random.randint(1_000_000, 50_000_000, n)
    }

    # Ensure high >= open, close and low <= open, close
    df = pd.DataFrame(data)
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df


@pytest.fixture
def sample_ohlcv_data(sample_dates, sp500_symbols) -> pd.DataFrame:
    """Generate sample OHLCV data for multiple symbols."""
    all_data = []
    base_prices = {
        'AAPL': 150, 'MSFT': 300, 'GOOGL': 100, 'AMZN': 120, 'META': 200,
        'NVDA': 250, 'TSLA': 200, 'BRK.B': 300, 'JPM': 140, 'JNJ': 160,
        'V': 220, 'UNH': 500, 'HD': 300, 'PG': 150, 'MA': 350,
        'DIS': 100, 'PYPL': 80, 'ADBE': 400, 'NFLX': 350, 'CRM': 180
    }

    for i, symbol in enumerate(sp500_symbols):
        base_price = base_prices.get(symbol, 100 + i * 10)
        df = generate_ohlcv_data(symbol, sample_dates, base_price, seed=i)
        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)


@pytest.fixture
def single_stock_data(sample_dates) -> pd.DataFrame:
    """Generate OHLCV data for a single stock (AAPL)."""
    return generate_ohlcv_data('AAPL', sample_dates, base_price=150, seed=42)


@pytest.fixture
def trending_up_data(short_date_range) -> pd.DataFrame:
    """Generate data with a clear upward trend."""
    n = len(short_date_range)
    # Strong upward trend
    returns = np.random.normal(0.01, 0.005, n)  # 1% daily return on average
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'date': short_date_range,
        'symbol': ['TREND_UP'] * n,
        'open': prices * 0.995,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, n)
    })
    return df


@pytest.fixture
def trending_down_data(short_date_range) -> pd.DataFrame:
    """Generate data with a clear downward trend."""
    n = len(short_date_range)
    # Strong downward trend
    returns = np.random.normal(-0.01, 0.005, n)  # -1% daily return on average
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'date': short_date_range,
        'symbol': ['TREND_DOWN'] * n,
        'open': prices * 1.005,
        'high': prices * 1.01,
        'low': prices * 0.99,
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, n)
    })
    return df


@pytest.fixture
def mean_reverting_data(short_date_range) -> pd.DataFrame:
    """Generate data that exhibits mean reversion characteristics."""
    n = len(short_date_range)
    mean_price = 100

    # Start above mean, should revert
    prices = [110]  # Start 10% above mean
    for i in range(1, n):
        # Mean reversion: pull toward 100
        reversion_force = 0.1 * (mean_price - prices[-1])
        noise = np.random.normal(0, 1)
        prices.append(prices[-1] + reversion_force + noise)

    prices = np.array(prices)

    df = pd.DataFrame({
        'date': short_date_range,
        'symbol': ['MEAN_REV'] * n,
        'open': prices * 0.998,
        'high': prices * 1.005,
        'low': prices * 0.995,
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, n)
    })
    return df


@pytest.fixture
def empty_data() -> pd.DataFrame:
    """Return an empty DataFrame with the correct schema."""
    return pd.DataFrame(columns=['date', 'symbol', 'open', 'high', 'low', 'close', 'volume'])


@pytest.fixture
def insufficient_data(short_date_range) -> pd.DataFrame:
    """Generate data with too few data points for meaningful analysis."""
    dates = short_date_range[:3]  # Only 3 data points
    return generate_ohlcv_data('SMALL', dates, base_price=100, seed=99)


# =============================================================================
# Mock API Response Fixtures
# =============================================================================

@pytest.fixture
def mock_yahoo_response() -> Dict[str, Any]:
    """Mock response from Yahoo Finance API."""
    return {
        'chart': {
            'result': [{
                'meta': {
                    'currency': 'USD',
                    'symbol': 'AAPL',
                    'exchangeName': 'NASDAQ',
                    'regularMarketPrice': 175.50,
                    'previousClose': 174.20,
                },
                'timestamp': [1672531200, 1672617600, 1672704000],
                'indicators': {
                    'quote': [{
                        'open': [175.0, 176.0, 177.0],
                        'high': [176.5, 177.5, 178.5],
                        'low': [174.0, 175.0, 176.0],
                        'close': [175.5, 176.5, 177.5],
                        'volume': [50000000, 48000000, 52000000]
                    }],
                    'adjclose': [{
                        'adjclose': [175.5, 176.5, 177.5]
                    }]
                }
            }],
            'error': None
        }
    }


@pytest.fixture
def mock_sp500_wiki_response() -> str:
    """Mock HTML response from Wikipedia S&P 500 page."""
    return """
    <html>
    <body>
    <table class="wikitable sortable" id="constituents">
        <thead>
            <tr>
                <th>Symbol</th>
                <th>Security</th>
                <th>GICS Sector</th>
                <th>GICS Sub-Industry</th>
                <th>Headquarters Location</th>
                <th>Date added</th>
                <th>CIK</th>
                <th>Founded</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><a href="/wiki/Apple_Inc.">AAPL</a></td>
                <td>Apple Inc.</td>
                <td>Information Technology</td>
                <td>Technology Hardware</td>
                <td>Cupertino, California</td>
                <td>1982-11-30</td>
                <td>320193</td>
                <td>1977</td>
            </tr>
            <tr>
                <td><a href="/wiki/Microsoft">MSFT</a></td>
                <td>Microsoft Corporation</td>
                <td>Information Technology</td>
                <td>Systems Software</td>
                <td>Redmond, Washington</td>
                <td>1994-06-01</td>
                <td>789019</td>
                <td>1975</td>
            </tr>
        </tbody>
    </table>
    </body>
    </html>
    """


@pytest.fixture
def mock_api_error_response() -> Dict[str, Any]:
    """Mock error response from API."""
    return {
        'chart': {
            'result': None,
            'error': {
                'code': 'Not Found',
                'description': 'No data found for this symbol'
            }
        }
    }


@pytest.fixture
def mock_rate_limit_response() -> Dict[str, Any]:
    """Mock rate limit response from API."""
    return {
        'error': {
            'code': 429,
            'message': 'Rate limit exceeded. Please try again later.'
        }
    }


# =============================================================================
# Portfolio and Position Fixtures
# =============================================================================

@pytest.fixture
def sample_positions() -> Dict[str, Dict[str, Any]]:
    """Return sample portfolio positions."""
    return {
        'AAPL': {
            'symbol': 'AAPL',
            'quantity': 100,
            'entry_price': 150.00,
            'current_price': 175.00,
            'entry_date': datetime(2023, 1, 15),
            'stop_loss': 142.50,  # 5% below entry
            'take_profit': 180.00
        },
        'MSFT': {
            'symbol': 'MSFT',
            'quantity': 50,
            'entry_price': 280.00,
            'current_price': 320.00,
            'entry_date': datetime(2023, 2, 1),
            'stop_loss': 266.00,
            'take_profit': 350.00
        },
        'GOOGL': {
            'symbol': 'GOOGL',
            'quantity': 75,
            'entry_price': 95.00,
            'current_price': 88.00,  # Losing position
            'entry_date': datetime(2023, 3, 10),
            'stop_loss': 85.50,
            'take_profit': 115.00
        }
    }


@pytest.fixture
def empty_positions() -> Dict[str, Dict[str, Any]]:
    """Return empty portfolio positions."""
    return {}


@pytest.fixture
def sample_trade_history() -> List[Dict[str, Any]]:
    """Return sample trade history."""
    return [
        {
            'symbol': 'AAPL',
            'action': 'BUY',
            'quantity': 100,
            'price': 145.00,
            'timestamp': datetime(2023, 1, 5),
            'commission': 0.0
        },
        {
            'symbol': 'AAPL',
            'action': 'SELL',
            'quantity': 50,
            'price': 160.00,
            'timestamp': datetime(2023, 2, 15),
            'commission': 0.0
        },
        {
            'symbol': 'MSFT',
            'action': 'BUY',
            'quantity': 30,
            'price': 290.00,
            'timestamp': datetime(2023, 2, 20),
            'commission': 0.0
        },
        {
            'symbol': 'GOOGL',
            'action': 'BUY',
            'quantity': 75,
            'price': 95.00,
            'timestamp': datetime(2023, 3, 10),
            'commission': 0.0
        },
        {
            'symbol': 'TSLA',
            'action': 'BUY',
            'quantity': 25,
            'price': 200.00,
            'timestamp': datetime(2023, 3, 25),
            'commission': 0.0
        },
        {
            'symbol': 'TSLA',
            'action': 'SELL',
            'quantity': 25,
            'price': 180.00,  # Loss
            'timestamp': datetime(2023, 4, 10),
            'commission': 0.0
        }
    ]


@pytest.fixture
def portfolio_config() -> Dict[str, Any]:
    """Return sample portfolio configuration."""
    return {
        'initial_capital': 100000.0,
        'max_position_size': 0.10,  # 10% max per position
        'max_portfolio_exposure': 0.80,  # 80% max invested
        'stop_loss_pct': 0.05,  # 5% stop loss
        'take_profit_pct': 0.15,  # 15% take profit
        'commission_rate': 0.0,  # No commission for simplicity
        'slippage_rate': 0.001,  # 0.1% slippage
        'rebalance_frequency': 'weekly'
    }


# =============================================================================
# Risk Management Fixtures
# =============================================================================

@pytest.fixture
def risk_params() -> Dict[str, Any]:
    """Return sample risk management parameters."""
    return {
        'max_position_size': 10000.0,  # Max $10k per position
        'max_portfolio_risk': 0.02,  # 2% max portfolio risk
        'max_single_trade_risk': 0.01,  # 1% max single trade risk
        'max_correlation': 0.7,  # Max correlation between positions
        'max_sector_exposure': 0.30,  # 30% max sector exposure
        'trailing_stop_pct': 0.08,  # 8% trailing stop
        'volatility_lookback': 20,  # Days for volatility calculation
        'var_confidence': 0.95,  # 95% VaR confidence
    }


@pytest.fixture
def high_volatility_data(short_date_range) -> pd.DataFrame:
    """Generate high volatility data for risk testing."""
    n = len(short_date_range)
    # High volatility: 5% daily swings
    returns = np.random.normal(0, 0.05, n)
    prices = 100 * np.cumprod(1 + returns)

    df = pd.DataFrame({
        'date': short_date_range,
        'symbol': ['HIGH_VOL'] * n,
        'open': prices * (1 + np.random.uniform(-0.02, 0.02, n)),
        'high': prices * (1 + np.random.uniform(0.02, 0.05, n)),
        'low': prices * (1 - np.random.uniform(0.02, 0.05, n)),
        'close': prices,
        'volume': np.random.randint(1_000_000, 10_000_000, n)
    })
    return df


# =============================================================================
# Backtest Fixtures
# =============================================================================

@pytest.fixture
def backtest_config() -> Dict[str, Any]:
    """Return sample backtest configuration."""
    return {
        'start_date': datetime(2023, 1, 1),
        'end_date': datetime(2023, 12, 31),
        'initial_capital': 100000.0,
        'benchmark': 'SPY',
        'strategy': 'momentum',
        'rebalance_frequency': 'weekly',
        'commission': 0.0,
        'slippage': 0.001,
        'risk_free_rate': 0.05,  # 5% annual
    }


@pytest.fixture
def mock_backtest_results() -> Dict[str, Any]:
    """Return mock backtest results for testing report generation."""
    return {
        'total_return': 0.15,  # 15% return
        'annualized_return': 0.18,
        'sharpe_ratio': 1.25,
        'sortino_ratio': 1.45,
        'max_drawdown': -0.12,  # -12% max drawdown
        'win_rate': 0.58,
        'profit_factor': 1.65,
        'total_trades': 45,
        'winning_trades': 26,
        'losing_trades': 19,
        'avg_win': 0.08,
        'avg_loss': -0.03,
        'avg_holding_period': 12.5,  # days
        'portfolio_values': [100000, 102000, 101500, 104000, 108000, 115000],
        'dates': [
            datetime(2023, 1, 1),
            datetime(2023, 3, 1),
            datetime(2023, 5, 1),
            datetime(2023, 7, 1),
            datetime(2023, 9, 1),
            datetime(2023, 12, 31)
        ],
        'benchmark_values': [100000, 101000, 102500, 103000, 105000, 110000]
    }


# =============================================================================
# Mock Object Fixtures
# =============================================================================

@pytest.fixture
def mock_data_fetcher():
    """Create a mock DataFetcher object."""
    fetcher = Mock()
    fetcher.get_sp500_symbols.return_value = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    fetcher.fetch_ohlcv.return_value = pd.DataFrame({
        'date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        'symbol': ['AAPL', 'AAPL'],
        'open': [150.0, 151.0],
        'high': [152.0, 153.0],
        'low': [149.0, 150.0],
        'close': [151.0, 152.0],
        'volume': [50000000, 48000000]
    })
    return fetcher


@pytest.fixture
def mock_strategy():
    """Create a mock Strategy object."""
    strategy = Mock()
    strategy.generate_signals.return_value = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'signal': [1, -1, 0],  # Buy, Sell, Hold
        'strength': [0.8, 0.6, 0.0]
    })
    return strategy


@pytest.fixture
def mock_risk_manager():
    """Create a mock RiskManager object."""
    risk_mgr = Mock()
    risk_mgr.calculate_position_size.return_value = 100
    risk_mgr.check_stop_loss.return_value = False
    risk_mgr.get_portfolio_risk.return_value = 0.015
    return risk_mgr


# =============================================================================
# Utility Functions for Tests
# =============================================================================

def assert_ohlcv_valid(df: pd.DataFrame) -> None:
    """Assert that OHLCV data is valid."""
    required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    assert all(col in df.columns for col in required_columns), \
        f"Missing columns. Expected: {required_columns}, Got: {list(df.columns)}"

    if len(df) > 0:
        # High should be >= max(open, close)
        assert (df['high'] >= df[['open', 'close']].max(axis=1)).all(), \
            "High should be >= max(open, close)"

        # Low should be <= min(open, close)
        assert (df['low'] <= df[['open', 'close']].min(axis=1)).all(), \
            "Low should be <= min(open, close)"

        # Volume should be non-negative
        assert (df['volume'] >= 0).all(), "Volume should be non-negative"

        # Prices should be positive
        for col in ['open', 'high', 'low', 'close']:
            assert (df[col] > 0).all(), f"{col} should be positive"


def assert_signals_valid(signals: pd.DataFrame) -> None:
    """Assert that trading signals are valid."""
    required_columns = ['symbol', 'signal']
    assert all(col in signals.columns for col in required_columns), \
        f"Missing columns in signals. Expected: {required_columns}"

    if len(signals) > 0:
        # Signals should be -1, 0, or 1
        valid_signals = signals['signal'].isin([-1, 0, 1])
        assert valid_signals.all(), "Signals should be -1 (sell), 0 (hold), or 1 (buy)"


# Make utility functions available to tests
@pytest.fixture
def ohlcv_validator():
    """Return the OHLCV validation function."""
    return assert_ohlcv_valid


@pytest.fixture
def signals_validator():
    """Return the signals validation function."""
    return assert_signals_valid

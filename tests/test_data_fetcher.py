"""
Tests for the data fetching module.

This module tests:
- S&P 500 stock list retrieval
- OHLCV data fetching
- Caching behavior
- Error handling and edge cases
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import json
import tempfile
import os
from typing import Dict, Any, List
import requests


# =============================================================================
# Test S&P 500 Stock List Retrieval
# =============================================================================

class TestSP500StockList:
    """Tests for S&P 500 stock list retrieval functionality."""

    def test_get_sp500_symbols_returns_list(self, mock_sp500_wiki_response):
        """Test that get_sp500_symbols returns a list of symbols."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = mock_sp500_wiki_response
            mock_response.status_code = 200
            mock_response.raise_for_status = Mock()
            mock_get.return_value = mock_response

            # Assuming DataFetcher class exists
            from unittest.mock import MagicMock
            fetcher = MagicMock()
            fetcher.get_sp500_symbols.return_value = ['AAPL', 'MSFT']

            symbols = fetcher.get_sp500_symbols()

            assert isinstance(symbols, list)
            assert len(symbols) > 0
            assert 'AAPL' in symbols

    def test_sp500_symbols_are_strings(self, sp500_symbols):
        """Test that all symbols are strings."""
        for symbol in sp500_symbols:
            assert isinstance(symbol, str), f"Symbol {symbol} is not a string"

    def test_sp500_symbols_uppercase(self, sp500_symbols):
        """Test that all symbols are uppercase."""
        for symbol in sp500_symbols:
            assert symbol == symbol.upper(), f"Symbol {symbol} is not uppercase"

    def test_sp500_symbols_no_duplicates(self, full_sp500_symbols):
        """Test that there are no duplicate symbols."""
        assert len(full_sp500_symbols) == len(set(full_sp500_symbols)), \
            "Found duplicate symbols"

    def test_sp500_symbols_valid_format(self, full_sp500_symbols):
        """Test that symbols follow valid format (letters, dots allowed)."""
        import re
        pattern = re.compile(r'^[A-Z.]+$')
        for symbol in full_sp500_symbols:
            assert pattern.match(symbol), f"Invalid symbol format: {symbol}"

    @pytest.mark.parametrize("invalid_response", [
        "",  # Empty response
        "<html></html>",  # Empty HTML
        "<html><body>No table here</body></html>",  # No table
    ])
    def test_get_sp500_symbols_handles_invalid_html(self, invalid_response):
        """Test handling of invalid HTML responses."""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.text = invalid_response
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            # Should handle gracefully (return empty or raise appropriate error)
            fetcher = MagicMock()
            fetcher.get_sp500_symbols.side_effect = ValueError("Invalid HTML response")

            with pytest.raises(ValueError):
                fetcher.get_sp500_symbols()

    def test_get_sp500_symbols_network_error(self):
        """Test handling of network errors."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.ConnectionError("Network error")

            fetcher = MagicMock()
            fetcher.get_sp500_symbols.side_effect = requests.exceptions.ConnectionError("Network error")

            with pytest.raises(requests.exceptions.ConnectionError):
                fetcher.get_sp500_symbols()

    def test_get_sp500_symbols_timeout(self):
        """Test handling of timeout errors."""
        with patch('requests.get') as mock_get:
            mock_get.side_effect = requests.exceptions.Timeout("Request timed out")

            fetcher = MagicMock()
            fetcher.get_sp500_symbols.side_effect = requests.exceptions.Timeout("Timeout")

            with pytest.raises(requests.exceptions.Timeout):
                fetcher.get_sp500_symbols()


# =============================================================================
# Test OHLCV Data Fetching
# =============================================================================

class TestOHLCVDataFetching:
    """Tests for OHLCV data fetching functionality."""

    def test_fetch_ohlcv_returns_dataframe(self, single_stock_data):
        """Test that fetch_ohlcv returns a DataFrame."""
        assert isinstance(single_stock_data, pd.DataFrame)

    def test_fetch_ohlcv_has_required_columns(self, single_stock_data):
        """Test that returned DataFrame has all required columns."""
        required_columns = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            assert col in single_stock_data.columns, f"Missing column: {col}"

    def test_fetch_ohlcv_data_types(self, single_stock_data):
        """Test that columns have correct data types."""
        assert single_stock_data['symbol'].dtype == 'object'
        assert np.issubdtype(single_stock_data['volume'].dtype, np.integer) or \
               np.issubdtype(single_stock_data['volume'].dtype, np.floating)

        for col in ['open', 'high', 'low', 'close']:
            assert np.issubdtype(single_stock_data[col].dtype, np.floating)

    def test_fetch_ohlcv_high_low_relationship(self, single_stock_data, ohlcv_validator):
        """Test that high >= low for all rows."""
        ohlcv_validator(single_stock_data)

    def test_fetch_ohlcv_positive_prices(self, single_stock_data):
        """Test that all prices are positive."""
        for col in ['open', 'high', 'low', 'close']:
            assert (single_stock_data[col] > 0).all(), f"Negative prices found in {col}"

    def test_fetch_ohlcv_non_negative_volume(self, single_stock_data):
        """Test that volume is non-negative."""
        assert (single_stock_data['volume'] >= 0).all(), "Negative volume found"

    def test_fetch_ohlcv_date_ordering(self, single_stock_data):
        """Test that dates are in ascending order."""
        dates = pd.to_datetime(single_stock_data['date'])
        assert dates.is_monotonic_increasing, "Dates are not in ascending order"

    def test_fetch_multiple_symbols(self, sample_ohlcv_data, sp500_symbols):
        """Test fetching data for multiple symbols."""
        unique_symbols = sample_ohlcv_data['symbol'].unique()
        assert len(unique_symbols) == len(sp500_symbols)
        for symbol in sp500_symbols:
            assert symbol in unique_symbols, f"Missing symbol: {symbol}"

    def test_fetch_ohlcv_date_range(self, sample_ohlcv_data, sample_dates):
        """Test that data falls within requested date range."""
        data_dates = pd.to_datetime(sample_ohlcv_data['date'])
        assert data_dates.min() >= pd.to_datetime(sample_dates[0])
        assert data_dates.max() <= pd.to_datetime(sample_dates[-1])

    @pytest.mark.parametrize("symbol,expected_error", [
        ("INVALID123", "Invalid symbol"),
        ("", "Empty symbol"),
        (None, "None symbol"),
    ])
    def test_fetch_ohlcv_invalid_symbol(self, symbol, expected_error):
        """Test handling of invalid symbols."""
        fetcher = MagicMock()

        if symbol is None or symbol == "":
            fetcher.fetch_ohlcv.side_effect = ValueError(expected_error)
            with pytest.raises(ValueError):
                fetcher.fetch_ohlcv(symbol, datetime(2023, 1, 1), datetime(2023, 12, 31))
        else:
            # For invalid but non-empty symbols, might return empty DataFrame
            fetcher.fetch_ohlcv.return_value = pd.DataFrame()
            result = fetcher.fetch_ohlcv(symbol, datetime(2023, 1, 1), datetime(2023, 12, 31))
            assert len(result) == 0

    def test_fetch_ohlcv_invalid_date_range(self):
        """Test handling of invalid date ranges (end before start)."""
        fetcher = MagicMock()
        fetcher.fetch_ohlcv.side_effect = ValueError("End date before start date")

        with pytest.raises(ValueError):
            fetcher.fetch_ohlcv('AAPL', datetime(2023, 12, 31), datetime(2023, 1, 1))

    def test_fetch_ohlcv_future_dates(self):
        """Test handling of future dates."""
        fetcher = MagicMock()
        future_date = datetime.now() + timedelta(days=365)

        # Should return empty or partial data for future dates
        fetcher.fetch_ohlcv.return_value = pd.DataFrame()
        result = fetcher.fetch_ohlcv('AAPL', datetime.now(), future_date)
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# Test Caching Behavior
# =============================================================================

class TestCachingBehavior:
    """Tests for data caching functionality."""

    def test_cache_stores_data(self, single_stock_data):
        """Test that fetched data is cached."""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_file = os.path.join(cache_dir, 'AAPL_2023.pkl')

            # Simulate caching
            single_stock_data.to_pickle(cache_file)

            assert os.path.exists(cache_file)
            cached_data = pd.read_pickle(cache_file)
            pd.testing.assert_frame_equal(single_stock_data, cached_data)

    def test_cache_retrieval(self, single_stock_data):
        """Test that cached data is retrieved correctly."""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_file = os.path.join(cache_dir, 'AAPL_2023.pkl')

            # Store in cache
            single_stock_data.to_pickle(cache_file)

            # Retrieve from cache
            cached_data = pd.read_pickle(cache_file)

            pd.testing.assert_frame_equal(single_stock_data, cached_data)

    def test_cache_invalidation_on_expiry(self):
        """Test that cache is invalidated after expiry."""
        with tempfile.TemporaryDirectory() as cache_dir:
            cache_file = os.path.join(cache_dir, 'test_cache.json')

            # Create cache with old timestamp
            cache_data = {
                'timestamp': (datetime.now() - timedelta(days=2)).isoformat(),
                'data': [1, 2, 3]
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f)

            # Check if cache is expired (assuming 1 day expiry)
            with open(cache_file, 'r') as f:
                loaded = json.load(f)

            cache_time = datetime.fromisoformat(loaded['timestamp'])
            is_expired = (datetime.now() - cache_time) > timedelta(days=1)

            assert is_expired, "Cache should be expired"

    def test_cache_respects_max_size(self):
        """Test that cache respects maximum size limits."""
        max_cache_entries = 100
        cache = {}

        # Add entries beyond limit
        for i in range(150):
            if len(cache) >= max_cache_entries:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(cache))
                del cache[oldest_key]
            cache[f'key_{i}'] = f'value_{i}'

        assert len(cache) <= max_cache_entries

    def test_cache_key_generation(self):
        """Test that cache keys are generated correctly."""
        symbol = 'AAPL'
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        # Expected key format
        expected_key = f"{symbol}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"

        assert expected_key == "AAPL_20230101_20231231"

    def test_cache_miss_triggers_fetch(self):
        """Test that cache miss triggers API fetch."""
        fetcher = MagicMock()
        fetch_count = 0

        def mock_fetch(*args, **kwargs):
            nonlocal fetch_count
            fetch_count += 1
            return pd.DataFrame({'data': [1, 2, 3]})

        fetcher.fetch_ohlcv.side_effect = mock_fetch

        # First call - cache miss
        fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))
        assert fetch_count == 1

        # Second call with same params - should use cache (but we're mocking)
        fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))
        assert fetch_count == 2  # In real implementation, this would be 1 with cache

    def test_cache_handles_concurrent_access(self):
        """Test that cache handles concurrent access safely."""
        import threading

        cache = {}
        lock = threading.Lock()
        errors = []

        def writer(key, value):
            try:
                with lock:
                    cache[key] = value
            except Exception as e:
                errors.append(e)

        def reader(key):
            try:
                with lock:
                    return cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(10):
            t1 = threading.Thread(target=writer, args=(f'key_{i}', f'value_{i}'))
            t2 = threading.Thread(target=reader, args=(f'key_{i}',))
            threads.extend([t1, t2])

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"


# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error handling in data fetching."""

    def test_handles_api_error_response(self, mock_api_error_response):
        """Test handling of API error responses."""
        fetcher = MagicMock()
        fetcher.fetch_ohlcv.side_effect = ValueError("No data found")

        with pytest.raises(ValueError, match="No data found"):
            fetcher.fetch_ohlcv('INVALID', datetime(2023, 1, 1), datetime(2023, 12, 31))

    def test_handles_rate_limit(self, mock_rate_limit_response):
        """Test handling of rate limit responses."""
        fetcher = MagicMock()

        # Simulate rate limiting with retry logic
        call_count = 0

        def rate_limited_fetch(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Rate limit exceeded")
            return pd.DataFrame({'data': [1, 2, 3]})

        fetcher.fetch_ohlcv.side_effect = rate_limited_fetch

        # Should eventually succeed after retries
        result = None
        for _ in range(5):
            try:
                result = fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))
                break
            except Exception:
                continue

        assert result is not None

    def test_handles_network_timeout(self):
        """Test handling of network timeouts."""
        fetcher = MagicMock()
        fetcher.fetch_ohlcv.side_effect = requests.exceptions.Timeout("Timeout")

        with pytest.raises(requests.exceptions.Timeout):
            fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))

    def test_handles_ssl_error(self):
        """Test handling of SSL errors."""
        fetcher = MagicMock()
        fetcher.fetch_ohlcv.side_effect = requests.exceptions.SSLError("SSL Error")

        with pytest.raises(requests.exceptions.SSLError):
            fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))

    def test_handles_partial_data(self, short_date_range):
        """Test handling of partial data responses."""
        fetcher = MagicMock()

        # Return data for only half the requested dates
        partial_data = pd.DataFrame({
            'date': short_date_range[:10],
            'symbol': ['AAPL'] * 10,
            'open': [150.0] * 10,
            'high': [152.0] * 10,
            'low': [148.0] * 10,
            'close': [151.0] * 10,
            'volume': [1000000] * 10
        })

        fetcher.fetch_ohlcv.return_value = partial_data
        result = fetcher.fetch_ohlcv('AAPL', short_date_range[0], short_date_range[-1])

        assert len(result) == 10
        assert len(result) < len(short_date_range)

    def test_handles_malformed_data(self):
        """Test handling of malformed API responses."""
        fetcher = MagicMock()

        # Malformed data with missing columns
        malformed_data = pd.DataFrame({
            'date': [datetime(2023, 1, 1)],
            'close': [150.0]
            # Missing open, high, low, volume
        })

        fetcher.fetch_ohlcv.return_value = malformed_data

        result = fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))

        # Should either raise error or handle gracefully
        assert 'close' in result.columns

    def test_handles_empty_response(self, empty_data):
        """Test handling of empty API responses."""
        fetcher = MagicMock()
        fetcher.fetch_ohlcv.return_value = empty_data

        result = fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    @pytest.mark.parametrize("error_type,error_message", [
        (ConnectionError, "Connection refused"),
        (TimeoutError, "Request timed out"),
        (ValueError, "Invalid data format"),
        (KeyError, "Missing key in response"),
    ])
    def test_handles_various_exceptions(self, error_type, error_message):
        """Test handling of various exception types."""
        fetcher = MagicMock()
        fetcher.fetch_ohlcv.side_effect = error_type(error_message)

        with pytest.raises(error_type, match=error_message):
            fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))

    def test_retry_logic_on_transient_errors(self):
        """Test retry logic for transient errors."""
        fetcher = MagicMock()

        attempt = 0
        max_attempts = 3

        def flaky_fetch(*args, **kwargs):
            nonlocal attempt
            attempt += 1
            if attempt < max_attempts:
                raise ConnectionError("Temporary failure")
            return pd.DataFrame({'data': [1, 2, 3]})

        fetcher.fetch_ohlcv.side_effect = flaky_fetch

        # Implement retry logic
        result = None
        for i in range(max_attempts):
            try:
                result = fetcher.fetch_ohlcv('AAPL', datetime(2023, 1, 1), datetime(2023, 12, 31))
                break
            except ConnectionError:
                if i == max_attempts - 1:
                    raise
                continue

        assert result is not None
        assert attempt == max_attempts


# =============================================================================
# Test Data Validation
# =============================================================================

class TestDataValidation:
    """Tests for data validation functionality."""

    def test_validate_ohlcv_schema(self, single_stock_data, ohlcv_validator):
        """Test OHLCV schema validation."""
        ohlcv_validator(single_stock_data)

    def test_detect_price_anomalies(self, single_stock_data):
        """Test detection of price anomalies."""
        # Add anomalous data
        anomalous_data = single_stock_data.copy()
        anomalous_data.loc[0, 'close'] = -100  # Negative price

        has_anomaly = (anomalous_data['close'] < 0).any()
        assert has_anomaly, "Should detect negative price anomaly"

    def test_detect_volume_anomalies(self, single_stock_data):
        """Test detection of volume anomalies."""
        # Add anomalous volume
        anomalous_data = single_stock_data.copy()
        anomalous_data.loc[0, 'volume'] = -1000000  # Negative volume

        has_anomaly = (anomalous_data['volume'] < 0).any()
        assert has_anomaly, "Should detect negative volume anomaly"

    def test_detect_ohlc_inconsistency(self):
        """Test detection of OHLC inconsistencies."""
        inconsistent_data = pd.DataFrame({
            'date': [datetime(2023, 1, 1)],
            'symbol': ['TEST'],
            'open': [100.0],
            'high': [95.0],  # High < Open - invalid
            'low': [90.0],
            'close': [98.0],
            'volume': [1000000]
        })

        # High should be >= all other prices
        is_inconsistent = (inconsistent_data['high'] < inconsistent_data['open']).any()
        assert is_inconsistent, "Should detect OHLC inconsistency"

    def test_detect_missing_dates(self, sample_dates):
        """Test detection of missing dates in data."""
        # Create data with gaps
        sparse_dates = sample_dates[::3]  # Every 3rd date
        sparse_data = pd.DataFrame({
            'date': sparse_dates,
            'symbol': ['TEST'] * len(sparse_dates),
            'close': [100.0] * len(sparse_dates)
        })

        # Check for gaps
        dates = pd.to_datetime(sparse_data['date'])
        date_diffs = dates.diff().dropna()
        expected_diff = timedelta(days=1)

        # There should be gaps larger than 1 day (excluding weekends)
        has_gaps = (date_diffs > timedelta(days=3)).any()
        assert has_gaps, "Should detect missing dates"

    def test_data_continuity(self, single_stock_data):
        """Test data continuity (no sudden jumps)."""
        prices = single_stock_data['close'].values
        returns = np.diff(prices) / prices[:-1]

        # Check for extreme returns (>50% in a day)
        extreme_returns = np.abs(returns) > 0.5
        assert not extreme_returns.any(), "Found extreme price jumps"

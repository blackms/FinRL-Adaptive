"""
S&P 500 Data Fetcher Module

Provides functionality to fetch and cache S&P 500 stock data using yfinance.
Handles rate limiting and provides caching capabilities for efficient data retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for data caching."""

    enabled: bool = True
    directory: Path = field(default_factory=lambda: Path("./cache"))
    expiry_hours: int = 24

    def __post_init__(self) -> None:
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    requests_per_minute: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    backoff_multiplier: float = 2.0


@dataclass
class StockData:
    """Container for stock OHLCV data."""

    symbol: str
    data: pd.DataFrame
    market_cap: float | None = None
    sector: str | None = None
    industry: str | None = None
    fetched_at: datetime = field(default_factory=datetime.now)

    @property
    def is_valid(self) -> bool:
        """Check if data is valid and non-empty."""
        return self.data is not None and not self.data.empty


class RateLimiter:
    """Rate limiter for API requests."""

    def __init__(self, config: RateLimitConfig) -> None:
        self.config = config
        self._request_times: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Acquire permission to make a request, waiting if necessary."""
        async with self._lock:
            now = time.time()
            minute_ago = now - 60

            # Remove old request times
            self._request_times = [t for t in self._request_times if t > minute_ago]

            if len(self._request_times) >= self.config.requests_per_minute:
                # Need to wait
                oldest = self._request_times[0]
                wait_time = 60 - (now - oldest) + 0.1
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)

            self._request_times.append(time.time())

    def acquire_sync(self) -> None:
        """Synchronous version of acquire."""
        now = time.time()
        minute_ago = now - 60

        self._request_times = [t for t in self._request_times if t > minute_ago]

        if len(self._request_times) >= self.config.requests_per_minute:
            oldest = self._request_times[0]
            wait_time = 60 - (now - oldest) + 0.1
            if wait_time > 0:
                logger.debug(f"Rate limit reached, waiting {wait_time:.2f}s")
                time.sleep(wait_time)

        self._request_times.append(time.time())


class SP500DataFetcher:
    """
    Fetcher for S&P 500 stock data.

    Provides methods to:
    - Get the list of S&P 500 constituents
    - Fetch top N stocks by market cap
    - Download OHLCV data with caching
    - Handle rate limiting gracefully

    Example:
        >>> fetcher = S&P500DataFetcher()
        >>> top_stocks = fetcher.get_top_stocks_by_market_cap(100)
        >>> data = fetcher.fetch_ohlcv(top_stocks, start="2023-01-01")
    """

    # S&P 500 constituents (simplified list - in production, fetch from Wikipedia or similar)
    SP500_SYMBOLS = [
        "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "UNH", "XOM",
        "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK", "ABBV", "LLY",
        "PEP", "KO", "PFE", "COST", "TMO", "AVGO", "MCD", "WMT", "CSCO", "ABT",
        "ACN", "DHR", "BAC", "CRM", "ADBE", "CMCSA", "NKE", "VZ", "NFLX", "INTC",
        "TXN", "WFC", "PM", "RTX", "AMD", "T", "NEE", "UPS", "MS", "BMY",
        "QCOM", "HON", "UNP", "SPGI", "IBM", "CAT", "BA", "DE", "INTU", "LOW",
        "GS", "ELV", "SBUX", "GE", "ISRG", "AMAT", "PLD", "BLK", "MDLZ", "AXP",
        "ADP", "LMT", "GILD", "TJX", "CVS", "BKNG", "SYK", "ADI", "MMC", "AMT",
        "VRTX", "C", "REGN", "CI", "ZTS", "NOW", "TMUS", "MO", "ETN", "CB",
        "SO", "BDX", "BSX", "DUK", "SCHW", "PGR", "LRCX", "SLB", "AON", "EQIX",
        "CME", "ITW", "CL", "MU", "SNPS", "ICE", "CDNS", "NOC", "WM", "HUM",
        "SHW", "FI", "EOG", "MMM", "ORLY", "CSX", "GD", "MCK", "FDX", "APD",
        "NSC", "COP", "ATVI", "EMR", "PXD", "F", "GM", "PSA", "TGT", "MAR",
        "USB", "PNC", "AEP", "TFC", "KLAC", "OXY", "MRNA", "AZO", "KMB", "MCO",
        "CTVA", "PSX", "SRE", "D", "CHTR", "AFL", "ECL", "EW", "CCI", "ADSK",
    ]

    def __init__(
        self,
        cache_config: CacheConfig | None = None,
        rate_limit_config: RateLimitConfig | None = None,
    ) -> None:
        """
        Initialize the data fetcher.

        Args:
            cache_config: Configuration for caching. Defaults to enabled with 24h expiry.
            rate_limit_config: Configuration for rate limiting. Defaults to 60 req/min.
        """
        self.cache_config = cache_config or CacheConfig()
        self.rate_limit_config = rate_limit_config or RateLimitConfig()
        self._rate_limiter = RateLimiter(self.rate_limit_config)
        self._market_cap_cache: dict[str, tuple[float, datetime]] = {}

    def _get_cache_key(self, symbols: list[str], start: str, end: str, interval: str) -> str:
        """Generate a cache key for the given parameters."""
        key_data = f"{sorted(symbols)}_{start}_{end}_{interval}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a given key."""
        return self.cache_config.directory / f"{cache_key}.parquet"

    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if a cache file exists and is not expired."""
        if not cache_path.exists():
            return False

        mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
        expiry = mtime + timedelta(hours=self.cache_config.expiry_hours)
        return datetime.now() < expiry

    def _load_from_cache(self, cache_path: Path) -> pd.DataFrame | None:
        """Load data from cache if valid."""
        if not self.cache_config.enabled:
            return None

        if self._is_cache_valid(cache_path):
            try:
                logger.debug(f"Loading from cache: {cache_path}")
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
        return None

    def _save_to_cache(self, data: pd.DataFrame, cache_path: Path) -> None:
        """Save data to cache."""
        if not self.cache_config.enabled or data.empty:
            return

        try:
            data.to_parquet(cache_path)
            logger.debug(f"Saved to cache: {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def get_sp500_symbols(self) -> list[str]:
        """
        Get the list of S&P 500 constituent symbols.

        Returns:
            List of stock ticker symbols.
        """
        return self.SP500_SYMBOLS.copy()

    def _fetch_market_cap_with_retry(self, symbol: str) -> float | None:
        """Fetch market cap for a single symbol with retry logic."""
        for attempt in range(self.rate_limit_config.retry_attempts):
            try:
                self._rate_limiter.acquire_sync()
                ticker = yf.Ticker(symbol)
                info = ticker.info
                market_cap = info.get("marketCap")
                if market_cap is not None:
                    return float(market_cap)
                return None
            except Exception as e:
                delay = self.rate_limit_config.retry_delay_seconds * (
                    self.rate_limit_config.backoff_multiplier ** attempt
                )
                logger.warning(
                    f"Failed to fetch market cap for {symbol} (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < self.rate_limit_config.retry_attempts - 1:
                    time.sleep(delay)
        return None

    def get_market_caps(
        self,
        symbols: list[str] | None = None,
        use_cache: bool = True,
    ) -> dict[str, float]:
        """
        Get market capitalizations for the given symbols.

        Args:
            symbols: List of ticker symbols. Defaults to all S&P 500 symbols.
            use_cache: Whether to use cached market cap values.

        Returns:
            Dictionary mapping symbols to their market caps.
        """
        symbols = symbols or self.SP500_SYMBOLS
        market_caps: dict[str, float] = {}
        cache_expiry = timedelta(hours=self.cache_config.expiry_hours)

        for symbol in symbols:
            # Check cache first
            if use_cache and symbol in self._market_cap_cache:
                cached_cap, cached_time = self._market_cap_cache[symbol]
                if datetime.now() - cached_time < cache_expiry:
                    market_caps[symbol] = cached_cap
                    continue

            # Fetch from API
            cap = self._fetch_market_cap_with_retry(symbol)
            if cap is not None:
                market_caps[symbol] = cap
                self._market_cap_cache[symbol] = (cap, datetime.now())

        return market_caps

    def get_top_stocks_by_market_cap(
        self,
        n: int = 100,
        symbols: list[str] | None = None,
    ) -> list[str]:
        """
        Get the top N stocks by market capitalization.

        Args:
            n: Number of stocks to return.
            symbols: List of symbols to filter from. Defaults to S&P 500.

        Returns:
            List of ticker symbols sorted by market cap (descending).
        """
        symbols = symbols or self.SP500_SYMBOLS
        market_caps = self.get_market_caps(symbols)

        # Sort by market cap descending
        sorted_symbols = sorted(
            market_caps.keys(),
            key=lambda s: market_caps[s],
            reverse=True,
        )

        return sorted_symbols[:n]

    def validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix OHLC data consistency.

        Ensures data integrity by:
        - Enforcing high >= max(open, close)
        - Enforcing low <= min(open, close)
        - Removing rows with zero/negative prices
        - Logging warnings for suspicious price gaps (>50% moves)

        Args:
            df: DataFrame with OHLCV data. Must have columns:
                open, high, low, close, volume (case-insensitive).

        Returns:
            Cleaned DataFrame with validated OHLCV data.

        Example:
            >>> validated_df = fetcher.validate_ohlcv(raw_df)
        """
        if df.empty:
            return df

        # Make a copy to avoid modifying original
        df = df.copy()

        # Normalize column names to lowercase
        col_mapping = {col: col.lower() for col in df.columns}
        df = df.rename(columns=col_mapping)

        # Check required columns exist
        required_cols = ["open", "high", "low", "close"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns for validation: {missing_cols}")
            return df

        # Ensure high >= max(open, close)
        df["high"] = df[["open", "close", "high"]].max(axis=1)

        # Ensure low <= min(open, close)
        df["low"] = df[["open", "close", "low"]].min(axis=1)

        # Remove rows with zero/negative prices
        original_len = len(df)
        df = df[(df["close"] > 0) & (df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0)]

        # Handle volume column if present
        if "volume" in df.columns:
            df = df[df["volume"] >= 0]

        removed_rows = original_len - len(df)
        if removed_rows > 0:
            logger.warning(
                f"Removed {removed_rows} rows with invalid prices (zero/negative)"
            )

        # Check for suspicious gaps (>50% overnight moves)
        if len(df) > 1:
            df["_pct_change"] = df["close"].pct_change().abs()
            suspicious = df[df["_pct_change"] > 0.5]

            if len(suspicious) > 0:
                logger.warning(
                    f"Found {len(suspicious)} suspicious price moves >50%: "
                    f"dates={suspicious['datetime'].tolist() if 'datetime' in suspicious.columns else 'N/A'}"
                )

            df = df.drop(columns=["_pct_change"], errors="ignore")

        return df

    def fetch_ohlcv(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> dict[str, StockData]:
        """
        Fetch OHLCV data for the given symbols.

        Args:
            symbols: List of ticker symbols.
            start: Start date (YYYY-MM-DD format or datetime).
            end: End date. Defaults to today.
            interval: Data interval ('1d', '1h', '5m', etc.).

        Returns:
            Dictionary mapping symbols to StockData objects.
        """
        # Normalize dates
        start_str = start if isinstance(start, str) else start.strftime("%Y-%m-%d")
        end_str = (
            end if isinstance(end, str)
            else (end.strftime("%Y-%m-%d") if end else datetime.now().strftime("%Y-%m-%d"))
        )

        # Check cache
        cache_key = self._get_cache_key(symbols, start_str, end_str, interval)
        cache_path = self._get_cache_path(cache_key)

        cached_data = self._load_from_cache(cache_path)
        if cached_data is not None:
            return self._dataframe_to_stock_data(cached_data, symbols)

        # Fetch data
        result: dict[str, StockData] = {}
        all_data: list[pd.DataFrame] = []

        for symbol in symbols:
            stock_data = self._fetch_single_ohlcv(symbol, start_str, end_str, interval)
            if stock_data.is_valid:
                result[symbol] = stock_data
                df = stock_data.data.copy()
                df["symbol"] = symbol
                all_data.append(df)

        # Cache combined data
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            self._save_to_cache(combined, cache_path)

        return result

    def _fetch_single_ohlcv(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> StockData:
        """Fetch OHLCV data for a single symbol with retry logic."""
        for attempt in range(self.rate_limit_config.retry_attempts):
            try:
                self._rate_limiter.acquire_sync()
                ticker = yf.Ticker(symbol)
                df = ticker.history(start=start, end=end, interval=interval)

                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    return StockData(symbol=symbol, data=pd.DataFrame())

                # Standardize column names
                df = df.reset_index()
                df.columns = [col.lower().replace(" ", "_") for col in df.columns]

                # Rename 'date' to 'datetime' if present
                if "date" in df.columns:
                    df = df.rename(columns={"date": "datetime"})

                # Validate and fix OHLCV data consistency
                df = self.validate_ohlcv(df)

                if df.empty:
                    logger.warning(f"No valid data after validation for {symbol}")
                    return StockData(symbol=symbol, data=pd.DataFrame())

                # Get additional info
                info = ticker.info
                market_cap = info.get("marketCap")
                sector = info.get("sector")
                industry = info.get("industry")

                return StockData(
                    symbol=symbol,
                    data=df,
                    market_cap=float(market_cap) if market_cap else None,
                    sector=sector,
                    industry=industry,
                )

            except Exception as e:
                delay = self.rate_limit_config.retry_delay_seconds * (
                    self.rate_limit_config.backoff_multiplier ** attempt
                )
                logger.warning(
                    f"Failed to fetch OHLCV for {symbol} (attempt {attempt + 1}): {e}. "
                    f"Retrying in {delay:.1f}s"
                )
                if attempt < self.rate_limit_config.retry_attempts - 1:
                    time.sleep(delay)

        return StockData(symbol=symbol, data=pd.DataFrame())

    def _dataframe_to_stock_data(
        self,
        df: pd.DataFrame,
        symbols: list[str],
    ) -> dict[str, StockData]:
        """Convert a combined DataFrame back to StockData objects."""
        result: dict[str, StockData] = {}

        for symbol in symbols:
            symbol_df = df[df["symbol"] == symbol].drop(columns=["symbol"])
            if not symbol_df.empty:
                result[symbol] = StockData(symbol=symbol, data=symbol_df.reset_index(drop=True))

        return result

    async def fetch_ohlcv_async(
        self,
        symbols: list[str],
        start: str | datetime,
        end: str | datetime | None = None,
        interval: str = "1d",
    ) -> dict[str, StockData]:
        """
        Asynchronously fetch OHLCV data for the given symbols.

        Args:
            symbols: List of ticker symbols.
            start: Start date (YYYY-MM-DD format or datetime).
            end: End date. Defaults to today.
            interval: Data interval ('1d', '1h', '5m', etc.).

        Returns:
            Dictionary mapping symbols to StockData objects.
        """
        # Run synchronous fetch in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.fetch_ohlcv(symbols, start, end, interval),
        )

    def clear_cache(self) -> None:
        """Clear all cached data."""
        if not self.cache_config.enabled:
            return

        for cache_file in self.cache_config.directory.glob("*.parquet"):
            try:
                cache_file.unlink()
                logger.debug(f"Deleted cache file: {cache_file}")
            except Exception as e:
                logger.warning(f"Failed to delete cache file {cache_file}: {e}")

        self._market_cap_cache.clear()
        logger.info("Cache cleared")


# Export the main class
__all__ = ["SP500DataFetcher", "StockData", "CacheConfig", "RateLimitConfig"]

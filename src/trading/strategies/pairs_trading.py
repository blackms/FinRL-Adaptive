"""
Pairs Trading Strategy Module

Implements a statistical arbitrage strategy based on cointegration analysis
using the Engle-Granger method for mean-reverting pair relationships.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .base import BaseStrategy, Signal, SignalType, StrategyConfig


@dataclass
class PairConfig:
    """
    Configuration for a single trading pair.

    Attributes:
        symbol_a: First symbol in the pair (typically the dependent variable).
        symbol_b: Second symbol in the pair (typically the independent variable).
        hedge_ratio: Hedge ratio for the pair (units of B per unit of A).
        half_life: Estimated mean-reversion half-life in periods.
    """

    symbol_a: str
    symbol_b: str
    hedge_ratio: float = 1.0
    half_life: float | None = None


@dataclass
class PairsTradingConfig(StrategyConfig):
    """
    Configuration for Pairs Trading Strategy.

    Attributes:
        pairs: List of PairConfig objects defining trading pairs.
        lookback_period: Period for cointegration test and spread calculation.
        zscore_entry_threshold: Z-score threshold for entry (default: 2.0).
        zscore_exit_threshold: Z-score threshold for exit (default: 0.0).
        zscore_stop_loss: Z-score threshold for stop loss (default: 4.0).
        min_half_life: Minimum acceptable half-life for mean reversion.
        max_half_life: Maximum acceptable half-life for mean reversion.
        recalc_period: Periods between hedge ratio recalculation.
        cointegration_pvalue: P-value threshold for cointegration test.
        use_dynamic_hedge_ratio: Whether to dynamically update hedge ratios.
        spread_ma_period: Period for spread moving average.
    """

    name: str = "PairsTradingStrategy"
    pairs: list[PairConfig] = field(default_factory=list)
    lookback_period: int = 60
    zscore_entry_threshold: float = 2.0
    zscore_exit_threshold: float = 0.0
    zscore_stop_loss: float = 4.0
    min_half_life: float = 5.0
    max_half_life: float = 120.0
    recalc_period: int = 20
    cointegration_pvalue: float = 0.05
    use_dynamic_hedge_ratio: bool = True
    spread_ma_period: int = 20
    min_data_points: int = 100


class PairsTradingStrategy(BaseStrategy):
    """
    Statistical arbitrage strategy based on pairs cointegration.

    Uses the Engle-Granger two-step cointegration test to identify
    mean-reverting spread relationships between asset pairs. Trades
    are entered when the spread z-score exceeds entry thresholds
    and exited when the spread reverts to the mean.

    The strategy:
    1. Tests for cointegration using Augmented Dickey-Fuller test
    2. Calculates optimal hedge ratio via OLS regression
    3. Monitors spread z-score for entry/exit signals
    4. Manages positions on both legs of the pair simultaneously

    Example:
        >>> pairs = [
        ...     PairConfig(symbol_a="GLD", symbol_b="GDX"),
        ...     PairConfig(symbol_a="XOM", symbol_b="CVX"),
        ... ]
        >>> config = PairsTradingConfig(
        ...     pairs=pairs,
        ...     zscore_entry_threshold=2.0,
        ...     zscore_exit_threshold=0.5,
        ... )
        >>> strategy = PairsTradingStrategy(config)
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, config: PairsTradingConfig | None = None) -> None:
        """
        Initialize the pairs trading strategy.

        Args:
            config: Strategy configuration. Defaults to PairsTradingConfig.
        """
        super().__init__(config or PairsTradingConfig())
        self._config: PairsTradingConfig = self.config  # type: ignore
        self._pair_states: dict[str, dict[str, Any]] = {}
        self._last_recalc: dict[str, int] = {}

    def _get_pair_key(self, pair: PairConfig) -> str:
        """Generate a unique key for a pair."""
        return f"{pair.symbol_a}_{pair.symbol_b}"

    def test_cointegration(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> tuple[bool, float, float]:
        """
        Test for cointegration using Engle-Granger two-step method.

        Step 1: Run OLS regression of A on B to get hedge ratio
        Step 2: Test residuals for stationarity using ADF test

        Args:
            prices_a: Price series for first asset.
            prices_b: Price series for second asset.

        Returns:
            Tuple of (is_cointegrated, pvalue, hedge_ratio).
        """
        try:
            from statsmodels.tsa.stattools import adfuller
        except ImportError:
            # Fallback to basic correlation if statsmodels not available
            correlation = prices_a.corr(prices_b)
            hedge_ratio = np.cov(prices_a, prices_b)[0, 1] / np.var(prices_b)
            # Simple stationarity proxy
            spread = prices_a - hedge_ratio * prices_b
            spread_diff = spread.diff().dropna()
            # Crude stationarity test using variance ratio
            is_stationary = spread_diff.std() < spread.std() * 0.5
            return is_stationary, 0.05 if is_stationary else 0.5, hedge_ratio

        # Step 1: OLS regression to get hedge ratio (beta)
        # A = alpha + beta * B + epsilon
        X = np.column_stack([np.ones(len(prices_b)), prices_b.values])
        y = prices_a.values

        # Solve normal equations
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        hedge_ratio = beta[1]

        # Step 2: Calculate spread (residuals)
        spread = prices_a - hedge_ratio * prices_b

        # Step 3: ADF test on spread
        adf_result = adfuller(spread.dropna(), maxlag=1, autolag=None)
        pvalue = adf_result[1]

        is_cointegrated = pvalue < self._config.cointegration_pvalue

        return is_cointegrated, pvalue, hedge_ratio

    def calculate_hedge_ratio(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> float:
        """
        Calculate optimal hedge ratio using OLS regression.

        The hedge ratio represents how many units of asset B to trade
        for each unit of asset A to create a stationary spread.

        Args:
            prices_a: Price series for first asset.
            prices_b: Price series for second asset.

        Returns:
            Hedge ratio (beta coefficient).
        """
        # Use rolling OLS for more recent weighting
        X = np.column_stack([np.ones(len(prices_b)), prices_b.values])
        y = prices_a.values

        # Solve using least squares
        beta = np.linalg.lstsq(X, y, rcond=None)[0]

        return float(beta[1])

    def calculate_spread(
        self,
        prices_a: pd.Series,
        prices_b: pd.Series,
        hedge_ratio: float,
    ) -> pd.Series:
        """
        Calculate the spread between the two assets.

        Spread = Price_A - hedge_ratio * Price_B

        Args:
            prices_a: Price series for first asset.
            prices_b: Price series for second asset.
            hedge_ratio: Hedge ratio for the pair.

        Returns:
            Series with spread values.
        """
        return prices_a - hedge_ratio * prices_b

    def calculate_zscore(
        self,
        spread: pd.Series,
        lookback: int | None = None,
    ) -> pd.Series:
        """
        Calculate rolling z-score of the spread.

        Z-score = (spread - mean) / std

        Args:
            spread: Spread series.
            lookback: Lookback period for mean/std calculation.

        Returns:
            Series with z-score values.
        """
        lookback = lookback or self._config.lookback_period

        spread_mean = spread.rolling(window=lookback).mean()
        spread_std = spread.rolling(window=lookback).std()

        zscore = (spread - spread_mean) / spread_std

        return zscore

    def estimate_half_life(self, spread: pd.Series) -> float:
        """
        Estimate mean-reversion half-life using Ornstein-Uhlenbeck model.

        Uses the formula: half_life = -log(2) / log(1 + theta)
        where theta is the mean-reversion speed from AR(1) regression.

        Args:
            spread: Spread series.

        Returns:
            Estimated half-life in periods.
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align series
        min_len = min(len(spread_lag), len(spread_diff))
        spread_lag = spread_lag.iloc[-min_len:]
        spread_diff = spread_diff.iloc[-min_len:]

        if len(spread_lag) < 10:
            return float("inf")

        # AR(1) regression: spread_t - spread_{t-1} = theta * spread_{t-1} + epsilon
        # This gives us theta (mean-reversion speed)
        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values

        try:
            theta = np.linalg.lstsq(X, y, rcond=None)[0][0]

            if theta >= 0:
                return float("inf")  # No mean reversion

            half_life = -np.log(2) / np.log(1 + theta)
            return max(1.0, half_life)  # At least 1 period

        except (np.linalg.LinAlgError, ValueError):
            return float("inf")

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate pairs trading indicators for all configured pairs.

        Args:
            data: DataFrame with price data for all symbols.
                  Expected to have multi-symbol data with 'symbol' column
                  or separate columns for each symbol's close price.

        Returns:
            DataFrame with spread and z-score for each pair.
        """
        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # Process each pair
        for pair in self._config.pairs:
            pair_key = self._get_pair_key(pair)

            # Try to get prices from data
            prices_a = self._extract_prices(df, pair.symbol_a)
            prices_b = self._extract_prices(df, pair.symbol_b)

            if prices_a is None or prices_b is None:
                continue

            # Align series
            aligned = pd.concat([prices_a, prices_b], axis=1).dropna()
            if len(aligned) < self._config.min_data_points:
                continue

            prices_a = aligned.iloc[:, 0]
            prices_b = aligned.iloc[:, 1]

            # Get or calculate hedge ratio
            hedge_ratio = self._get_hedge_ratio(pair, prices_a, prices_b)

            # Calculate spread and z-score
            spread = self.calculate_spread(prices_a, prices_b, hedge_ratio)
            zscore = self.calculate_zscore(spread)

            # Store in dataframe
            df[f"{pair_key}_spread"] = spread
            df[f"{pair_key}_zscore"] = zscore
            df[f"{pair_key}_hedge_ratio"] = hedge_ratio

            # Update pair state
            self._pair_states[pair_key] = {
                "hedge_ratio": hedge_ratio,
                "latest_zscore": zscore.iloc[-1] if len(zscore) > 0 else 0,
                "spread_mean": spread.rolling(self._config.spread_ma_period).mean().iloc[-1],
                "spread_std": spread.rolling(self._config.spread_ma_period).std().iloc[-1],
            }

        return df

    def _extract_prices(
        self,
        data: pd.DataFrame,
        symbol: str,
    ) -> pd.Series | None:
        """
        Extract price series for a symbol from the data.

        Args:
            data: DataFrame with price data.
            symbol: Symbol to extract.

        Returns:
            Price series or None if not found.
        """
        # Try different column naming conventions
        symbol_lower = symbol.lower()

        # Check for symbol-specific close column
        if f"{symbol_lower}_close" in data.columns:
            return data[f"{symbol_lower}_close"]

        if f"{symbol}_close" in data.columns:
            return data[f"{symbol}_close"]

        # Check for symbol column with filtering
        if "symbol" in data.columns:
            symbol_data = data[data["symbol"].str.upper() == symbol.upper()]
            if not symbol_data.empty and "close" in symbol_data.columns:
                return symbol_data["close"]

        # Check if symbol is a column itself (pivoted data)
        if symbol in data.columns:
            return data[symbol]

        if symbol_lower in data.columns:
            return data[symbol_lower]

        return None

    def _get_hedge_ratio(
        self,
        pair: PairConfig,
        prices_a: pd.Series,
        prices_b: pd.Series,
    ) -> float:
        """
        Get hedge ratio for a pair, recalculating if needed.

        Args:
            pair: Pair configuration.
            prices_a: Price series for first asset.
            prices_b: Price series for second asset.

        Returns:
            Hedge ratio for the pair.
        """
        pair_key = self._get_pair_key(pair)

        # Use configured hedge ratio if not dynamic
        if not self._config.use_dynamic_hedge_ratio:
            return pair.hedge_ratio

        # Check if recalculation is needed
        current_period = len(prices_a)
        last_recalc = self._last_recalc.get(pair_key, 0)

        if current_period - last_recalc >= self._config.recalc_period:
            hedge_ratio = self.calculate_hedge_ratio(prices_a, prices_b)
            self._last_recalc[pair_key] = current_period

            # Update pair config
            pair.hedge_ratio = hedge_ratio
            return hedge_ratio

        # Return existing hedge ratio
        return pair.hedge_ratio

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate pairs trading signals based on z-score thresholds.

        Entry signals:
        - Z-score > +entry_threshold: Short A, Long B (spread too high)
        - Z-score < -entry_threshold: Long A, Short B (spread too low)

        Exit signals:
        - Z-score crosses exit_threshold: Close positions

        Args:
            data: DataFrame with price data for all pair symbols.

        Returns:
            List of Signal objects for entries/exits.
        """
        if data is None or data.empty:
            return []

        signals: list[Signal] = []
        df = self.calculate_indicators(data)

        # Get timestamp
        timestamp = datetime.now()
        if "datetime" in df.columns:
            ts = df["datetime"].iloc[-1]
            timestamp = ts.to_pydatetime() if isinstance(ts, pd.Timestamp) else ts

        # Generate signals for each pair
        for pair in self._config.pairs:
            pair_key = self._get_pair_key(pair)

            zscore_col = f"{pair_key}_zscore"
            if zscore_col not in df.columns:
                continue

            zscore = df[zscore_col].iloc[-1]
            prev_zscore = df[zscore_col].iloc[-2] if len(df) > 1 else zscore

            if pd.isna(zscore):
                continue

            pair_state = self._pair_states.get(pair_key, {})
            hedge_ratio = pair_state.get("hedge_ratio", pair.hedge_ratio)

            # Get current prices
            prices_a = self._extract_prices(df, pair.symbol_a)
            prices_b = self._extract_prices(df, pair.symbol_b)

            if prices_a is None or prices_b is None:
                continue

            price_a = float(prices_a.iloc[-1])
            price_b = float(prices_b.iloc[-1])

            # Check for entry signals
            has_position_a = self.has_position(pair.symbol_a)
            has_position_b = self.has_position(pair.symbol_b)
            has_pair_position = has_position_a and has_position_b

            metadata = {
                "pair": pair_key,
                "zscore": zscore,
                "hedge_ratio": hedge_ratio,
                "price_a": price_a,
                "price_b": price_b,
            }

            # Entry: Spread too high (short A, long B)
            if (
                zscore > self._config.zscore_entry_threshold
                and not has_pair_position
            ):
                # Short symbol A
                signal_a = Signal(
                    symbol=pair.symbol_a,
                    signal_type=SignalType.SELL,
                    strength=-min(zscore / self._config.zscore_stop_loss, 1.0),
                    timestamp=timestamp,
                    price=price_a,
                    metadata={**metadata, "leg": "short", "action": "entry"},
                )
                signals.append(signal_a)
                self.add_signal(signal_a)

                # Long symbol B
                signal_b = Signal(
                    symbol=pair.symbol_b,
                    signal_type=SignalType.BUY,
                    strength=min(zscore / self._config.zscore_stop_loss, 1.0),
                    timestamp=timestamp,
                    price=price_b,
                    metadata={**metadata, "leg": "long", "action": "entry"},
                )
                signals.append(signal_b)
                self.add_signal(signal_b)

            # Entry: Spread too low (long A, short B)
            elif (
                zscore < -self._config.zscore_entry_threshold
                and not has_pair_position
            ):
                # Long symbol A
                signal_a = Signal(
                    symbol=pair.symbol_a,
                    signal_type=SignalType.BUY,
                    strength=min(abs(zscore) / self._config.zscore_stop_loss, 1.0),
                    timestamp=timestamp,
                    price=price_a,
                    metadata={**metadata, "leg": "long", "action": "entry"},
                )
                signals.append(signal_a)
                self.add_signal(signal_a)

                # Short symbol B
                signal_b = Signal(
                    symbol=pair.symbol_b,
                    signal_type=SignalType.SELL,
                    strength=-min(abs(zscore) / self._config.zscore_stop_loss, 1.0),
                    timestamp=timestamp,
                    price=price_b,
                    metadata={**metadata, "leg": "short", "action": "entry"},
                )
                signals.append(signal_b)
                self.add_signal(signal_b)

            # Exit: Z-score reverted to mean
            elif has_pair_position:
                # Check for exit condition (z-score crossed exit threshold)
                exit_condition = (
                    (prev_zscore > self._config.zscore_exit_threshold >= zscore)
                    or (prev_zscore < -self._config.zscore_exit_threshold <= zscore)
                    or abs(zscore) < self._config.zscore_exit_threshold
                )

                # Check for stop loss
                stop_condition = abs(zscore) > self._config.zscore_stop_loss

                if exit_condition or stop_condition:
                    action = "stop_loss" if stop_condition else "exit"

                    # Close position A
                    pos_a = self.get_position(pair.symbol_a)
                    if pos_a:
                        signal_a = Signal(
                            symbol=pair.symbol_a,
                            signal_type=SignalType.SELL if pos_a.is_long else SignalType.BUY,
                            strength=0.5,
                            timestamp=timestamp,
                            price=price_a,
                            metadata={**metadata, "action": action},
                        )
                        signals.append(signal_a)
                        self.add_signal(signal_a)

                    # Close position B
                    pos_b = self.get_position(pair.symbol_b)
                    if pos_b:
                        signal_b = Signal(
                            symbol=pair.symbol_b,
                            signal_type=SignalType.SELL if pos_b.is_long else SignalType.BUY,
                            strength=0.5,
                            timestamp=timestamp,
                            price=price_b,
                            metadata={**metadata, "action": action},
                        )
                        signals.append(signal_b)
                        self.add_signal(signal_b)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """
        Calculate position size for pairs trading.

        For pairs trading, position sizes are balanced based on the hedge ratio
        to ensure dollar-neutral exposure.

        Args:
            signal: Trading signal.
            capital: Available capital.
            risk_per_trade: Maximum risk per trade (defaults to config value).

        Returns:
            Number of shares to trade.
        """
        risk = risk_per_trade or self._config.risk_per_trade

        # Get pair information from metadata
        metadata = signal.metadata
        hedge_ratio = metadata.get("hedge_ratio", 1.0)
        leg = metadata.get("leg", "")

        # Base allocation per pair
        pair_allocation = capital * risk * 2  # 2x for both legs

        # Calculate shares based on leg
        if leg == "long" or leg == "short":
            # This is the B leg (hedged position)
            position_value = pair_allocation / 2 * hedge_ratio
        else:
            # This is the A leg (primary position)
            position_value = pair_allocation / 2

        # Calculate shares
        if signal.price <= 0:
            return 0.0

        shares = position_value / signal.price

        # Adjust sign based on signal type
        return shares if signal.is_buy else -shares

    def get_pair_statistics(self, pair: PairConfig) -> dict[str, Any]:
        """
        Get current statistics for a pair.

        Args:
            pair: Pair configuration.

        Returns:
            Dictionary with pair statistics.
        """
        pair_key = self._get_pair_key(pair)
        state = self._pair_states.get(pair_key, {})

        return {
            "pair": pair_key,
            "hedge_ratio": state.get("hedge_ratio", pair.hedge_ratio),
            "zscore": state.get("latest_zscore", 0),
            "spread_mean": state.get("spread_mean", 0),
            "spread_std": state.get("spread_std", 0),
            "has_position": (
                self.has_position(pair.symbol_a)
                and self.has_position(pair.symbol_b)
            ),
        }

    def add_pair(
        self,
        symbol_a: str,
        symbol_b: str,
        hedge_ratio: float = 1.0,
    ) -> PairConfig:
        """
        Add a new trading pair.

        Args:
            symbol_a: First symbol in the pair.
            symbol_b: Second symbol in the pair.
            hedge_ratio: Initial hedge ratio.

        Returns:
            Created PairConfig object.
        """
        pair = PairConfig(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            hedge_ratio=hedge_ratio,
        )
        self._config.pairs.append(pair)
        return pair

    def remove_pair(self, symbol_a: str, symbol_b: str) -> bool:
        """
        Remove a trading pair.

        Args:
            symbol_a: First symbol in the pair.
            symbol_b: Second symbol in the pair.

        Returns:
            True if pair was removed, False if not found.
        """
        for i, pair in enumerate(self._config.pairs):
            if pair.symbol_a == symbol_a and pair.symbol_b == symbol_b:
                self._config.pairs.pop(i)
                pair_key = self._get_pair_key(pair)
                self._pair_states.pop(pair_key, None)
                self._last_recalc.pop(pair_key, None)
                return True
        return False


__all__ = ["PairsTradingStrategy", "PairsTradingConfig", "PairConfig"]

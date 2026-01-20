"""
Aggressive Momentum Strategy Module

Implements an aggressive momentum-based trading strategy designed to beat
buy-and-hold in bull markets through early entries, trend-following bias,
pyramiding winners, and letting winners run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType, StrategyConfig, Position


@dataclass
class AggressiveMomentumConfig(StrategyConfig):
    """
    Configuration for Aggressive Momentum Strategy.

    Key differences from standard momentum:
    1. Lower entry thresholds - get in early
    2. Trend-following bias - stay long in uptrends
    3. Pyramid into winners - add to winning positions
    4. Quick entries, slow exits - let winners run

    Attributes:
        rsi_period: Period for RSI calculation.
        rsi_buy_threshold: RSI threshold for buy signals (lower = more aggressive).
        rsi_sell_threshold: RSI threshold for sell signals.
        fast_ma_period: Period for fast moving average.
        slow_ma_period: Period for slow moving average.
        trend_ma_period: Period for trend filter MA (50-day default).
        ma_type: Type of moving average ('sma', 'ema').
        macd_fast: MACD fast period.
        macd_slow: MACD slow period.
        macd_signal: MACD signal line period.
        min_market_exposure: Minimum portfolio exposure (0.0-1.0).
        base_position_size: Base position size per stock (fraction of portfolio).
        max_positions: Maximum number of concurrent positions.
        pyramid_threshold: Gain threshold to add to position (e.g., 0.10 = 10%).
        pyramid_size: Additional size per pyramid (fraction of base).
        max_pyramids: Maximum number of pyramids per position.
        trailing_stop_pct: Trailing stop percentage (e.g., 0.15 = 15%).
        rebalance_frequency: Days between rebalancing.
    """

    name: str = "AggressiveMomentumStrategy"

    # RSI Parameters (more aggressive than 70/30)
    rsi_period: int = 14
    rsi_buy_threshold: float = 55.0  # Buy when RSI crosses above 45 (entry)
    rsi_sell_threshold: float = 40.0  # Sell only when RSI drops below 40
    rsi_entry_trigger: float = 45.0  # RSI level to trigger buy consideration

    # Moving Average Parameters (faster signals)
    fast_ma_period: int = 5
    slow_ma_period: int = 15
    trend_ma_period: int = 50  # Trend filter
    ma_type: str = "ema"

    # MACD Parameters
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # Position Sizing
    min_market_exposure: float = 0.60  # Always keep 60% invested
    base_position_size: float = 0.20  # 20% per position (5 stocks = 100%)
    max_positions: int = 5

    # Pyramiding
    pyramid_threshold: float = 0.10  # Add when up 10%
    pyramid_size: float = 0.25  # Add 25% more
    max_pyramids: int = 3

    # Risk Management
    trailing_stop_pct: float = 0.15  # 15% trailing stop (give room)
    rebalance_frequency: int = 21  # Monthly rebalance (~21 trading days)

    # Signal Thresholds
    signal_threshold: float = 0.2  # Lower threshold for entries
    min_data_points: int = 60  # Need enough for 50-day MA


@dataclass
class PyramidLevel:
    """Tracks pyramid levels for a position."""

    entry_price: float
    quantity: float
    timestamp: datetime


@dataclass
class AggressivePosition(Position):
    """
    Extended position tracking for aggressive momentum strategy.

    Adds tracking for:
    - Pyramid levels
    - Trailing stop high watermark
    - Position ranking
    """

    pyramid_levels: list[PyramidLevel] = field(default_factory=list)
    high_watermark: float = 0.0
    trailing_stop_price: float = 0.0
    momentum_rank: float = 0.0

    def update_trailing_stop(self, current_price: float, stop_pct: float) -> None:
        """Update trailing stop based on new high."""
        if current_price > self.high_watermark:
            self.high_watermark = current_price
            self.trailing_stop_price = current_price * (1 - stop_pct)

    @property
    def pyramid_count(self) -> int:
        """Get number of pyramid levels."""
        return len(self.pyramid_levels)


class AggressiveMomentumStrategy(BaseStrategy):
    """
    Aggressive momentum strategy designed to beat buy-and-hold in bull markets.

    Key differences from standard momentum:
    1. Lower entry thresholds - get in early
    2. Trend-following bias - stay long in uptrends
    3. Pyramid into winners - add to winning positions
    4. Quick entries, slow exits - let winners run

    Signal Logic:
        BUY when: RSI crosses above 45 AND price > 50-day MA AND MACD > signal
        PYRAMID when: Position up >10%, add 25% more (max 3 pyramids)
        SELL only when: RSI < 40 OR price < 50-day MA OR trailing stop hit
        HOLD BIAS: Default to holding, not selling

    Position Sizing:
        - Base position: 20% of portfolio per stock (5 stocks = 100% invested)
        - Minimum exposure: Always keep 60% invested in best-ranked stocks
        - Rebalance monthly to equal-weight top performers

    Example:
        >>> config = AggressiveMomentumConfig(
        ...     rsi_buy_threshold=55,
        ...     pyramid_threshold=0.10,
        ...     trailing_stop_pct=0.15,
        ... )
        >>> strategy = AggressiveMomentumStrategy(config)
        >>> signals = strategy.generate_signals(data)
    """

    def __init__(self, config: AggressiveMomentumConfig | None = None) -> None:
        """
        Initialize the aggressive momentum strategy.

        Args:
            config: Strategy configuration. Defaults to AggressiveMomentumConfig.
        """
        super().__init__(config or AggressiveMomentumConfig())
        self._config: AggressiveMomentumConfig = self.config  # type: ignore
        self._aggressive_positions: dict[str, AggressivePosition] = {}
        self._last_rebalance: datetime | None = None
        self._days_since_rebalance: int = 0

    def calculate_rsi(self, prices: pd.Series, period: int | None = None) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        Args:
            prices: Series of prices (typically close prices).
            period: RSI period. Defaults to config value.

        Returns:
            Series with RSI values.
        """
        period = period or self._config.rsi_period

        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()

        # Avoid division by zero
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Default to neutral if undefined

    def calculate_moving_average(
        self,
        data: pd.Series,
        period: int,
        ma_type: str | None = None,
    ) -> pd.Series:
        """
        Calculate moving average.

        Args:
            data: Series of values.
            period: Moving average period.
            ma_type: Type of MA ('sma', 'ema'). Defaults to config value.

        Returns:
            Series with moving average values.
        """
        ma_type = ma_type or self._config.ma_type

        if ma_type == "sma":
            return data.rolling(window=period).mean()
        elif ma_type == "ema":
            return data.ewm(span=period, adjust=False).mean()
        else:
            raise ValueError(f"Unknown MA type: {ma_type}")

    def calculate_macd(
        self,
        prices: pd.Series,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD indicator.

        Args:
            prices: Series of prices.

        Returns:
            Tuple of (macd_line, signal_line, histogram).
        """
        fast_ema = prices.ewm(span=self._config.macd_fast, adjust=False).mean()
        slow_ema = prices.ewm(span=self._config.macd_slow, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=self._config.macd_signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def calculate_momentum_rank(self, data: pd.DataFrame) -> float:
        """
        Calculate momentum rank for a stock (0-100).

        Higher rank = stronger momentum characteristics.

        Args:
            data: DataFrame with OHLCV data and indicators.

        Returns:
            Momentum rank score (0-100).
        """
        if len(data) < 20:
            return 50.0

        latest = data.iloc[-1]
        score = 0.0
        max_score = 0.0

        # Price above trend MA (25 points)
        max_score += 25
        if latest.get("close", 0) > latest.get("trend_ma", 0):
            score += 25

        # RSI strength (25 points)
        max_score += 25
        rsi = latest.get("rsi", 50)
        if 45 <= rsi <= 70:  # Sweet spot for momentum
            score += 25 * ((rsi - 45) / 25)
        elif rsi > 70:  # Still bullish but might be extended
            score += 15

        # MACD above signal (25 points)
        max_score += 25
        if latest.get("macd", 0) > latest.get("macd_signal", 0):
            score += 25

        # Recent performance (25 points)
        max_score += 25
        returns_20d = data["close"].pct_change(20).iloc[-1] if len(data) >= 21 else 0
        if returns_20d > 0:
            score += min(25, returns_20d * 100)  # Cap at 25

        return (score / max_score) * 100 if max_score > 0 else 50.0

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all momentum indicators for aggressive strategy.

        Args:
            data: DataFrame with OHLCV data.

        Returns:
            DataFrame with added indicator columns.
        """
        df = data.copy()

        # Normalize column names to lowercase
        df.columns = [col.lower() for col in df.columns]

        # RSI
        df["rsi"] = self.calculate_rsi(df["close"])
        df["rsi_prev"] = df["rsi"].shift(1)

        # Moving averages
        df["fast_ma"] = self.calculate_moving_average(
            df["close"], self._config.fast_ma_period
        )
        df["slow_ma"] = self.calculate_moving_average(
            df["close"], self._config.slow_ma_period
        )
        df["trend_ma"] = self.calculate_moving_average(
            df["close"], self._config.trend_ma_period
        )

        # MA relationships
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        df["ma_diff_pct"] = df["ma_diff"] / df["slow_ma"] * 100
        df["above_trend"] = df["close"] > df["trend_ma"]
        df["above_trend_prev"] = df["above_trend"].shift(1)

        # MACD
        df["macd"], df["macd_signal"], df["macd_hist"] = self.calculate_macd(
            df["close"]
        )
        df["macd_bullish"] = df["macd"] > df["macd_signal"]
        df["macd_bullish_prev"] = df["macd_bullish"].shift(1)

        # Price momentum
        df["returns_1d"] = df["close"].pct_change(1)
        df["returns_5d"] = df["close"].pct_change(5)
        df["returns_20d"] = df["close"].pct_change(20)

        # Volatility for position sizing
        df["volatility"] = df["returns_1d"].rolling(window=20).std() * np.sqrt(252)

        # Volume analysis
        if "volume" in df.columns:
            df["volume_ma"] = df["volume"].rolling(window=20).mean()
            df["volume_ratio"] = df["volume"] / df["volume_ma"]
        else:
            df["volume_ratio"] = 1.0

        # High watermark for trailing stops
        df["high_20d"] = df["high"].rolling(window=20).max()

        return df

    def _check_buy_conditions(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        symbol: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if buy conditions are met.

        Buy when: RSI crosses above 45 AND price > 50-day MA AND MACD > signal

        Args:
            row: Current row of data.
            prev_row: Previous row of data.
            symbol: Stock symbol.

        Returns:
            Tuple of (should_buy, metadata).
        """
        metadata: dict[str, Any] = {}

        # Extract values safely
        rsi = row.get("rsi", 50)
        rsi_prev = prev_row.get("rsi", 50) if prev_row is not None else 50
        above_trend = row.get("above_trend", False)
        macd_bullish = row.get("macd_bullish", False)

        # Primary conditions
        rsi_entry = (
            rsi > self._config.rsi_entry_trigger
            and rsi_prev <= self._config.rsi_entry_trigger
        ) or rsi > self._config.rsi_buy_threshold

        trend_filter = above_trend
        macd_confirm = macd_bullish

        # All conditions must be met
        should_buy = rsi_entry and trend_filter and macd_confirm

        # Don't buy if already at max position
        if symbol in self._aggressive_positions:
            pos = self._aggressive_positions[symbol]
            if pos.pyramid_count >= self._config.max_pyramids:
                should_buy = False

        metadata = {
            "rsi_entry": rsi_entry,
            "trend_filter": trend_filter,
            "macd_confirm": macd_confirm,
            "rsi": rsi,
            "above_trend": above_trend,
            "macd_bullish": macd_bullish,
        }

        return should_buy, metadata

    def _check_pyramid_conditions(
        self,
        row: pd.Series,
        symbol: str,
    ) -> tuple[bool, dict[str, Any]]:
        """
        Check if pyramiding conditions are met.

        Pyramid when: Position up >10%, add 25% more (max 3 pyramids)

        Args:
            row: Current row of data.
            symbol: Stock symbol.

        Returns:
            Tuple of (should_pyramid, metadata).
        """
        if symbol not in self._aggressive_positions:
            return False, {}

        position = self._aggressive_positions[symbol]
        current_price = row.get("close", 0)

        # Update current price
        position.current_price = current_price

        # Check pyramid conditions
        gain_pct = position.unrealized_pnl_pct / 100  # Convert to decimal

        should_pyramid = (
            gain_pct >= self._config.pyramid_threshold
            and position.pyramid_count < self._config.max_pyramids
            and row.get("above_trend", False)  # Still in uptrend
            and row.get("macd_bullish", False)  # MACD still bullish
        )

        metadata = {
            "gain_pct": gain_pct,
            "pyramid_count": position.pyramid_count,
            "max_pyramids": self._config.max_pyramids,
        }

        return should_pyramid, metadata

    def _check_sell_conditions(
        self,
        row: pd.Series,
        prev_row: pd.Series,
        symbol: str,
    ) -> tuple[bool, str, dict[str, Any]]:
        """
        Check if sell conditions are met.

        Sell only when: RSI < 40 OR price < 50-day MA OR trailing stop hit

        Args:
            row: Current row of data.
            prev_row: Previous row of data.
            symbol: Stock symbol.

        Returns:
            Tuple of (should_sell, reason, metadata).
        """
        if symbol not in self._aggressive_positions:
            return False, "", {}

        position = self._aggressive_positions[symbol]
        current_price = row.get("close", 0)

        # Update trailing stop
        position.update_trailing_stop(current_price, self._config.trailing_stop_pct)

        # Extract values
        rsi = row.get("rsi", 50)
        above_trend = row.get("above_trend", True)

        # Sell conditions (more lenient - hold bias)
        rsi_exit = rsi < self._config.rsi_sell_threshold
        trend_break = not above_trend and prev_row.get("above_trend", True)
        trailing_stop_hit = (
            current_price <= position.trailing_stop_price
            and position.trailing_stop_price > 0
        )

        reason = ""
        if trailing_stop_hit:
            reason = "trailing_stop"
        elif trend_break:
            reason = "trend_break"
        elif rsi_exit:
            reason = "rsi_exit"

        should_sell = trailing_stop_hit or trend_break or rsi_exit

        metadata = {
            "rsi": rsi,
            "above_trend": above_trend,
            "trailing_stop_price": position.trailing_stop_price,
            "current_price": current_price,
            "trailing_stop_hit": trailing_stop_hit,
            "trend_break": trend_break,
            "rsi_exit": rsi_exit,
        }

        return should_sell, reason, metadata

    def _calculate_signal_strength(
        self,
        row: pd.Series,
        signal_type: str,
        metadata: dict[str, Any],
    ) -> float:
        """
        Calculate signal strength based on multiple factors.

        Args:
            row: Current row of data.
            signal_type: Type of signal ('buy', 'sell', 'pyramid').
            metadata: Signal metadata.

        Returns:
            Signal strength between -1.0 and 1.0.
        """
        if signal_type == "sell":
            # Sell strength based on urgency
            if metadata.get("trailing_stop_hit"):
                return -1.0  # Maximum urgency
            elif metadata.get("trend_break"):
                return -0.8
            else:
                return -0.6

        # Buy/Pyramid strength
        rsi = row.get("rsi", 50)
        macd_hist = row.get("macd_hist", 0)
        returns_20d = row.get("returns_20d", 0)

        # RSI component (0.4 weight)
        if 45 <= rsi <= 65:
            rsi_score = 0.4 * ((rsi - 45) / 20)  # Higher in sweet spot
        elif rsi > 65:
            rsi_score = 0.3  # Still positive but less confident
        else:
            rsi_score = 0.2

        # MACD histogram component (0.3 weight)
        macd_score = min(0.3, max(0, macd_hist * 10))

        # Momentum component (0.3 weight)
        momentum_score = min(0.3, max(0, returns_20d * 3))

        strength = rsi_score + macd_score + momentum_score

        # Pyramid signals are slightly weaker
        if signal_type == "pyramid":
            strength *= 0.8

        return min(1.0, strength)

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """
        Generate aggressive momentum-based trading signals.

        Signal priority:
        1. Trailing stop exits (protect profits)
        2. New buy entries (catch uptrends early)
        3. Pyramid additions (scale into winners)
        4. Trend break exits (respect the trend)

        Args:
            data: DataFrame with OHLCV data. Must include 'symbol' column
                  or process a single symbol.

        Returns:
            List of Signal objects.
        """
        if not self.validate_data(data):
            return []

        signals: list[Signal] = []
        df = self.calculate_indicators(data)

        # Get symbol from data or use default
        symbol_col = df.get("symbol", pd.Series(["UNKNOWN"] * len(df)))
        symbol = symbol_col.iloc[-1] if len(symbol_col) > 0 else "UNKNOWN"
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[-1]
        symbol = str(symbol)

        # Get latest and previous rows
        if len(df) < 2:
            return []

        latest = df.iloc[-1]
        prev = df.iloc[-2]

        # Get timestamp
        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        current_price = float(latest.get("close", 0))

        # Check for existing position
        has_position = symbol in self._aggressive_positions

        # Priority 1: Check trailing stop (protect profits)
        if has_position:
            should_sell, reason, sell_metadata = self._check_sell_conditions(
                latest, prev, symbol
            )

            if should_sell:
                strength = self._calculate_signal_strength(
                    latest, "sell", sell_metadata
                )
                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=strength,
                    timestamp=timestamp,
                    price=current_price,
                    metadata={
                        "reason": reason,
                        "signal_type": "exit",
                        **sell_metadata,
                    },
                )
                signals.append(signal)
                self.add_signal(signal)

                # Remove position tracking
                if symbol in self._aggressive_positions:
                    del self._aggressive_positions[symbol]

                return signals

        # Priority 2: Check for new entry (if no position)
        if not has_position:
            # Check if we have room for more positions
            if len(self._aggressive_positions) < self._config.max_positions:
                should_buy, buy_metadata = self._check_buy_conditions(
                    latest, prev, symbol
                )

                if should_buy:
                    strength = self._calculate_signal_strength(
                        latest, "buy", buy_metadata
                    )

                    if strength > self._config.signal_threshold:
                        signal = Signal(
                            symbol=symbol,
                            signal_type=SignalType.BUY,
                            strength=strength,
                            timestamp=timestamp,
                            price=current_price,
                            metadata={
                                "signal_type": "entry",
                                "momentum_rank": self.calculate_momentum_rank(df),
                                **buy_metadata,
                            },
                        )
                        signals.append(signal)
                        self.add_signal(signal)

                        # Create position tracking
                        self._aggressive_positions[symbol] = AggressivePosition(
                            symbol=symbol,
                            quantity=0,  # Will be set by position sizing
                            entry_price=current_price,
                            entry_time=timestamp,
                            current_price=current_price,
                            high_watermark=current_price,
                            trailing_stop_price=current_price
                            * (1 - self._config.trailing_stop_pct),
                            momentum_rank=self.calculate_momentum_rank(df),
                        )

                        return signals

        # Priority 3: Check for pyramid (if has position)
        if has_position:
            should_pyramid, pyramid_metadata = self._check_pyramid_conditions(
                latest, symbol
            )

            if should_pyramid:
                strength = self._calculate_signal_strength(
                    latest, "pyramid", pyramid_metadata
                )

                signal = Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    strength=strength,
                    timestamp=timestamp,
                    price=current_price,
                    metadata={
                        "signal_type": "pyramid",
                        "pyramid_level": pyramid_metadata.get("pyramid_count", 0) + 1,
                        **pyramid_metadata,
                    },
                )
                signals.append(signal)
                self.add_signal(signal)

                # Update position tracking
                position = self._aggressive_positions[symbol]
                position.pyramid_levels.append(
                    PyramidLevel(
                        entry_price=current_price,
                        quantity=0,  # Will be set by position sizing
                        timestamp=timestamp,
                    )
                )

                return signals

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """
        Calculate position size for aggressive momentum strategy.

        Position Sizing Rules:
        - Base position: 20% of portfolio per stock
        - Pyramid addition: 25% of base position
        - Minimum total exposure: 60% of portfolio

        Args:
            signal: Trading signal.
            capital: Available capital.
            risk_per_trade: Not used (position sizing is fixed-percentage based).

        Returns:
            Number of shares to trade.
        """
        if signal.price <= 0:
            return 0.0

        signal_type = signal.metadata.get("signal_type", "entry")

        if signal.is_sell:
            # Sell entire position
            if signal.symbol in self._aggressive_positions:
                return -self._aggressive_positions[signal.symbol].quantity
            return 0.0

        # Calculate base position value
        base_position_value = capital * self._config.base_position_size

        if signal_type == "pyramid":
            # Pyramid: Add 25% of base position
            position_value = base_position_value * self._config.pyramid_size
        else:
            # New entry: Full base position
            position_value = base_position_value

        # Adjust for signal strength (0.8x to 1.2x)
        strength_multiplier = 0.8 + (signal.strength * 0.4)
        adjusted_value = position_value * strength_multiplier

        # Calculate shares
        shares = adjusted_value / signal.price

        return shares

    def calculate_minimum_exposure_signals(
        self,
        portfolio_data: dict[str, pd.DataFrame],
        current_capital: float,
        current_invested: float,
    ) -> list[Signal]:
        """
        Generate signals to maintain minimum market exposure.

        If current exposure drops below 60%, generate buy signals
        for the highest-ranked stocks.

        Args:
            portfolio_data: Dict of symbol -> DataFrame with OHLCV data.
            current_capital: Total capital available.
            current_invested: Current invested amount.

        Returns:
            List of buy signals to meet minimum exposure.
        """
        signals: list[Signal] = []

        current_exposure = current_invested / current_capital if current_capital > 0 else 0
        target_exposure = self._config.min_market_exposure

        if current_exposure >= target_exposure:
            return signals

        # Calculate how much more to invest
        additional_investment_needed = (target_exposure - current_exposure) * current_capital

        # Rank all stocks by momentum
        ranked_stocks: list[tuple[str, float, pd.DataFrame]] = []

        for symbol, data in portfolio_data.items():
            if symbol in self._aggressive_positions:
                continue  # Skip stocks we already hold

            if not self.validate_data(data):
                continue

            df = self.calculate_indicators(data)
            rank = self.calculate_momentum_rank(df)

            # Only consider stocks in uptrend
            if df.iloc[-1].get("above_trend", False):
                ranked_stocks.append((symbol, rank, df))

        # Sort by rank (highest first)
        ranked_stocks.sort(key=lambda x: x[1], reverse=True)

        # Generate buy signals for top-ranked stocks
        invested_so_far = 0.0

        for symbol, rank, df in ranked_stocks:
            if invested_so_far >= additional_investment_needed:
                break

            if len(self._aggressive_positions) >= self._config.max_positions:
                break

            latest = df.iloc[-1]
            current_price = float(latest.get("close", 0))

            timestamp = latest.get("datetime", datetime.now())
            if isinstance(timestamp, pd.Timestamp):
                timestamp = timestamp.to_pydatetime()

            signal = Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=rank / 100,  # Normalize rank to 0-1
                timestamp=timestamp,
                price=current_price,
                metadata={
                    "signal_type": "minimum_exposure",
                    "momentum_rank": rank,
                    "reason": "maintaining_minimum_exposure",
                },
            )
            signals.append(signal)

            # Track investment
            position_value = current_capital * self._config.base_position_size
            invested_so_far += position_value

        return signals

    def get_portfolio_statistics(self) -> dict[str, Any]:
        """
        Get current portfolio statistics.

        Returns:
            Dictionary with portfolio metrics.
        """
        total_positions = len(self._aggressive_positions)
        total_value = sum(
            pos.market_value for pos in self._aggressive_positions.values()
        )
        total_pnl = sum(
            pos.unrealized_pnl for pos in self._aggressive_positions.values()
        )

        pyramid_count = sum(
            pos.pyramid_count for pos in self._aggressive_positions.values()
        )

        return {
            "total_positions": total_positions,
            "max_positions": self._config.max_positions,
            "total_value": total_value,
            "total_unrealized_pnl": total_pnl,
            "total_pyramid_levels": pyramid_count,
            "positions": {
                symbol: {
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "current_price": pos.current_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "unrealized_pnl_pct": pos.unrealized_pnl_pct,
                    "pyramid_count": pos.pyramid_count,
                    "trailing_stop": pos.trailing_stop_price,
                    "high_watermark": pos.high_watermark,
                }
                for symbol, pos in self._aggressive_positions.items()
            },
        }

    def reset(self) -> None:
        """Reset strategy state."""
        self._aggressive_positions.clear()
        self._last_rebalance = None
        self._days_since_rebalance = 0
        self.clear_history()


__all__ = [
    "AggressiveMomentumStrategy",
    "AggressiveMomentumConfig",
    "AggressivePosition",
    "PyramidLevel",
]

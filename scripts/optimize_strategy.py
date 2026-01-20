#!/usr/bin/env python3
"""
Multi-Strategy Backtest Comparison Script

This script tests multiple trading strategies against buy-and-hold benchmark
for the year 2024 using live market data for top tech stocks.

Strategies Tested:
- Buy and Hold (benchmark)
- Momentum (current)
- Aggressive Momentum (new)
- Trend Following (existing)
- Ensemble (new)
- Breakout (new)

Each strategy is tested with parameter grid search to find optimal configurations.

Usage:
    python scripts/optimize_strategy.py

Output:
    - Ranked comparison table of all strategies vs Buy-and-Hold
    - Best strategy configuration saved to /opt/FinRL/config/best_strategy.json
    - Equity curve comparison chart saved to /opt/FinRL/output/strategy_comparison.png
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading.data.fetcher import SP500DataFetcher, CacheConfig
from src.trading.strategies.momentum import MomentumStrategy, MomentumConfig
from src.trading.strategies.trend_following import TrendFollowingStrategy, TrendFollowingConfig
from src.trading.strategies.base import BaseStrategy, Signal, SignalType, StrategyConfig
from src.trading.backtest.engine import BacktestEngine, BacktestConfig, BacktestResult
from src.trading.risk.manager import RiskConfig, PositionSizingMethod, StopLossType

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise during optimization
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM STRATEGIES
# =============================================================================

@dataclass
class AggressiveMomentumConfig(StrategyConfig):
    """Configuration for Aggressive Momentum Strategy."""
    name: str = "AggressiveMomentumStrategy"
    rsi_period: int = 10
    rsi_overbought: float = 60.0
    rsi_oversold: float = 40.0
    fast_ma_period: int = 5
    slow_ma_period: int = 15
    ma_type: str = "ema"
    signal_threshold: float = 0.10
    use_volume_confirmation: bool = False
    volume_ma_period: int = 10
    position_size: float = 0.20


class AggressiveMomentumStrategy(BaseStrategy):
    """
    Aggressive momentum strategy with tighter parameters for more frequent trading.
    Uses shorter lookback periods and lower thresholds for quicker entry/exit.
    """

    def __init__(self, config: AggressiveMomentumConfig | None = None) -> None:
        cfg = config or AggressiveMomentumConfig()
        cfg.min_data_points = 20  # Reduce min data requirement
        cfg.lookback_period = 20
        super().__init__(cfg)
        self._config: AggressiveMomentumConfig = self.config

    def calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all momentum indicators."""
        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # RSI
        df["rsi"] = self.calculate_rsi(df["close"], self._config.rsi_period)

        # Fast and slow EMAs
        df["fast_ma"] = df["close"].ewm(span=self._config.fast_ma_period, adjust=False).mean()
        df["slow_ma"] = df["close"].ewm(span=self._config.slow_ma_period, adjust=False).mean()

        # MA crossover signals
        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]
        df["ma_diff_pct"] = df["ma_diff"] / df["slow_ma"] * 100
        df["ma_cross"] = np.sign(df["ma_diff"]).diff()

        # Price momentum
        df["momentum"] = df["close"].pct_change(periods=self._config.fast_ma_period)
        df["momentum_5d"] = df["close"].pct_change(periods=5)

        return df

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate aggressive momentum signals."""
        if not self.validate_data(data):
            return []

        signals: list[Signal] = []
        df = self.calculate_indicators(data)

        symbol = df.get("symbol", pd.Series(["UNKNOWN"])).iloc[-1]
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[-1]

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        if pd.isna(latest.get("rsi")) or pd.isna(latest.get("fast_ma")):
            return []

        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Calculate signal strength
        rsi = latest["rsi"]
        ma_diff_pct = latest["ma_diff_pct"]
        momentum = latest.get("momentum", 0)

        # RSI component
        if rsi >= self._config.rsi_overbought:
            rsi_signal = -(rsi - self._config.rsi_overbought) / (100 - self._config.rsi_overbought)
        elif rsi <= self._config.rsi_oversold:
            rsi_signal = (self._config.rsi_oversold - rsi) / self._config.rsi_oversold
        else:
            mid = (self._config.rsi_overbought + self._config.rsi_oversold) / 2
            rsi_signal = (mid - rsi) / (self._config.rsi_overbought - self._config.rsi_oversold)

        # Combined strength
        strength = 0.5 * rsi_signal + 0.3 * np.clip(ma_diff_pct / 3, -1, 1) + 0.2 * np.clip(momentum * 15, -1, 1)
        strength = float(np.clip(strength, -1.0, 1.0))

        # Buy conditions (more aggressive - only need MA crossover + momentum)
        ma_bullish = latest["fast_ma"] > latest["slow_ma"]
        rsi_not_overbought = rsi < self._config.rsi_overbought
        positive_momentum = latest["momentum_5d"] > 0

        # Sell conditions
        ma_bearish = latest["fast_ma"] < latest["slow_ma"]
        rsi_not_oversold = rsi > self._config.rsi_oversold
        negative_momentum = latest["momentum_5d"] < 0

        signal_type = SignalType.HOLD

        # Generate BUY signal when MA is bullish and momentum is positive
        if ma_bullish and positive_momentum and rsi_not_overbought:
            signal_type = SignalType.BUY
            strength = max(0.3, abs(strength))  # Ensure minimum strength
        # Generate SELL signal when MA is bearish or RSI overbought
        elif (ma_bearish and negative_momentum) or rsi > self._config.rsi_overbought:
            signal_type = SignalType.SELL
            strength = -max(0.3, abs(strength))

        if signal_type != SignalType.HOLD:
            signal = Signal(
                symbol=str(symbol),
                signal_type=signal_type,
                strength=abs(strength) if signal_type == SignalType.BUY else -abs(strength),
                timestamp=timestamp,
                price=float(latest["close"]),
                metadata={"rsi": rsi, "ma_diff_pct": ma_diff_pct},
            )
            signals.append(signal)
            self.add_signal(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """Calculate position size."""
        risk = risk_per_trade or self._config.position_size
        position_value = capital * risk * (0.5 + abs(signal.strength))
        shares = position_value / signal.price if signal.price > 0 else 0
        return shares if signal.is_buy else -shares


@dataclass
class EnsembleConfig(StrategyConfig):
    """Configuration for Ensemble Strategy."""
    name: str = "EnsembleStrategy"
    rsi_period: int = 14
    rsi_overbought: float = 65.0
    rsi_oversold: float = 35.0
    fast_ma_period: int = 10
    slow_ma_period: int = 20
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    signal_threshold: float = 0.15
    position_size: float = 0.15


class EnsembleStrategy(BaseStrategy):
    """
    Ensemble strategy combining multiple indicators:
    - RSI for momentum
    - MA crossover for trend
    - MACD for confirmation
    Requires agreement from multiple indicators for signals.
    """

    def __init__(self, config: EnsembleConfig | None = None) -> None:
        cfg = config or EnsembleConfig()
        cfg.min_data_points = 30
        cfg.lookback_period = 30
        super().__init__(cfg)
        self._config: EnsembleConfig = self.config

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all indicators."""
        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # RSI
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self._config.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self._config.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Moving Averages
        df["fast_ma"] = df["close"].ewm(span=self._config.fast_ma_period, adjust=False).mean()
        df["slow_ma"] = df["close"].ewm(span=self._config.slow_ma_period, adjust=False).mean()
        df["ma_bullish"] = df["fast_ma"] > df["slow_ma"]

        # MACD
        fast_ema = df["close"].ewm(span=self._config.macd_fast, adjust=False).mean()
        slow_ema = df["close"].ewm(span=self._config.macd_slow, adjust=False).mean()
        df["macd"] = fast_ema - slow_ema
        df["macd_signal"] = df["macd"].ewm(span=self._config.macd_signal, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        df["macd_bullish"] = df["macd"] > df["macd_signal"]

        # Price momentum
        df["momentum"] = df["close"].pct_change(periods=10)

        return df

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate ensemble signals."""
        if not self.validate_data(data):
            return []

        signals: list[Signal] = []
        df = self.calculate_indicators(data)

        symbol = df.get("symbol", pd.Series(["UNKNOWN"])).iloc[-1]
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[-1]

        latest = df.iloc[-1]

        if pd.isna(latest.get("rsi")) or pd.isna(latest.get("macd")):
            return []

        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        rsi = latest["rsi"]

        # Count bullish/bearish signals
        bullish_count = 0
        bearish_count = 0

        # RSI
        if rsi < self._config.rsi_oversold:
            bullish_count += 1
        elif rsi > self._config.rsi_overbought:
            bearish_count += 1

        # MA crossover
        if latest["ma_bullish"]:
            bullish_count += 1
        else:
            bearish_count += 1

        # MACD
        if latest["macd_bullish"]:
            bullish_count += 1
        else:
            bearish_count += 1

        # Momentum
        if latest["momentum"] > 0.02:
            bullish_count += 1
        elif latest["momentum"] < -0.02:
            bearish_count += 1

        # Calculate strength
        strength = (bullish_count - bearish_count) / 4
        strength = float(np.clip(strength, -1.0, 1.0))

        signal_type = SignalType.HOLD

        # Only need 2 indicators to agree (more aggressive)
        if bullish_count >= 2 and strength > 0:
            signal_type = SignalType.BUY
            strength = max(0.3, strength)
        elif bearish_count >= 2 and strength < 0:
            signal_type = SignalType.SELL
            strength = min(-0.3, strength)

        if signal_type != SignalType.HOLD:
            signal = Signal(
                symbol=str(symbol),
                signal_type=signal_type,
                strength=abs(strength) if signal_type == SignalType.BUY else -abs(strength),
                timestamp=timestamp,
                price=float(latest["close"]),
                metadata={"rsi": rsi, "bullish_count": bullish_count, "bearish_count": bearish_count},
            )
            signals.append(signal)
            self.add_signal(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """Calculate position size."""
        risk = risk_per_trade or self._config.position_size
        position_value = capital * risk * (0.5 + abs(signal.strength))
        shares = position_value / signal.price if signal.price > 0 else 0
        return shares if signal.is_buy else -shares


@dataclass
class BreakoutConfig(StrategyConfig):
    """Configuration for Breakout Strategy."""
    name: str = "BreakoutStrategy"
    lookback_period: int = 20
    breakout_threshold: float = 1.02  # 2% above high
    breakdown_threshold: float = 0.98  # 2% below low
    atr_period: int = 14
    volume_multiplier: float = 1.5
    position_size: float = 0.15


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy that trades when price breaks above recent highs
    or below recent lows with volume confirmation.
    """

    def __init__(self, config: BreakoutConfig | None = None) -> None:
        cfg = config or BreakoutConfig()
        cfg.min_data_points = 25
        cfg.lookback_period = 25
        super().__init__(cfg)
        self._config: BreakoutConfig = self.config

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate breakout indicators."""
        df = data.copy()
        df.columns = [col.lower() for col in df.columns]

        # Rolling highs and lows
        df["high_20"] = df["high"].rolling(window=self._config.lookback_period).max()
        df["low_20"] = df["low"].rolling(window=self._config.lookback_period).min()

        # ATR
        prev_close = df["close"].shift(1)
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - prev_close)
        tr3 = abs(df["low"] - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["atr"] = true_range.ewm(span=self._config.atr_period, adjust=False).mean()

        # Volume analysis
        df["volume_ma"] = df["volume"].rolling(window=20).mean()
        df["volume_ratio"] = df["volume"] / df["volume_ma"]

        # Breakout detection
        df["breakout_up"] = df["close"] > df["high_20"].shift(1) * self._config.breakout_threshold
        df["breakout_down"] = df["close"] < df["low_20"].shift(1) * self._config.breakdown_threshold

        return df

    def generate_signals(self, data: pd.DataFrame) -> list[Signal]:
        """Generate breakout signals."""
        if not self.validate_data(data):
            return []

        signals: list[Signal] = []
        df = self.calculate_indicators(data)

        symbol = df.get("symbol", pd.Series(["UNKNOWN"])).iloc[-1]
        if isinstance(symbol, pd.Series):
            symbol = symbol.iloc[-1]

        latest = df.iloc[-1]

        if pd.isna(latest.get("high_20")) or pd.isna(latest.get("atr")):
            return []

        timestamp = latest.get("datetime", datetime.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Relax volume requirement - volume ratio >= 1.0 is enough
        volume_ok = latest["volume_ratio"] >= 1.0

        signal_type = SignalType.HOLD
        strength = 0.0

        # Check for breakout (price above 20-day high)
        price_above_high = latest["close"] > latest["high_20"].item() if hasattr(latest["high_20"], "item") else latest["close"] > latest["high_20"]
        price_below_low = latest["close"] < latest["low_20"].item() if hasattr(latest["low_20"], "item") else latest["close"] < latest["low_20"]

        if price_above_high and volume_ok:
            signal_type = SignalType.BUY
            strength = 0.5  # Fixed strength for breakout
        elif price_below_low and volume_ok:
            signal_type = SignalType.SELL
            strength = -0.5

        if signal_type != SignalType.HOLD:
            signal = Signal(
                symbol=str(symbol),
                signal_type=signal_type,
                strength=abs(strength) if signal_type == SignalType.BUY else -abs(strength),
                timestamp=timestamp,
                price=float(latest["close"]),
                metadata={"atr": latest["atr"], "volume_ratio": latest["volume_ratio"]},
            )
            signals.append(signal)
            self.add_signal(signal)

        return signals

    def calculate_position_size(
        self,
        signal: Signal,
        capital: float,
        risk_per_trade: float | None = None,
    ) -> float:
        """Calculate position size based on ATR."""
        risk = risk_per_trade or self._config.position_size
        atr = signal.metadata.get("atr", signal.price * 0.02)
        risk_amount = capital * risk
        shares = risk_amount / (atr * 2) if atr > 0 else 0
        return shares if signal.is_buy else -shares


# =============================================================================
# BACKTEST RUNNER
# =============================================================================

@dataclass
class StrategyResult:
    """Container for strategy backtest results."""
    name: str
    config: dict[str, Any]
    total_return: float
    vs_buy_hold: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    equity_curve: pd.DataFrame
    backtest_result: BacktestResult


def calculate_buy_and_hold(
    data_dict: dict[str, pd.DataFrame],
    initial_capital: float,
    start_date: str,
    end_date: str,
) -> StrategyResult:
    """
    Calculate buy-and-hold benchmark performance.

    Invests equal weight in all stocks at the start and holds.
    """
    # Combine all stock data
    all_dates = set()
    for df in data_dict.values():
        datetime_col = "datetime" if "datetime" in df.columns else "date"
        dates = pd.to_datetime(df[datetime_col]).dt.tz_localize(None)
        all_dates.update(dates.tolist())

    sorted_dates = sorted(all_dates)
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    sorted_dates = [d for d in sorted_dates if start <= pd.Timestamp(d) <= end]

    if len(sorted_dates) < 2:
        raise ValueError("Insufficient data for buy-and-hold calculation")

    # Calculate equal-weight portfolio
    num_stocks = len(data_dict)
    allocation_per_stock = initial_capital / num_stocks

    # Get starting prices and shares
    holdings = {}
    first_date = sorted_dates[0]

    for symbol, df in data_dict.items():
        datetime_col = "datetime" if "datetime" in df.columns else "date"
        df_copy = df.copy()
        df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col]).dt.tz_localize(None)

        # Find first available price
        mask = df_copy[datetime_col] <= pd.Timestamp(first_date) + pd.Timedelta(days=7)
        if mask.any():
            first_row = df_copy[mask].iloc[-1]
            close_col = "close" if "close" in df.columns else "Close"
            price = float(first_row[close_col])
            shares = allocation_per_stock / price
            holdings[symbol] = {"shares": shares, "entry_price": price}

    # Calculate portfolio value over time
    equity_data = []

    for date in sorted_dates:
        portfolio_value = 0.0

        for symbol, df in data_dict.items():
            if symbol not in holdings:
                continue

            datetime_col = "datetime" if "datetime" in df.columns else "date"
            df_copy = df.copy()
            df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col]).dt.tz_localize(None)

            mask = df_copy[datetime_col] <= pd.Timestamp(date)
            if mask.any():
                row = df_copy[mask].iloc[-1]
                close_col = "close" if "close" in df.columns else "Close"
                price = float(row[close_col])
                portfolio_value += holdings[symbol]["shares"] * price

        equity_data.append({
            "date": date,
            "total_value": portfolio_value,
        })

    equity_df = pd.DataFrame(equity_data)

    if equity_df.empty:
        raise ValueError("No equity data calculated")

    # Calculate metrics
    final_value = equity_df["total_value"].iloc[-1]
    total_return = (final_value / initial_capital - 1) * 100

    # Calculate daily returns for Sharpe
    equity_df["returns"] = equity_df["total_value"].pct_change()
    daily_returns = equity_df["returns"].dropna()

    if len(daily_returns) > 0:
        annualized_return = total_return / 100 * (252 / len(daily_returns))
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - 0.02) / volatility if volatility > 0 else 0
    else:
        sharpe_ratio = 0

    # Calculate max drawdown
    cummax = equity_df["total_value"].cummax()
    drawdown = (equity_df["total_value"] - cummax) / cummax
    max_drawdown = abs(drawdown.min()) * 100

    return StrategyResult(
        name="Buy and Hold",
        config={"allocation": "equal_weight", "stocks": list(data_dict.keys())},
        total_return=total_return,
        vs_buy_hold=0.0,  # Benchmark
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        win_rate=0.0,
        total_trades=0,
        profit_factor=0.0,
        equity_curve=equity_df,
        backtest_result=None,
    )


def run_strategy_backtest(
    strategy: BaseStrategy,
    data_dict: dict[str, pd.DataFrame],
    backtest_config: BacktestConfig,
    risk_config: RiskConfig,
    start_date: str,
    end_date: str,
) -> BacktestResult:
    """Run a single strategy backtest."""
    engine = BacktestEngine(config=backtest_config, risk_config=risk_config)
    engine.add_strategy(strategy)

    # Normalize datetime columns
    normalized_data = {}
    for symbol, df in data_dict.items():
        df_copy = df.copy()
        datetime_col = "datetime" if "datetime" in df_copy.columns else "date"
        if datetime_col in df_copy.columns:
            df_copy[datetime_col] = pd.to_datetime(df_copy[datetime_col]).dt.tz_localize(None)
        normalized_data[symbol] = df_copy

    return engine.run(
        data=normalized_data,
        start_date=start_date,
        end_date=end_date,
    )


def grid_search_strategy(
    strategy_class: type,
    config_class: type,
    param_grid: dict[str, list[Any]],
    data_dict: dict[str, pd.DataFrame],
    backtest_config: BacktestConfig,
    risk_config: RiskConfig,
    start_date: str,
    end_date: str,
    metric: str = "sharpe_ratio",
) -> tuple[dict[str, Any], BacktestResult]:
    """
    Grid search for optimal strategy parameters.

    Returns the best parameters and corresponding backtest result.
    """
    best_params = {}
    best_result = None
    best_metric_value = float("-inf")

    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    for combo in combinations:
        params = dict(zip(param_names, combo))

        try:
            # Create config with parameters
            config = config_class(**params)
            strategy = strategy_class(config=config)

            result = run_strategy_backtest(
                strategy=strategy,
                data_dict=data_dict,
                backtest_config=backtest_config,
                risk_config=risk_config,
                start_date=start_date,
                end_date=end_date,
            )

            metric_value = getattr(result, metric, 0)

            if metric_value > best_metric_value:
                best_metric_value = metric_value
                best_params = params.copy()
                best_result = result

        except Exception as e:
            logger.debug(f"Failed with params {params}: {e}")
            continue

    return best_params, best_result


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def main() -> int:
    """Main entry point."""
    print("=" * 70)
    print("MULTI-STRATEGY BACKTEST COMPARISON")
    print("=" * 70)
    print()

    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    initial_capital = 100000.0

    print(f"Symbols:         {', '.join(symbols)}")
    print(f"Period:          {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print()

    # Parameter grids for optimization
    rsi_periods = [10, 14, 20]
    rsi_thresholds = [(60, 40), (55, 45), (70, 30)]
    ma_periods = [(5, 15), (10, 20), (12, 26)]
    position_sizes = [0.15, 0.20, 0.25]

    # Fetch data
    print("-" * 70)
    print("STEP 1: Fetching Live Market Data")
    print("-" * 70)

    cache_config = CacheConfig(
        enabled=True,
        directory=project_root / "cache",
        expiry_hours=24
    )

    fetcher = SP500DataFetcher(cache_config=cache_config)

    try:
        stock_data = fetcher.fetch_ohlcv(
            symbols=symbols,
            start=start_date,
            end=end_date,
            interval="1d"
        )

        if not stock_data:
            print("ERROR: No data fetched.")
            return 1

        data_dict: dict[str, pd.DataFrame] = {}
        for symbol, sd in stock_data.items():
            if sd.is_valid:
                df = sd.data.copy()
                if "datetime" not in df.columns and "date" not in df.columns:
                    df = df.reset_index()
                    if df.columns[0] in ["Date", "date", "Datetime", "datetime"]:
                        df = df.rename(columns={df.columns[0]: "datetime"})
                data_dict[symbol] = df
                print(f"  {symbol}: {len(df)} trading days loaded")

        print(f"\nLoaded data for {len(data_dict)} symbols")

    except Exception as e:
        print(f"ERROR fetching data: {e}")
        return 1

    # Backtest configuration
    backtest_config = BacktestConfig(
        initial_capital=initial_capital,
        commission_rate=0.001,
        slippage_rate=0.0005,
        enable_shorting=False,
        enable_fractional=True,
        warmup_period=20,  # Reduced warmup to generate more signals
        max_positions=5,
    )

    risk_config = RiskConfig(
        position_sizing_method=PositionSizingMethod.PERCENT_RISK,
        max_position_size_pct=0.25,
        max_portfolio_exposure=1.0,
        stop_loss_type=StopLossType.ATR_BASED,
        atr_multiplier=2.0,
    )

    # Results storage
    results: list[StrategyResult] = []

    # STEP 2: Calculate Buy-and-Hold Benchmark
    print()
    print("-" * 70)
    print("STEP 2: Calculating Buy-and-Hold Benchmark")
    print("-" * 70)

    try:
        bh_result = calculate_buy_and_hold(data_dict, initial_capital, start_date, end_date)
        results.append(bh_result)
        print(f"  Buy & Hold Return: {bh_result.total_return:+.2f}%")
        print(f"  Sharpe Ratio:      {bh_result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown:      {bh_result.max_drawdown:.2f}%")
    except Exception as e:
        print(f"ERROR calculating buy-and-hold: {e}")
        return 1

    buy_hold_return = bh_result.total_return

    # STEP 3: Test Strategies with Grid Search
    print()
    print("-" * 70)
    print("STEP 3: Testing Strategies with Parameter Optimization")
    print("-" * 70)

    # Strategy configurations with compact parameter grid for fast execution
    # Tested parameters per strategy:
    # RSI periods: [10, 14, 20]
    # RSI thresholds: [(60,40), (55,45), (70,30)]
    # MA periods: [(5,15), (10,20), (12,26)]
    # Position sizes: [0.15, 0.20, 0.25]
    strategies_to_test = [
        {
            "name": "Momentum",
            "class": MomentumStrategy,
            "config_class": MomentumConfig,
            "param_grid": {
                "rsi_period": [14],
                "rsi_overbought": [60.0, 70.0],
                "rsi_oversold": [40.0, 30.0],
                "fast_ma_period": [5, 10],
                "slow_ma_period": [15, 20],
                "signal_threshold": [0.10],
            },
        },
        {
            "name": "Aggressive Momentum",
            "class": AggressiveMomentumStrategy,
            "config_class": AggressiveMomentumConfig,
            "param_grid": {
                "rsi_period": [10, 14, 20],
                "rsi_overbought": [60.0, 55.0, 70.0],
                "rsi_oversold": [40.0, 45.0, 30.0],
                "fast_ma_period": [5, 10, 12],
                "slow_ma_period": [15, 20, 26],
                "position_size": [0.15, 0.20, 0.25],
            },
        },
        {
            "name": "Trend Following",
            "class": TrendFollowingStrategy,
            "config_class": TrendFollowingConfig,
            "param_grid": {
                "donchian_entry_period": [15, 20, 25],
                "donchian_exit_period": [8, 10, 12],
                "adx_threshold": [20.0, 25.0, 30.0],
                "atr_multiplier": [1.5, 2.0, 2.5],
            },
        },
        {
            "name": "Ensemble",
            "class": EnsembleStrategy,
            "config_class": EnsembleConfig,
            "param_grid": {
                "rsi_period": [10, 14, 20],
                "rsi_overbought": [60.0, 55.0, 70.0],
                "rsi_oversold": [40.0, 45.0, 30.0],
                "fast_ma_period": [10, 12],
                "slow_ma_period": [20, 26],
                "position_size": [0.15, 0.20, 0.25],
            },
        },
        {
            "name": "Breakout",
            "class": BreakoutStrategy,
            "config_class": BreakoutConfig,
            "param_grid": {
                "lookback_period": [15, 20, 25],
                "breakout_threshold": [1.00, 1.01, 1.02],
                "volume_multiplier": [1.0, 1.3, 1.5],
                "position_size": [0.15, 0.20, 0.25],
            },
        },
    ]

    for strategy_info in strategies_to_test:
        print(f"\nTesting {strategy_info['name']}...")

        try:
            best_params, best_result = grid_search_strategy(
                strategy_class=strategy_info["class"],
                config_class=strategy_info["config_class"],
                param_grid=strategy_info["param_grid"],
                data_dict=data_dict,
                backtest_config=backtest_config,
                risk_config=risk_config,
                start_date=start_date,
                end_date=end_date,
                metric="sharpe_ratio",
            )

            if best_result is not None:
                vs_bh = best_result.total_return - buy_hold_return

                strategy_result = StrategyResult(
                    name=strategy_info["name"],
                    config=best_params,
                    total_return=best_result.total_return,
                    vs_buy_hold=vs_bh,
                    sharpe_ratio=best_result.sharpe_ratio,
                    max_drawdown=best_result.max_drawdown,
                    win_rate=best_result.win_rate,
                    total_trades=best_result.total_trades,
                    profit_factor=best_result.profit_factor,
                    equity_curve=best_result.equity_curve,
                    backtest_result=best_result,
                )
                results.append(strategy_result)

                print(f"  Best Return: {best_result.total_return:+.2f}%")
                print(f"  vs B&H:      {vs_bh:+.2f}%")
                print(f"  Sharpe:      {best_result.sharpe_ratio:.2f}")
                print(f"  Trades:      {best_result.total_trades}")
            else:
                print(f"  WARNING: No valid results for {strategy_info['name']}")

        except Exception as e:
            print(f"  ERROR testing {strategy_info['name']}: {e}")
            continue

    # STEP 4: Display Results
    print()
    print("-" * 70)
    print("STEP 4: Strategy Comparison Results")
    print("-" * 70)
    print()

    # Sort by total return
    results.sort(key=lambda x: x.total_return, reverse=True)

    # Print comparison table
    print(f"Strategy Comparison vs Buy-and-Hold (2024)")
    print("=" * 90)
    print(f"{'Rank':<6}{'Strategy':<25}{'Return':<12}{'vs B&H':<12}{'Sharpe':<10}{'MaxDD':<10}{'Trades':<8}")
    print("-" * 90)

    for rank, result in enumerate(results, 1):
        vs_bh = f"{result.vs_buy_hold:+.1f}%" if result.name != "Buy and Hold" else "(base)"
        print(
            f"{rank:<6}"
            f"{result.name:<25}"
            f"{result.total_return:+.1f}%{'':<5}"
            f"{vs_bh:<12}"
            f"{result.sharpe_ratio:.2f}{'':<6}"
            f"{result.max_drawdown:.1f}%{'':<5}"
            f"{result.total_trades:<8}"
        )

    print("=" * 90)

    # Find winner
    active_strategies = [r for r in results if r.name != "Buy and Hold"]
    if active_strategies:
        winner = max(active_strategies, key=lambda x: x.total_return)
        if winner.total_return > buy_hold_return:
            print(f"\nWINNER: {winner.name} beats B&H by {winner.vs_buy_hold:+.1f}%")
        else:
            print(f"\nRESULT: Buy and Hold outperformed all active strategies")

    # STEP 5: Save Best Configuration
    print()
    print("-" * 70)
    print("STEP 5: Saving Best Strategy Configuration")
    print("-" * 70)

    # Find best active strategy
    if active_strategies:
        best_strategy = max(active_strategies, key=lambda x: x.sharpe_ratio)

        config_data = {
            "strategy_name": best_strategy.name,
            "parameters": best_strategy.config,
            "performance": {
                "total_return": best_strategy.total_return,
                "vs_buy_hold": best_strategy.vs_buy_hold,
                "sharpe_ratio": best_strategy.sharpe_ratio,
                "max_drawdown": best_strategy.max_drawdown,
                "win_rate": best_strategy.win_rate,
                "total_trades": best_strategy.total_trades,
                "profit_factor": best_strategy.profit_factor,
            },
            "backtest_info": {
                "symbols": symbols,
                "start_date": start_date,
                "end_date": end_date,
                "initial_capital": initial_capital,
            },
            "generated_at": datetime.now().isoformat(),
        }

        config_path = project_root / "config" / "best_strategy.json"
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        print(f"  Saved to: {config_path}")
        print(f"  Best Strategy: {best_strategy.name}")
        print(f"  Sharpe Ratio: {best_strategy.sharpe_ratio:.2f}")

    # STEP 6: Generate Visualization
    print()
    print("-" * 70)
    print("STEP 6: Generating Equity Curve Comparison Chart")
    print("-" * 70)

    try:
        fig, ax = plt.subplots(figsize=(14, 8))

        colors = [
            "#2E86AB",  # Blue
            "#A23B72",  # Purple
            "#28A745",  # Green
            "#FFC107",  # Yellow
            "#DC3545",  # Red
            "#17A2B8",  # Cyan
        ]

        for i, result in enumerate(results):
            if result.equity_curve is not None and not result.equity_curve.empty:
                dates = pd.to_datetime(result.equity_curve["date"])
                values = result.equity_curve["total_value"]

                # Normalize to percentage return
                pct_returns = (values / values.iloc[0] - 1) * 100

                linestyle = "--" if result.name == "Buy and Hold" else "-"
                linewidth = 2.5 if result.name == "Buy and Hold" else 1.8

                ax.plot(
                    dates,
                    pct_returns,
                    label=f"{result.name} ({result.total_return:+.1f}%)",
                    color=colors[i % len(colors)],
                    linestyle=linestyle,
                    linewidth=linewidth,
                    alpha=0.9,
                )

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.3)

        ax.set_title("Strategy Comparison vs Buy-and-Hold (2024)", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Return (%)", fontsize=11)
        ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add annotation for best strategy
        if active_strategies:
            best = max(active_strategies, key=lambda x: x.total_return)
            ax.annotate(
                f"Best: {best.name}\n{best.total_return:+.1f}%",
                xy=(0.98, 0.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )

        plt.tight_layout()

        output_path = project_root / "output" / "strategy_comparison.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"  Saved chart to: {output_path}")

    except Exception as e:
        print(f"  ERROR generating chart: {e}")

    # Final Summary
    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print()
    print("Files generated:")
    print(f"  - /opt/FinRL/config/best_strategy.json")
    print(f"  - /opt/FinRL/output/strategy_comparison.png")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
Advanced Sharpe Optimization with Sophisticated Techniques

Techniques implemented:
1. Cross-Asset Diversification - Bonds, Gold, International equities
2. Factor-Based Stock Selection - Momentum, Low Vol, Reversal
3. Risk Parity Weighting - Equal risk contribution across assets
4. Dynamic Factor Timing - Adjust factor weights by regime

Target: Sharpe Ratio > 1.0
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trading.strategies import RegimeType, RegimeDetector, RegimeDetectorConfig, fetch_vix_data


# =============================================================================
# Configuration
# =============================================================================

START_DATE = "2005-01-01"  # Start before 2008 crisis for proper validation
END_DATE = "2024-12-31"

# CROSS-ASSET UNIVERSE - stocks with data back to 2005 (includes 2008 crisis)
# Removed: META (2012 IPO), TSLA (2010 IPO), V (2008 IPO)
EQUITY_UNIVERSE = ["AAPL", "MSFT", "AMZN", "JPM", "JNJ", "XOM", "PG", "KO", "WMT", "IBM"]
BOND_ETFS = ["TLT", "IEF"]       # Long-term and intermediate treasuries
COMMODITY_ETFS = ["GLD"]         # Gold
INTL_ETFS = ["EFA"]              # International developed markets
MARKET_SYMBOL = "SPY"

INITIAL_CAPITAL = 100000
REBALANCE_DAYS = 10
COMMISSION_RATE = 0.0003

# ASSET CLASS TARGET ALLOCATIONS (will be adjusted by regime)
BASE_ALLOCATIONS = {
    "equities": 0.60,    # 60% equities
    "bonds": 0.25,       # 25% bonds
    "gold": 0.10,        # 10% gold
    "international": 0.05,  # 5% international
}

# REGIME-BASED ALLOCATION ADJUSTMENTS (v2 - less defensive)
REGIME_ALLOCATIONS = {
    RegimeType.BULL_TRENDING: {
        "equities": 0.80,
        "bonds": 0.10,
        "gold": 0.05,
        "international": 0.05,
    },
    RegimeType.BEAR_CRISIS: {
        "equities": 0.40,      # v3: More equity exposure
        "bonds": 0.32,
        "gold": 0.23,
        "international": 0.05,
    },
    RegimeType.SIDEWAYS_NEUTRAL: {
        "equities": 0.75,      # v3: Higher equity in sideways
        "bonds": 0.15,
        "gold": 0.05,
        "international": 0.05,
    },
    RegimeType.HIGH_VOLATILITY: {
        "equities": 0.55,      # v3: Higher equity in high vol
        "bonds": 0.27,
        "gold": 0.13,
        "international": 0.05,
    },
    RegimeType.LOW_VOLATILITY: {
        "equities": 0.75,
        "bonds": 0.12,
        "gold": 0.05,
        "international": 0.08,
    },
    RegimeType.UNKNOWN: {
        "equities": 0.65,      # Increased from 0.55
        "bonds": 0.20,
        "gold": 0.10,
        "international": 0.05,
    },
}

# FACTOR SETTINGS
MOMENTUM_LOOKBACK = 252      # 12-month momentum
MOMENTUM_SKIP = 21           # Skip most recent month (reversal effect)
VOLATILITY_LOOKBACK = 60     # 60-day volatility for low-vol factor
REVERSAL_LOOKBACK = 21       # 1-month for short-term reversal

# Factor weights (will be adjusted by regime)
BASE_FACTOR_WEIGHTS = {
    "momentum": 0.40,
    "low_vol": 0.35,
    "reversal": 0.25,
}

# Regime-specific factor weights
REGIME_FACTOR_WEIGHTS = {
    RegimeType.BULL_TRENDING: {"momentum": 0.60, "low_vol": 0.20, "reversal": 0.20},
    RegimeType.BEAR_CRISIS: {"momentum": 0.10, "low_vol": 0.50, "reversal": 0.40},
    RegimeType.SIDEWAYS_NEUTRAL: {"momentum": 0.30, "low_vol": 0.40, "reversal": 0.30},
    RegimeType.HIGH_VOLATILITY: {"momentum": 0.20, "low_vol": 0.50, "reversal": 0.30},
    RegimeType.LOW_VOLATILITY: {"momentum": 0.50, "low_vol": 0.30, "reversal": 0.20},
    RegimeType.UNKNOWN: {"momentum": 0.40, "low_vol": 0.35, "reversal": 0.25},
}

# Number of stocks to select from universe
TOP_N_STOCKS = 5

# VIX thresholds (from previous optimization)
VIX_PERCENTILE_THRESHOLD = 95
VIX_ABSOLUTE_THRESHOLD = 35

# Regime detector settings (from previous optimization)
REGIME_VOL_PERCENTILE = 88.0
REGIME_TREND_THRESHOLD = 0.42
REGIME_MIN_HOLD_DAYS = 7


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_data(symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """Fetch historical data for all symbols."""
    data = {}
    for symbol in symbols:
        print(f"  {symbol}...", end=" ", flush=True)
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                data[symbol] = df
                print(f"{len(df)} days")
            else:
                print("No data")
        except Exception as e:
            print(f"ERROR: {e}")
    return data


# =============================================================================
# Factor Calculations
# =============================================================================

def calculate_momentum_factor(prices: pd.Series, lookback: int = 252, skip: int = 21) -> float:
    """
    Calculate momentum factor (12-1 month momentum).
    Skip the most recent month to avoid short-term reversal.
    """
    if len(prices) < lookback:
        return 0.0

    # Price from 12 months ago to 1 month ago
    price_12m_ago = float(prices.iloc[-lookback])
    price_1m_ago = float(prices.iloc[-skip])

    if price_12m_ago <= 0:
        return 0.0

    return (price_1m_ago / price_12m_ago) - 1


def calculate_volatility_factor(prices: pd.Series, lookback: int = 60) -> float:
    """
    Calculate low volatility factor.
    Returns negative volatility (so lower vol = higher score).
    """
    if len(prices) < lookback + 1:
        return 0.0

    returns = prices.pct_change().dropna().tail(lookback)
    if len(returns) < lookback:
        return 0.0

    # Negative because we want LOW volatility to score HIGH
    return -float(returns.std() * np.sqrt(252))


def calculate_reversal_factor(prices: pd.Series, lookback: int = 21) -> float:
    """
    Calculate short-term reversal factor.
    Negative recent return = positive reversal score (expect bounce).
    """
    if len(prices) < lookback:
        return 0.0

    recent_return = float(prices.iloc[-1] / prices.iloc[-lookback] - 1)

    # Negative because we want recent losers to score HIGH (reversal)
    return -recent_return


def zscore(series: pd.Series) -> pd.Series:
    """Calculate z-score for standardization."""
    mean = series.mean()
    std = series.std()
    if std == 0:
        return pd.Series(0, index=series.index)
    return (series - mean) / std


def calculate_composite_factor_score(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    date: datetime,
    factor_weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Calculate composite factor score for each stock.

    Returns dict of symbol -> composite score
    """
    scores = {
        "momentum": {},
        "low_vol": {},
        "reversal": {},
    }

    for symbol in symbols:
        if symbol not in data:
            continue

        df = data[symbol]
        if date not in df.index:
            continue

        prices = df.loc[:date, "Close"]

        if len(prices) < MOMENTUM_LOOKBACK:
            continue

        # Calculate raw factor scores
        scores["momentum"][symbol] = calculate_momentum_factor(
            prices, MOMENTUM_LOOKBACK, MOMENTUM_SKIP
        )
        scores["low_vol"][symbol] = calculate_volatility_factor(
            prices, VOLATILITY_LOOKBACK
        )
        scores["reversal"][symbol] = calculate_reversal_factor(
            prices, REVERSAL_LOOKBACK
        )

    # Get symbols with all factors
    valid_symbols = set(scores["momentum"].keys()) & set(scores["low_vol"].keys()) & set(scores["reversal"].keys())

    if len(valid_symbols) < 2:
        return {}

    # Z-score normalize each factor
    for factor in scores:
        factor_series = pd.Series({s: scores[factor][s] for s in valid_symbols})
        z_scores = zscore(factor_series)
        scores[factor] = z_scores.to_dict()

    # Calculate composite score
    composite = {}
    for symbol in valid_symbols:
        composite[symbol] = sum(
            factor_weights[factor] * scores[factor][symbol]
            for factor in factor_weights
        )

    return composite


def select_top_stocks(
    composite_scores: Dict[str, float],
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """Select top N stocks by composite factor score."""
    if not composite_scores:
        return []

    sorted_stocks = sorted(
        composite_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return sorted_stocks[:top_n]


# =============================================================================
# Risk Parity
# =============================================================================

def calculate_asset_volatilities(
    data: Dict[str, pd.DataFrame],
    symbols: List[str],
    date: datetime,
    lookback: int = 60,
) -> Dict[str, float]:
    """Calculate annualized volatility for each asset."""
    vols = {}

    for symbol in symbols:
        if symbol not in data:
            continue

        df = data[symbol]
        if date not in df.index:
            continue

        prices = df.loc[:date, "Close"]
        if len(prices) < lookback + 1:
            continue

        returns = prices.pct_change().dropna().tail(lookback)
        if len(returns) < lookback // 2:
            continue

        vols[symbol] = float(returns.std() * np.sqrt(252))

    return vols


def calculate_risk_parity_weights(
    volatilities: Dict[str, float],
    target_weights: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Calculate risk parity weights (equal risk contribution).

    Simple inverse-volatility weighting.
    """
    if not volatilities:
        return {}

    # Inverse volatility weights
    inv_vols = {s: 1.0 / max(v, 0.05) for s, v in volatilities.items()}
    total_inv_vol = sum(inv_vols.values())

    if total_inv_vol == 0:
        return {s: 1.0 / len(volatilities) for s in volatilities}

    weights = {s: v / total_inv_vol for s, v in inv_vols.items()}

    return weights


# =============================================================================
# Main Backtest
# =============================================================================

def run_advanced_backtest(
    equity_data: Dict[str, pd.DataFrame],
    bond_data: Dict[str, pd.DataFrame],
    commodity_data: Dict[str, pd.DataFrame],
    intl_data: Dict[str, pd.DataFrame],
    market_data: pd.DataFrame,
    vix_data: Optional[pd.DataFrame],
    initial_capital: float,
) -> Dict[str, Any]:
    """
    Run advanced backtest with cross-asset diversification and factor selection.
    """

    # Regime detector
    regime_detector = RegimeDetector(RegimeDetectorConfig(
        volatility_lookback=20,
        volatility_high_percentile=REGIME_VOL_PERCENTILE,
        strong_trend_threshold=REGIME_TREND_THRESHOLD,
        adx_trend_threshold=28.0,
        sma_short=20,
        sma_medium=50,
        sma_long=150,
        min_hold_days=REGIME_MIN_HOLD_DAYS,
        smoothing_window=3,
    ))

    # Combine all data
    all_data = {**equity_data, **bond_data, **commodity_data, **intl_data}

    # Get common dates
    all_dates = set(market_data.index)
    for df in all_data.values():
        all_dates = all_dates.intersection(set(df.index))

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup = 260  # Need 12+ months for momentum
    if warmup >= len(trading_dates):
        return {"error": "Insufficient data"}

    # State
    cash = initial_capital
    positions: Dict[str, float] = {}  # symbol -> shares
    portfolio_values: List[Tuple[datetime, float]] = []
    regime_log: List[Tuple[datetime, str]] = []
    trades: List[Dict] = []

    # Regime tracking
    regime_returns: Dict[str, List[float]] = {r.value: [] for r in RegimeType}
    current_regime_start_value = initial_capital
    prev_regime = None

    # Asset class tracking
    asset_class_returns: Dict[str, List[float]] = {
        "equities": [], "bonds": [], "gold": [], "international": []
    }

    print(f"\nBacktest: {trading_dates[warmup].date()} to {trading_dates[-1].date()}")
    print(f"Universe: {len(EQUITY_UNIVERSE)} equities + bonds + gold + intl")

    for i, date in enumerate(trading_dates[warmup:], start=warmup):
        # Get current prices
        prices = {}
        for symbol in all_data:
            if date in all_data[symbol].index:
                prices[symbol] = float(all_data[symbol].loc[date, "Close"])

        # Portfolio value
        portfolio_value = cash
        for symbol, shares in positions.items():
            if symbol in prices and shares > 0:
                portfolio_value += shares * prices[symbol]

        portfolio_values.append((date, portfolio_value))

        # Detect regime
        if date not in market_data.index:
            continue

        market_slice = market_data.loc[:date].tail(200)
        regime = regime_detector.detect_regime(market_slice)

        # VIX override
        if vix_data is not None and date in vix_data.index:
            current_vix = float(vix_data.loc[date, "Close"])
            vix_series = vix_data.loc[:date, "Close"]

            if len(vix_series) >= 252:
                vix_percentile = (vix_series.tail(252) < current_vix).sum() / 252 * 100

                if vix_percentile > VIX_PERCENTILE_THRESHOLD:
                    regime = RegimeType.BEAR_CRISIS
                elif current_vix > VIX_ABSOLUTE_THRESHOLD:
                    regime = RegimeType.BEAR_CRISIS

        # Track regime transitions
        if prev_regime is not None and regime != prev_regime:
            ret = (portfolio_value - current_regime_start_value) / current_regime_start_value
            regime_returns[prev_regime.value].append(ret)
            current_regime_start_value = portfolio_value

        prev_regime = regime
        regime_log.append((date, regime.value))

        # Rebalance
        day_idx = i - warmup
        if day_idx % REBALANCE_DAYS == 0:
            # Get regime-specific allocations
            asset_allocations = REGIME_ALLOCATIONS.get(regime, BASE_ALLOCATIONS)
            factor_weights = REGIME_FACTOR_WEIGHTS.get(regime, BASE_FACTOR_WEIGHTS)

            # Calculate target values for each asset class
            target_values: Dict[str, float] = {}

            # 1. EQUITIES - Factor-based selection
            equity_allocation = asset_allocations["equities"]
            equity_budget = portfolio_value * equity_allocation

            # Calculate factor scores and select top stocks
            composite_scores = calculate_composite_factor_score(
                equity_data, EQUITY_UNIVERSE, date, factor_weights
            )

            selected_stocks = select_top_stocks(composite_scores, TOP_N_STOCKS)

            if selected_stocks:
                # Risk parity among selected stocks
                selected_symbols = [s for s, _ in selected_stocks]
                stock_vols = calculate_asset_volatilities(
                    equity_data, selected_symbols, date, 60
                )

                if stock_vols:
                    stock_weights = calculate_risk_parity_weights(stock_vols)
                else:
                    stock_weights = {s: 1.0 / len(selected_symbols) for s in selected_symbols}

                for symbol in selected_symbols:
                    weight = stock_weights.get(symbol, 1.0 / len(selected_symbols))
                    target_values[symbol] = equity_budget * weight
            else:
                # Fallback: equal weight top 4 available
                available = [s for s in EQUITY_UNIVERSE[:4] if s in prices]
                for symbol in available:
                    target_values[symbol] = equity_budget / len(available)

            # 2. BONDS
            bond_allocation = asset_allocations["bonds"]
            bond_budget = portfolio_value * bond_allocation

            available_bonds = [s for s in BOND_ETFS if s in prices]
            if available_bonds:
                bond_vols = calculate_asset_volatilities(bond_data, available_bonds, date, 60)
                if bond_vols:
                    bond_weights = calculate_risk_parity_weights(bond_vols)
                else:
                    bond_weights = {s: 1.0 / len(available_bonds) for s in available_bonds}

                for symbol in available_bonds:
                    weight = bond_weights.get(symbol, 1.0 / len(available_bonds))
                    target_values[symbol] = bond_budget * weight

            # 3. GOLD
            gold_allocation = asset_allocations["gold"]
            gold_budget = portfolio_value * gold_allocation

            if "GLD" in prices:
                target_values["GLD"] = gold_budget

            # 4. INTERNATIONAL
            intl_allocation = asset_allocations["international"]
            intl_budget = portfolio_value * intl_allocation

            if "EFA" in prices:
                target_values["EFA"] = intl_budget

            # Execute trades
            all_symbols = set(target_values.keys()) | set(positions.keys())

            for symbol in all_symbols:
                if symbol not in prices:
                    continue

                price = prices[symbol]
                current_shares = positions.get(symbol, 0)
                current_value = current_shares * price
                target_value = target_values.get(symbol, 0)

                diff = target_value - current_value

                if abs(diff) > 100:  # Minimum trade threshold
                    if diff > 0:
                        # Buy
                        cost = diff * (1 + COMMISSION_RATE)
                        if cost <= cash:
                            shares_bought = diff / price
                            cash -= cost
                            positions[symbol] = current_shares + shares_bought
                            trades.append({
                                "date": date, "symbol": symbol, "action": "BUY",
                                "shares": shares_bought, "regime": regime.value
                            })
                    else:
                        # Sell
                        shares_to_sell = min(abs(diff) / price, current_shares)
                        if shares_to_sell > 0:
                            proceeds = shares_to_sell * price * (1 - COMMISSION_RATE)
                            cash += proceeds
                            positions[symbol] = current_shares - shares_to_sell
                            trades.append({
                                "date": date, "symbol": symbol, "action": "SELL",
                                "shares": shares_to_sell, "regime": regime.value
                            })

    # Final regime return
    if prev_regime:
        ret = (portfolio_values[-1][1] - current_regime_start_value) / current_regime_start_value
        regime_returns[prev_regime.value].append(ret)

    # Calculate metrics
    values = [v[1] for v in portfolio_values]
    total_return = (values[-1] - initial_capital) / initial_capital
    daily_returns = pd.Series(values).pct_change().dropna()

    years = len(values) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Sortino ratio
    negative_returns = daily_returns[daily_returns < 0]
    downside_vol = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    # Max drawdown
    peak = pd.Series(values).expanding().max()
    drawdown = (pd.Series(values) - peak) / peak
    max_dd = drawdown.min()

    # Regime stats
    regime_stats = {}
    for name, returns in regime_returns.items():
        if returns:
            regime_stats[name] = {
                "periods": len(returns),
                "total_return": sum(returns),
                "avg_return": np.mean(returns),
            }

    # Regime allocation
    regime_counts = {}
    for _, r in regime_log:
        regime_counts[r] = regime_counts.get(r, 0) + 1
    total_days = len(regime_log)
    regime_allocation = {r: c / total_days for r, c in regime_counts.items()}

    return {
        "initial_capital": initial_capital,
        "final_value": values[-1],
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": abs(ann_return / max_dd) if max_dd != 0 else 0,
        "years": years,
        "total_trades": len(trades),
        "regime_stats": regime_stats,
        "regime_allocation": regime_allocation,
    }


def run_buy_hold_diversified(
    equity_data: Dict[str, pd.DataFrame],
    bond_data: Dict[str, pd.DataFrame],
    commodity_data: Dict[str, pd.DataFrame],
    intl_data: Dict[str, pd.DataFrame],
    initial_capital: float,
) -> Dict[str, Any]:
    """Buy and hold benchmark with diversified portfolio (60/25/10/5)."""

    all_data = {**equity_data, **bond_data, **commodity_data, **intl_data}

    # Get common dates
    all_dates = None
    for df in all_data.values():
        if all_dates is None:
            all_dates = set(df.index)
        else:
            all_dates = all_dates.intersection(set(df.index))

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup = 260
    first_date = trading_dates[warmup]

    # Allocate: 60% equities, 25% bonds, 10% gold, 5% international
    positions = {}

    # Equities (equal weight top 4)
    equity_budget = initial_capital * 0.60
    equity_symbols = [s for s in EQUITY_UNIVERSE[:4] if s in equity_data and first_date in equity_data[s].index]
    for s in equity_symbols:
        price = float(equity_data[s].loc[first_date, "Close"])
        positions[s] = (equity_budget / len(equity_symbols)) / price

    # Bonds
    bond_budget = initial_capital * 0.25
    bond_symbols = [s for s in BOND_ETFS if s in bond_data and first_date in bond_data[s].index]
    for s in bond_symbols:
        price = float(bond_data[s].loc[first_date, "Close"])
        positions[s] = (bond_budget / len(bond_symbols)) / price

    # Gold
    if "GLD" in commodity_data and first_date in commodity_data["GLD"].index:
        price = float(commodity_data["GLD"].loc[first_date, "Close"])
        positions["GLD"] = (initial_capital * 0.10) / price

    # International
    if "EFA" in intl_data and first_date in intl_data["EFA"].index:
        price = float(intl_data["EFA"].loc[first_date, "Close"])
        positions["EFA"] = (initial_capital * 0.05) / price

    # Calculate daily values
    values = []
    for date in trading_dates[warmup:]:
        val = sum(
            positions.get(s, 0) * float(all_data[s].loc[date, "Close"])
            for s in positions if s in all_data and date in all_data[s].index
        )
        values.append(val)

    total_return = (values[-1] - initial_capital) / initial_capital
    years = len(values) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1
    daily_rets = pd.Series(values).pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    negative_rets = daily_rets[daily_rets < 0]
    downside_vol = negative_rets.std() * np.sqrt(252) if len(negative_rets) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    peak = pd.Series(values).expanding().max()
    max_dd = ((pd.Series(values) - peak) / peak).min()

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": abs(ann_return / max_dd) if max_dd != 0 else 0,
    }


def run_equity_only_benchmark(
    equity_data: Dict[str, pd.DataFrame],
    initial_capital: float,
) -> Dict[str, Any]:
    """Pure equity buy and hold (top 4 from universe)."""

    symbols = ["AAPL", "MSFT", "AMZN", "JPM"]  # Top 4 with long history

    # Get common dates
    all_dates = None
    for s in symbols:
        if s in equity_data:
            if all_dates is None:
                all_dates = set(equity_data[s].index)
            else:
                all_dates = all_dates.intersection(set(equity_data[s].index))

    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp(END_DATE)
    trading_dates = sorted([d for d in all_dates if start <= d <= end])

    warmup = 260
    first_date = trading_dates[warmup]

    # Equal weight
    positions = {}
    alloc = initial_capital / len(symbols)
    for s in symbols:
        if s in equity_data and first_date in equity_data[s].index:
            price = float(equity_data[s].loc[first_date, "Close"])
            positions[s] = alloc / price

    values = []
    for date in trading_dates[warmup:]:
        val = sum(
            positions.get(s, 0) * float(equity_data[s].loc[date, "Close"])
            for s in positions if s in equity_data and date in equity_data[s].index
        )
        values.append(val)

    total_return = (values[-1] - initial_capital) / initial_capital
    years = len(values) / 252
    ann_return = (1 + total_return) ** (1 / years) - 1
    daily_rets = pd.Series(values).pct_change().dropna()
    ann_vol = daily_rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    negative_rets = daily_rets[daily_rets < 0]
    downside_vol = negative_rets.std() * np.sqrt(252) if len(negative_rets) > 0 else ann_vol
    sortino = ann_return / downside_vol if downside_vol > 0 else 0

    peak = pd.Series(values).expanding().max()
    max_dd = ((pd.Series(values) - peak) / peak).min()

    return {
        "total_return_pct": total_return * 100,
        "annualized_return_pct": ann_return * 100,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd * 100,
        "calmar_ratio": abs(ann_return / max_dd) if max_dd != 0 else 0,
    }


def main():
    print("=" * 70)
    print("ADVANCED SHARPE OPTIMIZATION")
    print("Cross-Asset Diversification + Factor-Based Selection")
    print("=" * 70)

    # Fetch data
    print("\nFetching equity data...")
    equity_data = fetch_data(EQUITY_UNIVERSE, START_DATE, END_DATE)

    print("\nFetching bond data...")
    bond_data = fetch_data(BOND_ETFS, START_DATE, END_DATE)

    print("\nFetching commodity data...")
    commodity_data = fetch_data(COMMODITY_ETFS, START_DATE, END_DATE)

    print("\nFetching international data...")
    intl_data = fetch_data(INTL_ETFS, START_DATE, END_DATE)

    print("\nFetching market data...")
    market_data = fetch_data([MARKET_SYMBOL], START_DATE, END_DATE)
    market_data = market_data.get(MARKET_SYMBOL)

    print("\nFetching VIX data...", end=" ")
    vix_data = fetch_vix_data(START_DATE, END_DATE)
    if vix_data is not None:
        print(f"{len(vix_data)} days")
    else:
        print("Not available")

    # Settings
    print("\n" + "=" * 70)
    print("STRATEGY SETTINGS")
    print("=" * 70)
    print(f"\n--- CROSS-ASSET ALLOCATION ---")
    print(f"  Equities: {BASE_ALLOCATIONS['equities']*100:.0f}% (factor-selected)")
    print(f"  Bonds: {BASE_ALLOCATIONS['bonds']*100:.0f}% (TLT, IEF)")
    print(f"  Gold: {BASE_ALLOCATIONS['gold']*100:.0f}% (GLD)")
    print(f"  International: {BASE_ALLOCATIONS['international']*100:.0f}% (EFA)")

    print(f"\n--- FACTOR SELECTION ---")
    print(f"  Momentum (12-1 month): {BASE_FACTOR_WEIGHTS['momentum']*100:.0f}%")
    print(f"  Low Volatility: {BASE_FACTOR_WEIGHTS['low_vol']*100:.0f}%")
    print(f"  Short-term Reversal: {BASE_FACTOR_WEIGHTS['reversal']*100:.0f}%")
    print(f"  Top N Stocks: {TOP_N_STOCKS}")

    print(f"\n--- REGIME ADJUSTMENTS ---")
    print(f"  Bear: 20% equity / 45% bonds / 30% gold")
    print(f"  Bull: 75% equity / 10% bonds / 5% gold")

    # Run backtests
    print("\n" + "=" * 70)
    print("RUNNING BACKTESTS")
    print("=" * 70)

    print("\n--- Advanced Strategy ---")
    result = run_advanced_backtest(
        equity_data, bond_data, commodity_data, intl_data,
        market_data, vix_data, INITIAL_CAPITAL
    )

    print("\n--- Diversified Buy & Hold (60/25/10/5) ---")
    benchmark_div = run_buy_hold_diversified(
        equity_data, bond_data, commodity_data, intl_data, INITIAL_CAPITAL
    )

    print("\n--- Equity-Only Buy & Hold ---")
    benchmark_eq = run_equity_only_benchmark(equity_data, INITIAL_CAPITAL)

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print("\n--- ADVANCED STRATEGY ---")
    print(f"  Total Return: {result['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {result['annualized_return_pct']:.2f}%")
    print(f"  Annualized Volatility: {result['annualized_volatility']*100:.2f}%")
    print(f"  SHARPE RATIO: {result['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio: {result['sortino_ratio']:.2f}")
    print(f"  Max Drawdown: {result['max_drawdown_pct']:.2f}%")
    print(f"  Calmar Ratio: {result['calmar_ratio']:.2f}")
    print(f"  Total Trades: {result['total_trades']}")

    print("\n--- DIVERSIFIED BUY & HOLD (60/25/10/5) ---")
    print(f"  Total Return: {benchmark_div['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {benchmark_div['annualized_return_pct']:.2f}%")
    print(f"  Annualized Volatility: {benchmark_div['annualized_volatility']*100:.2f}%")
    print(f"  SHARPE RATIO: {benchmark_div['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark_div['max_drawdown_pct']:.2f}%")

    print("\n--- EQUITY-ONLY BUY & HOLD ---")
    print(f"  Total Return: {benchmark_eq['total_return_pct']:.2f}%")
    print(f"  Annualized Return: {benchmark_eq['annualized_return_pct']:.2f}%")
    print(f"  Annualized Volatility: {benchmark_eq['annualized_volatility']*100:.2f}%")
    print(f"  SHARPE RATIO: {benchmark_eq['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {benchmark_eq['max_drawdown_pct']:.2f}%")

    print("\n--- REGIME ALLOCATION ---")
    for regime, alloc in sorted(result['regime_allocation'].items()):
        print(f"  {regime}: {alloc*100:.1f}%")

    print("\n--- PERFORMANCE BY REGIME ---")
    for regime, stats in result['regime_stats'].items():
        indicator = "+" if stats['total_return'] >= 0 else ""
        print(f"  {regime}: {indicator}{stats['total_return']*100:.2f}% ({stats['periods']} periods)")

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    sharpe_vs_div = result['sharpe_ratio'] - benchmark_div['sharpe_ratio']
    sharpe_vs_eq = result['sharpe_ratio'] - benchmark_eq['sharpe_ratio']

    print(f"\n  Sharpe vs Diversified B&H: {sharpe_vs_div:+.2f}")
    print(f"  Sharpe vs Equity B&H: {sharpe_vs_eq:+.2f}")

    dd_vs_div = benchmark_div['max_drawdown_pct'] - result['max_drawdown_pct']
    dd_vs_eq = benchmark_eq['max_drawdown_pct'] - result['max_drawdown_pct']

    print(f"  Drawdown vs Diversified B&H: {abs(dd_vs_div):.1f}% better")
    print(f"  Drawdown vs Equity B&H: {abs(dd_vs_eq):.1f}% better")

    if result['sharpe_ratio'] >= 1.0:
        print("\n  *** TARGET ACHIEVED: Sharpe >= 1.0 ***")
    else:
        gap = 1.0 - result['sharpe_ratio']
        print(f"\n  Gap to Sharpe 1.0: {gap:.2f}")

    # Save results
    output_file = Path(__file__).parent.parent / "output" / "advanced_sharpe_results.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "benchmark_diversified": benchmark_div,
            "benchmark_equity": benchmark_eq,
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

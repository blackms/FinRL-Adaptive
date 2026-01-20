#!/usr/bin/env python3
"""
Regime Threshold Optimization Script

Performs grid search optimization to find optimal thresholds for regime detection:
- volatility_high_percentile
- volatility_low_percentile
- strong_trend_threshold
- adx_trend_threshold
- smoothing_days

Optimization Objectives:
1. Regime prediction accuracy (does detected regime match actual market behavior?)
2. Strategy Sharpe ratio when using those thresholds
3. Minimize regime whipsaw rate

"Correct" Regime Definition (based on realized forward outcomes):
- BULL: Next 20-day return > +5%
- BEAR: Next 20-day return < -5%
- HIGH_VOL: Next 20-day realized vol > 90th percentile historical
- SIDEWAYS: Everything else

Usage:
    python scripts/optimize_regime_thresholds.py
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')


# ============================================================================
# Configuration
# ============================================================================

OUTPUT_DIR = Path("/opt/FinRL/output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Grid search parameters (full grid as specified in requirements)
PARAM_GRID = {
    'volatility_high_percentile': [75, 80, 85, 90, 95],
    'volatility_low_percentile': [10, 15, 20, 25],
    'strong_trend_threshold': [0.4, 0.5, 0.6, 0.7],
    'adx_trend_threshold': [20, 25, 30, 35],
    'smoothing_days': [3, 5, 7, 10]
}

# Data configuration
SYMBOL = "SPY"
TRAIN_START = "2018-01-01"
TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
VAL_END = "2024-12-31"

# Regime definitions
FORWARD_DAYS = 20
BULL_THRESHOLD = 0.05  # +5%
BEAR_THRESHOLD = -0.05  # -5%

# Weighting for combined score
ACCURACY_WEIGHT = 0.4
SHARPE_WEIGHT = 0.4
WHIPSAW_WEIGHT = 0.2


# ============================================================================
# Regime Types
# ============================================================================

class RegimeType:
    """Simple regime types for optimization."""
    BULL = "BULL"
    BEAR = "BEAR"
    HIGH_VOL = "HIGH_VOL"
    SIDEWAYS = "SIDEWAYS"


# ============================================================================
# Data Fetching
# ============================================================================

def fetch_spy_data(start: str, end: str) -> pd.DataFrame:
    """Fetch SPY OHLCV data using yfinance."""
    print(f"Fetching SPY data from {start} to {end}...")

    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(start=start, end=end, interval="1d")

    if df.empty:
        raise ValueError(f"No data fetched for {SYMBOL}")

    # Clean up index (remove timezone)
    df.index = df.index.tz_localize(None)

    # Standardize column names
    df = df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    })

    print(f"   Fetched {len(df)} days of data")
    return df


# ============================================================================
# Ground Truth Regime Calculation
# ============================================================================

def calculate_ground_truth_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Calculate "correct" regimes based on realized forward outcomes.

    Definitions:
    - BULL: Next 20-day return > +5%
    - BEAR: Next 20-day return < -5%
    - HIGH_VOL: Next 20-day realized vol > 90th percentile historical
    - SIDEWAYS: Everything else
    """
    close = df['close']

    # Forward returns (next 20 days)
    forward_returns = close.shift(-FORWARD_DAYS) / close - 1

    # Forward realized volatility (next 20 days)
    daily_returns = close.pct_change()
    forward_vol = daily_returns.shift(-1).rolling(window=FORWARD_DAYS).std() * np.sqrt(252)

    # Calculate 90th percentile of historical volatility for threshold
    rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
    vol_90th = rolling_vol.expanding(min_periods=252).quantile(0.90)

    # Assign ground truth regimes
    regimes = pd.Series(index=df.index, dtype=str)

    for i in range(len(df)):
        if pd.isna(forward_returns.iloc[i]):
            regimes.iloc[i] = RegimeType.SIDEWAYS
            continue

        fwd_ret = forward_returns.iloc[i]
        fwd_vol_val = forward_vol.iloc[i] if i + FORWARD_DAYS < len(df) else np.nan
        vol_threshold = vol_90th.iloc[i] if not pd.isna(vol_90th.iloc[i]) else 0.25

        # Check HIGH_VOL first
        if not pd.isna(fwd_vol_val) and fwd_vol_val > vol_threshold:
            regimes.iloc[i] = RegimeType.HIGH_VOL
        elif fwd_ret > BULL_THRESHOLD:
            regimes.iloc[i] = RegimeType.BULL
        elif fwd_ret < BEAR_THRESHOLD:
            regimes.iloc[i] = RegimeType.BEAR
        else:
            regimes.iloc[i] = RegimeType.SIDEWAYS

    return regimes


# ============================================================================
# Optimized Pre-Calculated Indicators
# ============================================================================

def calculate_all_indicators(df: pd.DataFrame) -> dict:
    """
    Pre-calculate all indicators needed for regime detection.
    This is done ONCE before grid search to avoid redundant calculations.
    """
    close = df['close']
    high = df['high']
    low = df['low']

    indicators = {}

    # Daily returns
    daily_returns = close.pct_change()
    indicators['daily_returns'] = daily_returns

    # Rolling volatility (20-day)
    rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252)
    indicators['rolling_vol'] = rolling_vol

    # Volatility percentile (vs 252-day history)
    vol_percentile = pd.Series(index=df.index, dtype=float)
    for i in range(252, len(df)):
        hist_vols = rolling_vol.iloc[max(0, i-252):i]
        current_vol = rolling_vol.iloc[i]
        if not pd.isna(current_vol) and len(hist_vols.dropna()) > 0:
            vol_percentile.iloc[i] = (hist_vols < current_vol).sum() / len(hist_vols.dropna()) * 100
        else:
            vol_percentile.iloc[i] = 50.0
    indicators['volatility_percentile'] = vol_percentile

    # ADX calculation
    period = 14
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.ewm(span=period, adjust=False).mean()
    atr_safe = atr.replace(0, np.nan).fillna(1e-10)

    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_safe)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_safe)

    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, np.nan).fillna(1e-10)
    dx = 100 * abs(plus_di - minus_di) / di_sum_safe
    adx = dx.ewm(span=period, adjust=False).mean()
    indicators['adx'] = adx

    # Moving averages
    sma_20 = close.rolling(window=20).mean()
    sma_50 = close.rolling(window=50).mean()
    sma_200 = close.rolling(window=min(200, len(df))).mean()

    # Trend direction
    trend_direction = pd.Series(index=df.index, dtype=float)
    for i in range(200, len(df)):
        price = close.iloc[i]
        s20, s50, s200 = sma_20.iloc[i], sma_50.iloc[i], sma_200.iloc[i]

        if pd.isna(s20) or pd.isna(s50) or pd.isna(s200):
            trend_direction.iloc[i] = 0
            continue

        above_20 = 1 if price > s20 else -1
        above_50 = 1 if price > s50 else -1
        above_200 = 1 if price > s200 else -1
        ma_aligned = 1 if s20 > s50 > s200 else (-1 if s20 < s50 < s200 else 0)
        trend_direction.iloc[i] = (above_20 + above_50 + above_200 + ma_aligned) / 4
    indicators['trend_direction'] = trend_direction

    # Trend strength
    trend_strength = pd.Series(index=df.index, dtype=float)
    for i in range(200, len(df)):
        adx_val = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 0
        adx_strength = min(adx_val / 50, 1.0)

        if i >= 10:
            slope_20 = (sma_20.iloc[i] - sma_20.iloc[i-5]) / sma_20.iloc[i-5] if sma_20.iloc[i-5] > 0 else 0
            slope_50 = (sma_50.iloc[i] - sma_50.iloc[i-10]) / sma_50.iloc[i-10] if sma_50.iloc[i-10] > 0 else 0

            if (slope_20 > 0 and slope_50 > 0) or (slope_20 < 0 and slope_50 < 0):
                slope_strength = min(abs(slope_20 * 10) + abs(slope_50 * 5), 1.0)
            else:
                slope_strength = 0.2
        else:
            slope_strength = 0.5

        trend_strength.iloc[i] = 0.6 * adx_strength + 0.4 * slope_strength
    indicators['trend_strength'] = trend_strength

    return indicators


# ============================================================================
# Fast Regime Detection (uses pre-calculated indicators)
# ============================================================================

def detect_regimes_fast(
    indicators: dict,
    df: pd.DataFrame,
    vol_high_pct: float,
    strong_trend_th: float,
    adx_trend_th: float,
    smoothing_days: int
) -> pd.Series:
    """
    Fast regime detection using pre-calculated indicators.
    Only the classification thresholds are varied.
    """
    vol_percentile = indicators['volatility_percentile']
    trend_strength = indicators['trend_strength']
    trend_direction = indicators['trend_direction']
    adx = indicators['adx']

    regimes = pd.Series(index=df.index, dtype=str)
    last_regime = None
    days_in_regime = 0

    for i in range(len(df)):
        if i < 200:  # Warmup
            regimes.iloc[i] = RegimeType.SIDEWAYS
            continue

        # Get indicator values
        vp = vol_percentile.iloc[i] if not pd.isna(vol_percentile.iloc[i]) else 50.0
        ts = trend_strength.iloc[i] if not pd.isna(trend_strength.iloc[i]) else 0.0
        td = trend_direction.iloc[i] if not pd.isna(trend_direction.iloc[i]) else 0.0
        ax = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 20.0

        # Classify raw regime
        if vp > vol_high_pct:
            raw_regime = RegimeType.HIGH_VOL
        elif ts > strong_trend_th or ax > adx_trend_th:
            if td > 0.1:
                raw_regime = RegimeType.BULL
            elif td < -0.1:
                raw_regime = RegimeType.BEAR
            else:
                raw_regime = RegimeType.SIDEWAYS
        else:
            raw_regime = RegimeType.SIDEWAYS

        # Apply smoothing
        if last_regime is None:
            final_regime = raw_regime
            last_regime = raw_regime
            days_in_regime = 1
        elif raw_regime == last_regime:
            final_regime = raw_regime
            days_in_regime += 1
        elif days_in_regime >= smoothing_days:
            final_regime = raw_regime
            last_regime = raw_regime
            days_in_regime = 1
        else:
            final_regime = last_regime
            days_in_regime += 1

        regimes.iloc[i] = final_regime

    return regimes


# ============================================================================
# Evaluation Functions
# ============================================================================

def calculate_accuracy(detected: pd.Series, ground_truth: pd.Series) -> float:
    """Calculate regime prediction accuracy."""
    common_idx = detected.index.intersection(ground_truth.index)
    detected = detected.loc[common_idx]
    ground_truth = ground_truth.loc[common_idx]

    valid_mask = ~(detected.isna() | ground_truth.isna())
    detected = detected[valid_mask]
    ground_truth = ground_truth[valid_mask]

    if len(detected) == 0:
        return 0.0

    correct = (detected == ground_truth).sum()
    return correct / len(detected)


def calculate_whipsaw_rate(regimes: pd.Series) -> float:
    """Calculate regime whipsaw rate (transitions per 100 days)."""
    valid_regimes = regimes[regimes != '']
    transitions = (valid_regimes != valid_regimes.shift(1)).sum()
    days = len(valid_regimes)

    if days == 0:
        return 100.0

    return (transitions / days) * 100


def calculate_strategy_sharpe(df: pd.DataFrame, detected_regimes: pd.Series) -> float:
    """
    Calculate Sharpe ratio of a simple regime-based strategy.

    Strategy exposure by regime:
    - BULL: 1.0
    - SIDEWAYS: 0.5
    - HIGH_VOL: 0.3
    - BEAR: 0.2
    """
    close = df['close']
    daily_returns = close.pct_change()

    exposure_map = {
        RegimeType.BULL: 1.0,
        RegimeType.SIDEWAYS: 0.5,
        RegimeType.HIGH_VOL: 0.3,
        RegimeType.BEAR: 0.2,
    }

    strategy_returns = []

    for i in range(1, len(df)):
        date = df.index[i]
        if date not in detected_regimes.index:
            continue

        regime = detected_regimes.loc[date]
        if pd.isna(regime) or regime == '':
            regime = RegimeType.SIDEWAYS

        exposure = exposure_map.get(regime, 0.5)
        ret = daily_returns.iloc[i]

        if not pd.isna(ret):
            strategy_returns.append(ret * exposure)

    if len(strategy_returns) < 20:
        return 0.0

    strategy_returns = pd.Series(strategy_returns)
    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()

    if std_ret == 0 or pd.isna(std_ret):
        return 0.0

    sharpe = np.sqrt(252) * (mean_ret - 0.05/252) / std_ret
    return sharpe


def calculate_combined_score(metrics: dict) -> float:
    """Calculate combined optimization score."""
    accuracy_score = metrics['accuracy']
    sharpe_score = max(0, metrics['sharpe']) / 2
    whipsaw_score = 1 - min(metrics['whipsaw'] / 20, 1)

    combined = (
        ACCURACY_WEIGHT * accuracy_score +
        SHARPE_WEIGHT * sharpe_score +
        WHIPSAW_WEIGHT * whipsaw_score
    )

    return combined


# ============================================================================
# Grid Search Optimization
# ============================================================================

def run_grid_search(
    df: pd.DataFrame,
    indicators: dict,
    ground_truth: pd.Series
) -> list:
    """Run grid search over all parameter combinations."""
    param_names = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())

    combinations = list(product(*param_values))
    total_combos = len(combinations)

    print(f"\nRunning grid search over {total_combos} parameter combinations...")

    results = []

    for i, combo in enumerate(combinations):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"   Progress: {i+1}/{total_combos} ({(i+1)/total_combos*100:.1f}%)")

        params = dict(zip(param_names, combo))

        # Detect regimes with these parameters
        detected = detect_regimes_fast(
            indicators, df,
            vol_high_pct=params['volatility_high_percentile'],
            strong_trend_th=params['strong_trend_threshold'],
            adx_trend_th=params['adx_trend_threshold'],
            smoothing_days=params['smoothing_days']
        )

        # Evaluate
        accuracy = calculate_accuracy(detected, ground_truth)
        whipsaw = calculate_whipsaw_rate(detected)
        sharpe = calculate_strategy_sharpe(df, detected)

        combined_score = calculate_combined_score({
            'accuracy': accuracy,
            'sharpe': sharpe,
            'whipsaw': whipsaw,
        })

        results.append({
            'params': params,
            'accuracy': accuracy,
            'sharpe': sharpe,
            'whipsaw': whipsaw,
            'combined_score': combined_score,
        })

    results.sort(key=lambda x: x['combined_score'], reverse=True)

    return results


# ============================================================================
# Main Optimization
# ============================================================================

def main():
    print("=" * 80)
    print("REGIME THRESHOLD OPTIMIZATION")
    print("=" * 80)

    # Fetch data
    print("\n1. Fetching SPY data...")
    df_train = fetch_spy_data(TRAIN_START, TRAIN_END)
    df_val = fetch_spy_data(VAL_START, VAL_END)

    # Calculate ground truth regimes
    print("\n2. Calculating ground truth regimes...")
    ground_truth_train = calculate_ground_truth_regimes(df_train)
    ground_truth_val = calculate_ground_truth_regimes(df_val)

    print("\n   Training set regime distribution:")
    train_dist = ground_truth_train.value_counts(normalize=True)
    for regime, pct in train_dist.items():
        print(f"      {regime}: {pct*100:.1f}%")

    print("\n   Validation set regime distribution:")
    val_dist = ground_truth_val.value_counts(normalize=True)
    for regime, pct in val_dist.items():
        print(f"      {regime}: {pct*100:.1f}%")

    # Pre-calculate indicators (ONCE)
    print("\n3. Pre-calculating indicators (one-time computation)...")
    train_indicators = calculate_all_indicators(df_train)
    val_indicators = calculate_all_indicators(df_val)
    print("   Indicators computed successfully!")

    # Evaluate baseline
    print("\n4. Evaluating baseline configuration...")
    baseline_params = {
        'volatility_high_percentile': 90.0,
        'volatility_low_percentile': 20.0,
        'strong_trend_threshold': 0.6,
        'adx_trend_threshold': 25.0,
        'smoothing_days': 5,
    }

    baseline_detected = detect_regimes_fast(
        train_indicators, df_train,
        vol_high_pct=baseline_params['volatility_high_percentile'],
        strong_trend_th=baseline_params['strong_trend_threshold'],
        adx_trend_th=baseline_params['adx_trend_threshold'],
        smoothing_days=baseline_params['smoothing_days']
    )

    baseline_metrics = {
        'accuracy': calculate_accuracy(baseline_detected, ground_truth_train),
        'sharpe': calculate_strategy_sharpe(df_train, baseline_detected),
        'whipsaw': calculate_whipsaw_rate(baseline_detected),
    }
    baseline_score = calculate_combined_score(baseline_metrics)

    print(f"   Baseline (Training):")
    print(f"      Accuracy:  {baseline_metrics['accuracy']*100:.2f}%")
    print(f"      Sharpe:    {baseline_metrics['sharpe']:.3f}")
    print(f"      Whipsaw:   {baseline_metrics['whipsaw']:.2f} transitions/100 days")
    print(f"      Combined:  {baseline_score:.4f}")

    # Run grid search
    print("\n5. Running grid search optimization...")
    results = run_grid_search(df_train, train_indicators, ground_truth_train)

    # Get best parameters
    best_result = results[0]
    best_params = best_result['params']

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print("\n   Top 5 Parameter Combinations:")
    print("-" * 80)

    for i, result in enumerate(results[:5]):
        print(f"\n   #{i+1}: Score={result['combined_score']:.4f}")
        print(f"      Params: {result['params']}")
        print(f"      Accuracy={result['accuracy']*100:.2f}%, "
              f"Sharpe={result['sharpe']:.3f}, "
              f"Whipsaw={result['whipsaw']:.2f}")

    # Validate on out-of-sample data
    print("\n6. Validating on out-of-sample data (2023-2024)...")

    # Baseline on validation
    baseline_val_detected = detect_regimes_fast(
        val_indicators, df_val,
        vol_high_pct=baseline_params['volatility_high_percentile'],
        strong_trend_th=baseline_params['strong_trend_threshold'],
        adx_trend_th=baseline_params['adx_trend_threshold'],
        smoothing_days=baseline_params['smoothing_days']
    )
    baseline_val_metrics = {
        'accuracy': calculate_accuracy(baseline_val_detected, ground_truth_val),
        'sharpe': calculate_strategy_sharpe(df_val, baseline_val_detected),
        'whipsaw': calculate_whipsaw_rate(baseline_val_detected),
    }
    baseline_val_score = calculate_combined_score(baseline_val_metrics)

    # Optimized on validation
    optimized_val_detected = detect_regimes_fast(
        val_indicators, df_val,
        vol_high_pct=best_params['volatility_high_percentile'],
        strong_trend_th=best_params['strong_trend_threshold'],
        adx_trend_th=best_params['adx_trend_threshold'],
        smoothing_days=best_params['smoothing_days']
    )
    optimized_val_metrics = {
        'accuracy': calculate_accuracy(optimized_val_detected, ground_truth_val),
        'sharpe': calculate_strategy_sharpe(df_val, optimized_val_detected),
        'whipsaw': calculate_whipsaw_rate(optimized_val_detected),
    }
    optimized_val_score = calculate_combined_score(optimized_val_metrics)

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS (Out-of-Sample)")
    print("=" * 80)

    print(f"\n{'Metric':<25} {'Baseline':>15} {'Optimized':>15} {'Improvement':>15}")
    print("-" * 70)

    acc_improvement = (optimized_val_metrics['accuracy'] - baseline_val_metrics['accuracy']) * 100
    print(f"{'Accuracy':<25} {baseline_val_metrics['accuracy']*100:>14.2f}% "
          f"{optimized_val_metrics['accuracy']*100:>14.2f}% "
          f"{acc_improvement:>+14.2f}%")

    sharpe_improvement = optimized_val_metrics['sharpe'] - baseline_val_metrics['sharpe']
    print(f"{'Sharpe Ratio':<25} {baseline_val_metrics['sharpe']:>15.3f} "
          f"{optimized_val_metrics['sharpe']:>15.3f} "
          f"{sharpe_improvement:>+15.3f}")

    whipsaw_improvement = baseline_val_metrics['whipsaw'] - optimized_val_metrics['whipsaw']
    print(f"{'Whipsaw Rate':<25} {baseline_val_metrics['whipsaw']:>15.2f} "
          f"{optimized_val_metrics['whipsaw']:>15.2f} "
          f"{whipsaw_improvement:>+15.2f}")

    combined_improvement = optimized_val_score - baseline_val_score
    print(f"{'Combined Score':<25} {baseline_val_score:>15.4f} "
          f"{optimized_val_score:>15.4f} "
          f"{combined_improvement:>+15.4f}")

    print("\n" + "=" * 80)
    print("OPTIMIZED PARAMETERS")
    print("=" * 80)

    print(f"\n   volatility_high_percentile: {best_params['volatility_high_percentile']}")
    print(f"   volatility_low_percentile:  {best_params['volatility_low_percentile']}")
    print(f"   strong_trend_threshold:     {best_params['strong_trend_threshold']}")
    print(f"   adx_trend_threshold:        {best_params['adx_trend_threshold']}")
    print(f"   smoothing_days:             {best_params['smoothing_days']}")

    # Save results
    output_data = {
        'optimization_date': datetime.now().isoformat(),
        'data_config': {
            'symbol': SYMBOL,
            'train_period': f"{TRAIN_START} to {TRAIN_END}",
            'validation_period': f"{VAL_START} to {VAL_END}",
        },
        'best_parameters': best_params,
        'baseline_parameters': baseline_params,
        'training_results': {
            'baseline': {
                'accuracy': baseline_metrics['accuracy'],
                'sharpe': baseline_metrics['sharpe'],
                'whipsaw': baseline_metrics['whipsaw'],
                'combined_score': baseline_score,
            },
            'optimized': {
                'accuracy': best_result['accuracy'],
                'sharpe': best_result['sharpe'],
                'whipsaw': best_result['whipsaw'],
                'combined_score': best_result['combined_score'],
            },
        },
        'validation_results': {
            'baseline': {
                'accuracy': baseline_val_metrics['accuracy'],
                'sharpe': baseline_val_metrics['sharpe'],
                'whipsaw': baseline_val_metrics['whipsaw'],
                'combined_score': baseline_val_score,
            },
            'optimized': {
                'accuracy': optimized_val_metrics['accuracy'],
                'sharpe': optimized_val_metrics['sharpe'],
                'whipsaw': optimized_val_metrics['whipsaw'],
                'combined_score': optimized_val_score,
            },
        },
        'improvements': {
            'accuracy_pct': acc_improvement,
            'sharpe_delta': sharpe_improvement,
            'whipsaw_reduction': whipsaw_improvement,
            'combined_score_delta': combined_improvement,
        },
        'top_5_combinations': [
            {
                'rank': i + 1,
                'params': r['params'],
                'accuracy': r['accuracy'],
                'sharpe': r['sharpe'],
                'whipsaw': r['whipsaw'],
                'combined_score': r['combined_score'],
            }
            for i, r in enumerate(results[:5])
        ],
    }

    output_path = OUTPUT_DIR / 'optimized_thresholds.json'
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n   Results saved to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("OPTIMIZATION SUMMARY")
    print("=" * 80)

    print(f"\n   Training Period: {TRAIN_START} to {TRAIN_END}")
    print(f"   Validation Period: {VAL_START} to {VAL_END}")
    print(f"   Parameter Combinations Tested: {len(results)}")

    print(f"\n   Accuracy Improvement: {acc_improvement:+.2f}%")
    print(f"   Sharpe Improvement:   {sharpe_improvement:+.3f}")
    print(f"   Whipsaw Reduction:    {whipsaw_improvement:+.2f} transitions/100 days")

    if combined_improvement > 0:
        print(f"\n   Overall: Optimization SUCCESSFUL (combined score +{combined_improvement:.4f})")
    else:
        print(f"\n   Overall: Optimization did not improve out-of-sample performance")

    print("\n" + "=" * 80)

    return output_data


if __name__ == "__main__":
    main()

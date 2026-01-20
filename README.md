# Adaptive Hedge Fund Trading System

An institutional-grade quantitative trading system combining multi-factor alpha generation with regime-adaptive exposure management.

## Key Results (2024 Out-of-Sample)

| Strategy | Return | Alpha vs B&H | Sharpe |
|----------|--------|--------------|--------|
| **Adaptive Hedge Fund** | +35.08% | -2.94% | 1.16 |
| Market-Neutral HF | +2.61% | -35.41% | -0.06 |
| Buy & Hold | +38.02% | - | 1.94 |

**Bear Market Performance (2022 Q2):** +10.32% vs -21.70% B&H = **+32% alpha**

## Features

- **Multi-Factor Alpha Model**: Momentum (50%), Quality (20%), Value (15%), Low Volatility (15%)
- **Regime-Adaptive Exposure**: 95% long in bull markets, 40% in bear markets
- **Risk Parity Position Sizing**: Equal risk contribution from each position
- **Volatility Targeting**: Dynamic leverage to maintain 22% annual volatility
- **Walk-Forward Validation**: Proper out-of-sample testing, no look-ahead bias
- **Realistic Transaction Costs**: Commission, slippage, and borrow costs modeled

## Installation

```bash
# Clone the repository
git clone https://github.com/blackms/FinRL-Adaptive.git
cd FinRL-Adaptive

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Run Hedge Fund Backtest

```bash
python scripts/hedge_fund_backtest.py
```

### Train RL Agent

```bash
python scripts/train_rl_agent.py --algorithm ppo --timesteps 100000
```

### Run Strategy Optimizer

```bash
python scripts/optimize_strategy.py
```

## Project Structure

```
├── src/trading/
│   ├── strategies/
│   │   ├── hedge_fund.py      # Adaptive hedge fund strategy
│   │   ├── momentum.py        # Momentum strategies
│   │   └── ensemble.py        # Ensemble strategies
│   ├── backtest/              # Backtesting engine
│   ├── data/                  # Data fetchers
│   └── rl/                    # Reinforcement learning environment
├── scripts/
│   ├── hedge_fund_backtest.py # Main backtest runner
│   ├── train_rl_agent.py      # RL training script
│   └── optimize_strategy.py   # Parameter optimization
├── docs/
│   └── adaptive_hedge_fund_strategy.md  # Full documentation
└── tests/                     # Test suite
```

## Strategy Overview

### Adaptive Exposure Mechanism

```
Bull Market (>5% return over 40 days):
  └── 95% net long exposure (capture upside)

Bear Market (<-5% return over 40 days):
  └── 40% net long exposure (defensive)

Neutral Market:
  └── 70% net long exposure (moderate)
```

### Factor Model

| Factor | Weight | Signal |
|--------|--------|--------|
| Momentum | 50% | 60-day price return, skip last 5 days |
| Quality | 20% | Trend consistency × positive return ratio |
| Value | 15% | Negative short-term momentum (contrarian) |
| Low Vol | 15% | Inverse realized volatility |

## Documentation

Full documentation with Mermaid diagrams available at:
- [Adaptive Hedge Fund Strategy](docs/adaptive_hedge_fund_strategy.md)

## Supported RL Algorithms

- PPO (Proximal Policy Optimization)
- A2C (Advantage Actor-Critic)
- SAC (Soft Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed DDPG)

## Performance

### Walk-Forward Validation (2020-2024)

| Period | Strategy | Buy & Hold | Alpha |
|--------|----------|------------|-------|
| 2022 Q2 (Bear) | +10.32% | -21.70% | **+32.02%** |
| 2023 Q3 | +13.77% | -2.87% | **+16.65%** |
| 2024 Q1 | +21.34% | +12.53% | **+8.82%** |

## Requirements

- Python 3.10+
- pandas, numpy, scipy
- yfinance (data fetching)
- stable-baselines3 (RL)
- matplotlib (visualization)

## License

MIT License

## Disclaimer

This software is for educational and research purposes only. Not financial advice. Past performance does not guarantee future results. Trading involves substantial risk of loss.

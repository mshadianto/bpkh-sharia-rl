# Sharia-Constrained RL Portfolio Optimization

**AI-Augmented Investment Decision Support for BPKH Hajj Fund Management**

Ph.D. Research — Sopian (MS Hadianto) · Supervisor: Dr. Indra Gunawan, Ph.D. — FEB UIII / BPKH

---

## Overview

Reinforcement Learning approach to Sharia-compliant portfolio optimization, calibrated to BPKH's Rp179.5T hajj fund portfolio. Compares PPO, A2C, and DQN agents against traditional benchmarks with an Explainable RL (XRL) governance layer.

## Key Results (Real JII Data — 200K Timesteps)

| Strategy | Cum. Return | Ann. Return | Sharpe | Max DD | Calmar | Sharia Score |
|---|---|---|---|---|---|---|
| **PPO** | **+3347.8%** | **+250.1%** | **5.447** | 11.57% | 21.619 | 1.0000 |
| **DQN** | +1411.3% | +161.5% | 4.721 | **8.35%** | 19.334 | 1.0000 |
| A2C | +54.6% | +16.7% | 0.534 | 23.20% | 0.719 | 1.0000 |
| Equal Weight | +38.3% | +12.2% | 0.357 | 25.05% | 0.486 | — |
| Momentum | -19.2% | -7.3% | -0.245 | 72.04% | -0.101 | — |
| RKAT 6.88% | +21.5% | +7.1% | 0.000 | 0.00% | 0.000 | — |

## Live Demos

- **Research Showcase**: [bpkh-sharia-rl.pages.dev](https://bpkh-sharia-rl.pages.dev)
- **Research Dashboard**: [bpkh-dashboard.pages.dev](https://bpkh-dashboard.pages.dev)

## Quick Start

```bash
# Create venv and install dependencies
python -m venv .venv
pip install yfinance pandas numpy gymnasium stable-baselines3 torch ta matplotlib shap pyarrow

# Run with real JII data
python bpkh_rl_portfolio.py --timesteps 50000

# Run with synthetic data (no internet needed)
python bpkh_rl_portfolio.py --synthetic --timesteps 10000

# Single algorithm with custom timesteps
python bpkh_rl_portfolio.py --algo PPO --timesteps 200000
```

## Files

| File | Description |
|---|---|
| `bpkh_rl_portfolio.py` | Main RL training pipeline — PPO, A2C, DQN with BPKH-calibrated environment |
| `sharia_rl_showcase_v3.jsx` | Interactive research proposal showcase (React + Recharts) |
| `bpkh_research_dashboard.jsx` | Experiment tracker + paper figure generator + XRL governance viewer |
| `output/results.json` | Full metrics from latest run |
| `output/equity_curves.png` | Portfolio equity curves visualization |
| `output/governance_proposal.json` | Multi-stakeholder XRL governance proposal |
| `output/backtest_results.csv` | Performance comparison table |
| `output/model_ppo.zip` | Trained PPO model |
| `output/model_dqn.zip` | Trained DQN model |
| `output/model_a2c.zip` | Trained A2C model |

## Novel Contributions

1. **First multi-algorithm RL comparison** (DQN vs PPO vs A2C) for Islamic portfolios
2. **Sharia compliance embedded in RL reward**: `R(t) = α·Sharpe + β·(1−MDD) + λ·SCS`
3. **Governance-aware XRL**: proposals for Badan Pelaksana → Dewan Pengawas workflow
4. **BPKH-calibrated environment**: parameters from actual LP3KH January 2026 data

## Governance Proposal Output

The `governance_proposal.json` contains differentiated outputs for:

1. **Badan Pelaksana** — Allocation recommendation + projected yield
2. **Komite Investasi & Penempatan** — Feasibility analysis
3. **Komite Manajemen Risiko & Syariah** — SCS score + compliance status
4. **Komite Audit** — Full audit trail + decision log
5. **Dewan Pengawas** — Executive summary dashboard

## Key Parameters

| Parameter | Default | Tune Range | Effect |
|-----------|---------|------------|--------|
| `reward_alpha` | 0.5 | 0.3–0.7 | Sharpe weight |
| `reward_beta` | 0.3 | 0.1–0.4 | Drawdown penalty |
| `reward_lambda` | 0.2 | 0.1–0.3 | Sharia compliance |
| `learning_rate` | 3e-4 | 1e-4–1e-3 | Training speed |
| `total_timesteps` | 100K | 50K–500K | Training duration |
| `transaction_cost_bps` | 15 | 10–25 | Realism |

## BPKH Calibration Source

- **LP3KH Januari 2026** (published 20 Feb 2026)
- **Laporan Tahunan BPKH 2024**
- **UU No. 34/2014** — Pengelolaan Keuangan Haji
- **PP No. 5/2018** — Pelaksanaan UU PKH

## Data Sources

- JII (Jakarta Islamic Index) — 20 tickers via yfinance
- BPKH LP3KH January 2026 (unaudited, 20 Feb 2026)
- RKAT 2026 target yield: 6.88%

---

*Governance-Aware RL for Sharia-Compliant Portfolio Optimization*
*Sopian (MS Hadianto) × Dr. Indra Gunawan, Ph.D.*
*FEB UIII / BPKH — 2026*

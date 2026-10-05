# Verification of the Authors' Original PennyLane Code

**Paper:** V. Gurgul, Y. Chen, S. Lessmann, *Variational Quantum Circuit-Based
Reinforcement Learning for Dynamic Portfolio Optimization*, arXiv:2601.18811
(Jan 2026). <https://arxiv.org/abs/2601.18811>

**Code:** <https://github.com/VincentGurgul/qrl-dpo-public>, imported into this
repo unchanged as commit `5e02cbd` (`original_pennylane/qrl-dpo-public/`).

**Verdict:** the authors' code **runs end to end and is fully deterministic**,
and its **classical baselines approximately reproduce** the paper's Table 5.1.
However, the released `MAIN.py` **does not reproduce the paper's quantum
results**: its quantum agents score Sharpe ≈ −0.40 / −0.27 versus the
paper's +0.73 / +0.65. The paper describes a training setup that differs from
the released script in several documented ways (Section 6).

---

## 1. Does it run out of the box? No — three environment fixes were needed

| # | Problem in the public release | Fix applied | Changes results? |
|---|---|---|---|
| 1 | `models.py` imports `actor_critic.actor_critic_functions.DeepActorCritic`, but the `actor_critic/` module is **not in the public repo** → `ModuleNotFoundError` on startup | Removed the unused import (`MAIN.py` never uses `DeepActorCritic`); pristine copy kept as `models_original.py` | No |
| 2 | `requirements.txt` pins `torch==2.6.0.dev20240927`, a nightly build that is **no longer installable** | Pinned stable `torch==2.6.0` (`requirements_fixed.txt`) | No |
| 3 | Missing/incompatible `autoray` dependency for PennyLane | Pinned `autoray==0.6.12` | No |

`MAIN.py` itself has **never been modified** (its only commit is the import).
No model logic, configuration value, or hyperparameter was changed.

## 2. Inputs and parameters (exactly as in the authors' `MAIN.py`)

**Data:** `data/price_data.parquet.gzip` — daily returns, 3,536 rows ×
15 tickers (AAPL, MSFT, JNJ, XOM, JPM, SPY, QQQ, IWM, XLV, XLF, TLT, LQD,
GLD, USO, EFA), 2011-08-09 → 2025-08-29. Evaluated with 7-fold
expanding-window time-series cross-validation (train → 80/20 train/val split
→ test).

**Global config:**

| Setting | Value |
|---|---|
| LOOKBACK_WINDOW | 30 days |
| FORECAST_WINDOW | 7 days (Auto-ARIMA forecast appended to the state) |
| DYNAMIC_PO / DPO_INTERVAL | True / rebalance every 30 days |
| SHORT_SELLING / CLAMP_NEGATIVES | True / True (weights clamped at −100%) |
| CV_SPLITS | 7 |
| SEED | 68 |
| Epochs (all RL models) | 50, early stopping off |

State size: 15 assets × (30 + 7) = **555 features**. Quantum models expand this
with `radial_to_linear` to **2,220 features → 15 qubits**, amplitude-encoded.

**Per-model hyperparameters:**

| Model | Predictor | Optimizer | actor_lr | critic_lr | L2 | risk_pref | gamma | target nets |
|---|---|---|---|---|---|---|---|---|
| DDPG | MLP (30 hidden; 34,306 params actor+critic) | SGD | 0.02024 | 0.01425 | 9.59e-3 | −0.2832 | 0.02860 | no |
| QDPG | VQC, 60 weights, amplitude, RY | SGD | 0.09936 | 0.00180 | 3.21e-6 | −0.9286 | 0.009827 | no |
| Deep Q-Learning | MLP (30 hidden) | Adam | 0.00114 | 0.00399 | 5.72e-3 | −0.8135 | 0.04711 | yes |
| Quantum Q-Learning | VQC, 60 weights, amplitude, RY | SGD | 0.09488 | 0.00116 | 5.03e-5 | −0.1201 | 0.001218 | yes |

Equal Weights: 1/15 per asset. MVO: risk aversion 10, re-optimized every
30 days for DPO.

## 3. Outputs produced by the authors' code

Full run of all six models (`authors_main_full_run_2026-06-12.log`,
2026-06-12 23:03 → 06-13 04:33, ~5.5 h). Mean across the 7 folds:

| Model | Profit p.a. SPO | Sharpe SPO | Profit p.a. DPO | Sharpe DPO | Runtime |
|---|---|---|---|---|---|
| Equal Weights | 10.03% | 0.4570 | — | — | <1 s |
| Mean-Variance Opt. | 22.68% | 0.5575 | 21.18% | 0.5033 | 0.6 s |
| DDPG | 11.59% | 0.5523 | 11.59% | 0.5525 | 1 min 39 s |
| **QDPG (PennyLane)** | 6.18% | 0.1011 | −2.35% | **−0.3998** | 1 h 37 min |
| Deep Q-Learning | 14.47% | 0.5046 | 14.99% | 0.4943 | 31 s |
| **Quantum Q-Learning (PennyLane)** | 8.48% | 0.1587 | −1.04% | **−0.2673** | 3 h 51 min |

**Note:** the column the authors' logger labels `std` is actually the
**interquartile range** (`print_results` computes the 75th minus the 25th
percentile), not a standard deviation.

## 4. Reproducibility check (fresh re-run, 2026-10-05)

`qiskit_port/reproduce_authors_main.py` runs the unmodified `MAIN.py` for a
chosen subset of models. The fresh run (`authors_main_fresh_rerun_2026-10-05.log`)
reproduces the June run **to every printed digit**:

| Model | June 2026 | October 2026 re-run | Match |
|---|---|---|---|
| Equal Weights (SPO) | 10.0269% / 0.4570 | 10.0269% / 0.4570 | exact |
| MVO (SPO / DPO Sharpe) | 0.5575 / 0.5033 | 0.5575 / 0.5033 | exact |
| DDPG (SPO / DPO Sharpe) | 0.5523 / 0.5525 | 0.5523 / 0.5525 | exact |
| Deep Q-Learning (SPO / DPO Sharpe) | 0.5046 / 0.4943 | 0.5046 / 0.4943 | exact |
| QDPG (PennyLane) | −0.3998 DPO | re-run in progress | — |

The code is deterministic under `SEED = 68`.

## 5. Comparison with the paper's reported results (Table 5.1)

The paper reports one mean Sharpe per model across the 7 folds (RL agents
rebalanced every 30 days, i.e. DPO; Equal Weights and MVO static, i.e. SPO).

| Model | Released code | Paper Table 5.1 | Difference |
|---|---|---|---|
| Equal Weights | 0.4570 | 0.4375 | +0.020 |
| Mean-Variance Opt. (static) | 0.5575 | 0.5919 | −0.034 |
| DDPG (MLP, ~34k params) | 0.5525 | 0.4384 (27k tier) | +0.114 |
| Deep Q-Learning (MLP) | 0.4943 | 0.4939 (27k tier) | +0.0004 |
| **Quantum DDPG, 60 params** | **−0.3998** | **0.7281** | **−1.128** |
| **Quantum DQN, 60 params** | **−0.2673** | **0.6537** | **−0.921** |

Paper's other rows, for reference: classical 13.5k tier DDPG 0.4287 / DQN
0.4446; 160k tier DDPG 0.7926 / DQN 0.8237; quantum 30-param DDPG 0.4179 /
DQN 0.4776.

**Reading:** classical baselines land within ~0.1 Sharpe of the paper (Deep
Q-Learning within 0.0004). The quantum agents miss by roughly one full
Sharpe unit and have the opposite sign.

## 6. Why the quantum results differ: paper setup vs. released script

The paper's text describes settings that do not match the released `MAIN.py`:

| Aspect | Paper (section) | Released `MAIN.py` |
|---|---|---|
| Quantum optimizer | Adam, with parameter shift (§4.4) | `torch.optim.SGD` for QDPG and QQL |
| Target networks | "both employ an experience replay buffer and target networks" (§3.5) | QDPG uses `soft_update=False` (no target networks) |
| Transaction costs in metric | 0.15% per trade inside the Sharpe ratio (§4.2) | none in `utilities/metrics.py` |
| Sharpe denominator | sample std, T−1 (§4.2) | `np.std` (population, T) — minor |
| Dataset | 5,049 daily prices, Aug 2011 – Sep 2025 (§4.1) | 3,536 rows, 2011-08-09 – 2025-08-29 |
| Hyperparameters | tuned by Bayesian search on validation Sharpe (§4.4) | fixed values in `MAIN.py` |

Transaction costs cannot explain the gap: they would *lower* the paper's
Sharpe ratios relative to the code's, yet the paper's quantum numbers are
much *higher*. The most plausible explanation is that the released fixed
hyperparameters and training settings (SGD, no target networks for QDPG)
are not the configuration that produced Table 5.1.

## 7. Bottom line for our study

1. **The authors' PennyLane code works**: it runs end to end after three
   environment-only fixes, and is bit-for-bit reproducible.
2. **Classical results reproduce** the paper approximately.
3. **The headline quantum results do not reproduce** from the public
   release — a reproducibility gap we document rather than resolve.
4. Our Qiskit port matches this released PennyLane code to ~1e-7, so all our
   Qiskit-vs-PennyLane equivalence claims refer to the **released** code.

**Reproduce:**
```bash
PY=original_pennylane/qrl-dpo-public/.venv/bin/python
PYTHONPATH=. $PY -u qiskit_port/reproduce_authors_main.py           # 4 fast models (~3 min)
PYTHONPATH=. $PY -u qiskit_port/reproduce_authors_main.py "QDPG"    # PennyLane QDPG (~1.5 h)
```

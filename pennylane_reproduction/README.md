# Reproducing the Authors' PennyLane Code

This folder answers one question: **does the authors' original PennyLane code
run the way the authors intended, and does it give the results in their
paper?** It is separate from `comparison_logs/`, which compares our Qiskit
port against PennyLane.

**Paper:** V. Gurgul, Y. Chen, S. Lessmann, *Variational Quantum Circuit-Based
Reinforcement Learning for Dynamic Portfolio Optimization*, arXiv:2601.18811
(Jan 2026). <https://arxiv.org/abs/2601.18811>

**Code:** <https://github.com/VincentGurgul/qrl-dpo-public>, vendored here
unchanged as `original_pennylane/qrl-dpo-public/` (commit `5e02cbd`).

## Folder contents

| Path | What it is |
|---|---|
| `reproduce_authors_main.py` | Runs the authors' `MAIN.py` for chosen models. `MAIN.py` on disk is never edited. `--paper-config` switches on the paper's fixed settings. |
| `summarize_runs.py` | Parses every run log into the stats tables below. |
| `logs/` | The authors' own result logs, named `<date>_<as_released or paper_config>_<models>.log`. |
| `results/stats_summary.md` | **All the stats**: per-model mean, IQR, min, max, runtime; reproducibility; comparison with the paper. |
| `results/stats_summary.csv` | The same numbers as a spreadsheet. |

## Verdict

1. **The code runs**, after three environment-only fixes (below).
2. **It is fully reproducible.** Re-running it four months later gave the same
   numbers to every printed digit.
3. **Its classical results match the paper closely.** Deep Q-Learning lands
   within 0.0004 Sharpe of the paper's value.
4. **Its quantum results, as released, do not match the paper.** Quantum DDPG
   scores Sharpe −0.40 against the paper's 0.73, and Quantum Q-Learning −0.27
   against 0.65.
5. **The released script skips two settings the paper says it used**: soft
   target-network updates and early stopping. Switching them on lifts the
   classical models sharply (DDPG 0.55 → 0.77). Quantum runs with the paper's
   settings are next.

## 1. Fixes needed before it would run

| Problem in the public release | Fix | Changes results? |
|---|---|---|
| `models.py` imports an `actor_critic` module that is not in the public repo, so it crashes at start-up | Removed the unused import (`MAIN.py` never uses it); the original is kept as `models_original.py` | No |
| `requirements.txt` pins a PyTorch nightly (`2.6.0.dev20240927`) that can no longer be installed | Pinned `torch==2.6.0` in `requirements_fixed.txt` | No |
| PennyLane needs a compatible `autoray` | Pinned `autoray==0.6.12` | No |

## 2. Inputs: the authors' `MAIN.py` vs. the paper

**Data (both):** daily returns for 15 assets (AAPL, EFA, GLD, IWM, JNJ, JPM,
LQD, MSFT, QQQ, SPY, TLT, USO, XLF, XLV, XOM), tested with 7-fold
expanding-window cross-validation.

| Setting | Paper | Authors' `MAIN.py` | Match? |
|---|---|---|---|
| Lookback / forecast window | 30 / 7 days (Appx. A.1) | 30 / 7 | yes |
| Rebalancing interval | 30 days (§4.1) | 30 | yes |
| Epochs | 50 (Appx. A.1) | 50 | yes |
| **Early stopping** | **on, patience 10** (Appx. A.1) | **off for every model** | **no** |
| **Soft target-network updates** | **τ = 0.005** (Appx. A.1) | **off for DDPG and QDPG** | **no** |
| DQN action samples / replay buffer | 10 / unlimited (Appx. A.1) | 10 / unlimited | yes |
| Quantum model size | 30 or 60 parameters (§4.3) | 60 weights (actor 15 qubits, critic 12) | yes (60) |
| Optimizer | tuned over {Adam, SGD} (Appx. A.2) | SGD (quantum, DDPG), Adam (DQN) | consistent |
| Learning rates, L2, risk preference, γ | tuned within ranges (Appx. A.2) | fixed values, all inside those ranges | consistent |
| Transaction costs in the Sharpe ratio | 0.15% per trade (§4.2) | none | no |
| Dataset size | 5,049 daily prices, to Sep 2025 (§4.1) | 3,536 rows, 2011-08-09 to 2025-08-29 | no |

The paper does not print its final tuned values. `MAIN.py`'s values all fall
inside the paper's search ranges, and their long decimals look like
Bayesian-search output, so they are probably the tuned values. The clear
mismatches are the fixed settings: early stopping and soft updates.

**`MAIN.py` hyperparameters, per model:**

| Model | Network | Optimizer | Actor lr | Critic lr | L2 | Risk pref. | γ |
|---|---|---|---|---|---|---|---|
| DDPG | MLP, 30 hidden units (34,306 params) | SGD | 0.02024 | 0.01425 | 9.59e-3 | −0.2832 | 0.02860 |
| QDPG | VQC, 60 weights (actor 15 qubits, critic 12) | SGD | 0.09936 | 0.00180 | 3.21e-6 | −0.9286 | 0.009827 |
| Deep Q-Learning | MLP, 30 hidden units | Adam | 0.00114 | 0.00399 | 5.72e-3 | −0.8135 | 0.04711 |
| Quantum Q-Learning | VQC, 60 weights (actor 15 qubits, critic 12) | SGD | 0.09488 | 0.00116 | 5.03e-5 | −0.1201 | 0.001218 |

Each model's state has 15 assets × (30 + 7) days = 555 features. The quantum
actor expands this to 2,220 features and amplitude-encodes them on 15 qubits
(4 rotation passes of its 60 weights). The quantum critic also sees the 15
portfolio weights (570 inputs, 2,280 features) and uses 12 qubits (5 passes).
Both use RY rotations with reverse-linear CNOT entanglement.

Defaults the trainers use without `MAIN.py` setting them: exploration noise
0.2, soft-update τ 0.005 (only active when soft updates are on), early-stopping
`min_delta` 0, and no L1 or weight decay.

## 3. Runs

| Date | Config | Models | Status |
|---|---|---|---|
| 2026-06-12 | as released | all six | done (about 5.5 hours) |
| 2026-10-05 | as released | Equal Weights, MVO, DDPG, Deep Q-Learning | done; identical to June |
| 2026-10-05 | as released | QDPG | running |
| 2026-10-05 | paper config | DDPG, Deep Q-Learning | done |
| next | paper config | QDPG, Quantum Q-Learning | not run yet |

## 4. Headline results (Sharpe ratio, mean over 7 folds)

Full statistics, including IQR, min, max and runtime for every run, are in
[`results/stats_summary.md`](results/stats_summary.md).

| Model | As released | Paper config | Paper Table 5.1 |
|---|---|---|---|
| Equal Weights (static) | 0.4570 | same (no training) | 0.4375 |
| Mean-Variance (static) | 0.5575 | same (no training) | 0.5919 |
| DDPG (rebalanced) | 0.5525 | **0.7708** | 0.4384 (27k) / 0.7926 (160k) |
| Deep Q-Learning (rebalanced) | 0.4943 | **0.6184** | 0.4939 (27k) / 0.8237 (160k) |
| **QDPG** (rebalanced) | **−0.3998** | not run yet | **0.7281** |
| **Quantum Q-Learning** (rebalanced) | **−0.2673** | not run yet | **0.6537** |

The paper reports classical models at three sizes (13.5k, 27k, 160k
parameters). The code's classical networks have 34k, closest to the 27k tier.

**Reading:**
- The as-released classical models match the paper's 27k tier; Deep
  Q-Learning is within 0.0004.
- Switching on the paper's two fixed settings moves the classical models up
  toward the paper's best (160k) results. So those settings matter a lot, and
  they are the prime suspect for the quantum gap.
- The paper's numbers include 0.15% transaction costs and the code's don't.
  Costs would only lower the paper's numbers, so they can't explain why the
  paper's quantum results are higher.

## 5. Other issues in the authors' code

- The authors' logger prints the **interquartile range** (75th minus 25th
  percentile) under the label "std". The stats here call it IQR.
- The paper's §4.4 says the quantum models used Adam, but Appendix A.2 lists
  the optimizer as a tuned choice between Adam and SGD. The code uses SGD.

## Reproduce

```bash
PY=original_pennylane/qrl-dpo-public/.venv/bin/python
$PY -u pennylane_reproduction/reproduce_authors_main.py                        # 4 fast models, as released (~3 min)
$PY -u pennylane_reproduction/reproduce_authors_main.py "QDPG"                 # quantum DDPG, as released (~1.5 h)
$PY -u pennylane_reproduction/reproduce_authors_main.py --paper-config "DDPG" "Deep Q-Learning"
$PY pennylane_reproduction/summarize_runs.py                                   # rebuild the stats tables
```

Each run writes a log to `original_pennylane/qrl-dpo-public/results_logs/`.
Copy it into `logs/` with the naming pattern above, then rerun
`summarize_runs.py`.

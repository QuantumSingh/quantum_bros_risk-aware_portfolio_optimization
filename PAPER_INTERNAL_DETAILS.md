# Paper Internal Details — Exact Parameters, Loss Curves, Date Ranges, Plot Guide, Stress & Tail Risk

Companion to `TECHNICAL_DOCUMENTATION.md`. This file records every
internal detail a reader (or reviewer) could ask for: the exact
configuration behind each experiment, what the loss curves are and which
ones the paper needs, the calendar dates of every data window, a
plot-by-plot sanity guide, and the stress-test / tail-risk results.

---

## 1. Exact experiment parameters

### 1.1 Benchmark configurations

| Parameter | Benchmark 1 | Benchmark 2 | Stress tests |
|---|---|---|---|
| Dataset | original repo returns | custom_nonpaper_10 (pct_change of prices) | original repo returns |
| Assets | first 4 (AAPL, EFA, GLD, IWM) | all 10 (NVDA..KO) | first 4 |
| Rows | trailing 240 | trailing 240 | 192 before each window + window |
| Split | 60/20/20 chronological (144/48/48) | same | 144 train / 48 val / window test |
| Lookback window | 10 days | 5 days | 10 days |
| Forecast window | 0 | 0 | 0 |
| num_weights (circuit) | 16 | 10 | 16 |
| Qubits (actor) | 8 (160 features) | 10 (200 features -> max(8, 10)) | 8 |
| max_epochs | 25 | 25 (Qiskit: 5, noted in table) | 25 |
| Early stopping | patience 5, min_delta 1e-4, on val critic loss | same | same |
| Exploration noise | sigma = 0.2 (Gaussian, DDPG trainer default) | same | same |
| Short selling | yes (activation w = x / sum(x)) | yes | yes |
| Negative clamping | reduce_negatives at evaluation | same | same |
| DPO rebalance interval | 10 days (=lookback); MVO: 30 | 5 days; MVO: 30 | 10 days |
| Transaction cost (metrics) | 10 bps one-way | same | same |
| Seed | 68 (torch + numpy, global) | 68 | 68 |

### 1.2 Per-model training hyperparameters (from the repo's tuning, unchanged)

| Model | actor_lr | critic_lr | optimizer | l2_lambda | soft_update | risk_preference (rho) | gamma |
|---|---|---|---|---|---|---|---|
| Classical DDPG | 0.020240 | 0.014249 | SGD | 9.586e-3 | no | -0.2832 | 0.02860 |
| PL / Qiskit QDPG | 0.099357 | 0.001804 | SGD | 3.207e-6 | no | -0.9286 | 0.009827 |
| Classical Deep Q-Learning | 0.001142 | 0.003991 | Adam | 5.716e-3 | yes (tau 0.005) | -0.8135 | 0.04711 |
| PL / Qiskit Quantum Q-Learning | 0.094882 | 0.001164 | SGD | 5.030e-5 | yes (tau 0.005) | -0.1201 | 0.001218 |

Additional: classical models use a (30,)-hidden-unit MLP predictor;
classical DQL uses num_action_samples = 10; quantum models use amplitude
encoding + radial_to_linear + rotation_axes="y" + reverse_linear
entanglement; DQL/QQL sample 10 random candidate actions through the
critic per window. Replay buffer: push each transition, sample 1
(uniform) per update. l1_lambda = 0 and weight_decay = 0 everywhere.

Note the strongly negative risk_preference of the quantum DDPG
(rho = -0.93): the reward r = mean(w.s) + rho*std(w.s) penalizes
volatility heavily, which is visible in the stress-test behavior
(Section 6).

### 1.3 Circuit parameters (per benchmark actor)

| | Benchmark 1 actor | Benchmark 2 actor | MAIN (reference) |
|---|---|---|---|
| Raw features | 40 (4 assets x lookback 10) | 50 (10 x 5) | 555 |
| After radial_to_linear | 160 | 200 | 2220 |
| Qubits | 8 | 10 | 15 |
| Hilbert dim | 256 | 1024 | 32768 |
| Trainable RY weights | 16 | 10 | 60 |
| Decomposed depth / CX (8q instance) | 1490 / 254 | larger (state prep ~2^n) | -- |

Critic input adds the action: e.g. Benchmark 1 critic sees
(40 + 4) x 4 = 176 features -> 8 qubits.

---

## 2. Loss curves: definitions, parameters, and what the paper needs

### 2.1 What each curve IS (definitions, so readers can interpret signs)

- **Actor loss** = -Q(s, mu(s)) + l1_lambda*||W||_1 + l2_lambda*||W||_2^2.
  It is the *negated critic value* of the actor's action: decreasing
  actor loss means the critic scores the policy's actions higher. It can
  legitimately be negative and is NOT a fit error — do not expect it to
  approach zero.
- **Critic loss** = (r + gamma * Q(s', mu(s')) - Q(s, a~))^2, the squared
  temporal-difference error on the noisy replayed transition. Approaches
  zero as the critic becomes self-consistent; occasional spikes are
  normal (replay sampling + exploration noise + the moving target).
- **Validation critic loss** = same TD error computed over the
  validation windows without exploration noise. This is the
  early-stopping signal (patience 5, min_delta 1e-4).

Parameters that shape these curves (record them next to any loss plot):
optimizer + learning rates, gamma, risk_preference, l2_lambda,
exploration noise sigma = 0.2, replay buffer sample size 1, soft-update
tau (DQL variants), epochs, early-stopping settings, seed. All values in
Section 1.2 — the curves are meaningless to a reader without them.

### 2.2 Which loss curves the paper should include (and already has)

1. **Per-model actor + critic curves, Benchmark 1** — figure
   `training_loss_curves.png` (left: actor, right: critic; all six
   trained models). Purpose: shows all models actually converge and
   lets readers compare classical vs quantum optimization behavior.
2. **Epoch-plateau curve with the early-stopping marker** — figure
   `epoch_plateau_loss_curve.png` (100-epoch PennyLane QDPG run, actor +
   critic + val-critic, red line at the patience-5 stop, epoch 24).
   Purpose: justifies the 25-epoch benchmark budget.
3. **Test metric vs epochs** — `metric_vs_epochs.png` (Sharpe and profit
   vs 10/25/50/100 epochs, two stacked panels). Purpose: the plateau in
   *test* space, which is what actually matters.
4. Optional (data already saved in `comparison_logs/series/*.npz` under
   `val_critic_loss`): a validation-loss overlay per model, and a
   PennyLane-vs-Qiskit loss-curve overlay for the same config — the
   latter visualizes the per-call-parity vs trajectory-divergence story.

What NOT to do: do not present actor loss as "error", and do not smooth
away critic-loss spikes — they are honest properties of single-sample
replay training; mention them in the caption.

---

## 3. Exact date ranges (calendar dates of every window)

### 3.1 Benchmark 1 (original dataset, 4 assets, 240 trailing rows)

| Split | Dates | Rows |
|---|---|---|
| Full | 2024-09-16 .. 2025-08-29 | 240 |
| Train | 2024-09-16 .. 2025-04-11 | 144 |
| Validation | 2025-04-14 .. 2025-06-23 | 48 |
| Test | **2025-06-24 .. 2025-08-29** | 48 |

### 3.2 Benchmark 2 (custom_nonpaper_10, 240 trailing rows)

| Split | Dates | Rows |
|---|---|---|
| Full | 2025-07-21 .. 2026-07-02 | 240 |
| Train | 2025-07-21 .. 2026-02-12 | 144 |
| Validation | 2026-02-13 .. 2026-04-23 | 48 |
| Test | **2026-04-24 .. 2026-07-02** | 48 |

### 3.3 Stress-test windows (original dataset; retrained per window on the
144+48 rows immediately preceding the test start)

| Window | Test dates | Character |
|---|---|---|
| covid_crash_2020 | 2020-02-15 .. 2020-06-30 | crash then rebound |
| bear_2022 | 2022-01-03 .. 2022-12-30 | grinding bear market |
| calm_2024 | 2024-01-02 .. 2024-06-28 | control (steady bull) |

### 3.4 Sweep windows

All original-dataset sweep slices are *trailing* windows ending
2025-08-29 (rows=120 starts 2025-03; rows=1768 starts 2018-08); the
custom-dataset run ends 2026-07-02. This is why the rows axis conflates
data volume with test-window regime — state it wherever the rows axis is
shown.

**Both benchmark test windows are short (48 days) and recent — that is
the paper's central smallness caveat. Annualized numbers from 48 days
are regime snapshots, not expected returns. The stress tests exist
precisely to complement them with adverse regimes.**

---

## 4. Plot-by-plot guide: what each figure shows and whether it "makes sense"

All in `comparison_logs/plots/`. Verdicts from direct visual inspection.

| Figure | What it shows | Verdict / caveat to state |
|---|---|---|
| benchmark1_cumulative_returns | 8 DPO equity curves, 48 test days | OK. Lines cluster within +-8%; EW/DDPG on top. Caption: short window. |
| benchmark1_runtime_bar | log-scale runtimes | OK. Note log scale + the 0.01s display floor for Equal Weight. |
| benchmark1_metrics_comparison | Sharpe/CAGR/vol/VaR bars | OK. Sharpe values >3 for EW reflect the short calm window — say so. |
| training_loss_curves | actor+critic per model | OK. Actor losses negative by construction (see 2.1). |
| custom_nonpaper_10_cumulative_returns | Benchmark 2 equity curves | OK and striking: quantum pairs (PL solid/Qiskit) visibly track each other — use as visual parity evidence. Classical lines compress near zero; acceptable, but consider a log-wealth version if reviewers ask. |
| custom_nonpaper_10_runtime_bar | runtimes | OK. Qiskit bars reflect 5-epoch budget — caption must say so. |
| custom_nonpaper_10_metrics_comparison | metric bars | OK. CAGR panel dominated by quantum; annualization artifact caveat. |
| epoch_plateau_loss_curve | 100-epoch losses + ES marker | OK. Spikes after epoch ~50 are replay/exploration variance, not divergence. |
| metric_vs_epochs | Sharpe & profit vs epochs (2 panels) | OK (dual-axis version replaced by stacked panels). |
| runtime_vs_rows / runtime_vs_epochs | scaling | OK. rows axis: regime confound caveat. |
| parameter_sweep_summary | 2x2 sweep overview | OK; same caveats. |
| new_metrics_bar_chart(_benchmark2) | 5 new metrics x 8 models | OK. |
| turnover_comparison | turnover per model x benchmark | OK. MVO's 3.07 = full flips at re-optimization; quantum ~0.85-1.5; classical DDPG ~0. |
| transaction_cost_adjusted_return | gross vs net (10 bps) | OK. Benchmark 2 bars dwarf Benchmark 1 — consider splitting per benchmark for the paper. |
| stress_test_cumulative | 3 stress windows, 6 models | OK and the best figure of the set: quantum shallowest COVID trough with recovery to +5%; MVO stuck at -29%; quantum lags in calm-2024. |
| stress_test_tail_risk | MDD + CVaR bars per window | OK. |
| tail_risk_comparison | MDD/CVaR/worst-day/kurtosis, both benchmarks | OK. |
| circuit_diagram | high-level VQC | OK (8-qubit instance drawn for readability; MAIN is 15 qubits). |

Global caveats to attach wherever relevant: (1) 48-day test windows;
(2) single seed for headline tables; (3) Benchmark 2 CAGRs are
annualization artifacts of a hot window; (4) Qiskit epochs=5 in
Benchmark 2; (5) rows-axis regime confound.

---

## 5. Tail-risk results (existing benchmarks, DPO daily returns)

From `comparison_logs/tail_risk_results.csv` (also .tex). Metrics: max
drawdown, time under water, VaR/CVaR at 5% and 1%, worst day, worst
week, skewness, excess kurtosis. Computed from the saved per-model
daily-return series — no retraining, so exactly consistent with the
benchmark tables. (Calmar / Sortino as headline metrics remain
teammate-owned; drawdown/CVaR here serve the dedicated tail analysis.)

## 6. Stress-test results (headline)

From `comparison_logs/stress_test_results.csv`. DPO Sharpe / max
drawdown / CVaR 5% per window:

| Model | COVID crash | 2022 bear | 2024 calm |
|---|---|---|---|
| Equal Weight | -0.15 / 29.5% / 7.1% | -1.04 / 23.0% / 2.8% | 0.85 / 3.2% / 1.6% |
| MVO | -1.08 / 44.3% / 11.2% | -1.39 / 30.1% / 3.1% | 1.36 / 10.5% / 2.9% |
| Classical DDPG | 0.09 / 23.8% / 5.8% | -0.96 / 23.0% / 2.3% | 0.86 / 3.2% / 1.6% |
| Classical DQL | 0.02 / 37.0% / 8.5% | -0.97 / 33.1% / 4.5% | 0.07 / 23.1% / 3.7% |
| PennyLane QDPG | **0.55 / 20.2% / 3.8%** | **-0.60 / 31.8% / 3.2%** | 0.19 / 7.3% / 2.3% |
| PennyLane QQL | **0.53 / 20.4% / 3.8%** | -0.65 / 32.5% / 3.3% | -0.07 / 6.6% / 2.1% |

Reading: the quantum policies are the only models with positive Sharpe
through the COVID crash, with the shallowest drawdowns and roughly half
the tail loss (CVaR) of Equal Weight; they are least-bad in the 2022
bear; and they underperform in the calm control. This is consistent
with their tuned risk_preference (rho = -0.93 for QDPG): the learned
policies are *defensive*. Frame it exactly that way — a risk profile,
not universal outperformance. (Qiskit models are omitted from stress
runs for runtime; per-call parity with PennyLane is the verified
transfer argument, stated in the table note.)

---

## 7. Conversion rationale and research value

Fully documented in `TECHNICAL_DOCUMENTATION.md` Section 16 (issues the
conversion surfaced; why Qiskit; how it strengthens the research line)
and Section 11 (parity methodology). The one-paragraph version for the
paper: *the conversion is an independent audit and portability proof —
it found real defects (endianness, a non-functional classical head,
unimplemented batching, data-semantics traps), demonstrated the model
trains under hardware-compatible gradient rules at a quantified ~100x
cost, and produced transpilable circuit artifacts that make the
hardware phase auditable. Every reported number is reproducible from a
seeded, committed pipeline.*

## 8. Reproduction

```bash
PY=original_pennylane/qrl-dpo-public/.venv/bin/python
PYTHONPATH=. $PY -u qiskit_port/run_stress_and_tail_tests.py   # stress + tail (~ 4 min)
PYTHONPATH=. $PY -u qiskit_port/generate_poster_plots.py       # all 20 figures
```

(Benchmarks, sweep, parity: see TECHNICAL_DOCUMENTATION.md Section 17.)

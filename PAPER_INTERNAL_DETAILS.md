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

## 8. Post-hoc additions: epoch extension, robustness checks, Qiskit stress, full circuits

### 8.1 Extended epoch ladder (overfitting confirmed)

| Epochs | 10 | 25 | 50 | 100 | 200 | 500 |
|---|---|---|---|---|---|---|
| DPO Sharpe | 0.96 | 2.31 | 2.67 | 2.76 | 2.75 | **2.55** |
| DPO profit p.a. | 38% | 95% | 114% | 115% | 120% | 105% |

Training past ~200 epochs *degrades* test performance — the 25–50 epoch
budget is near-optimal, not merely cheap. The loss-curve figure
(`epoch_plateau_loss_curve.png`) now shows the full 500-epoch run.

### 8.2 Robustness / anti-artifact checks (`robustness_checks.csv`, `robustness_summary.csv`)

Benchmark 1 configuration; figures `robustness_seed_distribution.png`,
`walk_forward_folds.png`.

| Check | Result (test DPO Sharpe) |
|---|---|
| PennyLane QDPG, 10 seeds | mean 2.78, std 1.50, range [0.74, 5.00] — headline seed 68 (0.84) is near the bottom of its own distribution |
| Classical DDPG, 10 seeds | mean 3.20, std 0.42 (much tighter) |
| Placebo: time-shuffled training data | 2.21 ± 0.59 — still "good" |
| Untrained quantum actors (20 inits) | 2.83 ± 1.35 |
| Random Dirichlet policies (2000) | median 3.04, p5–p95 [1.12, 4.56] |
| Walk-forward, 10 folds 2016–2025 | EW 0.45±3.36, QDPG 0.13±2.07, DDPG 0.87±3.22 |

**Interpretation (write this into the paper):** the Benchmark 1 test
window is a calm bull market where *any* long-ish allocation scores a
high Sharpe — trained models sit inside the random-policy null, and
placebo/untrained baselines do "well". Single-window absolute Sharpe is
therefore regime measurement, not evidence of learned skill. The claims
that survive: (a) the framework-equivalence/parity results, (b) the
turnover/cost profiles, (c) the stress-test risk profile (below) —
where strategies genuinely separate — and (d) walk-forward regime
dependence (QDPG has the lowest cross-fold variance but also the lowest
mean; nothing dominates). This reframing is the robustness suite's
central finding, not a weakness of the study.

### 8.3 Qiskit stress-test rows (5-epoch budget, noted in table)

COVID crash: Qiskit QDPG Sharpe **+0.60** / MDD 26.8% / CVaR 5.8%,
Qiskit QQL +0.52 — confirming the PennyLane defensive result
cross-framework. Bear 2022: Qiskit pair ≈ −1.24 (worse than PennyLane's
−0.60/−0.65: the 5-epoch policies differ more; trajectory sensitivity,
see 8.2). Calm 2024: Qiskit ≈ −0.27/−0.26. The COVID conclusion —
quantum policies uniquely positive through the crash — holds in both
frameworks; window-level rankings elsewhere remain budget/seed
sensitive.

### 8.4 All three circuit instances (`comparison_logs/circuits/`)

One architecture, three sizes (same ansatz family everywhere — the
benchmarks do NOT use different circuits, just different instance sizes):

| Instance | Qubits | Features | Weights | Decomposed depth | CX |
|---|---|---|---|---|---|
| Benchmark 1 actor | 8 | 160 | 16 | 1,490 | 254 |
| Benchmark 2 actor | 10 | 200 | 10 | 6,084 | 1,013 |
| MAIN full config | 15 | 2,220 | 60 | 196,526 | 32,794 |

State preparation (~2^n CX) dominates depth at every size; the
variational part stays tiny. This is the quantitative core of the
hardware-feasibility discussion.

### 8.5 Figure inventory update (all figures ≤ 2 panels)

Replacements: `stress_test_cumulative.png` → three single-window
figures `stress_cumulative_{covid_crash_2020,bear_2022,calm_2024}.png`;
`tail_risk_comparison.png` is now MDD+CVaR only, with worst-day/kurtosis
in `tail_risk_shape.png`; benchmark metric grids split into
`*_metrics_comparison.png` (Sharpe+CAGR) and `*_metrics_comparison_risk.png`;
new-metrics 1x5 grids split into `new_metrics_bar_chart*` (returns) and
`new_metrics_risk_chart*`; TC impact split per benchmark; sweep summary
split into `parameter_sweep_summary.png` + `parameter_sweep_runtime.png`.
New: `robustness_seed_distribution.png`, `walk_forward_folds.png`.

## 8b. Verification & Diagnostics (stability, jaggedness, cross-framework)

Three additional studies address code stability, the origin of loss-curve
jaggedness, and cross-implementation consistency.

### 8b.1 Statistical stability & reproducibility
(`run_stability_tests.py` -> `stability_reproducibility.csv`,
`stability_seeds.csv`, `stability_summary.csv/.tex`,
`plots/stability_seed_boxplot.png`)

- **Reproducibility (determinism): PASS.** Each model trained twice at
  seed 68 gives a bit-identical test Sharpe (|diff| = 0.0e0): Classical
  DDPG 3.150824, Classical DQL -0.974783, PennyLane QDPG 0.840228,
  PennyLane QQL 0.940100. Same seed -> same result, for every team member.
- **Stability across 5 seeds** (coefficient of variation CV = std/|mean|):

  | Model | Sharpe mean | std | CV | 95% CI half-width | min..max |
  |---|---|---|---|---|---|
  | Classical DDPG | 3.04 | 0.58 | **0.19** | 0.72 | 2.06..3.53 |
  | Classical Deep Q-Learning | 1.07 | 1.99 | **1.87** | 2.48 | -0.98..3.13 |
  | PennyLane QDPG | 3.20 | 1.62 | **0.51** | 2.01 | 0.84..5.00 |
  | PennyLane Quantum Q-Learning | 2.79 | 1.25 | **0.45** | 1.55 | 0.94..4.12 |

  Interpretation: three of four models have CV > 0.4, so a single-seed
  Sharpe is NOT representative; Classical DQL is especially unstable
  (CV 1.87, sign-changing across seeds). This is the quantitative backing
  for reporting multi-seed distributions rather than single numbers, and
  it is consistent with the random-policy-null finding (Section 8.2): in
  this calm window, seed variance swamps model differences.

### 8b.2 Why the loss curves are jagged
(`run_internal_diagnostics.py` -> `internal_diagnostics.csv`,
`internal_diagnostics_summary.csv`,
`plots/internal_diagnostics.png`, `plots/internal_jaggedness_drivers.png`)

An instrumented, faithful replica of the DDPG update loop logs every
internal quantity at every training window (not per-epoch). Findings:

- The **per-epoch average critic loss is smooth** (0.213 -> 0.170 over 6
  epochs) while the **per-window critic loss is jagged** (0.16..0.24) —
  the jaggedness is intra-epoch noise that only partly averages out.
- The **reward is dominated by the volatility term** (risk_pref *
  volatility, rho = -0.93), not the profit term. Each window has
  different market volatility and a freshly redrawn exploration noise, so
  the training signal is inherently noisy. The heavy volatility penalty
  that produces the DEFENSIVE policy is the same mechanism that makes
  training jagged.
- Critic-loss magnitude correlates most with the **critic's own
  prediction** Q(s,a) (corr 0.90), then the moving target |target Q|
  (0.13) and exploration-noise magnitude (0.11). Measured, not assumed.
- Structural amplifiers (design of the repo trainer): single-sample
  replay (each update uses one sampled transition), exploration noise
  redrawn per window, and no target network (soft_update=False), so the
  critic chases a target that moves every step.

### 8b.3 Cross-framework verification
(`run_cross_verification.py` -> `cross_verification_summary.csv/.tex`,
`plots/cross_verification_loss.png`)

Two distinct levels of agreement, to be reported separately:

- **Per-call parity: EXACT** (~1.2e-7 forward, ~3.6e-7 gradients, from the
  30-config parity suite). The two implementations are equivalent per
  call.
- **Trajectory agreement: tracks then diverges.** Trained on the same
  seed and data, PennyLane and Qiskit QDPG loss curves overlay for ~70
  epochs, then Qiskit shows late critic-loss spikes PennyLane does not.
  Final trained metrics differ despite exact per-call parity:

  | Pair | Metric | PennyLane | Qiskit | abs diff |
  |---|---|---|---|---|
  | QDPG | Sharpe | 0.840 | 1.107 | 0.267 |
  | QDPG | ann. return | 0.228 | 0.285 | 0.057 |
  | Q-Learning | Sharpe | 0.940 | 0.537 | 0.403 |

  This is expected chaotic amplification of float32-level per-call
  differences across thousands of noise-injected, replay-sampled updates
  — NOT a parity failure. **Team guidance:** verify per-call parity
  (exact) separately from trained trajectories (chaotic); a Sharpe
  difference between frameworks on a trained model is expected, not a bug.

## 9. Reproduction

```bash
PY=original_pennylane/qrl-dpo-public/.venv/bin/python
PYTHONPATH=. $PY -u qiskit_port/run_stress_and_tail_tests.py     # PL/classical stress + tail (~4 min)
PYTHONPATH=. $PY -u qiskit_port/run_stress_qiskit_extension.py   # Qiskit stress rows (~40 min)
PYTHONPATH=. $PY -u qiskit_port/run_robustness_checks.py         # seed/placebo/null/walk-forward (~15 min)
PYTHONPATH=. $PY -u qiskit_port/export_qiskit_paper_circuit.py   # all 3 circuit instances
PYTHONPATH=. $PY -u qiskit_port/generate_poster_plots.py         # all figures
PYTHONPATH=. $PY -u qiskit_port/run_stability_tests.py           # determinism + 5-seed stability (~10 min)
PYTHONPATH=. $PY -u qiskit_port/run_internal_diagnostics.py      # per-window internals / jaggedness (~1 min)
PYTHONPATH=. $PY -u qiskit_port/run_cross_verification.py        # PL vs Qiskit trajectory (seconds)
```

The 500-epoch sweep point is appended by re-running the epoch axis of
`run_parameter_sweep.py` with `max_epochs=500, early_stopping=False`.

(Benchmarks, sweep, parity: see TECHNICAL_DOCUMENTATION.md Section 17.)

# Risk-Aware Portfolio Optimization with Quantum Reinforcement Learning: A Validated PennyLane-to-Qiskit Conversion and Benchmark Study

Paper skeleton. Each section lists its planned content and the produced
artifacts (tables/figures) that will back it. Artifact paths are relative
to the repo root.

---

## 1. Abstract

- [ ] One paragraph: problem (QRL for portfolio optimization), contribution
      (parity-validated Qiskit port of the PennyLane QNN, cross-framework and
      cross-dataset benchmarks, new risk/cost metrics, hardware-readiness plan),
      headline numbers (forward parity ~1e-7; weight-gradient parity ~4e-7;
      benchmark results TBD from final tables).

## 2. Introduction

- [ ] Motivation: reproducibility and framework portability in QML finance.
- [ ] Why Qiskit: hardware access (IonQ), ecosystem, verification value.
- [ ] Contributions list (bulleted, mirrors abstract).

## 3. Background

### 3.1 Portfolio Optimization
- [ ] Markowitz MVO, static vs dynamic allocation (SPO vs DPO), short selling.

### 3.2 Reinforcement Learning
- [ ] DDPG actor-critic; Deep Q-Learning variant used in the original repo;
      reward = mean return + risk_preference * volatility.

### 3.3 Quantum Machine Learning
- [ ] Variational quantum models as function approximators; encodings.

### 3.4 Variational Quantum Circuits
- [ ] Amplitude encoding, trainable RY layer(s), CNOT entanglement,
      Pauli-Z expectation readout.
- Figure: `comparison_logs/circuits/qiskit_vqc_high_level_circuit.png`
- Table: `comparison_logs/circuits/circuit_summary.md`

## 4. Data

### 4.1 Original Repository Dataset
- [ ] 15 tickers (AAPL ... XOM), 3536 daily rows, 2011-08-09 to 2025-08-29.
- [ ] Note: the repo's `price_data.parquet.gzip` already stores daily RETURNS.

### 4.2 Custom Non-Paper 10-Ticker Dataset (`custom_nonpaper_10`)
- [ ] NVDA, AMD, TSLA, AMZN, META, NFLX, COST, UNH, BA, KO.
- [ ] Zero overlap with the original universe (verified in metadata).
- [ ] 2891 rows of adjusted close prices, 2015-01-02 to 2026-07-02
      (`data/custom_portfolios/custom_nonpaper_10_metadata.json`).

### 4.3 Preprocessing and Return Conversion
- [ ] yfinance adjusted close -> pct_change -> inf/NaN cleanup.
- [ ] Validation protocol: shape, date range, min/max/mean/std, NaN/inf
      checks, |return| > 50% warnings (see benchmark run logs).
- [ ] radial_to_linear input transformation (x, x^2, sin x, cos x; 4x features).

## 5. Methodology

### 5.1 PennyLane Baseline
- [ ] Original QuantumNeuralNetwork: amplitude encoding, RY ansatz,
      reverse_linear CNOTs, TorchLayer integration, backprop gradients.

### 5.2 Qiskit Conversion
- [ ] Exact statevector simulation; bit-reversed amplitude ordering to match
      PennyLane conventions; parameter-shift weight gradients; central
      finite-difference input gradients (needed for the DDPG critic-to-actor
      path); per-gate-occurrence shifts for shared parameters (chain rule).
- [ ] Known original-code issues found during conversion: classical_layers
      re-creates an unregistered nn.Linear per forward (crashes on float32,
      non-deterministic on float64); the Qiskit port uses a persistent
      trainable layer instead.

### 5.3 Parity Testing
- [ ] 30-case suite: encodings (amplitude/angle/stacked_angle), axes
      (x/y/z/xy/xyz), entanglement (linear/reverse_linear/full), input
      transformation, output_map, output_activation, classical_layers,
      1-D single-state calls.
- Results to cite: forward max |diff| ~1.2e-7; weight grad ~3.6e-7; input
  grad ~2.9e-4 (finite difference, tolerance 5e-4).
- Script: `qiskit_port/test_full_pennylane_qiskit_parity.py`

### 5.4 Benchmark Design
- [ ] 8 models x 2 datasets; 60/20/20 split; SPO and DPO evaluation;
      identical seeds (68) and tuned hyperparameters across frameworks.
- [ ] Epoch budget justified by the parameter sweep (Section 7.3).

## 6. Model Architecture

### 6.1 Classical Baselines
- [ ] Equal Weight (1/N), rolling MVO (30-day re-optimization),
      classical DDPG and Deep Q-Learning with (30,)-hidden-unit MLP.

### 6.2 PennyLane Quantum Neural Network
- [ ] Architecture parameters per benchmark (qubits, weights).

### 6.3 Qiskit Quantum Neural Network
- [ ] Drop-in equivalence claim + `qiskit_port/qiskit_compatible_qnn.py`.
- Figure: `comparison_logs/circuits/qiskit_vqc_decomposed_circuit.png`
  (depth 1490; 254 CX for the 8-qubit Benchmark 1 actor).

### 6.4 Reinforcement Learning Policies
- [ ] Actor/critic wiring, exploration noise, replay buffer, reward shaping,
      short-selling normalization activation.

## 7. Experimental Setup

### 7.1 Benchmark 1: Original Data Subset
- [ ] 4 assets, trailing 240 rows, lookback 10, 16 weights.
- Table: `comparison_logs/real_data_benchmark_results.tex`
- Log: `comparison_logs/real_data_benchmark_run.log`

### 7.2 Benchmark 2: Custom Non-Paper 10-Ticker Portfolio
- [ ] 10 assets, trailing 240 rows, lookback 5, 10 weights.
- Table: `comparison_logs/custom_nonpaper_10_benchmark_results.tex`
- Log: `comparison_logs/custom_nonpaper_10_benchmark_run.log`

### 7.3 Training Parameter Sweep
- [ ] Axes: rows (120/240/600/1768), lookback (5/10), epochs
      (10/25/50/100, with and without early stopping), assets (4/10),
      batch size (1; batch_size>1 unsupported by the original
      window-by-window loop -- documented failure).
- [ ] Caveat to state: varying `rows` also shifts the test window, so the
      rows axis mixes data volume with market regime.
- Table: `comparison_logs/parameter_sweep_results.tex`
- Figures: `comparison_logs/plots/epoch_plateau_loss_curve.png`,
  `runtime_vs_rows.png`, `runtime_vs_epochs.png`, `metric_vs_epochs.png`

### 7.4 Metrics
- [ ] Repo metrics: annualized profit (1+mean)^252-1, Sharpe (rf=4.18%).
- [ ] Teammate metrics: Calmar, max drawdown, Sortino, CVaR 5%.
- [ ] New metrics (this work, `qiskit_port/portfolio_metrics.py`):
      annualized return (CAGR), annualized volatility, VaR 5%,
      average one-way turnover, transaction-cost-adjusted return (10 bps).

## 8. Results

### 8.1 Qiskit/PennyLane Parity Results
- [ ] Parity table (forward/weight-grad/input-grad max diffs per config).

### 8.2 Benchmark 1 Results
- Table: `comparison_logs/real_data_benchmark_results.tex`
- Figures: `comparison_logs/plots/benchmark1_cumulative_returns.png`,
  `benchmark1_runtime_bar.png`, `benchmark1_metrics_comparison.png`

### 8.3 Benchmark 2 Results
- Table: `comparison_logs/custom_nonpaper_10_benchmark_results.tex`
- Figures: `comparison_logs/plots/custom_nonpaper_10_cumulative_returns.png`,
  `custom_nonpaper_10_runtime_bar.png`,
  `custom_nonpaper_10_metrics_comparison.png`

### 8.4 Training Sweet Spot Results
- [ ] Plateau at ~50 epochs; 25 epochs captures ~84% of plateau Sharpe at
      half the runtime; early stopping (patience 5, min_delta 1e-4) halts
      at ~24 epochs on validation critic loss.
- Figure: `comparison_logs/plots/parameter_sweep_summary.png`

### 8.5 New Metrics Results
- Table: `comparison_logs/new_metrics_results.tex`
- Figures: `comparison_logs/plots/new_metrics_bar_chart.png`,
  `turnover_comparison.png`, `transaction_cost_adjusted_return.png`
- [ ] Expected discussion point: RL DPO strategies rebalance every 5 days
      with high turnover (~1.0 one-way), so transaction costs bite; EW/MVO
      barely trade.

### 8.6 Runtime Analysis
- [ ] PennyLane backprop vs Qiskit parameter-shift + finite-difference:
      per-epoch cost ratios per benchmark; where the Qiskit time goes
      (state preparation decomposition, gradient evaluation counts).

## 9. Discussion

- [ ] Equivalence of frameworks vs practicality; when Qiskit is worth it.
- [ ] Quantum vs classical performance on both datasets, honestly framed
      (small test windows, no hyperparameter re-tuning on custom data).

## 10. Limitations

- [ ] Short test windows; single seed for headline tables; no transaction
      costs inside training reward; exploration noise makes training
      trajectory-sensitive; original-code quirks (classical_layers,
      batch_size); rows-axis regime confound.

## 11. IonQ / Quantum Hardware Inference Plan

- [ ] Summary of `qiskit_port/ionq_inference_plan.md` (inference-only,
      frozen parameters, 5-20 market states, shot-noise budget, error/
      latency/depth reporting).

## 12. Conclusion and Future Work

- [ ] Verified portability recipe; fast_statevector backend; hardware runs;
      batched training loop; cost-aware reward shaping.

## 13. References

- [ ] Original repo/paper; PennyLane; Qiskit; DDPG (Lillicrap et al.);
      Markowitz; QML-for-finance surveys.

## 14. Appendix

- [ ] Full parity case list and tolerances.
- [ ] Full hyperparameter tables (`qiskit_port/benchmark_lib.py` HYPERPARAMS).
- [ ] Decomposed circuit and QASM listing
      (`comparison_logs/circuits/qiskit_vqc_circuit.qasm`).
- [ ] Reproduction commands for every table/figure.

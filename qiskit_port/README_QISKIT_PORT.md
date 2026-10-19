# Qiskit Port — Overview and Deliverables Index

Qiskit conversion of the PennyLane quantum RL portfolio optimization
model, parity-validated and benchmarked. All commands run from the repo
root with:

```bash
PY=original_pennylane/qrl-dpo-public/.venv/bin/python
PYTHONPATH=. $PY -u <script>
```

## Core model files

| File | Purpose |
|---|---|
| `qiskit_exact_amplitude_finite_diff_qnn.py` | The Qiskit QNN: exact statevector sim, bit-reversed amplitude ordering, parameter-shift weight grads (per-gate-occurrence for shared params), finite-difference input grads |
| `qiskit_compatible_qnn.py` | `QiskitQuantumNeuralNetwork` — drop-in replacement class for the PennyLane `QuantumNeuralNetwork` |
| `portfolio_metrics.py` | Five new metrics (CAGR, ann. volatility, VaR 5%, turnover, TC-adjusted return) + weight-trajectory evaluators |
| `benchmark_lib.py` | Shared benchmark machinery (data validation, 8 models, loss capture, exports) |

## Validation (Task A)

- `test_full_pennylane_qiskit_parity.py` — 30-case parity suite.
  Forward max |diff| ~1.2e-7, weight grad ~3.6e-7, input grad ~2.9e-4
  (finite difference; tolerance 5e-4).
- `compare_pennylane_qiskit_forward.py`, `compare_pennylane_qiskit_gradients.py`,
  `compare_initial_qnn_weights.py` (weight init is bit-identical under seed).

## Benchmarks

| Script | Dataset | Outputs (comparison_logs/) |
|---|---|---|
| `run_real_data_benchmark.py` | original repo returns, 4 assets, 240 rows | `real_data_benchmark_results.{csv,tex,md}`, `real_data_benchmark_run.log` |
| `run_custom_nonpaper_10_benchmark.py` | custom_nonpaper_10 (NVDA, AMD, TSLA, AMZN, META, NFLX, COST, UNH, BA, KO) | `custom_nonpaper_10_benchmark_results.{csv,tex,md}`, `..._run.log` |
| `run_parameter_sweep.py` | sweet-spot experiments | `parameter_sweep_results.{csv,tex}`, `parameter_sweep_run.log` |

Data pipeline notes:
- The original repo's `price_data.parquet.gzip` already contains daily
  RETURNS despite its name — no pct_change there.
- `download_custom_nonpaper_10_data.py` saves raw adjusted-close PRICES;
  the benchmark converts with pct_change and validates (NaN/inf/extreme
  return checks) before training.
- `batch_size > 1` is not supported by the original window-by-window
  training loop (documented failure in the sweep results).

## Figures and paper assets

- `generate_poster_plots.py` → everything in `comparison_logs/plots/`
  (cumulative returns, runtime bars, loss curves, epoch plateau, metrics
  comparison, turnover, TC-adjusted returns, sweep summary) plus the
  Task E aggregates `new_metrics_results.{csv,tex}`.
- `export_qiskit_paper_circuit.py` → `comparison_logs/circuits/`
  (text/PNG diagrams, OpenQASM 2, `circuit_summary.md`).
- Paper skeleton: `../paper_outline.md`. Hardware plan: `ionq_inference_plan.md`.

## Known original-code quirks found during the port

- `classical_layers=True` in the original creates a fresh unregistered
  `nn.Linear` inside every `forward()` (crashes on float32 input,
  non-deterministic on float64). The Qiskit port uses a persistent
  trainable output layer instead; the parity test pins the RNG to compare.
- `radial_to_linear` standardizes over dim 0, so shape (1, N) batches
  produce NaN in both frameworks; single states must be 1-D vectors.

# IonQ Inference Plan (Task J)

Goal: demonstrate the trained quantum policy on real hardware WITHOUT
training on it. Training stays on the local exact simulator; IonQ runs
forward-pass inference circuits only.

## Why inference-only

Training the RL loop on hardware is infeasible and unnecessary:
one epoch of the Benchmark 1 QDPG performs hundreds of circuit
evaluations per window (parameter-shift weight gradients plus
finite-difference input gradients for the critic-to-actor path), and the
policy is already parity-verified against PennyLane on the simulator.
Hardware adds evidence about NISQ noise behavior, not about learning.

## Protocol

1. **Train locally on the exact simulator.**
   Use the Benchmark 1 configuration (4 assets, lookback 10 -> 160
   features after radial_to_linear -> 8 qubits, 16 RY weights,
   reverse_linear CNOTs), seed 68.

2. **Freeze trained parameters.**
   Save `model.weights` (16 floats) plus the full config to JSON next to
   the run log. The circuit for a given market state is then fully
   determined by (frozen weights, state amplitudes).

3. **Select 5-20 market states.**
   Take evenly spaced test-set windows from Benchmark 1 (every k-th
   rebalance date, k chosen to give ~10 states), plus the 2-3 states where
   the simulator allocation was most extreme (largest single-asset weight)
   so hardware error on "confident" decisions is visible.

4. **Build inference circuits.**
   For each state: unitary StatePreparation of the 256-dim amplitude
   vector (NOT `initialize`, which inserts resets), the frozen RY/CNOT
   ansatz, then measure qubits 0-3 in the computational basis.
   Reference export: `comparison_logs/circuits/qiskit_vqc_circuit.qasm`
   (depth 1490, {rz: 765, rx: 510, cx: 254, ry: 16} in the rx/ry/rz/cx
   basis before IonQ native-gate transpilation).

5. **Run on IonQ later** (via qiskit-ionq provider), starting with the
   ideal+noise-model cloud simulator, then QPU.
   - Shots: 1000 and 4000 per circuit (two budgets to expose shot-noise
     scaling; standard error of <Z> is sqrt((1-<Z>^2)/shots) ~ 0.03 at
     1000 shots).
   - Estimate <Z_i> for i=0..3 from bitstring marginals; apply the same
     short-selling normalization (w = z / sum(z)) downstream.

6. **Compare against the exact statevector output.**
   For each state: max and mean |<Z_i>_hardware - <Z_i>_statevector|,
   plus the induced portfolio-weight difference after normalization
   (the decision-relevant error), plus the resulting daily-return
   difference over that state's holding interval.

7. **Report.**
   Table per state: expectation error, weight error, shots, queue+execution
   latency, transpiled circuit depth and gate counts (IonQ native basis),
   qubit count. Aggregate: mean/max errors at each shot budget, and
   whether hardware-derived allocations change any rebalance decision
   materially (e.g. weight shift > 5 percentage points).

## Risks / notes

- 8-qubit amplitude encoding of an arbitrary 256-dim state costs ~254
  two-qubit gates after decomposition; on current trapped-ion fidelities
  (~99-99.5% 2q) the expected circuit fidelity is roughly
  0.99^250 ~ 8-30%, so expectation values will be noticeably damped.
  Mitigations to evaluate, in order: (a) reduce to a 4-6 qubit variant
  trained on fewer features, (b) truncate/compress the state preparation
  (drop near-zero amplitudes before synthesis), (c) IonQ debiasing /
  error mitigation options.
- Do NOT run the full RL training or the finite-difference gradients on
  hardware under any circumstances (cost blows up quadratically in
  input_dim x shots).
- Keep a per-run JSON manifest (job ids, timestamps, shots, transpiled
  depth) so latency and drift can be reported honestly.

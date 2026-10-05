# Technical Documentation — Qiskit Port of the PennyLane Quantum RL Portfolio Optimizer

This document explains everything that was built in this project: what was
done and why, the physics and mathematics behind every component, and how
the code works — including a line-by-line walkthrough of the core model
file. It is written to be readable top-to-bottom by someone who knows
Python and basic linear algebra; the quantum computing and finance
background is developed from scratch as it is needed.

Contents:

1.  What was done (project narrative)
2.  Repository map
3.  Quantum computing foundations
4.  The quantum model, end to end, as one formula
5.  Gradients: the parameter-shift rule and finite differences
6.  The reinforcement learning setup and why input gradients matter
7.  Line-by-line: `qiskit_port/qiskit_exact_amplitude_finite_diff_qnn.py`
8.  The drop-in wrapper: `qiskit_compatible_qnn.py`
9.  The metrics module: `portfolio_metrics.py` (with all formulas)
10. The benchmark machinery: `benchmark_lib.py`
11. Parity testing: methodology and tolerance rationale
12. The parameter sweep: design and findings
13. Benchmark results
14. Circuit export and the IonQ plan
15. Numerical gotchas and lessons learned
16. Why the conversion was worth it: issues surfaced, rationale, research value
17. How to reproduce everything

---

## 1. What was done (project narrative)

The starting point was an existing research repo implementing quantum
reinforcement learning (QRL) for portfolio optimization in **PennyLane**:
a variational quantum circuit (VQC) acts as the function approximator
(actor and critic) inside DDPG and Deep Q-Learning trainers that allocate
a portfolio of assets from windows of daily returns.

The project goals, in order:

1. **Port the quantum model to Qiskit** such that it is a functionally
   equivalent drop-in replacement — same forward outputs and same
   gradients within numerical tolerance, so `predictor=QuantumNeuralNetwork`
   can be swapped for `predictor=QiskitQuantumNeuralNetwork` with no other
   code changes.
2. **Prove the equivalence** with a systematic parity test suite
   (30 cases covering every encoding, rotation axis set, entanglement
   pattern, input transformation, output mapping, and calling convention).
3. **Benchmark** the Qiskit and PennyLane models against classical
   baselines (Equal Weight, Mean-Variance Optimization, classical DDPG,
   classical Deep Q-Learning) on two datasets: a subset of the original
   paper's data, and a new 10-ticker universe (`custom_nonpaper_10`:
   NVDA, AMD, TSLA, AMZN, META, NFLX, COST, UNH, BA, KO) that has zero
   overlap with the original paper's tickers.
4. **Justify training parameters** with a sweep over data size, lookback,
   epochs, early stopping, and batch size.
5. **Add five new portfolio metrics** (annualized return, annualized
   volatility, VaR 5%, turnover, transaction-cost-adjusted return) and
   compare all eight models under them.
6. **Export the circuit** (diagrams, QASM, summary) and produce all
   paper/poster figures and tables.
7. **Write the paper skeleton** and an **IonQ hardware inference plan**.

All of these were completed. The parity headline: forward outputs match
to ~1.2e-7, weight gradients to ~3.6e-7, input gradients to ~2.9e-4
(finite-difference path, tolerance 5e-4).

---

## 2. Repository map

```
qiskit_port/
  qiskit_exact_amplitude_finite_diff_qnn.py   The core Qiskit QNN (Section 7)
  qiskit_compatible_qnn.py                    Drop-in wrapper class (Section 8)
  portfolio_metrics.py                        Five new metrics + evaluators (Section 9)
  benchmark_lib.py                            Shared benchmark machinery (Section 10)
  test_full_pennylane_qiskit_parity.py        30-case parity suite (Section 11)
  compare_pennylane_qiskit_forward.py         Quick forward parity check
  compare_pennylane_qiskit_gradients.py       Quick gradient parity check
  compare_initial_qnn_weights.py              Seeded weight-init parity check
  run_parameter_sweep.py                      Task D sweep (Section 12)
  run_real_data_benchmark.py                  Benchmark 1 runner (Section 13)
  run_custom_nonpaper_10_benchmark.py         Benchmark 2 runner (Section 13)
  download_custom_nonpaper_10_data.py         yfinance downloader for the new universe
  export_qiskit_paper_circuit.py              Circuit TXT/PNG/QASM export (Section 14)
  generate_poster_plots.py                    All figures + metric aggregates
  ionq_inference_plan.md                      Hardware inference protocol (Section 14)

comparison_logs/
  real_data_benchmark_results.{csv,tex,md}    Benchmark 1 table
  custom_nonpaper_10_benchmark_results.*      Benchmark 2 table
  parameter_sweep_results.{csv,tex}           Sweep table
  new_metrics_results.{csv,tex}               Five-metric aggregate across benchmarks
  *_run.log                                   Full run logs
  series/*.npz                                Per-model daily returns, weights, loss curves
  series/*_row.json                           Per-model cached result rows (resume support)
  plots/*.png                                 All 17 figures
  circuits/*                                  Circuit diagrams, QASM, summary

data/custom_portfolios/
  custom_nonpaper_10_prices.{parquet.gzip,csv} Adjusted close PRICES (convert with pct_change)
  custom_nonpaper_10_metadata.json             Provenance and validation metadata

paper_outline.md                              14-section paper skeleton
original_pennylane/qrl-dpo-public/            The original repo (untouched model code)
```

---

## 3. Quantum computing foundations

### 3.1 Qubits and statevectors

A single qubit's state is a unit vector in C^2 (2-dimensional complex
space): |psi> = a|0> + b|1> with |a|^2 + |b|^2 = 1. The numbers a, b are
called **amplitudes**. When measured, the qubit yields 0 with probability
|a|^2 and 1 with probability |b|^2 (the Born rule).

A register of n qubits lives in the tensor product space C^(2^n): its
state is a unit vector of 2^n complex amplitudes,

    |psi> = sum_{i=0}^{2^n - 1} c_i |i>,     sum_i |c_i|^2 = 1,

where |i> is the computational basis state labeled by the n-bit binary
expansion of the integer i. This exponential dimension is what the model
exploits: **n qubits can carry 2^n features in their amplitudes**. The
project's Benchmark 1 actor stores 160 classical features in the
amplitudes of just 8 qubits (2^8 = 256 slots, the rest zero-padded).

Simulating this classically costs memory and time proportional to 2^n —
which is why the code's statevector arrays have length 256 (8 qubits) or
1024 (10 qubits), and why the 15-qubit MAIN configuration (32,768
amplitudes) is expensive.

### 3.2 Bit-ordering: the single most dangerous convention

Both frameworks index basis states by integers, but they disagree about
which qubit corresponds to which **bit** of the integer:

- **Qiskit is little-endian**: qubit q corresponds to bit q of the basis
  index. Basis state |i> has qubit 0's value in the least significant bit.
- **PennyLane is big-endian**: wire 0 corresponds to the most significant
  bit.

Consequently the *same physical state* is stored as two different vectors
whose entries are related by **reversing the bits of every index**. For
3 qubits, index 6 = binary 110 maps to binary 011 = index 3. If you feed
PennyLane-ordered amplitudes to Qiskit without this permutation, every
multi-qubit computation silently produces wrong answers that look
plausible. The port handles this once, at encoding time, with a
precomputed permutation (Section 7, `_bit_reverse_indices`).

### 3.3 Gates used by the model

Quantum gates are unitary matrices applied to the statevector. The model
uses four:

**Rotation gates** (single qubit), generated by the Pauli matrices

    X = [[0,1],[1,0]],  Y = [[0,-i],[i,0]],  Z = [[1,0],[0,-1]]

via the exponential map R_P(theta) = exp(-i * theta * P / 2):

    RY(theta) = [[cos(theta/2), -sin(theta/2)],
                 [sin(theta/2),  cos(theta/2)]]        (real rotation)

    RX(theta) = [[cos(theta/2), -i sin(theta/2)],
                 [-i sin(theta/2), cos(theta/2)]]

    RZ(theta) = [[e^{-i theta/2}, 0],
                 [0, e^{+i theta/2}]]                  (phase rotation)

Geometrically these rotate the qubit's Bloch-sphere vector about the
x/y/z axes by angle theta. RY is the workhorse here because it keeps
amplitudes real, which composes naturally with amplitude-encoded real
feature vectors. RZ alone cannot change measurement probabilities in the
Z basis (it only changes phases), which is why the original code warns
that a Z-only ansatz does not converge.

**CNOT (controlled-NOT)**, a two-qubit gate: flips the target qubit iff
the control qubit is 1:

    CNOT = [[1,0,0,0],
            [0,1,0,0],
            [0,0,0,1],
            [0,0,1,0]]   (control = first qubit, target = second)

CNOT is the **entangling** gate: after CNOT, the two qubits' states can
no longer be described independently. Without entangling gates a
multi-qubit circuit is just n independent single-qubit circuits and the
model would factorize — entanglement is what lets measurement outcomes
on qubit 0 depend on features encoded across all qubits.

The model arranges CNOTs in one of three patterns after each full pass
of rotations over the qubits:

- `linear`:          CNOT(0→1), CNOT(1→2), ..., CNOT(n-2 → n-1)
- `reverse_linear`:  CNOT(n-2 → n-1), ..., CNOT(1→2), CNOT(0→1)
  (same pairs, applied in reverse order — order matters because CNOTs
  on overlapping qubits do not commute)
- `full`:            CNOT(i→j) for every ordered pair i != j

### 3.4 Measurement: Pauli-Z expectation values

The model's outputs are not sampled bitstrings but exact **expectation
values** of the Pauli-Z operator on the first `output_size` qubits:

    <Z_q> = <psi| Z_q |psi>.

Z has eigenvalue +1 on |0> and -1 on |1>, so for the n-qubit state the
expectation reduces to a signed sum of basis-state probabilities:

    <Z_q> = sum_i |c_i|^2 * (-1)^{bit_q(i)},

where bit_q(i) is bit q of index i (in Qiskit's little-endian
convention). Each output is therefore a number in [-1, 1]: +1 means
qubit q is certainly 0, -1 certainly 1. This is exactly what the code
computes vectorized in `_z_expectations` (Section 7.9) — no measurement
gates, no shots, no sampling noise. On real hardware the same quantity
would be estimated from repeated measurements with standard error
sqrt((1 - <Z>^2)/shots).

### 3.5 Amplitude encoding

Classical features x in R^d are loaded into the state's amplitudes:

    x  ->  pad with zeros to length 2^n  ->  divide by L2 norm
        ->  |psi_x> = (1/||x~||) sum_i x~_i |i>.

The division by the norm is required by quantum mechanics (states are
unit vectors) and has a modeling consequence: **amplitude encoding is
scale-invariant** — x and 2x produce the identical state. Only the
direction of the feature vector matters. It also makes the encoding a
*nonlinear* function of x (because of the 1/||x|| factor), which is why
input gradients cannot use the parameter-shift rule (Section 5.3).

The circuit realization of "set the statevector to these 2^n numbers" is
non-trivial: Qiskit's `initialize`/`StatePreparation` synthesizes a
cascade of rotations and CNOTs; for 8 qubits the decomposition is ~254
CNOTs and ~1275 single-qubit rotations, total depth ~1490 (this is the
dominant cost of the model and the main challenge for hardware runs).

### 3.6 Angle encoding variants

For completeness the port also implements the repo's two other encodings:

- **angle**: feature i sets the rotation angle of one gate on qubit i
  (RX if 'x' is in `rotation_axes`, else RY). Needs one qubit per
  feature — cheap circuits, expensive qubit counts.
- **stacked_angle**: features wrap around the register (feature i goes
  to qubit i mod n) with an entanglement layer after each full pass —
  more features than qubits at the cost of depth. The original
  implementation has an idiosyncrasy in its "full" entanglement (it uses
  the *feature* index as one CNOT endpoint); the port reproduces it
  exactly, quirk included, because drop-in parity is the goal.

---

## 4. The quantum model, end to end, as one formula

Putting Section 3 together, for input x, trainable weights
theta = (theta_0, ..., theta_{W-1}), n qubits, and M measured qubits,
the model computes

    f_m(x, theta) = <psi(x, theta)| Z_m |psi(x, theta)>,   m = 0..M-1

with

    |psi(x, theta)> = A(theta) . E(T(x)) |0...0>,

where

- T is the optional classical input transformation. The benchmarks use
  `radial_to_linear`: standardize x, then concatenate
  [x, x^2, sin x, cos x] (4x the features). Its purpose is to break
  the radial symmetry that amplitude encoding's scale-invariance
  induces: two return-windows that differ only in overall magnitude
  become distinguishable through the nonlinear features.
- E prepares the encoded state (amplitude encoding: pad, normalize,
  bit-reorder, StatePreparation).
- A(theta) is the ansatz: for i = 0..W-1, apply R_axis(theta_i) on qubit
  (i mod n) for each requested axis, and after every full pass over the
  qubits (except a trailing one) apply the entanglement layer.

The PyTorch wrapper then optionally applies a classical linear head, an
output range map, and an activation:

    y = activation( range_map( Linear_or_Identity( f(x, theta) ) ) ).

In the RL usage, the **actor** has M = number of assets and its
activation normalizes outputs into portfolio weights; the **critic** has
M = 1 and consumes [state, action] concatenated.

---

## 5. Gradients: the parameter-shift rule and finite differences

Training needs dLoss/dtheta (weight gradients) and — for DDPG's actor
update — dCritic/dInput (input gradients). The two use different
mathematics, and the difference is the heart of this port.

### 5.1 The parameter-shift rule (exact, hardware-compatible)

For a gate G(theta) = exp(-i * theta * P / 2) whose generator P satisfies
P^2 = I (true for Pauli X, Y, Z), the expectation of any observable O
after the circuit,

    g(theta) = <0| U2(theta)^dagger O U2(theta) |0>   (theta appearing in one gate),

is an exact sinusoid in theta: g(theta) = A + B cos(theta) + C sin(theta).
Differentiating a sinusoid can be done by evaluating it at two shifted
points; the classical trigonometric identity gives the **parameter-shift
rule**:

    dg/dtheta = [ g(theta + pi/2) - g(theta - pi/2) ] / 2.

This is *not* an approximation — it is exact for gates with involutory
generators, and unlike backpropagation it only requires the ability to
*run the circuit*, which is why it also works on hardware.

**Shared parameters (the chain-rule subtlety).** In this model one weight
theta_i can drive several gates: with `rotation_axes="xy"`, RX(theta_i)
and RY(theta_i) both receive the same value on the same qubit. Writing
the output as g(a, b) with a = b = theta_i (a the RX angle, b the RY
angle), the total derivative is

    d/dtheta_i g(theta_i, theta_i) = (dg/da)|_{a=theta_i} + (dg/db)|_{b=theta_i}.

Each partial derivative must be obtained by shifting **only that gate
occurrence**, holding the other at its base value. Shifting all
occurrences together by pi/2 does NOT compute the derivative (the
function of a shared parameter appearing k times is a sum of harmonics
up to frequency k, and the two-point rule is only valid for a single
frequency). The implementation therefore passes a
`(weight_index, axis, delta)` override into circuit construction so
exactly one gate is shifted per evaluation, and sums the per-axis terms
(Section 7.4). This was verified against PennyLane's autodiff to ~4e-7
for "xy" and "xyz" configurations.

Cost: 2 circuit evaluations per weight per axis per sample. For 16
weights, 1 axis, that is 32 extra simulations per backward pass.

### 5.2 Central finite differences (for input gradients)

The derivative of the outputs with respect to each *input feature* is
approximated numerically:

    df/dx_j ≈ [ f(x + eps*e_j) - f(x - eps*e_j) ] / (2*eps),   eps = 1e-4.

Taylor expansion shows the central form's error is O(eps^2) in the step
(the symmetric evaluation cancels the O(eps) term), while float32
round-off contributes O(u/eps) with u ≈ 1e-7. eps = 1e-4 balances the
two, and empirically yields input-gradient agreement with PennyLane's
exact autodiff of ~2.9e-4 max — hence the parity suite's dedicated,
looser input-gradient tolerance (5e-4) compared to the weight-gradient
tolerance (1e-4).

### 5.3 Why two different schemes?

The trainable weights enter through rotation gates → parameter-shift is
exact and cheap(ish). The inputs enter through **amplitude encoding**,
which involves padding and division by the norm — not a fixed-generator
gate — so no shift rule exists for them; finite differences are the
honest fallback. Cost: 2 * input_dim circuit evaluations per sample,
which for the 160-feature actor is 320 simulations per backward — this
is why input gradients are only computed when actually needed
(Section 7.5).

PennyLane's original model never faces this: on `default.qubit` it
backpropagates *through the simulator's math* (autodiff of the actual
matrix algebra), which is exact and fast but impossible on hardware.
The Qiskit port's gradient path is the hardware-realistic one.

---

## 6. The reinforcement learning setup and why input gradients matter

The trainers come from the original repo (`ddpg/ddpg_functions.py`,
`q_learning/q_learning_functions.py`) and were deliberately not modified.

**State and action.** The state is the flattened lookback window of daily
returns: s_t in R^{L*A} for lookback L and A assets (Benchmark 1:
10 * 4 = 40). The action is the portfolio weight vector w in R^A,
produced by the actor with a normalizing activation: softmax (long-only)
or w / sum(w) (short selling allowed, used in the benchmarks).

**Reward.** For allocation w on window s,

    r(s, w) = mean_t( w . s_t ) + rho * std_t( w . s_t ),   rho < 0,

i.e. mean daily portfolio return penalized by volatility — the
"risk-aware" part of the title. rho is the tuned `risk_preference`
hyperparameter (e.g. -0.93 for the quantum DDPG).

**DDPG (Deep Deterministic Policy Gradient).**
Critic update: minimize the temporal-difference error

    L_critic = ( r + gamma * Q'(s', mu(s')) - Q(s, a~) )^2,

where a~ is the exploration-noised action stored in the replay buffer.
Actor update: deterministic policy gradient — maximize Q(s, mu(s)),
implemented as L_actor = -Q(s, mu_phi(s)) (+ L1/L2 regularization).
Its gradient with respect to actor parameters phi is, by the chain rule,

    dL_actor/dphi = - dQ/da |_{a=mu(s)} * dmu/dphi.

The first factor is **the derivative of the critic's output with respect
to part of its input** (the action half of [s, a]). When the critic is a
quantum model, this is exactly the finite-difference input gradient of
Section 5.2. Without it, the actor receives zero learning signal — this
is why `needs_input_grad` support in the Qiskit backward pass is not an
optional nicety but the difference between training and not training.

**Deep Q-Learning variant.** Same critic TD structure with soft target
networks (Polyak averaging, tau = 0.005); the actor update additionally
scores `num_action_samples=10` random candidate actions through the
critic per step — the reason the quantum Q-learning models are ~2.8x
slower to train than the DDPG ones.

**Evaluation.** Two protocols from the repo:
- **SPO** (static): one allocation from the last validation window, held
  for the whole test period.
- **DPO** (dynamic): re-allocate every `interval` days (the lookback
  length for RL models; 30 days for MVO) from the trailing window;
  within an interval the weights are constant.

---

## 7. Line-by-line: `qiskit_exact_amplitude_finite_diff_qnn.py`

The core file, 498 lines. Every functional line is explained; line
numbers refer to the current file.

### 7.1 Imports and the bit-reversal table (lines 1–20)

```python
 1  import re
 2  import numpy as np
 3  import torch
 4  import torch.nn as nn
 5
 6  from qiskit import QuantumCircuit
 7  from qiskit.quantum_info import Statevector
```

- `re` validates the `rotation_axes` string with a regex (line 191).
- `numpy` does all raw statevector math; `torch`/`nn` provide the
  trainable-parameter and autograd integration so the model plugs into
  the unmodified PyTorch RL trainers.
- `QuantumCircuit` builds circuits; `Statevector` simulates them exactly
  (no shots, full 2^n complex vector).

```python
11  def _bit_reverse_indices(num_qubits: int) -> np.ndarray:
13      size = 2 ** num_qubits
14      indices = np.zeros(size, dtype=int)
16      for i in range(size):
17          reversed_bits = format(i, f"0{num_qubits}b")[::-1]
18          indices[i] = int(reversed_bits, 2)
20      return indices
```

Builds the permutation that converts PennyLane (big-endian) amplitude
ordering into Qiskit (little-endian) ordering — Section 3.2. Line 17
formats integer `i` as an n-bit binary string, reverses the string, and
line 18 parses it back to an integer: `indices[i]` is where PennyLane's
i-th amplitude must go in the Qiskit vector. Computed once per model at
construction (line 235) and reused for every sample — an O(2^n) loop
that would be wasteful per-call.

### 7.2 The custom autograd Function: forward (lines 23–43)

```python
23  class _QiskitAmplitudeParameterShift(torch.autograd.Function):
24      @staticmethod
25      def forward(ctx, inputs, weights, module):
26          inputs_np = inputs.detach().cpu().numpy()
27          weights_np = weights.detach().cpu().numpy()
28
29          outputs = module._forward_numpy(inputs_np, weights_np)
```

PyTorch cannot differentiate through Qiskit, so the quantum evaluation is
wrapped in a `torch.autograd.Function` where we supply both directions by
hand. `forward` receives the input tensor, the weight tensor, and the
module itself (so backward can rebuild circuits with the same
configuration). Lines 26–27 drop out of the autograd graph
(`detach`) and convert to NumPy; line 29 runs the actual quantum
simulation batch (Section 7.10).

```python
31          # PennyLane preserves the input's dimensionality: a single 1-D
32          # state vector produces a 1-D output, not a batch of one.
33          if inputs_np.ndim == 1:
34              outputs = outputs[0]
```

DDPG feeds single states as 1-D vectors and PennyLane returns 1-D
outputs for them. `_forward_numpy` always computes a 2-D (batch, out)
array; this squeeze restores shape parity — without it, downstream
`torch.cat((state, action))` calls in the critic path would break.

```python
36          ctx.module = module
37          ctx.save_for_backward(inputs, weights)
39          return torch.tensor(outputs, dtype=inputs.dtype, device=inputs.device)
```

`ctx` is the container autograd hands back to `backward`: tensors go
through `save_for_backward` (lets PyTorch manage their memory), the
module reference rides on the context directly. The result tensor is
created fresh (constant from autograd's perspective — gradients flow
only through our hand-written backward).

### 7.3 backward: setup (lines 45–66)

```python
45      @staticmethod
46      def backward(ctx, grad_output):
47          inputs, weights = ctx.saved_tensors
50          original_inputs_shape = tuple(inputs.shape)
52          inputs_np = inputs.detach().cpu().numpy()
53          weights_np = weights.detach().cpu().numpy()
54          grad_output_np = grad_output.detach().cpu().numpy()
```

`backward` receives `grad_output` = dLoss/dOutputs (shape = output
shape) and must return dLoss/dInputs and dLoss/dWeights — the
**vector-Jacobian products**. Lines 59–63 normalize both inputs and
grad_output to 2-D so one code path handles single states and batches;
`original_inputs_shape` (line 50) lets the input gradient be reshaped
back at the end (line 135).

```python
65          num_weights = len(weights_np)
66          grad_weights = np.zeros_like(weights_np, dtype=np.float64)
```

Accumulator for dLoss/dtheta, in float64 to avoid float32 accumulation
error over many terms.

### 7.4 backward: weight gradients by per-occurrence parameter shift (lines 68–95)

```python
78          shift = np.pi / 2.0
79          axes_present = [a for a in "xyz" if a in module.rotation_axes]
81          for k in range(num_weights):
82              grad_k = 0.0
84              for axis in axes_present:
85                  out_plus = module._forward_numpy(
86                      inputs_np, weights_np, shift=(k, axis, shift))
88                  out_minus = module._forward_numpy(
89                      inputs_np, weights_np, shift=(k, axis, -shift))
92                  jac_k_axis = 0.5 * (out_plus - out_minus)
93                  grad_k += np.sum(grad_output_np * jac_k_axis)
95              grad_weights[k] = grad_k
```

Direct implementation of Section 5.1. For each weight k and each axis it
drives, run the whole batch twice with **only that one gate occurrence**
shifted by ±pi/2 (the `(k, axis, ±pi/2)` tuple is interpreted inside
circuit construction, Section 7.8). Line 92 is the shift rule
(f(+) - f(-))/2 giving the Jacobian slice d(outputs)/d(theta_k via this
axis); line 93 contracts it with grad_output (the vector-Jacobian
product, summing over both batch and output dimensions since theta_k is
shared across the batch); line 82/93's accumulation over axes is the
shared-parameter chain rule. Total cost: 2 * W * |axes| batched
simulations.

### 7.5 backward: lazy input gradients (lines 97–133)

```python
104         needs_input_grad = ctx.needs_input_grad[0] or getattr(
105             module, "compute_input_gradients", False)
107
108         if not needs_input_grad:
109             ...
114             return None, grad_weights_torch, None
```

`ctx.needs_input_grad[0]` is PyTorch's own bookkeeping: it is True
exactly when something upstream requires gradients with respect to this
function's first argument (the inputs). For the **actor's own update**
it is False (states are data) and the expensive block below is skipped;
for the **critic inside the actor loss** it is True (the action half of
the critic input is a function of actor parameters) and the gradient is
computed. `compute_input_gradients` forces it on for tests. Returning
`None` tells autograd "no gradient for this argument" (also for the
third argument, the module reference).

```python
116         grad_inputs = np.zeros_like(inputs_np, dtype=np.float64)
117         eps = 1e-4
119         batch_size, input_dim = inputs_np.shape
121         for b in range(batch_size):
122             for j in range(input_dim):
123                 inputs_plus = inputs_np.copy()
124                 inputs_minus = inputs_np.copy()
126                 inputs_plus[b, j] += eps
127                 inputs_minus[b, j] -= eps
129                 out_plus = module._forward_numpy(inputs_plus, weights_np)[b]
130                 out_minus = module._forward_numpy(inputs_minus, weights_np)[b]
132                 jac_bj = (out_plus - out_minus) / (2.0 * eps)
133                 grad_inputs[b, j] = np.sum(grad_output_np[b] * jac_bj)
```

Central finite differences (Section 5.2), one feature at a time: perturb
feature j of sample b by ±eps, rerun, form the symmetric difference
quotient (line 132), and contract with that sample's grad_output
(line 133). Note the `[b]` selections — only the perturbed sample's
outputs are affected, but `_forward_numpy` simulates the whole batch, so
there is an easy 2x saving here for batch>1 (left as a known
optimization; the RL loop uses batch 1).

Lines 135–149 reshape the input gradient back to the caller's original
shape (1-D stays 1-D) and convert both gradients to tensors of the
caller's dtype/device. The return order (inputs, weights, module)
mirrors forward's argument order — an autograd API requirement.

### 7.6 The module: constructor (lines 152–271)

```python
162     def __init__(self, input_size, output_size, num_qubits=None,
                     num_weights=None, output_map=None, rotation_axes="y",
                     classical_layers=False, output_activation=None,
                     encoding="amplitude", input_transformation=None,
                     entanglement="reverse_linear", device="cpu", seed=None,
                     compute_input_gradients=False, **kwargs):
```

The signature deliberately mirrors the PennyLane `QuantumNeuralNetwork`
(same names, same defaults) so the class is argument-compatible;
`compute_input_gradients` is the one Qiskit-specific addition and
`**kwargs` swallows anything else so trainer code never breaks.

```python
182         if seed is not None:
183             torch.manual_seed(seed)
184             np.random.seed(seed)
```

Matches the original's global seeding (`set_seeds`) so that, given the
same seed, the subsequent weight initialization draws identical values —
verified bit-identical by `compare_initial_qnn_weights.py`.

Lines 186–195: validation. The regex `[xyz]*` permits any combination
of axes; entanglement and encoding are checked against the supported
sets. Lines 197–207 store every configuration attribute.

```python
209         if input_transformation is not None:
210             name = getattr(input_transformation, "__name__", "")
211             if name == "radial_to_linear":
212                 self.actual_input_size = input_size * 4
213             elif name == "radial_to_linear_small":
214                 self.actual_input_size = input_size * 2
```

The transformations are applied *outside* the quantum part (in
`forward`) but change the feature count the circuit must accommodate:
`radial_to_linear` outputs [x, x^2, sin x, cos x] (4x), the small
variant [x, cos x] (2x). Detection by function name matches how the
original code special-cases the same two functions.

```python
220         if num_qubits is not None: ...
222         elif encoding in ("angle", "stacked_angle"):
223             self.num_qubits = max(self.actual_input_size, self.output_size)
224         else:
225             self.num_qubits = max(
226                 int(np.ceil(np.log2(self.actual_input_size))),
227                 self.output_size)
```

Qubit-count logic copied from the original: angle encodings need a qubit
per feature; amplitude encoding needs ceil(log2(features)) qubits to fit
the features into 2^n amplitudes — and never fewer qubits than outputs,
since each output is one qubit's <Z>.

Lines 230–244: amplitude-specific setup — `feature_dim = 2^n`, the
bit-reversal permutation (Section 7.1), and a capacity check; the angle
branch instead checks features <= qubits (mirroring the original's
error).

```python
259         self.weights = nn.Parameter(
260             torch.empty(self.num_weights, dtype=torch.float32))
262         nn.init.uniform_(self.weights, 0.0, 2.0 * np.pi)
```

The trainable parameters: a flat vector of W angles, uniformly
initialized on [0, 2pi) — the same distribution and RNG consumption as
PennyLane's `TorchLayer`, which is what makes seeded inits bit-identical.
`nn.Parameter` registers it with PyTorch so optimizers see it.

```python
264         if self.classical_layers:
265             self.output_layer = nn.Linear(self.num_qubits, self.output_size, ...)
270         else:
271             self.output_layer = nn.Identity()
```

The optional classical head. Deliberate deviation from the original,
which constructs a *fresh, unregistered* `nn.Linear` inside every
`forward()` call — making it untrainable, non-deterministic call-to-call,
and crash-prone on float32 inputs. The port owns one persistent,
trainable layer; the parity test pins the original's per-call random
layer by seeding to prove the rest of the pipeline matches (Section 11).

### 7.7 Amplitude preparation (lines 273–295)

```python
273     def _pad_and_normalize_numpy(self, x):
274         if x.ndim == 1: x = x.reshape(1, -1)
277         if x.shape[-1] != self.actual_input_size: raise ValueError(...)
283         padded = np.zeros((x.shape[0], self.feature_dim), dtype=np.float64)
284         padded[:, : self.actual_input_size] = x
286         norms = np.linalg.norm(padded, axis=1, keepdims=True)
287         norms = np.maximum(norms, 1e-12)
289         normalized = padded / norms
292         reordered = np.zeros_like(normalized)
293         reordered[:, self._amplitude_permutation] = normalized
295         return reordered
```

Implements Section 3.5 for a whole batch at once: zero-pad each row to
2^n entries (`pad_with=0` in PennyLane terms), L2-normalize each row
(the max with 1e-12 guards an all-zero input from dividing by zero —
`normalize=True` in PennyLane), and finally scatter through the
bit-reversal permutation: line 293 places PennyLane-ordered amplitude i
at Qiskit position `permutation[i]`. After this, Qiskit's simulator
produces the *same physical state* PennyLane would.

### 7.8 Circuit construction (lines 297–391)

```python
297     def _add_entanglement(self, qc):
```

Emits the CNOT pattern of Section 3.3 onto circuit `qc` — three
branches, loop bounds copied verbatim from the original
`ParameterizedQuantumCircuit` (including `reverse_linear`'s descending
order, which matters because overlapping CNOTs do not commute).

Lines 312–347 implement the two angle encodings, faithfully mirroring
the original's `compute_decomposition` methods — including two quirks
preserved intentionally: the encoding rotation is RX whenever 'x'
appears in `rotation_axes` (else RY, 'z' never chosen), and
stacked_angle's "full" entanglement uses the *feature* loop index as a
CNOT endpoint (`if i != j: qc.cx(i, j)` with i potentially a feature
index — line 345–347), reproducing the original's arguably-buggy wiring
because parity, not correction, is the goal here.

```python
349     def _build_numeric_circuit(self, sample, weights, shift=None):
359         qc = QuantumCircuit(self.num_qubits)
361         if self.encoding == "amplitude":
363             qc.initialize(sample, list(range(self.num_qubits)))
```

One circuit per sample. For amplitude encoding, `initialize` hands
Qiskit the prepared 2^n-amplitude vector; Qiskit synthesizes the state
preparation internally (this synthesis is the dominant runtime cost).

```python
369         gates_by_axis = (("x", qc.rx), ("y", qc.ry), ("z", qc.rz))
372         for i, param in enumerate(weights.flatten()):
373             q = i % self.num_qubits
374             base = float(param)
378             for axis, gate_fn in gates_by_axis:
379                 if axis not in self.rotation_axes: continue
382                 angle = base
383                 if shift is not None and shift[0] == i and shift[1] == axis:
384                     angle = base + shift[2]
386                 gate_fn(angle, q)
388             if (i + 1) % self.num_qubits == 0 and (i + 1) != len(weights):
389                 self._add_entanglement(qc)
```

The ansatz A(theta) of Section 4. Weight i lands on qubit i mod n
(line 373); every requested axis gets a gate with the same base value
(line 378–386) — matching the original where one parameter drives
RX and RY and RZ together. Lines 383–384 are the per-occurrence shift
mechanism from Section 5.1: during gradient evaluation, exactly the
(weight, axis) pair named in the `shift` tuple gets `base + delta`;
every other gate — including *other axes of the same weight* — stays at
base. Line 388 inserts the entanglement layer after each complete pass
over the qubits, except after the final weight (the `(i+1) != len`
guard), again matching the original's control flow exactly.

### 7.9 Measurement (lines 393–402)

```python
393     def _z_expectations(self, state, num_measurements):
394         probs = np.abs(state.data) ** 2
395         indices = np.arange(probs.shape[0])
397         expvals = np.empty(num_measurements, dtype=np.float64)
398         for q in range(num_measurements):
399             signs = 1.0 - 2.0 * ((indices >> q) & 1)
400             expvals[q] = np.dot(probs, signs)
402         return expvals
```

Section 3.4 vectorized. Line 394: Born-rule probabilities for all 2^n
basis states at once. Line 399: `(indices >> q) & 1` extracts bit q of
every index (Qiskit little-endian ⇒ that bit *is* qubit q's value);
`1 - 2*bit` maps bit 0 → +1, bit 1 → -1, i.e. the Z eigenvalues.
Line 400: <Z_q> as a single dot product. This replaced an earlier pure-
Python per-amplitude loop that dominated runtime at 20 qubits (a ~10^6-
iteration Python loop per qubit per evaluation).

### 7.10 Batch forward and output stages (lines 404–470)

```python
404     def _prepare_samples_numpy(self, inputs_np):
```

Encoding dispatch: amplitude inputs go through pad/normalize/reorder;
angle inputs are validated for width and passed through as float64
(angles are used directly as rotation parameters).

```python
419     def _forward_numpy(self, inputs_np, weights_np, shift=None):
424         num_measurements = self.num_qubits if self.classical_layers else self.output_size
426         for sample in encoded:
427             qc = self._build_numeric_circuit(sample, weights_np, shift=shift)
428             state = Statevector.from_instruction(qc)
430             outputs.append(self._z_expectations(state, num_measurements))
```

The simulation loop: one circuit build + exact statevector evolution +
measurement per sample. With `classical_layers` the model measures every
qubit (the linear head mixes them down to `output_size`), matching the
original's QNode. `Statevector.from_instruction` applies the circuit's
gates to |0...0> exactly — no shots, no noise.

```python
434     def _map_to_output_range(self, logits):
441         if self.output_map == (-float("inf"), float("inf")):
442             return 100 * torch.arctanh(torch.clamp(logits, -0.9999, 0.9999))
444         normalized_output = (logits + 1) / 2
445         return self.output_map[0] + normalized_output * (range width)
```

Reproduction of the original's `__map_to_output_range`: identity for
(-1, 1); a clamped, scaled arctanh for the infinite map (arctanh
stretches [-1,1] to R; the clamp bounds it; the 100 factor sets scale —
note this amplifies numerical differences by 100/(1 - logit^2), which is
why the parity suite uses scaled tolerances for this case); otherwise an
affine map of [-1,1] onto [lo, hi].

```python
450     def forward(self, tensor):
451         tensor = tensor.float()
453         if self.input_transformation is not None:
454             tensor = self.input_transformation(tensor)
456         tensor = _QiskitAmplitudeParameterShift.apply(tensor, self.weights, self)
462         tensor = self.output_layer(tensor)
464         if self.output_map is not None: tensor = self._map_to_output_range(tensor)
467         if self.output_activation is not None: tensor = self.output_activation(tensor)
470         return tensor
```

The nn.Module entry point, mirroring the original `forward` stage-for-
stage: cast to float32, classical transformation (differentiable in
torch, so its gradient chains automatically with our finite-difference
input gradient), the quantum block via the custom autograd Function,
classical head, range map, activation. In RL use, the activation is the
weight-normalization the trainer injects.

Lines 473–497 are a self-test (`python file.py` builds a small model,
runs forward and backward, prints shapes and gradients).

---

## 8. The drop-in wrapper: `qiskit_compatible_qnn.py`

```python
class QiskitQuantumNeuralNetwork(QiskitExactAmplitudeFiniteDiffQNN):
    def __init__(self, *args, backend="fast_statevector", **kwargs):
        self.backend = backend
        super().__init__(*args, **kwargs)
```

The class the swap targets: `predictor=QiskitQuantumNeuralNetwork` in
the trainers. It subclasses the core model unchanged and reserves a
`backend` argument for the planned split between `fast_statevector`
(training/benchmarks) and `qiskit_reference` (diagrams, QASM, IonQ
prep). Its docstring records the parity-verified feature matrix and the
classical_layers fix.

---

## 9. The metrics module: `portfolio_metrics.py`

All functions consume plain daily-return arrays r_1..r_T and daily
weight matrices W (T x A), so classical and quantum strategies are
measured identically. Conventions: 252 trading days/year, risk-free
rate 4.18% (matching the repo), 10 bps one-way transaction cost.

**The five new metrics:**

1. `annualized_return` — geometric CAGR:
   ( prod_t (1 + r_t) )^(252/T) - 1.
   Compounds the realized path; always ≤ the repo's arithmetic
   annualization (1 + mean)^252 - 1 under volatility (AM–GM), and the
   honest number for "what would this have compounded to".

2. `annualized_volatility` — sqrt(252) * std(r) (population std,
   matching the repo's np.std).

3. `value_at_risk_5` — historical one-day 5% VaR:
   -Percentile_5(r), reported positive. "5% of days lose more than
   this fraction."

4. `avg_turnover` — one-way turnover per rebalance:
   turnover_t = (1/2) * sum_i |w_{t,i} - w_{t-1,i}|; averaged over
   days where it is nonzero (rebalance events), so strategies with
   different rebalance frequencies are comparable. The 1/2 makes
   "sell everything, buy everything else" = 1.0. Static allocations
   score 0. Weight drift within holding periods is ignored, matching
   the repo's evaluation semantics (constant weights inside an
   interval).

5. `tc_adjusted_annual_return` — CAGR of net returns
   net_r_t = r_t - (cost_bps / 1e4) * turnover_t,
   i.e. each rebalance pays proportional costs on traded volume.

**Repo-compatible metrics** (`repo_profit_pa`, `repo_sharpe`) are
reimplemented locally — same formulas as `utilities/metrics.py`
(Sharpe = (252*mean - 0.0418) / (sqrt(252)*std + 1e-7)) — so the module
has no import dependency on the original repo's global ticker config.

**Weight-capturing evaluators.** The repo's `RLEvaluator` returns only
(profit, sharpe); turnover and transaction costs need the weight
trajectory. `evaluate_actor_with_weights` re-implements the DPO loop
with identical semantics (trailing-window state → actor → optional
negative-clamping renormalization → hold for `interval` days) but
returns `daily_returns` *and* the full `weights_by_day` matrix;
`evaluate_actor_spo_with_weights` and `constant_weights_evaluation`
cover the static cases, `weight_schedule_evaluation` covers rolling MVO.
Correctness was verified against the repo evaluator (identical
profit/sharpe) and with unit tests (static ⇒ turnover 0 and
TC-adjusted = gross; alternating full flips ⇒ turnover exactly 1.0).

---

## 10. The benchmark machinery: `benchmark_lib.py`

One module powers both benchmarks and the sweep:

- **Data loading + validation.** `load_original_returns` reads the
  original parquet (which already contains RETURNS — see Section 15)
  and slices assets/rows; `load_custom_nonpaper_10_returns` reads the
  price parquet and converts: pct_change → replace ±inf → dropna.
  `validate_returns` prints and checks shape, columns, date range,
  min/max/mean/std, NaN count, inf count, and warns on any |return| >
  50% — the guard that would have caught the prices-as-returns bug.
- **Split**: chronological 60/20/20 train/val/test (no shuffling —
  time series).
- **Training with loss capture.** The repo trainers print per-epoch
  losses but do not store them. `train_with_loss_capture` redirects
  stdout during `model.train(...)` and parses
  `Epoch i/N, Actor Loss: ..., Critic Loss: ...[, Val Critic Loss: ...]`
  with a regex into arrays — loss curves without touching original code.
- **The 8 models** are declared in `MODEL_SPECS`; `build_rl_model`
  constructs DDPG/DeepQLearning with the right predictor
  (NeuralNetwork / QuantumNeuralNetwork / Qiskit port) and
  `HYPERPARAMS` carries the repo's tuned learning rates, regularization,
  risk preference, and gamma per model family.
- **Qiskit budget control.** `effective_max_epochs` lets a config set
  `qiskit_max_epochs` lower than `max_epochs` (used in Benchmark 2:
  5 vs 25) with an automatic explanatory note in the results row.
- **Resume support.** After each model, its result row is persisted as
  `series/<prefix>_<model>_row.json` next to the `.npz` series
  (daily returns, weights, loss curves). On rerun, cached models are
  loaded and skipped. This exists because hour-long Qiskit runs were
  twice killed by session restarts; after the second kill the seven
  finished models were reconstructed from their series + logged
  runtimes and only the missing model re-trained.
- **Exports**: CSV (all columns), Markdown, and a LaTeX table of the
  headline DPO columns; per-model series for the plot generator.

---

## 11. Parity testing: methodology and tolerance rationale

`test_full_pennylane_qiskit_parity.py` builds the PennyLane and Qiskit
models with the same seed, copies the PennyLane weights into the Qiskit
model (eliminating init differences), runs identical inputs through
both, and compares forward outputs, then backpropagates identical
upstream gradients (`y.sum().backward()`) and compares weight and input
gradients.

Tolerances and why they are what they are:

| Comparison | Tolerance | Source of error |
|---|---|---|
| Forward | 1e-5 (observed ~1.2e-7) | float32 tensor plumbing around float64 simulations |
| Weight grad | 1e-4 (observed ~3.6e-7) | parameter shift is exact; float32 round-trip only |
| Input grad | 5e-4 (observed ~2.9e-4) | finite differences: O(eps^2) truncation + O(u/eps) round-off |
| (-inf,inf) output map | scaled up ~130x | 100*arctanh amplifies differences by 100/(1-logit^2) |

The 30 cases cover: amplitude/angle/stacked_angle encodings; rotation
axes y, x, z, xy, xyz (the multi-axis cases prove the shared-parameter
chain rule); linear/reverse_linear/full entanglement; radial_to_linear;
output_map (0,1) and (-inf,inf); tanh output activation;
classical_layers (with the RNG-pinning protocol below); and 1-D
single-state calls (the DDPG calling convention).

**The classical_layers protocol.** The original creates a fresh random
float64 `nn.Linear` inside every forward (unregistered, untrainable,
crashes on float32 input, non-deterministic on float64 — verified
empirically). Exact output parity with a non-deterministic function is
undefined, so the test seeds the global RNG, reproduces the exact Linear
the PennyLane forward will draw, loads those parameters into the Qiskit
model's persistent layer, and then compares — validating everything
around the bug while the port fixes it.

Also verified: weight initialization is **bit-identical** under the same
seed (both draw Uniform[0, 2pi) through the same torch RNG stream).

Findings that came out of parity work rather than being inputs to it:
the bit-ordering permutation, the classical_layers bug, the 1-D shape
convention, and the per-occurrence shift requirement for shared
parameters.

---

## 12. The parameter sweep: design and findings

`run_parameter_sweep.py` varies one axis at a time around a base config
(original data, 240 rows, 4 assets, lookback 5, 8 weights, seed 68),
using PennyLane QDPG as the subject (parity makes the conclusions
transfer to Qiskit; PennyLane is ~100x faster to train). Every run
records the full config, epochs completed, early-stopping trigger,
final actor/critic/validation losses, runtime, SPO/DPO performance, and
the new metrics; loss curves are saved per run.

Findings (`comparison_logs/parameter_sweep_results.csv`):

- **Epochs** (no early stopping): DPO profit 38% → 95% → 114% → 115%
  and Sharpe 0.96 → 2.31 → 2.67 → 2.76 across 10/25/50/100 epochs.
  Plateau ≈ 50; **25 epochs captures ~84% of plateau Sharpe at half the
  cost** — the benchmark setting, chosen for Qiskit affordability.
- **Early stopping** (patience 5, min_delta 1e-4 on validation critic
  loss) stops at epoch 24 but tests worse than training to 50 — the
  validation critic loss plateaus before test portfolio performance
  does. Kept as a guardrail, reported honestly.
- **Rows** (120/240/600/1768): runtime scales as expected; performance
  across this axis is confounded — changing the row count also moves
  the test window into a different market regime (flagged in the
  outline's limitations).
- **Lookback** 10 beats 5 on this window (Sharpe 4.90 vs 2.19).
- **Batch size**: `batch_size=4` fails with
  `shape [320] vs input size 80` — the original training loop flattens
  each batch into one vector, so batch training is **not implemented**
  in the original code. Documented, not fixed (architecture freeze).

---

## 13. Benchmark results

Both benchmarks: 60/20/20 split, ~48-day test window, seed 68, repo
hyperparameters, DPO = re-allocate every lookback days (RL) / 30 days
(MVO), transaction cost 10 bps one-way for the TC-adjusted column.

**Benchmark 1 — original data subset (4 assets, 240 rows, lookback 10,
25 epochs):**

| Model | Profit p.a. | Sharpe | Turnover | TC-adj | Runtime |
|---|---|---|---|---|---|
| Equal Weight | 52% | 3.38 | 0.00 | 51% | ~0s |
| Classical DDPG | 44% | 3.15 | 0.001 | 43% | 3.6s |
| Qiskit QDPG | 31% | 1.11 | 0.84 | 26% | 884s |
| PennyLane QQL | 28% | 0.94 | 0.85 | 23% | 33s |
| PennyLane QDPG | 26% | 0.84 | 0.87 | 21% | 9s |
| MVO | 23% | 0.97 | 1.26 | 20% | 0.5s |
| Qiskit QQL | 19% | 0.54 | 0.94 | 13% | 2347s |
| Classical DQL | -13% | -0.97 | 0.09 | -15% | 0.1s |

A rising-market window where 1/N is hard to beat; the quantum policies
churn (~0.85 turnover per 10-day rebalance) and pay for it.

**Benchmark 2 — custom_nonpaper_10 (10 assets, 240 rows, lookback 5;
Qiskit at 5 epochs, noted in-table):**

| Model | Sharpe | CAGR | TC-adj CAGR | Runtime |
|---|---|---|---|---|
| PennyLane QDPG | 3.64 | 30.1x | 28.3x | 15s |
| Qiskit QDPG | 3.57 | 24.2x | 22.6x | 2425s |
| PennyLane QQL | 3.55 | 32.5x | 30.5x | 34s |
| Qiskit QQL | 3.52 | 22.2x | 20.8x | 3231s |
| Classical DQL | 2.37 | 2.7x | 2.7x | 0.2s |
| Classical DDPG | 1.57 | 0.48 | 0.48 | 4.4s |
| MVO | 1.55 | 0.91 | 0.88 | 0.5s |
| Equal Weight | 1.37 | 0.33 | 0.33 | ~0s |

Quantum models dominate Sharpe here; the astronomical CAGRs are a
48-day NVDA-regime window annualized with short selling allowed —
report Sharpe as the headline, the returns as window artifacts.

**Cross-cutting observations:**

- **Per-call parity does not imply per-trajectory parity.** PennyLane
  and Qiskit agree to 1e-7 per evaluation, but after ~3,000
  noise-injected, replay-sampled updates the trained policies differ
  (Benchmark 1 Sharpe 0.84 vs 1.11). RL training is chaotic;
  float32-level differences compound. Expected, and worth a paragraph
  in the paper's discussion.
- **Runtime**: Qiskit ≈ 100x PennyLane per epoch — the price of
  hardware-realistic gradients (parameter shift + finite differences)
  and per-sample state-preparation synthesis versus PennyLane's
  simulator backprop. This is the quantitative case for the two-backend
  strategy and the planned fast_statevector work.
- **Turnover as a strategy fingerprint**: classical DDPG barely trades
  (0.001), quantum policies churn (0.84–1.48), rolling MVO churns worst
  (up to 3.07 — full portfolio flips at re-optimization).

---

## 14. Circuit export and the IonQ plan

`export_qiskit_paper_circuit.py` builds the Benchmark 1 actor circuit
(40 raw features → radial_to_linear → 160 features → 8 qubits, 16 RY
weights, reverse_linear CNOTs) and exports text and PNG diagrams at two
levels, OpenQASM 2, and `circuit_summary.md`. Two technical choices:

- The export swaps `initialize` for the **unitary** `StatePreparation`:
  `initialize` prepends a reset of every qubit (non-unitary), which is
  redundant from |0...0> and pollutes QASM/hardware translation. Same
  state, cleaner circuit.
- The decomposed form (basis rx/ry/rz/cx) has **depth 1490 with 254
  CNOTs, 765 RZ, 510 RX, 16 RY** — almost entirely the amplitude
  state-preparation cascade, not the 16-gate ansatz. This number drives
  the IonQ risk assessment: at ~99–99.5% two-qubit fidelity,
  0.99^254 ≈ 8–30% circuit fidelity, so `ionq_inference_plan.md`
  prescribes inference-only runs (5–20 frozen market states, 1000 and
  4000 shots, expectation/weight/latency/depth reporting) and lists
  mitigation options (fewer qubits, amplitude truncation, debiasing)
  rather than any hardware training.

---

## 15. Numerical gotchas and lessons learned

Recorded so nobody re-learns them the hard way:

1. **The original `price_data.parquet.gzip` contains RETURNS** (values
   in [-0.29, 0.17]) despite its name. The custom yfinance parquets
   contain PRICES and need pct_change. An earlier benchmark fed prices
   as returns and produced infinite profits and Sharpe ≈ 465 — the
   validation layer now hard-fails on NaN/inf and warns on |r| > 50%.
2. **Endianness**: PennyLane big-endian vs Qiskit little-endian
   amplitude ordering (Section 3.2) — fixed once with a precomputed
   bit-reversal permutation.
3. **Shared-parameter shifting**: shifting all occurrences of a weight
   together is *not* the parameter-shift rule; shift each gate
   occurrence separately and sum (Section 5.1).
4. **classical_layers in the original is broken** (fresh unregistered
   Linear per forward): crashes on float32, non-deterministic on
   float64, untrainable. Port fixes it; parity test pins the RNG to
   validate around it.
5. **`radial_to_linear` NaNs on shape (1, N)** inputs in both
   frameworks (std over a single batch row is 0 → 0/0). DDPG's 1-D
   states avoid it; synthetic benchmarks must too.
6. **batch_size > 1 silently unsupported** by the original trainers
   (states flattened across the batch dimension).
7. **1-D calling convention**: PennyLane returns 1-D outputs for 1-D
   inputs; the port must (and now does) match, or critic-input
   concatenation breaks.
8. **Log-scale bar charts with zero-valued bars** explode
   `bbox_inches="tight"` (a 0-height bar's text at y=0 pushes the
   bounding box toward -inf) — clip plotted values to a floor.
9. **Trained-model divergence across frameworks is not a parity
   failure** (Section 13) — distinguish per-call equivalence from
   trajectory equivalence when writing claims.

---

## 16. Why the conversion was worth it: issues surfaced, rationale, and research value

This section collects the argument for the whole exercise — what the
conversion *revealed*, why Qiskit was the right target, and what the
work contributes to the research line. It is written to be lifted
almost directly into the paper's Introduction and Discussion.

### 16.1 The conversion acted as an independent audit of the original research code

None of the following was known before the port. Every one of them is
invisible as long as the model lives in a single framework, and every
one surfaced *because* an independent reimplementation had to agree with
the original to seven decimal places:

1. **Amplitude bit-ordering (endianness).** PennyLane and Qiskit index
   basis states with opposite bit conventions. Feeding one framework's
   amplitude vector to the other produces plausible-looking but wrong
   outputs for every multi-qubit computation. Any future hardware or
   cross-framework work on the original model would have silently hit
   this; the port isolates and fixes it with a single documented
   permutation.
2. **The `classical_layers` option in the original is non-functional.**
   It constructs a fresh, randomly initialized, *unregistered*
   `nn.Linear` inside every `forward()` call: the layer is untrainable
   (its parameters never reach the optimizer), outputs are
   non-deterministic call-to-call, and float32 inputs crash outright.
   Consequence for the research line: any original result obtained with
   `classical_layers=True` measures a random projection, not a trained
   head — an important caveat when interpreting the original paper's
   ablations. The port implements the evidently intended behavior (one
   persistent trainable layer).
3. **Batch training is not actually implemented.** The original trainer
   flattens each batch into a single vector, so `batch_size>1` breaks
   on input dimensions. All original results are therefore
   window-by-window (stochastic single-sample) training — worth knowing
   when comparing to literature that batches.
4. **Dataset semantics were undocumented and dangerous.** The original
   `price_data.parquet.gzip` actually contains *returns*; naively
   extending the pipeline to new data (actual prices) produced infinite
   profits and Sharpe ratios near 465 in an early benchmark before the
   conversion's validation layer (NaN/inf hard-fails, |r| > 50%
   warnings) caught it. The rebuilt Benchmark 2 is trustworthy
   precisely because of this.
5. **`radial_to_linear` NaNs on batch-of-one 2-D inputs** in both
   frameworks (standardize over a single row divides 0 by 0) — a
   latent trap for anyone batching evaluation.
6. **Shared-parameter gradients are subtle under hardware rules.** With
   multi-axis ansatze (e.g. `rotation_axes="xy"`), one weight drives
   several gates. Simulator backprop (the original's path) never
   notices; the correct hardware-compatible gradient requires shifting
   each gate occurrence separately and summing (chain rule). The port
   implements and *verifies* this against autodiff to ~4e-7 — a
   correctness result the original setup could not even express.
7. **Calling-convention fragility.** The RL stack depends on the model
   returning 1-D outputs for 1-D state inputs; an implementation that
   silently promotes to batches breaks the critic's tensor plumbing.
   Now pinned by a dedicated parity case.

The reproducibility literature calls this pattern out: results that
exist only inside one codebase inherit that codebase's bugs. Items 2–4
directly sharpen how the original results should be interpreted; items
1 and 6 are the exact traps a hardware transition would have fallen
into. Surfacing them is a research contribution independent of any
benchmark number.

### 16.2 Why Qiskit specifically

Stated honestly (and pre-empting the obvious reviewer question): IonQ
can be reached from PennyLane directly (`pennylane-ionq`, or via AWS
Braket), so hardware *access* was not the reason. The reasons were:

1. **Framework-independent validation.** Reproducing forward passes and
   gradients to ~1e-7 in an independently written implementation is
   the strongest available evidence that the reported behavior is a
   property of the *model*, not of PennyLane's conventions, autodiff,
   or simulator internals.
2. **Hardware-realistic training semantics.** The original trains by
   backpropagating through the simulator's linear algebra — a technique
   that does not exist on any QPU. The Qiskit port trains the same
   model with parameter-shift and finite-difference gradients, the only
   rules available on hardware, and shows learning still works and
   what it costs (~100x runtime). That converts "this trains on a
   simulator" into "this would train under hardware rules" — a
   materially stronger claim for the research line.
3. **The hardware-facing toolchain.** Qiskit's transpiler (native-gate
   targeting, optimization levels), OpenQASM 2 export, Aer noise
   models, and resource estimation produce auditable artifacts — the
   exported circuit's depth (1490) and CNOT count (254) are what turn
   the IonQ plan from hand-waving into a costed protocol with a
   quantified fidelity risk (0.99^254 ≈ 8–30%).
4. **Vendor portability and audience.** The same QASM/circuit artifacts
   target IBM, IonQ, and Azure Quantum backends, and Qiskit is the most
   widely adopted stack — maximizing reuse of the group's work.

### 16.3 How this proves out the research line

- **The original claim is strengthened.** The QRL portfolio model's
  behavior now rests on two independent implementations that agree to
  numerical precision, with a 30-case regression suite that any future
  architecture change can be re-verified against. The research line
  gains a permanent correctness harness, not just a one-off check.
- **Generalization beyond the original universe.** Benchmark 2 tests
  the pipeline on ten tickers the original work never touched
  (zero-overlap by construction) with honest data validation — evidence
  the approach is not an artifact of one dataset, and a template for
  future out-of-universe tests.
- **A more complete economic evaluation.** The five added metrics
  expose what profit/Sharpe alone hide: the quantum policies' high
  turnover (~0.85–1.5 per rebalance) and its transaction-cost drag
  (~2–3pp annually at 10 bps), versus near-zero-turnover classical
  DDPG. Cost-aware reward shaping becomes an obvious, well-motivated
  next step for the line.
- **Feasibility is now quantified, not asserted.** The 100x
  simulator-vs-hardware-rules training gap, the state-preparation-
  dominated circuit depth, and the sweep's epoch plateau give the group
  concrete numbers for planning: what to train where (PennyLane for
  sweeps, Qiskit for verification), what to run on hardware
  (inference only), and what to optimize next (the fast_statevector
  backend, amplitude-encoding compression).
- **A publishable observation fell out for free.** Per-call parity does
  not imply per-trajectory parity: two numerically equivalent
  implementations, identically seeded, diverge after thousands of
  noise-injected RL updates (Benchmark 1 Sharpe 0.84 vs 1.11). This is
  a clean, demonstrable statement about the chaotic sensitivity of RL
  training with quantum function approximators — relevant to anyone
  claiming exact reproducibility in QRL.
- **The hardware chapter is de-risked before spending a dollar.** The
  IonQ plan proceeds from frozen, parity-verified parameters with known
  circuit costs and a shot-noise budget, so the eventual hardware
  results will be attributable to hardware — not to software ambiguity.

## 17. How to reproduce everything

All commands from the repo root (never the VS Code play button — the
interpreter and PYTHONPATH matter):

```bash
PY=original_pennylane/qrl-dpo-public/.venv/bin/python

# Parity (30 cases, ~7 min; the 20-qubit angle cases dominate)
PYTHONPATH=. $PY -u qiskit_port/test_full_pennylane_qiskit_parity.py

# Quick parity checks
PYTHONPATH=. $PY -u qiskit_port/compare_pennylane_qiskit_forward.py
PYTHONPATH=. $PY -u qiskit_port/compare_pennylane_qiskit_gradients.py
PYTHONPATH=. $PY -u qiskit_port/compare_initial_qnn_weights.py

# Dataset (network required)
$PY -u qiskit_port/download_custom_nonpaper_10_data.py

# Parameter sweep (~4 min, PennyLane)
PYTHONPATH=. $PY -u qiskit_port/run_parameter_sweep.py

# Benchmarks (resumable; Qiskit models dominate runtime:
#  Benchmark 1 ~55 min total, Benchmark 2 ~95 min total)
PYTHONPATH=. $PY -u qiskit_port/run_real_data_benchmark.py
PYTHONPATH=. $PY -u qiskit_port/run_custom_nonpaper_10_benchmark.py

# Circuit exports
PYTHONPATH=. $PY -u qiskit_port/export_qiskit_paper_circuit.py

# All figures + new-metric aggregate tables
PYTHONPATH=. $PY -u qiskit_port/generate_poster_plots.py
```

Determinism notes: everything is seeded (seed 68); classical and
PennyLane results reproduce exactly; Qiskit results reproduce exactly
given the same library versions; retraining after interruption resumes
from cached per-model rows (delete `comparison_logs/series/*_row.json`
to force full reruns).

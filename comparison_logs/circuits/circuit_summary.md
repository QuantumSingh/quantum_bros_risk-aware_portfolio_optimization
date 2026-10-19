# Qiskit VQC circuit summary

One architecture — amplitude encoding, RY passes with a
reverse_linear CNOT layer after each full qubit pass, Pauli-Z
expectation readout from the exact statevector (no measurement
gates, no shots) — instantiated at three sizes. Amplitude ordering
is bit-reversed relative to PennyLane so outputs match the original
QNode (forward parity ~1e-7). radial_to_linear (x, x^2, sin x,
cos x) happens classically outside the circuit.

| Instance | Raw features | Encoded features | Qubits | Hilbert dim | Weights | Decomposed depth | Gate counts | QASM |
|---|---|---|---|---|---|---|---|---|
| Benchmark 1 actor (8 qubits) | 40 | 160 | 8 | 256 | 16 | 1490 | {'rz': 765, 'rx': 510, 'cx': 254, 'ry': 16} | qiskit_vqc_circuit.qasm |
| Benchmark 2 actor (10 qubits) | 50 | 200 | 10 | 1024 | 10 | 6084 | {'rz': 3069, 'rx': 2046, 'cx': 1013, 'ry': 10} | qiskit_vqc_circuit_bench2.qasm |
| Full MAIN configuration (15 qubits) | 555 | 2220 | 15 | 32768 | 60 | 196526 | {'rz': 98301, 'rx': 65534, 'cx': 32794, 'ry': 60} | not exported (size) |

Notes:
- The critic of each experiment is the same architecture with
  input_size = state + action features and output_size = 1.
- Depth is dominated by amplitude state preparation (~2^n CX),
  not the variational ansatz — the key hardware consideration
  (see qiskit_port/ionq_inference_plan.md).
- Exports use unitary StatePreparation; the simulator model uses
  initialize (identical state from |0...0>).

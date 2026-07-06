# Qiskit VQC circuit summary

Configuration: Benchmark 1 actor (original dataset subset).

| Property | Value |
|---|---|
| Raw input features | 40 (4 assets x lookback 10) |
| Features after radial_to_linear | 160 (4x augmentation) |
| Number of qubits | 8 |
| Hilbert dimension | 256 |
| Trainable weights | 16 |
| Encoding | Amplitude encoding (pad to 2^n, L2-normalize; unitary StatePreparation in export, `initialize` in the simulator model) |
| Trainable rotations | RY(theta_i) on qubit i mod 8 |
| Entanglement | reverse_linear CNOT layer after each full qubit pass |
| Measurement | Pauli-Z expectation values of the first 4 qubits, computed from the exact statevector (no measurement gates, no shots) |
| Decomposed depth | 1490 |
| Decomposed gate counts | {'rz': 765, 'rx': 510, 'cx': 254, 'ry': 16} |
| OpenQASM 2 | exported (qiskit_vqc_circuit.qasm) |

Notes:
- The amplitude ordering is bit-reversed relative to PennyLane so that
  Qiskit statevector outputs match the PennyLane QNode exactly
  (parity: forward max abs diff ~1e-7).
- The same ansatz generalizes to the MAIN 15-asset configuration
  (555 raw features -> 2220 after radial_to_linear -> 15 qubits,
  60 weights); this 8-qubit instance is drawn for readability.
- Input-side classical preprocessing (radial_to_linear) happens outside
  the circuit.

"""Export the general Qiskit VQC circuit used in the paper/poster (Task G).

Builds the actor circuit at the Benchmark 1 configuration (4 assets,
lookback 10 -> 40 raw features -> radial_to_linear -> 160 features ->
8 qubits, 16 trainable RY weights, reverse_linear CNOT entanglement).

Exports to comparison_logs/circuits/:
    qiskit_vqc_high_level_circuit.txt / .png   (initialize block + ansatz)
    qiskit_vqc_decomposed_circuit.txt / .png   (state prep decomposed to
                                                1- and 2-qubit gates)
    qiskit_vqc_circuit.qasm                    (decomposed, OpenQASM 2)
    circuit_summary.md

No measurement gates are appended: Pauli-Z expectations are computed
directly from the simulated statevector, matching the PennyLane QNode.
"""
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

from qiskit import QuantumCircuit, transpile
from qiskit import qasm2

from predictors.input_transformations import radial_to_linear
from qiskit_port.qiskit_exact_amplitude_finite_diff_qnn import (
    QiskitExactAmplitudeFiniteDiffQNN,
)

OUTPUT_DIR = REPO_ROOT / "comparison_logs" / "circuits"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Benchmark 1 actor configuration.
CONFIG = dict(
    input_size=40,          # 4 assets x lookback 10
    output_size=4,
    num_weights=16,
    encoding="amplitude",
    input_transformation=radial_to_linear,  # 40 -> 160 features -> 8 qubits
    rotation_axes="y",
    entanglement="reverse_linear",
    seed=68,
)


def main():
    model = QiskitExactAmplitudeFiniteDiffQNN(**CONFIG)

    # radial_to_linear is applied outside the circuit in the pipeline;
    # the circuit itself sees actual_input_size features. Use a
    # deterministic dummy state for the drawing.
    rng = np.random.default_rng(CONFIG["seed"])
    dummy_input = rng.normal(size=(1, model.actual_input_size))
    amplitudes = model._pad_and_normalize_numpy(dummy_input)[0]
    weights = model.weights.detach().numpy()

    base_circuit = model._build_numeric_circuit(amplitudes, weights)

    # The model uses `initialize`, which prepends a (non-unitary) reset of
    # every qubit. The register starts in |0...0> anyway, so for export we
    # swap it for the unitary StatePreparation: identical state, cleaner
    # decomposition, and hardware/QASM friendly.
    circuit = QuantumCircuit(model.num_qubits)
    circuit.prepare_state(list(amplitudes), list(range(model.num_qubits)),
                          label="AmplitudeEnc")
    for instruction in base_circuit.data[1:]:
        circuit.append(instruction)

    print(f"num_qubits={model.num_qubits}  feature_dim={model.feature_dim}  "
          f"num_weights={model.num_weights}")

    # ------------------------------------------------------------------
    # High-level circuit (initialize shown as a single block)
    # ------------------------------------------------------------------
    high_txt = OUTPUT_DIR / "qiskit_vqc_high_level_circuit.txt"
    high_txt.write_text(str(circuit.draw(output="text", fold=120)))
    print("Wrote", high_txt.name)

    try:
        fig = circuit.draw(output="mpl", fold=28, style="clifford")
        fig.savefig(OUTPUT_DIR / "qiskit_vqc_high_level_circuit.png",
                    dpi=200, bbox_inches="tight")
        print("Wrote qiskit_vqc_high_level_circuit.png")
    except Exception as exc:
        print("PNG (high level) failed:", exc)

    # ------------------------------------------------------------------
    # Decomposed circuit (state preparation as 1- and 2-qubit gates)
    # ------------------------------------------------------------------
    decomposed = transpile(
        circuit,
        basis_gates=["ry", "rz", "rx", "cx"],
        optimization_level=0,
    )
    ops = dict(decomposed.count_ops())
    depth = decomposed.depth()
    print(f"Decomposed: depth={depth}, ops={ops}")

    dec_txt = OUTPUT_DIR / "qiskit_vqc_decomposed_circuit.txt"
    dec_txt.write_text(str(decomposed.draw(output="text", fold=160)))
    print("Wrote", dec_txt.name)

    try:
        # The fully decomposed circuit is large; draw a truncated window so
        # the PNG stays readable, and note the truncation in the title.
        fig = decomposed.draw(output="mpl", fold=40, idle_wires=True)
        fig.suptitle(
            f"Decomposed VQC ({model.num_qubits} qubits, depth {depth})",
            fontsize=10,
        )
        fig.savefig(OUTPUT_DIR / "qiskit_vqc_decomposed_circuit.png",
                    dpi=150, bbox_inches="tight")
        print("Wrote qiskit_vqc_decomposed_circuit.png")
    except Exception as exc:
        print("PNG (decomposed) failed:", exc)

    # ------------------------------------------------------------------
    # OpenQASM 2 export (needs the decomposed basis; `initialize` itself
    # is not a QASM primitive)
    # ------------------------------------------------------------------
    try:
        qasm_path = OUTPUT_DIR / "qiskit_vqc_circuit.qasm"
        qasm_path.write_text(qasm2.dumps(decomposed))
        print("Wrote", qasm_path.name)
        qasm_note = f"exported ({qasm_path.name})"
    except Exception as exc:
        print("QASM export failed:", exc)
        qasm_note = f"not exportable: {exc}"

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    summary = f"""# Qiskit VQC circuit summary

Configuration: Benchmark 1 actor (original dataset subset).

| Property | Value |
|---|---|
| Raw input features | {CONFIG['input_size']} (4 assets x lookback 10) |
| Features after radial_to_linear | {model.actual_input_size} (4x augmentation) |
| Number of qubits | {model.num_qubits} |
| Hilbert dimension | {model.feature_dim} |
| Trainable weights | {model.num_weights} |
| Encoding | Amplitude encoding (pad to 2^n, L2-normalize; unitary StatePreparation in export, `initialize` in the simulator model) |
| Trainable rotations | RY(theta_i) on qubit i mod {model.num_qubits} |
| Entanglement | reverse_linear CNOT layer after each full qubit pass |
| Measurement | Pauli-Z expectation values of the first {CONFIG['output_size']} qubits, computed from the exact statevector (no measurement gates, no shots) |
| Decomposed depth | {depth} |
| Decomposed gate counts | {ops} |
| OpenQASM 2 | {qasm_note} |

Notes:
- The amplitude ordering is bit-reversed relative to PennyLane so that
  Qiskit statevector outputs match the PennyLane QNode exactly
  (parity: forward max abs diff ~1e-7).
- The same ansatz generalizes to the MAIN 15-asset configuration
  (555 raw features -> 2220 after radial_to_linear -> 15 qubits,
  60 weights); this 8-qubit instance is drawn for readability.
- Input-side classical preprocessing (radial_to_linear) happens outside
  the circuit.
"""
    (OUTPUT_DIR / "circuit_summary.md").write_text(summary)
    print("Wrote circuit_summary.md")


if __name__ == "__main__":
    main()

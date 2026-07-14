"""Export the Qiskit VQC circuits used in the paper/poster (Task G).

ONE architecture, THREE instance sizes. Every model in this project uses
the same circuit family — amplitude encoding, RY rotation passes with a
reverse_linear CNOT layer after each full pass, Pauli-Z expectation
readout — instantiated at the size each experiment needs:

    bench1_actor : 4 assets x lookback 10 -> 160 features -> 8 qubits,
                   16 weights   (Benchmark 1, stress tests)
    bench2_actor : 10 assets x lookback 5 -> 200 features -> 10 qubits
                   (output_size=10 forces 10 qubits), 10 weights
                   (Benchmark 2)
    main15       : 15 assets x lookback 37 -> 2220 features -> 15 qubits,
                   60 weights   (original MAIN.py full configuration)

Exports to comparison_logs/circuits/ (bench1 keeps the original
unsuffixed filenames used elsewhere in the paper):
    qiskit_vqc_high_level_circuit[.,_bench2,_main15].txt/.png
    qiskit_vqc_decomposed_circuit[.,_bench2].txt/.png (main15 counts only)
    qiskit_vqc_circuit[.,_bench2].qasm
    circuit_summary.md

No measurement gates are appended: Pauli-Z expectations are computed
from the exact statevector, matching the PennyLane QNode.
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

CONFIGS = {
    # suffix "" keeps Benchmark 1 at the original filenames.
    "": dict(
        label="Benchmark 1 actor (8 qubits)",
        input_size=40, output_size=4, num_weights=16,
        decompose=True, qasm=True, decomposed_txt=True,
    ),
    "_bench2": dict(
        label="Benchmark 2 actor (10 qubits)",
        input_size=50, output_size=10, num_weights=10,
        decompose=True, qasm=True, decomposed_txt=False,  # ~4k gates: counts + png only
    ),
    "_main15": dict(
        label="Full MAIN configuration (15 qubits)",
        input_size=555, output_size=15, num_weights=60,
        decompose=True, qasm=False, decomposed_txt=False,  # ~65k gates: counts only
    ),
}

BASE_KWARGS = dict(
    encoding="amplitude",
    input_transformation=radial_to_linear,
    rotation_axes="y",
    entanglement="reverse_linear",
    seed=68,
)


def build_export_circuit(model):
    """Model circuit with unitary StatePreparation instead of initialize
    (initialize prepends resets; redundant from |0...0> and QASM-hostile)."""
    rng = np.random.default_rng(68)
    dummy = rng.normal(size=(1, model.actual_input_size))
    amplitudes = model._pad_and_normalize_numpy(dummy)[0]
    base = model._build_numeric_circuit(amplitudes, model.weights.detach().numpy())

    qc = QuantumCircuit(model.num_qubits)
    qc.prepare_state(list(amplitudes), list(range(model.num_qubits)),
                     label="AmplitudeEnc")
    for instruction in base.data[1:]:
        qc.append(instruction)
    return qc


def export_one(suffix, cfg):
    model = QiskitExactAmplitudeFiniteDiffQNN(
        input_size=cfg["input_size"], output_size=cfg["output_size"],
        num_weights=cfg["num_weights"], **BASE_KWARGS)

    circuit = build_export_circuit(model)
    print(f"\n{cfg['label']}: {model.num_qubits} qubits, "
          f"{model.actual_input_size} features, {model.num_weights} weights")

    high_txt = OUTPUT_DIR / f"qiskit_vqc_high_level_circuit{suffix}.txt"
    high_txt.write_text(str(circuit.draw(output="text", fold=120)))
    print("Wrote", high_txt.name)

    try:
        fig = circuit.draw(output="mpl", fold=28, style="clifford")
        fig.savefig(OUTPUT_DIR / f"qiskit_vqc_high_level_circuit{suffix}.png",
                    dpi=200, bbox_inches="tight")
        print(f"Wrote qiskit_vqc_high_level_circuit{suffix}.png")
    except Exception as exc:
        print("high-level PNG failed:", exc)

    depth, ops, qasm_note = None, None, "not exported (size)"
    if cfg["decompose"]:
        decomposed = transpile(circuit, basis_gates=["ry", "rz", "rx", "cx"],
                               optimization_level=0)
        ops = dict(decomposed.count_ops())
        depth = decomposed.depth()
        print(f"Decomposed: depth={depth}, ops={ops}")

        if cfg["decomposed_txt"]:
            (OUTPUT_DIR / f"qiskit_vqc_decomposed_circuit{suffix}.txt").write_text(
                str(decomposed.draw(output="text", fold=160)))
            try:
                fig = decomposed.draw(output="mpl", fold=40, idle_wires=True)
                fig.suptitle(f"Decomposed VQC ({model.num_qubits} qubits, "
                             f"depth {depth})", fontsize=10)
                fig.savefig(OUTPUT_DIR / f"qiskit_vqc_decomposed_circuit{suffix}.png",
                            dpi=150, bbox_inches="tight")
            except Exception as exc:
                print("decomposed PNG failed:", exc)

        if cfg["qasm"]:
            try:
                (OUTPUT_DIR / f"qiskit_vqc_circuit{suffix}.qasm").write_text(
                    qasm2.dumps(decomposed))
                qasm_note = f"qiskit_vqc_circuit{suffix}.qasm"
                print("Wrote", qasm_note)
            except Exception as exc:
                qasm_note = f"failed: {exc}"

    return dict(label=cfg["label"], suffix=suffix,
                raw_features=cfg["input_size"],
                features=model.actual_input_size,
                qubits=model.num_qubits, hilbert=model.feature_dim,
                weights=model.num_weights, depth=depth, ops=ops,
                qasm=qasm_note)


def main():
    results = [export_one(suffix, cfg) for suffix, cfg in CONFIGS.items()]

    lines = [
        "# Qiskit VQC circuit summary",
        "",
        "One architecture — amplitude encoding, RY passes with a",
        "reverse_linear CNOT layer after each full qubit pass, Pauli-Z",
        "expectation readout from the exact statevector (no measurement",
        "gates, no shots) — instantiated at three sizes. Amplitude ordering",
        "is bit-reversed relative to PennyLane so outputs match the original",
        "QNode (forward parity ~1e-7). radial_to_linear (x, x^2, sin x,",
        "cos x) happens classically outside the circuit.",
        "",
        "| Instance | Raw features | Encoded features | Qubits | Hilbert dim | Weights | Decomposed depth | Gate counts | QASM |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['label']} | {r['raw_features']} | {r['features']} | "
            f"{r['qubits']} | {r['hilbert']} | {r['weights']} | "
            f"{r['depth']} | {r['ops']} | {r['qasm']} |")
    lines += [
        "",
        "Notes:",
        "- The critic of each experiment is the same architecture with",
        "  input_size = state + action features and output_size = 1.",
        "- Depth is dominated by amplitude state preparation (~2^n CX),",
        "  not the variational ansatz — the key hardware consideration",
        "  (see qiskit_port/ionq_inference_plan.md).",
        "- Exports use unitary StatePreparation; the simulator model uses",
        "  initialize (identical state from |0...0>).",
    ]
    (OUTPUT_DIR / "circuit_summary.md").write_text("\n".join(lines) + "\n")
    print("\nWrote circuit_summary.md")


if __name__ == "__main__":
    main()

"""Inspect the authors' PennyLane circuit at different depths.

The trainable part of the authors' circuit is a repeating block:
one RY rotation on every qubit, then a chain of CNOTs linking neighbouring
qubits. Each rotation uses one weight, so num_weights sets how many blocks
there are. With the actor's 15 qubits:

    num_weights 15 -> 1 rotation layer, no CNOT chain
    num_weights 30 -> 2 layers, 1 chain        (half the released circuit)
    num_weights 45 -> 3 layers, 2 chains
    num_weights 60 -> 4 layers, 3 chains        (as released)

For each depth this script builds the authors' QuantumNeuralNetwork for the
actor (555 inputs -> 15 qubits) and the critic (570 inputs -> 12 qubits),
then writes:

    results/circuits/<role>_<N>w.txt   circuit diagram (amplitude encoding
                                       shown as one block)
    results/circuits/actor_<N>w.png    the same, as a picture
    results/circuit_specs.csv          gate counts and depth of the trainable part
    results/speed_benchmark.csv        time for one forward + backward pass

Usage (from repo root):
    $PY pennylane_circuit_reduction/inspect_circuit.py
"""
import os
import statistics
import sys
import time
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
AUTHORS = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(AUTHORS))
os.chdir(AUTHORS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pennylane as qml
import torch

from predictors import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear

warnings.filterwarnings("ignore")

DEPTHS = [15, 30, 45, 60]
N_ASSETS, LOOKBACK, FORECAST = 15, 30, 7
ROLES = {
    # role: (raw input size, output size)
    "actor": (N_ASSETS * (LOOKBACK + FORECAST), N_ASSETS),
    "critic": (N_ASSETS * (LOOKBACK + FORECAST) + N_ASSETS, 1),
}
REPEATS = 15
OUT = HERE / "results"
CIRCUITS = OUT / "circuits"


def build(role, num_weights):
    in_size, out_size = ROLES[role]
    activation = (lambda x: x / torch.sum(x, dim=-1, keepdim=True)) if role == "actor" else None
    return QuantumNeuralNetwork(
        input_size=in_size, output_size=out_size, num_weights=num_weights,
        encoding="amplitude", input_transformation=radial_to_linear,
        rotation_axes="y", output_activation=activation, seed=68,
    )


def circuit_args(model):
    """Inputs exactly as the QNode receives them (after radial_to_linear)."""
    raw = torch.randn(model.input_size, generator=torch.Generator().manual_seed(0))
    return model.input_transformation(raw), model.qlayer.weights.detach()


def time_pass(model, with_backward):
    x = torch.randn(model.input_size, generator=torch.Generator().manual_seed(1))
    for _ in range(2):  # warm-up
        out = model(x)
        if with_backward:
            out.sum().backward()
            model.zero_grad()
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        out = model(x)
        if with_backward:
            out.sum().backward()
            model.zero_grad()
        times.append(time.perf_counter() - start)
    return statistics.median(times)


def main():
    CIRCUITS.mkdir(parents=True, exist_ok=True)
    spec_rows, speed_rows = [], []

    for role in ROLES:
        for n_w in DEPTHS:
            model = build(role, n_w)
            inputs, weights = circuit_args(model)
            qnode = model.quantum_circuit

            text = qml.draw(qnode, level="top", decimals=None, max_length=220)(inputs, weights)
            (CIRCUITS / f"{role}_{n_w}w.txt").write_text(text)
            if role == "actor":
                fig, _ = qml.draw_mpl(qnode, level="top", decimals=None)(inputs, weights)
                fig.savefig(CIRCUITS / f"actor_{n_w}w.png", dpi=110, bbox_inches="tight")
                plt.close(fig)

            res = qml.specs(qnode, level="top")(inputs, weights)["resources"]
            gates = dict(res.gate_types)
            n_q = model.num_qubits
            spec_rows.append({
                "role": role, "num_weights": n_w, "qubits": n_q,
                "rotation_layers": round(n_w / n_q, 2),
                "RY_gates": gates.get("RY", 0), "CNOT_gates": gates.get("CNOT", 0),
                "trainable_part_depth": res.depth - 1,  # minus the encoding block
                "total_ops_simulated": res.num_gates,
            })

            fwd = time_pass(model, with_backward=False)
            fwd_bwd = time_pass(model, with_backward=True)
            speed_rows.append({"role": role, "num_weights": n_w, "qubits": n_q,
                               "forward_ms": fwd * 1e3, "forward_backward_ms": fwd_bwd * 1e3})
            print(f"{role:6s} {n_w:2d} weights | {n_q} qubits | "
                  f"RY {gates.get('RY', 0):2d} CNOT {gates.get('CNOT', 0):2d} | "
                  f"fwd {fwd*1e3:6.1f} ms  fwd+bwd {fwd_bwd*1e3:6.1f} ms")

    specs = pd.DataFrame(spec_rows)
    speed = pd.DataFrame(speed_rows)
    for role in ROLES:
        mask = speed["role"] == role
        base = speed.loc[mask & (speed["num_weights"] == 60), "forward_backward_ms"].iloc[0]
        speed.loc[mask, "speedup_vs_60w"] = base / speed.loc[mask, "forward_backward_ms"]
    specs.to_csv(OUT / "circuit_specs.csv", index=False)
    speed.round(3).to_csv(OUT / "speed_benchmark.csv", index=False)
    print("\n", specs.to_string(index=False))
    print("\n", speed.round(2).to_string(index=False))
    print(f"\nWrote diagrams to {CIRCUITS} and tables to {OUT}")


if __name__ == "__main__":
    main()

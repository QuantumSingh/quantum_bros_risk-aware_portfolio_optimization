"""Speed benchmark: PennyLane QuantumNeuralNetwork vs Qiskit port.

Times forward and forward+backward on two configs:
  - the parity-test base config (5 qubits, 8 weights)
  - the MAIN.py QDPG actor config (15 assets x 37 days = 555 inputs,
    radial_to_linear -> 2220 features -> 15 qubits, 60 weights)

Run:
    PYTHONPATH=. original_pennylane/qrl-dpo-public/.venv/bin/python -u \
        qiskit_port/benchmark_pennylane_vs_qiskit_speed.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

from predictors.quantum_neural_network import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
from qiskit_port.qiskit_compatible_qnn import QiskitQuantumNeuralNetwork


def time_forward(model, x, reps):
    with torch.no_grad():
        model(x)  # warmup
        start = time.perf_counter()
        for _ in range(reps):
            model(x)
    return (time.perf_counter() - start) / reps


def time_forward_backward(model, x, reps):
    start = time.perf_counter()
    for _ in range(reps):
        model.zero_grad(set_to_none=True)
        y = model(x)
        y.sum().backward()
    return (time.perf_counter() - start) / reps


def bench(label, reps_fwd, reps_bwd, single_state=False, **kwargs):
    print(f"\n--- {label} ---")

    torch.manual_seed(68)
    np.random.seed(68)
    pl_model = QuantumNeuralNetwork(**kwargs, seed=68)
    q_model = QiskitQuantumNeuralNetwork(**kwargs, seed=68)

    print(f"num_qubits={q_model.num_qubits}  num_weights={q_model.num_weights}")

    # DDPG feeds single states as 1-D vectors. Shape (1, N) would also
    # break radial_to_linear, whose standardize() runs over dim=0.
    if single_state:
        x = torch.randn(kwargs["input_size"])
    else:
        x = torch.randn(1, kwargs["input_size"])

    for name, model in [("PennyLane", pl_model), ("Qiskit", q_model)]:
        t_fwd = time_forward(model, x, reps_fwd)
        t_bwd = time_forward_backward(model, x, reps_bwd)
        print(f"{name:>10}: forward {t_fwd * 1e3:9.1f} ms   fwd+bwd {t_bwd * 1e3:9.1f} ms")


def main():
    bench(
        "parity base config (batch 1)",
        reps_fwd=10,
        reps_bwd=3,
        input_size=20,
        output_size=4,
        num_weights=8,
        encoding="amplitude",
        rotation_axes="y",
        entanglement="reverse_linear",
    )

    bench(
        "MAIN QDPG actor config (single 1-D state)",
        reps_fwd=3,
        reps_bwd=1,
        single_state=True,
        input_size=555,
        output_size=15,
        num_weights=60,
        encoding="amplitude",
        input_transformation=radial_to_linear,
        rotation_axes="y",
        entanglement="reverse_linear",
    )


if __name__ == "__main__":
    main()

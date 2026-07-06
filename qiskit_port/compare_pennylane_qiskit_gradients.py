import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

import torch
import numpy as np

from predictors.quantum_neural_network import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
from qiskit_port.qiskit_exact_amplitude_finite_diff_qnn import (
    QiskitExactAmplitudeFiniteDiffQNN,
)


def get_pl_weight_param(pl_model):
    for name, param in pl_model.named_parameters():
        if "weights" in name:
            return name, param
    raise RuntimeError("Could not find PennyLane weights.")


def run_test(use_radial):
    print("\n" + "=" * 80)
    print("Gradient parity test")
    print("use_radial:", use_radial)
    print("=" * 80)

    torch.manual_seed(68)
    np.random.seed(68)

    input_size = 5 if use_radial else 20
    output_size = 4
    num_weights = 8
    input_transformation = radial_to_linear if use_radial else None

    pl_model = QuantumNeuralNetwork(
        input_size=input_size,
        output_size=output_size,
        num_weights=num_weights,
        encoding="amplitude",
        input_transformation=input_transformation,
        rotation_axes="y",
        entanglement="reverse_linear",
        seed=68,
    )

    q_model = QiskitExactAmplitudeFiniteDiffQNN(
        input_size=input_size,
        output_size=output_size,
        num_weights=num_weights,
        encoding="amplitude",
        input_transformation=input_transformation,
        rotation_axes="y",
        entanglement="reverse_linear",
        seed=68,
        compute_input_gradients=True,
    )

    pl_weight_name, pl_weight = get_pl_weight_param(pl_model)

    with torch.no_grad():
        q_model.weights.copy_(pl_weight.detach().clone().float())

    x_base = torch.randn(2, input_size)

    x_pl = x_base.clone().detach().requires_grad_(True)
    x_q = x_base.clone().detach().requires_grad_(True)

    pl_model.zero_grad(set_to_none=True)
    q_model.zero_grad(set_to_none=True)

    y_pl = pl_model(x_pl)
    y_q = q_model(x_q)

    loss_pl = y_pl.sum()
    loss_q = y_q.sum()

    loss_pl.backward()
    loss_q.backward()

    print("Output max abs diff:", (y_pl.float() - y_q.float()).abs().max().item())

    print("\nPennyLane output:")
    print(y_pl)

    print("\nQiskit output:")
    print(y_q)

    print("\nInput grad max abs diff:")
    print((x_pl.grad.float() - x_q.grad.float()).abs().max().item())

    print("\nPennyLane input grad:")
    print(x_pl.grad)

    print("\nQiskit input grad:")
    print(x_q.grad)

    print("\nWeight grad max abs diff:")
    print((pl_weight.grad.float() - q_model.weights.grad.float()).abs().max().item())

    print("\nPennyLane weight grad:")
    print(pl_weight.grad)

    print("\nQiskit weight grad:")
    print(q_model.weights.grad)


if __name__ == "__main__":
    run_test(use_radial=False)
    run_test(use_radial=True)

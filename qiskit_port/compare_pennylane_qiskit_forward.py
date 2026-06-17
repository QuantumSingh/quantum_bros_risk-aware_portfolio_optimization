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


def copy_weights_from_pennylane_to_qiskit(pl_model, q_model):
    print("\nPennyLane parameters:")
    for name, param in pl_model.named_parameters():
        print(name, param.shape)

    print("\nQiskit parameters:")
    for name, param in q_model.named_parameters():
        print(name, param.shape)

    pl_weight = None
    for name, param in pl_model.named_parameters():
        if "weights" in name:
            pl_weight = param.detach().clone().float()
            print("\nUsing PennyLane weight parameter:", name)
            break

    if pl_weight is None:
        raise RuntimeError("Could not find PennyLane weights.")

    with torch.no_grad():
        q_model.weights.copy_(pl_weight)

    return pl_weight


def run_test(use_radial):
    print("\n" + "=" * 80)
    print("Forward parity test")
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
    )

    copy_weights_from_pennylane_to_qiskit(pl_model, q_model)

    x = torch.randn(2, input_size)

    with torch.no_grad():
        y_pl = pl_model(x)
        y_q = q_model(x)

    print("\nInput shape:", x.shape)
    print("PennyLane output:")
    print(y_pl)

    print("\nQiskit output:")
    print(y_q)

    diff = y_pl.float() - y_q.float()

    print("\nDifference:")
    print(diff)

    print("\nMax abs diff:", diff.abs().max().item())
    print("Mean abs diff:", diff.abs().mean().item())


if __name__ == "__main__":
    run_test(use_radial=False)
    run_test(use_radial=True)

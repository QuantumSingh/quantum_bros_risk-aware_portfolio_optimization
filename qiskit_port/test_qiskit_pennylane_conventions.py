import sys
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

from predictors.quantum_neural_network import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
from qiskit_port.qiskit_exact_amplitude_finite_diff_qnn import (
    QiskitExactAmplitudeFiniteDiffQNN,
)


def bit_reverse_index(i, n):
    out = 0
    for _ in range(n):
        out = (out << 1) | (i & 1)
        i >>= 1
    return out


class ConventionQNN(QiskitExactAmplitudeFiniteDiffQNN):
    def __init__(self, *args, reverse_amplitudes=False, reverse_measure=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse_amplitudes = reverse_amplitudes
        self.reverse_measure = reverse_measure

    def _pad_and_normalize_numpy(self, x):
        padded = super()._pad_and_normalize_numpy(x)

        if not self.reverse_amplitudes:
            return padded

        out = np.zeros_like(padded)
        for i in range(self.feature_dim):
            out[:, bit_reverse_index(i, self.num_qubits)] = padded[:, i]
        return out

    def _z_expectation(self, state, qubit):
        probs = np.abs(state.data) ** 2

        expval = 0.0
        for basis_index, prob in enumerate(probs):
            if self.reverse_measure:
                bit_position = self.num_qubits - 1 - qubit
            else:
                bit_position = qubit

            bit = (basis_index >> bit_position) & 1
            expval += prob * (1.0 if bit == 0 else -1.0)

        return expval


def copy_weights(pl_model, q_model):
    for name, param in pl_model.named_parameters():
        if "weights" in name:
            with torch.no_grad():
                q_model.weights.copy_(param.detach().clone().float())
            return
    raise RuntimeError("Could not find PennyLane weights.")


def run_one(use_radial, reverse_amplitudes, reverse_measure):
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

    q_model = ConventionQNN(
        input_size=input_size,
        output_size=output_size,
        num_weights=num_weights,
        encoding="amplitude",
        input_transformation=input_transformation,
        rotation_axes="y",
        entanglement="reverse_linear",
        seed=68,
        reverse_amplitudes=reverse_amplitudes,
        reverse_measure=reverse_measure,
    )

    copy_weights(pl_model, q_model)

    x = torch.randn(2, input_size)

    with torch.no_grad():
        y_pl = pl_model(x).float()
        y_q = q_model(x).float()

    diff = y_pl - y_q

    return diff.abs().max().item(), diff.abs().mean().item(), y_pl, y_q


def main():
    for use_radial in [False, True]:
        print("\n" + "=" * 80)
        print("use_radial:", use_radial)
        print("=" * 80)

        best = None

        for reverse_amplitudes in [False, True]:
            for reverse_measure in [False, True]:
                max_diff, mean_diff, y_pl, y_q = run_one(
                    use_radial=use_radial,
                    reverse_amplitudes=reverse_amplitudes,
                    reverse_measure=reverse_measure,
                )

                print(
                    "reverse_amplitudes=",
                    reverse_amplitudes,
                    "reverse_measure=",
                    reverse_measure,
                    "max_diff=",
                    max_diff,
                    "mean_diff=",
                    mean_diff,
                )

                row = (max_diff, mean_diff, reverse_amplitudes, reverse_measure, y_pl, y_q)
                if best is None or max_diff < best[0]:
                    best = row

        print("\nBEST:")
        print("max_diff:", best[0])
        print("mean_diff:", best[1])
        print("reverse_amplitudes:", best[2])
        print("reverse_measure:", best[3])
        print("PennyLane output:")
        print(best[4])
        print("Qiskit output:")
        print(best[5])


if __name__ == "__main__":
    main()

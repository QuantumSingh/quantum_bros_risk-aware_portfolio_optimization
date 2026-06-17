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

torch.manual_seed(68)
np.random.seed(68)

pl_model = QuantumNeuralNetwork(
    input_size=20,
    output_size=4,
    num_weights=8,
    encoding="amplitude",
    input_transformation=None,
    rotation_axes="y",
    entanglement="reverse_linear",
    seed=68,
)

torch.manual_seed(68)
np.random.seed(68)

q_model = QiskitExactAmplitudeFiniteDiffQNN(
    input_size=20,
    output_size=4,
    num_weights=8,
    encoding="amplitude",
    input_transformation=None,
    rotation_axes="y",
    entanglement="reverse_linear",
    seed=68,
)

print("PennyLane weights:")
for name, param in pl_model.named_parameters():
    print(name, param.detach())

print("\nQiskit weights:")
for name, param in q_model.named_parameters():
    print(name, param.detach())

import re
import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


def z_observable(num_qubits: int, qubit: int) -> SparsePauliOp:
    pauli = ["I"] * num_qubits
    pauli[num_qubits - 1 - qubit] = "Z"
    return SparsePauliOp.from_list([("".join(pauli), 1.0)])


class QiskitExactAmplitudeQNN(nn.Module):
    """
    Qiskit reproduction of the PennyLane QuantumNeuralNetwork for
    encoding='amplitude'.

    This is designed to match the paper's PennyLane QNN structure as closely
    as possible:
    - optional input_transformation
    - amplitude encoding with zero padding and normalization
    - RY/RX/RZ variational ansatz
    - reverse_linear/linear/full entanglement
    - Pauli-Z expectation outputs
    - optional output_map and output_activation
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_qubits: int | None = None,
        num_weights: int | None = None,
        output_map: tuple | None = None,
        rotation_axes: str = "y",
        classical_layers: bool = False,
        output_activation=None,
        encoding: str = "amplitude",
        input_transformation=None,
        entanglement: str = "reverse_linear",
        device: str = "cpu",
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if not bool(re.fullmatch(r"[xyz]*", rotation_axes)):
            raise ValueError(
                f'Rotation option "{rotation_axes}" not recognized. '
                'Please only use letters "x", "y", or "z".'
            )

        if entanglement not in ["linear", "reverse_linear", "full"]:
            raise ValueError(
                'Entanglement must be "linear", "reverse_linear", or "full".'
            )

        if encoding != "amplitude":
            raise ValueError(
                "QiskitExactAmplitudeQNN currently targets encoding='amplitude'."
            )

        self.input_size = input_size
        self.output_size = output_size
        self.num_weights = num_weights
        self.output_map = output_map
        self.rotation_axes = rotation_axes
        self.classical_layers = classical_layers
        self.output_activation = output_activation
        self.encoding = encoding
        self.input_transformation = input_transformation
        self.entanglement = entanglement
        self.device = device

        # Match PennyLane actual_input_size logic.
        if input_transformation is not None:
            name = getattr(input_transformation, "__name__", "")
            if name == "radial_to_linear":
                self.actual_input_size = input_size * 4
            elif name == "radial_to_linear_small":
                self.actual_input_size = input_size * 2
            else:
                self.actual_input_size = input_size
        else:
            self.actual_input_size = input_size

        # Match PennyLane amplitude qubit logic.
        if num_qubits is not None:
            self.num_qubits = num_qubits
        else:
            self.num_qubits = max(
                int(np.ceil(np.log2(self.actual_input_size))),
                self.output_size,
            )

        if self.num_weights is None:
            self.num_weights = self.num_qubits

        self.feature_dim = 2 ** self.num_qubits

        if self.actual_input_size > self.feature_dim:
            raise ValueError(
                f"actual_input_size={self.actual_input_size} cannot fit into "
                f"2^{self.num_qubits}={self.feature_dim} amplitudes."
            )

        self.feature_map = RawFeatureVector(feature_dimension=self.feature_dim)
        self.theta = ParameterVector("theta", self.num_weights)

        self.circuit = self._build_circuit()

        if self.classical_layers:
            num_observables = self.num_qubits
        else:
            num_observables = self.output_size

        observables = [
            z_observable(self.num_qubits, q)
            for q in range(num_observables)
        ]

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=Estimator(),
            input_params=list(self.feature_map.parameters),
            weight_params=list(self.theta),
            observables=observables,
            input_gradients=False,
        )

        self.qlayer = TorchConnector(self.qnn)

        if self.classical_layers:
            self.output_layer = nn.Linear(
                self.num_qubits,
                self.output_size,
                dtype=torch.float32,
            )
        else:
            self.output_layer = nn.Identity()

    def _add_entanglement(self, qc: QuantumCircuit) -> None:
        if self.entanglement == "linear":
            for j in range(self.num_qubits - 1):
                qc.cx(j, j + 1)

        elif self.entanglement == "reverse_linear":
            for j in range(self.num_qubits - 1, 0, -1):
                qc.cx(j - 1, j)

        elif self.entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    if i != j:
                        qc.cx(i, j)

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        # Amplitude encoding.
        qc.compose(self.feature_map, inplace=True)

        # Match PennyLane ParameterizedQuantumCircuit ansatz.
        for i, param in enumerate(self.theta):
            q = i % self.num_qubits

            if "x" in self.rotation_axes:
                qc.rx(param, q)
            if "y" in self.rotation_axes:
                qc.ry(param, q)
            if "z" in self.rotation_axes:
                qc.rz(param, q)

            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(self.theta):
                self._add_entanglement(qc)

        return qc

    def _pad_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != self.actual_input_size:
            raise ValueError(
                f"Provided tensor has shape {x.shape}, but actual_input_size "
                f"is configured to {self.actual_input_size}."
            )

        batch_size = x.shape[0]

        padded = torch.zeros(
            batch_size,
            self.feature_dim,
            dtype=x.dtype,
            device=x.device,
        )
        padded[:, : self.actual_input_size] = x

        norm = torch.linalg.norm(padded, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)

        return padded / norm

    def _map_to_output_range(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_map is None:
            return logits

        if self.output_map == (-1, 1):
            return logits

        if self.output_map == (-float("inf"), float("inf")):
            return 100 * torch.arctanh(torch.clamp(logits, -0.9999, 0.9999))

        normalized_output = (logits + 1) / 2
        return (
            self.output_map[0]
            + normalized_output * (self.output_map[1] - self.output_map[0])
        )

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.float()

        if self.input_transformation is not None:
            tensor = self.input_transformation(tensor)

        tensor = self._pad_and_normalize(tensor)

        tensor = self.qlayer(tensor).to(self.device)
        tensor = self.output_layer(tensor)

        if self.output_map is not None:
            tensor = self._map_to_output_range(tensor)

        if self.output_activation is not None:
            tensor = self.output_activation(tensor)

        return tensor


if __name__ == "__main__":
    model = QiskitExactAmplitudeQNN(
        input_size=20,
        output_size=4,
        num_weights=8,
        encoding="amplitude",
        rotation_axes="y",
        seed=68,
    )

    x = torch.randn(2, 20)
    y = model(x)

    print(model.circuit)
    print("input shape:", x.shape)
    print("actual_input_size:", model.actual_input_size)
    print("num_qubits:", model.num_qubits)
    print("feature_dim:", model.feature_dim)
    print("output shape:", y.shape)
    print("output:", y)

import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector


def z_observable(num_qubits: int, qubit: int) -> SparsePauliOp:
    pauli = ["I"] * num_qubits
    pauli[num_qubits - 1 - qubit] = "Z"
    return SparsePauliOp.from_list([("".join(pauli), 1.0)])


class QiskitTrainableQNN(nn.Module):
    """
    Qiskit QNN designed to train inside the original DDPG pipeline.

    This uses angle encoding instead of RawFeatureVector amplitude encoding
    because Qiskit's RawFeatureVector/ParameterizedInitialize fails during
    gradient backpropagation in EstimatorQNN.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_qubits: int | None = None,
        num_weights: int | None = 60,
        encoding: str = "angle",
        rotation_axes: str = "y",
        entanglement: str = "reverse_linear",
        output_map: tuple | None = None,
        output_activation=None,
        input_transformation=None,
        seed: int = 68,
        **kwargs,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        self.num_weights = num_weights
        self.entanglement = entanglement
        self.encoding = encoding
        self.rotation_axes = rotation_axes
        self.output_map = output_map
        self.output_activation = output_activation
        self.input_transformation = input_transformation

        if num_qubits is None:
            self.num_qubits = max(output_size, min(input_size, 8))
        else:
            self.num_qubits = num_qubits

        self.input_params = ParameterVector("x", self.input_size)
        self.theta = ParameterVector("theta", self.num_weights)

        self.circuit = self._build_circuit()

        observables = [
            z_observable(self.num_qubits, q)
            for q in range(self.num_qubits)
        ]

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=Estimator(),
            input_params=list(self.input_params),
            weight_params=list(self.theta),
            observables=observables,
            input_gradients=False,
        )

        self.qlayer = TorchConnector(self.qnn)

        if self.num_qubits != self.output_size:
            self.output_layer = nn.Linear(self.num_qubits, self.output_size)
        else:
            self.output_layer = nn.Identity()

    def _add_entanglement(self, qc: QuantumCircuit) -> None:
        if self.entanglement == "linear":
            for q in range(self.num_qubits - 1):
                qc.cx(q, q + 1)

        elif self.entanglement == "reverse_linear":
            for q in range(self.num_qubits - 1, 0, -1):
                qc.cx(q - 1, q)

        elif self.entanglement == "full":
            for i in range(self.num_qubits):
                for j in range(self.num_qubits):
                    if i != j:
                        qc.cx(i, j)

        else:
            raise ValueError(f"Unknown entanglement type: {self.entanglement}")

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        # Angle encoding: encode each input feature as an RY rotation.
        for i, x_i in enumerate(self.input_params):
            q = i % self.num_qubits
            qc.ry(x_i, q)

            if (i + 1) % self.num_qubits == 0:
                self._add_entanglement(qc)

        # Trainable variational layer.
        for i, theta_i in enumerate(self.theta):
            q = i % self.num_qubits

            if "x" in self.rotation_axes:
                qc.rx(theta_i, q)
            if "y" in self.rotation_axes:
                qc.ry(theta_i, q)
            if "z" in self.rotation_axes:
                qc.rz(theta_i, q)

            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(self.theta):
                self._add_entanglement(qc)

        return qc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        if self.input_transformation is not None:
            x = self.input_transformation(x)

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input last dimension {self.input_size}, got {x.shape[-1]}"
            )

        quantum_output = self.qlayer(x)
        final_output = self.output_layer(quantum_output)

        if self.output_activation is not None:
            final_output = self.output_activation(final_output)

        return final_output


if __name__ == "__main__":
    model = QiskitTrainableQNN(
        input_size=24,
        output_size=4,
        num_weights=60,
        rotation_axes="y",
        seed=68,
    )

    x = torch.randn(3, 24)
    y = model(x)

    print(model.circuit)
    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("output:", y)

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
    """
    Create a Pauli-Z observable on one qubit.

    Qiskit Pauli strings are ordered from highest qubit index on the left
    to qubit 0 on the right.
    """
    pauli = ["I"] * num_qubits
    pauli[num_qubits - 1 - qubit] = "Z"
    return SparsePauliOp.from_list([("".join(pauli), 1.0)])


class QiskitQuantumNeuralNetwork(nn.Module):
    """
    Qiskit version of the PennyLane QuantumNeuralNetwork prototype.

    This first version supports angle-style input encoding and returns
    expectation values of Pauli-Z observables, similar to the PennyLane QNN.

    Goal:
        input tensor -> Qiskit VQC -> output tensor
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_qubits: int | None = None,
        num_weights: int | None = None,
        output_map: tuple | None = None,
        rotation_axes: str = "y",
        entanglement: str = "reverse_linear",
        seed: int | None = None,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if entanglement not in ["linear", "reverse_linear", "full"]:
            raise ValueError("entanglement must be 'linear', 'reverse_linear', or 'full'")

        self.input_size = input_size
        self.output_size = output_size
        self.num_qubits = num_qubits if num_qubits is not None else max(input_size, output_size)
        self.num_weights = num_weights if num_weights is not None else self.num_qubits
        self.output_map = output_map
        self.rotation_axes = rotation_axes
        self.entanglement = entanglement

        self.input_params = ParameterVector("x", self.input_size)
        self.weight_params = ParameterVector("theta", self.num_weights)

        self.circuit = self._build_circuit()

        observables = [
            z_observable(self.num_qubits, i)
            for i in range(self.output_size)
        ]

        self.qnn = EstimatorQNN(
            circuit=self.circuit,
            estimator=Estimator(),
            input_params=list(self.input_params),
            weight_params=list(self.weight_params),
            observables=observables,
            input_gradients=True,
        )

        self.qlayer = TorchConnector(self.qnn)

    def _add_entanglement(self, qc: QuantumCircuit):
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

    def _build_circuit(self) -> QuantumCircuit:
        qc = QuantumCircuit(self.num_qubits)

        # Input encoding: map classical features into rotations.
        for i, x_i in enumerate(self.input_params):
            target_qubit = i % self.num_qubits

            if "x" in self.rotation_axes:
                qc.rx(x_i, target_qubit)
            else:
                qc.ry(x_i, target_qubit)

        self._add_entanglement(qc)

        # Trainable ansatz parameters.
        for i, theta_i in enumerate(self.weight_params):
            target_qubit = i % self.num_qubits

            if "x" in self.rotation_axes:
                qc.rx(theta_i, target_qubit)
            if "y" in self.rotation_axes:
                qc.ry(theta_i, target_qubit)
            if "z" in self.rotation_axes:
                qc.rz(theta_i, target_qubit)

            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(self.weight_params):
                self._add_entanglement(qc)

        return qc

    def _map_to_output_range(self, logits: torch.Tensor) -> torch.Tensor:
        if self.output_map is None:
            return logits

        low, high = self.output_map

        if self.output_map == (-1, 1):
            return logits

        if low == -float("inf") and high == float("inf"):
            return 100 * torch.atanh(torch.clamp(logits, -0.9999, 0.9999))

        normalized = (logits + 1) / 2
        return low + normalized * (high - low)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()

        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input shape ending in {self.input_size}, got {x.shape}"
            )

        out = self.qlayer(x)
        out = self._map_to_output_range(out)

        return out


if __name__ == "__main__":
    model = QiskitQuantumNeuralNetwork(
        input_size=4,
        output_size=2,
        num_qubits=4,
        num_weights=8,
        rotation_axes="y",
        entanglement="reverse_linear",
        seed=68,
    )

    x = torch.randn(3, 4)
    y = model(x)

    print(model.circuit)
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Output:", y)

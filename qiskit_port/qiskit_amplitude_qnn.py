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
    """
    Create a Pauli-Z observable on a selected qubit.
    """
    pauli = ["I"] * num_qubits
    pauli[num_qubits - 1 - qubit] = "Z"
    return SparsePauliOp.from_list([("".join(pauli), 1.0)])


class QiskitAmplitudeQNN(nn.Module):
    """
    Qiskit prototype of the paper's PennyLane QuantumNeuralNetwork
    when encoding='amplitude'.

    It does:

        classical input vector
            -> amplitude encoding
            -> trainable RY variational circuit
            -> Pauli-Z expectation values
            -> torch output tensor
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_weights: int = 60,
        num_qubits: int | None = None,
        entanglement: str = "reverse_linear",
        seed: int = 68,
    ):
        super().__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.input_size = input_size
        self.output_size = output_size
        self.num_weights = num_weights
        self.entanglement = entanglement

        if num_qubits is None:
            self.num_qubits = max(
                int(np.ceil(np.log2(input_size))),
                output_size,
            )
        else:
            self.num_qubits = num_qubits

        self.feature_dim = 2 ** self.num_qubits

        if input_size > self.feature_dim:
            raise ValueError(
                f"input_size={input_size} cannot fit into "
                f"2^{self.num_qubits}={self.feature_dim} amplitudes."
            )

        self.feature_map = RawFeatureVector(feature_dimension=self.feature_dim)
        self.theta = ParameterVector("theta", self.num_weights)

        self.circuit = self._build_circuit()

        observables = [
            z_observable(self.num_qubits, q)
            for q in range(self.output_size)
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

    def _add_entanglement(self, qc: QuantumCircuit) -> None:
        """
        Add entanglement gates.

        reverse_linear roughly mirrors the original PennyLane default.
        """
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
        """
        Build the amplitude-encoded Qiskit circuit.
        """
        qc = QuantumCircuit(self.num_qubits)

        qc.compose(self.feature_map, inplace=True)

        for i, param in enumerate(self.theta):
            q = i % self.num_qubits
            qc.ry(param, q)

            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(self.theta):
                self._add_entanglement(qc)

        return qc

    def _pad_and_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Amplitude encoding needs a vector of length 2^num_qubits.
        If the input is shorter, pad with zeros.
        Then normalize to unit length.
        """
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"Expected input last dimension {self.input_size}, got {x.shape[-1]}"
            )

        batch_size = x.shape[0]

        padded = torch.zeros(
            batch_size,
            self.feature_dim,
            dtype=x.dtype,
            device=x.device,
        )

        padded[:, : self.input_size] = x

        norm = torch.linalg.norm(padded, dim=1, keepdim=True)
        norm = torch.clamp(norm, min=1e-12)

        return padded / norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = self._pad_and_normalize(x)
        return self.qlayer(x)


if __name__ == "__main__":
    model = QiskitAmplitudeQNN(
        input_size=10,
        output_size=4,
        num_weights=60,
        seed=68,
    )

    x = torch.randn(3, 10)
    y = model(x)

    print(model.circuit)
    print("num_qubits:", model.num_qubits)
    print("feature_dim:", model.feature_dim)
    print("input shape:", x.shape)
    print("output shape:", y.shape)
    print("output:", y)

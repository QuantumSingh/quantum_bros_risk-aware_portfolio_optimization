import re
import numpy as np
import torch
import torch.nn as nn

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector



def _bit_reverse_indices(num_qubits: int) -> np.ndarray:
    """Map PennyLane amplitude ordering to Qiskit amplitude ordering."""
    size = 2 ** num_qubits
    indices = np.zeros(size, dtype=int)

    for i in range(size):
        reversed_bits = format(i, f"0{num_qubits}b")[::-1]
        indices[i] = int(reversed_bits, 2)

    return indices


class _QiskitAmplitudeParameterShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weights, module):
        inputs_np = inputs.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()

        outputs = module._forward_numpy(inputs_np, weights_np)

        ctx.module = module
        ctx.save_for_backward(inputs, weights)

        return torch.tensor(
            outputs,
            dtype=inputs.dtype,
            device=inputs.device,
        )

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weights = ctx.saved_tensors
        module = ctx.module

        original_inputs_shape = tuple(inputs.shape)

        inputs_np = inputs.detach().cpu().numpy()
        weights_np = weights.detach().cpu().numpy()
        grad_output_np = grad_output.detach().cpu().numpy()

        # DDPG sometimes calls the QNN with a single state vector [input_dim]
        # and sometimes with a batch [batch_size, input_dim].
        # Use 2D arrays internally, then reshape gradients back.
        if inputs_np.ndim == 1:
            inputs_np = inputs_np.reshape(1, -1)

        if grad_output_np.ndim == 1:
            grad_output_np = grad_output_np.reshape(1, -1)

        num_weights = len(weights_np)
        grad_weights = np.zeros_like(weights_np, dtype=np.float64)

        # ------------------------------------------------------------
        # 1. Weight gradients using exact parameter-shift.
        # ------------------------------------------------------------
        shift = np.pi / 2.0

        for k in range(num_weights):
            weights_plus = weights_np.copy()
            weights_minus = weights_np.copy()

            weights_plus[k] += shift
            weights_minus[k] -= shift

            out_plus = module._forward_numpy(inputs_np, weights_plus)
            out_minus = module._forward_numpy(inputs_np, weights_minus)

            jac_k = 0.5 * (out_plus - out_minus)
            grad_weights[k] = np.sum(grad_output_np * jac_k)

        # ------------------------------------------------------------
        # 2. Optional input gradients using central finite difference.
        # Full original settings are extremely slow with input gradients.
        # ------------------------------------------------------------
        if not getattr(module, "compute_input_gradients", False):
            grad_weights_torch = torch.tensor(
                grad_weights,
                dtype=weights.dtype,
                device=weights.device,
            )
            return None, grad_weights_torch, None

        grad_inputs = np.zeros_like(inputs_np, dtype=np.float64)
        eps = 1e-4

        batch_size, input_dim = inputs_np.shape

        for b in range(batch_size):
            for j in range(input_dim):
                inputs_plus = inputs_np.copy()
                inputs_minus = inputs_np.copy()

                inputs_plus[b, j] += eps
                inputs_minus[b, j] -= eps

                out_plus = module._forward_numpy(inputs_plus, weights_np)[b]
                out_minus = module._forward_numpy(inputs_minus, weights_np)[b]

                jac_bj = (out_plus - out_minus) / (2.0 * eps)
                grad_inputs[b, j] = np.sum(grad_output_np[b] * jac_bj)

        grad_inputs = grad_inputs.reshape(original_inputs_shape)

        grad_inputs_torch = torch.tensor(
            grad_inputs,
            dtype=inputs.dtype,
            device=inputs.device,
        )

        grad_weights_torch = torch.tensor(
            grad_weights,
            dtype=weights.dtype,
            device=weights.device,
        )

        return grad_inputs_torch, grad_weights_torch, None


class QiskitExactAmplitudeFiniteDiffQNN(nn.Module):
    """
    Exact-amplitude Qiskit QNN meant to reproduce the PennyLane QNN structure.

    This avoids Qiskit Machine Learning's RawFeatureVector backward issue by:
    - preparing numeric amplitude states directly,
    - simulating with Qiskit Statevector,
    - computing trainable-weight gradients with parameter shift.
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
        compute_input_gradients: bool = False,
        **kwargs,
    ):
        super().__init__()

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        if encoding != "amplitude":
            raise ValueError("This class is only for encoding='amplitude'.")

        if not bool(re.fullmatch(r"[xyz]*", rotation_axes)):
            raise ValueError("rotation_axes must only contain x, y, and/or z.")

        if rotation_axes != "y":
            raise ValueError(
                "For exact original-paper reproduction, this class currently "
                "supports rotation_axes='y'."
            )

        if entanglement not in ["linear", "reverse_linear", "full"]:
            raise ValueError("entanglement must be linear, reverse_linear, or full.")

        self.input_size = input_size
        self.output_size = output_size
        self.output_map = output_map
        self.rotation_axes = rotation_axes
        self.classical_layers = classical_layers
        self.output_activation = output_activation
        self.encoding = encoding
        self.input_transformation = input_transformation
        self.entanglement = entanglement
        self.device = device
        self.compute_input_gradients = compute_input_gradients

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

        if num_qubits is not None:
            self.num_qubits = num_qubits
        else:
            self.num_qubits = max(
                int(np.ceil(np.log2(self.actual_input_size))),
                self.output_size,
            )

        self.feature_dim = 2 ** self.num_qubits

        # PennyLane and Qiskit use opposite amplitude basis ordering.
        # Bit-reversing amplitudes makes Qiskit forward outputs match PennyLane.
        self._amplitude_permutation = _bit_reverse_indices(self.num_qubits)

        if self.actual_input_size > self.feature_dim:
            raise ValueError(
                f"actual_input_size={self.actual_input_size} cannot fit into "
                f"2^{self.num_qubits}={self.feature_dim} amplitudes."
            )

        if num_weights is None:
            num_weights = self.num_qubits

        self.num_weights = num_weights

        # Match PennyLane TorchLayer trainable weights.
        self.weights = nn.Parameter(
            torch.empty(self.num_weights, dtype=torch.float32)
        )
        nn.init.uniform_(self.weights, 0.0, 2.0 * np.pi)

        if self.classical_layers:
            self.output_layer = nn.Linear(
                self.num_qubits,
                self.output_size,
                dtype=torch.float32,
            )
        else:
            self.output_layer = nn.Identity()

    def _pad_and_normalize_numpy(self, x):
        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[-1] != self.actual_input_size:
            raise ValueError(
                f"Input has shape {x.shape}, expected final dim "
                f"{self.actual_input_size}."
            )

        padded = np.zeros((x.shape[0], self.feature_dim), dtype=np.float64)
        padded[:, : self.actual_input_size] = x

        norms = np.linalg.norm(padded, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)

        normalized = padded / norms

        # Match PennyLane amplitude ordering.
        reordered = np.zeros_like(normalized)
        reordered[:, self._amplitude_permutation] = normalized

        return reordered

    def _add_entanglement(self, qc):
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

    def _build_numeric_circuit(self, amplitudes, weights):
        qc = QuantumCircuit(self.num_qubits)

        # Numeric amplitude encoding: PennyLane AmplitudeEncoding equivalent.
        qc.initialize(amplitudes, list(range(self.num_qubits)))

        # Match PennyLane ParameterizedQuantumCircuit.
        for i, param in enumerate(weights.flatten()):
            q = i % self.num_qubits

            # Original paper uses rotation_axes='y'.
            qc.ry(float(param), q)

            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(weights):
                self._add_entanglement(qc)

        return qc

    def _z_expectation(self, state, qubit):
        probs = np.abs(state.data) ** 2

        expval = 0.0
        for basis_index, prob in enumerate(probs):
            bit = (basis_index >> qubit) & 1
            expval += prob * (1.0 if bit == 0 else -1.0)

        return expval

    def _forward_numpy(self, inputs_np, weights_np):
        encoded = self._pad_and_normalize_numpy(inputs_np)

        outputs = []

        num_measurements = self.num_qubits if self.classical_layers else self.output_size

        for sample in encoded:
            qc = self._build_numeric_circuit(sample, weights_np)
            state = Statevector.from_instruction(qc)

            sample_outputs = [
                self._z_expectation(state, q)
                for q in range(num_measurements)
            ]

            outputs.append(sample_outputs)

        return np.array(outputs, dtype=np.float64)

    def _map_to_output_range(self, logits):
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

    def forward(self, tensor):
        tensor = tensor.float()

        if self.input_transformation is not None:
            tensor = self.input_transformation(tensor)

        tensor = _QiskitAmplitudeParameterShift.apply(
            tensor,
            self.weights,
            self,
        )

        tensor = self.output_layer(tensor)

        if self.output_map is not None:
            tensor = self._map_to_output_range(tensor)

        if self.output_activation is not None:
            tensor = self.output_activation(tensor)

        return tensor


if __name__ == "__main__":
    model = QiskitExactAmplitudeFiniteDiffQNN(
        input_size=20,
        output_size=4,
        num_weights=8,
        encoding="amplitude",
        rotation_axes="y",
        seed=68,
    )

    x = torch.randn(2, 20)
    y = model(x)

    print("input shape:", x.shape)
    print("actual_input_size:", model.actual_input_size)
    print("num_qubits:", model.num_qubits)
    print("feature_dim:", model.feature_dim)
    print("output shape:", y.shape)
    print("output:", y)

    loss = y.sum()
    loss.backward()

    print("loss:", loss.item())
    print("weights grad:", model.weights.grad)

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

        # PennyLane preserves the input's dimensionality: a single 1-D
        # state vector produces a 1-D output, not a batch of one.
        if inputs_np.ndim == 1:
            outputs = outputs[0]

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
        #
        # When rotation_axes has multiple letters (e.g. "xy"), a single
        # weight drives multiple gates (RX(theta) and RY(theta)) on the
        # same qubit with the shared value theta. The total derivative is
        # then the sum of independent single-gate parameter-shift terms,
        # one per axis, each holding the other axis gates fixed at their
        # base angle (chain rule for a value reused by several gates).
        # ------------------------------------------------------------
        shift = np.pi / 2.0
        axes_present = [a for a in "xyz" if a in module.rotation_axes]

        for k in range(num_weights):
            grad_k = 0.0

            for axis in axes_present:
                out_plus = module._forward_numpy(
                    inputs_np, weights_np, shift=(k, axis, shift)
                )
                out_minus = module._forward_numpy(
                    inputs_np, weights_np, shift=(k, axis, -shift)
                )

                jac_k_axis = 0.5 * (out_plus - out_minus)
                grad_k += np.sum(grad_output_np * jac_k_axis)

            grad_weights[k] = grad_k

        # ------------------------------------------------------------
        # 2. Input gradients using central finite difference.
        # Computed when autograd actually needs them (e.g. the DDPG
        # critic backpropagating into the actor's action) or when
        # explicitly forced via compute_input_gradients. Skipped
        # otherwise because full finite difference is expensive.
        # ------------------------------------------------------------
        needs_input_grad = ctx.needs_input_grad[0] or getattr(
            module, "compute_input_gradients", False
        )

        if not needs_input_grad:
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

        if encoding not in ("amplitude", "angle", "stacked_angle"):
            raise ValueError(
                "encoding must be 'amplitude', 'angle', or 'stacked_angle'."
            )

        if not bool(re.fullmatch(r"[xyz]*", rotation_axes)):
            raise ValueError("rotation_axes must only contain x, y, and/or z.")

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
        elif encoding in ("angle", "stacked_angle"):
            self.num_qubits = max(self.actual_input_size, self.output_size)
        else:
            self.num_qubits = max(
                int(np.ceil(np.log2(self.actual_input_size))),
                self.output_size,
            )

        if encoding == "amplitude":
            self.feature_dim = 2 ** self.num_qubits

            # PennyLane and Qiskit use opposite amplitude basis ordering.
            # Bit-reversing amplitudes makes Qiskit forward outputs match PennyLane.
            self._amplitude_permutation = _bit_reverse_indices(self.num_qubits)

            if self.actual_input_size > self.feature_dim:
                raise ValueError(
                    f"actual_input_size={self.actual_input_size} cannot fit into "
                    f"2^{self.num_qubits}={self.feature_dim} amplitudes."
                )
        else:
            self.feature_dim = None
            self._amplitude_permutation = None

            if encoding == "angle" and self.actual_input_size > self.num_qubits:
                raise ValueError(
                    f"Angle encoding selected, but number of features "
                    f"({self.actual_input_size}) exceeds number of qubits "
                    f"({self.num_qubits})."
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

    def _encoding_rotation_gate(self, qc):
        # AngleEncoding/StackedAngleEncoding always pick RX if 'x' is in
        # rotation_axes, else RY -- independent of 'z'.
        return qc.rx if "x" in self.rotation_axes else qc.ry

    def _add_angle_encoding(self, qc, features):
        gate_fn = self._encoding_rotation_gate(qc)

        # Matches AngleEncoding.compute_decomposition: one rotation per
        # wire, indexed straight from the feature vector (no wraparound).
        for i in range(self.num_qubits):
            gate_fn(float(features[i]), i)

    def _add_stacked_angle_encoding(self, qc, features):
        gate_fn = self._encoding_rotation_gate(qc)
        num_qubits = self.num_qubits
        num_features = len(features)

        # Matches StackedAngleEncoding.compute_decomposition, including its
        # "full" entanglement quirk that reuses the outer feature index i
        # (rather than a qubit index) as one CNOT endpoint.
        for i in range(num_features):
            target = i % num_qubits
            gate_fn(float(features[i]), target)

            if (i + 1) % num_qubits == 0:
                if self.entanglement == "linear":
                    for j in range(num_qubits - 1):
                        qc.cx(j, j + 1)
                elif self.entanglement == "reverse_linear":
                    for j in range(num_qubits - 1, 0, -1):
                        qc.cx(j - 1, j)
                elif self.entanglement == "full":
                    for j in range(num_qubits):
                        if i != j:
                            qc.cx(i, j)

    def _build_numeric_circuit(self, sample, weights, shift=None):
        """Build the ansatz circuit.

        ``shift`` is an optional ``(weight_index, axis, delta)`` override
        used for per-gate parameter-shift evaluation: only the gate at
        ``weight_index`` for the given ``axis`` gets ``value + delta``,
        while any other gate driven by the same weight (e.g. the RY gate
        when rotation_axes="xy" and we're shifting the RX gate) stays at
        its unshifted base value.
        """
        qc = QuantumCircuit(self.num_qubits)

        if self.encoding == "amplitude":
            # Numeric amplitude encoding: PennyLane AmplitudeEncoding equivalent.
            qc.initialize(sample, list(range(self.num_qubits)))
        elif self.encoding == "angle":
            self._add_angle_encoding(qc, sample)
        elif self.encoding == "stacked_angle":
            self._add_stacked_angle_encoding(qc, sample)

        gates_by_axis = (("x", qc.rx), ("y", qc.ry), ("z", qc.rz))

        # Match PennyLane ParameterizedQuantumCircuit.
        for i, param in enumerate(weights.flatten()):
            q = i % self.num_qubits
            base = float(param)

            # Same weight value drives every requested axis on this qubit,
            # matching ParameterizedQuantumCircuit.__call__ in the original.
            for axis, gate_fn in gates_by_axis:
                if axis not in self.rotation_axes:
                    continue

                angle = base
                if shift is not None and shift[0] == i and shift[1] == axis:
                    angle = base + shift[2]

                gate_fn(angle, q)

            if (i + 1) % self.num_qubits == 0 and (i + 1) != len(weights):
                self._add_entanglement(qc)

        return qc

    def _z_expectations(self, state, num_measurements):
        probs = np.abs(state.data) ** 2
        indices = np.arange(probs.shape[0])

        expvals = np.empty(num_measurements, dtype=np.float64)
        for q in range(num_measurements):
            signs = 1.0 - 2.0 * ((indices >> q) & 1)
            expvals[q] = np.dot(probs, signs)

        return expvals

    def _prepare_samples_numpy(self, inputs_np):
        if self.encoding == "amplitude":
            return self._pad_and_normalize_numpy(inputs_np)

        if inputs_np.ndim == 1:
            inputs_np = inputs_np.reshape(1, -1)

        if inputs_np.shape[-1] != self.actual_input_size:
            raise ValueError(
                f"Input has shape {inputs_np.shape}, expected final dim "
                f"{self.actual_input_size}."
            )

        return inputs_np.astype(np.float64, copy=False)

    def _forward_numpy(self, inputs_np, weights_np, shift=None):
        encoded = self._prepare_samples_numpy(inputs_np)

        outputs = []

        num_measurements = self.num_qubits if self.classical_layers else self.output_size

        for sample in encoded:
            qc = self._build_numeric_circuit(sample, weights_np, shift=shift)
            state = Statevector.from_instruction(qc)

            outputs.append(self._z_expectations(state, num_measurements))

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

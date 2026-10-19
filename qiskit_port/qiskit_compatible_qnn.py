from qiskit_port.qiskit_exact_amplitude_finite_diff_qnn import (
    QiskitExactAmplitudeFiniteDiffQNN,
)


class QiskitQuantumNeuralNetwork(QiskitExactAmplitudeFiniteDiffQNN):
    """
    Drop-in Qiskit replacement for the PennyLane QuantumNeuralNetwork.

    Parity-verified against the original (see
    test_full_pennylane_qiskit_parity.py) for:
      - encoding: amplitude, angle, stacked_angle
      - rotation_axes: x, y, z, xy, xyz
      - entanglement: linear, reverse_linear, full
      - input_transformation (e.g. radial_to_linear)
      - output_map, output_activation
      - classical_layers (with a fix: the original re-creates an
        unregistered random nn.Linear inside every forward() call, so it
        crashes on float32 inputs and is non-deterministic on float64;
        this port owns a persistent trainable output Linear instead)

    Later phases add faster backend modes (backend="fast_statevector"
    for training, backend="qiskit_reference" for diagrams/QASM/IonQ).
    """

    def __init__(self, *args, backend="fast_statevector", **kwargs):
        self.backend = backend
        super().__init__(*args, **kwargs)

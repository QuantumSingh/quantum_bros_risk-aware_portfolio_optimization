import torch

from qiskit_quantum_neural_network import QiskitQuantumNeuralNetwork


def main():
    model = QiskitQuantumNeuralNetwork(
        input_size=6,
        output_size=3,
        num_qubits=6,
        num_weights=12,
        rotation_axes="y",
        entanglement="reverse_linear",
        seed=68,
    )

    x = torch.randn(5, 6)
    y = model(x)

    assert y.shape == (5, 3)

    print("Qiskit QNN test passed.")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()

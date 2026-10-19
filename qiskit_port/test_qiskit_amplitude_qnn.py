import torch

from qiskit_amplitude_qnn import QiskitAmplitudeQNN


def main():
    model = QiskitAmplitudeQNN(
        input_size=10,
        output_size=4,
        num_weights=60,
        seed=68,
    )

    x = torch.randn(5, 10)
    y = model(x)

    assert y.shape == (5, 4)

    print("Qiskit amplitude QNN test passed.")
    print("Input shape:", x.shape)
    print("Output shape:", y.shape)


if __name__ == "__main__":
    main()

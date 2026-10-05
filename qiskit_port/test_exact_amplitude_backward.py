import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from qiskit_port.qiskit_exact_amplitude_qnn import QiskitExactAmplitudeQNN

def main():
    print("Testing exact amplitude QNN backward pass...")

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

    print("Forward output shape:", y.shape)
    print("Forward output:", y)

    loss = y.sum()
    print("Loss:", loss.item())

    try:
        loss.backward()
        print("Backward pass worked.")

        for name, param in model.named_parameters():
            print(name, "grad:", param.grad)

    except Exception as e:
        print("Backward pass failed.")
        print(type(e).__name__)
        print(e)


if __name__ == "__main__":
    main()

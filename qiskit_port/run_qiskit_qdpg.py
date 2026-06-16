import os
import sys
from pathlib import Path

# Find repo root
REPO_ROOT = Path(__file__).resolve().parents[1]

# Original PennyLane paper code location
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

# Add original code to Python path
sys.path.insert(0, str(ORIGINAL_CODE))

# Add repo root so Python can import qiskit_port
sys.path.insert(0, str(REPO_ROOT))

# Change working directory because original MAIN.py uses relative paths like ./data/
os.chdir(ORIGINAL_CODE)

# Original DDPG pipeline
from ddpg.ddpg_functions import DDPG

# Your Qiskit replacement predictor
from qiskit_port.qiskit_amplitude_qnn import QiskitAmplitudeQNN


def main():
    print("Qiskit QDPG runner loaded successfully.")
    print("Repo root:", REPO_ROOT)
    print("Original code path:", ORIGINAL_CODE)
    print("DDPG class:", DDPG)
    print("Qiskit predictor:", QiskitAmplitudeQNN)


if __name__ == "__main__":
    main()

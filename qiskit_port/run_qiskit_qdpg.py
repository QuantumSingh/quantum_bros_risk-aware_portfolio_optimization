import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(ORIGINAL_CODE))
sys.path.insert(0, str(REPO_ROOT))

# Original code uses relative paths like ./data/price_data.parquet.gzip
os.chdir(ORIGINAL_CODE)

# ---------------------------------------------------------------------
# Original paper code imports
# ---------------------------------------------------------------------

from ddpg.ddpg_functions import DDPG
from qiskit_port.qiskit_trainable_qnn import QiskitTrainableQNN

# Try to import the same transformation used by the original QDPG block.
# If this import fails, we will temporarily use None and fix the import path.
try:
    from predictors.quantum_neural_network import radial_to_linear
except ImportError:
    radial_to_linear = None


def main():
    print("Starting Qiskit QDPG smoke test...")
    print("Original code path:", ORIGINAL_CODE)
    print("Using predictor:", QiskitTrainableQNN)
    print("radial_to_linear:", radial_to_linear)

    # -----------------------------------------------------------------
    # Small config for smoke testing
    # -----------------------------------------------------------------

    LOOKBACK_WINDOW = 5
    FORECAST_WINDOW = 0
    SHORT_SELLING = True
    CLAMP_NEGATIVES = True
    SEED = 68

    # -----------------------------------------------------------------
    # Load original generated data
    # -----------------------------------------------------------------

    price_data = pd.read_parquet("./data/price_data.parquet.gzip")
    print("Full price_data shape:", price_data.shape)

    # Use a small slice first so Qiskit does not take forever.
    # Later we can scale this back up to the full cross-validation setup.
    price_data = price_data.iloc[:, :4]
    price_data = price_data.tail(120)
    print("Smoke-test price_data shape:", price_data.shape)

    n = len(price_data)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)

    train_data = price_data.iloc[:train_end]
    val_data = price_data.iloc[train_end:val_end]
    test_data = price_data.iloc[val_end:]

    print("Train shape:", train_data.shape)
    print("Val shape:", val_data.shape)
    print("Test shape:", test_data.shape)

    # -----------------------------------------------------------------
    # Qiskit QDPG model
    # This copies the original QDPG setup, changing only predictor.
    # -----------------------------------------------------------------

    model = DDPG(
        lookback_window=LOOKBACK_WINDOW,
        forecast_window=FORECAST_WINDOW,
        batch_size=1,
        predictor=QiskitTrainableQNN,
        num_weights=8,
	num_qubits=4,
        encoding="angle",
        input_transformation=radial_to_linear,
        rotation_axes="y",
        short_selling=SHORT_SELLING,
        reduce_negatives=CLAMP_NEGATIVES,
        verbose=1,
        seed=SEED,
    )

    start_time = time.time()

    model.train(
        train_data=train_data,
        val_data=val_data,
        actor_lr=0.09935741130315447,
        critic_lr=0.0018039893844072358,
        optimizer=torch.optim.SGD,
        l2_lambda=3.2067524338595386e-06,
        soft_update=False,
        num_epochs=1,
        early_stopping=False,
    )

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.2f} seconds.")

    results = model.evaluate(
        test_data=test_data,
        dpo=True,
    )

    print("Qiskit QDPG smoke-test results:")
    print(results)


if __name__ == "__main__":
    main()

import os
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(ORIGINAL_CODE))
os.chdir(ORIGINAL_CODE)

from ddpg.ddpg_functions import DDPG
from predictors.quantum_neural_network import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
import utilities.metrics as metrics_module


def main():
    print("Starting PennyLane QDPG smoke baseline...")
    print("Original code path:", ORIGINAL_CODE)
    print("Using predictor:", QuantumNeuralNetwork)
    print("radial_to_linear:", radial_to_linear)

    LOOKBACK_WINDOW = 5
    FORECAST_WINDOW = 0
    SHORT_SELLING = True
    CLAMP_NEGATIVES = True
    SEED = 68

    torch.manual_seed(SEED)

    price_data = pd.read_parquet("./data/price_data.parquet.gzip")
    print("Full price_data shape:", price_data.shape)

    price_data = price_data.iloc[:, :4]
    price_data = price_data.tail(120)

    print("Smoke-test price_data shape:", price_data.shape)

    metrics_module.tickers = list(price_data.columns)
    print("Smoke-test tickers:", metrics_module.tickers)

    n = len(price_data)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_data = price_data.iloc[:train_end]
    val_data = price_data.iloc[train_end:val_end]
    test_data = price_data.iloc[val_end:]

    print("Train shape:", train_data.shape)
    print("Val shape:", val_data.shape)
    print("Test shape:", test_data.shape)

    model = DDPG(
        lookback_window=LOOKBACK_WINDOW,
        forecast_window=FORECAST_WINDOW,
        batch_size=1,
        predictor=QuantumNeuralNetwork,
        num_weights=8,
        encoding="amplitude",
        input_transformation=radial_to_linear,
        rotation_axes="y",
        short_selling=SHORT_SELLING,
        reduce_negatives=CLAMP_NEGATIVES,
        verbose=1,
        seed=SEED,
    )

    print("Training PennyLane QDPG baseline...")

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

    print("Evaluating PennyLane QDPG baseline...")

    results = model.evaluate(test_data=test_data, dpo=True)

    print("PennyLane QDPG smoke baseline results:")
    print(results)


if __name__ == "__main__":
    main()

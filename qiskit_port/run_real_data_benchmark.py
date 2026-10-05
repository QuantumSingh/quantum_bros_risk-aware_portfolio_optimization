"""Benchmark 1: original repo dataset subset (Task B).

Tests whether the Qiskit conversion reproduces PennyLane behavior under
the same data/model setup, against classical baselines, on a manageable
subset of the original 15-ticker returns dataset (first 4 assets,
trailing 240 rows).

The original parquet already contains daily RETURNS (values in roughly
[-0.3, 0.2]) despite its 'price_data' file name, so no pct_change is
applied here.

Outputs:
    comparison_logs/real_data_benchmark_results.csv / .tex / .md
    comparison_logs/real_data_benchmark_run.log
    comparison_logs/series/real_data_benchmark_<model>.npz
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

os.chdir(ORIGINAL_CODE)

from qiskit_port import benchmark_lib as bl


CONFIG = dict(
    dataset_name="original_paper_subset_4assets_240rows",
    rows=240,
    num_assets=4,
    lookback_window=10,
    forecast_window=0,
    num_weights=16,
    # Epoch budget and early stopping chosen from the Task D sweep
    # (validation critic loss plateaus early; see parameter_sweep_results.csv).
    max_epochs=25,
    early_stopping=True,
    patience=5,
    min_delta=1e-4,
    seed=68,
    short_selling=True,
    clamp_negatives=True,
)


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "real_data_benchmark_run.log")
    sys.stdout = logger

    try:
        print("Benchmark 1 config:")
        for key, val in CONFIG.items():
            print(f"  {key}: {val}")

        returns = bl.load_original_returns(CONFIG["rows"], CONFIG["num_assets"])
        train_data, val_data, test_data = bl.split_60_20_20(returns)

        bl.run_benchmark(
            CONFIG, train_data, val_data, test_data,
            output_prefix="real_data_benchmark",
        )
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()

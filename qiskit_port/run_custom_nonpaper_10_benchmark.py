"""Benchmark 2: custom_nonpaper_10 universe (Task C).

Ten tickers with zero overlap with the original paper dataset:
NVDA, AMD, TSLA, AMZN, META, NFLX, COST, UNH, BA, KO.

Prices (yfinance adjusted close) are converted to daily returns with
pct_change and validated before any model sees them.

Outputs:
    comparison_logs/custom_nonpaper_10_benchmark_results.csv / .tex / .md
    comparison_logs/custom_nonpaper_10_benchmark_run.log
    comparison_logs/series/custom_nonpaper_10_<model>.npz
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
    dataset_name="custom_nonpaper_10",
    rows=240,
    num_assets=10,
    lookback_window=5,
    forecast_window=0,
    num_weights=10,
    # Epoch budget and early stopping chosen from the Task D sweep
    # (validation critic loss plateaus early; see parameter_sweep_results.csv).
    max_epochs=25,
    # 10-asset config means 10-qubit circuits (~4x deeper state prep);
    # full-budget Qiskit training would take hours per model. Parity with
    # PennyLane is already proven, so Qiskit runs a reduced budget here
    # (noted in the results table) while classical/PennyLane models use
    # the full sweep-justified budget.
    qiskit_max_epochs=5,
    early_stopping=True,
    patience=5,
    min_delta=1e-4,
    seed=68,
    short_selling=True,
    clamp_negatives=True,
)


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "custom_nonpaper_10_benchmark_run.log")
    sys.stdout = logger

    try:
        print("Benchmark 2 config:")
        for key, val in CONFIG.items():
            print(f"  {key}: {val}")

        returns = bl.load_custom_nonpaper_10_returns(CONFIG["rows"])
        train_data, val_data, test_data = bl.split_60_20_20(returns)

        bl.run_benchmark(
            CONFIG, train_data, val_data, test_data,
            output_prefix="custom_nonpaper_10_benchmark",
        )
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()

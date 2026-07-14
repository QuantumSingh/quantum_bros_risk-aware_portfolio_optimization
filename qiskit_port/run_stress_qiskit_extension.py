"""Add the Qiskit models to the historical stress tests.

Runs Qiskit QDPG and Qiskit Quantum Q-Learning through the same three
stress windows as run_stress_and_tail_tests.py, with the reduced
5-epoch Qiskit budget used in Benchmark 2 (parity with PennyLane is the
verified transfer argument; the note lands in the results row).

Appends to comparison_logs/stress_test_results.csv (idempotent: skips
window/model pairs already present).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

os.chdir(ORIGINAL_CODE)

from qiskit_port import benchmark_lib as bl
from qiskit_port.run_stress_and_tail_tests import (
    STRESS_WINDOWS, STRESS_CONFIG, TRAIN_ROWS, VAL_ROWS,
    load_full_original_returns, tail_metrics,
)

QISKIT_MODELS = [
    ("Qiskit QDPG", "ddpg", "qiskit"),
    ("Qiskit Quantum Q-Learning", "dql", "qiskit"),
]


def main():
    csv_path = bl.COMPARISON_DIR / "stress_test_results.csv"
    df = pd.read_csv(csv_path)
    done = set(zip(df["window"], df["model"]))

    returns = load_full_original_returns()
    new_rows = []

    for window_name, (start, end) in STRESS_WINDOWS.items():
        test_data = returns.loc[start:end]
        history = returns.loc[:start].iloc[:-1]
        train_data = history.iloc[-(TRAIN_ROWS + VAL_ROWS):-VAL_ROWS]
        val_data = history.iloc[-VAL_ROWS:]

        config = dict(STRESS_CONFIG, dataset_name=window_name,
                      qiskit_max_epochs=5)

        for display_name, kind, framework in QISKIT_MODELS:
            if (window_name, display_name) in done:
                print(f"skip (cached): {window_name} / {display_name}")
                continue

            row, artifacts = bl.run_single_model(
                display_name, kind, framework, config,
                train_data, val_data, test_data,
            )
            dpo_r = artifacts["dpo"]["daily_returns"]
            row["window"] = window_name
            row["test_start"] = str(test_data.index[0].date())
            row["test_end"] = str(test_data.index[-1].date())
            row.update({f"tail_{k}": v for k, v in tail_metrics(dpo_r).items()})
            new_rows.append(row)

            np.savez_compressed(
                bl.SERIES_DIR / f"stress_{window_name}_{bl.safe_model_name(display_name)}.npz",
                dpo_daily_returns=dpo_r,
                dpo_weights=artifacts["dpo"]["weights_by_day"],
            )
            # Persist incrementally so an interrupted run keeps progress.
            pd.concat([pd.read_csv(csv_path), pd.DataFrame(new_rows[-1:])],
                      ignore_index=True).to_csv(csv_path, index=False)

    final = pd.read_csv(csv_path)
    tex_cols = ["window", "model", "test_start", "test_end",
                "dpo_profit_pa", "dpo_sharpe", "dpo_annualized_return",
                "tail_max_drawdown", "tail_cvar_5", "tail_worst_day",
                "tail_worst_week", "runtime_seconds"]
    final[[c for c in tex_cols if c in final.columns]].to_latex(
        bl.COMPARISON_DIR / "stress_test_results.tex",
        index=False, float_format="%.4f")
    print(final[["window", "model", "dpo_sharpe", "tail_max_drawdown",
                 "tail_cvar_5"]].to_string())


if __name__ == "__main__":
    main()

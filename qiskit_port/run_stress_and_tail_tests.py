"""Stress tests and tail-risk analysis for the paper.

Part 1 — Tail risk on the existing benchmarks:
    computes drawdown/CVaR/skew/kurtosis/worst-period statistics from the
    saved per-model daily-return series of Benchmark 1 and Benchmark 2
    (no retraining).

Part 2 — Historical stress tests:
    retrains the fast models (classical + PennyLane; Qiskit is
    parity-verified equivalent and skipped for runtime) on the 192 rows
    immediately preceding each stress window of the ORIGINAL dataset,
    then evaluates DPO through the window:

        covid_crash_2020 : 2020-02-15 .. 2020-06-30  (crash + rebound)
        bear_2022        : 2022-01-03 .. 2022-12-30  (grinding bear)
        calm_2024        : 2024-01-02 .. 2024-06-28  (control window)

    Model configuration mirrors Benchmark 1 (4 assets, lookback 10,
    16 weights, 25 epochs with patience-5 early stopping, seed 68).

Note: Calmar/max-drawdown/Sortino/CVaR-5% as *headline benchmark
metrics* are teammate-owned; here max drawdown and CVaR appear as parts
of the dedicated tail-risk analysis, alongside additions (CVaR 1%,
VaR 1%, skewness, excess kurtosis, worst day/week, time under water).

Outputs:
    comparison_logs/tail_risk_results.csv / .tex
    comparison_logs/stress_test_results.csv / .tex
    comparison_logs/stress_test_run.log
    comparison_logs/series/stress_<window>_<model>.npz
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

os.chdir(ORIGINAL_CODE)

from qiskit_port import benchmark_lib as bl
from qiskit_port import portfolio_metrics as pm


STRESS_WINDOWS = {
    "covid_crash_2020": ("2020-02-15", "2020-06-30"),
    "bear_2022": ("2022-01-03", "2022-12-30"),
    "calm_2024": ("2024-01-02", "2024-06-28"),
}

STRESS_MODELS = [
    ("Equal Weight", "equal_weight", "classical"),
    ("Mean Variance Optimization", "mvo", "classical"),
    ("Classical DDPG", "ddpg", "classical"),
    ("Classical Deep Q-Learning", "dql", "classical"),
    ("PennyLane QDPG", "ddpg", "pennylane"),
    ("PennyLane Quantum Q-Learning", "dql", "pennylane"),
]

STRESS_CONFIG = dict(
    dataset_name="stress",          # overwritten per window
    lookback_window=10,
    forecast_window=0,
    num_weights=16,
    max_epochs=25,
    early_stopping=True,
    patience=5,
    min_delta=1e-4,
    seed=68,
    short_selling=True,
    clamp_negatives=True,
)

NUM_ASSETS = 4
TRAIN_ROWS = 144
VAL_ROWS = 48


# ---------------------------------------------------------------------------
# Tail-risk statistics
# ---------------------------------------------------------------------------

def max_drawdown(daily_returns) -> float:
    """Largest peak-to-trough loss of the compounded wealth curve,
    reported as a positive fraction."""
    wealth = np.cumprod(1.0 + np.asarray(daily_returns, dtype=np.float64))
    running_peak = np.maximum.accumulate(wealth)
    drawdowns = 1.0 - wealth / running_peak
    return float(np.max(drawdowns))


def time_under_water(daily_returns) -> float:
    """Fraction of test days spent below the previous wealth peak."""
    wealth = np.cumprod(1.0 + np.asarray(daily_returns, dtype=np.float64))
    running_peak = np.maximum.accumulate(wealth)
    return float(np.mean(wealth < running_peak))


def var_alpha(daily_returns, alpha: float) -> float:
    """Historical daily VaR at level alpha (positive = loss)."""
    return float(-np.percentile(np.asarray(daily_returns), 100 * alpha))


def cvar_alpha(daily_returns, alpha: float) -> float:
    """Expected shortfall: mean loss on the worst alpha-fraction of days
    (positive = loss). CVaR >= VaR always."""
    r = np.asarray(daily_returns, dtype=np.float64)
    cutoff = np.percentile(r, 100 * alpha)
    tail = r[r <= cutoff]
    if len(tail) == 0:
        return float(-cutoff)
    return float(-np.mean(tail))


def worst_rolling(daily_returns, window: int) -> float:
    """Worst compounded return over any consecutive `window` days."""
    r = np.asarray(daily_returns, dtype=np.float64)
    if len(r) < window:
        return float(np.prod(1.0 + r) - 1.0)
    wealth = np.cumprod(1.0 + r)
    wealth = np.concatenate([[1.0], wealth])
    rolling = wealth[window:] / wealth[:-window] - 1.0
    return float(np.min(rolling))


def tail_metrics(daily_returns) -> dict:
    r = np.asarray(daily_returns, dtype=np.float64)
    return {
        "max_drawdown": max_drawdown(r),
        "time_under_water": time_under_water(r),
        "var_5": var_alpha(r, 0.05),
        "cvar_5": cvar_alpha(r, 0.05),
        "var_1": var_alpha(r, 0.01),
        "cvar_1": cvar_alpha(r, 0.01),
        "worst_day": float(np.min(r)),
        "worst_week": worst_rolling(r, 5),
        "skewness": float(sps.skew(r)),
        "excess_kurtosis": float(sps.kurtosis(r)),  # Fisher: normal = 0
        "num_days": int(len(r)),
    }


# ---------------------------------------------------------------------------
# Part 1: tail risk of the existing benchmark results
# ---------------------------------------------------------------------------

def tail_risk_from_saved_series() -> pd.DataFrame:
    rows = []
    for prefix, label in [
        ("real_data_benchmark", "benchmark1_original"),
        ("custom_nonpaper_10_benchmark", "benchmark2_custom_nonpaper_10"),
    ]:
        for display_name, _kind, _fw, _algo, _enc in bl.MODEL_SPECS:
            safe = bl.safe_model_name(display_name)
            npz_path = bl.SERIES_DIR / f"{prefix}_{safe}.npz"
            if not npz_path.exists():
                continue
            data = np.load(npz_path)
            r = data["dpo_daily_returns"]
            row = {"benchmark": label, "model": display_name}
            row.update(tail_metrics(r))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Part 2: historical stress windows
# ---------------------------------------------------------------------------

def load_full_original_returns() -> pd.DataFrame:
    returns = pd.read_parquet(ORIGINAL_CODE / "data" / "price_data.parquet.gzip")
    return returns.iloc[:, :NUM_ASSETS]


def run_stress_window(window_name, start, end, returns) -> list:
    print("\n" + "=" * 80)
    print(f"STRESS WINDOW {window_name}: {start} .. {end}")
    print("=" * 80)

    test_data = returns.loc[start:end]
    history = returns.loc[:start].iloc[:-1]  # strictly before the window

    if len(history) < TRAIN_ROWS + VAL_ROWS:
        raise ValueError(f"Not enough history before {start}")

    train_data = history.iloc[-(TRAIN_ROWS + VAL_ROWS):-VAL_ROWS]
    val_data = history.iloc[-VAL_ROWS:]

    print(f"train {train_data.index[0].date()} .. {train_data.index[-1].date()} "
          f"({len(train_data)} rows)")
    print(f"val   {val_data.index[0].date()} .. {val_data.index[-1].date()} "
          f"({len(val_data)} rows)")
    print(f"test  {test_data.index[0].date()} .. {test_data.index[-1].date()} "
          f"({len(test_data)} rows)")

    config = dict(STRESS_CONFIG, dataset_name=window_name)

    rows = []
    for display_name, kind, framework in STRESS_MODELS:
        row, artifacts = bl.run_single_model(
            display_name, kind, framework, config,
            train_data, val_data, test_data,
        )
        dpo_r = artifacts["dpo"]["daily_returns"]

        row["window"] = window_name
        row["test_start"] = str(test_data.index[0].date())
        row["test_end"] = str(test_data.index[-1].date())
        row.update({f"tail_{k}": v for k, v in tail_metrics(dpo_r).items()})
        rows.append(row)

        np.savez_compressed(
            bl.SERIES_DIR / f"stress_{window_name}_{bl.safe_model_name(display_name)}.npz",
            dpo_daily_returns=dpo_r,
            dpo_weights=artifacts["dpo"]["weights_by_day"],
        )
    return rows


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "stress_test_run.log")
    sys.stdout = logger
    try:
        # Part 1
        tail_df = tail_risk_from_saved_series()
        tail_csv = bl.COMPARISON_DIR / "tail_risk_results.csv"
        tail_df.to_csv(tail_csv, index=False)
        tail_df.to_latex(bl.COMPARISON_DIR / "tail_risk_results.tex",
                         index=False, float_format="%.4f")
        print("Tail-risk table (existing benchmarks):")
        print(tail_df.to_string())

        # Part 2
        returns = load_full_original_returns()
        all_rows = []
        for window_name, (start, end) in STRESS_WINDOWS.items():
            all_rows.extend(run_stress_window(window_name, start, end, returns))

        stress_df = pd.DataFrame(all_rows)
        stress_csv = bl.COMPARISON_DIR / "stress_test_results.csv"
        stress_df.to_csv(stress_csv, index=False)

        tex_cols = ["window", "model", "test_start", "test_end",
                    "dpo_profit_pa", "dpo_sharpe", "dpo_annualized_return",
                    "tail_max_drawdown", "tail_cvar_5", "tail_worst_day",
                    "tail_worst_week", "runtime_seconds"]
        stress_df[[c for c in tex_cols if c in stress_df.columns]].to_latex(
            bl.COMPARISON_DIR / "stress_test_results.tex",
            index=False, float_format="%.4f")

        print("\nStress-test summary:")
        print(stress_df[["window", "model", "dpo_sharpe",
                         "tail_max_drawdown", "tail_cvar_5",
                         "tail_worst_day"]].to_string())
        print("\nSaved:", tail_csv.name, "and", stress_csv.name)
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()

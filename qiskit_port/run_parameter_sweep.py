"""Training/data/epoch sweet-spot experiments (Task D).

Sweeps one axis at a time around a base configuration, using the
PennyLane QDPG model (the quantum model of interest; PennyLane is fast
enough for sweeps, and Qiskit is parity-verified equivalent, so the
plateau/runtime conclusions transfer).

Axes:
    rows        120, 240, 600, 1768 (~50% of the 3536-row original set)
    lookback    5, 10
    epochs      10, 25, 50, 100 without early stopping (plateau curves)
                + one early-stopping run (patience 5, min_delta 1e-4)
    assets      4 (original subset), 10 (custom_nonpaper_10)
    batch_size  1 (baseline) and 4 (expected unsupported; documented)

Outputs:
    comparison_logs/parameter_sweep_results.csv / .tex
    comparison_logs/parameter_sweep_run.log
    comparison_logs/series/sweep_<run_id>.npz  (loss curves)
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

os.chdir(ORIGINAL_CODE)

from qiskit_port import benchmark_lib as bl
from qiskit_port import portfolio_metrics as pm


BASE = dict(
    dataset="original_paper_subset",
    rows=240,
    num_assets=4,
    lookback_window=5,
    num_weights=8,
    batch_size=1,
    max_epochs=50,
    early_stopping=True,
    patience=5,
    min_delta=1e-4,
    seed=68,
)


def load_data(cfg):
    if cfg["dataset"] == "original_paper_subset":
        returns = bl.load_original_returns(cfg["rows"], cfg["num_assets"])
    elif cfg["dataset"] == "custom_nonpaper_10":
        returns = bl.load_custom_nonpaper_10_returns(cfg["rows"])
    else:
        raise ValueError(cfg["dataset"])
    return bl.split_60_20_20(returns)


def run_one(run_id: str, cfg: dict) -> dict:
    print("\n" + "=" * 80)
    print(f"SWEEP RUN {run_id}: " + ", ".join(
        f"{k}={v}" for k, v in cfg.items() if k != "seed"))
    print("=" * 80)

    record = {
        "run_id": run_id,
        "dataset": cfg["dataset"],
        "rows": cfg["rows"],
        "assets": cfg["num_assets"],
        "lookback_window": cfg["lookback_window"],
        "num_weights": cfg["num_weights"],
        "batch_size": cfg["batch_size"],
        "max_epochs": cfg["max_epochs"],
        "early_stopping": cfg["early_stopping"],
        "epochs_completed": None,
        "early_stopping_triggered": None,
        "final_actor_loss": None,
        "final_critic_loss": None,
        "final_val_critic_loss": None,
        "runtime_seconds": None,
        "spo_profit_pa": None,
        "spo_sharpe": None,
        "dpo_profit_pa": None,
        "dpo_sharpe": None,
        "dpo_annualized_return": None,
        "dpo_annualized_volatility": None,
        "dpo_var_5": None,
        "dpo_avg_turnover": None,
        "dpo_tc_adjusted_annual_return": None,
        "status": "ok",
    }

    try:
        train_data, val_data, test_data = load_data(cfg)

        torch.manual_seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        model_config = dict(
            lookback_window=cfg["lookback_window"],
            num_weights=cfg["num_weights"],
            seed=cfg["seed"],
            short_selling=True,
            clamp_negatives=True,
        )

        from models import DDPG
        from predictors import QuantumNeuralNetwork
        from predictors.input_transformations import radial_to_linear

        model = DDPG(
            lookback_window=cfg["lookback_window"],
            forecast_window=0,
            batch_size=cfg["batch_size"],
            predictor=QuantumNeuralNetwork,
            num_weights=cfg["num_weights"],
            encoding="amplitude",
            input_transformation=radial_to_linear,
            rotation_axes="y",
            short_selling=True,
            reduce_negatives=True,
            verbose=1,
            seed=cfg["seed"],
        )

        train_kwargs = dict(bl.HYPERPARAMS["ddpg_quantum"])
        train_kwargs.update(
            num_epochs=cfg["max_epochs"],
            early_stopping=cfg["early_stopping"],
            patience=cfg["patience"],
            min_delta=cfg["min_delta"],
        )

        start = time.time()
        text, history = bl.train_with_loss_capture(
            model, train_data, val_data, train_kwargs
        )
        train_runtime = time.time() - start

        epochs_completed = len(history["epoch"])
        record["epochs_completed"] = epochs_completed
        record["early_stopping_triggered"] = (
            cfg["early_stopping"] and epochs_completed < cfg["max_epochs"]
        )
        if epochs_completed:
            record["final_actor_loss"] = history["actor_loss"][-1]
            record["final_critic_loss"] = history["critic_loss"][-1]
            val_losses = [v for v in history["val_critic_loss"] if not np.isnan(v)]
            record["final_val_critic_loss"] = val_losses[-1] if val_losses else None

        spo = pm.evaluate_actor_spo_with_weights(model.actor, val_data, test_data)
        dpo = pm.evaluate_actor_with_weights(
            model.actor, val_data, test_data, interval=cfg["lookback_window"]
        )
        spo_m = pm.compute_all_metrics(spo["daily_returns"], spo["weights_by_day"])
        dpo_m = pm.compute_all_metrics(dpo["daily_returns"], dpo["weights_by_day"])

        record["runtime_seconds"] = train_runtime
        record["spo_profit_pa"] = spo_m["profit_pa"]
        record["spo_sharpe"] = spo_m["sharpe"]
        record["dpo_profit_pa"] = dpo_m["profit_pa"]
        record["dpo_sharpe"] = dpo_m["sharpe"]
        record["dpo_annualized_return"] = dpo_m["annualized_return"]
        record["dpo_annualized_volatility"] = dpo_m["annualized_volatility"]
        record["dpo_var_5"] = dpo_m["var_5"]
        record["dpo_avg_turnover"] = dpo_m["avg_turnover"]
        record["dpo_tc_adjusted_annual_return"] = dpo_m["tc_adjusted_annual_return"]

        np.savez_compressed(
            bl.SERIES_DIR / f"sweep_{run_id}.npz",
            actor_loss=np.array(history["actor_loss"]),
            critic_loss=np.array(history["critic_loss"]),
            val_critic_loss=np.array(history["val_critic_loss"]),
        )

        print(f"epochs {epochs_completed}/{cfg['max_epochs']}"
              f"{' (early stop)' if record['early_stopping_triggered'] else ''}, "
              f"train {train_runtime:.1f}s, "
              f"DPO profit {dpo_m['profit_pa']*100:.2f}%, sharpe {dpo_m['sharpe']:.3f}")

    except Exception as exc:
        record["status"] = f"FAILED: {type(exc).__name__}: {exc}"
        print("RUN FAILED:", record["status"])

    return record


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "parameter_sweep_run.log")
    sys.stdout = logger

    try:
        records = []

        # Axis 1: data range (rows), early stopping on
        for rows in [120, 240, 600, 1768]:
            cfg = dict(BASE, rows=rows)
            records.append(run_one(f"rows_{rows}", cfg))

        # Axis 2: lookback window
        cfg = dict(BASE, lookback_window=10)
        records.append(run_one("lookback_10", cfg))

        # Axis 3: epochs without early stopping (plateau observation)
        for epochs in [10, 25, 50, 100]:
            cfg = dict(BASE, max_epochs=epochs, early_stopping=False)
            records.append(run_one(f"epochs_{epochs}_noES", cfg))

        # Early stopping reference run at the largest epoch budget
        cfg = dict(BASE, max_epochs=100, early_stopping=True)
        records.append(run_one("epochs_100_ES", cfg))

        # Axis 4: asset count (custom non-paper 10-ticker universe)
        cfg = dict(BASE, dataset="custom_nonpaper_10", num_assets=10,
                   num_weights=10)
        records.append(run_one("assets_10_custom", cfg))

        # Axis 5: batch size (expected unsupported by the original
        # window-by-window training loop; documented as a finding)
        cfg = dict(BASE, batch_size=4, max_epochs=2)
        records.append(run_one("batch_4_probe", cfg))

        df = pd.DataFrame(records)

        csv_path = bl.COMPARISON_DIR / "parameter_sweep_results.csv"
        tex_path = bl.COMPARISON_DIR / "parameter_sweep_results.tex"
        df.to_csv(csv_path, index=False)

        tex_cols = ["run_id", "dataset", "rows", "assets", "lookback_window",
                    "max_epochs", "epochs_completed", "early_stopping_triggered",
                    "final_val_critic_loss", "runtime_seconds",
                    "dpo_profit_pa", "dpo_sharpe", "status"]
        df[[c for c in tex_cols if c in df.columns]].to_latex(
            tex_path, index=False, float_format="%.4f")

        print("\n" + "=" * 80)
        print(df.to_string())
        print("\nSaved:", csv_path, "and", tex_path)
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()

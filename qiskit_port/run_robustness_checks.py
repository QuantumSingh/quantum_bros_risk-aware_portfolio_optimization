"""Robustness / anti-artifact checks for the benchmark results.

Answers "is this the model, or luck/manipulation?" with five tests, all
on the Benchmark 1 configuration (original data, 4 assets, 240 rows,
lookback 10, 16 weights, 25 epochs + patience-5 early stopping):

  A. multi-seed        10 seeds x {PennyLane QDPG, Classical DDPG}:
                       distribution of test DPO Sharpe (is seed 68 typical?)
  B. placebo           train on time-SHUFFLED train/val rows (temporal
                       structure destroyed, cross-sectional intact),
                       evaluate on the real test window. If performance
                       survives, results are artifacts.
  C. random policies   2000 random Dirichlet weight schedules rebalanced
                       every 10 days on the real test window: the "no
                       skill" null distribution; report trained models'
                       percentile within it.
  D. untrained actors  20 freshly initialized quantum actors, no
                       training: the architecture-prior null.
  E. walk-forward      10 consecutive non-overlapping 240-row folds
                       (rolling origin through ~2016-2025), 60/20/20
                       inside each: does the ranking hold across
                       regimes, or only in the headline window?

Outputs:
    comparison_logs/robustness_checks.csv           (every run)
    comparison_logs/robustness_summary.csv / .tex   (aggregates)
    comparison_logs/robustness_run.log
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

CONFIG = dict(
    dataset_name="robustness_bench1_config",
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
ROWS = 240
SEEDS = [68, 1, 2, 3, 4, 5, 6, 7, 8, 9]
DPO_INTERVAL = CONFIG["lookback_window"]


def load_bench1_slice(returns_full, end_idx=None):
    r = returns_full if end_idx is None else returns_full.iloc[:end_idx]
    return r.tail(ROWS)


def split(returns):
    n = len(returns)
    return (returns.iloc[:int(0.6 * n)],
            returns.iloc[int(0.6 * n):int(0.8 * n)],
            returns.iloc[int(0.8 * n):])


def train_and_eval(kind, framework, seed, train_data, val_data, test_data,
                   config=None):
    """Train one RL model, return DPO metrics dict."""
    cfg = dict(config or CONFIG, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = bl.build_rl_model(kind, framework, cfg)
    train_kwargs = bl.make_train_kwargs(kind, framework, cfg)

    start = time.time()
    _, history = bl.train_with_loss_capture(model, train_data, val_data,
                                            train_kwargs)
    runtime = time.time() - start

    dpo = pm.evaluate_actor_with_weights(
        model.actor, val_data, test_data, interval=DPO_INTERVAL)
    metrics = pm.compute_all_metrics(dpo["daily_returns"], dpo["weights_by_day"])
    metrics["epochs_completed"] = len(history["epoch"])
    metrics["runtime_seconds"] = runtime
    return metrics


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "robustness_run.log")
    sys.stdout = logger
    rows = []
    try:
        returns_full = pd.read_parquet(
            ORIGINAL_CODE / "data" / "price_data.parquet.gzip"
        ).iloc[:, :NUM_ASSETS]

        bench = load_bench1_slice(returns_full)
        train_data, val_data, test_data = split(bench)

        # ------------------------------------------------------------
        # A. Multi-seed
        # ------------------------------------------------------------
        print("\n=== A. Multi-seed (10 seeds) ===")
        for label, kind, fw in [("PennyLane QDPG", "ddpg", "pennylane"),
                                ("Classical DDPG", "ddpg", "classical")]:
            for seed in SEEDS:
                m = train_and_eval(kind, fw, seed, train_data, val_data,
                                   test_data)
                rows.append(dict(test="A_multiseed", model=label, seed=seed,
                                 **m))
                print(f"{label} seed {seed}: sharpe {m['sharpe']:.3f} "
                      f"profit {m['profit_pa']*100:.1f}% "
                      f"({m['epochs_completed']} ep)")

        ew = pm.constant_weights_evaluation(test_data,
                                            np.ones(NUM_ASSETS) / NUM_ASSETS)
        ew_m = pm.compute_all_metrics(ew["daily_returns"], ew["weights_by_day"])
        rows.append(dict(test="A_multiseed", model="Equal Weight", seed=-1,
                         **ew_m))
        print(f"Equal Weight (deterministic): sharpe {ew_m['sharpe']:.3f}")

        # ------------------------------------------------------------
        # B. Placebo: time-shuffled training data
        # ------------------------------------------------------------
        print("\n=== B. Placebo (time-shuffled train/val) ===")
        for seed in [68, 1, 2]:
            rng = np.random.default_rng(seed)
            train_shuf = train_data.iloc[rng.permutation(len(train_data))]
            val_shuf = val_data.iloc[rng.permutation(len(val_data))]
            m = train_and_eval("ddpg", "pennylane", seed, train_shuf,
                               val_shuf, test_data)
            rows.append(dict(test="B_placebo", model="PennyLane QDPG",
                             seed=seed, **m))
            print(f"placebo seed {seed}: sharpe {m['sharpe']:.3f} "
                  f"profit {m['profit_pa']*100:.1f}%")

        # ------------------------------------------------------------
        # C. Random-policy null (Dirichlet weights, rebalanced like DPO)
        # ------------------------------------------------------------
        print("\n=== C. Random-policy null (2000 draws) ===")
        rng = np.random.default_rng(0)
        n_days = len(test_data)
        n_intervals = int(np.ceil(n_days / DPO_INTERVAL))
        sharpes = np.empty(2000)
        for i in range(2000):
            w_sched = rng.dirichlet(np.ones(NUM_ASSETS), size=n_intervals)
            w_daily = np.repeat(w_sched, DPO_INTERVAL, axis=0)[:n_days]
            daily = np.sum(test_data.values * w_daily, axis=1)
            sharpes[i] = pm.repo_sharpe(daily)
        for q in [5, 25, 50, 75, 95]:
            print(f"random-policy sharpe p{q}: {np.percentile(sharpes, q):.3f}")
        rows.append(dict(test="C_random_null", model="random_policies",
                         seed=-1, sharpe=float(np.median(sharpes)),
                         profit_pa=np.nan,
                         null_p5=float(np.percentile(sharpes, 5)),
                         null_p95=float(np.percentile(sharpes, 95))))
        np.save(bl.SERIES_DIR / "robustness_random_null_sharpes.npy", sharpes)

        # ------------------------------------------------------------
        # D. Untrained quantum actors (architecture prior)
        # ------------------------------------------------------------
        print("\n=== D. Untrained quantum actors (20 inits) ===")
        from predictors import QuantumNeuralNetwork
        from predictors.input_transformations import radial_to_linear
        activation = lambda x: x / torch.sum(x, dim=-1, keepdim=True)
        untrained = []
        for seed in range(20):
            torch.manual_seed(seed)
            np.random.seed(seed)
            actor = QuantumNeuralNetwork(
                input_size=NUM_ASSETS * CONFIG["lookback_window"],
                output_size=NUM_ASSETS,
                num_weights=CONFIG["num_weights"],
                encoding="amplitude",
                input_transformation=radial_to_linear,
                rotation_axes="y",
                output_activation=activation,
                seed=seed,
            )
            dpo = pm.evaluate_actor_with_weights(actor, val_data, test_data,
                                                 interval=DPO_INTERVAL)
            m = pm.compute_all_metrics(dpo["daily_returns"],
                                       dpo["weights_by_day"])
            untrained.append(m["sharpe"])
            rows.append(dict(test="D_untrained", model="Untrained QNN actor",
                             seed=seed, **m))
        print(f"untrained sharpe: mean {np.mean(untrained):.3f} "
              f"std {np.std(untrained):.3f} "
              f"range [{np.min(untrained):.3f}, {np.max(untrained):.3f}]")

        # ------------------------------------------------------------
        # E. Walk-forward folds (rolling origin)
        # ------------------------------------------------------------
        print("\n=== E. Walk-forward folds (10 x 240 rows) ===")
        n_folds = 10
        total = len(returns_full)
        for fold in range(n_folds):
            end_idx = total - (n_folds - 1 - fold) * ROWS
            fold_slice = load_bench1_slice(returns_full, end_idx=end_idx)
            ftr, fva, fte = split(fold_slice)
            fold_label = f"{fte.index[0].date()}..{fte.index[-1].date()}"

            few = pm.constant_weights_evaluation(
                fte, np.ones(NUM_ASSETS) / NUM_ASSETS)
            few_m = pm.compute_all_metrics(few["daily_returns"],
                                           few["weights_by_day"])
            rows.append(dict(test="E_walkforward", model="Equal Weight",
                             seed=-1, fold=fold, fold_test=fold_label,
                             **few_m))

            for label, kind, fw in [("PennyLane QDPG", "ddpg", "pennylane"),
                                    ("Classical DDPG", "ddpg", "classical")]:
                m = train_and_eval(kind, fw, 68, ftr, fva, fte)
                rows.append(dict(test="E_walkforward", model=label, seed=68,
                                 fold=fold, fold_test=fold_label, **m))
            last = [r for r in rows if r.get("fold") == fold]
            print(f"fold {fold} ({fold_label}): " + "  ".join(
                f"{r['model'].split()[0]}={r['sharpe']:.2f}" for r in last))

        # ------------------------------------------------------------
        # Exports + aggregates
        # ------------------------------------------------------------
        df = pd.DataFrame(rows)
        df.to_csv(bl.COMPARISON_DIR / "robustness_checks.csv", index=False)

        seed_df = df[df.test == "A_multiseed"]
        wf = df[df.test == "E_walkforward"]
        summary = []
        for label in ["PennyLane QDPG", "Classical DDPG"]:
            s = seed_df[seed_df.model == label]["sharpe"]
            summary.append(dict(
                metric=f"{label} multi-seed sharpe",
                mean=s.mean(), std=s.std(), minimum=s.min(), maximum=s.max()))
        s = df[df.test == "B_placebo"]["sharpe"]
        summary.append(dict(metric="Placebo (shuffled) sharpe",
                            mean=s.mean(), std=s.std(),
                            minimum=s.min(), maximum=s.max()))
        s = df[df.test == "D_untrained"]["sharpe"]
        summary.append(dict(metric="Untrained actor sharpe",
                            mean=s.mean(), std=s.std(),
                            minimum=s.min(), maximum=s.max()))
        for label in ["Equal Weight", "PennyLane QDPG", "Classical DDPG"]:
            s = wf[wf.model == label]["sharpe"]
            summary.append(dict(metric=f"{label} walk-forward sharpe (10 folds)",
                                mean=s.mean(), std=s.std(),
                                minimum=s.min(), maximum=s.max()))
        sdf = pd.DataFrame(summary)
        sdf.to_csv(bl.COMPARISON_DIR / "robustness_summary.csv", index=False)
        sdf.to_latex(bl.COMPARISON_DIR / "robustness_summary.tex",
                     index=False, float_format="%.4f")
        print("\n" + sdf.to_string())
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()

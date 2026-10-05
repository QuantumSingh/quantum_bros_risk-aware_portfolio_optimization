"""Statistical stability and reproducibility tests (Benchmark 1 config).

Two questions:

  A. REPRODUCIBILITY (is the code deterministic?)
     Train each model twice with the SAME seed and check the test DPO
     Sharpe is bit-identical. A non-zero difference means hidden
     nondeterminism (unseeded RNG, dict ordering, etc.).

  B. STABILITY (how much does the result move across seeds?)
     Train each model at N seeds and report mean, sample std, coefficient
     of variation (std/|mean|), and a 95% CI of the mean (t-based). High
     CV means the single-seed headline number is not representative.

Fast models only (classical + PennyLane); Qiskit is parity-verified
equivalent per-call and skipped for runtime.

Outputs:
    comparison_logs/stability_reproducibility.csv   (determinism check)
    comparison_logs/stability_seeds.csv             (per-seed raw)
    comparison_logs/stability_summary.csv / .tex    (aggregates)
    comparison_logs/stability_run.log
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))
os.chdir(ORIGINAL_CODE)

import torch
from qiskit_port import benchmark_lib as bl
from qiskit_port import portfolio_metrics as pm

MODELS = [
    ("Classical DDPG", "ddpg", "classical"),
    ("Classical Deep Q-Learning", "dql", "classical"),
    ("PennyLane QDPG", "ddpg", "pennylane"),
    ("PennyLane Quantum Q-Learning", "dql", "pennylane"),
]
SEEDS = [68, 1, 2, 3, 4]
NUM_ASSETS, ROWS, DPO_INTERVAL = 4, 240, 10

CONFIG = dict(
    dataset_name="stability", lookback_window=10, forecast_window=0,
    num_weights=16, max_epochs=25, early_stopping=True, patience=5,
    min_delta=1e-4, short_selling=True, clamp_negatives=True,
)


def train_eval(kind, framework, seed, tr, va, te):
    cfg = dict(CONFIG, seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = bl.build_rl_model(kind, framework, cfg)
    _, hist = bl.train_with_loss_capture(
        model, tr, va, bl.make_train_kwargs(kind, framework, cfg))
    dpo = pm.evaluate_actor_with_weights(model.actor, va, te, interval=DPO_INTERVAL)
    m = pm.compute_all_metrics(dpo["daily_returns"], dpo["weights_by_day"])
    m["epochs"] = len(hist["epoch"])
    return m


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "stability_run.log")
    sys.stdout = logger
    try:
        returns = bl.load_original_returns(ROWS, NUM_ASSETS)
        tr, va, te = bl.split_60_20_20(returns)

        # ---- A. Reproducibility: same seed twice ----
        print("\n=== A. Reproducibility (same seed 68, run twice) ===")
        repro = []
        for name, kind, fw in MODELS:
            m1 = train_eval(kind, fw, 68, tr, va, te)
            m2 = train_eval(kind, fw, 68, tr, va, te)
            d_sharpe = abs(m1["sharpe"] - m2["sharpe"])
            d_ret = abs(m1["annualized_return"] - m2["annualized_return"])
            ok = d_sharpe < 1e-9
            repro.append(dict(model=name, run1_sharpe=m1["sharpe"],
                              run2_sharpe=m2["sharpe"], abs_diff_sharpe=d_sharpe,
                              abs_diff_ann_return=d_ret,
                              deterministic=bool(ok)))
            print(f"{name:<30} run1={m1['sharpe']:.6f} run2={m2['sharpe']:.6f} "
                  f"|diff|={d_sharpe:.2e}  {'DETERMINISTIC' if ok else 'NONDETERMINISTIC'}")
        pd.DataFrame(repro).to_csv(
            bl.COMPARISON_DIR / "stability_reproducibility.csv", index=False)

        # ---- B. Stability across seeds ----
        print(f"\n=== B. Stability across {len(SEEDS)} seeds ===")
        raw = []
        for name, kind, fw in MODELS:
            for seed in SEEDS:
                t0 = time.time()
                m = train_eval(kind, fw, seed, tr, va, te)
                raw.append(dict(model=name, seed=seed, sharpe=m["sharpe"],
                                ann_return=m["annualized_return"],
                                profit_pa=m["profit_pa"], var_5=m["var_5"],
                                turnover=m["avg_turnover"], epochs=m["epochs"],
                                runtime_s=time.time() - t0))
            print(f"  {name}: done {len(SEEDS)} seeds")
        raw_df = pd.DataFrame(raw)
        raw_df.to_csv(bl.COMPARISON_DIR / "stability_seeds.csv", index=False)

        summary = []
        for name, _, _ in MODELS:
            s = raw_df[raw_df.model == name]["sharpe"].values
            n = len(s)
            mean, sd = s.mean(), s.std(ddof=1)
            cv = sd / abs(mean) if mean != 0 else np.nan
            tcrit = sps.t.ppf(0.975, n - 1)
            ci = tcrit * sd / np.sqrt(n)
            summary.append(dict(model=name, n=n, sharpe_mean=round(mean, 3),
                                sharpe_std=round(sd, 3), cv=round(cv, 3),
                                ci95_halfwidth=round(ci, 3),
                                sharpe_min=round(s.min(), 3),
                                sharpe_max=round(s.max(), 3)))
        sdf = pd.DataFrame(summary)
        sdf.to_csv(bl.COMPARISON_DIR / "stability_summary.csv", index=False)
        sdf.to_latex(bl.COMPARISON_DIR / "stability_summary.tex",
                     index=False, float_format="%.3f")
        print("\n" + sdf.to_string(index=False))
        print("\nCV = std/|mean|: >0.5 means the single-seed headline is "
              "not representative.")
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()

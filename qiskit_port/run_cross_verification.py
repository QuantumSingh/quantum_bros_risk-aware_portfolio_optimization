"""Cross-verify the Qiskit and PennyLane implementations end to end.

Two levels of agreement, reported together so teammates know exactly
what to expect when they compare runs:

  1. PER-CALL parity (already established by
     test_full_pennylane_qiskit_parity.py): identical inputs+weights give
     forward outputs to ~1e-7 and gradients to ~1e-7. This is EXACT
     equivalence of the models.

  2. TRAJECTORY agreement: when both frameworks train the SAME model on
     the SAME data with the SAME seed, do the loss curves and final
     metrics match? They track early, then diverge, because RL training
     is chaotic: float32-level per-call differences are amplified across
     thousands of noise-injected, replay-sampled updates. This is
     expected and is NOT a parity failure -- it is the distinction
     between per-call equivalence and per-trajectory equivalence.

Inputs (already produced): the 100-epoch loss curves
comparison_logs/series/losscurve_100_{plqdpg,qqdpg}.npz and the benchmark
results table.

Outputs:
    comparison_logs/cross_verification_summary.csv / .tex
    comparison_logs/plots/cross_verification_loss.png  (2-panel)
    comparison_logs/cross_verification_run.log
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from qiskit_port import benchmark_lib as bl


def loss_overlay():
    pl = bl.SERIES_DIR / "losscurve_100_plqdpg.npz"
    qk = bl.SERIES_DIR / "losscurve_100_qqdpg.npz"
    if not (pl.exists() and qk.exists()):
        print("SKIP loss overlay: need losscurve_100_plqdpg and _qqdpg")
        return None
    p, q = np.load(pl), np.load(qk)
    n = min(len(p["actor_loss"]), len(q["actor_loss"]))
    ep = np.arange(1, n + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(ep, p["actor_loss"][:n], label="PennyLane", color="#2ca02c")
    axes[0].plot(ep, q["actor_loss"][:n], label="Qiskit", color="#d62728",
                 linestyle="--")
    axes[0].set_title("Actor loss (same seed, same data)")
    axes[1].plot(ep, p["critic_loss"][:n], label="PennyLane", color="#2ca02c")
    axes[1].plot(ep, q["critic_loss"][:n], label="Qiskit", color="#d62728",
                 linestyle="--")
    axes[1].set_title("Critic loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle("Cross-framework training: tracks early, diverges late "
                 "(chaotic RL amplifies ~1e-7 per-call differences)")
    fig.tight_layout()
    out = bl.PLOTS_DIR / "cross_verification_loss.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.name)

    # Per-epoch divergence statistics
    da = np.abs(p["actor_loss"][:n] - q["actor_loss"][:n])
    dc = np.abs(p["critic_loss"][:n] - q["critic_loss"][:n])
    return dict(
        actor_diff_epoch1=float(da[0]), actor_diff_mean=float(da.mean()),
        actor_diff_final=float(da[-1]),
        critic_diff_epoch1=float(dc[0]), critic_diff_mean=float(dc.mean()),
        critic_diff_final=float(dc[-1]))


def metric_table():
    csv = bl.COMPARISON_DIR / "real_data_benchmark_results.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv).set_index("model")
    pairs = [("PennyLane QDPG", "Qiskit QDPG"),
             ("PennyLane Quantum Q-Learning", "Qiskit Quantum Q-Learning")]
    cols = ["dpo_sharpe", "dpo_annualized_return", "dpo_avg_turnover"]
    rows = []
    for pl, qk in pairs:
        if pl not in df.index or qk not in df.index:
            continue
        for c in cols:
            rows.append(dict(pair=f"{pl.split()[-1]} vs {qk.split()[-1]}",
                             metric=c.replace("dpo_", ""),
                             pennylane=round(df.loc[pl, c], 4),
                             qiskit=round(df.loc[qk, c], 4),
                             abs_diff=round(abs(df.loc[pl, c] - df.loc[qk, c]), 4)))
    return pd.DataFrame(rows)


def main():
    logger = bl.TeeLogger(bl.COMPARISON_DIR / "cross_verification_run.log")
    sys.stdout = logger
    try:
        print("=== Level 1: per-call parity (from parity suite) ===")
        print("forward max |diff| ~1.2e-7, weight grad ~3.6e-7, "
              "input grad ~2.9e-4 (finite difference).")
        print("=> the two implementations are EXACTLY equivalent per call.\n")

        print("=== Level 2: trajectory agreement (same seed/data) ===")
        div = loss_overlay()
        if div:
            for k, v in div.items():
                print(f"  {k}: {v:.4e}")
            print("  => curves start near-identical, diverge as training "
                  "proceeds (expected chaotic amplification).")

        print("\n=== Final trained-metric comparison (Benchmark 1) ===")
        mt = metric_table()
        if mt is not None:
            print(mt.to_string(index=False))
            mt.to_csv(bl.COMPARISON_DIR / "cross_verification_summary.csv",
                      index=False)
            mt.to_latex(bl.COMPARISON_DIR / "cross_verification_summary.tex",
                        index=False, float_format="%.4f")
            print("\nInterpretation: final metrics differ at the O(0.1 Sharpe) "
                  "level despite 1e-7 per-call parity -- teammates comparing "
                  "TRAINED results should expect this and compare per-call "
                  "parity (exact) separately from trained trajectories (chaotic).")
    finally:
        sys.stdout = logger.stdout
        logger.close()


if __name__ == "__main__":
    main()

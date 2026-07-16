"""Dedicated long training-loss-curve run (Benchmark 1 config).

The benchmark's loss curves are truncated: they use 25 epochs WITH early
stopping, so models that converge fast (e.g. Classical Deep Q-Learning)
stop after ~6 epochs and their curves look incomplete. This script
retrains with early stopping OFF at a fixed, longer budget so every
model shows its full trajectory.

Usage:
    PYTHONPATH=. python qiskit_port/run_training_loss_curves.py [budget] [model_keys...]

    budget       epoch budget (default 100)
    model_keys   subset of: cddpg cdql plqdpg plqql qqdpg qqql
                 (default: the 4 fast non-Qiskit models)

Resumable: each model's loss history is cached to
comparison_logs/series/losscurve_<budget>_<key>.npz and skipped on rerun.
Regenerates comparison_logs/plots/training_loss_curves_long.png from
whatever caches exist.
"""
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))
os.chdir(ORIGINAL_CODE)

from qiskit_port import benchmark_lib as bl

MODELS = {  # key -> (display name, kind, framework, color, linestyle)
    "cddpg": ("Classical DDPG", "ddpg", "classical", "#1f77b4", "-"),
    "cdql": ("Classical Deep Q-Learning", "dql", "classical", "#17becf", "--"),
    "plqdpg": ("PennyLane QDPG", "ddpg", "pennylane", "#2ca02c", "-"),
    "plqql": ("PennyLane Quantum Q-Learning", "dql", "pennylane", "#98df8a", "--"),
    "qqdpg": ("Qiskit QDPG", "ddpg", "qiskit", "#d62728", "-"),
    "qqql": ("Qiskit Quantum Q-Learning", "dql", "qiskit", "#ff9896", "--"),
}
FAST_DEFAULT = ["cddpg", "cdql", "plqdpg", "plqql"]

NUM_ASSETS = 4
ROWS = 240


def base_config(budget):
    return dict(
        dataset_name="loss_curve_bench1",
        lookback_window=10, forecast_window=0, num_weights=16,
        max_epochs=budget, early_stopping=False, patience=5, min_delta=1e-4,
        seed=68, short_selling=True, clamp_negatives=True,
    )


def train_one(key, budget, train_data, val_data):
    display, kind, framework, _, _ = MODELS[key]
    cache = bl.SERIES_DIR / f"losscurve_{budget}_{key}.npz"
    if cache.exists():
        print(f"cached: {display}")
        return
    print(f"training {display} for {budget} epochs (early stopping off)...")
    cfg = base_config(budget)
    model = bl.build_rl_model(kind, framework, cfg)
    train_kwargs = bl.make_train_kwargs(kind, framework, cfg)
    _, history = bl.train_with_loss_capture(model, train_data, val_data,
                                            train_kwargs)
    np.savez_compressed(
        cache,
        actor_loss=np.array(history["actor_loss"]),
        critic_loss=np.array(history["critic_loss"]),
    )
    print(f"  done: {len(history['epoch'])} epochs recorded")


def plot(budget):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plotted = False
    for key, (display, _, _, color, ls) in MODELS.items():
        cache = bl.SERIES_DIR / f"losscurve_{budget}_{key}.npz"
        if not cache.exists():
            continue
        d = np.load(cache)
        ep = np.arange(1, len(d["actor_loss"]) + 1)
        axes[0].plot(ep, d["actor_loss"], label=display, color=color, ls=ls)
        axes[1].plot(ep, d["critic_loss"], label=display, color=color, ls=ls)
        plotted = True
    if not plotted:
        print("nothing to plot yet")
        return
    axes[0].set_title("Actor loss")
    axes[1].set_title("Critic loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8.5)
    fig.suptitle(f"Benchmark 1 training loss curves — {budget} epochs, "
                 "early stopping off")
    fig.tight_layout()
    out = bl.PLOTS_DIR / "training_loss_curves_long.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    args = sys.argv[1:]
    budget = 100
    if args and args[0].isdigit():
        budget = int(args[0])
        args = args[1:]
    keys = args if args else FAST_DEFAULT

    returns = bl.load_original_returns(ROWS, NUM_ASSETS)
    train_data, val_data, _ = bl.split_60_20_20(returns)

    for key in keys:
        if key not in MODELS:
            print(f"unknown model key: {key} (valid: {list(MODELS)})")
            continue
        train_one(key, budget, train_data, val_data)

    plot(budget)


if __name__ == "__main__":
    main()

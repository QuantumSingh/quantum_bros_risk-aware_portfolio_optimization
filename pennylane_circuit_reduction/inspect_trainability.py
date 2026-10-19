"""Do deeper circuits give weaker training signals? (barren-plateau check)

A known problem with variational quantum circuits: as they get deeper, their
measured outputs crowd toward zero and their gradients shrink, so training
stalls ("barren plateau"). This script measures that trend for the authors'
circuits at each depth, before any training.

For each depth (15/30/45/60 weights) and role (actor: 15 qubits, critic: 12),
it builds the authors' QuantumNeuralNetwork with 20 different random
initializations and evaluates 5 random market states, recording:

    mean |output|        how far the raw circuit output (<Z> of qubit 0) is
                         from zero; near zero means a weak, uninformative signal
    gradient variance    variance of d<Z_0>/d(weight) across initializations and
                         weights; the standard barren-plateau measure

Outputs:
    results/trainability.csv
    results/plots/trainability_vs_depth.png

Usage (from repo root):
    $PY pennylane_circuit_reduction/inspect_trainability.py
"""
import os
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
AUTHORS = HERE.parent / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(AUTHORS))
os.chdir(AUTHORS)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from predictors import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear

warnings.filterwarnings("ignore")

DEPTHS = [15, 30, 45, 60]
ROLES = {"actor": (555, 15), "critic": (570, 1)}
N_INITS, N_STATES = 20, 5
OUT = HERE / "results"


def measure(role, n_w):
    in_size, out_size = ROLES[role]
    states = [torch.randn(in_size, generator=torch.Generator().manual_seed(100 + s))
              for s in range(N_STATES)]
    outputs, grads = [], []
    for seed in range(N_INITS):
        model = QuantumNeuralNetwork(
            input_size=in_size, output_size=out_size, num_weights=n_w,
            encoding="amplitude", input_transformation=radial_to_linear,
            rotation_axes="y", seed=seed)
        for x in states:
            model.zero_grad()
            z0 = model.qlayer(model.input_transformation(x))[0]  # raw <Z> on qubit 0
            z0.backward()
            outputs.append(abs(z0.item()))
            grads.append(model.qlayer.weights.grad.detach().numpy().copy())
    grads = np.concatenate(grads)
    return {"role": role, "num_weights": n_w, "qubits": model.num_qubits,
            "mean_abs_output": float(np.mean(outputs)),
            "grad_variance": float(np.var(grads)),
            "mean_abs_grad": float(np.mean(np.abs(grads)))}


def main():
    rows = []
    for role in ROLES:
        for n_w in DEPTHS:
            row = measure(role, n_w)
            rows.append(row)
            print(f"{role:6s} {n_w:2d} weights: mean|<Z0>| {row['mean_abs_output']:.4f}  "
                  f"grad variance {row['grad_variance']:.2e}  mean|grad| {row['mean_abs_grad']:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "trainability.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for role, color in [("actor", "#2b6cb0"), ("critic", "#2f855a")]:
        sub = df[df["role"] == role]
        label = f"{role} ({int(sub['qubits'].iloc[0])} qubits)"
        axes[0].plot(sub["num_weights"], sub["mean_abs_output"], "o-", color=color, lw=2, ms=7, label=label)
        axes[1].plot(sub["num_weights"], sub["grad_variance"], "o-", color=color, lw=2, ms=7, label=label)
    axes[0].set_ylabel("Mean |output| at initialization")
    axes[0].set_title("Output strength (near 0 = weak signal)")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Gradient variance (log scale)")
    axes[1].set_title("Gradient size (smaller = slower learning)")
    for ax in axes:
        ax.set_xlabel("Trainable weights")
        ax.set_xticks(DEPTHS)
        ax.grid(alpha=0.25)
        ax.legend()
    fig.suptitle(f"Trainability vs circuit depth ({N_INITS} random initializations × {N_STATES} market states)")
    fig.tight_layout()
    (OUT / "plots").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "plots" / "trainability_vs_depth.png", dpi=180, bbox_inches="tight")
    print(f"\nWrote {OUT / 'trainability.csv'} and plots/trainability_vs_depth.png")


if __name__ == "__main__":
    main()

"""Summarize the circuit-depth sweep: performance, speed, loss curves, and
confusion-matrix metrics.

Reads results/runs/<run>/fold_*.json (written by run_depth_sweep.py) and
results/speed_benchmark.csv (written by inspect_circuit.py). Works with
partly finished runs: each table says how many of the 7 folds are in.

Confusion matrix. This is a portfolio problem, not a classifier, so we ask a
yes/no question the model implicitly answers at every 30-day rebalance:
"will this asset beat the average of all 15 assets over the next 30 days?"

    model says YES  = it overweights the asset (weight above 1/15 = 6.67%)
    actual YES      = the asset's 30-day return beat the 15-asset average

    TP  overweighted, and it beat the average   (backed a winner)
    FP  overweighted, but it lagged             (backed a loser)
    FN  underweighted, but it beat the average  (missed a winner)
    TN  underweighted, and it lagged            (avoided a loser)

    accuracy  = (TP + TN) / all
    precision = TP / (TP + FP)   when it backs an asset, how often it wins
    recall    = TP / (TP + FN)   of all winners, how many it backed
    chance    = accuracy a coin flip with the same backing rate would get

Outputs (results/):
    sweep_summary.csv / .md
    plots/performance_vs_depth.png, speed_vs_depth.png, loss_curves.png,
    plots/classification_vs_depth.png, plots/confusion_<run>.png

Usage (from repo root):
    $PY pennylane_circuit_reduction/summarize_sweep.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RUNS = RESULTS / "runs"
PLOTS = RESULTS / "plots"
N_FOLDS = 7

# Paper Table 5.1 (arXiv:2601.18811): mean Sharpe (std over folds).
PAPER = {("QDPG", 30): (0.4179, 0.6113), ("QDPG", 60): (0.7281, 0.7395),
         ("QQL", 30): (0.4776, 0.6650), ("QQL", 60): (0.6537, 0.6155)}
# Authors' released MAIN.py, full 7-fold run (pennylane_reproduction/), 60 weights.
RELEASED_60W = {"QDPG": -0.3998, "QQL": -0.2673}

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10.5})


def load_runs():
    runs = {}
    for run_dir in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        folds = [json.loads(f.read_text()) for f in sorted(run_dir.glob("fold_*.json"))]
        if folds:
            runs[run_dir.name] = sorted(folds, key=lambda r: r["fold"])
    return runs


def confusion(folds):
    tp = fp = fn = tn = 0
    for fold in folds:
        for reb in fold["rebalances"]:
            w = np.array(reb["weights"])
            r = np.array(reb["asset_returns"])
            pred = w > 1.0 / len(w)
            actual = r > r.mean()
            tp += int(np.sum(pred & actual))
            fp += int(np.sum(pred & ~actual))
            fn += int(np.sum(~pred & actual))
            tn += int(np.sum(~pred & ~actual))
    total = tp + fp + fn + tn
    p_pred, p_true = (tp + fp) / total, (tp + fn) / total
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / total,
        "precision": tp / (tp + fp) if tp + fp else float("nan"),
        "recall": tp / (tp + fn) if tp + fn else float("nan"),
        "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0,
        "chance_accuracy": p_pred * p_true + (1 - p_pred) * (1 - p_true),
        "backing_rate": p_pred,
    }


def stats(values):
    v = np.array(values)
    return v.mean(), np.percentile(v, 75) - np.percentile(v, 25), v.min(), v.max()


def summarize(runs):
    rows = []
    for name, folds in runs.items():
        model = folds[0]["model"]
        n_w = folds[0]["num_weights"]
        row = {"run": name, "model": model, "num_weights": n_w,
               "folds_done": f"{len(folds)}/{N_FOLDS}",
               "train_minutes": sum(f["train_seconds"] for f in folds) / 60}
        for key in ("dpo_sharpe", "spo_sharpe", "dpo_profit", "spo_profit"):
            mean, iqr, lo, hi = stats([f[key] for f in folds])
            row.update({key: mean, f"{key}_iqr": iqr, f"{key}_min": lo, f"{key}_max": hi})
        row.update(confusion(folds))
        paper = PAPER.get((model, n_w))
        row["paper_sharpe"] = paper[0] if paper else float("nan")
        rows.append(row)
    df = pd.DataFrame(rows)
    order = {"DDPG": 0, "DQL": 1, "QDPG": 2, "QQL": 3}
    df["_o"] = df["model"].map(order)
    return df.sort_values(["_o", "num_weights"]).drop(columns="_o").reset_index(drop=True)


def plot_performance(df, quantum):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for ax, key, title in [(axes[0], "dpo_sharpe", "Rebalanced every 30 days (DPO)"),
                           (axes[1], "spo_sharpe", "Static allocation (SPO)")]:
        for model, color in [("QDPG", "#2b6cb0"), ("QQL", "#c05621")]:
            sub = quantum[quantum["model"] == model]
            if sub.empty:
                continue
            ax.plot(sub["num_weights"], sub[key], "o-", color=color, lw=2, ms=7,
                    label=f"{model} (this sweep)")
            ax.fill_between(sub["num_weights"], sub[f"{key}_min"], sub[f"{key}_max"],
                            color=color, alpha=0.10)
            if key == "dpo_sharpe":
                pts = [(n, PAPER[(model, n)][0]) for n in (30, 60) if (model, n) in PAPER]
                ax.plot(*zip(*pts), "D", color=color, mfc="white", ms=8, mew=2,
                        label=f"{model} (paper)")
        classical = df[df["model"] == "DDPG"]
        if not classical.empty:
            ax.axhline(classical[key].iloc[0], color="#555", ls="--", lw=1.2,
                       label="Classical DDPG")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks([15, 30, 45, 60])
        ax.set_xticklabels(["15\n(1 layer)", "30\n(2 layers)", "45\n(3 layers)", "60\n(4, released)"])
        ax.set_xlabel("Trainable weights (actor rotation layers)")
        ax.set_ylabel("Sharpe ratio, mean over folds")
        ax.set_title(title)
    axes[0].legend(fontsize=8.5, loc="best")
    fig.suptitle("Quantum DDPG performance vs circuit depth (shaded: min–max over folds)")
    fig.tight_layout()
    fig.savefig(PLOTS / "performance_vs_depth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_speed(quantum):
    bench_path = RESULTS / "speed_benchmark.csv"
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    if bench_path.exists():
        bench = pd.read_csv(bench_path)
        for role, color in [("actor", "#2b6cb0"), ("critic", "#2f855a")]:
            sub = bench[bench["role"] == role]
            axes[0].plot(sub["num_weights"], sub["forward_backward_ms"], "o-", color=color,
                         lw=2, ms=7, label=f"{role} ({int(sub['qubits'].iloc[0])} qubits)")
        axes[0].set_ylabel("ms per forward + backward pass")
        axes[0].set_title("One circuit evaluation (clean benchmark)")
        axes[0].legend()
    sub = quantum[quantum["model"] == "QDPG"]
    if not sub.empty:
        axes[1].bar(sub["num_weights"].astype(str), sub["train_minutes"], color="#2b6cb0", width=0.55)
        for x, (m, done) in enumerate(zip(sub["train_minutes"], sub["folds_done"])):
            axes[1].text(x, m, f"{m:.0f} min\n({done} folds)", ha="center", va="bottom", fontsize=8.5)
        axes[1].set_ylabel("Total training minutes")
        axes[1].set_title("Full Quantum DDPG training (runs overlapped)")
    for ax in axes:
        ax.set_xlabel("Trainable weights")
    fig.suptitle("Speed vs circuit depth")
    fig.tight_layout()
    fig.savefig(PLOTS / "speed_vs_depth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_losses(runs):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {15: "#90cdf4", 30: "#4299e1", 45: "#2b6cb0", 60: "#1a365d"}
    for name, folds in runs.items():
        if folds[0]["model"] != "QDPG":
            continue
        n_w = folds[0]["num_weights"]
        for ax, key in [(axes[0], "actor_loss"), (axes[1], "critic_loss")]:
            curves = np.array([f[key] for f in folds if len(f[key]) == 50])
            if not len(curves):
                continue
            epochs = np.arange(1, curves.shape[1] + 1)
            ax.plot(epochs, curves.mean(axis=0), color=colors.get(n_w, "gray"), lw=2,
                    label=f"{n_w} weights ({len(curves)} folds)")
            ax.fill_between(epochs, curves.min(axis=0), curves.max(axis=0),
                            color=colors.get(n_w, "gray"), alpha=0.08)
    axes[0].set_title("Actor loss (lower = critic rates the policy higher)")
    axes[1].set_title("Critic loss (prediction error, lower = better fit)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss, mean over folds")
        ax.legend(fontsize=8.5)
    fig.suptitle("Quantum DDPG training loss by circuit depth (shaded: min–max over folds)")
    fig.tight_layout()
    fig.savefig(PLOTS / "loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_classification(df):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    q = df[df["model"] == "QDPG"]
    if not q.empty:
        for metric, color in [("accuracy", "#2b6cb0"), ("precision", "#2f855a"),
                              ("recall", "#c05621"), ("f1", "#6b46c1")]:
            axes[0].plot(q["num_weights"], q[metric], "o-", color=color, lw=2, ms=7, label=metric)
        axes[0].plot(q["num_weights"], q["chance_accuracy"], "k--", lw=1.2, label="chance accuracy")
        axes[0].set_xticks([15, 30, 45, 60])
        axes[0].set_xlabel("Trainable weights")
        axes[0].set_ylim(0, 1)
        axes[0].set_title("Quantum DDPG: picking winners vs depth")
        axes[0].legend(fontsize=8.5)
    labels, accs, chances = [], [], []
    for _, r in df.iterrows():
        labels.append(r["run"].replace("_", "\n"))
        accs.append(r["accuracy"])
        chances.append(r["chance_accuracy"])
    x = np.arange(len(labels))
    axes[1].bar(x - 0.18, accs, 0.36, label="accuracy", color="#2b6cb0")
    axes[1].bar(x + 0.18, chances, 0.36, label="chance", color="#cbd5e0")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Accuracy vs chance, every run")
    axes[1].legend(fontsize=8.5)
    fig.suptitle("Does the model overweight the assets that go on to beat the average?")
    fig.tight_layout()
    fig.savefig(PLOTS / "classification_vs_depth.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(df):
    for _, r in df.iterrows():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        cm = np.array([[r["TP"], r["FN"]], [r["FP"], r["TN"]]])
        axes[0].imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            axes[0].text(j, i, f"{v}\n{v / cm.sum():.0%}", ha="center", va="center",
                         color="white" if v > cm.max() * 0.6 else "black", fontsize=11)
        axes[0].set_xticks([0, 1], ["model: overweight", "model: underweight"])
        axes[0].set_yticks([0, 1], ["actual: beat avg", "actual: lagged"])
        axes[0].set_title("Confusion matrix (all rebalances, all assets)")
        axes[0].grid(False)
        names = ["accuracy", "precision", "recall", "f1"]
        vals = [r[n] for n in names]
        axes[1].bar(names, vals, color=["#2b6cb0", "#2f855a", "#c05621", "#6b46c1"], width=0.55)
        axes[1].axhline(r["chance_accuracy"], color="black", ls="--", lw=1.2,
                        label=f"chance accuracy {r['chance_accuracy']:.2f}")
        for i, v in enumerate(vals):
            axes[1].text(i, v, f"{v:.2f}", ha="center", va="bottom")
        axes[1].set_ylim(0, 1)
        axes[1].legend(fontsize=8.5)
        axes[1].set_title("Classification metrics")
        fig.suptitle(f"{r['run']}  ({r['folds_done']} folds)")
        fig.tight_layout()
        fig.savefig(PLOTS / f"confusion_{r['run']}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def write_markdown(df):
    fmt = lambda x, nd=4: "—" if pd.isna(x) else f"{x:.{nd}f}"
    md = ["# Circuit-depth sweep results", "",
          "Generated by `summarize_sweep.py`. Same data, folds, hyperparameters and seed as the",
          "authors' `MAIN.py`; only the number of circuit weights changes. IQR = interquartile",
          "range over folds.", "",
          "## Performance and speed", "",
          "| Run | Weights | Folds | DPO Sharpe (IQR) | SPO Sharpe (IQR) | DPO profit p.a. | Paper Sharpe | Train min |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['run']} | {'—' if pd.isna(r['num_weights']) else int(r['num_weights'])} | "
                  f"{r['folds_done']} | {fmt(r['dpo_sharpe'])} ({fmt(r['dpo_sharpe_iqr'])}) | "
                  f"{fmt(r['spo_sharpe'])} ({fmt(r['spo_sharpe_iqr'])}) | "
                  f"{fmt(r['dpo_profit'] * 100, 2)}% | {fmt(r['paper_sharpe'])} | {r['train_minutes']:.0f} |")
    md += ["", "Released `MAIN.py` at 60 weights (pennylane_reproduction/): "
           f"QDPG DPO Sharpe {RELEASED_60W['QDPG']}, QQL {RELEASED_60W['QQL']}.", "",
           "## Picking winners (confusion matrix)", "",
           "At each 30-day rebalance, *overweight* (weight above 1/15) counts as predicting the",
           "asset beats the 15-asset average over the next 30 days.", "",
           "| Run | TP | FP | FN | TN | Accuracy | Chance | Precision | Recall | F1 |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {r['run']} | {r['TP']} | {r['FP']} | {r['FN']} | {r['TN']} | "
                  f"{fmt(r['accuracy'], 3)} | {fmt(r['chance_accuracy'], 3)} | "
                  f"{fmt(r['precision'], 3)} | {fmt(r['recall'], 3)} | {fmt(r['f1'], 3)} |")
    (RESULTS / "sweep_summary.md").write_text("\n".join(md) + "\n")


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        print("No runs yet.")
        return
    df = summarize(runs)
    df.to_csv(RESULTS / "sweep_summary.csv", index=False)
    quantum = df[df["model"].isin(["QDPG", "QQL"])]
    plot_performance(df, quantum)
    plot_speed(quantum)
    plot_losses(runs)
    plot_classification(df)
    plot_confusions(df)
    write_markdown(df)
    print((RESULTS / "sweep_summary.md").read_text())


if __name__ == "__main__":
    main()

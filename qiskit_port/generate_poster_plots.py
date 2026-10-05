"""Generate all paper/poster figures and aggregate tables (Task H).

Reads the CSVs and per-model series saved by the benchmark runners and
the parameter sweep, and writes every figure into comparison_logs/plots/.
Missing inputs are skipped with a warning so this can be re-run as
results land.

Also writes the Task E aggregate exports:
    comparison_logs/new_metrics_results.csv / .tex
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

COMPARISON_DIR = REPO_ROOT / "comparison_logs"
PLOTS_DIR = COMPARISON_DIR / "plots"
SERIES_DIR = COMPARISON_DIR / "series"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Paper/poster style: readable sizes, no chart junk, max 2 panels/figure.
plt.rcParams.update({
    "savefig.dpi": 200,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.labelsize": 11.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9.5,
    "legend.framealpha": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.9,
    "figure.autolayout": False,
})

MODEL_ORDER = [
    "Equal Weight",
    "Mean Variance Optimization",
    "Classical DDPG",
    "Classical Deep Q-Learning",
    "PennyLane QDPG",
    "PennyLane Quantum Q-Learning",
    "Qiskit QDPG",
    "Qiskit Quantum Q-Learning",
]

MODEL_COLORS = {
    "Equal Weight": "#888888",
    "Mean Variance Optimization": "#555555",
    "Classical DDPG": "#1f77b4",
    "Classical Deep Q-Learning": "#17becf",
    "PennyLane QDPG": "#2ca02c",
    "PennyLane Quantum Q-Learning": "#98df8a",
    "Qiskit QDPG": "#d62728",
    "Qiskit Quantum Q-Learning": "#ff9896",
}


def safe_name(model: str) -> str:
    return model.lower().replace(" ", "_").replace("-", "_")


def load_benchmark(prefix: str):
    csv_path = COMPARISON_DIR / f"{prefix}_results.csv"
    if not csv_path.exists():
        print(f"SKIP: {csv_path.name} not found")
        return None
    df = pd.read_csv(csv_path)
    if "dpo_annualized_return" not in df.columns:
        print(f"SKIP: {csv_path.name} has the old schema "
              "(re-run the benchmark to regenerate)")
        return None
    df["__order"] = df["model"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}).fillna(99)
    return df.sort_values("__order").drop(columns="__order")


def model_series(prefix: str, model: str):
    path = SERIES_DIR / f"{prefix}_{safe_name(model)}.npz"
    if not path.exists():
        return None
    return np.load(path)


def finish(fig, filename: str):
    out = PLOTS_DIR / filename
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out.relative_to(REPO_ROOT))


# ---------------------------------------------------------------------------
# Per-benchmark figures
# ---------------------------------------------------------------------------

def plot_cumulative_returns(prefix: str, out_name: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    for model in MODEL_ORDER:
        data = model_series(prefix, model)
        if data is None:
            continue
        r = data["dpo_daily_returns"]
        if len(r) == 0:
            continue
        curve = np.cumprod(1.0 + r) - 1.0
        ax.plot(np.arange(1, len(curve) + 1), curve * 100,
                label=model, color=MODEL_COLORS.get(model), linewidth=1.8,
                linestyle="--" if "Q-Learning" in model else "-")
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"SKIP: no series for {prefix}")
        return
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Test day")
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    finish(fig, out_name)


def plot_runtime_bar(df: pd.DataFrame, out_name: str, title: str):
    sub = df.dropna(subset=["runtime_seconds"])
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [MODEL_COLORS.get(m, "#333333") for m in sub["model"]]
    # Floor at 0.01s: a zero-height bar (Equal Weight) on a log axis puts
    # the tight bounding box at -inf and blows up the canvas.
    plotted_vals = sub["runtime_seconds"].clip(lower=0.01)
    ax.bar(sub["model"], plotted_vals, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Train + evaluate runtime (s, log scale)")
    ax.set_title(title)
    for i, (m, v, pv) in enumerate(zip(sub["model"], sub["runtime_seconds"],
                                       plotted_vals)):
        ax.text(i, pv, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    finish(fig, out_name)


def two_panel_bars(df: pd.DataFrame, metric_pair, out_name: str, title: str):
    """One figure, exactly two bar panels (max-2-panels rule)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    for ax, (col, label) in zip(axes, metric_pair):
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        colors = [MODEL_COLORS.get(m, "#333333") for m in sub["model"]]
        ax.bar(sub["model"], sub[col], color=colors)
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=40, labelsize=8)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        ax.grid(alpha=0.25, axis="y")
        ax.axhline(0, color="black", linewidth=0.6)
    fig.suptitle(title)
    fig.tight_layout()
    finish(fig, out_name)


def plot_metrics_comparison(df: pd.DataFrame, out_name: str, title: str):
    two_panel_bars(
        df,
        [("dpo_sharpe", "Sharpe (DPO)"),
         ("dpo_annualized_return", "Annualized return (CAGR)")],
        out_name, title)
    two_panel_bars(
        df,
        [("dpo_annualized_volatility", "Annualized volatility"),
         ("dpo_var_5", "VaR 5% (daily)")],
        out_name.replace(".png", "_risk.png"), title + " — risk")


def plot_training_loss_curves(prefix: str, out_name: str, title: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    plotted = False
    for model in MODEL_ORDER:
        data = model_series(prefix, model)
        if data is None or len(data["actor_loss"]) == 0:
            continue
        epochs = np.arange(1, len(data["actor_loss"]) + 1)
        axes[0].plot(epochs, data["actor_loss"],
                     label=model, color=MODEL_COLORS.get(model))
        axes[1].plot(epochs, data["critic_loss"],
                     label=model, color=MODEL_COLORS.get(model))
        plotted = True
    if not plotted:
        plt.close(fig)
        print(f"SKIP: no loss curves for {prefix}")
        return
    axes[0].set_title("Actor loss")
    axes[1].set_title("Critic loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    fig.suptitle(title)
    fig.tight_layout()
    finish(fig, out_name)


# ---------------------------------------------------------------------------
# Parameter sweep figures
# ---------------------------------------------------------------------------

def sweep_figures():
    csv_path = COMPARISON_DIR / "parameter_sweep_results.csv"
    if not csv_path.exists():
        print("SKIP: parameter sweep results not found")
        return
    df = pd.read_csv(csv_path)

    # Epoch plateau: loss curves from the LONGEST no-ES run available
    # (500 > 200 > 100 epochs).
    candidates = sorted(
        SERIES_DIR.glob("sweep_epochs_*_noES.npz"),
        key=lambda p: int(p.stem.split("_")[2]),
    )
    if candidates:
        npz_path = candidates[-1]
        n_epochs_run = int(npz_path.stem.split("_")[2])
        data = np.load(npz_path)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        epochs = np.arange(1, len(data["actor_loss"]) + 1)
        ax.plot(epochs, data["actor_loss"], label="Actor loss")
        ax.plot(epochs, data["critic_loss"], label="Critic loss")
        val = data["val_critic_loss"]
        if len(val) and not np.all(np.isnan(val)):
            ax.plot(epochs, val, label="Val critic loss", linestyle="--")
        es_row = df[df["run_id"] == "epochs_100_ES"]
        if len(es_row) and not pd.isna(es_row["epochs_completed"].iloc[0]):
            stop = int(es_row["epochs_completed"].iloc[0])
            ax.axvline(stop, color="red", linestyle=":",
                       label=f"Early stop (patience 5) @ {stop}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"PennyLane QDPG training losses over {n_epochs_run} "
                     "epochs (240 rows, 4 assets)")
        ax.legend()
        finish(fig, "epoch_plateau_loss_curve.png")

    rows_df = df[df["run_id"].str.startswith("rows_")].dropna(
        subset=["runtime_seconds"])
    if len(rows_df):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(rows_df["rows"], rows_df["runtime_seconds"], "o-")
        for _, r in rows_df.iterrows():
            ax.annotate(f"{int(r['epochs_completed'])} ep",
                        (r["rows"], r["runtime_seconds"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("Training rows")
        ax.set_ylabel("Training runtime (s)")
        ax.set_title("Runtime vs data range (early stopping on)")
        ax.grid(alpha=0.3)
        finish(fig, "runtime_vs_rows.png")

    ep_df = df[df["run_id"].str.contains("noES")].dropna(
        subset=["runtime_seconds"])
    if len(ep_df):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ep_df["max_epochs"], ep_df["runtime_seconds"], "o-")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Training runtime (s)")
        ax.set_title("Runtime vs epochs (240 rows, 4 assets, PennyLane)")
        ax.grid(alpha=0.3)
        finish(fig, "runtime_vs_epochs.png")

        # Two measures of different scale -> two stacked panels sharing
        # the x axis (never a dual-axis chart).
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7, 6.5), sharex=True)
        ax_top.plot(ep_df["max_epochs"], ep_df["dpo_sharpe"], "o-")
        ax_top.set_ylabel("DPO Sharpe")
        ax_bot.plot(ep_df["max_epochs"], ep_df["dpo_profit_pa"] * 100, "o-",
                    color="#d62728")
        ax_bot.set_ylabel("DPO profit p.a. (%)")
        ax_bot.set_xlabel("Epochs")
        ax_top.set_title("Test metrics vs training epochs (plateau ~50)")
        for ax in (ax_top, ax_bot):
            ax.grid(alpha=0.3)
        fig.tight_layout()
        finish(fig, "metric_vs_epochs.png")

    # Summary: two figures of two panels each (max-2-panels rule)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    if len(rows_df):
        axes[0].plot(rows_df["rows"], rows_df["dpo_sharpe"], "o-")
        axes[0].set_title("DPO Sharpe vs rows\n(caveat: test window shifts too)")
        axes[0].set_xlabel("Rows")
        axes[0].set_ylabel("DPO Sharpe")
    if len(ep_df):
        axes[1].plot(ep_df["max_epochs"], ep_df["dpo_sharpe"], "o-")
        axes[1].set_title("DPO Sharpe vs epochs (no ES)")
        axes[1].set_xlabel("Epochs")
    fig.suptitle("Parameter sweep summary (PennyLane QDPG)")
    fig.tight_layout()
    finish(fig, "parameter_sweep_summary.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    if len(ep_df):
        axes[0].plot(ep_df["max_epochs"], ep_df["runtime_seconds"], "o-")
        axes[0].set_title("Training runtime vs epochs")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Seconds")
    lb_df = df[df["run_id"].isin(["rows_240", "lookback_10"])]
    if len(lb_df) == 2:
        axes[1].bar(lb_df["lookback_window"].astype(str),
                    lb_df["dpo_sharpe"], width=0.5,
                    color=["#1f77b4", "#2ca02c"])
        axes[1].set_title("DPO Sharpe vs lookback window")
        axes[1].set_xlabel("Lookback")
        axes[1].set_ylabel("DPO Sharpe")
    fig.suptitle("Parameter sweep: cost and lookback (PennyLane QDPG)")
    fig.tight_layout()
    finish(fig, "parameter_sweep_runtime.png")


# ---------------------------------------------------------------------------
# Stress-test and tail-risk figures
# ---------------------------------------------------------------------------

STRESS_WINDOWS = ["covid_crash_2020", "bear_2022", "calm_2024"]
STRESS_TITLES = {
    "covid_crash_2020": "COVID crash (2020-02-15 .. 2020-06-30)",
    "bear_2022": "2022 bear market",
    "calm_2024": "2024 calm control",
}


def stress_figures():
    csv_path = COMPARISON_DIR / "stress_test_results.csv"
    if not csv_path.exists():
        print("SKIP: stress test results not found")
        return

    # One window per figure (max-2-panels rule; these read better solo).
    for window in STRESS_WINDOWS:
        fig, ax = plt.subplots(figsize=(9, 5))
        plotted = False
        for model in MODEL_ORDER:
            path = SERIES_DIR / f"stress_{window}_{safe_name(model)}.npz"
            if not path.exists():
                continue
            r = np.load(path)["dpo_daily_returns"]
            curve = (np.cumprod(1.0 + r) - 1.0) * 100
            ax.plot(np.arange(1, len(curve) + 1), curve,
                    label=model, color=MODEL_COLORS.get(model),
                    linestyle="--" if "Q-Learning" in model else "-")
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_title(f"Stress test — {STRESS_TITLES[window]}\n"
                     "(DPO; models retrained on data preceding the window)")
        ax.set_xlabel("Test day")
        ax.set_ylabel("Cumulative return (%)")
        ax.legend(loc="best", fontsize=8.5)
        finish(fig, f"stress_cumulative_{window}.png")

    df = pd.read_csv(csv_path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    width = 0.25
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    x = np.arange(len(models))
    for i, window in enumerate(STRESS_WINDOWS):
        sub = df[df["window"] == window].set_index("model")
        mdd = [sub.loc[m, "tail_max_drawdown"] if m in sub.index else np.nan
               for m in models]
        cvar = [sub.loc[m, "tail_cvar_5"] if m in sub.index else np.nan
                for m in models]
        axes[0].bar(x + (i - 1) * width, np.array(mdd) * 100, width,
                    label=STRESS_TITLES[window])
        axes[1].bar(x + (i - 1) * width, np.array(cvar) * 100, width,
                    label=STRESS_TITLES[window])
    axes[0].set_title("Max drawdown (%)", fontsize=10)
    axes[1].set_title("CVaR 5% (daily, %)", fontsize=10)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=30, ha="right", fontsize=7)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=7)
    fig.suptitle("Tail risk under stress windows")
    fig.tight_layout()
    finish(fig, "stress_test_tail_risk.png")


def tail_risk_figure():
    csv_path = COMPARISON_DIR / "tail_risk_results.csv"
    if not csv_path.exists():
        print("SKIP: tail risk results not found")
        return
    df = pd.read_csv(csv_path)

    benchmarks = list(df["benchmark"].unique())
    models = [m for m in MODEL_ORDER if m in set(df["model"])]
    x = np.arange(len(models))
    width = 0.38

    def tail_pair(metric_pair, out_name, suptitle):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (col, title, scale) in zip(axes, metric_pair):
            for i, bench in enumerate(benchmarks):
                sub = df[df["benchmark"] == bench].set_index("model")
                vals = [sub.loc[m, col] * scale if m in sub.index else np.nan
                        for m in models]
                ax.bar(x + (i - 0.5) * width, vals, width,
                       label=bench.replace("benchmark", "B"))
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=40, ha="right", fontsize=8)
            ax.grid(alpha=0.25, axis="y")
            ax.axhline(0, color="black", linewidth=0.6)
        axes[0].legend()
        fig.suptitle(suptitle)
        fig.tight_layout()
        finish(fig, out_name)

    tail_pair([("max_drawdown", "Max drawdown (%)", 100),
               ("cvar_5", "CVaR 5% (daily, %)", 100)],
              "tail_risk_comparison.png",
              "Tail-risk profile of the benchmark strategies (DPO)")
    tail_pair([("worst_day", "Worst day (%)", 100),
               ("excess_kurtosis", "Excess kurtosis", 1)],
              "tail_risk_shape.png",
              "Tail shape of the benchmark strategies (DPO)")


# ---------------------------------------------------------------------------
# Robustness / anti-artifact figures
# ---------------------------------------------------------------------------

def robustness_figures():
    csv_path = COMPARISON_DIR / "robustness_checks.csv"
    if not csv_path.exists():
        print("SKIP: robustness checks not found")
        return
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: multi-seed distributions vs placebo, with EW reference.
    seed_df = df[df.test == "A_multiseed"]
    groups, labels = [], []
    for label in ["PennyLane QDPG", "Classical DDPG"]:
        vals = seed_df[seed_df.model == label]["sharpe"].dropna().values
        if len(vals):
            groups.append(vals)
            labels.append(f"{label}\n(10 seeds)")
    placebo = df[df.test == "B_placebo"]["sharpe"].dropna().values
    if len(placebo):
        groups.append(placebo)
        labels.append("Placebo: shuffled\ntraining data")
    axes[0].boxplot(groups, tick_labels=labels, widths=0.5)
    for i, vals in enumerate(groups):
        axes[0].scatter(np.full(len(vals), i + 1) +
                        np.random.default_rng(0).uniform(-0.07, 0.07, len(vals)),
                        vals, s=18, alpha=0.7, color="#1f77b4", zorder=3)
    ew = seed_df[seed_df.model == "Equal Weight"]["sharpe"]
    if len(ew):
        axes[0].axhline(float(ew.iloc[0]), color="#888888", linestyle="--",
                        label="Equal Weight (deterministic)")
        axes[0].legend()
    axes[0].set_ylabel("Test DPO Sharpe")
    axes[0].set_title("Seed robustness and the shuffled-data placebo")

    # Panel 2: null distributions vs trained models.
    null_path = SERIES_DIR / "robustness_random_null_sharpes.npy"
    if null_path.exists():
        null = np.load(null_path)
        axes[1].hist(null, bins=60, alpha=0.55, color="#bbbbbb",
                     label="Random policies (2000)", density=True)
    untrained = df[df.test == "D_untrained"]["sharpe"].dropna().values
    if len(untrained):
        axes[1].hist(untrained, bins=12, alpha=0.6, color="#9edae5",
                     label="Untrained QNN actors (20)", density=True)
    qdpg68 = df[(df.test == "A_multiseed") & (df.model == "PennyLane QDPG")
                & (df.seed == 68)]["sharpe"]
    if len(qdpg68):
        axes[1].axvline(float(qdpg68.iloc[0]), color="#2ca02c", linewidth=2,
                        label="Trained PennyLane QDPG (seed 68)")
    axes[1].set_xlabel("Test DPO Sharpe")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Trained policy vs no-skill nulls")
    axes[1].legend(fontsize=8.5)
    fig.suptitle("Anti-artifact checks (Benchmark 1 configuration)")
    fig.tight_layout()
    finish(fig, "robustness_seed_distribution.png")

    # Walk-forward folds (single panel)
    wf = df[df.test == "E_walkforward"]
    if len(wf):
        fig, ax = plt.subplots(figsize=(10, 5))
        for label in ["Equal Weight", "Classical DDPG", "PennyLane QDPG"]:
            sub = wf[wf.model == label].sort_values("fold")
            ax.plot(sub["fold"], sub["sharpe"], "o-", label=label,
                    color=MODEL_COLORS.get(label))
        ax.axhline(0, color="black", linewidth=0.6)
        folds = wf[wf.model == "Equal Weight"].sort_values("fold")
        ax.set_xticks(folds["fold"])
        ax.set_xticklabels([str(t)[:10] for t in folds["fold_test"]],
                           rotation=40, ha="right", fontsize=8)
        ax.set_xlabel("Fold (test-window start date)")
        ax.set_ylabel("Test DPO Sharpe")
        ax.set_title("Walk-forward evaluation: 10 rolling 240-row folds")
        ax.legend()
        finish(fig, "walk_forward_folds.png")


# ---------------------------------------------------------------------------
# New-metrics aggregates (Task E exports)
# ---------------------------------------------------------------------------

NEW_METRIC_COLS = [
    "dpo_annualized_return", "dpo_annualized_volatility", "dpo_var_5",
    "dpo_avg_turnover", "dpo_tc_adjusted_annual_return",
]


def new_metrics_outputs(benchmarks: dict):
    frames = []
    for label, df in benchmarks.items():
        if df is None:
            continue
        cols = ["model", "framework", "algorithm", "dataset",
                "dpo_profit_pa", "dpo_sharpe", *NEW_METRIC_COLS,
                "runtime_seconds", "notes"]
        sub = df[[c for c in cols if c in df.columns]].copy()
        sub.insert(0, "benchmark", label)
        frames.append(sub)
    if not frames:
        print("SKIP: no benchmark results for new-metrics aggregation")
        return

    agg = pd.concat(frames, ignore_index=True)
    csv_path = COMPARISON_DIR / "new_metrics_results.csv"
    tex_path = COMPARISON_DIR / "new_metrics_results.tex"
    agg.to_csv(csv_path, index=False)
    agg.to_latex(tex_path, index=False, float_format="%.4f")
    print("Wrote", csv_path.name, "and", tex_path.name)

    # New metrics per benchmark: two 2-panel figures instead of one 1x5.
    for label, df in benchmarks.items():
        if df is None:
            continue
        two_panel_bars(
            df,
            [("dpo_annualized_return", "Annualized return (CAGR)"),
             ("dpo_tc_adjusted_annual_return",
              "TC-adjusted annual return (10 bps)")],
            f"new_metrics_bar_chart_{label}.png"
            if label != "benchmark1" else "new_metrics_bar_chart.png",
            f"Return metrics — {label}")
        two_panel_bars(
            df,
            [("dpo_annualized_volatility", "Annualized volatility"),
             ("dpo_var_5", "VaR 5% (daily)")],
            f"new_metrics_risk_chart_{label}.png"
            if label != "benchmark1" else "new_metrics_risk_chart.png",
            f"Risk metrics — {label}")

    # Turnover comparison across benchmarks (single panel)
    fig, ax = plt.subplots(figsize=(9.5, 5))
    width = 0.38
    labels = [m for m in MODEL_ORDER]
    for i, (label, df) in enumerate(benchmarks.items()):
        if df is None:
            continue
        vals = [df.loc[df["model"] == m, "dpo_avg_turnover"].squeeze()
                if (df["model"] == m).any() else np.nan for m in labels]
        ax.bar(np.arange(len(labels)) + i * width, vals, width, label=label)
    ax.set_xticks(np.arange(len(labels)) + width / 2)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Avg one-way turnover per rebalance")
    ax.set_title("Turnover comparison (DPO)")
    ax.legend()
    finish(fig, "turnover_comparison.png")

    # Gross vs TC-adjusted return: one figure per benchmark.
    for i, (label, df) in enumerate(benchmarks.items()):
        if df is None:
            continue
        fig, ax = plt.subplots(figsize=(9.5, 5))
        x = np.arange(len(labels))
        gross = [df.loc[df["model"] == m, "dpo_annualized_return"].squeeze()
                 if (df["model"] == m).any() else np.nan for m in labels]
        net = [df.loc[df["model"] == m, "dpo_tc_adjusted_annual_return"].squeeze()
               if (df["model"] == m).any() else np.nan for m in labels]
        ax.bar(x - 0.19, np.array(gross) * 100, 0.36, label="Gross")
        ax.bar(x + 0.19, np.array(net) * 100, 0.36,
               label="Net of 10 bps costs", alpha=0.75)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_ylabel("Annualized return (%)")
        ax.set_title(f"Transaction-cost impact (DPO) — {label}")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.legend()
        finish(fig, "transaction_cost_adjusted_return.png" if label == "benchmark1"
               else f"transaction_cost_adjusted_return_{label}.png")


# ---------------------------------------------------------------------------

def main():
    bench1 = load_benchmark("real_data_benchmark")
    bench2 = load_benchmark("custom_nonpaper_10_benchmark")

    if bench1 is not None:
        plot_cumulative_returns(
            "real_data_benchmark", "benchmark1_cumulative_returns.png",
            "Benchmark 1: cumulative test returns (DPO) — original data subset")
        plot_runtime_bar(bench1, "benchmark1_runtime_bar.png",
                         "Benchmark 1: runtime by model")
        plot_metrics_comparison(bench1, "benchmark1_metrics_comparison.png",
                                "Benchmark 1: metrics comparison")
        plot_training_loss_curves(
            "real_data_benchmark", "training_loss_curves.png",
            "Benchmark 1: training loss curves")

    if bench2 is not None:
        plot_cumulative_returns(
            "custom_nonpaper_10_benchmark",
            "custom_nonpaper_10_cumulative_returns.png",
            "Benchmark 2: cumulative test returns (DPO) — custom_nonpaper_10")
        plot_runtime_bar(bench2, "custom_nonpaper_10_runtime_bar.png",
                         "Benchmark 2: runtime by model")
        plot_metrics_comparison(
            bench2, "custom_nonpaper_10_metrics_comparison.png",
            "Benchmark 2: metrics comparison")

    sweep_figures()

    stress_figures()
    tail_risk_figure()
    robustness_figures()

    new_metrics_outputs({"benchmark1": bench1, "benchmark2": bench2})

    # One-stop poster assets: copy the circuit diagram into plots/
    circuit_png = COMPARISON_DIR / "circuits" / "qiskit_vqc_high_level_circuit.png"
    if circuit_png.exists():
        shutil.copy(circuit_png, PLOTS_DIR / "circuit_diagram.png")
        print("Copied circuit_diagram.png")

    print("\nAll available figures generated in", PLOTS_DIR.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()

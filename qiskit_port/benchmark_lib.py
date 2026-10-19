"""Shared machinery for the paper/poster benchmarks.

Used by:
    run_real_data_benchmark.py          (Benchmark 1: original dataset subset)
    run_custom_nonpaper_10_benchmark.py (Benchmark 2: custom non-paper tickers)
    run_parameter_sweep.py              (training sweet-spot experiments)

Responsibilities:
    - data loading + validation (the original repo parquet already holds
      RETURNS; custom yfinance parquets hold PRICES and need pct_change)
    - 60/20/20 train/val/test split
    - training the 8 model variants with captured epoch loss curves
    - SPO/DPO evaluation with full weight trajectories (for turnover and
      transaction-cost metrics, via qiskit_port.portfolio_metrics)
    - CSV / LaTeX / log / figure exports
"""
import io
import json
import re
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

for p in (str(REPO_ROOT), str(ORIGINAL_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from qiskit_port import portfolio_metrics as pm

COMPARISON_DIR = REPO_ROOT / "comparison_logs"
PLOTS_DIR = COMPARISON_DIR / "plots"
SERIES_DIR = COMPARISON_DIR / "series"

for d in (COMPARISON_DIR, PLOTS_DIR, SERIES_DIR):
    d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class TeeLogger:
    """Mirror stdout to a log file."""

    def __init__(self, log_path: Path):
        self.log_file = open(log_path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.log_file.write(text)

    def flush(self):
        self.stdout.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()


# ---------------------------------------------------------------------------
# Data loading and validation
# ---------------------------------------------------------------------------

def validate_returns(returns: pd.DataFrame, name: str) -> dict:
    """Print and return a validation report for a daily-returns DataFrame."""
    values = returns.values
    report = {
        "dataset": name,
        "shape": tuple(returns.shape),
        "columns": list(returns.columns),
        "start_date": str(returns.index[0]),
        "end_date": str(returns.index[-1]),
        "min_return": float(np.nanmin(values)),
        "max_return": float(np.nanmax(values)),
        "mean_return": float(np.nanmean(values)),
        "std_return": float(np.nanstd(values)),
        "num_nan": int(np.isnan(values).sum()),
        "num_inf": int(np.isinf(values).sum()),
    }

    print(f"\n--- Data validation: {name} ---")
    for key, val in report.items():
        print(f"{key}: {val}")

    if report["num_nan"] > 0:
        raise ValueError(f"{name}: {report['num_nan']} NaN values in returns")
    if report["num_inf"] > 0:
        raise ValueError(f"{name}: {report['num_inf']} inf values in returns")

    extreme = np.abs(values) > 0.5
    if extreme.any():
        rows_idx, cols_idx = np.where(extreme)
        warnings.warn(
            f"{name}: {extreme.sum()} daily returns exceed |50%| "
            f"(e.g. {returns.columns[cols_idx[0]]} on {returns.index[rows_idx[0]]}: "
            f"{values[rows_idx[0], cols_idx[0]]:.3f}); check for splits/data errors."
        )
        report["extreme_returns"] = int(extreme.sum())
    else:
        report["extreme_returns"] = 0

    print("Validation OK.")
    return report


def load_original_returns(rows: int, num_assets: int) -> pd.DataFrame:
    """Original repo dataset. The parquet already contains daily RETURNS
    (values in roughly [-0.3, 0.2]) despite the 'price_data' file name."""
    returns = pd.read_parquet(ORIGINAL_CODE / "data" / "price_data.parquet.gzip")
    returns = returns.iloc[:, :num_assets]
    if rows is not None:
        returns = returns.tail(rows)
    validate_returns(returns, f"original_paper_subset ({num_assets} assets, {len(returns)} rows)")
    return returns


def load_custom_nonpaper_10_returns(rows: int = None) -> pd.DataFrame:
    """custom_nonpaper_10 dataset: stored as adjusted close PRICES,
    converted here to daily returns."""
    prices = pd.read_parquet(
        REPO_ROOT / "data" / "custom_portfolios" / "custom_nonpaper_10_prices.parquet.gzip"
    )
    returns = prices.pct_change()
    returns = returns.replace([float("inf"), float("-inf")], pd.NA)
    returns = returns.dropna()
    returns = returns.astype(np.float64)
    if rows is not None:
        returns = returns.tail(rows)
    validate_returns(returns, f"custom_nonpaper_10 ({len(returns)} rows)")
    return returns


def split_60_20_20(returns: pd.DataFrame):
    n = len(returns)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)
    train = returns.iloc[:train_end]
    val = returns.iloc[train_end:val_end]
    test = returns.iloc[val_end:]
    print(f"Split: train {train.shape}, val {val.shape}, test {test.shape}")
    return train, val, test


# ---------------------------------------------------------------------------
# Training with captured epoch losses
# ---------------------------------------------------------------------------

EPOCH_RE = re.compile(
    r"Epoch (\d+)/(\d+), Actor Loss: ([-\d.eE+]+), Critic Loss: ([-\d.eE+]+)"
    r"(?:, Val Critic Loss: ([-\d.eE+]+))?"
)


def train_with_loss_capture(model, train_data, val_data, train_kwargs):
    """Run model.train while capturing stdout; parse per-epoch losses.

    Returns (captured_text, loss_history dict of lists).
    """
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        model.train(train_data=train_data, val_data=val_data, **train_kwargs)
    text = buffer.getvalue()

    history = {"epoch": [], "actor_loss": [], "critic_loss": [], "val_critic_loss": []}
    for match in EPOCH_RE.finditer(text):
        history["epoch"].append(int(match.group(1)))
        history["actor_loss"].append(float(match.group(3)))
        history["critic_loss"].append(float(match.group(4)))
        history["val_critic_loss"].append(
            float(match.group(5)) if match.group(5) else np.nan
        )
    return text, history


# ---------------------------------------------------------------------------
# MVO (self-contained, works for any asset subset)
# ---------------------------------------------------------------------------

def optimize_mvo_weights(data: pd.DataFrame, risk_aversion: float = 10.0,
                         short_selling: bool = True) -> np.ndarray:
    from scipy.optimize import minimize

    mu = data.mean().values
    cov = data.cov().values
    n_assets = len(mu)

    def objective(w):
        portfolio_return = np.dot(w, mu) * 252
        portfolio_variance = np.dot(w, np.dot(cov * 252, w))
        return -(portfolio_return - risk_aversion * portfolio_variance)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(-1.0, 1.0)] * n_assets if short_selling else [(0.0, 1.0)] * n_assets
    x0 = np.ones(n_assets) / n_assets

    result = minimize(objective, x0, method="SLSQP", bounds=bounds,
                      constraints=constraints,
                      options={"maxiter": 1000, "ftol": 1e-9})
    if not result.success:
        print("MVO warning:", result.message, "- falling back to equal weights")
        return x0
    return result.x


def mvo_dpo_evaluation(train_data, test_data, interval: int = 30,
                       risk_aversion: float = 10.0,
                       short_selling: bool = True) -> dict:
    """Rolling MVO re-optimization, returning daily returns AND weights."""
    rolling_data = pd.concat([train_data.tail(interval), test_data])
    num_intervals = len(test_data) // interval + 1

    daily_returns = []
    weights_rows = []

    for i in range(num_intervals):
        train_start = i * interval
        test_start = train_start + interval
        test_end = test_start + interval

        rolling_train = rolling_data.iloc[train_start:test_start]
        rolling_test = rolling_data.iloc[test_start:test_end]
        if len(rolling_train) == 0 or len(rolling_test) == 0:
            continue

        w = optimize_mvo_weights(rolling_train, risk_aversion, short_selling)
        daily_returns.extend(np.sum(rolling_test.values * w, axis=1))
        weights_rows.extend([w] * len(rolling_test))

    return {
        "daily_returns": np.asarray(daily_returns, dtype=np.float64),
        "weights_by_day": np.asarray(weights_rows, dtype=np.float64),
    }


# ---------------------------------------------------------------------------
# The 8 benchmark models
# ---------------------------------------------------------------------------

# Tuned hyperparameters carried over from the existing benchmark scripts
# (originally from the repo's hyperparameter optimization).
HYPERPARAMS = {
    "ddpg_classical": dict(
        actor_lr=0.020239765866555008, critic_lr=0.014249327834891122,
        optimizer=torch.optim.SGD, l2_lambda=0.009585823379719707,
        soft_update=False, risk_preference=-0.2832085400024138,
        gamma=0.028599514945159235,
    ),
    "ddpg_quantum": dict(
        actor_lr=0.09935741130315447, critic_lr=0.0018039893844072358,
        optimizer=torch.optim.SGD, l2_lambda=3.2067524338595386e-06,
        soft_update=False, risk_preference=-0.9286138365176491,
        gamma=0.009826640813865617,
    ),
    "dql_classical": dict(
        actor_lr=0.0011422800982086824, critic_lr=0.003990673146909851,
        optimizer=torch.optim.Adam, l2_lambda=0.005716467080685015,
        soft_update=True, risk_preference=-0.8134627523331615,
        gamma=0.04711405953074143,
    ),
    "dql_quantum": dict(
        actor_lr=0.09488160675184557, critic_lr=0.0011635537865649728,
        optimizer=torch.optim.SGD, l2_lambda=5.030375515009282e-05,
        soft_update=True, risk_preference=-0.12009396389629173,
        gamma=0.0012179475752639956,
    ),
}


def build_rl_model(kind: str, framework: str, config: dict):
    """kind: 'ddpg' or 'dql'; framework: 'classical', 'pennylane', 'qiskit'."""
    from models import DDPG, DeepQLearning
    from predictors import NeuralNetwork, QuantumNeuralNetwork
    from predictors.input_transformations import radial_to_linear
    from qiskit_port.qiskit_exact_amplitude_finite_diff_qnn import (
        QiskitExactAmplitudeFiniteDiffQNN,
    )

    algo_cls = DDPG if kind == "ddpg" else DeepQLearning

    common = dict(
        lookback_window=config["lookback_window"],
        forecast_window=config.get("forecast_window", 0),
        batch_size=1,
        short_selling=config.get("short_selling", True),
        reduce_negatives=config.get("clamp_negatives", True),
        verbose=1,
        seed=config["seed"],
    )

    if framework == "classical":
        return algo_cls(predictor=NeuralNetwork, hidden_sizes=(30,), **common)

    quantum_kwargs = dict(
        num_weights=config["num_weights"],
        encoding="amplitude",
        input_transformation=radial_to_linear,
        rotation_axes="y",
    )
    if framework == "pennylane":
        return algo_cls(predictor=QuantumNeuralNetwork, **quantum_kwargs, **common)
    if framework == "qiskit":
        return algo_cls(
            predictor=QiskitExactAmplitudeFiniteDiffQNN,
            compute_input_gradients=True,
            **quantum_kwargs,
            **common,
        )
    raise ValueError(framework)


def effective_max_epochs(framework: str, config: dict) -> int:
    """Qiskit runs can get a reduced epoch budget via config
    'qiskit_max_epochs' (parity with PennyLane is already proven, so the
    Qiskit entries verify the pipeline rather than re-derive the policy)."""
    if framework == "qiskit" and config.get("qiskit_max_epochs"):
        return config["qiskit_max_epochs"]
    return config["max_epochs"]


def make_train_kwargs(kind: str, framework: str, config: dict) -> dict:
    hp_key = f"{kind}_{'classical' if framework == 'classical' else 'quantum'}"
    kwargs = dict(HYPERPARAMS[hp_key])
    kwargs.update(
        num_epochs=effective_max_epochs(framework, config),
        early_stopping=config.get("early_stopping", True),
        patience=config.get("patience", 5),
        min_delta=config.get("min_delta", 1e-4),
    )
    if kind == "dql" and framework == "classical":
        kwargs["num_action_samples"] = 10
    return kwargs


MODEL_SPECS = [
    # (display name, kind, framework, algorithm label, encoding label)
    ("Equal Weight", "equal_weight", "classical", "Naive baseline", "N/A"),
    ("Mean Variance Optimization", "mvo", "classical", "Convex optimization", "N/A"),
    ("Classical DDPG", "ddpg", "classical", "DDPG", "N/A"),
    ("Classical Deep Q-Learning", "dql", "classical", "Deep Q-Learning", "N/A"),
    ("PennyLane QDPG", "ddpg", "pennylane", "DDPG", "Amplitude"),
    ("PennyLane Quantum Q-Learning", "dql", "pennylane", "Deep Q-Learning", "Amplitude"),
    ("Qiskit QDPG", "ddpg", "qiskit", "DDPG", "Amplitude"),
    ("Qiskit Quantum Q-Learning", "dql", "qiskit", "Deep Q-Learning", "Amplitude"),
]

FRAMEWORK_LABELS = {
    "classical": "PyTorch",
    "pennylane": "PennyLane",
    "qiskit": "Qiskit",
}


def run_single_model(display_name, kind, framework, config,
                     train_data, val_data, test_data):
    """Train and evaluate one model. Returns (result_row, artifacts)."""
    print("\n" + "=" * 80)
    print("Running:", display_name)
    print("=" * 80)

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    start = time.time()
    loss_history = None
    training_text = ""
    notes = []

    if kind == "equal_weight":
        n = test_data.shape[1]
        spo = pm.constant_weights_evaluation(test_data, np.ones(n) / n)
        dpo = spo
        notes.append("static 1/N allocation")
    elif kind == "mvo":
        risk_aversion = 10 if config.get("short_selling", True) else 5
        w = optimize_mvo_weights(train_data, risk_aversion,
                                 config.get("short_selling", True))
        spo = pm.constant_weights_evaluation(test_data, w)
        dpo = mvo_dpo_evaluation(train_data, test_data, interval=30,
                                 risk_aversion=risk_aversion,
                                 short_selling=config.get("short_selling", True))
        notes.append("DPO re-optimized every 30 days")
    else:
        model = build_rl_model(kind, framework, config)
        train_kwargs = make_train_kwargs(kind, framework, config)
        training_text, loss_history = train_with_loss_capture(
            model, train_data, val_data, train_kwargs
        )
        epochs_done = len(loss_history["epoch"])
        max_epochs = effective_max_epochs(framework, config)
        stopped_early = epochs_done < max_epochs
        notes.append(f"epochs={epochs_done}/{max_epochs}"
                     + (" (early stop)" if stopped_early else ""))
        if max_epochs != config["max_epochs"]:
            notes.append("reduced Qiskit epoch budget (parity-verified vs PennyLane)")

        spo = pm.evaluate_actor_spo_with_weights(
            model.actor, val_data, test_data,
            clamp_negatives=config.get("clamp_negatives", True),
        )
        dpo = pm.evaluate_actor_with_weights(
            model.actor, val_data, test_data,
            interval=config["lookback_window"],
            clamp_negatives=config.get("clamp_negatives", True),
        )
        notes.append(f"DPO rebalance every {config['lookback_window']} days")

    runtime = time.time() - start

    spo_metrics = pm.compute_all_metrics(spo["daily_returns"], spo["weights_by_day"])
    dpo_metrics = pm.compute_all_metrics(dpo["daily_returns"], dpo["weights_by_day"])

    row = {
        "model": display_name,
        "framework": FRAMEWORK_LABELS[framework],
        "algorithm": dict((m[0], m[3]) for m in MODEL_SPECS)[display_name],
        "encoding": dict((m[0], m[4]) for m in MODEL_SPECS)[display_name],
        "dataset": config["dataset_name"],
        "runtime_seconds": runtime,
        "notes": "; ".join(notes),
    }
    for key, val in spo_metrics.items():
        row[f"spo_{key}"] = val
    for key, val in dpo_metrics.items():
        row[f"dpo_{key}"] = val

    print(f"Runtime: {runtime:.1f}s")
    print(f"SPO profit p.a.: {spo_metrics['profit_pa']*100:.2f}%  sharpe {spo_metrics['sharpe']:.3f}")
    print(f"DPO profit p.a.: {dpo_metrics['profit_pa']*100:.2f}%  sharpe {dpo_metrics['sharpe']:.3f}")
    print(f"DPO turnover: {dpo_metrics['avg_turnover']:.4f}  "
          f"TC-adj return: {dpo_metrics['tc_adjusted_annual_return']*100:.2f}%")

    artifacts = {
        "spo": spo,
        "dpo": dpo,
        "loss_history": loss_history,
        "training_text": training_text,
    }
    return row, artifacts


def safe_model_name(display_name: str) -> str:
    return display_name.lower().replace(" ", "_").replace("-", "_")


def run_benchmark(config, train_data, val_data, test_data, output_prefix: str,
                  model_specs=None, resume: bool = True):
    """Run all models, save CSV/TeX/series, return the results DataFrame.

    Resumable: after each model, its result row is persisted to
    series/<prefix>_<model>_row.json. On restart with resume=True, models
    with an existing row JSON (and series npz) are loaded, not re-run.
    """
    specs = model_specs if model_specs is not None else MODEL_SPECS

    rows = []
    for display_name, kind, framework, _algo, _enc in specs:
        safe_name = safe_model_name(display_name)
        row_path = SERIES_DIR / f"{output_prefix}_{safe_name}_row.json"
        npz_path = SERIES_DIR / f"{output_prefix}_{safe_name}.npz"

        if resume and row_path.exists() and npz_path.exists():
            rows.append(json.loads(row_path.read_text()))
            print(f"\nResume: skipping {display_name} (cached row + series)")
            continue

        try:
            row, artifacts = run_single_model(
                display_name, kind, framework, config,
                train_data, val_data, test_data,
            )
        except Exception as exc:  # keep the benchmark alive if one model dies
            print(f"ERROR in {display_name}: {type(exc).__name__}: {exc}")
            rows.append({
                "model": display_name,
                "framework": FRAMEWORK_LABELS[framework],
                "dataset": config["dataset_name"],
                "notes": f"FAILED: {type(exc).__name__}: {exc}",
            })
            continue

        rows.append(row)

        series_path = npz_path
        np.savez_compressed(
            series_path,
            spo_daily_returns=artifacts["spo"]["daily_returns"],
            spo_weights=artifacts["spo"]["weights_by_day"],
            dpo_daily_returns=artifacts["dpo"]["daily_returns"],
            dpo_weights=artifacts["dpo"]["weights_by_day"],
            actor_loss=np.array(artifacts["loss_history"]["actor_loss"])
            if artifacts["loss_history"] else np.array([]),
            critic_loss=np.array(artifacts["loss_history"]["critic_loss"])
            if artifacts["loss_history"] else np.array([]),
            val_critic_loss=np.array(artifacts["loss_history"]["val_critic_loss"])
            if artifacts["loss_history"] else np.array([]),
        )
        row_path.write_text(json.dumps(row, default=float))

    df = pd.DataFrame(rows)

    csv_path = COMPARISON_DIR / f"{output_prefix}_results.csv"
    tex_path = COMPARISON_DIR / f"{output_prefix}_results.tex"
    md_path = COMPARISON_DIR / f"{output_prefix}_results.md"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)

    tex_cols = [
        "model", "framework", "algorithm", "dataset",
        "dpo_profit_pa", "dpo_sharpe", "dpo_annualized_return",
        "dpo_annualized_volatility", "dpo_var_5", "dpo_avg_turnover",
        "dpo_tc_adjusted_annual_return", "runtime_seconds", "notes",
    ]
    tex_df = df[[c for c in tex_cols if c in df.columns]]
    tex_df.to_latex(tex_path, index=False, float_format="%.4f")

    print("\nSaved:", csv_path.name, tex_path.name, md_path.name)
    return df

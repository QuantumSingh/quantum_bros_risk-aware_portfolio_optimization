"""Train the authors' models at a chosen circuit depth and record everything.

Reproduces one model block of the authors' MAIN.py exactly (same data, same
7-fold time-series cross-validation, same hyperparameters, seed 68), using
the authors' own classes. The one thing that varies is num_weights, which
sets how many repeated layers the quantum circuit has (see inspect_circuit.py).

Each test period ("fold") runs independently: the authors' code re-seeds
Python, NumPy and Torch when it builds each fold's model, so running folds
in separate processes gives the same numbers as MAIN.py's single run.
Finished folds are skipped, so an interrupted run picks up where it stopped.

Per fold it writes results/runs/<run>/fold_<k>.json containing:
  - the loss curve: actor and critic loss after every epoch
  - the authors' own SPO and DPO profit and Sharpe (model.evaluate)
  - every 30-day rebalance: the portfolio weights the model chose and each
    asset's actual return over the next 30 days (for the confusion matrix)
  - train and evaluation runtime

Usage (from repo root):
    $PY -u pennylane_circuit_reduction/run_depth_sweep.py --model QDPG --num-weights 30
    $PY -u pennylane_circuit_reduction/run_depth_sweep.py --model QDPG --num-weights 30 --folds 6 7
    $PY -u pennylane_circuit_reduction/run_depth_sweep.py --model DDPG      # classical check
"""
import argparse
import io
import json
import os
import re
import sys
import time
import warnings
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
AUTHORS = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(AUTHORS))
os.chdir(AUTHORS)

import numpy as np
import pandas as pd
import torch

from config import tickers
from models import DDPG, DeepQLearning
from predictors import NeuralNetwork, QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
from utilities.data_processing import TimeSeriesCrossValidation, arima_forecast
from utilities.metrics import calculate_test_performance, reduce_negatives

warnings.filterwarnings("ignore")

RUNS = HERE / "results" / "runs"

# GLOBAL_CONFIG from the authors' MAIN.py.
CFG = dict(LOOKBACK_WINDOW=30, FORECAST_WINDOW=7, DPO_INTERVAL=30,
           SHORT_SELLING=True, CLAMP_NEGATIVES=True, CV_SPLITS=7, SEED=68)

QUANTUM = dict(predictor=QuantumNeuralNetwork, encoding="amplitude",
               input_transformation=radial_to_linear, rotation_axes="y")
CLASSICAL = dict(predictor=NeuralNetwork, hidden_sizes=(30,))

# Per-model settings, copied from the authors' MAIN.py.
MODELS = {
    "DDPG": (DDPG, CLASSICAL, dict(
        actor_lr=0.020239765866555008, critic_lr=0.014249327834891122,
        optimizer=torch.optim.SGD, l2_lambda=0.009585823379719707,
        soft_update=False, risk_preference=-0.2832085400024138,
        gamma=0.028599514945159235)),
    "QDPG": (DDPG, QUANTUM, dict(
        actor_lr=0.09935741130315447, critic_lr=0.0018039893844072358,
        optimizer=torch.optim.SGD, l2_lambda=3.2067524338595386e-06,
        soft_update=False, risk_preference=-0.9286138365176491,
        gamma=0.009826640813865617)),
    "DQL": (DeepQLearning, CLASSICAL, dict(
        actor_lr=0.0011422800982086824, critic_lr=0.003990673146909851,
        optimizer=torch.optim.Adam, l2_lambda=0.005716467080685015,
        soft_update=True, risk_preference=-0.8134627523331615,
        gamma=0.04711405953074143, num_action_samples=10)),
    "QQL": (DeepQLearning, QUANTUM, dict(
        actor_lr=0.09488160675184557, critic_lr=0.0011635537865649728,
        optimizer=torch.optim.SGD, l2_lambda=5.030375515009282e-05,
        soft_update=True, risk_preference=-0.12009396389629173,
        gamma=0.0012179475752639956)),
}
COMMON_TRAIN = dict(num_epochs=50, early_stopping=False, patience=10)

EPOCH_RE = re.compile(r"Epoch (\d+)/\d+, Actor Loss: (\S+), Critic Loss: (\S+?),?\s")


def run_name(model, num_weights):
    return f"{model}_{num_weights}w" if model in ("QDPG", "QQL") else model


def folds():
    """The authors' 7 folds, as (fold number, train index, test index)."""
    price_data = pd.read_parquet("./data/price_data.parquet.gzip")
    cv = TimeSeriesCrossValidation(data=price_data, n_splits=CFG["CV_SPLITS"])
    with redirect_stdout(io.StringIO()):  # the authors' CV prints a progress bar
        splits = list(cv())
    return price_data, [(k, tr, te) for k, (tr, te) in enumerate(splits, start=1)]


def dpo_with_allocations(model, val_data, test_data):
    """The authors' RLEvaluator.evaluate_dpo, step for step, but keeping
    each rebalance's weights and the assets' returns over the holding period."""
    actor, forecast = model.actor, model.forecast_window
    window = int(actor.input_size / len(tickers)) - forecast
    interval = model.lookback_window
    merged = pd.concat((val_data.tail(window), test_data))
    actor.eval()
    rebalances, daily = [], []
    for i in range(len(test_data) // interval + 1):
        start = i * interval
        rolling_train = merged[start:start + interval]
        rolling_test = merged[start + interval:start + 2 * interval]
        with torch.no_grad():
            state = torch.tensor(
                arima_forecast(rolling_train.tail(window).values, forecast),
                dtype=torch.float32)
            weights = actor(state.flatten()).numpy().squeeze()
        if model.reduce_negatives:
            weights = reduce_negatives(weights)
        daily.extend(np.sum(rolling_test.values * weights, axis=1))
        if len(rolling_test):
            rebalances.append({
                "start": str(rolling_test.index[0].date()),
                "end": str(rolling_test.index[-1].date()),
                "weights": [float(w) for w in weights],
                "asset_returns": [float(r) for r in (1 + rolling_test).prod() - 1],
            })
    profit, sharpe = calculate_test_performance(np.array(daily))
    return rebalances, float(profit), float(sharpe)


def run_fold(model_name, num_weights, fold, price_data, train_index, test_index, out_dir):
    algo, predictor_kwargs, train_kwargs = MODELS[model_name]
    predictor_kwargs = dict(predictor_kwargs)
    if model_name in ("QDPG", "QQL"):
        predictor_kwargs["num_weights"] = num_weights

    val_split = int(len(price_data.iloc[train_index]) * 0.8)
    train_data = price_data.iloc[train_index][:val_split]
    val_data = price_data.iloc[train_index][val_split:]
    test_data = price_data.iloc[test_index]

    model = algo(
        lookback_window=CFG["LOOKBACK_WINDOW"], forecast_window=CFG["FORECAST_WINDOW"],
        batch_size=1, short_selling=CFG["SHORT_SELLING"],
        reduce_negatives=CFG["CLAMP_NEGATIVES"], verbose=1, seed=CFG["SEED"],
        **predictor_kwargs,
    )

    start = time.time()
    log = io.StringIO()
    with redirect_stdout(log):  # verbose=1 only adds per-epoch loss prints
        model.train(train_data=train_data, val_data=val_data,
                    **train_kwargs, **COMMON_TRAIN)
    train_s = time.time() - start
    losses = EPOCH_RE.findall(log.getvalue() + "\n")

    model.verbose = 0
    start = time.time()
    (spo_profit, spo_sharpe), (dpo_profit, dpo_sharpe) = model.evaluate(
        test_data=test_data, dpo=True)
    rebalances, mirror_profit, mirror_sharpe = dpo_with_allocations(model, val_data, test_data)
    eval_s = time.time() - start

    if not (np.isclose(mirror_profit, dpo_profit) and np.isclose(mirror_sharpe, dpo_sharpe)):
        raise RuntimeError(f"Fold {fold}: allocation replay ({mirror_sharpe}) does not "
                           f"match the authors' DPO evaluation ({dpo_sharpe})")

    result = {
        "model": model_name, "num_weights": num_weights if model_name in ("QDPG", "QQL") else None,
        "fold": fold,
        "train_period": [str(train_data.index[0].date()), str(train_data.index[-1].date())],
        "val_period": [str(val_data.index[0].date()), str(val_data.index[-1].date())],
        "test_period": [str(test_data.index[0].date()), str(test_data.index[-1].date())],
        "train_rows": len(train_data), "val_rows": len(val_data), "test_rows": len(test_data),
        "epochs": len(losses),
        "actor_loss": [float(a) for _, a, _ in losses],
        "critic_loss": [float(c) for _, _, c in losses],
        "spo_profit": float(spo_profit), "spo_sharpe": float(spo_sharpe),
        "dpo_profit": float(dpo_profit), "dpo_sharpe": float(dpo_sharpe),
        "train_seconds": train_s, "eval_seconds": eval_s,
        "rebalances": rebalances,
    }
    (out_dir / f"fold_{fold}.json").write_text(json.dumps(result))
    print(f"[{run_name(model_name, num_weights)}] fold {fold}: test {result['test_period'][0]}.."
          f"{result['test_period'][1]} | DPO Sharpe {dpo_sharpe:.4f} | SPO Sharpe {spo_sharpe:.4f} | "
          f"train {train_s/60:.1f} min", flush=True)


def write_summary(model_name, num_weights, out_dir):
    """MAIN.py-style summary (mean, IQR, min, max over folds) once all folds exist."""
    files = sorted(out_dir.glob("fold_*.json"))
    if len(files) < CFG["CV_SPLITS"]:
        return
    rows = [json.loads(f.read_text()) for f in files]
    summary = {"model": model_name, "num_weights": rows[0]["num_weights"], "folds": len(rows),
               "train_minutes_total": sum(r["train_seconds"] for r in rows) / 60}
    for key in ("spo_profit", "spo_sharpe", "dpo_profit", "dpo_sharpe"):
        vals = np.array([r[key] for r in rows])
        summary[key] = {"mean": float(vals.mean()),
                        "iqr": float(np.nanpercentile(vals, 75) - np.nanpercentile(vals, 25)),
                        "min": float(vals.min()), "max": float(vals.max())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{run_name(model_name, num_weights)}] all folds done: "
          f"DPO Sharpe {summary['dpo_sharpe']['mean']:.4f}, "
          f"SPO Sharpe {summary['spo_sharpe']['mean']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=list(MODELS), default="QDPG")
    parser.add_argument("--num-weights", type=int, default=60)
    parser.add_argument("--folds", type=int, nargs="*", default=None)
    args = parser.parse_args()

    out_dir = RUNS / run_name(args.model, args.num_weights)
    out_dir.mkdir(parents=True, exist_ok=True)
    price_data, all_folds = folds()
    wanted = set(args.folds) if args.folds else {k for k, _, _ in all_folds}

    for k, tr, te in all_folds:
        if k not in wanted:
            continue
        if (out_dir / f"fold_{k}.json").exists():
            print(f"[{run_name(args.model, args.num_weights)}] fold {k}: already done, skipping")
            continue
        run_fold(args.model, args.num_weights, k, price_data, tr, te, out_dir)

    write_summary(args.model, args.num_weights, out_dir)


if __name__ == "__main__":
    main()

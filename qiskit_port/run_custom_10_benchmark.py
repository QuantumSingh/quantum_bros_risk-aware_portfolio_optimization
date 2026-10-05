import os
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

os.chdir(ORIGINAL_CODE)

from models import EqualWeights, MeanVarianceOptimization, DDPG, DeepQLearning
from predictors import NeuralNetwork, QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
from qiskit_port.qiskit_exact_amplitude_finite_diff_qnn import (
    QiskitExactAmplitudeFiniteDiffQNN,
)
import utilities.metrics as metrics_module


BENCHMARK_CONFIG = {
    "rows": 120,
    "num_assets": 10,
    "lookback_window": 5,
    "forecast_window": 0,
    "num_weights": 10,
    "num_epochs": 1,
    "seed": 68,
    "short_selling": True,
    "clamp_negatives": True,
    "dpo": True,
}


def unpack_results(results):
    """
    Handles:
    - EqualWeights style: (profit, sharpe)
    - RL/MVO DPO style: ((spo_profit, spo_sharpe), (dpo_profit, dpo_sharpe))
    """
    if isinstance(results, tuple) and len(results) == 2:
        first, second = results

        if isinstance(first, tuple) or isinstance(first, list):
            spo_profit, spo_sharpe = first
            dpo_profit, dpo_sharpe = second
            return float(spo_profit), float(spo_sharpe), float(dpo_profit), float(dpo_sharpe)

        return float(first), float(second), None, None

    raise ValueError(f"Unexpected results format: {results}")


def add_result(rows, model_name, framework, algorithm, encoding, results, runtime):
    spo_profit, spo_sharpe, dpo_profit, dpo_sharpe = unpack_results(results)

    rows.append({
        "model": model_name,
        "framework": framework,
        "algorithm": algorithm,
        "encoding": encoding,
        "spo_profit": spo_profit,
        "spo_profit_pct": spo_profit * 100,
        "spo_sharpe": spo_sharpe,
        "dpo_profit": dpo_profit,
        "dpo_profit_pct": None if dpo_profit is None else dpo_profit * 100,
        "dpo_sharpe": dpo_sharpe,
        "runtime_seconds": runtime,
    })


def run_and_time(name, fn):
    print("\n" + "=" * 80)
    print("Running:", name)
    print("=" * 80)

    start = time.time()
    results = fn()
    runtime = time.time() - start

    print("Finished:", name)
    print("Runtime seconds:", runtime)
    print("Results:", results)

    return results, runtime


def main():
    torch.manual_seed(BENCHMARK_CONFIG["seed"])

    print("Benchmark config:")
    for k, v in BENCHMARK_CONFIG.items():
        print(f"{k}: {v}")

    price_data = pd.read_parquet(REPO_ROOT / "data/custom_portfolios/custom_10_mixed_prices.parquet.gzip")
    print("\nFull price_data shape:", price_data.shape)

    price_data = price_data.iloc[:, : BENCHMARK_CONFIG["num_assets"]]
    price_data = price_data.tail(BENCHMARK_CONFIG["rows"])

    metrics_module.tickers = list(price_data.columns)

    print("Benchmark price_data shape:", price_data.shape)
    print("Benchmark tickers:", list(price_data.columns))

    n = len(price_data)
    train_end = int(0.6 * n)
    val_end = int(0.8 * n)

    train_data = price_data.iloc[:train_end]
    val_data = price_data.iloc[train_end:val_end]
    test_data = price_data.iloc[val_end:]

    print("Train shape:", train_data.shape)
    print("Val shape:", val_data.shape)
    print("Test shape:", test_data.shape)

    rows = []

    # ---------------------------------------------------------------------
    # 1. Equal Weight baseline
    # ---------------------------------------------------------------------
    def run_equal_weight():
        # The original EqualWeights() class assumes the full 15-asset dataset.
        # This benchmark uses a sliced asset subset, so we compute equal weights
        # directly for the selected benchmark columns.
        weights = [1.0 / len(price_data.columns)] * len(price_data.columns)

        # SPO: hold equal weights over the full test period.
        spo = metrics_module.calculate_test_performance(test_data, weights)

        # DPO: periodically rebalance back to equal weights.
        # With no transaction costs, this is equivalent to the same equal-weight exposure.
        dpo = metrics_module.calculate_test_performance(test_data, weights)

        return spo, dpo

    results, runtime = run_and_time(
        "Equal Weight",
        run_equal_weight,
    )
    add_result(
        rows,
        "Equal Weight",
        "Classical",
        "Naive baseline",
        "N/A",
        results,
        runtime,
    )

    # ---------------------------------------------------------------------
    # 2. Mean Variance Optimization baseline
    # ---------------------------------------------------------------------
    def optimize_mvo_weights(data, risk_aversion=10.0, short_selling=True):
        """
        Self-contained MVO optimizer for the selected benchmark asset universe.
        This avoids the original repo's global 15-ticker assumptions.
        """
        from scipy.optimize import minimize

        mu = data.mean().values
        cov = data.cov().values
        n_assets = len(mu)

        def objective(w):
            portfolio_return = np.dot(w, mu) * 252
            portfolio_variance = np.dot(w, np.dot(cov * 252, w))
            return -(portfolio_return - risk_aversion * portfolio_variance)

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        if short_selling:
            bounds = [(-1.0, 1.0)] * n_assets
        else:
            bounds = [(0.0, 1.0)] * n_assets

        x0 = np.ones(n_assets) / n_assets

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        if not result.success:
            print("MVO warning:", result.message)
            print("Falling back to equal weights.")
            return x0

        return result.x

    def run_mvo():
        risk_aversion = 10 if BENCHMARK_CONFIG["short_selling"] else 5

        # Static portfolio optimization
        spo_weights = optimize_mvo_weights(
            train_data,
            risk_aversion=risk_aversion,
            short_selling=BENCHMARK_CONFIG["short_selling"],
        )
        spo = metrics_module.calculate_test_performance(test_data, spo_weights)

        # Dynamic portfolio optimization with periodic re-optimization
        interval = 30
        rolling_data = pd.concat([train_data.tail(interval), test_data])
        num_intervals = len(test_data) // interval + 1

        all_daily_returns = []

        for i in range(num_intervals):
            train_start = i * interval
            test_start = train_start + interval
            test_end = test_start + interval

            rolling_train = rolling_data.iloc[train_start:test_start]
            rolling_test = rolling_data.iloc[test_start:test_end]

            if len(rolling_train) == 0 or len(rolling_test) == 0:
                continue

            dpo_weights = optimize_mvo_weights(
                rolling_train,
                risk_aversion=risk_aversion,
                short_selling=BENCHMARK_CONFIG["short_selling"],
            )

            daily_returns = np.sum(rolling_test.values * dpo_weights, axis=1)
            all_daily_returns.extend(daily_returns)

        dpo = metrics_module.calculate_test_performance(np.array(all_daily_returns))

        return spo, dpo

    results, runtime = run_and_time("Mean Variance Optimization", run_mvo)
    add_result(
        rows,
        "Mean Variance Optimization",
        "Classical",
        "Convex optimization",
        "N/A",
        results,
        runtime,
    )

    # ---------------------------------------------------------------------
    # 3. Classical DDPG
    # ---------------------------------------------------------------------
    def run_classical_ddpg():
        model = DDPG(
            lookback_window=BENCHMARK_CONFIG["lookback_window"],
            forecast_window=BENCHMARK_CONFIG["forecast_window"],
            batch_size=1,
            predictor=NeuralNetwork,
            hidden_sizes=(30,),
            short_selling=BENCHMARK_CONFIG["short_selling"],
            reduce_negatives=BENCHMARK_CONFIG["clamp_negatives"],
            verbose=1,
            seed=BENCHMARK_CONFIG["seed"],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.020239765866555008,
            critic_lr=0.014249327834891122,
            optimizer=torch.optim.SGD,
            l2_lambda=0.009585823379719707,
            soft_update=False,
            risk_preference=-0.2832085400024138,
            gamma=0.028599514945159235,
            num_epochs=BENCHMARK_CONFIG["num_epochs"],
            early_stopping=False,
            patience=10,
        )
        return model.evaluate(test_data=test_data, dpo=BENCHMARK_CONFIG["dpo"])

    results, runtime = run_and_time("Classical DDPG", run_classical_ddpg)
    add_result(rows, "Classical DDPG", "PyTorch", "DDPG", "N/A", results, runtime)

    # ---------------------------------------------------------------------
    # 4. PennyLane QDPG
    # ---------------------------------------------------------------------
    def run_pennylane_qdpg():
        model = DDPG(
            lookback_window=BENCHMARK_CONFIG["lookback_window"],
            forecast_window=BENCHMARK_CONFIG["forecast_window"],
            batch_size=1,
            predictor=QuantumNeuralNetwork,
            num_weights=BENCHMARK_CONFIG["num_weights"],
            encoding="amplitude",
            input_transformation=radial_to_linear,
            rotation_axes="y",
            short_selling=BENCHMARK_CONFIG["short_selling"],
            reduce_negatives=BENCHMARK_CONFIG["clamp_negatives"],
            verbose=1,
            seed=BENCHMARK_CONFIG["seed"],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.09935741130315447,
            critic_lr=0.0018039893844072358,
            optimizer=torch.optim.SGD,
            l2_lambda=3.2067524338595386e-06,
            soft_update=False,
            risk_preference=-0.9286138365176491,
            gamma=0.009826640813865617,
            num_epochs=BENCHMARK_CONFIG["num_epochs"],
            early_stopping=False,
            patience=10,
        )
        return model.evaluate(test_data=test_data, dpo=BENCHMARK_CONFIG["dpo"])

    results, runtime = run_and_time("PennyLane QDPG", run_pennylane_qdpg)
    add_result(rows, "PennyLane QDPG", "PennyLane", "DDPG", "Amplitude", results, runtime)

    # ---------------------------------------------------------------------
    # 5. Qiskit QDPG
    # ---------------------------------------------------------------------
    def run_qiskit_qdpg():
        model = DDPG(
            lookback_window=BENCHMARK_CONFIG["lookback_window"],
            forecast_window=BENCHMARK_CONFIG["forecast_window"],
            batch_size=1,
            predictor=QiskitExactAmplitudeFiniteDiffQNN,
            num_weights=BENCHMARK_CONFIG["num_weights"],
            encoding="amplitude",
            input_transformation=radial_to_linear,
            rotation_axes="y",
            short_selling=BENCHMARK_CONFIG["short_selling"],
            reduce_negatives=BENCHMARK_CONFIG["clamp_negatives"],
            verbose=1,
            seed=BENCHMARK_CONFIG["seed"],
            compute_input_gradients=True,
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.09935741130315447,
            critic_lr=0.0018039893844072358,
            optimizer=torch.optim.SGD,
            l2_lambda=3.2067524338595386e-06,
            soft_update=False,
            risk_preference=-0.9286138365176491,
            gamma=0.009826640813865617,
            num_epochs=BENCHMARK_CONFIG["num_epochs"],
            early_stopping=False,
            patience=10,
        )
        return model.evaluate(test_data=test_data, dpo=BENCHMARK_CONFIG["dpo"])

    results, runtime = run_and_time("Qiskit QDPG", run_qiskit_qdpg)
    add_result(rows, "Qiskit QDPG", "Qiskit", "DDPG", "Amplitude", results, runtime)

    # ---------------------------------------------------------------------
    # 6. Classical Deep Q-Learning
    # ---------------------------------------------------------------------
    def run_classical_dql():
        model = DeepQLearning(
            lookback_window=BENCHMARK_CONFIG["lookback_window"],
            forecast_window=BENCHMARK_CONFIG["forecast_window"],
            batch_size=1,
            predictor=NeuralNetwork,
            hidden_sizes=(30,),
            short_selling=BENCHMARK_CONFIG["short_selling"],
            reduce_negatives=BENCHMARK_CONFIG["clamp_negatives"],
            verbose=1,
            seed=BENCHMARK_CONFIG["seed"],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.0011422800982086824,
            critic_lr=0.003990673146909851,
            optimizer=torch.optim.Adam,
            l2_lambda=0.005716467080685015,
            soft_update=True,
            risk_preference=-0.8134627523331615,
            gamma=0.04711405953074143,
            num_epochs=BENCHMARK_CONFIG["num_epochs"],
            early_stopping=False,
            patience=10,
            num_action_samples=10,
        )
        return model.evaluate(test_data=test_data, dpo=BENCHMARK_CONFIG["dpo"])

    results, runtime = run_and_time("Classical Deep Q-Learning", run_classical_dql)
    add_result(rows, "Classical Deep Q-Learning", "PyTorch", "Deep Q-Learning", "N/A", results, runtime)

    # ---------------------------------------------------------------------
    # 7. PennyLane Quantum Q-Learning
    # ---------------------------------------------------------------------
    def run_pennylane_qql():
        model = DeepQLearning(
            lookback_window=BENCHMARK_CONFIG["lookback_window"],
            forecast_window=BENCHMARK_CONFIG["forecast_window"],
            batch_size=1,
            predictor=QuantumNeuralNetwork,
            num_weights=BENCHMARK_CONFIG["num_weights"],
            encoding="amplitude",
            input_transformation=radial_to_linear,
            rotation_axes="y",
            short_selling=BENCHMARK_CONFIG["short_selling"],
            reduce_negatives=BENCHMARK_CONFIG["clamp_negatives"],
            verbose=1,
            seed=BENCHMARK_CONFIG["seed"],
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.09488160675184557,
            critic_lr=0.0011635537865649728,
            optimizer=torch.optim.SGD,
            l2_lambda=5.030375515009282e-05,
            soft_update=True,
            risk_preference=-0.12009396389629173,
            gamma=0.0012179475752639956,
            num_epochs=BENCHMARK_CONFIG["num_epochs"],
            early_stopping=False,
            patience=10,
        )
        return model.evaluate(test_data=test_data, dpo=BENCHMARK_CONFIG["dpo"])

    results, runtime = run_and_time("PennyLane Quantum Q-Learning", run_pennylane_qql)
    add_result(rows, "PennyLane Quantum Q-Learning", "PennyLane", "Deep Q-Learning", "Amplitude", results, runtime)

    # ---------------------------------------------------------------------
    # 8. Qiskit Quantum Q-Learning
    # ---------------------------------------------------------------------
    def run_qiskit_qql():
        model = DeepQLearning(
            lookback_window=BENCHMARK_CONFIG["lookback_window"],
            forecast_window=BENCHMARK_CONFIG["forecast_window"],
            batch_size=1,
            predictor=QiskitExactAmplitudeFiniteDiffQNN,
            num_weights=BENCHMARK_CONFIG["num_weights"],
            encoding="amplitude",
            input_transformation=radial_to_linear,
            rotation_axes="y",
            short_selling=BENCHMARK_CONFIG["short_selling"],
            reduce_negatives=BENCHMARK_CONFIG["clamp_negatives"],
            verbose=1,
            seed=BENCHMARK_CONFIG["seed"],
            compute_input_gradients=True,
        )
        model.train(
            train_data=train_data,
            val_data=val_data,
            actor_lr=0.09488160675184557,
            critic_lr=0.0011635537865649728,
            optimizer=torch.optim.SGD,
            l2_lambda=5.030375515009282e-05,
            soft_update=True,
            risk_preference=-0.12009396389629173,
            gamma=0.0012179475752639956,
            num_epochs=BENCHMARK_CONFIG["num_epochs"],
            early_stopping=False,
            patience=10,
        )
        return model.evaluate(test_data=test_data, dpo=BENCHMARK_CONFIG["dpo"])

    results, runtime = run_and_time("Qiskit Quantum Q-Learning", run_qiskit_qql)
    add_result(rows, "Qiskit Quantum Q-Learning", "Qiskit", "Deep Q-Learning", "Amplitude", results, runtime)

    # ---------------------------------------------------------------------
    # Save benchmark results
    # ---------------------------------------------------------------------
    output_dir = REPO_ROOT / "comparison_logs"
    output_dir.mkdir(exist_ok=True)

    df = pd.DataFrame(rows)

    csv_path = output_dir / "custom_10_benchmark_results.csv"
    md_path = output_dir / "custom_10_benchmark_results.md"
    tex_path = output_dir / "custom_10_benchmark_results.tex"

    df.to_csv(csv_path, index=False)
    df.to_markdown(md_path, index=False)
    df.to_latex(tex_path, index=False, float_format="%.4f")

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS")
    print("=" * 80)
    print(df)

    print("\nSaved:")
    print(csv_path)
    print(md_path)
    print(tex_path)


if __name__ == "__main__":
    main()

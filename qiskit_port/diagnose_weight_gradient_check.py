"""Compare PennyLane autograd weight gradients with central finite differences."""
import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"
sys.path.insert(0, str(ORIGINAL_CODE))

from predictors import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear


PLOTS_DIR = REPO_ROOT / "comparison_logs" / "plots"


def parse_positive_ints(value: str) -> list[int]:
    try:
        values = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected comma-separated integers.") from exc
    if not values or any(number < 1 for number in values):
        raise argparse.ArgumentTypeError("All values must be positive integers.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Values must not contain duplicates.")
    return values


def parse_positive_floats(value: str) -> list[float]:
    try:
        values = [float(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "Expected comma-separated positive numbers."
        ) from exc
    if not values or any(not math.isfinite(number) or number <= 0 for number in values):
        raise argparse.ArgumentTypeError("All step sizes must be finite and positive.")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("Step sizes must not contain duplicates.")
    return values


def load_fixed_windows(
    data_path: Path,
    input_type: str,
    rows: int,
    num_assets: int,
    lookback: int,
    num_windows: int,
) -> np.ndarray:
    data = pd.read_parquet(data_path)
    if input_type == "prices":
        data = data.pct_change().dropna()
    returns = data.iloc[:, :num_assets].tail(rows)
    values = returns.to_numpy(dtype=np.float32)
    if values.shape[1] != num_assets or not np.isfinite(values).all():
        raise ValueError(f"Invalid or incomplete return data in {data_path}")

    train_returns = values[:int(0.6 * len(values))]
    available = len(train_returns) - lookback + 1
    if available < 2:
        raise ValueError("Not enough training rows for the requested lookback.")
    if num_windows < 2 or num_windows > available:
        raise ValueError(
            f"Choose 2 to {available} input windows; received {num_windows}."
        )

    starts = np.linspace(0, available - 1, num_windows, dtype=int)
    windows = np.stack(
        [train_returns[start:start + lookback].reshape(-1) for start in starts]
    )
    return windows


def build_model(
    input_size: int, num_assets: int, depth: int, seed: int
) -> QuantumNeuralNetwork:
    num_qubits = max(math.ceil(math.log2(input_size * 4)), num_assets)
    return QuantumNeuralNetwork(
        input_size=input_size,
        output_size=num_assets,
        num_qubits=num_qubits,
        num_weights=num_qubits * depth,
        encoding="amplitude",
        input_transformation=radial_to_linear,
        rotation_axes="y",
        entanglement="reverse_linear",
        seed=seed,
    )


def measure_gradients(
    windows: np.ndarray,
    depth: int,
    seed: int,
    num_assets: int,
    output_index: int,
    step_sizes: list[float],
) -> tuple[np.ndarray, dict[float, np.ndarray]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_model(windows.shape[1], num_assets, depth, seed).double()
    inputs = torch.tensor(windows, dtype=torch.float64)
    outputs = model(inputs)
    if not torch.isfinite(outputs).all():
        raise ValueError(f"Non-finite model output at depth {depth}.")

    weights = dict(model.named_parameters())["qlayer.weights"]
    scalar_output = outputs[0, output_index]
    autograd = torch.autograd.grad(scalar_output, weights)[0].detach().cpu().numpy()
    if not np.isfinite(autograd).all():
        raise ValueError(f"Non-finite autograd gradient at depth {depth}.")

    base_weights = weights.detach().clone()
    finite_differences: dict[float, np.ndarray] = {}
    try:
        for step in step_sizes:
            finite_difference = np.empty_like(autograd)
            for weight_index in range(weights.numel()):
                with torch.no_grad():
                    weights.copy_(base_weights)
                    weights[weight_index] += step
                with torch.no_grad():
                    output_plus = model(inputs)[0, output_index].item()

                with torch.no_grad():
                    weights.copy_(base_weights)
                    weights[weight_index] -= step
                with torch.no_grad():
                    output_minus = model(inputs)[0, output_index].item()

                finite_difference[weight_index] = (
                    output_plus - output_minus
                ) / (2 * step)
            finite_differences[step] = finite_difference
    finally:
        with torch.no_grad():
            weights.copy_(base_weights)

    return autograd, finite_differences


def summarize_comparison(
    depth: int, step: float, autograd: np.ndarray, finite_difference: np.ndarray
) -> dict:
    error = finite_difference - autograd
    rmse = float(np.sqrt(np.mean(error**2)))
    autograd_rms = float(np.sqrt(np.mean(autograd**2)))
    denominator = float(np.linalg.norm(autograd) * np.linalg.norm(finite_difference))
    cosine_similarity = (
        float(np.dot(autograd, finite_difference) / denominator)
        if denominator
        else float("nan")
    )
    return {
        "depth_layers": depth,
        "num_weights": len(autograd),
        "epsilon": step,
        "max_absolute_error": float(np.max(np.abs(error))),
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "rmse": rmse,
        "autograd_rms": autograd_rms,
        "normalized_rmse": rmse / autograd_rms if autograd_rms else float("nan"),
        "cosine_similarity": cosine_similarity,
        "all_gradients_finite": bool(
            np.isfinite(autograd).all() and np.isfinite(finite_difference).all()
        ),
    }


def plot_results(
    summaries: list[dict],
    comparisons: list[tuple[int, float, np.ndarray, np.ndarray]],
    scatter_path: Path,
    error_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(6, 6))
    for depth, step, autograd, finite_difference in comparisons:
        axis.scatter(
            autograd,
            finite_difference,
            s=24,
            alpha=0.7,
            label=f"depth {depth}, eps={step:g}",
        )
    all_values = np.concatenate([
        np.concatenate((autograd, finite_difference))
        for _, _, autograd, finite_difference in comparisons
    ])
    extent = max(float(np.max(np.abs(all_values))), 1e-8)
    axis.plot([-extent, extent], [-extent, extent], "k--", linewidth=1,
              label="y = x")
    axis.set(xlabel="Autograd", ylabel="Central finite difference",
             title="Autograd vs. numerical weight gradients")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    scatter_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(scatter_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    for step in sorted({record["epsilon"] for record in summaries}):
        selected = sorted(
            (record for record in summaries if record["epsilon"] == step),
            key=lambda record: record["depth_layers"],
        )
        axis.plot(
            [record["depth_layers"] for record in selected],
            [record["rmse"] for record in selected],
            marker="o",
            label=f"eps={step:g}",
        )
    axis.set(
        xlabel="Circuit depth (layers)",
        ylabel="Gradient RMSE",
        title="Finite-difference error by depth",
    )
    axis.set_yscale("log")
    axis.set_xticks(sorted({record["depth_layers"] for record in summaries}))
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    error_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(error_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=parse_positive_ints, default=[1, 2, 4],
                        help="Ansatz layers to check.")
    parser.add_argument("--epsilon", type=parse_positive_floats,
                        default=[1e-2, 1e-3, 1e-4],
                        help="Finite-difference step sizes.")
    parser.add_argument("--seed", type=int, default=68,
                        help="Random initialization seed.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "custom_portfolios"
        / "custom_nonpaper_10_prices.parquet.gzip",
        help="Parquet input file.",
    )
    parser.add_argument("--input-type", choices=("prices", "returns"),
                        default="prices")
    parser.add_argument("--windows", type=int, default=4,
                        help="Fixed training windows; at least two are required.")
    parser.add_argument("--rows", type=int, default=240,
                        help="Trailing input rows.")
    parser.add_argument("--assets", type=int, default=4,
                        help="Number of leading assets.")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Returns per input state.")
    parser.add_argument(
        "--details-output",
        type=Path,
        default=REPO_ROOT / "comparison_logs"
        / "weight_gradient_check_details.csv",
        help="Per-weight gradient output CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=REPO_ROOT / "comparison_logs"
        / "weight_gradient_check_summary.csv",
        help="Aggregate error metrics CSV.",
    )
    parser.add_argument(
        "--scatter-output",
        type=Path,
        default=PLOTS_DIR / "weight_gradient_check_scatter.png",
        help="Autograd-versus-finite-difference plot.",
    )
    parser.add_argument(
        "--error-plot-output",
        type=Path,
        default=PLOTS_DIR / "weight_gradient_check_error.png",
        help="Error-by-depth plot.",
    )
    args = parser.parse_args()

    if args.seed < 0:
        parser.error("--seed must be non-negative.")
    if args.windows < 2 or args.rows < 2 or args.assets < 1 or args.lookback < 1:
        parser.error("--windows must be >=2; rows, assets, and lookback >=1.")
    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}")

    windows = load_fixed_windows(
        args.data, args.input_type, args.rows, args.assets,
        args.lookback, args.windows,
    )
    output_index = 0
    print("PennyLane weight-gradient diagnostic")
    print(f"Data: {args.data} ({args.input_type})")
    print(f"Fixed input batch: {windows.shape}; output checked: window 0, asset 0")
    print(f"Seed: {args.seed}; depths: {args.depths}; epsilons: {args.epsilon}")
    print("All perturbed evaluations use the same fixed input batch and preprocessing.")

    summaries = []
    comparisons = []
    detail_rows = []
    for depth in args.depths:
        num_qubits = max(math.ceil(math.log2(windows.shape[1] * 4)), args.assets)
        print(f"Checking depth {depth} ({num_qubits * depth} weights)...",
              flush=True)
        autograd, finite_differences = measure_gradients(
            windows, depth, args.seed, args.assets, output_index, args.epsilon
        )
        for step, finite_difference in finite_differences.items():
            summary = summarize_comparison(
                depth, step, autograd, finite_difference
            )
            summaries.append(summary)
            comparisons.append((depth, step, autograd, finite_difference))
            for weight_index, (auto_value, finite_value) in enumerate(
                zip(autograd, finite_difference)
            ):
                detail_rows.append({
                    "depth_layers": depth,
                    "num_weights": len(autograd),
                    "epsilon": step,
                    "weight_index": weight_index,
                    "autograd": float(auto_value),
                    "finite_difference": float(finite_value),
                    "absolute_error": float(abs(finite_value - auto_value)),
                })

    args.details_output.parent.mkdir(parents=True, exist_ok=True)
    with args.details_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(detail_rows[0]))
        writer.writeheader()
        writer.writerows(detail_rows)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    plot_results(
        summaries, comparisons, args.scatter_output, args.error_plot_output
    )

    print("\nSummary by depth and epsilon:")
    for record in summaries:
        print(
            f"  depth={record['depth_layers']:>2}, eps={record['epsilon']:.0e}: "
            f"RMSE={record['rmse']:.3e}, "
            f"max|error|={record['max_absolute_error']:.3e}, "
            f"normalized RMSE={record['normalized_rmse']:.3e}, "
            f"cosine={record['cosine_similarity']:.6f}, "
            f"finite={record['all_gradients_finite']}"
        )
    print(f"\nSaved per-weight data: {args.details_output}")
    print(f"Saved summary: {args.summary_output}")
    print(f"Saved comparison plot: {args.scatter_output}")
    print(f"Saved depth-error plot: {args.error_plot_output}")
    print("Interpretation: compare errors across step sizes; very small "
          "finite-difference steps can amplify floating-point error.")


if __name__ == "__main__":
    main()

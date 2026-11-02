"""Measure how the PennyLane actor's gradient signal changes with depth.

This is a diagnostic, not a training benchmark. It computes the Jacobian of
the actor outputs with respect to circuit weights for fixed real-data input
windows, random initializations, and a range of ansatz depths.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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


def load_input_windows(
    data_path: Path,
    input_type: str,
    rows: int,
    num_assets: int,
    lookback: int,
    num_windows: int,
) -> tuple[np.ndarray, int]:
    data = pd.read_parquet(data_path)
    if input_type == "prices":
        data = data.pct_change().dropna()
    returns = data.iloc[:, :num_assets].tail(rows)
    values = returns.to_numpy(dtype=np.float32)
    if values.shape[1] != num_assets or not np.isfinite(values).all():
        raise ValueError(f"Invalid or incomplete return data in {data_path}")

    train_end = int(0.6 * len(values))
    train_returns = values[:train_end]
    available = len(train_returns) - lookback + 1
    if available < 1:
        raise ValueError("Not enough training rows for the requested lookback.")
    if num_windows > available:
        raise ValueError(
            f"Requested {num_windows} windows, but only {available} are available."
        )

    starts = np.linspace(0, available - 1, num_windows, dtype=int)
    windows = np.stack(
        [train_returns[start:start + lookback].reshape(-1) for start in starts]
    )
    return windows, train_end


def measure_depth(
    windows: np.ndarray,
    depth: int,
    seeds: list[int],
    num_qubits: int,
    num_assets: int,
) -> np.ndarray:
    gradients = []
    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = QuantumNeuralNetwork(
            input_size=windows.shape[1],
            output_size=num_assets,
            num_qubits=num_qubits,
            num_weights=num_qubits * depth,
            encoding="amplitude",
            input_transformation=radial_to_linear,
            rotation_axes="y",
            entanglement="reverse_linear",
            seed=seed,
        )
        model.eval()

        inputs = torch.tensor(windows, dtype=torch.float32)
        outputs = model(inputs)
        weights = dict(model.named_parameters())["qlayer.weights"]

        sample_gradients = []
        for output_index in range(outputs.numel()):
            grad = torch.autograd.grad(
                outputs.reshape(-1)[output_index],
                weights,
                retain_graph=output_index < outputs.numel() - 1,
            )[0]
            sample_gradients.append(grad.detach().cpu().numpy())
        gradients.append(
            np.stack(sample_gradients).reshape(
                len(windows), outputs.shape[1], num_qubits * depth
            )
        )
        del model

    return np.stack(gradients)


def plot_results(records: list[dict], output_path: Path) -> None:
    depths = [record["depth_layers"] for record in records]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(
        depths,
        [record["gradient_rms_per_weight"] for record in records],
        marker="o",
        label="Gradient RMS per weight",
    )
    axes[0].plot(
        depths,
        [record["mean_abs_gradient"] for record in records],
        marker="s",
        label="Mean absolute gradient",
    )
    axes[0].set_title("Gradient magnitude")
    axes[0].set_ylabel("Gradient")
    axes[0].legend()

    axes[1].plot(
        depths,
        [record["mean_parameter_gradient_variance"] for record in records],
        marker="o",
        color="#9467bd",
    )
    axes[1].set_title("Gradient variance across parameters")
    axes[1].set_ylabel("Mean parameter-gradient variance")

    for axis in axes:
        axis.set_xlabel("Circuit depth (layers)")
        axis.set_xticks(depths)
        axis.grid(alpha=0.25)

    fig.suptitle("PennyLane actor gradient scaling with circuit depth")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depths", type=parse_positive_ints, default=[1, 2, 4, 8, 16],
                        help="Ansatz layers (one layer has one rotation per qubit).")
    parser.add_argument("--seeds", type=parse_positive_ints, default=[68, 1, 2, 3, 4],
                        help="Comma-separated positive initialization seeds.")
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "custom_portfolios"
        / "custom_nonpaper_10_prices.parquet.gzip",
        help="Parquet data file (default: bundled custom 10-ticker prices).",
    )
    parser.add_argument(
        "--input-type",
        choices=("prices", "returns"),
        default="prices",
        help="Convert prices to daily returns, or use returns as stored.",
    )
    parser.add_argument("--windows", type=int, default=12,
                        help="Number of fixed training windows to sample.")
    parser.add_argument("--rows", type=int, default=240,
                        help="Trailing rows of returns to inspect.")
    parser.add_argument("--assets", type=int, default=4,
                        help="Number of leading assets to inspect.")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Returns per actor state.")
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "comparison_logs" / "gradient_scaling_diagnostic.csv",
        help="CSV output path.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=PLOTS_DIR / "gradient_scaling_diagnostic.png",
        help="Plot output path.",
    )
    args = parser.parse_args()

    if args.windows < 2 or args.rows < 2 or args.assets < 1 or args.lookback < 1:
        parser.error("--windows must be >=2; rows, assets, and lookback must be >=1.")

    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}")
    windows, train_rows = load_input_windows(
        args.data,
        args.input_type,
        args.rows,
        args.assets,
        args.lookback,
        args.windows,
    )
    num_qubits = max(math.ceil(math.log2(windows.shape[1] * 4)), args.assets)
    print("Gradient-scaling diagnostic (PennyLane exact simulator; no training)")
    print(f"Data: {args.data} ({args.input_type}); first {args.assets} assets, "
          f"trailing {args.rows} rows; "
          f"first {train_rows} rows used for input windows only")
    print(f"Fixed input windows: {len(windows)}; seeds: {args.seeds}")
    print(f"Depths: {args.depths} layers ({num_qubits} weights per layer)")
    print("Input preprocessing: radial_to_linear, standardized across the "
          "fixed window batch")
    print("Metric: RMS of d(policy outputs)/d(circuit weights), normalized "
          "per weight")

    records = []
    for depth in args.depths:
        print(f"Measuring depth {depth} ({num_qubits * depth} weights)...", flush=True)
        gradients = measure_depth(
            windows, depth, args.seeds, num_qubits, args.assets
        )
        parameter_variances = np.var(gradients, axis=(0, 1, 2), ddof=1)
        records.append({
            "depth_layers": depth,
            "num_qubits": num_qubits,
            "num_weights": num_qubits * depth,
            "num_seeds": len(args.seeds),
            "num_input_windows": len(windows),
            "gradient_rms_per_weight": float(np.sqrt(np.mean(gradients ** 2))),
            "mean_abs_gradient": float(np.mean(np.abs(gradients))),
            "mean_parameter_gradient_variance": float(np.mean(parameter_variances)),
            "median_policy_jacobian_norm": float(
                np.median(np.linalg.norm(gradients, axis=(2, 3)))
            ),
        })

    reference_rms = records[0]["gradient_rms_per_weight"]
    for record in records:
        record["rms_ratio_to_shallowest"] = (
            record["gradient_rms_per_weight"] / reference_rms
            if reference_rms else float("nan")
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    plot_results(records, args.plot_output)

    print("\nResults:")
    for record in records:
        print(
            f"  depth={record['depth_layers']:>2} "
            f"RMS/weight={record['gradient_rms_per_weight']:.3e} "
            f"variance={record['mean_parameter_gradient_variance']:.3e} "
            f"ratio={record['rms_ratio_to_shallowest']:.3f}"
        )
    print(f"Saved {args.output}")
    print(f"Saved plot {args.plot_output}")
    print("Interpretation: declining gradient RMS/weight and gradient variance "
          "with depth support a weakening-signal hypothesis; this diagnostic "
          "alone does not establish a barren plateau or test learning.")


if __name__ == "__main__":
    main()

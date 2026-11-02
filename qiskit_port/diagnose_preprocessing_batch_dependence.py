"""Check whether radial_to_linear depends on the other samples in its batch."""
import argparse
import csv
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

from predictors.input_transformations import radial_to_linear


PLOTS_DIR = REPO_ROOT / "comparison_logs" / "plots"


def load_windows(
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
    if num_windows < 7 or num_windows > available:
        raise ValueError(
            f"Choose 7 to {available} candidate windows; received {num_windows}."
        )

    starts = np.linspace(0, available - 1, num_windows, dtype=int)
    return np.stack([
        train_returns[start:start + lookback].reshape(-1) for start in starts
    ])


def transform(batch: np.ndarray) -> np.ndarray:
    inputs = torch.tensor(batch, dtype=torch.float32)
    with torch.no_grad():
        return radial_to_linear(inputs).cpu().numpy()


def save_plot(
    first: np.ndarray,
    second: np.ndarray,
    plot_path: Path,
) -> None:
    feature_index = np.arange(len(first))
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    axes[0].scatter(first, second, alpha=0.7, s=22)
    lower = float(min(first.min(), second.min()))
    upper = float(max(first.max(), second.max()))
    axes[0].plot([lower, upper], [lower, upper], "k--", label="equal representation")
    axes[0].set(
        xlabel="Anchor transformed with batch A",
        ylabel="Anchor transformed with batch B",
        title="Same market window, different batch companions",
    )
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(feature_index, np.abs(first - second), linewidth=1)
    axes[1].set(
        xlabel="Transformed feature index",
        ylabel="Absolute feature difference",
        title="Change in the anchor's representation",
    )
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPO_ROOT / "data" / "custom_portfolios"
        / "custom_nonpaper_10_prices.parquet.gzip",
        help="Parquet input file.",
    )
    parser.add_argument(
        "--input-type",
        choices=("prices", "returns"),
        default="prices",
        help="Convert prices to returns, or use returns as stored.",
    )
    parser.add_argument("--rows", type=int, default=240,
                        help="Trailing rows to inspect.")
    parser.add_argument("--assets", type=int, default=4,
                        help="Number of leading assets.")
    parser.add_argument("--lookback", type=int, default=10,
                        help="Returns per input window.")
    parser.add_argument("--windows", type=int, default=8,
                        help="Candidate windows used to build the two batches.")
    parser.add_argument(
        "--feature-output",
        type=Path,
        default=REPO_ROOT / "comparison_logs"
        / "preprocessing_batch_dependence_features.csv",
        help="Per-feature comparison CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=REPO_ROOT / "comparison_logs"
        / "preprocessing_batch_dependence_summary.csv",
        help="Diagnostic summary CSV.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=PLOTS_DIR / "preprocessing_batch_dependence.png",
        help="Diagnostic plot.",
    )
    args = parser.parse_args()

    if args.rows < 2 or args.assets < 1 or args.lookback < 1:
        parser.error("--rows must be >=2; --assets and --lookback must be >=1.")
    if not args.data.is_file():
        parser.error(f"Data file not found: {args.data}")

    windows = load_windows(
        args.data,
        args.input_type,
        args.rows,
        args.assets,
        args.lookback,
        args.windows,
    )

    anchor = windows[0]
    batch_a = np.concatenate((anchor[None, :], windows[1:4]), axis=0)
    batch_b = np.concatenate((anchor[None, :], windows[-3:]), axis=0)
    transformed_a = transform(batch_a)
    transformed_a_repeat = transform(batch_a)
    transformed_b = transform(batch_b)
    transformed_single = transform(anchor[None, :])

    expected_features = anchor.size * 4
    shape_ok = (
        transformed_a.shape == (4, expected_features)
        and transformed_b.shape == (4, expected_features)
    )
    finite_batches = (
        bool(np.isfinite(transformed_a).all())
        and bool(np.isfinite(transformed_b).all())
    )
    deterministic = bool(np.array_equal(transformed_a, transformed_a_repeat))
    anchor_a = transformed_a[0]
    anchor_b = transformed_b[0]
    absolute_difference = np.abs(anchor_a - anchor_b)
    max_difference = float(absolute_difference.max())
    mean_difference = float(absolute_difference.mean())
    batch_dependent = max_difference > 1e-6
    singleton_finite = bool(np.isfinite(transformed_single).all())

    feature_path = args.feature_output
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "feature_index",
                "batch_a_value",
                "batch_b_value",
                "absolute_difference",
            ],
        )
        writer.writeheader()
        for index, (value_a, value_b, difference) in enumerate(
            zip(anchor_a, anchor_b, absolute_difference)
        ):
            writer.writerow({
                "feature_index": index,
                "batch_a_value": float(value_a),
                "batch_b_value": float(value_b),
                "absolute_difference": float(difference),
            })

    summary = {
        "data": str(args.data),
        "input_type": args.input_type,
        "assets": args.assets,
        "lookback": args.lookback,
        "batch_size": len(batch_a),
        "transformed_feature_count": expected_features,
        "shape_check_passed": shape_ok,
        "multi_sample_batches_finite": finite_batches,
        "same_batch_repeatable": deterministic,
        "anchor_max_absolute_difference": max_difference,
        "anchor_mean_absolute_difference": mean_difference,
        "anchor_batch_dependent": batch_dependent,
        "singleton_batch_finite": singleton_finite,
    }

    summary_path = args.summary_output
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)

    save_plot(anchor_a, anchor_b, args.plot_output)

    print("radial_to_linear batch-dependence diagnostic")
    print(f"Data: {args.data} ({args.input_type})")
    print(f"Batch shapes: {batch_a.shape} and {batch_b.shape}")
    print(f"Expected transformed shape: (4, {expected_features})")
    print(f"Shape check: {'PASS' if shape_ok else 'FAIL'}")
    print(f"Multi-sample batches finite: {'PASS' if finite_batches else 'FAIL'}")
    print(f"Repeating the same batch gives identical output: "
          f"{'PASS' if deterministic else 'FAIL'}")
    print(f"Same anchor, different companions: max |difference|={max_difference:.6g}, "
          f"mean |difference|={mean_difference:.6g}")
    print(f"Batch-dependent representation detected: {batch_dependent}")
    print(f"Singleton batch finite: {singleton_finite}")
    if not singleton_finite:
        print("Note: singleton standardization has zero degrees of freedom and "
              "produces non-finite values in the current transform.")
    print(f"Saved feature comparison: {feature_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved plot: {args.plot_output}")

    if not shape_ok or not finite_batches or not deterministic:
        raise SystemExit("A required preprocessing diagnostic check failed.")


if __name__ == "__main__":
    main()

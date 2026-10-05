import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_CODE = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ORIGINAL_CODE))

from predictors.quantum_neural_network import QuantumNeuralNetwork
from predictors.input_transformations import radial_to_linear
from qiskit_port.qiskit_exact_amplitude_finite_diff_qnn import (
    QiskitExactAmplitudeFiniteDiffQNN,
)


ATOL_FORWARD = 1e-5
ATOL_WEIGHT_GRAD = 1e-4
ATOL_INPUT_GRAD = 5e-4


def get_pl_weights(model):
    for name, param in model.named_parameters():
        if "weights" in name:
            return param
    raise RuntimeError("No PennyLane weights found")


def assert_close(name, a, b, atol):
    diff = (a.float() - b.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    print(f"{name} max abs diff: {max_diff}")
    print(f"{name} mean abs diff: {mean_diff}")

    if not torch.allclose(a.float(), b.float(), atol=atol, rtol=0):
        raise AssertionError(f"{name} mismatch: max diff {max_diff}")


def build_models(**kwargs):
    torch.manual_seed(68)
    np.random.seed(68)

    pl_model = QuantumNeuralNetwork(**kwargs, seed=68)

    q_model = QiskitExactAmplitudeFiniteDiffQNN(
        **kwargs,
        seed=68,
        compute_input_gradients=True,
    )

    pl_weights = get_pl_weights(pl_model)

    with torch.no_grad():
        q_model.weights.copy_(pl_weights.detach().clone().float())

    return pl_model, q_model, pl_weights


def run_forward_case(label, atol_forward=ATOL_FORWARD, **kwargs):
    print("\n" + "=" * 80)
    print("FORWARD CASE:", label)
    print("=" * 80)

    pl_model, q_model, _ = build_models(**kwargs)

    x = torch.randn(2, kwargs["input_size"])

    with torch.no_grad():
        y_pl = pl_model(x)
        y_q = q_model(x)

    print("PennyLane output:")
    print(y_pl)

    print("Qiskit output:")
    print(y_q)

    assert_close("forward", y_pl, y_q, atol_forward)


def run_gradient_case(
    label,
    atol_forward=ATOL_FORWARD,
    atol_weight=ATOL_WEIGHT_GRAD,
    atol_input=ATOL_INPUT_GRAD,
    **kwargs,
):
    print("\n" + "=" * 80)
    print("GRADIENT CASE:", label)
    print("=" * 80)

    pl_model, q_model, pl_weights = build_models(**kwargs)

    x = torch.randn(2, kwargs["input_size"])

    x_pl = x.clone().detach().requires_grad_(True)
    x_q = x.clone().detach().requires_grad_(True)

    pl_model.zero_grad(set_to_none=True)
    q_model.zero_grad(set_to_none=True)

    y_pl = pl_model(x_pl)
    y_q = q_model(x_q)

    assert_close("forward before backward", y_pl, y_q, atol_forward)

    y_pl.sum().backward()
    y_q.sum().backward()

    assert_close("weight grad", pl_weights.grad, q_model.weights.grad, atol_weight)
    assert_close("input grad", x_pl.grad, x_q.grad, atol_input)


def run_1d_input_case(label, **kwargs):
    """DDPG feeds single 1-D state vectors [input_size]; PennyLane
    preserves the 1-D shape end to end, so the port must too."""
    print("\n" + "=" * 80)
    print("1-D INPUT CASE:", label)
    print("=" * 80)

    pl_model, q_model, pl_weights = build_models(**kwargs)

    x = torch.randn(kwargs["input_size"])

    x_pl = x.clone().detach().requires_grad_(True)
    x_q = x.clone().detach().requires_grad_(True)

    y_pl = pl_model(x_pl)
    y_q = q_model(x_q)

    if tuple(y_pl.shape) != tuple(y_q.shape):
        raise AssertionError(
            f"1-D output shape mismatch: PennyLane {tuple(y_pl.shape)} "
            f"vs Qiskit {tuple(y_q.shape)}"
        )

    assert_close("forward", y_pl, y_q, ATOL_FORWARD)

    y_pl.sum().backward()
    y_q.sum().backward()

    assert_close("weight grad", pl_weights.grad, q_model.weights.grad, ATOL_WEIGHT_GRAD)
    assert_close("input grad", x_pl.grad, x_q.grad, ATOL_INPUT_GRAD)


def run_classical_layers_case(label, **kwargs):
    """Parity test for classical_layers=True.

    The original PennyLane model creates a *fresh, randomly initialized*
    nn.Linear inside every forward() call (its parameters are drawn from
    the global torch RNG, never registered, and never trained), and it
    only runs at all with float64 inputs. The Qiskit port instead owns a
    persistent trainable Linear. To compare the two, we seed the RNG,
    reproduce the exact Linear the PennyLane forward will draw, and load
    those parameters into the Qiskit model's output layer.
    """
    print("\n" + "=" * 80)
    print("CLASSICAL LAYERS CASE:", label)
    print("=" * 80)

    pl_model, q_model, pl_weights = build_models(**kwargs)

    linear_seed = 123
    torch.manual_seed(linear_seed)
    ref_linear = torch.nn.Linear(
        pl_model.num_qubits, pl_model.output_size, dtype=torch.float64
    )

    with torch.no_grad():
        q_model.output_layer.weight.copy_(ref_linear.weight.float())
        q_model.output_layer.bias.copy_(ref_linear.bias.float())

    x = torch.randn(2, kwargs["input_size"])

    # The original's in-forward float64 Linear rejects float32 inputs.
    x_pl = x.clone().detach().double().requires_grad_(True)
    x_q = x.clone().detach().requires_grad_(True)

    torch.manual_seed(linear_seed)
    y_pl = pl_model(x_pl)
    y_q = q_model(x_q)

    assert_close("forward before backward", y_pl, y_q, ATOL_FORWARD)

    y_pl.sum().backward()
    y_q.sum().backward()

    assert_close("weight grad", pl_weights.grad, q_model.weights.grad, ATOL_WEIGHT_GRAD)
    assert_close("input grad", x_pl.grad, x_q.grad, ATOL_INPUT_GRAD)


def main():
    base = dict(
        input_size=20,
        output_size=4,
        num_weights=8,
        encoding="amplitude",
        input_transformation=None,
        rotation_axes="y",
        entanglement="reverse_linear",
    )

    run_forward_case("amplitude / y / reverse_linear", **base)
    run_gradient_case("amplitude / y / reverse_linear", **base)

    radial = dict(
        input_size=5,
        output_size=4,
        num_weights=8,
        encoding="amplitude",
        input_transformation=radial_to_linear,
        rotation_axes="y",
        entanglement="reverse_linear",
    )

    run_forward_case("amplitude / radial_to_linear / y / reverse_linear", **radial)
    run_gradient_case("amplitude / radial_to_linear / y / reverse_linear", **radial)

    for axes in ["x", "z", "xy", "xyz"]:
        case = dict(base, rotation_axes=axes)
        label = f"amplitude / {axes} / reverse_linear"
        run_forward_case(label, **case)
        run_gradient_case(label, **case)

    for ent in ["linear", "full"]:
        case = dict(base, entanglement=ent)
        label = f"amplitude / y / {ent}"
        run_forward_case(label, **case)
        run_gradient_case(label, **case)

    for enc in ["angle", "stacked_angle"]:
        case = dict(base, encoding=enc)
        label = f"{enc} / y / reverse_linear"
        run_forward_case(label, **case)
        run_gradient_case(label, **case)

    # stacked_angle's "full" entanglement has different wiring semantics
    # than the ansatz's "full" (it reuses the feature index as one CNOT
    # endpoint instead of an all-pairs mesh), so exercise it explicitly.
    stacked_full = dict(base, encoding="stacked_angle", entanglement="full")
    run_forward_case("stacked_angle / y / full", **stacked_full)
    run_gradient_case("stacked_angle / y / full", **stacked_full)

    map_case = dict(base, output_map=(0.0, 1.0))
    run_forward_case("amplitude / y / output_map (0,1)", **map_case)
    run_gradient_case("amplitude / y / output_map (0,1)", **map_case)

    # The (-inf, inf) map applies 100*arctanh, which amplifies any
    # numerical difference by ~100/(1 - logit^2), roughly 130x at these
    # logit magnitudes, so tolerances scale accordingly.
    inf_map_case = dict(base, output_map=(-float("inf"), float("inf")))
    run_forward_case(
        "amplitude / y / output_map (-inf,inf)",
        atol_forward=1e-3,
        **inf_map_case,
    )
    run_gradient_case(
        "amplitude / y / output_map (-inf,inf)",
        atol_forward=1e-3,
        atol_weight=1e-2,
        atol_input=5e-2,
        **inf_map_case,
    )

    act_case = dict(base, output_activation=torch.tanh)
    run_forward_case("amplitude / y / tanh activation", **act_case)
    run_gradient_case("amplitude / y / tanh activation", **act_case)

    classical_case = dict(base, classical_layers=True)
    run_classical_layers_case("amplitude / y / classical_layers", **classical_case)

    run_1d_input_case("amplitude / y / 1-D single state", **base)

    print("\nAll currently supported parity tests passed.")


if __name__ == "__main__":
    main()
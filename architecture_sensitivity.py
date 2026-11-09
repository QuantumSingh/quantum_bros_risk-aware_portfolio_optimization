"""
Architecture sensitivity: circuit depth x entanglement topology.

TWO STAGES
Stage 1 (MEASURED): build small variational circuits in PennyLane with 1/2/4
  layers and Linear/Full entanglement, then measure how much their gradients
  vary across random initializations. Small gradients = harder to train
  ("barren plateau" behavior). This is an empirical result, not an assumption.
Stage 2 (STILL A MODEL): convert the measured gradient variance into a noise
  level and push it through a paired portfolio simulation (prices, weights and
  noise draws shared across variants), then report Sharpe, CAGR, crash return
  and interaction with 95% CIs.

WHAT IS MEASURED VS ASSUMED
- Measured: gradient variance vs depth/topology for THIS ansatz and cost.
- Assumed: that lower gradient variance translates into noisier trading
  decisions, via noise ~ 1/std(gradient). See noise_from_gradients().
- Still not done: no training, no real prices, not your repo's QNN. The repo's
  QuantumNeuralNetwork (amplitude encoding, num_weights=60) differs from the
  ansatz used here, so treat results as indicative, not as results for QDPG.

Requires: pennylane, numpy, matplotlib.
"""

import itertools
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFIG = {
    # --- Stage 1: circuit measurement ---
    "N_QUBITS": 5,             # matches the 5 assets in the market simulation
    "N_INIT": 300,             # random parameter initializations per variant
    "COST": "global",          # "global" (Z on all qubits) or "local" (Z on qubit 0).
                               # Global costs show barren plateaus more strongly;
                               # local costs are more trainable. Worth reporting both.
    "CIRCUIT_SEED": 7,

    # --- Stage 2: portfolio simulation ---
    "N_SEEDS": 300,
    "BASE_SEED": 68,           # same base seed as the original repo
    "N_DAYS": 400,
    "N_ASSETS": 5,
    "MU_DAILY": 0.0004,
    "SIGMA_DAILY": 0.012,
    "CRASH_START": 200,
    "CRASH_END": 230,          # exclusive
    "CRASH_SHOCK": -0.038,     # harsh (~-68% over 30 days); lower for realism
    "NOISE_BASE": 0.0015,      # daily return noise for the reference variant
                               # (1 layer, Linear). Still a chosen scale; only
                               # the RELATIVE noise across variants is measured.
}

LAYERS = [1, 2, 4]
TOPOS = ["Linear", "Full"]
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Stage 1: measure gradient variance in simulated circuits
# ---------------------------------------------------------------------------

def build_circuit(n_layers, topo):
    """Return a QNode. Each layer = RY and RZ rotation on every qubit, then CNOTs.
    Linear: CNOT chain (0-1, 1-2, ...). Full: CNOT between every qubit pair.
    Full uses many more two-qubit gates per layer (10 vs 4 at 5 qubits), which is
    the structural reason it can be harder to train and, on real hardware,
    noisier."""
    n = CONFIG["N_QUBITS"]
    dev = qml.device("default.qubit", wires=n)  # noiseless simulator

    if CONFIG["COST"] == "global":
        obs = qml.prod(*[qml.PauliZ(i) for i in range(n)])
    else:
        obs = qml.PauliZ(0)

    @qml.qnode(dev, interface="autograd", diff_method="backprop")
    def circuit(params):
        for layer in range(n_layers):
            for q in range(n):
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)
            if topo == "Linear":
                pairs = [(i, i + 1) for i in range(n - 1)]
            else:
                pairs = list(itertools.combinations(range(n), 2))
            for a, b in pairs:
                qml.CNOT(wires=[a, b])
        return qml.expval(obs)

    return circuit


def measure_gradient_std():
    """For each variant, draw many random parameter sets, compute the full
    gradient, and return the average (over parameters) standard deviation of
    the gradient across initializations. Dict: (layers, topo) -> std."""
    rng = np.random.default_rng(CONFIG["CIRCUIT_SEED"])
    n = CONFIG["N_QUBITS"]
    stds = {}
    for layers, topo in itertools.product(LAYERS, TOPOS):
        circuit = build_circuit(layers, topo)
        grad_fn = qml.grad(circuit)
        grads = []
        for _ in range(CONFIG["N_INIT"]):
            p = pnp.array(rng.uniform(0, 2 * np.pi, size=(layers, n, 2)),
                          requires_grad=True)
            grads.append(np.array(grad_fn(p)).ravel())
        # grads: (N_INIT, n_params). Variance across inits per parameter, averaged.
        var = np.var(np.stack([g for g in grads]), axis=0, ddof=1).mean()
        stds[(layers, topo)] = float(np.sqrt(var))
    return stds


def noise_from_gradients(grad_std):
    """ASSUMPTION: noise in the policy's decisions scales inversely with the
    gradient signal strength, relative to the reference (1 layer, Linear).
    Smaller gradients -> weaker learning signal -> noisier decisions.
    If a deeper circuit has LARGER gradient std than the reference, it gets LESS
    noise, so this mapping can favor deeper circuits (that would be a real,
    non-circular result of the measurement)."""
    ref = grad_std[(1, "Linear")]
    return {k: CONFIG["NOISE_BASE"] * ref / v for k, v in grad_std.items()}


# ---------------------------------------------------------------------------
# Stage 2: paired portfolio simulation
# ---------------------------------------------------------------------------

def generate_returns(rng):
    """i.i.d. Gaussian simple daily returns with a crash window (same shock on
    every asset). Not GBM, not real data."""
    r = rng.normal(CONFIG["MU_DAILY"], CONFIG["SIGMA_DAILY"],
                   size=(CONFIG["N_DAYS"], CONFIG["N_ASSETS"]))
    r[CONFIG["CRASH_START"]:CONFIG["CRASH_END"], :] += CONFIG["CRASH_SHOCK"]
    return r


def metrics_normal(x):
    """CAGR and arithmetic Sharpe (mean/std * sqrt(252), risk-free rate 0)."""
    cagr = np.prod(1 + x) ** (TRADING_DAYS / len(x)) - 1
    sharpe = x.mean() / (x.std(ddof=1) + 1e-12) * np.sqrt(TRADING_DAYS)
    return cagr, sharpe


def run_seed(seed, sigmas):
    """One repeat. Prices, portfolio weights and the base noise draw are shared
    by all variants; only sigma differs. Returns
    {(layers, topo): (sharpe, cagr, crash_cum_return)}."""
    rng = np.random.default_rng(seed)
    asset_ret = generate_returns(rng)
    w = rng.dirichlet(np.ones(CONFIG["N_ASSETS"]))  # fixed random portfolio
    z = rng.standard_normal(CONFIG["N_DAYS"])        # shared noise draw
    port = asset_ret @ w

    cs, ce = CONFIG["CRASH_START"], CONFIG["CRASH_END"]
    mask = np.ones(CONFIG["N_DAYS"], dtype=bool)
    mask[cs:ce] = False

    out = {}
    for key in itertools.product(LAYERS, TOPOS):
        r = port + sigmas[key] * z
        cagr, sharpe = metrics_normal(r[mask])
        crash_cum = np.prod(1 + r[cs:ce]) - 1  # not annualized (only 30 days)
        out[key] = (sharpe, cagr, crash_cum)
    return out


def mean_ci(a):
    a = np.asarray(a)
    return a.mean(), 1.96 * a.std(ddof=1) / np.sqrt(len(a))


def main():
    # ---- Stage 1 ----
    print(f"Stage 1: measuring gradient std ({CONFIG['N_QUBITS']} qubits, "
          f"{CONFIG['N_INIT']} inits, {CONFIG['COST']} cost)...")
    grad_std = measure_gradient_std()
    sigmas = noise_from_gradients(grad_std)
    print(f"\n{'Variant':<12}{'grad std':>12}{'noise sigma':>14}")
    for topo in TOPOS:
        for L in LAYERS:
            print(f"{topo:<6} L={L}  {grad_std[(L, topo)]:12.5f}{sigmas[(L, topo)]:14.5f}")

    # ---- Stage 2 ----
    seeds = [CONFIG["BASE_SEED"] + i for i in range(CONFIG["N_SEEDS"])]
    runs = [run_seed(s, sigmas) for s in seeds]

    def col(key, idx):
        return np.array([r[key][idx] for r in runs])

    names = ["Sharpe (out-of-crash)", "CAGR % (out-of-crash)", "Cum. return % (crash window)"]
    scale = [1, 100, 100]
    print(f"\nStage 2: {CONFIG['N_SEEDS']} paired seeds. Values: mean +/- 95% CI")
    for k, nm in enumerate(names):
        print(f"\n--- {nm} ---")
        for topo in TOPOS:
            for L in LAYERS:
                m, ci = mean_ci(col((L, topo), k) * scale[k])
                print(f"{topo:<6} L={L} | {m:9.3f} +/- {ci:.3f}")

    # Interaction: does the Full-vs-Linear gap change with depth (vs L=1)?
    print("\n--- Interaction (DiD on Sharpe, paired) ---")
    did = {}
    for L in [2, 4]:
        d = (col((L, "Full"), 0) - col((L, "Linear"), 0)) - \
            (col((1, "Full"), 0) - col((1, "Linear"), 0))
        did[L] = mean_ci(d)
        print(f"L={L} vs L=1 | {did[L][0]:8.4f} +/- {did[L][1]:.4f}")
    print("Here the noise scales come from measured gradients, so any interaction "
          "is inherited from the circuits (plus Sharpe's nonlinearity), not coded in.")

    # ---- Plot ----
    fig, axs = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle(f"Measured gradient variance -> noise -> portfolio metrics "
                 f"({CONFIG['COST']} cost, {CONFIG['N_QUBITS']} qubits)",
                 fontsize=13, fontweight="bold")
    colors = {"Linear": "tab:blue", "Full": "tab:red"}

    ax = axs[0, 0]  # measured gradient std
    for topo in TOPOS:
        ax.plot(LAYERS, [grad_std[(L, topo)] for L in LAYERS], marker="o",
                label=topo, color=colors[topo])
    ax.set_yscale("log")
    ax.set_title("MEASURED: gradient std vs depth", fontweight="bold")
    ax.set_ylabel("Gradient std (log scale)")

    ax = axs[0, 1]  # resulting noise
    for topo in TOPOS:
        ax.plot(LAYERS, [sigmas[(L, topo)] for L in LAYERS], marker="o",
                label=topo, color=colors[topo])
    ax.set_title("Implied return noise (daily sigma)", fontweight="bold")
    ax.set_ylabel("Noise sigma")

    panels = [(0, "Sharpe (out-of-crash)", 1), (1, "CAGR % (out-of-crash)", 100),
              (2, "Crash-window cum. return %", 100)]
    targets = [axs[0, 2], axs[1, 0], axs[1, 1]]
    for ax, (k, title, sc) in zip(targets, panels):
        for topo in TOPOS:
            ms, cis = zip(*[mean_ci(col((L, topo), k) * sc) for L in LAYERS])
            ax.errorbar(LAYERS, ms, yerr=cis, marker="o", capsize=4,
                        label=topo, color=colors[topo])
        ax.set_title(title, fontweight="bold")

    ax = axs[1, 2]
    Ls = [2, 4]
    ax.bar([f"L={L} vs L=1" for L in Ls], [did[L][0] for L in Ls],
           yerr=[did[L][1] for L in Ls], capsize=5, color="tab:green", alpha=0.85, width=0.4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title("Interaction: DiD on Sharpe", fontweight="bold")
    ax.set_ylabel("Delta Sharpe")

    for ax in axs.ravel():
        ax.grid(linestyle="--", alpha=0.5)
        if ax is not axs[1, 2]:
            ax.set_xticks(LAYERS)
            ax.set_xlabel("Circuit layers")
            ax.legend(title="Entanglement")

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    fn = "architecture_sensitivity.png"
    plt.savefig(fn, dpi=300)
    print(f"\nFigure saved as '{fn}'.")


if __name__ == "__main__":
    main()
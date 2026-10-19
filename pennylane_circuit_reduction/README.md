# Circuit Depth Reduction (PennyLane)

Next steps from the PI: **cut the quantum circuit to about half by removing
repeated layers, rerun, then add layers back one at a time** to see how
performance and speed change. Study the loss curves and confusion-matrix
metrics (accuracy, precision, recall). Stay with PennyLane; trying a different
optimizer comes later.

Everything here runs the **authors' own PennyLane code** (`original_pennylane/`)
with the authors' exact settings. The only thing that changes is the circuit
depth. It is kept separate from `pennylane_reproduction/` (does the code run as
the authors intended?) and `comparison_logs/` (Qiskit vs PennyLane).

## What the circuit does

Each quantum model is a circuit of qubits that turns market data into numbers:

1. **Encode** (`|Ψ⟩` block, fixed). The last 30 days of returns plus a 7-day
   forecast for 15 assets (555 numbers) are expanded to 2,220 features and
   packed into the *amplitudes* of 15 qubits (2^15 = 32,768 slots). Nothing is
   learned here.
2. **Transform** (trainable). Rotation gates (`RY`) turn each qubit by a learned
   angle, and CNOT gates link neighbouring qubits so information mixes across
   all 15. Each rotation angle is one trainable weight.
3. **Read out.** Each qubit's average Z value (between −1 and 1) is measured.
   For the actor, the 15 values are normalized into 15 portfolio weights.

**The repeating block** is one layer of `RY` rotations (one per qubit) followed
by a staircase of CNOTs. In `results/circuits/actor_60w.png` the three diagonal
staircases are the repeats. The number of weights sets how many blocks there are:

| Weights | Actor (15 qubits) | Critic (12 qubits) | Gates simulated (actor) | Speed per pass |
|---|---|---|---|---|
| **60** (as released) | 4 rotation layers, 3 CNOT staircases | 5 layers, 4 staircases | 103 | 1× |
| 45 | 3 layers, 2 staircases | 3.75 layers | 74 | 1.3× faster |
| **30** (half) | 2 layers, 1 staircase | 2.5 layers | 45 | ~2× faster |
| 15 | 1 layer, no CNOTs | 1.25 layers | 16 | ~3.8× faster |

The paper itself reports quantum models at **30 and 60** weights, so the
half-size circuit has a published comparison point.

**Caveat on real hardware:** in simulation, cost tracks these gate counts, so
fewer layers means faster training. On a real quantum computer the fixed
encoding step dominates: it decomposes into roughly 2^15 CNOTs, which dwarfs
the 42 CNOTs in the trainable part. Shrinking the trainable layers therefore
speeds up *simulation* far more than it would shorten a hardware circuit.

## The experiment

`run_depth_sweep.py` reproduces the authors' `MAIN.py` exactly: 3,536 days of
returns for 15 assets, 7 test periods ("folds") of expanding-window
cross-validation, lookback 30 plus a 7-day forecast, rebalancing every 30 days,
50 epochs, seed 68, and the authors' tuned hyperparameters. Only
`num_weights` changes.

**Checked against the authors' code:** run on the classical DDPG, it
reproduces `MAIN.py` to every printed digit (Sharpe 0.5525 rebalanced, 0.5523
static, plus IQR, min and max). Each fold re-seeds all random numbers, so folds
can run as separate jobs and still give identical results. Finished folds are
saved, so an interrupted run resumes where it stopped.

For each fold it records the loss curve (actor and critic loss per epoch), the
authors' own static and rebalanced Sharpe and profit, and every rebalance
decision.

## Confusion matrix: turning portfolio choices into yes/no predictions

This is a portfolio problem, not a classifier, so the confusion matrix needs a
yes/no question. At every 30-day rebalance, the model implicitly answers one
for each asset: *will this asset beat the average of all 15 assets over the
next 30 days?*

- The model says **yes** if it **overweights** the asset (more than
  1/15 = 6.67% of the portfolio).
- The **actual** answer is yes if the asset's 30-day return beat the
  15-asset average.

| | Actually beat the average | Actually lagged |
|---|---|---|
| **Model overweighted** | TP: backed a winner | FP: backed a loser |
| **Model underweighted** | FN: missed a winner | TN: avoided a loser |

- **Accuracy** = (TP + TN) / all: how often the tilt was right.
- **Precision** = TP / (TP + FP): when the model backs an asset, how often it wins.
- **Recall** = TP / (TP + FN): of all the winners, how many the model backed.
- **Chance accuracy**: what random guessing would score at the same backing
  rate, about 50%. A model only shows skill if it beats this.

## Files

| Path | What it is |
|---|---|
| `inspect_circuit.py` | Draws the circuit at each depth, counts gates, times one forward and backward pass. |
| `run_depth_sweep.py` | Trains and evaluates one model at one depth, fold by fold. |
| `summarize_sweep.py` | Builds all tables and plots from finished runs. |
| `results/circuits/` | Circuit diagrams (text and PNG) at each depth. |
| `results/circuit_specs.csv`, `results/speed_benchmark.csv` | Gate counts and timing per depth. |
| `results/runs/<run>/fold_<k>.json` | Raw per-fold results. |
| `results/sweep_summary.md` | **The results tables.** |
| `results/plots/` | Performance vs depth, speed, loss curves, confusion matrices. |

## Run it

```bash
PY=original_pennylane/qrl-dpo-public/.venv/bin/python
$PY pennylane_circuit_reduction/inspect_circuit.py                              # diagrams, gate counts, timing (~2 min)
$PY -u pennylane_circuit_reduction/run_depth_sweep.py --model QDPG --num-weights 30
$PY -u pennylane_circuit_reduction/run_depth_sweep.py --model QDPG --num-weights 30 --folds 6 7
$PY pennylane_circuit_reduction/summarize_sweep.py                              # tables and plots
```

Approximate Quantum DDPG training time for all 7 folds on a quiet Mac:
15 weights ~25 min, 30 ~50 min, 45 ~75 min, 60 ~1h40m. Quantum Q-Learning
takes about 2.5 times as long.

## Running the long jobs on the UMD HEP cluster

The 45- and 60-weight runs and all Quantum Q-Learning runs take hours, so the
cluster is the right place for them. Never put your password in a script or
commit it.

1. Log in with your own terpmail username: `ssh -XY <username>@hepcms-henrietta.umd.edu`
2. Change the initial password right away (your PI's `ldappasswd` command).
3. Get the code:
   `git clone https://github.com/QuantumSingh/quantum_bros_risk-aware_portfolio_optimization.git`,
   then `cd` into it and `git checkout qiskit-port` (or `main` after the PR is merged).
4. Make a Python environment: check `python3 --version` (the code targets
   3.12; ask your PI which module provides it if needed), then
   `python3 -m venv venv && source venv/bin/activate && pip install -r original_pennylane/qrl-dpo-public/requirements_fixed.txt`.
   The data file `original_pennylane/qrl-dpo-public/data/price_data.parquet.gzip`
   is ignored by git, so copy it up from your Mac with
   `scp original_pennylane/qrl-dpo-public/data/price_data.parquet.gzip <username>@hepcms-henrietta.umd.edu:<repo path>/original_pennylane/qrl-dpo-public/data/`.
5. Start a run that survives logging out:
   `nohup python -u pennylane_circuit_reduction/run_depth_sweep.py --model QDPG --num-weights 60 > qdpg60.log 2>&1 &`
   Folds are independent, so separate folds (`--folds 1`, `--folds 2`, ...)
   can run at the same time. If the cluster has a batch system (HEP clusters
   often use HTCondor), ask your PI how to submit jobs instead.
6. Bring results back by copying `pennylane_circuit_reduction/results/runs/`
   to your Mac (`scp -r`) or committing them, then run `summarize_sweep.py`.

"""Run the authors' UNMODIFIED MAIN.py for a chosen subset of models.

Reads original_pennylane/qrl-dpo-public/MAIN.py, replaces only the
RUN_MODELS list in memory, and executes it from the authors' directory.
MAIN.py on disk is never modified, so every config value and
hyperparameter is exactly the authors' own. Results are written by the
authors' own logger to results_logs/<timestamp>.log.

Usage (from repo root):
    PYTHONPATH=. $PY -u qiskit_port/reproduce_authors_main.py              # 4 fast models
    PYTHONPATH=. $PY -u qiskit_port/reproduce_authors_main.py "QDPG"       # one quantum model

Valid names: "Equal Weights", "Mean Variance Optimization", "DDPG",
"QDPG", "Deep Q-Learning", "Quantum Q-Learning".
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORS = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

FAST_DEFAULT = ["Equal Weights", "Mean Variance Optimization",
                "DDPG", "Deep Q-Learning"]


def main():
    models = sys.argv[1:] or FAST_DEFAULT
    src = (AUTHORS / "MAIN.py").read_text()
    new_list = "RUN_MODELS = [\n" + "".join(f"    {m!r},\n" for m in models) + "]"
    patched, n = re.subn(r"RUN_MODELS = \[.*?\]", new_list, src,
                         count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("Could not locate RUN_MODELS in MAIN.py")

    print(f"Running authors' MAIN.py (unmodified on disk) for: {models}")
    os.chdir(AUTHORS)
    sys.path.insert(0, str(AUTHORS))
    exec(compile(patched, str(AUTHORS / "MAIN.py"), "exec"),
         {"__name__": "__main__"})


if __name__ == "__main__":
    main()

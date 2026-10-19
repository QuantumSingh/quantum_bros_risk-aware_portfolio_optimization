"""Run the authors' UNMODIFIED MAIN.py for a chosen subset of models.

Reads original_pennylane/qrl-dpo-public/MAIN.py, replaces only the
RUN_MODELS list in memory, and executes it from the authors' directory.
MAIN.py on disk is never modified, so every config value and
hyperparameter is exactly the authors' own. Results are written by the
authors' own logger to original_pennylane/qrl-dpo-public/results_logs/;
copy finished logs into pennylane_reproduction/logs/ named
<date>_<as_released|paper_config>_<models>.log, then run
summarize_runs.py.

Usage (from repo root):
    $PY -u pennylane_reproduction/reproduce_authors_main.py              # 4 fast models
    $PY -u pennylane_reproduction/reproduce_authors_main.py "QDPG"       # one quantum model
    $PY -u pennylane_reproduction/reproduce_authors_main.py --paper-config "QDPG"

Valid names: "Equal Weights", "Mean Variance Optimization", "DDPG",
"QDPG", "Deep Q-Learning", "Quantum Q-Learning".

--paper-config applies the paper's fixed hyperparameters (arXiv:2601.18811,
Appendix A.1) that the released MAIN.py does not use: soft target-network
updates (tau = 0.005, the trainers' default) and early stopping with
patience 10, for every RL model. All tuned hyperparameters are kept.
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORS = REPO_ROOT / "original_pennylane" / "qrl-dpo-public"

FAST_DEFAULT = ["Equal Weights", "Mean Variance Optimization",
                "DDPG", "Deep Q-Learning"]


def apply_paper_config(src: str) -> str:
    """Switch every RL model to the paper's Appendix A.1 fixed settings."""
    n_soft = src.count("soft_update=False")
    n_es = src.count("early_stopping=False")
    if n_es != 4:
        raise RuntimeError(f"Expected 4 early_stopping=False, found {n_es}")
    src = src.replace("soft_update=False", "soft_update=True")
    src = src.replace("early_stopping=False", "early_stopping=True")
    print(f"Paper config: soft_update enabled for {n_soft} model(s) "
          f"(tau=0.005), early stopping enabled for {n_es} (patience 10)")
    return src


def main():
    args = sys.argv[1:]
    paper_config = "--paper-config" in args
    models = [a for a in args if a != "--paper-config"] or FAST_DEFAULT

    src = (AUTHORS / "MAIN.py").read_text()
    if paper_config:
        src = apply_paper_config(src)
    new_list = "RUN_MODELS = [\n" + "".join(f"    {m!r},\n" for m in models) + "]"
    patched, n = re.subn(r"RUN_MODELS = \[.*?\]", new_list, src,
                         count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("Could not locate RUN_MODELS in MAIN.py")

    label = "PAPER CONFIG" if paper_config else "as released"
    print(f"Running authors' MAIN.py (unmodified on disk, {label}) for: {models}")
    os.chdir(AUTHORS)
    sys.path.insert(0, str(AUTHORS))
    exec(compile(patched, str(AUTHORS / "MAIN.py"), "exec"),
         {"__name__": "__main__"})


if __name__ == "__main__":
    main()

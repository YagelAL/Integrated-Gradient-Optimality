"""Reproduce the short paper's tables with the README commands, in order.

Usage (from the repository root):
    python scripts/reproduce.py              # Table 1 and Table 2
    python scripts/reproduce.py --table 1    # synthetic data only
    python scripts/reproduce.py --table 2    # real data only (CSVs in Data/Raw_Data/)
    python scripts/reproduce.py --train      # also retrain the models (optional)
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
REAL = ["ml1m", "pinterest", "yahoo"]
RAW = {"ml1m": "Movielens-1m.csv", "pinterest": "pinterest_data.csv", "yahoo": "Yahoo_ratings.csv"}


def run(*args):
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def table1(train):
    run("scripts/make_synth_data.py", "--seed", "0")
    if train:
        run("scripts/train.py", "+final=mlp_synth")
        run("scripts/train.py", "+final=mlp_deep_synth")
    dirs = []
    for m in ["mlp", "mlp_deep"]:
        for obj in ["del_ke", "ins_ke"]:
            out = f"results/runs/synth/{m}/{obj}_oracle"
            dirs.append(out)
            run("scripts/synth_gt_audit.py", f"+final={m}_synth", "budget=q50",
                f"hydra.run.dir={out}", "++audit_explainers='ig,pls'",
                f"++all_explainers.pls.params.selection_objective={obj}",
                "++all_explainers.pls.params.eval_budget=50473",
                "++all_explainers.pls.params.temp_decay=0",
                "++all_explainers.pls.params.num_restarts=1")
    print("\n=== Table 1")
    run("experiments/synthetic/synth_match_table.py", *dirs)


def table2(train):
    raw = Path(os.environ.get("RAW_DATA_DIR", ROOT / "Data" / "Raw_Data"))
    missing = [f for f in RAW.values() if not (raw / f).exists()]
    if missing:
        sys.exit(f"missing {missing} in {raw}: see the Datasets section of the README")
    run("scripts/preprocess.py", "dataset=ml1m", "split=random_fraction")
    for d in ["pinterest", "yahoo"]:
        run("scripts/preprocess.py", f"dataset={d}", "split=random_fraction",
            "filters.drop_consecutive_repeats=false")
    if train:
        for d in REAL:
            run("scripts/train.py", f"+final=mlp_{d}")
    print("\n=== Table 2, IG row (ig/del_ke/rank=0 lines)")
    for d in REAL:
        run("scripts/run_explainer_eval.py", f"+final=mlp_{d}", "budget=q1000",
            "++active_explainer=ig", "fidelity.item_ranks=[0]",
            f"hydra.run.dir=results/runs/mlp_{d}/q1000")
    print("\n=== Table 2, oracle-like row")
    run("experiments/real_data/floor_oracle.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["1", "2"])
    ap.add_argument("--train", action="store_true", help="also retrain the models")
    a = ap.parse_args()
    if a.table in (None, "1"):
        table1(a.train)
    if a.table in (None, "2"):
        table2(a.train)

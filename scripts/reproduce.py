"""Reproduce the short paper's tables with the README commands, in order.

Usage (from the repository root):
    python scripts/reproduce.py              # Tables 1, 2 and 3 for mlp / mlp_deep (CPU)
    python scripts/reproduce.py --table 1    # synthetic data only (Tables 1 and 2)
    python scripts/reproduce.py --table 2    # real data only (Table 3; CSVs in Data/Raw_Data/)
    python scripts/reproduce.py --train      # also retrain the models (optional)
    python scripts/reproduce.py --ncf        # also run the synthetic ncf_gmf / ncf audits (CUDA recommended)
    python scripts/reproduce.py --real-ncf   # also run the real-data NCF run of Table 4 (CUDA recommended)
    python scripts/reproduce.py --table none --real-ncf   # only the Table 4 run
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
_prepared = False


def run(*args):
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run([PY, *args], cwd=ROOT, check=True)


def warn_if_no_cuda():
    import torch

    if not torch.cuda.is_available():
        print("WARNING: no CUDA device found; the NCF runs need a GPU and take hours on CPU.", flush=True)


def table1(train, ncf):
    models = ["mlp", "mlp_deep"] + (["ncf_gmf", "ncf"] if ncf else [])
    if ncf:
        warn_if_no_cuda()
    run("scripts/make_synth_data.py", "--seed", "0")
    if train:
        for m in models:
            run("scripts/train.py", f"+final={m}_synth")
    dirs = []
    for m in models:
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
    print("\n=== Table 2, regret vs the exhaustive optimum")
    run("experiments/synthetic/synth_regret_table.py", *dirs)


def preprocess_real():
    global _prepared
    if _prepared:
        return
    raw = Path(os.environ.get("RAW_DATA_DIR", ROOT / "Data" / "Raw_Data"))
    missing = [f for f in RAW.values() if not (raw / f).exists()]
    if missing:
        sys.exit(f"missing {missing} in {raw}: see the Datasets section of the README")
    run("scripts/preprocess.py", "dataset=ml1m", "split=random_fraction")
    for d in ["pinterest", "yahoo"]:
        run("scripts/preprocess.py", f"dataset={d}", "split=random_fraction",
            "filters.drop_consecutive_repeats=false")
    _prepared = True


def table2(train):
    preprocess_real()
    if train:
        for d in REAL:
            run("scripts/train.py", f"+final=mlp_{d}")
    print("\n=== Table 3, IG and SpinRec rows (ig / spinrec_del_ke, del_ke/rank=0 lines)")
    for d in REAL:
        run("scripts/run_explainer_eval.py", f"+final=mlp_{d}", "budget=q1000",
            "++active_explainer='ig,spinrec_del_ke'", "fidelity.item_ranks=[0]",
            f"hydra.run.dir=results/runs/mlp_{d}/q1000")
    print("\n=== Table 3, oracle approx. row")
    run("experiments/real_data/floor_oracle.py")
    print("\n=== Table 3, PLS-only (without IG)")
    run("experiments/real_data/floor_oracle.py", "--exclude-ig")


def real_ncf(train):
    warn_if_no_cuda()
    preprocess_real()
    if train:
        for d in REAL:
            run("scripts/train.py", f"+final=ncf_{d}")
    print("\n=== Table 4, IG and SpinRec rows (ig / spinrec_del_ke, del_ke/rank=0 lines)")
    for d in REAL:
        run("scripts/run_explainer_eval.py", f"+final=ncf_{d}", "budget=q1000",
            "++active_explainer='ig,spinrec_del_ke'", "fidelity.item_ranks=[0]",
            f"hydra.run.dir=results/runs/ncf_{d}/q1000")
    print("\n=== Table 4, oracle approx. row (fixed per-user values)")
    run("experiments/real_data/floor_oracle.py", "--model=ncf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", choices=["1", "2", "none"], help="restrict to one table group; none = only the opt-in add-ons")
    ap.add_argument("--train", action="store_true", help="also retrain the models")
    ap.add_argument("--ncf", action="store_true", help="also run the synthetic ncf_gmf / ncf audits (CUDA recommended)")
    ap.add_argument("--real-ncf", action="store_true", help="also run the real-data NCF run of Table 4 (CUDA recommended)")
    a = ap.parse_args()
    if a.table in (None, "1"):
        table1(a.train, a.ncf)
    if a.table in (None, "2"):
        table2(a.train)
    if a.table == "none" and a.ncf:
        table1(a.train, True)
    if a.real_ncf:
        real_ncf(a.train)

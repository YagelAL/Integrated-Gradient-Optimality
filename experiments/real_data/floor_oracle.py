"""Oracle approximation row of the real-data tables: per-user floor of DEL@k_e (rank 0; mlp by default, ncf with --model=ncf).

Per dataset and k_e, takes every run in the per-user CSV (IG, SpinRec for ncf, and the PLS
searches; for mlp those of configs/budget/pls_oracle_<dataset>.yaml), keeps the runs that
cover all test users, and averages over users the per-user minimum of
`monotonic_value` (DEL@k_e after the cummin over k_e). Duplicate rows of a run
were averaged when the CSVs were built.

Usage: python experiments/real_data/floor_oracle.py [per_user_dir] [--exclude-ig] [--model=mlp|ncf]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXCL = "--exclude-ig" in sys.argv
MODEL = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--model=")), "mlp")
args = [a for a in sys.argv[1:] if a != "--exclude-ig" and not a.startswith("--model=")]
DIR = Path(args[0] if args else "results/real_data/per_user")

rows = []
for ds in ["ml1m", "pinterest", "yahoo"]:
    d = pd.read_csv(DIR / f"{ds}_{MODEL}_del_ke_rank0.csv.gz")
    if EXCL:
        d = d[d.kind != "explainer"]
    for ke, x in d.groupby("ke"):
        runs = [g.set_index("user_id").monotonic_value for _, g in x.groupby(["source", "explainer"])]
        n = max(len(s) for s in runs)
        runs = [s for s in runs if len(s) == n]
        idx = runs[0].index
        floor = np.nanmin(np.vstack([s.reindex(idx).values for s in runs]), 0).mean()
        rows.append(dict(dataset=ds, n_test_users=n, k_e=ke, **{"pls_only" if EXCL else "oracle_approx": round(float(floor), 4)}))
print(pd.DataFrame(rows).to_string(index=False))

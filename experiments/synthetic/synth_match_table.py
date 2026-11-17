"""% match with true items (mean item-level overlap x100) per explainer x ke, for synth_gt_audit.py run dirs.

A row whose objective value equals the exhaustive optimum (|p_sel - p_pls| <= 1e-6) counts as full
agreement (distinct sets can tie), except the `pls` rows, which stay the raw `precision_model`
(their reference is the closed-form set, not p_pls). Writes the `ig` rows to results/synthetic/ig_overlap.csv.

Usage: python synth_match_table.py <run_dir> [<run_dir> ...]
"""
import glob
import sys
from pathlib import Path

import pandas as pd

TOL = 1e-6
OUT = Path(__file__).resolve().parents[2] / "results" / "synthetic" / "ig_overlap.csv"

rows = []
for d in sys.argv[1:]:
    csvs = glob.glob(f"{d}/synth_gt_audit_*.csv")
    assert len(csvs) == 1, f"{d}: expected 1 csv, found {csvs}"
    df = pd.read_csv(csvs[0])
    tie = ((df["p_sel"] - df["p_pls"]).abs() <= TOL) & (df["explainer"] != "pls")
    df["overlap"] = df["precision_model"].where(~tie, 1.0)
    table = (100 * df.groupby(["explainer", "ke"])["overlap"].mean().unstack()).round(1)
    n = df["user_id"].nunique()
    print(f"== {d}  (n_users={n})")
    print(table.to_string())
    model, metric = Path(d).parts[-2], Path(d).parts[-1].removesuffix("_oracle")
    rows += [dict(model=model, metric=metric, k=k, ig_overlap_pct=v, n=n) for k, v in table.loc["ig"].items()]
OUT.parent.mkdir(parents=True, exist_ok=True)
new = pd.DataFrame(rows)
if OUT.exists():  # keep rows of models not rerun here
    old = pd.read_csv(OUT)
    new = pd.concat([old[~old["model"].isin(new["model"])], new], ignore_index=True)
new.to_csv(OUT, index=False)
print(f"wrote {OUT}")

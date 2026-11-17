"""Value regret of IG's set vs the exhaustive optimum (per-k_e, no cummin), for synth_gt_audit.py run dirs.

DEL regret = p(IG set) - p(optimal set); INS regret = p(optimal set) - p(IG set); relative = regret / p_orig.
Also prints, on `mlp`, the objective gap between the exhaustive and the closed-form sets.
Writes results/synthetic/ig_regret.csv.

Usage: python synth_regret_table.py <run_dir> [<run_dir> ...]
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-6
OUT = Path(__file__).resolve().parents[2] / "results" / "synthetic" / "ig_regret.csv"


def boot_ci(v, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(v, size=len(v), replace=True).mean() for _ in range(n_boot)])
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


rows = []
for d in sys.argv[1:]:
    csvs = glob.glob(f"{d}/synth_gt_audit_*.csv")
    assert len(csvs) == 1, f"{d}: expected 1 csv, found {csvs}"
    df = pd.read_csv(csvs[0])
    assert not df.duplicated(["user_id", "explainer", "ke"]).any(), f"{d}: duplicate (user, explainer, ke)"
    obj = df["pls_selection_objective"].dropna().unique()
    assert len(obj) == 1 and obj[0] in ("del_ke", "ins_ke"), f"{d}: objective {obj}"
    sign = 1.0 if obj[0] == "del_ke" else -1.0
    ig = df[df.explainer == "ig"]
    assert not ig[["p_sel", "p_pls", "p_orig"]].isna().any().any(), f"{d}: missing p values"
    assert df.hist_len.max() <= 24, f"{d}: hist_len > 24, enumeration not exhaustive"
    print(f"== {d}  {obj[0]}  per-k_e, no cummin")
    for ke, g in ig.groupby("ke"):
        reg = sign * (g.p_sel - g.p_pls).to_numpy() + 0.0
        assert reg.min() >= -TOL, f"{d}: ke={ke} pls not optimal (min regret {reg.min():.3g})"
        rel = reg / g.p_orig.to_numpy()
        lo, hi = boot_ci(rel)
        rows.append(dict(
            run=d, objective=obj[0], ke=ke, n=len(g), mean_rel_regret=rel.mean(), ci_lo=lo, ci_hi=hi,
            median_rel_regret=np.median(rel), max_rel_regret=rel.max(),
            pct_regret_le_tol=100 * (reg <= TOL).mean(), min_regret=reg.min(),
        ))
        print(f"  ke={ke} n={len(g)} mean={100 * rel.mean():.4f}% CI=[{100 * lo:.4f},{100 * hi:.4f}] "
              f"median={100 * np.median(rel):.4f}% max={100 * rel.max():.4f}% "
              f"zero(<={TOL:g})={100 * (reg <= TOL).mean():.1f}% min_regret={reg.min():.3g}")
    if "p_closed" in df and df.p_closed.notna().any():
        pls = df[df.explainer == "pls"]
        gap = (pls.p_pls - pls.p_closed).abs()
        diff = pls.precision_model < 1
        print(f"  tie gap |p_pls - p_closed|: sets differ in {int(diff.sum())}/{len(pls)} rows, "
              f"max over those {gap[diff].max() if diff.any() else 0:.3g}, max over all rows {gap.max():.3g}")
OUT.parent.mkdir(parents=True, exist_ok=True)
new = pd.DataFrame(rows)
if OUT.exists():  # keep rows of runs not rerun here
    old = pd.read_csv(OUT)
    new = pd.concat([old[~old["run"].isin(new["run"])], new], ignore_index=True)
new.to_csv(OUT, index=False)
print(f"wrote {OUT}")

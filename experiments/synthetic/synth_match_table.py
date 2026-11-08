"""% match with true items (mean precision_model x100) per explainer x ke, for synth_gt_audit.py run dirs.

Usage: python synth_match_table.py <run_dir> [<run_dir> ...]
"""
import glob
import sys

import pandas as pd

for d in sys.argv[1:]:
    csvs = glob.glob(f"{d}/synth_gt_audit_*.csv")
    assert len(csvs) == 1, f"{d}: expected 1 csv, found {csvs}"
    df = pd.read_csv(csvs[0])
    table = (100 * df.groupby(["explainer", "ke"])["precision_model"].mean().unstack()).round(1)
    print(f"== {d}  (n_users={df['user_id'].nunique()})")
    print(table.to_string())

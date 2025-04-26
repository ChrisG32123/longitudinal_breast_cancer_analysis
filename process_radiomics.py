#!/usr/bin/env python3
"""
Aggregate individual radiomics CSVs into one master file,
adding days_from_T0 and percent-change columns.
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ——— Paths —————————————————————————————————————————————————————
IN_DIR  = Path("/mnt/home/gerlac37/ISPY2/data/patient_radiomics_v2")
OUT_CSV = Path("/mnt/home/gerlac37/ISPY2/data/radiomics_processed_v2.csv")

# ——— Read & concatenate all per-patient CSVs —————————————————————————
all_files = sorted(IN_DIR.glob("radiomics_*.csv"))
if not all_files:
    raise FileNotFoundError(f"No files matching {IN_DIR}/radiomics_*.csv")

valid_files = []
for file in all_files:
    try:
        pd.read_csv(file)
        valid_files.append(file)
    except pd.errors.EmptyDataError:
        continue

df = pd.concat((pd.read_csv(f) for f in valid_files), ignore_index=True)

# ——— Sort so T0 is first per patient —————————————————————————————
df = df.sort_values(["patient_id", "date_ordinal"])

# ——— Compute days_from_T0 and percent changes ——————————————————————
def add_baseline_metrics(group):
    # baseline = the first (earliest) row in this group
    baseline = group.iloc[0]
    # days since baseline
    group["days_from_T0"] = group["date_ordinal"] - baseline["date_ordinal"]
    # percent change from baseline, e.g. (current / baseline - 1) * 100
    group["vol_pch_from_T0"] = (group["volume"]          / baseline["volume"]          - 1) * 100
    group["sph_pch_from_T0"] = (group["sphericity"]      / baseline["sphericity"]      - 1) * 100
    group["ld_pch_from_T0"]  = (group["longest_diameter"]/ baseline["longest_diameter"] - 1) * 100
    return group

df = df.groupby("patient_id", group_keys=False).apply(add_baseline_metrics)

# ——— Reorder columns to match your spec —————————————————————————————
cols = [
    "patient_id","date_str","date_ordinal",
    "volume","sphericity","longest_diameter","time_label",
    "days_from_T0","vol_pch_from_T0","sph_pch_from_T0","ld_pch_from_T0"
]
df = df[cols]

# ——— Write out the master CSV —————————————————————————————————————
df.to_csv(OUT_CSV, index=False)
print(f"Wrote {len(df)} rows to {OUT_CSV}")

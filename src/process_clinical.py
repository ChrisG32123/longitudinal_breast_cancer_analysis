#!/usr/bin/env python3

##################################################
# Run from:  ~/ISPY2$ python process_clinical.py #
##################################################

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MultiLabelBinarizer

def fix_col_strs(series, old_strs, new_str):
    """
    Replaces any occurrence of each string in `old_strs` with `new_str` in the given Pandas Series.
    """
    arr = series.to_numpy(dtype=str)
    for old_str in old_strs:
        mask = (arr == old_str)
        arr[mask] = new_str
    return pd.Series(arr, index=series.index)

def main():
    parser = argparse.ArgumentParser(description="Preprocess I-SPY2 clinical data and save a cleaned CSV.")
    parser.add_argument("--input_csv", default="data/ISPY2-Imaging-Cohort-1-Clinical-Data.xlsx", help="Path to the raw clinical CSV file")
    parser.add_argument("--output_csv", default="data/clinical_processed.csv", help="Where to save the preprocessed CSV")
    args = parser.parse_args()

    # Load Data
    df_clin = pd.read_excel(args.input_csv)
    print(f"[INFO] Loaded dataframe with shape {df_clin.shape}")

    # Basic cleaning of NaNs
    # Convert literal "nan" strings into actual NaN
    df_clin.replace("nan", np.nan, inplace=True)
    df_clin = df_clin.dropna(how="any")
    print(f"[INFO] After dropping rows with missing menopoause, race, or ethnicity, shape: {df_clin.shape}")

    # Preprocess Arm
    def process_arm_col(arm_str):
        # Convert to set, removing the plus sign
        items = arm_str.split()
        s = set(items).difference({"+", ""})
        return s

    df_clin["Arm"] = df_clin["Arm"].astype(str).apply(process_arm_col)
    mlb_arm = MultiLabelBinarizer()
    arm_arr = mlb_arm.fit_transform(df_clin["Arm"])
    arm_df = pd.DataFrame(arm_arr, index=df_clin.index, columns=mlb_arm.classes_)

    # Preprocess Race
    old_strs_pcfc_isla = ["Native Hawaiian or Other Pacific Islande"]
    df_clin["Race"] = fix_col_strs(df_clin["Race"], old_strs_pcfc_isla, "Native Hawaiian or Pacific Islander")

    # Replace ; with , and split on comma to handle multi-labeled race
    df_clin["Race"] = df_clin["Race"].astype(str).str.replace(";", ",")
    df_clin["Race"] = df_clin["Race"].apply(lambda x: [item.strip() for item in x.split(",") if item.strip()])

    mlb_race = MultiLabelBinarizer()
    race_arr = mlb_race.fit_transform(df_clin["Race"])
    race_df = pd.DataFrame(race_arr, index=df_clin.index, columns=mlb_race.classes_)

    # Preprocess Menopausal Status
    old_strs_pre = [
        "Premenopausal(< 6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)",
        "Premenopausal(<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)"
    ]
    new_str_pre = "Premenopausal (<6 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)"
    df_clin["menopausal_status"] = fix_col_strs(df_clin["menopausal_status"], old_strs_pre, new_str_pre)

    old_strs_peri = ["Perimenopausal(6-12 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)"]
    new_str_peri = "Perimenopausal (6-12 months since LMP AND no prior bilateral ovariectomy AND not on estrogen replacement)"
    df_clin["menopausal_status"] = fix_col_strs(df_clin["menopausal_status"], old_strs_peri, new_str_peri)

    meno_ohe = pd.get_dummies(df_clin["menopausal_status"], prefix="MENO", drop_first=False)

    # Preprocess Ethnicity => binary 1 if "Hispanic or Latino"
    df_clin["ethnicity"] = np.where(df_clin["ethnicity"] == "Hispanic or Latino", 1, 0)

    # Merge everything back
    df_merged = pd.concat([df_clin, arm_df, race_df, meno_ohe], axis=1)

    # Drop old columns
    df_merged.drop(columns=["Arm", "Race", "menopausal_status"], inplace=True)

    # Set True/False to 1/0
    bool_cols = df_merged.select_dtypes(include=["bool"]).columns
    for c in bool_cols:
        df_merged[c] = df_merged[c].astype(int)

    # Save to CSV
    df_merged.to_csv(args.output_csv, index=False)
    print(f"[INFO] Final preprocessed CSV saved to: {args.output_csv}")
    print(f"[INFO] Final shape: {df_merged.shape}")

if __name__ == "__main__":
    main()

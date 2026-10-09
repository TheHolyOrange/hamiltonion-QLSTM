"""
Data preprocessing for the UniEload dataset (GSTU campus substation, hourly,
Aug 2023 - Sep 2025, electrical measurements + weather):
https://data.mendeley.com/datasets/d8r95wmzms/3, file
GSTU_Data_Final_With_Weather.csv (sha256 99c845d8...48239).

Pipeline (split / scaling / windowing reuse src/preprocessing.py, so models
trained here go through exactly the same harness as on ETTh1):
  1. Parse Date + Time into a timestamp and sort. The raw file lists each
     day's 09:00-23:00 hours before its 00:00-08:00 hours; after sorting the
     series is a complete, gap-free hourly grid (19,008 rows, all 1h steps).
  2. Repair the single obvious data-entry error: Current_Y_A = 4566 A on
     2025-06-25 08:00 (other phases 464 / 520 A; next-largest Y reading in
     the whole file is 890 A). It is set to NaN and linearly interpolated
     from its neighbours. No other values are touched.
  3. Target: three-phase active power,
         Load_kW = sqrt(3) * V_line * mean(I_R, I_Y, I_B) * pf / 1000.
     The dataset has no explicit load column; this is the standard balanced-
     approximation formula from the measured quantities.
  4. Features: Load_kW, the five weather variables, a vacation flag (from the
     Notes column, case-insensitive), and cyclical hour / day-of-week terms --
     11 features, the same count as the ETTh1 setup.
  5. Most-recent n_rows subsample, chronological 70/15/15 split, StandardScaler
     fit on train only, sliding windows -> next-hour Load_kW.
"""
import numpy as np
import pandas as pd

from src.preprocessing import (
    ETTSequenceDataset,
    add_time_features,
    chronological_split,
    inverse_transform_target,
    subsample_recent,
)
from sklearn.preprocessing import StandardScaler

RAW_CSV = "data/GSTU_Data_Final_With_Weather.csv"
TARGET = "Load_kW"
WEATHER = ["Temperature_C", "Humidity_%", "Precipitation_mm", "Wind_Speed_mps", "Solar_Radiation_Wm2"]
FEATURE_COLS = [TARGET] + WEATHER + ["vacation", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
PHASES = ["Current_R_A", "Current_Y_A", "Current_B_A"]


def load_clean(path=RAW_CSV):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%m/%d/%Y %I:%M %p")
    df = df.sort_values("date").reset_index(drop=True)

    bad = df["Current_Y_A"] > 2000
    df.loc[bad, "Current_Y_A"] = np.nan
    df["Current_Y_A"] = df["Current_Y_A"].interpolate(method="linear")

    df[TARGET] = np.sqrt(3) * df["System_V"] * df[PHASES].mean(axis=1) * df["pf"] / 1000.0
    df["vacation"] = df["Notes"].fillna("").str.strip().str.lower().eq("vacation").astype(float)
    return add_time_features(df)


def build_datasets(n_rows=3500, sequence_length=24, train_frac=0.7, val_frac=0.15):
    """Same return signature as src.preprocessing.build_datasets."""
    df = subsample_recent(load_clean(), n_rows)
    target_idx = FEATURE_COLS.index(TARGET)

    train_df, val_df, test_df = chronological_split(df, train_frac, val_frac)

    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_df[FEATURE_COLS].values)
    val_arr = scaler.transform(val_df[FEATURE_COLS].values)
    test_arr = scaler.transform(test_df[FEATURE_COLS].values)

    meta = {
        "feature_cols": FEATURE_COLS,
        "target_idx": target_idx,
        "scaler": scaler,
        "n_rows_used": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "date_range": (str(df["date"].iloc[0]), str(df["date"].iloc[-1])),
        "split_dates": {
            "train": (str(train_df["date"].iloc[0]), str(train_df["date"].iloc[-1])),
            "val": (str(val_df["date"].iloc[0]), str(val_df["date"].iloc[-1])),
            "test": (str(test_df["date"].iloc[0]), str(test_df["date"].iloc[-1])),
        },
    }
    return (
        ETTSequenceDataset(train_arr, target_idx, sequence_length),
        ETTSequenceDataset(val_arr, target_idx, sequence_length),
        ETTSequenceDataset(test_arr, target_idx, sequence_length),
        meta,
    )


__all__ = ["build_datasets", "inverse_transform_target", "load_clean", "FEATURE_COLS", "TARGET"]


if __name__ == "__main__":
    train_ds, val_ds, test_ds, meta = build_datasets()
    print("features:", meta["feature_cols"])
    print("rows used:", meta["n_rows_used"], meta["date_range"])
    print("splits:", meta["split_dates"])
    print("dataset lens (windows):", len(train_ds), len(val_ds), len(test_ds))

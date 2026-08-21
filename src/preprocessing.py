"""
Data preprocessing for the ETTh1 (Electricity Transformer Temperature - hourly)
dataset used to train the QLSTM component of the Dynamical Hamiltonian QLSTM project.

Pipeline:
  1. Load raw CSV (date, 6 load features, OT target).
  2. Optionally subsample a contiguous recent window (quantum-circuit simulation
     is expensive, so the 30% checkpoint trains on a manageable slice rather
     than the full 17,420-hour series; the full series is left as future work).
  3. Add cyclical time-of-day / day-of-week features so the model has explicit
     access to the periodic structure that drives higher-order temporal
     dependencies in transformer oil temperature.
  4. Chronological (non-shuffled) train/val/test split.
  5. StandardScaler fit on train split only, applied to all splits.
  6. Sliding-window Dataset producing (sequence_length, n_features) -> next OT value.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

RAW_CSV = "data/ETTh1.csv"
TARGET = "OT"
BASE_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]


def load_raw(path=RAW_CSV):
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def add_time_features(df):
    df = df.copy()
    hour = df["date"].dt.hour
    dow = df["date"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    return df


def subsample_recent(df, n_rows):
    if n_rows is None or n_rows >= len(df):
        return df.reset_index(drop=True)
    return df.iloc[-n_rows:].reset_index(drop=True)


def chronological_split(df, train_frac=0.7, val_frac=0.15):
    n = len(df)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    train = df.iloc[:n_train].reset_index(drop=True)
    val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test = df.iloc[n_train + n_val:].reset_index(drop=True)
    return train, val, test


class ETTSequenceDataset(Dataset):
    """Sliding-window dataset: X = past `sequence_length` steps of all features,
    y = OT value at the step immediately following the window."""

    def __init__(self, array, target_idx, sequence_length):
        self.X = torch.tensor(array, dtype=torch.float32)
        self.target_idx = target_idx
        self.sequence_length = sequence_length

    def __len__(self):
        return max(0, self.X.shape[0] - self.sequence_length)

    def __getitem__(self, i):
        window = self.X[i: i + self.sequence_length]
        target = self.X[i + self.sequence_length, self.target_idx]
        return window, target


def build_datasets(n_rows=3500, sequence_length=24, train_frac=0.7, val_frac=0.15):
    """Returns train/val/test ETTSequenceDataset objects, the fitted scaler,
    the feature list and the target column index (for inverse-transform)."""
    df = load_raw()
    df = add_time_features(df)
    df = subsample_recent(df, n_rows)

    feature_cols = BASE_FEATURES + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target_idx = feature_cols.index(TARGET)

    train_df, val_df, test_df = chronological_split(df, train_frac, val_frac)

    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train_df[feature_cols].values)
    val_arr = scaler.transform(val_df[feature_cols].values)
    test_arr = scaler.transform(test_df[feature_cols].values)

    train_ds = ETTSequenceDataset(train_arr, target_idx, sequence_length)
    val_ds = ETTSequenceDataset(val_arr, target_idx, sequence_length)
    test_ds = ETTSequenceDataset(test_arr, target_idx, sequence_length)

    meta = {
        "feature_cols": feature_cols,
        "target_idx": target_idx,
        "scaler": scaler,
        "n_rows_used": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }
    return train_ds, val_ds, test_ds, meta


def inverse_transform_target(values, scaler, target_idx, n_features):
    """Undo StandardScaler for a 1D array of target-only values."""
    dummy = np.zeros((len(values), n_features))
    dummy[:, target_idx] = values
    return scaler.inverse_transform(dummy)[:, target_idx]


if __name__ == "__main__":
    train_ds, val_ds, test_ds, meta = build_datasets()
    print("features:", meta["feature_cols"])
    print("rows used:", meta["n_rows_used"])
    print("train/val/test sizes:", meta["train_size"], meta["val_size"], meta["test_size"])
    print("dataset lens (windows):", len(train_ds), len(val_ds), len(test_ds))
    x, y = train_ds[0]
    print("sample window shape:", x.shape, "target:", y.item())

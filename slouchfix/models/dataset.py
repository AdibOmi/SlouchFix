"""Loads the CSVs written by `data_collection.py` and builds a
person-disjoint train/val/test split, per the pitch deck's evaluation
methodology: splitting by person (not by row) so the test set measures
generalization to unseen users, not just unseen frames of people the model
already trained on.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..features import FEATURE_NAMES


def load_raw_dataset(raw_dir: Path = config.DATA_RAW_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No labeled CSVs found in {raw_dir}. Run `python -m slouchfix.data_collection "
            "--person <id>` for several people first."
        )
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def person_disjoint_split(
    df: pd.DataFrame, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    persons = df["person_id"].unique().tolist()
    if len(persons) < 3:
        raise ValueError(
            f"Need at least 3 distinct person_ids for a person-disjoint split, found {len(persons)}. "
            "Collect sessions from more people."
        )

    rng = np.random.default_rng(seed)
    persons = list(persons)
    rng.shuffle(persons)

    n = len(persons)
    n_test = max(1, round(n * test_frac))
    n_val = max(1, round(n * val_frac))
    n_test, n_val = min(n_test, n - 2), min(n_val, n - n_test - 1)

    test_persons = set(persons[:n_test])
    val_persons = set(persons[n_test : n_test + n_val])
    train_persons = set(persons[n_test + n_val :])

    train_df = df[df["person_id"].isin(train_persons)].reset_index(drop=True)
    val_df = df[df["person_id"].isin(val_persons)].reset_index(drop=True)
    test_df = df[df["person_id"].isin(test_persons)].reset_index(drop=True)
    return train_df, val_df, test_df


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (X features, y posture label strings, y distance_cm)."""
    x = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_label = df["label"].to_numpy()
    y_distance = df["distance_cm"].to_numpy(dtype=np.float32)
    return x, y_label, y_distance


def fit_standard_scaler(x_train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std == 0] = 1.0
    return mean, std


def apply_standard_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std

"""Ablation studies (Required Feature 10): controlled comparisons that
isolate the effect of one design choice at a time, reusing the same
person-disjoint train/val/test split and the same `PostureMLP` architecture
as `scripts/train.py` -- only the *input features* change between the two
arms of a given ablation, so any accuracy/F1/MAE difference is attributable
to that one choice, not to a different model or a different split.

Ablation A -- calibration-relative vs raw-absolute features
-------------------------------------------------------------
Tests whether comparing against a personal baseline (the architectural
choice this whole project is built around -- see the orientation/mounting
corner case in the build brief) helps the *trained* model, not just the
rule-based baseline that already does this by construction. "Calibration"
here is approximated per person from that person's own good_posture rows
(mean pitch/yaw/roll/face_center_y_frac/nose_to_eye_ratio), mirroring what
`calibration.py` computes live from a short reference clip. This uses only
each person's own frames to build their own baseline -- exactly what a real
calibration step does before any classification happens for that person --
so it is not test-label leakage across people; splits stay person-disjoint
throughout.

Ablation B -- full feature set vs motion-feature-ablated
-------------------------------------------------------------
The build brief's other suggested ablation (landmark-only vs raw-frame CNN
features) needs a second training pipeline entirely -- image tensors, a CNN
backbone, its own augmentation -- which is out of scope for this
lightweight Phase 1 pass (see README/TECH_STACK for why CNN input is
deferred rather than silently dropped). Substituting a feature-ablation
answerable with the same pipeline: `motion_score` is the only feature
requiring frame history (a rolling window of landmark positions); removing
it tests whether that temporal signal earns its complexity, or whether
single-frame geometric features already carry the discriminative signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import FEATURE_NAMES
from . import dataset, evaluate, train_mlp

CALIBRATION_RELATIVE_COLS = ["pitch_deg", "yaw_deg", "roll_deg", "face_center_y_frac", "nose_to_eye_ratio"]


def compute_person_baselines(df: pd.DataFrame) -> dict[str, pd.Series]:
    """One row of mean feature values per person, computed from that
    person's own good_posture frames (falling back to all of that person's
    frames if none are labeled good_posture)."""
    baselines: dict[str, pd.Series] = {}
    for person_id, group in df.groupby("person_id"):
        good = group[group["label"] == "good_posture"]
        reference = good if len(good) > 0 else group
        baselines[person_id] = reference[CALIBRATION_RELATIVE_COLS].mean()
    return baselines


def make_calibration_relative(df: pd.DataFrame, baselines: dict[str, pd.Series]) -> pd.DataFrame:
    df = df.copy()
    for col in CALIBRATION_RELATIVE_COLS:
        baseline_per_row = df["person_id"].map(lambda pid: baselines[pid][col])
        df[col] = df[col] - baseline_per_row
    return df


def _train_and_eval_mlp(
    x_train: np.ndarray,
    y_label_train: np.ndarray,
    y_distance_train: np.ndarray,
    x_val: np.ndarray,
    y_label_val: np.ndarray,
    y_distance_val: np.ndarray,
    x_test: np.ndarray,
    y_label_test: np.ndarray,
    y_distance_test: np.ndarray,
) -> dict:
    models = train_mlp.train(x_train, y_label_train, y_distance_train, x_val, y_label_val, y_distance_val)
    pred_labels, pred_distances = train_mlp.predict(models, x_test)
    cls_report = evaluate.classification_report(y_label_test, pred_labels)
    dist_report = evaluate.distance_report(y_distance_test, pred_distances)
    return {"classification": cls_report, "distance": dist_report}


def run_calibration_ablation(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    # Arm 1: raw absolute features (the default pipeline in scripts/train.py).
    x_train_raw, y_label_train, y_distance_train = dataset.to_xy(train_df)
    x_val_raw, y_label_val, y_distance_val = dataset.to_xy(val_df)
    x_test_raw, y_label_test, y_distance_test = dataset.to_xy(test_df)

    mean, std = dataset.fit_standard_scaler(x_train_raw)
    without_calibration = _train_and_eval_mlp(
        dataset.apply_standard_scaler(x_train_raw, mean, std), y_label_train, y_distance_train,
        dataset.apply_standard_scaler(x_val_raw, mean, std), y_label_val, y_distance_val,
        dataset.apply_standard_scaler(x_test_raw, mean, std), y_label_test, y_distance_test,
    )

    # Arm 2: calibration-relative features. Each split's baselines are
    # computed only from that split's own people (splits are person-disjoint
    # already, so this can't leak across train/val/test).
    train_rel = make_calibration_relative(train_df, compute_person_baselines(train_df))
    val_rel = make_calibration_relative(val_df, compute_person_baselines(val_df))
    test_rel = make_calibration_relative(test_df, compute_person_baselines(test_df))

    x_train_rel, _, _ = dataset.to_xy(train_rel)
    x_val_rel, _, _ = dataset.to_xy(val_rel)
    x_test_rel, _, _ = dataset.to_xy(test_rel)

    mean_rel, std_rel = dataset.fit_standard_scaler(x_train_rel)
    with_calibration = _train_and_eval_mlp(
        dataset.apply_standard_scaler(x_train_rel, mean_rel, std_rel), y_label_train, y_distance_train,
        dataset.apply_standard_scaler(x_val_rel, mean_rel, std_rel), y_label_val, y_distance_val,
        dataset.apply_standard_scaler(x_test_rel, mean_rel, std_rel), y_label_test, y_distance_test,
    )

    return {"with_calibration": with_calibration, "without_calibration": without_calibration}


def run_feature_subset_ablation(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> dict:
    x_train_raw, y_label_train, y_distance_train = dataset.to_xy(train_df)
    x_val_raw, y_label_val, y_distance_val = dataset.to_xy(val_df)
    x_test_raw, y_label_test, y_distance_test = dataset.to_xy(test_df)

    mean, std = dataset.fit_standard_scaler(x_train_raw)
    full_features = _train_and_eval_mlp(
        dataset.apply_standard_scaler(x_train_raw, mean, std), y_label_train, y_distance_train,
        dataset.apply_standard_scaler(x_val_raw, mean, std), y_label_val, y_distance_val,
        dataset.apply_standard_scaler(x_test_raw, mean, std), y_label_test, y_distance_test,
    )

    motion_idx = FEATURE_NAMES.index("motion_score")
    keep = [i for i in range(len(FEATURE_NAMES)) if i != motion_idx]

    x_train_nm, x_val_nm, x_test_nm = x_train_raw[:, keep], x_val_raw[:, keep], x_test_raw[:, keep]
    mean_nm, std_nm = dataset.fit_standard_scaler(x_train_nm)
    no_motion_feature = _train_and_eval_mlp(
        dataset.apply_standard_scaler(x_train_nm, mean_nm, std_nm), y_label_train, y_distance_train,
        dataset.apply_standard_scaler(x_val_nm, mean_nm, std_nm), y_label_val, y_distance_val,
        dataset.apply_standard_scaler(x_test_nm, mean_nm, std_nm), y_label_test, y_distance_test,
    )

    return {"full_features": full_features, "no_motion_feature": no_motion_feature}

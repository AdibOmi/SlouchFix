"""Runs the two no-training baselines (Required Features 4 and 5) over a
labeled dataframe from `data/raw/`, reusing the exact `baseline.classify`
and `naive_pose_baseline.classify_naive` functions the live app calls per
frame -- so "baseline accuracy" here means the same code path, not a
re-implementation that could quietly drift from what actually runs.

The labeled CSVs store the feature vector (`FEATURE_NAMES`) but not a raw
pixel `inter_eye_px` or a live `Calibration` object, so both are
reconstructed per row/person:
  - `inter_eye_px` = `inter_eye_px_norm * config.FRAME_WIDTH` (the same
    frame width used when the norm was computed at collection time).
  - Per-person `Calibration` (for Baseline 1) is built from that person's
    own good_posture rows, exactly mirroring what a live calibration clip
    would capture -- see `models/ablations.py` for the same pattern used
    for the calibration-ablation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..baseline import PostureReading, classify as rule_based_classify
from ..calibration import Calibration
from ..features import FrameFeatures
from ..naive_pose_baseline import classify_naive
from . import evaluate
from .ablations import compute_person_baselines


def _row_to_features(row: pd.Series) -> FrameFeatures:
    inter_eye_px_norm = float(row["inter_eye_px_norm"])
    return FrameFeatures(
        pitch_deg=float(row["pitch_deg"]),
        yaw_deg=float(row["yaw_deg"]),
        roll_deg=float(row["roll_deg"]),
        inter_eye_px=inter_eye_px_norm * config.FRAME_WIDTH,
        inter_eye_px_norm=inter_eye_px_norm,
        nose_to_eye_ratio=float(row["nose_to_eye_ratio"]),
        bbox_width_frac=float(row["bbox_width_frac"]),
        bbox_height_frac=float(row["bbox_height_frac"]),
        bbox_area_frac=float(row["bbox_area_frac"]),
        face_center_y_frac=float(row["face_center_y_frac"]),
        motion_score=float(row["motion_score"]),
    )


def _build_person_calibrations(df: pd.DataFrame) -> dict[str, Calibration]:
    baselines = compute_person_baselines(df)
    calibrations: dict[str, Calibration] = {}
    for person_id, group in df.groupby("person_id"):
        good = group[group["label"] == "good_posture"]
        reference = good if len(good) > 0 else group
        mean_inter_eye_px = float(reference["inter_eye_px_norm"].mean()) * config.FRAME_WIDTH
        mean_distance_cm = float(reference["distance_cm"].mean())
        b = baselines[person_id]
        calibrations[person_id] = Calibration(
            pixel_to_cm_k=mean_distance_cm * mean_inter_eye_px,
            baseline_pitch_deg=float(b["pitch_deg"]),
            baseline_yaw_deg=float(b["yaw_deg"]),
            baseline_roll_deg=float(b["roll_deg"]),
            baseline_nose_to_eye_ratio=float(b["nose_to_eye_ratio"]),
            baseline_face_center_y_frac=float(b["face_center_y_frac"]),
            calibration_distance_cm=mean_distance_cm,
        )
    return calibrations


def evaluate_rule_based_baseline(test_df: pd.DataFrame) -> dict:
    """Baseline 1: rule-based thresholds relative to a per-person
    calibration, via the app's actual `baseline.classify`."""
    calibrations = _build_person_calibrations(test_df)

    pred_labels, pred_distances, true_labels, true_distances = [], [], [], []
    for _, row in test_df.iterrows():
        features = _row_to_features(row)
        calib = calibrations[row["person_id"]]
        reading = rule_based_classify(features, calib)
        pred_labels.append(reading.label)
        pred_distances.append(reading.distance_cm)
        true_labels.append(row["label"])
        true_distances.append(row["distance_cm"])

    cls_report = evaluate.classification_report(np.array(true_labels), np.array(pred_labels))
    dist_report = evaluate.distance_report(np.array(true_distances), np.array(pred_distances))

    sample_features = _row_to_features(test_df.iloc[0])
    sample_calib = calibrations[test_df.iloc[0]["person_id"]]
    latency = evaluate.benchmark_latency(lambda _x: rule_based_classify(sample_features, sample_calib), np.zeros(1))
    return {"classification": cls_report, "distance": dist_report, "latency": latency}


def evaluate_naive_pose_baseline(test_df: pd.DataFrame) -> dict:
    """Baseline 2: naive off-the-shelf pose thresholds, no calibration."""
    pred_labels, pred_distances, true_labels, true_distances = [], [], [], []
    for _, row in test_df.iterrows():
        features = _row_to_features(row)
        reading = classify_naive(features)
        pred_labels.append(reading.label)
        pred_distances.append(reading.distance_cm)
        true_labels.append(row["label"])
        true_distances.append(row["distance_cm"])

    cls_report = evaluate.classification_report(np.array(true_labels), np.array(pred_labels))
    dist_report = evaluate.distance_report(np.array(true_distances), np.array(pred_distances))

    sample_features = _row_to_features(test_df.iloc[0])
    latency = evaluate.benchmark_latency(lambda _x: classify_naive(sample_features), np.zeros(1))
    return {"classification": cls_report, "distance": dist_report, "latency": latency}

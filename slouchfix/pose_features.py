"""Feature engineering: turn MoveNet keypoints into the 4-d angle vector
consumed by the trained SVM -- the exact representation the ablation study
selected as its Stage-2 winner (see paper Section III-C / V-B).

Kept in one place deliberately, same principle the retired `features.py`
docstring stated: this module runs identically at training time
(`models/dataset.py`) and inference time (`app.py`), so a train/deploy
feature mismatch is a naming bug, not a possibility.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import config
from .pose import PoseResult

FEATURE_NAMES = [
    "elbow_left_deg",
    "elbow_right_deg",
    "neck_deg",
    "shoulder_tilt_deg",
]


@dataclass
class PoseAngleFeatures:
    elbow_left_deg: float
    elbow_right_deg: float
    neck_deg: float
    shoulder_tilt_deg: float
    distance_cm: float  # informational only -- never part of the feature vector fed to the model

    def to_vector(self) -> np.ndarray:
        return np.array(
            [self.elbow_left_deg, self.elbow_right_deg, self.neck_deg, self.shoulder_tilt_deg],
            dtype=np.float32,
        )


def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Interior angle at vertex b, formed by rays b->a and b->c, in degrees."""
    v1 = a - b
    v2 = c - b
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def _angle_from_vertical(vec: np.ndarray) -> float:
    """Angle between `vec` and the vertical (0, -1) image-space axis, in degrees."""
    vertical = np.array([0.0, -1.0])
    cos_theta = np.dot(vec, vertical) / (np.linalg.norm(vec) + 1e-9)
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return math.degrees(math.acos(cos_theta))


def has_required_joints(pose: PoseResult) -> bool:
    return pose.has_joints(config.REQUIRED_ANGLE_JOINTS, config.MIN_KEYPOINT_SCORE)


def extract(pose: PoseResult) -> PoseAngleFeatures:
    """Compute the 4-d angle vector plus the informational distance estimate.
    Caller must check `has_required_joints(pose)` first -- this raises if a
    required joint is missing, by design (no imputation in the live path)."""
    if not has_required_joints(pose):
        raise ValueError("Pose is missing one or more required joints for angle extraction.")

    left_shoulder = pose.point(config.LEFT_SHOULDER)
    right_shoulder = pose.point(config.RIGHT_SHOULDER)
    left_elbow = pose.point(config.LEFT_ELBOW)
    right_elbow = pose.point(config.RIGHT_ELBOW)
    left_wrist = pose.point(config.LEFT_WRIST)
    right_wrist = pose.point(config.RIGHT_WRIST)
    nose = pose.point(config.NOSE)

    elbow_left_deg = _angle_between(left_shoulder, left_elbow, left_wrist)
    elbow_right_deg = _angle_between(right_shoulder, right_elbow, right_wrist)

    mid_shoulder = (left_shoulder + right_shoulder) / 2.0
    neck_vec = nose - mid_shoulder
    neck_deg = _angle_from_vertical(neck_vec)

    shoulder_vec = right_shoulder - left_shoulder
    shoulder_tilt_deg = abs(math.degrees(math.atan2(shoulder_vec[1], shoulder_vec[0])))
    if shoulder_tilt_deg > 90.0:
        shoulder_tilt_deg = 180.0 - shoulder_tilt_deg

    shoulder_width_px = float(np.linalg.norm(shoulder_vec))
    distance_cm = _estimate_distance_cm(shoulder_width_px, pose.frame_width)

    return PoseAngleFeatures(
        elbow_left_deg=elbow_left_deg,
        elbow_right_deg=elbow_right_deg,
        neck_deg=neck_deg,
        shoulder_tilt_deg=shoulder_tilt_deg,
        distance_cm=distance_cm,
    )


def _estimate_distance_cm(shoulder_width_px: float, frame_width: int) -> float:
    """Pinhole distance estimate using an assumed average shoulder width and
    treating frame width as focal length in pixels -- deliberately
    approximate (see config.AVG_SHOULDER_WIDTH_CM), informational only, never
    fed to the classifier."""
    if shoulder_width_px <= 1e-6:
        return 0.0
    focal_length_px = float(frame_width)
    return config.AVG_SHOULDER_WIDTH_CM * focal_length_px / shoulder_width_px

"""Feature engineering: turn raw FaceLandmarker output into the feature
vector consumed by both the rule-based baseline and the trained models.

Kept in one place deliberately: this module runs identically at training
time (feeding `data_collection.py`) and at inference time (`app.py`), so a
train/deploy feature mismatch is a naming bug, not a possibility.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from . import config
from .landmarks import FaceResult

FEATURE_NAMES = [
    "pitch_deg",
    "yaw_deg",
    "roll_deg",
    "inter_eye_px_norm",
    "nose_to_eye_ratio",
    "bbox_width_frac",
    "bbox_height_frac",
    "bbox_area_frac",
    "face_center_y_frac",
    "motion_score",
]


@dataclass
class FrameFeatures:
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    inter_eye_px: float          # raw pixel distance, used for distance calibration
    inter_eye_px_norm: float     # normalized by frame width, used as an ML feature
    nose_to_eye_ratio: float
    bbox_width_frac: float
    bbox_height_frac: float
    bbox_area_frac: float
    face_center_y_frac: float
    motion_score: float

    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.pitch_deg,
                self.yaw_deg,
                self.roll_deg,
                self.inter_eye_px_norm,
                self.nose_to_eye_ratio,
                self.bbox_width_frac,
                self.bbox_height_frac,
                self.bbox_area_frac,
                self.face_center_y_frac,
                self.motion_score,
            ],
            dtype=np.float32,
        )


def _solve_head_pose(face: FaceResult) -> tuple[float, float, float]:
    """Return (pitch, yaw, roll) in degrees via solvePnP against a canonical
    3D face model. Approximates the camera as a simple pinhole with focal
    length equal to frame width and no lens distortion — fine for relative
    thresholds, not for photogrammetry-grade accuracy."""
    image_points = np.array(
        [face.point(idx) for idx in config.POSE_LANDMARK_IDS], dtype=np.float64
    )
    model_points = np.array(config.MODEL_POINTS_3D, dtype=np.float64)

    focal_length = face.frame_width
    center = (face.frame_width / 2, face.frame_height / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1))

    ok, rotation_vec, _translation_vec = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rotation_vec)
    sy = math.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], sy)
        roll = 0.0

    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class FeatureExtractor:
    """Stateful extractor: holds a short rolling history for the
    "landmark movement over time" feature, so it must be reused across
    frames of the same session rather than constructed fresh each call."""

    def __init__(self, window: int = config.MOTION_WINDOW_FRAMES) -> None:
        self._history: deque[np.ndarray] = deque(maxlen=window)

    def reset(self) -> None:
        self._history.clear()

    def extract(self, face: FaceResult) -> FrameFeatures:
        pitch, yaw, roll = _solve_head_pose(face)

        left_eye = face.point(config.LEFT_EYE_OUTER)
        right_eye = face.point(config.RIGHT_EYE_OUTER)
        inter_eye_px = float(np.linalg.norm(right_eye - left_eye))
        inter_eye_px_norm = inter_eye_px / face.frame_width

        nose = face.point(config.NOSE_TIP)
        eye_line_y = (left_eye[1] + right_eye[1]) / 2.0
        nose_to_eye_ratio = float((nose[1] - eye_line_y) / max(inter_eye_px, 1e-6))

        xs = face.points_px[:, 0]
        ys = face.points_px[:, 1]
        bbox_w = float(xs.max() - xs.min())
        bbox_h = float(ys.max() - ys.min())
        bbox_width_frac = bbox_w / face.frame_width
        bbox_height_frac = bbox_h / face.frame_height
        bbox_area_frac = bbox_width_frac * bbox_height_frac

        face_center_y_frac = float(ys.mean() / face.frame_height)

        motion_points = np.array(
            [face.point(idx) for idx in config.MOTION_LANDMARK_IDS], dtype=np.float32
        )
        motion_score = self._update_motion(motion_points, inter_eye_px)

        return FrameFeatures(
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            inter_eye_px=inter_eye_px,
            inter_eye_px_norm=inter_eye_px_norm,
            nose_to_eye_ratio=nose_to_eye_ratio,
            bbox_width_frac=bbox_width_frac,
            bbox_height_frac=bbox_height_frac,
            bbox_area_frac=bbox_area_frac,
            face_center_y_frac=face_center_y_frac,
            motion_score=motion_score,
        )

    def _update_motion(self, points: np.ndarray, inter_eye_px: float) -> float:
        score = 0.0
        if self._history:
            prev = self._history[-1]
            displacement = np.linalg.norm(points - prev, axis=1).mean()
            score = float(displacement / max(inter_eye_px, 1e-6))
        self._history.append(points)
        return score

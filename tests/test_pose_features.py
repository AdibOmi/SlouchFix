"""Unit tests for pose_features.py's angle math, using synthetic keypoint
arrays with known expected angles rather than a real MoveNet detection."""

from __future__ import annotations

import numpy as np
import pytest

from slouchfix import config
from slouchfix.pose import PoseResult
from slouchfix.pose_features import extract, has_required_joints


def _make_pose(overrides: dict[int, tuple[float, float]], frame_width: int = 640, frame_height: int = 480) -> PoseResult:
    points = np.zeros((17, 2), dtype=np.float32)
    scores = np.full(17, 0.9, dtype=np.float32)
    for idx, (x, y) in overrides.items():
        points[idx] = (x, y)
    return PoseResult(points_px=points, scores=scores, frame_width=frame_width, frame_height=frame_height)


def _upright_pose() -> PoseResult:
    """A person facing the camera, arms straight down, shoulders level."""
    return _make_pose(
        {
            config.NOSE: (320, 100),
            config.LEFT_SHOULDER: (280, 150),
            config.RIGHT_SHOULDER: (360, 150),
            config.LEFT_ELBOW: (280, 220),
            config.RIGHT_ELBOW: (360, 220),
            config.LEFT_WRIST: (280, 290),
            config.RIGHT_WRIST: (360, 290),
        }
    )


def test_has_required_joints_true_for_complete_pose() -> None:
    assert has_required_joints(_upright_pose())


def test_has_required_joints_false_when_low_confidence() -> None:
    pose = _upright_pose()
    pose.scores[config.LEFT_WRIST] = 0.0
    assert not has_required_joints(pose)


def test_straight_arm_gives_near_180_elbow_angle() -> None:
    # Shoulder, elbow, and wrist are collinear (arm straight down) -> ~180 degrees.
    features = extract(_upright_pose())
    assert features.elbow_left_deg == pytest.approx(180.0, abs=1.0)
    assert features.elbow_right_deg == pytest.approx(180.0, abs=1.0)


def test_bent_elbow_gives_smaller_angle() -> None:
    pose = _upright_pose()
    # Bend the left elbow so the wrist comes back up toward the shoulder.
    pose.points_px[config.LEFT_WRIST] = (330, 160)
    features = extract(pose)
    assert features.elbow_left_deg < 120.0


def test_upright_neck_gives_near_zero_neck_angle() -> None:
    # Nose directly above the mid-shoulder point -> neck angle near 0 degrees.
    features = extract(_upright_pose())
    assert features.neck_deg == pytest.approx(0.0, abs=2.0)


def test_forward_head_gives_larger_neck_angle() -> None:
    pose = _upright_pose()
    pose.points_px[config.NOSE] = (450, 160)  # nose pushed far forward, roughly level with shoulders
    features = extract(pose)
    assert features.neck_deg > 45.0


def test_level_shoulders_give_near_zero_tilt() -> None:
    features = extract(_upright_pose())
    assert features.shoulder_tilt_deg == pytest.approx(0.0, abs=1.0)


def test_tilted_shoulders_give_nonzero_tilt() -> None:
    pose = _upright_pose()
    pose.points_px[config.RIGHT_SHOULDER] = (360, 190)  # right shoulder dropped
    features = extract(pose)
    assert features.shoulder_tilt_deg > 10.0


def test_extract_raises_when_required_joint_missing() -> None:
    pose = _upright_pose()
    pose.scores[config.RIGHT_ELBOW] = 0.0
    with pytest.raises(ValueError):
        extract(pose)


def test_feature_vector_order_matches_feature_names() -> None:
    from slouchfix.pose_features import FEATURE_NAMES

    features = extract(_upright_pose())
    vector = features.to_vector()
    assert len(vector) == len(FEATURE_NAMES) == 4
    assert vector[0] == pytest.approx(features.elbow_left_deg)
    assert vector[1] == pytest.approx(features.elbow_right_deg)
    assert vector[2] == pytest.approx(features.neck_deg)
    assert vector[3] == pytest.approx(features.shoulder_tilt_deg)

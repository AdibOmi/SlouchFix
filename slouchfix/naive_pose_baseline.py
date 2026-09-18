"""Baseline 2: naive off-the-shelf pose-estimation baseline.

Uses the exact same MediaPipe landmarks + solvePnP head pose as the rest of
the pipeline (`features.py`), but classifies against *fixed, universal*
angle/distance thresholds instead of the personal calibration baseline in
`calibration.py` / `baseline.py`. This is the "point an off-the-shelf pose
estimator at hand-picked thresholds and hope it generalizes" baseline —
required as a second, non-trained comparison point precisely because it is
expected to lose to the calibrated rule-based baseline.

The reason it loses is the whole architectural argument of this project:
desks, monitor height, laptop lid angle, and lap-vs-table setups all shift
a person's *neutral* head pose and camera-relative distance. A fixed
"pitch must stay near 0 degrees" rule has no way to tell "this person's
laptop lid is tilted back 20 degrees" apart from "this person is leaning
forward 20 degrees" — it can only compare against an assumed-universal
neutral pose, not the individual's own baseline. Distance suffers the same
problem: without a personal calibration constant, the only way to turn
pixels into centimeters is to assume a population-average interpupillary
distance, which is wrong by construction for anyone whose actual IPD
differs from that average.
"""

from __future__ import annotations

from dataclasses import dataclass

from .baseline import PostureReading
from .features import FrameFeatures

# Population-average interpupillary distance. Real human IPD ranges from
# about 50mm to 75mm, so this single constant is the main source of this
# baseline's per-person distance error -- deliberately, to demonstrate why
# `calibration.py`'s per-user pixel_to_cm_k constant is needed.
AVG_INTERPUPILLARY_MM = 63.0

# Fixed "universal" thresholds, assuming a neutral head pose of exactly
# (pitch=0, yaw=0, roll=0) and a face vertically centered in frame. No
# session calibration is consulted at all.
NAIVE_TOO_CLOSE_CM = 40.0
NAIVE_YAW_DEG = 20.0
NAIVE_ROLL_DEG = 15.0
NAIVE_PITCH_DEG = 12.0
NAIVE_SLOUCH_FACE_CENTER_Y = 0.62


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def estimate_distance_cm_naive(features: FrameFeatures) -> float:
    """Pinhole distance estimate using an assumed average IPD and treating
    frame width as focal length in pixels -- the same approximation
    `calibration.py` uses, minus the personal calibration constant."""
    frame_width_px = features.inter_eye_px / max(features.inter_eye_px_norm, 1e-6)
    return (AVG_INTERPUPILLARY_MM / 10.0) * frame_width_px / max(features.inter_eye_px, 1e-6)


@dataclass
class _NaiveScore:
    label: str
    score: float


def classify_naive(features: FrameFeatures) -> PostureReading:
    """Baseline 2. No calibration argument by design -- see module docstring."""
    distance_cm = estimate_distance_cm_naive(features)

    scores = {
        "too_close": _clip01((NAIVE_TOO_CLOSE_CM - distance_cm) / NAIVE_TOO_CLOSE_CM)
        if distance_cm < NAIVE_TOO_CLOSE_CM
        else 0.0,
        "looking_away": _clip01((abs(features.yaw_deg) - NAIVE_YAW_DEG) / NAIVE_YAW_DEG)
        if abs(features.yaw_deg) > NAIVE_YAW_DEG
        else 0.0,
        "head_tilted": _clip01((abs(features.roll_deg) - NAIVE_ROLL_DEG) / NAIVE_ROLL_DEG)
        if abs(features.roll_deg) > NAIVE_ROLL_DEG
        else 0.0,
        "leaning_forward": _clip01((features.pitch_deg - NAIVE_PITCH_DEG) / NAIVE_PITCH_DEG)
        if features.pitch_deg > NAIVE_PITCH_DEG
        else 0.0,
        "slouched": _clip01(
            (features.face_center_y_frac - NAIVE_SLOUCH_FACE_CENTER_Y) / NAIVE_SLOUCH_FACE_CENTER_Y
        )
        if features.face_center_y_frac > NAIVE_SLOUCH_FACE_CENTER_Y
        else 0.0,
    }

    label, score = max(scores.items(), key=lambda kv: kv[1])
    if score <= 0.0:
        confidence = 1.0 - max(scores.values(), default=0.0)
        return PostureReading(label="good_posture", confidence=_clip01(confidence), distance_cm=distance_cm)

    confidence = _clip01(0.5 + 0.5 * score)
    return PostureReading(label=label, confidence=confidence, distance_cm=distance_cm)

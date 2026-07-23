"""Rule-based threshold baseline.

This is the "beat this before justifying the neural approach" baseline from
the tech stack doc, and it's also what makes the app usable on day one, with
zero training data: it needs only a calibration, no model file.

Each rule produces a score in [0, 1] (0 = not triggered, 1 = strongly
triggered). The highest-scoring triggered rule wins; if nothing triggers,
the frame is "good_posture" with confidence based on how comfortably it sits
inside every threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from .calibration import Calibration
from .features import FrameFeatures
from .settings import Settings


@dataclass
class PostureReading:
    label: str
    confidence: float
    distance_cm: float


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _score_too_close(distance_cm: float, settings: Settings) -> float:
    threshold = settings.too_close_distance_cm
    if distance_cm >= threshold:
        return 0.0
    deficit = threshold - distance_cm
    return _clip01(deficit / threshold)


def _score_looking_away(features: FrameFeatures, calib: Calibration, settings: Settings) -> float:
    threshold = settings.yaw_looking_away_deg
    deviation = abs(features.yaw_deg - calib.baseline_yaw_deg)
    if deviation <= threshold:
        return 0.0
    return _clip01((deviation - threshold) / threshold)


def _score_head_tilted(features: FrameFeatures, calib: Calibration, settings: Settings) -> float:
    threshold = settings.roll_head_tilted_deg
    deviation = abs(features.roll_deg - calib.baseline_roll_deg)
    if deviation <= threshold:
        return 0.0
    return _clip01((deviation - threshold) / threshold)


def _score_leaning_forward(features: FrameFeatures, calib: Calibration, settings: Settings) -> float:
    threshold = settings.pitch_leaning_forward_deg
    pitch_forward = features.pitch_deg - calib.baseline_pitch_deg
    if pitch_forward <= threshold:
        return 0.0
    return _clip01((pitch_forward - threshold) / threshold)


def _score_slouched(features: FrameFeatures, calib: Calibration, settings: Settings) -> float:
    threshold = settings.slouch_face_drop_ratio
    drop = features.face_center_y_frac - calib.baseline_face_center_y_frac
    if drop <= threshold:
        return 0.0
    return _clip01((drop - threshold) / threshold)


def classify(features: FrameFeatures, calib: Calibration, settings: Settings | None = None) -> PostureReading:
    settings = settings or Settings()
    distance_cm = calib.estimate_distance_cm(features)

    scores = {
        "too_close": _score_too_close(distance_cm, settings),
        "looking_away": _score_looking_away(features, calib, settings),
        "head_tilted": _score_head_tilted(features, calib, settings),
        "leaning_forward": _score_leaning_forward(features, calib, settings),
        "slouched": _score_slouched(features, calib, settings),
    }

    label, score = max(scores.items(), key=lambda kv: kv[1])
    if score <= 0.0:
        confidence = 1.0 - max(scores.values(), default=0.0)
        return PostureReading(label="good_posture", confidence=_clip01(confidence), distance_cm=distance_cm)

    confidence = _clip01(0.5 + 0.5 * score)  # triggered rules start at 0.5 confidence, scale up with severity
    return PostureReading(label=label, confidence=confidence, distance_cm=distance_cm)

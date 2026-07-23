"""One-time calibration: records the user's "good posture" baseline and the
pixel-to-cm ratio needed to turn inter-eye pixel distance into an estimated
screen distance.

Monocular distance estimation without a depth sensor relies on the pinhole
camera relationship: apparent size in pixels is inversely proportional to
distance for a fixed focal length and a fixed real-world size (here,
interpupillary distance). Calibrating at one known distance folds
`real_eye_distance_cm * focal_length_px` into a single constant `k`:

    k = calibration_distance_cm * inter_eye_px_at_calibration
    estimated_distance_cm = k / current_inter_eye_px
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass

import numpy as np

from . import config
from .capture import WebcamCapture
from .features import FeatureExtractor, FrameFeatures
from .landmarks import FaceLandmarkDetector

DEFAULT_CALIBRATION_DISTANCE_CM = 50.0  # roughly "arm's length" from the screen
CALIBRATION_SECONDS = 2.5


@dataclass
class Calibration:
    pixel_to_cm_k: float
    baseline_pitch_deg: float
    baseline_yaw_deg: float
    baseline_roll_deg: float
    baseline_nose_to_eye_ratio: float
    baseline_face_center_y_frac: float
    calibration_distance_cm: float

    def estimate_distance_cm(self, features: FrameFeatures) -> float:
        return self.pixel_to_cm_k / max(features.inter_eye_px, 1e-6)

    def save(self) -> None:
        config.APP_DIR.mkdir(parents=True, exist_ok=True)
        config.CALIBRATION_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Calibration | None":
        if not config.CALIBRATION_PATH.exists():
            return None
        data = json.loads(config.CALIBRATION_PATH.read_text())
        return cls(**data)


def run_calibration(
    distance_cm: float = DEFAULT_CALIBRATION_DISTANCE_CM,
    seconds: float = CALIBRATION_SECONDS,
    show_preview: bool = True,
) -> Calibration:
    """Interactive calibration: sit still, in good posture, at `distance_cm`
    from the screen (measure it once, e.g. with a ruler or tape) while this
    runs. Averages features over `seconds` to reduce single-frame noise."""
    import cv2

    print(
        f"Calibration: sit in good posture at about {distance_cm:.0f} cm from the "
        f"screen and look at the camera. Hold still for {seconds:.1f}s..."
    )

    samples: list[FrameFeatures] = []
    with WebcamCapture() as cam, FaceLandmarkDetector() as detector:
        extractor = FeatureExtractor()
        start = time.time()
        while time.time() - start < seconds:
            frame = cam.read()
            if frame is None:
                continue
            face = detector.process(frame)
            if face is not None:
                samples.append(extractor.extract(face))
                if show_preview:
                    cv2.imshow("SlouchFix calibration", frame)
            if show_preview and cv2.waitKey(1) & 0xFF == 27:
                break
        if show_preview:
            cv2.destroyAllWindows()

    if not samples:
        raise RuntimeError("No face detected during calibration. Check camera framing and lighting.")

    inter_eye_px = float(np.mean([s.inter_eye_px for s in samples]))
    calibration = Calibration(
        pixel_to_cm_k=distance_cm * inter_eye_px,
        baseline_pitch_deg=float(np.mean([s.pitch_deg for s in samples])),
        baseline_yaw_deg=float(np.mean([s.yaw_deg for s in samples])),
        baseline_roll_deg=float(np.mean([s.roll_deg for s in samples])),
        baseline_nose_to_eye_ratio=float(np.mean([s.nose_to_eye_ratio for s in samples])),
        baseline_face_center_y_frac=float(np.mean([s.face_center_y_frac for s in samples])),
        calibration_distance_cm=distance_cm,
    )
    calibration.save()
    print(f"Calibration saved to {config.CALIBRATION_PATH}")
    return calibration


def get_or_run_calibration() -> Calibration:
    existing = Calibration.load()
    if existing is not None:
        return existing
    return run_calibration()

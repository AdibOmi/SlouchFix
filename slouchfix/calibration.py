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
from dataclasses import asdict, dataclass, field

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


def _build_calibration(samples: list[FrameFeatures], distance_cm: float) -> Calibration:
    """Averages a batch of samples collected at a known distance into a
    saved-shape `Calibration`. Pure/no I/O so both the blocking CLI flow
    below and `CalibrationSession` (driven step-by-step by the local
    server, for the Flutter UI) can share the same math."""
    if not samples:
        raise RuntimeError("No face detected during calibration. Check camera framing and lighting.")

    inter_eye_px = float(np.mean([s.inter_eye_px for s in samples]))
    return Calibration(
        pixel_to_cm_k=distance_cm * inter_eye_px,
        baseline_pitch_deg=float(np.mean([s.pitch_deg for s in samples])),
        baseline_yaw_deg=float(np.mean([s.yaw_deg for s in samples])),
        baseline_roll_deg=float(np.mean([s.roll_deg for s in samples])),
        baseline_nose_to_eye_ratio=float(np.mean([s.nose_to_eye_ratio for s in samples])),
        baseline_face_center_y_frac=float(np.mean([s.face_center_y_frac for s in samples])),
        calibration_distance_cm=distance_cm,
    )


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

    calibration = _build_calibration(samples, distance_cm)
    calibration.save()
    print(f"Calibration saved to {config.CALIBRATION_PATH}")
    return calibration


def get_or_run_calibration() -> Calibration:
    existing = Calibration.load()
    if existing is not None:
        return existing
    return run_calibration()


class CalibrationSession:
    """Step-driven calibration for callers that can't block on a native cv2
    window (the local server, for the Flutter UI): the caller reads a frame
    at a time via `step()` and can stream it/a progress fraction to a client
    instead. Owns its own camera + detector, so it must not be used while a
    `SlouchFixApp` is also holding the camera open -- for recalibrating an
    already-running app, use `SlouchFixApp.start_recalibration()` instead,
    which reuses the app's existing camera handle."""

    def __init__(
        self,
        distance_cm: float = DEFAULT_CALIBRATION_DISTANCE_CM,
        seconds: float = CALIBRATION_SECONDS,
        camera_index: int = config.CAMERA_INDEX,
    ) -> None:
        self.distance_cm = distance_cm
        self.seconds = seconds
        self.camera_index = camera_index
        self._cam: WebcamCapture | None = None
        self._detector: FaceLandmarkDetector | None = None
        self._extractor = FeatureExtractor()
        self._samples: list[FrameFeatures] = []
        self._start: float | None = None
        self.latest_frame: np.ndarray | None = None
        self.result: Calibration | None = None
        self.error: str | None = None

    def open(self) -> "CalibrationSession":
        self._cam = WebcamCapture(camera_index=self.camera_index).open()
        self._detector = FaceLandmarkDetector()
        self._start = time.time()
        return self

    @property
    def progress(self) -> float:
        if self._start is None:
            return 0.0
        return min(1.0, (time.time() - self._start) / self.seconds)

    @property
    def done(self) -> bool:
        return self.result is not None or self.error is not None

    def step(self) -> np.ndarray | None:
        """Reads and processes one frame, returning it for preview (or None
        if the camera gave nothing this call). Once `seconds` has elapsed,
        computes and saves the result -- check `.done`/`.result`/`.error`
        after. A `RuntimeError` (no face seen for the whole window) is
        caught and stored in `.error` rather than raised, since this is
        meant to be polled from a background thread the caller doesn't
        directly supervise."""
        assert self._cam is not None and self._detector is not None and self._start is not None
        if self.done:
            return self.latest_frame

        frame = self._cam.read()
        if frame is not None:
            self.latest_frame = frame
            face = self._detector.process(frame)
            if face is not None:
                self._samples.append(self._extractor.extract(face))

        if time.time() - self._start >= self.seconds:
            try:
                calibration = _build_calibration(self._samples, self.distance_cm)
                calibration.save()
                self.result = calibration
            except RuntimeError as exc:
                self.error = str(exc)

        return self.latest_frame

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None
        if self._cam is not None:
            self._cam.close()
            self._cam = None

    def __enter__(self) -> "CalibrationSession":
        return self.open()

    def __exit__(self, *exc_info) -> None:
        self.close()


@dataclass
class RecalibrationMonitor:
    """Flags the active calibration as possibly stale when frame-to-frame
    landmark scale/angle changes far exceed ordinary micro-movement noise
    (e.g. the laptop or camera got physically bumped or repositioned mid
    session). This is the "lightweight recalibration trigger" from the
    orientation/mounting-angle corner case: a sudden jump means the
    baseline this frame is being compared against may no longer describe
    the current camera-to-face geometry.

    Phase 1 scope: log/print the flag so it's visible during development
    and demos. A real "recalibrate now?" prompt is future UI work -- see
    the build brief's corner-case section.
    """

    scale_jump_frac: float = 0.35  # relative jump in inter_eye_px_norm between consecutive frames
    angle_jump_deg: float = 20.0   # jump in pitch/yaw/roll between consecutive frames
    _prev: "FrameFeatures | None" = field(default=None, init=False, repr=False)

    def update(self, features: FrameFeatures) -> bool:
        """Returns True if this frame's jump from the previous frame looks
        like a physical setup change rather than normal head movement."""
        flagged = False
        prev = self._prev
        if prev is not None:
            scale_delta = abs(features.inter_eye_px_norm - prev.inter_eye_px_norm) / max(
                prev.inter_eye_px_norm, 1e-6
            )
            angle_delta = max(
                abs(features.pitch_deg - prev.pitch_deg),
                abs(features.yaw_deg - prev.yaw_deg),
                abs(features.roll_deg - prev.roll_deg),
            )
            if scale_delta > self.scale_jump_frac or angle_delta > self.angle_jump_deg:
                flagged = True
        self._prev = features
        return flagged

    def reset(self) -> None:
        self._prev = None

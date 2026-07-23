"""MediaPipe FaceLandmarker (Tasks API) wrapper.

Older code samples use `mp.solutions.face_mesh`, which was removed from the
`mediapipe` package in this project's pinned version (0.10.35) in favor of
the Tasks API. This wraps `FaceLandmarker` so the rest of the codebase can
work with a plain list of (x, y) pixel coordinates, same as the legacy API
produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

ASSET_PATH = Path(__file__).resolve().parent / "assets" / "face_landmarker.task"

NUM_LANDMARKS = 478  # FaceLandmarker output includes iris refinement (468 + 10)


@dataclass
class FaceResult:
    """Pixel-space landmarks for a single detected face in one frame."""

    points_px: np.ndarray  # shape (N, 2), float32, (x, y) in pixel coordinates
    frame_width: int
    frame_height: int

    def point(self, idx: int) -> np.ndarray:
        return self.points_px[idx]


class FaceLandmarkDetector:
    """Runs MediaPipe FaceLandmarker in VIDEO mode over a sequence of frames."""

    def __init__(self, model_path: Path = ASSET_PATH, max_faces: int = 1) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker model not found at {model_path}. "
                "Download it from "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                "face_landmarker/float16/latest/face_landmarker.task"
            )
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame_bgr: np.ndarray) -> FaceResult | None:
        """Run detection on one BGR frame (as returned by cv2.VideoCapture)."""
        height, width = frame_bgr.shape[:2]
        rgb = frame_bgr[:, :, ::-1]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        self._timestamp_ms += 1  # monotonically increasing, unit doesn't matter for VIDEO mode
        result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)

        if not result.face_landmarks:
            return None

        face = result.face_landmarks[0]
        points_px = np.array(
            [(lm.x * width, lm.y * height) for lm in face],
            dtype=np.float32,
        )
        return FaceResult(points_px=points_px, frame_width=width, frame_height=height)

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "FaceLandmarkDetector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

"""MoveNet (SinglePose Lightning) wrapper via tflite-runtime.

Replaces the old MediaPipe FaceLandmarker (`landmarks.py`, retired): the
winning ablation pipeline detects body pose, not face landmarks. Chosen
over TensorFlow Hub's MoveNet loader (too heavy for a background app) and
over an ONNX re-export of MoveNet (an unnecessary conversion step) -- see
`docs/superpowers/specs/2026-09-21-pose-pipeline-integration-design.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ASSET_PATH = Path(__file__).resolve().parent / "assets" / "movenet_lightning.tflite"

NUM_KEYPOINTS = 17  # COCO-17 order; see config.py for named indices
INPUT_SIZE = 192  # MoveNet Lightning's fixed square input resolution


@dataclass
class PoseResult:
    """Pixel-space keypoints for the single detected person in one frame."""

    points_px: np.ndarray  # shape (17, 2), float32, (x, y) in pixel coordinates
    scores: np.ndarray  # shape (17,), float32, per-joint confidence in [0, 1]
    frame_width: int
    frame_height: int

    def point(self, idx: int) -> np.ndarray:
        return self.points_px[idx]

    def score(self, idx: int) -> float:
        return float(self.scores[idx])

    def has_joints(self, indices: list[int], min_score: float) -> bool:
        return all(self.score(idx) >= min_score for idx in indices)


class PoseDetector:
    """Runs MoveNet Lightning over a sequence of frames via tflite-runtime."""

    def __init__(self, model_path: Path = ASSET_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"MoveNet model not found at {model_path}. Download it from "
                "https://storage.googleapis.com/tfhub-lite-models/google/lite-model/"
                "movenet/singlepose/lightning/tflite/float16/4.tflite "
                f"and save it as {model_path}."
            )
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from ai_edge_litert.interpreter import Interpreter  # tflite-runtime's successor package

        self._interpreter = Interpreter(model_path=str(model_path))
        self._interpreter.allocate_tensors()
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()

    def process(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Run detection on one BGR frame (as returned by cv2.VideoCapture)."""
        height, width = frame_bgr.shape[:2]

        rgb = frame_bgr[:, :, ::-1]
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        input_dtype = self._input_details[0]["dtype"]
        if input_dtype == np.uint8:
            input_tensor = resized.astype(np.uint8)
        else:
            input_tensor = (resized.astype(np.float32) - 127.5) / 127.5
        input_tensor = np.expand_dims(input_tensor, axis=0)

        self._interpreter.set_tensor(self._input_details[0]["index"], input_tensor)
        self._interpreter.invoke()
        keypoints = self._interpreter.get_tensor(self._output_details[0]["index"])
        # Output shape (1, 1, 17, 3): [y, x, score] normalized to [0, 1].
        keypoints = keypoints[0, 0]

        points_px = np.stack([keypoints[:, 1] * width, keypoints[:, 0] * height], axis=1).astype(np.float32)
        scores = keypoints[:, 2].astype(np.float32)

        if scores.max() < 0.1:
            return None

        return PoseResult(points_px=points_px, scores=scores, frame_width=width, frame_height=height)

    def close(self) -> None:
        pass  # tflite-runtime's Interpreter has no explicit teardown

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

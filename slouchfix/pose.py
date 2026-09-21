"""MoveNet (SinglePose Lightning) wrapper via onnxruntime.

Replaces the old MediaPipe FaceLandmarker (`landmarks.py`, retired): the
winning ablation pipeline detects body pose, not face landmarks. Runs on
onnxruntime rather than a separate tflite-runtime interpreter, so the live
app only ever talks to one inference runtime -- the same one `inference.py`
uses for the SVM classifier -- at the cost of a one-time conversion to get
MoveNet into ONNX in the first place (see
docs/superpowers/specs/2026-09-21-pose-pipeline-integration-design.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ASSET_PATH = Path(__file__).resolve().parent / "assets" / "movenet_lightning.onnx"

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
    """Runs MoveNet Lightning (ONNX) over a sequence of frames via onnxruntime."""

    def __init__(self, model_path: Path = ASSET_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                f"MoveNet model not found at {model_path}. Download it from "
                "https://huggingface.co/Xenova/movenet-singlepose-lightning/resolve/main/onnx/model.onnx "
                f"and save it as {model_path}."
            )
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def process(self, frame_bgr: np.ndarray) -> PoseResult | None:
        """Run detection on one BGR frame (as returned by cv2.VideoCapture)."""
        height, width = frame_bgr.shape[:2]

        rgb = frame_bgr[:, :, ::-1]
        resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        # This ONNX export takes raw 0-255 pixel values as int32, NHWC -- no
        # normalization, unlike a typical float32 model input.
        input_tensor = np.expand_dims(resized.astype(np.int32), axis=0)

        outputs = self._session.run(None, {self._input_name: input_tensor})
        keypoints = outputs[0][0, 0]  # shape (1, 1, 17, 3) -> (17, 3): [y, x, score], normalized to [0, 1]

        points_px = np.stack([keypoints[:, 1] * width, keypoints[:, 0] * height], axis=1).astype(np.float32)
        scores = keypoints[:, 2].astype(np.float32)

        if scores.max() < 0.1:
            return None

        return PoseResult(points_px=points_px, scores=scores, frame_width=width, frame_height=height)

    def close(self) -> None:
        pass

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

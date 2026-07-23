"""Unified posture inference: prefers a trained ONNX model if one has been
exported to `data/models/posture_model.onnx`, otherwise falls back to the
rule-based baseline. This is the only switch `app.py` needs to flip once a
model exists — nothing else in the real-time loop changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import baseline, config
from .calibration import Calibration
from .features import FEATURE_NAMES, FrameFeatures
from .settings import Settings

ONNX_MODEL_PATH = config.MODELS_DIR / "posture_model.onnx"
ONNX_META_PATH = config.MODELS_DIR / "posture_model_meta.json"


class PostureEngine:
    """label + confidence + distance_cm, from whichever backend is available."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.load()
        self._session = None
        self._labels: list[str] | None = None
        self._feature_mean: np.ndarray | None = None
        self._feature_std: np.ndarray | None = None
        self._try_load_onnx()

    def _try_load_onnx(self) -> None:
        if not (ONNX_MODEL_PATH.exists() and ONNX_META_PATH.exists()):
            return
        try:
            import onnxruntime as ort
        except ImportError:
            print("[SlouchFix] onnxruntime not installed; using rule-based baseline instead.")
            return

        meta = json.loads(ONNX_META_PATH.read_text())
        if meta.get("feature_names") != FEATURE_NAMES:
            print(
                "[SlouchFix] Trained model's feature order doesn't match the current "
                "feature extractor; using rule-based baseline instead."
            )
            return

        self._session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
        self._labels = meta["labels"]
        self._feature_mean = np.array(meta["feature_mean"], dtype=np.float32)
        self._feature_std = np.array(meta["feature_std"], dtype=np.float32)
        print(f"[SlouchFix] Loaded trained model from {ONNX_MODEL_PATH}")

    @property
    def using_trained_model(self) -> bool:
        return self._session is not None

    def classify(self, features: FrameFeatures, calib: Calibration) -> baseline.PostureReading:
        if self._session is None:
            return baseline.classify(features, calib, self.settings)

        x = features.to_vector()
        x = (x - self._feature_mean) / np.where(self._feature_std == 0, 1.0, self._feature_std)
        x = x.reshape(1, -1).astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        outputs = self._session.run(None, {input_name: x})
        logits = outputs[0][0]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        idx = int(np.argmax(probs))
        label = self._labels[idx]
        confidence = float(probs[idx])

        distance_cm = calib.estimate_distance_cm(features)
        # A second regression head for distance is optional; if the model
        # only exports classification logits, fall back to the calibrated
        # geometric estimate, which is already fairly accurate on its own.
        if len(outputs) > 1:
            distance_cm = float(np.asarray(outputs[1]).reshape(-1)[0])

        return baseline.PostureReading(label=label, confidence=confidence, distance_cm=distance_cm)

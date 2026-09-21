"""Posture inference: loads the bundled SVM (exported to
`data/models/posture_model.onnx`) and classifies the 4-d pose-angle
feature vector. No rule-based fallback -- the bundled model is required
(see docs/superpowers/specs/2026-09-21-pose-pipeline-integration-design.md,
decision 5); the app fails fast at startup if it's missing or its feature
schema doesn't match the current `pose_features.FEATURE_NAMES`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from . import config
from .pose_features import FEATURE_NAMES, PoseAngleFeatures

ONNX_MODEL_PATH = config.MODELS_DIR / "posture_model.onnx"
ONNX_META_PATH = config.MODELS_DIR / "posture_model_meta.json"


@dataclass
class PostureReading:
    label: str
    confidence: float
    distance_cm: float


class PostureEngine:
    """label + confidence + distance_cm, from the bundled SVM."""

    def __init__(self) -> None:
        if not (ONNX_MODEL_PATH.exists() and ONNX_META_PATH.exists()):
            raise FileNotFoundError(
                f"No trained posture model found at {ONNX_MODEL_PATH}. Run "
                "`python -m slouchfix.models.train_svm` followed by "
                "`python -m slouchfix.models.export_onnx` first."
            )

        import onnxruntime as ort

        meta = json.loads(ONNX_META_PATH.read_text())
        if meta.get("feature_names") != FEATURE_NAMES:
            raise ValueError(
                f"Trained model's feature order {meta.get('feature_names')} doesn't match "
                f"the current feature extractor {FEATURE_NAMES}. Retrain and re-export the model."
            )

        self._session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=["CPUExecutionProvider"])
        self._labels: list[str] = meta["labels"]

    def classify(self, features: PoseAngleFeatures) -> PostureReading:
        x = features.to_vector().reshape(1, -1)

        input_name = self._session.get_inputs()[0].name
        # export_onnx.py exports with zipmap=False, so output order is
        # [predicted_label, probabilities] and probabilities is a plain
        # (1, num_classes) float array, not a list of dicts.
        _predicted_label, probabilities = self._session.run(None, {input_name: x})
        probs = np.asarray(probabilities[0], dtype=np.float32)

        idx = int(np.argmax(probs))
        label = self._labels[idx]
        confidence = float(probs[idx])

        return PostureReading(label=label, confidence=confidence, distance_cm=features.distance_cm)

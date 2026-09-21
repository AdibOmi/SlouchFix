"""Fixture test: train the SVM pipeline on a tiny synthetic sample, export
it via export_onnx.py, and confirm inference.py can load and classify
against the result -- catches schema drift between training and inference
without needing a real webcam or the full 2,000-image dataset."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from slouchfix import config
from slouchfix.models import export_onnx
from slouchfix.pose_features import PoseAngleFeatures


@pytest.fixture()
def tiny_trained_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)

    rng = np.random.default_rng(42)
    # "good": near-straight elbows, small neck/tilt angles.
    good = rng.normal(loc=[170, 170, 5, 2], scale=[5, 5, 2, 1], size=(30, 4))
    # "slouched": bent elbows, larger neck angle.
    slouched = rng.normal(loc=[120, 120, 40, 5], scale=[5, 5, 5, 2], size=(30, 4))
    X = np.concatenate([good, slouched]).astype(np.float32)
    y = np.array(["good"] * 30 + ["slouched"] * 30)

    pipeline = Pipeline([("scaler", StandardScaler()), ("svm", SVC(kernel="rbf", probability=True, random_state=42))])
    pipeline.fit(X, y)

    labels = sorted(pipeline.named_steps["svm"].classes_.tolist())
    export_onnx.export(pipeline, labels)
    return X, y


def test_exported_model_round_trips_through_inference(tiny_trained_pipeline, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr("slouchfix.inference.ONNX_MODEL_PATH", tmp_path / "posture_model.onnx")
    monkeypatch.setattr("slouchfix.inference.ONNX_META_PATH", tmp_path / "posture_model_meta.json")

    from slouchfix.inference import PostureEngine

    X, y = tiny_trained_pipeline
    engine = PostureEngine()

    good_features = PoseAngleFeatures(
        elbow_left_deg=170.0, elbow_right_deg=170.0, neck_deg=5.0, shoulder_tilt_deg=2.0, distance_cm=60.0
    )
    slouched_features = PoseAngleFeatures(
        elbow_left_deg=120.0, elbow_right_deg=120.0, neck_deg=40.0, shoulder_tilt_deg=5.0, distance_cm=60.0
    )

    good_reading = engine.classify(good_features)
    slouched_reading = engine.classify(slouched_features)

    assert good_reading.label == "good"
    assert slouched_reading.label == "slouched"
    assert good_reading.distance_cm == 60.0


def test_engine_raises_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("slouchfix.inference.ONNX_MODEL_PATH", tmp_path / "nonexistent.onnx")
    monkeypatch.setattr("slouchfix.inference.ONNX_META_PATH", tmp_path / "nonexistent.json")

    from slouchfix.inference import PostureEngine

    with pytest.raises(FileNotFoundError):
        PostureEngine()

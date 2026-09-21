"""Exports the trained sklearn SVM pipeline to ONNX for `onnxruntime`-based
inference in the desktop app (see `slouchfix/inference.py`), plus a
metadata JSON recording the feature and label order. The StandardScaler is
embedded in the exported ONNX graph by skl2onnx, so (unlike the retired
MLP-based exporter) no separate feature_mean/feature_std needs to travel
in the metadata -- the graph does its own scaling.
"""

from __future__ import annotations

import argparse
import json

import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.pipeline import Pipeline

from .. import config
from ..pose_features import FEATURE_NAMES
from .train_svm import MODEL_PICKLE_PATH


def export(pipeline: Pipeline, labels: list[str]) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = config.MODELS_DIR / "posture_model.onnx"
    meta_path = config.MODELS_DIR / "posture_model_meta.json"

    initial_type = [("features", FloatTensorType([None, len(FEATURE_NAMES)]))]
    onnx_model = convert_sklearn(
        pipeline,
        initial_types=initial_type,
        options={id(pipeline.named_steps["svm"]): {"zipmap": False}},
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())

    meta = {
        "feature_names": FEATURE_NAMES,
        "labels": list(labels),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Exported metadata to {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=str, default=str(MODEL_PICKLE_PATH), help="path to the trained pipeline (.joblib) to export"
    )
    args = parser.parse_args()

    pipeline: Pipeline = joblib.load(args.model)
    labels = sorted(pipeline.named_steps["svm"].classes_.tolist())
    export(pipeline, labels)


if __name__ == "__main__":
    main()

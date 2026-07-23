"""Exports the trained PyTorch MLP to ONNX for `onnxruntime`-based inference
in the desktop app (see `slouchfix/inference.py`), plus a metadata JSON
recording the feature order, label order, and the standardization
mean/std used at train time -- the app must apply the exact same scaling.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from .. import config
from ..features import FEATURE_NAMES
from .train_mlp import PostureMLP


def export(
    model: PostureMLP,
    label_classes: list[str],
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> None:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = config.MODELS_DIR / "posture_model.onnx"
    meta_path = config.MODELS_DIR / "posture_model_meta.json"

    model.eval()
    dummy_input = torch.zeros((1, len(FEATURE_NAMES)), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["features"],
        output_names=["posture_logits", "distance_cm"],
        dynamic_axes={"features": {0: "batch"}, "posture_logits": {0: "batch"}, "distance_cm": {0: "batch"}},
        opset_version=17,
        dynamo=False,  # legacy TorchScript-based exporter: simpler and dependency-light for this small MLP
    )

    meta = {
        "feature_names": FEATURE_NAMES,
        "labels": list(label_classes),
        "feature_mean": feature_mean.tolist(),
        "feature_std": feature_std.tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Exported metadata to {meta_path}")

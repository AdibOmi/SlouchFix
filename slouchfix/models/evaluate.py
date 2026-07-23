"""Metrics from the pitch deck's Evaluation slide: accuracy, macro-F1,
confusion matrix, MAE (cm) for distance, and an on-device latency/FPS
benchmark (measured against the actual `predict_fn`, not just "in theory").
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from .. import config


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=config.LABELS, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=config.LABELS).tolist(),
        "labels_order": config.LABELS,
    }


def distance_report(y_true_cm: np.ndarray, y_pred_cm: np.ndarray) -> dict:
    return {"mae_cm": float(mean_absolute_error(y_true_cm, y_pred_cm))}


def benchmark_latency(predict_fn: Callable[[np.ndarray], object], x_sample: np.ndarray, n: int = 200) -> dict:
    """`predict_fn` should take a single feature row (shape (n_features,) or
    (1, n_features)) and run one forward pass, mirroring the per-frame call
    in the real-time app."""
    for _ in range(10):
        predict_fn(x_sample)

    start = time.perf_counter()
    for _ in range(n):
        predict_fn(x_sample)
    elapsed = time.perf_counter() - start

    latency_ms = (elapsed / n) * 1000
    fps = 1000.0 / latency_ms if latency_ms > 0 else float("inf")
    return {"latency_ms": latency_ms, "fps": fps}


def print_report(name: str, cls_report: dict, dist_report: dict, latency: dict | None = None) -> None:
    print(f"\n=== {name} ===")
    print(f"  accuracy   : {cls_report['accuracy']:.4f}")
    print(f"  macro F1   : {cls_report['macro_f1']:.4f}")
    print(f"  distance MAE: {dist_report['mae_cm']:.2f} cm")
    if latency is not None:
        print(f"  latency    : {latency['latency_ms']:.2f} ms/frame  ({latency['fps']:.1f} FPS)")

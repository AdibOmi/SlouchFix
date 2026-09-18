"""Shared "train every model, evaluate on the held-out test split" routine,
used by both `scripts/train.py` (prints results, exports the MLP to ONNX for
the live app) and `scripts/evaluate_report.py` (also builds baselines,
ablations, and `reports/evaluation_report.md` around the same numbers) so
the two never drift into computing metrics two different ways.
"""

from __future__ import annotations

import pandas as pd

from . import dataset, evaluate, train_baseline, train_mlp, train_rf, train_xgb

MODEL_MODULES = {
    "Logistic Regression (baseline)": train_baseline,
    "Random Forest": train_rf,
    "XGBoost": train_xgb,
    "MLP (PyTorch)": train_mlp,
}


def train_and_evaluate_all(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Returns {model_name: {"models", "classification", "distance", "latency"}}
    plus "_scaler" (mean, std) and "_mlp" (the trained MLP's raw model dict,
    for ONNX export) under those two reserved keys."""
    x_train_raw, y_label_train, y_distance_train = dataset.to_xy(train_df)
    x_val_raw, y_label_val, y_distance_val = dataset.to_xy(val_df)
    x_test_raw, y_label_test, y_distance_test = dataset.to_xy(test_df)

    mean, std = dataset.fit_standard_scaler(x_train_raw)
    x_train = dataset.apply_standard_scaler(x_train_raw, mean, std)
    x_val = dataset.apply_standard_scaler(x_val_raw, mean, std)
    x_test = dataset.apply_standard_scaler(x_test_raw, mean, std)

    results: dict = {"_scaler": (mean, std)}

    for name, module in MODEL_MODULES.items():
        if module is train_mlp:
            models = module.train(x_train, y_label_train, y_distance_train, x_val, y_label_val, y_distance_val)
            results["_mlp"] = models
        else:
            models = module.train(x_train, y_label_train, y_distance_train)

        pred_labels, pred_distances = module.predict(models, x_test)
        cls_report = evaluate.classification_report(y_label_test, pred_labels)
        dist_report = evaluate.distance_report(y_distance_test, pred_distances)
        latency = evaluate.benchmark_latency(lambda x: module.predict(models, x), x_test[:1])
        results[name] = {
            "models": models,
            "classification": cls_report,
            "distance": dist_report,
            "latency": latency,
        }

    return results

"""Train-and-compare orchestrator: loads the labeled CSVs from
`data/raw/`, does a person-disjoint split, trains the logistic-regression
baseline, Random Forest, XGBoost, and the PyTorch MLP, evaluates all four
on the held-out test split, and exports the MLP to ONNX for the app to use.

Usage:
    python scripts/train.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from slouchfix.models import dataset, evaluate, export_onnx, train_baseline, train_mlp, train_rf, train_xgb

MODEL_MODULES = {
    "Logistic Regression (baseline)": train_baseline,
    "Random Forest": train_rf,
    "XGBoost": train_xgb,
    "MLP (PyTorch)": train_mlp,
}


def main() -> None:
    print("Loading labeled dataset from data/raw/ ...")
    df = dataset.load_raw_dataset()
    print(f"Loaded {len(df)} labeled frames from {df['person_id'].nunique()} people.")

    train_df, val_df, test_df = dataset.person_disjoint_split(df)
    print(
        f"Split -> train: {len(train_df)} rows / {train_df['person_id'].nunique()} people, "
        f"val: {len(val_df)} rows / {val_df['person_id'].nunique()} people, "
        f"test: {len(test_df)} rows / {test_df['person_id'].nunique()} people"
    )

    x_train_raw, y_label_train, y_distance_train = dataset.to_xy(train_df)
    x_val_raw, y_label_val, y_distance_val = dataset.to_xy(val_df)
    x_test_raw, y_label_test, y_distance_test = dataset.to_xy(test_df)

    mean, std = dataset.fit_standard_scaler(x_train_raw)
    x_train = dataset.apply_standard_scaler(x_train_raw, mean, std)
    x_val = dataset.apply_standard_scaler(x_val_raw, mean, std)
    x_test = dataset.apply_standard_scaler(x_test_raw, mean, std)

    results = {}
    trained_mlp = None

    for name, module in MODEL_MODULES.items():
        print(f"\nTraining {name} ...")
        if module is train_mlp:
            models = module.train(
                x_train, y_label_train, y_distance_train, x_val, y_label_val, y_distance_val
            )
            trained_mlp = models
        else:
            models = module.train(x_train, y_label_train, y_distance_train)

        pred_labels, pred_distances = module.predict(models, x_test)
        cls_report = evaluate.classification_report(y_label_test, pred_labels)
        dist_report = evaluate.distance_report(y_distance_test, pred_distances)
        latency = evaluate.benchmark_latency(lambda x: module.predict(models, x), x_test[:1])
        evaluate.print_report(name, cls_report, dist_report, latency)
        results[name] = (cls_report, dist_report)

    print("\n=== Summary (test set, person-disjoint) ===")
    for name, (cls_report, dist_report) in results.items():
        print(f"  {name:32s} acc={cls_report['accuracy']:.3f}  f1={cls_report['macro_f1']:.3f}  MAE={dist_report['mae_cm']:.1f}cm")

    if trained_mlp is not None:
        print("\nExporting MLP to ONNX for the desktop app ...")
        export_onnx.export(trained_mlp["model"], list(trained_mlp["label_encoder"].classes_), mean, std)


if __name__ == "__main__":
    main()

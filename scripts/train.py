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

from slouchfix.models import dataset, evaluate, train_all

try:
    from slouchfix.models import export_onnx
except ImportError:
    export_onnx = None


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

    print(f"\nTraining {', '.join(train_all.MODEL_MODULES)} ...")
    all_results = train_all.train_and_evaluate_all(train_df, val_df, test_df)

    print("\n=== Summary (test set, person-disjoint) ===")
    for name in train_all.MODEL_MODULES:
        r = all_results[name]
        evaluate.print_report(name, r["classification"], r["distance"], r["latency"])
        print(
            f"  {name:32s} acc={r['classification']['accuracy']:.3f}  "
            f"f1={r['classification']['macro_f1']:.3f}  MAE={r['distance']['mae_cm']:.1f}cm"
        )

    mean, std = all_results["_scaler"]
    trained_mlp = all_results.get("_mlp")
    if trained_mlp is not None and export_onnx is not None:
        print("\nExporting MLP to ONNX for the desktop app ...")
        export_onnx.export(trained_mlp["model"], list(trained_mlp["label_encoder"].classes_), mean, std)


if __name__ == "__main__":
    main()

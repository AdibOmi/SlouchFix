"""Trains + evaluates one or more of the project's models on whatever's in
`data/raw/`, using the exact same loading/split/metric code as
`scripts/train.py` / `scripts/evaluate_report.py` (`slouchfix/models/*`), so
results are apples-to-apples comparable across separate runs.

Excludes the legacy flat synthetic dataset (`synth_p0*_s1.csv`, from
`scripts/_make_synthetic_dataset.py`) by default -- that generator's own
docstring calls it a dev-only placeholder to delete before real use, and
mixing it with the newer persona-based generator (`scripts/dataset.py`)
would blend two different synthetic methodologies into one run.

Usage:
    python scripts/run_model_comparison.py --model xgboost
    python scripts/run_model_comparison.py --model xgboost rf mlp
    python scripts/run_model_comparison.py --model xgboost --include-legacy-synth
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slouchfix import config
from slouchfix.models import dataset, evaluate, train_baseline, train_rf

MODEL_CHOICES = {
    "logreg": ("Logistic Regression (baseline)", train_baseline),
    "rf": ("Random Forest", train_rf),
}
try:
    from slouchfix.models import train_xgb
    MODEL_CHOICES["xgboost"] = ("XGBoost", train_xgb)
except ImportError:
    pass
try:
    from slouchfix.models import train_mlp
    MODEL_CHOICES["mlp"] = ("MLP (PyTorch)", train_mlp)
except ImportError:
    pass

REPORTS_DIR = config.PROJECT_ROOT / "reports"
RESULTS_JSON = REPORTS_DIR / "model_comparison_results.json"


def _dataset_fingerprint(df) -> dict:
    return {
        "n_rows": len(df),
        "n_people": int(df["person_id"].nunique()),
        "people": sorted(str(p) for p in df["person_id"].unique()),
    }


def load_dataset(include_legacy_synth: bool):
    df = dataset.load_raw_dataset()
    total_rows = len(df)
    if not include_legacy_synth:
        legacy_mask = df["person_id"].astype(str).str.startswith("synth_")
        n_legacy = int(legacy_mask.sum())
        df = df[~legacy_mask].reset_index(drop=True)
        if n_legacy:
            print(f"Excluded {n_legacy} rows from the legacy flat synthetic set (synth_p0*_s1.csv).")
    print(f"Loaded {len(df)}/{total_rows} labeled frames from {df['person_id'].nunique()} people: "
          f"{sorted(df['person_id'].unique())}")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train + evaluate SlouchFix models for comparison")
    parser.add_argument("--model", nargs="+", choices=sorted(MODEL_CHOICES), required=True)
    parser.add_argument("--include-legacy-synth", action="store_true",
                         help="Include the old flat synth_p0*_s1.csv files too (off by default)")
    args = parser.parse_args()

    df = load_dataset(args.include_legacy_synth)
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

    RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _dataset_fingerprint(df)
    stored = json.loads(RESULTS_JSON.read_text(encoding="utf-8")) if RESULTS_JSON.exists() else {}
    all_results = stored.get("models", {})
    if stored.get("dataset") and stored["dataset"] != fingerprint:
        print(
            f"\n[Notice] Dataset changed since the last saved comparison "
            f"({stored['dataset']['n_people']} people / {stored['dataset']['n_rows']} rows -> "
            f"{fingerprint['n_people']} people / {fingerprint['n_rows']} rows). "
            "Resetting model_comparison_results.json so old and new models aren't compared "
            "across different datasets.\n"
        )
        all_results = {}

    for key in args.model:
        name, module = MODEL_CHOICES[key]
        print(f"\n=== Training {name} ===")
        if key == "mlp":
            models = module.train(x_train, y_label_train, y_distance_train, x_val, y_label_val, y_distance_val)
        else:
            models = module.train(x_train, y_label_train, y_distance_train)

        pred_labels, pred_distances = module.predict(models, x_test)
        cls_report = evaluate.classification_report(y_label_test, pred_labels)
        dist_report = evaluate.distance_report(y_distance_test, pred_distances)
        latency = evaluate.benchmark_latency(lambda x, m=models, mod=module: mod.predict(m, x), x_test[:1])
        evaluate.print_report(name, cls_report, dist_report, latency)

        all_results[name] = {
            "classification": cls_report,
            "distance": dist_report,
            "latency": latency,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            "n_people_train": int(train_df["person_id"].nunique()),
            "n_people_val": int(val_df["person_id"].nunique()),
            "n_people_test": int(test_df["person_id"].nunique()),
        }

    RESULTS_JSON.write_text(
        json.dumps({"dataset": fingerprint, "models": all_results}, indent=2), encoding="utf-8"
    )
    print(f"\nSaved/updated results for {len(args.model)} model(s) to {RESULTS_JSON}")

    print("\n=== All models recorded so far (this dataset) ===")
    for name, r in all_results.items():
        print(
            f"  {name:32s} acc={r['classification']['accuracy']:.3f}  "
            f"f1={r['classification']['macro_f1']:.3f}  "
            f"MAE={r['distance']['mae_cm']:.1f}cm  "
            f"latency={r['latency']['latency_ms']:.2f}ms ({r['latency']['fps']:.0f} FPS)"
        )


if __name__ == "__main__":
    main()

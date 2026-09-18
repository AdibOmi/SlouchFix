"""Compares 3 candidate models (Logistic Regression, XGBoost, MLP) on the
real `datasets/*.csv` files the user collected, using the project's own
training/eval code (`slouchfix/models/*`) so metrics are computed the same
way as `reports/evaluation_report.md`.

NOT the official pipeline (`scripts/train.py` / `scripts/evaluate_report.py`).
Those require the exact `slouchfix.features.FEATURE_NAMES` schema and >=3
distinct person_ids; `datasets/*.csv` has neither (see chat discussion).
This script is a best-effort bridge to get comparable numbers now:

  - Renames/derives what it can from the actual columns present.
  - DROPS `inter_eye_px_norm` and `nose_to_eye_ratio` -- there is no source
    data for them at all -- rather than faking values. Trains on an
    8-feature set, not the app's real 10.
  - `motion_score` is approximated as sqrt(vel_x^2 + vel_y^2), a different
    underlying computation than the real multi-landmark displacement score.
  - Only 2 real person_ids exist (p07 x4 sessions, p12 x1 session), so this
    uses a person-aware split (p12's session as test, 3 of p07's sessions as
    train, 1 as val) instead of the official >=3-person-disjoint split --
    directionally similar but far less statistically robust (n=1 test
    person). Do not treat these numbers as production-grade accuracy.
  - A model trained here will NOT load into the live app: `inference.py`
    checks the exported feature list against `features.FEATURE_NAMES`
    exactly and falls back to the rule-based baseline on any mismatch.

Usage:
    python scripts/compare_models_real_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from slouchfix import config
from slouchfix.models import dataset, evaluate, train_baseline, train_mlp, train_xgb

DATASETS_DIR = config.PROJECT_ROOT / "datasets"
REPORT_PATH = config.PROJECT_ROOT / "reports" / "real_data_model_comparison.md"

REDUCED_FEATURE_NAMES = [
    "pitch_deg",
    "yaw_deg",
    "roll_deg",
    "bbox_width_frac",
    "bbox_height_frac",
    "bbox_area_frac",
    "face_center_y_frac",
    "motion_score_proxy",
]

MODEL_MODULES = {
    "XGBoost": train_xgb,
    "MLP (PyTorch)": train_mlp,
    "Logistic Regression (baseline)": train_baseline,
}


def load_and_convert(datasets_dir: Path = DATASETS_DIR) -> pd.DataFrame:
    files = sorted(datasets_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {datasets_dir}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        # session_id isn't a column in this schema -- recover it from the
        # filename, e.g. "p07_20260910_091503.csv" -> "20260910_091503".
        session_id = "_".join(f.stem.split("_")[1:])
        df["person_id"] = df["volunteer_id"]
        df["session_id"] = session_id
        df["bbox_width_frac"] = df["face_width_norm"]
        df["bbox_height_frac"] = df["face_height_norm"]
        df["bbox_area_frac"] = df["face_width_norm"] * df["face_height_norm"]
        df["face_center_y_frac"] = df["face_center_y"]
        df["pitch_deg"] = df["head_pitch_deg"]
        df["yaw_deg"] = df["head_yaw_deg"]
        df["roll_deg"] = df["head_roll_deg"]
        df["motion_score_proxy"] = np.sqrt(df["vel_x"] ** 2 + df["vel_y"] ** 2)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def person_aware_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """p12's single session -> test (the only unseen-person signal we have).
    p07's 4 sessions, sorted chronologically by session_id: last -> val,
    the other 3 -> train."""
    p07_sessions = sorted(df.loc[df["person_id"] == "p07", "session_id"].unique())
    val_session = p07_sessions[-1]
    train_sessions = p07_sessions[:-1]

    train_df = df[(df["person_id"] == "p07") & (df["session_id"].isin(train_sessions))].reset_index(drop=True)
    val_df = df[(df["person_id"] == "p07") & (df["session_id"] == val_session)].reset_index(drop=True)
    test_df = df[df["person_id"] == "p12"].reset_index(drop=True)
    return train_df, val_df, test_df


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df[REDUCED_FEATURE_NAMES].to_numpy(dtype=np.float32)
    y_label = df["label"].to_numpy()
    y_distance = df["distance_cm"].to_numpy(dtype=np.float32)
    return x, y_label, y_distance


def main() -> None:
    print("Loading + converting datasets/*.csv ...")
    df = load_and_convert()
    print(f"Loaded {len(df)} frames from {df['person_id'].nunique()} people: "
          f"{sorted(df['person_id'].unique())}")

    train_df, val_df, test_df = person_aware_split(df)
    print(
        f"Split -> train: {len(train_df)} rows (p07, {train_df['session_id'].nunique()} sessions), "
        f"val: {len(val_df)} rows (p07, held-out session), "
        f"test: {len(test_df)} rows (p12, fully unseen person)"
    )

    x_train_raw, y_label_train, y_distance_train = to_xy(train_df)
    x_val_raw, y_label_val, y_distance_val = to_xy(val_df)
    x_test_raw, y_label_test, y_distance_test = to_xy(test_df)

    mean, std = dataset.fit_standard_scaler(x_train_raw)
    x_train = dataset.apply_standard_scaler(x_train_raw, mean, std)
    x_val = dataset.apply_standard_scaler(x_val_raw, mean, std)
    x_test = dataset.apply_standard_scaler(x_test_raw, mean, std)

    results: dict = {}
    for name, module in MODEL_MODULES.items():
        print(f"\nTraining {name} ...")
        if module is train_mlp:
            models = module.train(x_train, y_label_train, y_distance_train, x_val, y_label_val, y_distance_val)
        else:
            models = module.train(x_train, y_label_train, y_distance_train)

        pred_labels, pred_distances = module.predict(models, x_test)
        cls_report = evaluate.classification_report(y_label_test, pred_labels)
        dist_report = evaluate.distance_report(y_distance_test, pred_distances)
        latency = evaluate.benchmark_latency(lambda x, m=models, mod=module: mod.predict(m, x), x_test[:1])
        results[name] = {"classification": cls_report, "distance": dist_report, "latency": latency}
        evaluate.print_report(name, cls_report, dist_report, latency)

    print("\n=== Summary (test = p12, unseen person) ===")
    for name in MODEL_MODULES:
        r = results[name]
        print(
            f"  {name:32s} acc={r['classification']['accuracy']:.3f}  "
            f"f1={r['classification']['macro_f1']:.3f}  "
            f"MAE={r['distance']['mae_cm']:.1f}cm  "
            f"latency={r['latency']['latency_ms']:.2f}ms"
        )

    _write_report(df, train_df, val_df, test_df, results)


def _write_report(df, train_df, val_df, test_df, results: dict) -> None:
    lines: list[str] = []
    lines.append("# SlouchFix -- Real-Data Model Comparison (provisional)")
    lines.append("")
    lines.append(
        "> **Not the official pipeline.** This compares 3 models "
        "(Logistic Regression, XGBoost, MLP) trained on `datasets/*.csv`, converted from a "
        "non-matching schema. Caveats:\n"
        "> - Trained on an **8-feature reduced set** -- `inter_eye_px_norm` and "
        "`nose_to_eye_ratio` have no source data in these files and were dropped rather "
        "than faked. The app's real feature vector has 10 features; a model trained here "
        "**will not load into the live app** (`inference.py` rejects any feature-list mismatch "
        "and falls back to the rule-based baseline).\n"
        "> - `motion_score_proxy` is `sqrt(vel_x^2+vel_y^2)`, a different underlying signal "
        "than the app's real multi-landmark motion score.\n"
        "> - Only **2 real people** exist in the data (p07 x4 sessions, p12 x1 session), "
        "below the project's own >=3-person minimum for a person-disjoint split. Test set here "
        "is p12's single session -- an n=1 unseen-person test, not statistically robust. "
        "Treat these numbers as directional only.\n"
    )
    lines.append("")
    lines.append(
        f"Dataset: **{len(df)} frames**, people: {sorted(df['person_id'].unique())}. "
        f"Train: {len(train_df)} rows (p07, {train_df['session_id'].nunique()} sessions) / "
        f"Val: {len(val_df)} rows (p07, held-out session) / "
        f"Test: {len(test_df)} rows (p12, unseen person)."
    )
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 | Distance MAE (cm) | Latency |")
    lines.append("|---|---|---|---|---|")
    for name in MODEL_MODULES:
        r = results[name]
        lat = f"{r['latency']['latency_ms']:.2f} ms ({r['latency']['fps']:.0f} FPS)"
        lines.append(
            f"| {name} | {r['classification']['accuracy']:.3f} | "
            f"{r['classification']['macro_f1']:.3f} | {r['distance']['mae_cm']:.1f} | {lat} |"
        )
    lines.append("")

    best_name = max(MODEL_MODULES, key=lambda n: results[n]["classification"]["macro_f1"])
    lines.append(f"**Best of these 3 on this run:** {best_name}.")
    lines.append("")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH}")


if __name__ == "__main__":
    main()

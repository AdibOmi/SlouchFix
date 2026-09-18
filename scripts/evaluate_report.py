"""Builds `reports/evaluation_report.md` (Required Feature 12): baselines vs
trained models, ablations, and a confusion matrix chart, all computed from
whatever's currently in `data/raw/`.

Usage:
    python scripts/evaluate_report.py

Honesty note (per the Phase 1 build brief): if `data/raw/` only has the
synthetic dataset (`scripts/_make_synthetic_dataset.py`), or otherwise too
few real people/frames, the report says so explicitly rather than
presenting synthetic numbers as if they were real-world results.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from slouchfix import config
from slouchfix.models import ablations, baseline_eval, dataset, train_all

REPORTS_DIR = config.PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def _is_synthetic_only(df) -> bool:
    return all(str(p).startswith("synth_") for p in df["person_id"].unique())


def _fmt_row(name: str, cls: dict, dist: dict, latency: dict | None = None) -> str:
    lat = f"{latency['latency_ms']:.2f} ms ({latency['fps']:.0f} FPS)" if latency else "n/a"
    return f"| {name} | {cls['accuracy']:.3f} | {cls['macro_f1']:.3f} | {dist['mae_cm']:.1f} | {lat} |"


def _plot_confusion_matrix(cm: list[list[int]], labels: list[str], title: str, out_path: Path) -> None:
    cm_arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_arr, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm_arr[i, j]), ha="center", va="center",
                     color="white" if cm_arr[i, j] > cm_arr.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _ablation_table(rows: dict[str, dict]) -> str:
    lines = ["| Arm | Accuracy | Macro F1 | Distance MAE (cm) |", "|---|---|---|---|"]
    for name, r in rows.items():
        cls, dist = r["classification"], r["distance"]
        lines.append(f"| {name} | {cls['accuracy']:.3f} | {cls['macro_f1']:.3f} | {dist['mae_cm']:.1f} |")
    return "\n".join(lines)


def main() -> None:
    print("Loading labeled dataset from data/raw/ ...")
    df = dataset.load_raw_dataset()
    synthetic_only = _is_synthetic_only(df)
    n_people = df["person_id"].nunique()
    print(f"Loaded {len(df)} labeled frames from {n_people} people (synthetic_only={synthetic_only}).")

    train_df, val_df, test_df = dataset.person_disjoint_split(df)
    print(
        f"Split -> train: {len(train_df)} rows / {train_df['person_id'].nunique()} people, "
        f"val: {len(val_df)} rows / {val_df['person_id'].nunique()} people, "
        f"test: {len(test_df)} rows / {test_df['person_id'].nunique()} people"
    )

    print("\nEvaluating Baseline 1 (rule-based, calibrated) ...")
    baseline1 = baseline_eval.evaluate_rule_based_baseline(test_df)
    print("Evaluating Baseline 2 (naive off-the-shelf pose, uncalibrated) ...")
    baseline2 = baseline_eval.evaluate_naive_pose_baseline(test_df)

    print("\nTraining and evaluating LogReg / RF / XGBoost / MLP ...")
    model_results = train_all.train_and_evaluate_all(train_df, val_df, test_df)

    print("\nRunning ablation A (calibration-relative vs raw features) ...")
    calib_ablation = ablations.run_calibration_ablation(train_df, val_df, test_df)
    print("Running ablation B (full features vs motion-feature-ablated) ...")
    feature_ablation = ablations.run_feature_subset_ablation(train_df, val_df, test_df)

    best_name = max(
        (n for n in train_all.MODEL_MODULES),
        key=lambda n: model_results[n]["classification"]["macro_f1"],
    )
    best = model_results[best_name]
    cm_path = FIGURES_DIR / "confusion_matrix_best_model.png"
    _plot_confusion_matrix(
        best["classification"]["confusion_matrix"],
        best["classification"]["labels_order"],
        f"Confusion matrix -- {best_name} (test set)",
        cm_path,
    )

    best_beats_rule_based = best["classification"]["macro_f1"] > baseline1["classification"]["macro_f1"]

    lines: list[str] = []
    lines.append("# SlouchFix -- Evaluation Report")
    lines.append("")
    lines.append(f"Dataset: **{len(df)} labeled frames** from **{n_people} people** in `data/raw/`.")
    if synthetic_only:
        lines.append(
            "\n> **Synthetic dataset notice:** every person_id in this run comes from "
            "`scripts/_make_synthetic_dataset.py`, not a real recorded person. These numbers "
            "validate that the training/evaluation/ablation pipeline runs correctly end-to-end "
            "and are **not evidence the trained model works on real posture data** -- the "
            "synthetic label profiles are cleanly separated by construction, so high scores here "
            "are expected and not a claim of real-world accuracy. Re-run this script after "
            "collecting real sessions with `python -m slouchfix.data_collection --person <id>` "
            "(need at least 3 distinct real people for the person-disjoint split) to get numbers "
            "that mean anything about real-world performance.\n"
        )
    elif n_people < 5 or len(df) < 500:
        lines.append(
            f"\n> **Small-dataset notice:** only {n_people} people / {len(df)} frames were available "
            "for this run -- smaller than the 5-10 people the full data-collection protocol calls "
            "for. Treat these numbers as directional, not final.\n"
        )
    lines.append("")

    lines.append("## Baselines vs trained models (test set, person-disjoint)")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 | Distance MAE (cm) | Latency |")
    lines.append("|---|---|---|---|---|")
    lines.append(_fmt_row("Baseline 1: rule-based (calibrated)", baseline1["classification"], baseline1["distance"], baseline1["latency"]))
    lines.append(_fmt_row("Baseline 2: naive pose (uncalibrated)", baseline2["classification"], baseline2["distance"], baseline2["latency"]))
    for name in train_all.MODEL_MODULES:
        r = model_results[name]
        lines.append(_fmt_row(name, r["classification"], r["distance"], r["latency"]))
    lines.append("")

    if best_beats_rule_based:
        lines.append(
            f"**Result:** the best trained model (**{best_name}**, macro F1="
            f"{best['classification']['macro_f1']:.3f}) beats the calibrated rule-based baseline "
            f"(macro F1={baseline1['classification']['macro_f1']:.3f})."
        )
    else:
        lines.append(
            f"**Result (honest negative finding):** no trained model beat the calibrated rule-based "
            f"baseline on this dataset -- best trained macro F1 is {best['classification']['macro_f1']:.3f} "
            f"({best_name}) vs {baseline1['classification']['macro_f1']:.3f} for the rule-based baseline. "
            "With a bootstrap-sized dataset this is an expected and legitimate Phase 1 outcome, not a bug: "
            "the rule-based baseline is hand-tuned per the exact class definitions, while the trained "
            "models have only a few hundred frames per class to learn the same boundaries from scratch. "
            "This is expected to flip once the full 5-10-person, multi-session dataset is collected."
        )
    lines.append("")

    lines.append(f"## Confusion matrix -- {best_name} (best trained model)")
    lines.append("")
    lines.append(f"![confusion matrix](figures/{cm_path.name})")
    lines.append("")

    lines.append("## Ablation A: calibration-relative vs raw-absolute features (MLP)")
    lines.append("")
    lines.append(
        "Tests whether normalizing features against a per-person baseline -- the same "
        "architectural idea behind the rule-based baseline's calibration step -- also helps the "
        "*trained* model. See `slouchfix/models/ablations.py` for exactly how the per-person "
        "baseline is computed."
    )
    lines.append("")
    lines.append(_ablation_table(calib_ablation))
    lines.append("")

    lines.append("## Ablation B: full feature set vs motion-feature-ablated (MLP)")
    lines.append("")
    lines.append(
        "The build brief's suggested second ablation (landmark-only vs raw-frame CNN features) needs "
        "a separate image-input training pipeline that's out of scope for this lightweight Phase 1 pass. "
        "Substituted: full 10-feature set vs the same set with `motion_score` (the only feature requiring "
        "frame-history, i.e. the only 'temporal' feature) removed, to see whether that signal earns its "
        "complexity."
    )
    lines.append("")
    lines.append(_ablation_table(feature_ablation))
    lines.append("")

    lines.append("## Qualitative analysis")
    lines.append("")
    if not synthetic_only:
        lines.append(
            "See `reports/qualitative/` for screenshots. Screenshots are captured live: run "
            "`python scripts/run_app.py --preview`, hold the posture you want to illustrate, and "
            "press `s` to save the current preview frame (labeled with the currently-detected state "
            "and a timestamp) into `reports/qualitative/`. Capture at least good_posture plus one "
            "example of each flagged case for the write-up."
        )
    else:
        lines.append(
            "Not included in this run: qualitative screenshots require an actual live webcam session, "
            "which this automated evaluation script does not perform (there's no real person or camera "
            "in a synthetic-dataset run). Once real sessions exist, run "
            "`python scripts/run_app.py --preview` and press `s` during good_posture and each flagged "
            "case (slouched, leaning_forward, too_close, head_tilted, looking_away) to save labeled "
            "screenshots to `reports/qualitative/`, then reference them here."
        )
    lines.append("")

    lines.append("## Known limitations (Phase 1 scope)")
    lines.append("")
    lines.append(
        "- Distance regression MAE and posture accuracy above are only as good as the bootstrap "
        "dataset; see the dataset notice at the top of this report.\n"
        "- The orientation/mounting-angle corner case (lid angle, lap vs. table, riser height) is "
        "handled by calibrating relative to a personal baseline (`slouchfix/calibration.py`) rather "
        "than fixed absolute thresholds, plus a frame-to-frame drift monitor "
        "(`slouchfix.calibration.RecalibrationMonitor`) that flags when the current session's "
        "calibration may be stale. Accelerometer-based absolute gravity fusion is deferred to a "
        "Phase 2 mobile build; `slouchfix.baseline.classify` already accepts an optional "
        "`device_tilt_angle` parameter (default `None`) as that future seam.\n"
        "- No CNN/raw-frame model was trained this phase (see Ablation B); model export is PyTorch -> "
        "ONNX only, no TFLite/mobile export yet."
    )
    lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "evaluation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()

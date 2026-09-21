# Pose-pipeline integration: design spec

Date: 2026-09-21

## Context

The live `slouchfix/` app currently runs a face-landmark pipeline: MediaPipe
`FaceLandmarker` → solvePnP head pose (pitch/yaw/roll) + bbox/motion features
→ a personally-calibrated, 5-rule threshold baseline (`too_close`,
`looking_away`, `head_tilted`, `leaning_forward`, `slouched`,
`good_posture`). It requires a per-user calibration step
(`calibration.py`) because every rule is scored relative to that user's own
neutral baseline.

The ablation study (`paper/slouchfix_paper.tex`, `result_notebook/`)
identified a different, better pipeline for this problem: MoveNet (body
pose) → a 4-dimensional joint-angle representation (2 elbow angles, 1 neck
angle, 1 shoulder-tilt angle) → an RBF-kernel SVM, binary `good`/`slouched`
only, 97.7% held-out accuracy on the 2,000-image synthetic dataset, no
per-user calibration required. This spec wires that pipeline into the live
app, replacing the face-landmark system entirely.

## Decisions (from brainstorming)

1. **Full replacement.** `too_close` / `looking_away` / `head_tilted` /
   `leaning_forward` and the calibration system are removed, not kept
   alongside the new pipeline. The app becomes binary `good` / `slouched`.
2. **Both training-data sources.** The shipped default model is trained on
   the same 2,000-image synthetic (Stable-Diffusion) dataset the paper
   used. A personal-data collection path is also built, so the model can
   optionally be retrained on the user's own webcam sessions later. These
   two sources share one feature schema (the 4-d angle vector), so they
   can be combined in a single training run without translation.
3. **MoveNet runtime: ONNX via `onnxruntime`.** Not TensorFlow (too heavy
   for a background app) and not a separate `tflite-runtime` interpreter
   (revised from an earlier version of this decision) — the live app uses
   `onnxruntime` for both the pose estimator and the SVM classifier, one
   inference runtime for everything. Uses the public
   `Xenova/movenet-singlepose-lightning` ONNX export (verified: input
   `(1,192,192,3)` int32 raw RGB pixels, output `(1,1,17,3)`
   `[y, x, score]` normalized to `[0,1]` — identical keypoint contract to
   the original `.tflite`).
4. **Distance becomes a non-ML, informational heuristic.** Shoulder-width
   in pixels vs. an assumed average shoulder width, shown in the UI, never
   fed to the classifier or used in any decision (the same
   "population-average, deliberately approximate" pattern the old
   `naive_pose_baseline.py` used for interpupillary distance).
5. **No rule-based fallback.** `inference.py` requires the bundled ONNX SVM
   to be present; there is no zero-model, threshold-based fallback path
   (the old `baseline.py` behavior is fully retired, not kept as a safety
   net).

## Removed

- `slouchfix/landmarks.py` (MediaPipe FaceLandmarker wrapper)
- `slouchfix/features.py` (face-based 10-feature extractor)
- `slouchfix/baseline.py` (rule-based threshold classifier)
- `slouchfix/naive_pose_baseline.py` (fixed-threshold comparison baseline)
- `slouchfix/calibration.py` (per-user calibration)
- `slouchfix/models/train_baseline.py`, `train_mlp.py`, `train_rf.py`,
  `train_xgb.py`, `ablations.py`, `baseline_eval.py`, `dataset.py`,
  `evaluate.py` (all built around the retired 10-feature face vector)
- `config.py`: face-mesh landmark indices, 3D model points, the 5
  threshold constants, `CALIBRATION_PATH`
- `settings.py`: `too_close_distance_cm`, `yaw_looking_away_deg`,
  `roll_head_tilted_deg`, `pitch_leaning_forward_deg`,
  `slouch_face_drop_ratio`
- `app.py`: calibration acquisition/recalibration state and flow
- `server.py`: `/api/calibration/*` endpoints, `needs_calibration` /
  `recalibrating` / `recalibration_progress` response fields, the
  threshold-settings request model
- `ui.py`: Recalibrate button, threshold-editing fields, calibration-stale
  banner
- `tray.py`: Recalibrate menu item

## New / rewritten

### `slouchfix/pose.py` (replaces `landmarks.py`)

MoveNet-Lightning wrapper via `onnxruntime`, loading a bundled
`movenet_lightning.onnx` from `slouchfix/assets/` (not committed to git —
see Dependencies). Given a BGR frame, returns the 17 COCO-style keypoints
(pixel coords + per-joint confidence) MoveNet natively outputs, or `None`
if no person is detected above threshold. No cross-backend
canonical-schema remapping is needed — the live app only ever runs this
one backend, unlike the ablation study's six-backend comparison.

### `slouchfix/pose_features.py` (replaces `features.py`)

Computes the exact 4-d angle vector from the paper directly from MoveNet's
keypoints:

- Two elbow angles (shoulder–elbow–wrist, per side)
- Neck angle (vertical vs. nose-to-mid-shoulder vector)
- Shoulder-tilt angle (shoulder line vs. horizontal)

Also computes the separate, non-feature distance heuristic
(shoulder-width-px vs. assumed average shoulder width → `distance_cm`),
kept out of the feature vector entirely. `FEATURE_NAMES` is now
`["elbow_left_deg", "elbow_right_deg", "neck_deg", "shoulder_tilt_deg"]`.

Shared between training (`models/dataset.py`) and inference (`app.py`), so
a train/deploy feature mismatch stays a naming bug, not a possibility —
same principle the retired `features.py` docstring stated.

### `slouchfix/models/dataset.py` (rewritten)

Loads the 2,000-image HF subset (via `scripts/download_hf_posture_subset.py`,
downloading it if not already present locally), runs each image through
`pose.py` + `pose_features.py`, and produces a `(N, 4)` feature matrix +
binary labels. Skips images where MoveNet doesn't detect all four required
joints on either side (no imputation in this rewritten path — the ablation
study's imputation logic was for handling six heterogeneous backends across
a full dataset characterization; the live training path only needs a clean
fit set).

### `slouchfix/models/train_svm.py` (new)

Trains `sklearn.pipeline.Pipeline([StandardScaler(), SVC(kernel="rbf",
probability=True)])` on the angle features, matching the notebook's
winning arm's hyperparameters. Default: synthetic dataset only. `--with-
personal <csv>` flag concatenates a personal-session CSV (same 4-column
schema) before fitting.

### `slouchfix/models/export_onnx.py` (rewritten)

Exports the trained sklearn pipeline via `skl2onnx.convert_sklearn` (the
`StandardScaler` is embedded in the exported graph, so the earlier
`feature_mean`/`feature_std` fields in the metadata JSON are dropped — the
ONNX graph does its own scaling). Metadata JSON now contains only
`feature_names` and `labels`.

### `slouchfix/inference.py` (rewritten)

`PostureEngine.__init__` requires `posture_model.onnx` +
`posture_model_meta.json` to exist and match `pose_features.FEATURE_NAMES`;
raises a clear startup error if not (per decision 5 — no fallback).
`classify(angle_features)` runs the ONNX session, returns
`PostureReading(label, confidence, distance_cm)` with `distance_cm` filled
in from the separate heuristic in `pose_features.py`, not from the model.

### `slouchfix/data_collection.py` (rewritten)

Runs live MoveNet against the webcam; the user labels short sessions with
a keypress (`g` = good, `s` = slouched); appends rows to a personal CSV in
the same 4-angle schema the synthetic dataset uses, so it can be passed
straight to `train_svm.py --with-personal`.

## Unaffected

`tracker.py` (already treats `label` as an opaque string — no change
needed), `history.py` (schema is already just
`[timestamp, label, confidence, distance_cm]`), `notifier.py`'s structure
(only its per-label message dict shrinks to one `"slouched"` entry),
`capture.py`.

## Dependencies

Add: `skl2onnx`, `scikit-learn`, `joblib` (training-time only) —
`onnxruntime` was already a dependency and now also runs the pose
estimator. Drop `mediapipe` from the live app's runtime path (no longer
used at inference time); leave it in `requirements.txt` for now rather
than force-removing it, since the ablation notebook/paper tooling still
references it.

`slouchfix/assets/movenet_lightning.onnx` (the pose estimator's weights,
~9MB) is gitignored, not committed — same pattern as the retired
`face_landmarker.task`. It's fetched from the public
`Xenova/movenet-singlepose-lightning` Hugging Face repo (see `pose.py`'s
error message for the exact URL) rather than downloaded automatically on
first run, matching decision 5's "require bundled, no silent fallback"
posture: a missing asset should be a clear, explicit setup step.

## Error handling

- Missing MoveNet `.tflite` asset or missing/mismatched ONNX model: fail
  fast at startup with a clear message (no silent fallback).
- No person detected in frame: skip the frame, same branch shape as the
  old `face is None` handling, re-pointed at pose detection.
- Person detected but a required joint is missing/low-confidence: skip the
  frame (no partial-feature imputation in the live path).

## Testing

- Unit tests for `pose_features.py`'s angle math against synthetic
  keypoint arrays with known expected angles.
- A small fixture test: train `train_svm.py` on a tiny synthetic sample,
  export via `export_onnx.py`, confirm `inference.py` can load and
  classify against it.
- Manual end-to-end pass with `--preview` against a real webcam before
  calling the work done.

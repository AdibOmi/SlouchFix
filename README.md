# SlouchFix
<<<<<<< HEAD

On-device posture and screen-distance detection for programmers, running as
a Windows/PC background app. See `SlouchFix.pdf` for the original pitch and
`TECH_STACK.md` for the (PC-adapted) technology and algorithm stack.

The app monitors webcam-derived facial landmarks in real time, entirely
on-device (no video ever leaves the machine), and nudges you with a desktop
toast when you're too close to the screen, leaning forward, slouched, head
tilted, or looking away.

## Status

- **Real-time app (rule-based)**: done, and usable right now with just a
  webcam and one calibration step — no training data required.
- **ML pipeline** (dataset loader, person-disjoint split, 4 model types,
  metrics, ONNX export): done and validated end-to-end against a synthetic
  dataset (see `scripts/_make_synthetic_dataset.py`) — plumbing works, but
  no model is trained on real people yet, because that requires actually
  recording sessions (see "Collecting training data" below).
- The app runs with the rule-based baseline until a real model is trained
  and exported; at that point it switches automatically (`inference.py`
  prefers a trained ONNX model if `data/models/posture_model.onnx` exists).

## Setup

A Python 3.12 virtual environment is already set up at `.venv/` (MediaPipe
doesn't yet ship wheels for very new Python releases like 3.14, which is why
this pins 3.12). To recreate it elsewhere:

```
py -3.12 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

The MediaPipe FaceLandmarker model asset (`slouchfix/assets/face_landmarker.task`,
~3.6MB) is already downloaded and checked into this working copy.

## Running the app

First run does a one-time calibration: sit in good posture at a distance you
measure once (default assumption is ~50cm / arm's length — measure yours
and pass `--distance` if different), and stay still for ~2.5 seconds.

```
.venv\Scripts\python scripts\run_app.py --preview
```

- `--preview` opens a debug window showing the camera feed with the current
  detected state overlaid (label, confidence, estimated distance). Omit it
  for the real headless mode.
- `--no-notifications` prints alerts to the console instead of sending
  Windows toasts (useful when developing).
- `--recalibrate` forces a fresh calibration before starting.

For normal background use, run the system tray version instead — it has no
window, lives in the tray, and exposes Pause/Recalibrate/Quit from a right-click menu:

```
.venv\Scripts\python scripts\run_tray.py
```

Calibration is stored at `~/.slouchfix/calibration.json`. Delete it (or use
`--recalibrate`) if you change desks, monitor height, or camera position.

## Dashboard

A small window for everything that used to require the tray's right-click
menu or editing `config.py` by hand: starting/stopping/pausing monitoring,
recalibrating, browsing posture history, viewing time-in-each-state stats,
and tuning thresholds.

```
.venv\Scripts\python scripts\run_dashboard.py
```

- **Status** — current posture label, confidence, distance, time in state,
  Start/Stop, Pause/Resume, Recalibrate, and a notifications on/off toggle.
- **History** — a table of recent posture-state changes and sent alerts,
  newest first.
- **Stats** — time spent in each posture state and alert counts, for today /
  last 7 days / all time.
- **Settings** — the rule-based thresholds (too-close distance, tilt/lean/yaw
  angles), notification cooldown, debounce frame count, and camera index.
  Saved to `~/.slouchfix/settings.json`; changes apply the next time
  monitoring is (re)started.

Every run (dashboard or tray) appends to the same history log at
`data/history/`, so either one feeds the same History/Stats view.

## Collecting training data

To eventually beat the rule-based baseline with a trained model, record
labeled sessions from several people (the pitch deck calls for 5-10, across
different monitors, lighting, distances, and camera angles):

```
.venv\Scripts\python scripts\collect_data.py --person p01 --distance 50
```

Controls while it's running: `1`-`6` set the current posture label
(good_posture / slouched / leaning_forward / too_close / head_tilted /
looking_away), `r` toggles recording on/off, `+`/`-` adjust the recorded
distance_cm if you deliberately change how far you're sitting, `q`/Esc
quits. Each session is written to `data/raw/<person>_<session>.csv`. Use a
consistent `--person` id across that person's sessions and a different one
per person — the training split is person-disjoint (whole people held out
for validation/test), so `person_id` has to be trustworthy.

## Training

Once there's real labeled data in `data/raw/`:

```
.venv\Scripts\python scripts\train.py
```

This loads every CSV in `data/raw/`, splits by person (not by row), trains
the logistic-regression baseline, Random Forest, XGBoost, and a small
PyTorch MLP, prints accuracy / macro-F1 / distance MAE / latency for each on
the held-out test people, and exports the MLP to
`data/models/posture_model.onnx` (+ `posture_model_meta.json`). The app
picks that up automatically on the next run.

`scripts/_make_synthetic_dataset.py` and `scripts/smoke_test.py` are dev-only
helpers used to validate this whole pipeline without a webcam or real
people; they're not part of the shipped product.

## Project layout

```
slouchfix/
  capture.py         webcam frame acquisition (OpenCV)
  landmarks.py        MediaPipe FaceLandmarker wrapper
  features.py          feature engineering shared by training + inference
  calibration.py         one-time pixel-to-cm + baseline-pose calibration
  baseline.py             rule-based threshold classifier
  tracker.py               debouncing + duration-in-state tracking
  notifier.py               non-intrusive Windows toast notifications
  inference.py              picks rule-based vs. trained ONNX model
  app.py                    real-time console loop
  tray.py                    system tray wrapper around app.py
  ui.py                       Tkinter dashboard (status/history/stats/settings)
  settings.py                  user-adjustable thresholds, persisted to ~/.slouchfix/settings.json
  history.py                    usage log (state changes + alerts) for the dashboard
  data_collection.py             labeled session recorder
  models/
    dataset.py                 CSV loading + person-disjoint split
    train_baseline.py           logistic regression + linear regression
    train_rf.py                  random forest
    train_xgb.py                  XGBoost
    train_mlp.py                   PyTorch MLP (two heads: class + distance)
    evaluate.py                     accuracy/F1/MAE/latency metrics
    export_onnx.py                   PyTorch -> ONNX export for the app
scripts/               thin CLI entry points + dev helpers
data/
  raw/                 one CSV per recorded session (gitignored)
  processed/            (reserved for combined/cleaned datasets)
  models/                trained model exports (gitignored)
  history/                dashboard usage log: states.csv + alerts.csv (gitignored)
```
=======
A privacy-first desktop app that detects poor posture and unsafe screen distance in real time while you code.
>>>>>>> fed8e4caa5a2be4bf88167f45b598cb9889af636

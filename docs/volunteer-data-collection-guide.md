# SlouchFix Volunteer Data Collection Guide

This is the step-by-step protocol for volunteers recording labeled posture
sessions for the SlouchFix training dataset (see `README.md` →
"Collecting training data"). It covers device/camera position, lighting,
distance measurement, and exactly how to perform each of the six posture
labels so the recorded data is usable for training.

A polished, shareable version of this same guide (with diagrams) is meant to
be handed to volunteers directly — see the published Artifact link. This
file is the version-controlled source of truth.

## 0. What actually gets recorded (privacy)

`scripts/collect_data.py` never saves video or images. For each frame while
recording is on, it writes one row of *numbers* to a CSV: a timestamp, your
person id, the label you set, the distance you measured, and ten derived
measurements (head pitch/yaw/roll angles, face position and size as a
fraction of the frame, a motion score). The live camera window is only a
local preview so you can see your own framing — it is never saved or sent
anywhere. Only the CSV file leaves your machine.

## 1. Before you start

- Python 3.12 environment set up per the project `README.md` (`py -3.12 -m
  venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`). On
  Mac/Linux: `python3.12 -m venv .venv` and `source .venv/bin/activate`.
- A ruler or tape measure — you'll measure your real eye-to-screen distance,
  not guess it.
- A `person_id` assigned to you by the project coordinator (e.g. `p07`).
  **Use exactly this id for every session you record, and never reuse
  someone else's** — the training split holds out entire people for
  validation, so a wrong or shared id corrupts that split.
- 15–20 minutes somewhere you won't be interrupted (you'll be making
  deliberately bad-posture faces at a webcam; pick a private room if that
  matters to you).

## 2. Set up your camera position and angle

Camera position varies a lot in real use, and that's on purpose — the model
needs to see that variety, and each volunteer's *own* calibration is what
corrects for it (see `slouchfix/calibration.py`). Record separate sessions
(section 6) across the setups you can access:

- **Laptop flat on a desk** — the most common case. The camera ends up
  slightly below eye level, tilted up at your face.
- **Laptop on a stand/riser, or an external monitor with a webcam clipped to
  the top** — camera at or slightly above eye level.
- **Laptop on your lap or a low table** — camera well below eye level,
  looking up at a steep angle.

Within a single session, keep the camera fixed — don't reposition it
mid-recording, since a physical bump mid-session is exactly what
`RecalibrationMonitor` is designed to flag as invalid.

## 3. Frame yourself correctly

- Center your face in frame, close enough that your face and shoulders are
  both visible.
- Only one face in frame — the detector tracks a single primary face.
- Face a light source (a window or lamp in front of or beside you). Avoid
  sitting with a bright window or light directly *behind* you — backlighting
  silhouettes your face and the landmark detector loses it.
- Avoid near-total darkness. Normal room lighting is fine; you don't need
  studio lighting.

## 4. Measure your distance

Before each session, sit in your normal position and measure the distance
from your eyes to the screen with a ruler or tape measure (the app defaults
to assuming ~50cm / arm's length — measure yours). Pass it as `--distance`
when you start the recorder. If you deliberately move closer or farther
during the session (e.g. to record `too_close`), use the `+`/`-` keys live
to keep the logged number accurate — don't leave it at the starting value.

## 5. Run the recorder

```
.venv\Scripts\python scripts\collect_data.py --person p07 --distance 50
```

(replace `p07` and `50` with your assigned id and your measured distance)

A preview window opens showing your camera feed and an overlay of the
current label, recording status, distance, and frame count. Controls:

| Key | Action |
|---|---|
| `1`–`6` | Set the current posture label (see table below) |
| `r` | Toggle recording on/off |
| `+` / `-` | Adjust the logged distance by 1cm (use if you move) |
| `q` or `Esc` | Stop and save the session |

Recording only writes rows while both a face is detected **and** recording
is toggled on (`REC` shown in the overlay, and the frame counter climbing).
If the counter isn't moving, the detector isn't seeing your face — recheck
section 3 before continuing.

## 6. The six posture labels

Change the label *before* you change your posture, hold each pose steadily
for several seconds before moving to the next, and always start a session
with a stretch of genuine `good_posture` — that's the reference the model
(and your own live app calibration) compares everything else against.

| Key | Label | What to actually do |
|---|---|---|
| `1` | `good_posture` | Sit the way you actually would while focused on work: back reasonably upright, head level, at your normal measured distance, looking at the screen. |
| `2` | `slouched` | Let your back and shoulders round and sink down, so your face visibly drops lower in frame — a real slump, not a lean forward. |
| `3` | `leaning_forward` | Push your head and torso toward the screen while keeping your back angle similar to normal — a forward head push, not a distance change. |
| `4` | `too_close` | Scoot noticeably closer than your normal distance — aim for well under 40cm — and update the distance with `-`. |
| `5` | `head_tilted` | Tilt your head sideways toward one shoulder (ear-to-shoulder), not turning it — a clear, noticeable tilt. |
| `6` | `looking_away` | Turn your head/gaze clearly to the side or down — e.g. glancing at a phone or a second monitor — not just a small eye movement. |

Make each pose unambiguous but still like something you'd actually do —
avoid holding an identical, frozen exaggerated pose for the entire label
duration; natural small movement within the pose is fine and realistic.

## 7. Recording plan (vary conditions across sessions, not within one)

Run several short sessions rather than one long one, changing one variable
between sessions:

| Session | Vary this | Keep everything else the same |
|---|---|---|
| 1 | Normal desk setup, daytime light, your usual distance | — |
| 2 | Evening / lamp lighting | Same desk setup and distance |
| 3 | Laptop on your lap (low camera angle) | Same lighting as session 1 |
| 4 | External monitor or raised laptop (camera at/above eye level), if available | Same lighting as session 1 |

Each session: run through all six labels, roughly 20–30 seconds per label.
More sessions and more varied conditions are better than longer sessions —
the model learns from variety across people and setups, not raw volume from
one person in one spot.

## 8. Quality checklist before you finish

- [ ] Frame counter increased while `REC` was on for every label (not just
      `good_posture`)
- [ ] Distance was measured with a ruler/tape, not guessed
- [ ] `+`/`-` used to correct distance whenever you physically moved
- [ ] Each label matches what your body actually did (see section 6), not
      just a facial expression
- [ ] Same `person_id` used across all of your sessions
- [ ] No backlighting / no other people's faces in frame

## 9. Send back your files

Each session is saved to `data/raw/<person_id>_<session_timestamp>.csv` (one
file per session, nothing else needed — no video files exist to send).
Zip the files under `data/raw/` matching your `person_id` and send them to
**[project coordinator — fill in email or shared drive link]**.

## 10. Troubleshooting

- **"Could not open webcam"** — close any other app using the camera (Zoom,
  Teams, browser tabs with camera access) and try again.
- **Frame count stuck at 0 while recording** — the detector can't find a
  face: improve lighting, move closer to center-frame, or reduce backlight.
- **Wrong Python version errors** — MediaPipe needs Python 3.12 specifically;
  re-check `python --version` inside the activated `.venv`.

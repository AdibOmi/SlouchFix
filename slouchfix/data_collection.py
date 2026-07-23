"""Labeled data collection tool for building the training dataset described
in the pitch deck: 5-10 people, multiple monitors/lighting/distances/camera
angles, labeled good_posture / slouched / leaning_forward / too_close /
head_tilted / looking_away.

There's no ground-truth sensor for screen distance, so the person running a
session measures their own distance once (e.g. with a tape measure) and
types it in; they can nudge it with +/- if they deliberately move during the
session. Posture label is set live with number keys. Every processed frame
while "recording" is on gets appended to a per-session CSV under
`data/raw/`, keyed by person_id and session_id so training can do a
person-disjoint split later.

Controls while running:
  1-6   set posture label (see LABEL_KEYS below)
  r     toggle recording on/off
  +/-   adjust current distance_cm by 1cm
  q/Esc quit and save
"""

from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from . import config
from .capture import WebcamCapture
from .features import FEATURE_NAMES, FeatureExtractor
from .landmarks import FaceLandmarkDetector

LABEL_KEYS = {
    ord("1"): "good_posture",
    ord("2"): "slouched",
    ord("3"): "leaning_forward",
    ord("4"): "too_close",
    ord("5"): "head_tilted",
    ord("6"): "looking_away",
}

CSV_HEADER = ["timestamp", "person_id", "session_id", "label", "distance_cm", *FEATURE_NAMES]


def _open_writer(person_id: str, session_id: str) -> tuple[csv.writer, "Path", object]:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.DATA_RAW_DIR / f"{person_id}_{session_id}.csv"
    file = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(CSV_HEADER)
    return writer, out_path, file


def collect(person_id: str, initial_distance_cm: float = 50.0) -> Path:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer, out_path, file = _open_writer(person_id, session_id)

    label = "good_posture"
    distance_cm = initial_distance_cm
    recording = False
    frame_count = 0

    print(f"Recording to {out_path}")
    print("Keys: 1-6 = label, r = toggle recording, +/- = adjust distance_cm, q/Esc = quit")

    try:
        with WebcamCapture() as cam, FaceLandmarkDetector() as detector:
            extractor = FeatureExtractor()
            while True:
                frame = cam.read()
                if frame is None:
                    continue

                face = detector.process(frame)
                if face is not None and recording:
                    features = extractor.extract(face)
                    writer.writerow(
                        [
                            time.time(),
                            person_id,
                            session_id,
                            label,
                            distance_cm,
                            *features.to_vector().tolist(),
                        ]
                    )
                    frame_count += 1
                elif face is None:
                    extractor.reset()

                status = "REC" if recording else "paused"
                overlay = f"[{status}] label={label} dist={distance_cm:.0f}cm frames={frame_count}"
                cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("SlouchFix data collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key in LABEL_KEYS:
                    label = LABEL_KEYS[key]
                elif key == ord("r"):
                    recording = not recording
                elif key in (ord("+"), ord("=")):
                    distance_cm += 1
                elif key in (ord("-"), ord("_")):
                    distance_cm = max(1.0, distance_cm - 1)
                elif key == 27 or key == ord("q"):
                    break
    finally:
        file.close()
        cv2.destroyAllWindows()

    print(f"Saved {frame_count} labeled frames to {out_path}")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SlouchFix labeled data collection")
    parser.add_argument("--person", required=True, help="person id, e.g. p01 (must be consistent across that person's sessions)")
    parser.add_argument("--distance", type=float, default=50.0, help="measured starting distance from screen, in cm")
    args = parser.parse_args()

    collect(person_id=args.person, initial_distance_cm=args.distance)


if __name__ == "__main__":
    main()

"""Personal-session data collection tool: labels live webcam frames as
good/slouched via keypress, appending rows to a CSV in the same 4-angle
schema `models/dataset.py` uses for the synthetic set -- so this CSV can be
passed straight to `models/train_svm.py --with-personal` and combined with
the synthetic data in one training run, per the design spec's "both data
sources" decision.

Controls while running:
  g     set label to "good"
  s     set label to "slouched"
  r     toggle recording on/off
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
from .pose import PoseDetector
from .pose_features import FEATURE_NAMES, extract, has_required_joints

CSV_HEADER = ["timestamp", "label", *FEATURE_NAMES]


def collect(out_path: Path | None = None) -> Path:
    if out_path is None:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
        out_path = config.DATA_RAW_DIR / f"personal_session_{session_id}.csv"

    file = out_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(file)
    writer.writerow(CSV_HEADER)

    label = "good"
    recording = False
    frame_count = 0

    print(f"Recording to {out_path}")
    print("Keys: g = good, s = slouched, r = toggle recording, q/Esc = quit")

    try:
        with WebcamCapture() as cam, PoseDetector() as detector:
            while True:
                frame = cam.read()
                if frame is None:
                    continue

                pose = detector.process(frame)
                if pose is not None and has_required_joints(pose) and recording:
                    features = extract(pose)
                    writer.writerow([time.time(), label, *features.to_vector().tolist()])
                    frame_count += 1

                status = "REC" if recording else "paused"
                overlay = f"[{status}] label={label} frames={frame_count}"
                cv2.putText(frame, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.imshow("SlouchFix data collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("g"):
                    label = "good"
                elif key == ord("s"):
                    label = "slouched"
                elif key == ord("r"):
                    recording = not recording
                elif key == 27 or key == ord("q"):
                    break
    finally:
        file.close()
        cv2.destroyAllWindows()

    print(f"Saved {frame_count} labeled frames to {out_path}")
    return out_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SlouchFix personal-session data collection")
    parser.add_argument("--out", type=Path, default=None, help="output CSV path (default: a timestamped file under data/raw/)")
    args = parser.parse_args()

    collect(out_path=args.out)


if __name__ == "__main__":
    main()

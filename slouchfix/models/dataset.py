"""Builds the (features, labels) training set from the 2,000-image HF
subset, running each image through the exact same `pose.py` +
`pose_features.py` code the live app uses at inference time -- see
`pose_features.py`'s docstring for why that sharing matters.

Downloads the subset on first use via `scripts/download_hf_posture_subset.py`
if it isn't already present locally.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from .. import config
from ..pose import PoseDetector
from ..pose_features import extract, has_required_joints

IMAGES_DIR = config.DATA_RAW_DIR / "hf_posture_images"
CLASS_FOLDERS = {"good": "good", "slouched": "slouched"}


def ensure_downloaded(n_per_class: int = 1000) -> None:
    if IMAGES_DIR.exists() and all((IMAGES_DIR / folder).exists() for folder in CLASS_FOLDERS.values()):
        return
    script = Path(__file__).resolve().parent.parent.parent / "scripts" / "download_hf_posture_subset.py"
    subprocess.run([sys.executable, str(script), "--n", str(n_per_class)], check=True)


def build_dataset(images_dir: Path = IMAGES_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y): X is (N, 4) float32 angle features, y is (N,) string labels.
    Images where MoveNet detects no person, or is missing a required joint,
    are skipped -- no imputation in this path (see design spec's Error
    handling section)."""
    features: list[np.ndarray] = []
    labels: list[str] = []
    skipped = 0

    with PoseDetector() as detector:
        for label, folder_name in CLASS_FOLDERS.items():
            folder = images_dir / folder_name
            image_paths = sorted(folder.glob("*.jpg"))
            for path in image_paths:
                frame = cv2.imread(str(path))
                if frame is None:
                    skipped += 1
                    continue
                pose = detector.process(frame)
                if pose is None or not has_required_joints(pose):
                    skipped += 1
                    continue
                angle_features = extract(pose)
                features.append(angle_features.to_vector())
                labels.append(label)
            print(f"  {label}: {sum(1 for lab in labels if lab == label)} usable / {len(image_paths)} images")

    print(f"Skipped {skipped} images (no person detected or missing required joint).")
    return np.stack(features), np.array(labels)


def load_personal_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads a personal-session CSV produced by `data_collection.py`, in the
    same 4-angle-column + label schema as the synthetic dataset."""
    import csv

    features: list[list[float]] = []
    labels: list[str] = []
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(
                [
                    float(row["elbow_left_deg"]),
                    float(row["elbow_right_deg"]),
                    float(row["neck_deg"]),
                    float(row["shoulder_tilt_deg"]),
                ]
            )
            labels.append(row["label"])
    return np.array(features, dtype=np.float32), np.array(labels)

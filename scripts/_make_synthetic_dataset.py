"""Dev-only helper: generates synthetic labeled CSVs shaped exactly like
`data_collection.py` output, so `scripts/train.py` can be exercised
end-to-end before any real person's data has been collected. Not part of
the product -- delete data/raw/synth_*.csv before collecting real data.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from slouchfix import config
from slouchfix.features import FEATURE_NAMES

LABEL_PROFILES = {
    "good_posture": dict(pitch=0, yaw=0, roll=0, inter_eye=90, nose_eye=0.55, bbox_w=0.35, bbox_h=0.5, area=0.18, cy=0.5, motion=0.01, dist=50),
    "slouched": dict(pitch=5, yaw=0, roll=1, inter_eye=88, nose_eye=0.5, bbox_w=0.34, bbox_h=0.48, area=0.17, cy=0.62, motion=0.015, dist=52),
    "leaning_forward": dict(pitch=18, yaw=0, roll=0, inter_eye=110, nose_eye=0.4, bbox_w=0.45, bbox_h=0.6, area=0.27, cy=0.5, motion=0.02, dist=35),
    "too_close": dict(pitch=8, yaw=0, roll=0, inter_eye=150, nose_eye=0.5, bbox_w=0.6, bbox_h=0.75, area=0.45, cy=0.5, motion=0.02, dist=22),
    "head_tilted": dict(pitch=0, yaw=2, roll=22, inter_eye=90, nose_eye=0.55, bbox_w=0.35, bbox_h=0.5, area=0.18, cy=0.5, motion=0.01, dist=50),
    "looking_away": dict(pitch=2, yaw=40, roll=2, inter_eye=85, nose_eye=0.55, bbox_w=0.33, bbox_h=0.48, area=0.16, cy=0.5, motion=0.03, dist=50),
}


def make_rows(person_id: str, session_id: str, rng: np.random.Generator, n_per_label: int = 120) -> list[dict]:
    rows = []
    for label, p in LABEL_PROFILES.items():
        for _ in range(n_per_label):
            row = {
                "timestamp": 0.0,
                "person_id": person_id,
                "session_id": session_id,
                "label": label,
                "distance_cm": p["dist"] + rng.normal(0, 3),
                "pitch_deg": p["pitch"] + rng.normal(0, 2.5),
                "yaw_deg": p["yaw"] + rng.normal(0, 2.5),
                "roll_deg": p["roll"] + rng.normal(0, 2.0),
                "inter_eye_px_norm": (p["inter_eye"] + rng.normal(0, 4)) / config.FRAME_WIDTH,
                "nose_to_eye_ratio": p["nose_eye"] + rng.normal(0, 0.03),
                "bbox_width_frac": p["bbox_w"] + rng.normal(0, 0.02),
                "bbox_height_frac": p["bbox_h"] + rng.normal(0, 0.02),
                "bbox_area_frac": p["area"] + rng.normal(0, 0.015),
                "face_center_y_frac": p["cy"] + rng.normal(0, 0.02),
                "motion_score": max(0.0, p["motion"] + rng.normal(0, 0.005)),
            }
            assert set(FEATURE_NAMES) <= set(row.keys())
            rows.append(row)
    return rows


def main() -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i, person_id in enumerate(["synth_p01", "synth_p02", "synth_p03", "synth_p04", "synth_p05"]):
        rows = make_rows(person_id, "s1", np.random.default_rng(100 + i))
        df = pd.DataFrame(rows)
        out = config.DATA_RAW_DIR / f"{person_id}_s1.csv"
        df.to_csv(out, index=False)
        print(f"Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

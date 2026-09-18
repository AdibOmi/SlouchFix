"""Dataset generator for SlouchFix.

Generates a large, physically and ergonomically grounded posture dataset modeling
real-world programming setups across diverse subjects:
  - Configurable number of people (default: 100 people, p001 to p100).
  - 10 distinct workstation archetypes (low laptop, laptop stand, 24-27" monitor,
    34" ultrawide, dual-screen setups with left/right webcams, standing desks,
    ergonomic recliners, glasses wearers, and fidgety thinkers).
  - Anthropometric variations (inter-pupillary distance, seated height, posture tendencies).
  - Continuous behavioral timelines with sustained posture episodes, smooth
    transitions, and autoregressive (AR-1) physiological micro-movements.
  - Generates per-session CSVs in data/raw/ and consolidated datasets/dataset.csv.

Usage:
    python scripts/dataset.py
    python scripts/dataset.py --num-people 100 --sessions-per-person 2 --clean
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from slouchfix import config
from slouchfix.features import FEATURE_NAMES


@dataclass
class Persona:
    person_id: str
    name: str
    description: str
    ipd_scale: float           # Inter-pupillary distance scale (0.88 to 1.15)
    base_distance_cm: float    # Natural working distance in good posture
    base_pitch_deg: float      # Camera mount tilt bias (+: low laptop, -: high monitor)
    base_yaw_deg: float        # Camera horizontal offset bias (e.g. dual monitor)
    base_cy_frac: float        # Baseline seated face vertical center (frame fraction)
    fidget_factor: float       # Multiplier for micro-motion and movement jitter
    slouch_tendency: float     # Relative propensity for slouching when fatigued


ARCHETYPES = [
    {
        "type": "laptop_desk",
        "description": "Laptop flat on desk, low camera looking up at chin (+7° to +11° pitch)",
        "pitch_bias": (6.0, 11.0),
        "yaw_bias": (-2.0, 2.0),
        "dist_base": (42.0, 48.0),
        "cy_base": (0.49, 0.54),
        "slouch_tend": (1.0, 1.4),
        "fidget": (0.8, 1.2),
    },
    {
        "type": "laptop_stand",
        "description": "Laptop on elevated stand, eye-level camera (-2° to +3° pitch)",
        "pitch_bias": (-2.0, 3.0),
        "yaw_bias": (-2.0, 2.0),
        "dist_base": (46.0, 56.0),
        "cy_base": (0.45, 0.50),
        "slouch_tend": (0.8, 1.2),
        "fidget": (0.8, 1.2),
    },
    {
        "type": "external_monitor_24_27",
        "description": "24-27 inch external monitor, top-mounted webcam (-7° to -3° pitch)",
        "pitch_bias": (-8.0, -3.0),
        "yaw_bias": (-3.0, 3.0),
        "dist_base": (54.0, 66.0),
        "cy_base": (0.42, 0.48),
        "slouch_tend": (0.9, 1.3),
        "fidget": (0.8, 1.1),
    },
    {
        "type": "ultrawide_curved",
        "description": "34-inch ultrawide curved monitor, wide gaze angles (-8° to -4° pitch)",
        "pitch_bias": (-8.0, -4.0),
        "yaw_bias": (-4.0, 4.0),
        "dist_base": (60.0, 72.0),
        "cy_base": (0.42, 0.47),
        "slouch_tend": (0.8, 1.2),
        "fidget": (0.9, 1.3),
    },
    {
        "type": "dual_monitor_left_cam",
        "description": "Dual monitor setup, webcam on left laptop (-16° to -10° yaw bias)",
        "pitch_bias": (-4.0, 2.0),
        "yaw_bias": (-17.0, -10.0),
        "dist_base": (50.0, 60.0),
        "cy_base": (0.45, 0.51),
        "slouch_tend": (0.8, 1.2),
        "fidget": (0.9, 1.3),
    },
    {
        "type": "dual_monitor_right_cam",
        "description": "Dual monitor setup, webcam on right laptop (+10° to +16° yaw bias)",
        "pitch_bias": (-4.0, 2.0),
        "yaw_bias": (10.0, 17.0),
        "dist_base": (50.0, 60.0),
        "cy_base": (0.45, 0.51),
        "slouch_tend": (0.8, 1.2),
        "fidget": (0.9, 1.3),
    },
    {
        "type": "standing_desk",
        "description": "Standing desk setup, eye-level camera, high micro-motion (1.5x - 2.0x)",
        "pitch_bias": (-2.0, 2.0),
        "yaw_bias": (-2.0, 2.0),
        "dist_base": (52.0, 64.0),
        "cy_base": (0.44, 0.49),
        "slouch_tend": (0.4, 0.7),
        "fidget": (1.5, 2.1),
    },
    {
        "type": "ergonomic_sloucher",
        "description": "Ergonomic recliner / high-back chair, prone to deep slouching (1.5x - 2.1x)",
        "pitch_bias": (-5.0, 1.0),
        "yaw_bias": (-3.0, 3.0),
        "dist_base": (50.0, 62.0),
        "cy_base": (0.44, 0.49),
        "slouch_tend": (1.5, 2.1),
        "fidget": (0.7, 1.0),
    },
    {
        "type": "glasses_reader",
        "description": "Glasses wearer, tends to lean forward closer to screen (38-48cm)",
        "pitch_bias": (2.0, 6.0),
        "yaw_bias": (-2.0, 2.0),
        "dist_base": (40.0, 48.0),
        "cy_base": (0.47, 0.52),
        "slouch_tend": (1.1, 1.5),
        "fidget": (0.9, 1.2),
    },
    {
        "type": "fidgety_thinker",
        "description": "Frequent chin-on-hand thinker with prominent lateral head tilt",
        "pitch_bias": (-3.0, 2.0),
        "yaw_bias": (-4.0, 4.0),
        "dist_base": (48.0, 56.0),
        "cy_base": (0.47, 0.52),
        "slouch_tend": (0.9, 1.2),
        "fidget": (1.2, 1.6),
    },
]


def generate_personas(num_people: int = 100, seed: int = 2026) -> list[Persona]:
    """Generates `num_people` diverse personas sampled across real-life archetypes."""
    rng = np.random.default_rng(seed)
    personas = []
    for i in range(1, num_people + 1):
        arch = rng.choice(ARCHETYPES)
        person_id = f"p{i:03d}"
        ipd_scale = float(np.clip(rng.normal(1.0, 0.055), 0.88, 1.15))
        base_dist = float(rng.uniform(*arch["dist_base"]))
        base_pitch = float(rng.uniform(*arch["pitch_bias"]))
        base_yaw = float(rng.uniform(*arch["yaw_bias"]))
        base_cy = float(rng.uniform(*arch["cy_base"]))
        slouch_tend = float(rng.uniform(*arch["slouch_tend"]))
        fidget = float(rng.uniform(*arch["fidget"]))

        personas.append(
            Persona(
                person_id=person_id,
                name=f"Person_{i:03d}",
                description=f"{arch['description']} (IPD={ipd_scale:.2f}, base_dist={base_dist:.0f}cm)",
                ipd_scale=ipd_scale,
                base_distance_cm=base_dist,
                base_pitch_deg=base_pitch,
                base_yaw_deg=base_yaw,
                base_cy_frac=base_cy,
                fidget_factor=fidget,
                slouch_tendency=slouch_tend,
            )
        )
    return personas


class AR1NoiseGenerator:
    """Autoregressive AR(1) noise generator simulating physiological micro-tremor and drift."""

    def __init__(self, rho: float = 0.85, rng: np.random.Generator | None = None) -> None:
        self.rho = rho
        self.rng = rng or np.random.default_rng()
        self.state = 0.0

    def step(self, scale: float = 1.0) -> float:
        innov = self.rng.normal(0, scale * math.sqrt(1.0 - self.rho**2))
        self.state = self.rho * self.state + innov
        return self.state


def _smooth_step(t: float) -> float:
    """Cubic smoothstep for natural posture transitions."""
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def _generate_posture_target(
    persona: Persona,
    label: str,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Generates physical state targets for a given posture and persona."""
    base_dist = persona.base_distance_cm
    base_pitch = persona.base_pitch_deg
    base_yaw = persona.base_yaw_deg
    base_cy = persona.base_cy_frac

    if label == "good_posture":
        dist = base_dist + rng.uniform(-2.0, 2.0)
        pitch = base_pitch + rng.uniform(-2.5, 2.5)
        yaw = base_yaw + rng.uniform(-3.0, 3.0)
        roll = rng.uniform(-2.0, 2.0)
        cy = base_cy + rng.uniform(-0.015, 0.015)
        motion = 0.009 * persona.fidget_factor

    elif label == "slouched":
        dist = base_dist + rng.uniform(1.0, 5.0)
        pitch = base_pitch + rng.uniform(-2.0, 4.0)
        yaw = base_yaw + rng.uniform(-4.0, 4.0)
        roll = rng.uniform(-3.5, 3.5)
        cy = base_cy + rng.uniform(0.12, 0.18) * persona.slouch_tendency
        motion = 0.012 * persona.fidget_factor

    elif label == "leaning_forward":
        dist = rng.uniform(32.0, 42.0)
        pitch = base_pitch + rng.uniform(14.0, 22.0)
        yaw = base_yaw + rng.uniform(-4.0, 4.0)
        roll = rng.uniform(-3.0, 3.0)
        cy = base_cy + rng.uniform(0.01, 0.05)
        motion = 0.015 * persona.fidget_factor

    elif label == "too_close":
        dist = rng.uniform(20.0, 31.0)
        pitch = base_pitch + rng.uniform(6.0, 15.0)
        yaw = base_yaw + rng.uniform(-5.0, 5.0)
        roll = rng.uniform(-4.0, 4.0)
        cy = base_cy + rng.uniform(-0.02, 0.04)
        motion = 0.018 * persona.fidget_factor

    elif label == "head_tilted":
        direction = rng.choice([-1.0, 1.0])
        dist = base_dist + rng.uniform(-2.0, 3.0)
        pitch = base_pitch + rng.uniform(-3.0, 3.0)
        yaw = base_yaw + rng.uniform(-5.0, 5.0)
        roll = direction * rng.uniform(18.0, 30.0)
        cy = base_cy + rng.uniform(-0.01, 0.03)
        motion = 0.010 * persona.fidget_factor

    elif label == "looking_away":
        direction = rng.choice([-1.0, 1.0])
        dist = base_dist + rng.uniform(-3.0, 4.0)
        pitch = base_pitch + rng.uniform(-4.0, 4.0)
        yaw = base_yaw + direction * rng.uniform(32.0, 55.0)
        roll = rng.uniform(-4.0, 4.0)
        cy = base_cy + rng.uniform(-0.02, 0.02)
        motion = 0.022 * persona.fidget_factor

    else:
        raise ValueError(f"Unknown label: {label}")

    return {
        "distance_cm": dist,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "roll_deg": roll,
        "face_center_y_frac": cy,
        "base_motion": motion,
    }


def _render_frame(
    persona: Persona,
    state: dict[str, float],
    label: str,
    noise: dict[str, AR1NoiseGenerator],
    transition_factor: float = 0.0,
) -> dict[str, float | str]:
    """Applies pinhole projection optics and geometric correlations to render feature vector."""
    dist = max(15.0, state["distance_cm"] + noise["dist"].step(0.4))

    pitch = state["pitch_deg"] + noise["pitch"].step(0.6)
    yaw = state["yaw_deg"] + noise["yaw"].step(0.6)
    roll = state["roll_deg"] + noise["roll"].step(0.5)

    cos_yaw = max(0.4, math.cos(math.radians(yaw)))
    cos_pitch = max(0.6, math.cos(math.radians(pitch)))

    k_inter = 4500.0 * persona.ipd_scale
    inter_eye_px = (k_inter / dist) * cos_yaw
    inter_eye_px_norm = inter_eye_px / config.FRAME_WIDTH

    base_ratio = 0.55 - (pitch * 0.008)
    nose_to_eye_ratio = max(0.30, min(0.75, base_ratio + noise["ratio"].step(0.01)))

    bbox_width_frac = inter_eye_px_norm * 2.38 * cos_yaw
    bbox_height_frac = inter_eye_px_norm * 3.45 * cos_pitch
    bbox_width_frac = max(0.12, min(0.95, bbox_width_frac + noise["bbox_w"].step(0.005)))
    bbox_height_frac = max(0.16, min(0.98, bbox_height_frac + noise["bbox_h"].step(0.006)))
    bbox_area_frac = bbox_width_frac * bbox_height_frac

    face_center_y_frac = max(0.20, min(0.85, state["face_center_y_frac"] + noise["cy"].step(0.004)))

    motion_spike = transition_factor * 0.06
    motion_score = max(0.002, state["base_motion"] + motion_spike + abs(noise["motion"].step(0.003)))

    return {
        "distance_cm": round(dist, 1),
        "pitch_deg": round(pitch, 2),
        "yaw_deg": round(yaw, 2),
        "roll_deg": round(roll, 2),
        "inter_eye_px_norm": round(inter_eye_px_norm, 5),
        "nose_to_eye_ratio": round(nose_to_eye_ratio, 4),
        "bbox_width_frac": round(bbox_width_frac, 4),
        "bbox_height_frac": round(bbox_height_frac, 4),
        "bbox_area_frac": round(bbox_area_frac, 4),
        "face_center_y_frac": round(face_center_y_frac, 4),
        "motion_score": round(motion_score, 4),
    }


def generate_session(
    persona: Persona,
    session_id: str,
    session_type: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generates a complete session recording timeline with continuous posture episodes."""
    if session_type == "morning_alert":
        block_labels = [
            ("good_posture", int(rng.integers(80, 120))),
            ("leaning_forward", int(rng.integers(45, 75))),
            ("good_posture", int(rng.integers(70, 100))),
            ("looking_away", int(rng.integers(35, 55))),
            ("good_posture", int(rng.integers(60, 90))),
            ("head_tilted", int(rng.integers(40, 65))),
            ("good_posture", int(rng.integers(70, 100))),
            ("too_close", int(rng.integers(35, 55))),
            ("good_posture", int(rng.integers(50, 80))),
        ]
    else:
        block_labels = [
            ("good_posture", int(rng.integers(55, 85))),
            ("slouched", int(rng.integers(80, 125))),
            ("leaning_forward", int(rng.integers(55, 95))),
            ("slouched", int(rng.integers(70, 110))),
            ("head_tilted", int(rng.integers(50, 85))),
            ("good_posture", int(rng.integers(45, 75))),
            ("too_close", int(rng.integers(45, 70))),
            ("looking_away", int(rng.integers(45, 70))),
            ("slouched", int(rng.integers(65, 100))),
        ]

    noise = {
        "dist": AR1NoiseGenerator(0.88, rng),
        "pitch": AR1NoiseGenerator(0.85, rng),
        "yaw": AR1NoiseGenerator(0.85, rng),
        "roll": AR1NoiseGenerator(0.85, rng),
        "ratio": AR1NoiseGenerator(0.80, rng),
        "bbox_w": AR1NoiseGenerator(0.85, rng),
        "bbox_h": AR1NoiseGenerator(0.85, rng),
        "cy": AR1NoiseGenerator(0.90, rng),
        "motion": AR1NoiseGenerator(0.75, rng),
    }

    start_time = datetime.now() - timedelta(minutes=int(rng.integers(30, 300)))
    fps = config.TARGET_FPS  # 15 fps
    frame_dt = 1.0 / fps

    rows = []
    current_state = _generate_posture_target(persona, block_labels[0][0], rng)

    current_timestamp = start_time.timestamp()

    for idx, (target_label, duration_frames) in enumerate(block_labels):
        target_state = _generate_posture_target(persona, target_label, rng)
        transition_len = 12 if idx > 0 else 0

        for f in range(duration_frames):
            if f < transition_len and idx > 0:
                alpha = _smooth_step(f / transition_len)
                interp_state = {
                    k: (1.0 - alpha) * current_state[k] + alpha * target_state[k]
                    for k in target_state
                }
                trans_factor = math.sin(alpha * math.pi)
            else:
                interp_state = target_state
                trans_factor = 0.0

            rendered = _render_frame(
                persona=persona,
                state=interp_state,
                label=target_label,
                noise=noise,
                transition_factor=trans_factor,
            )

            row = {
                "timestamp": round(current_timestamp, 3),
                "person_id": persona.person_id,
                "session_id": session_id,
                "label": target_label,
                **rendered,
            }
            rows.append(row)
            current_timestamp += frame_dt

        current_state = target_state

    df = pd.DataFrame(rows)
    cols = ["timestamp", "person_id", "session_id", "label", "distance_cm", *FEATURE_NAMES]
    return df[cols]


def main() -> None:
    parser = argparse.ArgumentParser(description="SlouchFix dataset generator")
    parser.add_argument("--num-people", type=int, default=100, help="Number of distinct people/personas (default: 100)")
    parser.add_argument("--sessions-per-person", type=int, default=2, help="Sessions per person (default: 2)")
    parser.add_argument("--clean", action="store_true", default=True, help="Remove existing raw session CSVs before generating (default: True)")
    parser.add_argument("--no-clean", action="store_false", dest="clean", help="Keep existing raw session CSVs")
    args = parser.parse_args()

    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    personas = generate_personas(num_people=args.num_people, seed=2026)

    if args.clean:
        for old_file in config.DATA_RAW_DIR.glob("*.csv"):
            try:
                old_file.unlink()
            except OSError:
                pass

    print(f"Generating dataset for {len(personas)} people across real-world setups...")
    print(f"Sessions per person: {args.sessions_per_person}")
    print(f"Output directory: {config.DATA_RAW_DIR}\n")

    total_frames = 0
    generated_files = []

    for idx, persona in enumerate(personas):
        session_types = ["morning_alert", "afternoon_fatigue"]
        for s_idx in range(args.sessions_per_person):
            stype = session_types[s_idx % len(session_types)]
            s_name = "morning" if s_idx % 2 == 0 else "afternoon"
            s_id = f"s{s_idx+1:02d}_{s_name}_{20260918 + idx}"
            df_session = generate_session(persona, s_id, stype, rng)
            out_path = config.DATA_RAW_DIR / f"{persona.person_id}_{s_id}.csv"
            df_session.to_csv(out_path, index=False)
            total_frames += len(df_session)
            generated_files.append(out_path)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(personas):
            print(f"  Processed {idx + 1:3d}/{len(personas)} people ({total_frames:6d} frames generated so far)...")

    # Build consolidated single dataset.csv
    print("\nAssembling consolidated dataset.csv ...")
    all_dfs = [pd.read_csv(f) for f in generated_files]
    combined_df = pd.concat(all_dfs, ignore_index=True)

    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    processed_dataset_path = config.DATA_PROCESSED_DIR / "dataset.csv"
    combined_df.to_csv(processed_dataset_path, index=False)

    datasets_dir = PROJECT_ROOT / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    datasets_csv_path = datasets_dir / "dataset.csv"
    combined_df.to_csv(datasets_csv_path, index=False)

    print("\n" + "=" * 65)
    print(f"Successfully generated {len(generated_files)} session CSV files.")
    print(f"Total people: {len(personas)}")
    print(f"Total frames: {total_frames:,}")
    print(f"Consolidated dataset saved to:")
    print(f"  * {datasets_csv_path}")
    print(f"  * {processed_dataset_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()

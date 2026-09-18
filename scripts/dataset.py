"""Realistic dataset synthesis for SlouchFix.

Generates physically and ergonomically realistic session CSVs modeling real-world
programming scenarios:
  - 8 diverse personas across different hardware setups (laptops, external monitors,
    dual-screen setups, standing desks, varying anthropometrics).
  - Optics-grounded landmark scaling (inter-ocular distance vs. screen distance).
  - Continuous behavioral timelines with sustained posture blocks, realistic
    posture transitions, and autoregressive physiological micro-movements.
  - Compatible with slouchfix.models.dataset and scripts/train.py.

Usage:
    python scripts/generate_realistic_dataset.py
"""

from __future__ import annotations

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
    ipd_scale: float           # Inter-pupillary distance scale (0.90 to 1.12)
    base_distance_cm: float    # Natural working distance in good posture
    base_pitch_deg: float      # Camera mount tilt bias (+: low laptop, -: high monitor)
    base_yaw_deg: float        # Camera horizontal offset bias (e.g. dual monitor)
    base_cy_frac: float        # Baseline seated face vertical center (frame fraction)
    fidget_factor: float       # Multiplier for micro-motion and movement jitter
    slouch_tendency: float     # Relative propensity for slouching when fatigued


PERSONAS = [
    Persona(
        person_id="p01_alex",
        name="Alex",
        description="Tall developer, 27-inch 4K monitor with top-mounted webcam (-6° pitch)",
        ipd_scale=1.08,
        base_distance_cm=62.0,
        base_pitch_deg=-6.0,
        base_yaw_deg=0.0,
        base_cy_frac=0.44,
        fidget_factor=0.9,
        slouch_tendency=1.0,
    ),
    Persona(
        person_id="p02_maya",
        name="Maya",
        description="Laptop on flat desk, low camera looking up at chin (+8° pitch)",
        ipd_scale=0.94,
        base_distance_cm=46.0,
        base_pitch_deg=8.0,
        base_yaw_deg=1.0,
        base_cy_frac=0.51,
        fidget_factor=1.0,
        slouch_tendency=1.1,
    ),
    Persona(
        person_id="p03_david",
        name="David",
        description="Dual-screen setup, laptop webcam on the left looking at main monitor (-14° yaw)",
        ipd_scale=1.02,
        base_distance_cm=55.0,
        base_pitch_deg=-2.0,
        base_yaw_deg=-14.0,
        base_cy_frac=0.48,
        fidget_factor=1.1,
        slouch_tendency=0.9,
    ),
    Persona(
        person_id="p04_sarah",
        name="Sarah",
        description="Active standing desk setup with eye-level webcam, higher micro-motion",
        ipd_scale=0.98,
        base_distance_cm=58.0,
        base_pitch_deg=0.0,
        base_yaw_deg=0.0,
        base_cy_frac=0.46,
        fidget_factor=1.7,
        slouch_tendency=0.6,
    ),
    Persona(
        person_id="p05_ken",
        name="Ken",
        description="Glasses wearer with small editor fonts, prone to leaning forward and too-close",
        ipd_scale=0.92,
        base_distance_cm=44.0,
        base_pitch_deg=4.0,
        base_yaw_deg=-1.0,
        base_cy_frac=0.49,
        fidget_factor=1.0,
        slouch_tendency=1.3,
    ),
    Persona(
        person_id="p06_elena",
        name="Elena",
        description="Deep-slouch prone in ergonomic high-back chair during code reviews",
        ipd_scale=1.00,
        base_distance_cm=54.0,
        base_pitch_deg=-3.0,
        base_yaw_deg=0.0,
        base_cy_frac=0.47,
        fidget_factor=0.8,
        slouch_tendency=1.8,
    ),
    Persona(
        person_id="p07_liam",
        name="Liam",
        description="Frequent chin-on-hand thinker with prominent head tilt",
        ipd_scale=1.04,
        base_distance_cm=50.0,
        base_pitch_deg=-1.0,
        base_yaw_deg=2.0,
        base_cy_frac=0.50,
        fidget_factor=1.2,
        slouch_tendency=1.0,
    ),
    Persona(
        person_id="p08_priya",
        name="Priya",
        description="Ultrawide 34-inch curved monitor, wide gaze angles, frequent looking away",
        ipd_scale=0.96,
        base_distance_cm=65.0,
        base_pitch_deg=-4.0,
        base_yaw_deg=0.0,
        base_cy_frac=0.45,
        fidget_factor=1.1,
        slouch_tendency=0.9,
    ),
]


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
        # Sinking down in chair: face center drops dramatically, distance slightly increases
        dist = base_dist + rng.uniform(1.0, 5.0)
        pitch = base_pitch + rng.uniform(-2.0, 4.0)
        yaw = base_yaw + rng.uniform(-4.0, 4.0)
        roll = rng.uniform(-3.5, 3.5)
        cy = base_cy + rng.uniform(0.12, 0.18) * persona.slouch_tendency
        motion = 0.012 * persona.fidget_factor

    elif label == "leaning_forward":
        # Torso leans in towards screen: distance decreases, positive pitch, nose foreshortening
        dist = rng.uniform(32.0, 42.0)
        pitch = base_pitch + rng.uniform(14.0, 22.0)
        yaw = base_yaw + rng.uniform(-4.0, 4.0)
        roll = rng.uniform(-3.0, 3.0)
        cy = base_cy + rng.uniform(0.01, 0.05)
        motion = 0.015 * persona.fidget_factor

    elif label == "too_close":
        # Dangerously close to screen: <32 cm
        dist = rng.uniform(20.0, 31.0)
        pitch = base_pitch + rng.uniform(6.0, 15.0)
        yaw = base_yaw + rng.uniform(-5.0, 5.0)
        roll = rng.uniform(-4.0, 4.0)
        cy = base_cy + rng.uniform(-0.02, 0.04)
        motion = 0.018 * persona.fidget_factor

    elif label == "head_tilted":
        # Head tilted sideways (lateral neck flexion, resting chin on hand)
        direction = rng.choice([-1.0, 1.0])
        dist = base_dist + rng.uniform(-2.0, 3.0)
        pitch = base_pitch + rng.uniform(-3.0, 3.0)
        yaw = base_yaw + rng.uniform(-5.0, 5.0)
        roll = direction * rng.uniform(18.0, 30.0)
        cy = base_cy + rng.uniform(-0.01, 0.03)
        motion = 0.010 * persona.fidget_factor

    elif label == "looking_away":
        # Turned head away towards secondary screen or phone
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
    # Distance with micro-fluctuations
    dist = max(15.0, state["distance_cm"] + noise["dist"].step(0.4))

    # Head pose angles with tremor
    pitch = state["pitch_deg"] + noise["pitch"].step(0.6)
    yaw = state["yaw_deg"] + noise["yaw"].step(0.6)
    roll = state["roll_deg"] + noise["roll"].step(0.5)

    # Optical projection of inter-ocular distance
    # At 50cm with 640px width, inter_eye_px ~ 90px
    # Perspective foreshortening from yaw: cos(yaw)
    cos_yaw = max(0.4, math.cos(math.radians(yaw)))
    cos_pitch = max(0.6, math.cos(math.radians(pitch)))

    k_inter = 4500.0 * persona.ipd_scale
    inter_eye_px = (k_inter / dist) * cos_yaw
    inter_eye_px_norm = inter_eye_px / config.FRAME_WIDTH

    # Nose-to-eye ratio (foreshortened by pitch angle)
    # Pitching forward compresses vertical nose-to-eye distance
    base_ratio = 0.55 - (pitch * 0.008)
    nose_to_eye_ratio = max(0.30, min(0.75, base_ratio + noise["ratio"].step(0.01)))

    # Bounding box geometry
    bbox_width_frac = inter_eye_px_norm * 2.38 * cos_yaw
    bbox_height_frac = inter_eye_px_norm * 3.45 * cos_pitch
    bbox_width_frac = max(0.12, min(0.95, bbox_width_frac + noise["bbox_w"].step(0.005)))
    bbox_height_frac = max(0.16, min(0.98, bbox_height_frac + noise["bbox_h"].step(0.006)))
    bbox_area_frac = bbox_width_frac * bbox_height_frac

    # Face center vertical position
    face_center_y_frac = max(0.20, min(0.85, state["face_center_y_frac"] + noise["cy"].step(0.004)))

    # Motion score: base physiological micro-motion + transition boost
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
        # Morning session: high good_posture ratio, occasional leaning forward and looking away
        block_labels = [
            ("good_posture", 120),
            ("leaning_forward", 70),
            ("good_posture", 90),
            ("looking_away", 50),
            ("good_posture", 80),
            ("head_tilted", 60),
            ("good_posture", 90),
            ("too_close", 45),
            ("good_posture", 60),
        ]
    else:
        # Afternoon fatigue session: more slouched, leaning forward, head tilted
        block_labels = [
            ("good_posture", 80),
            ("slouched", 110),
            ("leaning_forward", 85),
            ("slouched", 95),
            ("head_tilted", 75),
            ("good_posture", 60),
            ("too_close", 60),
            ("looking_away", 60),
            ("slouched", 80),
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
            # Transition interpolation
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
    # Ensure correct column ordering matching slouchfix.data_collection
    cols = ["timestamp", "person_id", "session_id", "label", "distance_cm", *FEATURE_NAMES]
    return df[cols]


def main() -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)

    print(f"Synthesizing realistic dataset for {len(PERSONAS)} personas across real-world setups...")
    print(f"Output directory: {config.DATA_RAW_DIR}\n")

    total_frames = 0
    generated_files = []

    for idx, persona in enumerate(PERSONAS):
        print(f"[{idx+1}/{len(PERSONAS)}] Persona: {persona.name} ({persona.person_id})")
        print(f"    Setup: {persona.description}")

        # Session 1: Morning Focus
        s1_id = f"s01_morning_{20260918 + idx}"
        df_s1 = generate_session(persona, s1_id, "morning_alert", rng)
        out_s1 = config.DATA_RAW_DIR / f"{persona.person_id}_{s1_id}.csv"
        df_s1.to_csv(out_s1, index=False)
        total_frames += len(df_s1)
        generated_files.append(out_s1)
        print(f"    -> Session 1: {len(df_s1)} frames saved to {out_s1.name}")

        # Session 2: Afternoon Fatigue
        s2_id = f"s02_afternoon_{20260918 + idx}"
        df_s2 = generate_session(persona, s2_id, "afternoon_fatigue", rng)
        out_s2 = config.DATA_RAW_DIR / f"{persona.person_id}_{s2_id}.csv"
        df_s2.to_csv(out_s2, index=False)
        total_frames += len(df_s2)
        generated_files.append(out_s2)
        print(f"    -> Session 2: {len(df_s2)} frames saved to {out_s2.name}")

    print("\n" + "=" * 60)
    print(f"Successfully generated {len(generated_files)} CSV files ({total_frames} frames total).")
    print("Personas covered:")
    for p in PERSONAS:
        print(f"  * {p.person_id:12s} - {p.name}: {p.description}")
    print("=" * 60)


if __name__ == "__main__":
    main()

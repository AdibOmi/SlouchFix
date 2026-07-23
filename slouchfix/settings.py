"""User-adjustable settings, persisted to ``~/.slouchfix/settings.json`` so
the dashboard's Settings tab can change thresholds/behavior without editing
`config.py`. Anything not listed here (camera resolution, target FPS,
landmark indices) is a fixed constant, not meant to be end-user tunable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields

from . import config

SETTINGS_PATH = config.APP_DIR / "settings.json"


@dataclass
class Settings:
    too_close_distance_cm: float = config.TOO_CLOSE_DISTANCE_CM
    yaw_looking_away_deg: float = config.YAW_LOOKING_AWAY_DEG
    roll_head_tilted_deg: float = config.ROLL_HEAD_TILTED_DEG
    pitch_leaning_forward_deg: float = config.PITCH_LEANING_FORWARD_DEG
    slouch_face_drop_ratio: float = config.SLOUCH_FACE_DROP_RATIO
    confirm_frames: int = config.STATE_CONFIRM_FRAMES
    notification_cooldown_sec: float = config.NOTIFICATION_COOLDOWN_SEC
    good_posture_reminder_sec: float = config.GOOD_POSTURE_REMINDER_SEC
    notifications_enabled: bool = True
    camera_index: int = config.CAMERA_INDEX

    def save(self) -> None:
        config.APP_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls) -> "Settings":
        if not SETTINGS_PATH.exists():
            return cls()
        data = json.loads(SETTINGS_PATH.read_text())
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

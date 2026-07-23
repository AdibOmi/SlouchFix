"""Debounces raw per-frame posture readings into a stable state with a
duration, so a single noisy frame doesn't flip the label or fire an alert.
Matches the "duration of poor posture" output field from the pitch deck.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from . import config
from .baseline import PostureReading


@dataclass
class TrackedState:
    label: str
    confidence: float
    distance_cm: float
    duration_sec: float
    just_changed: bool


class PostureTracker:
    def __init__(self, confirm_frames: int = config.STATE_CONFIRM_FRAMES) -> None:
        self.confirm_frames = confirm_frames
        self._pending_label: str | None = None
        self._pending_count = 0
        self._confirmed_label: str | None = None
        self._confirmed_confidence = 0.0
        self._confirmed_since = time.monotonic()

    def update(self, reading: PostureReading) -> TrackedState:
        now = time.monotonic()

        if reading.label == self._pending_label:
            self._pending_count += 1
        else:
            self._pending_label = reading.label
            self._pending_count = 1

        just_changed = False
        if self._pending_count >= self.confirm_frames and reading.label != self._confirmed_label:
            self._confirmed_label = reading.label
            self._confirmed_since = now
            just_changed = True

        self._confirmed_confidence = reading.confidence
        duration = now - self._confirmed_since

        return TrackedState(
            label=self._confirmed_label or reading.label,
            confidence=self._confirmed_confidence,
            distance_cm=reading.distance_cm,
            duration_sec=duration,
            just_changed=just_changed,
        )

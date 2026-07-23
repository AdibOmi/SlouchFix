"""Usage history logging: appends confirmed posture-state changes and sent
alerts to CSV files so the dashboard's History/Stats tabs have something to
read. This is the app's own usage log, separate from the labeled training
recordings in `data/raw/` (those come from `data_collection.py` and are for
model training, not for showing the user their own history).
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import config
from .tracker import TrackedState

HISTORY_DIR = config.PROJECT_ROOT / "data" / "history"
STATES_PATH = HISTORY_DIR / "states.csv"
ALERTS_PATH = HISTORY_DIR / "alerts.csv"

STATE_FIELDS = ["timestamp", "label", "confidence", "distance_cm"]
ALERT_FIELDS = ["timestamp", "label", "title", "message"]


class HistoryLogger:
    def __init__(self, states_path: Path = STATES_PATH, alerts_path: Path = ALERTS_PATH) -> None:
        self.states_path = states_path
        self.alerts_path = alerts_path
        self.states_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_header(self.states_path, STATE_FIELDS)
        self._ensure_header(self.alerts_path, ALERT_FIELDS)

    @staticmethod
    def _ensure_header(path: Path, fields: list[str]) -> None:
        if not path.exists():
            with path.open("w", newline="") as f:
                csv.writer(f).writerow(fields)

    def log_state(self, state: TrackedState) -> None:
        with self.states_path.open("a", newline="") as f:
            csv.writer(f).writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    state.label,
                    f"{state.confidence:.3f}",
                    f"{state.distance_cm:.1f}",
                ]
            )

    def log_alert(self, label: str, title: str, message: str) -> None:
        with self.alerts_path.open("a", newline="") as f:
            csv.writer(f).writerow([datetime.now().isoformat(timespec="seconds"), label, title, message])

    def load_states(self, since: datetime | None = None) -> list[dict]:
        return self._load(self.states_path, since)

    def load_alerts(self, since: datetime | None = None) -> list[dict]:
        return self._load(self.alerts_path, since)

    @staticmethod
    def _load(path: Path, since: datetime | None) -> list[dict]:
        if not path.exists():
            return []
        rows = []
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                if since is not None and datetime.fromisoformat(row["timestamp"]) < since:
                    continue
                rows.append(row)
        return rows


@dataclass
class PeriodStats:
    tracked_minutes: float
    label_seconds: dict[str, float]
    alert_counts: Counter


def compute_label_durations(states: list[dict]) -> dict[str, float]:
    """Approximate seconds spent in each label from consecutive state-change
    timestamps: the gap between one state's start and the next one's start is
    that label's duration. The final still-open interval is not counted."""
    durations: dict[str, float] = {}
    for i in range(len(states) - 1):
        t0 = datetime.fromisoformat(states[i]["timestamp"])
        t1 = datetime.fromisoformat(states[i + 1]["timestamp"])
        label = states[i]["label"]
        durations[label] = durations.get(label, 0.0) + (t1 - t0).total_seconds()
    return durations


def compute_period_stats(history: HistoryLogger, since: datetime | None = None) -> PeriodStats:
    states = history.load_states(since=since)
    alerts = history.load_alerts(since=since)
    label_seconds = compute_label_durations(states)
    return PeriodStats(
        tracked_minutes=sum(label_seconds.values()) / 60.0,
        label_seconds=label_seconds,
        alert_counts=Counter(row["label"] for row in alerts),
    )

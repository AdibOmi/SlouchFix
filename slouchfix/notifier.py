"""Non-intrusive desktop notifications.

Two rules keep this from becoming nag-ware: a per-label cooldown (don't
repeat the same complaint more than once every `NOTIFICATION_COOLDOWN_SEC`),
and alerts only fire once a state has been confirmed by the PostureTracker's
debounce window, so a single bad frame never triggers a toast.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from .settings import Settings
from .tracker import TrackedState

if TYPE_CHECKING:
    from .history import HistoryLogger

MESSAGES = {
    "too_close": lambda s, settings: (
        "Too Close",
        f"Move back approximately {max(settings.too_close_distance_cm - s.distance_cm, 1):.0f} cm",
    ),
    "leaning_forward": lambda s, settings: (
        "Leaning Forward",
        f"You have been leaning for {s.duration_sec:.0f} seconds",
    ),
    "slouched": lambda s, settings: (
        "Slouched",
        f"You have been slouching for {s.duration_sec:.0f} seconds",
    ),
    "head_tilted": lambda s, settings: ("Head Tilted", "Straighten your head and neck"),
    "looking_away": lambda s, settings: ("Looking Away", "Face not centered on screen"),
}


class Notifier:
    def __init__(
        self,
        app_name: str = "SlouchFix",
        enabled: bool = True,
        settings: Settings | None = None,
        history: "HistoryLogger | None" = None,
    ) -> None:
        self.app_name = app_name
        self.settings = settings or Settings.load()
        self.enabled = enabled and self.settings.notifications_enabled
        self.history = history
        self._last_notified: dict[str, float] = {}
        self._last_good_reminder = 0.0
        self._backend_warned = False

    def _send(self, title: str, message: str, label: str = "") -> None:
        if not self.enabled:
            return
        try:
            from plyer import notification

            notification.notify(title=title, message=message, app_name=self.app_name, timeout=6)
        except Exception as exc:  # pragma: no cover - depends on OS notification backend
            if not self._backend_warned:
                print(f"[SlouchFix] Desktop notifications unavailable ({exc}); printing instead.")
                self._backend_warned = True
            print(f"[SlouchFix] {title}: {message}")

        if self.history is not None:
            self.history.log_alert(label, title, message)

    def maybe_notify(self, state: TrackedState, session_minutes: float) -> None:
        now = time.monotonic()

        if state.label == "good_posture":
            if now - self._last_good_reminder >= self.settings.good_posture_reminder_sec:
                self._last_good_reminder = now
                self._send(
                    "Good Posture",
                    f"Distance: {state.distance_cm:.0f} cm | Focus session: {session_minutes:.0f} min",
                    label="good_posture",
                )
            return

        builder = MESSAGES.get(state.label)
        if builder is None:
            return

        last = self._last_notified.get(state.label, 0.0)
        if now - last < self.settings.notification_cooldown_sec:
            return

        self._last_notified[state.label] = now
        title, message = builder(state, self.settings)
        self._send(title, message, label=state.label)

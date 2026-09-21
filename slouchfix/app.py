"""Main real-time loop: capture -> pose -> pose_features -> inference ->
tracker -> notifier. This is the console/headless entry point; `tray.py`
wraps this in a background thread with a system tray icon for normal use.
"""

from __future__ import annotations

import argparse
import time

from . import config
from .capture import WebcamCapture
from .history import HistoryLogger
from .inference import PostureEngine
from .notifier import Notifier
from .pose import PoseDetector
from .pose_features import extract, has_required_joints
from .settings import Settings
from .tracker import PostureTracker, TrackedState


class SlouchFixApp:
    def __init__(
        self,
        show_preview: bool = False,
        notifications: bool = True,
        settings: Settings | None = None,
        history: HistoryLogger | None = None,
    ) -> None:
        self.show_preview = show_preview
        self.settings = settings or Settings.load()
        self.history = history or HistoryLogger()
        self.engine = PostureEngine()
        self.tracker = PostureTracker(confirm_frames=self.settings.confirm_frames)
        self.notifier = Notifier(enabled=notifications, settings=self.settings, history=self.history)
        self.paused = False
        self._stop = False
        self._session_start = time.monotonic()
        self.state: TrackedState | None = None
        self.latest_frame = None

    @property
    def session_seconds(self) -> float:
        return time.monotonic() - self._session_start

    def stop(self) -> None:
        self._stop = True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def run(self) -> None:
        with WebcamCapture(camera_index=self.settings.camera_index) as cam, PoseDetector() as detector:
            print("SlouchFix running. Press Ctrl+C (or Esc in preview) to stop.")
            frame_interval = 1.0 / config.TARGET_FPS

            while not self._stop:
                loop_start = time.monotonic()
                frame = cam.read()
                if frame is None:
                    continue
                self.latest_frame = frame

                if self.paused:
                    if self.show_preview:
                        self._show(frame, banner="PAUSED")
                    time.sleep(frame_interval)
                    continue

                pose = detector.process(frame)
                if pose is None or not has_required_joints(pose):
                    if self.show_preview:
                        self._show(frame, banner="no person detected")
                    time.sleep(frame_interval)
                    continue

                features = extract(pose)
                reading = self.engine.classify(features)
                state = self.tracker.update(reading)
                self.state = state
                if state.just_changed:
                    self.history.log_state(state)
                session_minutes = (time.monotonic() - self._session_start) / 60.0
                self.notifier.maybe_notify(state, session_minutes)

                if self.show_preview:
                    banner = f"{state.label} ({state.confidence:.2f}) dist={state.distance_cm:.0f}cm"
                    self._show(frame, banner=banner, label_for_screenshot=state.label)

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, frame_interval - elapsed))

        if self.show_preview:
            import cv2

            cv2.destroyAllWindows()

    def _show(self, frame, banner: str, label_for_screenshot: str | None = None) -> None:
        import cv2

        cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SlouchFix", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            self._stop = True
        elif key == ord("s") and label_for_screenshot is not None:
            self._save_qualitative_screenshot(frame, label_for_screenshot)

    def _save_qualitative_screenshot(self, frame, label: str) -> None:
        """Saves the current preview frame under reports/qualitative/ -- the
        easiest way to produce the "3 to 5 qualitative screenshots" the
        evaluation report asks for: run with --preview, hold the posture you
        want to illustrate, and press 's'."""
        import cv2

        out_dir = config.PROJECT_ROOT / "reports" / "qualitative"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(str(out_path), frame)
        print(f"[SlouchFix] Saved qualitative screenshot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SlouchFix -- posture monitor")
    parser.add_argument("--preview", action="store_true", help="show a debug camera preview window")
    parser.add_argument(
        "--no-notifications", action="store_true", help="disable desktop notifications (console-only)"
    )
    args = parser.parse_args()

    app = SlouchFixApp(show_preview=args.preview, notifications=not args.no_notifications)
    try:
        app.run()
    except KeyboardInterrupt:
        print("SlouchFix stopped.")


if __name__ == "__main__":
    main()

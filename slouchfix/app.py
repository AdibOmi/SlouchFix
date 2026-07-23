"""Main real-time loop: capture -> landmarks -> features -> inference ->
tracker -> notifier. This is the console/headless entry point; `tray.py`
wraps this in a background thread with a system tray icon for normal use.
"""

from __future__ import annotations

import argparse
import time

from . import config
from .calibration import get_or_run_calibration, run_calibration
from .capture import WebcamCapture
from .features import FeatureExtractor
from .history import HistoryLogger
from .inference import PostureEngine
from .landmarks import FaceLandmarkDetector
from .notifier import Notifier
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
        self.calibration = get_or_run_calibration()
        self.engine = PostureEngine(self.settings)
        self.tracker = PostureTracker(confirm_frames=self.settings.confirm_frames)
        self.notifier = Notifier(enabled=notifications, settings=self.settings, history=self.history)
        self.paused = False
        self._stop = False
        self._session_start = time.monotonic()
        self.state: TrackedState | None = None

    @property
    def using_trained_model(self) -> bool:
        return self.engine.using_trained_model

    @property
    def session_seconds(self) -> float:
        return time.monotonic() - self._session_start

    def stop(self) -> None:
        self._stop = True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def recalibrate(self) -> None:
        was_paused = self.paused
        self.paused = True
        try:
            self.calibration = run_calibration()
        finally:
            self.paused = was_paused

    def run(self) -> None:
        with WebcamCapture(camera_index=self.settings.camera_index) as cam, FaceLandmarkDetector() as detector:
            extractor = FeatureExtractor()
            print("SlouchFix running. Press Ctrl+C (or Esc in preview) to stop.")
            frame_interval = 1.0 / config.TARGET_FPS

            while not self._stop:
                loop_start = time.monotonic()
                frame = cam.read()
                if frame is None:
                    continue

                if self.paused:
                    if self.show_preview:
                        self._show(frame, banner="PAUSED")
                    time.sleep(frame_interval)
                    continue

                face = detector.process(frame)
                if face is None:
                    extractor.reset()
                    if self.show_preview:
                        self._show(frame, banner="no face detected")
                    time.sleep(frame_interval)
                    continue

                features = extractor.extract(face)
                reading = self.engine.classify(features, self.calibration)
                state = self.tracker.update(reading)
                self.state = state
                if state.just_changed:
                    self.history.log_state(state)
                session_minutes = (time.monotonic() - self._session_start) / 60.0
                self.notifier.maybe_notify(state, session_minutes)

                if self.show_preview:
                    banner = f"{state.label} ({state.confidence:.2f}) dist={state.distance_cm:.0f}cm"
                    self._show(frame, banner=banner)

                elapsed = time.monotonic() - loop_start
                time.sleep(max(0.0, frame_interval - elapsed))

        if self.show_preview:
            import cv2

            cv2.destroyAllWindows()

    def _show(self, frame, banner: str) -> None:
        import cv2

        cv2.putText(frame, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("SlouchFix", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            self._stop = True


def main() -> None:
    parser = argparse.ArgumentParser(description="SlouchFix -- posture and screen-distance monitor")
    parser.add_argument("--preview", action="store_true", help="show a debug camera preview window")
    parser.add_argument(
        "--no-notifications", action="store_true", help="disable desktop notifications (console-only)"
    )
    parser.add_argument("--recalibrate", action="store_true", help="force a fresh calibration before starting")
    args = parser.parse_args()

    if args.recalibrate:
        run_calibration()

    app = SlouchFixApp(show_preview=args.preview, notifications=not args.no_notifications)
    try:
        app.run()
    except KeyboardInterrupt:
        print("SlouchFix stopped.")


if __name__ == "__main__":
    main()

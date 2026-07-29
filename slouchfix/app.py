"""Main real-time loop: capture -> landmarks -> features -> inference ->
tracker -> notifier. This is the console/headless entry point; `tray.py`
wraps this in a background thread with a system tray icon for normal use.
"""

from __future__ import annotations

import argparse
import time

from . import config
from .calibration import (
    CALIBRATION_SECONDS,
    DEFAULT_CALIBRATION_DISTANCE_CM,
    RecalibrationMonitor,
    _build_calibration,
    get_or_run_calibration,
    run_calibration,
)
from .capture import WebcamCapture
from .features import FeatureExtractor, FrameFeatures
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
        self.drift_monitor = RecalibrationMonitor()
        self.notifier = Notifier(enabled=notifications, settings=self.settings, history=self.history)
        self.paused = False
        self._stop = False
        self._session_start = time.monotonic()
        self.state: TrackedState | None = None
        self.latest_frame = None
        self.calibration_stale = False

        self._recalibrating = False
        self._recal_samples: list[FrameFeatures] = []
        self._recal_start = 0.0
        self._recal_seconds = 0.0
        self._recal_distance_cm = 0.0
        self.recalibration_error: str | None = None

    @property
    def using_trained_model(self) -> bool:
        return self.engine.using_trained_model

    @property
    def session_seconds(self) -> float:
        return time.monotonic() - self._session_start

    @property
    def recalibrating(self) -> bool:
        return self._recalibrating

    @property
    def recalibration_progress(self) -> float:
        if not self._recalibrating or self._recal_seconds <= 0:
            return 0.0
        return min(1.0, (time.monotonic() - self._recal_start) / self._recal_seconds)

    def stop(self) -> None:
        self._stop = True

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def recalibrate(self) -> None:
        was_paused = self.paused
        self.paused = True
        try:
            self.calibration = run_calibration()
            self.drift_monitor.reset()
        finally:
            self.paused = was_paused

    def start_recalibration(
        self,
        distance_cm: float = DEFAULT_CALIBRATION_DISTANCE_CM,
        seconds: float = CALIBRATION_SECONDS,
    ) -> None:
        """Non-blocking recalibration for callers (the local server, for the
        Flutter UI) that can't tolerate `recalibrate()`'s native cv2 window:
        samples are collected from frames `run()` is already reading, over
        the next `seconds`, then folded into a new baseline in place.
        Poll `recalibrating`/`recalibration_progress` for UI feedback."""
        self._recal_samples = []
        self._recal_start = time.monotonic()
        self._recal_seconds = seconds
        self._recal_distance_cm = distance_cm
        self.recalibration_error = None
        self._recalibrating = True

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
                self.latest_frame = frame

                if self.paused and not self._recalibrating:
                    if self.show_preview:
                        self._show(frame, banner="PAUSED")
                    time.sleep(frame_interval)
                    continue

                face = detector.process(frame)
                if face is None:
                    extractor.reset()
                    self.drift_monitor.reset()
                    if self.show_preview:
                        self._show(frame, banner="no face detected")
                    time.sleep(frame_interval)
                    continue

                features = extractor.extract(face)

                if self._recalibrating:
                    self._recal_samples.append(features)
                    if time.monotonic() - self._recal_start >= self._recal_seconds:
                        try:
                            self.calibration = _build_calibration(self._recal_samples, self._recal_distance_cm)
                            self.calibration.save()
                            self.drift_monitor.reset()
                            self.calibration_stale = False
                        except RuntimeError as exc:
                            self.recalibration_error = str(exc)
                        self._recalibrating = False
                    if self.show_preview:
                        self._show(frame, banner=f"Recalibrating... {self.recalibration_progress:.0%}")
                    time.sleep(frame_interval)
                    continue

                if self.drift_monitor.update(features):
                    self.calibration_stale = True
                    print(
                        "[SlouchFix] Large frame-to-frame landmark jump detected -- "
                        "calibration may be stale (camera/laptop moved?). Consider --recalibrate."
                    )
                reading = self.engine.classify(features, self.calibration)
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

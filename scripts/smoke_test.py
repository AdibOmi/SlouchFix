"""Sanity check without a webcam: exercises feature math, the rule-based
baseline, tracker debouncing, and notifier plumbing using synthetic
landmark data shaped like real FaceLandmarker output.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from slouchfix import config
from slouchfix.baseline import classify
from slouchfix.calibration import Calibration
from slouchfix.features import FeatureExtractor
from slouchfix.landmarks import NUM_LANDMARKS, FaceResult
from slouchfix.notifier import Notifier
from slouchfix.tracker import PostureTracker


def make_face(frame_w=640, frame_h=480, jitter=0.0, y_offset=0.0, x_offset=0.0) -> FaceResult:
    rng = np.random.default_rng(0)
    pts = rng.normal(loc=(frame_w / 2, frame_h / 2), scale=(frame_w * 0.1, frame_h * 0.12), size=(NUM_LANDMARKS, 2))
    pts += np.array([x_offset, y_offset])
    pts += rng.normal(scale=jitter, size=pts.shape)

    # Pin the specific landmarks features.py / config.py rely on to plausible
    # face-like positions so solvePnP and the ratios behave sensibly.
    def set_point(idx, x, y):
        pts[idx] = (frame_w / 2 + x + x_offset, frame_h / 2 + y + y_offset)

    set_point(config.NOSE_TIP, 0, 10)
    set_point(config.CHIN, 0, 70)
    set_point(config.LEFT_EYE_OUTER, -45, -20)
    set_point(config.RIGHT_EYE_OUTER, 45, -20)
    set_point(config.LEFT_MOUTH_CORNER, -25, 45)
    set_point(config.RIGHT_MOUTH_CORNER, 25, 45)
    set_point(config.LEFT_EYE_INNER, -15, -20)
    set_point(config.RIGHT_EYE_INNER, 15, -20)
    set_point(config.FOREHEAD, 0, -60)

    return FaceResult(points_px=pts.astype(np.float32), frame_width=frame_w, frame_height=frame_h)


def main() -> None:
    extractor = FeatureExtractor()

    print("1) Feature extraction on a neutral synthetic face ...")
    face = make_face()
    features = extractor.extract(face)
    print(f"   pitch={features.pitch_deg:.1f} yaw={features.yaw_deg:.1f} roll={features.roll_deg:.1f} "
          f"inter_eye_px={features.inter_eye_px:.1f}")
    assert features.to_vector().shape == (10,)

    print("2) Calibration from that neutral face at 50cm ...")
    calib = Calibration(
        pixel_to_cm_k=50.0 * features.inter_eye_px,
        baseline_pitch_deg=features.pitch_deg,
        baseline_yaw_deg=features.yaw_deg,
        baseline_roll_deg=features.roll_deg,
        baseline_nose_to_eye_ratio=features.nose_to_eye_ratio,
        baseline_face_center_y_frac=features.face_center_y_frac,
        calibration_distance_cm=50.0,
    )
    reading = classify(features, calib)
    print(f"   classify(neutral) -> {reading.label} conf={reading.confidence:.2f} dist={reading.distance_cm:.1f}cm")
    assert reading.label == "good_posture"
    assert abs(reading.distance_cm - 50.0) < 1.0

    print("3) A face that moved closer to the camera (bigger inter-eye distance) ...")
    extractor.reset()
    closer_face = make_face()
    closer_face.points_px = (closer_face.points_px - [320, 240]) * 1.6 + [320, 240]
    closer_features = extractor.extract(closer_face)
    closer_reading = classify(closer_features, calib)
    print(f"   classify(closer) -> {closer_reading.label} conf={closer_reading.confidence:.2f} dist={closer_reading.distance_cm:.1f}cm")
    assert closer_reading.label == "too_close"
    assert closer_reading.distance_cm < 40.0

    print("4) A head-tilted (rolled) face ...")
    extractor.reset()
    tilted_face = make_face()
    theta = np.radians(25)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    center = np.array([320, 240])
    tilted_face.points_px = ((tilted_face.points_px - center) @ rot.T + center).astype(np.float32)
    tilted_features = extractor.extract(tilted_face)
    tilted_reading = classify(tilted_features, calib)
    print(f"   classify(tilted) -> {tilted_reading.label} conf={tilted_reading.confidence:.2f} roll={tilted_features.roll_deg:.1f}")
    assert tilted_reading.label == "head_tilted"

    print("5) PostureTracker debouncing (needs config.STATE_CONFIRM_FRAMES consecutive frames) ...")
    tracker = PostureTracker(confirm_frames=3)
    from slouchfix.baseline import PostureReading

    r1 = tracker.update(PostureReading("slouched", 0.9, 55.0))
    assert r1.label != "slouched" or not r1.just_changed  # not confirmed yet on frame 1
    tracker.update(PostureReading("slouched", 0.9, 55.0))
    r3 = tracker.update(PostureReading("slouched", 0.9, 55.0))
    assert r3.label == "slouched" and r3.just_changed
    print(f"   confirmed after 3 frames -> {r3.label} duration={r3.duration_sec:.3f}s")

    print("6) Notifier plumbing (disabled backend, should not raise) ...")
    notifier = Notifier(enabled=False)
    notifier.maybe_notify(r3, session_minutes=5.0)

    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()

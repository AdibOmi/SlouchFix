"""Central configuration: paths, camera settings, landmark indices, thresholds."""

from pathlib import Path

# --- Paths -------------------------------------------------------------

APP_DIR = Path.home() / ".slouchfix"
CALIBRATION_PATH = APP_DIR / "calibration.json"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# --- Camera --------------------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15

# --- MediaPipe FaceMesh landmark indices ---------------------------------
# Canonical subset used for solvePnP head-pose estimation (standard mapping
# for the 468-point MediaPipe FaceMesh topology).
NOSE_TIP = 4
CHIN = 152
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_MOUTH_CORNER = 61
RIGHT_MOUTH_CORNER = 291

# Extra points used directly in feature engineering.
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362
FOREHEAD = 10

POSE_LANDMARK_IDS = [NOSE_TIP, CHIN, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, LEFT_MOUTH_CORNER, RIGHT_MOUTH_CORNER]

# 3D canonical face model points (mm, arbitrary consistent scale) matched
# 1:1 with POSE_LANDMARK_IDS above, in the same order. Standard values used
# widely for MediaPipe/OpenCV solvePnP head-pose demos.
MODEL_POINTS_3D = [
    (0.0, 0.0, 0.0),          # nose tip
    (0.0, -63.6, -12.5),      # chin
    (-43.3, 32.7, -26.0),     # left eye outer corner
    (43.3, 32.7, -26.0),      # right eye outer corner
    (-28.9, -28.9, -24.1),    # left mouth corner
    (28.9, -28.9, -24.1),     # right mouth corner
]

# Landmarks tracked frame-to-frame for the "movement over time" feature.
MOTION_LANDMARK_IDS = [NOSE_TIP, LEFT_EYE_OUTER, RIGHT_EYE_OUTER, FOREHEAD]
MOTION_WINDOW_FRAMES = 10  # ~0.6-1s of history at target FPS

# --- Posture labels --------------------------------------------------------

LABELS = [
    "good_posture",
    "slouched",
    "leaning_forward",
    "too_close",
    "head_tilted",
    "looking_away",
]

# --- Rule-based baseline thresholds (relative to calibration) -----------

TOO_CLOSE_DISTANCE_CM = 40.0
YAW_LOOKING_AWAY_DEG = 30.0
ROLL_HEAD_TILTED_DEG = 15.0
PITCH_LEANING_FORWARD_DEG = 12.0   # forward pitch beyond calibrated baseline
SLOUCH_FACE_DROP_RATIO = 0.12      # fraction of frame height the face center may sink before "slouched"

# --- Temporal smoothing / non-intrusiveness ------------------------------

STATE_CONFIRM_FRAMES = 8           # consecutive frames needed before a state change is accepted
NOTIFICATION_COOLDOWN_SEC = 45.0   # minimum gap between two notifications for the same label
GOOD_POSTURE_REMINDER_SEC = 900.0  # occasional positive reinforcement, not just nagging

"""Central configuration: paths, camera settings, pose-keypoint indices."""

from pathlib import Path

# --- Paths -------------------------------------------------------------

APP_DIR = Path.home() / ".slouchfix"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# --- Camera --------------------------------------------------------------

CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 15

# --- MoveNet (COCO-17) keypoint indices -----------------------------------
# Standard MoveNet/COCO ordering: nose, left/right eye, left/right ear,
# left/right shoulder, left/right elbow, left/right wrist, left/right hip,
# left/right knee, left/right ankle.
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

# Joints required to compute the 4-d angle feature vector (see
# pose_features.py). A frame missing any of these below MIN_KEYPOINT_SCORE
# confidence is skipped rather than imputed.
REQUIRED_ANGLE_JOINTS = [
    NOSE,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    RIGHT_ELBOW,
    LEFT_WRIST,
    RIGHT_WRIST,
]
MIN_KEYPOINT_SCORE = 0.3

# Assumed average shoulder width (cm), used only for the informational,
# non-ML distance estimate in pose_features.py -- deliberately approximate,
# same "population-average" pattern the old naive_pose_baseline.py used for
# interpupillary distance.
AVG_SHOULDER_WIDTH_CM = 40.0

# --- Posture labels --------------------------------------------------------

LABELS = ["good", "slouched"]

# --- Temporal smoothing / non-intrusiveness ------------------------------

STATE_CONFIRM_FRAMES = 8           # consecutive frames needed before a state change is accepted
NOTIFICATION_COOLDOWN_SEC = 45.0   # minimum gap between two notifications for the same label
GOOD_POSTURE_REMINDER_SEC = 900.0  # occasional positive reinforcement, not just nagging

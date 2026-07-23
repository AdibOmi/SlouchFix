# SlouchFix — Technology & Algorithm Stack

Adapted from the original `SlouchFix_TechStack.docx` (which targeted a Flutter
mobile demo). SlouchFix is now a **Windows/PC desktop app**, so Part 2 has
been replaced with a desktop-native stack. Part 1 (the ML/Pattern Recognition
core) is unchanged — it's the graded part and the runtime target doesn't
affect it.

## Part 1: ML / Pattern Recognition

### Course Topics Applied

| Course Topic | How it is used in SlouchFix |
|---|---|
| Linear classifiers | Baseline model — logistic regression on raw landmark distances, to beat before justifying the neural approach |
| Optimization | Loss design: cross-entropy for posture class, MSE/Huber loss for distance regression; Adam optimizer with LR scheduling |
| Neural networks | Core MLP/small feed-forward net mapping landmark features to posture class + distance |
| Backpropagation | Training the network end-to-end; used to explain/justify architecture choices in ablations |
| Image classification | Framing posture detection as multi-class classification (good / slouched / leaning forward / too close / head tilted / looking away) |
| CNNs & CNN architectures | Optional stage using raw face-crop images instead of/alongside landmarks (e.g. a small MobileNet-style backbone, or distillation target) |

### Languages & Core Libraries

- **Python 3.12** — all model development and the shipped app run on the same interpreter (simplifies deployment — no Python/Flutter split)
- **PyTorch** — model definition and training loop
- **MediaPipe (Python) FaceMesh** — pretrained landmark extraction, same library used live in the app
- **scikit-learn** — baseline linear classifier, train/val/test splitting utilities, metrics (F1, confusion matrix)
- **NumPy / Pandas** — feature engineering (pixel distances, angles) and dataset bookkeeping
- **OpenCV** — video capture, frame preprocessing, calibration overlays
- **Matplotlib / Seaborn** — qualitative analysis plots, ablation charts

### Model Export for On-Device Use

- **ONNX** — conversion path from the trained PyTorch model to `onnxruntime` for fast CPU inference in the desktop app (no PyTorch dependency needed at runtime)
- Since training and inference both run on the same PC/Python stack, there is no
  second mobile-runtime conversion step (no TFLite, no MediaPipe Tasks
  Android/iOS bindings) — train in PyTorch, export once to ONNX, run with
  `onnxruntime` in the app.

### Experiment Tracking

- Jupyter / notebooks — training notebooks
- TensorBoard (optional) — logging loss curves and ablation runs

### Metrics

- Accuracy, macro-F1 — per posture class, since classes are imbalanced
- MAE (centimeters) — for distance regression
- Inference latency (ms) / FPS — measured on-device (the actual PC target), not just in a notebook
- False-alert rate — non-intrusive-notification requirement from the pitch

## Part 2: Desktop App (Windows/PC)

Replaces the original Flutter mobile demo. Same webcam-in, notification-out
loop, but native to the PC the user codes on — no phone, no second camera
angle, matches the pitch's actual use case ("during coding sessions").

| Concern | Stack |
|---|---|
| Language & runtime | Python 3.12 (one interpreter for training and the shipped app) |
| Camera & real-time pipeline | OpenCV (`cv2.VideoCapture`) — webcam frame capture and preprocessing |
| Landmark extraction | MediaPipe FaceMesh (Python) — identical to the training-side extractor, so there is no train/deploy feature skew |
| On-device inference | `onnxruntime` (CPU) running the exported model; rule-based threshold baseline runs with no model file at all, so the app is usable before any training happens |
| State management | Plain Python state machine (`PostureTracker`) tracking current label, confidence, and duration-in-state — no UI framework needed since there's no visible window most of the time |
| Local storage (calibration & session data) | JSON file (`~/.slouchfix/calibration.json`) for the one-time pixel-to-cm calibration; CSV under `data/` for logged posture history |
| UI/UX | System tray icon (`pystray`) for background operation; native Windows toast notifications (`plyer` / `win11toast`) for non-intrusive nudges — matches "Non-intrusive desktop notifications" from the pitch directly, more so than a phone push |
| Packaging (future) | PyInstaller — single-exe distribution for Windows once the model is trained and stable |

### Why This Split Matters

Keep the ML/PR section focused on what satisfies the rubric (loss function,
metrics, baselines, ablations) — the app stack is just the deployment target
proving the model works on-device, not the graded deep-learning core. Moving
the deployment target from Flutter/mobile to Python/PC only changes Part 2,
and it actually *simplifies* the project: one language, one interpreter, one
feature-extraction code path shared between training and inference, and no
mobile build toolchain to maintain.

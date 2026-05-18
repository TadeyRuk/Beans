WINDOW_SIZE = (600, 400)
WINDOW_MARGIN = 20
TARGET_FPS = 60
WEBCAM_INDEX = 0
PROCESS_RESOLUTION = (640, 480)
MAX_HANDS = 2
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.5

DEPTH_SCALE = 0.3
SMOOTHING = 0.35          # EMA weight for fresh landmark samples (lower = smoother)
HAND_FIT_SCALE = 0.85     # shrink 1-hand layout so mesh doesn't clip window edges
USE_VSYNC = True          # lock render loop to display refresh via swap_interval
CAPTURE_TICK = 1.0 / 60  # capture thread sleep cap (prevents CPU burn)
SHOW_FPS = False          # set True to show FPS counter in top-left corner

# Camera-feed background — mesh overlaid on live webcam
BG_COLOR = (0.0, 0.0, 0.0, 1.0)          # cleared behind camera quad (unused visually)
FACE_COLOR = (0.0, 0.7, 0.85, 0.22)       # translucent cyan — camera shows through faces
EDGE_COLOR_GLOW = (0.0, 0.8, 0.9, 0.25)
EDGE_COLOR_BRIGHT = (0.6, 1.0, 1.0, 1.0)
JOINT_COLOR = (1.0, 1.0, 1.0, 1.0)
PULSE_COLOR = (0.0, 0.7, 0.85, 0.55)

GLOW_THICKNESS = 5.0
EDGE_THICKNESS = 1.5
JOINT_SIZE = 6.0

# Low-poly triangle mesh topology over MediaPipe's 21 landmarks.
#   0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
HAND_TRIANGLES = [
    # Palm fan
    (0, 1, 5), (0, 5, 9), (0, 9, 13), (0, 13, 17),
    # Gaps between finger bases
    (5, 6, 9), (9, 10, 13), (13, 14, 17),
    # Thumb segments
    (1, 2, 5), (2, 3, 4),
    # Index finger
    (5, 6, 7), (6, 7, 8),
    # Middle finger
    (9, 10, 11), (10, 11, 12),
    # Ring finger
    (13, 14, 15), (14, 15, 16),
    # Pinky
    (17, 18, 19), (18, 19, 20),
]

WAKE_WORD_PATH = "beans/assets/hey-beans.ppn"
PORCUPINE_KEY_ENV = "PORCUPINE_KEY"
WAKE_WORD_ENABLED = False  # v1.1

PINCH_SENSITIVITY = 150   # delta multiplier for medium responsiveness
PINCH_ACTIVATE = 0.60     # normalized thumb-to-index distance below which we engage
VOL_FADE_RATE = 1 / 120   # 2-second fade at 60fps

# Volume gesture line (index tip → wrist) — 0-255 RGBA for pyglet
VOL_LINE_GLOW_COLOR = (0, 210, 60, 90)      # outer glow
VOL_LINE_CORE_COLOR = (60, 255, 100, 240)   # bright core
VOL_LABEL_COLOR     = (100, 255, 140, 240)  # "NN%" text

import threading
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarker,
    HandLandmarkerOptions,
    HandLandmarksConnections,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from beans import config

HAND_CONNECTIONS = [(c.start, c.end) for c in HandLandmarksConnections.HAND_CONNECTIONS]

_MODEL_PATH = "beans/assets/hand_landmarker.task"


class HandTracker:
    def __init__(self):
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode=VisionTaskRunningMode.IMAGE,
            num_hands=config.MAX_HANDS,
            min_hand_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_hand_presence_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._landmarker = HandLandmarker.create_from_options(options)

    def process(self, frame_bgr: np.ndarray) -> tuple[list, list[str], np.ndarray]:
        """Return (landmarks_list, handedness_list, flipped_rgb_frame).
        landmarks_list: 0-2 arrays of shape (21, 3) in normalized coords.
        handedness_list: matching list of 'Left' or 'Right' strings.
        flipped_rgb_frame: the horizontally-flipped RGB frame used for detection.
        """
        frame_bgr = cv2.flip(frame_bgr, 1)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect(mp_image)
        hands = []
        handedness = []
        if result.hand_landmarks:
            for i, hand_lm in enumerate(result.hand_landmarks):
                lm = np.array([(p.x, p.y, p.z) for p in hand_lm], dtype=np.float32)
                hands.append(lm)
                label = result.handedness[i][0].category_name if result.handedness else "Right"
                handedness.append(label)
        return hands, handedness, frame_rgb

    def close(self):
        self._landmarker.close()


class LatestFrame:
    """Thread-safe single-slot for the most recent (landmarks, handedness, rgb_frame) tuple."""
    def __init__(self):
        self._lock = threading.Lock()
        self._hands: list = []
        self._handedness: list[str] = []
        self._frame: np.ndarray | None = None

    def set(self, hands: list, handedness: list[str], frame: np.ndarray):
        with self._lock:
            self._hands = hands
            self._handedness = handedness
            self._frame = frame

    def get(self) -> tuple[list, list[str], np.ndarray | None]:
        with self._lock:
            return self._hands, self._handedness, self._frame


# Keep old name as alias so main.py import still resolves
LatestLandmarks = LatestFrame


class CaptureLoop(threading.Thread):
    def __init__(self, slot: LatestFrame, stop_event: threading.Event):
        super().__init__(daemon=True, name="beans-capture")
        self.slot = slot
        self.stop_event = stop_event

    def run(self):
        tracker = HandTracker()
        cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        w, h = config.PROCESS_RESOLUTION
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, config.TARGET_FPS)

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                hands, handedness, rgb = tracker.process(frame)
                self.slot.set(hands, handedness, rgb)
        finally:
            cap.release()
            tracker.close()

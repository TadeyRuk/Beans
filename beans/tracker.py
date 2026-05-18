import threading
import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
import numpy as np

from beans import config


class HandTracker:
    def __init__(self):
        self._hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=config.MAX_HANDS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def process(self, frame_bgr: np.ndarray) -> list:
        """Return list of 0-2 arrays shaped (21, 3) in normalized image coords."""
        # Flip horizontally for selfie mirror, convert to RGB
        frame_rgb = cv2.cvtColor(cv2.flip(frame_bgr, 1), cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._hands.process(frame_rgb)
        if not results.multi_hand_landmarks:
            return []
        hands = []
        for hand_lm in results.multi_hand_landmarks:
            lm = np.array(
                [(p.x, p.y, p.z) for p in hand_lm.landmark], dtype=np.float32
            )
            hands.append(lm)
        return hands

    def close(self):
        self._hands.close()


class LatestLandmarks:
    def __init__(self):
        self._lock = threading.Lock()
        self._value: list = []

    def set(self, hands: list):
        with self._lock:
            self._value = hands

    def get(self) -> list:
        with self._lock:
            return self._value


class CaptureLoop(threading.Thread):
    def __init__(self, slot: LatestLandmarks, stop_event: threading.Event):
        super().__init__(daemon=True, name="beans-capture")
        self.slot = slot
        self.stop_event = stop_event
        self._tracker = HandTracker()

    def run(self):
        cap = cv2.VideoCapture(config.WEBCAM_INDEX)
        w, h = config.PROCESS_RESOLUTION
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    continue
                # Resize to processing resolution in case webcam reported different
                if frame.shape[1] != w or frame.shape[0] != h:
                    frame = cv2.resize(frame, (w, h))
                hands = self._tracker.process(frame)
                self.slot.set(hands)
        finally:
            cap.release()
            self._tracker.close()

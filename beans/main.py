import os
import sys
import time
import threading
import signal
from collections import deque

# Force XWayland so GLFW can set window position and drag on GNOME/Wayland
os.environ.setdefault("GDK_BACKEND", "x11")
os.environ.setdefault("GLFW_PLATFORM", "x11")

import glfw
import moderngl

from beans import config
from beans import window as win_mod
from beans.tracker import CaptureLoop, LatestFrame
from beans.gesture import classify, _pinch_distance
from beans.volume import PinchVolumeController
from beans.renderer import WireframeRenderer


class AppState:
    def __init__(self):
        self.visible = True
        self.stop = threading.Event()
        self.lock = threading.Lock()


def main():
    state = AppState()

    def _sigint(_sig, _frame):
        state.stop.set()
    signal.signal(signal.SIGINT, _sigint)

    frame_slot = LatestFrame()
    capture = CaptureLoop(frame_slot, state.stop)
    capture.start()

    window = win_mod.create_window("beans", config.WINDOW_SIZE)
    glfw.make_context_current(window)
    glfw.swap_interval(1 if config.USE_VSYNC else 0)

    ctx = moderngl.create_context()
    renderer = WireframeRenderer(ctx)
    volume_ctrl = PinchVolumeController()

    def _on_key(win, key, _scancode, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            state.stop.set()

    glfw.set_key_callback(window, _on_key)

    t_start = time.monotonic()
    frame_times: deque[float] = deque(maxlen=30)
    last_frame_time = time.monotonic()

    # TODO(v1.1): wake-word integration
    # if config.WAKE_WORD_ENABLED:
    #     from beans.wake_word import WakeWordListener
    #     listener = WakeWordListener(state, config.WAKE_WORD_PATH,
    #                                 os.environ[config.PORCUPINE_KEY_ENV])
    #     listener.start()

    try:
        while not state.stop.is_set():
            glfw.poll_events()
            if glfw.window_should_close(window):
                state.stop.set()
                break

            hands, handedness, rgb_frame = frame_slot.get()
            gestures = [classify(h) for h in hands]
            right_lm = next((lm for lm, h in zip(hands, handedness) if h == "Right"), None)
            if right_lm is not None:
                is_pinching = classify(right_lm) == "pinch"
                new_vol = volume_ctrl.update(is_pinching, _pinch_distance(right_lm))
                if new_vol is not None:
                    renderer.set_volume_display(new_vol)
                else:
                    renderer.tick_volume_fade()
            else:
                renderer.tick_volume_fade()
            t = time.monotonic() - t_start

            # Upload camera frame to GPU texture
            renderer.update_camera_frame(rgb_frame)

            # Rolling FPS
            now = time.monotonic()
            frame_times.append(now - last_frame_time)
            last_frame_time = now
            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else None

            renderer.render(
                hands, gestures, t,
                fps=fps if config.SHOW_FPS else None,
            )
            glfw.swap_buffers(window)

            if not config.USE_VSYNC:
                elapsed = time.monotonic() - now
                remaining = (1.0 / config.TARGET_FPS) - elapsed
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        state.stop.set()
        capture.join(timeout=2.0)
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()

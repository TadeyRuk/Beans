import os
import sys
import time
import threading
import signal

import glfw
import moderngl

from beans import config
from beans import window as win_mod
from beans.tracker import CaptureLoop, LatestLandmarks
from beans.gesture import classify
from beans.renderer import WireframeRenderer


class AppState:
    def __init__(self):
        self.visible = True
        self.stop = threading.Event()
        self.lock = threading.Lock()


def main():
    state = AppState()

    # Graceful shutdown on Ctrl-C
    def _sigint(_sig, _frame):
        state.stop.set()
    signal.signal(signal.SIGINT, _sigint)

    landmarks = LatestLandmarks()
    capture = CaptureLoop(landmarks, state.stop)
    capture.start()

    window = win_mod.create_window("beans", config.WINDOW_SIZE)
    glfw.make_context_current(window)
    glfw.swap_interval(0)  # we pace manually

    ctx = moderngl.create_context()
    renderer = WireframeRenderer(ctx)

    def _on_key(win, key, _scancode, action, _mods):
        if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
            state.stop.set()

    glfw.set_key_callback(window, _on_key)

    frame_duration = 1.0 / config.TARGET_FPS
    t_start = time.monotonic()

    # TODO(v1.1): wake-word integration
    # if config.WAKE_WORD_ENABLED:
    #     from beans.wake_word import WakeWordListener
    #     listener = WakeWordListener(state, config.WAKE_WORD_PATH,
    #                                 os.environ[config.PORCUPINE_KEY_ENV])
    #     listener.start()

    try:
        while not state.stop.is_set():
            frame_start = time.monotonic()

            glfw.poll_events()
            if glfw.window_should_close(window):
                state.stop.set()
                break

            hands = landmarks.get()
            gestures = [classify(h) for h in hands]
            t = frame_start - t_start

            renderer.render(hands, gestures, t)
            glfw.swap_buffers(window)

            elapsed = time.monotonic() - frame_start
            sleep_for = frame_duration - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
    finally:
        state.stop.set()
        capture.join(timeout=2.0)
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()

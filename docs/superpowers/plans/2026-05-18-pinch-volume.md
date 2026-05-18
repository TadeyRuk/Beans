# Pinch-to-Volume Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Right-hand index-thumb pinch gesture controls system volume via delta-based distance tracking, with a teal volume bar overlay that fades out 2 seconds after release.

**Architecture:** A new `beans/volume.py` handles all system volume I/O (`pactl`) and the pinch-delta state machine. `tracker.py` is extended to surface MediaPipe handedness labels. `main.py` filters to the right hand and drives the controller each frame. `renderer.py` draws a pyglet volume bar + percentage label with alpha fade.

**Tech Stack:** Python 3, MediaPipe Tasks API, pactl (PipeWire/PulseAudio), pyglet shapes, moderngl (no new GL programs).

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `beans/volume.py` | **Create** | `get_volume`, `set_volume`, `PinchVolumeController` |
| `beans/config.py` | **Modify** | Add `PINCH_SENSITIVITY`, `VOL_FADE_RATE` |
| `beans/tracker.py` | **Modify** | Return handedness labels from `process()` and `LatestFrame` |
| `beans/main.py` | **Modify** | Right-hand selection, per-frame volume controller update |
| `beans/renderer.py` | **Modify** | Volume bar + label overlay, fade state, `set_volume_display`, `tick_volume_fade` |

---

### Task 1: Config constants

**Files:**
- Modify: `beans/config.py`

- [ ] **Step 1: Add constants to config**

Open `beans/config.py` and append at the end:

```python
PINCH_SENSITIVITY = 150   # delta multiplier for medium responsiveness
VOL_FADE_RATE = 1 / 120   # 2-second fade at 60fps
```

- [ ] **Step 2: Verify compile**

```bash
python -m py_compile beans/config.py && echo "OK"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add beans/config.py
git commit -m "config: add PINCH_SENSITIVITY and VOL_FADE_RATE"
```

---

### Task 2: System volume module

**Files:**
- Create: `beans/volume.py`

- [ ] **Step 1: Create `beans/volume.py`**

```python
import re
import subprocess

from beans import config


def get_volume() -> int:
    """Return current default sink volume as int 0-100. Returns 50 on failure."""
    try:
        out = subprocess.run(
            ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
            capture_output=True, text=True, timeout=1,
        ).stdout
        m = re.search(r"(\d+)%", out)
        return int(m.group(1)) if m else 50
    except Exception:
        return 50


def set_volume(pct: int) -> None:
    """Set default sink volume to pct (0-100). Silently ignores failures."""
    pct = max(0, min(100, pct))
    try:
        subprocess.run(
            ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{pct}%"],
            capture_output=True, timeout=1,
        )
    except Exception:
        pass


class PinchVolumeController:
    """Delta-based pinch-to-volume controller.

    Call update() every frame with whether the hand is pinching and the
    current normalised pinch distance (|tip4 - tip8| / palm_width).
    Returns the new volume int when active, None when not pinching.
    """

    def __init__(self) -> None:
        self._volume: int = get_volume()
        self._last_dist: float | None = None

    def update(self, is_pinching: bool, norm_dist: float) -> int | None:
        if not is_pinching:
            self._last_dist = None
            return None

        if self._last_dist is None:
            # First pinch frame — anchor without changing volume.
            self._last_dist = norm_dist
            return self._volume

        delta = (norm_dist - self._last_dist) * config.PINCH_SENSITIVITY
        self._volume = max(0, min(100, int(self._volume + delta)))
        self._last_dist = norm_dist
        set_volume(self._volume)
        return self._volume
```

- [ ] **Step 2: Verify compile**

```bash
python -m py_compile beans/volume.py && echo "OK"
```
Expected: `OK`

- [ ] **Step 3: Smoke-test volume I/O manually**

```bash
source .venv/bin/activate
python -c "from beans.volume import get_volume, set_volume; v = get_volume(); print('current:', v); set_volume(v)"
```
Expected: prints `current: <some number 0-100>` with no error. Volume unchanged.

- [ ] **Step 4: Commit**

```bash
git add beans/volume.py
git commit -m "feat: add volume.py with get/set_volume and PinchVolumeController"
```

---

### Task 3: Extend tracker to surface handedness

**Files:**
- Modify: `beans/tracker.py`

- [ ] **Step 1: Update `HandTracker.process()` to return handedness**

Replace the entire `process` method and the `LatestFrame` class in `beans/tracker.py`:

```python
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
```

- [ ] **Step 2: Update `LatestFrame` to carry handedness**

Replace the `LatestFrame` class:

```python
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
```

- [ ] **Step 3: Update `CaptureLoop.run()` to pass handedness to slot**

In `CaptureLoop.run()`, change the line that calls `tracker.process` and `self.slot.set`:

```python
hands, handedness, rgb = tracker.process(frame)
self.slot.set(hands, handedness, rgb)
```

- [ ] **Step 4: Verify compile**

```bash
python -m py_compile beans/tracker.py && echo "OK"
```
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add beans/tracker.py
git commit -m "feat(tracker): surface handedness labels alongside landmarks"
```

---

### Task 4: Wire volume controller into main loop

**Files:**
- Modify: `beans/main.py`

- [ ] **Step 1: Import volume module and `_pinch_distance`**

At the top of `beans/main.py`, add these imports after the existing `from beans.gesture import classify` line:

```python
from beans.gesture import classify, _pinch_distance
from beans.volume import PinchVolumeController
```

- [ ] **Step 2: Instantiate controller in `main()`**

After `renderer = WireframeRenderer(ctx)`, add:

```python
volume_ctrl = PinchVolumeController()
```

- [ ] **Step 3: Update the frame unpack to include handedness**

Change:
```python
hands, rgb_frame = frame_slot.get()
```
To:
```python
hands, handedness, rgb_frame = frame_slot.get()
```

- [ ] **Step 4: Add right-hand volume logic after `gestures = ...`**

After `gestures = [classify(h) for h in hands]`, add:

```python
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
```

- [ ] **Step 5: Verify compile**

```bash
python -m py_compile beans/main.py && echo "OK"
```
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add beans/main.py
git commit -m "feat(main): wire PinchVolumeController to right-hand detection"
```

---

### Task 5: Volume overlay in renderer

**Files:**
- Modify: `beans/renderer.py`

- [ ] **Step 1: Add volume state to `__init__`**

In `WireframeRenderer.__init__`, after `self._label: object = None` (around line 228), add:

```python
self._vol_level: int = 0
self._vol_alpha: float = 0.0
self._vol_fading: bool = False
```

- [ ] **Step 2: Add `set_volume_display` method**

Add this method to `WireframeRenderer` (after `_draw_fps`):

```python
def set_volume_display(self, level: int) -> None:
    self._vol_level = level
    self._vol_alpha = 1.0
    self._vol_fading = False
```

- [ ] **Step 3: Add `tick_volume_fade` method**

```python
def tick_volume_fade(self) -> None:
    if self._vol_alpha <= 0.0:
        return
    if not self._vol_fading:
        self._vol_fading = True
    self._vol_alpha = max(0.0, self._vol_alpha - config.VOL_FADE_RATE)
```

- [ ] **Step 4: Add `_draw_volume_bar` method**

```python
def _draw_volume_bar(self) -> None:
    if self._vol_alpha <= 0.0:
        return
    try:
        import pyglet.shapes
    except ImportError:
        return

    w, h = config.WINDOW_SIZE
    bar_w = 12
    pad_x = 10
    pad_y = 30
    bar_h = h - pad_y * 2
    bar_x = w - pad_x - bar_w
    bar_y = pad_y

    a = self._vol_alpha

    # Background track
    bg = pyglet.shapes.Rectangle(
        bar_x, bar_y, bar_w, bar_h,
        color=(20, 20, 30, int(120 * a)),
    )
    bg.draw()

    # Fill
    fill_h = int(bar_h * self._vol_level / 100)
    if fill_h > 0:
        fill = pyglet.shapes.Rectangle(
            bar_x, bar_y, bar_w, fill_h,
            color=(0, 180, 210, int(220 * a)),
        )
        fill.draw()

    # Percentage label above bar
    if self._fps_label is not None:
        import pyglet.text
        lbl = pyglet.text.Label(
            f"{self._vol_level}%",
            font_name="monospace",
            font_size=11,
            x=bar_x + bar_w // 2,
            y=bar_y + bar_h + 12,
            anchor_x="center",
            anchor_y="center",
            color=(220, 255, 255, int(220 * a)),
        )
        lbl.draw()
    else:
        try:
            import pyglet.text
            lbl = pyglet.text.Label(
                f"{self._vol_level}%",
                font_name="monospace",
                font_size=11,
                x=bar_x + bar_w // 2,
                y=bar_y + bar_h + 12,
                anchor_x="center",
                anchor_y="center",
                color=(220, 255, 255, int(220 * a)),
            )
            lbl.draw()
        except Exception:
            pass
```

- [ ] **Step 5: Call `_draw_volume_bar` in `render()`**

At the very end of `render()`, after `self._draw_fps(fps)` (both the early-return path and the normal path), add:

```python
self._draw_volume_bar()
```

For the early-return path (no hands), the method currently returns after `self._draw_fps(fps)`. Change that block so it calls `_draw_volume_bar` before returning:

```python
if not hands:
    self._pulse_prog["u_time"] = t
    self._pulse_prog["u_color"] = config.PULSE_COLOR
    self._pulse_prog["u_resolution"] = (rx, ry)
    self._pulse_vao.render(moderngl.TRIANGLE_STRIP)
    self._draw_search_label()
    self._draw_fps(fps)
    self._draw_volume_bar()
    return
```

And at the end of the normal render path (after `self._draw_fps(fps)`):

```python
self._draw_gesture_label(gestures)
self._draw_fps(fps)
self._draw_volume_bar()
```

- [ ] **Step 6: Verify compile**

```bash
python -m py_compile beans/renderer.py && echo "OK"
```
Expected: `OK`

- [ ] **Step 7: Compile-check all modules**

```bash
python -m py_compile beans/*.py && echo "ALL OK"
```
Expected: `ALL OK`

- [ ] **Step 8: Commit**

```bash
git add beans/renderer.py
git commit -m "feat(renderer): volume bar overlay with alpha fade"
```

---

### Task 6: End-to-end verification

- [ ] **Step 1: Launch the app**

```bash
cd /home/tadey/Documents/Beans && source .venv/bin/activate && python -m beans.main
```

- [ ] **Step 2: No hands → no volume bar visible**

Wave nothing in front of the camera. Volume bar should be invisible.

- [ ] **Step 3: Right hand open → no volume bar**

Show open right hand (no pinch). Volume bar should remain invisible.

- [ ] **Step 4: Pinch right hand → bar appears**

Bring right-hand index tip and thumb tip together. Volume bar should appear at current system volume level.

- [ ] **Step 5: Spread fingers while pinching → volume rises**

While keeping pinch gesture, slowly spread index and thumb apart. Bar should fill up; system volume should increase (verify with audio playing).

- [ ] **Step 6: Squeeze fingers while pinching → volume drops**

While pinching, bring fingers closer. Bar drains; system volume decreases.

- [ ] **Step 7: Release pinch → bar fades out over ~2 seconds**

Open hand. Bar should fade to invisible over approximately 2 seconds.

- [ ] **Step 8: Left-hand pinch → no effect**

Show only left hand and pinch. Volume should not change, bar should not appear.

- [ ] **Step 9: Both hands, pinch right → volume changes; left pinch ignored**

Show both hands. Pinch only the right. Volume responds. Pinch only the left. Volume does not change.

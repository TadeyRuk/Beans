# Pinch-to-Volume Control — Design Spec
**Date:** 2026-05-18  
**Project:** Beans (Kouri System)

---

## Overview

Right-hand index-thumb pinch gesture controls system volume in real-time. While pinching, moving fingers apart raises volume and squeezing them together lowers it (delta-based). A volume indicator overlaid on the Beans panel fades out 2 seconds after the pinch is released.

---

## Architecture

Three concerns, three places:

1. **`beans/volume.py`** — system volume I/O + pinch-delta state machine  
2. **`beans/main.py`** — right-hand selection, per-frame controller update  
3. **`beans/renderer.py`** — volume bar + label overlay, fade timer

---

## `beans/volume.py`

### System volume
- `get_volume() -> int` — shells out to `pactl get-sink-volume @DEFAULT_SINK@`, parses the first `XX%` value, returns int 0–100.
- `set_volume(pct: int)` — calls `pactl set-sink-volume @DEFAULT_SINK@ {pct}%`, clamped to 0–100.
- Both use `subprocess.run` with `capture_output=True`; failures are silently swallowed (no volume change).

### `PinchVolumeController`
State: `_last_dist: float | None`, `_volume: int` (seeded from `get_volume()` at construction).

```
update(is_pinching: bool, norm_dist: float) -> int | None
```
- If `is_pinching` is False: reset `_last_dist = None`, return `None`.
- If first pinch frame (`_last_dist is None`): record `_last_dist = norm_dist`, return current volume (no jump).
- Otherwise: `delta = (norm_dist - _last_dist) * SENSITIVITY`, where `SENSITIVITY = 150` gives medium responsiveness (a ~0.27 normalized-distance sweep covers ~40% volume).
- New volume = clamp(`_volume + delta`, 0, 100). Call `set_volume()`. Update `_last_dist`. Return new volume int.

`norm_dist` is `|tip[4] - tip[8]|` divided by palm width (already computed by `_pinch_distance` in `gesture.py` — reuse that helper).

---

## `beans/main.py`

### Right-hand selection
MediaPipe `HandLandmarker` result includes `handedness` alongside `hand_landmarks`. In `tracker.py`, `HandTracker.process()` currently returns only landmarks. Extend it to also return handedness labels so `main.py` can filter.

- `tracker.py`: return `(hands, handedness_labels, rgb_frame)` where `handedness_labels` is a list of `"Left"` or `"Right"` strings (one per detected hand, same order as `hands`).
- `main.py`: after receiving the frame, find the hand whose label is `"Right"`. Pass it alone to `PinchVolumeController.update()`.
- `LatestFrame` slot: extend to carry `handedness` list alongside `hands`.

### Per-frame update
```python
right_lm = next((lm for lm, h in zip(hands, handedness) if h == "Right"), None)
if right_lm is not None:
    is_pinching = classify(right_lm) == "pinch"
    norm_dist = _pinch_distance(right_lm)  # reuse from gesture.py
    new_vol = volume_ctrl.update(is_pinching, norm_dist)
    if new_vol is not None:
        renderer.set_volume_display(new_vol, active=True)
    else:
        renderer.tick_volume_fade()  # advance fade timer when not pinching
```

---

## `beans/renderer.py` — Volume Overlay

### State added to `WireframeRenderer`
- `_vol_level: int` — 0–100, last known volume
- `_vol_alpha: float` — 0.0–1.0, current opacity
- `_vol_fading: bool` — True after pinch release, False while pinching

### API
- `set_volume_display(level: int, active: bool)` — sets `_vol_level = level`, `_vol_alpha = 1.0`, `_vol_fading = False`.
- `tick_volume_fade()` — if `_vol_fading` is False, starts the fade (`_vol_fading = True`). Decrements `_vol_alpha` by `FADE_RATE` per frame (2-second fade at 60fps → `FADE_RATE = 1/120`). Clamps to 0.

### Visual design
Drawn in `render()` after all mesh passes, using pyglet (same as gesture label — no new GL programs needed):

**Volume bar** — right edge of the panel, 12px wide, inset 10px from right, full height minus 30px top/bottom padding. Drawn as two pyglet shapes:
- Background track: dark translucent rectangle (`(20, 20, 30, 120)`).
- Fill: teal rectangle from bottom, height proportional to `_vol_level / 100`. Color `(0, 180, 210, 220)` — matches the mesh edge palette.

**Label** — `"{vol}%"` in monospace 11pt, centered above the bar, color `(220, 255, 255, 220)`.

Both drawn at `_vol_alpha` opacity. When `_vol_alpha <= 0`, nothing is drawn.

---

## Config additions (`beans/config.py`)
```python
PINCH_SENSITIVITY = 150   # delta multiplier; medium feel
VOL_FADE_RATE = 1 / 120   # 2-second fade at 60fps
```

---

## Files modified
| File | Change |
|------|--------|
| `beans/volume.py` | **new** — `get_volume`, `set_volume`, `PinchVolumeController` |
| `beans/tracker.py` | Return handedness alongside landmarks |
| `beans/main.py` | Right-hand selection, volume controller integration |
| `beans/renderer.py` | Volume bar + label overlay with fade |
| `beans/config.py` | `PINCH_SENSITIVITY`, `VOL_FADE_RATE` |

---

## Verification
1. `python -m py_compile beans/*.py` — clean.
2. Run app, no hands visible — no volume bar shown.
3. Show right hand, no pinch — no volume bar.
4. Pinch right hand index+thumb — volume bar appears at current volume.
5. Spread fingers while pinching — bar fills up, system volume rises (test with audio playing).
6. Squeeze fingers while pinching — bar drains, volume drops.
7. Release pinch — bar fades out over ~2 seconds.
8. Show only left hand and pinch — volume does NOT change.
9. Show both hands, pinch right — only right hand controls volume.

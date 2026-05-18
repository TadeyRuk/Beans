# Beans

**Beans** is an ambient hand tracking visualizer — a module of the [Kouri System](https://github.com/kouri-system).

It opens a small, floating, transparent window on your desktop showing a real-time low-poly wireframe mesh of your hand(s), captured via webcam and tracked with MediaPipe. Designed for Fedora + Hyprland, but works on any Wayland-based Linux.

---

## Status

| Feature | v1.0 (now) | v1.1 (soon) |
|---|---|---|
| Hand tracking + wireframe rendering | ✅ | — |
| Gesture detection + labels | ✅ | — |
| "Hey Beans" wake word toggle | 🔲 scaffolded | ✅ |
| Gesture → system actions | — | v2 |

---

## What it does now

- Opens immediately on launch as a transparent 400×400 floating window
- Webcam → MediaPipe → 21-landmark hand mesh, up to 2 hands simultaneously
- Cyan/white glowing wireframe edges, bright joint dots
- Gesture label displayed at bottom center: `open_palm`, `fist`, `point`, `pinch`, `peace`
- 0 hands: soft pulsing ring + "searching..." indicator

---

## Requirements

- Linux (Fedora + Hyprland recommended; any Wayland compositor works)
- Python 3.11+
- Webcam
- Microphone *(for wake word in v1.1)*

---

## Install

```bash
# For wake word audio support (v1.1) — install now so the env is ready
sudo dnf install portaudio-devel

# Create venv and install deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For v1.1 wake word, you will also need a [Picovoice access key](https://console.picovoice.ai/) and a `hey-beans.ppn` wake word model placed in `beans/assets/`.

---

## Run

```bash
python -m beans.main
```

Press **ESC** or close the window to exit.

---

## Hyprland configuration

Add to your `~/.config/hypr/hyprland.conf`:

```
windowrulev2 = float, class:^(beans)$
windowrulev2 = pin, class:^(beans)$
windowrulev2 = noborder, class:^(beans)$
windowrulev2 = noshadow, class:^(beans)$
windowrulev2 = size 400 400, class:^(beans)$
windowrulev2 = move 100%-420 100%-420, class:^(beans)$
```

This pins Beans to the bottom-right corner of your screen, floating and always on top.

> **Note on dragging:** On Wayland, GLFW cannot reposition windows programmatically. Use your normal Hyprland window drag binding (default: `SUPER + left-click drag`) to move the window.

---

## Configuration

Edit `beans/config.py` to tune:

| Key | Default | Description |
|---|---|---|
| `WEBCAM_INDEX` | `0` | Camera device index |
| `TARGET_FPS` | `30` | Render frame rate |
| `PROCESS_RESOLUTION` | `(640, 480)` | MediaPipe input resolution |
| `MAX_HANDS` | `2` | Maximum simultaneous hands |
| `GLOW_THICKNESS` | `6.0` | Glow pass line width (px) |
| `EDGE_THICKNESS` | `1.5` | Bright edge line width (px) |

---

## Project structure

```
beans/
├── main.py          # entry point, render loop
├── tracker.py       # MediaPipe hand tracking + capture thread
├── renderer.py      # ModernGL wireframe mesh + glow
├── gesture.py       # geometry-based gesture classification
├── window.py        # GLFW transparent window setup
├── wake_word.py     # Porcupine wake word (scaffold — v1.1)
├── config.py        # constants
└── assets/
    └── hey-beans.ppn  # user-provided Porcupine model (v1.1)
```

---

## Roadmap

- **v1.1** — "Hey Beans" wake word via Porcupine toggles window visibility
- **v2** — Gesture → system action mapping (scroll, workspace switch, media control)

---

## License

MIT — see [LICENSE](LICENSE).

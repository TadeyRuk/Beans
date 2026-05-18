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

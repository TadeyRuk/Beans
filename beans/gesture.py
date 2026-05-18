import numpy as np

# Finger landmark indices: [MCP, PIP, DIP, TIP]
_FINGER_INDICES = [
    [1, 2, 3, 4],   # thumb  (special handling)
    [5, 6, 7, 8],   # index
    [9, 10, 11, 12], # middle
    [13, 14, 15, 16], # ring
    [17, 18, 19, 20], # pinky
]

GESTURE_ACTIONS: dict[str, str] = {
    "open_palm": "idle",
    "fist": "—",
    "point": "—",
    "pinch": "—",
    "peace": "—",
    "unknown": "—",
}


def _palm_width(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[5] - lm[17])) + 1e-6


def _finger_extended(lm: np.ndarray, finger_idx: int) -> bool:
    mcp, pip, dip, tip = _FINGER_INDICES[finger_idx]
    if finger_idx == 0:
        # Thumb: compare tip x vs IP joint x (works for mirrored selfie)
        return float(lm[tip][0]) < float(lm[pip][0])
    tip_to_mcp = np.linalg.norm(lm[tip] - lm[mcp])
    pip_to_mcp = np.linalg.norm(lm[pip] - lm[mcp])
    return tip_to_mcp > pip_to_mcp * 1.1


def _pinch_distance(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[4] - lm[8])) / _palm_width(lm)


def classify(lm: np.ndarray) -> str:
    """Classify gesture from a single hand's 21×3 landmark array."""
    pinch = _pinch_distance(lm)
    if pinch < 0.25:
        return "pinch"

    extended = [_finger_extended(lm, i) for i in range(5)]
    # open palm: all non-thumb fingers extended
    if all(extended[1:]):
        return "open_palm"
    # fist: no fingers extended
    if not any(extended[1:]):
        return "fist"
    # point: only index extended
    if extended[1] and not any(extended[2:]):
        return "point"
    # peace: index + middle extended, ring + pinky curled
    if extended[1] and extended[2] and not extended[3] and not extended[4]:
        return "peace"
    return "unknown"

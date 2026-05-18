import math
import numpy as np
import moderngl
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

from beans import config

# All edges from MediaPipe's standard hand topology
_CONNECTIONS = list(mp_hands.HAND_CONNECTIONS)


_VERT_LINES = """
#version 330
in vec2 in_pos;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_GEOM_LINES = """
#version 330
layout(lines) in;
layout(triangle_strip, max_vertices = 4) out;

uniform vec2 u_resolution;
uniform float u_thickness;

void main() {
    vec2 p0 = gl_in[0].gl_Position.xy;
    vec2 p1 = gl_in[1].gl_Position.xy;

    // Convert NDC to pixels
    vec2 dir = normalize((p1 - p0) * u_resolution * 0.5);
    vec2 normal = vec2(-dir.y, dir.x);
    vec2 offset = (normal / (u_resolution * 0.5)) * u_thickness * 0.5;

    gl_Position = vec4(p0 - offset, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(p0 + offset, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(p1 - offset, 0.0, 1.0); EmitVertex();
    gl_Position = vec4(p1 + offset, 0.0, 1.0); EmitVertex();
    EndPrimitive();
}
"""

_FRAG_FLAT = """
#version 330
uniform vec4 u_color;
out vec4 f_color;
void main() {
    f_color = u_color;
}
"""

_VERT_POINTS = """
#version 330
in vec2 in_pos;
uniform float u_point_size;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    gl_PointSize = u_point_size;
}
"""

_FRAG_POINTS = """
#version 330
uniform vec4 u_color;
out vec4 f_color;
void main() {
    vec2 c = gl_PointCoord - 0.5;
    if (dot(c, c) > 0.25) discard;
    f_color = u_color;
}
"""

_VERT_QUAD = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG_PULSE = """
#version 330
in vec2 v_uv;
uniform float u_time;
uniform vec4 u_color;
out vec4 f_color;
void main() {
    vec2 c = v_uv - 0.5;
    float d = length(c);
    float radius = 0.25 + 0.03 * sin(u_time * 3.0);
    float ring = 1.0 - smoothstep(0.0, 0.015, abs(d - radius));
    f_color = vec4(u_color.rgb, u_color.a * ring);
}
"""


def _lm_to_ndc(lm: np.ndarray) -> np.ndarray:
    """Convert MediaPipe normalized coords to NDC. x is already mirrored by tracker."""
    xy = lm[:, :2].copy()
    xy[:, 0] = xy[:, 0] * 2.0 - 1.0
    xy[:, 1] = 1.0 - xy[:, 1] * 2.0
    return xy.astype(np.float32)


def _layout(ndc: np.ndarray, slot: int, total: int) -> np.ndarray:
    if total == 1:
        return ndc
    # Two hands: shift left (slot 0) or right (slot 1), scale down slightly
    scale = 0.6
    offset_x = -0.4 if slot == 0 else 0.4
    out = ndc.copy()
    out[:, 0] = ndc[:, 0] * scale + offset_x
    out[:, 1] = ndc[:, 1] * scale
    return out


class WireframeRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        w, h = config.WINDOW_SIZE

        self._line_prog = ctx.program(
            vertex_shader=_VERT_LINES,
            geometry_shader=_GEOM_LINES,
            fragment_shader=_FRAG_FLAT,
        )
        self._point_prog = ctx.program(
            vertex_shader=_VERT_POINTS,
            fragment_shader=_FRAG_POINTS,
        )
        self._pulse_prog = ctx.program(
            vertex_shader=_VERT_QUAD,
            fragment_shader=_FRAG_PULSE,
        )

        max_pts = config.MAX_HANDS * 21
        self._pt_vbo = ctx.buffer(reserve=max_pts * 2 * 4)
        self._pt_vao = ctx.vertex_array(self._point_prog, [(self._pt_vbo, "2f", "in_pos")])

        max_edges = config.MAX_HANDS * len(_CONNECTIONS) * 2
        self._edge_vbo = ctx.buffer(reserve=max_edges * 2 * 4)
        self._edge_vao = ctx.vertex_array(self._line_prog, [(self._edge_vbo, "2f", "in_pos")])

        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        self._quad_vbo = ctx.buffer(quad.tobytes())
        self._quad_vao = ctx.vertex_array(self._pulse_prog, [(self._quad_vbo, "2f", "in_pos")])

        self._resolution = (float(w), float(h))

        # Text labels via pyglet
        self._label: object = None
        self._search_label: object = None
        self._init_labels()

    def _init_labels(self):
        try:
            import pyglet
            self._label = pyglet.text.Label(
                "",
                font_name="monospace",
                font_size=11,
                x=config.WINDOW_SIZE[0] // 2,
                y=18,
                anchor_x="center",
                anchor_y="center",
                color=(180, 255, 255, 200),
            )
            self._search_label = pyglet.text.Label(
                "searching...",
                font_name="monospace",
                font_size=11,
                x=config.WINDOW_SIZE[0] // 2,
                y=18,
                anchor_x="center",
                anchor_y="center",
                color=(100, 200, 210, 160),
            )
        except Exception:
            pass  # labels are nice-to-have; rendering continues without them

    def _write_hand_data(self, hands: list):
        all_pts: list[np.ndarray] = []
        all_edges: list[np.ndarray] = []
        n = len(hands)
        for i, lm in enumerate(hands):
            ndc = _layout(_lm_to_ndc(lm), i, n)
            all_pts.append(ndc)
            for a, b in _CONNECTIONS:
                all_edges.append(ndc[a])
                all_edges.append(ndc[b])
        if all_pts:
            pts = np.concatenate(all_pts, axis=0).astype(np.float32)
            self._pt_vbo.write(pts.tobytes())
        if all_edges:
            edges = np.array(all_edges, dtype=np.float32)
            self._edge_vbo.write(edges.tobytes())
        return (
            sum(len(h) for h in hands),
            len(all_edges) // 2,
        )

    def render(self, hands: list, gestures: list[str], t: float):
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.clear(*config.BG_COLOR)

        rx, ry = self._resolution

        if not hands:
            # Idle pulse ring
            self._pulse_prog["u_time"] = t
            self._pulse_prog["u_color"] = config.PULSE_COLOR
            self._quad_vao.render(moderngl.TRIANGLE_STRIP)
            self._draw_search_label()
            return

        n_pts, n_edges = self._write_hand_data(hands)

        # Glow pass — thick, dim
        self._line_prog["u_resolution"] = (rx, ry)
        self._line_prog["u_thickness"] = config.GLOW_THICKNESS
        self._line_prog["u_color"] = config.EDGE_COLOR_GLOW
        self._edge_vao.render(moderngl.LINES, vertices=n_edges * 2)

        # Bright pass — thin
        self._line_prog["u_thickness"] = config.EDGE_THICKNESS
        self._line_prog["u_color"] = config.EDGE_COLOR_BRIGHT
        self._edge_vao.render(moderngl.LINES, vertices=n_edges * 2)

        # Joint points
        self._point_prog["u_point_size"] = config.JOINT_SIZE
        self._point_prog["u_color"] = config.JOINT_COLOR
        self._pt_vao.render(moderngl.POINTS, vertices=n_pts)

        self._draw_gesture_label(gestures)

    def _draw_gesture_label(self, gestures: list[str]):
        if not gestures or self._label is None:
            return
        from beans.gesture import GESTURE_ACTIONS
        parts = []
        for g in gestures:
            action = GESTURE_ACTIONS.get(g, "—")
            parts.append(f"{g}  {action}")
        self._label.text = "  |  ".join(parts)
        self._label.draw()

    def _draw_search_label(self):
        if self._search_label is not None:
            self._search_label.draw()

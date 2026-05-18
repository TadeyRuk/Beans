import numpy as np
import moderngl

from beans import config
from beans.tracker import HAND_CONNECTIONS as _CONNECTIONS


# ---------------------------------------------------------------------------
# Shaders
# ---------------------------------------------------------------------------

# Camera feed — fullscreen quad, samples the webcam texture
_VERT_CAMERA = """
#version 330
in vec2 in_pos;
out vec2 v_uv;
void main() {
    v_uv = in_pos * 0.5 + 0.5;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

_FRAG_CAMERA = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 f_color;
void main() {
    // v_uv already matches the flipped (selfie-mirror) frame from tracker
    f_color = vec4(texture(u_tex, v_uv).rgb, 1.0);
}
"""

_VERT_FACE = """
#version 330
in vec3 in_pos;
uniform float u_depth_scale;
out float v_depth;
void main() {
    float z = in_pos.z;
    vec2 pos = in_pos.xy + vec2(0.0, z * u_depth_scale);
    gl_Position = vec4(pos, 0.0, 1.0);
    v_depth = z;
}
"""

_FRAG_FACE = """
#version 330
in float v_depth;
uniform vec4 u_color;
out vec4 f_color;
void main() {
    float brightness = 1.0 + v_depth * 0.6;
    f_color = vec4(clamp(u_color.rgb * brightness, 0.0, 1.0), u_color.a);
}
"""

_VERT_LINES = """
#version 330
in vec3 in_pos;
uniform float u_depth_scale;
void main() {
    float z = in_pos.z;
    vec2 pos = in_pos.xy + vec2(0.0, z * u_depth_scale);
    gl_Position = vec4(pos, 0.0, 1.0);
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
in vec3 in_pos;
uniform float u_depth_scale;
uniform float u_point_size;
void main() {
    float z = in_pos.z;
    vec2 pos = in_pos.xy + vec2(0.0, z * u_depth_scale);
    gl_Position = vec4(pos, 0.0, 1.0);
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
uniform vec2 u_resolution;
out vec4 f_color;
void main() {
    vec2 c = v_uv - 0.5;
    c.x *= u_resolution.x / u_resolution.y;
    float d = length(c);
    float radius = 0.18 + 0.02 * sin(u_time * 3.0);
    float ring = 1.0 - smoothstep(0.0, 0.012, abs(d - radius));
    f_color = vec4(u_color.rgb, u_color.a * ring);
}
"""


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _lm_to_ndc(lm: np.ndarray) -> np.ndarray:
    out = np.empty((21, 3), dtype=np.float32)
    out[:, 0] = lm[:, 0] * 2.0 - 1.0
    out[:, 1] = 1.0 - lm[:, 1] * 2.0
    out[:, 2] = lm[:, 2]
    return out


def _layout(ndc: np.ndarray, slot: int, total: int) -> np.ndarray:
    # No transformation — the mesh aligns 1:1 with the camera feed underneath.
    # Each landmark sits exactly where the webcam saw it in normalized image space.
    return ndc


def _sort_tris_back_to_front(ndc: np.ndarray) -> list[tuple]:
    return sorted(
        config.HAND_TRIANGLES,
        key=lambda t: (ndc[t[0], 2] + ndc[t[1], 2] + ndc[t[2], 2]) / 3.0,
    )


def _px_to_ndc(px: tuple[int, int], w: int, h: int) -> tuple[float, float]:
    """Convert pixel coords (origin bottom-left, as pyglet uses) to NDC [-1,1]."""
    return (px[0] / w * 2.0 - 1.0, px[1] / h * 2.0 - 1.0)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class WireframeRenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        w, h = config.WINDOW_SIZE

        self._camera_prog = ctx.program(vertex_shader=_VERT_CAMERA, fragment_shader=_FRAG_CAMERA)
        self._face_prog   = ctx.program(vertex_shader=_VERT_FACE,   fragment_shader=_FRAG_FACE)
        self._line_prog   = ctx.program(
            vertex_shader=_VERT_LINES,
            geometry_shader=_GEOM_LINES,
            fragment_shader=_FRAG_FLAT,
        )
        self._point_prog  = ctx.program(vertex_shader=_VERT_POINTS, fragment_shader=_FRAG_POINTS)
        self._pulse_prog  = ctx.program(vertex_shader=_VERT_QUAD,   fragment_shader=_FRAG_PULSE)

        # Fullscreen quad used for both camera bg and pulse overlay
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=np.float32)
        self._quad_vbo = ctx.buffer(quad.tobytes())
        self._camera_vao = ctx.vertex_array(self._camera_prog, [(self._quad_vbo, "2f", "in_pos")])
        self._pulse_vao  = ctx.vertex_array(self._pulse_prog,  [(self._quad_vbo, "2f", "in_pos")])

        # Camera texture — sized to PROCESS_RESOLUTION, updated each frame
        cw, ch = config.PROCESS_RESOLUTION
        self._cam_tex = ctx.texture((cw, ch), 3)  # RGB
        self._cam_tex.filter = moderngl.LINEAR, moderngl.LINEAR
        self._cam_tex_dirty = False

        nh = config.MAX_HANDS
        nt = len(config.HAND_TRIANGLES)
        nc = len(_CONNECTIONS)

        self._face_vbo = ctx.buffer(reserve=nh * nt * 3 * 3 * 4)
        self._face_vao = ctx.vertex_array(self._face_prog, [(self._face_vbo, "3f", "in_pos")])

        self._edge_vbo = ctx.buffer(reserve=nh * nc * 2 * 3 * 4)
        self._edge_vao = ctx.vertex_array(self._line_prog, [(self._edge_vbo, "3f", "in_pos")])

        self._pt_vbo = ctx.buffer(reserve=nh * 21 * 3 * 4)
        self._pt_vao = ctx.vertex_array(self._point_prog, [(self._pt_vbo, "3f", "in_pos")])

        # Small reusable VBO for 2-point overlay lines (pinch gap, vol line)
        self._overlay_line_vbo = ctx.buffer(reserve=2 * 3 * 4)
        self._overlay_line_vao = ctx.vertex_array(
            self._line_prog, [(self._overlay_line_vbo, "3f", "in_pos")]
        )
        # Small reusable VBO for single overlay points
        self._overlay_pt_vbo = ctx.buffer(reserve=2 * 3 * 4)
        self._overlay_pt_vao = ctx.vertex_array(
            self._point_prog, [(self._overlay_pt_vbo, "3f", "in_pos")]
        )

        self._resolution = (float(w), float(h))

        # EMA smoothing cache
        self._smoothed_hands: list[np.ndarray] = []

        self._label: object = None
        self._vol_level: int = 0
        self._pinch_thumb_px: tuple[int, int] | None = None
        self._pinch_index_px: tuple[int, int] | None = None
        self._search_label: object = None
        self._fps_label: object = None
        self._vol_label: object = None
        self._init_labels()

    def _init_labels(self):
        try:
            import pyglet
            cx = config.WINDOW_SIZE[0] // 2
            h = config.WINDOW_SIZE[1]
            self._label = pyglet.text.Label(
                "",
                font_name="monospace",
                font_size=13,
                x=cx,
                y=18,
                anchor_x="center",
                anchor_y="center",
                color=(220, 255, 255, 230),
            )
            self._search_label = pyglet.text.Label(
                "searching...",
                font_name="monospace",
                font_size=13,
                x=cx,
                y=18,
                anchor_x="center",
                anchor_y="center",
                color=(180, 230, 240, 200),
            )
            if config.SHOW_FPS:
                self._fps_label = pyglet.text.Label(
                    "",
                    font_name="monospace",
                    font_size=10,
                    x=8,
                    y=h - 14,
                    anchor_x="left",
                    anchor_y="center",
                    color=(220, 255, 255, 200),
                )
            self._vol_label = pyglet.text.Label(
                "",
                font_name="monospace",
                font_size=14,
                bold=True,
                x=0,
                y=0,
                anchor_x="left",
                anchor_y="center",
                color=config.VOL_LABEL_COLOR,
            )
        except Exception:
            pass

    def update_camera_frame(self, rgb_frame: np.ndarray):
        """Upload a new webcam RGB frame to the GPU texture. Call from main thread."""
        if rgb_frame is None:
            return
        # Flip vertically — OpenGL UV origin is bottom-left, image origin is top-left
        flipped = np.ascontiguousarray(np.flipud(rgb_frame))
        self._cam_tex.write(flipped.tobytes())

    def _write_hand_data(self, hands: list) -> tuple:
        # Sort by wrist x so hand identity is stable across frames regardless of
        # MediaPipe's arbitrary ordering, preventing EMA from blending across hands.
        hands = sorted(hands, key=lambda h: h[0, 0])

        # EMA smoothing
        if len(self._smoothed_hands) != len(hands):
            self._smoothed_hands = [h.copy() for h in hands]
        else:
            a = config.SMOOTHING
            self._smoothed_hands = [
                a * fresh + (1.0 - a) * cached
                for fresh, cached in zip(hands, self._smoothed_hands)
            ]
        hands = self._smoothed_hands

        all_faces: list[np.ndarray] = []
        all_edges: list[np.ndarray] = []
        all_pts: list[np.ndarray] = []
        n = len(hands)

        for i, lm in enumerate(hands):
            ndc = _layout(_lm_to_ndc(lm), i, n)
            all_pts.append(ndc)
            for a_idx, b_idx in _CONNECTIONS:
                all_edges.append(ndc[a_idx])
                all_edges.append(ndc[b_idx])
            for tri in _sort_tris_back_to_front(ndc):
                a, b, c = tri
                all_faces.extend([ndc[a], ndc[b], ndc[c]])

        n_faces = n_edges = n_pts = 0
        if all_faces:
            self._face_vbo.write(np.array(all_faces, dtype=np.float32).tobytes())
            n_faces = len(all_faces) // 3
        if all_edges:
            self._edge_vbo.write(np.array(all_edges, dtype=np.float32).tobytes())
            n_edges = len(all_edges)
        if all_pts:
            self._pt_vbo.write(np.concatenate(all_pts).astype(np.float32).tobytes())
            n_pts = sum(len(h) for h in hands)

        return n_pts, n_edges, n_faces

    def render(self, hands: list, gestures: list[str], t: float, fps: float | None = None):
        ctx = self.ctx
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        ctx.clear(0.0, 0.0, 0.0, 1.0)

        rx, ry = self._resolution

        # --- Draw live camera feed as background ---
        self._cam_tex.use(location=0)
        self._camera_prog["u_tex"] = 0
        self._camera_vao.render(moderngl.TRIANGLE_STRIP)

        if not hands:
            # Pulse ring overlay on top of camera
            self._pulse_prog["u_time"] = t
            self._pulse_prog["u_color"] = config.PULSE_COLOR
            self._pulse_prog["u_resolution"] = (rx, ry)
            self._pulse_vao.render(moderngl.TRIANGLE_STRIP)
            self._draw_search_label()
            self._draw_fps(fps)
            return

        n_pts, n_edges, n_faces = self._write_hand_data(hands)

        # Filled triangle faces
        self._face_prog["u_depth_scale"] = config.DEPTH_SCALE
        self._face_prog["u_color"] = config.FACE_COLOR
        self._face_vao.render(moderngl.TRIANGLES, vertices=n_faces * 3)

        # Glow pass
        self._line_prog["u_resolution"] = (rx, ry)
        self._line_prog["u_depth_scale"] = config.DEPTH_SCALE
        self._line_prog["u_thickness"] = config.GLOW_THICKNESS
        self._line_prog["u_color"] = config.EDGE_COLOR_GLOW
        self._edge_vao.render(moderngl.LINES, vertices=n_edges)

        # Bright edge pass
        self._line_prog["u_thickness"] = config.EDGE_THICKNESS
        self._line_prog["u_color"] = config.EDGE_COLOR_BRIGHT
        self._edge_vao.render(moderngl.LINES, vertices=n_edges)

        # Joint dots
        self._point_prog["u_depth_scale"] = config.DEPTH_SCALE
        self._point_prog["u_point_size"] = config.JOINT_SIZE
        self._point_prog["u_color"] = config.JOINT_COLOR
        self._pt_vao.render(moderngl.POINTS, vertices=n_pts)

        self._draw_gesture_label(gestures)
        self._draw_fps(fps)
        self._draw_pinch_gap()
        self._draw_volume_bar()

    def _draw_gesture_label(self, gestures: list[str]):
        if not gestures or self._label is None:
            return
        from beans.gesture import GESTURE_ACTIONS
        parts = [f"{g}  {GESTURE_ACTIONS.get(g, '—')}" for g in gestures]
        self._label.text = "  |  ".join(parts)
        self._label.draw()

    def _draw_search_label(self):
        if self._search_label is not None:
            self._search_label.draw()

    def _draw_fps(self, fps: float | None):
        if self._fps_label is not None and fps is not None:
            self._fps_label.text = f"{fps:.0f} fps"
            self._fps_label.draw()

    def set_volume_display(self, level: int) -> None:
        self._vol_level = level

    def set_pinch_gap(self, thumb_px: tuple[int, int], index_px: tuple[int, int], **_) -> None:
        self._pinch_thumb_px = thumb_px
        self._pinch_index_px = index_px

    def clear_pinch_gap(self) -> None:
        self._pinch_thumb_px = None
        self._pinch_index_px = None

    def _overlay_line(self, a_px, b_px, color_01: tuple, thickness: float) -> None:
        """Draw a single line between two pixel-coord points using the hand line shader."""
        w, h = config.WINDOW_SIZE
        ax, ay = _px_to_ndc(a_px, w, h)
        bx, by = _px_to_ndc(b_px, w, h)
        verts = np.array([[ax, ay, 0.0], [bx, by, 0.0]], dtype=np.float32)
        self._overlay_line_vbo.write(verts.tobytes())
        rx, ry = self._resolution
        self._line_prog["u_resolution"] = (rx, ry)
        self._line_prog["u_depth_scale"] = 0.0
        self._line_prog["u_thickness"] = thickness
        self._line_prog["u_color"] = color_01
        self._overlay_line_vao.render(moderngl.LINES, vertices=2)

    def _overlay_dots(self, *px_coords, color_01: tuple, size: float) -> None:
        """Draw dots at the given pixel-coord positions using the point shader."""
        w, h = config.WINDOW_SIZE
        verts = np.array(
            [[*_px_to_ndc(p, w, h), 0.0] for p in px_coords], dtype=np.float32
        )
        self._overlay_pt_vbo.write(verts.tobytes())
        self._point_prog["u_depth_scale"] = 0.0
        self._point_prog["u_point_size"] = size
        self._point_prog["u_color"] = color_01
        self._overlay_pt_vao.render(moderngl.POINTS, vertices=len(px_coords))

    def _draw_pinch_gap(self) -> None:
        if self._pinch_thumb_px is None or self._pinch_index_px is None:
            return
        c = tuple(v / 255.0 for v in config.PINCH_GAP_ACTIVE_COLOR)
        self._overlay_line(self._pinch_thumb_px, self._pinch_index_px, c, thickness=2.5)
        self._overlay_dots(self._pinch_thumb_px, self._pinch_index_px, color_01=c, size=10.0)

    def _draw_volume_bar(self) -> None:
        # Show volume % label near the pinch gap midpoint when hand is visible
        if self._vol_label is None:
            return
        if self._pinch_thumb_px is None or self._pinch_index_px is None:
            return
        tx, ty = self._pinch_thumb_px
        ix, iy = self._pinch_index_px
        self._vol_label.text = f"{self._vol_level}%"
        self._vol_label.x = (tx + ix) // 2 + 14
        self._vol_label.y = (ty + iy) // 2
        self._vol_label.color = config.VOL_LABEL_COLOR
        self._vol_label.draw()

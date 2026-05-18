import glfw


_drag_start_cursor: tuple[float, float] | None = None
_drag_start_win: tuple[int, int] | None = None


def _on_mouse_button(window, button, action, mods):
    global _drag_start_cursor, _drag_start_win
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            _drag_start_cursor = glfw.get_cursor_pos(window)
            _drag_start_win = glfw.get_window_pos(window)
        else:
            _drag_start_cursor = None
            _drag_start_win = None


def _on_cursor_pos(window, x, y):
    if _drag_start_cursor is None:
        return
    dx = x - _drag_start_cursor[0]
    dy = y - _drag_start_cursor[1]
    wx, wy = _drag_start_win
    glfw.set_window_pos(window, int(wx + dx), int(wy + dy))


def create_window(title: str = "beans", size: tuple[int, int] = (400, 400)):
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.TRANSPARENT_FRAMEBUFFER, glfw.TRUE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw.window_hint(glfw.FLOATING, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.SAMPLES, 4)

    # Set app ID so Hyprland windowrulev2 can target class:^(beans)$
    glfw.window_hint_string(glfw.X11_CLASS_NAME, "beans")
    glfw.window_hint_string(glfw.X11_INSTANCE_NAME, "beans")
    glfw.window_hint_string(glfw.WAYLAND_APP_ID, "beans")

    window = glfw.create_window(size[0], size[1], title, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.set_mouse_button_callback(window, _on_mouse_button)
    glfw.set_cursor_pos_callback(window, _on_cursor_pos)

    return window


def show(window):
    glfw.show_window(window)


def hide(window):
    glfw.hide_window(window)

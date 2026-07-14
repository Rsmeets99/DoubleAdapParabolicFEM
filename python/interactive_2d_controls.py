from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np


@dataclass
class CyclicNavigator:
    count: int
    index: int = 0

    def clamp(self) -> int:
        if self.count <= 0:
            self.index = 0
        else:
            self.index %= self.count
        return self.index

    def move(self, delta: int) -> int:
        self.index += delta
        return self.clamp()

    def set_count(self, count: int) -> int:
        self.count = count
        return self.clamp()


def is_interactive_backend(fig=None) -> bool:
    backend = ""
    if fig is not None:
        try:
            backend = fig.canvas.get_backend()
        except Exception:
            backend = ""
    if not backend:
        try:
            import matplotlib

            backend = matplotlib.get_backend()
        except Exception:
            return False
    return backend.lower() != "agg"


def set_window_title(fig, title: str) -> None:
    manager = getattr(fig.canvas, "manager", None)
    if manager is not None and hasattr(manager, "set_window_title"):
        manager.set_window_title(title)


def add_status_text(fig, text: str = "", *, x: float = 0.01, y: float = 0.015):
    return fig.text(x, y, text, ha="left", va="bottom", fontsize=9)


def set_status(status_artist, text: str, fig=None) -> None:
    if status_artist is None:
        return
    status_artist.set_text(text)
    if fig is not None:
        fig.canvas.draw_idle()


def set_textbox_value(textbox, text: str) -> None:
    if textbox is None:
        return
    try:
        textbox.set_val(str(text))
    except Exception:
        text_disp = getattr(textbox, "text_disp", None)
        if text_disp is not None:
            text_disp.set_text(str(text))


def install_keymap(
    fig,
    handlers: Mapping[str, Callable[[], str | None]],
    *,
    help_text: str | None = None,
    status_artist=None,
    print_help: bool = True,
) -> int:
    def on_key(event):
        if event.key == "h" and help_text is not None:
            if print_help:
                print(help_text)
            set_status(status_artist, help_text, fig)
            return
        handler = handlers.get(event.key)
        if handler is None:
            return
        message = handler()
        if message is not None:
            set_status(status_artist, str(message), fig)
        else:
            fig.canvas.draw_idle()

    return fig.canvas.mpl_connect("key_press_event", on_key)


def remove_artists(artists: Sequence) -> None:
    for artist in list(artists):
        if artist is None:
            continue
        try:
            artist.remove()
        except (NotImplementedError, ValueError, RuntimeError):
            pass


def style_axis_2d(ax, title: str | None = None) -> None:
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#e5e7eb", linewidth=0.6)


def reset_axis_2d(
    ax,
    points: np.ndarray,
    *,
    pad: float = 0.05,
    fallback: tuple[float, float, float, float] = (-0.05, 1.05, -0.05, 1.05),
) -> None:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        ax.set_xlim(fallback[0], fallback[1])
        ax.set_ylim(fallback[2], fallback[3])
        return
    points = points.reshape((-1, 2))
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        ax.set_xlim(fallback[0], fallback[1])
        ax.set_ylim(fallback[2], fallback[3])
        return
    visible = points[finite]
    xmin, ymin = visible.min(axis=0)
    xmax, ymax = visible.max(axis=0)
    dx = max(float(xmax - xmin), 1.0e-12)
    dy = max(float(ymax - ymin), 1.0e-12)
    ax.set_xlim(float(xmin) - pad * dx, float(xmax) + pad * dx)
    ax.set_ylim(float(ymin) - pad * dy, float(ymax) + pad * dy)


def style_axis_3d(
    ax,
    title: str | None = None,
    *,
    elev: float = 23,
    azim: float = -55,
    roll: float | None = 0.0,
) -> None:
    if title is not None:
        ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    if roll is None:
        ax.view_init(elev=elev, azim=azim)
        return
    try:
        ax.view_init(elev=elev, azim=azim, roll=roll)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)


def reset_axis_3d(ax, points: np.ndarray, *, min_radius: float = 0.5) -> None:
    points = np.asarray(points, dtype=float)
    if points.size == 0:
        return
    points = points.reshape((-1, 3))
    finite = np.all(np.isfinite(points), axis=1)
    if not np.any(finite):
        return
    visible = points[finite]
    mins = visible.min(axis=0)
    maxs = visible.max(axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = min_radius
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    ax.set_box_aspect((1, 1, 1))

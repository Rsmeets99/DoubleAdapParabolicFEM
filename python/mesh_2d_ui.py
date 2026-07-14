from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import TextBox
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from binary_readers import require_mesh_is_2p1d
from interactive_2d_controls import (
    add_status_text,
    install_keymap,
    reset_axis_3d,
    set_status,
    set_textbox_value,
    set_window_title,
    style_axis_3d,
)
from mesh_2d_geometry import (
    all_cell_faces,
    cell_centers,
    dof_rows_for_cell,
    resolve_cell_row,
)


def set_axes_equal_3d(ax, points: np.ndarray) -> None:
    reset_axis_3d(ax, points)


def style_3d_axis(
    ax,
    title: str | None,
    *,
    elev: float = 23,
    azim: float = -55,
    roll: float | None = 0.0,
    show_axes: bool = True,
) -> None:
    style_axis_3d(ax, title, elev=elev, azim=azim, roll=roll)
    if not show_axes:
        ax.set_axis_off()


def draw_prisms_3d(
    ax,
    mesh_data: dict,
    rows: np.ndarray | None = None,
    *,
    alpha: float = 0.22,
    color_by_cell_index: bool = True,
    selected_row: int | None = None,
    selected_alpha: float = 0.56,
    reset_points: np.ndarray | None = None,
) -> Poly3DCollection:
    faces, owners = all_cell_faces(mesh_data, rows)
    n_cells = mesh_data["cell_ids"].shape[0]

    cmap = plt.get_cmap("viridis")
    facecolors = []
    for owner in owners:
        if color_by_cell_index:
            color = np.array(cmap(0.12 + 0.78 * (owner / max(n_cells - 1, 1))))
            color[3] = alpha
        else:
            color = np.array([0.88, 0.88, 0.88, alpha])
        if selected_row is not None and owner == selected_row:
            color = np.array([1.0, 0.46, 0.10, selected_alpha])
        facecolors.append(color)

    collection = Poly3DCollection(
        faces,
        facecolors=facecolors,
        edgecolors=(0.05, 0.05, 0.05, 0.72),
        linewidths=0.65,
    )
    ax.add_collection3d(collection)

    points = (
        np.asarray(reset_points, dtype=np.float64)
        if reset_points is not None
        else np.vstack([face for face in faces])
        if faces
        else np.empty((0, 3))
    )
    set_axes_equal_3d(ax, points)
    return collection


def draw_dofs_3d(
    ax,
    dof_data: dict | None,
    *,
    gids: np.ndarray | None = None,
    show_labels: bool = False,
) -> None:
    if dof_data is None:
        return

    coords = dof_data["dof_coords"]
    flags = dof_data["is_constrained"]

    if gids is None:
        gids = np.arange(coords.shape[0], dtype=np.int32)

    if gids.size == 0:
        return

    unconstrained = gids[~flags[gids]]
    constrained = gids[flags[gids]]

    if unconstrained.size:
        ax.scatter(
            coords[unconstrained, 0],
            coords[unconstrained, 1],
            coords[unconstrained, 2],
            s=18,
            marker="o",
            c="#1f77b4",
            depthshade=False,
        )

    if constrained.size:
        ax.scatter(
            coords[constrained, 0],
            coords[constrained, 1],
            coords[constrained, 2],
            s=34,
            marker="x",
            c="#d62728",
            depthshade=False,
        )

    if show_labels:
        for gid in gids:
            x, y, t = coords[int(gid)]
            ax.text(x, y, t, str(int(gid)), fontsize=7)


def plot_mesh_with_cell_selector(
    mesh_data: dict,
    dof_data: dict | None = None,
    *,
    initial_cell: int | None = None,
    show_dof_labels: bool = False,
    overview_only: bool = False,
    show_axes: bool = True,
    view_elev: float = 23,
    view_azim: float = -55,
    view_roll: float | None = 0.0,
    color_by_cell_index: bool = True,
):
    require_mesh_is_2p1d(mesh_data, "plot_mesh_with_cell_selector")

    n_cells = mesh_data["cell_ids"].shape[0]
    initial_row = 0
    if initial_cell is not None:
        initial_row = resolve_cell_row(mesh_data, str(initial_cell))

    fig = plt.figure(figsize=(7.6, 7.0) if overview_only else (13.5, 6.8))
    set_window_title(fig, "2+1D mesh viewer")
    if overview_only:
        ax_full = fig.add_subplot(1, 1, 1, projection="3d")
        ax_cell = None
        fig.subplots_adjust(bottom=0.02, left=0.02, right=0.98, top=0.98)
        textbox = None
        status = None
    else:
        ax_full = fig.add_subplot(1, 2, 1, projection="3d")
        ax_cell = fig.add_subplot(1, 2, 2, projection="3d")
        fig.subplots_adjust(bottom=0.16, left=0.03, right=0.98, top=0.92, wspace=0.08)

        textbox_ax = fig.add_axes([0.40, 0.045, 0.20, 0.045])
        textbox = TextBox(
            textbox_ax,
            "Cell id",
            initial=str(int(mesh_data["cell_ids"][initial_row])),
        )
        status = add_status_text(fig, x=0.62, y=0.048)
    help_text = (
        "left/right: previous/next cell | l: DoF labels | "
        "r: reset/redraw | h: help"
    )

    centers = cell_centers(mesh_data)
    all_faces, _ = all_cell_faces(mesh_data)
    all_points = np.vstack(all_faces) if all_faces else np.empty((0, 3))
    state = {
        "row": initial_row,
        "show_dof_labels": show_dof_labels,
        "syncing_textbox": False,
    }

    def sync_textbox() -> None:
        state["syncing_textbox"] = True
        try:
            set_textbox_value(textbox, str(int(mesh_data["cell_ids"][state["row"]])))
        finally:
            state["syncing_textbox"] = False

    def redraw() -> None:
        row = state["row"]
        cell_id = int(mesh_data["cell_ids"][row])
        sync_textbox()

        ax_full.clear()
        draw_prisms_3d(
            ax_full,
            mesh_data,
            color_by_cell_index=color_by_cell_index,
            selected_row=None if overview_only else row,
            reset_points=all_points,
        )
        draw_dofs_3d(ax_full, dof_data)
        if not overview_only:
            ax_full.text(
                centers[row, 0],
                centers[row, 1],
                centers[row, 2],
                str(cell_id),
                fontsize=9,
                color="#111111",
            )
        style_3d_axis(
            ax_full,
            None if not show_axes else f"2+1D mesh | {n_cells} cells",
            elev=view_elev,
            azim=view_azim,
            roll=view_roll,
            show_axes=show_axes,
        )

        if ax_cell is None:
            fig.canvas.draw_idle()
            return

        ax_cell.clear()
        draw_prisms_3d(
            ax_cell,
            mesh_data,
            np.asarray([row], dtype=np.int32),
            alpha=0.62,
            color_by_cell_index=color_by_cell_index,
            selected_row=row,
            selected_alpha=0.72,
        )
        draw_dofs_3d(
            ax_cell,
            dof_data,
            gids=(
                dof_rows_for_cell(mesh_data, dof_data, cell_id)
                if dof_data is not None
                else None
            ),
            show_labels=bool(state["show_dof_labels"]),
        )
        style_3d_axis(
            ax_cell,
            None if not show_axes else f"cell id {cell_id} | row {row}",
            elev=view_elev,
            azim=view_azim,
            roll=view_roll,
            show_axes=show_axes,
        )
        set_status(
            status,
            f"cell id {cell_id} | row {row + 1}/{n_cells}. {help_text}",
        )
        fig.canvas.draw_idle()

    def on_submit(text: str) -> None:
        if state["syncing_textbox"]:
            return
        try:
            state["row"] = resolve_cell_row(mesh_data, text)
        except Exception as exc:
            set_status(status, str(exc), fig)
            return
        redraw()

    def previous_cell():
        state["row"] = (int(state["row"]) - 1) % n_cells
        redraw()
        return None

    def next_cell():
        state["row"] = (int(state["row"]) + 1) % n_cells
        redraw()
        return None

    def toggle_labels():
        state["show_dof_labels"] = not state["show_dof_labels"]
        redraw()
        return None

    def reset_view():
        redraw()
        return None

    handlers = {
        "previous_cell": previous_cell,
        "next_cell": next_cell,
        "toggle_labels": toggle_labels,
        "reset_view": reset_view,
    }

    if textbox is not None:
        textbox.on_submit(on_submit)
        install_keymap(
            fig,
            {
                "left": handlers["previous_cell"],
                "right": handlers["next_cell"],
                "l": handlers["toggle_labels"],
                "r": handlers["reset_view"],
            },
            help_text=help_text,
            status_artist=status,
        )
    fig._adappfem_2d_state = state
    fig._adappfem_2d_redraw = redraw
    fig._adappfem_2d_handlers = handlers
    redraw()
    axes = (ax_full,) if ax_cell is None else (ax_full, ax_cell)
    return fig, axes


_set_axes_equal_3d = set_axes_equal_3d
_style_axis = style_3d_axis
_draw_prisms = draw_prisms_3d
_draw_dofs = draw_dofs_3d

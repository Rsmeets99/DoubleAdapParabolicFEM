from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from artifact_paths import repository_data_dir, require_generator_file
from binary_readers import load_dofs_binary, load_mesh_binary
from mesh_2d_ui import plot_mesh_with_cell_selector
from plot_cli import add_display_options, finish_figures

VIEW_PRESETS = {
    "default": (23.0, -55.0, 0.0),
    "top": (90.0, -90.0, 0.0),
    "bottom": (-90.0, -90.0, 0.0),
}


def load_default_artifacts(policy: str) -> tuple[dict, dict]:
    data_dir = repository_data_dir()
    mesh_path = require_generator_file(
        data_dir / f"mesh_2d_{policy}.bin",
        "mesh_2d_generator",
    )
    dofs_path = require_generator_file(
        data_dir / f"dofs_2d_{policy}.bin",
        "mesh_2d_generator",
    )
    return load_mesh_binary(mesh_path), load_dofs_binary(dofs_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 2+1D triangular-prism mesh artifacts.")
    parser.add_argument("--mesh", type=Path, help="Path to a 2+1D mesh binary.")
    parser.add_argument("--dofs", type=Path, help="Path to the matching 2+1D DoF binary.")
    parser.add_argument(
        "--policy",
        choices=("spaceonly", "spacetime"),
        default="spaceonly",
        help="Default generated artifact pair to load when --mesh is omitted.",
    )
    parser.add_argument("--cell", type=int, default=None, help="Initial cell id or row.")
    parser.add_argument("--no-dofs", action="store_true", help="Do not plot DoFs.")
    parser.add_argument("--dof-labels", action="store_true", help="Label selected-cell DoFs.")
    parser.add_argument(
        "--left-only",
        "--overview-only",
        dest="overview_only",
        action="store_true",
        help="Show only the full-mesh overview panel, without the selected-cell panel.",
    )
    parser.add_argument(
        "--hide-axes",
        "--no-axes",
        dest="show_axes",
        action="store_false",
        help="Hide axis decorations and titles in the 3D view.",
    )
    parser.set_defaults(show_axes=True)
    parser.add_argument(
        "--uniform-color",
        "--no-cell-colors",
        dest="color_by_cell_index",
        action="store_false",
        help="Use one uniform mesh color instead of row-index cell coloring.",
    )
    parser.set_defaults(color_by_cell_index=True)
    parser.add_argument(
        "--view-preset",
        choices=tuple(VIEW_PRESETS),
        default="default",
        help="Initial 3D viewing preset. 'bottom' views the mesh from the t=0 side.",
    )
    parser.add_argument(
        "--view-elev",
        type=float,
        default=None,
        help="Initial Matplotlib 3D elevation angle in degrees. Overrides --view-preset.",
    )
    parser.add_argument(
        "--view-azim",
        type=float,
        default=None,
        help="Initial Matplotlib 3D azimuth angle in degrees. Overrides --view-preset.",
    )
    parser.add_argument(
        "--view-roll",
        type=float,
        default=None,
        help="Initial Matplotlib 3D roll angle in degrees. Overrides --view-preset.",
    )
    add_display_options(parser)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Construct, navigate, redraw, and close the viewer without showing it.",
    )
    return parser.parse_args()


def resolve_view_angles(args: argparse.Namespace) -> tuple[float, float, float]:
    elev, azim, roll = VIEW_PRESETS[args.view_preset]
    if args.view_elev is not None:
        elev = float(args.view_elev)
    if args.view_azim is not None:
        azim = float(args.view_azim)
    if args.view_roll is not None:
        roll = float(args.view_roll)
    return elev, azim, roll


def main() -> None:
    args = parse_args()

    if args.mesh is None:
        mesh_data, dof_data = load_default_artifacts(args.policy)
    else:
        mesh_data = load_mesh_binary(args.mesh)
        dof_data = None if args.dofs is None else load_dofs_binary(args.dofs)

    if args.no_dofs:
        dof_data = None

    print("Loaded 2+1D mesh:")
    for key, value in mesh_data["header"].items():
        print(f"  {key}: {value}")

    if dof_data is not None:
        print("Loaded 2+1D dofs:")
        for key, value in dof_data["header"].items():
            print(f"  {key}: {value}")

    view_elev, view_azim, view_roll = resolve_view_angles(args)
    plot_mesh_with_cell_selector(
        mesh_data,
        dof_data,
        initial_cell=args.cell,
        show_dof_labels=args.dof_labels,
        overview_only=args.overview_only,
        show_axes=args.show_axes,
        view_elev=view_elev,
        view_azim=view_azim,
        view_roll=view_roll,
        color_by_cell_index=args.color_by_cell_index,
    )
    if args.self_test:
        fig = plt.gcf()
        handlers = getattr(fig, "_adappfem_2d_handlers", {})
        for name in ("next_cell", "previous_cell", "toggle_labels", "reset_view"):
            handler = handlers.get(name)
            if handler is not None:
                handler()
        redraw = getattr(fig, "_adappfem_2d_redraw", None)
        if redraw is not None:
            redraw()
        expected_axes = 1 if args.overview_only else 2
        if len(fig.axes) < expected_axes:
            raise RuntimeError("expected overview and selected-cell axes")
        if not any(ax.collections for ax in fig.axes):
            raise RuntimeError("expected 2+1D mesh artists were not constructed")
        fig.canvas.draw()
        plt.close(fig)
        print("OK: 2+1D mesh viewer constructed and redrawn.")
        return

    finish_figures(show=not args.no_show, save_png=args.save_png)


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

from artifact_paths import repository_data_dir, require_generator_file
from binary_readers import (
    load_dofs_binary,
    load_mesh_binary,
    require_1p1d_space_time,
)
from mesh_plot_geometry import mesh_polygons_1p1d
from mesh_plot_ui import (
    plot_dofs_1p1d,
    plot_local_cell_dofs_1p1d,
    plot_mesh_1p1d,
    plot_mesh_with_neighbour_navigation,
    print_dof_info,
)
from plot_cli import add_display_options, finish_figures, install_noop_show_when


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="View 1+1D mesh and DoF generator artifacts.",
    )
    parser.add_argument("--mesh", type=Path, default=None, help="Path to a 1+1D mesh binary.")
    parser.add_argument("--dofs", type=Path, default=None, help="Path to the matching DoF binary.")
    parser.add_argument(
        "--policy",
        choices=("spacetime", "spaceonly"),
        default="spacetime",
        help="Default generated artifact pair to load when --mesh is omitted.",
    )
    parser.add_argument("--no-dofs", action="store_true", help="Do not load or plot DoFs.")
    parser.add_argument("--dof-labels", action="store_true", help="Label selected-cell DoFs.")
    parser.add_argument(
        "--mesh-only",
        "--static-mesh",
        dest="mesh_only",
        action="store_true",
        help="Draw only the mesh, without neighbour navigation, selection, or DoF overlays.",
    )
    parser.add_argument(
        "--hide-axes",
        "--no-axes",
        dest="show_axes",
        action="store_false",
        help="Hide axis decorations and titles.",
    )
    parser.set_defaults(show_axes=True)
    parser.add_argument(
        "--tight-frame",
        action="store_true",
        help="Remove subplot padding around the mesh.",
    )
    parser.add_argument(
        "--uniform-color",
        "--no-cell-colors",
        dest="color_by_cell_index",
        action="store_false",
        help="Use one uniform mesh color instead of row-index Viridis coloring.",
    )
    parser.add_argument(
        "--cell-colors",
        "--viridis-cell-colors",
        dest="color_by_cell_index",
        action="store_true",
        help="Color cells by row index with Viridis shades.",
    )
    parser.set_defaults(color_by_cell_index=True)
    add_display_options(parser)
    return parser.parse_args(argv)


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        args = parse_args()

    restore_show = install_noop_show_when(args.no_show or args.save_png is not None)

    data_dir = repository_data_dir()
    if args.mesh is None:
        mesh_path = require_generator_file(
            data_dir / f"mesh_test_{args.policy}.bin",
            "mesh_generator",
        )
        dof_path = require_generator_file(
            data_dir / f"dofs_test_{args.policy}.bin",
            "mesh_generator",
        )
    else:
        mesh_path = args.mesh
        dof_path = args.dofs

    mesh_data = load_mesh_binary(mesh_path)
    dof_data = (
        None
        if args.no_dofs or args.mesh_only or dof_path is None
        else load_dofs_binary(dof_path)
    )

    print("Loaded mesh:")
    for key, value in mesh_data["header"].items():
        print(f"  {key}: {value}")

    if dof_data is not None:
        print("Loaded dofs:")
        for key, value in dof_data["header"].items():
            print(f"  {key}: {value}")

        print_dof_info(dof_data)

    if args.mesh_only:
        plot_mesh_1p1d(
            mesh_data,
            show_axes=args.show_axes,
            tight_frame=args.tight_frame,
            color_by_cell_index=args.color_by_cell_index,
        )
    else:
        plot_mesh_with_neighbour_navigation(
            mesh_data,
            show_axes=args.show_axes,
            tight_frame=args.tight_frame,
            color_by_cell_index=args.color_by_cell_index,
        )

    if dof_data is not None and not args.mesh_only:
        plot_local_cell_dofs_1p1d(
            mesh_data,
            dof_data,
            show_labels=args.dof_labels,
            jitter=0.0,
            label_offset_fraction=0.012,
        )

    if args.mesh_only or args.no_show or args.save_png is not None:
        restore_show()
        finish_figures(show=not args.no_show, save_png=args.save_png)


if __name__ == "__main__":
    main()

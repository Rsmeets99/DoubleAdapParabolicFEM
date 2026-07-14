import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection

from mesh_plot_geometry import (
    build_neighbour_masks,
    mesh_polygons_1p1d,
    reconstruct_local_occurrence_coordinates_1p1d,
)


def _mesh_facecolors(n_cells: int, *, color_by_cell_index: bool) -> np.ndarray:
    if not color_by_cell_index:
        return np.tile(np.array([0.88, 0.88, 0.88, 1.0]), (n_cells, 1))

    cmap = plt.get_cmap("viridis")
    rows = np.arange(n_cells, dtype=float)
    denom = max(n_cells - 1, 1)
    return np.asarray([cmap(0.12 + 0.78 * (row / denom)) for row in rows])


def _set_tight_mesh_limits(ax, polys: np.ndarray) -> None:
    points = np.asarray(polys, dtype=float).reshape((-1, 2))
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    dx = max(float(xmax - xmin), 1.0e-12)
    dy = max(float(ymax - ymin), 1.0e-12)
    pad = 0.0025 * max(dx, dy)
    ax.set_xlim(float(xmin) - pad, float(xmax) + pad)
    ax.set_ylim(float(ymin) - pad, float(ymax) + pad)


def plot_mesh_1p1d(
    mesh_data: dict,
    ax=None,
    *,
    show_axes: bool = True,
    tight_frame: bool = False,
    color_by_cell_index: bool = True,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    polys = mesh_polygons_1p1d(mesh_data)
    collection = PolyCollection(
        polys,
        edgecolors="black",
        linewidths=0.8,
        facecolors=_mesh_facecolors(
            len(polys),
            color_by_cell_index=color_by_cell_index,
        ),
    )

    ax.add_collection(collection)
    ax.autoscale()
    ax.set_aspect("equal")
    if show_axes:
        ax.set_xlabel("Space")
        ax.set_ylabel("Time")
        ax.set_title("1+1D Mesh")
    else:
        ax.set_axis_off()
        ax.set_title("")
    if tight_frame:
        _set_tight_mesh_limits(ax, polys)
        collection.set_clip_on(False)
        fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        fig._adappfem_savefig_pad_inches = 0.0

    return fig, ax, collection


def plot_mesh_with_neighbour_navigation(
    mesh_data: dict,
    *,
    show_axes: bool = True,
    tight_frame: bool = False,
    color_by_cell_index: bool = True,
):
    n_cells = mesh_data["cell_ids"].shape[0]
    spatial_mask, temporal_mask = build_neighbour_masks(mesh_data)

    fig, ax, collection = plot_mesh_1p1d(
        mesh_data,
        show_axes=show_axes,
        tight_frame=tight_frame,
        color_by_cell_index=color_by_cell_index,
    )

    base_colors = _mesh_facecolors(
        n_cells,
        color_by_cell_index=color_by_cell_index,
    )
    selected_color = np.array([1.0, 0.55, 0.0, 1.0])
    spatial_color = np.array([0.2, 0.4, 1.0, 1.0])
    temporal_color = np.array([0.2, 0.8, 0.2, 1.0])

    facecolors = base_colors.copy()
    collection.set_facecolors(facecolors)

    state = {"current": 0}
    if show_axes:
        ax.set_title("1+1D Mesh Viewer (left/right)")

    def update(row_index: int):
        facecolors[:] = base_colors
        facecolors[row_index] = selected_color
        facecolors[spatial_mask[row_index]] = spatial_color
        facecolors[temporal_mask[row_index]] = temporal_color
        collection.set_facecolors(facecolors)

        cell_id = int(mesh_data["cell_ids"][row_index])
        if show_axes:
            ax.set_title(f"1+1D Mesh Viewer (left/right) | selected cell id = {cell_id}")
        fig.canvas.draw_idle()

    def on_key(event):
        if event.key == "right":
            state["current"] = (state["current"] + 1) % n_cells
            update(state["current"])
        elif event.key == "left":
            state["current"] = (state["current"] - 1) % n_cells
            update(state["current"])

    fig.canvas.mpl_connect("key_press_event", on_key)
    update(0)
    plt.show()


def print_dof_info(dof_data: dict):
    coords = dof_data["dof_coords"]
    is_constrained = dof_data["is_constrained"]
    offsets = dof_data["constraint_offsets"]
    masters = dof_data["constraint_masters"]
    weights = dof_data["constraint_weights"]

    print("\n=== DoF info ===")
    for gid in range(coords.shape[0]):
        x, t = coords[gid]
        print(
            f"DoF {gid} @ ({x:.6g}, {t:.6g}) | "
            f"constrained = {bool(is_constrained[gid])}",
            end="",
        )

        if is_constrained[gid]:
            start = offsets[gid]
            end = offsets[gid + 1]
            dof_masters = masters[start:end]
            dof_weights = weights[start:end]

            master_str = ", ".join(str(int(m)) for m in dof_masters)
            weight_str = ", ".join(f"{w:.6g}" for w in dof_weights)

            print(f" | masters = [{master_str}] | weights = [{weight_str}]")
        else:
            print()


def plot_dofs_1p1d(
    mesh_data: dict,
    dof_data: dict,
    show_labels: bool = False,
    show_constrained: bool = True,
    ax=None,
):
    if ax is None:
        fig, ax, _ = plot_mesh_1p1d(mesh_data)
    else:
        fig = ax.figure
        plot_mesh_1p1d(mesh_data, ax=ax)

    dof_coords = dof_data["dof_coords"]
    is_constrained = dof_data["is_constrained"]

    unconstrained = ~is_constrained
    constrained = is_constrained

    if np.any(unconstrained):
        ax.scatter(
            dof_coords[unconstrained, 0],
            dof_coords[unconstrained, 1],
            s=28,
            marker="o",
            label="unconstrained DoFs",
            zorder=3,
        )

    if show_constrained and np.any(constrained):
        ax.scatter(
            dof_coords[constrained, 0],
            dof_coords[constrained, 1],
            s=42,
            marker="x",
            label="constrained DoFs",
            zorder=4,
        )

    if show_labels:
        for gid, (x, t) in enumerate(dof_coords):
            ax.text(x, t, str(gid), fontsize=8, ha="left", va="bottom")

    ax.set_title("1+1D Mesh with global DoFs")
    ax.legend()
    return fig, ax


def plot_constraints_1p1d(
    mesh_data: dict,
    dof_data: dict,
    show_labels: bool = False,
    label_weights: bool = False,
):
    fig, ax = plot_dofs_1p1d(
        mesh_data,
        dof_data,
        show_labels=show_labels,
        show_constrained=True,
    )

    coords = dof_data["dof_coords"]
    is_constrained = dof_data["is_constrained"]
    offsets = dof_data["constraint_offsets"]
    masters = dof_data["constraint_masters"]
    weights = dof_data["constraint_weights"]

    segments = []
    segment_midpoints = []
    segment_labels = []

    for gid in range(coords.shape[0]):
        if not is_constrained[gid]:
            continue

        start = offsets[gid]
        end = offsets[gid + 1]

        slave_pt = coords[gid]
        for master_gid, weight in zip(masters[start:end], weights[start:end]):
            master_pt = coords[master_gid]
            segments.append([slave_pt, master_pt])

            midpoint = 0.5 * (slave_pt + master_pt)
            segment_midpoints.append(midpoint)
            segment_labels.append(weight)

    if segments:
        line_collection = LineCollection(
            segments,
            linewidths=0.9,
            linestyles="dashed",
            alpha=0.55,
            zorder=2,
        )
        ax.add_collection(line_collection)

        if label_weights:
            for midpoint, w in zip(segment_midpoints, segment_labels):
                ax.text(
                    midpoint[0],
                    midpoint[1],
                    f"{w:.3g}",
                    fontsize=8,
                    ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.15", alpha=0.7),
                    zorder=5,
                )

    ax.set_title("1+1D Mesh with global DoFs and constraint relations")
    plt.show()


def plot_local_cell_dofs_1p1d(
    mesh_data: dict,
    dof_data: dict,
    show_labels: bool = True,
    jitter: float = 0.0,
    label_offset_fraction: float = 0.015,
):
    fig, ax, _ = plot_mesh_1p1d(mesh_data)

    xs, ts, gids, cell_ids, local_indices, constrained_flags = \
        reconstruct_local_occurrence_coordinates_1p1d(mesh_data, dof_data)

    spatial_vertices = mesh_data["spatial_vertices"]
    temporal_vertices = mesh_data["temporal_vertices"]
    cell_spatial_vertex_ids = mesh_data["cell_spatial_vertex_ids"]
    cell_temporal_vertex_ids = mesh_data["cell_temporal_vertex_ids"]
    mesh_cell_id_to_row = mesh_data["cell_id_to_row"]

    plot_xs = xs.copy()
    plot_ts = ts.copy()

    if jitter != 0.0:
        dx = np.array([
            jitter * (((c * 37 + l * 17) % 11) - 5) / 5.0
            for c, l in zip(cell_ids, local_indices)
        ])
        dt = np.array([
            jitter * (((c * 19 + l * 23) % 11) - 5) / 5.0
            for c, l in zip(cell_ids, local_indices)
        ])
        plot_xs += dx
        plot_ts += dt

    unconstrained = ~constrained_flags
    constrained = constrained_flags

    if np.any(unconstrained):
        ax.scatter(
            plot_xs[unconstrained],
            plot_ts[unconstrained],
            s=28,
            marker="o",
            label="local DoF -> unconstrained gid",
            zorder=3,
        )

    if np.any(constrained):
        ax.scatter(
            plot_xs[constrained],
            plot_ts[constrained],
            s=42,
            marker="x",
            label="local DoF -> constrained gid",
            zorder=4,
        )

    texts = []

    if show_labels:
        for gid, x, t in zip(gids, xs, ts):
            txt = ax.text(
                x,
                t,
                f"{gid}",
                fontsize=7,
                ha="center",
                va="center",
                zorder=5,
            )
            texts.append(txt)

        def update_label_positions(event=None):
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()

            dx_view = xlim[1] - xlim[0]
            dt_view = ylim[1] - ylim[0]
            offset_len = label_offset_fraction * min(abs(dx_view), abs(dt_view))

            for txt, x, t, cell_id in zip(texts, xs, ts, cell_ids):
                mesh_row = mesh_cell_id_to_row[int(cell_id)]
                spatial_ids = cell_spatial_vertex_ids[mesh_row]
                temporal_ids = cell_temporal_vertex_ids[mesh_row]

                x0 = spatial_vertices[spatial_ids[0], 0]
                x1 = spatial_vertices[spatial_ids[1], 0]
                t0 = temporal_vertices[temporal_ids[0], 0]
                t1 = temporal_vertices[temporal_ids[1], 0]

                xc = 0.5 * (x0 + x1)
                tc = 0.5 * (t0 + t1)

                dir_x = xc - x
                dir_t = tc - t

                norm = np.hypot(dir_x, dir_t)
                if norm > 0.0:
                    dir_x /= norm
                    dir_t /= norm
                else:
                    dir_x, dir_t = 0.0, 0.0

                txt.set_position((x + offset_len * dir_x, t + offset_len * dir_t))

            fig.canvas.draw_idle()

        update_label_positions()
        ax.callbacks.connect("xlim_changed", update_label_positions)
        ax.callbacks.connect("ylim_changed", update_label_positions)

    ax.set_title("1+1D Mesh with local cell DoF occurrences")
    ax.legend()
    plt.show()


def check_identity_constraints(dof_data: dict, tol: float = 1e-12):
    is_constrained = dof_data["is_constrained"]
    offsets = dof_data["constraint_offsets"]
    masters = dof_data["constraint_masters"]
    weights = dof_data["constraint_weights"]

    suspicious = []

    for gid in range(len(is_constrained)):
        if not is_constrained[gid]:
            continue

        start = offsets[gid]
        end = offsets[gid + 1]

        local_masters = masters[start:end]
        local_weights = weights[start:end]

        nz = [
            (int(m), float(w))
            for m, w in zip(local_masters, local_weights)
            if abs(w) > tol
        ]

        if len(nz) == 1 and abs(nz[0][1] - 1.0) < tol:
            suspicious.append((gid, nz[0][0], nz[0][1]))

    print("\n=== Identity-constraint check ===")
    if not suspicious:
        print("No constrained DoFs of the form slave = 1 * master were found.")
    else:
        print("Found constrained DoFs that should likely have been collapsed:")
        for gid, master, weight in suspicious:
            print(f"  DoF {gid} = {weight:.16g} * DoF {master}")


def _duplicate_coordinate_groups(
    coords: np.ndarray,
    tol: float,
) -> list[list[int]]:
    buckets = {}
    for gid, pt in enumerate(coords):
        key = tuple(int(round(float(x) / tol)) for x in pt)
        buckets.setdefault(key, []).append(gid)
    return [gids for gids in buckets.values() if len(gids) > 1]


def _print_duplicate_policy_summary(policy: str) -> None:
    if policy.lower() == "spaceonly":
        print(
            "SpaceOnlyPolicy: duplicate coordinate groups are allowed "
            "and shown for inspection."
        )
    else:
        print("SpaceTimePolicy: duplicate coordinate groups are suspicious.")


def check_duplicate_coordinates(dof_data: dict, policy: str, tol: float = 1e-12):
    coords = dof_data["dof_coords"]
    is_constrained = dof_data["is_constrained"]

    duplicate_groups = _duplicate_coordinate_groups(coords, tol)

    print("\n=== Duplicate-coordinate check ===")
    if not duplicate_groups:
        print("No duplicate global DoF coordinates found.")
        return

    if policy.lower() == "spaceonly":
        print(
            "Duplicate global DoF coordinates found, but this is allowed under "
            "SpaceOnlyPolicy because continuity in time is not enforced."
        )
        for gids in duplicate_groups:
            pt = coords[gids[0]]
            flags = ["C" if is_constrained[gid] else "U" for gid in gids]
            print(
                f"  coord=({pt[0]:.16g}, {pt[1]:.16g})"
                f" -> dofs={gids}, status={flags}"
            )
        return

    print(f"Found {len(duplicate_groups)} suspicious duplicate-coordinate group(s):")
    for gids in duplicate_groups:
        pt = coords[gids[0]]
        flags = ["C" if is_constrained[gid] else "U" for gid in gids]
        print(
            f"  coord=({pt[0]:.16g}, {pt[1]:.16g})"
            f" -> dofs={gids}, status={flags}"
        )


def report_duplicate_coordinate_owners(
    dof_data: dict,
    policy: str,
    tol: float = 1e-12,
):
    coords = dof_data["dof_coords"]
    is_constrained = dof_data["is_constrained"]

    duplicate_groups = _duplicate_coordinate_groups(coords, tol)

    print("\n=== Duplicate-coordinate owners ===")
    if not duplicate_groups:
        print("No duplicate global DoF coordinates found.")
        return

    _print_duplicate_policy_summary(policy)

    for gids in duplicate_groups:
        pt = coords[gids[0]]
        print(f"coord=({pt[0]:.16g}, {pt[1]:.16g})")
        for gid in gids:
            status = "C" if is_constrained[gid] else "U"
            print(f"  gid={gid} status={status}")


def report_duplicate_coordinate_cell_owners(
    dof_data: dict,
    policy: str,
    tol: float = 1e-12,
):
    coords = dof_data["dof_coords"]
    is_constrained = dof_data["is_constrained"]
    cell_to_dofs = dof_data["cell_to_dofs"]
    cell_ids = dof_data["cell_ids"]

    duplicate_groups = _duplicate_coordinate_groups(coords, tol)

    print("\n=== Duplicate-coordinate local owners ===")
    if not duplicate_groups:
        print("No duplicate global DoF coordinates found.")
        return

    _print_duplicate_policy_summary(policy)

    for gids in duplicate_groups:
        pt = coords[gids[0]]
        print(f"\ncoord=({pt[0]:.16g}, {pt[1]:.16g})")

        for gid in gids:
            status = "C" if is_constrained[gid] else "U"
            print(f"  gid={gid} status={status}")

            owners = np.argwhere(cell_to_dofs == gid)
            for row, local_index in owners:
                print(
                    f"    owner cell_id={int(cell_ids[row])} "
                    f"local_index={int(local_index)}"
                )


def run_dof_consistency_checks(dof_data: dict, policy: str, tol: float = 1e-12):
    print(f"\nRunning DoF consistency checks for policy = {policy}")
    check_identity_constraints(dof_data, tol=tol)
    check_duplicate_coordinates(dof_data, policy=policy, tol=tol)

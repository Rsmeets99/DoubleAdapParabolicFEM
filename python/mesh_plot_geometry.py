import numpy as np

from binary_readers import require_mesh_is_1p1d


def mesh_polygons_1p1d(mesh_data: dict) -> np.ndarray:
    require_mesh_is_1p1d(mesh_data, "mesh_polygons_1p1d")

    spatial_vertices = mesh_data["spatial_vertices"]
    temporal_vertices = mesh_data["temporal_vertices"]
    cell_spatial_vertex_ids = mesh_data["cell_spatial_vertex_ids"]
    cell_temporal_vertex_ids = mesh_data["cell_temporal_vertex_ids"]

    x = spatial_vertices[cell_spatial_vertex_ids][:, :, 0]
    t = temporal_vertices[cell_temporal_vertex_ids][:, :, 0]

    return np.stack(
        [
            np.column_stack([x[:, 0], t[:, 0]]),
            np.column_stack([x[:, 1], t[:, 0]]),
            np.column_stack([x[:, 1], t[:, 1]]),
            np.column_stack([x[:, 0], t[:, 1]]),
        ],
        axis=1,
    )


def build_neighbour_masks(mesh_data: dict):
    n_cells = mesh_data["cell_ids"].shape[0]
    cell_id_to_row = mesh_data["cell_id_to_row"]

    spatial_neigh = mesh_data["spatial_neighbours"]
    temporal_neigh = mesh_data["temporal_neighbours"]

    spatial_mask = np.zeros((n_cells, n_cells), dtype=bool)
    temporal_mask = np.zeros((n_cells, n_cells), dtype=bool)

    for row in range(n_cells):
        for face in range(spatial_neigh.shape[1]):
            for neigh_id in spatial_neigh[row, face]:
                if neigh_id >= 0:
                    spatial_mask[row, cell_id_to_row[int(neigh_id)]] = True

        for face in range(temporal_neigh.shape[1]):
            for neigh_id in temporal_neigh[row, face]:
                if neigh_id >= 0:
                    temporal_mask[row, cell_id_to_row[int(neigh_id)]] = True

    return spatial_mask, temporal_mask


def reconstruct_local_occurrence_coordinates_1p1d(mesh_data: dict, dof_data: dict):
    spatial_vertices = mesh_data["spatial_vertices"]
    temporal_vertices = mesh_data["temporal_vertices"]
    cell_spatial_vertex_ids = mesh_data["cell_spatial_vertex_ids"]
    cell_temporal_vertex_ids = mesh_data["cell_temporal_vertex_ids"]
    mesh_cell_id_to_row = mesh_data["cell_id_to_row"]

    cell_to_dofs = dof_data["cell_to_dofs"]
    local_reference_coords = dof_data["local_reference_coords"]
    is_constrained = dof_data["is_constrained"]
    dof_cell_ids = dof_data["cell_ids"]

    n_cells, dofs_per_cell = cell_to_dofs.shape

    xs = []
    ts = []
    gids = []
    cell_ids = []
    local_indices = []
    constrained_flags = []

    for row in range(n_cells):
        cell_id = int(dof_cell_ids[row])
        mesh_row = mesh_cell_id_to_row[cell_id]

        spatial_ids = cell_spatial_vertex_ids[mesh_row]
        temporal_ids = cell_temporal_vertex_ids[mesh_row]

        x0 = spatial_vertices[spatial_ids[0], 0]
        x1 = spatial_vertices[spatial_ids[1], 0]
        t0 = temporal_vertices[temporal_ids[0], 0]
        t1 = temporal_vertices[temporal_ids[1], 0]

        for local_index in range(dofs_per_cell):
            gid = cell_to_dofs[row, local_index]
            if gid < 0:
                continue

            xi = local_reference_coords[local_index]
            x = x0 + (x1 - x0) * xi[0]
            t = t0 + (t1 - t0) * xi[1]

            xs.append(x)
            ts.append(t)
            gids.append(int(gid))
            cell_ids.append(cell_id)
            local_indices.append(local_index)
            constrained_flags.append(bool(is_constrained[gid]))

    return (
        np.asarray(xs),
        np.asarray(ts),
        np.asarray(gids, dtype=np.int32),
        np.asarray(cell_ids, dtype=np.int32),
        np.asarray(local_indices, dtype=np.int32),
        np.asarray(constrained_flags, dtype=bool),
    )

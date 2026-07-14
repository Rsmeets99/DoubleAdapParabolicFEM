from __future__ import annotations

import numpy as np

from binary_readers import require_mesh_is_2p1d


def cell_prism_vertices(mesh_data: dict, row: int) -> np.ndarray:
    require_mesh_is_2p1d(mesh_data, "cell_prism_vertices")

    spatial_vertices = mesh_data["spatial_vertices"]
    temporal_vertices = mesh_data["temporal_vertices"]
    spatial_ids = mesh_data["cell_spatial_vertex_ids"][row]
    temporal_ids = mesh_data["cell_temporal_vertex_ids"][row]

    xy = spatial_vertices[spatial_ids]
    t0 = float(temporal_vertices[temporal_ids[0], 0])
    t1 = float(temporal_vertices[temporal_ids[1], 0])

    bottom = np.column_stack([xy[:, 0], xy[:, 1], np.full(3, t0)])
    top = np.column_stack([xy[:, 0], xy[:, 1], np.full(3, t1)])
    return np.vstack([bottom, top])


def cell_prism_faces(mesh_data: dict, row: int) -> list[np.ndarray]:
    vertices = cell_prism_vertices(mesh_data, row)
    return [
        vertices[[0, 1, 2]],
        vertices[[3, 5, 4]],
        vertices[[0, 1, 4, 3]],
        vertices[[1, 2, 5, 4]],
        vertices[[2, 0, 3, 5]],
    ]


def all_cell_faces(
    mesh_data: dict,
    rows: np.ndarray | None = None,
) -> tuple[list[np.ndarray], np.ndarray]:
    require_mesh_is_2p1d(mesh_data, "all_cell_faces")

    if rows is None:
        rows = np.arange(mesh_data["cell_ids"].shape[0], dtype=np.int32)

    faces: list[np.ndarray] = []
    owners: list[int] = []
    for row in rows:
        row_int = int(row)
        cell_faces = cell_prism_faces(mesh_data, row_int)
        faces.extend(cell_faces)
        owners.extend([row_int] * len(cell_faces))

    return faces, np.asarray(owners, dtype=np.int32)


def cell_centers(mesh_data: dict) -> np.ndarray:
    centers = []
    for row in range(mesh_data["cell_ids"].shape[0]):
        centers.append(cell_prism_vertices(mesh_data, row).mean(axis=0))
    return np.asarray(centers, dtype=np.float64)


def dof_rows_for_cell(mesh_data: dict, dof_data: dict, cell_id: int) -> np.ndarray:
    dof_row = dof_data["cell_id_to_row"].get(int(cell_id))
    if dof_row is None:
        return np.empty((0,), dtype=np.int32)

    gids = np.asarray(dof_data["cell_to_dofs"][dof_row], dtype=np.int32)
    return np.unique(gids[gids >= 0])


def resolve_cell_row(mesh_data: dict, typed: str) -> int:
    text = typed.strip()
    if not text:
        raise ValueError("empty cell id")

    value = int(text)
    cell_id_to_row = mesh_data["cell_id_to_row"]
    if value in cell_id_to_row:
        return int(cell_id_to_row[value])

    if 0 <= value < mesh_data["cell_ids"].shape[0]:
        return value

    raise ValueError(f"unknown cell id or row {value}")

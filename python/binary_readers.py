from pathlib import Path

import numpy as np

MESH_BINARY_FORMAT_VERSION = 1
DOFS_BINARY_FORMAT_VERSION = 1
TIME_SLAB_SPACE_METADATA_BINARY_FORMAT_VERSION = 1
TIME_SLAB_PROVENANCE_BINARY_FORMAT_VERSION = 1
TIME_SLAB_EDGE_PATCHES_BINARY_FORMAT_VERSION = 1
TIME_SLAB_VERTEX_PATCHES_BINARY_FORMAT_VERSION = 2


def require_size(array: np.ndarray, expected: int, label: str) -> np.ndarray:
    if array.size != expected:
        raise ValueError(f"{label}: expected {expected} entries, got {array.size}.")
    return array


def read_exact_array(
    handle,
    dtype,
    count: int,
    label: str,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = require_size(
        np.fromfile(handle, dtype=dtype, count=count),
        count,
        label,
    )
    if shape is not None:
        return array.reshape(shape)
    return array


def require_no_trailing_bytes(handle, message: str) -> None:
    trailing = np.fromfile(handle, dtype=np.uint8)
    if trailing.size != 0:
        raise ValueError(message)


def read_versioned_int32_header(
    handle,
    filename: Path,
    label: str,
    payload_count: int,
    expected_version: int,
) -> np.ndarray:
    header = read_exact_array(
        handle,
        np.int32,
        payload_count + 1,
        f"{filename}: {label}",
    )

    version = int(header[0])
    if version != expected_version:
        raise ValueError(
            f"{filename}: unsupported {label} format version {version}. "
            f"Expected version {expected_version}."
        )

    return header[1:]


def require_1p1d_space_time(
    dim_space_v: int,
    dim_time_v: int,
    label: str,
) -> None:
    if dim_space_v != 1 or dim_time_v != 1:
        raise ValueError(f"{label}: only 1+1D space-time data is currently supported.")


def require_mesh_is_1p1d(mesh_data: dict, label: str) -> None:
    header = mesh_data["header"]
    require_1p1d_space_time(
        int(header["dim_space_v"]),
        int(header["dim_time_v"]),
        label,
    )


def require_mesh_is_2p1d(mesh_data: dict, label: str) -> None:
    header = mesh_data["header"]
    if int(header["dim_space_v"]) != 2 or int(header["dim_time_v"]) != 1:
        raise ValueError(f"{label}: expected 2+1D triangular-prism mesh data.")


def load_mesh_binary(filename: Path | str) -> dict:
    filename = Path(filename)

    with open(filename, "rb") as f:
        header = read_versioned_int32_header(
            f,
            filename,
            "mesh header",
            13,
            MESH_BINARY_FORMAT_VERSION,
        )

        (
            n_spatial_vertices,
            n_temporal_vertices,
            dim_space_v,
            dim_time_v,
            n_cells,
            vertices_per_spatial_face,
            vertices_per_temporal_face,
            Tp_vertices,
            Ip_vertices,
            spatial_faces,
            max_spatial_neigh,
            temporal_faces,
            max_temporal_neigh,
        ) = header.tolist()

        spatial_vertices = read_exact_array(
            f,
            np.float64,
            n_spatial_vertices * dim_space_v,
            f"{filename}: spatial vertices",
            (n_spatial_vertices, dim_space_v),
        )

        temporal_vertices = read_exact_array(
            f,
            np.float64,
            n_temporal_vertices * dim_time_v,
            f"{filename}: temporal vertices",
            (n_temporal_vertices, dim_time_v),
        )

        cell_ids = read_exact_array(
            f,
            np.int32,
            n_cells,
            f"{filename}: cell ids",
        )

        cell_spatial_vertex_ids = read_exact_array(
            f,
            np.int32,
            n_cells * Tp_vertices,
            f"{filename}: cell spatial vertex ids",
            (n_cells, Tp_vertices),
        )

        cell_temporal_vertex_ids = read_exact_array(
            f,
            np.int32,
            n_cells * Ip_vertices,
            f"{filename}: cell temporal vertex ids",
            (n_cells, Ip_vertices),
        )

        spatial_neighbours = read_exact_array(
            f,
            np.int32,
            n_cells * spatial_faces * max_spatial_neigh,
            f"{filename}: spatial neighbours",
            (n_cells, spatial_faces, max_spatial_neigh),
        )

        temporal_neighbours = read_exact_array(
            f,
            np.int32,
            n_cells * temporal_faces * max_temporal_neigh,
            f"{filename}: temporal neighbours",
            (n_cells, temporal_faces, max_temporal_neigh),
        )

        require_no_trailing_bytes(
            f,
            f"{filename}: trailing bytes detected in mesh binary.",
        )

    cell_id_to_row = {int(cell_id): row for row, cell_id in enumerate(cell_ids.tolist())}

    return {
        "spatial_vertices": spatial_vertices,
        "temporal_vertices": temporal_vertices,
        "cell_ids": cell_ids,
        "cell_spatial_vertex_ids": cell_spatial_vertex_ids,
        "cell_temporal_vertex_ids": cell_temporal_vertex_ids,
        "spatial_neighbours": spatial_neighbours,
        "temporal_neighbours": temporal_neighbours,
        "cell_id_to_row": cell_id_to_row,
        "header": {
            "format_version": MESH_BINARY_FORMAT_VERSION,
            "n_spatial_vertices": n_spatial_vertices,
            "n_temporal_vertices": n_temporal_vertices,
            "dim_space_v": dim_space_v,
            "dim_time_v": dim_time_v,
            "n_cells": n_cells,
            "vertices_per_spatial_face": vertices_per_spatial_face,
            "vertices_per_temporal_face": vertices_per_temporal_face,
            "Tp_vertices": Tp_vertices,
            "Ip_vertices": Ip_vertices,
            "spatial_faces": spatial_faces,
            "max_spatial_neigh": max_spatial_neigh,
            "temporal_faces": temporal_faces,
            "max_temporal_neigh": max_temporal_neigh,
        },
    }


def load_dofs_binary(filename: Path | str) -> dict:
    filename = Path(filename)

    with open(filename, "rb") as f:
        header = read_versioned_int32_header(
            f,
            filename,
            "dof header",
            7,
            DOFS_BINARY_FORMAT_VERSION,
        )

        (
            n_cells,
            dofs_per_cell,
            n_dofs,
            dim_v,
            p_space,
            p_time,
            n_constraint_entries,
        ) = header.tolist()

        dof_coords = read_exact_array(
            f,
            np.float64,
            n_dofs * dim_v,
            f"{filename}: dof coordinates",
            (n_dofs, dim_v),
        )

        cell_ids = read_exact_array(
            f,
            np.int32,
            n_cells,
            f"{filename}: dof cell ids",
        )

        cell_to_dofs = read_exact_array(
            f,
            np.int32,
            n_cells * dofs_per_cell,
            f"{filename}: cell-to-dofs",
            (n_cells, dofs_per_cell),
        )

        local_reference_coords = read_exact_array(
            f,
            np.float64,
            dofs_per_cell * dim_v,
            f"{filename}: local reference coordinates",
            (dofs_per_cell, dim_v),
        )

        is_constrained = read_exact_array(
            f,
            np.uint8,
            n_dofs,
            f"{filename}: constrained flags",
        ).astype(bool)

        constraint_offsets = read_exact_array(
            f,
            np.int32,
            n_dofs + 1,
            f"{filename}: constraint offsets",
        )

        constraint_masters = read_exact_array(
            f,
            np.int32,
            n_constraint_entries,
            f"{filename}: constraint masters",
        )

        constraint_weights = read_exact_array(
            f,
            np.float64,
            n_constraint_entries,
            f"{filename}: constraint weights",
        )

        require_no_trailing_bytes(
            f,
            f"{filename}: trailing bytes detected in dof binary.",
        )

    cell_id_to_row = {int(cell_id): row for row, cell_id in enumerate(cell_ids.tolist())}

    return {
        "dof_coords": dof_coords,
        "cell_ids": cell_ids,
        "cell_to_dofs": cell_to_dofs,
        "local_reference_coords": local_reference_coords,
        "is_constrained": is_constrained,
        "constraint_offsets": constraint_offsets,
        "constraint_masters": constraint_masters,
        "constraint_weights": constraint_weights,
        "cell_id_to_row": cell_id_to_row,
        "header": {
            "format_version": DOFS_BINARY_FORMAT_VERSION,
            "n_cells": n_cells,
            "dofs_per_cell": dofs_per_cell,
            "n_dofs": n_dofs,
            "dim_v": dim_v,
            "p_space": p_space,
            "p_time": p_time,
            "n_constraint_entries": n_constraint_entries,
        },
    }

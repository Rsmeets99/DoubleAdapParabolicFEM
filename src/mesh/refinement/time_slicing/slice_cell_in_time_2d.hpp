#pragma once

#include <stdexcept>

#include "../../../core/exceptions.hpp"
#include "../../cell.hpp"
#include "../../mesh.hpp"
#include "../../topology/faces.hpp"
#include "../../topology/orientation.hpp"

namespace mesh::refinement::time_slicing
{
    template<typename GeomTraits>
    [[nodiscard]] inline typename mesh::MeshTypes<GeomTraits>::cell_id_type
    slice_cell_in_time_2d(
        mesh::Mesh<GeomTraits>& target_mesh,
        const mesh::Mesh<GeomTraits>& source_mesh,
        typename mesh::MeshTypes<GeomTraits>::cell_id_type source_cell_id,
        double t_begin,
        double t_end,
        double tol = 1.0e-14)
    {
        using Types             = mesh::MeshTypes<GeomTraits>;
        using CellType          = mesh::Cell<GeomTraits>;
        using cell_id_type      = typename Types::cell_id_type;
        using vertex_id_type    = typename Types::vertex_id_type;
        using TemporalPoint     = typename Types::TemporalPoint;
        using SpatialVertexIds  = typename Types::SpatialVertexIds;
        using TemporalVertexIds = typename Types::TemporalVertexIds;

        static_assert(GeomTraits::dim_space_v == 2,
                      "slice_cell_in_time_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "slice_cell_in_time_2d requires dim_time_v == 1.");

        if (!(t_begin < t_end))
            throw std::runtime_error(
                "slice_cell_in_time_2d: expected t_begin < t_end.");

        const auto& source_cell = source_mesh.cell(source_cell_id);

        const auto t0_id_src = source_cell.temporal_vertex_ids[0];
        const auto t1_id_src = source_cell.temporal_vertex_ids[1];

        const double t0 = source_mesh.temporal_vertices()[t0_id_src][0];
        const double t1 = source_mesh.temporal_vertices()[t1_id_src][0];

        if (t_begin < t0 - tol || t_end > t1 + tol)
            throw std::runtime_error(
                "slice_cell_in_time_2d: requested slice is not contained in source cell.");

        SpatialVertexIds spatial_ids{};
        for (int v = 0; v < Types::n_spatial_vertices; ++v)
        {
            const auto source_vertex_id =
                source_cell.spatial_vertex_ids[static_cast<std::size_t>(v)];
            spatial_ids[static_cast<std::size_t>(v)] =
                target_mesh.get_or_create_spatial_vertex(
                    source_mesh.spatial_vertices()[static_cast<std::size_t>(source_vertex_id)]);
        }

        TemporalPoint tb{};
        TemporalPoint te{};
        tb[0] = t_begin;
        te[0] = t_end;

        const vertex_id_type tb_id = target_mesh.get_or_create_temporal_vertex(tb);
        const vertex_id_type te_id = target_mesh.get_or_create_temporal_vertex(te);

        TemporalVertexIds temporal_ids{};
        temporal_ids[0] = tb_id;
        temporal_ids[1] = te_id;

        CellType sliced{};
        sliced.cell_id     = static_cast<cell_id_type>(target_mesh.unsafe_cells_ref().size());
        sliced.parent_id   = -1;
        sliced.children.clear();
        sliced.is_leaf     = true;

        // Sliced cells are synthetic slab-local cells. They keep refinement
        // metadata for diagnostics/lookup, but are not part of the source tree.
        sliced.generation      = source_cell.generation;
        sliced.spatial_level   = source_cell.spatial_level;
        sliced.temporal_level  = source_cell.temporal_level;
        sliced.last_split_type = source_cell.last_split_type;

        sliced.spatial_vertex_ids            = spatial_ids;
        sliced.temporal_vertex_ids           = temporal_ids;
        sliced.spatial_refinement_edge_local = source_cell.spatial_refinement_edge_local;

        mesh::topology::ensure_spatial_vertex_orientation(
            sliced, target_mesh.spatial_vertices());
        mesh::topology::ensure_temporal_vertex_orientation(
            sliced, target_mesh.temporal_vertices());
        mesh::topology::fill_faces(sliced);

        // Spatial boundary is inherited from the source cell. The slab-local
        // mesh has no root-boundary reconstruction step, so this is the
        // authoritative boundary information for slab FESpaces.
        sliced.spatial_boundary = source_cell.spatial_boundary;

        // In a slab-local problem every sliced cell spans the full slab interval.
        sliced.temporal_boundary[0] = true;
        sliced.temporal_boundary[1] = true;

        target_mesh.unsafe_cells_ref().push_back(sliced);
        return sliced.cell_id;
    }
}

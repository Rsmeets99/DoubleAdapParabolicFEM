#pragma once

#include <array>

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "../detail/vertex_registry.hpp"
#include "../topology/boundary.hpp"
#include "../topology/faces.hpp"
#include "../topology/orientation.hpp"
#include "../types.hpp"

namespace mesh::initialization
{
    template<typename GeomTraits>
    [[nodiscard]] inline Cell<GeomTraits> make_root_cell(
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        detail::VertexRegistry<GeomTraits>& registry,
        const typename MeshTypes<GeomTraits>::SpatialSimplexPoints& spatial_points,
        const typename MeshTypes<GeomTraits>::TemporalPoint& t0,
        const typename MeshTypes<GeomTraits>::TemporalPoint& t1,
        typename MeshTypes<GeomTraits>::cell_id_type root_id = 0)
    {
        Cell<GeomTraits> cell{};
        cell.cell_id        = root_id;
        cell.parent_id      = -1;
        cell.is_leaf        = true;
        cell.generation     = 0;
        cell.spatial_level  = 0;
        cell.temporal_level = 0;
        cell.last_split_type = RefinementType::none;
        cell.spatial_refinement_edge_local = {0, 1};

        for (int i = 0; i < MeshTypes<GeomTraits>::n_spatial_vertices; ++i)
        {
            cell.spatial_vertex_ids[static_cast<std::size_t>(i)] =
                registry.get_or_create_spatial_vertex(
                    spatial_vertices,
                    spatial_points[static_cast<std::size_t>(i)]);
        }

        cell.temporal_vertex_ids[0] =
            registry.get_or_create_temporal_vertex(temporal_vertices, t0);
        cell.temporal_vertex_ids[1] =
            registry.get_or_create_temporal_vertex(temporal_vertices, t1);

        topology::ensure_spatial_vertex_orientation(cell, spatial_vertices);
        topology::ensure_temporal_vertex_orientation(cell, temporal_vertices);
        topology::fill_faces(cell);

        return cell;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline Cell<GeomTraits> make_root_cell(
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        detail::VertexRegistry<GeomTraits>& registry,
        std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        const typename MeshTypes<GeomTraits>::SpatialPoint& x0,
        const typename MeshTypes<GeomTraits>::SpatialPoint& x1,
        const typename MeshTypes<GeomTraits>::TemporalPoint& t0,
        const typename MeshTypes<GeomTraits>::TemporalPoint& t1,
        typename MeshTypes<GeomTraits>::cell_id_type root_id = 0)
    {
        if constexpr (!(GeomTraits::dim_space_v == 1))
        {
            throw core::dimension_not_supported_error(
                "make_root_cell is currently only implemented for 1+1D.");
        }
        else
        {
            const typename MeshTypes<GeomTraits>::SpatialSimplexPoints spatial_points{
                x0,
                x1
            };

            auto cell = make_root_cell<GeomTraits>(
                spatial_vertices,
                temporal_vertices,
                registry,
                spatial_points,
                t0,
                t1,
                root_id);

            spatial_boundary_vertex_ids = {
                cell.spatial_vertex_ids[0],
                cell.spatial_vertex_ids[1]
            };

            temporal_boundary_vertex_ids = {
                cell.temporal_vertex_ids[0],
                cell.temporal_vertex_ids[1]
            };

            topology::fill_boundary(
                cell,
                spatial_boundary_vertex_ids,
                temporal_boundary_vertex_ids);

            return cell;
        }
    }
}

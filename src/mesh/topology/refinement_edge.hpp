#pragma once

#include <algorithm>

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "../types.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    [[nodiscard]] inline typename MeshTypes<GeomTraits>::SpatialEdgeVertexIds
    spatial_refinement_edge_vertex_ids(const Cell<GeomTraits>& cell)
    {
        using Types = MeshTypes<GeomTraits>;
        typename Types::SpatialEdgeVertexIds edge{};

        const int i0 = cell.spatial_refinement_edge_local[0];
        const int i1 = cell.spatial_refinement_edge_local[1];

        if (i0 < 0 || i0 >= Types::n_spatial_vertices ||
            i1 < 0 || i1 >= Types::n_spatial_vertices)
        {
            throw core::invalid_mesh_error(
                "spatial_refinement_edge_vertex_ids: local refinement-edge indices out of range.");
        }

        edge[0] = cell.spatial_vertex_ids[static_cast<std::size_t>(i0)];
        edge[1] = cell.spatial_vertex_ids[static_cast<std::size_t>(i1)];
        return edge;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool cell_contains_spatial_edge(
        const Cell<GeomTraits>& cell,
        const typename MeshTypes<GeomTraits>::SpatialEdgeVertexIds& edge,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            const auto a_id = cell.spatial_vertex_ids[0];
            const auto b_id = cell.spatial_vertex_ids[1];

            const double xa = spatial_vertices[static_cast<std::size_t>(a_id)][0];
            const double xb = spatial_vertices[static_cast<std::size_t>(b_id)][0];

            const double xe0 = spatial_vertices[static_cast<std::size_t>(edge[0])][0];
            const double xe1 = spatial_vertices[static_cast<std::size_t>(edge[1])][0];

            const double cell_min = std::min(xa, xb);
            const double cell_max = std::max(xa, xb);

            const double edge_min = std::min(xe0, xe1);
            const double edge_max = std::max(xe0, xe1);

            return cell_min <= edge_min && edge_max <= cell_max;
        }
        else
        {
            throw core::dimension_not_supported_error(
                "cell_contains_spatial_edge is only implemented for dim_space_v == 1.");
        }
    }
}
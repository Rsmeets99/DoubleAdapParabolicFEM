#pragma once

#include <algorithm>
#include <vector>

#include "../cell.hpp"
#include "../types.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    inline void ensure_spatial_vertex_orientation_1d(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        static_assert(GeomTraits::dim_space_v == 1,
                      "ensure_spatial_vertex_orientation_1d requires dim_space_v == 1.");

        const double x0 = spatial_vertices[cell.spatial_vertex_ids[0]][0];
        const double x1 = spatial_vertices[cell.spatial_vertex_ids[1]][0];

        if (x1 < x0)
            std::swap(cell.spatial_vertex_ids[0], cell.spatial_vertex_ids[1]);
    }

    template<typename GeomTraits>
    inline void ensure_temporal_vertex_orientation_1d(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices)
    {
        static_assert(GeomTraits::dim_time_v == 1,
                      "ensure_temporal_vertex_orientation_1d requires dim_time_v == 1.");

        const double t0 = temporal_vertices[cell.temporal_vertex_ids[0]][0];
        const double t1 = temporal_vertices[cell.temporal_vertex_ids[1]][0];

        if (t1 < t0)
            std::swap(cell.temporal_vertex_ids[0], cell.temporal_vertex_ids[1]);
    }
}
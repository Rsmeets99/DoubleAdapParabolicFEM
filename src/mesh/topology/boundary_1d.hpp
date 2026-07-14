#pragma once

#include <algorithm>
#include <vector>

#include "../cell.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    inline void fill_spatial_boundary_1d(
        Cell<GeomTraits>& cell,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids)
    {
        static_assert(GeomTraits::dim_space_v == 1,
                      "fill_spatial_boundary_1d requires dim_space_v == 1.");

        for (int i = 0; i < 2; ++i)
        {
            const auto vid = cell.spatial_faces[i].spatial_vertex_ids[0];
            cell.spatial_boundary[i] =
                (std::find(spatial_boundary_vertex_ids.begin(),
                           spatial_boundary_vertex_ids.end(),
                           vid) != spatial_boundary_vertex_ids.end());
        }
    }

    template<typename GeomTraits>
    inline void fill_temporal_boundary_1d(
        Cell<GeomTraits>& cell,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        static_assert(GeomTraits::dim_time_v == 1,
                      "fill_temporal_boundary_1d requires dim_time_v == 1.");

        for (int i = 0; i < 2; ++i)
        {
            const auto vid = cell.temporal_faces[i].temporal_vertex_id;
            cell.temporal_boundary[i] =
                (std::find(temporal_boundary_vertex_ids.begin(),
                           temporal_boundary_vertex_ids.end(),
                           vid) != temporal_boundary_vertex_ids.end());
        }
    }

    template<typename GeomTraits>
    inline void fill_boundary_1d(
        Cell<GeomTraits>& cell,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        fill_spatial_boundary_1d(cell, spatial_boundary_vertex_ids);
        fill_temporal_boundary_1d(cell, temporal_boundary_vertex_ids);
    }
}
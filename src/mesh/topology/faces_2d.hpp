#pragma once

#include "../cell.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    inline void fill_spatial_faces_2d(Cell<GeomTraits>& cell)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "fill_spatial_faces_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "fill_spatial_faces_2d requires dim_time_v == 1.");

        cell.spatial_faces[0].spatial_vertex_ids = {
            cell.spatial_vertex_ids[0],
            cell.spatial_vertex_ids[1]
        };
        cell.spatial_faces[0].temporal_vertex_ids = {
            cell.temporal_vertex_ids[0],
            cell.temporal_vertex_ids[1]
        };

        cell.spatial_faces[1].spatial_vertex_ids = {
            cell.spatial_vertex_ids[1],
            cell.spatial_vertex_ids[2]
        };
        cell.spatial_faces[1].temporal_vertex_ids = {
            cell.temporal_vertex_ids[0],
            cell.temporal_vertex_ids[1]
        };

        cell.spatial_faces[2].spatial_vertex_ids = {
            cell.spatial_vertex_ids[2],
            cell.spatial_vertex_ids[0]
        };
        cell.spatial_faces[2].temporal_vertex_ids = {
            cell.temporal_vertex_ids[0],
            cell.temporal_vertex_ids[1]
        };
    }

    template<typename GeomTraits>
    inline void fill_temporal_faces_2d(Cell<GeomTraits>& cell)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "fill_temporal_faces_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "fill_temporal_faces_2d requires dim_time_v == 1.");

        cell.temporal_faces[0].spatial_vertex_ids = {
            cell.spatial_vertex_ids[0],
            cell.spatial_vertex_ids[1],
            cell.spatial_vertex_ids[2]
        };
        cell.temporal_faces[0].temporal_vertex_id = cell.temporal_vertex_ids[0];

        cell.temporal_faces[1].spatial_vertex_ids = {
            cell.spatial_vertex_ids[0],
            cell.spatial_vertex_ids[1],
            cell.spatial_vertex_ids[2]
        };
        cell.temporal_faces[1].temporal_vertex_id = cell.temporal_vertex_ids[1];
    }

    template<typename GeomTraits>
    inline void fill_faces_2d(Cell<GeomTraits>& cell)
    {
        fill_spatial_faces_2d(cell);
        fill_temporal_faces_2d(cell);
    }
}

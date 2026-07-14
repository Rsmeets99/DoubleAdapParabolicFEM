#pragma once

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "../types.hpp"
#include "orientation_1d.hpp"
#include "orientation_2d.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    inline void ensure_spatial_vertex_orientation(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            ensure_spatial_vertex_orientation_1d(cell, spatial_vertices);
        }
        else if constexpr (GeomTraits::dim_space_v == 2)
        {
            ensure_spatial_vertex_orientation_2d(cell, spatial_vertices);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "ensure_spatial_vertex_orientation is only implemented for dim_space_v = 1 or 2.");
        }
    }

    template<typename GeomTraits>
    inline void ensure_temporal_vertex_orientation(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices)
    {
        if constexpr (GeomTraits::dim_time_v == 1)
        {
            ensure_temporal_vertex_orientation_1d(cell, temporal_vertices);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "ensure_temporal_vertex_orientation is only implemented for dim_time_v == 1.");
        }
    }
}

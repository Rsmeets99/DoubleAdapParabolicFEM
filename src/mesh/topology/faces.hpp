#pragma once

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "faces_1d.hpp"
#include "faces_2d.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    inline void fill_spatial_faces(Cell<GeomTraits>& cell)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            fill_spatial_faces_1d(cell);
        }
        else if constexpr (GeomTraits::dim_space_v == 2)
        {
            fill_spatial_faces_2d(cell);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_spatial_faces is only implemented for dim_space_v = 1 or 2.");
        }
    }

    template<typename GeomTraits>
    inline void fill_temporal_faces(Cell<GeomTraits>& cell)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            fill_temporal_faces_1d(cell);
        }
        else if constexpr (GeomTraits::dim_space_v == 2)
        {
            fill_temporal_faces_2d(cell);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_temporal_faces is only implemented for dim_space_v = 1 or 2.");
        }
    }

    template<typename GeomTraits>
    inline void fill_faces(Cell<GeomTraits>& cell)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            fill_faces_1d(cell);
        }
        else if constexpr (GeomTraits::dim_space_v == 2)
        {
            fill_faces_2d(cell);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_faces is only implemented for dim_space_v = 1 or 2.");
        }
    }
}

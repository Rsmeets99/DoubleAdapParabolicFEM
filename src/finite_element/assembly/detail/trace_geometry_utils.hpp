#pragma once

#include <cmath>

namespace finite_element::assembly::detail
{
    template<class Geometry>
    double initial_trace_measure(const typename Geometry::Data& geom) noexcept
    {
        if constexpr (Geometry::SpaceType::GT::dim_space_v == 1)
        {
            // Bottom face is the spatial interval at t = 0.
            return geom.hx;
        }
        else if constexpr (Geometry::SpaceType::GT::dim_space_v == 2)
        {
            // Bottom face is the spatial triangle at t = 0.
            const double detJ = geom.J00 * geom.J11 - geom.J01 * geom.J10;
            return std::abs(detJ);
        }
        else
        {
            static_assert(Geometry::SpaceType::GT::dim_space_v == 1 ||
                          Geometry::SpaceType::GT::dim_space_v == 2,
                          "initial_trace_measure only supports dim_space_v = 1 or 2.");
            return 0.0;
        }
    }

    template<class Geometry, class BottomRefPoint>
    auto map_bottom_qp_to_physical(
        const typename Geometry::Data& geom,
        const BottomRefPoint& xi_bottom)
    {
        if constexpr (Geometry::SpaceType::GT::dim_space_v == 1)
        {
            return Geometry::map_to_physical(
                geom,
                typename Geometry::RefPoint{xi_bottom[0], 0.0});
        }
        else
        {
            return Geometry::map_to_physical(
                geom,
                typename Geometry::RefPoint{xi_bottom[0], xi_bottom[1], 0.0});
        }
    }

    template<class Geometry>
    auto spatial_argument_from_space_time_point(
        const typename Geometry::SpaceTimePoint& x)
    {
        if constexpr (Geometry::SpaceType::GT::dim_space_v == 1)
        {
            return x[0];
        }
        else
        {
            return typename Geometry::SpaceType::SpatialPoint{x[0], x[1]};
        }
    }
}
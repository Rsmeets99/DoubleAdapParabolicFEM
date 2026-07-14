#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "../types.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    [[nodiscard]] inline double signed_area_twice_2d(
        const Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "signed_area_twice_2d requires dim_space_v == 2.");

        const auto& v0 = spatial_vertices[static_cast<std::size_t>(cell.spatial_vertex_ids[0])];
        const auto& v1 = spatial_vertices[static_cast<std::size_t>(cell.spatial_vertex_ids[1])];
        const auto& v2 = spatial_vertices[static_cast<std::size_t>(cell.spatial_vertex_ids[2])];

        return (v1[0] - v0[0]) * (v2[1] - v0[1])
             - (v1[1] - v0[1]) * (v2[0] - v0[0]);
    }

    template<typename GeomTraits>
    inline void remap_refinement_edge_after_local_swap_2d(
        Cell<GeomTraits>& cell,
        int a,
        int b) noexcept
    {
        for (int& local : cell.spatial_refinement_edge_local)
        {
            if (local == a)
                local = b;
            else if (local == b)
                local = a;
        }
    }

    template<typename GeomTraits>
    inline void ensure_spatial_vertex_orientation_2d(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "ensure_spatial_vertex_orientation_2d requires dim_space_v == 2.");

        const double area2 = signed_area_twice_2d(cell, spatial_vertices);
        if (std::abs(area2) < 1e-15)
            throw core::invalid_mesh_error("Degenerate 2D spatial triangle.");

        if (area2 < 0.0)
        {
            std::swap(cell.spatial_vertex_ids[1], cell.spatial_vertex_ids[2]);
            remap_refinement_edge_after_local_swap_2d(cell, 1, 2);
        }
    }
}

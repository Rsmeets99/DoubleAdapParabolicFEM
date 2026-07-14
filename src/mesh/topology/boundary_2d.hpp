#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#include "../cell.hpp"
#include "../types.hpp"
#include "boundary_1d.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    [[nodiscard]] inline typename MeshTypes<GeomTraits>::SpatialFaceVertexIds
    sorted_spatial_face_vertex_ids_2d(
        typename MeshTypes<GeomTraits>::SpatialFaceVertexIds face)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "sorted_spatial_face_vertex_ids_2d requires dim_space_v == 2.");
        std::sort(face.begin(), face.end());
        return face;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool point_on_segment_2d(
        const typename MeshTypes<GeomTraits>::SpatialPoint& p,
        const typename MeshTypes<GeomTraits>::SpatialPoint& a,
        const typename MeshTypes<GeomTraits>::SpatialPoint& b,
        double tol = 1e-12)
    {
        const double abx = b[0] - a[0];
        const double aby = b[1] - a[1];
        const double apx = p[0] - a[0];
        const double apy = p[1] - a[1];

        const double cross = abx * apy - aby * apx;
        if (std::abs(cross) > tol)
            return false;

        const double dot = apx * abx + apy * aby;
        const double len2 = abx * abx + aby * aby;
        return -tol <= dot && dot <= len2 + tol;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool spatial_edge_lies_on_boundary_edge_2d(
        const typename MeshTypes<GeomTraits>::SpatialFaceVertexIds& edge,
        const typename MeshTypes<GeomTraits>::SpatialFaceVertexIds& boundary_edge,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        const auto& e0 = spatial_vertices[static_cast<std::size_t>(edge[0])];
        const auto& e1 = spatial_vertices[static_cast<std::size_t>(edge[1])];
        const auto& b0 = spatial_vertices[static_cast<std::size_t>(boundary_edge[0])];
        const auto& b1 = spatial_vertices[static_cast<std::size_t>(boundary_edge[1])];

        return point_on_segment_2d<GeomTraits>(e0, b0, b1) &&
               point_on_segment_2d<GeomTraits>(e1, b0, b1);
    }

    template<typename GeomTraits>
    inline void fill_spatial_boundary_2d(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "fill_spatial_boundary_2d requires dim_space_v == 2.");

        for (int face = 0; face < 3; ++face)
        {
            cell.spatial_boundary[static_cast<std::size_t>(face)] = false;

            const auto edge =
                sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                    cell.spatial_faces[static_cast<std::size_t>(face)].spatial_vertex_ids);

            for (const auto& boundary_edge : spatial_boundary_face_vertex_ids)
            {
                const auto sorted_boundary_edge =
                    sorted_spatial_face_vertex_ids_2d<GeomTraits>(boundary_edge);

                if (edge == sorted_boundary_edge ||
                    spatial_edge_lies_on_boundary_edge_2d<GeomTraits>(
                        edge,
                        sorted_boundary_edge,
                        spatial_vertices))
                {
                    cell.spatial_boundary[static_cast<std::size_t>(face)] = true;
                    break;
                }
            }
        }
    }

    template<typename GeomTraits>
    inline void fill_boundary_2d(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        fill_spatial_boundary_2d(cell, spatial_boundary_face_vertex_ids, spatial_vertices);
        fill_temporal_boundary_1d(cell, temporal_boundary_vertex_ids);
    }
}

#pragma once

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "boundary_1d.hpp"
#include "boundary_2d.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    inline void fill_spatial_boundary(
        Cell<GeomTraits>& cell,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            fill_spatial_boundary_1d(cell, spatial_boundary_vertex_ids);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_spatial_boundary is only implemented for dim_space_v == 1.");
        }
    }

    template<typename GeomTraits>
    inline void fill_temporal_boundary(
        Cell<GeomTraits>& cell,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        if constexpr (GeomTraits::dim_time_v == 1)
        {
            fill_temporal_boundary_1d(cell, temporal_boundary_vertex_ids);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_temporal_boundary is only implemented for dim_time_v == 1.");
        }
    }

    template<typename GeomTraits>
    inline void fill_boundary(
        Cell<GeomTraits>& cell,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            fill_boundary_1d(cell, spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_boundary is only implemented for 1+1D.");
        }
    }

    template<typename GeomTraits>
    inline void fill_spatial_boundary(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        if constexpr (GeomTraits::dim_space_v == 2)
        {
            fill_spatial_boundary_2d(
                cell,
                spatial_boundary_face_vertex_ids,
                spatial_vertices);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_spatial_boundary with face ids is only implemented for dim_space_v == 2.");
        }
    }

    template<typename GeomTraits>
    inline void fill_boundary(
        Cell<GeomTraits>& cell,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        if constexpr (GeomTraits::dim_space_v == 2)
        {
            fill_boundary_2d(
                cell,
                spatial_boundary_face_vertex_ids,
                spatial_vertices,
                temporal_boundary_vertex_ids);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "fill_boundary with spatial face ids is only implemented for dim_space_v == 2.");
        }
    }
}

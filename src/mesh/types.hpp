#pragma once

#include <array>

#include "../core/ids.hpp"

namespace mesh
{
    template<typename GeomTraits>
    struct MeshTypes
    {
        static constexpr int dim_space_v = GeomTraits::dim_space_v;
        static constexpr int dim_time_v  = GeomTraits::dim_time_v;
        static constexpr int dim_v       = GeomTraits::dim_v;
        static constexpr int n_spatial_vertices       = GeomTraits::Tp_vertices;
        static constexpr int n_temporal_vertices      = GeomTraits::Ip_vertices;
        static constexpr int n_spatial_faces          = GeomTraits::Tp_faces;
        static constexpr int n_temporal_faces         = GeomTraits::Ip_faces;
        static constexpr int n_spatial_face_vertices  = GeomTraits::vertices_per_spatial_face;
        static constexpr int n_temporal_face_vertices = GeomTraits::vertices_per_temporal_face;

        using cell_id_type   = core::CellId;
        using vertex_id_type = core::VertexId;

        using SpatialPoint   = std::array<double, dim_space_v>;
        using TemporalPoint  = std::array<double, dim_time_v>;
        using SpaceTimePoint = std::array<double, dim_v>;

        using SpatialVertexIds      = std::array<vertex_id_type, n_spatial_vertices>;
        using TemporalVertexIds     = std::array<vertex_id_type, n_temporal_vertices>;
        using SpatialFaceVertexIds  = std::array<vertex_id_type, n_spatial_face_vertices>;
        using SpatialEdgeVertexIds  = std::array<vertex_id_type, 2>;
        using LocalEdgeIndices      = std::array<int, 2>;
        using SpatialSimplexPoints  = std::array<SpatialPoint, n_spatial_vertices>;

        using SpatialPointKey   = std::array<long long, dim_space_v>;
        using TemporalPointKey  = std::array<long long, dim_time_v>;
    };
}

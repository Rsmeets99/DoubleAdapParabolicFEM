#pragma once

#include <cstddef>

namespace mesh
{
    template<int DimSpace>
    struct MeshTraits
    {
        static_assert(DimSpace >= 1, "MeshTraits requires DimSpace >= 1.");

        static constexpr int dim_space_v = DimSpace;
        static constexpr int dim_time_v  = 1;
        static constexpr int dim_v       = dim_space_v + dim_time_v;

        // Spatial simplex data
        static constexpr int Tp_vertices = (DimSpace == 1 ? 2 : 3);
        static constexpr int Tp_faces    = (DimSpace == 1 ? 2 : 3);

        // Temporal interval data
        static constexpr int Ip_vertices = 2;
        static constexpr int Ip_faces    = 2;

        // Derived prism counts used by FE tables and binary exporters.
        static constexpr int vertices_per_cell = Tp_vertices * Ip_vertices;
        static constexpr int faces_per_cell    = Tp_faces + Ip_faces;
        static constexpr int vertices_per_spatial_face = (DimSpace == 1 ? 1 : 2);
        static constexpr int vertices_per_temporal_face = 1;
    };
}

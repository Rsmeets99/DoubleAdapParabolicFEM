#pragma once

#include "../../../mesh/types.hpp"

namespace finite_element::fespace
{
    template<typename GeomTraits>
    struct InterfaceBase
    {
        int master_cell = -1;
        int master_face = -1;
        int slave_cell  = -1;
        int slave_face  = -1;

        bool is_boundary = false;
        bool is_hanging  = false;
    };

    template<typename GeomTraits>
    struct SpatialInterface : InterfaceBase<GeomTraits>
    {
        using Types = mesh::MeshTypes<GeomTraits>;
        using SpatialFaceVertexIds = typename Types::SpatialFaceVertexIds;
        using TemporalVertexIds    = typename Types::TemporalVertexIds;

        SpatialFaceVertexIds master_spatial_vertex_ids{};
        TemporalVertexIds    master_temporal_vertex_ids{};
        SpatialFaceVertexIds slave_spatial_vertex_ids{};
        TemporalVertexIds    slave_temporal_vertex_ids{};
    };

    template<typename GeomTraits>
    struct TemporalInterface : InterfaceBase<GeomTraits>
    {
        using Types = mesh::MeshTypes<GeomTraits>;
        using SpatialVertexIds = typename Types::SpatialVertexIds;
        using vertex_id_type   = typename Types::vertex_id_type;

        SpatialVertexIds master_spatial_vertex_ids{};
        vertex_id_type   master_temporal_vertex_id = -1;
        SpatialVertexIds slave_spatial_vertex_ids{};
        vertex_id_type   slave_temporal_vertex_id = -1;
    };
}

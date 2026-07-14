#pragma once

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "../detail/vertex_registry.hpp"
#include "../types.hpp"
#include "refinement_type.hpp"
#include "refine_1d.hpp"
#include "refine_2d.hpp"

namespace mesh::refinement
{
    template<typename GeomTraits>
    inline void refine(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        typename MeshTypes<GeomTraits>::cell_id_type cell_id,
        RefinementType refinement_type = RefinementType::none)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            refine_1d(
                cells,
                spatial_vertices,
                temporal_vertices,
                registry,
                spatial_boundary_vertex_ids,
                temporal_boundary_vertex_ids,
                cell_id,
                refinement_type);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "refine is only implemented for 1+1D.");
        }
    }

    template<typename GeomTraits>
    inline void refine(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        typename MeshTypes<GeomTraits>::cell_id_type cell_id,
        RefinementType refinement_type = RefinementType::none)
    {
        if constexpr (GeomTraits::dim_space_v == 2)
        {
            refine_2d(
                cells,
                spatial_vertices,
                temporal_vertices,
                registry,
                spatial_boundary_face_vertex_ids,
                temporal_boundary_vertex_ids,
                cell_id,
                refinement_type);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "refine with spatial boundary faces is only implemented for 2+1D.");
        }
    }

    template<typename GeomTraits>
    inline void refine(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& cell_ids,
        RefinementType refinement_type = RefinementType::none)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            refine_1d(
                cells,
                spatial_vertices,
                temporal_vertices,
                registry,
                spatial_boundary_vertex_ids,
                temporal_boundary_vertex_ids,
                cell_ids,
                refinement_type);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "refine is only implemented for 1+1D.");
        }
    }

    template<typename GeomTraits>
    inline void refine(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& cell_ids,
        RefinementType refinement_type = RefinementType::none)
    {
        if constexpr (GeomTraits::dim_space_v == 2)
        {
            refine_2d(
                cells,
                spatial_vertices,
                temporal_vertices,
                registry,
                spatial_boundary_face_vertex_ids,
                temporal_boundary_vertex_ids,
                cell_ids,
                refinement_type);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "refine with spatial boundary faces is only implemented for 2+1D.");
        }
    }
}

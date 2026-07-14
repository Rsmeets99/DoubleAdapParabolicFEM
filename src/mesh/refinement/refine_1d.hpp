#pragma once

#include <vector>

#include "../../core/exceptions.hpp"
#include "../cell.hpp"
#include "../detail/vertex_registry.hpp"
#include "../topology/boundary.hpp"
#include "../topology/faces.hpp"
#include "../topology/orientation.hpp"
#include "../types.hpp"
#include "refinement_type.hpp"
#include "split_policy.hpp"

namespace mesh::refinement::impl
{
    template<typename GeomTraits>
    inline void finalize_child_cell_1d(
        Cell<GeomTraits>& child,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        topology::ensure_spatial_vertex_orientation(child, spatial_vertices);
        topology::ensure_temporal_vertex_orientation(child, temporal_vertices);
        topology::fill_faces(child);
        topology::fill_boundary(child, spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline typename MeshTypes<GeomTraits>::SpatialPoint spatial_midpoint_1d(
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        typename MeshTypes<GeomTraits>::vertex_id_type left_id,
        typename MeshTypes<GeomTraits>::vertex_id_type right_id)
    {
        using Types = MeshTypes<GeomTraits>;
        typename Types::SpatialPoint xmid{};
        xmid[0] = 0.5 * (spatial_vertices[left_id][0] + spatial_vertices[right_id][0]);
        return xmid;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline typename MeshTypes<GeomTraits>::TemporalPoint temporal_midpoint_1d(
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        typename MeshTypes<GeomTraits>::vertex_id_type t0_id,
        typename MeshTypes<GeomTraits>::vertex_id_type t1_id)
    {
        using Types = MeshTypes<GeomTraits>;
        typename Types::TemporalPoint tmid{};
        tmid[0] = 0.5 * (temporal_vertices[t0_id][0] + temporal_vertices[t1_id][0]);
        return tmid;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline Cell<GeomTraits> make_child_cell(
        typename MeshTypes<GeomTraits>::cell_id_type child_id,
        typename MeshTypes<GeomTraits>::cell_id_type parent_id,
        int generation,
        int spatial_level,
        int temporal_level,
        RefinementType last_split_type,
        const typename MeshTypes<GeomTraits>::SpatialVertexIds& spatial_vertex_ids,
        const typename MeshTypes<GeomTraits>::TemporalVertexIds& temporal_vertex_ids,
        const typename MeshTypes<GeomTraits>::LocalEdgeIndices& spatial_refinement_edge_local)
    {
        Cell<GeomTraits> child{};
        child.cell_id                     = child_id;
        child.parent_id                   = parent_id;
        child.is_leaf                     = true;
        child.generation                  = generation;
        child.spatial_level               = spatial_level;
        child.temporal_level              = temporal_level;
        child.last_split_type             = last_split_type;
        child.spatial_vertex_ids          = spatial_vertex_ids;
        child.temporal_vertex_ids         = temporal_vertex_ids;
        child.spatial_refinement_edge_local = spatial_refinement_edge_local;
        return child;
    }
}

namespace mesh::refinement
{
    template<typename GeomTraits>
    inline void refine_1d(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        mesh::detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        typename MeshTypes<GeomTraits>::cell_id_type cell_id,
        RefinementType refinement_type = RefinementType::none)
    {
        using Types = MeshTypes<GeomTraits>;
        using cell_id_type      = typename Types::cell_id_type;
        using vertex_id_type    = typename Types::vertex_id_type;
        using SpatialVertexIds  = typename Types::SpatialVertexIds;
        using TemporalVertexIds = typename Types::TemporalVertexIds;
        using LocalEdgeIndices  = typename Types::LocalEdgeIndices;

        static_assert(GeomTraits::dim_space_v == 1,
                      "refine_1d is only implemented for 1+1D.");

        if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= cells.size())
            throw core::invalid_mesh_error("refine_1d: cell_id out of range.");

        auto& parent = cells[static_cast<std::size_t>(cell_id)];
        if (!parent.is_leaf)
            throw core::invalid_mesh_error("refine_1d: can only refine a leaf cell.");

        if (refinement_type == RefinementType::none)
            refinement_type = next_split_type<GeomTraits>(parent.generation);

        const vertex_id_type xl_id = parent.spatial_vertex_ids[0];
        const vertex_id_type xr_id = parent.spatial_vertex_ids[1];
        const vertex_id_type t0_id = parent.temporal_vertex_ids[0];
        const vertex_id_type t1_id = parent.temporal_vertex_ids[1];

        const int child_generation = parent.generation + 1;
        const cell_id_type next_id = static_cast<cell_id_type>(cells.size());
        const LocalEdgeIndices child_edge_local{ {0, 1} };

        if (refinement_type == RefinementType::spatial)
        {
            const auto xmid = impl::spatial_midpoint_1d<GeomTraits>(spatial_vertices, xl_id, xr_id);
            const vertex_id_type xm_id = registry.get_or_create_spatial_vertex(spatial_vertices, xmid);

            SpatialVertexIds sv0{};
            SpatialVertexIds sv1{};
            TemporalVertexIds tv{};

            sv0[0] = xl_id; sv0[1] = xm_id;
            sv1[0] = xm_id; sv1[1] = xr_id;
            tv[0]  = t0_id; tv[1]  = t1_id;

            auto c0 = impl::make_child_cell<GeomTraits>(
                next_id,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level,
                refinement_type,
                sv0,
                tv,
                child_edge_local);

            auto c1 = impl::make_child_cell<GeomTraits>(
                next_id + 1,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level,
                refinement_type,
                sv1,
                tv,
                child_edge_local);

            impl::finalize_child_cell_1d(c0, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);
            impl::finalize_child_cell_1d(c1, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);

            parent.is_leaf = false;
            parent.last_split_type = refinement_type;
            parent.children = {c0.cell_id, c1.cell_id};

            cells.push_back(c0);
            cells.push_back(c1);
            return;
        }

        if (refinement_type == RefinementType::temporal)
        {
            const auto tmid = impl::temporal_midpoint_1d<GeomTraits>(temporal_vertices, t0_id, t1_id);
            const vertex_id_type tm_id = registry.get_or_create_temporal_vertex(temporal_vertices, tmid);

            SpatialVertexIds sv{};
            TemporalVertexIds tv0{};
            TemporalVertexIds tv1{};

            sv[0]  = xl_id; sv[1]  = xr_id;
            tv0[0] = t0_id; tv0[1] = tm_id;
            tv1[0] = tm_id; tv1[1] = t1_id;

            auto c0 = impl::make_child_cell<GeomTraits>(
                next_id,
                parent.cell_id,
                child_generation,
                parent.spatial_level,
                parent.temporal_level + 1,
                refinement_type,
                sv,
                tv0,
                child_edge_local);

            auto c1 = impl::make_child_cell<GeomTraits>(
                next_id + 1,
                parent.cell_id,
                child_generation,
                parent.spatial_level,
                parent.temporal_level + 1,
                refinement_type,
                sv,
                tv1,
                child_edge_local);

            impl::finalize_child_cell_1d(c0, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);
            impl::finalize_child_cell_1d(c1, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);

            parent.is_leaf = false;
            parent.last_split_type = refinement_type;
            parent.children = {c0.cell_id, c1.cell_id};

            cells.push_back(c0);
            cells.push_back(c1);
            return;
        }

        if (refinement_type == RefinementType::spacetime)
        {
            const auto xmid = impl::spatial_midpoint_1d<GeomTraits>(spatial_vertices, xl_id, xr_id);
            const auto tmid = impl::temporal_midpoint_1d<GeomTraits>(temporal_vertices, t0_id, t1_id);

            const vertex_id_type xm_id = registry.get_or_create_spatial_vertex(spatial_vertices, xmid);
            const vertex_id_type tm_id = registry.get_or_create_temporal_vertex(temporal_vertices, tmid);

            SpatialVertexIds sv_left{};
            SpatialVertexIds sv_right{};
            TemporalVertexIds tv_bottom{};
            TemporalVertexIds tv_top{};

            sv_left[0]   = xl_id; sv_left[1]   = xm_id;
            sv_right[0]  = xm_id; sv_right[1]  = xr_id;
            tv_bottom[0] = t0_id; tv_bottom[1] = tm_id;
            tv_top[0]    = tm_id; tv_top[1]    = t1_id;

            auto c00 = impl::make_child_cell<GeomTraits>(
                next_id,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level + 1,
                refinement_type,
                sv_left,
                tv_bottom,
                child_edge_local);

            auto c10 = impl::make_child_cell<GeomTraits>(
                next_id + 1,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level + 1,
                refinement_type,
                sv_right,
                tv_bottom,
                child_edge_local);

            auto c01 = impl::make_child_cell<GeomTraits>(
                next_id + 2,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level + 1,
                refinement_type,
                sv_left,
                tv_top,
                child_edge_local);

            auto c11 = impl::make_child_cell<GeomTraits>(
                next_id + 3,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level + 1,
                refinement_type,
                sv_right,
                tv_top,
                child_edge_local);

            impl::finalize_child_cell_1d(c00, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);
            impl::finalize_child_cell_1d(c10, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);
            impl::finalize_child_cell_1d(c01, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);
            impl::finalize_child_cell_1d(c11, spatial_vertices, temporal_vertices,
                                         spatial_boundary_vertex_ids, temporal_boundary_vertex_ids);

            parent.is_leaf = false;
            parent.last_split_type = refinement_type;
            parent.children = {c00.cell_id, c10.cell_id, c01.cell_id, c11.cell_id};

            cells.push_back(c00);
            cells.push_back(c10);
            cells.push_back(c01);
            cells.push_back(c11);
            return;
        }

        throw core::invalid_mesh_error("refine_1d: unsupported RefinementType.");
    }

    template<typename GeomTraits>
    inline void refine_1d(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        mesh::detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<core::VertexId>& spatial_boundary_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& cell_ids,
        RefinementType refinement_type = RefinementType::none)
    {
        for (const auto id : cell_ids)
        {
            refine_1d(
                cells,
                spatial_vertices,
                temporal_vertices,
                registry,
                spatial_boundary_vertex_ids,
                temporal_boundary_vertex_ids,
                id,
                refinement_type);
        }
    }
}
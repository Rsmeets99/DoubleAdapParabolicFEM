#pragma once

#include <array>
#include <cstddef>
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
    inline void finalize_child_cell_2d(
        Cell<GeomTraits>& child,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids)
    {
        topology::ensure_spatial_vertex_orientation(child, spatial_vertices);
        topology::ensure_temporal_vertex_orientation(child, temporal_vertices);
        topology::fill_faces(child);
        topology::fill_boundary(
            child,
            spatial_boundary_face_vertex_ids,
            spatial_vertices,
            temporal_boundary_vertex_ids);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline typename MeshTypes<GeomTraits>::SpatialPoint spatial_midpoint_2d(
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        typename MeshTypes<GeomTraits>::vertex_id_type a_id,
        typename MeshTypes<GeomTraits>::vertex_id_type b_id)
    {
        using Types = MeshTypes<GeomTraits>;
        typename Types::SpatialPoint mid{};
        mid[0] = 0.5 * (spatial_vertices[static_cast<std::size_t>(a_id)][0] +
                        spatial_vertices[static_cast<std::size_t>(b_id)][0]);
        mid[1] = 0.5 * (spatial_vertices[static_cast<std::size_t>(a_id)][1] +
                        spatial_vertices[static_cast<std::size_t>(b_id)][1]);
        return mid;
    }

    [[nodiscard]] inline int opposite_local_vertex_2d(
        const std::array<int, 2>& edge_local)
    {
        for (int local = 0; local < 3; ++local)
        {
            if (local != edge_local[0] && local != edge_local[1])
                return local;
        }

        throw core::invalid_mesh_error("opposite_local_vertex_2d: invalid local edge.");
    }

    [[nodiscard]] inline std::array<int, 2> oriented_refinement_edge_2d(
        const std::array<int, 2>& edge_local)
    {
        const int a = edge_local[0];
        const int b = edge_local[1];

        if ((a == 0 && b == 1) || (a == 1 && b == 0))
            return {0, 1};
        if ((a == 1 && b == 2) || (a == 2 && b == 1))
            return {1, 2};
        if ((a == 2 && b == 0) || (a == 0 && b == 2))
            return {2, 0};

        throw core::invalid_mesh_error("oriented_refinement_edge_2d: invalid local edge.");
    }

    template<typename GeomTraits>
    [[nodiscard]] inline typename MeshTypes<GeomTraits>::TemporalPoint temporal_midpoint_2d(
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        typename MeshTypes<GeomTraits>::vertex_id_type t0_id,
        typename MeshTypes<GeomTraits>::vertex_id_type t1_id)
    {
        using Types = MeshTypes<GeomTraits>;
        typename Types::TemporalPoint mid{};
        mid[0] = 0.5 * (temporal_vertices[static_cast<std::size_t>(t0_id)][0] +
                        temporal_vertices[static_cast<std::size_t>(t1_id)][0]);
        return mid;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline std::array<typename MeshTypes<GeomTraits>::SpatialVertexIds, 2>
    spatial_nvb_child_vertices_2d(
        const Cell<GeomTraits>& parent,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& mutable_spatial_vertices,
        mesh::detail::VertexRegistry<GeomTraits>& registry)
    {
        using Types = MeshTypes<GeomTraits>;
        using vertex_id_type = typename Types::vertex_id_type;
        using SpatialVertexIds = typename Types::SpatialVertexIds;

        if (parent.spatial_refinement_edge_local[0] < 0 ||
            parent.spatial_refinement_edge_local[0] >= 3 ||
            parent.spatial_refinement_edge_local[1] < 0 ||
            parent.spatial_refinement_edge_local[1] >= 3 ||
            parent.spatial_refinement_edge_local[0] == parent.spatial_refinement_edge_local[1])
        {
            throw core::invalid_mesh_error("spatial_nvb_child_vertices_2d: invalid spatial refinement edge.");
        }

        const auto refinement_edge_local =
            oriented_refinement_edge_2d(parent.spatial_refinement_edge_local);
        const int edge_local_0 = refinement_edge_local[0];
        const int edge_local_1 = refinement_edge_local[1];
        const int opposite_local =
            opposite_local_vertex_2d(refinement_edge_local);

        const vertex_id_type a_id =
            parent.spatial_vertex_ids[static_cast<std::size_t>(edge_local_0)];
        const vertex_id_type b_id =
            parent.spatial_vertex_ids[static_cast<std::size_t>(edge_local_1)];
        const vertex_id_type c_id =
            parent.spatial_vertex_ids[static_cast<std::size_t>(opposite_local)];

        const auto midpoint =
            spatial_midpoint_2d<GeomTraits>(spatial_vertices, a_id, b_id);
        const vertex_id_type m_id =
            registry.get_or_create_spatial_vertex(mutable_spatial_vertices, midpoint);

        SpatialVertexIds sv0{};
        SpatialVertexIds sv1{};

        sv0[0] = m_id;
        sv0[1] = c_id;
        sv0[2] = a_id;

        sv1[0] = m_id;
        sv1[1] = b_id;
        sv1[2] = c_id;

        return {sv0, sv1};
    }

    template<typename GeomTraits>
    [[nodiscard]] inline Cell<GeomTraits> make_child_cell_2d(
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
        child.cell_id                       = child_id;
        child.parent_id                     = parent_id;
        child.is_leaf                       = true;
        child.generation                    = generation;
        child.spatial_level                 = spatial_level;
        child.temporal_level                = temporal_level;
        child.last_split_type               = last_split_type;
        child.spatial_vertex_ids            = spatial_vertex_ids;
        child.temporal_vertex_ids           = temporal_vertex_ids;
        child.spatial_refinement_edge_local = spatial_refinement_edge_local;
        return child;
    }
}

namespace mesh::refinement
{
    template<typename GeomTraits>
    inline void refine_2d(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        mesh::detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        typename MeshTypes<GeomTraits>::cell_id_type cell_id,
        RefinementType refinement_type = RefinementType::none)
    {
        using Types = MeshTypes<GeomTraits>;
        using cell_id_type     = typename Types::cell_id_type;
        using vertex_id_type   = typename Types::vertex_id_type;
        using TemporalVertexIds = typename Types::TemporalVertexIds;
        using LocalEdgeIndices = typename Types::LocalEdgeIndices;

        static_assert(GeomTraits::dim_space_v == 2,
                      "refine_2d is only implemented for 2+1D.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "refine_2d requires dim_time_v == 1.");

        if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= cells.size())
            throw core::invalid_mesh_error("refine_2d: cell_id out of range.");

        auto& parent = cells[static_cast<std::size_t>(cell_id)];
        if (!parent.is_leaf)
            throw core::invalid_mesh_error("refine_2d: can only refine a leaf cell.");

        if (refinement_type == RefinementType::none)
            refinement_type = next_split_type<GeomTraits>(parent.generation);

        if (refinement_type != RefinementType::spatial &&
            refinement_type != RefinementType::temporal &&
            refinement_type != RefinementType::spacetime)
        {
            throw core::invalid_mesh_error(
                "refine_2d: unsupported RefinementType for 2+1D.");
        }

        const int child_generation = parent.generation + 1;
        const cell_id_type next_id = static_cast<cell_id_type>(cells.size());

        // Deterministic NVB convention for this codebase:
        // the midpoint/newest vertex is stored at local index 0, and each child
        // refines next along the edge opposite that newest vertex.
        const LocalEdgeIndices child_edge_local{{1, 2}};

        if (refinement_type == RefinementType::spatial)
        {
            const auto spatial_child_vertices =
                impl::spatial_nvb_child_vertices_2d<GeomTraits>(
                    parent,
                    spatial_vertices,
                    spatial_vertices,
                    registry);
            const TemporalVertexIds tv = parent.temporal_vertex_ids;

            auto c0 = impl::make_child_cell_2d<GeomTraits>(
                next_id,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level,
                refinement_type,
                spatial_child_vertices[0],
                tv,
                child_edge_local);

            auto c1 = impl::make_child_cell_2d<GeomTraits>(
                next_id + 1,
                parent.cell_id,
                child_generation,
                parent.spatial_level + 1,
                parent.temporal_level,
                refinement_type,
                spatial_child_vertices[1],
                tv,
                child_edge_local);

            impl::finalize_child_cell_2d(
                c0,
                spatial_vertices,
                temporal_vertices,
                spatial_boundary_face_vertex_ids,
                temporal_boundary_vertex_ids);
            impl::finalize_child_cell_2d(
                c1,
                spatial_vertices,
                temporal_vertices,
                spatial_boundary_face_vertex_ids,
                temporal_boundary_vertex_ids);

            parent.is_leaf = false;
            parent.last_split_type = refinement_type;
            parent.children = {c0.cell_id, c1.cell_id};

            cells.push_back(c0);
            cells.push_back(c1);
            return;
        }

        const vertex_id_type t0_id = parent.temporal_vertex_ids[0];
        const vertex_id_type t1_id = parent.temporal_vertex_ids[1];
        const auto tmid =
            impl::temporal_midpoint_2d<GeomTraits>(temporal_vertices, t0_id, t1_id);
        const vertex_id_type tm_id =
            registry.get_or_create_temporal_vertex(temporal_vertices, tmid);

        TemporalVertexIds tv_bottom{};
        TemporalVertexIds tv_top{};

        tv_bottom[0] = t0_id;
        tv_bottom[1] = tm_id;
        tv_top[0] = tm_id;
        tv_top[1] = t1_id;

        if (refinement_type == RefinementType::temporal)
        {
            const auto child_edge = parent.spatial_refinement_edge_local;
            auto c0 = impl::make_child_cell_2d<GeomTraits>(
                next_id,
                parent.cell_id,
                child_generation,
                parent.spatial_level,
                parent.temporal_level + 1,
                refinement_type,
                parent.spatial_vertex_ids,
                tv_bottom,
                child_edge);

            auto c1 = impl::make_child_cell_2d<GeomTraits>(
                next_id + 1,
                parent.cell_id,
                child_generation,
                parent.spatial_level,
                parent.temporal_level + 1,
                refinement_type,
                parent.spatial_vertex_ids,
                tv_top,
                child_edge);

            impl::finalize_child_cell_2d(
                c0,
                spatial_vertices,
                temporal_vertices,
                spatial_boundary_face_vertex_ids,
                temporal_boundary_vertex_ids);
            impl::finalize_child_cell_2d(
                c1,
                spatial_vertices,
                temporal_vertices,
                spatial_boundary_face_vertex_ids,
                temporal_boundary_vertex_ids);

            parent.is_leaf = false;
            parent.last_split_type = refinement_type;
            parent.children = {c0.cell_id, c1.cell_id};

            cells.push_back(c0);
            cells.push_back(c1);
            return;
        }

        const auto spatial_child_vertices =
            impl::spatial_nvb_child_vertices_2d<GeomTraits>(
                parent,
                spatial_vertices,
                spatial_vertices,
                registry);

        auto c00 = impl::make_child_cell_2d<GeomTraits>(
            next_id,
            parent.cell_id,
            child_generation,
            parent.spatial_level + 1,
            parent.temporal_level + 1,
            refinement_type,
            spatial_child_vertices[0],
            tv_bottom,
            child_edge_local);

        auto c10 = impl::make_child_cell_2d<GeomTraits>(
            next_id + 1,
            parent.cell_id,
            child_generation,
            parent.spatial_level + 1,
            parent.temporal_level + 1,
            refinement_type,
            spatial_child_vertices[1],
            tv_bottom,
            child_edge_local);

        auto c01 = impl::make_child_cell_2d<GeomTraits>(
            next_id + 2,
            parent.cell_id,
            child_generation,
            parent.spatial_level + 1,
            parent.temporal_level + 1,
            refinement_type,
            spatial_child_vertices[0],
            tv_top,
            child_edge_local);

        auto c11 = impl::make_child_cell_2d<GeomTraits>(
            next_id + 3,
            parent.cell_id,
            child_generation,
            parent.spatial_level + 1,
            parent.temporal_level + 1,
            refinement_type,
            spatial_child_vertices[1],
            tv_top,
            child_edge_local);

        impl::finalize_child_cell_2d(
            c00,
            spatial_vertices,
            temporal_vertices,
            spatial_boundary_face_vertex_ids,
            temporal_boundary_vertex_ids);
        impl::finalize_child_cell_2d(
            c10,
            spatial_vertices,
            temporal_vertices,
            spatial_boundary_face_vertex_ids,
            temporal_boundary_vertex_ids);
        impl::finalize_child_cell_2d(
            c01,
            spatial_vertices,
            temporal_vertices,
            spatial_boundary_face_vertex_ids,
            temporal_boundary_vertex_ids);
        impl::finalize_child_cell_2d(
            c11,
            spatial_vertices,
            temporal_vertices,
            spatial_boundary_face_vertex_ids,
            temporal_boundary_vertex_ids);

        parent.is_leaf = false;
        parent.last_split_type = refinement_type;
        parent.children = {c00.cell_id, c10.cell_id, c01.cell_id, c11.cell_id};

        cells.push_back(c00);
        cells.push_back(c10);
        cells.push_back(c01);
        cells.push_back(c11);
    }

    template<typename GeomTraits>
    inline void refine_2d(
        std::vector<Cell<GeomTraits>>& cells,
        std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices,
        mesh::detail::VertexRegistry<GeomTraits>& registry,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>&
            spatial_boundary_face_vertex_ids,
        const std::vector<core::VertexId>& temporal_boundary_vertex_ids,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& cell_ids,
        RefinementType refinement_type = RefinementType::none)
    {
        for (const auto id : cell_ids)
        {
            refine_2d(
                cells,
                spatial_vertices,
                temporal_vertices,
                registry,
                spatial_boundary_face_vertex_ids,
                temporal_boundary_vertex_ids,
                id,
                refinement_type);
        }
    }
}

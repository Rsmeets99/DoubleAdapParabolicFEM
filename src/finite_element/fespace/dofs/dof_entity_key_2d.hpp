#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <tuple>

#include "../../../mesh/topology/boundary_2d.hpp"
#include "physical_dof_coords.hpp"

namespace finite_element::fespace
{
    enum class DofSpatialEntityKind2D
    {
        vertex,
        edge,
        triangle_interior
    };

    enum class DofTemporalEntityKind2D
    {
        bottom_endpoint,
        top_endpoint,
        interval_interior
    };

    struct DofEntityKey2D
    {
        DofSpatialEntityKind2D spatial_kind =
            DofSpatialEntityKind2D::triangle_interior;
        std::array<int, 3> spatial_entity_ids{-1, -1, -1};
        std::array<int, 3> spatial_node_tuple{-1, -1, -1};
        int spatial_edge_node_ordinal = -1;

        DofTemporalEntityKind2D temporal_kind =
            DofTemporalEntityKind2D::interval_interior;
        int temporal_vertex_id = -1;
        std::array<int, 2> temporal_interval_ids{-1, -1};
        int temporal_node_ordinal = -1;

        bool discontinuous_time = false;

        [[nodiscard]] bool operator<(const DofEntityKey2D& other) const noexcept
        {
            return std::tie(
                       spatial_kind,
                       spatial_entity_ids,
                       spatial_node_tuple,
                       spatial_edge_node_ordinal,
                       temporal_kind,
                       temporal_vertex_id,
                       temporal_interval_ids,
                       temporal_node_ordinal,
                       discontinuous_time) <
                   std::tie(
                       other.spatial_kind,
                       other.spatial_entity_ids,
                       other.spatial_node_tuple,
                       other.spatial_edge_node_ordinal,
                       other.temporal_kind,
                       other.temporal_vertex_id,
                       other.temporal_interval_ids,
                       other.temporal_node_ordinal,
                       other.discontinuous_time);
        }

        [[nodiscard]] bool operator==(const DofEntityKey2D&) const noexcept =
            default;
    };

    struct DofEntityKey2DHash
    {
        [[nodiscard]] std::size_t operator()(
            const DofEntityKey2D& key) const noexcept
        {
            std::size_t seed = 0;
            const auto combine = [&](const int value)
            {
                const std::size_t h = std::hash<int>{}(value);
                seed ^= h + 0x9e3779b97f4a7c15ULL + (seed << 6U) + (seed >> 2U);
            };
            const auto combine_array = [&](const auto& values)
            {
                for (const int value : values)
                    combine(value);
            };

            combine(static_cast<int>(key.spatial_kind));
            combine_array(key.spatial_entity_ids);
            combine_array(key.spatial_node_tuple);
            combine(key.spatial_edge_node_ordinal);
            combine(static_cast<int>(key.temporal_kind));
            combine(key.temporal_vertex_id);
            combine_array(key.temporal_interval_ids);
            combine(key.temporal_node_ordinal);
            combine(key.discontinuous_time ? 1 : 0);

            return seed;
        }
    };

    template<typename SpaceTimePoint>
    struct DofEntityKeyDebug2D
    {
        DofEntityKey2D key{};
        SpaceTimePoint physical_coord{};
        bool spatial_boundary_eliminated = false;
    };

    namespace detail
    {
        [[nodiscard]] inline std::array<int, 2> sorted_edge_ids(
            int a,
            int b) noexcept
        {
            if (b < a)
                std::swap(a, b);
            return {a, b};
        }

        [[nodiscard]] inline std::array<int, 3> sorted_triangle_ids(
            std::array<int, 3> ids) noexcept
        {
            std::sort(ids.begin(), ids.end());
            return ids;
        }

        [[nodiscard]] inline int local_face_vertex_0_2d(const int face)
        {
            switch (face)
            {
                case 0: return 0;
                case 1: return 1;
                case 2: return 2;
                default:
                    throw std::runtime_error(
                        "DofEntityKey2D: invalid local spatial face.");
            }
        }

        [[nodiscard]] inline int local_face_vertex_1_2d(const int face)
        {
            switch (face)
            {
                case 0: return 1;
                case 1: return 2;
                case 2: return 0;
                default:
                    throw std::runtime_error(
                        "DofEntityKey2D: invalid local spatial face.");
            }
        }

        template<class Space, class Point>
        [[nodiscard]] bool point_on_spatial_boundary_2d(
            const Space& space,
            const Point& p)
        {
            using GT = typename Space::GT;
            typename Space::SpatialPoint x{};
            x[0] = p[0];
            x[1] = p[1];

            const auto& mesh = space.mesh_ref();
            for (const auto& boundary_edge :
                 mesh.spatial_boundary_face_vertex_ids())
            {
                const auto& a =
                    mesh.spatial_vertices()[
                        static_cast<std::size_t>(boundary_edge[0])];
                const auto& b =
                    mesh.spatial_vertices()[
                        static_cast<std::size_t>(boundary_edge[1])];
                if (mesh::topology::point_on_segment_2d<GT>(x, a, b))
                    return true;
            }

            return false;
        }

        template<class Space>
        [[nodiscard]] bool local_dof_on_spatial_boundary_2d(
            const Space& space,
            const int cell_id,
            const int local_index)
        {
            using ElemTables = typename Space::ElemTables;
            const auto& cell = space.mesh_ref().cell(cell_id);
            const auto& meta = ElemTables::meta(local_index);

            for (int k = 0; k < meta.num_spatial_faces; ++k)
            {
                const int face = meta.spatial_faces[static_cast<std::size_t>(k)];
                if (face >= 0 && cell.spatial_boundary[static_cast<std::size_t>(face)])
                    return true;
            }

            const auto p =
                finite_element::fespace::physical_dof_coord(
                    space,
                    cell_id,
                    local_index);
            return point_on_spatial_boundary_2d(space, p);
        }

        template<class Space>
        [[nodiscard]] DofEntityKey2D make_spatial_entity_key_part_2d(
            const Space& space,
            const int cell_id,
            const int local_index)
        {
            using FETraits = typename Space::FETraitsType;
            using ElemTables = typename Space::ElemTables;
            using SpatialNodes = typename FETraits::SpatialNodes;

            static_assert(Space::GT::dim_space_v == 2,
                          "DofEntityKey2D requires 2+1D spaces.");

            const auto& cell = space.mesh_ref().cell(cell_id);
            const int spatial_node = ElemTables::spatial_node_id(local_index);
            const auto spatial_meta =
                SpatialNodes::node_meta[static_cast<std::size_t>(spatial_node)];

            DofEntityKey2D key{};

            if (spatial_meta.vertex >= 0)
            {
                key.spatial_kind = DofSpatialEntityKind2D::vertex;
                key.spatial_entity_ids = {
                    cell.spatial_vertex_ids[
                        static_cast<std::size_t>(spatial_meta.vertex)],
                    -1,
                    -1};
                key.spatial_node_tuple = {0, 0, 0};
                key.spatial_edge_node_ordinal = -1;
                return key;
            }

            if (spatial_meta.num_spatial_faces == 1)
            {
                const int face = spatial_meta.spatial_faces[0];
                const int local_v0 = local_face_vertex_0_2d(face);
                const int local_v1 = local_face_vertex_1_2d(face);
                const int v0 =
                    cell.spatial_vertex_ids[static_cast<std::size_t>(local_v0)];
                const int v1 =
                    cell.spatial_vertex_ids[static_cast<std::size_t>(local_v1)];
                const auto edge_ids = sorted_edge_ids(v0, v1);

                int edge_ordinal =
                    SpatialNodes::face_ordinal
                        [static_cast<std::size_t>(spatial_node)]
                        [static_cast<std::size_t>(face)];
                if (edge_ordinal < 0)
                {
                    throw std::runtime_error(
                        "DofEntityKey2D: edge node lacks face ordinal.");
                }

                if (v0 != edge_ids[0])
                    edge_ordinal = FETraits::p_space_v - edge_ordinal;

                key.spatial_kind = DofSpatialEntityKind2D::edge;
                key.spatial_entity_ids = {edge_ids[0], edge_ids[1], -1};
                key.spatial_node_tuple = {-1, -1, -1};
                key.spatial_edge_node_ordinal = edge_ordinal;
                return key;
            }

            key.spatial_kind = DofSpatialEntityKind2D::triangle_interior;
            key.spatial_entity_ids =
                sorted_triangle_ids(cell.spatial_vertex_ids);
            key.spatial_edge_node_ordinal = -1;

            const auto local_bary =
                SpatialNodes::barycentric_tuples[
                    static_cast<std::size_t>(spatial_node)];
            key.spatial_node_tuple = {-1, -1, -1};
            for (int local_vertex = 0; local_vertex < 3; ++local_vertex)
            {
                const int global_vertex =
                    cell.spatial_vertex_ids[
                        static_cast<std::size_t>(local_vertex)];
                const auto it = std::find(
                    key.spatial_entity_ids.begin(),
                    key.spatial_entity_ids.end(),
                    global_vertex);
                if (it == key.spatial_entity_ids.end())
                {
                    throw std::runtime_error(
                        "DofEntityKey2D: local vertex missing from canonical triangle.");
                }

                const std::size_t canonical_index =
                    static_cast<std::size_t>(
                        std::distance(key.spatial_entity_ids.begin(), it));
                key.spatial_node_tuple[canonical_index] =
                    local_bary[static_cast<std::size_t>(local_vertex)];
            }

            return key;
        }

        template<class Space>
        void fill_temporal_entity_key_part_2d(
            const Space& space,
            const int cell_id,
            const int local_index,
            DofEntityKey2D& key)
        {
            using FETraits = typename Space::FETraitsType;
            using ElemTables = typename Space::ElemTables;

            const auto& cell = space.mesh_ref().cell(cell_id);
            const int temporal_node =
                ElemTables::temporal_node_id(local_index);

            key.discontinuous_time =
                !Space::PolicyType::continuous_in_time;

            if constexpr (!Space::PolicyType::continuous_in_time)
            {
                key.temporal_interval_ids = {
                    cell.temporal_vertex_ids[0],
                    cell.temporal_vertex_ids[1]};
                key.temporal_node_ordinal = temporal_node;

                if (temporal_node == 0)
                {
                    key.temporal_kind =
                        DofTemporalEntityKind2D::bottom_endpoint;
                    key.temporal_vertex_id = cell.temporal_vertex_ids[0];
                    return;
                }

                if (temporal_node == FETraits::p_time_v)
                {
                    key.temporal_kind = DofTemporalEntityKind2D::top_endpoint;
                    key.temporal_vertex_id = cell.temporal_vertex_ids[1];
                    return;
                }

                key.temporal_kind = DofTemporalEntityKind2D::interval_interior;
                key.temporal_vertex_id = -1;
                return;
            }

            key.temporal_interval_ids = {
                cell.temporal_vertex_ids[0],
                cell.temporal_vertex_ids[1]};
            key.temporal_node_ordinal = temporal_node;

            if (temporal_node == 0)
            {
                key.temporal_kind = DofTemporalEntityKind2D::bottom_endpoint;
                key.temporal_vertex_id = cell.temporal_vertex_ids[0];
                key.temporal_interval_ids = {-1, -1};
                key.temporal_node_ordinal = 0;
                return;
            }

            if (temporal_node == FETraits::p_time_v)
            {
                key.temporal_kind =
                    Space::PolicyType::continuous_in_time
                        ? DofTemporalEntityKind2D::bottom_endpoint
                        : DofTemporalEntityKind2D::top_endpoint;
                key.temporal_vertex_id = cell.temporal_vertex_ids[1];
                key.temporal_interval_ids = {-1, -1};
                key.temporal_node_ordinal = 0;
                return;
            }

            key.temporal_kind = DofTemporalEntityKind2D::interval_interior;
            key.temporal_vertex_id = -1;
        }
    }

    template<class Space>
    [[nodiscard]] DofEntityKeyDebug2D<typename Space::SpaceTimePoint>
    make_dof_entity_key_2d(
        const Space& space,
        const int cell_id,
        const int local_index)
    {
        static_assert(Space::GT::dim_space_v == 2,
                      "make_dof_entity_key_2d requires 2+1D spaces.");

        auto key =
            detail::make_spatial_entity_key_part_2d(
                space,
                cell_id,
                local_index);
        detail::fill_temporal_entity_key_part_2d(
            space,
            cell_id,
            local_index,
            key);

        return {
            key,
            finite_element::fespace::physical_dof_coord(
                space,
                cell_id,
                local_index),
            detail::local_dof_on_spatial_boundary_2d(
                space,
                cell_id,
                local_index)};
    }
}

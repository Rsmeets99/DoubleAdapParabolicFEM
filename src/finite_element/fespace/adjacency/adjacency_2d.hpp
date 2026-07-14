#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "interface.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../mesh/topology/spatial_edge_adjacency_2d.hpp"
#include "../../../mesh/topology/temporal_keys.hpp"

namespace finite_element::fespace
{
    template<typename GeomTraits, typename Policy>
    struct Adjacency;

    namespace detail::adjacency_impl
    {
        template<typename GeomTraits>
        struct SpatialFaceRecord2D
        {
            using Types = mesh::MeshTypes<GeomTraits>;
            using Edge = typename Types::SpatialFaceVertexIds;

            int cell_id = -1;
            int face_id = -1;

            Edge edge{};
            Edge sorted_edge{};

            int t0_id = -1;
            int t1_id = -1;
            mesh::topology::TimeIntervalIdKey interval_id_key{};
            double t0 = 0.0;
            double t1 = 0.0;

            bool boundary = false;
        };

        template<typename GeomTraits>
        struct TemporalFaceRecord2D
        {
            using Types = mesh::MeshTypes<GeomTraits>;
            using SpatialVertexIds = typename Types::SpatialVertexIds;

            int cell_id = -1;
            int face_id = -1;

            SpatialVertexIds vertices{};
            SpatialVertexIds sorted_vertices{};

            int temporal_vertex_id = -1;
            bool boundary = false;
        };

        struct TemporalMatchingStats2D
        {
            std::size_t groups = 0;
            std::size_t records = 0;
            std::size_t max_group_records = 0;
            std::size_t old_all_pairs = 0;
            std::size_t grid_candidate_pairs = 0;
            std::size_t exact_overlap_tests = 0;
            std::size_t bbox_rejected_pairs = 0;
            std::size_t same_face_or_cell_rejected_pairs = 0;
            std::size_t grid_buckets = 0;
            std::size_t grid_entries = 0;
            std::size_t fallback_pairwise_groups = 0;
            std::size_t fallback_degenerate_bbox_groups = 0;
        };

        struct SpatialMatchingStats2D
        {
            std::size_t groups = 0;
            std::size_t records = 0;
            std::size_t max_group_records = 0;
            std::size_t old_all_pairs = 0;
            std::size_t interval_candidate_pairs = 0;
            std::size_t positive_overlap_tests = 0;
            std::size_t disjoint_breaks = 0;
            std::size_t disjoint_skipped_pairs = 0;
            std::size_t same_cell_rejected_pairs = 0;
            std::size_t debug_reference_groups = 0;
            std::size_t debug_reference_mismatches = 0;
        };

        template<typename GeomTraits>
        struct SpatialBoundingBox2D
        {
            double min_x = 0.0;
            double max_x = 0.0;
            double min_y = 0.0;
            double max_y = 0.0;
        };

        template<typename GeomTraits>
        [[nodiscard]] inline bool same_time_interval_2d(
            const SpatialFaceRecord2D<GeomTraits>& a,
            const SpatialFaceRecord2D<GeomTraits>& b) noexcept
        {
            return a.interval_id_key == b.interval_id_key;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool contains_time_interval_2d(
            const SpatialFaceRecord2D<GeomTraits>& outer,
            const SpatialFaceRecord2D<GeomTraits>& inner) noexcept
        {
            return (outer.t0 <= inner.t0 && inner.t1 <= outer.t1) &&
                   !same_time_interval_2d(outer, inner);
        }

        template<typename GeomTraits>
        inline void sort_spatial_face_group_2d(
            std::vector<SpatialFaceRecord2D<GeomTraits>>& group)
        {
            std::sort(
                group.begin(),
                group.end(),
                [](const auto& a, const auto& b)
                {
                    if (a.t0 != b.t0) return a.t0 < b.t0;
                    if (a.t1 != b.t1) return a.t1 > b.t1;
                    if (a.interval_id_key != b.interval_id_key)
                        return a.interval_id_key < b.interval_id_key;
                    if (a.cell_id != b.cell_id) return a.cell_id < b.cell_id;
                    return a.face_id < b.face_id;
                });
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool point_in_triangle_2d(
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& p,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& a,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& b,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& c,
            double tol = 1.0e-12)
        {
            const double area =
                (b[0] - a[0]) * (c[1] - a[1]) -
                (b[1] - a[1]) * (c[0] - a[0]);

            const double w0 =
                ((b[0] - p[0]) * (c[1] - p[1]) -
                 (b[1] - p[1]) * (c[0] - p[0])) / area;
            const double w1 =
                ((c[0] - p[0]) * (a[1] - p[1]) -
                 (c[1] - p[1]) * (a[0] - p[0])) / area;
            const double w2 = 1.0 - w0 - w1;

            return w0 >= -tol && w1 >= -tol && w2 >= -tol;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool triangle_contains_triangle_2d(
            const typename mesh::MeshTypes<GeomTraits>::SpatialVertexIds& outer,
            const typename mesh::MeshTypes<GeomTraits>::SpatialVertexIds& inner,
            const std::vector<typename mesh::MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
        {
            if (mesh::topology::sorted_spatial_cell_vertex_ids_2d<GeomTraits>(outer) ==
                mesh::topology::sorted_spatial_cell_vertex_ids_2d<GeomTraits>(inner))
            {
                return false;
            }

            const auto& a = spatial_vertices[static_cast<std::size_t>(outer[0])];
            const auto& b = spatial_vertices[static_cast<std::size_t>(outer[1])];
            const auto& c = spatial_vertices[static_cast<std::size_t>(outer[2])];

            for (const auto vid : inner)
            {
                const auto& p = spatial_vertices[static_cast<std::size_t>(vid)];
                if (!point_in_triangle_2d<GeomTraits>(p, a, b, c))
                    return false;
            }

            return true;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline double cross_2d(
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& a,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& b,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& c) noexcept
        {
            return (b[0] - a[0]) * (c[1] - a[1]) -
                   (b[1] - a[1]) * (c[0] - a[0]);
        }

        template<typename GeomTraits>
        [[nodiscard]] inline double polygon_area_2d(
            const std::vector<typename mesh::MeshTypes<GeomTraits>::SpatialPoint>& polygon) noexcept
        {
            if (polygon.size() < 3)
                return 0.0;

            double area_twice = 0.0;
            for (std::size_t i = 0; i < polygon.size(); ++i)
            {
                const auto& a = polygon[i];
                const auto& b = polygon[(i + 1) % polygon.size()];
                area_twice += a[0] * b[1] - a[1] * b[0];
            }

            return 0.5 * std::abs(area_twice);
        }

        template<typename GeomTraits>
        [[nodiscard]] inline typename mesh::MeshTypes<GeomTraits>::SpatialPoint
        line_intersection_2d(
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& p0,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& p1,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& q0,
            const typename mesh::MeshTypes<GeomTraits>::SpatialPoint& q1,
            double tol = 1.0e-14) noexcept
        {
            const double rx = p1[0] - p0[0];
            const double ry = p1[1] - p0[1];
            const double sx = q1[0] - q0[0];
            const double sy = q1[1] - q0[1];
            const double denom = rx * sy - ry * sx;

            if (std::abs(denom) <= tol)
                return p1;

            const double qpx = q0[0] - p0[0];
            const double qpy = q0[1] - p0[1];
            const double t = (qpx * sy - qpy * sx) / denom;
            return {p0[0] + t * rx, p0[1] + t * ry};
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool triangles_overlap_positive_area_2d(
            const typename mesh::MeshTypes<GeomTraits>::SpatialVertexIds& a_vertices,
            const typename mesh::MeshTypes<GeomTraits>::SpatialVertexIds& b_vertices,
            const std::vector<typename mesh::MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
            double area_tol = 1.0e-14)
        {
            using SpatialPoint = typename mesh::MeshTypes<GeomTraits>::SpatialPoint;

            std::vector<SpatialPoint> polygon{
                spatial_vertices[static_cast<std::size_t>(a_vertices[0])],
                spatial_vertices[static_cast<std::size_t>(a_vertices[1])],
                spatial_vertices[static_cast<std::size_t>(a_vertices[2])]
            };

            const SpatialPoint clip_vertices[3]{
                spatial_vertices[static_cast<std::size_t>(b_vertices[0])],
                spatial_vertices[static_cast<std::size_t>(b_vertices[1])],
                spatial_vertices[static_cast<std::size_t>(b_vertices[2])]
            };

            const double orientation =
                cross_2d<GeomTraits>(
                    clip_vertices[0],
                    clip_vertices[1],
                    clip_vertices[2]);
            if (std::abs(orientation) <= area_tol)
                return false;

            const double sign = orientation >= 0.0 ? 1.0 : -1.0;

            for (int edge = 0; edge < 3; ++edge)
            {
                if (polygon.empty())
                    return false;

                const auto& c0 = clip_vertices[edge];
                const auto& c1 = clip_vertices[(edge + 1) % 3];
                std::vector<SpatialPoint> clipped;
                clipped.reserve(polygon.size() + 1);

                const auto inside = [&](const SpatialPoint& p)
                {
                    return sign * cross_2d<GeomTraits>(c0, c1, p) >=
                           -area_tol;
                };

                SpatialPoint previous = polygon.back();
                bool previous_inside = inside(previous);
                for (const auto& current : polygon)
                {
                    const bool current_inside = inside(current);
                    if (current_inside != previous_inside)
                    {
                        clipped.push_back(
                            line_intersection_2d<GeomTraits>(
                                previous,
                                current,
                                c0,
                                c1));
                    }
                    if (current_inside)
                        clipped.push_back(current);

                    previous = current;
                    previous_inside = current_inside;
                }

                polygon = std::move(clipped);
            }

            return polygon_area_2d<GeomTraits>(polygon) > area_tol;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline std::unordered_map<
            mesh::topology::SpatialEdgeKey2D<GeomTraits>,
            std::vector<SpatialFaceRecord2D<GeomTraits>>,
            mesh::topology::SpatialEdgeKey2DHash<GeomTraits>>
        make_spatial_face_groups_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<int>& active_cells)
        {
            using EdgeKey = mesh::topology::SpatialEdgeKey2D<GeomTraits>;
            using EdgeHash = mesh::topology::SpatialEdgeKey2DHash<GeomTraits>;

            std::unordered_map<EdgeKey, std::vector<SpatialFaceRecord2D<GeomTraits>>, EdgeHash> groups;
            groups.reserve(active_cells.size() * 3);

            for (const int cell_id : active_cells)
            {
                const auto& cell = mesh.cell(cell_id);
                for (int face = 0; face < 3; ++face)
                {
                    SpatialFaceRecord2D<GeomTraits> rec;
                    rec.cell_id = cell_id;
                    rec.face_id = face;
                    rec.edge = cell.spatial_faces[face].spatial_vertex_ids;
                    rec.sorted_edge =
                        mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                            rec.edge);
                    rec.t0_id = cell.spatial_faces[face].temporal_vertex_ids[0];
                    rec.t1_id = cell.spatial_faces[face].temporal_vertex_ids[1];
                    rec.interval_id_key =
                        mesh::topology::make_time_interval_id_key(
                            rec.t0_id,
                            rec.t1_id);
                    rec.t0 = mesh.temporal_vertices()[static_cast<std::size_t>(rec.t0_id)][0];
                    rec.t1 = mesh.temporal_vertices()[static_cast<std::size_t>(rec.t1_id)][0];
                    rec.boundary = cell.spatial_boundary[static_cast<std::size_t>(face)];

                    groups[mesh::topology::SpatialEdgeKey2D<GeomTraits>{rec.sorted_edge}]
                        .push_back(rec);
                }
            }

            for (auto& [key, group] : groups)
            {
                (void)key;
                sort_spatial_face_group_2d(group);
            }

            return groups;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline SpatialInterface<GeomTraits>
        make_spatial_interface_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const SpatialFaceRecord2D<GeomTraits>& master_rec,
            const SpatialFaceRecord2D<GeomTraits>* slave_rec,
            const bool is_boundary,
            const bool is_hanging)
        {
            SpatialInterface<GeomTraits> iface;
            const auto& master = mesh.cell(master_rec.cell_id);

            iface.master_cell = master_rec.cell_id;
            iface.master_face = master_rec.face_id;
            iface.is_boundary = is_boundary;
            iface.is_hanging = is_hanging;
            iface.master_spatial_vertex_ids =
                master.spatial_faces[master_rec.face_id].spatial_vertex_ids;
            iface.master_temporal_vertex_ids =
                master.spatial_faces[master_rec.face_id].temporal_vertex_ids;

            if (slave_rec != nullptr)
            {
                const auto& slave = mesh.cell(slave_rec->cell_id);
                iface.slave_cell = slave_rec->cell_id;
                iface.slave_face = slave_rec->face_id;
                iface.slave_spatial_vertex_ids =
                    slave.spatial_faces[slave_rec->face_id].spatial_vertex_ids;
                iface.slave_temporal_vertex_ids =
                    slave.spatial_faces[slave_rec->face_id].temporal_vertex_ids;
            }

            return iface;
        }

        template<typename GeomTraits, class PushSpatial>
        inline bool match_spatial_face_pair_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const SpatialFaceRecord2D<GeomTraits>& a,
            const SpatialFaceRecord2D<GeomTraits>& b,
            PushSpatial&& push_spatial,
            SpatialMatchingStats2D* stats = nullptr)
        {
            if (a.cell_id == b.cell_id)
            {
                if (stats != nullptr)
                    ++stats->same_cell_rejected_pairs;
                return false;
            }

            if (stats != nullptr)
                ++stats->interval_candidate_pairs;

            if (same_time_interval_2d(a, b))
            {
                const auto& master_rec = (a.cell_id < b.cell_id) ? a : b;
                const auto& slave_rec  = (a.cell_id < b.cell_id) ? b : a;
                push_spatial(
                    make_spatial_interface_2d<GeomTraits>(
                        mesh,
                        master_rec,
                        &slave_rec,
                        false,
                        false));
                return true;
            }

            const bool a_contains_b = contains_time_interval_2d(a, b);
            const bool b_contains_a = contains_time_interval_2d(b, a);

            if (!a_contains_b && !b_contains_a)
            {
                if (stats != nullptr)
                    ++stats->positive_overlap_tests;

                if (mesh::topology::temporal_intervals_overlap_positive_2d(
                        a.t0,
                        a.t1,
                        b.t0,
                        b.t1))
                {
                    throw std::runtime_error(
                        "compute_adjacency_2d: invalid spatial interface with "
                        "partially overlapping time intervals.");
                }

                if (stats != nullptr)
                    ++stats->disjoint_skipped_pairs;
                return false;
            }

            const auto& master_rec = a_contains_b ? a : b;
            const auto& slave_rec = a_contains_b ? b : a;
            push_spatial(
                make_spatial_interface_2d<GeomTraits>(
                    mesh,
                    master_rec,
                    &slave_rec,
                    false,
                    true));
            return true;
        }

        template<typename GeomTraits, class PushSpatial>
        inline void append_unmatched_spatial_boundary_faces_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<SpatialFaceRecord2D<GeomTraits>>& group,
            const std::vector<char>& matched,
            PushSpatial&& push_spatial)
        {
            for (int i = 0; i < static_cast<int>(group.size()); ++i)
            {
                if (matched[static_cast<std::size_t>(i)])
                    continue;

                const auto& rec = group[static_cast<std::size_t>(i)];
                if (!rec.boundary)
                    continue;

                push_spatial(
                    make_spatial_interface_2d<GeomTraits>(
                        mesh,
                        rec,
                        nullptr,
                        true,
                        false));
            }
        }

        template<typename GeomTraits, class PushSpatial>
        inline void match_spatial_face_group_pairwise_reference_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<SpatialFaceRecord2D<GeomTraits>>& group,
            PushSpatial&& push_spatial,
            SpatialMatchingStats2D* stats = nullptr,
            const bool record_group_stats = true)
        {
            if (stats != nullptr && record_group_stats)
            {
                ++stats->groups;
                stats->records += group.size();
                stats->max_group_records =
                    std::max(stats->max_group_records, group.size());
                stats->old_all_pairs += group.size() * (group.size() - 1U) / 2U;
            }

            std::vector<char> matched(group.size(), 0);

            for (int i = 0; i < static_cast<int>(group.size()); ++i)
            {
                for (int j = i + 1; j < static_cast<int>(group.size()); ++j)
                {
                    if (match_spatial_face_pair_2d<GeomTraits>(
                            mesh,
                            group[static_cast<std::size_t>(i)],
                            group[static_cast<std::size_t>(j)],
                            push_spatial,
                            stats))
                    {
                        matched[static_cast<std::size_t>(i)] = 1;
                        matched[static_cast<std::size_t>(j)] = 1;
                    }
                }
            }

            append_unmatched_spatial_boundary_faces_2d<GeomTraits>(
                mesh,
                group,
                matched,
                push_spatial);
        }

        template<typename GeomTraits>
        [[nodiscard]] inline auto spatial_interface_debug_key_2d(
            const SpatialInterface<GeomTraits>& iface)
        {
            return std::make_tuple(
                iface.master_cell,
                iface.master_face,
                iface.slave_cell,
                iface.slave_face,
                iface.is_boundary,
                iface.is_hanging,
                iface.master_spatial_vertex_ids,
                iface.master_temporal_vertex_ids,
                iface.slave_spatial_vertex_ids,
                iface.slave_temporal_vertex_ids);
        }

        template<typename GeomTraits>
        [[nodiscard]] inline auto spatial_interface_debug_keys_2d(
            std::vector<SpatialInterface<GeomTraits>> interfaces)
        {
            using SpatialFaceVertexIds =
                typename SpatialInterface<GeomTraits>::SpatialFaceVertexIds;
            using TemporalVertexIds =
                typename SpatialInterface<GeomTraits>::TemporalVertexIds;
            using Key = std::tuple<
                int,
                int,
                int,
                int,
                bool,
                bool,
                SpatialFaceVertexIds,
                TemporalVertexIds,
                SpatialFaceVertexIds,
                TemporalVertexIds>;

            std::vector<Key> keys;
            keys.reserve(interfaces.size());
            for (const auto& iface : interfaces)
                keys.push_back(spatial_interface_debug_key_2d(iface));
            std::sort(keys.begin(), keys.end());
            return keys;
        }

        template<typename GeomTraits, class PushSpatial>
        inline void match_spatial_face_group_interval_overlay_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<SpatialFaceRecord2D<GeomTraits>>& group,
            PushSpatial&& push_spatial,
            SpatialMatchingStats2D* stats = nullptr,
            const bool debug_compare_with_pairwise = false)
        {
            if (stats != nullptr)
            {
                ++stats->groups;
                stats->records += group.size();
                stats->max_group_records =
                    std::max(stats->max_group_records, group.size());
                stats->old_all_pairs += group.size() * (group.size() - 1U) / 2U;
            }

            std::vector<SpatialInterface<GeomTraits>> overlay_debug_interfaces;
            const bool capture_overlay =
                debug_compare_with_pairwise && group.size() <= 32U;
            if (capture_overlay)
                overlay_debug_interfaces.reserve(group.size());

            const auto push_overlay =
                [&](const SpatialInterface<GeomTraits>& iface)
                {
                    if (capture_overlay)
                        overlay_debug_interfaces.push_back(iface);
                    push_spatial(iface);
                };

            std::vector<char> matched(group.size(), 0);

            for (int i = 0; i < static_cast<int>(group.size()); ++i)
            {
                const auto& a = group[static_cast<std::size_t>(i)];
                for (int j = i + 1; j < static_cast<int>(group.size()); ++j)
                {
                    const auto& b = group[static_cast<std::size_t>(j)];
                    if (b.t0 >= a.t1)
                    {
                        if (stats != nullptr)
                            ++stats->disjoint_breaks;
                        break;
                    }

                    if (match_spatial_face_pair_2d<GeomTraits>(
                            mesh,
                            a,
                            b,
                            push_overlay,
                            stats))
                    {
                        matched[static_cast<std::size_t>(i)] = 1;
                        matched[static_cast<std::size_t>(j)] = 1;
                    }
                }
            }

            append_unmatched_spatial_boundary_faces_2d<GeomTraits>(
                mesh,
                group,
                matched,
                push_overlay);

            if (capture_overlay)
            {
                if (stats != nullptr)
                    ++stats->debug_reference_groups;

                std::vector<SpatialInterface<GeomTraits>> reference;
                reference.reserve(overlay_debug_interfaces.size());
                SpatialMatchingStats2D reference_stats;
                match_spatial_face_group_pairwise_reference_2d<GeomTraits>(
                    mesh,
                    group,
                    [&](const SpatialInterface<GeomTraits>& iface)
                    {
                        reference.push_back(iface);
                    },
                    &reference_stats);

                if (spatial_interface_debug_keys_2d(
                        std::move(overlay_debug_interfaces)) !=
                    spatial_interface_debug_keys_2d(std::move(reference)))
                {
                    if (stats != nullptr)
                        ++stats->debug_reference_mismatches;
                    throw std::runtime_error(
                        "compute_adjacency_2d: indexed spatial interval "
                        "overlay differs from pairwise reference.");
                }
            }
        }

        template<typename GeomTraits>
        [[nodiscard]] inline std::unordered_map<int, std::vector<TemporalFaceRecord2D<GeomTraits>>>
        make_temporal_face_groups_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<int>& active_cells)
        {
            std::unordered_map<int, std::vector<TemporalFaceRecord2D<GeomTraits>>> groups;
            groups.reserve(active_cells.size() * 2);

            for (const int cell_id : active_cells)
            {
                const auto& cell = mesh.cell(cell_id);
                for (int face = 0; face < 2; ++face)
                {
                    TemporalFaceRecord2D<GeomTraits> rec;
                    rec.cell_id = cell_id;
                    rec.face_id = face;
                    rec.vertices = cell.temporal_faces[face].spatial_vertex_ids;
                    rec.sorted_vertices =
                        mesh::topology::sorted_spatial_cell_vertex_ids_2d<GeomTraits>(
                            rec.vertices);
                    rec.temporal_vertex_id = cell.temporal_faces[face].temporal_vertex_id;
                    rec.boundary = cell.temporal_boundary[static_cast<std::size_t>(face)];

                    groups[rec.temporal_vertex_id].push_back(rec);
                }
            }

            for (auto& [temporal_vertex_id, group] : groups)
            {
                (void)temporal_vertex_id;
                std::sort(
                    group.begin(),
                    group.end(),
                    [](const auto& a, const auto& b)
                    {
                        if (a.sorted_vertices != b.sorted_vertices)
                            return a.sorted_vertices < b.sorted_vertices;
                        if (a.cell_id != b.cell_id)
                            return a.cell_id < b.cell_id;
                        return a.face_id < b.face_id;
                    });
            }

            return groups;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool temporal_face_records_overlap_positive_2d(
            const TemporalFaceRecord2D<GeomTraits>& a,
            const TemporalFaceRecord2D<GeomTraits>& b,
            const std::vector<typename mesh::MeshTypes<GeomTraits>::SpatialPoint>&
                spatial_vertices,
            std::size_t* overlap_tests = nullptr)
        {
            if (a.cell_id == b.cell_id && a.face_id == b.face_id)
                return false;

            if (overlap_tests != nullptr)
                ++(*overlap_tests);

            if (a.sorted_vertices == b.sorted_vertices)
                return true;

            if (triangle_contains_triangle_2d<GeomTraits>(
                    a.vertices,
                    b.vertices,
                    spatial_vertices))
            {
                return true;
            }

            if (triangle_contains_triangle_2d<GeomTraits>(
                    b.vertices,
                    a.vertices,
                    spatial_vertices))
            {
                return true;
            }

            return triangles_overlap_positive_area_2d<GeomTraits>(
                a.vertices,
                b.vertices,
                spatial_vertices);
        }

        template<typename GeomTraits>
        [[nodiscard]] inline SpatialBoundingBox2D<GeomTraits>
        temporal_face_bounding_box_2d(
            const TemporalFaceRecord2D<GeomTraits>& rec,
            const std::vector<typename mesh::MeshTypes<GeomTraits>::SpatialPoint>&
                spatial_vertices)
        {
            const auto& p0 =
                spatial_vertices[static_cast<std::size_t>(rec.vertices[0])];
            SpatialBoundingBox2D<GeomTraits> box{
                p0[0],
                p0[0],
                p0[1],
                p0[1]
            };
            for (int local = 1; local < 3; ++local)
            {
                const auto& p =
                    spatial_vertices[
                        static_cast<std::size_t>(rec.vertices[local])];
                box.min_x = std::min(box.min_x, p[0]);
                box.max_x = std::max(box.max_x, p[0]);
                box.min_y = std::min(box.min_y, p[1]);
                box.max_y = std::max(box.max_y, p[1]);
            }
            return box;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool bounding_boxes_overlap_2d(
            const SpatialBoundingBox2D<GeomTraits>& a,
            const SpatialBoundingBox2D<GeomTraits>& b,
            const double tol = 1.0e-14) noexcept
        {
            return a.min_x <= b.max_x + tol &&
                   b.min_x <= a.max_x + tol &&
                   a.min_y <= b.max_y + tol &&
                   b.min_y <= a.max_y + tol;
        }

        template<typename GeomTraits, class PushTemporal>
        inline bool match_temporal_face_pair_exact_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const TemporalFaceRecord2D<GeomTraits>& a,
            const TemporalFaceRecord2D<GeomTraits>& b,
            PushTemporal&& push_temporal)
        {
            const auto& spatial_vertices = mesh.spatial_vertices();

            if (a.sorted_vertices == b.sorted_vertices)
            {
                const auto& master_rec =
                    (a.cell_id < b.cell_id) ? a : b;
                const auto& slave_rec =
                    (a.cell_id < b.cell_id) ? b : a;

                TemporalInterface<GeomTraits> iface;
                const auto& master = mesh.cell(master_rec.cell_id);
                const auto& slave = mesh.cell(slave_rec.cell_id);

                iface.master_cell = master_rec.cell_id;
                iface.master_face = master_rec.face_id;
                iface.slave_cell = slave_rec.cell_id;
                iface.slave_face = slave_rec.face_id;
                iface.master_spatial_vertex_ids =
                    master.temporal_faces[master_rec.face_id]
                        .spatial_vertex_ids;
                iface.master_temporal_vertex_id =
                    master.temporal_faces[master_rec.face_id]
                        .temporal_vertex_id;
                iface.slave_spatial_vertex_ids =
                    slave.temporal_faces[slave_rec.face_id]
                        .spatial_vertex_ids;
                iface.slave_temporal_vertex_id =
                    slave.temporal_faces[slave_rec.face_id]
                        .temporal_vertex_id;

                push_temporal(iface);
                return true;
            }

            const bool a_contains_b =
                triangle_contains_triangle_2d<GeomTraits>(
                    a.vertices,
                    b.vertices,
                    spatial_vertices);
            const bool b_contains_a =
                triangle_contains_triangle_2d<GeomTraits>(
                    b.vertices,
                    a.vertices,
                    spatial_vertices);

            if (!a_contains_b && !b_contains_a)
            {
                if (triangles_overlap_positive_area_2d<GeomTraits>(
                        a.vertices,
                        b.vertices,
                        spatial_vertices))
                {
                    throw std::runtime_error(
                        "compute_adjacency_2d: invalid temporal interface with "
                        "partially overlapping spatial faces.");
                }
                return false;
            }

            const auto& master_rec = a_contains_b ? a : b;
            const auto& slave_rec = a_contains_b ? b : a;

            TemporalInterface<GeomTraits> iface;
            const auto& master = mesh.cell(master_rec.cell_id);
            const auto& slave = mesh.cell(slave_rec.cell_id);

            iface.master_cell = master_rec.cell_id;
            iface.master_face = master_rec.face_id;
            iface.slave_cell = slave_rec.cell_id;
            iface.slave_face = slave_rec.face_id;
            iface.is_hanging = true;
            iface.master_spatial_vertex_ids =
                master.temporal_faces[master_rec.face_id].spatial_vertex_ids;
            iface.master_temporal_vertex_id =
                master.temporal_faces[master_rec.face_id].temporal_vertex_id;
            iface.slave_spatial_vertex_ids =
                slave.temporal_faces[slave_rec.face_id].spatial_vertex_ids;
            iface.slave_temporal_vertex_id =
                slave.temporal_faces[slave_rec.face_id].temporal_vertex_id;

            push_temporal(iface);
            return true;
        }

        template<typename GeomTraits, class PushTemporal>
        inline void match_temporal_face_group_pairwise_reference_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<TemporalFaceRecord2D<GeomTraits>>& group,
            PushTemporal&& push_temporal,
            TemporalMatchingStats2D* stats = nullptr,
            const bool record_group_stats = true)
        {
            if (stats != nullptr && record_group_stats)
            {
                ++stats->groups;
                stats->records += group.size();
                stats->max_group_records =
                    std::max(stats->max_group_records, group.size());
                stats->old_all_pairs += group.size() * (group.size() - 1U) / 2U;
                ++stats->fallback_pairwise_groups;
            }

            std::vector<char> matched(group.size(), 0);

            for (int i = 0; i < static_cast<int>(group.size()); ++i)
            {
                for (int j = i + 1; j < static_cast<int>(group.size()); ++j)
                {
                    const auto& a = group[static_cast<std::size_t>(i)];
                    const auto& b = group[static_cast<std::size_t>(j)];

                    if (a.cell_id == b.cell_id || a.face_id == b.face_id)
                    {
                        if (stats != nullptr)
                            ++stats->same_face_or_cell_rejected_pairs;
                        continue;
                    }

                    if (stats != nullptr)
                    {
                        ++stats->grid_candidate_pairs;
                        ++stats->exact_overlap_tests;
                    }

                    if (match_temporal_face_pair_exact_2d<GeomTraits>(
                            mesh,
                            a,
                            b,
                            push_temporal))
                    {
                        matched[static_cast<std::size_t>(i)] = 1;
                        matched[static_cast<std::size_t>(j)] = 1;
                    }
                }
            }

            for (int i = 0; i < static_cast<int>(group.size()); ++i)
            {
                if (matched[static_cast<std::size_t>(i)])
                    continue;

                const auto& rec = group[static_cast<std::size_t>(i)];
                if (!rec.boundary)
                    continue;

                TemporalInterface<GeomTraits> iface;
                const auto& cell = mesh.cell(rec.cell_id);

                iface.master_cell = rec.cell_id;
                iface.master_face = rec.face_id;
                iface.is_boundary = true;
                iface.master_spatial_vertex_ids =
                    cell.temporal_faces[rec.face_id].spatial_vertex_ids;
                iface.master_temporal_vertex_id =
                    cell.temporal_faces[rec.face_id].temporal_vertex_id;

                push_temporal(iface);
            }
        }

        template<typename GeomTraits, class PushTemporal>
        inline void match_temporal_face_group_2d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<TemporalFaceRecord2D<GeomTraits>>& group,
            PushTemporal&& push_temporal,
            TemporalMatchingStats2D* stats = nullptr)
        {
            if (stats != nullptr)
            {
                ++stats->groups;
                stats->records += group.size();
                stats->max_group_records =
                    std::max(stats->max_group_records, group.size());
                stats->old_all_pairs += group.size() * (group.size() - 1U) / 2U;
            }

            if (group.size() < 2)
            {
                match_temporal_face_group_pairwise_reference_2d(
                    mesh,
                    group,
                    push_temporal,
                    nullptr);
                return;
            }

            const auto& spatial_vertices = mesh.spatial_vertices();
            std::vector<SpatialBoundingBox2D<GeomTraits>> boxes;
            boxes.reserve(group.size());

            SpatialBoundingBox2D<GeomTraits> domain;
            bool have_domain = false;
            bool valid_boxes = true;
            for (const auto& rec : group)
            {
                auto box = temporal_face_bounding_box_2d<GeomTraits>(
                    rec,
                    spatial_vertices);
                const bool finite =
                    std::isfinite(box.min_x) &&
                    std::isfinite(box.max_x) &&
                    std::isfinite(box.min_y) &&
                    std::isfinite(box.max_y);
                valid_boxes = valid_boxes && finite;
                if (!have_domain)
                {
                    domain = box;
                    have_domain = true;
                }
                else
                {
                    domain.min_x = std::min(domain.min_x, box.min_x);
                    domain.max_x = std::max(domain.max_x, box.max_x);
                    domain.min_y = std::min(domain.min_y, box.min_y);
                    domain.max_y = std::max(domain.max_y, box.max_y);
                }
                boxes.push_back(box);
            }

            const double span_x = domain.max_x - domain.min_x;
            const double span_y = domain.max_y - domain.min_y;
            if (!valid_boxes || span_x <= 0.0 || span_y <= 0.0)
            {
                if (stats != nullptr)
                    ++stats->fallback_degenerate_bbox_groups;
                match_temporal_face_group_pairwise_reference_2d(
                    mesh,
                    group,
                    push_temporal,
                    stats,
                    false);
                return;
            }

            const int grid_dim =
                std::max(
                    1,
                    std::min(
                        512,
                        static_cast<int>(std::ceil(
                            std::sqrt(static_cast<double>(group.size()))))));
            const std::size_t bucket_count =
                static_cast<std::size_t>(grid_dim) *
                static_cast<std::size_t>(grid_dim);
            std::vector<std::vector<int>> buckets(bucket_count);
            if (stats != nullptr)
                stats->grid_buckets += bucket_count;

            const auto clamp_index = [grid_dim](int idx)
            {
                return std::max(0, std::min(grid_dim - 1, idx));
            };
            const auto coordinate_to_index =
                [&](double value, double min_value, double span)
            {
                const double scaled =
                    (value - min_value) * static_cast<double>(grid_dim) / span;
                return clamp_index(static_cast<int>(std::floor(scaled)));
            };
            const auto box_grid_range =
                [&](const SpatialBoundingBox2D<GeomTraits>& box)
            {
                constexpr double tol = 1.0e-14;
                const int ix0 = coordinate_to_index(
                    box.min_x - tol,
                    domain.min_x,
                    span_x);
                const int ix1 = coordinate_to_index(
                    box.max_x + tol,
                    domain.min_x,
                    span_x);
                const int iy0 = coordinate_to_index(
                    box.min_y - tol,
                    domain.min_y,
                    span_y);
                const int iy1 = coordinate_to_index(
                    box.max_y + tol,
                    domain.min_y,
                    span_y);
                return std::array<int, 4>{
                    std::min(ix0, ix1),
                    std::max(ix0, ix1),
                    std::min(iy0, iy1),
                    std::max(iy0, iy1)
                };
            };
            const auto bucket_id = [grid_dim](int ix, int iy)
            {
                return static_cast<std::size_t>(iy) *
                           static_cast<std::size_t>(grid_dim) +
                       static_cast<std::size_t>(ix);
            };

            for (int record_id = 0;
                 record_id < static_cast<int>(group.size());
                 ++record_id)
            {
                const auto range =
                    box_grid_range(boxes[static_cast<std::size_t>(record_id)]);
                for (int iy = range[2]; iy <= range[3]; ++iy)
                {
                    for (int ix = range[0]; ix <= range[1]; ++ix)
                    {
                        buckets[bucket_id(ix, iy)].push_back(record_id);
                        if (stats != nullptr)
                            ++stats->grid_entries;
                    }
                }
            }

            std::vector<char> matched(group.size(), 0);
            std::vector<int> candidate_marker(group.size(), 0);
            int candidate_marker_token = 0;
            std::vector<int> candidate_records;

            for (int i = 0; i < static_cast<int>(group.size()); ++i)
            {
                ++candidate_marker_token;
                if (candidate_marker_token == 0)
                {
                    std::fill(
                        candidate_marker.begin(),
                        candidate_marker.end(),
                        0);
                    candidate_marker_token = 1;
                }
                candidate_records.clear();

                const auto range =
                    box_grid_range(boxes[static_cast<std::size_t>(i)]);
                for (int iy = range[2]; iy <= range[3]; ++iy)
                {
                    for (int ix = range[0]; ix <= range[1]; ++ix)
                    {
                        for (const int candidate_id :
                             buckets[bucket_id(ix, iy)])
                        {
                            if (candidate_id <= i)
                                continue;
                            auto candidate_idx =
                                static_cast<std::size_t>(candidate_id);
                            if (candidate_marker[candidate_idx] ==
                                candidate_marker_token)
                            {
                                continue;
                            }
                            candidate_marker[candidate_idx] =
                                candidate_marker_token;
                            candidate_records.push_back(candidate_id);
                        }
                    }
                }
                std::sort(candidate_records.begin(), candidate_records.end());
                if (stats != nullptr)
                    stats->grid_candidate_pairs += candidate_records.size();

                for (const int j : candidate_records)
                {
                    const auto& a = group[static_cast<std::size_t>(i)];
                    const auto& b = group[static_cast<std::size_t>(j)];

                    if (a.cell_id == b.cell_id || a.face_id == b.face_id)
                    {
                        if (stats != nullptr)
                            ++stats->same_face_or_cell_rejected_pairs;
                        continue;
                    }

                    if (!bounding_boxes_overlap_2d<GeomTraits>(
                            boxes[static_cast<std::size_t>(i)],
                            boxes[static_cast<std::size_t>(j)]))
                    {
                        if (stats != nullptr)
                            ++stats->bbox_rejected_pairs;
                        continue;
                    }

                    if (stats != nullptr)
                        ++stats->exact_overlap_tests;

                    if (match_temporal_face_pair_exact_2d<GeomTraits>(
                            mesh,
                            a,
                            b,
                            push_temporal))
                    {
                        matched[static_cast<std::size_t>(i)] = 1;
                        matched[static_cast<std::size_t>(j)] = 1;
                    }
                }
            }

            for (int i = 0; i < static_cast<int>(group.size()); ++i)
            {
                if (matched[static_cast<std::size_t>(i)])
                    continue;

                const auto& rec = group[static_cast<std::size_t>(i)];
                if (!rec.boundary)
                    continue;

                TemporalInterface<GeomTraits> iface;
                const auto& cell = mesh.cell(rec.cell_id);

                iface.master_cell = rec.cell_id;
                iface.master_face = rec.face_id;
                iface.is_boundary = true;
                iface.master_spatial_vertex_ids =
                    cell.temporal_faces[rec.face_id].spatial_vertex_ids;
                iface.master_temporal_vertex_id =
                    cell.temporal_faces[rec.face_id].temporal_vertex_id;

                push_temporal(iface);
            }
        }

        template<typename GeomTraits, typename Policy>
        inline void initialize_adjacency_cell_maps_2d(
            Adjacency<GeomTraits, Policy>& adjacency,
            const std::vector<int>& active_cells)
        {
            adjacency.cell_to_spatial.clear();
            adjacency.cell_to_temporal.clear();
            adjacency.cell_to_spatial.reserve(active_cells.size());
            adjacency.cell_to_temporal.reserve(active_cells.size());
            for (const int cell_id : active_cells)
            {
                adjacency.cell_to_spatial[cell_id] = {};
                adjacency.cell_to_temporal[cell_id] = {};
            }
        }

        template<typename GeomTraits, typename Policy>
        inline void rebuild_adjacency_cell_maps_2d(
            Adjacency<GeomTraits, Policy>& adjacency,
            const std::vector<int>& active_cells)
        {
            initialize_adjacency_cell_maps_2d(adjacency, active_cells);

            for (int iface_id = 0;
                 iface_id < static_cast<int>(adjacency.spatial_interfaces.size());
                 ++iface_id)
            {
                const auto& iface =
                    adjacency.spatial_interfaces[static_cast<std::size_t>(iface_id)];
                adjacency.cell_to_spatial[iface.master_cell][iface.master_face]
                    .push_back(iface_id);
                if (!iface.is_boundary && iface.slave_cell >= 0)
                {
                    adjacency.cell_to_spatial[iface.slave_cell][iface.slave_face]
                        .push_back(iface_id);
                }
            }

            for (int iface_id = 0;
                 iface_id < static_cast<int>(adjacency.temporal_interfaces.size());
                 ++iface_id)
            {
                const auto& iface =
                    adjacency.temporal_interfaces[static_cast<std::size_t>(iface_id)];
                adjacency.cell_to_temporal[iface.master_cell][iface.master_face]
                    .push_back(iface_id);
                if (!iface.is_boundary && iface.slave_cell >= 0)
                {
                    adjacency.cell_to_temporal[iface.slave_cell][iface.slave_face]
                        .push_back(iface_id);
                }
            }
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool interface_has_cell_2d(
            const InterfaceBase<GeomTraits>& iface,
            const std::vector<char>& marker)
        {
            if (iface.master_cell >= 0 &&
                static_cast<std::size_t>(iface.master_cell) < marker.size() &&
                marker[static_cast<std::size_t>(iface.master_cell)])
            {
                return true;
            }
            return iface.slave_cell >= 0 &&
                   static_cast<std::size_t>(iface.slave_cell) < marker.size() &&
                   marker[static_cast<std::size_t>(iface.slave_cell)];
        }

        template<typename GeomTraits>
        inline void mark_interface_active_cells_2d(
            const InterfaceBase<GeomTraits>& iface,
            const std::vector<char>& active_marker,
            std::vector<char>& marker)
        {
            if (iface.master_cell >= 0 &&
                static_cast<std::size_t>(iface.master_cell) < active_marker.size() &&
                active_marker[static_cast<std::size_t>(iface.master_cell)])
            {
                marker[static_cast<std::size_t>(iface.master_cell)] = 1;
            }
            if (iface.slave_cell >= 0 &&
                static_cast<std::size_t>(iface.slave_cell) < active_marker.size() &&
                active_marker[static_cast<std::size_t>(iface.slave_cell)])
            {
                marker[static_cast<std::size_t>(iface.slave_cell)] = 1;
            }
        }
    }

    template<typename GeomTraits, typename Policy>
    inline void build_spatial_lateral_adjacency_2d(
        Adjacency<GeomTraits, Policy>& adjacency,
        const std::vector<int>& active_cells,
        const mesh::Mesh<GeomTraits>& mesh,
        const typename Adjacency<GeomTraits, Policy>::TimingCallback&
            timing_callback = {})
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "compute_adjacency_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "compute_adjacency_2d requires dim_time_v == 1.");

        using Clock = std::chrono::steady_clock;
        const auto seconds_since = [](const Clock::time_point& begin)
        {
            return std::chrono::duration<double>(
                       Clock::now() - begin)
                .count();
        };
        const auto record = [&](std::string_view phase, double seconds)
        {
            if (timing_callback)
                timing_callback(phase, seconds);
        };

        const auto push_spatial = [&](const SpatialInterface<GeomTraits>& iface)
        {
            adjacency.spatial_interfaces.push_back(iface);
        };

        {
            using SpatialGroups = decltype(
                detail::adjacency_impl::make_spatial_face_groups_2d(
                    mesh,
                    active_cells));
            SpatialGroups groups;
            {
                const auto start = Clock::now();
                groups =
                    detail::adjacency_impl::make_spatial_face_groups_2d(
                        mesh,
                        active_cells);
                std::size_t spatial_records = 0;
                for (const auto& [edge_key, group] : groups)
                {
                    (void)edge_key;
                    spatial_records += group.size();
                }
                record(
                    "fespace.adjacency_compute.spatial_records.count",
                    static_cast<double>(spatial_records));
                record(
                    "adjacency.spatial_face_records.count",
                    static_cast<double>(spatial_records));
                record(
                    "fespace.adjacency_compute.spatial_grouping",
                    seconds_since(start));
            }

            const auto match_start = Clock::now();
            detail::adjacency_impl::SpatialMatchingStats2D spatial_stats;
#ifndef NDEBUG
            constexpr bool debug_compare_spatial_matching = true;
#else
            constexpr bool debug_compare_spatial_matching = false;
#endif
            for (const auto& [edge_key, group] : groups)
            {
                (void)edge_key;
                detail::adjacency_impl::match_spatial_face_group_interval_overlay_2d(
                    mesh,
                    group,
                    push_spatial,
                    &spatial_stats,
                    debug_compare_spatial_matching);
            }
            const double spatial_match_seconds = seconds_since(match_start);
            record(
                "fespace.adjacency_compute.spatial_interface_matching",
                spatial_match_seconds);
            record(
                "fespace.adjacency_compute.interface_matching",
                spatial_match_seconds);
            record(
                "fespace.adjacency_compute.spatial_groups.count",
                static_cast<double>(spatial_stats.groups));
            record(
                "adjacency.spatial_edge_groups.count",
                static_cast<double>(spatial_stats.groups));
            record(
                "adjacency.spatial_interfaces.count",
                static_cast<double>(adjacency.spatial_interfaces.size()));
            record(
                "fespace.adjacency_compute.spatial_max_group_records.count",
                static_cast<double>(spatial_stats.max_group_records));
            record(
                "fespace.adjacency_compute.spatial_old_all_pairs.count",
                static_cast<double>(spatial_stats.old_all_pairs));
            record(
                "fespace.adjacency_compute.spatial_interval_candidate_pairs.count",
                static_cast<double>(spatial_stats.interval_candidate_pairs));
            record(
                "fespace.adjacency_compute.spatial_positive_overlap_tests.count",
                static_cast<double>(spatial_stats.positive_overlap_tests));
            record(
                "fespace.adjacency_compute.spatial_disjoint_breaks.count",
                static_cast<double>(spatial_stats.disjoint_breaks));
            record(
                "fespace.adjacency_compute.spatial_disjoint_skipped_pairs.count",
                static_cast<double>(spatial_stats.disjoint_skipped_pairs));
            record(
                "fespace.adjacency_compute.spatial_same_cell_rejected_pairs.count",
                static_cast<double>(spatial_stats.same_cell_rejected_pairs));
            record(
                "fespace.adjacency_compute.spatial_debug_reference_groups.count",
                static_cast<double>(spatial_stats.debug_reference_groups));
            record(
                "fespace.adjacency_compute.spatial_debug_reference_mismatches.count",
                static_cast<double>(spatial_stats.debug_reference_mismatches));
        }
    }

    template<typename GeomTraits, typename Policy>
    inline void build_temporal_horizontal_adjacency_2d(
        Adjacency<GeomTraits, Policy>& adjacency,
        const std::vector<int>& active_cells,
        const mesh::Mesh<GeomTraits>& mesh,
        const typename Adjacency<GeomTraits, Policy>::TimingCallback&
            timing_callback = {})
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "build_temporal_horizontal_adjacency_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "build_temporal_horizontal_adjacency_2d requires dim_time_v == 1.");

        using Clock = std::chrono::steady_clock;
        const auto seconds_since = [](const Clock::time_point& begin)
        {
            return std::chrono::duration<double>(
                       Clock::now() - begin)
                .count();
        };
        const auto record = [&](std::string_view phase, double seconds)
        {
            if (timing_callback)
                timing_callback(phase, seconds);
        };

        record(
            "fespace.adjacency_compute.temporal_adjacency_skipped_by_policy.count",
            0.0);
        record(
            "adjacency.temporal_adjacency_skipped_by_policy.count",
            0.0);

        const auto push_temporal = [&](const TemporalInterface<GeomTraits>& iface)
        {
            adjacency.temporal_interfaces.push_back(iface);
        };

        using TemporalGroups = decltype(
            detail::adjacency_impl::make_temporal_face_groups_2d(
                mesh,
                active_cells));
        TemporalGroups groups;
        {
            const auto start = Clock::now();
            groups =
                detail::adjacency_impl::make_temporal_face_groups_2d(
                    mesh,
                    active_cells);
            record(
                "fespace.adjacency_compute.temporal_grouping",
                seconds_since(start));
        }
        const auto match_start = Clock::now();
        detail::adjacency_impl::TemporalMatchingStats2D temporal_stats;
        for (const auto& [temporal_vertex_id, group] : groups)
        {
            (void)temporal_vertex_id;
            detail::adjacency_impl::match_temporal_face_group_2d(
                mesh,
                group,
                push_temporal,
                &temporal_stats);
        }
        const double temporal_match_seconds = seconds_since(match_start);
        record(
            "fespace.adjacency_compute.temporal_interface_matching",
            temporal_match_seconds);
        record(
            "fespace.adjacency_compute.interface_matching",
            temporal_match_seconds);
        record(
            "fespace.adjacency_compute.temporal_face_overlap_tests.count",
            static_cast<double>(temporal_stats.exact_overlap_tests));
        record(
            "fespace.adjacency_compute.temporal_groups.count",
            static_cast<double>(temporal_stats.groups));
        record(
            "fespace.adjacency_compute.temporal_records.count",
            static_cast<double>(temporal_stats.records));
        record(
            "adjacency.temporal_face_records.count",
            static_cast<double>(temporal_stats.records));
        record(
            "adjacency.temporal_interfaces.count",
            static_cast<double>(adjacency.temporal_interfaces.size()));
        record(
            "fespace.adjacency_compute.temporal_max_group_records.count",
            static_cast<double>(temporal_stats.max_group_records));
        record(
            "fespace.adjacency_compute.temporal_old_all_pairs.count",
            static_cast<double>(temporal_stats.old_all_pairs));
        record(
            "fespace.adjacency_compute.temporal_grid_candidate_pairs.count",
            static_cast<double>(temporal_stats.grid_candidate_pairs));
        record(
            "fespace.adjacency_compute.temporal_bbox_rejected_pairs.count",
            static_cast<double>(temporal_stats.bbox_rejected_pairs));
        record(
            "fespace.adjacency_compute.temporal_same_face_or_cell_rejected_pairs.count",
            static_cast<double>(
                temporal_stats.same_face_or_cell_rejected_pairs));
        record(
            "fespace.adjacency_compute.temporal_grid_buckets.count",
            static_cast<double>(temporal_stats.grid_buckets));
        record(
            "fespace.adjacency_compute.temporal_grid_entries.count",
            static_cast<double>(temporal_stats.grid_entries));
        record(
            "fespace.adjacency_compute.temporal_index_fallback_pairwise_groups.count",
            static_cast<double>(temporal_stats.fallback_pairwise_groups));
        record(
            "fespace.adjacency_compute.temporal_index_fallback_degenerate_bbox_groups.count",
            static_cast<double>(
                temporal_stats.fallback_degenerate_bbox_groups));
    }

    template<typename GeomTraits, typename Policy>
    inline void record_temporal_adjacency_skipped_by_policy_2d(
        const typename Adjacency<GeomTraits, Policy>::TimingCallback&
            timing_callback)
    {
        const auto record = [&](std::string_view phase, double value)
        {
            if (timing_callback)
                timing_callback(phase, value);
        };

        record(
            "fespace.adjacency_compute.temporal_adjacency_skipped_by_policy.count",
            1.0);
        record(
            "adjacency.temporal_adjacency_skipped_by_policy.count",
            1.0);
        record(
            "fespace.adjacency_compute.temporal_records.count",
            0.0);
        record(
            "adjacency.temporal_face_records.count",
            0.0);
        record(
            "fespace.adjacency_compute.temporal_groups.count",
            0.0);
        record(
            "adjacency.temporal_interfaces.count",
            0.0);
    }

    template<typename GeomTraits, typename Policy>
    inline void compute_adjacency_2d(
        Adjacency<GeomTraits, Policy>& adjacency,
        const std::vector<int>& active_cells,
        const mesh::Mesh<GeomTraits>& mesh,
        const typename Adjacency<GeomTraits, Policy>::TimingCallback&
            timing_callback = {})
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "compute_adjacency_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "compute_adjacency_2d requires dim_time_v == 1.");

        adjacency.clear();
        adjacency.spatial_interfaces.reserve(active_cells.size() * 3);
        if constexpr (Policy::continuous_in_time)
            adjacency.temporal_interfaces.reserve(active_cells.size() * 2);

        if constexpr (Policy::continuous_in_space)
        {
            build_spatial_lateral_adjacency_2d<GeomTraits, Policy>(
                adjacency,
                active_cells,
                mesh,
                timing_callback);
        }

        if constexpr (Policy::continuous_in_time)
        {
            build_temporal_horizontal_adjacency_2d<GeomTraits, Policy>(
                adjacency,
                active_cells,
                mesh,
                timing_callback);
            detail::adjacency_impl::rebuild_adjacency_cell_maps_2d(
                adjacency,
                active_cells);
        }
        else
        {
            record_temporal_adjacency_skipped_by_policy_2d<GeomTraits, Policy>(
                timing_callback);
            detail::adjacency_impl::rebuild_adjacency_cell_maps_2d(
                adjacency,
                active_cells);
            adjacency.cell_to_temporal.clear();
        }
    }

    template<typename GeomTraits, typename Policy>
    inline void compute_adjacency_2d_incremental(
        Adjacency<GeomTraits, Policy>& adjacency,
        const std::vector<int>& active_cells,
        const mesh::Mesh<GeomTraits>& mesh,
        const std::vector<int>& changed_cells,
        const typename Adjacency<GeomTraits, Policy>::TimingCallback&
            timing_callback = {},
        double fallback_fraction = 0.75)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "compute_adjacency_2d_incremental requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "compute_adjacency_2d_incremental requires dim_time_v == 1.");

        using Clock = std::chrono::steady_clock;
        const auto seconds_since = [](const Clock::time_point& begin)
        {
            return std::chrono::duration<double>(
                       Clock::now() - begin)
                .count();
        };
        const auto record = [&](std::string_view phase, double value)
        {
            if (timing_callback)
                timing_callback(phase, value);
        };

        const auto start = Clock::now();
        const std::size_t n_cells = static_cast<std::size_t>(mesh.n_cells());
        std::vector<char> active_marker(n_cells, 0);
        for (const int cell_id : active_cells)
        {
            if (cell_id >= 0 && static_cast<std::size_t>(cell_id) < n_cells)
                active_marker[static_cast<std::size_t>(cell_id)] = 1;
        }

        if (changed_cells.empty() ||
            (adjacency.spatial_interfaces.empty() &&
             adjacency.temporal_interfaces.empty()))
        {
            record("fespace.rebuild.incremental.fallback_full_rebuild.count", 1.0);
            compute_adjacency_2d(adjacency, active_cells, mesh, timing_callback);
            return;
        }

        std::vector<char> changed_marker(n_cells, 0);
        for (const int cell_id : changed_cells)
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= n_cells)
                continue;
            changed_marker[static_cast<std::size_t>(cell_id)] = 1;
        }

        Adjacency<GeomTraits, Policy> merged;
        merged.spatial_interfaces.reserve(active_cells.size() * 3);
        merged.temporal_interfaces.reserve(active_cells.size() * 2);

        const auto push_temporal = [&](const TemporalInterface<GeomTraits>& iface)
        {
            merged.temporal_interfaces.push_back(iface);
        };

        build_spatial_lateral_adjacency_2d<GeomTraits, Policy>(
            merged,
            active_cells,
            mesh,
            timing_callback);

        if constexpr (!Policy::continuous_in_time)
        {
            static_cast<void>(push_temporal);
            record_temporal_adjacency_skipped_by_policy_2d<GeomTraits, Policy>(
                timing_callback);
            detail::adjacency_impl::rebuild_adjacency_cell_maps_2d(
                merged,
                active_cells);
            merged.cell_to_temporal.clear();
            adjacency = std::move(merged);

            record("fespace.rebuild.incremental.fallback_full_rebuild.count", 0.0);
            record("fespace.rebuild.incremental.rebuild_active_cells.count",
                   static_cast<double>(changed_cells.size()));
            record("fespace.rebuild.incremental.adjacency_reuse",
                   seconds_since(start));
            return;
        }

        using TemporalRecord =
            detail::adjacency_impl::TemporalFaceRecord2D<GeomTraits>;

        std::vector<TemporalRecord> active_records;
        active_records.reserve(active_cells.size() * 2);
        std::vector<std::array<int, 2>> active_face_to_record(
            n_cells,
            std::array<int, 2>{{-1, -1}});
        std::unordered_map<int, std::vector<int>> records_by_temporal_vertex;
        records_by_temporal_vertex.reserve(active_cells.size());
        std::unordered_map<std::uint64_t, std::vector<int>>
            records_by_temporal_spatial_vertex;
        records_by_temporal_spatial_vertex.reserve(active_cells.size() * 8);
        const auto temporal_spatial_vertex_key =
            [](int temporal_vertex_id, int spatial_vertex_id) -> std::uint64_t
        {
            return (static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(temporal_vertex_id))
                    << 32U) ^
                   static_cast<std::uint32_t>(spatial_vertex_id);
        };
        const auto append_unique_vertex =
            [](std::vector<int>& vertices, int vertex_id)
        {
            if (std::find(vertices.begin(), vertices.end(), vertex_id) ==
                vertices.end())
            {
                vertices.push_back(vertex_id);
            }
        };
        const auto collect_spatial_lineage_vertices =
            [&](int cell_id)
        {
            std::vector<int> vertices;
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= n_cells)
                return vertices;

            int current = cell_id;
            while (current >= 0 &&
                   static_cast<std::size_t>(current) < n_cells)
            {
                const auto& cell = mesh.cell(current);
                for (const int vertex_id : cell.spatial_vertex_ids)
                    append_unique_vertex(vertices, vertex_id);
                current = cell.parent_id;
            }
            return vertices;
        };

        {
            const auto phase_start = Clock::now();
            for (const int cell_id : active_cells)
            {
                if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= n_cells)
                    continue;
                const auto& cell = mesh.cell(cell_id);
                for (int face = 0; face < 2; ++face)
                {
                    TemporalRecord rec;
                    rec.cell_id = cell_id;
                    rec.face_id = face;
                    rec.vertices =
                        cell.temporal_faces[static_cast<std::size_t>(face)]
                            .spatial_vertex_ids;
                    rec.sorted_vertices =
                        mesh::topology::sorted_spatial_cell_vertex_ids_2d<
                            GeomTraits>(rec.vertices);
                    rec.temporal_vertex_id =
                        cell.temporal_faces[static_cast<std::size_t>(face)]
                            .temporal_vertex_id;
                    rec.boundary =
                        cell.temporal_boundary[static_cast<std::size_t>(face)];

                    const int record_id =
                        static_cast<int>(active_records.size());
                    active_face_to_record[static_cast<std::size_t>(cell_id)]
                                         [static_cast<std::size_t>(face)] =
                        record_id;
                    records_by_temporal_vertex[rec.temporal_vertex_id]
                        .push_back(record_id);
                    for (const int vertex_id :
                         collect_spatial_lineage_vertices(cell_id))
                    {
                        records_by_temporal_spatial_vertex[
                            temporal_spatial_vertex_key(
                                rec.temporal_vertex_id,
                                vertex_id)]
                            .push_back(record_id);
                    }
                    active_records.push_back(rec);
                }
            }
            record(
                "fespace.adjacency_compute.temporal_index_build",
                seconds_since(phase_start));
            record(
                "fespace.adjacency_compute.incremental.temporal_faces_scanned.count",
                static_cast<double>(active_records.size()));
        }

        std::vector<std::array<char, 2>> affected_face(
            n_cells,
            std::array<char, 2>{{0, 0}});
        std::vector<char> seed_record_marker(active_records.size(), 0);
        std::vector<int> seed_records;
        seed_records.reserve(changed_cells.size() * 2);

        const auto mark_seed_face = [&](int cell_id, int face_id)
        {
            if (cell_id < 0 ||
                face_id < 0 ||
                face_id >= 2 ||
                static_cast<std::size_t>(cell_id) >= n_cells ||
                !active_marker[static_cast<std::size_t>(cell_id)])
            {
                return;
            }

            const int record_id =
                active_face_to_record[static_cast<std::size_t>(cell_id)]
                                     [static_cast<std::size_t>(face_id)];
            if (record_id < 0)
                return;

            const auto record_idx = static_cast<std::size_t>(record_id);
            if (!seed_record_marker[record_idx])
            {
                seed_record_marker[record_idx] = 1;
                seed_records.push_back(record_id);
            }
        };

        for (const int cell_id : changed_cells)
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= n_cells)
                continue;
            if (!active_marker[static_cast<std::size_t>(cell_id)])
                continue;
            mark_seed_face(cell_id, 0);
            mark_seed_face(cell_id, 1);
        }

        for (const auto& iface : adjacency.temporal_interfaces)
        {
            if (!detail::adjacency_impl::interface_has_cell_2d(
                    iface,
                    changed_marker))
            {
                continue;
            }

            mark_seed_face(iface.master_cell, iface.master_face);
            if (!iface.is_boundary && iface.slave_cell >= 0)
                mark_seed_face(iface.slave_cell, iface.slave_face);
        }

        if (seed_records.empty())
        {
            record("fespace.rebuild.incremental.fallback_full_rebuild.count", 1.0);
            record(
                "fespace.adjacency_compute.incremental.fallback_reason.empty_seed.count",
                1.0);
            compute_adjacency_2d(adjacency, active_cells, mesh, timing_callback);
            return;
        }

        const double seed_fraction =
            active_records.empty()
                ? 0.0
                : static_cast<double>(seed_records.size()) /
                      static_cast<double>(active_records.size());
        record(
            "fespace.adjacency_compute.incremental.temporal_seed_faces.count",
            static_cast<double>(seed_records.size()));
        record(
            "fespace.adjacency_compute.incremental.temporal_seed_fraction",
            seed_fraction);
        if (seed_fraction > 0.10)
        {
            record("fespace.rebuild.incremental.fallback_full_rebuild.count", 1.0);
            record(
                "fespace.adjacency_compute.incremental.fallback_reason.too_many_temporal_seed_faces.count",
                1.0);
            compute_adjacency_2d(adjacency, active_cells, mesh, timing_callback);
            return;
        }

        std::vector<char> included_record(active_records.size(), 0);
        std::vector<int> queue;
        queue.reserve(seed_records.size());
        std::unordered_set<int> affected_temporal_vertices;
        affected_temporal_vertices.reserve(seed_records.size());
        std::vector<std::vector<TemporalRecord>> local_temporal_components;
        local_temporal_components.reserve(seed_records.size());
        std::size_t temporal_candidate_overlap_tests = 0;
        std::size_t temporal_candidate_record_visits = 0;
        std::vector<int> candidate_marker(active_records.size(), 0);
        int candidate_marker_token = 0;
        std::vector<int> candidate_records;

        const auto include_record =
            [&](int record_id, std::vector<int>& component_record_ids)
        {
            if (record_id < 0 ||
                static_cast<std::size_t>(record_id) >= active_records.size())
            {
                return false;
            }
            const auto idx = static_cast<std::size_t>(record_id);
            if (included_record[idx])
                return false;

            included_record[idx] = 1;
            queue.push_back(record_id);
            component_record_ids.push_back(record_id);

            const auto& rec = active_records[idx];
            affected_face[static_cast<std::size_t>(rec.cell_id)]
                         [static_cast<std::size_t>(rec.face_id)] = 1;
            affected_temporal_vertices.insert(rec.temporal_vertex_id);
            return true;
        };

        const auto& spatial_vertices = mesh.spatial_vertices();
        for (const int seed_record_id : seed_records)
        {
            if (seed_record_id < 0 ||
                static_cast<std::size_t>(seed_record_id) >=
                    included_record.size() ||
                included_record[static_cast<std::size_t>(seed_record_id)])
            {
                continue;
            }

            queue.clear();
            std::vector<int> component_record_ids;
            component_record_ids.reserve(16);
            (void)include_record(seed_record_id, component_record_ids);

            for (std::size_t queue_pos = 0; queue_pos < queue.size(); ++queue_pos)
            {
                const auto& seed =
                    active_records[static_cast<std::size_t>(queue[queue_pos])];

                ++candidate_marker_token;
                if (candidate_marker_token == 0)
                {
                    std::fill(
                        candidate_marker.begin(),
                        candidate_marker.end(),
                        0);
                    candidate_marker_token = 1;
                }
                candidate_records.clear();

                for (const int vertex_id :
                     collect_spatial_lineage_vertices(seed.cell_id))
                {
                    const auto it = records_by_temporal_spatial_vertex.find(
                        temporal_spatial_vertex_key(
                            seed.temporal_vertex_id,
                            vertex_id));
                    if (it == records_by_temporal_spatial_vertex.end())
                        continue;

                    for (const int candidate_id : it->second)
                    {
                        if (candidate_id < 0 ||
                            static_cast<std::size_t>(candidate_id) >=
                                candidate_marker.size())
                        {
                            continue;
                        }
                        if (candidate_marker[
                                static_cast<std::size_t>(candidate_id)] ==
                            candidate_marker_token)
                        {
                            continue;
                        }

                        candidate_marker[static_cast<std::size_t>(candidate_id)] =
                            candidate_marker_token;
                        candidate_records.push_back(candidate_id);
                    }
                }

                temporal_candidate_record_visits += candidate_records.size();
                for (const int candidate_id : candidate_records)
                {
                    if (candidate_id == queue[queue_pos])
                        continue;
                    if (included_record[static_cast<std::size_t>(candidate_id)])
                        continue;

                    const auto& candidate =
                        active_records[static_cast<std::size_t>(candidate_id)];
                    if (detail::adjacency_impl::
                            temporal_face_records_overlap_positive_2d<
                                GeomTraits>(
                                seed,
                                candidate,
                                spatial_vertices,
                                &temporal_candidate_overlap_tests))
                    {
                        (void)include_record(
                            candidate_id,
                            component_record_ids);
                    }
                }
            }

            std::vector<TemporalRecord> component;
            component.reserve(component_record_ids.size());
            for (const int record_id : component_record_ids)
                component.push_back(
                    active_records[static_cast<std::size_t>(record_id)]);
            local_temporal_components.push_back(std::move(component));
        }

        std::size_t included_record_count = 0;
        std::unordered_set<int> touched_cells;
        touched_cells.reserve(queue.size());
        for (std::size_t record_id = 0; record_id < included_record.size(); ++record_id)
        {
            if (!included_record[record_id])
                continue;
            ++included_record_count;
            touched_cells.insert(active_records[record_id].cell_id);
        }

        const double local_fraction =
            active_records.empty()
                ? 0.0
                : static_cast<double>(included_record_count) /
                      static_cast<double>(active_records.size());
        record("fespace.rebuild.incremental.active_cells_total.count",
               static_cast<double>(active_cells.size()));
        record("fespace.rebuild.incremental.active_cells_touched.count",
               static_cast<double>(touched_cells.size()));
        record("fespace.rebuild.incremental.local_fraction",
               local_fraction);

        if (local_fraction > fallback_fraction)
        {
            record("fespace.rebuild.incremental.fallback_full_rebuild.count", 1.0);
            record(
                "fespace.adjacency_compute.incremental.fallback_reason.too_many_temporal_faces.count",
                1.0);
            compute_adjacency_2d(adjacency, active_cells, mesh, timing_callback);
            return;
        }

        std::size_t reused_temporal_interfaces = 0;
        for (const auto& iface : adjacency.temporal_interfaces)
        {
            const bool active_master =
                iface.master_cell >= 0 &&
                static_cast<std::size_t>(iface.master_cell) < active_marker.size() &&
                active_marker[static_cast<std::size_t>(iface.master_cell)];
            const bool active_slave =
                iface.is_boundary ||
                (iface.slave_cell >= 0 &&
                 static_cast<std::size_t>(iface.slave_cell) < active_marker.size() &&
                 active_marker[static_cast<std::size_t>(iface.slave_cell)]);
            if (!active_master || !active_slave)
                continue;

            const bool affected_master =
                iface.master_cell >= 0 &&
                static_cast<std::size_t>(iface.master_cell) < affected_face.size() &&
                affected_face[static_cast<std::size_t>(iface.master_cell)]
                             [static_cast<std::size_t>(iface.master_face)];
            const bool affected_slave =
                !iface.is_boundary &&
                iface.slave_cell >= 0 &&
                static_cast<std::size_t>(iface.slave_cell) < affected_face.size() &&
                affected_face[static_cast<std::size_t>(iface.slave_cell)]
                             [static_cast<std::size_t>(iface.slave_face)];
            if (affected_master || affected_slave)
                continue;

            push_temporal(iface);
            ++reused_temporal_interfaces;
        }

        detail::adjacency_impl::TemporalMatchingStats2D temporal_match_stats;
        const auto temporal_match_start = Clock::now();
        for (auto& group : local_temporal_components)
        {
            std::sort(
                group.begin(),
                group.end(),
                [](const auto& a, const auto& b)
                {
                    if (a.sorted_vertices != b.sorted_vertices)
                        return a.sorted_vertices < b.sorted_vertices;
                    if (a.cell_id != b.cell_id)
                        return a.cell_id < b.cell_id;
                    return a.face_id < b.face_id;
                });
            detail::adjacency_impl::match_temporal_face_group_2d(
                mesh,
                group,
                push_temporal,
                &temporal_match_stats);
        }
        const double temporal_match_seconds = seconds_since(temporal_match_start);
        record(
            "fespace.adjacency_compute.temporal_interface_matching",
            temporal_match_seconds);
        record(
            "fespace.adjacency_compute.interface_matching",
            temporal_match_seconds);

        detail::adjacency_impl::rebuild_adjacency_cell_maps_2d(
            merged,
            active_cells);
        adjacency = std::move(merged);

        record("fespace.rebuild.incremental.adjacency_interfaces_reused.count",
               static_cast<double>(reused_temporal_interfaces));
        record("fespace.rebuild.incremental.adjacency_interfaces_rebuilt.count",
               static_cast<double>(
                   adjacency.spatial_interfaces.size() +
                   adjacency.temporal_interfaces.size() -
                   reused_temporal_interfaces));
        record("fespace.rebuild.incremental.fallback_full_rebuild.count", 0.0);
        record("fespace.rebuild.incremental.rebuild_active_cells.count",
               static_cast<double>(touched_cells.size()));
        record("fespace.rebuild.incremental.adjacency_reuse",
               seconds_since(start));
        record(
            "fespace.adjacency_compute.incremental.affected_temporal_vertices.count",
            static_cast<double>(affected_temporal_vertices.size()));
        record(
            "fespace.adjacency_compute.incremental.affected_temporal_groups.count",
            static_cast<double>(local_temporal_components.size()));
        record(
            "fespace.adjacency_compute.incremental.temporal_groups_reused.count",
            static_cast<double>(
                records_by_temporal_vertex.size() > affected_temporal_vertices.size()
                    ? records_by_temporal_vertex.size() -
                          affected_temporal_vertices.size()
                    : 0));
        record(
            "fespace.adjacency_compute.incremental.temporal_groups_rebuilt.count",
            static_cast<double>(local_temporal_components.size()));
        record(
            "fespace.adjacency_compute.incremental.temporal_face_overlap_tests.count",
            static_cast<double>(
                temporal_candidate_overlap_tests +
                temporal_match_stats.exact_overlap_tests));
        record(
            "fespace.adjacency_compute.incremental.temporal_candidate_overlap_tests.count",
            static_cast<double>(temporal_candidate_overlap_tests));
        record(
            "fespace.adjacency_compute.incremental.temporal_candidate_record_visits.count",
            static_cast<double>(temporal_candidate_record_visits));
        record(
            "fespace.adjacency_compute.incremental.temporal_matching_overlap_tests.count",
            static_cast<double>(temporal_match_stats.exact_overlap_tests));
        record(
            "fespace.adjacency_compute.incremental.temporal_matching_old_all_pairs.count",
            static_cast<double>(temporal_match_stats.old_all_pairs));
        record(
            "fespace.adjacency_compute.incremental.temporal_matching_grid_candidate_pairs.count",
            static_cast<double>(temporal_match_stats.grid_candidate_pairs));
        record(
            "fespace.adjacency_compute.incremental.temporal_matching_bbox_rejected_pairs.count",
            static_cast<double>(temporal_match_stats.bbox_rejected_pairs));
    }
}

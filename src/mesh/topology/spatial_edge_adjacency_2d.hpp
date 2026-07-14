#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../core/hash.hpp"
#include "../cell.hpp"
#include "../types.hpp"
#include "boundary_2d.hpp"
#include "temporal_keys.hpp"

namespace mesh
{
    template<typename GeomTraits>
    class Mesh;
}

namespace mesh::topology
{
    template<typename GeomTraits>
    struct SpatialEdgeKey2D
    {
        using Edge = typename MeshTypes<GeomTraits>::SpatialFaceVertexIds;

        Edge vertex_ids{};

        bool operator==(const SpatialEdgeKey2D&) const noexcept = default;
    };

    template<typename GeomTraits>
    struct SpatialEdgeKey2DHash
    {
        [[nodiscard]] std::size_t operator()(const SpatialEdgeKey2D<GeomTraits>& key) const noexcept
        {
            std::size_t seed = 0;
            for (const auto vid : key.vertex_ids)
                core::hash_combine(seed, vid);
            return seed;
        }
    };

    template<typename GeomTraits>
    struct SpatialCellKey2D
    {
        using CellVertices = typename MeshTypes<GeomTraits>::SpatialVertexIds;

        CellVertices vertex_ids{};

        bool operator==(const SpatialCellKey2D&) const noexcept = default;
    };

    template<typename GeomTraits>
    struct SpatialCellKey2DHash
    {
        [[nodiscard]] std::size_t operator()(const SpatialCellKey2D<GeomTraits>& key) const noexcept
        {
            std::size_t seed = 0;
            for (const auto vid : key.vertex_ids)
                core::hash_combine(seed, vid);
            return seed;
        }
    };

    template<typename GeomTraits>
    struct ActiveSpatialEdgeRecord2D
    {
        using Edge = typename MeshTypes<GeomTraits>::SpatialFaceVertexIds;

        int edge_id = -1;
        int cell_id = -1;
        int face_id = -1;

        Edge vertex_ids{};
        Edge sorted_vertex_ids{};

        bool boundary_hint = false;

        std::vector<int> exact_match_edge_ids{};
        std::vector<int> contained_edge_ids{};
        std::vector<int> containing_edge_ids{};

        [[nodiscard]] bool has_exact_neighbor() const noexcept
        {
            return !exact_match_edge_ids.empty();
        }

        [[nodiscard]] bool is_nonconforming() const noexcept
        {
            return !contained_edge_ids.empty() || !containing_edge_ids.empty();
        }

        [[nodiscard]] bool is_boundary() const noexcept
        {
            return boundary_hint && !has_exact_neighbor() && !is_nonconforming();
        }

        [[nodiscard]] bool is_conforming_interior() const noexcept
        {
            return exact_match_edge_ids.size() == 1 && !is_nonconforming();
        }

        [[nodiscard]] bool is_singular() const noexcept
        {
            return exact_match_edge_ids.size() > 1 ||
                   (!boundary_hint && !has_exact_neighbor() && !is_nonconforming());
        }
    };

    template<typename GeomTraits>
    struct ActiveSpatialEdgeContainment2D
    {
        int container_edge_id = -1;
        int contained_edge_id = -1;
    };

    template<typename GeomTraits>
    struct ActiveSpatialEdgeAdjacency2D
    {
        using EdgeKey = SpatialEdgeKey2D<GeomTraits>;
        using EdgeRecord = ActiveSpatialEdgeRecord2D<GeomTraits>;
        using Containment = ActiveSpatialEdgeContainment2D<GeomTraits>;

        std::vector<EdgeRecord> edges{};

        std::unordered_map<
            EdgeKey,
            std::vector<int>,
            SpatialEdgeKey2DHash<GeomTraits>> edge_ids_by_key{};

        std::vector<Containment> containments{};
        std::vector<int> boundary_edge_ids{};
        std::vector<int> conforming_interior_edge_ids{};
        std::vector<int> nonconforming_edge_ids{};
        std::vector<int> singular_edge_ids{};

        [[nodiscard]] const EdgeRecord& edge(int edge_id) const
        {
            if (edge_id < 0 || static_cast<std::size_t>(edge_id) >= edges.size())
                throw std::runtime_error("ActiveSpatialEdgeAdjacency2D::edge: edge id out of range.");
            return edges[static_cast<std::size_t>(edge_id)];
        }

        [[nodiscard]] int edge_id_for_cell_face(int cell_id, int face_id) const noexcept
        {
            for (const auto& record : edges)
            {
                if (record.cell_id == cell_id && record.face_id == face_id)
                    return record.edge_id;
            }
            return -1;
        }

        [[nodiscard]] const EdgeRecord* edge_for_cell_face(int cell_id, int face_id) const noexcept
        {
            const int edge_id = edge_id_for_cell_face(cell_id, face_id);
            if (edge_id < 0)
                return nullptr;
            return &edges[static_cast<std::size_t>(edge_id)];
        }
    };

    template<typename GeomTraits>
    struct SlicewiseActiveSpatialEdgeAdjacencySlice2D
    {
        using cell_id_type = typename MeshTypes<GeomTraits>::cell_id_type;

        double t0 = 0.0;
        double t1 = 0.0;
        int t0_id = -1;
        int t1_id = -1;
        TimeIntervalIdKey interval_id_key{};
        std::vector<cell_id_type> active_cell_ids{};
        std::vector<cell_id_type> representative_cell_ids{};
        ActiveSpatialEdgeAdjacency2D<GeomTraits> adjacency{};
    };

    template<typename GeomTraits>
    struct SlicewiseActiveSpatialEdgeAdjacency2D
    {
        using Slice = SlicewiseActiveSpatialEdgeAdjacencySlice2D<GeomTraits>;

        std::vector<Slice> slices{};

        [[nodiscard]] bool has_singular_edges() const noexcept
        {
            for (const auto& slice : slices)
                if (!slice.adjacency.singular_edge_ids.empty())
                    return true;
            return false;
        }

        [[nodiscard]] bool has_nonconforming_edges() const noexcept
        {
            for (const auto& slice : slices)
                if (!slice.adjacency.nonconforming_edge_ids.empty())
                    return true;
            return false;
        }

        [[nodiscard]] bool is_conforming() const noexcept
        {
            return !has_singular_edges() && !has_nonconforming_edges();
        }
    };

    struct SlicewiseActiveSpatialEdgeAdjacencyStats2D
    {
        std::size_t time_slices_built = 0;
        std::size_t edge_records_built = 0;
    };

    struct LocalSpatialClosureStats2D
    {
        std::size_t active_cells_scanned = 0;
        std::size_t seed_cells_scanned = 0;
        std::size_t edge_records_built = 0;
        std::size_t seed_edge_records = 0;
        std::size_t candidate_edge_visits = 0;
        std::size_t edge_comparisons = 0;
        std::size_t time_overlap_tests = 0;
        std::size_t same_spatial_overlap_scans = 0;
    };

    struct LocalSpatialConformityVerificationStats2D
    {
        std::size_t active_cells_scanned = 0;
        std::size_t seed_cells = 0;
        std::size_t seed_cells_scanned = 0;
        std::size_t active_edge_records_built = 0;
        std::size_t seed_edge_records = 0;
        std::size_t candidate_edge_visits = 0;
        std::size_t candidate_cells = 0;
        double candidate_cells_ratio = 0.0;
        std::size_t time_slices_built = 0;
        std::size_t edge_records_built = 0;
        std::size_t local_edge_records_built = 0;
        std::size_t fallback_to_full_check = 0;
        std::size_t seed_edge_records_checked = 0;
        std::size_t singular_seed_edges = 0;
        std::size_t nonconforming_seed_edges = 0;
        double active_edge_record_construction_seconds = 0.0;
        double active_cell_vertex_index_construction_seconds = 0.0;
        double seed_candidate_discovery_seconds = 0.0;
        double local_slicewise_adjacency_rebuild_seconds = 0.0;
        double seed_edge_conformity_check_seconds = 0.0;
        double local_check_seconds = 0.0;
        double full_check_seconds = 0.0;
    };

    template<typename GeomTraits>
    struct LocalSpatialClosureResult2D
    {
        using cell_id_type = typename MeshTypes<GeomTraits>::cell_id_type;

        std::vector<cell_id_type> forced_cell_ids{};
        LocalSpatialClosureStats2D stats{};
    };

    template<typename GeomTraits>
    struct LocalSpatialConformityVerificationResult2D
    {
        bool is_conforming = true;
        LocalSpatialConformityVerificationStats2D stats{};
    };

    template<typename GeomTraits>
    [[nodiscard]] inline SlicewiseActiveSpatialEdgeAdjacencyStats2D
    adjacency_stats_2d(
        const SlicewiseActiveSpatialEdgeAdjacency2D<GeomTraits>& adjacency)
    {
        SlicewiseActiveSpatialEdgeAdjacencyStats2D stats;
        stats.time_slices_built = adjacency.slices.size();
        for (const auto& slice : adjacency.slices)
            stats.edge_records_built += slice.adjacency.edges.size();
        return stats;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline SpatialEdgeKey2D<GeomTraits>
    make_spatial_edge_key_2d(typename MeshTypes<GeomTraits>::SpatialFaceVertexIds edge)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "make_spatial_edge_key_2d requires dim_space_v == 2.");

        return {sorted_spatial_face_vertex_ids_2d<GeomTraits>(edge)};
    }

    template<typename GeomTraits>
    [[nodiscard]] inline typename MeshTypes<GeomTraits>::SpatialVertexIds
    sorted_spatial_cell_vertex_ids_2d(
        typename MeshTypes<GeomTraits>::SpatialVertexIds vertices)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "sorted_spatial_cell_vertex_ids_2d requires dim_space_v == 2.");
        std::sort(vertices.begin(), vertices.end());
        return vertices;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool same_spatial_cell_vertices_2d(
        const Cell<GeomTraits>& a,
        const Cell<GeomTraits>& b)
    {
        return sorted_spatial_cell_vertex_ids_2d<GeomTraits>(a.spatial_vertex_ids) ==
               sorted_spatial_cell_vertex_ids_2d<GeomTraits>(b.spatial_vertex_ids);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline std::pair<double, double> temporal_interval_bounds_2d(
        const Mesh<GeomTraits>& mesh,
        const Cell<GeomTraits>& cell)
    {
        const double a =
            mesh.temporal_vertices()[static_cast<std::size_t>(cell.temporal_vertex_ids[0])][0];
        const double b =
            mesh.temporal_vertices()[static_cast<std::size_t>(cell.temporal_vertex_ids[1])][0];

        if (a <= b)
            return {a, b};
        return {b, a};
    }

    template<typename GeomTraits>
    [[nodiscard]] inline TimeIntervalIdKey temporal_interval_id_key_2d(
        const Cell<GeomTraits>& cell) noexcept
    {
        return make_time_interval_id_key(
            cell.temporal_vertex_ids[0],
            cell.temporal_vertex_ids[1]);
    }

    [[nodiscard]] inline bool temporal_intervals_overlap_positive_2d(
        double a0,
        double a1,
        double b0,
        double b1,
        double time_tol = 1.0e-12) noexcept
    {
        const double left = std::max(std::min(a0, a1), std::min(b0, b1));
        const double right = std::min(std::max(a0, a1), std::max(b0, b1));
        return right > left + time_tol;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool temporal_intervals_overlap_positive_2d(
        const Mesh<GeomTraits>& mesh,
        const Cell<GeomTraits>& a,
        const Cell<GeomTraits>& b,
        double time_tol = 1.0e-12)
    {
        const auto [a0, a1] = temporal_interval_bounds_2d<GeomTraits>(mesh, a);
        const auto [b0, b1] = temporal_interval_bounds_2d<GeomTraits>(mesh, b);
        return temporal_intervals_overlap_positive_2d(a0, a1, b0, b1, time_tol);
    }

    inline void deduplicate_temporal_endpoints_2d(
        std::vector<double>& endpoints,
        double time_tol = 1.0e-12)
    {
        std::sort(endpoints.begin(), endpoints.end());

        std::vector<double> unique;
        unique.reserve(endpoints.size());

        for (const double t : endpoints)
        {
            if (unique.empty() || std::abs(t - unique.back()) > time_tol)
                unique.push_back(t);
        }

        endpoints = std::move(unique);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline std::vector<typename MeshTypes<GeomTraits>::cell_id_type>
    projected_active_spatial_cell_ids_2d(
        const Mesh<GeomTraits>& mesh,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& active_cells)
    {
        using Types = MeshTypes<GeomTraits>;
        using SpatialVertexIds = typename Types::SpatialVertexIds;
        using cell_id_type = typename Types::cell_id_type;

        static_assert(GeomTraits::dim_space_v == 2,
                      "projected_active_spatial_cell_ids_2d requires dim_space_v == 2.");

        std::vector<cell_id_type> representatives;
        std::vector<SpatialVertexIds> seen;

        representatives.reserve(active_cells.size());
        seen.reserve(active_cells.size());

        for (const auto cell_id : active_cells)
        {
            const auto key =
                sorted_spatial_cell_vertex_ids_2d<GeomTraits>(
                    mesh.cell(cell_id).spatial_vertex_ids);

            if (std::find(seen.begin(), seen.end(), key) != seen.end())
                continue;

            seen.push_back(key);
            representatives.push_back(cell_id);
        }

        return representatives;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool spatial_edge_contains_edge_2d(
        const typename MeshTypes<GeomTraits>::SpatialFaceVertexIds& container,
        const typename MeshTypes<GeomTraits>::SpatialFaceVertexIds& contained,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        if (sorted_spatial_face_vertex_ids_2d<GeomTraits>(container) ==
            sorted_spatial_face_vertex_ids_2d<GeomTraits>(contained))
        {
            return false;
        }

        return spatial_edge_lies_on_boundary_edge_2d<GeomTraits>(
            contained,
            container,
            spatial_vertices);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool spatial_edges_overlap_colinear_positive_2d(
        const typename MeshTypes<GeomTraits>::SpatialFaceVertexIds& a,
        const typename MeshTypes<GeomTraits>::SpatialFaceVertexIds& b,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices,
        double tol = 1.0e-12)
    {
        const auto& a0 = spatial_vertices[static_cast<std::size_t>(a[0])];
        const auto& a1 = spatial_vertices[static_cast<std::size_t>(a[1])];
        const auto& b0 = spatial_vertices[static_cast<std::size_t>(b[0])];
        const auto& b1 = spatial_vertices[static_cast<std::size_t>(b[1])];

        const bool colinear_overlap =
            point_on_segment_2d<GeomTraits>(a0, b0, b1, tol) ||
            point_on_segment_2d<GeomTraits>(a1, b0, b1, tol) ||
            point_on_segment_2d<GeomTraits>(b0, a0, a1, tol) ||
            point_on_segment_2d<GeomTraits>(b1, a0, a1, tol);
        if (!colinear_overlap)
            return false;

        const double ax = std::abs(a1[0] - a0[0]);
        const double ay = std::abs(a1[1] - a0[1]);
        const int component = ax >= ay ? 0 : 1;

        const double amin = std::min(a0[component], a1[component]);
        const double amax = std::max(a0[component], a1[component]);
        const double bmin = std::min(b0[component], b1[component]);
        const double bmax = std::max(b0[component], b1[component]);

        return std::min(amax, bmax) > std::max(amin, bmin) + tol;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool spatial_edge_lies_on_mesh_boundary_2d(
        const typename MeshTypes<GeomTraits>::SpatialFaceVertexIds& edge,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialFaceVertexIds>& boundary_edges,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        for (const auto& boundary_edge : boundary_edges)
        {
            const auto sorted_boundary_edge =
                sorted_spatial_face_vertex_ids_2d<GeomTraits>(boundary_edge);

            if (sorted_spatial_face_vertex_ids_2d<GeomTraits>(edge) == sorted_boundary_edge ||
                spatial_edge_lies_on_boundary_edge_2d<GeomTraits>(
                    edge,
                    sorted_boundary_edge,
                    spatial_vertices))
            {
                return true;
            }
        }

        return false;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline ActiveSpatialEdgeAdjacency2D<GeomTraits>
    build_active_spatial_edge_adjacency_2d(
        const Mesh<GeomTraits>& mesh,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& active_cells)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "build_active_spatial_edge_adjacency_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "build_active_spatial_edge_adjacency_2d requires dim_time_v == 1.");

        using Index = ActiveSpatialEdgeAdjacency2D<GeomTraits>;
        using EdgeRecord = typename Index::EdgeRecord;

        Index index;
        index.edges.reserve(active_cells.size() * 3);
        index.edge_ids_by_key.reserve(active_cells.size() * 3);

        const auto& spatial_vertices = mesh.spatial_vertices();
        const auto& boundary_edges = mesh.spatial_boundary_face_vertex_ids();

        for (const auto cell_id : active_cells)
        {
            const auto& cell = mesh.cell(cell_id);

            for (int face_id = 0; face_id < 3; ++face_id)
            {
                EdgeRecord record;
                record.edge_id = static_cast<int>(index.edges.size());
                record.cell_id = cell_id;
                record.face_id = face_id;
                record.vertex_ids =
                    cell.spatial_faces[static_cast<std::size_t>(face_id)].spatial_vertex_ids;
                record.sorted_vertex_ids =
                    sorted_spatial_face_vertex_ids_2d<GeomTraits>(record.vertex_ids);
                record.boundary_hint =
                    cell.spatial_boundary[static_cast<std::size_t>(face_id)] ||
                    spatial_edge_lies_on_mesh_boundary_2d<GeomTraits>(
                        record.sorted_vertex_ids,
                        boundary_edges,
                        spatial_vertices);

                index.edge_ids_by_key[
                    SpatialEdgeKey2D<GeomTraits>{record.sorted_vertex_ids}]
                    .push_back(record.edge_id);
                index.edges.push_back(std::move(record));
            }
        }

        for (const auto& [key, edge_ids] : index.edge_ids_by_key)
        {
            (void)key;
            for (const int edge_id : edge_ids)
            {
                auto& record = index.edges[static_cast<std::size_t>(edge_id)];
                record.exact_match_edge_ids.reserve(edge_ids.size() > 0 ? edge_ids.size() - 1 : 0);
                for (const int other_edge_id : edge_ids)
                {
                    if (other_edge_id != edge_id)
                        record.exact_match_edge_ids.push_back(other_edge_id);
                }
            }
        }

        for (int i = 0; i < static_cast<int>(index.edges.size()); ++i)
        {
            for (int j = i + 1; j < static_cast<int>(index.edges.size()); ++j)
            {
                const auto& edge_i = index.edges[static_cast<std::size_t>(i)];
                const auto& edge_j = index.edges[static_cast<std::size_t>(j)];

                if (edge_i.sorted_vertex_ids == edge_j.sorted_vertex_ids)
                    continue;

                if (spatial_edge_contains_edge_2d<GeomTraits>(
                        edge_i.sorted_vertex_ids,
                        edge_j.sorted_vertex_ids,
                        spatial_vertices))
                {
                    index.edges[static_cast<std::size_t>(i)].contained_edge_ids.push_back(j);
                    index.edges[static_cast<std::size_t>(j)].containing_edge_ids.push_back(i);
                    index.containments.push_back({i, j});
                    continue;
                }

                if (spatial_edge_contains_edge_2d<GeomTraits>(
                        edge_j.sorted_vertex_ids,
                        edge_i.sorted_vertex_ids,
                        spatial_vertices))
                {
                    index.edges[static_cast<std::size_t>(j)].contained_edge_ids.push_back(i);
                    index.edges[static_cast<std::size_t>(i)].containing_edge_ids.push_back(j);
                    index.containments.push_back({j, i});
                }
            }
        }

        for (const auto& record : index.edges)
        {
            if (record.is_boundary())
                index.boundary_edge_ids.push_back(record.edge_id);
            if (record.is_conforming_interior())
                index.conforming_interior_edge_ids.push_back(record.edge_id);
            if (record.is_nonconforming())
                index.nonconforming_edge_ids.push_back(record.edge_id);
            if (record.is_singular())
                index.singular_edge_ids.push_back(record.edge_id);
        }

        return index;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline SlicewiseActiveSpatialEdgeAdjacency2D<GeomTraits>
    build_slicewise_active_spatial_edge_adjacency_2d(
        const Mesh<GeomTraits>& mesh,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& active_cells,
        double time_tol = 1.0e-12)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "build_slicewise_active_spatial_edge_adjacency_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "build_slicewise_active_spatial_edge_adjacency_2d requires dim_time_v == 1.");

        using Index = SlicewiseActiveSpatialEdgeAdjacency2D<GeomTraits>;
        using Slice = typename Index::Slice;

        Index index;
        if (active_cells.empty())
            return index;

        struct EndpointRecord
        {
            int vertex_id = -1;
            double time = 0.0;
        };

        std::vector<EndpointRecord> endpoints;
        endpoints.reserve(active_cells.size() * 2);
        for (const auto cell_id : active_cells)
        {
            const auto& cell = mesh.cell(cell_id);
            for (const int temporal_vertex_id : cell.temporal_vertex_ids)
            {
                endpoints.push_back(
                    EndpointRecord{
                        temporal_vertex_id,
                        mesh.temporal_vertices()[
                            static_cast<std::size_t>(temporal_vertex_id)][0]});
            }
        }

        std::sort(
            endpoints.begin(),
            endpoints.end(),
            [](const EndpointRecord& a, const EndpointRecord& b)
            {
                if (a.time != b.time)
                    return a.time < b.time;
                return a.vertex_id < b.vertex_id;
            });
        endpoints.erase(
            std::unique(
                endpoints.begin(),
                endpoints.end(),
                [](const EndpointRecord& a, const EndpointRecord& b)
                {
                    return a.vertex_id == b.vertex_id ||
                           a.time == b.time;
                }),
            endpoints.end());
        if (endpoints.size() < 2)
            return index;

        index.slices.reserve(endpoints.size() - 1);

        for (std::size_t i = 0; i + 1 < endpoints.size(); ++i)
        {
            const double t0 = endpoints[i].time;
            const double t1 = endpoints[i + 1].time;
            if (t1 <= t0 + time_tol)
                continue;

            Slice slice;
            slice.t0 = t0;
            slice.t1 = t1;
            slice.t0_id = endpoints[i].vertex_id;
            slice.t1_id = endpoints[i + 1].vertex_id;
            slice.interval_id_key =
                make_time_interval_id_key(slice.t0_id, slice.t1_id);
            slice.active_cell_ids.reserve(active_cells.size());

            for (const auto cell_id : active_cells)
            {
                const auto [cell_t0, cell_t1] =
                    temporal_interval_bounds_2d<GeomTraits>(mesh, mesh.cell(cell_id));
                if (temporal_intervals_overlap_positive_2d(
                        cell_t0,
                        cell_t1,
                        t0,
                        t1,
                        time_tol))
                {
                    slice.active_cell_ids.push_back(cell_id);
                }
            }

            slice.representative_cell_ids =
                projected_active_spatial_cell_ids_2d(mesh, slice.active_cell_ids);
            slice.adjacency =
                build_active_spatial_edge_adjacency_2d(
                    mesh,
                    slice.representative_cell_ids);
            index.slices.push_back(std::move(slice));
        }

        return index;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline LocalSpatialClosureResult2D<GeomTraits>
    collect_local_spatial_closure_forced_cells_2d(
        const Mesh<GeomTraits>& mesh,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& active_cells,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& seed_cells,
        double time_tol = 1.0e-12)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "collect_local_spatial_closure_forced_cells_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "collect_local_spatial_closure_forced_cells_2d requires dim_time_v == 1.");

        using Types = MeshTypes<GeomTraits>;
        using cell_id_type = typename Types::cell_id_type;
        using Edge = typename Types::SpatialFaceVertexIds;
        using CellKey = SpatialCellKey2D<GeomTraits>;

        struct LocalEdgeRecord
        {
            cell_id_type cell_id = -1;
            int face_id = -1;
            Edge sorted_vertex_ids{};
        };

        LocalSpatialClosureResult2D<GeomTraits> result;
        if (active_cells.empty() || seed_cells.empty())
            return result;

        const auto n_cells = static_cast<std::size_t>(mesh.n_cells());
        std::vector<char> is_active(n_cells, 0);
        std::vector<char> forced_seen(n_cells, 0);

        std::vector<LocalEdgeRecord> active_edges;
        active_edges.reserve(active_cells.size() * 3);

        std::vector<std::vector<int>> edge_ids_by_spatial_vertex(
            mesh.spatial_vertices().size());

        std::unordered_map<
            CellKey,
            std::vector<cell_id_type>,
            SpatialCellKey2DHash<GeomTraits>> active_cells_by_spatial_key;
        active_cells_by_spatial_key.reserve(active_cells.size());

        for (const auto cell_id : active_cells)
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= n_cells)
                continue;

            is_active[static_cast<std::size_t>(cell_id)] = 1;
            ++result.stats.active_cells_scanned;

            const auto& cell = mesh.cell(cell_id);
            active_cells_by_spatial_key[
                CellKey{
                    sorted_spatial_cell_vertex_ids_2d<GeomTraits>(
                        cell.spatial_vertex_ids)}]
                .push_back(cell_id);

            for (int face_id = 0; face_id < 3; ++face_id)
            {
                const auto sorted_edge =
                    sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                        cell.spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);
                const int edge_id = static_cast<int>(active_edges.size());
                active_edges.push_back(
                    LocalEdgeRecord{cell_id, face_id, sorted_edge});

                for (const auto vid : sorted_edge)
                {
                    if (vid >= 0 &&
                        static_cast<std::size_t>(vid) <
                            edge_ids_by_spatial_vertex.size())
                    {
                        edge_ids_by_spatial_vertex[
                            static_cast<std::size_t>(vid)]
                            .push_back(edge_id);
                    }
                }
            }
        }
        result.stats.edge_records_built = active_edges.size();

        auto enqueue_same_spatial_overlaps =
            [&](cell_id_type container_cell_id, cell_id_type contained_cell_id)
            {
                if (container_cell_id < 0 ||
                    contained_cell_id < 0 ||
                    static_cast<std::size_t>(container_cell_id) >= n_cells ||
                    static_cast<std::size_t>(contained_cell_id) >= n_cells)
                {
                    return;
                }

                const auto& container_cell = mesh.cell(container_cell_id);
                const auto key =
                    CellKey{
                        sorted_spatial_cell_vertex_ids_2d<GeomTraits>(
                            container_cell.spatial_vertex_ids)};
                const auto it = active_cells_by_spatial_key.find(key);
                if (it == active_cells_by_spatial_key.end())
                    return;

                const auto& contained_cell = mesh.cell(contained_cell_id);
                for (const auto same_spatial_cell_id : it->second)
                {
                    ++result.stats.same_spatial_overlap_scans;
                    if (same_spatial_cell_id < 0 ||
                        static_cast<std::size_t>(same_spatial_cell_id) >=
                            n_cells)
                    {
                        continue;
                    }

                    ++result.stats.time_overlap_tests;
                    if (!temporal_intervals_overlap_positive_2d(
                            mesh,
                            mesh.cell(same_spatial_cell_id),
                            contained_cell,
                            time_tol))
                    {
                        continue;
                    }

                    const auto idx =
                        static_cast<std::size_t>(same_spatial_cell_id);
                    if (!forced_seen[idx])
                    {
                        forced_seen[idx] = 1;
                        result.forced_cell_ids.push_back(
                            same_spatial_cell_id);
                    }
                }
            };

        std::vector<int> candidate_edge_ids;
        std::vector<int> edge_marker(active_edges.size(), 0);
        int marker_token = 0;

        for (const auto seed_cell_id : seed_cells)
        {
            if (seed_cell_id < 0 ||
                static_cast<std::size_t>(seed_cell_id) >= n_cells ||
                !is_active[static_cast<std::size_t>(seed_cell_id)])
            {
                continue;
            }

            ++result.stats.seed_cells_scanned;
            const auto& seed_cell = mesh.cell(seed_cell_id);

            for (int face_id = 0; face_id < 3; ++face_id)
            {
                ++result.stats.seed_edge_records;
                const auto seed_edge =
                    sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                        seed_cell
                            .spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);

                candidate_edge_ids.clear();
                ++marker_token;
                if (marker_token == 0)
                {
                    std::fill(edge_marker.begin(), edge_marker.end(), 0);
                    marker_token = 1;
                }

                for (const auto vid : seed_edge)
                {
                    if (vid < 0 ||
                        static_cast<std::size_t>(vid) >=
                            edge_ids_by_spatial_vertex.size())
                    {
                        continue;
                    }

                    for (const int edge_id :
                         edge_ids_by_spatial_vertex[
                             static_cast<std::size_t>(vid)])
                    {
                        if (edge_id < 0 ||
                            static_cast<std::size_t>(edge_id) >=
                                edge_marker.size())
                        {
                            continue;
                        }

                        if (edge_marker[static_cast<std::size_t>(edge_id)] ==
                            marker_token)
                        {
                            continue;
                        }

                        edge_marker[static_cast<std::size_t>(edge_id)] =
                            marker_token;
                        candidate_edge_ids.push_back(edge_id);
                    }
                }

                result.stats.candidate_edge_visits +=
                    candidate_edge_ids.size();

                for (const int candidate_edge_id : candidate_edge_ids)
                {
                    const auto& candidate =
                        active_edges[static_cast<std::size_t>(
                            candidate_edge_id)];
                    if (candidate.cell_id == seed_cell_id)
                        continue;

                    ++result.stats.edge_comparisons;
                    ++result.stats.time_overlap_tests;
                    if (!temporal_intervals_overlap_positive_2d(
                            mesh,
                            mesh.cell(candidate.cell_id),
                            seed_cell,
                            time_tol))
                    {
                        continue;
                    }

                    if (spatial_edge_contains_edge_2d<GeomTraits>(
                            candidate.sorted_vertex_ids,
                            seed_edge,
                            mesh.spatial_vertices()))
                    {
                        enqueue_same_spatial_overlaps(
                            candidate.cell_id,
                            seed_cell_id);
                        continue;
                    }

                    if (spatial_edge_contains_edge_2d<GeomTraits>(
                            seed_edge,
                            candidate.sorted_vertex_ids,
                            mesh.spatial_vertices()))
                    {
                        enqueue_same_spatial_overlaps(
                            seed_cell_id,
                            candidate.cell_id);
                    }
                }
            }
        }

        std::sort(result.forced_cell_ids.begin(), result.forced_cell_ids.end());
        return result;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline LocalSpatialConformityVerificationResult2D<GeomTraits>
    verify_local_spatial_conforming_2d(
        const Mesh<GeomTraits>& mesh,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& active_cells,
        const std::vector<typename MeshTypes<GeomTraits>::cell_id_type>& seed_cells,
        double time_tol = 1.0e-12,
        double full_check_fallback_candidate_fraction = 0.75)
    {
        static_assert(GeomTraits::dim_space_v == 2,
                      "verify_local_spatial_conforming_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "verify_local_spatial_conforming_2d requires dim_time_v == 1.");

        using Types = MeshTypes<GeomTraits>;
        using cell_id_type = typename Types::cell_id_type;
        using Edge = typename Types::SpatialFaceVertexIds;

        struct EdgeQuery
        {
            Edge sorted_vertex_ids{};
        };

        LocalSpatialConformityVerificationResult2D<GeomTraits> result;
        result.stats.seed_cells = seed_cells.size();
        if (active_cells.empty() || seed_cells.empty())
            return result;

        using Clock = std::chrono::steady_clock;
        const auto function_start = Clock::now();
        const auto elapsed_seconds =
            [](const Clock::time_point& begin)
        {
            return std::chrono::duration<double>(
                       Clock::now() - begin)
                .count();
        };

        auto phase_start = Clock::now();
        const auto n_cells = static_cast<std::size_t>(mesh.n_cells());
        std::vector<char> active_seen(n_cells, 0);

        std::vector<std::vector<cell_id_type>> active_cells_by_spatial_vertex(
            mesh.spatial_vertices().size());

        for (const auto cell_id : active_cells)
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= n_cells)
                continue;

            active_seen[static_cast<std::size_t>(cell_id)] = 1;
            ++result.stats.active_cells_scanned;
            const auto& cell = mesh.cell(cell_id);

            for (const auto vid : cell.spatial_vertex_ids)
            {
                if (vid >= 0 &&
                    static_cast<std::size_t>(vid) <
                        active_cells_by_spatial_vertex.size())
                {
                    active_cells_by_spatial_vertex[
                        static_cast<std::size_t>(vid)]
                        .push_back(cell_id);
                }
            }
        }
        result.stats.active_edge_records_built = 0;
        result.stats.active_cell_vertex_index_construction_seconds =
            elapsed_seconds(phase_start);

        std::vector<cell_id_type> raw_candidate_cell_ids;
        std::vector<cell_id_type> local_active_cells;
        std::vector<int> raw_cell_marker(n_cells, 0);
        std::vector<int> local_cell_marker(n_cells, 0);
        int raw_marker_token = 0;
        int local_marker_token = 0;
        const auto& spatial_vertices = mesh.spatial_vertices();

        auto edge_already_present =
            [](const std::vector<EdgeQuery>& queries, const Edge& edge)
        {
            return std::find_if(
                       queries.begin(),
                       queries.end(),
                       [&](const EdgeQuery& query)
                       {
                           return query.sorted_vertex_ids == edge;
                       }) != queries.end();
        };

        auto seed_edge_queries =
            [&](cell_id_type seed_cell_id, const Edge& seed_edge)
        {
            std::vector<EdgeQuery> queries;
            queries.reserve(8);
            queries.push_back(EdgeQuery{seed_edge});

            cell_id_type ancestor_id = mesh.cell(seed_cell_id).parent_id;
            while (ancestor_id >= 0 &&
                   static_cast<std::size_t>(ancestor_id) < n_cells)
            {
                const auto& ancestor = mesh.cell(ancestor_id);
                for (int face_id = 0; face_id < 3; ++face_id)
                {
                    const auto ancestor_edge =
                        sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                            ancestor
                                .spatial_faces[static_cast<std::size_t>(face_id)]
                                .spatial_vertex_ids);
                    if (!spatial_edge_lies_on_boundary_edge_2d<GeomTraits>(
                            seed_edge,
                            ancestor_edge,
                            spatial_vertices))
                    {
                        continue;
                    }
                    if (!edge_already_present(queries, ancestor_edge))
                        queries.push_back(EdgeQuery{ancestor_edge});
                }

                ancestor_id = ancestor.parent_id;
            }

            return queries;
        };

        auto candidate_cell_touches_seed_edge =
            [&](cell_id_type candidate_cell_id, const Edge& seed_edge)
        {
            if (candidate_cell_id < 0 ||
                static_cast<std::size_t>(candidate_cell_id) >= n_cells)
            {
                return false;
            }

            const auto& candidate_cell = mesh.cell(candidate_cell_id);
            for (int face_id = 0; face_id < 3; ++face_id)
            {
                const auto candidate_edge =
                    sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                        candidate_cell
                            .spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);
                if (spatial_edges_overlap_colinear_positive_2d<GeomTraits>(
                        candidate_edge,
                        seed_edge,
                        spatial_vertices))
                {
                    return true;
                }
            }

            return false;
        };

        auto collect_seed_cell_candidates =
            [&](cell_id_type seed_cell_id)
        {
            ++local_marker_token;
            if (local_marker_token == 0)
            {
                std::fill(local_cell_marker.begin(), local_cell_marker.end(), 0);
                local_marker_token = 1;
            }

            local_active_cells.clear();
            local_cell_marker[static_cast<std::size_t>(seed_cell_id)] =
                local_marker_token;
            local_active_cells.push_back(seed_cell_id);

            const auto& seed_cell = mesh.cell(seed_cell_id);
            for (int face_id = 0; face_id < 3; ++face_id)
            {
                ++result.stats.seed_edge_records;
                const auto seed_edge =
                    sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                        seed_cell
                            .spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);

                ++raw_marker_token;
                if (raw_marker_token == 0)
                {
                    std::fill(raw_cell_marker.begin(), raw_cell_marker.end(), 0);
                    raw_marker_token = 1;
                }
                raw_candidate_cell_ids.clear();

                const auto queries = seed_edge_queries(seed_cell_id, seed_edge);
                for (const auto& query : queries)
                {
                    for (const auto vid : query.sorted_vertex_ids)
                    {
                        if (vid < 0 ||
                            static_cast<std::size_t>(vid) >=
                                active_cells_by_spatial_vertex.size())
                        {
                            continue;
                        }

                        for (const auto candidate_cell_id :
                             active_cells_by_spatial_vertex[
                                 static_cast<std::size_t>(vid)])
                        {
                            if (candidate_cell_id < 0 ||
                                static_cast<std::size_t>(candidate_cell_id) >=
                                    n_cells)
                            {
                                continue;
                            }

                            const auto idx =
                                static_cast<std::size_t>(candidate_cell_id);
                            if (raw_cell_marker[idx] == raw_marker_token)
                                continue;

                            raw_cell_marker[idx] = raw_marker_token;
                            raw_candidate_cell_ids.push_back(candidate_cell_id);
                        }
                    }
                }

                result.stats.candidate_edge_visits +=
                    raw_candidate_cell_ids.size();
                for (const auto candidate_cell_id : raw_candidate_cell_ids)
                {
                    if (candidate_cell_id < 0 ||
                        static_cast<std::size_t>(candidate_cell_id) >=
                            n_cells)
                    {
                        continue;
                    }

                    if (!temporal_intervals_overlap_positive_2d(
                            mesh,
                            mesh.cell(candidate_cell_id),
                            seed_cell,
                            time_tol))
                    {
                        continue;
                    }

                    if (!candidate_cell_touches_seed_edge(
                            candidate_cell_id,
                            seed_edge))
                    {
                        continue;
                    }

                    const auto idx =
                        static_cast<std::size_t>(candidate_cell_id);
                    if (local_cell_marker[idx] == local_marker_token)
                        continue;

                    local_cell_marker[idx] = local_marker_token;
                    local_active_cells.push_back(candidate_cell_id);
                }
            }

            return local_active_cells;
        };

        auto check_seed_cell_records =
            [&](cell_id_type seed_cell_id,
                const SlicewiseActiveSpatialEdgeAdjacency2D<GeomTraits>& adjacency)
        {
            for (const auto& slice : adjacency.slices)
            {
                for (const auto& record : slice.adjacency.edges)
                {
                    if (record.cell_id != seed_cell_id)
                        continue;

                    ++result.stats.seed_edge_records_checked;
                    if (record.is_singular())
                        ++result.stats.singular_seed_edges;
                    if (record.is_nonconforming())
                        ++result.stats.nonconforming_seed_edges;
                }
            }
        };

        for (const auto seed_cell_id : seed_cells)
        {
            if (seed_cell_id < 0 ||
                static_cast<std::size_t>(seed_cell_id) >= n_cells ||
                !active_seen[static_cast<std::size_t>(seed_cell_id)])
            {
                continue;
            }

            ++result.stats.seed_cells_scanned;

            phase_start = Clock::now();
            const auto candidates = collect_seed_cell_candidates(seed_cell_id);
            result.stats.seed_candidate_discovery_seconds +=
                elapsed_seconds(phase_start);

            result.stats.candidate_cells =
                std::max(result.stats.candidate_cells, candidates.size());
            if (result.stats.active_cells_scanned > 0)
            {
                result.stats.candidate_cells_ratio =
                    std::max(
                        result.stats.candidate_cells_ratio,
                        static_cast<double>(candidates.size()) /
                            static_cast<double>(
                                result.stats.active_cells_scanned));
            }

            if (result.stats.candidate_cells_ratio >
                full_check_fallback_candidate_fraction)
            {
                result.stats.fallback_to_full_check = 1;
                phase_start = Clock::now();
                const auto full_adjacency =
                    build_slicewise_active_spatial_edge_adjacency_2d(
                        mesh,
                        active_cells,
                        time_tol);
                const auto full_stats = adjacency_stats_2d(full_adjacency);
                result.stats.time_slices_built = full_stats.time_slices_built;
                result.stats.edge_records_built = full_stats.edge_records_built;
                result.stats.full_check_seconds = elapsed_seconds(phase_start);
                result.is_conforming = full_adjacency.is_conforming();
                result.stats.local_check_seconds = elapsed_seconds(function_start);
                return result;
            }

            phase_start = Clock::now();
            const auto local_adjacency =
                build_slicewise_active_spatial_edge_adjacency_2d(
                    mesh,
                    candidates,
                    time_tol);
            const auto local_stats = adjacency_stats_2d(local_adjacency);
            result.stats.time_slices_built += local_stats.time_slices_built;
            result.stats.edge_records_built += local_stats.edge_records_built;
            result.stats.local_edge_records_built +=
                local_stats.edge_records_built;
            result.stats.local_slicewise_adjacency_rebuild_seconds +=
                elapsed_seconds(phase_start);

            phase_start = Clock::now();
            check_seed_cell_records(seed_cell_id, local_adjacency);
            result.stats.seed_edge_conformity_check_seconds +=
                elapsed_seconds(phase_start);
        }

        result.is_conforming =
            result.stats.singular_seed_edges == 0 &&
            result.stats.nonconforming_seed_edges == 0;
        result.stats.local_check_seconds = elapsed_seconds(function_start);
        return result;
    }
}

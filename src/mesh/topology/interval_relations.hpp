#pragma once

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "../../core/hash.hpp"
#include "../cell.hpp"
#include "../types.hpp"

namespace mesh
{
    template<typename GeomTraits>
    class Mesh;
}

namespace mesh::topology
{
    template<typename GeomTraits>
    [[nodiscard]] inline bool temporal_intervals_overlap_1d(
        const Cell<GeomTraits>& a,
        const Cell<GeomTraits>& b,
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices)
    {
        static_assert(GeomTraits::dim_time_v == 1,
                      "temporal_intervals_overlap_1d requires dim_time_v == 1.");

        const double a0 = temporal_vertices[a.temporal_vertex_ids[0]][0];
        const double a1 = temporal_vertices[a.temporal_vertex_ids[1]][0];
        const double b0 = temporal_vertices[b.temporal_vertex_ids[0]][0];
        const double b1 = temporal_vertices[b.temporal_vertex_ids[1]][0];

        return !(a1 <= b0 || b1 <= a0);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool temporal_interval_contains_1d(
        const Cell<GeomTraits>& outer,
        const Cell<GeomTraits>& inner,
        const std::vector<typename MeshTypes<GeomTraits>::TemporalPoint>& temporal_vertices)
    {
        static_assert(GeomTraits::dim_time_v == 1,
                      "temporal_interval_contains_1d requires dim_time_v == 1.");

        const double o0 = temporal_vertices[outer.temporal_vertex_ids[0]][0];
        const double o1 = temporal_vertices[outer.temporal_vertex_ids[1]][0];
        const double i0 = temporal_vertices[inner.temporal_vertex_ids[0]][0];
        const double i1 = temporal_vertices[inner.temporal_vertex_ids[1]][0];

        return (o0 <= i0 && i1 <= o1);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool spatial_intervals_overlap_1d(
        const Cell<GeomTraits>& a,
        const Cell<GeomTraits>& b,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        static_assert(GeomTraits::dim_space_v == 1,
                      "spatial_intervals_overlap_1d requires dim_space_v == 1.");

        const double a0 = spatial_vertices[a.spatial_vertex_ids[0]][0];
        const double a1 = spatial_vertices[a.spatial_vertex_ids[1]][0];
        const double b0 = spatial_vertices[b.spatial_vertex_ids[0]][0];
        const double b1 = spatial_vertices[b.spatial_vertex_ids[1]][0];

        return !(a1 <= b0 || b1 <= a0);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool spatial_interval_contains_1d(
        const Cell<GeomTraits>& outer,
        const Cell<GeomTraits>& inner,
        const std::vector<typename MeshTypes<GeomTraits>::SpatialPoint>& spatial_vertices)
    {
        static_assert(GeomTraits::dim_space_v == 1,
                      "spatial_interval_contains_1d requires dim_space_v == 1.");

        const double o0 = spatial_vertices[outer.spatial_vertex_ids[0]][0];
        const double o1 = spatial_vertices[outer.spatial_vertex_ids[1]][0];
        const double i0 = spatial_vertices[inner.spatial_vertex_ids[0]][0];
        const double i1 = spatial_vertices[inner.spatial_vertex_ids[1]][0];

        return (o0 <= i0 && i1 <= o1);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool intervals_overlap_1d(
        double a0, double a1,
        double b0, double b1) noexcept
    {
        static_assert(GeomTraits::dim_space_v == 1 || GeomTraits::dim_time_v == 1,
                      "intervals_overlap_1d is for 1D intervals.");
        return !(a1 <= b0 || b1 <= a0);
    }

    // -------------------------------------------------------------------------
    // Exact interval keys
    // -------------------------------------------------------------------------

    template<typename GeomTraits>
    struct SpatialIntervalKey1D
    {
        using Types = MeshTypes<GeomTraits>;
        using SpatialVertexIds = typename Types::SpatialVertexIds;

        SpatialVertexIds vertex_ids{};

        bool operator==(const SpatialIntervalKey1D&) const noexcept = default;
    };

    template<typename GeomTraits>
    struct TemporalIntervalKey1D
    {
        using Types = MeshTypes<GeomTraits>;
        using TemporalVertexIds = typename Types::TemporalVertexIds;

        TemporalVertexIds vertex_ids{};

        bool operator==(const TemporalIntervalKey1D&) const noexcept = default;
    };

    template<typename GeomTraits>
    [[nodiscard]] inline SpatialIntervalKey1D<GeomTraits>
    make_spatial_interval_key_1d(typename MeshTypes<GeomTraits>::SpatialVertexIds ids)
    {
        static_assert(GeomTraits::dim_space_v == 1,
                      "make_spatial_interval_key_1d requires dim_space_v == 1.");
        std::sort(ids.begin(), ids.end());
        return {ids};
    }

    template<typename GeomTraits>
    [[nodiscard]] inline TemporalIntervalKey1D<GeomTraits>
    make_temporal_interval_key_1d(typename MeshTypes<GeomTraits>::TemporalVertexIds ids)
    {
        static_assert(GeomTraits::dim_time_v == 1,
                      "make_temporal_interval_key_1d requires dim_time_v == 1.");
        std::sort(ids.begin(), ids.end());
        return {ids};
    }

    // -------------------------------------------------------------------------
    // Active prism records for 1+1D
    // -------------------------------------------------------------------------

    template<typename GeomTraits>
    struct PrismRecord1D
    {
        using Types = MeshTypes<GeomTraits>;
        using SpatialVertexIds  = typename Types::SpatialVertexIds;
        using TemporalVertexIds = typename Types::TemporalVertexIds;

        int cell_id = -1;

        SpatialVertexIds  spatial_vertex_ids{};
        TemporalVertexIds temporal_vertex_ids{};

        double x0 = 0.0;
        double x1 = 0.0;
        double t0 = 0.0;
        double t1 = 0.0;

        int generation = 0;
    };

    template<typename GeomTraits>
    [[nodiscard]] inline std::vector<PrismRecord1D<GeomTraits>>
    make_active_prism_records_1d(
        const std::vector<int>& active_cells,
        const Mesh<GeomTraits>& mesh)
    {
        static_assert(GeomTraits::dim_space_v == 1,
                      "make_active_prism_records_1d requires dim_space_v == 1.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "make_active_prism_records_1d requires dim_time_v == 1.");

        std::vector<PrismRecord1D<GeomTraits>> out;
        out.reserve(active_cells.size());

        const auto& spatial_vertices  = mesh.spatial_vertices();
        const auto& temporal_vertices = mesh.temporal_vertices();

        for (const int cell_id : active_cells)
        {
            const auto& c = mesh.cell(cell_id);

            PrismRecord1D<GeomTraits> rec;
            rec.cell_id = cell_id;
            rec.spatial_vertex_ids  = c.spatial_vertex_ids;
            rec.temporal_vertex_ids = c.temporal_vertex_ids;
            rec.x0 = spatial_vertices[c.spatial_vertex_ids[0]][0];
            rec.x1 = spatial_vertices[c.spatial_vertex_ids[1]][0];
            rec.t0 = temporal_vertices[c.temporal_vertex_ids[0]][0];
            rec.t1 = temporal_vertices[c.temporal_vertex_ids[1]][0];
            rec.generation = c.generation;
            out.push_back(rec);
        }

        return out;
    }

    template<typename GeomTraits>
    struct ActivePrismIndex1D
    {
        using Record = PrismRecord1D<GeomTraits>;

        std::vector<Record> records;

        std::unordered_map<int, std::size_t> by_cell_id;

        // Exact temporal slab -> all active prisms on that slab, sorted by x-interval.
        std::unordered_map<TemporalIntervalKey1D<GeomTraits>, std::vector<const Record*>> by_temporal_interval;

        // Exact spatial slab -> all active prisms on that slab, sorted by t-interval.
        std::unordered_map<SpatialIntervalKey1D<GeomTraits>, std::vector<const Record*>> by_spatial_interval;

        ActivePrismIndex1D() = default;

        ActivePrismIndex1D(const std::vector<int>& active_cells,
                           const Mesh<GeomTraits>& mesh)
        {
            rebuild(active_cells, mesh);
        }

        void rebuild(const std::vector<int>& active_cells,
                     const Mesh<GeomTraits>& mesh)
        {
            records = make_active_prism_records_1d(active_cells, mesh);

            by_cell_id.clear();
            by_temporal_interval.clear();
            by_spatial_interval.clear();

            by_cell_id.reserve(records.size());
            by_temporal_interval.reserve(records.size());
            by_spatial_interval.reserve(records.size());

            for (std::size_t i = 0; i < records.size(); ++i)
                by_cell_id.emplace(records[i].cell_id, i);

            for (const auto& rec : records)
            {
                by_temporal_interval[make_temporal_interval_key_1d<GeomTraits>(rec.temporal_vertex_ids)].push_back(&rec);
                by_spatial_interval[make_spatial_interval_key_1d<GeomTraits>(rec.spatial_vertex_ids)].push_back(&rec);
            }

            for (auto& [key, group] : by_temporal_interval)
            {
                std::sort(group.begin(), group.end(),
                    [](const Record* a, const Record* b)
                    {
                        if (a->x0 != b->x0) return a->x0 < b->x0;
                        if (a->x1 != b->x1) return a->x1 < b->x1;
                        return a->cell_id < b->cell_id;
                    });
            }

            for (auto& [key, group] : by_spatial_interval)
            {
                std::sort(group.begin(), group.end(),
                    [](const Record* a, const Record* b)
                    {
                        if (a->t0 != b->t0) return a->t0 < b->t0;
                        if (a->t1 != b->t1) return a->t1 < b->t1;
                        return a->cell_id < b->cell_id;
                    });
            }
        }

        [[nodiscard]] const Record* find(int cell_id) const
        {
            const auto it = by_cell_id.find(cell_id);
            if (it == by_cell_id.end())
                return nullptr;
            return &records[it->second];
        }
    };

    // -------------------------------------------------------------------------
    // Ancestor exact-interval keys from the cell parent chain
    // -------------------------------------------------------------------------

    template<typename GeomTraits>
    [[nodiscard]] inline std::vector<TemporalIntervalKey1D<GeomTraits>>
    temporal_ancestor_keys_1d(const Mesh<GeomTraits>& mesh, int cell_id)
    {
        std::vector<TemporalIntervalKey1D<GeomTraits>> out;

        int cur = cell_id;
        while (cur >= 0)
        {
            const auto& c = mesh.cell(cur);
            const auto key = make_temporal_interval_key_1d<GeomTraits>(c.temporal_vertex_ids);

            if (out.empty() || !(out.back() == key))
                out.push_back(key);

            cur = c.parent_id;
        }

        return out;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline std::vector<SpatialIntervalKey1D<GeomTraits>>
    spatial_ancestor_keys_1d(const Mesh<GeomTraits>& mesh, int cell_id)
    {
        std::vector<SpatialIntervalKey1D<GeomTraits>> out;

        int cur = cell_id;
        while (cur >= 0)
        {
            const auto& c = mesh.cell(cur);
            const auto key = make_spatial_interval_key_1d<GeomTraits>(c.spatial_vertex_ids);

            if (out.empty() || !(out.back() == key))
                out.push_back(key);

            cur = c.parent_id;
        }

        return out;
    }

    // -------------------------------------------------------------------------
    // Lower-bound helpers for overlap scans on sorted groups
    // -------------------------------------------------------------------------

    template<typename GeomTraits>
    [[nodiscard]] inline std::size_t
    first_possible_x_overlap_1d(const std::vector<const PrismRecord1D<GeomTraits>*>& group,
                                double x0)
    {
        const auto it = std::lower_bound(
            group.begin(), group.end(), x0,
            [](const PrismRecord1D<GeomTraits>* rec, double value)
            {
                return rec->x0 < value;
            });

        if (it == group.begin())
            return 0;

        return static_cast<std::size_t>((it - group.begin()) - 1);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline std::size_t
    first_possible_t_overlap_1d(const std::vector<const PrismRecord1D<GeomTraits>*>& group,
                                double t0)
    {
        const auto it = std::lower_bound(
            group.begin(), group.end(), t0,
            [](const PrismRecord1D<GeomTraits>* rec, double value)
            {
                return rec->t0 < value;
            });

        if (it == group.begin())
            return 0;

        return static_cast<std::size_t>((it - group.begin()) - 1);
    }
}

namespace std
{
    template<typename GeomTraits>
    struct hash<mesh::topology::SpatialIntervalKey1D<GeomTraits>>
    {
        std::size_t operator()(const mesh::topology::SpatialIntervalKey1D<GeomTraits>& key) const noexcept
        {
            std::size_t seed = 0;
            for (const auto v : key.vertex_ids)
                core::hash_combine(seed, v);
            return seed;
        }
    };

    template<typename GeomTraits>
    struct hash<mesh::topology::TemporalIntervalKey1D<GeomTraits>>
    {
        std::size_t operator()(const mesh::topology::TemporalIntervalKey1D<GeomTraits>& key) const noexcept
        {
            std::size_t seed = 0;
            for (const auto v : key.vertex_ids)
                core::hash_combine(seed, v);
            return seed;
        }
    };
}
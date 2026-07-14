#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../../core/hash.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../mesh/topology/spatial_edge_adjacency_2d.hpp"

namespace finite_element::fespace::topology
{
    template<typename GeomTraits>
    class ActiveEdgeIntervalIndex2D
    {
    public:
        using MeshType = mesh::Mesh<GeomTraits>;
        using EdgeKey = mesh::topology::SpatialEdgeKey2D<GeomTraits>;
        using EdgeHash = mesh::topology::SpatialEdgeKey2DHash<GeomTraits>;
        using RecordId = std::size_t;

        struct Record
        {
            int cell_id = -1;
            int face_id = -1;
            EdgeKey edge_key{};
            int temporal_v0 = -1;
            int temporal_v1 = -1;
            double t0 = 0.0;
            double t1 = 0.0;
            int support_component = 0;
            double support_s0 = 0.0;
            double support_s1 = 0.0;
            std::uint8_t containment_direction = 0;
        };

        struct QueryQualityStats
        {
            std::uint64_t query_count = 0;
            std::uint64_t candidate_records = 0;
            std::uint64_t true_records_returned = 0;
            std::uint64_t spatial_rejects = 0;
            std::uint64_t time_rejects = 0;
            std::uint64_t inactive_rejects = 0;
            std::uint64_t duplicate_rejects = 0;
            std::uint64_t max_candidates_single_query = 0;
            std::uint64_t record_contains_query_candidates = 0;
            std::uint64_t query_contains_record_candidates = 0;
            std::uint64_t record_contains_query_true_records = 0;
            std::uint64_t query_contains_record_true_records = 0;
            std::uint64_t record_contains_query_spatial_rejects = 0;
            std::uint64_t query_contains_record_spatial_rejects = 0;
            std::uint64_t
                support_line_query_candidates_before_spatial_prune = 0;
            std::uint64_t
                support_line_query_candidates_after_spatial_prune = 0;
            std::uint64_t
                support_line_query_candidates_after_time_prune = 0;
            std::uint64_t support_line_interval_index_hits = 0;
            std::uint64_t support_line_interval_index_misses = 0;
            double containment_prune_seconds = 0.0;

            [[nodiscard]] double candidate_to_true_ratio() const noexcept
            {
                return true_records_returned == 0
                           ? 0.0
                           : static_cast<double>(candidate_records) /
                                 static_cast<double>(true_records_returned);
            }

            [[nodiscard]] double mean_candidates() const noexcept
            {
                return query_count == 0
                           ? 0.0
                           : static_cast<double>(candidate_records) /
                                 static_cast<double>(query_count);
            }
        };

        struct SupportLineGroupStats
        {
            std::uint64_t group_count = 0;
            std::uint64_t total_records = 0;
            std::uint64_t active_records = 0;
            std::uint64_t inactive_records = 0;
            std::uint64_t max_size = 0;
            std::uint64_t max_inactive_records = 0;
            double mean_size = 0.0;
            double tombstone_fraction = 0.0;
            double max_inactive_fraction = 0.0;
        };

        struct CompactionOptions
        {
            double global_inactive_fraction_threshold = 0.35;
            double support_line_group_inactive_fraction_threshold = 0.125;
            double edge_group_inactive_fraction_threshold = 0.125;
            double spatial_vertex_group_inactive_fraction_threshold = 0.20;
            std::size_t min_group_records_for_compaction = 128;
            std::size_t max_inactive_records_global = 100000;
            std::size_t max_inactive_records_per_support_line_group = 1024;
            std::size_t max_inactive_records_per_edge_group = 256;
            std::size_t max_inactive_records_per_spatial_vertex_group = 512;
        };

        void clear()
        {
            mesh_ = nullptr;
            records_.clear();
            record_active_.clear();
            records_by_edge_.clear();
            records_by_spatial_vertex_.clear();
            records_by_support_line_.clear();
            record_ids_by_cell_.clear();
            bulk_rebuilding_ = false;
            n_active_records_ = 0;
            n_inactive_records_ = 0;
            rebuild_count_ = 0;
            add_count_ = 0;
            remove_count_ = 0;
            remove_cell_records_touched_total_ = 0;
            compaction_count_ = 0;
            compaction_seconds_total_ = 0.0;
            support_line_group_compaction_count_ = 0;
            support_line_group_compaction_records_removed_ = 0;
            support_line_group_compaction_seconds_total_ = 0.0;
            edge_group_compaction_count_ = 0;
            edge_group_compaction_records_removed_ = 0;
            edge_group_compaction_seconds_total_ = 0.0;
            spatial_vertex_group_compaction_count_ = 0;
            spatial_vertex_group_compaction_records_removed_ = 0;
            spatial_vertex_group_compaction_seconds_total_ = 0.0;
            group_compaction_memory_before_bytes_total_ = 0;
            group_compaction_memory_after_bytes_total_ = 0;
            global_compaction_memory_before_bytes_total_ = 0;
            global_compaction_memory_after_bytes_total_ = 0;
            last_query_records_visited_ = 0;
            last_query_quality_ = {};
        }

        void set_compaction_options(const CompactionOptions& options) noexcept
        {
            compaction_options_ = options;
        }

        [[nodiscard]] const CompactionOptions&
        compaction_options() const noexcept
        {
            return compaction_options_;
        }

        void rebuild(
            const MeshType& mesh,
            const std::vector<int>& active_cells)
        {
            clear();
            mesh_ = &mesh;
            records_.reserve(active_cells.size() * 3U);
            record_active_.reserve(active_cells.size() * 3U);
            records_by_edge_.reserve(active_cells.size() * 3U);
            records_by_spatial_vertex_.reserve(active_cells.size() * 3U);
            records_by_support_line_.reserve(active_cells.size() * 3U);
            record_ids_by_cell_.reserve(active_cells.size());

            bulk_rebuilding_ = true;
            for (const int cell_id : active_cells)
                add_cell(mesh, cell_id);
            bulk_rebuilding_ = false;
            sort_all_support_line_groups_();

            ++rebuild_count_;
        }

        void add_cell(const MeshType& mesh, const int cell_id)
        {
            static_assert(GeomTraits::dim_space_v == 2,
                          "ActiveEdgeIntervalIndex2D requires dim_space_v == 2.");
            static_assert(GeomTraits::dim_time_v == 1,
                          "ActiveEdgeIntervalIndex2D requires dim_time_v == 1.");

            bind_mesh_(mesh);
            if (!mesh.valid_cell_id(cell_id))
                throw std::runtime_error(
                    "ActiveEdgeIntervalIndex2D::add_cell: invalid cell id.");

            remove_cell(mesh, cell_id);

            const auto& cell = mesh.cell(cell_id);
            const auto [t0, t1] =
                mesh::topology::temporal_interval_bounds_2d<GeomTraits>(
                    mesh,
                    cell);

            auto& record_ids = record_ids_by_cell_[cell_id];
            record_ids.reserve(static_cast<std::size_t>(cell.n_spatial_faces));

            for (int face_id = 0; face_id < cell.n_spatial_faces; ++face_id)
            {
                const auto edge_key =
                    mesh::topology::make_spatial_edge_key_2d<GeomTraits>(
                        cell.spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);

                Record record;
                record.cell_id = cell_id;
                record.face_id = face_id;
                record.edge_key = edge_key;
                record.temporal_v0 = cell.temporal_vertex_ids[0];
                record.temporal_v1 = cell.temporal_vertex_ids[1];
                record.t0 = t0;
                record.t1 = t1;

                const auto support_line_key =
                    make_support_line_key_(edge_key);
                auto& support_group =
                    records_by_support_line_[support_line_key];
                if (support_group.component < 0)
                {
                    support_group.component =
                        support_component_for_edge_(edge_key);
                    support_group.sorted = !bulk_rebuilding_;
                }
                const auto support_interval =
                    support_interval_for_edge_(
                        edge_key,
                        support_group.component);
                record.support_component = support_group.component;
                record.support_s0 = support_interval.first;
                record.support_s1 = support_interval.second;

                const RecordId record_id = records_.size();
                records_.push_back(record);
                record_active_.push_back(1);
                records_by_edge_[edge_key].push_back(record_id);
                for (const int vertex_id : edge_key.vertex_ids)
                    records_by_spatial_vertex_[vertex_id].push_back(record_id);
                insert_support_line_record_(support_group, record_id);
                record_ids.push_back(record_id);
                ++n_active_records_;
            }

            ++add_count_;
        }

        void remove_cell(const MeshType& mesh, const int cell_id)
        {
            static_assert(GeomTraits::dim_space_v == 2,
                          "ActiveEdgeIntervalIndex2D requires dim_space_v == 2.");
            static_assert(GeomTraits::dim_time_v == 1,
                          "ActiveEdgeIntervalIndex2D requires dim_time_v == 1.");

            bind_mesh_(mesh);

            const auto cell_it = record_ids_by_cell_.find(cell_id);
            if (cell_it == record_ids_by_cell_.end())
                return;

            std::size_t touched = 0;
            for (const RecordId record_id : cell_it->second)
            {
                if (record_id >= record_active_.size() ||
                    record_active_[record_id] == 0)
                {
                    continue;
                }

                record_active_[record_id] = 0;
                ++touched;
                --n_active_records_;
                ++n_inactive_records_;
            }

            remove_cell_records_touched_total_ += touched;
            record_ids_by_cell_.erase(cell_it);
            ++remove_count_;

            compact_if_needed_();
        }

        [[nodiscard]] std::vector<int> overlap_cells(
            const EdgeKey& edge_key,
            const double t0,
            const double t1) const
        {
            std::vector<int> cells;
            const auto records = overlap_records(edge_key, t0, t1);

            std::unordered_set<int> seen;
            seen.reserve(records.size());

            for (const auto& record : records)
            {
                if (seen.insert(record.cell_id).second)
                    cells.push_back(record.cell_id);
            }

            std::sort(cells.begin(), cells.end());
            return cells;
        }

        [[nodiscard]] std::vector<Record> overlap_records(
            const EdgeKey& edge_key,
            const double t0,
            const double t1) const
        {
            std::vector<Record> records;

            const auto group_it = records_by_edge_.find(edge_key);
            if (group_it == records_by_edge_.end())
            {
                last_query_records_visited_ = 0;
                last_query_quality_ = {};
                last_query_quality_.query_count = 1;
                return records;
            }

            auto& group = group_it->second;
            compact_edge_group_if_needed_(group);

            last_query_records_visited_ = group_it->second.size();
            last_query_quality_ = {};
            last_query_quality_.query_count = 1;
            last_query_quality_.candidate_records = group.size();
            last_query_quality_.max_candidates_single_query =
                group.size();

            for (const RecordId record_id : group)
            {
                if (!record_is_active_(record_id))
                {
                    ++last_query_quality_.inactive_rejects;
                    continue;
                }

                const auto& record = records_[record_id];
                if (!mesh::topology::temporal_intervals_overlap_positive_2d(
                        record.t0,
                        record.t1,
                        t0,
                        t1))
                {
                    ++last_query_quality_.time_rejects;
                    continue;
                }

                records.push_back(record);
            }
            last_query_quality_.true_records_returned = records.size();
            return records;
        }

        [[nodiscard]] std::vector<Record>
        overlap_records_containing_spatial_edge(
            const EdgeKey& query_edge,
            const double t0,
            const double t1) const
        {
            return overlap_records_with_spatial_containment_(
                query_edge,
                t0,
                t1,
                false);
        }

        [[nodiscard]] std::vector<Record>
        overlap_records_with_spatial_edge_containment(
            const EdgeKey& query_edge,
            const double t0,
            const double t1) const
        {
            return overlap_records_with_spatial_containment_(
                query_edge,
                t0,
                t1,
                true);
        }

        [[nodiscard]] std::vector<Record> overlap_records_touching_edge_vertices(
            const EdgeKey& edge_key,
            const double t0,
            const double t1) const
        {
            std::vector<Record> records;
            std::unordered_set<RecordId> seen;
            std::size_t visited = 0;
            last_query_quality_ = {};

            for (const int vertex_id : edge_key.vertex_ids)
            {
                const auto vertex_it =
                    records_by_spatial_vertex_.find(vertex_id);
                if (vertex_it == records_by_spatial_vertex_.end())
                    continue;

                auto& vertex_group = vertex_it->second;
                compact_spatial_vertex_group_if_needed_(vertex_group);

                visited += vertex_group.size();
                for (const RecordId record_id : vertex_group)
                {
                    if (!seen.insert(record_id).second)
                    {
                        ++last_query_quality_.duplicate_rejects;
                        continue;
                    }

                    if (!record_is_active_(record_id))
                    {
                        ++last_query_quality_.inactive_rejects;
                        continue;
                    }

                    const auto& record = records_[record_id];
                    if (!mesh::topology::temporal_intervals_overlap_positive_2d(
                            record.t0,
                            record.t1,
                            t0,
                            t1))
                    {
                        ++last_query_quality_.time_rejects;
                        continue;
                    }

                    records.push_back(record);
                }
            }

            last_query_records_visited_ = visited;
            last_query_quality_.query_count = 1;
            last_query_quality_.candidate_records = visited;
            last_query_quality_.true_records_returned = records.size();
            last_query_quality_.max_candidates_single_query = visited;
            return records;
        }

        [[nodiscard]] std::size_t n_records() const noexcept
        {
            return n_active_records_;
        }

        [[nodiscard]] std::size_t n_active_records() const noexcept
        {
            return n_active_records_;
        }

        [[nodiscard]] std::size_t n_inactive_records() const noexcept
        {
            return n_inactive_records_;
        }

        [[nodiscard]] std::size_t n_edge_groups() const noexcept
        {
            return records_by_edge_.size();
        }

        [[nodiscard]] std::uint64_t rebuild_count() const noexcept
        {
            return rebuild_count_;
        }

        [[nodiscard]] std::uint64_t add_count() const noexcept
        {
            return add_count_;
        }

        [[nodiscard]] std::uint64_t remove_count() const noexcept
        {
            return remove_count_;
        }

        [[nodiscard]] std::uint64_t
        remove_cell_records_touched_total() const noexcept
        {
            return remove_cell_records_touched_total_;
        }

        [[nodiscard]] std::uint64_t compaction_count() const noexcept
        {
            return compaction_count_;
        }

        [[nodiscard]] double compaction_seconds_total() const noexcept
        {
            return compaction_seconds_total_;
        }

        [[nodiscard]] std::uint64_t
        support_line_group_compaction_count() const noexcept
        {
            return support_line_group_compaction_count_;
        }

        [[nodiscard]] std::uint64_t
        support_line_group_compaction_records_removed() const noexcept
        {
            return support_line_group_compaction_records_removed_;
        }

        [[nodiscard]] double
        support_line_group_compaction_seconds_total() const noexcept
        {
            return support_line_group_compaction_seconds_total_;
        }

        [[nodiscard]] std::uint64_t edge_group_compaction_count() const noexcept
        {
            return edge_group_compaction_count_;
        }

        [[nodiscard]] std::uint64_t
        edge_group_compaction_records_removed() const noexcept
        {
            return edge_group_compaction_records_removed_;
        }

        [[nodiscard]] double edge_group_compaction_seconds_total() const noexcept
        {
            return edge_group_compaction_seconds_total_;
        }

        [[nodiscard]] std::uint64_t
        spatial_vertex_group_compaction_count() const noexcept
        {
            return spatial_vertex_group_compaction_count_;
        }

        [[nodiscard]] std::uint64_t
        spatial_vertex_group_compaction_records_removed() const noexcept
        {
            return spatial_vertex_group_compaction_records_removed_;
        }

        [[nodiscard]] double
        spatial_vertex_group_compaction_seconds_total() const noexcept
        {
            return spatial_vertex_group_compaction_seconds_total_;
        }

        [[nodiscard]] std::uint64_t
        group_compaction_memory_before_bytes_total() const noexcept
        {
            return group_compaction_memory_before_bytes_total_;
        }

        [[nodiscard]] std::uint64_t
        group_compaction_memory_after_bytes_total() const noexcept
        {
            return group_compaction_memory_after_bytes_total_;
        }

        [[nodiscard]] std::uint64_t
        global_compaction_memory_before_bytes_total() const noexcept
        {
            return global_compaction_memory_before_bytes_total_;
        }

        [[nodiscard]] std::uint64_t
        global_compaction_memory_after_bytes_total() const noexcept
        {
            return global_compaction_memory_after_bytes_total_;
        }

        [[nodiscard]] std::size_t last_query_records_visited() const noexcept
        {
            return last_query_records_visited_;
        }

        [[nodiscard]] const QueryQualityStats& last_query_quality() const noexcept
        {
            return last_query_quality_;
        }

        [[nodiscard]] SupportLineGroupStats support_line_group_stats() const
        {
            SupportLineGroupStats stats;
            stats.group_count = records_by_support_line_.size();
            if (records_by_support_line_.empty())
                return stats;

            for (const auto& [key, group] : records_by_support_line_)
            {
                (void)key;
                const auto size =
                    static_cast<std::uint64_t>(group.by_min.size());
                std::uint64_t inactive = 0;
                stats.max_size = std::max(stats.max_size, size);
                for (const RecordId record_id : group.by_min)
                {
                    if (!record_is_active_(record_id))
                        ++inactive;
                }
                stats.total_records += size;
                stats.inactive_records += inactive;
                stats.max_inactive_records =
                    std::max(stats.max_inactive_records, inactive);
                const double inactive_fraction =
                    size == 0
                        ? 0.0
                        : static_cast<double>(inactive) /
                              static_cast<double>(size);
                stats.max_inactive_fraction =
                    std::max(stats.max_inactive_fraction, inactive_fraction);
            }

            stats.active_records =
                stats.total_records - stats.inactive_records;
            stats.mean_size =
                static_cast<double>(stats.total_records) /
                static_cast<double>(stats.group_count);
            stats.tombstone_fraction =
                stats.total_records == 0
                    ? 0.0
                    : static_cast<double>(stats.inactive_records) /
                          static_cast<double>(stats.total_records);
            return stats;
        }

        [[nodiscard]] SupportLineGroupStats edge_group_stats() const
        {
            return id_vector_group_stats_(records_by_edge_);
        }

        [[nodiscard]] SupportLineGroupStats spatial_vertex_group_stats() const
        {
            return id_vector_group_stats_(records_by_spatial_vertex_);
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            std::size_t bytes = 0;
            bytes += records_.capacity() * sizeof(Record);
            bytes += record_active_.capacity() * sizeof(char);

            const auto vector_payload_bytes =
                [](const auto& map) -> std::size_t
                {
                    std::size_t total = 0;
                    for (const auto& [key, ids] : map)
                    {
                        (void)key;
                        total += ids.capacity() * sizeof(RecordId);
                    }
                    return total;
                };

            bytes += records_by_edge_.bucket_count() * sizeof(void*);
            bytes += records_by_edge_.size() *
                     (sizeof(EdgeKey) + sizeof(std::vector<RecordId>) +
                      3U * sizeof(void*));
            bytes += vector_payload_bytes(records_by_edge_);

            bytes += records_by_spatial_vertex_.bucket_count() * sizeof(void*);
            bytes += records_by_spatial_vertex_.size() *
                     (sizeof(int) + sizeof(std::vector<RecordId>) +
                      3U * sizeof(void*));
            bytes += vector_payload_bytes(records_by_spatial_vertex_);

            bytes += records_by_support_line_.bucket_count() * sizeof(void*);
            bytes += records_by_support_line_.size() *
                     (sizeof(SupportLineKey) + sizeof(SupportLineGroup) +
                      3U * sizeof(void*));
            for (const auto& [key, group] : records_by_support_line_)
            {
                (void)key;
                bytes += group.by_min.capacity() * sizeof(RecordId);
                bytes += group.by_max.capacity() * sizeof(RecordId);
                bytes += group.interval_tree.capacity() *
                         sizeof(typename SupportLineGroup::IntervalTreeNode);
                for (const auto& node : group.interval_tree)
                {
                    bytes += node.spanning_by_min.capacity() *
                             sizeof(RecordId);
                    bytes += node.spanning_by_max_asc.capacity() *
                             sizeof(RecordId);
                    bytes += node.spanning_by_max_desc.capacity() *
                             sizeof(RecordId);
                    bytes += node.spanning_by_t0_asc.capacity() *
                             sizeof(RecordId);
                    bytes += node.spanning_by_t1_desc.capacity() *
                             sizeof(RecordId);
                }
            }

            bytes += record_ids_by_cell_.bucket_count() * sizeof(void*);
            bytes += record_ids_by_cell_.size() *
                     (sizeof(int) + sizeof(std::vector<RecordId>) +
                      3U * sizeof(void*));
            bytes += vector_payload_bytes(record_ids_by_cell_);
            return bytes;
        }

    private:
        using Clock = std::chrono::steady_clock;

        struct SupportLineKey
        {
            long long a = 0;
            long long b = 0;
            long long c = 0;

            bool operator==(const SupportLineKey&) const noexcept = default;
        };

        struct SupportLineKeyHash
        {
            [[nodiscard]] std::size_t operator()(
                const SupportLineKey& key) const noexcept
            {
                std::size_t seed = 0;
                core::hash_combine(seed, key.a);
                core::hash_combine(seed, key.b);
                core::hash_combine(seed, key.c);
                return seed;
            }
        };

        struct SupportLineGroup
        {
            struct IntervalTreeNode
            {
                double center = 0.0;
                std::vector<RecordId> spanning_by_min{};
                std::vector<RecordId> spanning_by_max_asc{};
                std::vector<RecordId> spanning_by_max_desc{};
                std::vector<RecordId> spanning_by_t0_asc{};
                std::vector<RecordId> spanning_by_t1_desc{};
                int left = -1;
                int right = -1;
            };

            std::vector<RecordId> by_min{};
            std::vector<RecordId> by_max{};
            mutable std::vector<IntervalTreeNode> interval_tree{};
            int component = -1;
            bool sorted = false;
            mutable bool interval_tree_current = false;
        };

        struct RecordIdRange
        {
            const std::vector<RecordId>* ids = nullptr;
            std::size_t begin = 0;
            std::size_t end = 0;

            [[nodiscard]] std::size_t size() const noexcept
            {
                return end >= begin ? end - begin : 0;
            }
        };

        [[nodiscard]] static RecordIdRange whole_range_(
            const std::vector<RecordId>& ids) noexcept
        {
            return RecordIdRange{&ids, 0U, ids.size()};
        }

        [[nodiscard]] static RecordIdRange prefix_range_(
            const std::vector<RecordId>& ids,
            const typename std::vector<RecordId>::const_iterator end_it)
        {
            return RecordIdRange{
                &ids,
                0U,
                static_cast<std::size_t>(std::distance(ids.begin(), end_it))
            };
        }

        [[nodiscard]] static RecordIdRange suffix_range_(
            const std::vector<RecordId>& ids,
            const typename std::vector<RecordId>::const_iterator begin_it)
        {
            return RecordIdRange{
                &ids,
                static_cast<std::size_t>(
                    std::distance(ids.begin(), begin_it)),
                ids.size()
            };
        }

        [[nodiscard]] static RecordIdRange smaller_range_(
            const RecordIdRange a,
            const RecordIdRange b) noexcept
        {
            return a.size() <= b.size() ? a : b;
        }

        template<typename Visitor>
        static void visit_record_id_range_(
            const RecordIdRange range,
            const bool record_must_contain_query,
            Visitor& visitor)
        {
            if (range.ids == nullptr)
                return;
            for (std::size_t i = range.begin; i < range.end; ++i)
                visitor((*range.ids)[i], record_must_contain_query);
        }

        template<typename MapType>
        [[nodiscard]] SupportLineGroupStats id_vector_group_stats_(
            const MapType& groups) const
        {
            SupportLineGroupStats stats;
            stats.group_count = groups.size();
            if (groups.empty())
                return stats;

            for (const auto& [key, ids] : groups)
            {
                (void)key;
                const auto size =
                    static_cast<std::uint64_t>(ids.size());
                std::uint64_t inactive = 0;
                for (const RecordId record_id : ids)
                {
                    if (!record_is_active_(record_id))
                        ++inactive;
                }
                stats.total_records += size;
                stats.inactive_records += inactive;
                stats.max_size = std::max(stats.max_size, size);
                stats.max_inactive_records =
                    std::max(stats.max_inactive_records, inactive);
                const double inactive_fraction =
                    size == 0
                        ? 0.0
                        : static_cast<double>(inactive) /
                              static_cast<double>(size);
                stats.max_inactive_fraction =
                    std::max(stats.max_inactive_fraction, inactive_fraction);
            }

            stats.active_records =
                stats.total_records - stats.inactive_records;
            stats.mean_size =
                static_cast<double>(stats.total_records) /
                static_cast<double>(stats.group_count);
            stats.tombstone_fraction =
                stats.total_records == 0
                    ? 0.0
                    : static_cast<double>(stats.inactive_records) /
                          static_cast<double>(stats.total_records);
            return stats;
        }

        [[nodiscard]] bool group_should_compact_(
            const std::size_t total,
            const std::size_t inactive,
            const double inactive_fraction_threshold,
            const std::size_t max_inactive_records) const noexcept
        {
            if (inactive == 0)
                return false;
            if (inactive >= max_inactive_records)
                return true;
            if (total < compaction_options_.min_group_records_for_compaction)
                return false;
            return static_cast<double>(inactive) /
                       static_cast<double>(total) >=
                   inactive_fraction_threshold;
        }

        [[nodiscard]] static std::uint64_t
        record_id_vector_memory_bytes_(const std::vector<RecordId>& ids) noexcept
        {
            return static_cast<std::uint64_t>(
                ids.capacity() * sizeof(RecordId));
        }

        [[nodiscard]] static std::uint64_t
        support_line_group_memory_bytes_(
            const SupportLineGroup& group) noexcept
        {
            std::uint64_t bytes = 0;
            bytes += record_id_vector_memory_bytes_(group.by_min);
            bytes += record_id_vector_memory_bytes_(group.by_max);
            bytes += static_cast<std::uint64_t>(
                group.interval_tree.capacity() *
                sizeof(typename SupportLineGroup::IntervalTreeNode));
            for (const auto& node : group.interval_tree)
            {
                bytes += record_id_vector_memory_bytes_(node.spanning_by_min);
                bytes += record_id_vector_memory_bytes_(
                    node.spanning_by_max_asc);
                bytes += record_id_vector_memory_bytes_(
                    node.spanning_by_max_desc);
                bytes += record_id_vector_memory_bytes_(
                    node.spanning_by_t0_asc);
                bytes += record_id_vector_memory_bytes_(
                    node.spanning_by_t1_desc);
            }
            return bytes;
        }

        [[nodiscard]] std::size_t inactive_record_count_(
            const std::vector<RecordId>& ids) const noexcept
        {
            std::size_t inactive = 0;
            for (const RecordId record_id : ids)
            {
                if (!record_is_active_(record_id))
                    ++inactive;
            }
            return inactive;
        }

        void compact_record_id_vector_if_needed_(
            std::vector<RecordId>& ids,
            const double inactive_fraction_threshold,
            const std::size_t max_inactive_records,
            std::uint64_t& compaction_count,
            std::uint64_t& records_removed,
            double& seconds_total) const
        {
            const std::size_t total = ids.size();
            const std::size_t inactive = inactive_record_count_(ids);
            if (!group_should_compact_(
                    total,
                    inactive,
                    inactive_fraction_threshold,
                    max_inactive_records))
            {
                return;
            }

            const auto memory_before = record_id_vector_memory_bytes_(ids);
            const auto start = Clock::now();
            ids.erase(
                std::remove_if(
                    ids.begin(),
                    ids.end(),
                    [&](const RecordId record_id)
                    {
                        return !record_is_active_(record_id);
                    }),
                ids.end());
            ids.shrink_to_fit();
            const double seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            const auto memory_after = record_id_vector_memory_bytes_(ids);

            ++compaction_count;
            records_removed += inactive;
            seconds_total += seconds;
            ++compaction_count_;
            compaction_seconds_total_ += seconds;
            group_compaction_memory_before_bytes_total_ += memory_before;
            group_compaction_memory_after_bytes_total_ += memory_after;
        }

        void compact_edge_group_if_needed_(
            std::vector<RecordId>& ids) const
        {
            compact_record_id_vector_if_needed_(
                ids,
                compaction_options_.edge_group_inactive_fraction_threshold,
                compaction_options_.max_inactive_records_per_edge_group,
                edge_group_compaction_count_,
                edge_group_compaction_records_removed_,
                edge_group_compaction_seconds_total_);
        }

        void compact_spatial_vertex_group_if_needed_(
            std::vector<RecordId>& ids) const
        {
            compact_record_id_vector_if_needed_(
                ids,
                compaction_options_
                    .spatial_vertex_group_inactive_fraction_threshold,
                compaction_options_
                    .max_inactive_records_per_spatial_vertex_group,
                spatial_vertex_group_compaction_count_,
                spatial_vertex_group_compaction_records_removed_,
                spatial_vertex_group_compaction_seconds_total_);
        }

        [[nodiscard]] double record_support_min_(
            const RecordId record_id) const noexcept
        {
            return records_[record_id].support_s0;
        }

        [[nodiscard]] double record_support_max_(
            const RecordId record_id) const noexcept
        {
            return records_[record_id].support_s1;
        }

        [[nodiscard]] double record_temporal_min_(
            const RecordId record_id) const noexcept
        {
            return records_[record_id].t0;
        }

        [[nodiscard]] double record_temporal_max_(
            const RecordId record_id) const noexcept
        {
            return records_[record_id].t1;
        }

        [[nodiscard]] int build_support_line_interval_tree_(
            SupportLineGroup& group,
            std::vector<RecordId> record_ids) const
        {
            if (record_ids.empty())
                return -1;

            std::vector<double> midpoints;
            midpoints.reserve(record_ids.size());
            for (const RecordId record_id : record_ids)
                midpoints.push_back(
                    0.5 *
                    (record_support_min_(record_id) +
                     record_support_max_(record_id)));

            auto median_it =
                midpoints.begin() +
                static_cast<std::ptrdiff_t>(midpoints.size() / 2U);
            std::nth_element(
                midpoints.begin(),
                median_it,
                midpoints.end());
            const double center = *median_it;

            constexpr double interval_tol = 1.0e-12;
            std::vector<RecordId> left;
            std::vector<RecordId> right;
            typename SupportLineGroup::IntervalTreeNode node;
            node.center = center;
            left.reserve(record_ids.size() / 2U);
            right.reserve(record_ids.size() / 2U);
            node.spanning_by_min.reserve(record_ids.size());

            for (const RecordId record_id : record_ids)
            {
                const double s0 = record_support_min_(record_id);
                const double s1 = record_support_max_(record_id);
                if (s1 < center - interval_tol)
                    left.push_back(record_id);
                else if (s0 > center + interval_tol)
                    right.push_back(record_id);
                else
                    node.spanning_by_min.push_back(record_id);
            }

            if (node.spanning_by_min.empty())
            {
                const RecordId pivot = record_ids[record_ids.size() / 2U];
                node.spanning_by_min.push_back(pivot);
                left.clear();
                right.clear();
                for (const RecordId record_id : record_ids)
                {
                    if (record_id == pivot)
                        continue;
                    if (record_support_max_(record_id) < center)
                        left.push_back(record_id);
                    else
                        right.push_back(record_id);
                }
            }

            const auto by_min_less =
                [&](const RecordId a, const RecordId b)
                {
                    const double a_min = record_support_min_(a);
                    const double b_min = record_support_min_(b);
                    if (a_min != b_min)
                        return a_min < b_min;
                    const double a_max = record_support_max_(a);
                    const double b_max = record_support_max_(b);
                    if (a_max != b_max)
                        return a_max < b_max;
                    return a < b;
                };
            const auto by_max_asc_less =
                [&](const RecordId a, const RecordId b)
                {
                    const double a_max = record_support_max_(a);
                    const double b_max = record_support_max_(b);
                    if (a_max != b_max)
                        return a_max < b_max;
                    const double a_min = record_support_min_(a);
                    const double b_min = record_support_min_(b);
                    if (a_min != b_min)
                        return a_min < b_min;
                    return a < b;
                };
            const auto by_max_desc_less =
                [&](const RecordId a, const RecordId b)
                {
                    const double a_max = record_support_max_(a);
                    const double b_max = record_support_max_(b);
                    if (a_max != b_max)
                        return a_max > b_max;
                    const double a_min = record_support_min_(a);
                    const double b_min = record_support_min_(b);
                    if (a_min != b_min)
                        return a_min < b_min;
                    return a < b;
                };
            const auto by_t0_asc_less =
                [&](const RecordId a, const RecordId b)
                {
                    const double a_t0 = record_temporal_min_(a);
                    const double b_t0 = record_temporal_min_(b);
                    if (a_t0 != b_t0)
                        return a_t0 < b_t0;
                    const double a_t1 = record_temporal_max_(a);
                    const double b_t1 = record_temporal_max_(b);
                    if (a_t1 != b_t1)
                        return a_t1 < b_t1;
                    return a < b;
                };
            const auto by_t1_desc_less =
                [&](const RecordId a, const RecordId b)
                {
                    const double a_t1 = record_temporal_max_(a);
                    const double b_t1 = record_temporal_max_(b);
                    if (a_t1 != b_t1)
                        return a_t1 > b_t1;
                    const double a_t0 = record_temporal_min_(a);
                    const double b_t0 = record_temporal_min_(b);
                    if (a_t0 != b_t0)
                        return a_t0 < b_t0;
                    return a < b;
                };

            std::sort(
                node.spanning_by_min.begin(),
                node.spanning_by_min.end(),
                by_min_less);
            node.spanning_by_max_asc = node.spanning_by_min;
            node.spanning_by_max_desc = node.spanning_by_min;
            node.spanning_by_t0_asc = node.spanning_by_min;
            node.spanning_by_t1_desc = node.spanning_by_min;
            std::sort(
                node.spanning_by_max_asc.begin(),
                node.spanning_by_max_asc.end(),
                by_max_asc_less);
            std::sort(
                node.spanning_by_max_desc.begin(),
                node.spanning_by_max_desc.end(),
                by_max_desc_less);
            std::sort(
                node.spanning_by_t0_asc.begin(),
                node.spanning_by_t0_asc.end(),
                by_t0_asc_less);
            std::sort(
                node.spanning_by_t1_desc.begin(),
                node.spanning_by_t1_desc.end(),
                by_t1_desc_less);

            const int node_id =
                static_cast<int>(group.interval_tree.size());
            group.interval_tree.push_back(std::move(node));

            group.interval_tree[static_cast<std::size_t>(node_id)].left =
                build_support_line_interval_tree_(group, std::move(left));
            group.interval_tree[static_cast<std::size_t>(node_id)].right =
                build_support_line_interval_tree_(group, std::move(right));
            return node_id;
        }

        void ensure_support_line_group_interval_tree_(
            SupportLineGroup& group) const
        {
            if (group.interval_tree_current)
                return;

            group.interval_tree.clear();
            group.interval_tree.reserve(group.by_min.size());
            const int root_id =
                build_support_line_interval_tree_(group, group.by_min);
            (void)root_id;
            group.interval_tree_current = true;
        }

        template<typename Visitor>
        void query_record_contains_interval_tree_(
            const SupportLineGroup& group,
            const int node_id,
            const double query_min,
            const double query_max,
            const double query_t0,
            const double query_t1,
            Visitor& visitor,
            QueryQualityStats& quality) const
        {
            if (node_id < 0)
                return;

            constexpr double interval_tol = 1.0e-12;
            const auto& node =
                group.interval_tree[static_cast<std::size_t>(node_id)];

            const auto visit_by_min_prefix =
                [&]() -> RecordIdRange
                {
                    const auto end =
                        std::partition_point(
                            node.spanning_by_min.begin(),
                            node.spanning_by_min.end(),
                            [&](const RecordId record_id)
                            {
                                return record_support_min_(record_id) <=
                                       query_min + interval_tol;
                            });
                    return prefix_range_(node.spanning_by_min, end);
                };
            const auto visit_by_max_prefix =
                [&]() -> RecordIdRange
                {
                    const auto end =
                        std::partition_point(
                            node.spanning_by_max_desc.begin(),
                            node.spanning_by_max_desc.end(),
                            [&](const RecordId record_id)
                            {
                                return record_support_max_(record_id) >=
                                       query_max - interval_tol;
                            });
                    return prefix_range_(node.spanning_by_max_desc, end);
                };
            const auto temporal_t0_prefix =
                [&]() -> RecordIdRange
                {
                    const auto end =
                        std::partition_point(
                            node.spanning_by_t0_asc.begin(),
                            node.spanning_by_t0_asc.end(),
                            [&](const RecordId record_id)
                            {
                                return record_temporal_min_(record_id) <
                                       query_t1 - interval_tol;
                            });
                    return prefix_range_(node.spanning_by_t0_asc, end);
                };
            const auto temporal_t1_prefix =
                [&]() -> RecordIdRange
                {
                    const auto end =
                        std::partition_point(
                            node.spanning_by_t1_desc.begin(),
                            node.spanning_by_t1_desc.end(),
                            [&](const RecordId record_id)
                            {
                                return record_temporal_max_(record_id) >
                                       query_t0 + interval_tol;
                            });
                    return prefix_range_(node.spanning_by_t1_desc, end);
                };
            const auto visit_pruned =
                [&](const RecordIdRange spatial_range)
                {
                    quality
                        .support_line_query_candidates_before_spatial_prune +=
                        node.spanning_by_min.size();
                    quality
                        .support_line_query_candidates_after_spatial_prune +=
                        spatial_range.size();
                    const RecordIdRange temporal_range =
                        smaller_range_(
                            temporal_t0_prefix(),
                            temporal_t1_prefix());
                    const RecordIdRange pruned_range =
                        smaller_range_(spatial_range, temporal_range);
                    quality
                        .support_line_query_candidates_after_time_prune +=
                        pruned_range.size();
                    visit_record_id_range_(pruned_range, true, visitor);
                };

            if (query_max < node.center - interval_tol)
            {
                visit_pruned(visit_by_min_prefix());
                query_record_contains_interval_tree_(
                    group,
                    node.left,
                    query_min,
                    query_max,
                    query_t0,
                    query_t1,
                    visitor,
                    quality);
                return;
            }
            if (query_min > node.center + interval_tol)
            {
                visit_pruned(visit_by_max_prefix());
                query_record_contains_interval_tree_(
                    group,
                    node.right,
                    query_min,
                    query_max,
                    query_t0,
                    query_t1,
                    visitor,
                    quality);
                return;
            }

            const auto by_min_end =
                std::partition_point(
                    node.spanning_by_min.begin(),
                    node.spanning_by_min.end(),
                    [&](const RecordId record_id)
                    {
                        return record_support_min_(record_id) <=
                               query_min + interval_tol;
                    });
            const auto by_max_end =
                std::partition_point(
                    node.spanning_by_max_desc.begin(),
                    node.spanning_by_max_desc.end(),
                    [&](const RecordId record_id)
                    {
                        return record_support_max_(record_id) >=
                               query_max - interval_tol;
                    });
            const auto by_min_count =
                static_cast<std::size_t>(
                    std::distance(node.spanning_by_min.begin(), by_min_end));
            const auto by_max_count =
                static_cast<std::size_t>(
                    std::distance(
                        node.spanning_by_max_desc.begin(),
                        by_max_end));

            const RecordIdRange spatial_range =
                by_min_count <= by_max_count
                    ? prefix_range_(node.spanning_by_min, by_min_end)
                    : prefix_range_(node.spanning_by_max_desc, by_max_end);
            visit_pruned(spatial_range);
        }

        template<typename Visitor>
        void query_interval_contains_record_tree_(
            const SupportLineGroup& group,
            const int node_id,
            const double query_min,
            const double query_max,
            const double query_t0,
            const double query_t1,
            Visitor& visitor,
            QueryQualityStats& quality) const
        {
            if (node_id < 0)
                return;

            constexpr double interval_tol = 1.0e-12;
            const auto& node =
                group.interval_tree[static_cast<std::size_t>(node_id)];

            if (query_max < node.center - interval_tol)
            {
                query_interval_contains_record_tree_(
                    group,
                    node.left,
                    query_min,
                    query_max,
                    query_t0,
                    query_t1,
                    visitor,
                    quality);
                return;
            }
            if (query_min > node.center + interval_tol)
            {
                query_interval_contains_record_tree_(
                    group,
                    node.right,
                    query_min,
                    query_max,
                    query_t0,
                    query_t1,
                    visitor,
                    quality);
                return;
            }

            const auto by_min_begin =
                std::lower_bound(
                    node.spanning_by_min.begin(),
                    node.spanning_by_min.end(),
                    query_min - interval_tol,
                    [&](const RecordId record_id, const double value)
                    {
                        return record_support_min_(record_id) < value;
                    });
            const auto by_max_end =
                std::partition_point(
                    node.spanning_by_max_asc.begin(),
                    node.spanning_by_max_asc.end(),
                    [&](const RecordId record_id)
                    {
                        return record_support_max_(record_id) <=
                               query_max + interval_tol;
                    });
            const auto by_min_count =
                static_cast<std::size_t>(
                    std::distance(by_min_begin, node.spanning_by_min.end()));
            const auto by_max_count =
                static_cast<std::size_t>(
                    std::distance(
                        node.spanning_by_max_asc.begin(),
                        by_max_end));
            const auto temporal_t0_end =
                std::partition_point(
                    node.spanning_by_t0_asc.begin(),
                    node.spanning_by_t0_asc.end(),
                    [&](const RecordId record_id)
                    {
                        return record_temporal_min_(record_id) <
                               query_t1 - interval_tol;
                    });
            const auto temporal_t1_end =
                std::partition_point(
                    node.spanning_by_t1_desc.begin(),
                    node.spanning_by_t1_desc.end(),
                    [&](const RecordId record_id)
                    {
                        return record_temporal_max_(record_id) >
                               query_t0 + interval_tol;
                    });

            const RecordIdRange spatial_range =
                by_min_count <= by_max_count
                    ? suffix_range_(node.spanning_by_min, by_min_begin)
                    : prefix_range_(node.spanning_by_max_asc, by_max_end);
            const RecordIdRange temporal_range =
                smaller_range_(
                    prefix_range_(node.spanning_by_t0_asc, temporal_t0_end),
                    prefix_range_(node.spanning_by_t1_desc, temporal_t1_end));
            const RecordIdRange pruned_range =
                smaller_range_(spatial_range, temporal_range);
            quality.support_line_query_candidates_before_spatial_prune +=
                node.spanning_by_min.size();
            quality.support_line_query_candidates_after_spatial_prune +=
                spatial_range.size();
            quality.support_line_query_candidates_after_time_prune +=
                pruned_range.size();
            visit_record_id_range_(pruned_range, false, visitor);

            query_interval_contains_record_tree_(
                group,
                node.left,
                query_min,
                query_max,
                query_t0,
                query_t1,
                visitor,
                quality);
            query_interval_contains_record_tree_(
                group,
                node.right,
                query_min,
                query_max,
                query_t0,
                query_t1,
                visitor,
                quality);
        }

        [[nodiscard]] std::vector<Record>
        overlap_records_with_spatial_containment_(
            const EdgeKey& query_edge,
            const double t0,
            const double t1,
            const bool bidirectional) const
        {
            std::vector<Record> records;
            if (mesh_ == nullptr)
            {
                last_query_records_visited_ = 0;
                last_query_quality_ = {};
                last_query_quality_.query_count = 1;
                return records;
            }

            const auto line_key = make_support_line_key_(query_edge);
            const auto group_it = records_by_support_line_.find(line_key);
            if (group_it == records_by_support_line_.end())
            {
                last_query_records_visited_ = 0;
                last_query_quality_ = {};
                last_query_quality_.query_count = 1;
                return records;
            }

            auto& group = group_it->second;
            compact_support_line_group_if_needed_(group);
            ensure_support_line_group_sorted_(group);

            if (group.by_min.empty())
            {
                last_query_records_visited_ = 0;
                last_query_quality_ = {};
                last_query_quality_.query_count = 1;
                return records;
            }

            const int component =
                group.component >= 0
                    ? group.component
                    : support_component_for_edge_(query_edge);
            const auto query_interval =
                support_interval_for_edge_(query_edge, component);
            const double query_min = query_interval.first;
            const double query_max = query_interval.second;
            constexpr double interval_tol = 1.0e-12;
            constexpr std::size_t interval_tree_min_group_size = 1024U;
            const bool use_interval_tree =
                group.by_min.size() >= interval_tree_min_group_size;
            if (use_interval_tree)
                ensure_support_line_group_interval_tree_(group);

            std::size_t visited = 0;
            std::vector<RecordId> seen;
            if (bidirectional)
                seen.reserve(16);
            QueryQualityStats quality;
            quality.query_count = 1;
            if (use_interval_tree)
                ++quality.support_line_interval_index_hits;
            else
                ++quality.support_line_interval_index_misses;

            const auto add_record_if_valid =
                [&](const RecordId record_id, const bool record_must_contain_query)
                {
                    ++visited;
                    if (record_must_contain_query)
                        ++quality.record_contains_query_candidates;
                    else
                        ++quality.query_contains_record_candidates;
                    if (!record_is_active_(record_id))
                    {
                        ++quality.inactive_rejects;
                        return;
                    }

                    const auto& record = records_[record_id];
                    if (record.edge_key == query_edge)
                    {
                        ++quality.duplicate_rejects;
                        return;
                    }

                    if (record_must_contain_query)
                    {
                        if (record.support_s0 > query_min + interval_tol ||
                            record.support_s1 < query_max - interval_tol)
                        {
                            ++quality.spatial_rejects;
                            ++quality.record_contains_query_spatial_rejects;
                            return;
                        }
                    }
                    else
                    {
                        if (query_min > record.support_s0 + interval_tol ||
                            query_max < record.support_s1 - interval_tol)
                        {
                            ++quality.spatial_rejects;
                            ++quality.query_contains_record_spatial_rejects;
                            return;
                        }
                    }

                    if (!mesh::topology::temporal_intervals_overlap_positive_2d(
                            record.t0,
                            record.t1,
                            t0,
                            t1))
                    {
                        ++quality.time_rejects;
                        return;
                    }

                    const auto& spatial_vertices = mesh_->spatial_vertices();
                    const bool contains =
                        record_must_contain_query
                            ? mesh::topology::spatial_edge_contains_edge_2d<
                                  GeomTraits>(
                                  record.edge_key.vertex_ids,
                                  query_edge.vertex_ids,
                                  spatial_vertices)
                            : mesh::topology::spatial_edge_contains_edge_2d<
                                  GeomTraits>(
                                  query_edge.vertex_ids,
                                  record.edge_key.vertex_ids,
                                  spatial_vertices);
                    if (!contains)
                    {
                        ++quality.spatial_rejects;
                        if (record_must_contain_query)
                            ++quality.record_contains_query_spatial_rejects;
                        else
                            ++quality.query_contains_record_spatial_rejects;
                        return;
                    }

                    if (bidirectional)
                    {
                        if (std::find(seen.begin(), seen.end(), record_id) !=
                            seen.end())
                        {
                            ++quality.duplicate_rejects;
                            return;
                        }
                        seen.push_back(record_id);
                    }
                    Record returned_record = record;
                    returned_record.containment_direction =
                        record_must_contain_query ? 1U : 2U;
                    records.push_back(returned_record);
                    if (record_must_contain_query)
                        ++quality.record_contains_query_true_records;
                    else
                        ++quality.query_contains_record_true_records;
                };

            const auto visit_record_contains_query_sorted =
                [&]()
                {
                    const auto prefix_end =
                        std::partition_point(
                            group.by_min.begin(),
                            group.by_min.end(),
                            [&](const RecordId record_id)
                            {
                                return record_support_min_(record_id) <=
                                       query_min + interval_tol;
                            });
                    const auto suffix_begin =
                        std::lower_bound(
                            group.by_max.begin(),
                            group.by_max.end(),
                            query_max - interval_tol,
                            [&](const RecordId record_id, const double value)
                            {
                                return record_support_max_(record_id) < value;
                            });
                    const auto prefix_count =
                        static_cast<std::size_t>(
                            std::distance(group.by_min.begin(), prefix_end));
                    const auto suffix_count =
                        static_cast<std::size_t>(
                            std::distance(suffix_begin, group.by_max.end()));
                    const RecordIdRange spatial_range =
                        prefix_count <= suffix_count
                            ? prefix_range_(group.by_min, prefix_end)
                            : suffix_range_(group.by_max, suffix_begin);
                    quality
                        .support_line_query_candidates_before_spatial_prune +=
                        group.by_min.size();
                    quality
                        .support_line_query_candidates_after_spatial_prune +=
                        spatial_range.size();
                    quality
                        .support_line_query_candidates_after_time_prune +=
                        spatial_range.size();
                    visit_record_id_range_(
                        spatial_range,
                        true,
                        add_record_if_valid);
                };

            const auto visit_query_contains_record_sorted =
                [&]()
                {
                    const auto suffix_begin =
                        std::lower_bound(
                            group.by_min.begin(),
                            group.by_min.end(),
                            query_min - interval_tol,
                            [&](const RecordId record_id, const double value)
                            {
                                return record_support_min_(record_id) < value;
                            });
                    const auto prefix_end =
                        std::upper_bound(
                            group.by_max.begin(),
                            group.by_max.end(),
                            query_max + interval_tol,
                            [&](const double value, const RecordId record_id)
                            {
                                return value < record_support_max_(record_id);
                            });
                    const auto suffix_count =
                        static_cast<std::size_t>(
                            std::distance(suffix_begin, group.by_min.end()));
                    const auto prefix_count =
                        static_cast<std::size_t>(
                            std::distance(group.by_max.begin(), prefix_end));
                    const RecordIdRange spatial_range =
                        suffix_count <= prefix_count
                            ? suffix_range_(group.by_min, suffix_begin)
                            : prefix_range_(group.by_max, prefix_end);
                    quality
                        .support_line_query_candidates_before_spatial_prune +=
                        group.by_min.size();
                    quality
                        .support_line_query_candidates_after_spatial_prune +=
                        spatial_range.size();
                    quality
                        .support_line_query_candidates_after_time_prune +=
                        spatial_range.size();
                    visit_record_id_range_(
                        spatial_range,
                        false,
                        add_record_if_valid);
                };

            const auto prune_start = Clock::now();
            if (use_interval_tree)
            {
                query_record_contains_interval_tree_(
                    group,
                    group.interval_tree.empty() ? -1 : 0,
                    query_min,
                    query_max,
                    t0,
                    t1,
                    add_record_if_valid,
                    quality);
            }
            else
            {
                visit_record_contains_query_sorted();
            }

            if (bidirectional)
            {
                if (use_interval_tree)
                {
                    query_interval_contains_record_tree_(
                        group,
                        group.interval_tree.empty() ? -1 : 0,
                        query_min,
                        query_max,
                        t0,
                        t1,
                        add_record_if_valid,
                        quality);
                }
                else
                {
                    visit_query_contains_record_sorted();
                }
            }
            quality.containment_prune_seconds =
                std::chrono::duration<double>(Clock::now() - prune_start)
                    .count();

            last_query_records_visited_ = visited;
            quality.candidate_records = visited;
            quality.true_records_returned = records.size();
            quality.max_candidates_single_query = visited;
            last_query_quality_ = quality;
            return records;
        }

        void ensure_support_line_group_sorted_(
            SupportLineGroup& group) const
        {
            if (group.sorted)
                return;

            const auto by_min_less =
                [&](const RecordId a, const RecordId b)
                {
                    const auto& ra = records_[a];
                    const auto& rb = records_[b];
                    if (ra.support_s0 != rb.support_s0)
                        return ra.support_s0 < rb.support_s0;
                    if (ra.support_s1 != rb.support_s1)
                        return ra.support_s1 < rb.support_s1;
                    return a < b;
                };
            const auto by_max_less =
                [&](const RecordId a, const RecordId b)
                {
                    const auto& ra = records_[a];
                    const auto& rb = records_[b];
                    if (ra.support_s1 != rb.support_s1)
                        return ra.support_s1 < rb.support_s1;
                    if (ra.support_s0 != rb.support_s0)
                        return ra.support_s0 < rb.support_s0;
                    return a < b;
                };

            std::sort(group.by_min.begin(), group.by_min.end(), by_min_less);
            std::sort(group.by_max.begin(), group.by_max.end(), by_max_less);
            group.sorted = true;
        }

        void sort_all_support_line_groups_()
        {
            for (auto& [line_key, group] : records_by_support_line_)
            {
                (void)line_key;
                ensure_support_line_group_sorted_(group);
            }
        }

        void insert_support_line_record_(
            SupportLineGroup& group,
            const RecordId record_id)
        {
            group.interval_tree_current = false;
            group.interval_tree.clear();
            if (!group.sorted)
            {
                group.by_min.push_back(record_id);
                group.by_max.push_back(record_id);
                return;
            }

            const auto by_min_less =
                [&](const RecordId a, const RecordId b)
                {
                    const auto& ra = records_[a];
                    const auto& rb = records_[b];
                    if (ra.support_s0 != rb.support_s0)
                        return ra.support_s0 < rb.support_s0;
                    if (ra.support_s1 != rb.support_s1)
                        return ra.support_s1 < rb.support_s1;
                    return a < b;
                };
            const auto by_max_less =
                [&](const RecordId a, const RecordId b)
                {
                    const auto& ra = records_[a];
                    const auto& rb = records_[b];
                    if (ra.support_s1 != rb.support_s1)
                        return ra.support_s1 < rb.support_s1;
                    if (ra.support_s0 != rb.support_s0)
                        return ra.support_s0 < rb.support_s0;
                    return a < b;
                };

            group.by_min.insert(
                std::lower_bound(
                    group.by_min.begin(),
                    group.by_min.end(),
                    record_id,
                    by_min_less),
                record_id);
            group.by_max.insert(
                std::lower_bound(
                    group.by_max.begin(),
                    group.by_max.end(),
                    record_id,
                    by_max_less),
                record_id);
        }

        void compact_support_line_group_if_needed_(
            SupportLineGroup& group) const
        {
            const std::size_t total = group.by_min.size();
            const std::size_t inactive = inactive_record_count_(group.by_min);
            if (!group_should_compact_(
                    total,
                    inactive,
                    compaction_options_
                        .support_line_group_inactive_fraction_threshold,
                    compaction_options_
                        .max_inactive_records_per_support_line_group))
            {
                return;
            }

            const auto memory_before = support_line_group_memory_bytes_(group);
            const auto start = Clock::now();
            const auto keep_active =
                [&](const RecordId record_id)
                {
                    return record_is_active_(record_id);
                };
            group.by_min.erase(
                std::remove_if(
                    group.by_min.begin(),
                    group.by_min.end(),
                    [&](const RecordId record_id)
                    {
                        return !keep_active(record_id);
                    }),
                group.by_min.end());
            group.by_max.erase(
                std::remove_if(
                    group.by_max.begin(),
                    group.by_max.end(),
                    [&](const RecordId record_id)
                    {
                        return !keep_active(record_id);
                    }),
                group.by_max.end());
            group.interval_tree.clear();
            group.by_min.shrink_to_fit();
            group.by_max.shrink_to_fit();
            group.interval_tree.shrink_to_fit();
            group.interval_tree_current = false;

            const double seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            const auto memory_after = support_line_group_memory_bytes_(group);
            ++support_line_group_compaction_count_;
            support_line_group_compaction_records_removed_ += inactive;
            support_line_group_compaction_seconds_total_ += seconds;
            ++compaction_count_;
            compaction_seconds_total_ += seconds;
            group_compaction_memory_before_bytes_total_ += memory_before;
            group_compaction_memory_after_bytes_total_ += memory_after;
        }

        [[nodiscard]] bool record_is_active_(const RecordId record_id) const noexcept
        {
            return record_id < record_active_.size() &&
                   record_active_[record_id] != 0;
        }

        [[nodiscard]] SupportLineKey make_support_line_key_(
            const EdgeKey& edge_key) const
        {
            if (mesh_ == nullptr)
                throw std::runtime_error(
                    "ActiveEdgeIntervalIndex2D: support-line query on an unbound index.");

            constexpr double line_tol = 1.0e-10;
            const auto& spatial_vertices = mesh_->spatial_vertices();
            const auto& p0 =
                spatial_vertices[
                    static_cast<std::size_t>(edge_key.vertex_ids[0])];
            const auto& p1 =
                spatial_vertices[
                    static_cast<std::size_t>(edge_key.vertex_ids[1])];

            const double dx = p1[0] - p0[0];
            const double dy = p1[1] - p0[1];
            const double norm = std::sqrt(dx * dx + dy * dy);
            if (!(norm > 0.0))
                throw std::runtime_error(
                    "ActiveEdgeIntervalIndex2D: zero-length spatial edge.");

            double a = dy / norm;
            double b = -dx / norm;
            double c = -(a * p0[0] + b * p0[1]);

            if (a < -line_tol ||
                (std::abs(a) <= line_tol && b < -line_tol))
            {
                a = -a;
                b = -b;
                c = -c;
            }

            const auto quantize =
                [](const double value) -> long long
                {
                    constexpr double line_tol = 1.0e-10;
                    const double scaled = value / line_tol;
                    if (std::abs(scaled) < 0.5)
                        return 0;
                    return static_cast<long long>(std::llround(scaled));
                };

            return SupportLineKey{
                quantize(a),
                quantize(b),
                quantize(c)
            };
        }

        [[nodiscard]] int support_component_for_edge_(
            const EdgeKey& edge_key) const
        {
            if (mesh_ == nullptr)
                throw std::runtime_error(
                    "ActiveEdgeIntervalIndex2D: support component on an unbound index.");

            const auto& spatial_vertices = mesh_->spatial_vertices();
            const auto& p0 =
                spatial_vertices[
                    static_cast<std::size_t>(edge_key.vertex_ids[0])];
            const auto& p1 =
                spatial_vertices[
                    static_cast<std::size_t>(edge_key.vertex_ids[1])];

            return std::abs(p1[0] - p0[0]) >=
                           std::abs(p1[1] - p0[1])
                       ? 0
                       : 1;
        }

        [[nodiscard]] std::pair<double, double> support_interval_for_edge_(
            const EdgeKey& edge_key,
            const int component) const
        {
            if (mesh_ == nullptr)
                throw std::runtime_error(
                    "ActiveEdgeIntervalIndex2D: support interval on an unbound index.");

            const auto& spatial_vertices = mesh_->spatial_vertices();
            const double a =
                spatial_vertices[
                    static_cast<std::size_t>(edge_key.vertex_ids[0])]
                                [static_cast<std::size_t>(component)];
            const double b =
                spatial_vertices[
                    static_cast<std::size_t>(edge_key.vertex_ids[1])]
                                [static_cast<std::size_t>(component)];
            return {std::min(a, b), std::max(a, b)};
        }

        void compact_if_needed_()
        {
            const std::size_t total_records = records_.size();
            if (!group_should_compact_(
                    total_records,
                    n_inactive_records_,
                    compaction_options_.global_inactive_fraction_threshold,
                    compaction_options_.max_inactive_records_global))
            {
                return;
            }

            const auto memory_before = estimated_memory_bytes();
            const auto start = Clock::now();

            std::vector<Record> compact_records;
            compact_records.reserve(n_active_records_);
            std::vector<char> compact_active;
            compact_active.reserve(n_active_records_);

            std::unordered_map<EdgeKey, std::vector<RecordId>, EdgeHash>
                compact_by_edge;
            compact_by_edge.reserve(records_by_edge_.size());

            std::unordered_map<int, std::vector<RecordId>>
                compact_by_spatial_vertex;
            compact_by_spatial_vertex.reserve(records_by_spatial_vertex_.size());

            std::unordered_map<SupportLineKey, SupportLineGroup, SupportLineKeyHash>
                compact_by_support_line;
            compact_by_support_line.reserve(records_by_support_line_.size());

            std::unordered_map<int, std::vector<RecordId>>
                compact_by_cell;
            compact_by_cell.reserve(record_ids_by_cell_.size());

            for (RecordId old_id = 0; old_id < records_.size(); ++old_id)
            {
                if (!record_is_active_(old_id))
                    continue;

                const RecordId new_id = compact_records.size();
                compact_records.push_back(records_[old_id]);
                compact_active.push_back(1);

                const auto& record = compact_records.back();
                compact_by_edge[record.edge_key].push_back(new_id);
                for (const int vertex_id : record.edge_key.vertex_ids)
                    compact_by_spatial_vertex[vertex_id].push_back(new_id);
                auto& support_group =
                    compact_by_support_line[
                        make_support_line_key_(record.edge_key)];
                if (support_group.component < 0)
                    support_group.component = record.support_component;
                support_group.by_min.push_back(new_id);
                support_group.by_max.push_back(new_id);
                support_group.sorted = false;
                compact_by_cell[record.cell_id].push_back(new_id);
            }

            records_ = std::move(compact_records);
            record_active_ = std::move(compact_active);
            records_by_edge_ = std::move(compact_by_edge);
            records_by_spatial_vertex_ =
                std::move(compact_by_spatial_vertex);
            records_by_support_line_ = std::move(compact_by_support_line);
            sort_all_support_line_groups_();
            record_ids_by_cell_ = std::move(compact_by_cell);
            n_active_records_ = records_.size();
            n_inactive_records_ = 0;

            ++compaction_count_;
            compaction_seconds_total_ +=
                std::chrono::duration<double>(Clock::now() - start).count();
            global_compaction_memory_before_bytes_total_ += memory_before;
            global_compaction_memory_after_bytes_total_ +=
                estimated_memory_bytes();
        }

        void bind_mesh_(const MeshType& mesh)
        {
            if (mesh_ == nullptr)
            {
                mesh_ = &mesh;
                return;
            }

            if (mesh_ != &mesh)
                throw std::runtime_error(
                    "ActiveEdgeIntervalIndex2D: index is bound to a different mesh.");
        }

        const MeshType* mesh_ = nullptr;

        std::vector<Record> records_{};
        std::vector<char> record_active_{};
        mutable std::unordered_map<EdgeKey, std::vector<RecordId>, EdgeHash>
            records_by_edge_{};
        mutable std::unordered_map<int, std::vector<RecordId>>
            records_by_spatial_vertex_{};
        mutable std::unordered_map<SupportLineKey, SupportLineGroup, SupportLineKeyHash>
            records_by_support_line_{};
        std::unordered_map<int, std::vector<RecordId>> record_ids_by_cell_{};

        bool bulk_rebuilding_ = false;
        std::size_t n_active_records_ = 0;
        std::size_t n_inactive_records_ = 0;
        std::uint64_t rebuild_count_ = 0;
        std::uint64_t add_count_ = 0;
        std::uint64_t remove_count_ = 0;
        std::uint64_t remove_cell_records_touched_total_ = 0;
        mutable std::uint64_t compaction_count_ = 0;
        mutable double compaction_seconds_total_ = 0.0;
        CompactionOptions compaction_options_{};
        mutable std::uint64_t support_line_group_compaction_count_ = 0;
        mutable std::uint64_t support_line_group_compaction_records_removed_ = 0;
        mutable double support_line_group_compaction_seconds_total_ = 0.0;
        mutable std::uint64_t edge_group_compaction_count_ = 0;
        mutable std::uint64_t edge_group_compaction_records_removed_ = 0;
        mutable double edge_group_compaction_seconds_total_ = 0.0;
        mutable std::uint64_t spatial_vertex_group_compaction_count_ = 0;
        mutable std::uint64_t spatial_vertex_group_compaction_records_removed_ = 0;
        mutable double spatial_vertex_group_compaction_seconds_total_ = 0.0;
        mutable std::uint64_t group_compaction_memory_before_bytes_total_ = 0;
        mutable std::uint64_t group_compaction_memory_after_bytes_total_ = 0;
        mutable std::uint64_t global_compaction_memory_before_bytes_total_ = 0;
        mutable std::uint64_t global_compaction_memory_after_bytes_total_ = 0;
        mutable std::size_t last_query_records_visited_ = 0;
        mutable QueryQualityStats last_query_quality_{};
    };
}

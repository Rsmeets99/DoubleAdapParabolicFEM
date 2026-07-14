#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../mesh/mesh.hpp"
#include "../../mesh/refinement/refinement_type.hpp"
#include "../../mesh/topology/temporal_keys.hpp"
#include "topology/active_edge_interval_index_2d.hpp"

namespace finite_element::fespace
{
    enum class SearchIndexBuildMode
    {
        Eager,
        Lazy,
        Disabled
    };

    template<typename GeomTraits>
    class PartitionView
    {
    public:
        using MeshType = mesh::Mesh<GeomTraits>;
        using Types = mesh::MeshTypes<GeomTraits>;
        using SpaceTimePoint = typename Types::SpaceTimePoint;
        using ActiveEdgeIntervalIndex2DType =
            finite_element::fespace::topology::ActiveEdgeIntervalIndex2D<
                GeomTraits>;

        struct ActiveCellSearchIndex
        {
            std::vector<double> time_breaks{};
            std::vector<mesh::topology::TimePointIdKey> time_break_id_keys{};
            std::vector<mesh::topology::DyadicTimePointKey> time_break_dyadic_keys{};
            std::vector<std::vector<int>> time_slab_candidates{};
            std::vector<int> active_rank_by_cell{};

            void clear()
            {
                time_breaks.clear();
                time_break_id_keys.clear();
                time_break_dyadic_keys.clear();
                time_slab_candidates.clear();
                active_rank_by_cell.clear();
            }

            [[nodiscard]] bool empty() const noexcept
            {
                return time_breaks.size() < 2 || time_slab_candidates.empty();
            }

            [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
            {
                std::size_t bytes = 0;
                bytes += time_breaks.capacity() * sizeof(double);
                bytes += time_break_id_keys.capacity() *
                         sizeof(mesh::topology::TimePointIdKey);
                bytes += time_break_dyadic_keys.capacity() *
                         sizeof(mesh::topology::DyadicTimePointKey);
                bytes += time_slab_candidates.capacity() *
                         sizeof(std::vector<int>);
                for (const auto& candidates : time_slab_candidates)
                    bytes += candidates.capacity() * sizeof(int);
                bytes += active_rank_by_cell.capacity() * sizeof(int);
                return bytes;
            }
        };

        struct UpdateTiming
        {
            double total = 0.0;
            double membership_refresh = 0.0;
            double membership_remove = 0.0;
            double membership_add = 0.0;
            double active_vector_rebuild = 0.0;
            double active_search_index_rebuild = 0.0;
            double edge_index_remove_total = 0.0;
            double edge_index_add_total = 0.0;
            double edge_index_rebuild_total = 0.0;
            double edge_index_compaction_seconds = 0.0;
            std::uint64_t edge_index_remove_cell_records_touched = 0;
            std::uint64_t edge_index_compaction_count = 0;

            void clear() noexcept
            {
                *this = {};
            }
        };

        struct SearchIndexStats
        {
            std::uint64_t build_eager_count = 0;
            std::uint64_t build_lazy_count = 0;
            std::uint64_t disabled_count = 0;
            std::uint64_t fallback_scan_count = 0;
            double rebuild_seconds = 0.0;
        };

        enum class ActiveUpdateMode
        {
            MembershipOnly,
            MembershipAndEdgeIndex,
            MembershipAndSearchIndex,
            FullRebuild
        };

        explicit PartitionView(MeshType& mesh)
            : mesh_(mesh)
        {}

        [[nodiscard]] const std::vector<int>& active_cells() const noexcept
        {
            return active_cells_;
        }

        [[nodiscard]] bool is_active_cell(const int cell_id) const noexcept
        {
            return cell_id >= 0 &&
                   static_cast<std::size_t>(cell_id) < is_active_.size() &&
                   is_active_[static_cast<std::size_t>(cell_id)] != 0;
        }

        void set_active_cells(
            const std::vector<int>& cells,
            const SearchIndexBuildMode search_index_mode =
                SearchIndexBuildMode::Lazy,
            const bool build_edge_interval_index = true)
        {
            unsafe_set_active_cells(
                cells,
                search_index_mode,
                build_edge_interval_index);
            if (!validate_antichain())
                throw std::runtime_error(
                    "PartitionView::set_active_cells: invalid active antichain.");
        }

        void unsafe_set_active_cells(
            const std::vector<int>& cells,
            const SearchIndexBuildMode search_index_mode =
                SearchIndexBuildMode::Lazy,
            const bool build_edge_interval_index = true)
        {
            last_update_timing_.clear();
            const auto total_start = Clock::now();
            search_index_mode_ = search_index_mode;

            last_update_timing_.active_vector_rebuild +=
                measure_seconds_([&]() { active_cells_ = cells; });
            last_update_timing_.membership_add +=
                measure_seconds_(
                    [&]() { rebuild_activity_membership_from_active_cells_(); });

            if (search_index_mode == SearchIndexBuildMode::Eager)
            {
                last_update_timing_.active_search_index_rebuild +=
                    rebuild_active_cell_search_index_counted_(
                        SearchIndexBuildMode::Eager);
            }
            else if (search_index_mode == SearchIndexBuildMode::Lazy)
            {
                search_index_.clear();
                active_search_index_dirty_ = true;
            }
            else
            {
                search_index_.clear();
                active_search_index_dirty_ = false;
                ++search_index_stats_.disabled_count;
            }

            if (build_edge_interval_index)
            {
                last_update_timing_.edge_index_rebuild_total +=
                    measure_seconds_(
                        [&]() { rebuild_active_edge_interval_index_2d_(); });
            }
            else
            {
                clear_active_edge_interval_index_2d_();
            }
            last_update_timing_.total = seconds_since_(total_start);
            ++active_version_;
        }

        void unsafe_update_active_cells(
            const std::unordered_set<int>& add_ids,
            const std::unordered_set<int>& remove_ids,
            const ActiveUpdateMode mode =
                ActiveUpdateMode::MembershipAndEdgeIndex)
        {
            last_update_timing_.clear();
            const auto total_start = Clock::now();

            last_update_timing_.membership_refresh +=
                measure_seconds_([&]() { ensure_activity_capacity_(); });

            std::uint64_t edge_records_touched_before = 0;
            std::uint64_t edge_compaction_count_before = 0;
            double edge_compaction_seconds_before = 0.0;
            if constexpr (GeomTraits::dim_space_v == 2 &&
                          GeomTraits::dim_time_v == 1)
            {
                edge_records_touched_before =
                    active_edge_interval_index_2d_
                        .remove_cell_records_touched_total();
                edge_compaction_count_before =
                    active_edge_interval_index_2d_.compaction_count();
                edge_compaction_seconds_before =
                    active_edge_interval_index_2d_
                        .compaction_seconds_total();
            }

            const bool update_edge_incrementally =
                mode == ActiveUpdateMode::MembershipAndEdgeIndex ||
                mode == ActiveUpdateMode::MembershipAndSearchIndex;
            const bool rebuild_edge_index =
                mode == ActiveUpdateMode::FullRebuild;
            const bool rebuild_search_index =
                mode == ActiveUpdateMode::MembershipAndSearchIndex ||
                mode == ActiveUpdateMode::FullRebuild;

            std::vector<int> sorted_remove_ids(
                remove_ids.begin(),
                remove_ids.end());
            std::sort(sorted_remove_ids.begin(), sorted_remove_ids.end());
            std::vector<int> sorted_add_ids(add_ids.begin(), add_ids.end());
            std::sort(sorted_add_ids.begin(), sorted_add_ids.end());

            const double edge_remove_before =
                last_update_timing_.edge_index_remove_total;
            const auto remove_start = Clock::now();
            for (const int c : sorted_remove_ids)
            {
                if (c < 0 ||
                    static_cast<std::size_t>(c) >= is_active_.size() ||
                    is_active_[static_cast<std::size_t>(c)] == 0)
                {
                    continue;
                }

                if constexpr (GeomTraits::dim_space_v == 2 &&
                              GeomTraits::dim_time_v == 1)
                {
                    if (update_edge_incrementally)
                    {
                        last_update_timing_.edge_index_remove_total +=
                            measure_seconds_(
                                [&]()
                                {
                                    active_edge_interval_index_2d_
                                        .remove_cell(mesh_, c);
                                });
                    }
                }

                remove_active_cell_by_position_(c);
            }
            last_update_timing_.membership_remove +=
                std::max(
                    0.0,
                    seconds_since_(remove_start) -
                        (last_update_timing_.edge_index_remove_total -
                         edge_remove_before));

            const double edge_add_before =
                last_update_timing_.edge_index_add_total;
            const auto add_start = Clock::now();
            for (const int c : sorted_add_ids)
            {
                if (!mesh_.valid_cell_id(c))
                    throw std::runtime_error(
                        "PartitionView::unsafe_update_active_cells: invalid cell id.");
                ensure_activity_capacity_();
                if (is_active_[static_cast<std::size_t>(c)] != 0)
                    continue;

                add_active_cell_by_position_(c);
                if constexpr (GeomTraits::dim_space_v == 2 &&
                              GeomTraits::dim_time_v == 1)
                {
                    if (update_edge_incrementally)
                    {
                        last_update_timing_.edge_index_add_total +=
                            measure_seconds_(
                                [&]()
                                {
                                    active_edge_interval_index_2d_.add_cell(
                                        mesh_,
                                        c);
                                });
                    }
                }
            }
            last_update_timing_.membership_add +=
                std::max(
                    0.0,
                    seconds_since_(add_start) -
                        (last_update_timing_.edge_index_add_total -
                         edge_add_before));

            if (rebuild_edge_index)
            {
                last_update_timing_.edge_index_rebuild_total +=
                    measure_seconds_(
                        [&]() { rebuild_active_edge_interval_index_2d_(); });
            }

            if (rebuild_search_index)
            {
                if (search_index_mode_ == SearchIndexBuildMode::Disabled)
                {
                    search_index_.clear();
                    active_search_index_dirty_ = false;
                    ++search_index_stats_.disabled_count;
                }
                else
                {
                    last_update_timing_.active_search_index_rebuild +=
                        rebuild_active_cell_search_index_counted_(
                            SearchIndexBuildMode::Eager);
                }
            }
            else
            {
                mark_active_cell_search_index_dirty_();
            }

            if constexpr (GeomTraits::dim_space_v == 2 &&
                          GeomTraits::dim_time_v == 1)
            {
                last_update_timing_.edge_index_remove_cell_records_touched =
                    active_edge_interval_index_2d_
                        .remove_cell_records_touched_total() -
                    edge_records_touched_before;
                last_update_timing_.edge_index_compaction_count =
                    active_edge_interval_index_2d_.compaction_count() -
                    edge_compaction_count_before;
                last_update_timing_.edge_index_compaction_seconds =
                    active_edge_interval_index_2d_
                        .compaction_seconds_total() -
                    edge_compaction_seconds_before;
            }

            last_update_timing_.total = seconds_since_(total_start);
            ++active_version_;
        }

        void rebuild_active_search_index_if_needed()
        {
            if (!active_search_index_dirty_)
                return;
            last_update_timing_.active_search_index_rebuild +=
                rebuild_active_cell_search_index_counted_(
                    SearchIndexBuildMode::Lazy);
        }

        [[nodiscard]] MeshType& mesh_ref() noexcept
        {
            return mesh_;
        }

        [[nodiscard]] const MeshType& mesh_ref() const noexcept
        {
            return mesh_;
        }

        [[nodiscard]] std::vector<int>& unsafe_active_cells_ref() noexcept
        {
            return active_cells_;
        }

        [[nodiscard]] bool validate_antichain() const
        {
            if (is_active_.size() > mesh_.n_cells())
                return false;

            std::unordered_set<int> active_set;
            active_set.reserve(active_cells_.size());

            for (const int cell_id : active_cells_)
            {
                if (!mesh_.valid_cell_id(cell_id))
                    return false;
                if (!active_set.insert(cell_id).second)
                    return false;
                if (!is_active_cell(cell_id))
                    return false;
                if (static_cast<std::size_t>(cell_id) >=
                    active_pos_by_cell_.size())
                    return false;
                const int active_pos =
                    active_pos_by_cell_[static_cast<std::size_t>(cell_id)];
                if (active_pos < 0 ||
                    static_cast<std::size_t>(active_pos) >=
                        active_cells_.size() ||
                    active_cells_[static_cast<std::size_t>(active_pos)] !=
                        cell_id)
                {
                    return false;
                }
            }

            for (std::size_t i = 0; i < is_active_.size(); ++i)
            {
                if (is_active_[i] != 0 &&
                    active_set.count(static_cast<int>(i)) == 0)
                {
                    return false;
                }
                if (i < active_pos_by_cell_.size())
                {
                    const int pos = active_pos_by_cell_[i];
                    if (is_active_[i] == 0 && pos >= 0)
                        return false;
                    if (is_active_[i] != 0 &&
                        (pos < 0 ||
                         static_cast<std::size_t>(pos) >=
                             active_cells_.size() ||
                         active_cells_[static_cast<std::size_t>(pos)] !=
                             static_cast<int>(i)))
                    {
                        return false;
                    }
                }
                else if (is_active_[i] != 0)
                {
                    return false;
                }
            }

            for (const int cell_id : active_cells_)
            {
                int ancestor = mesh_.cell(cell_id).parent_id;
                std::size_t guard = 0;
                while (ancestor >= 0)
                {
                    if (!mesh_.valid_cell_id(ancestor))
                        return false;
                    if (is_active_cell(ancestor))
                        return false;

                    ancestor = mesh_.cell(ancestor).parent_id;
                    ++guard;
                    if (guard > mesh_.n_cells())
                        return false;
                }
            }

            return true;
        }

        [[nodiscard]] bool refines(const PartitionView& coarse) const
        {
            if (&mesh_ != &coarse.mesh_)
                return false;
            if (!validate_antichain() || !coarse.validate_antichain())
                return false;

            std::vector<char> coarse_covered(mesh_.n_cells(), 0);

            for (const int cell_id : active_cells_)
            {
                int current = cell_id;
                bool has_coarse_ancestor_or_self = false;
                std::size_t guard = 0;
                while (current >= 0)
                {
                    if (!mesh_.valid_cell_id(current))
                        return false;

                    if (coarse.is_active_cell(current))
                    {
                        coarse_covered[static_cast<std::size_t>(current)] = 1;
                        has_coarse_ancestor_or_self = true;
                        break;
                    }

                    current = mesh_.cell(current).parent_id;
                    ++guard;
                    if (guard > mesh_.n_cells())
                        return false;
                }

                if (!has_coarse_ancestor_or_self)
                    return false;
            }

            for (const int cell_id : coarse.active_cells_)
            {
                if (!coarse_covered[static_cast<std::size_t>(cell_id)])
                    return false;
            }

            return true;
        }

        void ensure_refines(const PartitionView& coarse)
        {
            if (&mesh_ != &coarse.mesh_)
                throw std::runtime_error(
                    "PartitionView::ensure_refines: partitions use different meshes.");
            if (!coarse.validate_antichain())
                throw std::runtime_error(
                    "PartitionView::ensure_refines: coarse partition is invalid.");
            if (!validate_antichain())
                throw std::runtime_error(
                    "PartitionView::ensure_refines: fine partition is invalid.");

            std::size_t guard = 0;
            while (!refines(coarse))
            {
                std::vector<int> cells_to_refine;
                std::unordered_set<int> seen;
                cells_to_refine.reserve(active_cells_.size());

                for (const int cell_id : active_cells_)
                {
                    if (has_active_ancestor_or_self_in_(coarse, cell_id))
                        continue;
                    if (!has_active_descendant_or_self_in_(coarse, cell_id))
                    {
                        throw std::runtime_error(
                            "PartitionView::ensure_refines: active cell is outside coarse partition.");
                    }
                    if (seen.insert(cell_id).second)
                        cells_to_refine.push_back(cell_id);
                }

                for (const int coarse_cell_id : coarse.active_cells_)
                {
                    if (is_covered_by_active_self_or_descendant_(coarse_cell_id))
                        continue;

                    const int active_ancestor =
                        find_active_ancestor_or_self_(coarse_cell_id);
                    if (active_ancestor < 0)
                    {
                        throw std::runtime_error(
                            "PartitionView::ensure_refines: coarse cell is not covered by this partition.");
                    }
                    if (seen.insert(active_ancestor).second)
                        cells_to_refine.push_back(active_ancestor);
                }

                if (cells_to_refine.empty())
                    throw std::runtime_error(
                        "PartitionView::ensure_refines: no repair refinement found.");

                refine_active_cells_once_(cells_to_refine);

                ++guard;
                if (guard > mesh_.n_cells() + 1)
                {
                    throw std::runtime_error(
                        "PartitionView::ensure_refines: repair did not converge.");
                }
            }
        }

        [[nodiscard]] std::uint64_t active_version() const noexcept
        {
            return active_version_;
        }

        [[nodiscard]] const UpdateTiming& last_update_timing() const noexcept
        {
            return last_update_timing_;
        }

        [[nodiscard]] const SearchIndexStats&
        search_index_stats() const noexcept
        {
            return search_index_stats_;
        }

        [[nodiscard]] SearchIndexBuildMode
        search_index_build_mode() const noexcept
        {
            return search_index_mode_;
        }

        [[nodiscard]] bool active_cell_search_index_is_dirty() const noexcept
        {
            return active_search_index_dirty_;
        }

        [[nodiscard]] bool active_cell_search_index_is_disabled() const noexcept
        {
            return search_index_mode_ == SearchIndexBuildMode::Disabled;
        }

        [[nodiscard]] bool active_cell_search_index_is_empty() const noexcept
        {
            return search_index_.empty();
        }

        void record_search_index_fallback_scan() const noexcept
        {
            ++search_index_stats_.fallback_scan_count;
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            std::size_t bytes = 0;
            bytes += active_cells_.capacity() * sizeof(int);
            bytes += is_active_.capacity() * sizeof(char);
            bytes += active_pos_by_cell_.capacity() * sizeof(int);
            bytes += search_index_.estimated_memory_bytes();
            if constexpr (GeomTraits::dim_space_v == 2 &&
                          GeomTraits::dim_time_v == 1)
            {
                bytes +=
                    active_edge_interval_index_2d_.estimated_memory_bytes();
            }
            return bytes;
        }

        [[nodiscard]] const ActiveEdgeIntervalIndex2DType&
        active_edge_interval_index_2d() const noexcept
        {
            static_assert(GeomTraits::dim_space_v == 2,
                          "active_edge_interval_index_2d requires dim_space_v == 2.");
            static_assert(GeomTraits::dim_time_v == 1,
                          "active_edge_interval_index_2d requires dim_time_v == 1.");
            return active_edge_interval_index_2d_;
        }

        void rebuild_active_cell_search_index()
        {
            search_index_mode_ = SearchIndexBuildMode::Eager;
            static_cast<void>(
                rebuild_active_cell_search_index_counted_(
                    SearchIndexBuildMode::Eager));
        }

        void rebuild_active_cell_search_index_impl_()
        {
            search_index_.clear();
            active_search_index_dirty_ = false;

            if constexpr (GeomTraits::dim_time_v != 1 ||
                          (GeomTraits::dim_space_v != 1 &&
                           GeomTraits::dim_space_v != 2))
            {
                return;
            }

            if (active_cells_.empty())
                return;

            search_index_.active_rank_by_cell.assign(mesh_.n_cells(), -1);
            for (int rank = 0;
                 rank < static_cast<int>(active_cells_.size());
                 ++rank)
            {
                const int cell_id =
                    active_cells_[static_cast<std::size_t>(rank)];
                if (cell_id >= 0 &&
                    static_cast<std::size_t>(cell_id) <
                        search_index_.active_rank_by_cell.size())
                {
                    search_index_.active_rank_by_cell[
                        static_cast<std::size_t>(cell_id)] = rank;
                }
            }

            struct EndpointRecord
            {
                int temporal_vertex_id = -1;
                double time = 0.0;
                mesh::topology::DyadicTimePointKey dyadic_key{};
            };

            std::vector<EndpointRecord> endpoints;
            endpoints.reserve(2 * active_cells_.size());

            for (const int cell_id : active_cells_)
            {
                const auto& cell = mesh_.cell(cell_id);
                const auto dyadic_endpoint_keys =
                    mesh_.dyadic_temporal_endpoint_keys(cell_id);
                for (int endpoint = 0; endpoint < 2; ++endpoint)
                {
                    const int temporal_vertex_id =
                        cell.temporal_vertex_ids[
                            static_cast<std::size_t>(endpoint)];
                    endpoints.push_back(
                        EndpointRecord{
                            temporal_vertex_id,
                            mesh_.temporal_vertices()[
                                static_cast<std::size_t>(
                                    temporal_vertex_id)][0],
                            dyadic_endpoint_keys[
                                static_cast<std::size_t>(endpoint)]});
                }
            }

            std::sort(
                endpoints.begin(),
                endpoints.end(),
                [](const EndpointRecord& a, const EndpointRecord& b)
                {
                    if (a.time != b.time)
                        return a.time < b.time;
                    return a.temporal_vertex_id < b.temporal_vertex_id;
                });

            std::unordered_map<
                mesh::topology::TimePointIdKey,
                int,
                mesh::topology::TimePointIdKeyHash>
                endpoint_rank_by_id;
            endpoint_rank_by_id.reserve(endpoints.size());

            for (const auto& endpoint : endpoints)
            {
                const auto id_key =
                    mesh::topology::make_time_point_id_key(
                        endpoint.temporal_vertex_id);
                if (endpoint_rank_by_id.find(id_key) !=
                    endpoint_rank_by_id.end())
                {
                    continue;
                }

                if (search_index_.time_breaks.empty() ||
                    endpoint.time != search_index_.time_breaks.back())
                {
                    search_index_.time_breaks.push_back(endpoint.time);
                    search_index_.time_break_id_keys.push_back(id_key);
                    search_index_.time_break_dyadic_keys.push_back(
                        endpoint.dyadic_key);
                }

                endpoint_rank_by_id.emplace(
                    id_key,
                    static_cast<int>(search_index_.time_breaks.size()) - 1);
            }

            if (search_index_.time_breaks.size() < 2)
                return;

            search_index_.time_slab_candidates.assign(
                search_index_.time_breaks.size() - 1,
                {});

            for (const int cell_id : active_cells_)
            {
                const auto& cell = mesh_.cell(cell_id);
                const auto begin_it =
                    endpoint_rank_by_id.find(
                        mesh::topology::make_time_point_id_key(
                            cell.temporal_vertex_ids[0]));
                const auto end_it =
                    endpoint_rank_by_id.find(
                        mesh::topology::make_time_point_id_key(
                            cell.temporal_vertex_ids[1]));
                if (begin_it == endpoint_rank_by_id.end() ||
                    end_it == endpoint_rank_by_id.end())
                {
                    continue;
                }

                const int begin_idx = begin_it->second;
                const int end_idx = end_it->second;

                for (int k = begin_idx; k < end_idx; ++k)
                {
                    search_index_.time_slab_candidates[
                        static_cast<std::size_t>(k)].push_back(cell_id);
                }
            }

            for (auto& candidates : search_index_.time_slab_candidates)
            {
                std::sort(
                    candidates.begin(),
                    candidates.end(),
                    [&](const int a, const int b)
                    {
                        const int ga = mesh_.cell(a).generation;
                        const int gb = mesh_.cell(b).generation;
                        if (ga != gb)
                            return ga > gb;
                        return a < b;
                    });
            }
        }

        [[nodiscard]] int find_active_cell_from_search_index(
            const SpaceTimePoint& p) const
        {
            if (search_index_mode_ == SearchIndexBuildMode::Disabled)
            {
                ++search_index_stats_.disabled_count;
                throw std::runtime_error(
                    "PartitionView::find_active_cell_from_search_index: active-cell search index is disabled for this FE space.");
            }

            ensure_active_cell_search_index_current_();
            if (search_index_.empty())
                return -1;

            const int interval_idx =
                find_interval_index_(
                    search_index_.time_breaks,
                    p[GeomTraits::dim_space_v]);
            if (interval_idx < 0)
                return -1;

            const auto& candidates =
                search_index_.time_slab_candidates[
                    static_cast<std::size_t>(interval_idx)];

            int best_boundary_match = -1;
            int best_boundary_rank = std::numeric_limits<int>::max();
            for (const int cell_id : candidates)
            {
                if (!mesh_.contains_coord(cell_id, p))
                    continue;

                if (point_is_strictly_inside_cell_(cell_id, p))
                    return cell_id;

                const int rank = active_rank_for_search_(cell_id);
                if (rank < best_boundary_rank)
                {
                    best_boundary_match = cell_id;
                    best_boundary_rank = rank;
                }
            }

            return best_boundary_match;
        }

    private:
        using Clock = std::chrono::steady_clock;

        template<class Fn>
        static double measure_seconds_(Fn&& fn)
        {
            const auto start = Clock::now();
            std::forward<Fn>(fn)();
            return seconds_since_(start);
        }

        [[nodiscard]] static double seconds_since_(
            const Clock::time_point start)
        {
            return std::chrono::duration<double>(
                       Clock::now() - start)
                .count();
        }

        [[nodiscard]] int active_rank_for_search_(const int cell_id) const noexcept
        {
            if (cell_id < 0 ||
                static_cast<std::size_t>(cell_id) >=
                    search_index_.active_rank_by_cell.size())
            {
                return std::numeric_limits<int>::max();
            }

            const int rank =
                search_index_.active_rank_by_cell[
                    static_cast<std::size_t>(cell_id)];
            if (rank < 0)
                return std::numeric_limits<int>::max();
            return rank;
        }

        [[nodiscard]] bool point_is_strictly_inside_cell_(
            const int cell_id,
            const SpaceTimePoint& p) const
        {
            constexpr double tol = 1.0e-12;

            if (!mesh_.valid_cell_id(cell_id))
                return false;

            const auto& cell = mesh_.cell(cell_id);
            const double t0 =
                mesh_.temporal_vertices()[
                    static_cast<std::size_t>(cell.temporal_vertex_ids[0])][0];
            const double t1 =
                mesh_.temporal_vertices()[
                    static_cast<std::size_t>(cell.temporal_vertex_ids[1])][0];
            const double t = p[GeomTraits::dim_space_v];
            if (!(t0 + tol < t && t < t1 - tol))
                return false;

            if constexpr (GeomTraits::dim_space_v == 1)
            {
                const double x0 =
                    mesh_.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[0])][0];
                const double x1 =
                    mesh_.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[1])][0];
                const double xl = std::min(x0, x1);
                const double xr = std::max(x0, x1);
                return xl + tol < p[0] && p[0] < xr - tol;
            }
            else if constexpr (GeomTraits::dim_space_v == 2)
            {
                const auto& v0 =
                    mesh_.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[0])];
                const auto& v1 =
                    mesh_.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[1])];
                const auto& v2 =
                    mesh_.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[2])];

                const double J00 = v1[0] - v0[0];
                const double J01 = v2[0] - v0[0];
                const double J10 = v1[1] - v0[1];
                const double J11 = v2[1] - v0[1];
                const double det = J00 * J11 - J01 * J10;
                if (std::abs(det) < 1.0e-15)
                    return false;

                const double dx = p[0] - v0[0];
                const double dy = p[1] - v0[1];
                const double inv_det = 1.0 / det;
                const double xi  = ( J11 * dx - J01 * dy) * inv_det;
                const double eta = (-J10 * dx + J00 * dy) * inv_det;
                return xi > tol && eta > tol && xi + eta < 1.0 - tol;
            }
            else
            {
                return false;
            }
        }

        [[nodiscard]] static int find_interval_index_(
            const std::vector<double>& breaks,
            const double value)
        {
            if (breaks.size() < 2)
                return -1;

            auto it = std::lower_bound(breaks.begin(), breaks.end(), value);
            if (it == breaks.end())
            {
                if (value == breaks.back())
                    return static_cast<int>(breaks.size()) - 2;
                return -1;
            }

            const int idx =
                static_cast<int>(std::distance(breaks.begin(), it));
            if (*it == value)
            {
                if (idx == static_cast<int>(breaks.size()) - 1)
                    return idx - 1;
                return idx;
            }

            if (idx == 0)
                return -1;

            return idx - 1;
        }

        void refresh_activity_mask()
        {
            rebuild_activity_membership_from_active_cells_();
        }

        void rebuild_activity_membership_from_active_cells_()
        {
            is_active_.assign(mesh_.n_cells(), 0);
            active_pos_by_cell_.assign(mesh_.n_cells(), -1);
            for (std::size_t pos = 0; pos < active_cells_.size(); ++pos)
            {
                const int c = active_cells_[pos];
                if (!mesh_.valid_cell_id(c))
                    throw std::runtime_error(
                        "PartitionView::set_active_cells: invalid cell id.");
                is_active_[static_cast<std::size_t>(c)] = 1;
                active_pos_by_cell_[static_cast<std::size_t>(c)] =
                    static_cast<int>(pos);
            }
        }

        void ensure_activity_capacity_()
        {
            const auto n_cells = mesh_.n_cells();
            if (is_active_.size() < n_cells)
                is_active_.resize(n_cells, 0);
            if (active_pos_by_cell_.size() < n_cells)
                active_pos_by_cell_.resize(n_cells, -1);
        }

        void remove_active_cell_by_position_(const int cell_id)
        {
            const auto idx = static_cast<std::size_t>(cell_id);
            if (idx >= active_pos_by_cell_.size())
                return;

            const int pos = active_pos_by_cell_[idx];
            if (pos < 0 ||
                static_cast<std::size_t>(pos) >= active_cells_.size())
            {
                is_active_[idx] = 0;
                active_pos_by_cell_[idx] = -1;
                return;
            }

            const int moved_cell = active_cells_.back();
            active_cells_[static_cast<std::size_t>(pos)] = moved_cell;
            active_pos_by_cell_[static_cast<std::size_t>(moved_cell)] = pos;
            active_cells_.pop_back();

            is_active_[idx] = 0;
            active_pos_by_cell_[idx] = -1;
        }

        void add_active_cell_by_position_(const int cell_id)
        {
            const auto idx = static_cast<std::size_t>(cell_id);
            is_active_[idx] = 1;
            active_pos_by_cell_[idx] =
                static_cast<int>(active_cells_.size());
            active_cells_.push_back(cell_id);
        }

        void ensure_active_cell_search_index_current_() const
        {
            if (!active_search_index_dirty_)
                return;
            auto* self = const_cast<PartitionView*>(this);
            self->last_update_timing_.active_search_index_rebuild +=
                self->rebuild_active_cell_search_index_counted_(
                    SearchIndexBuildMode::Lazy);
        }

        void mark_active_cell_search_index_dirty_()
        {
            if (search_index_mode_ == SearchIndexBuildMode::Disabled)
            {
                search_index_.clear();
                active_search_index_dirty_ = false;
            }
            else
            {
                active_search_index_dirty_ = true;
            }
        }

        [[nodiscard]] double rebuild_active_cell_search_index_counted_(
            const SearchIndexBuildMode trigger)
        {
            const auto start = Clock::now();
            rebuild_active_cell_search_index_impl_();
            const double seconds = seconds_since_(start);
            search_index_stats_.rebuild_seconds += seconds;
            if (trigger == SearchIndexBuildMode::Lazy)
                ++search_index_stats_.build_lazy_count;
            else
                ++search_index_stats_.build_eager_count;
            return seconds;
        }

        void rebuild_active_edge_interval_index_2d_()
        {
            if constexpr (GeomTraits::dim_space_v == 2 &&
                          GeomTraits::dim_time_v == 1)
            {
                active_edge_interval_index_2d_.rebuild(
                    mesh_,
                    active_cells_);
            }
        }

        void clear_active_edge_interval_index_2d_()
        {
            if constexpr (GeomTraits::dim_space_v == 2 &&
                          GeomTraits::dim_time_v == 1)
            {
                active_edge_interval_index_2d_.clear();
            }
        }

        [[nodiscard]] bool has_active_ancestor_or_self_in_(
            const PartitionView& other,
            const int cell_id) const
        {
            int current = cell_id;
            std::size_t guard = 0;
            while (current >= 0)
            {
                if (!mesh_.valid_cell_id(current))
                    return false;
                if (other.is_active_cell(current))
                    return true;

                current = mesh_.cell(current).parent_id;
                ++guard;
                if (guard > mesh_.n_cells())
                    return false;
            }
            return false;
        }

        [[nodiscard]] bool has_active_descendant_or_self_in_(
            const PartitionView& other,
            const int cell_id) const
        {
            if (other.is_active_cell(cell_id))
                return true;

            for (const int other_cell_id : other.active_cells_)
            {
                if (mesh_.is_ancestor(cell_id, other_cell_id))
                    return true;
            }

            return false;
        }

        [[nodiscard]] bool is_covered_by_active_self_or_descendant_(
            const int coarse_cell_id) const
        {
            if (is_active_cell(coarse_cell_id))
                return true;

            for (const int cell_id : active_cells_)
            {
                if (mesh_.is_ancestor(coarse_cell_id, cell_id))
                    return true;
            }

            return false;
        }

        [[nodiscard]] int find_active_ancestor_or_self_(const int cell_id) const
        {
            int current = cell_id;
            std::size_t guard = 0;
            while (current >= 0)
            {
                if (!mesh_.valid_cell_id(current))
                    return -1;
                if (is_active_cell(current))
                    return current;

                current = mesh_.cell(current).parent_id;
                ++guard;
                if (guard > mesh_.n_cells())
                    return -1;
            }

            return -1;
        }

        void refine_active_cells_once_(const std::vector<int>& cells_to_refine)
        {
            std::unordered_set<int> add_ids;
            std::unordered_set<int> remove_ids;

            for (const int cell_id : cells_to_refine)
            {
                if (!is_active_cell(cell_id))
                    continue;
                if (!mesh_.valid_cell_id(cell_id))
                    throw std::runtime_error(
                        "PartitionView::ensure_refines: invalid active cell id.");

                const auto children =
                    mesh_.create_children_if_needed(
                        cell_id,
                        mesh::RefinementType::none);
                if (children.empty())
                    throw std::runtime_error(
                        "PartitionView::ensure_refines: refined cell has no children.");

                remove_ids.insert(cell_id);
                for (const int child_id : children)
                    add_ids.insert(child_id);
            }

            if (!add_ids.empty() || !remove_ids.empty())
                unsafe_update_active_cells(add_ids, remove_ids);
        }

        MeshType& mesh_;
        std::vector<int> active_cells_{};
        std::vector<char> is_active_{};
        std::vector<int> active_pos_by_cell_{};
        mutable ActiveCellSearchIndex search_index_{};
        mutable bool active_search_index_dirty_ = false;
        mutable SearchIndexStats search_index_stats_{};
        SearchIndexBuildMode search_index_mode_ = SearchIndexBuildMode::Lazy;
        ActiveEdgeIntervalIndex2DType active_edge_interval_index_2d_{};
        UpdateTiming last_update_timing_{};
        std::uint64_t active_version_ = 0;
    };
}

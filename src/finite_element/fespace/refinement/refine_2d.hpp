#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../../mesh/refinement/refinement_type.hpp"
#include "../../../mesh/refinement/split_policy.hpp"
#include "../../../mesh/topology/spatial_edge_adjacency_2d.hpp"

namespace finite_element::fespace::detail::refinement_impl
{
    template<typename FESpaceType, class SlicewiseAdjacency>
    inline void record_slicewise_adjacency_metrics_2d(
        FESpaceType& space,
        std::string_view phase_prefix,
        std::size_t active_cells_scanned,
        const SlicewiseAdjacency& adjacency)
    {
        const auto stats = mesh::topology::adjacency_stats_2d(adjacency);
        const std::string phase(phase_prefix);
        space.record_timing_metric(
            phase + ".active_cells_scanned.count",
            static_cast<double>(active_cells_scanned));
        space.record_timing_metric(
            phase + ".time_slices_built.count",
            static_cast<double>(stats.time_slices_built));
        space.record_timing_metric(
            phase + ".edge_records_built.count",
            static_cast<double>(stats.edge_records_built));
    }

    template<typename FESpaceType>
    [[nodiscard]] inline std::vector<int>
    deduplicate_active_marked_cells_2d(
        const FESpaceType& space,
        const std::vector<int>& marked)
    {
        std::vector<int> unique_marked;
        unique_marked.reserve(marked.size());

        std::unordered_set<int> seen;
        for (const int cell_id : marked)
        {
            if (space.is_active_cell(cell_id) && seen.insert(cell_id).second)
                unique_marked.push_back(cell_id);
        }

        return unique_marked;
    }

    template<typename FESpaceType>
    inline void assert_active_spatial_conforming_full_2d(FESpaceType& space)
    {
        const auto& active_cells = space.active_cells();
        const auto adjacency =
            mesh::topology::build_slicewise_active_spatial_edge_adjacency_2d(
                space.mesh_ref(),
                active_cells);
        record_slicewise_adjacency_metrics_2d(
            space,
            "fespace.refinement.assert_active_spatial_conforming",
            active_cells.size(),
            adjacency);

        if (adjacency.has_singular_edges())
            throw std::runtime_error(
                "FESpace::refine_2d: active spatial edge adjacency is singular.");

        if (adjacency.has_nonconforming_edges())
            throw std::runtime_error(
                "FESpace::refine_2d: active spatial refinement closure left hanging edges.");
    }

    template<typename FESpaceType>
    inline void record_local_verification_metrics_2d(
        FESpaceType& space,
        std::string_view phase_prefix,
        const mesh::topology::LocalSpatialConformityVerificationStats2D& stats)
    {
        const std::string phase(phase_prefix);
        space.record_timing_metric(
            phase + ".active_cells_scanned.count",
            static_cast<double>(stats.active_cells_scanned));
        space.record_timing_metric(
            phase + ".time_slices_built.count",
            static_cast<double>(stats.time_slices_built));
        space.record_timing_metric(
            phase + ".edge_records_built.count",
            static_cast<double>(stats.edge_records_built));
        space.record_timing_metric(
            phase + ".local_edge_records_built.count",
            static_cast<double>(stats.local_edge_records_built));
        space.record_timing_metric(
            phase + ".fallback_to_full_check.count",
            static_cast<double>(stats.fallback_to_full_check));
        space.record_timing_metric(
            phase + ".seed_cells.count",
            static_cast<double>(stats.seed_cells));
        space.record_timing_metric(
            phase + ".seed_cells_scanned.count",
            static_cast<double>(stats.seed_cells_scanned));
        space.record_timing_metric(
            phase + ".active_edge_records_built.count",
            static_cast<double>(stats.active_edge_records_built));
        space.record_timing_metric(
            phase + ".seed_edge_records.count",
            static_cast<double>(stats.seed_edge_records));
        space.record_timing_metric(
            phase + ".candidate_edge_visits.count",
            static_cast<double>(stats.candidate_edge_visits));
        space.record_timing_metric(
            phase + ".candidate_cells.count",
            static_cast<double>(stats.candidate_cells));
        space.record_timing_metric(
            phase + ".candidate_cells_ratio",
            stats.candidate_cells_ratio);
        space.record_timing_metric(
            phase + ".seed_edge_records_checked.count",
            static_cast<double>(stats.seed_edge_records_checked));
        space.record_timing_metric(
            phase + ".singular_seed_edges.count",
            static_cast<double>(stats.singular_seed_edges));
        space.record_timing_metric(
            phase + ".nonconforming_seed_edges.count",
            static_cast<double>(stats.nonconforming_seed_edges));
        space.record_timing_metric(
            phase + ".active_edge_record_construction",
            stats.active_edge_record_construction_seconds);
        space.record_timing_metric(
            phase + ".active_cell_vertex_index_construction",
            stats.active_cell_vertex_index_construction_seconds);
        space.record_timing_metric(
            phase + ".seed_candidate_discovery",
            stats.seed_candidate_discovery_seconds);
        space.record_timing_metric(
            phase + ".local_slicewise_adjacency_rebuild",
            stats.local_slicewise_adjacency_rebuild_seconds);
        space.record_timing_metric(
            phase + ".seed_edge_conformity_check",
            stats.seed_edge_conformity_check_seconds);
        space.record_timing_metric(
            phase + ".local_check_time",
            stats.local_check_seconds);
        space.record_timing_metric(
            phase + ".full_check_time",
            stats.full_check_seconds);
    }

    template<typename FESpaceType>
    inline void assert_active_spatial_conforming_local_2d(
        FESpaceType& space,
        const std::vector<int>& seed_cells)
    {
        const auto result =
            mesh::topology::verify_local_spatial_conforming_2d(
                space.mesh_ref(),
                space.active_cells(),
                seed_cells);
        record_local_verification_metrics_2d(
            space,
            "fespace.refinement.assert_active_spatial_conforming",
            result.stats);

        if (!result.is_conforming)
            throw std::runtime_error(
                "FESpace::refine_2d: local active spatial refinement closure check failed.");
    }

    template<typename FESpaceType>
    inline void record_local_closure_metrics_2d(
        FESpaceType& space,
        std::string_view phase_prefix,
        const mesh::topology::LocalSpatialClosureStats2D& stats)
    {
        const std::string phase(phase_prefix);
        space.record_timing_metric(
            phase + ".active_cells_scanned.count",
            static_cast<double>(stats.active_cells_scanned));
        space.record_timing_metric(
            phase + ".time_slices_built.count",
            0.0);
        space.record_timing_metric(
            phase + ".edge_records_built.count",
            static_cast<double>(stats.edge_records_built));
        space.record_timing_metric(
            phase + ".seed_cells_scanned.count",
            static_cast<double>(stats.seed_cells_scanned));
        space.record_timing_metric(
            phase + ".seed_edge_records.count",
            static_cast<double>(stats.seed_edge_records));
        space.record_timing_metric(
            phase + ".candidate_edge_visits.count",
            static_cast<double>(stats.candidate_edge_visits));
        space.record_timing_metric(
            phase + ".edge_comparisons.count",
            static_cast<double>(stats.edge_comparisons));
        space.record_timing_metric(
            phase + ".time_overlap_tests.count",
            static_cast<double>(stats.time_overlap_tests));
        space.record_timing_metric(
            phase + ".same_spatial_overlap_scans.count",
            static_cast<double>(stats.same_spatial_overlap_scans));
    }

    template<typename FESpaceType>
    [[deprecated("Debug/testing oracle only. Normal 2+1D FE-active refinement uses the indexed queue path.")]]
    inline void collect_active_spatial_forced_refinements_full_2d(
        FESpaceType& space,
        std::vector<int>& forced)
    {
        const auto& active_cells = space.active_cells();
        const auto slicewise_adjacency =
            mesh::topology::build_slicewise_active_spatial_edge_adjacency_2d(
                space.mesh_ref(),
                active_cells);
        record_slicewise_adjacency_metrics_2d(
            space,
            "fespace.refinement.collect_active_spatial_forced",
            active_cells.size(),
            slicewise_adjacency);

        std::unordered_set<int> forced_seen;
        const auto forced_before = forced.size();
        for (const auto& slice : slicewise_adjacency.slices)
        {
            if (!slice.adjacency.singular_edge_ids.empty())
                throw std::runtime_error(
                    "FESpace::refine_2d: active spatial edge adjacency is singular.");

            for (const auto& containment : slice.adjacency.containments)
            {
                const auto& container =
                    slice.adjacency.edge(containment.container_edge_id);
                const auto& container_cell =
                    space.mesh_ref().cell(container.cell_id);

                for (const int active_cell_id : slice.active_cell_ids)
                {
                    if (!mesh::topology::same_spatial_cell_vertices_2d(
                            space.mesh_ref().cell(active_cell_id),
                            container_cell))
                    {
                        continue;
                    }

                    if (space.is_active_cell(active_cell_id) &&
                        forced_seen.insert(active_cell_id).second)
                    {
                        forced.push_back(active_cell_id);
                    }
                }
            }
        }
        space.record_timing_metric(
            "fespace.refinement.collect_active_spatial_forced.forced_refinements_added.count",
            static_cast<double>(forced.size() - forced_before));
    }

    template<typename FESpaceType>
    [[deprecated("Debug/testing oracle only. Normal 2+1D FE-active refinement uses the active edge/time interval index.")]]
    inline void collect_active_spatial_forced_refinements_2d(
        FESpaceType& space,
        const std::vector<int>& seed_cells,
        std::vector<int>& forced)
    {
        const auto forced_before = forced.size();
        const auto result =
            mesh::topology::collect_local_spatial_closure_forced_cells_2d(
                space.mesh_ref(),
                space.active_cells(),
                seed_cells);
        record_local_closure_metrics_2d(
            space,
            "fespace.refinement.collect_active_spatial_forced",
            result.stats);

        std::unordered_set<int> forced_seen;
        forced_seen.reserve(forced.size() + result.forced_cell_ids.size());
        for (const int cell_id : forced)
            forced_seen.insert(cell_id);

        for (const int cell_id : result.forced_cell_ids)
        {
            if (space.is_active_cell(cell_id) &&
                forced_seen.insert(cell_id).second)
            {
                forced.push_back(cell_id);
            }
        }

        space.record_timing_metric(
            "fespace.refinement.collect_active_spatial_forced.forced_refinements_added.count",
            static_cast<double>(forced.size() - forced_before));
    }

    struct ActiveWaveRefinementResult2D
    {
        bool changed = false;
        std::vector<int> added_active_cells{};
        std::vector<int> removed_active_cells{};
    };

    enum class EdgeQueryMode2D : std::uint8_t
    {
        ExactEdge = 0,
        AncestorEdge = 1,
        Containment = 2,
        OverlapCells = 3
    };

    enum class EdgeQueryCallSite2D : std::uint8_t
    {
        MainQueue = 0,
        PostFlushForcedClosure = 1,
        LocalVerifier = 2,
        Other = 3
    };

    enum class PostFlushEdgeProvenance2D : std::uint8_t
    {
        SplitRefinementEdgeChildSubedge = 0,
        InheritedParentBoundaryEdge = 1,
        InternalSiblingEdge = 2,
        NewlyCreatedInternalEdge = 3,
        TemporalOrNonLateralFace = 4,
        UnknownEdgeProvenance = 5
    };

    [[nodiscard]] inline constexpr std::size_t edge_query_mode_index_2d(
        const EdgeQueryMode2D mode) noexcept
    {
        return static_cast<std::size_t>(mode);
    }

    [[nodiscard]] inline constexpr std::string_view edge_query_mode_name_2d(
        const EdgeQueryMode2D mode) noexcept
    {
        switch (mode)
        {
        case EdgeQueryMode2D::ExactEdge:
            return "exact_edge";
        case EdgeQueryMode2D::AncestorEdge:
            return "ancestor_edge";
        case EdgeQueryMode2D::Containment:
            return "containment";
        case EdgeQueryMode2D::OverlapCells:
            return "same_spatial_overlap";
        }
        return "unknown";
    }

    [[nodiscard]] inline constexpr std::size_t edge_query_callsite_index_2d(
        const EdgeQueryCallSite2D call_site) noexcept
    {
        return static_cast<std::size_t>(call_site);
    }

    [[nodiscard]] inline constexpr std::string_view edge_query_callsite_name_2d(
        const EdgeQueryCallSite2D call_site) noexcept
    {
        switch (call_site)
        {
        case EdgeQueryCallSite2D::MainQueue:
            return "main_queue";
        case EdgeQueryCallSite2D::PostFlushForcedClosure:
            return "post_flush_forced_closure";
        case EdgeQueryCallSite2D::LocalVerifier:
            return "local_verifier";
        case EdgeQueryCallSite2D::Other:
            return "other";
        }
        return "unknown";
    }

    [[nodiscard]] inline constexpr std::size_t
    post_flush_edge_provenance_index_2d(
        const PostFlushEdgeProvenance2D provenance) noexcept
    {
        return static_cast<std::size_t>(provenance);
    }

    [[nodiscard]] inline constexpr std::string_view
    post_flush_edge_provenance_name_2d(
        const PostFlushEdgeProvenance2D provenance) noexcept
    {
        switch (provenance)
        {
        case PostFlushEdgeProvenance2D::SplitRefinementEdgeChildSubedge:
            return "split_refinement_edge_child_subedge";
        case PostFlushEdgeProvenance2D::InheritedParentBoundaryEdge:
            return "inherited_parent_boundary_edge";
        case PostFlushEdgeProvenance2D::InternalSiblingEdge:
            return "internal_sibling_edge";
        case PostFlushEdgeProvenance2D::NewlyCreatedInternalEdge:
            return "newly_created_internal_edge";
        case PostFlushEdgeProvenance2D::TemporalOrNonLateralFace:
            return "temporal_or_non_lateral_face";
        case PostFlushEdgeProvenance2D::UnknownEdgeProvenance:
            return "unknown_edge_provenance";
        }
        return "unknown";
    }

    struct EdgeQueryModeMetrics2D
    {
        std::size_t calls = 0;
        std::size_t cache_hits = 0;
        std::size_t cache_misses = 0;
        std::size_t candidate_records_visited = 0;
        std::size_t true_records_returned = 0;
        std::size_t duplicate_rejects = 0;
        std::size_t inactive_rejects = 0;
        std::size_t spatial_rejects = 0;
        std::size_t time_rejects = 0;
        std::size_t max_candidates_single_query = 0;
        double wall_seconds = 0.0;
    };

    struct EdgeQueryCallSiteMetrics2D
    {
        std::size_t calls = 0;
        std::size_t cache_hits = 0;
        std::size_t cache_misses = 0;
        std::size_t exact_edge_candidates = 0;
        std::size_t ancestor_edge_candidates = 0;
        std::size_t containment_candidates = 0;
        std::size_t true_records_returned = 0;
        std::size_t spatial_rejects = 0;
        std::size_t time_rejects = 0;
        std::size_t duplicate_rejects = 0;
        std::size_t record_contains_query_candidates = 0;
        std::size_t query_contains_record_candidates = 0;
        std::size_t record_contains_query_true_records = 0;
        std::size_t query_contains_record_true_records = 0;
        std::size_t record_contains_query_spatial_rejects = 0;
        std::size_t query_contains_record_spatial_rejects = 0;
        std::size_t bidirectional_records_later_discarded_by_main_closure = 0;
        double wall_seconds = 0.0;
    };

    struct PostFlushEdgeProvenanceMetrics2D
    {
        std::size_t considered_count = 0;
        std::size_t query_count = 0;
        std::size_t skipped_count = 0;
        std::size_t containment_candidates = 0;
        std::size_t true_records_returned = 0;
        std::size_t forced_cells_found = 0;
        std::size_t spatial_rejects = 0;
        std::size_t time_rejects = 0;
        std::size_t duplicate_rejects = 0;
        std::size_t parent_split_spatial_queries = 0;
        std::size_t parent_split_spacetime_queries = 0;
        std::size_t parent_split_temporal_queries = 0;
        std::size_t parent_split_other_queries = 0;
        std::array<std::size_t, 3> face_id_queries{};
        double wall_seconds = 0.0;
    };

    struct ForcedCellSourceMetrics2D
    {
        std::size_t total = 0;
        std::size_t from_split_edge = 0;
        std::size_t from_inherited_edge = 0;
        std::size_t discoverable_exact_child_edge = 0;
        std::size_t discoverable_exact_parent_edge = 0;
        std::size_t discoverable_presplit_parent_face = 0;
        std::size_t discoverable_split_batch = 0;
        std::size_t discoverable_same_spatial_time_overlap = 0;
        std::size_t discoverable_ancestor_edge_exact = 0;
        std::size_t discoverable_dyadic_edge_lookup = 0;
        std::size_t requires_broad_support_line = 0;
        std::size_t requires_query_contains_record = 0;
        std::size_t requires_record_contains_query = 0;
    };

    struct ActiveIndexedRefinementMetrics2D
    {
        std::size_t initially_marked_active_cells = 0;
        std::size_t queue_pops = 0;
        std::size_t unique_pending_cells_seen = 0;
        std::size_t repeated_pending_cell_pops = 0;
        std::size_t requeued_due_to_blockers = 0;
        std::size_t blockers_found = 0;
        std::size_t blockers_already_seen = 0;
        std::size_t closure_decision_cache_possible_count = 0;
        std::size_t blocker_cells = 0;
        std::size_t actually_split_active_cells = 0;
        std::size_t existing_children_reused = 0;
        std::size_t storage_children_created = 0;
        std::size_t edge_interval_index_queries = 0;
        std::size_t edge_interval_records_visited = 0;
        std::size_t edge_query_call_count = 0;
        std::size_t edge_query_count = 0;
        std::size_t edge_query_cache_hits = 0;
        std::size_t edge_query_cache_misses = 0;
        std::size_t edge_query_candidate_records = 0;
        std::size_t edge_query_true_records_returned = 0;
        std::size_t edge_query_true_records_returned_logical = 0;
        std::size_t edge_query_spatial_rejects = 0;
        std::size_t edge_query_time_rejects = 0;
        std::size_t edge_query_inactive_rejects = 0;
        std::size_t edge_query_duplicate_rejects = 0;
        std::size_t edge_query_duplicates_rejected = 0;
        std::size_t edge_query_max_candidates_single_query = 0;
        std::size_t edge_query_unique_query_keys = 0;
        std::size_t edge_query_repeated_same_key_count = 0;
        std::size_t edge_query_cache_entries = 0;
        std::size_t edge_query_cache_clear_count = 0;
        std::size_t edge_query_cache_memory_bytes_estimated = 0;
        std::size_t edge_query_visited_set_size_max = 0;
        std::size_t edge_query_exact_edge_queries = 0;
        std::size_t edge_query_ancestor_edge_queries = 0;
        std::size_t edge_query_containment_queries = 0;
        std::size_t edge_query_overlap_cell_queries = 0;
        std::size_t edge_query_ancestor_queries_skipped = 0;
        std::size_t edge_query_containment_queries_skipped = 0;
        std::size_t edge_query_early_time_rejects = 0;
        std::size_t edge_query_early_spatial_rejects = 0;
        std::size_t
            support_line_query_candidates_before_spatial_prune = 0;
        std::size_t
            support_line_query_candidates_after_spatial_prune = 0;
        std::size_t support_line_query_candidates_after_time_prune = 0;
        std::size_t support_line_interval_index_hits = 0;
        std::size_t support_line_interval_index_misses = 0;
        double containment_prune_wall_seconds = 0.0;
        double edge_query_duplicate_filter_wall_seconds = 0.0;
        std::array<EdgeQueryModeMetrics2D, 4> edge_query_mode_metrics{};
        std::array<EdgeQueryCallSiteMetrics2D, 4>
            edge_query_callsite_metrics{};
        std::array<PostFlushEdgeProvenanceMetrics2D, 6>
            post_flush_edge_provenance_metrics{};
        std::size_t post_flush_off_debug_mode = 0;
        std::size_t post_flush_split_edges_only_debug_mode = 0;
        std::size_t post_flush_split_and_inherited_edges_mode = 0;
        std::size_t post_flush_all_faces_debug_mode = 0;
        std::size_t post_flush_affected_edges_mode = 0;
        std::size_t post_flush_considered_edges_count = 0;
        std::size_t post_flush_query_count = 0;
        std::size_t post_flush_affected_edges_count = 0;
        std::size_t post_flush_all_faces_query_count = 0;
        std::size_t post_flush_skipped_edges_count = 0;
        std::size_t post_flush_skipped_internal_sibling_edges = 0;
        std::size_t post_flush_skipped_newly_created_internal_edges = 0;
        std::size_t post_flush_skipped_temporal_or_non_lateral_faces = 0;
        std::size_t post_flush_skipped_unknown_edges = 0;
        std::size_t post_flush_forced_cells_found = 0;
        std::size_t post_flush_directed_query_mode_used = 0;
        std::size_t post_flush_bidirectional_query_mode_used = 0;
        std::size_t post_flush_directed_query_candidates = 0;
        std::size_t post_flush_fallback_bidirectional_candidates = 0;
        std::size_t post_flush_affected_edge_cache_hits = 0;
        std::size_t post_flush_affected_edge_cache_misses = 0;
        std::size_t post_flush_exact_ancestor_queries_skipped = 0;
        std::size_t post_flush_presplit_neighbour_mode = 0;
        std::size_t post_flush_presplit_parent_face_queries = 0;
        std::size_t post_flush_presplit_parent_face_records = 0;
        std::size_t post_flush_presplit_records_used = 0;
        std::size_t post_flush_presplit_exact_child_ancestor_records_used = 0;
        std::size_t post_flush_forced_discovery_count = 0;
        std::size_t post_flush_unique_forced_cell_count = 0;
        std::size_t post_flush_forced_cells_returned = 0;
        std::size_t post_flush_forced_cells_enqueued = 0;
        std::size_t post_flush_forced_cells_split_later = 0;
        std::size_t post_flush_forced_cells_skipped_already_pending = 0;
        std::size_t post_flush_forced_cells_skipped_inactive = 0;
        std::size_t post_flush_forced_cells_skipped_already_split = 0;
        std::size_t post_flush_forced_cells_discovered_multiple_edges = 0;
        ForcedCellSourceMetrics2D forced_cell_sources{};
        ForcedCellSourceMetrics2D forced_cell_sources_split_edge{};
        ForcedCellSourceMetrics2D forced_cell_sources_inherited_edge{};
        std::size_t full_active_scans_in_normal_path = 0;
        std::size_t partition_update_calls = 0;
        std::size_t batched_split_cells = 0;
        std::size_t max_batch_size = 0;
        std::size_t batch_target_split_cells = 0;
        std::size_t flush_due_to_batch_limit_count = 0;
        std::size_t flush_due_to_queue_empty_count = 0;
        std::size_t flush_due_to_dependency_count = 0;
        std::size_t flush_due_to_cache_invalidation_count = 0;
    };

    template<typename FESpaceType>
    inline void record_indexed_refinement_metrics_2d(
        FESpaceType& space,
        const ActiveIndexedRefinementMetrics2D& metrics)
    {
        const auto record_count =
            [&](std::string_view name, const std::size_t value)
            {
                const double count = static_cast<double>(value);
                space.record_timing_metric(name, count);
                std::string count_name(name);
                count_name += ".count";
                space.record_timing_metric(count_name, count);
            };

        record_count(
            "refinement.initially_marked_active_cells",
            metrics.initially_marked_active_cells);
        record_count(
            "refinement.marked_cells",
            metrics.initially_marked_active_cells);
        record_count("refinement.queue_pops", metrics.queue_pops);
        record_count(
            "refinement.unique_pending_cells_seen",
            metrics.unique_pending_cells_seen);
        record_count(
            "refinement.repeated_pending_cell_pops",
            metrics.repeated_pending_cell_pops);
        record_count(
            "refinement.requeued_due_to_blockers",
            metrics.requeued_due_to_blockers);
        record_count(
            "refinement.blockers_found",
            metrics.blockers_found);
        record_count(
            "refinement.blockers_already_seen",
            metrics.blockers_already_seen);
        record_count(
            "refinement.closure_decision_cache_possible_count",
            metrics.closure_decision_cache_possible_count);
        record_count("refinement.blocker_cells", metrics.blocker_cells);
        record_count("refinement.closure_cells", metrics.blocker_cells);
        record_count(
            "refinement.actually_split_active_cells",
            metrics.actually_split_active_cells);
        record_count(
            "refinement.actually_split_cells",
            metrics.actually_split_active_cells);
        record_count(
            "refinement.existing_children_reused",
            metrics.existing_children_reused);
        record_count(
            "refinement.storage_children_created",
            metrics.storage_children_created);
        record_count(
            "refinement.edge_interval_index_queries",
            metrics.edge_interval_index_queries);
        record_count(
            "edge_interval_queries",
            metrics.edge_interval_index_queries);
        record_count(
            "refinement.edge_interval_records_visited",
            metrics.edge_interval_records_visited);
        record_count(
            "edge_interval_records_visited",
            metrics.edge_interval_records_visited);
        record_count("edge_query_count", metrics.edge_query_count);
        record_count(
            "edge_query_candidate_records",
            metrics.edge_query_candidate_records);
        record_count(
            "edge_query_true_records_returned",
            metrics.edge_query_true_records_returned);
        record_count(
            "edge_query_spatial_rejects",
            metrics.edge_query_spatial_rejects);
        record_count(
            "edge_query_time_rejects",
            metrics.edge_query_time_rejects);
        record_count(
            "edge_query_inactive_rejects",
            metrics.edge_query_inactive_rejects);
        record_count(
            "edge_query_duplicate_rejects",
            metrics.edge_query_duplicate_rejects);
        record_count(
            "edge_query_max_candidates_single_query",
            metrics.edge_query_max_candidates_single_query);
        record_count("edge_query.call_count", metrics.edge_query_call_count);
        record_count("edge_query.cache_hits", metrics.edge_query_cache_hits);
        record_count("edge_query.cache_misses", metrics.edge_query_cache_misses);
        record_count(
            "edge_query.candidate_records_visited",
            metrics.edge_query_candidate_records);
        record_count(
            "edge_query.true_records_returned",
            metrics.edge_query_true_records_returned_logical);
        record_count(
            "edge_query.duplicates_rejected",
            metrics.edge_query_duplicates_rejected);
        record_count(
            "edge_query.inactive_records_rejected",
            metrics.edge_query_inactive_rejects);
        record_count(
            "edge_query.spatial_rejects",
            metrics.edge_query_spatial_rejects);
        record_count(
            "edge_query.time_rejects",
            metrics.edge_query_time_rejects);
        record_count(
            "edge_query.max_candidates_single_query",
            metrics.edge_query_max_candidates_single_query);
        record_count(
            "edge_query.repeated_same_key_count",
            metrics.edge_query_repeated_same_key_count);
        record_count(
            "edge_query.unique_query_keys",
            metrics.edge_query_unique_query_keys);
        record_count("edge_query.cache_entries", metrics.edge_query_cache_entries);
        record_count(
            "edge_query.cache_clear_count",
            metrics.edge_query_cache_clear_count);
        space.record_timing_metric(
            "edge_query.cache_memory_bytes_estimated",
            static_cast<double>(
                metrics.edge_query_cache_memory_bytes_estimated));
        record_count(
            "edge_query.visited_set_size_max",
            metrics.edge_query_visited_set_size_max);
        record_count(
            "edge_query.exact_edge_queries",
            metrics.edge_query_exact_edge_queries);
        record_count(
            "edge_query.ancestor_edge_queries",
            metrics.edge_query_ancestor_edge_queries);
        record_count(
            "edge_query.containment_queries",
            metrics.edge_query_containment_queries);
        record_count(
            "edge_query.overlap_cell_queries",
            metrics.edge_query_overlap_cell_queries);
        record_count(
            "edge_query.ancestor_queries_skipped",
            metrics.edge_query_ancestor_queries_skipped);
        record_count(
            "edge_query.containment_queries_skipped",
            metrics.edge_query_containment_queries_skipped);
        record_count(
            "edge_query.early_time_rejects",
            metrics.edge_query_early_time_rejects);
        record_count(
            "edge_query.early_spatial_rejects",
            metrics.edge_query_early_spatial_rejects);
        record_count(
            "support_line_query_candidates_before_spatial_prune",
            metrics.support_line_query_candidates_before_spatial_prune);
        record_count(
            "support_line_query_candidates_after_spatial_prune",
            metrics.support_line_query_candidates_after_spatial_prune);
        record_count(
            "support_line_query_candidates_after_time_prune",
            metrics.support_line_query_candidates_after_time_prune);
        record_count(
            "support_line_interval_index_hits",
            metrics.support_line_interval_index_hits);
        record_count(
            "support_line_interval_index_misses",
            metrics.support_line_interval_index_misses);
        space.record_timing_metric(
            "containment_prune_wall",
            metrics.containment_prune_wall_seconds);
        space.record_timing_metric(
            "edge_query.duplicate_filter_wall",
            metrics.edge_query_duplicate_filter_wall_seconds);

        const double edge_query_candidate_to_true_ratio =
            metrics.edge_query_true_records_returned == 0
                ? 0.0
                : static_cast<double>(metrics.edge_query_candidate_records) /
                      static_cast<double>(metrics.edge_query_true_records_returned);
        const double edge_query_candidate_to_true_ratio_logical =
            metrics.edge_query_true_records_returned_logical == 0
                ? 0.0
                : static_cast<double>(metrics.edge_query_candidate_records) /
                      static_cast<double>(
                          metrics.edge_query_true_records_returned_logical);
        const double edge_query_mean_candidates =
            metrics.edge_query_count == 0
                ? 0.0
                : static_cast<double>(metrics.edge_query_candidate_records) /
                      static_cast<double>(metrics.edge_query_count);
        const double edge_query_mean_true_records =
            metrics.edge_query_call_count == 0
                ? 0.0
                : static_cast<double>(
                      metrics.edge_query_true_records_returned_logical) /
                      static_cast<double>(metrics.edge_query_call_count);
        const double edge_query_key_reuse_ratio =
            metrics.edge_query_call_count == 0
                ? 0.0
                : static_cast<double>(
                      metrics.edge_query_repeated_same_key_count) /
                      static_cast<double>(metrics.edge_query_call_count);
        space.record_timing_metric(
            "edge_query_candidate_to_true_ratio",
            edge_query_candidate_to_true_ratio);
        space.record_timing_metric(
            "edge_query_mean_candidates",
            edge_query_mean_candidates);
        space.record_timing_metric(
            "edge_query.candidate_to_true_ratio",
            edge_query_candidate_to_true_ratio_logical);
        space.record_timing_metric(
            "edge_query.mean_candidates_per_query",
            metrics.edge_query_call_count == 0
                ? 0.0
                : static_cast<double>(metrics.edge_query_candidate_records) /
                      static_cast<double>(metrics.edge_query_call_count));
        space.record_timing_metric(
            "edge_query.mean_true_records_per_query",
            edge_query_mean_true_records);
        space.record_timing_metric(
            "edge_query.query_key_reuse_ratio",
            edge_query_key_reuse_ratio);

        const auto record_mode_metrics =
            [&](const EdgeQueryMode2D mode)
            {
                const auto& mode_metrics =
                    metrics.edge_query_mode_metrics[
                        edge_query_mode_index_2d(mode)];
                const std::string prefix =
                    std::string("edge_query.mode.") +
                    std::string(edge_query_mode_name_2d(mode));

                record_count(prefix + ".calls", mode_metrics.calls);
                record_count(prefix + ".cache_hits", mode_metrics.cache_hits);
                record_count(
                    prefix + ".cache_misses",
                    mode_metrics.cache_misses);
                space.record_timing_metric(
                    prefix + ".wall",
                    mode_metrics.wall_seconds);
                record_count(
                    prefix + ".candidate_records_visited",
                    mode_metrics.candidate_records_visited);
                record_count(
                    prefix + ".true_records_returned",
                    mode_metrics.true_records_returned);
                record_count(
                    prefix + ".duplicate_rejects",
                    mode_metrics.duplicate_rejects);
                record_count(
                    prefix + ".inactive_rejects",
                    mode_metrics.inactive_rejects);
                record_count(
                    prefix + ".spatial_rejects",
                    mode_metrics.spatial_rejects);
                record_count(
                    prefix + ".time_rejects",
                    mode_metrics.time_rejects);
                record_count(
                    prefix + ".max_candidates_single_query",
                    mode_metrics.max_candidates_single_query);

                const double candidate_to_true_ratio =
                    mode_metrics.true_records_returned == 0
                        ? 0.0
                        : static_cast<double>(
                              mode_metrics.candidate_records_visited) /
                              static_cast<double>(
                                  mode_metrics.true_records_returned);
                const double mean_candidates =
                    mode_metrics.calls == 0
                        ? 0.0
                        : static_cast<double>(
                              mode_metrics.candidate_records_visited) /
                              static_cast<double>(mode_metrics.calls);
                space.record_timing_metric(
                    prefix + ".candidate_to_true_ratio",
                    candidate_to_true_ratio);
                space.record_timing_metric(
                    prefix + ".mean_candidates_per_query",
                    mean_candidates);
            };
        record_mode_metrics(EdgeQueryMode2D::ExactEdge);
        record_mode_metrics(EdgeQueryMode2D::AncestorEdge);
        record_mode_metrics(EdgeQueryMode2D::Containment);
        record_mode_metrics(EdgeQueryMode2D::OverlapCells);

        const auto record_callsite_metrics =
            [&](const EdgeQueryCallSite2D call_site)
            {
                const auto& callsite_metrics =
                    metrics.edge_query_callsite_metrics[
                        edge_query_callsite_index_2d(call_site)];
                const std::string prefix =
                    std::string("edge_query.callsite.") +
                    std::string(edge_query_callsite_name_2d(call_site));

                record_count(prefix + ".calls", callsite_metrics.calls);
                space.record_timing_metric(
                    prefix + ".wall",
                    callsite_metrics.wall_seconds);
                record_count(
                    prefix + ".cache_hits",
                    callsite_metrics.cache_hits);
                record_count(
                    prefix + ".cache_misses",
                    callsite_metrics.cache_misses);
                record_count(
                    prefix + ".exact_edge_candidates",
                    callsite_metrics.exact_edge_candidates);
                record_count(
                    prefix + ".ancestor_edge_candidates",
                    callsite_metrics.ancestor_edge_candidates);
                record_count(
                    prefix + ".containment_candidates",
                    callsite_metrics.containment_candidates);
                record_count(
                    prefix + ".true_records_returned",
                    callsite_metrics.true_records_returned);
                record_count(
                    prefix + ".spatial_rejects",
                    callsite_metrics.spatial_rejects);
                record_count(
                    prefix + ".time_rejects",
                    callsite_metrics.time_rejects);
                record_count(
                    prefix + ".duplicate_rejects",
                    callsite_metrics.duplicate_rejects);
                record_count(
                    prefix + ".record_contains_query_candidates",
                    callsite_metrics.record_contains_query_candidates);
                record_count(
                    prefix + ".query_contains_record_candidates",
                    callsite_metrics.query_contains_record_candidates);
                record_count(
                    prefix + ".record_contains_query_true_records",
                    callsite_metrics.record_contains_query_true_records);
                record_count(
                    prefix + ".query_contains_record_true_records",
                    callsite_metrics.query_contains_record_true_records);
                record_count(
                    prefix + ".record_contains_query_spatial_rejects",
                    callsite_metrics.record_contains_query_spatial_rejects);
                record_count(
                    prefix + ".query_contains_record_spatial_rejects",
                    callsite_metrics.query_contains_record_spatial_rejects);
                record_count(
                    prefix +
                        ".bidirectional_records_later_discarded_by_main_closure",
                    callsite_metrics
                        .bidirectional_records_later_discarded_by_main_closure);

                const double candidate_to_true_ratio =
                    callsite_metrics.true_records_returned == 0
                        ? 0.0
                        : static_cast<double>(
                              callsite_metrics.exact_edge_candidates +
                              callsite_metrics.ancestor_edge_candidates +
                              callsite_metrics.containment_candidates) /
                              static_cast<double>(
                                  callsite_metrics.true_records_returned);
                space.record_timing_metric(
                    prefix + ".candidate_to_true_ratio",
                    candidate_to_true_ratio);
            };
        record_callsite_metrics(EdgeQueryCallSite2D::MainQueue);
        record_callsite_metrics(EdgeQueryCallSite2D::PostFlushForcedClosure);
        record_callsite_metrics(EdgeQueryCallSite2D::LocalVerifier);
        record_callsite_metrics(EdgeQueryCallSite2D::Other);

        const auto record_post_flush_edge_provenance_metrics =
            [&](const PostFlushEdgeProvenance2D provenance)
            {
                const auto& provenance_metrics =
                    metrics.post_flush_edge_provenance_metrics[
                        post_flush_edge_provenance_index_2d(provenance)];
                const std::string prefix =
                    std::string("post_flush.edge_provenance.") +
                    std::string(post_flush_edge_provenance_name_2d(provenance));

                record_count(
                    prefix + ".considered_count",
                    provenance_metrics.considered_count);
                record_count(
                    prefix + ".query_count",
                    provenance_metrics.query_count);
                record_count(
                    prefix + ".skipped_count",
                    provenance_metrics.skipped_count);
                record_count(
                    prefix + ".containment_candidates",
                    provenance_metrics.containment_candidates);
                record_count(
                    prefix + ".true_records_returned",
                    provenance_metrics.true_records_returned);
                record_count(
                    prefix + ".forced_cells_found",
                    provenance_metrics.forced_cells_found);
                record_count(
                    prefix + ".spatial_rejects",
                    provenance_metrics.spatial_rejects);
                record_count(
                    prefix + ".time_rejects",
                    provenance_metrics.time_rejects);
                record_count(
                    prefix + ".duplicate_rejects",
                    provenance_metrics.duplicate_rejects);
                space.record_timing_metric(
                    prefix + ".wall",
                    provenance_metrics.wall_seconds);
                record_count(
                    prefix + ".parent_split_spatial_queries",
                    provenance_metrics.parent_split_spatial_queries);
                record_count(
                    prefix + ".parent_split_spacetime_queries",
                    provenance_metrics.parent_split_spacetime_queries);
                record_count(
                    prefix + ".parent_split_temporal_queries",
                    provenance_metrics.parent_split_temporal_queries);
                record_count(
                    prefix + ".parent_split_other_queries",
                    provenance_metrics.parent_split_other_queries);
                record_count(
                    prefix + ".face0_queries",
                    provenance_metrics.face_id_queries[0]);
                record_count(
                    prefix + ".face1_queries",
                    provenance_metrics.face_id_queries[1]);
                record_count(
                    prefix + ".face2_queries",
                    provenance_metrics.face_id_queries[2]);

                const double candidate_to_true_ratio =
                    provenance_metrics.true_records_returned == 0
                        ? 0.0
                        : static_cast<double>(
                              provenance_metrics.containment_candidates) /
                              static_cast<double>(
                                  provenance_metrics.true_records_returned);
                const double candidate_to_forced_cell_ratio =
                    provenance_metrics.forced_cells_found == 0
                        ? 0.0
                        : static_cast<double>(
                              provenance_metrics.containment_candidates) /
                              static_cast<double>(
                                  provenance_metrics.forced_cells_found);
                space.record_timing_metric(
                    prefix + ".candidate_to_true_ratio",
                    candidate_to_true_ratio);
                space.record_timing_metric(
                    prefix + ".candidate_to_forced_cell_ratio",
                    candidate_to_forced_cell_ratio);
            };
        record_post_flush_edge_provenance_metrics(
            PostFlushEdgeProvenance2D::SplitRefinementEdgeChildSubedge);
        record_post_flush_edge_provenance_metrics(
            PostFlushEdgeProvenance2D::InheritedParentBoundaryEdge);
        record_post_flush_edge_provenance_metrics(
            PostFlushEdgeProvenance2D::InternalSiblingEdge);
        record_post_flush_edge_provenance_metrics(
            PostFlushEdgeProvenance2D::NewlyCreatedInternalEdge);
        record_post_flush_edge_provenance_metrics(
            PostFlushEdgeProvenance2D::TemporalOrNonLateralFace);
        record_post_flush_edge_provenance_metrics(
            PostFlushEdgeProvenance2D::UnknownEdgeProvenance);

        const auto record_forced_cell_source_metrics =
            [&](const std::string& prefix,
                const ForcedCellSourceMetrics2D& source_metrics)
            {
                record_count(prefix + ".total", source_metrics.total);
                record_count(
                    prefix + ".from_split_edge",
                    source_metrics.from_split_edge);
                record_count(
                    prefix + ".from_inherited_edge",
                    source_metrics.from_inherited_edge);
                record_count(
                    prefix + ".discoverable_exact_child_edge",
                    source_metrics.discoverable_exact_child_edge);
                record_count(
                    prefix + ".discoverable_exact_parent_edge",
                    source_metrics.discoverable_exact_parent_edge);
                record_count(
                    prefix + ".discoverable_presplit_parent_face",
                    source_metrics.discoverable_presplit_parent_face);
                record_count(
                    prefix + ".discoverable_split_batch",
                    source_metrics.discoverable_split_batch);
                record_count(
                    prefix + ".discoverable_same_spatial_time_overlap",
                    source_metrics.discoverable_same_spatial_time_overlap);
                record_count(
                    prefix + ".discoverable_ancestor_edge_exact",
                    source_metrics.discoverable_ancestor_edge_exact);
                record_count(
                    prefix + ".discoverable_dyadic_edge_lookup",
                    source_metrics.discoverable_dyadic_edge_lookup);
                record_count(
                    prefix + ".requires_broad_support_line",
                    source_metrics.requires_broad_support_line);
                record_count(
                    prefix + ".requires_query_contains_record",
                    source_metrics.requires_query_contains_record);
                record_count(
                    prefix + ".requires_record_contains_query",
                    source_metrics.requires_record_contains_query);
            };
        record_forced_cell_source_metrics(
            "forced_cells",
            metrics.forced_cell_sources);
        record_forced_cell_source_metrics(
            "forced_cells.split_edge",
            metrics.forced_cell_sources_split_edge);
        record_forced_cell_source_metrics(
            "forced_cells.inherited_edge",
            metrics.forced_cell_sources_inherited_edge);

        record_count(
            "post_flush.off_debug_mode",
            metrics.post_flush_off_debug_mode);
        record_count(
            "post_flush.split_edges_only_debug_mode",
            metrics.post_flush_split_edges_only_debug_mode);
        record_count(
            "post_flush.split_and_inherited_edges_mode",
            metrics.post_flush_split_and_inherited_edges_mode);
        record_count(
            "post_flush.all_faces_debug_mode",
            metrics.post_flush_all_faces_debug_mode);
        record_count(
            "post_flush.affected_edges_mode",
            metrics.post_flush_affected_edges_mode);
        record_count(
            "post_flush.considered_edges_count",
            metrics.post_flush_considered_edges_count);
        record_count(
            "post_flush.query_count",
            metrics.post_flush_query_count);
        record_count(
            "post_flush.affected_edges_count",
            metrics.post_flush_affected_edges_count);
        record_count(
            "post_flush.all_faces_query_count",
            metrics.post_flush_all_faces_query_count);
        record_count(
            "post_flush.skipped_edges_count",
            metrics.post_flush_skipped_edges_count);
        record_count(
            "post_flush.skipped_internal_sibling_edges",
            metrics.post_flush_skipped_internal_sibling_edges);
        record_count(
            "post_flush.skipped_newly_created_internal_edges",
            metrics.post_flush_skipped_newly_created_internal_edges);
        record_count(
            "post_flush.skipped_temporal_or_non_lateral_faces",
            metrics.post_flush_skipped_temporal_or_non_lateral_faces);
        record_count(
            "post_flush.skipped_unknown_edges",
            metrics.post_flush_skipped_unknown_edges);
        record_count(
            "post_flush.forced_cells_found",
            metrics.post_flush_forced_cells_found);
        record_count(
            "post_flush.directed_query_mode_used",
            metrics.post_flush_directed_query_mode_used);
        record_count(
            "post_flush.bidirectional_query_mode_used",
            metrics.post_flush_bidirectional_query_mode_used);
        record_count(
            "post_flush.directed_query_candidates",
            metrics.post_flush_directed_query_candidates);
        record_count(
            "post_flush.fallback_bidirectional_candidates",
            metrics.post_flush_fallback_bidirectional_candidates);
        record_count(
            "post_flush.affected_edge_cache_hits",
            metrics.post_flush_affected_edge_cache_hits);
        record_count(
            "post_flush.affected_edge_cache_misses",
            metrics.post_flush_affected_edge_cache_misses);
        record_count(
            "post_flush.exact_ancestor_queries_skipped",
            metrics.post_flush_exact_ancestor_queries_skipped);
        record_count(
            "post_flush.presplit_neighbour_mode",
            metrics.post_flush_presplit_neighbour_mode);
        record_count(
            "post_flush.presplit_parent_face_queries",
            metrics.post_flush_presplit_parent_face_queries);
        record_count(
            "post_flush.presplit_parent_face_records",
            metrics.post_flush_presplit_parent_face_records);
        record_count(
            "post_flush.presplit_records_used",
            metrics.post_flush_presplit_records_used);
        record_count(
            "post_flush.presplit_exact_child_ancestor_records_used",
            metrics.post_flush_presplit_exact_child_ancestor_records_used);
        record_count(
            "post_flush.forced_discovery_count",
            metrics.post_flush_forced_discovery_count);
        record_count(
            "post_flush.unique_forced_cell_count",
            metrics.post_flush_unique_forced_cell_count);
        record_count(
            "post_flush.forced_cells_returned",
            metrics.post_flush_forced_cells_returned);
        record_count(
            "post_flush.forced_cells_enqueued",
            metrics.post_flush_forced_cells_enqueued);
        record_count(
            "post_flush.forced_cells_split_later",
            metrics.post_flush_forced_cells_split_later);
        record_count(
            "post_flush.forced_cells_skipped_already_pending",
            metrics.post_flush_forced_cells_skipped_already_pending);
        record_count(
            "post_flush.forced_cells_skipped_inactive",
            metrics.post_flush_forced_cells_skipped_inactive);
        record_count(
            "post_flush.forced_cells_skipped_already_split",
            metrics.post_flush_forced_cells_skipped_already_split);
        record_count(
            "post_flush.forced_cells_discovered_multiple_edges",
            metrics.post_flush_forced_cells_discovered_multiple_edges);
        space.record_timing_metric(
            "post_flush.candidates_per_affected_edge",
            metrics.post_flush_affected_edges_count == 0
                ? 0.0
                : static_cast<double>(
                      metrics.post_flush_fallback_bidirectional_candidates +
                      metrics.post_flush_directed_query_candidates) /
                      static_cast<double>(
                          metrics.post_flush_affected_edges_count));
        space.record_timing_metric(
            "post_flush.forced_cells_per_affected_edge",
            metrics.post_flush_affected_edges_count == 0
                ? 0.0
                : static_cast<double>(
                      metrics.post_flush_forced_cells_found) /
                      static_cast<double>(
                          metrics.post_flush_affected_edges_count));

        space.record_timing_metric(
            "exact_query_wall",
            metrics.edge_query_mode_metrics[
                edge_query_mode_index_2d(EdgeQueryMode2D::ExactEdge)]
                .wall_seconds);
        space.record_timing_metric(
            "ancestor_query_wall",
            metrics.edge_query_mode_metrics[
                edge_query_mode_index_2d(EdgeQueryMode2D::AncestorEdge)]
                .wall_seconds);
        space.record_timing_metric(
            "containment_query_wall",
            metrics.edge_query_mode_metrics[
                edge_query_mode_index_2d(EdgeQueryMode2D::Containment)]
                .wall_seconds);

        space.record_timing_metric(
            "refinement.queries_per_actually_split_cell",
            metrics.actually_split_active_cells == 0
                ? 0.0
                : static_cast<double>(metrics.edge_query_call_count) /
                      static_cast<double>(
                          metrics.actually_split_active_cells));
        space.record_timing_metric(
            "refinement.queries_per_marked_cell",
            metrics.initially_marked_active_cells == 0
                ? 0.0
                : static_cast<double>(metrics.edge_query_call_count) /
                      static_cast<double>(
                          metrics.initially_marked_active_cells));

        record_count(
            "refinement.active_cells",
            space.active_cells().size());

        const auto record_hash =
            [&](std::string_view name, const std::uint64_t value)
            {
                record_count(
                    std::string(name) + "_low32",
                    static_cast<std::size_t>(value & 0xffffffffULL));
                record_count(
                    std::string(name) + "_high32",
                    static_cast<std::size_t>(
                        (value >> 32) & 0xffffffffULL));
            };
        const auto hash_combine =
            [](std::uint64_t& seed, const std::uint64_t value)
            {
                seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) +
                        (seed >> 2);
            };

        std::vector<int> active_ids = space.active_cells();
        std::sort(active_ids.begin(), active_ids.end());
        active_ids.erase(
            std::unique(active_ids.begin(), active_ids.end()),
            active_ids.end());

        std::uint64_t active_cell_set_hash = 1469598103934665603ULL;
        std::uint64_t active_partition_hash = 1099511628211ULL;
        std::vector<std::uint64_t> active_geometry_cell_hashes;
        active_geometry_cell_hashes.reserve(active_ids.size());
        std::uint64_t generation_histogram_hash = 7809847782465536322ULL;
        std::vector<std::size_t> generation_histogram;
        int generation_min =
            active_ids.empty() ? 0 : std::numeric_limits<int>::max();
        int generation_max =
            active_ids.empty() ? 0 : std::numeric_limits<int>::min();
        const auto& mesh = space.mesh_ref();
        for (const int cell_id : active_ids)
        {
            hash_combine(
                active_cell_set_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(cell_id)));
            if (!mesh.valid_cell_id(cell_id))
                continue;
            const auto& cell = mesh.cell(cell_id);
            std::uint64_t cell_geometry_hash = 1469598103934665603ULL;
            generation_min = std::min(generation_min, cell.generation);
            generation_max = std::max(generation_max, cell.generation);
            if (cell.generation >= 0)
            {
                const auto generation =
                    static_cast<std::size_t>(cell.generation);
                if (generation_histogram.size() <= generation)
                    generation_histogram.resize(generation + 1, 0);
                ++generation_histogram[generation];
            }
            hash_combine(
                active_partition_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(cell_id)));
            hash_combine(
                active_partition_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(cell.parent_id)));
            hash_combine(
                active_partition_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(cell.generation)));
            hash_combine(
                active_partition_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(
                        cell.temporal_vertex_ids[0])));
            hash_combine(
                active_partition_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(
                        cell.temporal_vertex_ids[1])));
            for (const auto& face : cell.spatial_faces)
            {
                hash_combine(
                    active_partition_hash,
                    static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(
                            face.spatial_vertex_ids[0])));
                hash_combine(
                    active_partition_hash,
                    static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(
                            face.spatial_vertex_ids[1])));
            }

            using SpatialPoint =
                typename FESpaceType::MeshType::SpatialPoint;
            std::array<SpatialPoint, 3> spatial_points{};
            for (std::size_t i = 0; i < spatial_points.size(); ++i)
            {
                spatial_points[i] =
                    mesh.spatial_vertices()[static_cast<std::size_t>(
                        cell.spatial_vertex_ids[i])];
            }
            std::sort(
                spatial_points.begin(),
                spatial_points.end(),
                [](const SpatialPoint& a, const SpatialPoint& b)
                {
                    if (a[0] != b[0])
                        return a[0] < b[0];
                    return a[1] < b[1];
                });
            for (const auto& point : spatial_points)
            {
                hash_combine(
                    cell_geometry_hash,
                    std::bit_cast<std::uint64_t>(point[0]));
                hash_combine(
                    cell_geometry_hash,
                    std::bit_cast<std::uint64_t>(point[1]));
            }
            const double t0 =
                mesh.temporal_vertices()[static_cast<std::size_t>(
                    cell.temporal_vertex_ids[0])][0];
            const double t1 =
                mesh.temporal_vertices()[static_cast<std::size_t>(
                    cell.temporal_vertex_ids[1])][0];
            hash_combine(
                cell_geometry_hash,
                std::bit_cast<std::uint64_t>(std::min(t0, t1)));
            hash_combine(
                cell_geometry_hash,
                std::bit_cast<std::uint64_t>(std::max(t0, t1)));
            hash_combine(
                cell_geometry_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(cell.generation)));
            hash_combine(
                cell_geometry_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(cell.spatial_level)));
            hash_combine(
                cell_geometry_hash,
                static_cast<std::uint64_t>(
                    static_cast<std::uint32_t>(cell.temporal_level)));
            active_geometry_cell_hashes.push_back(cell_geometry_hash);
        }
        std::sort(
            active_geometry_cell_hashes.begin(),
            active_geometry_cell_hashes.end());
        std::uint64_t active_partition_geometry_hash =
            1469598103934665603ULL;
        for (const auto cell_hash : active_geometry_cell_hashes)
            hash_combine(active_partition_geometry_hash, cell_hash);
        for (std::size_t generation = 0;
             generation < generation_histogram.size();
             ++generation)
        {
            hash_combine(generation_histogram_hash, generation);
            hash_combine(
                generation_histogram_hash,
                generation_histogram[generation]);
        }
        record_count("refinement.active_cell_count_after", active_ids.size());
        record_hash("refinement.active_cell_set_hash", active_cell_set_hash);
        record_hash("refinement.active_partition_hash", active_partition_hash);
        record_hash(
            "refinement.active_partition_geometry_hash",
            active_partition_geometry_hash);
        record_hash(
            "refinement.generation_histogram_hash",
            generation_histogram_hash);
        record_count(
            "refinement.generation_min_after",
            static_cast<std::size_t>(std::max(0, generation_min)));
        record_count(
            "refinement.generation_max_after",
            static_cast<std::size_t>(std::max(0, generation_max)));
        record_count(
            "refinement.generation_distinct_after",
            generation_histogram.size());

        const auto& edge_index =
            space.partition_view().active_edge_interval_index_2d();
        record_count(
            "refinement.edge_index_active_records",
            edge_index.n_active_records());
        record_count(
            "refinement.edge_index_inactive_records",
            edge_index.n_inactive_records());

        if (space.detailed_fespace_diagnostics_enabled())
        {
            space.time_phase(
                "partition.metrics.edge_index_stats_wall",
                [&]()
                {
                    const auto support_stats =
                        edge_index.support_line_group_stats();
                    record_count(
                        "support_line_group_count",
                        static_cast<std::size_t>(support_stats.group_count));
                    record_count(
                        "support_line_group_max_size",
                        static_cast<std::size_t>(support_stats.max_size));
                    record_count(
                        "support_line_group_active_records",
                        static_cast<std::size_t>(support_stats.active_records));
                    record_count(
                        "support_line_group_inactive_records",
                        static_cast<std::size_t>(
                            support_stats.inactive_records));
                    record_count(
                        "support_line_group_max_inactive_records",
                        static_cast<std::size_t>(
                            support_stats.max_inactive_records));
                    space.record_timing_metric(
                        "support_line_group_mean_size",
                        support_stats.mean_size);
                    space.record_timing_metric(
                        "support_line_group_tombstone_fraction",
                        support_stats.tombstone_fraction);
                    space.record_timing_metric(
                        "support_line_group_max_inactive_fraction",
                        support_stats.max_inactive_fraction);

                    const auto edge_group_stats =
                        edge_index.edge_group_stats();
                    record_count(
                        "edge_group_count",
                        static_cast<std::size_t>(
                            edge_group_stats.group_count));
                    record_count(
                        "edge_group_max_size",
                        static_cast<std::size_t>(edge_group_stats.max_size));
                    record_count(
                        "edge_group_active_records",
                        static_cast<std::size_t>(
                            edge_group_stats.active_records));
                    record_count(
                        "edge_group_inactive_records",
                        static_cast<std::size_t>(
                            edge_group_stats.inactive_records));
                    record_count(
                        "edge_group_max_inactive_records",
                        static_cast<std::size_t>(
                            edge_group_stats.max_inactive_records));
                    space.record_timing_metric(
                        "edge_group_mean_size",
                        edge_group_stats.mean_size);
                    space.record_timing_metric(
                        "edge_group_tombstone_fraction",
                        edge_group_stats.tombstone_fraction);
                    space.record_timing_metric(
                        "edge_group_max_inactive_fraction",
                        edge_group_stats.max_inactive_fraction);

                    const auto spatial_vertex_group_stats =
                        edge_index.spatial_vertex_group_stats();
                    record_count(
                        "spatial_vertex_group_count",
                        static_cast<std::size_t>(
                            spatial_vertex_group_stats.group_count));
                    record_count(
                        "spatial_vertex_group_max_size",
                        static_cast<std::size_t>(
                            spatial_vertex_group_stats.max_size));
                    record_count(
                        "spatial_vertex_group_active_records",
                        static_cast<std::size_t>(
                            spatial_vertex_group_stats.active_records));
                    record_count(
                        "spatial_vertex_group_inactive_records",
                        static_cast<std::size_t>(
                            spatial_vertex_group_stats.inactive_records));
                    record_count(
                        "spatial_vertex_group_max_inactive_records",
                        static_cast<std::size_t>(
                            spatial_vertex_group_stats
                                .max_inactive_records));
                    space.record_timing_metric(
                        "spatial_vertex_group_mean_size",
                        spatial_vertex_group_stats.mean_size);
                    space.record_timing_metric(
                        "spatial_vertex_group_tombstone_fraction",
                        spatial_vertex_group_stats.tombstone_fraction);
                    space.record_timing_metric(
                        "spatial_vertex_group_max_inactive_fraction",
                        spatial_vertex_group_stats.max_inactive_fraction);
                });
        }
        else
        {
            space.record_timing_metric(
                "partition.metrics.edge_index_stats_wall",
                0.0);
        }

        record_count(
            "refinement.full_active_scans_in_normal_path",
            metrics.full_active_scans_in_normal_path);
        record_count(
            "refinement.full_active_scans",
            metrics.full_active_scans_in_normal_path);
        record_count(
            "full_active_scans",
            metrics.full_active_scans_in_normal_path);
        record_count(
            "refinement.partition_update_calls",
            metrics.partition_update_calls);
        record_count(
            "refinement.batched_split_cells",
            metrics.batched_split_cells);
        record_count(
            "refinement.max_batch_size",
            metrics.max_batch_size);
        record_count(
            "refinement.batch_target_split_cells",
            metrics.batch_target_split_cells);
        record_count(
            "refinement.flush_due_to_batch_limit_count",
            metrics.flush_due_to_batch_limit_count);
        record_count(
            "refinement.flush_due_to_queue_empty_count",
            metrics.flush_due_to_queue_empty_count);
        record_count(
            "refinement.flush_due_to_dependency_count",
            metrics.flush_due_to_dependency_count);
        record_count(
            "refinement.flush_due_to_cache_invalidation_count",
            metrics.flush_due_to_cache_invalidation_count);
        const double average_batch_size =
            metrics.partition_update_calls == 0
                ? 0.0
                : static_cast<double>(metrics.batched_split_cells) /
                      static_cast<double>(metrics.partition_update_calls);
        space.record_timing_metric(
            "refinement.average_batch_size",
            average_batch_size);
    }

    template<typename ActiveEdgeIntervalIndex2DType>
    inline void accumulate_edge_query_quality_metrics_2d(
        ActiveIndexedRefinementMetrics2D& metrics,
        const ActiveEdgeIntervalIndex2DType& index)
    {
        const auto& quality = index.last_query_quality();
        metrics.edge_query_count +=
            static_cast<std::size_t>(quality.query_count);
        metrics.edge_query_candidate_records +=
            static_cast<std::size_t>(quality.candidate_records);
        metrics.edge_query_true_records_returned +=
            static_cast<std::size_t>(quality.true_records_returned);
        metrics.edge_query_spatial_rejects +=
            static_cast<std::size_t>(quality.spatial_rejects);
        metrics.edge_query_time_rejects +=
            static_cast<std::size_t>(quality.time_rejects);
        metrics.edge_query_inactive_rejects +=
            static_cast<std::size_t>(quality.inactive_rejects);
        metrics.edge_query_duplicate_rejects +=
            static_cast<std::size_t>(quality.duplicate_rejects);
        metrics.edge_query_max_candidates_single_query =
            std::max(
                metrics.edge_query_max_candidates_single_query,
                static_cast<std::size_t>(
                    quality.max_candidates_single_query));
        metrics.support_line_query_candidates_before_spatial_prune +=
            static_cast<std::size_t>(
                quality
                    .support_line_query_candidates_before_spatial_prune);
        metrics.support_line_query_candidates_after_spatial_prune +=
            static_cast<std::size_t>(
                quality
                    .support_line_query_candidates_after_spatial_prune);
        metrics.support_line_query_candidates_after_time_prune +=
            static_cast<std::size_t>(
                quality.support_line_query_candidates_after_time_prune);
        metrics.support_line_interval_index_hits +=
            static_cast<std::size_t>(
                quality.support_line_interval_index_hits);
        metrics.support_line_interval_index_misses +=
            static_cast<std::size_t>(
                quality.support_line_interval_index_misses);
        metrics.containment_prune_wall_seconds +=
            quality.containment_prune_seconds;
    }

    template<typename ActiveEdgeIntervalIndex2DType>
    inline void accumulate_edge_query_mode_quality_metrics_2d(
        EdgeQueryModeMetrics2D& mode_metrics,
        const ActiveEdgeIntervalIndex2DType& index)
    {
        const auto& quality = index.last_query_quality();
        mode_metrics.candidate_records_visited +=
            static_cast<std::size_t>(quality.candidate_records);
        mode_metrics.true_records_returned +=
            static_cast<std::size_t>(quality.true_records_returned);
        mode_metrics.spatial_rejects +=
            static_cast<std::size_t>(quality.spatial_rejects);
        mode_metrics.time_rejects +=
            static_cast<std::size_t>(quality.time_rejects);
        mode_metrics.inactive_rejects +=
            static_cast<std::size_t>(quality.inactive_rejects);
        mode_metrics.duplicate_rejects +=
            static_cast<std::size_t>(quality.duplicate_rejects);
        mode_metrics.max_candidates_single_query =
            std::max(
                mode_metrics.max_candidates_single_query,
                static_cast<std::size_t>(
                    quality.max_candidates_single_query));
    }

    template<typename ActiveEdgeIntervalIndex2DType>
    inline void accumulate_edge_query_callsite_quality_metrics_2d(
        EdgeQueryCallSiteMetrics2D& callsite_metrics,
        const EdgeQueryMode2D mode,
        const ActiveEdgeIntervalIndex2DType& index)
    {
        const auto& quality = index.last_query_quality();
        const std::size_t candidates =
            static_cast<std::size_t>(quality.candidate_records);
        switch (mode)
        {
        case EdgeQueryMode2D::ExactEdge:
            callsite_metrics.exact_edge_candidates += candidates;
            break;
        case EdgeQueryMode2D::AncestorEdge:
            callsite_metrics.ancestor_edge_candidates += candidates;
            break;
        case EdgeQueryMode2D::Containment:
            callsite_metrics.containment_candidates += candidates;
            callsite_metrics.record_contains_query_candidates +=
                static_cast<std::size_t>(
                    quality.record_contains_query_candidates);
            callsite_metrics.query_contains_record_candidates +=
                static_cast<std::size_t>(
                    quality.query_contains_record_candidates);
            callsite_metrics.record_contains_query_true_records +=
                static_cast<std::size_t>(
                    quality.record_contains_query_true_records);
            callsite_metrics.query_contains_record_true_records +=
                static_cast<std::size_t>(
                    quality.query_contains_record_true_records);
            callsite_metrics.record_contains_query_spatial_rejects +=
                static_cast<std::size_t>(
                    quality.record_contains_query_spatial_rejects);
            callsite_metrics.query_contains_record_spatial_rejects +=
                static_cast<std::size_t>(
                    quality.query_contains_record_spatial_rejects);
            break;
        case EdgeQueryMode2D::OverlapCells:
            break;
        }
        callsite_metrics.spatial_rejects +=
            static_cast<std::size_t>(quality.spatial_rejects);
        callsite_metrics.time_rejects +=
            static_cast<std::size_t>(quality.time_rejects);
        callsite_metrics.duplicate_rejects +=
            static_cast<std::size_t>(quality.duplicate_rejects);
    }

    template<typename GeomTraits>
    struct EdgeQueryCacheKey2D
    {
        using EdgeKey = mesh::topology::SpatialEdgeKey2D<GeomTraits>;

        EdgeKey edge_key{};
        int temporal_v0 = -1;
        int temporal_v1 = -1;
        std::uint64_t active_index_version = 0;
        EdgeQueryMode2D mode = EdgeQueryMode2D::ExactEdge;

        bool operator==(const EdgeQueryCacheKey2D&) const noexcept = default;
    };

    template<typename GeomTraits>
    struct EdgeQueryCacheKeyHash2D
    {
        [[nodiscard]] std::size_t operator()(
            const EdgeQueryCacheKey2D<GeomTraits>& key) const noexcept
        {
            std::size_t seed =
                mesh::topology::SpatialEdgeKey2DHash<GeomTraits>{}(
                    key.edge_key);
            const auto combine =
                [&](const auto value)
                {
                    using ValueType = std::decay_t<decltype(value)>;
                    seed ^= std::hash<ValueType>{}(value) +
                            0x9e3779b97f4a7c15ULL + (seed << 6) +
                            (seed >> 2);
                };
            combine(key.temporal_v0);
            combine(key.temporal_v1);
            combine(key.active_index_version);
            combine(static_cast<std::uint8_t>(key.mode));
            return seed;
        }
    };

    struct EdgeRecordVisitKey2D
    {
        int cell_id = -1;
        int face_id = -1;

        bool operator==(const EdgeRecordVisitKey2D&) const noexcept = default;
    };

    struct EdgeRecordVisitKeyHash2D
    {
        [[nodiscard]] std::size_t operator()(
            const EdgeRecordVisitKey2D& key) const noexcept
        {
            std::size_t seed = std::hash<int>{}(key.cell_id);
            seed ^= std::hash<int>{}(key.face_id) + 0x9e3779b97f4a7c15ULL +
                    (seed << 6) + (seed >> 2);
            return seed;
        }
    };

    template<typename FESpaceType>
    struct ActiveEdgeQueryContext2D
    {
        using GeomTraits = typename FESpaceType::GT;
        using EdgeKey = mesh::topology::SpatialEdgeKey2D<GeomTraits>;
        using Key = EdgeQueryCacheKey2D<GeomTraits>;
        using KeyHash = EdgeQueryCacheKeyHash2D<GeomTraits>;
        using Record =
            typename FESpaceType::PartitionViewType
                ::ActiveEdgeIntervalIndex2DType::Record;

        bool cache_enabled = true;
        std::uint64_t active_index_version = 0;
        std::unordered_map<Key, std::vector<Record>, KeyHash> cache{};
        std::unordered_set<Key, KeyHash> seen_query_keys{};

        void record_query_key(
            const Key& key,
            ActiveIndexedRefinementMetrics2D& metrics)
        {
            if (seen_query_keys.insert(key).second)
                metrics.edge_query_unique_query_keys =
                    seen_query_keys.size();
            else
                ++metrics.edge_query_repeated_same_key_count;
        }

        void clear_after_active_index_update(
            ActiveIndexedRefinementMetrics2D& metrics)
        {
            ++active_index_version;
            cache.clear();
            ++metrics.edge_query_cache_clear_count;
        }

        [[nodiscard]] std::size_t estimated_cache_memory_bytes() const
        {
            std::size_t bytes =
                cache.bucket_count() *
                (sizeof(void*) + sizeof(std::size_t));
            for (const auto& [key, records] : cache)
            {
                static_cast<void>(key);
                bytes += sizeof(Key) + sizeof(std::vector<Record>) +
                         records.capacity() * sizeof(Record);
            }
            return bytes;
        }
    };

    template<typename FESpaceType, typename QueryFn>
    [[nodiscard]] inline std::vector<
        typename FESpaceType::PartitionViewType::ActiveEdgeIntervalIndex2DType::Record>
    cached_edge_query_records_2d(
        FESpaceType& space,
        ActiveEdgeQueryContext2D<FESpaceType>& query_context,
        ActiveIndexedRefinementMetrics2D& metrics,
        const EdgeQueryCallSite2D call_site,
        const EdgeQueryCacheKey2D<typename FESpaceType::GT>& key,
        QueryFn&& query_fn)
    {
        using Record =
            typename FESpaceType::PartitionViewType
                ::ActiveEdgeIntervalIndex2DType::Record;
        using Clock = std::chrono::steady_clock;

        const auto mode_start = Clock::now();
        auto& mode_metrics =
            metrics.edge_query_mode_metrics[edge_query_mode_index_2d(
                key.mode)];
        auto& callsite_metrics =
            metrics.edge_query_callsite_metrics[
                edge_query_callsite_index_2d(call_site)];
        ++mode_metrics.calls;
        ++callsite_metrics.calls;
        ++metrics.edge_query_call_count;
        query_context.record_query_key(key, metrics);

        if (query_context.cache_enabled)
        {
            const auto cached = query_context.cache.find(key);
            if (cached != query_context.cache.end())
            {
                ++mode_metrics.cache_hits;
                ++callsite_metrics.cache_hits;
                ++metrics.edge_query_cache_hits;
                mode_metrics.true_records_returned += cached->second.size();
                const double wall_seconds =
                    std::chrono::duration<double>(
                        Clock::now() - mode_start)
                        .count();
                mode_metrics.wall_seconds += wall_seconds;
                callsite_metrics.wall_seconds += wall_seconds;
                callsite_metrics.true_records_returned += cached->second.size();
                metrics.edge_query_true_records_returned_logical +=
                    cached->second.size();
                return cached->second;
            }
        }

        ++mode_metrics.cache_misses;
        ++callsite_metrics.cache_misses;
        ++metrics.edge_query_cache_misses;
        ++metrics.edge_interval_index_queries;
        std::vector<Record> records;
        space.time_phase(
            "refinement.edge_index_query_wall",
            [&]()
            {
                space.time_phase(
                    "edge_query.wall",
                    [&]()
                    {
                        space.time_phase(
                            "edge_index.query_total",
                            [&]()
                            {
                                records = std::forward<QueryFn>(query_fn)();
                            });
                    });
            });

        metrics.edge_interval_records_visited +=
            space.partition_view()
                .active_edge_interval_index_2d()
                .last_query_records_visited();
        accumulate_edge_query_quality_metrics_2d(
            metrics,
            space.partition_view().active_edge_interval_index_2d());
        accumulate_edge_query_mode_quality_metrics_2d(
            mode_metrics,
            space.partition_view().active_edge_interval_index_2d());
        accumulate_edge_query_callsite_quality_metrics_2d(
            callsite_metrics,
            key.mode,
            space.partition_view().active_edge_interval_index_2d());
        metrics.edge_query_true_records_returned_logical += records.size();
        callsite_metrics.true_records_returned += records.size();

        if (query_context.cache_enabled)
        {
            query_context.cache.emplace(key, records);
            metrics.edge_query_cache_entries =
                std::max(
                    metrics.edge_query_cache_entries,
                    query_context.cache.size());
            metrics.edge_query_cache_memory_bytes_estimated =
                std::max(
                    metrics.edge_query_cache_memory_bytes_estimated,
                    query_context.estimated_cache_memory_bytes());
        }

        const double wall_seconds =
            std::chrono::duration<double>(Clock::now() - mode_start).count();
        mode_metrics.wall_seconds += wall_seconds;
        callsite_metrics.wall_seconds += wall_seconds;

        return records;
    }

    struct PendingActiveRefinementBatch2D
    {
        std::unordered_set<int> add_ids{};
        std::unordered_set<int> remove_ids{};
        std::vector<int> added_seed_cells{};
        std::size_t split_cell_count = 0;

        [[nodiscard]] bool empty() const noexcept
        {
            return add_ids.empty() && remove_ids.empty();
        }

        void clear()
        {
            add_ids.clear();
            remove_ids.clear();
            added_seed_cells.clear();
            split_cell_count = 0;
        }
    };

    [[nodiscard]] inline bool refinement_has_spatial_part_2d(
        const mesh::RefinementType refinement_type) noexcept
    {
        return refinement_type == mesh::RefinementType::spatial ||
               refinement_type == mesh::RefinementType::spacetime;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline mesh::RefinementType effective_refinement_type_2d(
        const mesh::Mesh<GeomTraits>& mesh,
        const int cell_id,
        const mesh::RefinementType requested_refinement_type)
    {
        if (requested_refinement_type == mesh::RefinementType::none)
            return mesh::refinement::next_split_type<GeomTraits>(
                mesh.cell(cell_id).generation);
        return requested_refinement_type;
    }

    [[nodiscard]] inline bool supported_active_refinement_type_2d(
        const mesh::RefinementType refinement_type) noexcept
    {
        return refinement_type == mesh::RefinementType::spatial ||
               refinement_type == mesh::RefinementType::temporal ||
               refinement_type == mesh::RefinementType::spacetime;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline mesh::topology::SpatialEdgeKey2D<GeomTraits>
    active_spatial_refinement_edge_key_2d(
        const mesh::Mesh<GeomTraits>& mesh,
        const int cell_id)
    {
        return mesh::topology::make_spatial_edge_key_2d<GeomTraits>(
            mesh.spatial_refinement_edge(cell_id));
    }

    template<typename GeomTraits>
    [[nodiscard]] inline std::pair<double, double>
    active_temporal_interval_bounds_2d(
        const mesh::Mesh<GeomTraits>& mesh,
        const int cell_id)
    {
        return mesh::topology::temporal_interval_bounds_2d<GeomTraits>(
            mesh,
            mesh.cell(cell_id));
    }

    template<typename FESpaceType>
    [[nodiscard]] inline std::vector<
        typename FESpaceType::PartitionViewType::ActiveEdgeIntervalIndex2DType::Record>
    overlap_records_for_cell_edge_indexed_2d(
        FESpaceType& space,
        const int cell_id,
        const typename mesh::MeshTypes<typename FESpaceType::GT>::SpatialFaceVertexIds&
            edge,
        const double t0,
        const double t1,
        ActiveIndexedRefinementMetrics2D& metrics,
        ActiveEdgeQueryContext2D<FESpaceType>& query_context,
        const EdgeQueryCallSite2D call_site)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;
        using Record =
            typename FESpaceType::PartitionViewType
                ::ActiveEdgeIntervalIndex2DType::Record;
        using Clock = std::chrono::steady_clock;

        auto& mesh = space.mesh_ref();
        const auto& query_cell = mesh.cell(cell_id);
        const int temporal_v0 =
            static_cast<int>(query_cell.temporal_vertex_ids[0]);
        const int temporal_v1 =
            static_cast<int>(query_cell.temporal_vertex_ids[1]);
        std::unordered_set<EdgeRecordVisitKey2D, EdgeRecordVisitKeyHash2D>
            visited_records;

        const auto append_records =
            [&](std::vector<Record>& out,
                const std::vector<Record>& records,
                const EdgeQueryMode2D mode)
            {
                const auto start = Clock::now();
                auto& mode_metrics =
                    metrics.edge_query_mode_metrics[
                        edge_query_mode_index_2d(mode)];
                auto& callsite_metrics =
                    metrics.edge_query_callsite_metrics[
                        edge_query_callsite_index_2d(call_site)];
                visited_records.reserve(
                    std::max(
                        visited_records.size() + records.size(),
                        std::size_t{8}));
                for (const auto& record : records)
                {
                    const EdgeRecordVisitKey2D key{
                        record.cell_id,
                        record.face_id};
                    if (visited_records.insert(key).second)
                        out.push_back(record);
                    else
                    {
                        ++mode_metrics.duplicate_rejects;
                        ++callsite_metrics.duplicate_rejects;
                        ++metrics.edge_query_duplicate_rejects;
                        ++metrics.edge_query_duplicates_rejected;
                    }
                }
                metrics.edge_query_visited_set_size_max =
                    std::max(
                        metrics.edge_query_visited_set_size_max,
                        visited_records.size());
                metrics.edge_query_duplicate_filter_wall_seconds +=
                    std::chrono::duration<double>(Clock::now() - start)
                        .count();
            };

        std::vector<Edge> query_edges;
        query_edges.reserve(8);
        const Edge sorted_edge =
            mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                edge);
        query_edges.push_back(sorted_edge);

        int ancestor_id = mesh.cell(cell_id).parent_id;
        while (ancestor_id >= 0 && mesh.valid_cell_id(ancestor_id))
        {
            const auto& ancestor = mesh.cell(ancestor_id);
            for (int face_id = 0; face_id < 3; ++face_id)
            {
                const Edge ancestor_edge =
                    mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                        ancestor
                            .spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);
                if (!mesh::topology::spatial_edge_lies_on_boundary_edge_2d<
                        GeomTraits>(
                        sorted_edge,
                        ancestor_edge,
                        mesh.spatial_vertices()))
                {
                    continue;
                }

                const auto exists =
                    std::find(query_edges.begin(), query_edges.end(), ancestor_edge) !=
                    query_edges.end();
                if (!exists)
                    query_edges.push_back(ancestor_edge);
            }

            ancestor_id = ancestor.parent_id;
        }

        std::vector<Record> records;
        for (std::size_t query_edge_index = 0;
             query_edge_index < query_edges.size();
             ++query_edge_index)
        {
            const Edge& query_edge = query_edges[query_edge_index];
            const auto query_key =
                mesh::topology::SpatialEdgeKey2D<GeomTraits>{query_edge};
            const bool exact_query = query_edge_index == 0;
            if (exact_query)
                ++metrics.edge_query_exact_edge_queries;
            else
                ++metrics.edge_query_ancestor_edge_queries;
            const EdgeQueryCacheKey2D<GeomTraits> cache_key{
                query_key,
                temporal_v0,
                temporal_v1,
                query_context.active_index_version,
                exact_query
                    ? EdgeQueryMode2D::ExactEdge
                    : EdgeQueryMode2D::AncestorEdge};
            const auto exact_records =
                cached_edge_query_records_2d(
                    space,
                    query_context,
                    metrics,
                    call_site,
                    cache_key,
                    [&]()
                    {
                        return space.partition_view()
                            .active_edge_interval_index_2d()
                            .overlap_records(query_key, t0, t1);
                    });
            append_records(records, exact_records, cache_key.mode);
        }
        if (query_edges.size() == 1)
            ++metrics.edge_query_ancestor_queries_skipped;

        if (call_site == EdgeQueryCallSite2D::MainQueue &&
            !space.main_closure_query_uses_containment_2d())
        {
            return records;
        }

        const auto edge_key =
            mesh::topology::SpatialEdgeKey2D<GeomTraits>{sorted_edge};
        ++metrics.edge_query_containment_queries;
        const EdgeQueryCacheKey2D<GeomTraits> containment_cache_key{
            edge_key,
            temporal_v0,
            temporal_v1,
            query_context.active_index_version,
            EdgeQueryMode2D::Containment};
        const auto containment_records =
            cached_edge_query_records_2d(
                space,
                query_context,
                metrics,
                call_site,
                containment_cache_key,
                [&]()
                {
                    return space.partition_view()
                        .active_edge_interval_index_2d()
                        .overlap_records_with_spatial_edge_containment(
                            edge_key,
                            t0,
                            t1);
                });
        append_records(
            records,
            containment_records,
            EdgeQueryMode2D::Containment);

        return records;
    }

    template<typename GeomTraits>
    [[nodiscard]] inline std::vector<int>
    mesh_create_children_if_needed_for_active_refinement(
        mesh::Mesh<GeomTraits>& mesh,
        const int cell_id,
        const mesh::RefinementType requested_refinement_type,
        ActiveIndexedRefinementMetrics2D& metrics)
    {
        if (!mesh.valid_cell_id(cell_id))
            throw std::runtime_error(
                "FESpace::refine_2d: active refinement cell id out of range.");

        const auto effective_type =
            effective_refinement_type_2d(
                mesh,
                cell_id,
                requested_refinement_type);
        if (!supported_active_refinement_type_2d(effective_type))
            throw std::runtime_error(
                "FESpace::refine_2d: unsupported 2D active refinement type.");

        const auto& cell = mesh.cell(cell_id);
        const bool was_storage_leaf = cell.is_leaf;
        if (!was_storage_leaf)
        {
            if (cell.children.empty())
                throw std::runtime_error(
                    "FESpace::refine_2d: active non-leaf cell has no children to reuse.");
            if (requested_refinement_type != mesh::RefinementType::none &&
                cell.last_split_type != effective_type)
            {
                throw std::runtime_error(
                    "FESpace::refine_2d: active non-leaf children use an incompatible split type.");
            }
        }

        auto children =
            mesh.create_children_if_needed(cell_id, requested_refinement_type);
        if (was_storage_leaf)
            ++metrics.storage_children_created;
        else
            ++metrics.existing_children_reused;

        if (children.empty())
            throw std::runtime_error(
                "FESpace::refine_2d: refined active cell has no children.");
        return children;
    }

    template<typename FESpaceType>
    [[nodiscard]] inline PostFlushEdgeProvenance2D
    classify_post_flush_edge_provenance_2d(
        FESpaceType& space,
        const int child_cell_id,
        const typename mesh::MeshTypes<typename FESpaceType::GT>::SpatialFaceVertexIds&
            child_edge)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;

        auto& mesh = space.mesh_ref();
        if (!mesh.valid_cell_id(child_cell_id))
            return PostFlushEdgeProvenance2D::UnknownEdgeProvenance;

        const auto& child = mesh.cell(child_cell_id);
        const int parent_id = child.parent_id;
        if (!mesh.valid_cell_id(parent_id))
            return PostFlushEdgeProvenance2D::UnknownEdgeProvenance;

        const auto& parent = mesh.cell(parent_id);
        if (parent.last_split_type == mesh::RefinementType::temporal)
            return PostFlushEdgeProvenance2D::TemporalOrNonLateralFace;

        const Edge sorted_child_edge =
            mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                child_edge);
        const auto& vertices = mesh.spatial_vertices();

        if (parent.last_split_type == mesh::RefinementType::spatial ||
            parent.last_split_type == mesh::RefinementType::spacetime)
        {
            const Edge parent_refinement_edge =
                mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                    mesh.spatial_refinement_edge(parent_id));
            if (mesh::topology::spatial_edge_contains_edge_2d<GeomTraits>(
                    parent_refinement_edge,
                    sorted_child_edge,
                    vertices))
            {
                return PostFlushEdgeProvenance2D::
                    SplitRefinementEdgeChildSubedge;
            }

            bool same_spatial_sibling_edge = false;
            for (const int sibling_id : parent.children)
            {
                if (sibling_id == child_cell_id ||
                    !mesh.valid_cell_id(sibling_id))
                {
                    continue;
                }
                const auto& sibling = mesh.cell(sibling_id);
                for (int sibling_face_id = 0; sibling_face_id < 3;
                     ++sibling_face_id)
                {
                    const Edge sibling_edge =
                        mesh::topology::sorted_spatial_face_vertex_ids_2d<
                            GeomTraits>(
                            sibling
                                .spatial_faces[
                                    static_cast<std::size_t>(sibling_face_id)]
                                .spatial_vertex_ids);
                    if (!(sibling_edge == sorted_child_edge))
                        continue;
                    if (sibling.temporal_vertex_ids ==
                        child.temporal_vertex_ids)
                    {
                        return PostFlushEdgeProvenance2D::InternalSiblingEdge;
                    }
                    same_spatial_sibling_edge = true;
                }
            }

            for (int parent_face_id = 0; parent_face_id < 3; ++parent_face_id)
            {
                const Edge parent_edge =
                    mesh::topology::sorted_spatial_face_vertex_ids_2d<
                        GeomTraits>(
                        parent
                            .spatial_faces[
                                static_cast<std::size_t>(parent_face_id)]
                            .spatial_vertex_ids);
                if (parent_edge == sorted_child_edge ||
                    mesh::topology::spatial_edge_lies_on_boundary_edge_2d<
                        GeomTraits>(
                        sorted_child_edge,
                        parent_edge,
                        vertices))
                {
                    return PostFlushEdgeProvenance2D::
                        InheritedParentBoundaryEdge;
                }
            }

            if (same_spatial_sibling_edge)
                return PostFlushEdgeProvenance2D::NewlyCreatedInternalEdge;
        }

        return PostFlushEdgeProvenance2D::UnknownEdgeProvenance;
    }

    template<typename FESpaceType>
    inline void record_post_flush_parent_split_type_2d(
        FESpaceType& space,
        const int child_cell_id,
        PostFlushEdgeProvenanceMetrics2D& provenance_metrics)
    {
        auto& mesh = space.mesh_ref();
        if (!mesh.valid_cell_id(child_cell_id))
        {
            ++provenance_metrics.parent_split_other_queries;
            return;
        }
        const int parent_id = mesh.cell(child_cell_id).parent_id;
        if (!mesh.valid_cell_id(parent_id))
        {
            ++provenance_metrics.parent_split_other_queries;
            return;
        }
        switch (mesh.cell(parent_id).last_split_type)
        {
        case mesh::RefinementType::spatial:
            ++provenance_metrics.parent_split_spatial_queries;
            break;
        case mesh::RefinementType::spacetime:
            ++provenance_metrics.parent_split_spacetime_queries;
            break;
        case mesh::RefinementType::temporal:
            ++provenance_metrics.parent_split_temporal_queries;
            break;
        default:
            ++provenance_metrics.parent_split_other_queries;
            break;
        }
    }

    [[nodiscard]] inline bool
    post_flush_edge_provenance_is_affected_2d(
        const PostFlushEdgeProvenance2D provenance) noexcept
    {
        return provenance ==
                   PostFlushEdgeProvenance2D::
                       SplitRefinementEdgeChildSubedge ||
               provenance ==
                   PostFlushEdgeProvenance2D::InheritedParentBoundaryEdge;
    }

    [[nodiscard]] inline bool
    post_flush_edge_provenance_is_split_edge_2d(
        const PostFlushEdgeProvenance2D provenance) noexcept
    {
        return provenance ==
            PostFlushEdgeProvenance2D::SplitRefinementEdgeChildSubedge;
    }

    template<typename GeomTraits>
    struct AffectedPostFlushEdge2D
    {
        using Edge =
            typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;
        using EdgeKey = mesh::topology::SpatialEdgeKey2D<GeomTraits>;

        int source_parent_cell_id = -1;
        int child_cell_id = -1;
        int face_id = -1;
        Edge edge{};
        EdgeKey edge_key{};
        int temporal_v0 = -1;
        int temporal_v1 = -1;
        double t0 = 0.0;
        double t1 = 0.0;
        mesh::RefinementType split_type = mesh::RefinementType::none;
        PostFlushEdgeProvenance2D provenance =
            PostFlushEdgeProvenance2D::UnknownEdgeProvenance;
    };

    template<typename FESpaceType>
    struct PreSplitParentFaceNeighbourInfo2D
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge =
            typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;
        using EdgeKey = mesh::topology::SpatialEdgeKey2D<GeomTraits>;
        using Record =
            typename FESpaceType::PartitionViewType
                ::ActiveEdgeIntervalIndex2DType::Record;

        int parent_cell_id = -1;
        int parent_face_id = -1;
        Edge parent_face_edge{};
        EdgeKey parent_face_edge_key{};
        int temporal_v0 = -1;
        int temporal_v1 = -1;
        double t0 = 0.0;
        double t1 = 0.0;
        mesh::RefinementType split_type = mesh::RefinementType::none;
        std::vector<Record> exact_records{};
    };

    template<typename FESpaceType>
    [[nodiscard]] inline bool record_cell_can_discover_forced_cell_2d(
        FESpaceType& space,
        const int record_cell_id,
        const int source_record_cell_id,
        const int forced_cell_id)
    {
        if (record_cell_id == source_record_cell_id ||
            record_cell_id == forced_cell_id)
        {
            return true;
        }

        auto& mesh = space.mesh_ref();
        if (!mesh.valid_cell_id(record_cell_id) ||
            !mesh.valid_cell_id(forced_cell_id))
        {
            return false;
        }

        return mesh::topology::same_spatial_cell_vertices_2d(
                   mesh.cell(record_cell_id),
                   mesh.cell(forced_cell_id)) &&
               mesh::topology::temporal_intervals_overlap_positive_2d(
                   mesh,
                   mesh.cell(record_cell_id),
                   mesh.cell(forced_cell_id));
    }

    template<typename FESpaceType, typename Records>
    [[nodiscard]] inline bool records_can_discover_forced_cell_2d(
        FESpaceType& space,
        const Records& records,
        const int source_record_cell_id,
        const int forced_cell_id)
    {
        for (const auto& record : records)
        {
            if (record_cell_can_discover_forced_cell_2d(
                    space,
                    record.cell_id,
                    source_record_cell_id,
                    forced_cell_id))
            {
                return true;
            }
        }
        return false;
    }

    template<typename FESpaceType>
    [[nodiscard]] inline bool parent_source_edge_for_post_flush_edge_2d(
        FESpaceType& space,
        const AffectedPostFlushEdge2D<typename FESpaceType::GT>& affected_edge,
        typename mesh::MeshTypes<typename FESpaceType::GT>::SpatialFaceVertexIds&
            out_edge)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;

        auto& mesh = space.mesh_ref();
        const int parent_id = affected_edge.source_parent_cell_id;
        if (!mesh.valid_cell_id(parent_id))
            return false;

        if (affected_edge.provenance ==
            PostFlushEdgeProvenance2D::SplitRefinementEdgeChildSubedge)
        {
            out_edge =
                mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                    mesh.spatial_refinement_edge(parent_id));
            return true;
        }

        if (affected_edge.provenance ==
            PostFlushEdgeProvenance2D::InheritedParentBoundaryEdge)
        {
            const auto& parent = mesh.cell(parent_id);
            for (int parent_face_id = 0; parent_face_id < 3;
                 ++parent_face_id)
            {
                const Edge parent_edge =
                    mesh::topology::sorted_spatial_face_vertex_ids_2d<
                        GeomTraits>(
                        parent
                            .spatial_faces[
                                static_cast<std::size_t>(parent_face_id)]
                            .spatial_vertex_ids);
                if (parent_edge == affected_edge.edge ||
                    mesh::topology::spatial_edge_lies_on_boundary_edge_2d<
                        GeomTraits>(
                        affected_edge.edge,
                        parent_edge,
                        mesh.spatial_vertices()))
                {
                    out_edge = parent_edge;
                    return true;
                }
            }
        }

        return false;
    }

    template<typename FESpaceType>
    [[nodiscard]] inline std::vector<
        typename mesh::MeshTypes<typename FESpaceType::GT>::SpatialFaceVertexIds>
    ancestor_edges_for_post_flush_edge_2d(
        FESpaceType& space,
        const AffectedPostFlushEdge2D<typename FESpaceType::GT>& affected_edge)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;

        auto& mesh = space.mesh_ref();
        std::vector<Edge> edges;
        int ancestor_id = mesh.valid_cell_id(affected_edge.child_cell_id)
                              ? mesh.cell(affected_edge.child_cell_id).parent_id
                              : -1;
        while (ancestor_id >= 0 && mesh.valid_cell_id(ancestor_id))
        {
            const auto& ancestor = mesh.cell(ancestor_id);
            for (int face_id = 0; face_id < 3; ++face_id)
            {
                const Edge ancestor_edge =
                    mesh::topology::sorted_spatial_face_vertex_ids_2d<
                        GeomTraits>(
                        ancestor
                            .spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);
                if (!mesh::topology::spatial_edge_lies_on_boundary_edge_2d<
                        GeomTraits>(
                        affected_edge.edge,
                        ancestor_edge,
                        mesh.spatial_vertices()))
                {
                    continue;
                }
                if (ancestor_edge == affected_edge.edge)
                    continue;
                if (std::find(edges.begin(), edges.end(), ancestor_edge) ==
                    edges.end())
                {
                    edges.push_back(ancestor_edge);
                }
            }
            ancestor_id = ancestor.parent_id;
        }
        return edges;
    }

    template<typename FESpaceType>
    [[nodiscard]] inline bool exact_edge_source_discovers_forced_cell_2d(
        FESpaceType& space,
        const typename mesh::MeshTypes<typename FESpaceType::GT>::SpatialFaceVertexIds&
            edge,
        const double t0,
        const double t1,
        const int source_record_cell_id,
        const int forced_cell_id)
    {
        using GeomTraits = typename FESpaceType::GT;

        const auto edge_key =
            mesh::topology::SpatialEdgeKey2D<GeomTraits>{edge};
        const auto records =
            space.partition_view().active_edge_interval_index_2d()
                .overlap_records(edge_key, t0, t1);
        return records_can_discover_forced_cell_2d(
            space,
            records,
            source_record_cell_id,
            forced_cell_id);
    }

    template<typename FESpaceType>
    inline void record_forced_cell_source_2d(
        FESpaceType& space,
        const AffectedPostFlushEdge2D<typename FESpaceType::GT>& affected_edge,
        const typename FESpaceType::PartitionViewType
            ::ActiveEdgeIntervalIndex2DType::Record& source_record,
        const int forced_cell_id,
        const std::unordered_set<int>& split_batch_cells,
        const bool record_contains_query,
        ActiveIndexedRefinementMetrics2D& metrics)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;

        auto& total = metrics.forced_cell_sources;
        ForcedCellSourceMetrics2D* provenance_sources = nullptr;
        if (affected_edge.provenance ==
            PostFlushEdgeProvenance2D::SplitRefinementEdgeChildSubedge)
        {
            provenance_sources = &metrics.forced_cell_sources_split_edge;
        }
        else if (
            affected_edge.provenance ==
            PostFlushEdgeProvenance2D::InheritedParentBoundaryEdge)
        {
            provenance_sources = &metrics.forced_cell_sources_inherited_edge;
        }

        const auto add_if =
            [&](const bool condition,
                std::size_t ForcedCellSourceMetrics2D::* field)
            {
                if (!condition)
                    return;
                ++(total.*field);
                if (provenance_sources != nullptr)
                    ++((*provenance_sources).*field);
            };

        ++total.total;
        if (provenance_sources != nullptr)
            ++provenance_sources->total;
        add_if(
            affected_edge.provenance ==
                PostFlushEdgeProvenance2D::
                    SplitRefinementEdgeChildSubedge,
            &ForcedCellSourceMetrics2D::from_split_edge);
        add_if(
            affected_edge.provenance ==
                PostFlushEdgeProvenance2D::InheritedParentBoundaryEdge,
            &ForcedCellSourceMetrics2D::from_inherited_edge);

        const bool exact_child =
            exact_edge_source_discovers_forced_cell_2d(
                space,
                affected_edge.edge,
                affected_edge.t0,
                affected_edge.t1,
                source_record.cell_id,
                forced_cell_id);

        Edge parent_edge{};
        const bool has_parent_edge =
            parent_source_edge_for_post_flush_edge_2d(
                space,
                affected_edge,
                parent_edge);
        const bool exact_parent =
            has_parent_edge &&
            exact_edge_source_discovers_forced_cell_2d(
                space,
                parent_edge,
                affected_edge.t0,
                affected_edge.t1,
                source_record.cell_id,
                forced_cell_id);

        bool ancestor_exact = false;
        for (const Edge& ancestor_edge :
             ancestor_edges_for_post_flush_edge_2d(space, affected_edge))
        {
            if (exact_edge_source_discovers_forced_cell_2d(
                    space,
                    ancestor_edge,
                    affected_edge.t0,
                    affected_edge.t1,
                    source_record.cell_id,
                    forced_cell_id))
            {
                ancestor_exact = true;
                break;
            }
        }

        const bool split_batch =
            split_batch_cells.find(forced_cell_id) != split_batch_cells.end();
        const bool same_spatial_time_overlap =
            record_cell_can_discover_forced_cell_2d(
                space,
                source_record.cell_id,
                source_record.cell_id,
                forced_cell_id);
        const bool dyadic_lookup =
            mesh::topology::spatial_edge_contains_edge_2d<GeomTraits>(
                source_record.edge_key.vertex_ids,
                affected_edge.edge,
                space.mesh_ref().spatial_vertices()) ||
            mesh::topology::spatial_edge_contains_edge_2d<GeomTraits>(
                affected_edge.edge,
                source_record.edge_key.vertex_ids,
                space.mesh_ref().spatial_vertices());
        const bool broad_support_line_required =
            !exact_child && !exact_parent && !ancestor_exact && !split_batch;

        add_if(
            exact_child,
            &ForcedCellSourceMetrics2D::discoverable_exact_child_edge);
        add_if(
            exact_parent,
            &ForcedCellSourceMetrics2D::discoverable_exact_parent_edge);
        add_if(
            exact_parent,
            &ForcedCellSourceMetrics2D::discoverable_presplit_parent_face);
        add_if(
            split_batch,
            &ForcedCellSourceMetrics2D::discoverable_split_batch);
        add_if(
            same_spatial_time_overlap,
            &ForcedCellSourceMetrics2D::
                discoverable_same_spatial_time_overlap);
        add_if(
            ancestor_exact,
            &ForcedCellSourceMetrics2D::discoverable_ancestor_edge_exact);
        add_if(
            dyadic_lookup,
            &ForcedCellSourceMetrics2D::discoverable_dyadic_edge_lookup);
        add_if(
            broad_support_line_required,
            &ForcedCellSourceMetrics2D::requires_broad_support_line);
        add_if(
            !record_contains_query,
            &ForcedCellSourceMetrics2D::requires_query_contains_record);
        add_if(
            record_contains_query,
            &ForcedCellSourceMetrics2D::requires_record_contains_query);
    }

    inline void record_skipped_post_flush_edge_2d(
        ActiveIndexedRefinementMetrics2D& metrics,
        PostFlushEdgeProvenanceMetrics2D& provenance_metrics,
        const PostFlushEdgeProvenance2D provenance)
    {
        ++provenance_metrics.skipped_count;
        ++metrics.post_flush_skipped_edges_count;
        switch (provenance)
        {
        case PostFlushEdgeProvenance2D::InternalSiblingEdge:
            ++metrics.post_flush_skipped_internal_sibling_edges;
            break;
        case PostFlushEdgeProvenance2D::NewlyCreatedInternalEdge:
            ++metrics.post_flush_skipped_newly_created_internal_edges;
            break;
        case PostFlushEdgeProvenance2D::TemporalOrNonLateralFace:
            ++metrics.post_flush_skipped_temporal_or_non_lateral_faces;
            break;
        case PostFlushEdgeProvenance2D::UnknownEdgeProvenance:
            ++metrics.post_flush_skipped_unknown_edges;
            break;
        case PostFlushEdgeProvenance2D::SplitRefinementEdgeChildSubedge:
        case PostFlushEdgeProvenance2D::InheritedParentBoundaryEdge:
            break;
        }
    }

    template<typename FESpaceType>
    [[nodiscard]] inline std::vector<
        typename FESpaceType::PartitionViewType::ActiveEdgeIntervalIndex2DType::Record>
    overlap_records_for_affected_post_flush_edge_2d(
        FESpaceType& space,
        const AffectedPostFlushEdge2D<typename FESpaceType::GT>& affected_edge,
        ActiveIndexedRefinementMetrics2D& metrics,
        ActiveEdgeQueryContext2D<FESpaceType>& query_context)
    {
        using GeomTraits = typename FESpaceType::GT;

        ++metrics.edge_query_containment_queries;
        ++metrics.post_flush_bidirectional_query_mode_used;
        ++metrics.post_flush_exact_ancestor_queries_skipped;

        const EdgeQueryCacheKey2D<GeomTraits> cache_key{
            affected_edge.edge_key,
            affected_edge.temporal_v0,
            affected_edge.temporal_v1,
            query_context.active_index_version,
            EdgeQueryMode2D::Containment};
        const bool cache_hit =
            query_context.cache_enabled &&
            query_context.cache.find(cache_key) != query_context.cache.end();
        if (cache_hit)
            ++metrics.post_flush_affected_edge_cache_hits;
        else
            ++metrics.post_flush_affected_edge_cache_misses;

        const auto before_callsite_metrics =
            metrics.edge_query_callsite_metrics[
                edge_query_callsite_index_2d(
                    EdgeQueryCallSite2D::PostFlushForcedClosure)];
        const auto records =
            cached_edge_query_records_2d(
                space,
                query_context,
                metrics,
                EdgeQueryCallSite2D::PostFlushForcedClosure,
                cache_key,
                [&]()
                {
                    return space.partition_view()
                        .active_edge_interval_index_2d()
                        .overlap_records_with_spatial_edge_containment(
                            affected_edge.edge_key,
                            affected_edge.t0,
                            affected_edge.t1);
                });
        const auto& after_callsite_metrics =
            metrics.edge_query_callsite_metrics[
                edge_query_callsite_index_2d(
                    EdgeQueryCallSite2D::PostFlushForcedClosure)];
        metrics.post_flush_fallback_bidirectional_candidates +=
            after_callsite_metrics.containment_candidates -
            before_callsite_metrics.containment_candidates;
        return records;
    }

    template<typename FESpaceType>
    [[nodiscard]] inline std::vector<
        typename FESpaceType::PartitionViewType::ActiveEdgeIntervalIndex2DType::Record>
    exact_child_and_ancestor_records_for_affected_post_flush_edge_2d(
        FESpaceType& space,
        const AffectedPostFlushEdge2D<typename FESpaceType::GT>& affected_edge,
        ActiveIndexedRefinementMetrics2D& metrics,
        ActiveEdgeQueryContext2D<FESpaceType>& query_context)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;
        using Record =
            typename FESpaceType::PartitionViewType
                ::ActiveEdgeIntervalIndex2DType::Record;
        using Clock = std::chrono::steady_clock;

        auto& mesh = space.mesh_ref();
        std::vector<Edge> query_edges;
        query_edges.reserve(8);
        query_edges.push_back(affected_edge.edge);
        for (const Edge& ancestor_edge :
             ancestor_edges_for_post_flush_edge_2d(space, affected_edge))
        {
            if (std::find(
                    query_edges.begin(),
                    query_edges.end(),
                    ancestor_edge) == query_edges.end())
            {
                query_edges.push_back(ancestor_edge);
            }
        }

        std::vector<Record> records;
        std::unordered_set<EdgeRecordVisitKey2D, EdgeRecordVisitKeyHash2D>
            visited_records;
        const auto append_records =
            [&](const std::vector<Record>& source_records,
                const EdgeQueryMode2D mode)
            {
                const auto start = Clock::now();
                auto& mode_metrics =
                    metrics.edge_query_mode_metrics[
                        edge_query_mode_index_2d(mode)];
                auto& callsite_metrics =
                    metrics.edge_query_callsite_metrics[
                        edge_query_callsite_index_2d(
                            EdgeQueryCallSite2D::PostFlushForcedClosure)];
                visited_records.reserve(
                    std::max(
                        visited_records.size() + source_records.size(),
                        std::size_t{8}));
                for (const auto& record : source_records)
                {
                    const EdgeRecordVisitKey2D key{
                        record.cell_id,
                        record.face_id};
                    if (visited_records.insert(key).second)
                        records.push_back(record);
                    else
                    {
                        ++mode_metrics.duplicate_rejects;
                        ++callsite_metrics.duplicate_rejects;
                        ++metrics.edge_query_duplicate_rejects;
                        ++metrics.edge_query_duplicates_rejected;
                    }
                }
                metrics.edge_query_visited_set_size_max =
                    std::max(
                        metrics.edge_query_visited_set_size_max,
                        visited_records.size());
                metrics.edge_query_duplicate_filter_wall_seconds +=
                    std::chrono::duration<double>(Clock::now() - start)
                        .count();
            };

        for (std::size_t query_edge_index = 0;
             query_edge_index < query_edges.size();
             ++query_edge_index)
        {
            const Edge& query_edge = query_edges[query_edge_index];
            const auto query_key =
                mesh::topology::SpatialEdgeKey2D<GeomTraits>{query_edge};
            const bool exact_query = query_edge_index == 0;
            if (exact_query)
                ++metrics.edge_query_exact_edge_queries;
            else
                ++metrics.edge_query_ancestor_edge_queries;
            const EdgeQueryCacheKey2D<GeomTraits> cache_key{
                query_key,
                affected_edge.temporal_v0,
                affected_edge.temporal_v1,
                query_context.active_index_version,
                exact_query
                    ? EdgeQueryMode2D::ExactEdge
                    : EdgeQueryMode2D::AncestorEdge};
            const auto exact_records =
                cached_edge_query_records_2d(
                    space,
                    query_context,
                    metrics,
                    EdgeQueryCallSite2D::PostFlushForcedClosure,
                    cache_key,
                    [&]()
                    {
                        return space.partition_view()
                            .active_edge_interval_index_2d()
                            .overlap_records(
                                query_key,
                                affected_edge.t0,
                                affected_edge.t1);
                    });
            append_records(exact_records, cache_key.mode);
        }

        if (query_edges.size() == 1)
            ++metrics.edge_query_ancestor_queries_skipped;
        return records;
    }

    template<typename FESpaceType>
    inline void capture_presplit_parent_face_neighbours_2d(
        FESpaceType& space,
        const int parent_cell_id,
        const mesh::RefinementType split_type,
        ActiveIndexedRefinementMetrics2D& metrics,
        ActiveEdgeQueryContext2D<FESpaceType>& query_context,
        std::vector<PreSplitParentFaceNeighbourInfo2D<FESpaceType>>& out)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;

        if (!refinement_has_spatial_part_2d(split_type))
            return;

        auto& mesh = space.mesh_ref();
        if (!mesh.valid_cell_id(parent_cell_id))
            return;

        const auto& parent_cell = mesh.cell(parent_cell_id);
        const auto [t0, t1] =
            mesh::topology::temporal_interval_bounds_2d<GeomTraits>(
                mesh,
                parent_cell);
        const int temporal_v0 =
            static_cast<int>(parent_cell.temporal_vertex_ids[0]);
        const int temporal_v1 =
            static_cast<int>(parent_cell.temporal_vertex_ids[1]);

        for (int face_id = 0; face_id < 3; ++face_id)
        {
            const Edge parent_edge =
                mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                    parent_cell
                        .spatial_faces[static_cast<std::size_t>(face_id)]
                        .spatial_vertex_ids);
            const auto edge_key =
                mesh::topology::SpatialEdgeKey2D<GeomTraits>{parent_edge};

            ++metrics.edge_query_exact_edge_queries;
            ++metrics.post_flush_presplit_parent_face_queries;
            const EdgeQueryCacheKey2D<GeomTraits> cache_key{
                edge_key,
                temporal_v0,
                temporal_v1,
                query_context.active_index_version,
                EdgeQueryMode2D::ExactEdge};
            auto records =
                cached_edge_query_records_2d(
                    space,
                    query_context,
                    metrics,
                    EdgeQueryCallSite2D::PostFlushForcedClosure,
                    cache_key,
                    [&]()
                    {
                        return space.partition_view()
                            .active_edge_interval_index_2d()
                            .overlap_records(edge_key, t0, t1);
                    });
            metrics.post_flush_presplit_parent_face_records +=
                records.size();
            out.push_back(
                PreSplitParentFaceNeighbourInfo2D<FESpaceType>{
                    parent_cell_id,
                    face_id,
                    parent_edge,
                    edge_key,
                    temporal_v0,
                    temporal_v1,
                    t0,
                    t1,
                    split_type,
                    std::move(records)});
        }
    }

    template<typename FESpaceType>
    inline void collect_same_spatial_overlap_cells_indexed_2d(
        FESpaceType& space,
        const int spatial_cell_id,
        const int time_cell_id,
        std::unordered_set<int>& out,
        ActiveIndexedRefinementMetrics2D& metrics,
        ActiveEdgeQueryContext2D<FESpaceType>& query_context,
        const EdgeQueryCallSite2D call_site,
        std::vector<int>* newly_inserted_cells = nullptr)
    {
        using GeomTraits = typename FESpaceType::GT;

        auto& mesh = space.mesh_ref();
        if (!mesh.valid_cell_id(spatial_cell_id) ||
            !mesh.valid_cell_id(time_cell_id))
        {
            return;
        }

        const auto& spatial_cell = mesh.cell(spatial_cell_id);
        const auto& time_cell = mesh.cell(time_cell_id);
        const auto [t0, t1] =
            mesh::topology::temporal_interval_bounds_2d<GeomTraits>(
                mesh,
                time_cell);
        const auto edge_key =
            mesh::topology::make_spatial_edge_key_2d<GeomTraits>(
                spatial_cell.spatial_faces[0].spatial_vertex_ids);
        const int temporal_v0 =
            static_cast<int>(time_cell.temporal_vertex_ids[0]);
        const int temporal_v1 =
            static_cast<int>(time_cell.temporal_vertex_ids[1]);

        ++metrics.edge_query_overlap_cell_queries;
        const EdgeQueryCacheKey2D<GeomTraits> cache_key{
            edge_key,
            temporal_v0,
            temporal_v1,
            query_context.active_index_version,
            EdgeQueryMode2D::OverlapCells};
        const auto records =
            cached_edge_query_records_2d(
                space,
                query_context,
                metrics,
                call_site,
                cache_key,
                [&]()
                {
                    return space.partition_view()
                        .active_edge_interval_index_2d()
                        .overlap_records(edge_key, t0, t1);
                });

        std::unordered_set<int> candidates_seen;
        candidates_seen.reserve(records.size());
        for (const auto& record : records)
        {
            const int candidate_id = record.cell_id;
            if (!candidates_seen.insert(candidate_id).second)
            {
                auto& mode_metrics =
                    metrics.edge_query_mode_metrics[
                        edge_query_mode_index_2d(
                            EdgeQueryMode2D::OverlapCells)];
                auto& callsite_metrics =
                    metrics.edge_query_callsite_metrics[
                        edge_query_callsite_index_2d(call_site)];
                ++mode_metrics.duplicate_rejects;
                ++callsite_metrics.duplicate_rejects;
                ++metrics.edge_query_duplicate_rejects;
                ++metrics.edge_query_duplicates_rejected;
                continue;
            }
            if (!space.is_active_cell(candidate_id))
                continue;
            if (!mesh::topology::same_spatial_cell_vertices_2d(
                    mesh.cell(candidate_id),
                    spatial_cell))
            {
                continue;
            }
            if (!mesh::topology::temporal_intervals_overlap_positive_2d(
                    mesh,
                    mesh.cell(candidate_id),
                    time_cell))
            {
                continue;
            }

            ++metrics.post_flush_forced_discovery_count;
            if (out.insert(candidate_id).second)
            {
                ++metrics.post_flush_unique_forced_cell_count;
                if (newly_inserted_cells != nullptr)
                    newly_inserted_cells->push_back(candidate_id);
            }
            else
            {
                ++metrics.post_flush_forced_cells_discovered_multiple_edges;
            }
        }
    }

    template<typename FESpaceType>
    [[nodiscard]] inline std::vector<int>
    collect_indexed_spatial_closure_forced_cells_2d(
        FESpaceType& space,
        const std::vector<int>& seed_cells,
        const std::vector<PreSplitParentFaceNeighbourInfo2D<FESpaceType>>&
            presplit_infos,
        ActiveIndexedRefinementMetrics2D& metrics,
        ActiveEdgeQueryContext2D<FESpaceType>& query_context)
    {
        using GeomTraits = typename FESpaceType::GT;
        using Edge = typename mesh::MeshTypes<GeomTraits>::SpatialFaceVertexIds;
        using AffectedEdge = AffectedPostFlushEdge2D<GeomTraits>;

        auto& mesh = space.mesh_ref();
        std::unordered_set<int> forced;
        std::unordered_set<int> split_batch_cells;
        split_batch_cells.reserve(seed_cells.size());
        for (const int seed_cell_id : seed_cells)
            split_batch_cells.insert(seed_cell_id);
        std::vector<AffectedEdge> post_flush_worklist;
        const bool all_faces_debug =
            space.post_flush_closure_all_faces_debug_2d();
        const bool off_debug =
            space.post_flush_closure_off_debug_2d();
        const bool split_edges_only_debug =
            space.post_flush_closure_split_edges_only_debug_2d();
        const bool split_and_inherited_edges =
            space.post_flush_closure_split_and_inherited_edges_2d();
        const bool presplit_neighbour =
            space.post_flush_closure_presplit_neighbour_2d();
        if (off_debug)
            ++metrics.post_flush_off_debug_mode;
        else if (split_edges_only_debug)
            ++metrics.post_flush_split_edges_only_debug_mode;
        else if (split_and_inherited_edges)
            ++metrics.post_flush_split_and_inherited_edges_mode;
        else if (presplit_neighbour)
            ++metrics.post_flush_presplit_neighbour_mode;
        if (all_faces_debug)
            ++metrics.post_flush_all_faces_debug_mode;
        else if (!off_debug && !split_edges_only_debug &&
                 !split_and_inherited_edges && !presplit_neighbour)
            ++metrics.post_flush_affected_edges_mode;

        if (off_debug)
            return {};

        for (const int seed_cell_id : seed_cells)
        {
            if (!space.is_active_cell(seed_cell_id) ||
                !mesh.valid_cell_id(seed_cell_id))
            {
                continue;
            }

            const auto& seed_cell = mesh.cell(seed_cell_id);
            const auto [seed_t0, seed_t1] =
                mesh::topology::temporal_interval_bounds_2d<GeomTraits>(
                    mesh,
                    seed_cell);

            for (int face_id = 0; face_id < 3; ++face_id)
            {
                const Edge seed_edge =
                    mesh::topology::sorted_spatial_face_vertex_ids_2d<GeomTraits>(
                        seed_cell
                            .spatial_faces[static_cast<std::size_t>(face_id)]
                            .spatial_vertex_ids);
                const PostFlushEdgeProvenance2D provenance =
                    classify_post_flush_edge_provenance_2d(
                        space,
                        seed_cell_id,
                        seed_edge);
                auto& provenance_metrics =
                    metrics.post_flush_edge_provenance_metrics[
                        post_flush_edge_provenance_index_2d(provenance)];
                ++metrics.post_flush_considered_edges_count;
                ++provenance_metrics.considered_count;
                const bool should_query =
                    all_faces_debug ||
                    (split_edges_only_debug
                         ? post_flush_edge_provenance_is_split_edge_2d(
                               provenance)
                         : post_flush_edge_provenance_is_affected_2d(
                               provenance));

                if (!should_query)
                {
                    record_skipped_post_flush_edge_2d(
                        metrics,
                        provenance_metrics,
                        provenance);
                    continue;
                }

                ++metrics.post_flush_query_count;
                if (all_faces_debug)
                    ++metrics.post_flush_all_faces_query_count;
                else
                    ++metrics.post_flush_affected_edges_count;
                ++provenance_metrics.query_count;
                if (0 <= face_id && face_id < 3)
                {
                    ++provenance_metrics
                          .face_id_queries[static_cast<std::size_t>(face_id)];
                }
                record_post_flush_parent_split_type_2d(
                    space,
                    seed_cell_id,
                    provenance_metrics);

                const int parent_id = seed_cell.parent_id;
                const mesh::RefinementType split_type =
                    mesh.valid_cell_id(parent_id)
                        ? mesh.cell(parent_id).last_split_type
                        : mesh::RefinementType::none;
                post_flush_worklist.push_back(
                    AffectedEdge{
                        parent_id,
                        seed_cell_id,
                        face_id,
                        seed_edge,
                        mesh::topology::SpatialEdgeKey2D<GeomTraits>{
                            seed_edge},
                        static_cast<int>(seed_cell.temporal_vertex_ids[0]),
                        static_cast<int>(seed_cell.temporal_vertex_ids[1]),
                        seed_t0,
                        seed_t1,
                        split_type,
                        provenance});
            }
        }

        for (const auto& affected_edge : post_flush_worklist)
        {
            using Record =
                typename FESpaceType::PartitionViewType
                    ::ActiveEdgeIntervalIndex2DType::Record;
            auto& provenance_metrics =
                metrics.post_flush_edge_provenance_metrics[
                    post_flush_edge_provenance_index_2d(
                        affected_edge.provenance)];
            const auto before_callsite_metrics =
                metrics.edge_query_callsite_metrics[
                    edge_query_callsite_index_2d(
                        EdgeQueryCallSite2D::PostFlushForcedClosure)];
            std::vector<Record> records;
            if (presplit_neighbour)
            {
                std::unordered_set<
                    EdgeRecordVisitKey2D,
                    EdgeRecordVisitKeyHash2D>
                    visited_records;
                const auto append_presplit_records =
                    [&](const std::vector<Record>& source_records,
                        const bool from_presplit)
                    {
                        visited_records.reserve(
                            std::max(
                                visited_records.size() +
                                    source_records.size(),
                                std::size_t{8}));
                        for (const auto& record : source_records)
                        {
                            const EdgeRecordVisitKey2D key{
                                record.cell_id,
                                record.face_id};
                            if (!visited_records.insert(key).second)
                            {
                                continue;
                            }
                            records.push_back(record);
                            if (from_presplit)
                                ++metrics.post_flush_presplit_records_used;
                            else
                                ++metrics
                                      .post_flush_presplit_exact_child_ancestor_records_used;
                        }
                    };

                Edge parent_edge{};
                const bool has_parent_edge =
                    parent_source_edge_for_post_flush_edge_2d(
                        space,
                        affected_edge,
                        parent_edge);
                if (has_parent_edge)
                {
                    const auto parent_edge_key =
                        mesh::topology::SpatialEdgeKey2D<GeomTraits>{
                            parent_edge};
                    for (const auto& info : presplit_infos)
                    {
                        if (info.parent_cell_id !=
                            affected_edge.source_parent_cell_id)
                        {
                            continue;
                        }
                        if (!(info.parent_face_edge_key == parent_edge_key))
                            continue;
                        append_presplit_records(info.exact_records, true);
                    }
                }

                const auto exact_child_ancestor_records =
                    exact_child_and_ancestor_records_for_affected_post_flush_edge_2d(
                        space,
                        affected_edge,
                        metrics,
                        query_context);
                append_presplit_records(exact_child_ancestor_records, false);
            }
            else if (all_faces_debug ||
                !space.post_flush_affected_containment_only())
            {
                records =
                    overlap_records_for_cell_edge_indexed_2d(
                        space,
                        affected_edge.child_cell_id,
                        affected_edge.edge,
                        affected_edge.t0,
                        affected_edge.t1,
                        metrics,
                        query_context,
                        EdgeQueryCallSite2D::PostFlushForcedClosure);
            }
            else
            {
                records =
                    overlap_records_for_affected_post_flush_edge_2d(
                        space,
                        affected_edge,
                        metrics,
                        query_context);
            }
            const auto& after_callsite_metrics =
                metrics.edge_query_callsite_metrics[
                    edge_query_callsite_index_2d(
                        EdgeQueryCallSite2D::PostFlushForcedClosure)];
                provenance_metrics.containment_candidates +=
                    after_callsite_metrics.containment_candidates -
                    before_callsite_metrics.containment_candidates;
                provenance_metrics.true_records_returned +=
                    after_callsite_metrics.true_records_returned -
                    before_callsite_metrics.true_records_returned;
                provenance_metrics.spatial_rejects +=
                    after_callsite_metrics.spatial_rejects -
                    before_callsite_metrics.spatial_rejects;
                provenance_metrics.time_rejects +=
                    after_callsite_metrics.time_rejects -
                    before_callsite_metrics.time_rejects;
                provenance_metrics.duplicate_rejects +=
                    after_callsite_metrics.duplicate_rejects -
                    before_callsite_metrics.duplicate_rejects;
                provenance_metrics.wall_seconds +=
                    after_callsite_metrics.wall_seconds -
                    before_callsite_metrics.wall_seconds;

                for (const auto& record : records)
                {
                    if (record.cell_id == affected_edge.child_cell_id ||
                        !space.is_active_cell(record.cell_id))
                    {
                        continue;
                    }

                    if (record.edge_key == affected_edge.edge_key)
                        continue;

                    if (mesh::topology::spatial_edge_contains_edge_2d<GeomTraits>(
                            record.edge_key.vertex_ids,
                            affected_edge.edge,
                            mesh.spatial_vertices()))
                    {
                        const auto forced_before = forced.size();
                        std::vector<int> newly_forced_cells;
                        collect_same_spatial_overlap_cells_indexed_2d(
                            space,
                            record.cell_id,
                            affected_edge.child_cell_id,
                            forced,
                            metrics,
                            query_context,
                            EdgeQueryCallSite2D::PostFlushForcedClosure,
                            &newly_forced_cells);
                        for (const int forced_cell_id : newly_forced_cells)
                        {
                            record_forced_cell_source_2d(
                                space,
                                affected_edge,
                                record,
                                forced_cell_id,
                                split_batch_cells,
                                true,
                                metrics);
                        }
                        provenance_metrics.forced_cells_found +=
                            forced.size() - forced_before;
                        metrics.post_flush_forced_cells_found +=
                            forced.size() - forced_before;
                        continue;
                    }

                    if (mesh::topology::spatial_edge_contains_edge_2d<GeomTraits>(
                            affected_edge.edge,
                            record.edge_key.vertex_ids,
                            mesh.spatial_vertices()))
                    {
                        const auto forced_before = forced.size();
                        std::vector<int> newly_forced_cells;
                        collect_same_spatial_overlap_cells_indexed_2d(
                            space,
                            affected_edge.child_cell_id,
                            record.cell_id,
                            forced,
                            metrics,
                            query_context,
                            EdgeQueryCallSite2D::PostFlushForcedClosure,
                            &newly_forced_cells);
                        for (const int forced_cell_id : newly_forced_cells)
                        {
                            record_forced_cell_source_2d(
                                space,
                                affected_edge,
                                record,
                                forced_cell_id,
                                split_batch_cells,
                                false,
                                metrics);
                        }
                        provenance_metrics.forced_cells_found +=
                            forced.size() - forced_before;
                        metrics.post_flush_forced_cells_found +=
                            forced.size() - forced_before;
                    }
                }
        }

        std::vector<int> forced_cells;
        forced_cells.reserve(forced.size());
        for (const int cell_id : forced)
            forced_cells.push_back(cell_id);
        std::sort(forced_cells.begin(), forced_cells.end());
        metrics.post_flush_forced_cells_returned += forced_cells.size();
        return forced_cells;
    }

    template<typename FESpaceType>
    [[nodiscard]] inline ActiveWaveRefinementResult2D refine_active_wave_2d(
        FESpaceType& space,
        const std::vector<int>& wave,
        mesh::RefinementType requested_refinement_type)
    {
        ActiveWaveRefinementResult2D result;
        if (wave.empty())
            return result;

        auto& mesh = space.unsafe_mesh_ref();

        std::unordered_map<int, std::vector<int>> children_by_cell;
        children_by_cell.reserve(wave.size());
        space.time_phase(
            "refinement.child_creation_wall",
            [&]()
            {
                space.time_phase(
                    "fespace.refinement.storage_child_creation",
                    [&]()
                    {
                        space.time_phase(
                            "refinement.child_creation_total",
                            [&]()
                            {
                                for (const int cell_id : wave)
                                {
                                    if (!space.is_active_cell(cell_id))
                                        continue;

                                    if (!mesh.valid_cell_id(cell_id))
                                        throw std::runtime_error(
                                            "FESpace::refine_2d: marked cell id out of range.");

                                    children_by_cell.emplace(
                                        cell_id,
                                        mesh.create_children_if_needed(
                                            cell_id,
                                            requested_refinement_type));
                                }
                            });
                    });
            });

        std::unordered_set<int> add_ids;
        std::unordered_set<int> remove_ids;

        for (const auto& [cell_id, children] : children_by_cell)
        {
            if (!space.is_active_cell(cell_id))
                continue;
            if (children.empty())
                throw std::runtime_error(
                    "FESpace::refine_2d: refined active cell has no children.");

            remove_ids.insert(cell_id);
            for (const int child_id : children)
                add_ids.insert(child_id);
        }

        if (add_ids.empty() && remove_ids.empty())
            return result;

        space.time_phase(
            "refinement.partition_update_wall",
            [&]()
            {
                space.time_phase(
                    "fespace.refinement.active_partition_update",
                    [&]()
                    {
                        space.time_phase(
                            "refinement.active_partition_update_total",
                            [&]()
                            {
                                space.unsafe_update_active_cells(
                                    add_ids,
                                    remove_ids);
                            });
                    });
            });

        result.changed = true;
        result.added_active_cells.reserve(add_ids.size());
        for (const int cell_id : add_ids)
            result.added_active_cells.push_back(cell_id);
        std::sort(
            result.added_active_cells.begin(),
            result.added_active_cells.end());
        result.removed_active_cells.reserve(remove_ids.size());
        for (const int cell_id : remove_ids)
            result.removed_active_cells.push_back(cell_id);
        std::sort(
            result.removed_active_cells.begin(),
            result.removed_active_cells.end());
        return result;
    }

    template<typename FESpaceType>
    [[nodiscard]] inline ActiveWaveRefinementResult2D
    refine_active_indexed_queue_2d(
        FESpaceType& space,
        const std::vector<int>& marked,
        const mesh::RefinementType requested_refinement_type,
        ActiveIndexedRefinementMetrics2D& metrics)
    {
        using GeomTraits = typename FESpaceType::GT;
        using EdgeKey = mesh::topology::SpatialEdgeKey2D<GeomTraits>;

        ActiveWaveRefinementResult2D result;
        if (marked.empty())
            return result;

        auto& mesh = space.unsafe_mesh_ref();

        std::deque<int> queue;
        std::vector<char> queued(mesh.n_cells(), 0);
        std::unordered_map<int, mesh::RefinementType> requested_by_cell;
        requested_by_cell.reserve(marked.size());
        PendingActiveRefinementBatch2D pending_batch;
        ActiveEdgeQueryContext2D<FESpaceType> query_context;
        query_context.cache_enabled =
            space.refinement_edge_query_cache_enabled();
        const std::size_t max_pending_split_cells_before_flush =
            std::max<std::size_t>(
                std::size_t{1},
                space.refinement_batch_target_split_cells());
        metrics.batch_target_split_cells =
            max_pending_split_cells_before_flush;

        const auto resize_queued =
            [&]()
            {
                if (queued.size() < mesh.n_cells())
                    queued.resize(mesh.n_cells(), 0);
            };

        const auto is_effectively_active =
            [&](const int cell_id)
            {
                return space.is_active_cell(cell_id) &&
                       pending_batch.remove_ids.count(cell_id) == 0;
            };

        const auto enqueue =
            [&](const int cell_id)
            {
                if (!is_effectively_active(cell_id))
                    return;
                if (!mesh.valid_cell_id(cell_id))
                    throw std::runtime_error(
                        "FESpace::refine_2d: queued cell id out of range.");
                resize_queued();
                const auto idx = static_cast<std::size_t>(cell_id);
                if (queued[idx])
                    return;
                queued[idx] = 1;
                queue.push_back(cell_id);
            };

        for (const int cell_id : marked)
        {
            if (!space.is_active_cell(cell_id))
                continue;
            requested_by_cell[cell_id] = requested_refinement_type;
            enqueue(cell_id);
        }

        std::unordered_set<int> global_added;
        std::unordered_set<int> global_removed;
        std::unordered_set<int> pending_cells_seen;
        std::unordered_set<int> blockers_seen;
        std::unordered_set<int> forced_enqueued_cells;
        std::vector<PreSplitParentFaceNeighbourInfo2D<FESpaceType>>
            pending_presplit_infos;
        std::uint64_t guard = 0;

        enum class FlushReason
        {
            QueueEmpty,
            Dependency,
            BatchLimit
        };

        const auto flush_pending_batch =
            [&](const FlushReason reason)
            {
                if (pending_batch.empty())
                    return;

                std::vector<int> added_seed_cells =
                    pending_batch.added_seed_cells;
                std::vector<PreSplitParentFaceNeighbourInfo2D<FESpaceType>>
                    presplit_infos = std::move(pending_presplit_infos);
                std::sort(added_seed_cells.begin(), added_seed_cells.end());
                added_seed_cells.erase(
                    std::unique(
                        added_seed_cells.begin(),
                        added_seed_cells.end()),
                    added_seed_cells.end());

                metrics.partition_update_calls += 1;
                metrics.batched_split_cells +=
                    pending_batch.split_cell_count;
                metrics.max_batch_size =
                    std::max(
                        metrics.max_batch_size,
                        pending_batch.split_cell_count);
                switch (reason)
                {
                case FlushReason::QueueEmpty:
                    ++metrics.flush_due_to_queue_empty_count;
                    break;
                case FlushReason::Dependency:
                    ++metrics.flush_due_to_dependency_count;
                    break;
                case FlushReason::BatchLimit:
                    ++metrics.flush_due_to_batch_limit_count;
                    break;
                }

                space.time_phase(
                    "refinement.partition_update_wall",
                    [&]()
                    {
                        space.time_phase(
                            "fespace.refinement.active_partition_update",
                            [&]()
                            {
                                space.time_phase(
                                    "refinement.active_partition_update_total",
                                    [&]()
                                    {
                                        space.unsafe_update_active_cells(
                                            pending_batch.add_ids,
                                            pending_batch.remove_ids);
                                    });
                            });
                    });

                result.changed = true;
                pending_batch.clear();
                pending_presplit_infos.clear();
                resize_queued();
                query_context.clear_after_active_index_update(metrics);
                ++metrics.flush_due_to_cache_invalidation_count;

                const auto forced_cells =
                    collect_indexed_spatial_closure_forced_cells_2d(
                        space,
                        added_seed_cells,
                        presplit_infos,
                        metrics,
                        query_context);
                metrics.blocker_cells += forced_cells.size();
                for (const int forced_cell_id : forced_cells)
                {
                    if (global_removed.count(forced_cell_id) != 0)
                    {
                        ++metrics
                              .post_flush_forced_cells_skipped_already_split;
                        continue;
                    }
                    if (!mesh.valid_cell_id(forced_cell_id) ||
                        !space.is_active_cell(forced_cell_id))
                    {
                        ++metrics.post_flush_forced_cells_skipped_inactive;
                        continue;
                    }
                    resize_queued();
                    const auto forced_idx =
                        static_cast<std::size_t>(forced_cell_id);
                    if (forced_idx < queued.size() && queued[forced_idx])
                    {
                        ++metrics
                              .post_flush_forced_cells_skipped_already_pending;
                        continue;
                    }
                    enqueue(forced_cell_id);
                    if (forced_idx < queued.size() && queued[forced_idx])
                    {
                        ++metrics.post_flush_forced_cells_enqueued;
                        forced_enqueued_cells.insert(forced_cell_id);
                    }
                    else
                    {
                        ++metrics.post_flush_forced_cells_skipped_inactive;
                    }
                }
            };

        while (!queue.empty() || !pending_batch.empty())
        {
            if (queue.empty())
            {
                flush_pending_batch(FlushReason::QueueEmpty);
                continue;
            }

            const int cell_id = queue.front();
            queue.pop_front();
            resize_queued();
            if (mesh.valid_cell_id(cell_id))
                queued[static_cast<std::size_t>(cell_id)] = 0;

            if (!is_effectively_active(cell_id))
                continue;
            if (!mesh.valid_cell_id(cell_id))
                throw std::runtime_error(
                    "FESpace::refine_2d: queued active cell id out of range.");

            ++metrics.queue_pops;
            if (pending_cells_seen.insert(cell_id).second)
                metrics.unique_pending_cells_seen = pending_cells_seen.size();
            else
            {
                ++metrics.repeated_pending_cell_pops;
                ++metrics.closure_decision_cache_possible_count;
            }
            ++guard;
            if (guard > 1000000)
                throw std::runtime_error(
                    "FESpace::refine_2d: indexed refinement queue did not converge.");

            const auto requested_it = requested_by_cell.find(cell_id);
            const auto cell_requested_type =
                requested_it == requested_by_cell.end()
                    ? mesh::RefinementType::none
                    : requested_it->second;
            const auto cell_effective_type =
                effective_refinement_type_2d(
                    mesh,
                    cell_id,
                    cell_requested_type);

            std::vector<int> split_cells;
            if (!refinement_has_spatial_part_2d(cell_effective_type))
            {
                split_cells.push_back(cell_id);
            }
            else
            {
                const EdgeKey edge_key =
                    active_spatial_refinement_edge_key_2d(mesh, cell_id);
                const auto [t0, t1] =
                    active_temporal_interval_bounds_2d(mesh, cell_id);

                const auto overlap_records =
                    overlap_records_for_cell_edge_indexed_2d(
                        space,
                        cell_id,
                        mesh.spatial_refinement_edge(cell_id),
                        t0,
                        t1,
                        metrics,
                        query_context,
                        EdgeQueryCallSite2D::MainQueue);

                std::unordered_set<int> exact_split_seen;
                exact_split_seen.reserve(overlap_records.size() + 1);
                exact_split_seen.insert(cell_id);
                split_cells.push_back(cell_id);
                std::vector<int> blockers;
                blockers.reserve(overlap_records.size());
                for (const auto& record : overlap_records)
                {
                    const int candidate_id = record.cell_id;
                    if (!is_effectively_active(candidate_id))
                        continue;

                    if (record.edge_key == edge_key)
                    {
                        if (exact_split_seen.insert(candidate_id).second)
                            split_cells.push_back(candidate_id);
                        continue;
                    }

                    if (!mesh::topology::spatial_edge_contains_edge_2d<GeomTraits>(
                            record.edge_key.vertex_ids,
                            edge_key.vertex_ids,
                            mesh.spatial_vertices()))
                    {
                        if (record.containment_direction == 2U)
                        {
                            auto& callsite_metrics =
                                metrics.edge_query_callsite_metrics[
                                    edge_query_callsite_index_2d(
                                        EdgeQueryCallSite2D::MainQueue)];
                            ++callsite_metrics
                                  .bidirectional_records_later_discarded_by_main_closure;
                        }
                        continue;
                    }

                    const EdgeKey candidate_refinement_edge =
                        active_spatial_refinement_edge_key_2d(
                            mesh,
                            candidate_id);
                    if (!(candidate_refinement_edge == edge_key))
                        blockers.push_back(candidate_id);
                }

                std::sort(blockers.begin(), blockers.end());
                blockers.erase(
                    std::unique(blockers.begin(), blockers.end()),
                    blockers.end());

                if (!blockers.empty())
                {
                    flush_pending_batch(FlushReason::Dependency);
                    metrics.blockers_found += blockers.size();
                    metrics.requeued_due_to_blockers += blockers.size() + 1;
                    metrics.blocker_cells += blockers.size();
                    for (const int blocker_id : blockers)
                    {
                        if (!blockers_seen.insert(blocker_id).second)
                        {
                            ++metrics.blockers_already_seen;
                            ++metrics.closure_decision_cache_possible_count;
                        }
                        enqueue(blocker_id);
                    }
                    enqueue(cell_id);
                    continue;
                }
            }

            std::sort(split_cells.begin(), split_cells.end());
            split_cells.erase(
                std::unique(split_cells.begin(), split_cells.end()),
                split_cells.end());

            std::unordered_set<int> add_ids;
            std::unordered_set<int> remove_ids;
            for (const int split_cell_id : split_cells)
            {
                if (!is_effectively_active(split_cell_id))
                    continue;

                const auto split_requested_it =
                    requested_by_cell.find(split_cell_id);
                const auto split_requested_type =
                    split_requested_it == requested_by_cell.end()
                        ? mesh::RefinementType::none
                        : split_requested_it->second;
                const auto split_effective_type =
                    effective_refinement_type_2d(
                        mesh,
                        split_cell_id,
                        split_requested_type);

                if (space.post_flush_closure_presplit_neighbour_2d())
                {
                    capture_presplit_parent_face_neighbours_2d(
                        space,
                        split_cell_id,
                        split_effective_type,
                        metrics,
                        query_context,
                        pending_presplit_infos);
                }
                if (forced_enqueued_cells.erase(split_cell_id) != 0)
                    ++metrics.post_flush_forced_cells_split_later;

                std::vector<int> children;
                space.time_phase(
                    "refinement.child_creation_wall",
                    [&]()
                    {
                        space.time_phase(
                            "refinement.child_creation_total",
                            [&]()
                            {
                                children =
                                    mesh_create_children_if_needed_for_active_refinement(
                                        mesh,
                                        split_cell_id,
                                        split_requested_type,
                                        metrics);
                            });
                    });

                remove_ids.insert(split_cell_id);
                global_removed.insert(split_cell_id);
                pending_batch.remove_ids.insert(split_cell_id);
                for (const int child_id : children)
                {
                    add_ids.insert(child_id);
                    global_added.insert(child_id);
                    pending_batch.add_ids.insert(child_id);
                    pending_batch.added_seed_cells.push_back(child_id);
                }
                ++metrics.actually_split_active_cells;
                ++pending_batch.split_cell_count;
            }

            if (add_ids.empty() && remove_ids.empty())
                continue;

            if (pending_batch.split_cell_count >=
                max_pending_split_cells_before_flush)
            {
                flush_pending_batch(FlushReason::BatchLimit);
            }
        }

        result.added_active_cells.reserve(global_added.size());
        for (const int cell_id : global_added)
            result.added_active_cells.push_back(cell_id);
        std::sort(
            result.added_active_cells.begin(),
            result.added_active_cells.end());

        result.removed_active_cells.reserve(global_removed.size());
        for (const int cell_id : global_removed)
            result.removed_active_cells.push_back(cell_id);
        std::sort(
            result.removed_active_cells.begin(),
            result.removed_active_cells.end());

        return result;
    }

    template<typename FESpaceType>
    inline void refine_2d(
        FESpaceType& space,
        const std::vector<int>& marked,
        mesh::RefinementType requested_refinement_type =
            mesh::RefinementType::none)
    {
        using GeomTraits = typename FESpaceType::GT;
        static_assert(GeomTraits::dim_space_v == 2, "refine_2d requires dim_space_v == 2.");
        static_assert(GeomTraits::dim_time_v == 1, "refine_2d requires dim_time_v == 1.");

        std::vector<int> pending =
            deduplicate_active_marked_cells_2d(space, marked);

        if (pending.empty())
            return;

        ActiveIndexedRefinementMetrics2D metrics;
        metrics.initially_marked_active_cells = pending.size();

        ActiveWaveRefinementResult2D refinement_result;
        space.time_phase(
            "refinement.queue_closure_wall",
            [&]()
            {
                space.time_phase(
                    "fespace.refinement.active_indexed_queue",
                    [&]()
                    {
                        space.time_phase(
                            "refinement.queue_closure_total",
                            [&]()
                            {
                                refinement_result =
                                    refine_active_indexed_queue_2d(
                                        space,
                                        pending,
                                        requested_refinement_type,
                                        metrics);
                            });
                    });
            });
        record_indexed_refinement_metrics_2d(space, metrics);

        std::vector<int> verification_seed_cells =
            refinement_result.added_active_cells;
        std::vector<int> rebuild_changed_cells =
            refinement_result.added_active_cells;
        rebuild_changed_cells.insert(
            rebuild_changed_cells.end(),
            refinement_result.removed_active_cells.begin(),
            refinement_result.removed_active_cells.end());

        std::sort(
            verification_seed_cells.begin(),
                verification_seed_cells.end());
        std::sort(
            rebuild_changed_cells.begin(),
            rebuild_changed_cells.end());
        rebuild_changed_cells.erase(
            std::unique(
                rebuild_changed_cells.begin(),
                rebuild_changed_cells.end()),
            rebuild_changed_cells.end());
        verification_seed_cells.erase(
            std::unique(
                verification_seed_cells.begin(),
                verification_seed_cells.end()),
            verification_seed_cells.end());

        space.time_phase(
            "refinement.local_conformity_check_wall",
            [&]()
            {
                space.time_phase(
                    "fespace.refinement.assert_active_spatial_conforming",
                    [&]()
                    {
                        space.time_phase(
                            "refinement.local_conformity_check_total",
                            [&]()
                        {
                            try
                            {
                                assert_active_spatial_conforming_local_2d(
                                    space,
                                    verification_seed_cells);
                                space.record_timing_metric(
                                    "refinement.old_local_verifier_disagrees",
                                    0.0);
                            }
                            catch (...)
                            {
                                space.record_timing_metric(
                                    "refinement.old_local_verifier_disagrees",
                                    1.0);
                                throw;
                            }
                        });
                    });
            });
        const bool run_full_conformity_check =
            space.full_conformity_check_after_refinement()
#ifndef NDEBUG
            || true
#endif
            ;
        if (run_full_conformity_check)
        {
            space.time_phase(
                "refinement.full_conformity_check_wall",
                [&]()
                {
                    space.time_phase(
                        "fespace.refinement.assert_active_spatial_conforming_full_debug",
                        [&]()
                        {
                            assert_active_spatial_conforming_full_2d(space);
                        });
                });
        }
        else
        {
            space.record_timing_metric(
                "refinement.full_conformity_check_wall",
                0.0);
        }
        space.time_phase(
            "fespace.refinement.rebuild",
            [&]()
            {
                space.rebuild_incremental_after_refinement(
                    rebuild_changed_cells);
            });
    }
}

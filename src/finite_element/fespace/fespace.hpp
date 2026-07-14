#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../mesh/mesh.hpp"
#include "../detail/memory_usage.hpp"
#include "adjacency/adjacency.hpp"
#include "dofs/dof_entity_key_2d.hpp"
#include "dofs/dof_distribution.hpp"
#include "dofs/dof_handler.hpp"
#include "dofs/physical_dof_coords.hpp"
#include "partition_view.hpp"
#include "policies.hpp"
#include "refinement/refine.hpp"

namespace finite_element
{
    using SearchIndexBuildMode =
        finite_element::fespace::SearchIndexBuildMode;

    struct FESpaceInitializationOptions
    {
        SearchIndexBuildMode search_index = SearchIndexBuildMode::Lazy;
        bool build_refinement_indices = true;
        bool build_edge_interval_index = true;
    };

    enum class FESpaceDiagnosticLevel
    {
        None,
        Summary,
        Detailed
    };

    enum class FESpaceMainClosureQueryMode2D
    {
        ExactAndAncestor,
        ExactAncestorPlusContainment,
        OldBidirectionalDebug
    };

    enum class FESpacePostFlushClosureMode2D
    {
        OffDebug,
        SplitEdgesOnlyDebug,
        SplitAndInheritedEdges,
        AffectedEdges,
        PreSplitNeighbour,
        AllFacesDebug
    };

    // FESpace is defined by its active-cell partition. Those active cells may be mesh
    // leaves, but they do not need to be: a non-leaf mesh cell can remain active in the
    // FE space while its already-existing descendants stay inactive until the active
    // partition is refined further. Mesh leaves are therefore a mesh-storage notion,
    // while active cells are the FE-space notion of "current elements".
    template<typename GeomTraits,
             typename FETraits,
             typename Policy>
    class FESpace
    {
    public:
        using GT = GeomTraits;
        using FETraitsType = FETraits;
        using PolicyType = Policy;
        using MeshType = mesh::Mesh<GeomTraits>;
        using Types = mesh::MeshTypes<GeomTraits>;

        using SpaceTimePoint = typename Types::SpaceTimePoint;
        using SpatialPoint   = typename Types::SpatialPoint;
        using TemporalPoint  = typename Types::TemporalPoint;

        using AdjacencyType = finite_element::fespace::Adjacency<GeomTraits, Policy>;
        using DoFHandlerType = finite_element::fespace::DoFHandler<GeomTraits, FETraits>;
        using DoFType = typename DoFHandlerType::DoFType;
        using ElemTables = typename DoFHandlerType::ElemTables;
        using PartitionViewType = finite_element::fespace::PartitionView<GeomTraits>;
        using TimingCallback = std::function<void(std::string_view, double)>;
        using DofEntityKeyDebug2DType =
            finite_element::fespace::DofEntityKeyDebug2D<SpaceTimePoint>;

        struct ActiveCellHint
        {
            int cell_id = -1;

            [[nodiscard]] bool is_valid() const noexcept
            {
                return cell_id >= 0;
            }

            void reset() noexcept
            {
                cell_id = -1;
            }
        };

        explicit FESpace(MeshType& mesh)
            : partition_(mesh)
        {}

        void initialize(
            const std::vector<int>& active_cells,
            const FESpaceInitializationOptions& options = {})
        {
            auto timer = scoped_timing_("fespace.initialize");
            set_active_cells(active_cells, options);
        }

        // Rebuild all FE-space data derived from the current active partition:
        // first recompute adjacency on the active cells, then fully redistribute
        // and classify DoFs/constraints on that same active partition. The active
        // cell ids are the FE-space notion of "current elements"; mesh leaves are
        // only a lower-level mesh-storage detail and are used much less directly.
        void rebuild()
        {
            const auto rss_before =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.rebuild.memory.rss_before_bytes",
                static_cast<double>(rss_before));
            auto timer = scoped_timing_("fespace.rebuild");
            build_adjacency();
            distribute_dofs();
            timer.stop();
            const auto rss_after =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.rebuild.memory.rss_after_bytes",
                static_cast<double>(rss_after));
            record_timing_metric(
                "fespace.rebuild.memory.rss_delta_bytes",
                static_cast<double>(rss_after) -
                    static_cast<double>(rss_before));
        }

        void rebuild_incremental_after_refinement(
            const std::vector<int>& changed_cells)
        {
            if constexpr (GT::dim_space_v != 2 || GT::dim_time_v != 1)
            {
                static_cast<void>(changed_cells);
                rebuild();
            }
            else
            {
                const auto rss_before =
                    finite_element::detail::current_process_rss_bytes();
                record_timing_metric(
                    "fespace.rebuild.memory.rss_before_bytes",
                    static_cast<double>(rss_before));
                auto timer = scoped_timing_("fespace.rebuild");
                build_adjacency_incremental(changed_cells);
                distribute_dofs_incremental_after_refinement(changed_cells);
                timer.stop();
                const auto rss_after =
                    finite_element::detail::current_process_rss_bytes();
                record_timing_metric(
                    "fespace.rebuild.memory.rss_after_bytes",
                    static_cast<double>(rss_after));
                record_timing_metric(
                    "fespace.rebuild.memory.rss_delta_bytes",
                    static_cast<double>(rss_after) -
                        static_cast<double>(rss_before));
            }
        }

        // Safe replacement of the active partition: after this call the adjacency,
        // DoF layout, and search index all match the new active-cell set.
        void set_active_cells(
            const std::vector<int>& cells,
            const FESpaceInitializationOptions& options = {})
        {
            unsafe_set_active_cells(cells, options);
            rebuild();
        }

        // Lower-level hook for refinement code that updates the active partition
        // incrementally and takes responsibility for rebuilding FE data before
        // exposing the modified space to downstream algorithms.
        void unsafe_set_active_cells(
            const std::vector<int>& cells,
            const FESpaceInitializationOptions& options = {})
        {
            {
                auto timer = scoped_timing_("fespace.active_cell_update");
                partition_.unsafe_set_active_cells(
                    cells,
                    options.search_index,
                    options.build_refinement_indices &&
                        options.build_edge_interval_index);
                record_partition_update_metrics_();
                record_partition_metrics_();
                record_partition_search_index_metrics_();
            }
        }

        // Lower-level hook for refinement code that incrementally swaps cells in
        // the active partition. Callers must rebuild adjacency/DoFs afterwards.
        void unsafe_update_active_cells(
            const std::unordered_set<int>& cells_to_add,
            const std::unordered_set<int>& cells_to_remove,
            typename PartitionViewType::ActiveUpdateMode mode =
                PartitionViewType::ActiveUpdateMode::MembershipAndEdgeIndex)
        {
            {
                auto timer = scoped_timing_("fespace.active_cell_update");
                partition_.unsafe_update_active_cells(
                    cells_to_add,
                    cells_to_remove,
                    mode);
                record_partition_update_metrics_();
                record_partition_metrics_();
                record_partition_search_index_metrics_();
            }
        }

        void build_adjacency()
        {
            auto timer = scoped_timing_("fespace.adjacency_compute");
            auto refinement_timer =
                scoped_timing_("refinement.adjacency_rebuild_wall");
            adjacency_.compute_adjacency(
                active_cells(),
                mesh_ref(),
                [this](std::string_view phase, double seconds)
                {
                    record_timing_(phase, seconds);
                });
            ++adjacency_version_;
        }

        void build_adjacency_incremental(
            const std::vector<int>& changed_cells)
        {
            auto timer = scoped_timing_("fespace.adjacency_compute");
            auto refinement_timer =
                scoped_timing_("refinement.adjacency_rebuild_wall");
            adjacency_.compute_adjacency_incremental(
                active_cells(),
                mesh_ref(),
                changed_cells,
                [this](std::string_view phase, double seconds)
                {
                    record_timing_(phase, seconds);
                });
            ++adjacency_version_;
        }

        void distribute_dofs()
        {
            const auto rss_before =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.dof_distribution.memory.rss_before_bytes",
                static_cast<double>(rss_before));
            auto timer = scoped_timing_("fespace.dof_distribution");
            auto refinement_timer =
                scoped_timing_("refinement.dof_redistribution_wall");
            finite_element::fespace::distribute_dofs(*this);
            ++dof_distribution_version_;
            refinement_timer.stop();
            timer.stop();
            const auto rss_after =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.dof_distribution.memory.rss_after_bytes",
                static_cast<double>(rss_after));
            record_timing_metric(
                "fespace.dof_distribution.memory.rss_delta_bytes",
                static_cast<double>(rss_after) -
                    static_cast<double>(rss_before));
            record_timing_metric(
                "fespace.dof_distribution.memory.dof_handler_estimated_bytes",
                static_cast<double>(dof_handler_.estimated_memory_bytes()));
            record_timing_metric(
                "fespace.dof_distribution.memory.constraints_estimated_bytes",
                static_cast<double>(
                    dof_handler_.constraints_estimated_memory_bytes()));
            record_timing_metric(
                "fespace.dof_distribution.cell_slot_count",
                static_cast<double>(dof_handler_.cell_slot_count()));
            record_timing_metric(
                "fespace.dof_distribution.active_cell_slot_count",
                static_cast<double>(dof_handler_.active_cell_slot_count()));
            if constexpr (GT::dim_space_v != 2 || GT::dim_time_v != 1)
            {
                record_timing_metric(
                    "dof.true_dofs.count",
                    static_cast<double>(dof_handler_.n_true_dofs()));
                record_timing_metric(
                    "dof.constrained_keys.count",
                    static_cast<double>(dof_handler_.n_constrained_dofs()));
                record_timing_metric(
                    "dof.prolongation_nnz.count",
                    static_cast<double>(dof_handler_.prolongation_nonzeros()));
            }
        }

        void distribute_dofs_incremental_after_refinement(
            const std::vector<int>& changed_cells)
        {
            if constexpr (GT::dim_space_v != 2 || GT::dim_time_v != 1)
            {
                static_cast<void>(changed_cells);
                distribute_dofs();
            }
            else
            {
                if (timing_callback_)
                    record_dof_rebuild_diagnostics_2d_(changed_cells);

                // Canonical entity keys are cached per immutable mesh cell and
                // reused by distribute_dofs_2d. True/constrained DoF graphs are
                // still rebuilt globally here because constraint routes may
                // change through neighboring hanging interfaces after a local
                // refinement wave. Keeping this as an explicit fallback prevents
                // accidental partial reuse of stale constraints.
                if (timing_callback_)
                {
                    record_timing_metric("dof_rebuild.fallback_full", 1.0);
                    record_timing_metric("dof_rebuild.fallback_reason", 1.0);
                    record_timing_metric(
                        "dof_rebuild.fallback_reason.global_constraint_rebuild.count",
                        1.0);
                    record_timing_metric(
                        "dof_rebuild.incremental_constraint_reuse_enabled",
                        0.0);
                }
                distribute_dofs();
            }
        }

        void refine(const std::vector<int>& marked)
        {
            // Physical FE refinement: `RefinementType::none` is resolved by
            // the mesh split policy. In 2+1D this gives the generation-based
            // spatial/spacetime alternation.
            const auto rss_before =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.refinement.memory.rss_before_bytes",
                static_cast<double>(rss_before));
            auto timer = scoped_timing_("fespace.refinement");
            auto refinement_timer = scoped_timing_("refinement.total_wall");
            finite_element::fespace::refine(*this, marked);
            refinement_timer.stop();
            timer.stop();
            const auto rss_after =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.refinement.memory.rss_after_bytes",
                static_cast<double>(rss_after));
            record_timing_metric(
                "fespace.refinement.memory.rss_delta_bytes",
                static_cast<double>(rss_after) -
                    static_cast<double>(rss_before));
        }

        void refine(
            const std::vector<int>& marked,
            mesh::RefinementType requested_refinement_type)
        {
            // Explicit refinement requests are reserved for tests, legacy
            // mesh workflows, or named auxiliary algorithms. In particular,
            // `RefinementType::temporal` is not the default physical X/Y path.
            const auto rss_before =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.refinement.memory.rss_before_bytes",
                static_cast<double>(rss_before));
            auto timer = scoped_timing_("fespace.refinement");
            auto refinement_timer = scoped_timing_("refinement.total_wall");
            finite_element::fespace::refine(
                *this,
                marked,
                requested_refinement_type);
            refinement_timer.stop();
            timer.stop();
            const auto rss_after =
                finite_element::detail::current_process_rss_bytes();
            record_timing_metric(
                "fespace.refinement.memory.rss_after_bytes",
                static_cast<double>(rss_after));
            record_timing_metric(
                "fespace.refinement.memory.rss_delta_bytes",
                static_cast<double>(rss_after) -
                    static_cast<double>(rss_before));
        }

        void set_timing_callback(TimingCallback callback)
        {
            timing_callback_ = std::move(callback);
        }

        void set_diagnostic_level(FESpaceDiagnosticLevel level) noexcept
        {
            diagnostic_level_ = level;
        }

        void set_refinement_edge_query_cache_enabled(
            const bool enabled) noexcept
        {
            refinement_edge_query_cache_enabled_ = enabled;
        }

        [[nodiscard]] bool
        refinement_edge_query_cache_enabled() const noexcept
        {
            return refinement_edge_query_cache_enabled_;
        }

        void set_refinement_batch_target_split_cells(
            const std::size_t target) noexcept
        {
            refinement_batch_target_split_cells_ = target == 0 ? 1 : target;
        }

        [[nodiscard]] std::size_t
        refinement_batch_target_split_cells() const noexcept
        {
            return refinement_batch_target_split_cells_;
        }

        void set_post_flush_closure_mode_2d(
            const FESpacePostFlushClosureMode2D mode) noexcept
        {
            post_flush_closure_mode_2d_ = mode;
        }

        [[nodiscard]] FESpacePostFlushClosureMode2D
        post_flush_closure_mode_2d() const noexcept
        {
            return post_flush_closure_mode_2d_;
        }

        [[nodiscard]] bool post_flush_closure_off_debug_2d() const noexcept
        {
            return post_flush_closure_mode_2d_ ==
                FESpacePostFlushClosureMode2D::OffDebug;
        }

        [[nodiscard]] bool
        post_flush_closure_split_edges_only_debug_2d() const noexcept
        {
            return post_flush_closure_mode_2d_ ==
                FESpacePostFlushClosureMode2D::SplitEdgesOnlyDebug;
        }

        [[nodiscard]] bool
        post_flush_closure_split_and_inherited_edges_2d() const noexcept
        {
            return post_flush_closure_mode_2d_ ==
                FESpacePostFlushClosureMode2D::SplitAndInheritedEdges;
        }

        [[nodiscard]] bool
        post_flush_closure_all_faces_debug_2d() const noexcept
        {
            return post_flush_closure_mode_2d_ ==
                FESpacePostFlushClosureMode2D::AllFacesDebug;
        }

        [[nodiscard]] bool
        post_flush_closure_presplit_neighbour_2d() const noexcept
        {
            return post_flush_closure_mode_2d_ ==
                FESpacePostFlushClosureMode2D::PreSplitNeighbour;
        }

        void set_post_flush_affected_containment_only(
            const bool enabled) noexcept
        {
            post_flush_affected_containment_only_ = enabled;
        }

        [[nodiscard]] bool
        post_flush_affected_containment_only() const noexcept
        {
            return post_flush_affected_containment_only_;
        }

        void set_full_conformity_check_after_refinement(
            const bool enabled) noexcept
        {
            full_conformity_check_after_refinement_ = enabled;
        }

        [[nodiscard]] bool
        full_conformity_check_after_refinement() const noexcept
        {
            return full_conformity_check_after_refinement_;
        }

        void set_main_closure_query_mode_2d(
            const FESpaceMainClosureQueryMode2D mode) noexcept
        {
            main_closure_query_mode_2d_ = mode;
        }

        [[nodiscard]] FESpaceMainClosureQueryMode2D
        main_closure_query_mode_2d() const noexcept
        {
            return main_closure_query_mode_2d_;
        }

        [[nodiscard]] bool
        main_closure_query_uses_containment_2d() const noexcept
        {
            return main_closure_query_mode_2d_ !=
                FESpaceMainClosureQueryMode2D::ExactAndAncestor;
        }

        [[nodiscard]] FESpaceDiagnosticLevel diagnostic_level() const noexcept
        {
            return diagnostic_level_;
        }

        [[nodiscard]] bool detailed_fespace_diagnostics_enabled() const noexcept
        {
            return diagnostic_level_ == FESpaceDiagnosticLevel::Detailed;
        }

        void clear_timing_callback()
        {
            timing_callback_ = {};
        }

        template<class Fn>
        void time_phase(std::string_view phase, Fn&& fn)
        {
            auto timer = scoped_timing_(phase);
            std::forward<Fn>(fn)();
        }

        void record_timing_metric(std::string_view phase, double value)
        {
            record_timing_(phase, value);
        }

        [[nodiscard]] bool has_cached_dof_entity_key_2d(
            int cell_id,
            int local_index) const
        {
            if constexpr (GT::dim_space_v != 2 || GT::dim_time_v != 1)
            {
                static_cast<void>(cell_id);
                static_cast<void>(local_index);
                return false;
            }
            else
            {
                return cell_id >= 0 &&
                       local_index >= 0 &&
                       local_index < ElemTables::dofs_per_cell &&
                       static_cast<std::size_t>(cell_id) <
                           dof_entity_key_cache_valid_2d_.size() &&
                       dof_entity_key_cache_valid_2d_[static_cast<std::size_t>(cell_id)]
                           [static_cast<std::size_t>(local_index)] != 0;
            }
        }

        [[nodiscard]] const DofEntityKeyDebug2DType&
        cached_dof_entity_key_2d(int cell_id, int local_index)
        {
            static_assert(GT::dim_space_v == 2 && GT::dim_time_v == 1,
                          "cached_dof_entity_key_2d is only for 2+1D spaces.");
            if (cell_id < 0 ||
                local_index < 0 ||
                local_index >= ElemTables::dofs_per_cell)
            {
                throw std::runtime_error(
                    "cached_dof_entity_key_2d: invalid cell/local index.");
            }

            const auto needed = static_cast<std::size_t>(mesh_ref().n_cells());
            if (dof_entity_key_cache_2d_.size() < needed)
            {
                dof_entity_key_cache_2d_.resize(needed);
                dof_entity_key_cache_valid_2d_.resize(needed);
            }

            const auto cell_idx = static_cast<std::size_t>(cell_id);
            const auto local_idx = static_cast<std::size_t>(local_index);
            if (!dof_entity_key_cache_valid_2d_[cell_idx][local_idx])
            {
                dof_entity_key_cache_2d_[cell_idx][local_idx] =
                    finite_element::fespace::make_dof_entity_key_2d(
                        *this,
                        cell_id,
                        local_index);
                dof_entity_key_cache_valid_2d_[cell_idx][local_idx] = 1;
            }

            return dof_entity_key_cache_2d_[cell_idx][local_idx];
        }

        [[nodiscard]] bool is_active_cell(int cell_id) const
        {
            return partition_.is_active_cell(cell_id);
        }

        [[nodiscard]] int find_active_cell(const SpaceTimePoint& p) const
        {
            record_find_active_cell_metric_("find_active_cell.query_count", 1.0);
            return find_active_cell_unhinted_(p);
        }

        [[nodiscard]] int find_active_cell(int cell_id, const SpaceTimePoint& p) const
        {
            record_find_active_cell_metric_("find_active_cell.query_count", 1.0);

            const int hinted_match =
                find_active_cell_from_hint_unmetered_(cell_id, p);
            if (hinted_match >= 0)
            {
                const int canonical_match =
                    canonicalize_hinted_active_cell_match_(
                        hinted_match,
                        p);
                if (canonical_match == hinted_match)
                {
                    record_find_active_cell_metric_("find_active_cell.hint_hits", 1.0);
                }
                return canonical_match;
            }

            return find_active_cell_unhinted_(p);
        }

        [[nodiscard]] int find_active_cell(const SpaceTimePoint& p, ActiveCellHint& hint) const
        {
            record_find_active_cell_metric_("find_active_cell.query_count", 1.0);

            if (hint.is_valid())
            {
                const int hinted_cell =
                    find_active_cell_from_hint_unmetered_(hint.cell_id, p);
                if (hinted_cell >= 0)
                {
                    const int canonical_cell =
                        canonicalize_hinted_active_cell_match_(
                            hinted_cell,
                            p);
                    if (canonical_cell == hinted_cell)
                    {
                        record_find_active_cell_metric_("find_active_cell.hint_hits", 1.0);
                    }
                    hint.cell_id = canonical_cell;
                    return canonical_cell;
                }
            }

            const int cell_id = find_active_cell_unhinted_(p);
            if (cell_id >= 0)
                hint.cell_id = cell_id;
            else
                hint.reset();

            return cell_id;
        }

        [[nodiscard]] int find_active_cell_by_scan(
            const SpaceTimePoint& p) const
        {
            record_find_active_cell_metric_("find_active_cell.query_count", 1.0);
            record_find_active_cell_metric_(
                "find_active_cell.fallback_full_scans",
                1.0);
            partition_.record_search_index_fallback_scan();
            record_partition_search_index_metrics_();

            int best_boundary_match = -1;
            int best_boundary_rank = std::numeric_limits<int>::max();
            for (int rank = 0;
                 rank < static_cast<int>(active_cells().size());
                 ++rank)
            {
                const int cell_id =
                    active_cells()[static_cast<std::size_t>(rank)];
                if (!mesh_ref().contains_coord(cell_id, p))
                    continue;

                if (point_is_strictly_inside_active_cell_(cell_id, p))
                    return cell_id;

                if (rank < best_boundary_rank)
                {
                    best_boundary_match = cell_id;
                    best_boundary_rank = rank;
                }
            }
            return best_boundary_match;
        }

        [[nodiscard]] const std::vector<int>& active_cells() const noexcept
        {
            return partition_.active_cells();
        }

        [[nodiscard]] const MeshType& mesh_ref() const noexcept
        {
            return partition_.mesh_ref();
        }

        [[nodiscard]] std::vector<int>& unsafe_active_cells_ref() noexcept
        {
            return partition_.unsafe_active_cells_ref();
        }

        [[nodiscard]] MeshType& unsafe_mesh_ref() noexcept
        {
            return partition_.mesh_ref();
        }

        [[nodiscard]] PartitionViewType& partition_view() noexcept
        {
            return partition_;
        }

        [[nodiscard]] const PartitionViewType& partition_view() const noexcept
        {
            return partition_;
        }

        [[nodiscard]] std::uint64_t active_version() const noexcept
        {
            return partition_.active_version();
        }

        [[nodiscard]] std::uint64_t adjacency_version() const noexcept
        {
            return adjacency_version_;
        }

        [[nodiscard]] std::uint64_t dof_distribution_version() const noexcept
        {
            return dof_distribution_version_;
        }

        [[nodiscard]] AdjacencyType& adjacency_ref() noexcept { return adjacency_; }
        [[nodiscard]] const AdjacencyType& adjacency_ref() const noexcept { return adjacency_; }

        [[nodiscard]] DoFHandlerType& dof_handler_ref() noexcept { return dof_handler_; }
        [[nodiscard]] const DoFHandlerType& dof_handler_ref() const noexcept { return dof_handler_; }

        void write_mesh_binary(const std::filesystem::path& output_dir, const std::string& filename) const;
        void write_dofs_binary(const std::filesystem::path& output_dir, const std::string& filename) const;

    private:
        void record_partition_metrics_()
        {
            auto metrics_timer =
                scoped_timing_("refinement.partition_metric_recording_wall");

            if (diagnostic_level_ == FESpaceDiagnosticLevel::None)
                return;

            record_timing_metric(
                "partition.active_cells.count",
                static_cast<double>(partition_.active_cells().size()));

            if (diagnostic_level_ == FESpaceDiagnosticLevel::Detailed)
            {
                auto memory_timer =
                    scoped_timing_("partition.metrics.memory_estimate_wall");
                record_timing_metric(
                    "partition.memory.estimated_bytes",
                    static_cast<double>(partition_.estimated_memory_bytes()));
                if constexpr (GT::dim_space_v == 2 && GT::dim_time_v == 1)
                {
                    const auto& edge_index =
                        partition_.active_edge_interval_index_2d();
                    record_timing_metric(
                        "edge_index.memory.estimated_bytes",
                        static_cast<double>(
                            edge_index.estimated_memory_bytes()));
                }
                else
                {
                    record_timing_metric(
                        "edge_index.memory.estimated_bytes",
                        0.0);
                }
            }
            else
            {
                record_timing_metric(
                    "partition.metrics.memory_estimate_wall",
                    0.0);
            }

            if constexpr (GT::dim_space_v == 2 && GT::dim_time_v == 1)
            {
                const auto& edge_index =
                    partition_.active_edge_interval_index_2d();
                record_timing_metric(
                    "partition.edge_interval_records.count",
                    static_cast<double>(edge_index.n_records()));
                record_timing_metric(
                    "edge_index.records_active",
                    static_cast<double>(edge_index.n_active_records()));
                record_timing_metric(
                    "edge_index.records_active.count",
                    static_cast<double>(edge_index.n_active_records()));
                record_timing_metric(
                    "edge_index.records_inactive",
                    static_cast<double>(edge_index.n_inactive_records()));
                record_timing_metric(
                    "edge_index.records_inactive.count",
                    static_cast<double>(edge_index.n_inactive_records()));
                record_timing_metric(
                    "edge_index.support_line_group_compaction_count",
                    static_cast<double>(
                        edge_index.support_line_group_compaction_count()));
                record_timing_metric(
                    "edge_index.support_line_group_compaction_count.count",
                    static_cast<double>(
                        edge_index.support_line_group_compaction_count()));
                record_timing_metric(
                    "edge_index.support_line_group_compaction_records_removed",
                    static_cast<double>(
                        edge_index
                            .support_line_group_compaction_records_removed()));
                record_timing_metric(
                    "edge_index.support_line_group_compaction_records_removed.count",
                    static_cast<double>(
                        edge_index
                            .support_line_group_compaction_records_removed()));
                record_timing_metric(
                    "edge_index.support_line_group_compaction_seconds",
                    edge_index.support_line_group_compaction_seconds_total());
                record_timing_metric(
                    "edge_index.edge_group_compaction_count",
                    static_cast<double>(edge_index.edge_group_compaction_count()));
                record_timing_metric(
                    "edge_index.edge_group_compaction_count.count",
                    static_cast<double>(edge_index.edge_group_compaction_count()));
                record_timing_metric(
                    "edge_index.edge_group_compaction_records_removed",
                    static_cast<double>(
                        edge_index.edge_group_compaction_records_removed()));
                record_timing_metric(
                    "edge_index.edge_group_compaction_records_removed.count",
                    static_cast<double>(
                        edge_index.edge_group_compaction_records_removed()));
                record_timing_metric(
                    "edge_index.edge_group_compaction_seconds",
                    edge_index.edge_group_compaction_seconds_total());
                record_timing_metric(
                    "edge_index.spatial_vertex_group_compaction_count",
                    static_cast<double>(
                        edge_index.spatial_vertex_group_compaction_count()));
                record_timing_metric(
                    "edge_index.spatial_vertex_group_compaction_count.count",
                    static_cast<double>(
                        edge_index.spatial_vertex_group_compaction_count()));
                record_timing_metric(
                    "edge_index.spatial_vertex_group_compaction_records_removed",
                    static_cast<double>(
                        edge_index
                            .spatial_vertex_group_compaction_records_removed()));
                record_timing_metric(
                    "edge_index.spatial_vertex_group_compaction_records_removed.count",
                    static_cast<double>(
                        edge_index
                            .spatial_vertex_group_compaction_records_removed()));
                record_timing_metric(
                    "edge_index.spatial_vertex_group_compaction_seconds",
                    edge_index
                        .spatial_vertex_group_compaction_seconds_total());

                if (diagnostic_level_ == FESpaceDiagnosticLevel::Detailed)
                {
                    record_timing_metric(
                        "edge_index.group_compaction_memory_before_bytes",
                        static_cast<double>(
                            edge_index
                                .group_compaction_memory_before_bytes_total()));
                    record_timing_metric(
                        "edge_index.group_compaction_memory_after_bytes",
                        static_cast<double>(
                            edge_index
                                .group_compaction_memory_after_bytes_total()));
                    record_timing_metric(
                        "edge_index.global_compaction_memory_before_bytes",
                        static_cast<double>(
                            edge_index
                                .global_compaction_memory_before_bytes_total()));
                    record_timing_metric(
                        "edge_index.global_compaction_memory_after_bytes",
                        static_cast<double>(
                            edge_index
                                .global_compaction_memory_after_bytes_total()));
                }
            }
            else
            {
                record_timing_metric(
                    "partition.edge_interval_records.count",
                    0.0);
                record_timing_metric("edge_index.records_active", 0.0);
                record_timing_metric("edge_index.records_active.count", 0.0);
                record_timing_metric("edge_index.records_inactive", 0.0);
                record_timing_metric("edge_index.records_inactive.count", 0.0);
            }
        }

        void record_partition_update_metrics_()
        {
            const auto& timing = partition_.last_update_timing();
            record_timing_metric("partition.update.total", timing.total);
            record_timing_metric(
                "partition.update.membership_refresh",
                timing.membership_refresh);
            record_timing_metric(
                "partition.update.membership_remove",
                timing.membership_remove);
            record_timing_metric(
                "partition.update.membership_add",
                timing.membership_add);
            record_timing_metric(
                "partition.update.active_vector_rebuild",
                timing.active_vector_rebuild);
            record_timing_metric(
                "partition.update.active_search_index_rebuild",
                timing.active_search_index_rebuild);
            record_timing_metric(
                "edge_index.remove_total",
                timing.edge_index_remove_total);
            record_timing_metric(
                "edge_index.add_total",
                timing.edge_index_add_total);
            record_timing_metric(
                "edge_index.rebuild_total",
                timing.edge_index_rebuild_total);
            record_timing_metric(
                "edge_index.remove_cell_records_touched",
                static_cast<double>(
                    timing.edge_index_remove_cell_records_touched));
            record_timing_metric(
                "edge_index.remove_cell_records_touched.count",
                static_cast<double>(
                    timing.edge_index_remove_cell_records_touched));
            record_timing_metric(
                "edge_index.compaction_count",
                static_cast<double>(timing.edge_index_compaction_count));
            record_timing_metric(
                "edge_index.compaction_count.count",
                static_cast<double>(timing.edge_index_compaction_count));
            record_timing_metric(
                "edge_index.compaction_seconds",
                timing.edge_index_compaction_seconds);
            record_timing_metric(
                "refinement.edge_index_update_wall",
                timing.edge_index_remove_total + timing.edge_index_add_total +
                    timing.edge_index_rebuild_total +
                    timing.edge_index_compaction_seconds);

            // Compatibility name used by earlier benchmark reports.
            record_timing_metric(
                "fespace.active_cell_search_index_rebuild",
                timing.active_search_index_rebuild);
        }

        void record_partition_search_index_metrics_() const
        {
            const auto metrics_start = std::chrono::steady_clock::now();
            const auto& stats = partition_.search_index_stats();
            const auto record_count =
                [&](std::string_view name, const std::uint64_t value)
                {
                    const double count = static_cast<double>(value);
                    record_find_active_cell_metric_(name, count);
                    std::string count_name(name);
                    count_name += ".count";
                    record_find_active_cell_metric_(count_name, count);
                };

            record_count(
                "partition.search_index.build_eager_count",
                stats.build_eager_count);
            record_count(
                "partition.search_index.build_lazy_count",
                stats.build_lazy_count);
            record_count(
                "partition.search_index.disabled_count",
                stats.disabled_count);
            record_count(
                "partition.search_index.fallback_scan_count",
                stats.fallback_scan_count);
            record_find_active_cell_metric_(
                "partition.search_index.rebuild_seconds",
                stats.rebuild_seconds);
            record_find_active_cell_metric_(
                "partition.metrics.search_index_stats_wall",
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - metrics_start)
                    .count());
        }

        void record_find_active_cell_metric_(
            std::string_view phase,
            double value) const
        {
            if (timing_callback_)
                timing_callback_(phase, value);
        }

        [[nodiscard]] int find_active_cell_unhinted_(
            const SpaceTimePoint& p) const
        {
            int indexed_match = -1;
            try
            {
                indexed_match =
                    partition_.find_active_cell_from_search_index(p);
                record_partition_search_index_metrics_();
            }
            catch (...)
            {
                record_partition_search_index_metrics_();
                throw;
            }
            if (indexed_match >= 0)
            {
                record_find_active_cell_metric_(
                    "find_active_cell.index_hits",
                    1.0);
                return indexed_match;
            }

            for (const int root_cell_id : mesh_ref().root_cell_ids())
            {
                const int descendant_match =
                    find_active_descendant_cell_iterative(root_cell_id, p);
                if (descendant_match >= 0)
                    return descendant_match;
            }

            record_find_active_cell_metric_(
                "find_active_cell.fallback_full_scans",
                1.0);
            partition_.record_search_index_fallback_scan();
            record_partition_search_index_metrics_();
            for (const int cell_id : active_cells())
            {
                if (mesh_ref().contains_coord(cell_id, p))
                    return cell_id;
            }
            return -1;
        }

        [[nodiscard]] int find_active_cell_from_hint_unmetered_(
            const int cell_id,
            const SpaceTimePoint& p) const
        {
            const auto& mesh = mesh_ref();
            if (!mesh.valid_cell_id(cell_id))
                return -1;
            if (!mesh.contains_coord(cell_id, p))
                return -1;

            const int descendant_match =
                find_active_descendant_cell_iterative(cell_id, p);
            if (descendant_match >= 0)
                return descendant_match;

            int ancestor = mesh.cell(cell_id).parent_id;
            std::size_t guard = 0;
            while (ancestor >= 0)
            {
                if (!mesh.valid_cell_id(ancestor))
                    return -1;
                if (is_active_cell(ancestor) &&
                    mesh.contains_coord(ancestor, p))
                {
                    return ancestor;
                }

                ancestor = mesh.cell(ancestor).parent_id;
                ++guard;
                if (guard > mesh.n_cells())
                    return -1;
            }

            return -1;
        }

        [[nodiscard]] int canonicalize_hinted_active_cell_match_(
            const int hinted_match,
            const SpaceTimePoint& p) const
        {
            int indexed_match = -1;
            try
            {
                indexed_match =
                    partition_.find_active_cell_from_search_index(p);
                record_partition_search_index_metrics_();
            }
            catch (...)
            {
                record_partition_search_index_metrics_();
                throw;
            }
            if (indexed_match >= 0 && indexed_match != hinted_match)
            {
                record_find_active_cell_metric_(
                    "find_active_cell.index_hits",
                    1.0);
                return indexed_match;
            }
            return hinted_match;
        }

        [[nodiscard]] bool point_is_strictly_inside_active_cell_(
            const int cell_id,
            const SpaceTimePoint& p) const
        {
            constexpr double tol = 1.0e-12;

            if (!mesh_ref().valid_cell_id(cell_id))
                return false;

            const auto& mesh = mesh_ref();
            const auto& cell = mesh.cell(cell_id);
            const double t0 =
                mesh.temporal_vertices()[
                    static_cast<std::size_t>(
                        cell.temporal_vertex_ids[0])][0];
            const double t1 =
                mesh.temporal_vertices()[
                    static_cast<std::size_t>(
                        cell.temporal_vertex_ids[1])][0];
            const double t = p[GT::dim_space_v];
            if (!(t0 + tol < t && t < t1 - tol))
                return false;

            if constexpr (GT::dim_space_v == 1)
            {
                const double x0 =
                    mesh.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[0])][0];
                const double x1 =
                    mesh.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[1])][0];
                const double xl = std::min(x0, x1);
                const double xr = std::max(x0, x1);
                return xl + tol < p[0] && p[0] < xr - tol;
            }
            else if constexpr (GT::dim_space_v == 2)
            {
                const auto& v0 =
                    mesh.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[0])];
                const auto& v1 =
                    mesh.spatial_vertices()[
                        static_cast<std::size_t>(
                            cell.spatial_vertex_ids[1])];
                const auto& v2 =
                    mesh.spatial_vertices()[
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

        [[nodiscard]] int find_active_descendant_cell_iterative(
            int root_cell_id,
            const SpaceTimePoint& p) const
        {
            const auto& mesh = mesh_ref();
            if (root_cell_id < 0 ||
                static_cast<std::size_t>(root_cell_id) >= mesh.n_cells())
                return -1;
            if (!mesh.contains_coord(root_cell_id, p))
                return -1;

            std::vector<int> stack;
            stack.push_back(root_cell_id);

            int best_match = -1;

            while (!stack.empty())
            {
                const int cell_id = stack.back();
                stack.pop_back();

                const auto& cell = mesh.cell(cell_id);
                if (is_active_cell(cell_id))
                    best_match = cell_id;

                for (auto it = cell.children.rbegin(); it != cell.children.rend(); ++it)
                {
                    if (*it < 0 ||
                        static_cast<std::size_t>(*it) >= mesh.n_cells())
                        continue;
                    if (mesh.contains_coord(*it, p))
                        stack.push_back(*it);
                }
            }

            return best_match;
        }

        void record_dof_rebuild_diagnostics_2d_(
            const std::vector<int>& changed_cells)
        {
            static_assert(GT::dim_space_v == 2 && GT::dim_time_v == 1,
                          "2D DoF rebuild diagnostics require 2+1D spaces.");

            std::unordered_set<int> touched_cells;
            touched_cells.reserve(changed_cells.size() * 4U + 16U);
            for (const int cell_id : changed_cells)
            {
                if (cell_id >= 0)
                    touched_cells.insert(cell_id);
            }

            for (const auto& iface : adjacency_.spatial_interfaces)
            {
                if (touched_cells.count(iface.slave_cell) != 0 ||
                    touched_cells.count(iface.master_cell) != 0)
                {
                    if (iface.slave_cell >= 0)
                        touched_cells.insert(iface.slave_cell);
                    if (iface.master_cell >= 0)
                        touched_cells.insert(iface.master_cell);
                }
            }

            for (const auto& iface : adjacency_.temporal_interfaces)
            {
                if (touched_cells.count(iface.slave_cell) != 0 ||
                    touched_cells.count(iface.master_cell) != 0)
                {
                    if (iface.slave_cell >= 0)
                        touched_cells.insert(iface.slave_cell);
                    if (iface.master_cell >= 0)
                        touched_cells.insert(iface.master_cell);
                }
            }

            std::size_t active_touched = 0;
            for (const int cell_id : touched_cells)
            {
                if (is_active_cell(cell_id))
                    ++active_touched;
            }

            record_timing_metric(
                "dof_rebuild.active_cells_total",
                static_cast<double>(active_cells().size()));
            record_timing_metric(
                "dof_rebuild.active_cells_touched",
                static_cast<double>(active_touched));
            record_timing_metric(
                "dof_rebuild.changed_cells",
                static_cast<double>(changed_cells.size()));
            record_timing_metric(
                "dof_rebuild.touched_cells",
                static_cast<double>(touched_cells.size()));
            record_timing_metric(
                "dof_rebuild.touched_active_fraction",
                active_cells().empty()
                    ? 0.0
                    : static_cast<double>(active_touched) /
                          static_cast<double>(active_cells().size()));
        }

        class ScopedPhaseTimer
        {
        public:
            ScopedPhaseTimer() noexcept = default;

            ScopedPhaseTimer(const ScopedPhaseTimer&) = delete;
            ScopedPhaseTimer& operator=(const ScopedPhaseTimer&) = delete;

            ScopedPhaseTimer(ScopedPhaseTimer&& other) noexcept
                : space_(std::exchange(other.space_, nullptr)),
                  phase_(other.phase_),
                  start_(other.start_),
                  active_(std::exchange(other.active_, false))
            {}

            ScopedPhaseTimer& operator=(ScopedPhaseTimer&& other) noexcept
            {
                if (this == &other)
                    return *this;

                stop_noexcept_();
                space_ = std::exchange(other.space_, nullptr);
                phase_ = other.phase_;
                start_ = other.start_;
                active_ = std::exchange(other.active_, false);
                return *this;
            }

            ~ScopedPhaseTimer() noexcept
            {
                stop_noexcept_();
            }

        private:
            using Clock = std::chrono::steady_clock;

            friend class FESpace;

            ScopedPhaseTimer(FESpace& space, std::string_view phase)
                : space_(&space),
                  phase_(phase),
                  start_(Clock::now()),
                  active_(true)
            {}

            void stop()
            {
                if (!active_ || space_ == nullptr)
                    return;

                const auto end = Clock::now();
                const double seconds =
                    std::chrono::duration<double>(end - start_).count();
                active_ = false;
                space_->record_timing_(phase_, seconds);
            }

            void stop_noexcept_() noexcept
            {
                try
                {
                    stop();
                }
                catch (...)
                {
                }
            }

            FESpace* space_ = nullptr;
            std::string_view phase_{};
            Clock::time_point start_{};
            bool active_ = false;
        };

        [[nodiscard]] ScopedPhaseTimer scoped_timing_(std::string_view phase)
        {
            if (!timing_callback_)
                return {};

            return ScopedPhaseTimer(*this, phase);
        }

        void record_timing_(std::string_view phase, double seconds)
        {
            if (timing_callback_)
                timing_callback_(phase, seconds);
        }

        PartitionViewType partition_;

        AdjacencyType adjacency_{};
        DoFHandlerType dof_handler_{};
        std::uint64_t adjacency_version_ = 0;
        std::uint64_t dof_distribution_version_ = 0;
        FESpaceDiagnosticLevel diagnostic_level_ =
            FESpaceDiagnosticLevel::Summary;
        bool refinement_edge_query_cache_enabled_ = true;
        std::size_t refinement_batch_target_split_cells_ = 32;
        FESpacePostFlushClosureMode2D post_flush_closure_mode_2d_ =
            FESpacePostFlushClosureMode2D::PreSplitNeighbour;
        bool post_flush_affected_containment_only_ = false;
        bool full_conformity_check_after_refinement_ = false;
        FESpaceMainClosureQueryMode2D main_closure_query_mode_2d_ =
            FESpaceMainClosureQueryMode2D::ExactAndAncestor;
        std::vector<
            std::array<DofEntityKeyDebug2DType, ElemTables::dofs_per_cell>>
            dof_entity_key_cache_2d_{};
        std::vector<std::array<char, ElemTables::dofs_per_cell>>
            dof_entity_key_cache_valid_2d_{};
        TimingCallback timing_callback_{};
    };
}

#include "io/write_mesh_binary.hpp"
#include "io/write_dofs_binary.hpp"

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cctype>
#include <cstdint>
#include <cstddef>
#include <exception>
#include <functional>
#include <iomanip>
#include <memory>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../core/openmp.hpp"
#include "../../linear_algebra/concepts/solver.hpp"
#include "../../linear_algebra/operations/linalg_ops.hpp"
#include "../../linear_algebra/system/saddle_point_system.hpp"
#include "../../linear_algebra/system/solve.hpp"

#include "../assembly/detail/active_cell_locator_time_slab.hpp"
#include "../assembly/detail/openmp_assembly.hpp"
#include "../assembly/error_system/assemble_local_error_problem.hpp"
#include "../assembly/error_system/cell_first_local_error_blocks_2d.hpp"
#include "../assembly/error_system/dense_local_error_blocks_2d.hpp"
#include "../assembly/error_system/local_unified_cell_state_cache_2d.hpp"
#include "../assembly/error_system/shared_local_error_context_1d.hpp"
#include "../assembly/error_system/shared_local_error_context_2d.hpp"
#include "../coefficients/diffusion_coefficient.hpp"
#include "../detail/timing.hpp"
#include "../detail/memory_usage.hpp"
#include "../detail/space_time_capabilities.hpp"
#include "../detail/cell_geometry_cache.hpp"
#include "../error_fespace/patch_flux_function_1d.hpp"
#include "../error_fespace/patch_flux_space_1d.hpp"
#include "../error_fespace/patch_functions_2d.hpp"
#include "../error_fespace/patch_rt_flux_space_time_2d.hpp"
#include "../error_fespace/patch_scalar_function.hpp"
#include "../error_fespace/patch_scalar_space.hpp"
#include "../error_fespace/patch_scalar_space_time_2d.hpp"

#include "time_slab_builder.hpp"
#include "detail/time_slab_error_indicator_detail.hpp"
#include "time_slab_cellwise_errors.hpp"
#include "time_slab_edge_patch_builder.hpp"
#include "time_slab_vertex_patch_builder.hpp"

#ifndef APF_DISABLE_FUSED_FLUX_DIAGNOSTICS_2D
#define APF_DISABLE_FUSED_FLUX_DIAGNOSTICS_2D 0
#endif

namespace finite_element::time_slabs
{
    namespace detail
    {
        struct TimeSlabLocalErrorPatchProfile
        {
            double slab_count = 0.0;
            double patch_count = 0.0;
            double patch_cells_min = 0.0;
            double patch_cells_max = 0.0;
            double patch_cells_mean = 0.0;
            double patch_cells_median = 0.0;
            double patch_dofs_min = 0.0;
            double patch_dofs_max = 0.0;
            double patch_dofs_mean = 0.0;
            double patch_dofs_distinct_sizes = 0.0;
            double patch_dofs_mode_size = 0.0;
            double patch_dofs_mode_count = 0.0;
            double flux_dofs_distinct_sizes = 0.0;
            double flux_dofs_mode_size = 0.0;
            double flux_dofs_mode_count = 0.0;
            double scalar_dofs_distinct_sizes = 0.0;
            double scalar_dofs_mode_size = 0.0;
            double scalar_dofs_mode_count = 0.0;
            double reduced_scalar_basis_candidate_count = 0.0;
        };

        inline double median_or_zero(std::vector<double> values)
        {
            if (values.empty())
                return 0.0;
            std::sort(values.begin(), values.end());
            const auto mid = values.size() / 2u;
            if (values.size() % 2u == 1u)
                return values[mid];
            return 0.5 * (values[mid - 1u] + values[mid]);
        }

        struct SizeModeSummary
        {
            double distinct_sizes = 0.0;
            double mode_size = 0.0;
            double mode_count = 0.0;
        };

        inline SizeModeSummary size_mode_summary(std::vector<double> values)
        {
            SizeModeSummary summary;
            if (values.empty())
                return summary;

            std::sort(values.begin(), values.end());
            double current_value = values.front();
            std::size_t current_count = 0;
            std::size_t best_count = 0;
            double best_value = current_value;
            std::size_t distinct = 0;
            for (const double value : values)
            {
                if (value != current_value)
                {
                    ++distinct;
                    if (current_count > best_count)
                    {
                        best_count = current_count;
                        best_value = current_value;
                    }
                    current_value = value;
                    current_count = 0;
                }
                ++current_count;
            }
            ++distinct;
            if (current_count > best_count)
            {
                best_count = current_count;
                best_value = current_value;
            }

            summary.distinct_sizes = static_cast<double>(distinct);
            summary.mode_size = best_value;
            summary.mode_count = static_cast<double>(best_count);
            return summary;
        }

        template<class PatchSetType, class FluxSpaces, class ScalarSpaces>
        TimeSlabLocalErrorPatchProfile make_local_error_patch_profile(
            const PatchSetType& patch_set,
            const FluxSpaces& flux_spaces,
            const ScalarSpaces& scalar_spaces,
            int slab_count)
        {
            TimeSlabLocalErrorPatchProfile profile;
            profile.slab_count = static_cast<double>(std::max(0, slab_count));
            profile.patch_count =
                static_cast<double>(std::max(0, patch_set.n_patches()));

            std::vector<double> cell_counts;
            std::vector<double> dof_counts;
            std::vector<double> flux_dof_counts;
            std::vector<double> scalar_dof_counts;
            cell_counts.reserve(
                static_cast<std::size_t>(std::max(0, patch_set.n_patches())));
            dof_counts.reserve(cell_counts.capacity());
            flux_dof_counts.reserve(cell_counts.capacity());
            scalar_dof_counts.reserve(cell_counts.capacity());

            int patch_id = 0;
            for (const auto& patch : patch_set.patches())
            {
                cell_counts.push_back(static_cast<double>(patch.n_cells));
                double n_flux_dofs = 0.0;
                double n_scalar_dofs = 0.0;
                const auto index = static_cast<std::size_t>(patch_id);
                if (index < flux_spaces.size())
                    n_flux_dofs =
                        static_cast<double>(flux_spaces[index].n_dofs());
                if (index < scalar_spaces.size())
                    n_scalar_dofs =
                        static_cast<double>(scalar_spaces[index].n_dofs());
                const double n_dofs = n_flux_dofs + n_scalar_dofs;
                dof_counts.push_back(n_dofs);
                flux_dof_counts.push_back(n_flux_dofs);
                scalar_dof_counts.push_back(n_scalar_dofs);
                if (n_scalar_dofs > 1.0)
                    profile.reduced_scalar_basis_candidate_count += 1.0;
                ++patch_id;
            }

            if (!cell_counts.empty())
            {
                const auto [min_it, max_it] =
                    std::minmax_element(cell_counts.begin(), cell_counts.end());
                profile.patch_cells_min = *min_it;
                profile.patch_cells_max = *max_it;
                double sum = 0.0;
                for (const double value : cell_counts)
                    sum += value;
                profile.patch_cells_mean =
                    sum / static_cast<double>(cell_counts.size());
                profile.patch_cells_median =
                    median_or_zero(std::move(cell_counts));
            }

            if (!dof_counts.empty())
            {
                const auto [min_it, max_it] =
                    std::minmax_element(dof_counts.begin(), dof_counts.end());
                profile.patch_dofs_min = *min_it;
                profile.patch_dofs_max = *max_it;
                double sum = 0.0;
                for (const double value : dof_counts)
                    sum += value;
                profile.patch_dofs_mean =
                    sum / static_cast<double>(dof_counts.size());
            }
            const auto patch_mode = size_mode_summary(dof_counts);
            profile.patch_dofs_distinct_sizes = patch_mode.distinct_sizes;
            profile.patch_dofs_mode_size = patch_mode.mode_size;
            profile.patch_dofs_mode_count = patch_mode.mode_count;
            const auto flux_mode = size_mode_summary(std::move(flux_dof_counts));
            profile.flux_dofs_distinct_sizes = flux_mode.distinct_sizes;
            profile.flux_dofs_mode_size = flux_mode.mode_size;
            profile.flux_dofs_mode_count = flux_mode.mode_count;
            const auto scalar_mode =
                size_mode_summary(std::move(scalar_dof_counts));
            profile.scalar_dofs_distinct_sizes = scalar_mode.distinct_sizes;
            profile.scalar_dofs_mode_size = scalar_mode.mode_size;
            profile.scalar_dofs_mode_count = scalar_mode.mode_count;

            return profile;
        }

        inline void record_local_error_patch_profile(
            const finite_element::detail::TimingRecorder& timing,
            const TimeSlabLocalErrorPatchProfile& profile,
            bool vertex_patches)
        {
            timing.add(
                "time_slab.local_error_solves.time_slab_count.count",
                profile.slab_count);
            timing.add(
                "time_slab.local_error_solves.vertex_patch_count.count",
                vertex_patches ? profile.patch_count : 0.0);
            timing.add(
                "time_slab.local_error_solves.edge_patch_count.count",
                vertex_patches ? 0.0 : profile.patch_count);
            timing.add(
                "time_slab.local_error_solves.patch_cells_min.count",
                profile.patch_cells_min);
            timing.add(
                "time_slab.local_error_solves.patch_cells_max.count",
                profile.patch_cells_max);
            timing.add(
                "time_slab.local_error_solves.patch_cells_mean.count",
                profile.patch_cells_mean);
            timing.add(
                "time_slab.local_error_solves.patch_cells_median.count",
                profile.patch_cells_median);
            timing.add(
                "time_slab.local_error_solves.patch_dofs_min.count",
                profile.patch_dofs_min);
            timing.add(
                "time_slab.local_error_solves.patch_dofs_max.count",
                profile.patch_dofs_max);
            timing.add(
                "time_slab.local_error_solves.patch_dofs_mean.count",
                profile.patch_dofs_mean);
            timing.add(
                "time_slab.local_error_solves.patch_dofs_distinct_sizes.count",
                profile.patch_dofs_distinct_sizes);
            timing.add(
                "time_slab.local_error_solves.patch_dofs_mode_size.count",
                profile.patch_dofs_mode_size);
            timing.add(
                "time_slab.local_error_solves.patch_dofs_mode_count.count",
                profile.patch_dofs_mode_count);
            timing.add(
                "time_slab.local_error_solves.flux_dofs_distinct_sizes.count",
                profile.flux_dofs_distinct_sizes);
            timing.add(
                "time_slab.local_error_solves.flux_dofs_mode_size.count",
                profile.flux_dofs_mode_size);
            timing.add(
                "time_slab.local_error_solves.flux_dofs_mode_count.count",
                profile.flux_dofs_mode_count);
            timing.add(
                "time_slab.local_error_solves.scalar_dofs_distinct_sizes.count",
                profile.scalar_dofs_distinct_sizes);
            timing.add(
                "time_slab.local_error_solves.scalar_dofs_mode_size.count",
                profile.scalar_dofs_mode_size);
            timing.add(
                "time_slab.local_error_solves.scalar_dofs_mode_count.count",
                profile.scalar_dofs_mode_count);
            timing.add(
                "time_slab.local_error_solves.reduced_scalar_basis_candidate_count.count",
                profile.reduced_scalar_basis_candidate_count);

            timing.add("local_error.n_patches", profile.patch_count);
            timing.add(
                "local_error.n_patch_cells",
                profile.patch_count * profile.patch_cells_mean);
            timing.add(
                "local_error.n_patch_dofs",
                profile.patch_count * profile.patch_dofs_mean);
            timing.add("local_error.max_patch_dofs", profile.patch_dofs_max);
            timing.add("local_error.mean_patch_dofs", profile.patch_dofs_mean);
            timing.add(
                "local_error.patch_dofs_distinct_sizes",
                profile.patch_dofs_distinct_sizes);
            timing.add(
                "local_error.patch_dofs_mode_size",
                profile.patch_dofs_mode_size);
            timing.add(
                "local_error.patch_dofs_mode_count",
                profile.patch_dofs_mode_count);
            timing.add(
                "local_error.flux_dofs_distinct_sizes",
                profile.flux_dofs_distinct_sizes);
            timing.add(
                "local_error.flux_dofs_mode_size",
                profile.flux_dofs_mode_size);
            timing.add(
                "local_error.flux_dofs_mode_count",
                profile.flux_dofs_mode_count);
            timing.add(
                "local_error.scalar_dofs_distinct_sizes",
                profile.scalar_dofs_distinct_sizes);
            timing.add(
                "local_error.scalar_dofs_mode_size",
                profile.scalar_dofs_mode_size);
            timing.add(
                "local_error.scalar_dofs_mode_count",
                profile.scalar_dofs_mode_count);
            timing.add(
                "local_error.reduced_scalar_basis_candidate_count",
                profile.reduced_scalar_basis_candidate_count);
            timing.add("local_error.max_patch_cells", profile.patch_cells_max);
            timing.add(
                "local_error.mean_patch_cells",
                profile.patch_cells_mean);
        }

        inline void record_local_error_v4_timing_aliases(
            const finite_element::detail::TimingRecorder& timing,
            const finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats)
        {
            const double patch_matrix_assembly =
                stats.assemble_A_seconds + stats.assemble_B_seconds +
                stats.assemble_C_seconds;
            const double patch_rhs_assembly =
                stats.assemble_f_seconds + stats.assemble_g_seconds;
            const double chunk_setup =
                stats.tile_cell_order_construction_seconds +
                stats.cell_requests_construction_seconds +
                stats.patch_cells_by_local_patch_construction_seconds +
                stats.chunk_table_construction_seconds +
                stats.tile_dense_block_allocation_seconds;
            const double dense_patch_solves =
                stats.workspace_patch_systems + stats.fallback_patch_systems;
            const double current_mean_dimension =
                stats.patch_solver_current_dimension_count > 0.0
                    ? stats.patch_solver_current_dimension_sum /
                          stats.patch_solver_current_dimension_count
                    : 0.0;
            const double reduced_mean_dimension =
                stats.patch_solver_reduced_dimension_count > 0.0
                    ? stats.patch_solver_reduced_dimension_sum /
                          stats.patch_solver_reduced_dimension_count
                    : 0.0;
            const double reduced_dimension_ratio =
                current_mean_dimension > 0.0 && reduced_mean_dimension > 0.0
                    ? reduced_mean_dimension / current_mean_dimension
                    : 0.0;

            timing.add(
                "local_error.cell_state_construction_wall",
                stats.chunk_cell_state_construction_seconds);
            timing.add(
                "local_error.streaming_tile_assembly_wall",
                stats.streaming_tile_assembly_seconds);
            timing.add(
                "local_error.patch_matrix_assembly_wall",
                patch_matrix_assembly);
            timing.add(
                "local_error.patch_rhs_assembly_wall",
                patch_rhs_assembly);
            timing.add(
                "local_error.patch_solve_wall",
                stats.solve_patch_systems_seconds);
            timing.add(
                "local_error.patch_factorization_wall",
                stats.patch_solve_factorization_seconds);
            timing.add(
                "local_error.patch_backsolve_wall",
                stats.patch_solve_apply_seconds);
            timing.add(
                "local_error.patch_solve_workspace_allocation_wall",
                stats.patch_solve_workspace_allocation_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_solve_workspace_allocation",
                stats.patch_solve_workspace_allocation_seconds);
            timing.add(
                "local_error.patch_function_update_wall",
                stats.coefficient_writeback_seconds);
            timing.add("local_error.flux_function_evaluation_wall", 0.0);
            timing.add("local_error.chunk_setup_wall", chunk_setup);
            timing.add("local_error.cache_lookup_wall", 0.0);
            timing.add("local_error.n_patch_solves", dense_patch_solves);
            timing.add("local_error.n_dense_patch_solves", dense_patch_solves);
            timing.add("local_error.n_sparse_patch_solves", 0.0);
            timing.add("patch_solver.mode", stats.patch_solver_mode);
            timing.add(
                "patch_solver.current_dense_count",
                stats.patch_solver_current_dense_count);
            timing.add(
                "patch_solver.reduced_scalar_dense_count",
                stats.patch_solver_reduced_scalar_dense_count);
            timing.add(
                "patch_solver.reduced_fallback_count",
                stats.patch_solver_reduced_fallback_count);
            timing.add(
                "patch_solver.reduced_residual_fail_count",
                stats.patch_solver_reduced_residual_fail_count);
            timing.add(
                "patch_solver.current_mean_dimension",
                current_mean_dimension);
            timing.add(
                "patch_solver.reduced_mean_dimension",
                reduced_mean_dimension);
            timing.add(
                "patch_solver.reduced_dimension_ratio",
                reduced_dimension_ratio);
            timing.add(
                "time_slab.local_error_solves.patch_solver.current_dense_count",
                stats.patch_solver_current_dense_count);
            timing.add(
                "time_slab.local_error_solves.patch_solver.reduced_scalar_dense_count",
                stats.patch_solver_reduced_scalar_dense_count);
            timing.add(
                "time_slab.local_error_solves.patch_solver.reduced_fallback_count",
                stats.patch_solver_reduced_fallback_count);
            timing.add(
                "time_slab.local_error_solves.patch_solver.reduced_residual_fail_count",
                stats.patch_solver_reduced_residual_fail_count);
            timing.add(
                "time_slab.local_error_solves.patch_solver.current_mean_dimension",
                current_mean_dimension);
            timing.add(
                "time_slab.local_error_solves.patch_solver.reduced_mean_dimension",
                reduced_mean_dimension);
            timing.add(
                "time_slab.local_error_solves.patch_solver.reduced_dimension_ratio",
                reduced_dimension_ratio);
            timing.add("local_error.hardware_threads", stats.hardware_threads);
            timing.add(
                "local_error.configured_max_threads",
                stats.configured_max_threads);
            timing.add(
                "local_error.candidate_threads",
                stats.candidate_threads);
            timing.add(
                "local_error.selected_threads",
                stats.selected_threads);
            timing.add("local_error.memory_limited", stats.memory_limited);
            timing.add(
                "local_error.nested_parallel_disabled",
                stats.nested_parallel_disabled);
            timing.add(
                "local_error.worker_context_mode",
                stats.worker_context_mode);
            timing.add(
                "local_error.worker_context_mode_effective",
                stats.worker_context_mode);
            timing.add(
                "local_error.worker_contexts_constructed",
                stats.thread_context_construction_count);
            timing.add(
                "local_error.worker_context_construction_wall",
                stats.thread_context_construction_seconds);
            timing.add(
                "local_error.x_geometry_cache_construction_wall",
                stats.geometry_cache_construction_seconds);
            timing.add(
                "local_error.context_memory_mb",
                stats.worker_context_memory_mb);
            timing.add(
                "local_error.thread_context_construction_count",
                stats.thread_context_construction_count);
            timing.add(
                "local_error.thread_context_construction_wall",
                stats.thread_context_construction_seconds);
            timing.add(
                "local_error.geometry_cache_construction_wall",
                stats.geometry_cache_construction_seconds);
            timing.add(
                "local_error.ancestor_cache_construction_wall",
                stats.ancestor_cache_construction_seconds);
            timing.add(
                "local_error.slab_geometry_cache_construction_wall",
                stats.slab_geometry_cache_construction_seconds);
            timing.add(
                "local_error.state_prepare.total_wall",
                stats.state_prepare_total_seconds);
            timing.add(
                "local_error.state_prepare.unique_count_wall",
                stats.state_prepare_unique_count_seconds);
            timing.add(
                "local_error.state_prepare.set_allocation_wall",
                stats.state_prepare_set_allocation_seconds);
            timing.add(
                "local_error.state_prepare.map_index_build_wall",
                stats.state_prepare_map_index_build_seconds);
            timing.add(
                "local_error.state_prepare.ordinal_map_build_wall",
                stats.state_prepare_ordinal_map_build_seconds);
            timing.add(
                "local_error.state_prepare.cell_vector_allocation_wall",
                stats.state_prepare_cell_vector_allocation_seconds);
            timing.add(
                "local_error.state_prepare.request_collection_wall",
                stats.state_prepare_request_collection_seconds);
            timing.add(
                "local_error.state_prepare.debug_duplicate_request_count",
                stats.state_prepare_debug_duplicate_request_count);
            timing.add(
                "local_error.state_prepare.memory_mb",
                stats.state_prepare_memory_mb);
            timing.add(
                "local_error.state_index_mode",
                stats.state_index_mode);
            timing.add(
                "local_error.state_index.flat_lookup_count",
                stats.state_index_flat_lookup_count);
            timing.add(
                "local_error.state_index.map_lookup_count",
                stats.state_index_map_lookup_count);
            timing.add(
                "local_error.state_index.fallback_hash_lookup_count",
                stats.state_index_fallback_hash_lookup_count);
            timing.add(
                "local_error.state_fill.total_wall",
                stats.state_fill_total_seconds);
            timing.add(
                "local_error.state_fill.active_ancestor_lookup_wall",
                stats.state_fill_active_ancestor_lookup_seconds);
            timing.add(
                "local_error.state_fill.geometry_lookup_wall",
                stats.state_fill_geometry_lookup_seconds);
            timing.add(
                "local_error.state_fill.affine_map_wall",
                stats.state_fill_affine_map_seconds);
            timing.add(
                "local_error.state_fill.time_basis_wall",
                stats.state_fill_time_basis_seconds);
            timing.add(
                "local_error.state_fill.spatial_rt_basis_wall",
                stats.state_fill_spatial_rt_basis_seconds);
            timing.add(
                "local_error.state_fill.scalar_basis_wall",
                stats.state_fill_scalar_basis_seconds);
            timing.add(
                "local_error.state_fill.partition_of_unity_wall",
                stats.state_fill_partition_of_unity_seconds);
            timing.add(
                "local_error.state_fill.lambda_gradient_wall",
                stats.state_fill_lambda_gradient_seconds);
            timing.add(
                "local_error.state_fill.u_gradient_wall",
                stats.state_fill_u_gradient_seconds);
            timing.add(
                "local_error.state_fill.load_evaluation_wall",
                stats.state_fill_load_evaluation_seconds);
            timing.add(
                "local_error.state_fill.diffusion_evaluation_wall",
                stats.state_fill_diffusion_evaluation_seconds);
            timing.add(
                "local_error.state_fill.diffusion_inverse_wall",
                stats.state_fill_diffusion_inverse_seconds);
            timing.add(
                "local_error.state_fill.local_A_assembly_wall",
                stats.state_fill_local_A_assembly_seconds);
            timing.add(
                "local_error.state_fill.local_B_assembly_wall",
                stats.state_fill_local_B_assembly_seconds);
            timing.add(
                "local_error.state_fill.qpoints_processed",
                stats.state_fill_qpoints_processed);
            timing.add(
                "local_error.state_fill.sampled_qpoints",
                stats.state_fill_sampled_qpoints);
            timing.add(
                "local_error.coefficient_fast_path.enabled",
                stats.coefficient_fast_path_enabled);
            timing.add(
                "local_error.coefficient_fast_path.identity_diffusion_cells",
                stats.coefficient_fast_path_identity_diffusion_cells);
            timing.add(
                "local_error.coefficient_fast_path.constant_diffusion_cells",
                stats.coefficient_fast_path_constant_diffusion_cells);
            timing.add(
                "local_error.coefficient_fast_path.zero_load_cells",
                stats.coefficient_fast_path_zero_load_cells);
            timing.add(
                "local_error.coefficient_fast_path.generic_cells",
                stats.coefficient_fast_path_generic_cells);
            timing.add(
                "local_error.operator_builder_mode",
                stats.operator_builder_mode);
            timing.add(
                "local_error.local_A_identity_reference_fast_path_count",
                stats.local_A_identity_reference_fast_path_count);
            timing.add(
                "local_error.local_A_constant_reference_fast_path_count",
                stats.local_A_constant_reference_fast_path_count);
            timing.add(
                "local_error.local_A_variable_generic_path_count",
                stats.local_A_variable_generic_path_count);
            timing.add(
                "local_error.local_B_reference_fast_path_count",
                stats.local_B_reference_fast_path_count);
            timing.add(
                "local_error.local_A_build_wall",
                stats.local_A_build_seconds);
            timing.add(
                "local_error.local_B_build_wall",
                stats.local_B_build_seconds);
            timing.add(
                "local_error.local_A_debug_max_abs_diff",
                stats.local_A_debug_max_abs_diff);
            timing.add(
                "local_error.local_A_debug_rel_frobenius_diff",
                stats.local_A_debug_rel_frobenius_diff);
            timing.add(
                "local_error.local_B_debug_max_abs_diff",
                stats.local_B_debug_max_abs_diff);
            timing.add(
                "local_error.local_B_debug_rel_frobenius_diff",
                stats.local_B_debug_rel_frobenius_diff);
            timing.add(
                "local_error.compact_state_shadow.enabled",
                stats.compact_state_shadow_enabled);
            timing.add(
                "local_error.compact_state_shadow.sample_count",
                stats.compact_state_shadow_sample_count);
            timing.add(
                "local_error.compact_state_shadow.reference_rt_basis_max_abs_diff",
                stats.compact_state_reference_rt_basis_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.reference_scalar_basis_max_abs_diff",
                stats.compact_state_reference_scalar_basis_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.reference_partition_value_max_abs_diff",
                stats.compact_state_reference_partition_value_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.reference_partition_gradient_max_abs_diff",
                stats.compact_state_reference_partition_gradient_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.local_A_max_abs_diff",
                stats.compact_state_local_A_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.local_B_max_abs_diff",
                stats.compact_state_local_B_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.rhs_f_max_abs_diff",
                stats.compact_state_rhs_f_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.rhs_g_max_abs_diff",
                stats.compact_state_rhs_g_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.grad_theta_max_abs_diff",
                stats.compact_state_grad_theta_max_abs_diff);
            timing.add(
                "local_error.compact_state_shadow.u_time_derivative_max_abs_diff",
                stats.compact_state_u_time_derivative_max_abs_diff);
            timing.add(
                "local_error.old_cell_data_bytes_per_cell",
                stats.old_cell_data_bytes_per_cell);
            timing.add(
                "local_error.operator_state_bytes_per_cell",
                stats.operator_state_bytes_per_cell);
            timing.add(
                "local_error.rhs_state_bytes_per_cell",
                stats.rhs_state_bytes_per_cell);
            timing.add(
                "local_error.flux_diagnostic_state_bytes_per_cell",
                stats.flux_diagnostic_state_bytes_per_cell);
            timing.add(
                "local_error.reference_table_memory_mb",
                stats.reference_table_memory_mb);
            timing.add(
                "local_error.monolithic_cell_data_constructed_count",
                stats.monolithic_cell_data_constructed_count);
            timing.add(
                "local_error.compact_operator_state_constructed_count",
                stats.compact_operator_state_constructed_count);
            timing.add(
                "local_error.compact_rhs_state_constructed_count",
                stats.compact_rhs_state_constructed_count);
            timing.add(
                "local_error.monolithic_debug_path_used_count",
                stats.monolithic_debug_path_used_count);
            timing.add(
                "local_error.estimated_old_full_cache_gib",
                stats.estimated_full_cache_bytes /
                    (1024.0 * 1024.0 * 1024.0));
            timing.add(
                "local_error.estimated_compact_full_cache_gib",
                stats.estimated_compact_full_cache_gib);
            timing.add(
                "local_error.estimated_lifetime_window_cache_mb",
                stats.estimated_lifetime_window_cache_mb);
            timing.add(
                "local_error.state_fill.time_per_state",
                stats.actual_state_constructions > 0.0
                    ? stats.state_fill_total_seconds /
                          stats.actual_state_constructions
                    : 0.0);
            timing.add(
                "local_error.state_fill.time_per_qpoint",
                stats.state_fill_qpoints_processed > 0.0
                    ? stats.state_fill_total_seconds /
                          stats.state_fill_qpoints_processed
                    : 0.0);
            timing.add(
                "local_error.state_fill.state_bytes_per_cell",
                stats.state_bytes_per_cell);
            timing.add(
                "local_error.state_fill.estimated_full_cache_bytes",
                stats.estimated_full_cache_bytes);
            timing.add(
                "local_error.state_fill.configured_cache_limit_bytes",
                stats.configured_cache_limit_bytes);
            timing.add(
                "local_error.cell_state_cache.mode",
                stats.cell_state_cache_mode);
            timing.add(
                "local_error.cell_state_cache.budget_mb",
                stats.cell_state_cache_budget_mb);
            timing.add(
                "local_error.cell_state_cache.entries",
                stats.cell_state_cache_entries);
            timing.add(
                "local_error.cell_state_cache.memory_mb",
                stats.cell_state_cache_memory_mb);
            timing.add(
                "local_error.cell_state_cache.hits",
                stats.cell_state_cache_hits);
            timing.add(
                "local_error.cell_state_cache.misses",
                stats.cell_state_cache_misses);
            timing.add(
                "local_error.cell_state_cache.evictions",
                stats.cell_state_cache_evictions);
            timing.add(
                "local_error.cell_state_cache.hit_rate",
                stats.cell_state_cache_hit_rate);
            timing.add(
                "local_error.cell_state_cache.cross_tile_rebuilds_avoided",
                stats.cell_state_cache_cross_tile_rebuilds_avoided);
            timing.add(
                "local_error.cell_state_cache.stale_state_detected_count",
                stats.cell_state_cache_stale_state_detected_count);
        }
    } // namespace detail

    // Current equilibrated-flux reconstruction is the 1+1D edge-patch
    // implementation. Keep the explicit 1plus1d name here so it cannot be
    // confused with the polynomial order p.
    template<class Backend, class XSpaceType, class SourceYSpaceType>
    class TimeSlabEquilibratedFluxReconstruction1plus1d
    {
    public:
        using VectorType       = typename Backend::Vector;
        using SparseMatrixType = typename Backend::SparseMatrix;
        using SolverType       = typename Backend::Solver;
        using DenseMatrixType  = typename Backend::DenseMatrix;
        using DenseSolverType  = typename Backend::DenseSolver;

        using XSpace           = XSpaceType;
        using SourceYSpace     = SourceYSpaceType;
        using GT               = typename SourceYSpace::GT;
        using FETraits         = typename SourceYSpace::FETraitsType;

        using SlabSpaceType    = TimeSlabSpace<GT, FETraits>;
        using LocalSlabSpaceType = typename SlabSpaceType::SlabType::SpaceType;
        using PatchSetType     = TimeSlabEdgePatchSet<GT, FETraits>;
        using PatchType        = typename PatchSetType::PatchType;

        using FluxSpaceType =
            error_fespace::PatchFluxSpace1D<
                PatchType,
                FETraits::p_space_v,
                FETraits::p_time_v>;
        using ScalarSpaceType =
            error_fespace::PatchScalarSpace<
                PatchType,
                FETraits::p_space_v,
                FETraits::p_time_v>;

        using FluxFunctionType =
            error_fespace::PatchFluxFunction1D<FluxSpaceType, VectorType>;
        using ScalarFunctionType =
            error_fespace::PatchScalarFunction<ScalarSpaceType, VectorType>;

        using SpaceTimePoint = typename SlabSpaceType::SpaceTimePoint;

        static_assert(
            finite_element::detail::is_currently_supported_1plus1d_space_time_v<GT>,
            "TimeSlabEquilibratedFluxReconstruction1plus1d is a 1+1D edge-patch estimator.");
        static_assert(
            time_slabs::is_time_slab_edge_patch_v<PatchType>,
            "TimeSlabEquilibratedFluxReconstruction1plus1d must use edge patches.");

        struct FluxEvaluation
        {
            double sigma = 0.0;
            double div_sigma = 0.0;
        };

        TimeSlabEquilibratedFluxReconstruction1plus1d(
            const SourceYSpace& source_y_space,
            const XSpace& x_space)
            : source_y_space_(&source_y_space),
              x_space_(&x_space)
        {
            finite_element::detail::require_1plus1d_time_slab_estimator_capability<GT>();
        }

        TimeSlabEquilibratedFluxReconstruction1plus1d(
            const SlabSpaceType& slab_space,
            const XSpace& x_space)
            : x_space_(&x_space),
              slab_space_ptr_(&slab_space)
        {
            finite_element::detail::require_1plus1d_time_slab_estimator_capability<GT>();
        }

        void initialize(double time_tol = 0.0)
        {
            reset_local_state_();

            if (source_y_space_)
            {
                // Rebuild the owned slab space on every call so repeated
                // initialize() invocations follow changes in the source Y-space.
                owned_slab_space_.emplace(*source_y_space_);
                TimeSlabBuilder<GT, FETraits>::initialize(*owned_slab_space_, time_tol);
                slab_space_ptr_ = &*owned_slab_space_;
            }
            else if (!slab_space_ptr_)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d::initialize: no source Y-space or slab-space available.");
            }

            patch_set_.emplace(TimeSlabEdgePatchBuilder<GT, FETraits>::build(slab_space_ref()));
            build_local_spaces_();
            initialized_ = true;
        }

        [[nodiscard]] const SourceYSpace& source_y_space() const
        {
            if (!source_y_space_)
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d::source_y_space: no source Y-space stored.");
            return *source_y_space_;
        }

        [[nodiscard]] const XSpace& x_space() const noexcept
        {
            return *x_space_;
        }

        [[nodiscard]] const SlabSpaceType& slab_space_ref() const
        {
            if (!slab_space_ptr_)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d::slab_space_ref: slab space not initialized.");
            }
            return *slab_space_ptr_;
        }

        [[nodiscard]] const PatchSetType& patch_set() const
        {
            ensure_initialized_();
            return *patch_set_;
        }

        [[nodiscard]] int n_patches() const
        {
            ensure_initialized_();
            return patch_set_->n_patches();
        }

        [[nodiscard]] const FluxSpaceType& flux_space(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return flux_spaces_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] const ScalarSpaceType& scalar_space(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return scalar_spaces_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] const FluxFunctionType& flux_function(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return flux_functions_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] const ScalarFunctionType& scalar_function(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return scalar_functions_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] bool last_solve_all_patches_used_openmp() const noexcept
        {
            return last_solve_all_patches_used_openmp_;
        }

        void set_local_error_context_storage(std::string mode)
        {
            for (char& ch : mode)
                ch = static_cast<char>(
                    std::tolower(static_cast<unsigned char>(ch)));
            local_error_context_storage_ =
                (mode == "per_chunk_debug" ||
                 mode == "persistent_per_thread_debug" ||
                 mode == "shared_immutable" ||
                 mode == "shared_immutable_shadow")
                    ? mode
                    : "shared_immutable";
        }

        [[nodiscard]] const std::string& local_error_context_storage()
            const noexcept
        {
            return local_error_context_storage_;
        }

        void set_shared_context_validation(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';

                    return static_cast<char>(std::tolower(c));
                });
            shared_context_validation_ =
                mode == "sample" || mode == "full_debug" ? mode : "off";
        }

        [[nodiscard]] const std::string&
        shared_context_validation() const noexcept
        {
            return shared_context_validation_;
        }

        void set_local_error_coefficient_fast_path(bool enabled) noexcept
        {
            local_error_coefficient_fast_path_ = enabled;
        }

        [[nodiscard]] bool
        local_error_coefficient_fast_path() const noexcept
        {
            return local_error_coefficient_fast_path_;
        }

        void set_local_error_compact_state_shadow(bool enabled) noexcept
        {
            local_error_compact_state_shadow_ = enabled;
        }

        [[nodiscard]] bool
        local_error_compact_state_shadow() const noexcept
        {
            return local_error_compact_state_shadow_;
        }

        void set_local_error_cell_state_representation(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';
                    return static_cast<char>(std::tolower(c));
                });
            if (mode != "compact_split" && mode != "monolithic_debug")
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d: invalid local-error cell-state representation.");
            local_error_cell_state_representation_ = std::move(mode);
        }

        [[nodiscard]] const std::string&
        local_error_cell_state_representation() const noexcept
        {
            return local_error_cell_state_representation_;
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch(
            int patch_id,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15)
        {
            ensure_initialized_();

            finite_element::detail::CellGeometryCache<XSpace> x_geometry_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace> ancestor_cache(
                *x_space_);
            auto slab_geometry_caches = make_slab_geometry_caches_();
            last_solve_all_patches_used_openmp_ = false;

            solve_patch_impl_<QSpace, QTime>(
                patch_id,
                lambda_tilde,
                ell,
                u_delta,
                M,
                x_geometry_cache,
                slab_geometry_caches,
                ancestor_cache,
                solver,
                options,
                zero_tol);
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch_sparse_reference(
            int patch_id,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15)
        {
            ensure_initialized_();

            finite_element::detail::CellGeometryCache<XSpace> x_geometry_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace> ancestor_cache(
                *x_space_);
            auto slab_geometry_caches = make_slab_geometry_caches_();
            last_solve_all_patches_used_openmp_ = false;
            solve_patch_sparse_reference_impl_<QSpace, QTime>(
                patch_id,
                lambda_tilde,
                ell,
                u_delta,
                M,
                x_geometry_cache,
                slab_geometry_caches,
                ancestor_cache,
                solver,
                options,
                zero_tol);
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_all_patches(
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            ensure_initialized_();

            const int patch_count = n_patches();
            last_solve_all_patches_used_openmp_ = false;
            finite_element::assembly::error_system::LocalErrorProblemTimingStats
                timing_stats;
            auto shared_context_storage =
                build_shared_context_if_requested_(timing_stats);
            const bool use_shared_context =
                shared_context_storage.has_value() &&
                local_error_context_storage_ == "shared_immutable";
            detail::record_local_error_patch_profile(
                timing,
                detail::make_local_error_patch_profile(
                    *patch_set_,
                    flux_spaces_,
                    scalar_spaces_,
                    slab_space_ref().n_slabs()),
                false);

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
            if constexpr (std::default_initializable<SolverType>)
            {
                const int n_threads =
                    finite_element::assembly::detail::
                        recommended_openmp_threads_for_patch_solves(
                            patch_count);
                if (n_threads > 1)
                {
                    last_solve_all_patches_used_openmp_ = true;
                    std::exception_ptr error;

#pragma omp parallel num_threads(n_threads)
                    {
                        finite_element::assembly::error_system::
                            LocalErrorProblemTimingStats thread_timing_stats;
                        try
                        {
                            SolverType thread_solver;
                            if (use_shared_context)
                            {
                                const auto context =
                                    finite_element::assembly::error_system::
                                        LocalErrorProblemContext<
                                            XSpace,
                                            SlabSpaceType>{
                                            x_space_,
                                            slab_space_ptr_,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            &*shared_context_storage};

                                // The local patch problems are independent
                                // after initialize(): each thread uses the
                                // immutable context and writes a disjoint
                                // patch id.
#pragma omp for schedule(static)
                                for (int patch_id = 0;
                                     patch_id < patch_count;
                                     ++patch_id)
                                {
                                    solve_patch_with_context_<QSpace, QTime>(
                                        patch_id,
                                        context,
                                        lambda_tilde,
                                        ell,
                                        u_delta,
                                        M,
                                        thread_solver,
                                        options,
                                        zero_tol,
                                        &thread_timing_stats);
                                }
                            }
                            else
                            {
                                finite_element::detail::CellGeometryCache<XSpace>
                                    x_geometry_cache(*x_space_);
                                finite_element::assembly::detail::
                                    SourceActiveAncestorCache<XSpace>
                                        ancestor_cache(*x_space_);
                                auto slab_geometry_caches =
                                    make_slab_geometry_caches_();

                                // The local patch problems are independent
                                // after initialize(): each thread works on
                                // its own caches/solver and writes a disjoint
                                // patch id.
#pragma omp for schedule(static)
                                for (int patch_id = 0;
                                     patch_id < patch_count;
                                     ++patch_id)
                                {
                                    solve_patch_impl_<QSpace, QTime>(
                                        patch_id,
                                        lambda_tilde,
                                        ell,
                                        u_delta,
                                        M,
                                        x_geometry_cache,
                                        slab_geometry_caches,
                                        ancestor_cache,
                                        thread_solver,
                                        options,
                                        zero_tol,
                                        &thread_timing_stats);
                                }
                            }
                        }
                        catch (...)
                        {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                            {
                                if (!error)
                                    error = std::current_exception();
                            }
                        }

#pragma omp critical(adap_parabolic_fem_local_error_timing)
                        {
                            timing_stats.add(thread_timing_stats);
                        }
                    }

                    finite_element::assembly::detail::rethrow_parallel_exception(error);
                    record_local_error_timing_(timing, timing_stats);
                    return;
                }
            }
#endif

            if (use_shared_context)
            {
                const auto context =
                    finite_element::assembly::error_system::
                        LocalErrorProblemContext<XSpace, SlabSpaceType>{
                            x_space_,
                            slab_space_ptr_,
                            nullptr,
                            nullptr,
                            nullptr,
                            nullptr,
                            &*shared_context_storage};

                for (int patch_id = 0; patch_id < patch_count; ++patch_id)
                {
                    solve_patch_with_context_<QSpace, QTime>(
                        patch_id,
                        context,
                        lambda_tilde,
                        ell,
                        u_delta,
                        M,
                        solver,
                        options,
                        zero_tol,
                        &timing_stats);
                }

                record_local_error_timing_(timing, timing_stats);
                return;
            }

            finite_element::detail::CellGeometryCache<XSpace> x_geometry_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace> ancestor_cache(
                *x_space_);
            auto slab_geometry_caches = make_slab_geometry_caches_();

            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                solve_patch_impl_<QSpace, QTime>(
                    patch_id,
                    lambda_tilde,
                    ell,
                    u_delta,
                    M,
                    x_geometry_cache,
                    slab_geometry_caches,
                    ancestor_cache,
                    solver,
                    options,
                    zero_tol,
                    &timing_stats);
            }

            record_local_error_timing_(timing, timing_stats);
        }

        [[nodiscard]] double sigma_on_slab_cell(
            int slab_id,
            int slab_cell_id,
            const SpaceTimePoint& p) const
        {
            return sigma_and_div_sigma_on_slab_cell(slab_id, slab_cell_id, p).sigma;
        }

        [[nodiscard]] double div_sigma_on_slab_cell(
            int slab_id,
            int slab_cell_id,
            const SpaceTimePoint& p) const
        {
            return sigma_and_div_sigma_on_slab_cell(slab_id, slab_cell_id, p).div_sigma;
        }

        [[nodiscard]] FluxEvaluation sigma_and_div_sigma_on_slab_cell(
            int slab_id,
            int slab_cell_id,
            const SpaceTimePoint& p) const
        {
            ensure_initialized_();

            FluxEvaluation evaluation;
            const int membership_count =
                patch_set_->cell_patch_count(slab_id, slab_cell_id);

            for (int i = 0; i < membership_count; ++i)
            {
                const auto membership =
                    patch_set_->cell_patch_membership(slab_id, slab_cell_id, i);
                const auto patch_evaluation =
                    flux_function(membership.patch_id).evaluate_on_cell(
                        membership.patch_cell_index,
                        p);

                evaluation.sigma += patch_evaluation.value;
                evaluation.div_sigma += patch_evaluation.divergence;
            }

            return evaluation;
        }

        [[nodiscard]] double sigma(const SpaceTimePoint& p) const
        {
            const auto loc = slab_space_ref().find_active_cell(p);
            if (!loc.is_valid())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d::sigma: point not found in any slab cell.");
            }

            return sigma_on_slab_cell(loc.slab_id, loc.cell_id, p);
        }

        [[nodiscard]] double div_sigma(const SpaceTimePoint& p) const
        {
            const auto loc = slab_space_ref().find_active_cell(p);
            if (!loc.is_valid())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d::div_sigma: point not found in any slab cell.");
            }

            return div_sigma_on_slab_cell(loc.slab_id, loc.cell_id, p);
        }

    private:
        struct ActiveSlabCellRef
        {
            int slab_id = -1;
            int slab_cell_id = -1;
        };

        [[nodiscard]] std::vector<ActiveSlabCellRef>
        collect_active_slab_cell_refs_() const
        {
            std::vector<ActiveSlabCellRef> active_slab_cells;
            for (int slab_id = 0;
                 slab_id < slab_space_ref().n_slabs();
                 ++slab_id)
            {
                const auto& slab = slab_space_ref().slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                    active_slab_cells.push_back(
                        ActiveSlabCellRef{slab_id, slab_cell_id});
            }
            return active_slab_cells;
        }

        [[nodiscard]] std::optional<
            finite_element::assembly::error_system::
                SharedLocalErrorContext1D<XSpace, SlabSpaceType>>
        build_shared_context_if_requested_(
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& timing_stats) const
        {
            if (local_error_context_storage_ != "shared_immutable" &&
                local_error_context_storage_ != "shared_immutable_shadow")
            {
                return std::nullopt;
            }

            const auto collect_begin = std::chrono::steady_clock::now();
            const auto active_slab_cells = collect_active_slab_cell_refs_();
            timing_stats.active_slab_cell_collection_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - collect_begin)
                    .count();

            using SharedContext =
                finite_element::assembly::error_system::
                    SharedLocalErrorContext1D<XSpace, SlabSpaceType>;

            timing_stats.shared_context_shadow_enabled = 1.0;
            const auto shared_context_begin =
                std::chrono::steady_clock::now();
            auto shared_context =
                SharedContext::build(
                    *x_space_,
                    slab_space_ref(),
                    active_slab_cells);
            timing_stats.shared_context_build_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    shared_context_begin)
                    .count();
            timing_stats.shared_context_memory_mb =
                static_cast<double>(
                    shared_context.estimated_memory_bytes()) /
                (1024.0 * 1024.0);
            timing_stats.shared_context_x_geometry_count =
                static_cast<double>(shared_context.x_geometry_count());
            timing_stats.shared_context_slab_geometry_count =
                static_cast<double>(shared_context.slab_geometry_count());
            timing_stats.shared_context_ancestor_count =
                static_cast<double>(shared_context.ancestor_count());
            timing_stats.shared_context_active_slab_cells =
                static_cast<double>(shared_context.active_slab_cell_count());

            const bool validate_shared_context =
                local_error_context_storage_ == "shared_immutable_shadow" ||
                shared_context_validation_ == "sample" ||
                shared_context_validation_ == "full_debug";
            timing_stats.shared_context_validation_enabled =
                validate_shared_context ? 1.0 : 0.0;
            if (validate_shared_context)
            {
                const auto validation_begin =
                    std::chrono::steady_clock::now();
                finite_element::detail::CellGeometryCache<XSpace>
                    comparison_x_geometry_cache(*x_space_);
                finite_element::assembly::detail::
                    SourceActiveAncestorCache<XSpace>
                        comparison_ancestor_cache(*x_space_);
                auto comparison_slab_geometry_caches =
                    make_slab_geometry_caches_();
                timing_stats
                    .shared_context_comparison_mutable_caches_constructed +=
                    1.0;
                const auto comparison =
                    shared_context.compare_sample(
                        comparison_x_geometry_cache,
                        comparison_ancestor_cache,
                        comparison_slab_geometry_caches);
                timing_stats.shared_context_sample_geometry_max_abs_diff =
                    comparison.sample_geometry_max_abs_diff;
                timing_stats.shared_context_sample_slab_geometry_max_abs_diff =
                    comparison.sample_slab_geometry_max_abs_diff;
                timing_stats.shared_context_sample_ancestor_mismatch_count =
                    comparison.sample_ancestor_mismatch_count;
                timing_stats.shared_context_sample_count =
                    comparison.sample_count;
                timing_stats.shared_context_validation_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - validation_begin)
                        .count();
            }

            return shared_context;
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch_with_context_(
            int patch_id,
            const finite_element::assembly::error_system::
                LocalErrorProblemContext<XSpace, SlabSpaceType>& context,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            double zero_tol,
            finite_element::assembly::error_system::LocalErrorProblemTimingStats*
                timing_stats = nullptr)
        {
            ensure_patch_index_(patch_id);

            auto blocks =
                finite_element::assembly::error_system::assemble_local_error_problem<
                    QSpace,
                    QTime,
                    Backend>(
                        flux_space(patch_id),
                        scalar_space(patch_id),
                        context,
                        lambda_tilde,
                        u_delta,
                        ell,
                        M,
                        zero_tol,
                        timing_stats);

            la::saddle::SaddlePointSolution<Backend> split;
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    solve_timer(
                        timing_stats != nullptr
                            ? &timing_stats->solve_patch_systems_seconds
                            : nullptr);
                split = la::saddle::solve_and_split<Backend>(
                    blocks,
                    solver,
                    options);
            }

            flux_functions_[static_cast<std::size_t>(patch_id)]
                .update_coefficients(split.lambda);
            scalar_functions_[static_cast<std::size_t>(patch_id)]
                .update_coefficients(split.u);
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch_impl_(
            int patch_id,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            finite_element::detail::CellGeometryCache<XSpace>& x_geometry_cache,
            std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>&
                slab_geometry_caches,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            double zero_tol,
            finite_element::assembly::error_system::LocalErrorProblemTimingStats*
                timing_stats = nullptr)
        {
            const auto context =
                finite_element::assembly::error_system::LocalErrorProblemContext<
                    XSpace,
                    SlabSpaceType>{
                        x_space_,
                        slab_space_ptr_,
                        &x_geometry_cache,
                        &slab_geometry_caches,
                        &ancestor_cache};

            solve_patch_with_context_<QSpace, QTime>(
                patch_id,
                context,
                lambda_tilde,
                ell,
                u_delta,
                M,
                solver,
                options,
                zero_tol,
                timing_stats);
        }

        static void record_local_error_timing_(
            const finite_element::detail::TimingRecorder& timing,
            const finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats)
        {
            detail::record_local_error_v4_timing_aliases(timing, stats);

            timing.add(
                "time_slab.local_error_solves.patch_matrix_assembly",
                stats.assemble_A_seconds + stats.assemble_B_seconds +
                    stats.assemble_C_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_rhs_assembly",
                stats.assemble_f_seconds + stats.assemble_g_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_solve",
                stats.solve_patch_systems_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_A",
                stats.assemble_A_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_B",
                stats.assemble_B_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_C",
                stats.assemble_C_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_f",
                stats.assemble_f_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_g",
                stats.assemble_g_seconds);
            timing.add(
                "time_slab.local_error_solves.quadrature_table_construction",
                stats.quadrature_table_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.compose_with_constraints",
                stats.compose_with_constraints_seconds);
            timing.add(
                "time_slab.local_error_solves.reduced_basis_transform",
                stats.reduced_basis_transform_seconds);
            timing.add(
                "time_slab.local_error_solves.solve_patch_systems",
                stats.solve_patch_systems_seconds);
            timing.add(
                "time_slab.local_error_solves.coefficient_writeback",
                stats.coefficient_writeback_seconds);
            timing.add(
                "time_slab.local_error_solves.solve_patch_systems.factorization_time",
                stats.patch_solve_factorization_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_factorization",
                stats.patch_solve_factorization_seconds);
            timing.add(
                "time_slab.local_error_solves.solve_patch_systems.solve_apply_time",
                stats.patch_solve_apply_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_solve_apply",
                stats.patch_solve_apply_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_solve_groups.count",
                stats.patch_solve_groups);
            timing.add(
                "time_slab.local_error_solves.patch_solve_largest_group_size.count",
                stats.patch_solve_largest_group_size);
            timing.add(
                "time_slab.local_error_solves.batched_patch_systems.count",
                stats.batched_patch_systems);
            timing.add(
                "time_slab.local_error_solves.workspace_patch_systems.count",
                stats.workspace_patch_systems);
            timing.add(
                "time_slab.local_error_solves.legacy_patch_systems.count",
                stats.fallback_patch_systems);
            timing.add(
                "time_slab.local_error_solves.memory.dense_solver_workspace_bytes",
                stats.dense_solver_workspace_bytes);
            timing.add(
                "shared_context.shadow_enabled",
                stats.shared_context_shadow_enabled);
            timing.add(
                "shared_context.build_wall",
                stats.shared_context_build_seconds);
            timing.add(
                "shared_context.memory_mb",
                stats.shared_context_memory_mb);
            timing.add(
                "shared_context.x_geometry_count",
                stats.shared_context_x_geometry_count);
            timing.add(
                "shared_context.slab_geometry_count",
                stats.shared_context_slab_geometry_count);
            timing.add(
                "shared_context.ancestor_count",
                stats.shared_context_ancestor_count);
            timing.add(
                "shared_context.active_slab_cells",
                stats.shared_context_active_slab_cells);
            timing.add(
                "shared_context.sample_geometry_max_abs_diff",
                stats.shared_context_sample_geometry_max_abs_diff);
            timing.add(
                "shared_context.sample_slab_geometry_max_abs_diff",
                stats.shared_context_sample_slab_geometry_max_abs_diff);
            timing.add(
                "shared_context.sample_ancestor_mismatch_count",
                stats.shared_context_sample_ancestor_mismatch_count);
            timing.add(
                "shared_context.sample_count",
                stats.shared_context_sample_count);
            timing.add(
                "shared_context.validation_enabled",
                stats.shared_context_validation_enabled);
            timing.add(
                "shared_context.validation_wall",
                stats.shared_context_validation_seconds);
            timing.add(
                "shared_context.comparison_mutable_caches_constructed",
                stats.shared_context_comparison_mutable_caches_constructed);
            timing.add(
                "local_error.mutable_rhs_context_constructed_count",
                stats.mutable_rhs_context_constructed_count);
            timing.add(
                "local_error.mutable_rhs_context_construction_wall",
                stats.mutable_rhs_context_construction_seconds);
            timing.add(
                "local_error.shared_rhs_context_used_count",
                stats.shared_rhs_context_used_count);
            timing.add(
                "time_slab.local_error_solves.active_slab_cell_collection",
                stats.active_slab_cell_collection_seconds);
            timing.add(
                "time_slab.local_error_solves.cell_coloring",
                stats.cell_coloring_seconds);
            timing.add(
                "time_slab.local_error_solves.tile_cell_order_construction",
                stats.tile_cell_order_construction_seconds);
            timing.add(
                "local_error.tile_cell_order_construction_wall",
                stats.tile_cell_order_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.cell_requests_construction",
                stats.cell_requests_construction_seconds);
            timing.add(
                "local_error.cell_requests_construction_wall",
                stats.cell_requests_construction_seconds);
            timing.add(
                "local_error.membership_plan_construction_wall",
                stats.membership_plan_construction_seconds);
            timing.add(
                "local_error.patch_membership_lookup_count",
                stats.patch_membership_lookup_count);
            timing.add(
                "local_error.patch_membership_lookup_count_after_plan",
                stats.patch_membership_lookup_count_after_plan);
            timing.add(
                "local_error.tile_plan.build_wall",
                stats.tile_plan_build_seconds);
            timing.add(
                "local_error.tile_plan.memory_mb",
                stats.tile_plan_memory_mb);
            timing.add(
                "local_error.tile_plan.cells",
                stats.tile_plan_cells);
            timing.add(
                "local_error.tile_plan.memberships",
                stats.tile_plan_memberships);
            timing.add(
                "local_error.tile_plan.tile_count",
                stats.tile_plan_tile_count);
            timing.add(
                "local_error.tile_plan.chunk_count",
                stats.tile_plan_chunk_count);
            timing.add(
                "local_error.tile_plan.membership_scans_avoided",
                stats.tile_plan_membership_scans_avoided);
            timing.add(
                "time_slab.local_error_solves.patch_cells_by_local_patch_construction",
                stats.patch_cells_by_local_patch_construction_seconds);
            timing.add(
                "local_error.unused_patch_cells_by_local_patch_removed",
                stats.unused_patch_cells_by_local_patch_removed);
            timing.add(
                "time_slab.local_error_solves.chunk_table_construction",
                stats.chunk_table_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.chunk_cell_state_construction",
                stats.chunk_cell_state_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.streaming_chunk_state_construction",
                stats.streaming_chunk_state_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.streaming_tile_assembly",
                stats.streaming_tile_assembly_seconds);
            timing.add(
                "time_slab.local_error_solves.tile_dense_block_allocation",
                stats.tile_dense_block_allocation_seconds);
        }

        void build_local_spaces_()
        {
            if (!patch_set_.has_value())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d::build_local_spaces_: patch set not initialized.");
            }

            flux_spaces_.clear();
            scalar_spaces_.clear();
            flux_functions_.clear();
            scalar_functions_.clear();

            const int patch_count = patch_set_->n_patches();

            flux_spaces_.reserve(static_cast<std::size_t>(patch_count));
            scalar_spaces_.reserve(static_cast<std::size_t>(patch_count));
            for (const auto& patch : patch_set_->patches())
            {
                flux_spaces_.emplace_back(patch);
                scalar_spaces_.emplace_back(patch);
            }

            flux_functions_.reserve(static_cast<std::size_t>(patch_count));
            scalar_functions_.reserve(static_cast<std::size_t>(patch_count));
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                flux_functions_.emplace_back(
                    flux_spaces_[static_cast<std::size_t>(patch_id)]);
                scalar_functions_.emplace_back(
                    scalar_spaces_[static_cast<std::size_t>(patch_id)]);
            }
        }

        void reset_local_state_()
        {
            initialized_ = false;
            patch_set_.reset();
            flux_spaces_.clear();
            scalar_spaces_.clear();
            flux_functions_.clear();
            scalar_functions_.clear();
            last_solve_all_patches_used_openmp_ = false;
        }

        [[nodiscard]] std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>
        make_slab_geometry_caches_() const
        {
            std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>
                caches;
            caches.reserve(static_cast<std::size_t>(slab_space_ref().n_slabs()));

            for (int slab_id = 0; slab_id < slab_space_ref().n_slabs(); ++slab_id)
                caches.emplace_back(slab_space_ref().slab(slab_id).fespace_ref());

            return caches;
        }

        void ensure_initialized_() const
        {
            if (!initialized_)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d: call initialize() first.");
            }
        }

        void ensure_patch_index_(int patch_id) const
        {
            ensure_initialized_();

            if (patch_id < 0 || patch_id >= n_patches())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction1plus1d: patch id out of range.");
            }
        }

        const SourceYSpace* source_y_space_ = nullptr;
        const XSpace* x_space_              = nullptr;
        const SlabSpaceType* slab_space_ptr_ = nullptr;
        std::optional<SlabSpaceType> owned_slab_space_{};
        std::optional<PatchSetType> patch_set_{};

        std::vector<FluxSpaceType> flux_spaces_{};
        std::vector<ScalarSpaceType> scalar_spaces_{};
        std::vector<FluxFunctionType> flux_functions_{};
        std::vector<ScalarFunctionType> scalar_functions_{};

        bool initialized_ = false;
        bool last_solve_all_patches_used_openmp_ = false;
        std::string local_error_context_storage_ = "shared_immutable";
        std::string shared_context_validation_ = "off";
        bool local_error_coefficient_fast_path_ = true;
        bool local_error_compact_state_shadow_ = false;
        std::string local_error_cell_state_representation_ = "compact_split";
    };

    template<class Backend, class XSpaceType, class SourceYSpaceType>
    class TimeSlabEquilibratedFluxReconstruction2plus1d
    {
    public:
        using VectorType       = typename Backend::Vector;
        using SparseMatrixType = typename Backend::SparseMatrix;
        using SolverType       = typename Backend::Solver;
        using DenseMatrixType  = typename Backend::DenseMatrix;
        using DenseSolverType  = typename Backend::DenseSolver;

        using XSpace           = XSpaceType;
        using SourceYSpace     = SourceYSpaceType;
        using GT               = typename SourceYSpace::GT;
        using FETraits         = typename SourceYSpace::FETraitsType;

        using SlabSpaceType      = TimeSlabSpace<GT, FETraits>;
        using LocalSlabSpaceType = typename SlabSpaceType::SlabType::SpaceType;
        using PatchSetType       = TimeSlabVertexPatchSet<GT, FETraits>;
        using PatchType          = typename PatchSetType::PatchType;

        using FluxSpaceType =
            error_fespace::PatchRTFluxSpaceTime2D<
                PatchType,
                FETraits::p_space_v,
                FETraits::p_time_v>;
        using ScalarSpaceType =
            error_fespace::PatchScalarSpaceTime2D<
                PatchType,
                FETraits::p_space_v,
                FETraits::p_time_v>;

        using FluxFunctionType =
            error_fespace::PatchRTFluxFunctionTime2D<
                FluxSpaceType,
                VectorType>;
        using ScalarFunctionType =
            error_fespace::PatchScalarFunctionTime2D<
                ScalarSpaceType,
                VectorType>;

        using SpaceTimePoint = typename SlabSpaceType::SpaceTimePoint;
        using VectorValue    = typename FluxSpaceType::VectorValue;
        using FluxDiagnostics = CellwiseEquilibratedFluxError<int>;

        static_assert(GT::dim_space_v == 2 && GT::dim_time_v == 1,
                      "TimeSlabEquilibratedFluxReconstruction2plus1d requires a 2+1D space-time discretization.");
        static_assert(
            time_slabs::is_time_slab_vertex_patch_v<PatchType>,
            "TimeSlabEquilibratedFluxReconstruction2plus1d must use vertex patches.");

        struct FluxEvaluation
        {
            VectorValue sigma{0.0, 0.0};
            double div_sigma = 0.0;
        };

        struct FluxDiagnosticsRuntimeStats
        {
            double seconds = 0.0;
            double qpoints = 0.0;
            double reused_cell_state = 0.0;
            double extra_cell_state_rebuilds = 0.0;
        };

    private:
        struct LocalOperatorCache
        {
            struct OperatorData
            {
                int n_lambda = 0;
                int n_u = 0;
                int n_constraints = 0;
                int explicit_system_size = 0;
                DenseMatrixType A{};
                DenseMatrixType B{};
                DenseMatrixType C{};
                DenseMatrixType explicit_constraint_matrix{};
                std::shared_ptr<DenseSolverType>
                    explicit_constraint_solver{};

                [[nodiscard]] bool has_explicit_constraint_factorization()
                    const noexcept
                {
                    return explicit_constraint_solver != nullptr &&
                           explicit_system_size > 0 &&
                           explicit_constraint_matrix.rows() ==
                               explicit_system_size &&
                           explicit_constraint_matrix.cols() ==
                               explicit_system_size;
                }
            };

            std::unordered_map<std::size_t, OperatorData> operators{};

            void clear()
            {
                operators.clear();
            }

            [[nodiscard]] std::size_t size() const noexcept
            {
                return operators.size();
            }
        };

        struct LocalErrorWorkerContext2D
        {
            std::unique_ptr<
                finite_element::detail::CellGeometryCache<XSpace>>
                x_geometry_cache;
            std::unique_ptr<
                finite_element::assembly::detail::
                    SourceActiveAncestorCache<XSpace>>
                ancestor_cache;
            std::vector<
                finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>
                slab_geometry_caches;
            finite_element::assembly::error_system::
                LocalErrorProblemContext<XSpace, SlabSpaceType>
                problem_context;

            LocalErrorWorkerContext2D(
                const XSpace& x_space,
                const SlabSpaceType& slab_space,
                std::unique_ptr<
                    finite_element::detail::CellGeometryCache<XSpace>>&&
                    x_cache,
                std::unique_ptr<
                    finite_element::assembly::detail::
                        SourceActiveAncestorCache<XSpace>>&&
                    ancestor,
                std::vector<
                    finite_element::detail::CellGeometryCache<
                        LocalSlabSpaceType>>&& slab_caches)
                : x_geometry_cache(std::move(x_cache)),
                  ancestor_cache(std::move(ancestor)),
                  slab_geometry_caches(std::move(slab_caches)),
                  problem_context{
                      &x_space,
                      &slab_space,
                      x_geometry_cache.get(),
                      &slab_geometry_caches,
                      ancestor_cache.get()}
            {}
        };

    public:
        TimeSlabEquilibratedFluxReconstruction2plus1d(
            const SourceYSpace& source_y_space,
            const XSpace& x_space)
            : source_y_space_(&source_y_space),
              x_space_(&x_space)
        {}

        TimeSlabEquilibratedFluxReconstruction2plus1d(
            const SlabSpaceType& slab_space,
            const XSpace& x_space)
            : x_space_(&x_space),
              slab_space_ptr_(&slab_space)
        {}

        void set_local_error_patch_tile_size(int tile_size) noexcept
        {
            local_error_patch_tile_size_ = tile_size > 0 ? tile_size : 0;
        }

        [[nodiscard]] int local_error_patch_tile_size() const noexcept
        {
            return local_error_patch_tile_size_;
        }

        void set_local_error_cell_chunk_size(int chunk_size) noexcept
        {
            local_error_cell_chunk_size_ = chunk_size > 0 ? chunk_size : 0;
        }

        [[nodiscard]] int local_error_cell_chunk_size() const noexcept
        {
            return local_error_cell_chunk_size_;
        }

        void set_local_error_max_threads(int max_threads) noexcept
        {
            local_error_max_threads_ = max_threads > 0 ? max_threads : 0;
        }

        [[nodiscard]] int local_error_max_threads() const noexcept
        {
            return local_error_max_threads_;
        }

        void set_local_error_memory_budget_mb(double budget_mb) noexcept
        {
            local_error_memory_budget_mb_ = budget_mb > 0.0 ? budget_mb : 0.0;
        }

        [[nodiscard]] double local_error_memory_budget_mb() const noexcept
        {
            return local_error_memory_budget_mb_;
        }

        void set_local_error_worker_context_mode(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';

                    return static_cast<char>(std::tolower(c));
                });
            local_error_worker_context_mode_ =
                mode == "per_chunk_debug" ||
                        mode == "persistent_all_p_debug"
                    ? mode
                    : "persistent";
        }

        [[nodiscard]] const std::string&
        local_error_worker_context_mode() const noexcept
        {
            return local_error_worker_context_mode_;
        }

        void set_local_error_context_storage(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';

                    return static_cast<char>(std::tolower(c));
                });
            local_error_context_storage_ =
                mode == "per_chunk_debug" ||
                        mode == "persistent_per_thread_debug" ||
                        mode == "shared_immutable" ||
                        mode == "shared_immutable_shadow"
                    ? mode
                    : "shared_immutable";
        }

        [[nodiscard]] const std::string&
        local_error_context_storage() const noexcept
        {
            return local_error_context_storage_;
        }

        void set_local_error_state_index_mode(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';

                    return static_cast<char>(std::tolower(c));
                });
            local_error_state_index_mode_ =
                mode == "map_debug" ? "map_debug" : "flat";
        }

        [[nodiscard]] const std::string&
        local_error_state_index_mode() const noexcept
        {
            return local_error_state_index_mode_;
        }

        void set_local_error_cell_state_cache_mode(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';

                    return static_cast<char>(std::tolower(c));
                });
            local_error_cell_state_cache_mode_ =
                mode == "off" || mode == "tile" ||
                        mode == "bounded_lru" ||
                        mode == "lifetime_window" ||
                        mode == "full_if_fits"
                    ? mode
                    : "lifetime_window";
        }

        [[nodiscard]] const std::string&
        local_error_cell_state_cache_mode() const noexcept
        {
            return local_error_cell_state_cache_mode_;
        }

        void set_local_error_cell_state_cache_budget_mb(
            double budget_mb) noexcept
        {
            local_error_cell_state_cache_budget_mb_ =
                budget_mb > 0.0 ? budget_mb : 0.0;
        }

        [[nodiscard]] double
        local_error_cell_state_cache_budget_mb() const noexcept
        {
            return local_error_cell_state_cache_budget_mb_;
        }

        void set_local_error_coefficient_fast_path(bool enabled) noexcept
        {
            local_error_coefficient_fast_path_ = enabled;
        }

        [[nodiscard]] bool
        local_error_coefficient_fast_path() const noexcept
        {
            return local_error_coefficient_fast_path_;
        }

        void set_local_error_compact_state_shadow(bool enabled) noexcept
        {
            local_error_compact_state_shadow_ = enabled;
        }

        [[nodiscard]] bool
        local_error_compact_state_shadow() const noexcept
        {
            return local_error_compact_state_shadow_;
        }

        void set_local_error_cell_state_representation(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';
                    return static_cast<char>(std::tolower(c));
                });
            if (mode != "compact_split" && mode != "monolithic_debug")
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: invalid local-error cell-state representation.");
            local_error_cell_state_representation_ = std::move(mode);
        }

        [[nodiscard]] const std::string&
        local_error_cell_state_representation() const noexcept
        {
            return local_error_cell_state_representation_;
        }

        void set_local_error_flux_diagnostics_mode(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';
                    return static_cast<char>(std::tolower(c));
                });
            if (mode != "auto" && mode != "standalone" &&
                mode != "streaming_reuse")
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: invalid local-error flux-diagnostics mode.");
            }
            local_error_flux_diagnostics_mode_ = std::move(mode);
        }

        [[nodiscard]] const std::string&
        local_error_flux_diagnostics_mode() const noexcept
        {
            return local_error_flux_diagnostics_mode_;
        }

        void set_local_error_patch_solver(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';
                    return static_cast<char>(std::tolower(c));
                });
            if (mode != "current_dense" &&
                mode != "reduced_scalar_dense" &&
                mode != "auto")
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: invalid local-error patch solver.");
            }
            local_error_patch_solver_ = std::move(mode);
        }

        [[nodiscard]] const std::string&
        local_error_patch_solver() const noexcept
        {
            return local_error_patch_solver_;
        }

        void set_shared_context_validation(std::string mode)
        {
            std::transform(
                mode.begin(),
                mode.end(),
                mode.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';

                    return static_cast<char>(std::tolower(c));
                });
            shared_context_validation_ =
                mode == "sample" || mode == "full_debug" ? mode : "off";
        }

        [[nodiscard]] const std::string&
        shared_context_validation() const noexcept
        {
            return shared_context_validation_;
        }

        void set_fused_error_and_flux_diagnostics(bool enabled) noexcept
        {
            fused_error_and_flux_diagnostics_ = enabled;
        }

        [[nodiscard]] bool fused_error_and_flux_diagnostics() const noexcept
        {
            return fused_error_and_flux_diagnostics_;
        }

        void set_local_error_reuse_patch_solve_workspace(bool enabled) noexcept
        {
            local_error_reuse_patch_solve_workspace_ = enabled;
        }

        [[nodiscard]] bool local_error_reuse_patch_solve_workspace()
            const noexcept
        {
            return local_error_reuse_patch_solve_workspace_;
        }

        void initialize(double time_tol = 0.0)
        {
            reset_local_state_();

            if (source_y_space_)
            {
                owned_slab_space_.emplace(*source_y_space_);
                TimeSlabBuilder<GT, FETraits>::initialize(
                    *owned_slab_space_,
                    time_tol);
                slab_space_ptr_ = &*owned_slab_space_;
            }
            else if (!slab_space_ptr_)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d::initialize: no source Y-space or slab-space available.");
            }

            patch_set_.emplace(
                TimeSlabVertexPatchBuilder<GT, FETraits>::build(
                    slab_space_ref()));
            build_local_spaces_();
            initialized_ = true;
        }

        [[nodiscard]] const SourceYSpace& source_y_space() const
        {
            if (!source_y_space_)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d::source_y_space: no source Y-space stored.");
            }
            return *source_y_space_;
        }

        [[nodiscard]] const XSpace& x_space() const noexcept
        {
            return *x_space_;
        }

        [[nodiscard]] const SlabSpaceType& slab_space_ref() const
        {
            if (!slab_space_ptr_)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d::slab_space_ref: slab space not initialized.");
            }
            return *slab_space_ptr_;
        }

        [[nodiscard]] const PatchSetType& patch_set() const
        {
            ensure_initialized_();
            return *patch_set_;
        }

        [[nodiscard]] int n_patches() const
        {
            ensure_initialized_();
            return patch_set_->n_patches();
        }

        [[nodiscard]] const FluxSpaceType& flux_space(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return flux_spaces_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] const ScalarSpaceType& scalar_space(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return scalar_spaces_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] const FluxFunctionType& flux_function(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return flux_functions_[static_cast<std::size_t>(patch_id)];
        }

        void update_patch_flux_coefficients(
            int patch_id,
            const VectorType& coefficients)
        {
            ensure_patch_index_(patch_id);
            flux_functions_[static_cast<std::size_t>(patch_id)]
                .update_coefficients(coefficients);
        }

        [[nodiscard]] const ScalarFunctionType& scalar_function(int patch_id) const
        {
            ensure_patch_index_(patch_id);
            return scalar_functions_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] bool last_solve_all_patches_used_openmp() const noexcept
        {
            return last_solve_all_patches_used_openmp_;
        }

        [[nodiscard]] bool has_fused_flux_diagnostics() const noexcept
        {
            return last_fused_flux_diagnostics_.has_value();
        }

        [[nodiscard]] const FluxDiagnostics& fused_flux_diagnostics() const
        {
            if (!last_fused_flux_diagnostics_.has_value())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d::fused_flux_diagnostics: no fused diagnostics are available.");
            }
            return *last_fused_flux_diagnostics_;
        }

        [[nodiscard]] FluxDiagnosticsRuntimeStats
        fused_flux_diagnostics_runtime_stats() const noexcept
        {
            return last_fused_flux_diagnostics_runtime_stats_;
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch(
            int patch_id,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15)
        {
            ensure_initialized_();

            finite_element::detail::CellGeometryCache<XSpace> x_geometry_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace> ancestor_cache(
                *x_space_);
            auto slab_geometry_caches = make_slab_geometry_caches_();
            last_solve_all_patches_used_openmp_ = false;
            last_fused_flux_diagnostics_.reset();

            solve_patch_impl_<QSpace, QTime>(
                patch_id,
                lambda_tilde,
                ell,
                u_delta,
                M,
                x_geometry_cache,
                slab_geometry_caches,
                ancestor_cache,
                solver,
                options,
                zero_tol);
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_all_patches(
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            ensure_initialized_();

            const int patch_count = n_patches();
            last_solve_all_patches_used_openmp_ = false;
            last_fused_flux_diagnostics_.reset();
            last_fused_flux_diagnostics_runtime_stats_ = {};
            finite_element::assembly::error_system::LocalErrorProblemTimingStats
                timing_stats;
            detail::record_local_error_patch_profile(
                timing,
                detail::make_local_error_patch_profile(
                    *patch_set_,
                    flux_spaces_,
                    scalar_spaces_,
                    slab_space_ref().n_slabs()),
                true);
            using RTCellCache =
                finite_element::assembly::detail::
                    LocalErrorRTCellQuadratureCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType>;
            using ABElementCache =
                finite_element::assembly::error_system::
                    LocalABElementCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using RHSStateCache =
                finite_element::assembly::error_system::
                    LocalRHSStateCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType>;
            using UnifiedCellStateCache =
                finite_element::assembly::error_system::
                    LocalUnifiedCellStateCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using QpointStateCache =
                finite_element::assembly::error_system::
                    LocalErrorQpointStateCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using Tables =
                finite_element::assembly::detail::
                    LocalErrorQuadratureTables2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;

            if (local_operator_cache_ == nullptr &&
                local_error_patch_tile_size_ > 0 &&
                (local_error_patch_tile_size_ < patch_count ||
                 local_error_cell_state_representation_ == "compact_split"))
            {
                solve_all_patches_tiled_explicit_<
                    QSpace,
                    QTime>(
                    lambda_tilde,
                    ell,
                    u_delta,
                    M,
                    zero_tol,
                    timing);
                return;
            }

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
            {
                const auto thread_policy =
                    select_local_error_thread_policy_(
                        patch_count,
                        dense_blocks_bytes_for_patch_range_(0, patch_count) +
                            estimate_scalar_reduction_basis_bytes_());
                record_local_error_thread_policy_(timing, thread_policy);
                const int n_threads = thread_policy.selected_threads;
                if (n_threads > 1)
                {
                    struct SlabCellRef
                    {
                        int slab_id = -1;
                        int slab_cell_id = -1;
                    };

                    std::vector<SlabCellRef> active_slab_cells;
                    for (int slab_id = 0;
                         slab_id < slab_space_ref().n_slabs();
                         ++slab_id)
                    {
                        const auto& slab = slab_space_ref().slab(slab_id);
                        for (const int slab_cell_id : slab.active_cells())
                        {
                            active_slab_cells.push_back(
                                SlabCellRef{slab_id, slab_cell_id});
                        }
                    }

                    std::vector<std::vector<int>> cell_color_classes;
                    std::vector<int> patch_color_marker(
                        static_cast<std::size_t>(patch_count),
                        -1);
                    for (int cell_index = 0;
                         cell_index <
                             static_cast<int>(active_slab_cells.size());
                         ++cell_index)
                    {
                        const auto cell =
                            active_slab_cells[
                                static_cast<std::size_t>(cell_index)];
                        const int membership_count =
                            patch_set_->cell_patch_count(
                                cell.slab_id,
                                cell.slab_cell_id);
                        int color = 0;
                        for (;; ++color)
                        {
                            bool conflict = false;
                            for (int membership_index = 0;
                                 membership_index < membership_count;
                                 ++membership_index)
                            {
                                const auto& membership =
                                    patch_set_->cell_patch_membership(
                                        cell.slab_id,
                                        cell.slab_cell_id,
                                        membership_index);
                                if (patch_color_marker[
                                        static_cast<std::size_t>(
                                            membership.patch_id)] == color)
                                {
                                    conflict = true;
                                    break;
                                }
                            }

                            if (!conflict)
                                break;
                        }

                        if (color >=
                            static_cast<int>(cell_color_classes.size()))
                        {
                            cell_color_classes.resize(
                                static_cast<std::size_t>(color + 1));
                        }
                        cell_color_classes[
                            static_cast<std::size_t>(color)]
                            .push_back(cell_index);

                        for (int membership_index = 0;
                             membership_index < membership_count;
                             ++membership_index)
                        {
                            const auto& membership =
                                patch_set_->cell_patch_membership(
                                    cell.slab_id,
                                    cell.slab_cell_id,
                                    membership_index);
                            patch_color_marker[
                                static_cast<std::size_t>(
                                    membership.patch_id)] = color;
                        }
                    }

                    auto shared_context_storage =
                        build_shared_context_if_requested_(
                        active_slab_cells,
                        timing_stats);
                    const bool use_shared_context_for_state =
                        local_error_context_storage_ == "shared_immutable" &&
                        shared_context_storage.has_value();
                    const auto shared_rhs_context =
                        finite_element::assembly::error_system::
                            LocalErrorProblemContext<
                                XSpace,
                                SlabSpaceType>{
                                x_space_,
                                slab_space_ptr_,
                                nullptr,
                                nullptr,
                                nullptr,
                                use_shared_context_for_state
                                    ? &*shared_context_storage
                                    : nullptr};

                    QpointStateCache qpoint_state_cache;
                    qpoint_state_cache.prepare_from_spaces(
                        flux_spaces_,
                        scalar_spaces_);
                    std::vector<finite_element::assembly::error_system::
                                    DenseLocalErrorBlocks>
                        dense_blocks;
                    std::vector<std::size_t> operator_cache_keys;
                    std::vector<char> operator_cache_hits(
                        static_cast<std::size_t>(patch_count),
                        0);
                    double operator_cache_hit_count = 0.0;
                    double operator_cache_miss_count = 0.0;
                    double operator_cache_lookup_seconds = 0.0;
                    std::vector<const typename LocalOperatorCache::OperatorData*>
                        factor_cache_entries;
                    std::vector<char> factor_cache_hits(
                        static_cast<std::size_t>(patch_count),
                        0);
                    double factor_cache_hit_count = 0.0;
                    double factor_cache_miss_count = 0.0;
                    double factor_cache_lookup_seconds = 0.0;
                    if (local_operator_cache_ == nullptr)
                    {
                        dense_blocks.resize(
                            static_cast<std::size_t>(patch_count));
                    }
                    else
                    {
                        const double cache_lookup_start = omp_get_wtime();
                        initialize_dense_blocks_and_load_operator_cache_<
                            QSpace,
                            QTime,
                            MFunction>(
                            dense_blocks,
                            operator_cache_keys,
                            operator_cache_hits,
                            operator_cache_hit_count,
                            operator_cache_miss_count);
                        operator_cache_lookup_seconds +=
                            omp_get_wtime() - cache_lookup_start;
                        const double factor_lookup_start = omp_get_wtime();
                        initialize_operator_factor_cache_status_(
                            operator_cache_keys,
                            factor_cache_entries,
                            factor_cache_hits,
                            factor_cache_hit_count,
                            factor_cache_miss_count);
                        factor_cache_lookup_seconds +=
                            omp_get_wtime() - factor_lookup_start;
                    }
                    const double ab_build_requests_skipped = 0.0;
                    std::vector<la::saddle::SaddlePointSolution<Backend>>
                        patch_solutions(static_cast<std::size_t>(patch_count));
                    std::vector<typename LocalOperatorCache::OperatorData>
                        pending_factor_cache_entries(
                            static_cast<std::size_t>(patch_count));
                    std::vector<char> pending_factor_cache_ready(
                        static_cast<std::size_t>(patch_count),
                        0);

                    last_solve_all_patches_used_openmp_ = true;
                    static_cast<void>(solver);
                    static_cast<void>(options);

                    double fused_region_seconds = 0.0;
                    double thread_context_seconds = 0.0;
                    double rt_cache_seconds = 0.0;
                    double ab_cache_seconds = 0.0;
                    double rhs_cache_seconds = 0.0;
                    double unified_cell_state_seconds = 0.0;
                    double table_seconds = 0.0;
                    double workspace_seconds = 0.0;
                    double assemble_A_seconds = 0.0;
                    double assemble_B_seconds = 0.0;
                    double assemble_f_seconds = 0.0;
                    double assemble_g_seconds = 0.0;
                    double factor_cache_factorization_seconds = 0.0;
                    double factor_cache_store_seconds = 0.0;
                    double factor_cache_solve_seconds = 0.0;
                    double factor_cache_reusable_miss_count = 0.0;
                    double reduced_basis_transform_seconds = 0.0;
                    double patch_solve_factorization_seconds = 0.0;
                    double patch_solve_apply_seconds = 0.0;
                    double patch_solver_current_dense_count = 0.0;
                    double patch_solver_reduced_scalar_dense_count = 0.0;
                    double patch_solver_reduced_fallback_count = 0.0;
                    double patch_solver_reduced_residual_fail_count = 0.0;
                    double patch_solver_current_dimension_sum = 0.0;
                    double patch_solver_current_dimension_count = 0.0;
                    double patch_solver_reduced_dimension_sum = 0.0;
                    double patch_solver_reduced_dimension_count = 0.0;
                    double solve_seconds = 0.0;
                    double writeback_seconds = 0.0;
                    auto batch_stats =
                        patch_solve_batch_stats_(0, patch_count, n_threads);
                    using ExplicitSolveWorkspace =
                        finite_element::assembly::error_system::
                            DenseLocalErrorExplicitSolveWorkspace2D<Backend>;
                    const auto solve_workspace_alloc_start =
                        std::chrono::steady_clock::now();
                    std::vector<ExplicitSolveWorkspace> solve_workspaces(
                        static_cast<std::size_t>(n_threads));
                    batch_stats.workspace_allocation_seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            solve_workspace_alloc_start)
                            .count();
                    const auto patch_solve_order =
                        patch_solve_order_(0, patch_count, n_threads);
                    std::vector<typename QpointStateCache::AuditStats>
                        thread_qpoint_audit_stats(
                            static_cast<std::size_t>(n_threads));

                    const double fused_start = omp_get_wtime();
                    double phase_start = 0.0;

#pragma omp parallel num_threads(n_threads) shared(phase_start)
                    {
                        int thread_id = 0;
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
                        thread_id = omp_get_thread_num();
#endif
                        auto& thread_qpoint_audit =
                            thread_qpoint_audit_stats[
                                static_cast<std::size_t>(thread_id)];

#pragma omp single
                        {
                            phase_start = omp_get_wtime();
                        }

                        std::optional<
                            finite_element::detail::CellGeometryCache<XSpace>>
                            thread_x_geometry_cache;
                        std::optional<
                            finite_element::assembly::detail::
                                SourceActiveAncestorCache<XSpace>>
                            thread_ancestor_cache;
                        std::vector<
                            finite_element::detail::CellGeometryCache<
                                LocalSlabSpaceType>>
                            thread_slab_geometry_caches;
                        finite_element::assembly::error_system::
                            LocalErrorProblemContext<XSpace, SlabSpaceType>
                            thread_rhs_context{};
                        const finite_element::assembly::error_system::
                            LocalErrorProblemContext<XSpace, SlabSpaceType>*
                            active_rhs_context = &shared_rhs_context;
                        if (!use_shared_context_for_state)
                        {
                            thread_x_geometry_cache.emplace(*x_space_);
                            thread_ancestor_cache.emplace(*x_space_);
                            thread_slab_geometry_caches =
                                make_slab_geometry_caches_();
                            thread_rhs_context =
                                finite_element::assembly::error_system::
                                    LocalErrorProblemContext<
                                        XSpace,
                                        SlabSpaceType>{
                                        x_space_,
                                        slab_space_ptr_,
                                        &*thread_x_geometry_cache,
                                        &thread_slab_geometry_caches,
                                        &*thread_ancestor_cache};
                            active_rhs_context = &thread_rhs_context;
                        }

#pragma omp barrier
#pragma omp single
                        {
                            thread_context_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

#pragma omp for schedule(dynamic, 8)
                        for (int request_id = 0;
                             request_id <
                                 qpoint_state_cache.n_build_requests();
                             ++request_id)
                        {
                            qpoint_state_cache.fill_build_request(
                                request_id,
                                flux_spaces_,
                                scalar_spaces_,
                                *active_rhs_context,
                                lambda_tilde,
                                u_delta,
                                ell,
                                M,
                                &thread_qpoint_audit,
                                local_error_coefficient_fast_path_,
                                local_error_compact_state_shadow_);
                        }

#pragma omp single
                        {
                            unified_cell_state_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

#pragma omp single
                        {
                            table_seconds = 0.0;
                            phase_start = omp_get_wtime();
                        }

#pragma omp for schedule(static)
                        for (int patch_id = 0;
                             patch_id < patch_count;
                             ++patch_id)
                        {
                            auto& block =
                                dense_blocks[
                                    static_cast<std::size_t>(patch_id)];
                            if (local_operator_cache_ == nullptr)
                            {
                                block.resize(
                                    flux_spaces_[
                                        static_cast<std::size_t>(patch_id)]
                                        .n_dofs(),
                                    scalar_spaces_[
                                        static_cast<std::size_t>(patch_id)]
                                        .n_dofs());
                            }
                            else
                            {
                                block.f.set_zero();
                                block.g.set_zero();
                            }
                        }

#pragma omp single
                        {
                            workspace_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

                        for (const auto& color_cells : cell_color_classes)
                        {
#pragma omp for schedule(static)
                            for (int color_cell_index = 0;
                                 color_cell_index <
                                     static_cast<int>(color_cells.size());
                                 ++color_cell_index)
                            {
                                const auto cell =
                                    active_slab_cells[
                                        static_cast<std::size_t>(
                                            color_cells[
                                                static_cast<std::size_t>(
                                                    color_cell_index)])];
                                const int membership_count =
                                    patch_set_->cell_patch_count(
                                        cell.slab_id,
                                        cell.slab_cell_id);
                                const auto& state_cell =
                                    qpoint_state_cache.cell(
                                        cell.slab_id,
                                        cell.slab_cell_id);
                                const auto& local_A =
                                    finite_element::assembly::error_system::
                                        local_rt_mass_matrix_from_qpoint_state_2d<
                                            FluxSpaceType>(
                                            state_cell);
                                for (int membership_index = 0;
                                     membership_index < membership_count;
                                     ++membership_index)
                                {
                                    const auto& membership =
                                        patch_set_->cell_patch_membership(
                                            cell.slab_id,
                                            cell.slab_cell_id,
                                            membership_index);
                                    const int patch_id = membership.patch_id;
                                    if (operator_cache_hits[
                                            static_cast<std::size_t>(
                                                patch_id)] != 0)
                                    {
                                        continue;
                                    }

                                    finite_element::assembly::error_system::
                                        scatter_rt_local_matrix_dense_2d(
                                            dense_blocks[
                                                static_cast<std::size_t>(
                                                    patch_id)]
                                                .A,
                                            local_A,
                                            flux_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)],
                                            membership.patch_cell_index,
                                            zero_tol);
                                }
                            }
                        }

#pragma omp single
                        {
                            assemble_A_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

                        for (const auto& color_cells : cell_color_classes)
                        {
#pragma omp for schedule(static)
                            for (int color_cell_index = 0;
                                 color_cell_index <
                                     static_cast<int>(color_cells.size());
                                 ++color_cell_index)
                            {
                                const auto cell =
                                    active_slab_cells[
                                        static_cast<std::size_t>(
                                            color_cells[
                                                static_cast<std::size_t>(
                                                    color_cell_index)])];
                                const int membership_count =
                                    patch_set_->cell_patch_count(
                                        cell.slab_id,
                                        cell.slab_cell_id);
                                const auto& state_cell =
                                    qpoint_state_cache.cell(
                                        cell.slab_id,
                                        cell.slab_cell_id);
                                const auto& local_B =
                                    finite_element::assembly::error_system::
                                        local_divergence_matrix_from_qpoint_state_2d<
                                            ScalarSpaceType,
                                            FluxSpaceType>(
                                            state_cell);
                                for (int membership_index = 0;
                                     membership_index < membership_count;
                                     ++membership_index)
                                {
                                    const auto& membership =
                                        patch_set_->cell_patch_membership(
                                            cell.slab_id,
                                            cell.slab_cell_id,
                                            membership_index);
                                    const int patch_id = membership.patch_id;
                                    if (operator_cache_hits[
                                            static_cast<std::size_t>(
                                                patch_id)] != 0)
                                    {
                                        continue;
                                    }

                                    finite_element::assembly::error_system::
                                        scatter_divergence_local_matrix_dense_2d(
                                            dense_blocks[
                                                static_cast<std::size_t>(
                                                    patch_id)]
                                                .B,
                                            local_B,
                                            scalar_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)],
                                            flux_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)],
                                            membership.patch_cell_index,
                                            zero_tol);
                                }
                            }
                        }

#pragma omp single
                        {
                            assemble_B_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

                        for (const auto& color_cells : cell_color_classes)
                        {
#pragma omp for schedule(static)
                            for (int color_cell_index = 0;
                                 color_cell_index <
                                     static_cast<int>(color_cells.size());
                                 ++color_cell_index)
                            {
                                const auto cell =
                                    active_slab_cells[
                                        static_cast<std::size_t>(
                                            color_cells[
                                                static_cast<std::size_t>(
                                                    color_cell_index)])];
                                const int membership_count =
                                    patch_set_->cell_patch_count(
                                        cell.slab_id,
                                        cell.slab_cell_id);
                                for (int membership_index = 0;
                                     membership_index < membership_count;
                                     ++membership_index)
                                {
                                    const auto& membership =
                                        patch_set_->cell_patch_membership(
                                            cell.slab_id,
                                            cell.slab_cell_id,
                                            membership_index);
                                    const int patch_id = membership.patch_id;
                                    const auto& flux_space =
                                        flux_spaces_[
                                            static_cast<std::size_t>(
                                                patch_id)];

                                    la::local::FixedLocalVector<
                                        FluxSpaceType::local_dofs_v> local_f;
                                    finite_element::assembly::detail::
                                        zero_local_vector(local_f);
                                    const auto& state_cell =
                                        qpoint_state_cache.cell(
                                            cell.slab_id,
                                            cell.slab_cell_id);
                                    finite_element::assembly::error_system::
                                        accumulate_patch_flux_rhs_from_qpoint_state_2d(
                                            local_f,
                                            flux_space,
                                            state_cell,
                                            membership.patch_cell_index);
                                    finite_element::assembly::error_system::
                                        scatter_rt_local_vector_dense_2d(
                                            dense_blocks[
                                                static_cast<std::size_t>(
                                                    patch_id)]
                                                .f,
                                            local_f,
                                            flux_space,
                                            membership.patch_cell_index);
                                }
                            }
                        }

#pragma omp single
                        {
                            assemble_f_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

                        for (const auto& color_cells : cell_color_classes)
                        {
#pragma omp for schedule(static)
                            for (int color_cell_index = 0;
                                 color_cell_index <
                                     static_cast<int>(color_cells.size());
                                 ++color_cell_index)
                            {
                                const auto cell =
                                    active_slab_cells[
                                        static_cast<std::size_t>(
                                            color_cells[
                                                static_cast<std::size_t>(
                                                    color_cell_index)])];
                                const int membership_count =
                                    patch_set_->cell_patch_count(
                                        cell.slab_id,
                                        cell.slab_cell_id);
                                for (int membership_index = 0;
                                     membership_index < membership_count;
                                     ++membership_index)
                                {
                                    const auto& membership =
                                        patch_set_->cell_patch_membership(
                                            cell.slab_id,
                                            cell.slab_cell_id,
                                            membership_index);
                                    const int patch_id = membership.patch_id;
                                    const auto& scalar_space =
                                        scalar_spaces_[
                                            static_cast<std::size_t>(
                                                patch_id)];

                                    la::local::FixedLocalVector<
                                        ScalarSpaceType::local_dofs_v> local_g;
                                    finite_element::assembly::detail::
                                        zero_local_vector(local_g);
                                    const auto& state_cell =
                                        qpoint_state_cache.cell(
                                            cell.slab_id,
                                            cell.slab_cell_id);
                                    finite_element::assembly::error_system::
                                        accumulate_patch_scalar_rhs_from_qpoint_state_2d(
                                            local_g,
                                            scalar_space,
                                            state_cell,
                                            membership.patch_cell_index);
                                    finite_element::assembly::error_system::
                                        scatter_scalar_local_vector_dense_2d(
                                            dense_blocks[
                                                static_cast<std::size_t>(
                                                    patch_id)]
                                                .g,
                                            local_g,
                                            scalar_space,
                                            membership.patch_cell_index);
                                }
                            }
                        }

#pragma omp single
                        {
                            assemble_g_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

                        if (local_operator_cache_ != nullptr)
                        {
#pragma omp for schedule(static) reduction(+:factor_cache_factorization_seconds, factor_cache_reusable_miss_count)
                            for (int patch_id = 0;
                                 patch_id < patch_count;
                                 ++patch_id)
                            {
                                if (factor_cache_hits[
                                        static_cast<std::size_t>(
                                            patch_id)] != 0)
                                {
                                    continue;
                                }

                                double local_factorization_seconds = 0.0;
                                typename LocalOperatorCache::OperatorData
                                    data;
                                const bool reusable =
                                    build_operator_factor_cache_entry_(
                                        dense_blocks[
                                            static_cast<std::size_t>(
                                                patch_id)],
                                        scalar_spaces_[
                                            static_cast<std::size_t>(
                                                patch_id)],
                                        zero_tol,
                                        data,
                                        local_factorization_seconds);
                                factor_cache_factorization_seconds +=
                                    local_factorization_seconds;
                                if (reusable)
                                {
                                    pending_factor_cache_entries[
                                        static_cast<std::size_t>(
                                            patch_id)] = std::move(data);
                                    pending_factor_cache_ready[
                                        static_cast<std::size_t>(
                                            patch_id)] = 1;
                                    factor_cache_reusable_miss_count += 1.0;
                                }
                            }

#pragma omp single
                            {
                                const double store_start = omp_get_wtime();
                                store_operator_factor_cache_entries_(
                                    pending_factor_cache_entries,
                                    pending_factor_cache_ready,
                                    operator_cache_keys);
                                factor_cache_store_seconds +=
                                    omp_get_wtime() - store_start;

                                const double factor_lookup_start =
                                    omp_get_wtime();
                                load_operator_factor_cache_entries_(
                                    operator_cache_keys,
                                    factor_cache_entries);
                                factor_cache_lookup_seconds +=
                                    omp_get_wtime() - factor_lookup_start;
                                phase_start = omp_get_wtime();
                            }
                        }
                        else
                        {
#pragma omp single
                            {
                                phase_start = omp_get_wtime();
                            }
                        }

#pragma omp for schedule(static) reduction(+:factor_cache_solve_seconds, reduced_basis_transform_seconds, patch_solve_factorization_seconds, patch_solve_apply_seconds, patch_solver_current_dense_count, patch_solver_reduced_scalar_dense_count, patch_solver_reduced_fallback_count, patch_solver_reduced_residual_fail_count, patch_solver_current_dimension_sum, patch_solver_current_dimension_count, patch_solver_reduced_dimension_sum, patch_solver_reduced_dimension_count)
                        for (int solve_index = 0;
                             solve_index <
                                 static_cast<int>(patch_solve_order.size());
                             ++solve_index)
                        {
                            const int patch_id =
                                patch_solve_order[
                                    static_cast<std::size_t>(solve_index)];
                            const double patch_solve_start =
                                omp_get_wtime();
                            double local_reduced_transform_seconds = 0.0;
                            double local_factorization_seconds = 0.0;
                            double local_solve_apply_seconds = 0.0;
                            double local_current_dense_count = 0.0;
                            double local_reduced_scalar_dense_count = 0.0;
                            double local_reduced_fallback_count = 0.0;
                            double local_reduced_residual_fail_count = 0.0;
                            double local_current_dimension_sum = 0.0;
                            double local_current_dimension_count = 0.0;
                            double local_reduced_dimension_sum = 0.0;
                            double local_reduced_dimension_count = 0.0;
                            patch_solutions[
                                static_cast<std::size_t>(patch_id)] =
                                [&]() {
                                    finite_element::assembly::error_system::
                                        DenseLocalErrorExplicitSolveWorkspace2D<
                                            Backend>* workspace = nullptr;
                                    if (use_local_error_patch_solve_workspace_() &&
                                        local_operator_cache_ == nullptr)
                                    {
                                        const int workspace_id =
                                            omp_get_thread_num();
                                        workspace =
                                            &solve_workspaces[
                                                static_cast<std::size_t>(
                                                    workspace_id)];
                                    }
                                    return
                                        solve_dense_local_error_patch_with_selected_solver_(
                                            dense_blocks[
                                                static_cast<std::size_t>(
                                                    patch_id)],
                                            scalar_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)],
                                            workspace,
                                            local_operator_cache_ != nullptr
                                                ? factor_cache_entries[
                                                      static_cast<std::size_t>(
                                                          patch_id)]
                                                : nullptr,
                                            zero_tol,
                                            &local_reduced_transform_seconds,
                                            &local_factorization_seconds,
                                            &local_solve_apply_seconds,
                                            &local_current_dense_count,
                                            &local_reduced_scalar_dense_count,
                                            &local_reduced_fallback_count,
                                            &local_reduced_residual_fail_count,
                                            &local_current_dimension_sum,
                                            &local_current_dimension_count,
                                            &local_reduced_dimension_sum,
                                            &local_reduced_dimension_count);
                                }();
                            reduced_basis_transform_seconds +=
                                local_reduced_transform_seconds;
                            patch_solve_factorization_seconds +=
                                local_factorization_seconds;
                            patch_solve_apply_seconds +=
                                local_solve_apply_seconds;
                            patch_solver_current_dense_count +=
                                local_current_dense_count;
                            patch_solver_reduced_scalar_dense_count +=
                                local_reduced_scalar_dense_count;
                            patch_solver_reduced_fallback_count +=
                                local_reduced_fallback_count;
                            patch_solver_reduced_residual_fail_count +=
                                local_reduced_residual_fail_count;
                            patch_solver_current_dimension_sum +=
                                local_current_dimension_sum;
                            patch_solver_current_dimension_count +=
                                local_current_dimension_count;
                            patch_solver_reduced_dimension_sum +=
                                local_reduced_dimension_sum;
                            patch_solver_reduced_dimension_count +=
                                local_reduced_dimension_count;
                            if (local_operator_cache_ != nullptr &&
                                factor_cache_entries[
                                    static_cast<std::size_t>(patch_id)] !=
                                    nullptr)
                            {
                                factor_cache_solve_seconds +=
                                    omp_get_wtime() - patch_solve_start;
                            }
                        }

#pragma omp single
                        {
                            solve_seconds =
                                omp_get_wtime() - phase_start;
                            phase_start = omp_get_wtime();
                        }

#pragma omp for schedule(static)
                        for (int patch_id = 0;
                             patch_id < patch_count;
                             ++patch_id)
                        {
                            flux_functions_[static_cast<std::size_t>(patch_id)]
                                .update_coefficients(
                                    patch_solutions[
                                        static_cast<std::size_t>(patch_id)]
                                        .lambda);
                            scalar_functions_[static_cast<std::size_t>(patch_id)]
                                .update_coefficients(
                                    patch_solutions[
                                        static_cast<std::size_t>(patch_id)]
                                        .u);
                        }

#pragma omp single
                        {
                            writeback_seconds =
                                omp_get_wtime() - phase_start;
                        }
                    }

                    fused_region_seconds = omp_get_wtime() - fused_start;
                    for (const auto& thread_qpoint_audit :
                         thread_qpoint_audit_stats)
                    {
                        add_local_error_qpoint_state_audit_stats_(
                            timing_stats,
                            thread_qpoint_audit);
                    }
                    add_local_error_qpoint_state_counters_(
                        timing_stats,
                        qpoint_state_cache);

#if APF_DISABLE_FUSED_FLUX_DIAGNOSTICS_2D == 0
                    if (fused_error_and_flux_diagnostics_)
                    {
                        FluxDiagnosticsRuntimeStats diagnostics_runtime_stats;
                        last_fused_flux_diagnostics_ =
                            compute_fused_flux_diagnostics_from_qpoint_state_<
                                QSpace,
                                QTime>(
                                qpoint_state_cache,
                                timing,
                                n_threads,
                                &diagnostics_runtime_stats);
                        last_fused_flux_diagnostics_runtime_stats_ =
                            diagnostics_runtime_stats;
                    }
#endif

                    record_local_error_memory_counters_(
                        timing,
                        dense_blocks,
                        0,
                        qpoint_state_cache.estimated_memory_bytes(),
                        patch_solutions_bytes_(patch_solutions),
                        n_threads);

                    timing.add(
                        "time_slab.local_error_solves.fused_openmp_used",
                        0.0);
                    timing.add(
                        "time_slab.local_error_solves.lock_free_colored_cell_assembly",
                        0.0);
                    timing.add(
                        "time_slab.local_error_solves.lock_free_cell_colors.count",
                        static_cast<double>(cell_color_classes.size()));
                    timing.add(
                        "time_slab.local_error_solves.fused_openmp_threads.count",
                        static_cast<double>(n_threads));
                    timing.add(
                        "time_slab.local_error_solves.fused_openmp_parallel_region",
                        fused_region_seconds);
                    timing.add(
                        "time_slab.local_error_solves.fused_thread_context_construction",
                        thread_context_seconds);
                    timing.add(
                        "time_slab.local_error_solves.unified_cell_state_construction",
                        unified_cell_state_seconds);
                    timing.add(
                        "time_slab.local_error_solves.rt_cell_cache_construction",
                        rt_cache_seconds);
                    timing.add(
                        "time_slab.local_error_solves.rt_cell_cache.requested_patch_cells.count",
                        static_cast<double>(qpoint_state_cache.requested_patch_cells()));
                    timing.add(
                        "time_slab.local_error_solves.rt_cell_cache.unique_slab_cells.count",
                        static_cast<double>(qpoint_state_cache.unique_slab_cells()));
                    timing.add(
                        "time_slab.local_error_solves.rt_cell_cache.duplicate_patch_cells.count",
                        static_cast<double>(qpoint_state_cache.duplicate_patch_cells()));
                    timing.add(
                        "time_slab.local_error_solves.ab_element_cache_construction",
                        ab_cache_seconds);
                    timing.add(
                        "time_slab.local_error_solves.ab_element_cache.requested_patch_cells.count",
                        static_cast<double>(qpoint_state_cache.requested_patch_cells()));
                    timing.add(
                        "time_slab.local_error_solves.ab_element_cache.unique_slab_cells.count",
                        static_cast<double>(qpoint_state_cache.unique_slab_cells()));
                    timing.add(
                        "time_slab.local_error_solves.ab_element_cache.duplicate_patch_cells.count",
                        static_cast<double>(qpoint_state_cache.duplicate_patch_cells()));
                    timing.add(
                        "time_slab.local_error_solves.rhs_state_cache_construction",
                        rhs_cache_seconds);
                    timing.add(
                        "time_slab.local_error_solves.rhs_state_cache.requested_patch_cells.count",
                        static_cast<double>(qpoint_state_cache.requested_patch_cells()));
                    timing.add(
                        "time_slab.local_error_solves.rhs_state_cache.unique_slab_cells.count",
                        static_cast<double>(qpoint_state_cache.unique_slab_cells()));
                    timing.add(
                        "time_slab.local_error_solves.rhs_state_cache.duplicate_patch_cells.count",
                        static_cast<double>(qpoint_state_cache.duplicate_patch_cells()));
                    timing.add(
                        "time_slab.local_error_solves.qpoint_state_cache.diffusion_tensor_evaluations.count",
                        static_cast<double>(
                            qpoint_state_cache.diffusion_tensor_evaluations()));
                    record_local_error_reuse_summary_(
                        timing,
                        qpoint_state_cache.requested_patch_cells(),
                        qpoint_state_cache.unique_slab_cells());
                    record_local_error_global_reuse_summary_(
                        timing,
                        active_slab_cells);
                    timing.add(
                        "time_slab.local_error_solves.coefficient_writeback",
                        writeback_seconds);
                    timing.add(
                        "time_slab.local_error_solves.operator_cache_hits.count",
                        operator_cache_hit_count);
                    timing.add(
                        "time_slab.local_error_solves.patch_pattern_reused.count",
                        operator_cache_hit_count);
                    timing.add(
                        "time_slab.local_error_solves.operator_cache_misses.count",
                        operator_cache_miss_count);
                    timing.add(
                        "time_slab.local_error_solves.operator_cache_ab_build_requests_skipped.count",
                        ab_build_requests_skipped);
                    if (local_operator_cache_ != nullptr)
                    {
                        timing.add(
                            "time_slab.local_error_solves.operator_cache_lookup",
                            operator_cache_lookup_seconds);
                        timing.add(
                            "time_slab.local_error_solves.factor_cache_hits.count",
                            factor_cache_hit_count);
                        timing.add(
                            "time_slab.local_error_solves.patch_factorization_reused.count",
                            factor_cache_hit_count);
                        timing.add(
                            "time_slab.local_error_solves.factor_cache_misses.count",
                            factor_cache_miss_count);
                        timing.add(
                            "time_slab.local_error_solves.factor_cache_reusable_misses.count",
                            factor_cache_reusable_miss_count);
                        timing.add(
                            "time_slab.local_error_solves.factor_cache_lookup",
                            factor_cache_lookup_seconds);
                        timing.add(
                            "local_error.cache_lookup_wall",
                            operator_cache_lookup_seconds +
                                factor_cache_lookup_seconds);
                        timing.add(
                            "time_slab.local_error_solves.factor_cache_factorization",
                            factor_cache_factorization_seconds);
                        timing.add(
                            "time_slab.local_error_solves.factor_cache_store",
                            factor_cache_store_seconds);
                        timing.add(
                            "time_slab.local_error_solves.factor_cache_solve",
                            factor_cache_solve_seconds);
                    }

                    timing_stats.quadrature_table_construction_seconds =
                        table_seconds;
                    timing_stats.assemble_A_seconds = assemble_A_seconds;
                    timing_stats.assemble_B_seconds = assemble_B_seconds;
                    timing_stats.assemble_C_seconds = workspace_seconds;
                    timing_stats.assemble_f_seconds = assemble_f_seconds;
                    timing_stats.assemble_g_seconds = assemble_g_seconds;
                    timing_stats.reduced_basis_transform_seconds =
                        reduced_basis_transform_seconds;
                    timing_stats.solve_patch_systems_seconds = solve_seconds;
                    timing_stats.patch_solver_mode =
                        patch_solver_mode_code_();
                    timing_stats.patch_solver_current_dense_count =
                        patch_solver_current_dense_count;
                    timing_stats.patch_solver_reduced_scalar_dense_count =
                        patch_solver_reduced_scalar_dense_count;
                    timing_stats.patch_solver_reduced_fallback_count =
                        patch_solver_reduced_fallback_count;
                    timing_stats.patch_solver_reduced_residual_fail_count =
                        patch_solver_reduced_residual_fail_count;
                    timing_stats.patch_solver_current_dimension_sum =
                        patch_solver_current_dimension_sum;
                    timing_stats.patch_solver_current_dimension_count =
                        patch_solver_current_dimension_count;
                    timing_stats.patch_solver_reduced_dimension_sum =
                        patch_solver_reduced_dimension_sum;
                    timing_stats.patch_solver_reduced_dimension_count =
                        patch_solver_reduced_dimension_count;
                    batch_stats.factorization_seconds =
                        patch_solve_factorization_seconds;
                    batch_stats.solve_apply_seconds =
                        patch_solve_apply_seconds;
                    if (batch_stats.dense_solver_workspace_bytes == 0)
                    {
                        std::size_t workspace_bytes = 0;
                        for (const auto& workspace : solve_workspaces)
                            workspace_bytes += workspace.estimated_memory_bytes();
                        batch_stats.dense_solver_workspace_bytes =
                            workspace_bytes;
                    }
                    add_patch_solve_batch_stats_(
                        timing_stats,
                        batch_stats);

                    if (false)
                        record_reduced_mean_zero_dimension_counts_(timing);
                    record_local_error_timing_(timing, timing_stats);
                    return;
                }
            }
#endif

            std::vector<finite_element::assembly::error_system::
                            DenseLocalErrorBlocks>
                dense_blocks;
            std::vector<std::size_t> operator_cache_keys;
            std::vector<char> operator_cache_hits(
                static_cast<std::size_t>(patch_count),
                0);
            double operator_cache_hit_count = 0.0;
            double operator_cache_miss_count = 0.0;
            std::vector<const typename LocalOperatorCache::OperatorData*>
                factor_cache_entries;
            std::vector<char> factor_cache_hits(
                static_cast<std::size_t>(patch_count),
                0);
            double factor_cache_hit_count = 0.0;
            double factor_cache_miss_count = 0.0;
            if (local_operator_cache_ == nullptr)
            {
                dense_blocks.reserve(static_cast<std::size_t>(patch_count));
                for (int patch_id = 0; patch_id < patch_count; ++patch_id)
                {
                    dense_blocks.emplace_back(
                        flux_spaces_[static_cast<std::size_t>(patch_id)]
                            .n_dofs(),
                        scalar_spaces_[static_cast<std::size_t>(patch_id)]
                            .n_dofs());
                }
            }
            else
            {
                auto cache_timer =
                    timing.scoped(
                        "time_slab.local_error_solves.operator_cache_lookup");
                initialize_dense_blocks_and_load_operator_cache_<
                    QSpace,
                    QTime,
                    MFunction>(
                    dense_blocks,
                    operator_cache_keys,
                    operator_cache_hits,
                    operator_cache_hit_count,
                    operator_cache_miss_count);
                initialize_operator_factor_cache_status_(
                    operator_cache_keys,
                    factor_cache_entries,
                    factor_cache_hits,
                    factor_cache_hit_count,
                    factor_cache_miss_count);
            }

            std::optional<QpointStateCache> qpoint_state_cache;
            double ab_build_requests_skipped = 0.0;
            const auto fallback_active_slab_cells =
                collect_active_slab_cell_refs_();
            auto shared_context_storage =
                build_shared_context_if_requested_(
                    fallback_active_slab_cells,
                    timing_stats);
            const bool use_shared_context_for_state =
                local_error_context_storage_ == "shared_immutable" &&
                shared_context_storage.has_value();
            finite_element::detail::CellGeometryCache<XSpace>
                rhs_x_geometry_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>
                rhs_ancestor_cache(*x_space_);
            auto rhs_slab_geometry_caches = make_slab_geometry_caches_();
            const auto rhs_context =
                finite_element::assembly::error_system::LocalErrorProblemContext<
                    XSpace,
                    SlabSpaceType>{
                        x_space_,
                        slab_space_ptr_,
                        &rhs_x_geometry_cache,
                        &rhs_slab_geometry_caches,
                        &rhs_ancestor_cache,
                        nullptr
                    };
            const auto shared_rhs_context =
                finite_element::assembly::error_system::LocalErrorProblemContext<
                    XSpace,
                    SlabSpaceType>{
                    x_space_,
                    slab_space_ptr_,
                    nullptr,
                    nullptr,
                    nullptr,
                    use_shared_context_for_state
                        ? &*shared_context_storage
                        : nullptr};
            const auto& active_rhs_context =
                use_shared_context_for_state ? shared_rhs_context : rhs_context;

            {
                auto cache_timer =
                    timing.scoped(
                        "time_slab.local_error_solves.unified_cell_state_construction");
                typename QpointStateCache::AuditStats qpoint_audit_stats;
                qpoint_state_cache.emplace();
                qpoint_state_cache->prepare_from_spaces(
                    flux_spaces_,
                    scalar_spaces_);
                for (int request_id = 0;
                     request_id < qpoint_state_cache->n_build_requests();
                     ++request_id)
                {
                    qpoint_state_cache->fill_build_request(
                        request_id,
                        flux_spaces_,
                        scalar_spaces_,
                        active_rhs_context,
                        lambda_tilde,
                        u_delta,
                        ell,
                        M,
                        &qpoint_audit_stats,
                        local_error_coefficient_fast_path_,
                        local_error_compact_state_shadow_);
                }
                add_local_error_qpoint_state_audit_stats_(
                    timing_stats,
                    qpoint_audit_stats);
            }
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.requested_patch_cells.count",
                static_cast<double>(qpoint_state_cache->requested_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.unique_slab_cells.count",
                static_cast<double>(qpoint_state_cache->unique_slab_cells()));
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.duplicate_patch_cells.count",
                static_cast<double>(qpoint_state_cache->duplicate_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.requested_patch_cells.count",
                static_cast<double>(qpoint_state_cache->requested_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.unique_slab_cells.count",
                static_cast<double>(qpoint_state_cache->unique_slab_cells()));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.duplicate_patch_cells.count",
                static_cast<double>(qpoint_state_cache->duplicate_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.requested_patch_cells.count",
                static_cast<double>(qpoint_state_cache->requested_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.unique_slab_cells.count",
                static_cast<double>(qpoint_state_cache->unique_slab_cells()));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.duplicate_patch_cells.count",
                static_cast<double>(qpoint_state_cache->duplicate_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.diffusion_tensor_evaluations.count",
                static_cast<double>(
                    qpoint_state_cache->diffusion_tensor_evaluations()));
            record_local_error_reuse_summary_(
                timing,
                qpoint_state_cache->requested_patch_cells(),
                qpoint_state_cache->unique_slab_cells());
            add_local_error_qpoint_state_counters_(
                timing_stats,
                *qpoint_state_cache);
            timing_stats.state_bytes_per_cell =
                static_cast<double>(
                    sizeof(typename QpointStateCache::CellData));
            timing_stats.estimated_full_cache_bytes =
                static_cast<double>(qpoint_state_cache->unique_slab_cells()) *
                timing_stats.state_bytes_per_cell;
            const double compact_bytes_per_cell =
                timing_stats.operator_state_bytes_per_cell +
                timing_stats.rhs_state_bytes_per_cell +
                timing_stats.flux_diagnostic_state_bytes_per_cell;
            if (compact_bytes_per_cell > 0.0)
            {
                timing_stats.estimated_compact_full_cache_gib =
                    static_cast<double>(
                        qpoint_state_cache->unique_slab_cells()) *
                    compact_bytes_per_cell /
                    (1024.0 * 1024.0 * 1024.0);
            }

            finite_element::assembly::error_system::
                assemble_dense_local_error_blocks_cell_first_time_2d_into_from_qpoint_state<
                    QSpace,
                    QTime>(
                    *patch_set_,
                    flux_spaces_,
                    scalar_spaces_,
                    *qpoint_state_cache,
                    dense_blocks,
                    &operator_cache_hits,
                    &timing_stats,
                    zero_tol);
            double factor_cache_reusable_miss_count = 0.0;
            if (local_operator_cache_ != nullptr)
            {
                std::vector<typename LocalOperatorCache::OperatorData>
                    pending_factor_cache_entries(
                        static_cast<std::size_t>(patch_count));
                std::vector<char> pending_factor_cache_ready(
                    static_cast<std::size_t>(patch_count),
                    0);
                {
                    auto factor_timer =
                        timing.scoped(
                            "time_slab.local_error_solves.factor_cache_factorization");
                    for (int patch_id = 0;
                         patch_id < patch_count;
                         ++patch_id)
                    {
                        if (factor_cache_hits[
                                static_cast<std::size_t>(patch_id)] != 0)
                        {
                            continue;
                        }

                        double factorization_seconds = 0.0;
                        const bool reusable =
                            build_operator_factor_cache_entry_(
                                dense_blocks[
                                    static_cast<std::size_t>(patch_id)],
                                scalar_space(patch_id),
                                zero_tol,
                                pending_factor_cache_entries[
                                    static_cast<std::size_t>(patch_id)],
                                factorization_seconds);
                        static_cast<void>(factorization_seconds);
                        if (reusable)
                        {
                            pending_factor_cache_ready[
                                static_cast<std::size_t>(patch_id)] = 1;
                            factor_cache_reusable_miss_count += 1.0;
                        }
                    }
                }
                {
                    auto store_timer =
                        timing.scoped(
                            "time_slab.local_error_solves.factor_cache_store");
                    store_operator_factor_cache_entries_(
                        pending_factor_cache_entries,
                        pending_factor_cache_ready,
                        operator_cache_keys);
                }
                {
                    auto lookup_timer =
                        timing.scoped(
                            "time_slab.local_error_solves.factor_cache_lookup");
                    load_operator_factor_cache_entries_(
                        operator_cache_keys,
                        factor_cache_entries);
                }
            }
            timing.add(
                "time_slab.local_error_solves.operator_cache_hits.count",
                operator_cache_hit_count);
            timing.add(
                "time_slab.local_error_solves.patch_pattern_reused.count",
                operator_cache_hit_count);
            timing.add(
                "time_slab.local_error_solves.operator_cache_misses.count",
                operator_cache_miss_count);
            timing.add(
                "time_slab.local_error_solves.operator_cache_ab_build_requests_skipped.count",
                ab_build_requests_skipped);
            if (local_operator_cache_ != nullptr)
            {
                timing.add(
                    "time_slab.local_error_solves.factor_cache_hits.count",
                    factor_cache_hit_count);
                timing.add(
                    "time_slab.local_error_solves.patch_factorization_reused.count",
                    factor_cache_hit_count);
                timing.add(
                    "time_slab.local_error_solves.factor_cache_misses.count",
                    factor_cache_miss_count);
                timing.add(
                    "time_slab.local_error_solves.factor_cache_reusable_misses.count",
                    factor_cache_reusable_miss_count);
            }

            static_cast<void>(lambda_tilde);
            static_cast<void>(ell);
            static_cast<void>(u_delta);
            static_cast<void>(M);
            static_cast<void>(solver);
            static_cast<void>(options);
            double reduced_basis_transform_seconds = 0.0;
            double patch_solver_current_dense_count = 0.0;
            double patch_solver_reduced_scalar_dense_count = 0.0;
            double patch_solver_reduced_fallback_count = 0.0;
            double patch_solver_reduced_residual_fail_count = 0.0;
            double patch_solver_current_dimension_sum = 0.0;
            double patch_solver_current_dimension_count = 0.0;
            double patch_solver_reduced_dimension_sum = 0.0;
            double patch_solver_reduced_dimension_count = 0.0;
            auto batch_stats =
                patch_solve_batch_stats_(0, patch_count, 1);
            finite_element::assembly::error_system::
                DenseLocalErrorExplicitSolveWorkspace2D<Backend>
                    solve_workspace;
            const auto patch_solve_order =
                patch_solve_order_(0, patch_count, 1);
            for (const int patch_id : patch_solve_order)
            {
                la::saddle::SaddlePointSolution<Backend> split;
                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming solve_timer(
                            &timing_stats.solve_patch_systems_seconds);
                    double local_reduced_transform_seconds = 0.0;
                    split =
                        [&]() {
                            finite_element::assembly::error_system::
                                DenseLocalErrorExplicitSolveWorkspace2D<
                                    Backend>* workspace = nullptr;
                            if (use_local_error_patch_solve_workspace_() &&
                                local_operator_cache_ == nullptr)
                            {
                                workspace = &solve_workspace;
                            }
                            return
                                solve_dense_local_error_patch_with_selected_solver_(
                                    dense_blocks[
                                        static_cast<std::size_t>(patch_id)],
                                    scalar_space(patch_id),
                                    workspace,
                                    local_operator_cache_ != nullptr
                                        ? factor_cache_entries[
                                              static_cast<std::size_t>(
                                                  patch_id)]
                                        : nullptr,
                                    zero_tol,
                                    &local_reduced_transform_seconds,
                                    &batch_stats.factorization_seconds,
                                    &batch_stats.solve_apply_seconds,
                                    &patch_solver_current_dense_count,
                                    &patch_solver_reduced_scalar_dense_count,
                                    &patch_solver_reduced_fallback_count,
                                    &patch_solver_reduced_residual_fail_count,
                                    &patch_solver_current_dimension_sum,
                                    &patch_solver_current_dimension_count,
                                    &patch_solver_reduced_dimension_sum,
                                    &patch_solver_reduced_dimension_count);
                        }();
                    reduced_basis_transform_seconds +=
                        local_reduced_transform_seconds;
                }

                flux_functions_[static_cast<std::size_t>(patch_id)]
                    .update_coefficients(split.lambda);
                scalar_functions_[static_cast<std::size_t>(patch_id)]
                    .update_coefficients(split.u);
            }
            timing_stats.reduced_basis_transform_seconds =
                reduced_basis_transform_seconds;
            timing_stats.patch_solver_mode = patch_solver_mode_code_();
            timing_stats.patch_solver_current_dense_count =
                patch_solver_current_dense_count;
            timing_stats.patch_solver_reduced_scalar_dense_count =
                patch_solver_reduced_scalar_dense_count;
            timing_stats.patch_solver_reduced_fallback_count =
                patch_solver_reduced_fallback_count;
            timing_stats.patch_solver_reduced_residual_fail_count =
                patch_solver_reduced_residual_fail_count;
            timing_stats.patch_solver_current_dimension_sum =
                patch_solver_current_dimension_sum;
            timing_stats.patch_solver_current_dimension_count =
                patch_solver_current_dimension_count;
            timing_stats.patch_solver_reduced_dimension_sum =
                patch_solver_reduced_dimension_sum;
            timing_stats.patch_solver_reduced_dimension_count =
                patch_solver_reduced_dimension_count;
            if (batch_stats.dense_solver_workspace_bytes == 0)
                batch_stats.dense_solver_workspace_bytes =
                    solve_workspace.estimated_memory_bytes();
            add_patch_solve_batch_stats_(timing_stats, batch_stats);

#if APF_DISABLE_FUSED_FLUX_DIAGNOSTICS_2D == 0
            if (fused_error_and_flux_diagnostics_)
            {
                FluxDiagnosticsRuntimeStats diagnostics_runtime_stats;
                last_fused_flux_diagnostics_ =
                    compute_fused_flux_diagnostics_from_qpoint_state_<
                        QSpace,
                        QTime>(
                        *qpoint_state_cache,
                        timing,
                        1,
                        &diagnostics_runtime_stats);
                last_fused_flux_diagnostics_runtime_stats_ =
                    diagnostics_runtime_stats;
            }
#endif

            record_local_error_memory_counters_(
                timing,
                dense_blocks,
                0,
                qpoint_state_cache->estimated_memory_bytes(),
                sequential_patch_solution_peak_bytes_(),
                1);

            if (false)
                record_reduced_mean_zero_dimension_counts_(timing);
            record_local_error_timing_(timing, timing_stats);
            return;
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch_sparse_reference(
            int patch_id,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15)
        {
            ensure_initialized_();

            finite_element::detail::CellGeometryCache<XSpace> x_geometry_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace> ancestor_cache(
                *x_space_);
            auto slab_geometry_caches = make_slab_geometry_caches_();
            last_solve_all_patches_used_openmp_ = false;

            solve_patch_sparse_reference_impl_<QSpace, QTime>(
                patch_id,
                lambda_tilde,
                ell,
                u_delta,
                M,
                x_geometry_cache,
                slab_geometry_caches,
                ancestor_cache,
                solver,
                options,
                zero_tol);
        }

        [[nodiscard]] VectorValue sigma_on_slab_cell(
            int slab_id,
            int slab_cell_id,
            const SpaceTimePoint& p) const
        {
            return sigma_and_div_sigma_on_slab_cell(slab_id, slab_cell_id, p).sigma;
        }

        [[nodiscard]] double div_sigma_on_slab_cell(
            int slab_id,
            int slab_cell_id,
            const SpaceTimePoint& p) const
        {
            return sigma_and_div_sigma_on_slab_cell(slab_id, slab_cell_id, p).div_sigma;
        }

        [[nodiscard]] FluxEvaluation sigma_and_div_sigma_on_slab_cell(
            int slab_id,
            int slab_cell_id,
            const SpaceTimePoint& p) const
        {
            ensure_initialized_();

            FluxEvaluation evaluation;
            const int membership_count =
                patch_set_->cell_patch_count(slab_id, slab_cell_id);

            for (int i = 0; i < membership_count; ++i)
            {
                const auto membership =
                    patch_set_->cell_patch_membership(slab_id, slab_cell_id, i);
                const auto patch_evaluation =
                    flux_function(membership.patch_id).evaluate_on_cell(
                        membership.patch_cell_index,
                        p);

                evaluation.sigma[0] += patch_evaluation.value[0];
                evaluation.sigma[1] += patch_evaluation.value[1];
                evaluation.div_sigma += patch_evaluation.divergence;
            }

            return evaluation;
        }

        [[nodiscard]] VectorValue sigma(const SpaceTimePoint& p) const
        {
            const auto loc = slab_space_ref().find_active_cell(p);
            if (!loc.is_valid())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d::sigma: point not found in any slab cell.");
            }

            return sigma_on_slab_cell(loc.slab_id, loc.cell_id, p);
        }

        [[nodiscard]] double div_sigma(const SpaceTimePoint& p) const
        {
            const auto loc = slab_space_ref().find_active_cell(p);
            if (!loc.is_valid())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d::div_sigma: point not found in any slab cell.");
            }

            return div_sigma_on_slab_cell(loc.slab_id, loc.cell_id, p);
        }

    private:
        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch_impl_(
            int patch_id,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            finite_element::detail::CellGeometryCache<XSpace>& x_geometry_cache,
            std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>&
                slab_geometry_caches,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            double zero_tol,
            const finite_element::assembly::detail::
                LocalErrorRTCellQuadratureCache2D<
                    QSpace,
                    QTime,
                    FluxSpaceType>* rt_cell_cache = nullptr,
            const finite_element::assembly::error_system::
                LocalABElementCache2D<
                    QSpace,
                    QTime,
                    FluxSpaceType,
                    ScalarSpaceType>* ab_element_cache = nullptr,
            const finite_element::assembly::error_system::
                LocalRHSStateCache2D<
                    QSpace,
                    QTime,
                    FluxSpaceType>* rhs_state_cache = nullptr,
            finite_element::assembly::error_system::LocalErrorProblemTimingStats*
                timing_stats = nullptr)
        {
            ensure_patch_index_(patch_id);

            const auto context =
                finite_element::assembly::error_system::LocalErrorProblemContext<
                    XSpace,
                    SlabSpaceType>{
                        x_space_,
                        slab_space_ptr_,
                        &x_geometry_cache,
                        &slab_geometry_caches,
                        &ancestor_cache
                    };

            const auto& patch_flux_space = flux_space(patch_id);
            const auto& patch_scalar_space = scalar_space(patch_id);
            using Tables =
                finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                    QSpace,
                    QTime,
                    FluxSpaceType,
                    ScalarSpaceType>;

            std::optional<Tables> tables;
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    table_timer(
                        timing_stats != nullptr
                            ? &timing_stats->quadrature_table_construction_seconds
                            : nullptr);
                if (rt_cell_cache != nullptr)
                    tables.emplace(patch_flux_space, patch_scalar_space, *rt_cell_cache);
                else
                    tables.emplace(patch_flux_space, patch_scalar_space);
            }

            finite_element::assembly::error_system::DenseLocalErrorBlocks blocks(
                patch_flux_space.n_dofs(),
                patch_scalar_space.n_dofs());
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    timer(
                        timing_stats != nullptr
                            ? &timing_stats->assemble_A_seconds
                            : nullptr);
                if (ab_element_cache != nullptr)
                {
                    finite_element::assembly::error_system::
                        assemble_dense_mat_A_time_2d_from_ab_cache<
                            QSpace,
                            QTime>(
                            blocks.A,
                            patch_flux_space,
                            patch_scalar_space,
                            *ab_element_cache,
                            zero_tol);
                }
                else
                {
                    finite_element::assembly::error_system::
                        assemble_dense_mat_A_time_2d<QSpace, QTime>(
                            blocks.A,
                            patch_flux_space,
                            patch_scalar_space,
                            *tables,
                            M,
                            zero_tol);
                }
            }
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    timer(
                        timing_stats != nullptr
                            ? &timing_stats->assemble_B_seconds
                            : nullptr);
                if (ab_element_cache != nullptr)
                {
                    finite_element::assembly::error_system::
                        assemble_dense_mat_B_time_2d_from_ab_cache<
                            QSpace,
                            QTime>(
                            blocks.B,
                            patch_flux_space,
                            patch_scalar_space,
                            *ab_element_cache,
                            zero_tol);
                }
                else
                {
                    finite_element::assembly::error_system::
                        assemble_dense_mat_B_time_2d<QSpace, QTime>(
                            blocks.B,
                            patch_flux_space,
                            patch_scalar_space,
                            *tables,
                            zero_tol);
                }
            }
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    timer(
                        timing_stats != nullptr
                            ? &timing_stats->assemble_C_seconds
                            : nullptr);
                blocks.C.set_zero(patch_scalar_space.n_dofs(),
                                 patch_scalar_space.n_dofs());
            }
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    timer(
                        timing_stats != nullptr
                            ? &timing_stats->assemble_f_seconds
                            : nullptr);
                if (rhs_state_cache != nullptr)
                {
                    finite_element::assembly::error_system::
                        assemble_dense_vec_f_time_2d_from_rhs_cache<
                            QSpace,
                            QTime>(
                            blocks.f,
                            patch_flux_space,
                            patch_scalar_space,
                            *tables,
                            *rhs_state_cache);
                }
                else
                {
                    finite_element::assembly::error_system::
                        assemble_dense_vec_f_time_2d<QSpace, QTime>(
                            blocks.f,
                            patch_flux_space,
                            patch_scalar_space,
                            *tables,
                            context,
                            lambda_tilde,
                            u_delta);
                }
            }
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    timer(
                        timing_stats != nullptr
                            ? &timing_stats->assemble_g_seconds
                            : nullptr);
                if (rhs_state_cache != nullptr)
                {
                    finite_element::assembly::error_system::
                        assemble_dense_vec_g_time_2d_from_rhs_cache<
                            QSpace,
                            QTime>(
                            blocks.g,
                            patch_flux_space,
                            patch_scalar_space,
                            *tables,
                            *rhs_state_cache);
                }
                else
                {
                    finite_element::assembly::error_system::
                        assemble_dense_vec_g_time_2d<QSpace, QTime>(
                            blocks.g,
                            patch_flux_space,
                            patch_scalar_space,
                            *tables,
                            context,
                            lambda_tilde,
                            u_delta,
                            ell,
                            M);
                }
            }

            static_cast<void>(solver);
            static_cast<void>(options);
            la::saddle::SaddlePointSolution<Backend> split;
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    solve_timer(
                        timing_stats != nullptr
                            ? &timing_stats->solve_patch_systems_seconds
                            : nullptr);
                double transform_seconds = 0.0;
                double factorization_seconds = 0.0;
                double solve_apply_seconds = 0.0;
                double current_dense_count = 0.0;
                double reduced_scalar_dense_count = 0.0;
                double reduced_fallback_count = 0.0;
                double reduced_residual_fail_count = 0.0;
                double current_dimension_sum = 0.0;
                double current_dimension_count = 0.0;
                double reduced_dimension_sum = 0.0;
                double reduced_dimension_count = 0.0;
                split =
                    solve_dense_local_error_patch_with_selected_solver_(
                        blocks,
                        patch_scalar_space,
                        nullptr,
                        nullptr,
                        zero_tol,
                        &transform_seconds,
                        &factorization_seconds,
                        &solve_apply_seconds,
                        &current_dense_count,
                        &reduced_scalar_dense_count,
                        &reduced_fallback_count,
                        &reduced_residual_fail_count,
                        &current_dimension_sum,
                        &current_dimension_count,
                        &reduced_dimension_sum,
                        &reduced_dimension_count);
                if (timing_stats != nullptr)
                {
                    timing_stats->patch_solver_mode =
                        patch_solver_mode_code_();
                    timing_stats->reduced_basis_transform_seconds +=
                        transform_seconds;
                    timing_stats->patch_solve_factorization_seconds +=
                        factorization_seconds;
                    timing_stats->patch_solve_apply_seconds +=
                        solve_apply_seconds;
                    timing_stats->patch_solver_current_dense_count +=
                        current_dense_count;
                    timing_stats->patch_solver_reduced_scalar_dense_count +=
                        reduced_scalar_dense_count;
                    timing_stats->patch_solver_reduced_fallback_count +=
                        reduced_fallback_count;
                    timing_stats->patch_solver_reduced_residual_fail_count +=
                        reduced_residual_fail_count;
                    timing_stats->patch_solver_current_dimension_sum +=
                        current_dimension_sum;
                    timing_stats->patch_solver_current_dimension_count +=
                        current_dimension_count;
                    timing_stats->patch_solver_reduced_dimension_sum +=
                        reduced_dimension_sum;
                    timing_stats->patch_solver_reduced_dimension_count +=
                        reduced_dimension_count;
                }
            }

            flux_functions_[static_cast<std::size_t>(patch_id)]
                .update_coefficients(split.lambda);
            scalar_functions_[static_cast<std::size_t>(patch_id)]
                .update_coefficients(split.u);
        }

        void record_reduced_mean_zero_dimension_counts_(
            const finite_element::detail::TimingRecorder& timing) const
        {
            double interior_patch_count = 0.0;
            double explicit_system_size_sum = 0.0;
            double reduced_system_size_sum = 0.0;
            double full_scalar_dofs_sum = 0.0;
            double reduced_scalar_dofs_sum = 0.0;
            double eliminated_unknowns_sum = 0.0;

            const int patch_count =
                patch_set_.has_value() ? patch_set_->n_patches() : 0;
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                const auto& flux =
                    flux_spaces_[static_cast<std::size_t>(patch_id)];
                const auto& scalar =
                    scalar_spaces_[static_cast<std::size_t>(patch_id)];
                const int n_lambda = flux.n_dofs();
                const int n_u = scalar.n_dofs();
                const int n_constraints =
                    scalar.has_mean_zero_constraint()
                        ? scalar.n_mean_zero_constraints()
                        : 0;
                const int n_u_reduced =
                    finite_element::assembly::error_system::
                        reduced_scalar_dimension_2d(scalar, n_u);
                const int explicit_size = n_lambda + n_u + n_constraints;
                const int reduced_size = n_lambda + n_u_reduced;

                if (n_constraints > 0)
                    interior_patch_count += 1.0;
                explicit_system_size_sum +=
                    static_cast<double>(explicit_size);
                reduced_system_size_sum +=
                    static_cast<double>(reduced_size);
                full_scalar_dofs_sum += static_cast<double>(n_u);
                reduced_scalar_dofs_sum +=
                    static_cast<double>(n_u_reduced);
                eliminated_unknowns_sum +=
                    static_cast<double>(explicit_size - reduced_size);
            }

            timing.add(
                "time_slab.local_error_solves.reduced_mean_zero.interior_patches.count",
                interior_patch_count);
            timing.add(
                "time_slab.local_error_solves.reduced_mean_zero.explicit_system_size_sum.count",
                explicit_system_size_sum);
            timing.add(
                "time_slab.local_error_solves.reduced_mean_zero.reduced_system_size_sum.count",
                reduced_system_size_sum);
            timing.add(
                "time_slab.local_error_solves.reduced_mean_zero.full_scalar_dofs_sum.count",
                full_scalar_dofs_sum);
            timing.add(
                "time_slab.local_error_solves.reduced_mean_zero.reduced_scalar_dofs_sum.count",
                reduced_scalar_dofs_sum);
            timing.add(
                "time_slab.local_error_solves.reduced_mean_zero.eliminated_unknowns_sum.count",
                eliminated_unknowns_sum);
        }

        template<class T>
        static void hash_combine_(std::size_t& seed, const T& value)
        {
            const std::size_t h = std::hash<T>{}(value);
            seed ^= h + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
        }

        struct ActiveSlabCellRef
        {
            int slab_id = -1;
            int slab_cell_id = -1;
        };

        [[nodiscard]] std::vector<ActiveSlabCellRef>
        collect_active_slab_cell_refs_() const
        {
            std::vector<ActiveSlabCellRef> active_slab_cells;
            for (int slab_id = 0;
                 slab_id < slab_space_ref().n_slabs();
                 ++slab_id)
            {
                const auto& slab = slab_space_ref().slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    active_slab_cells.push_back(
                        ActiveSlabCellRef{slab_id, slab_cell_id});
                }
            }
            return active_slab_cells;
        }

        template<class RTCellCache, class RHSStateCache>
        void accumulate_fused_flux_diagnostics_from_cached_cells_(
            const std::vector<ActiveSlabCellRef>& active_slab_cells,
            const RTCellCache& rt_cell_cache,
            const RHSStateCache& rhs_state_cache,
            FluxDiagnostics& result) const
        {
            for (const auto& cell : active_slab_cells)
            {
                const int membership_count =
                    patch_set_->cell_patch_count(
                        cell.slab_id,
                        cell.slab_cell_id);
                if (membership_count <= 0)
                {
                    throw std::runtime_error(
                        "TimeSlabEquilibratedFluxReconstruction2plus1d: unified streaming diagnostics encountered a slab cell without vertex-patch memberships.");
                }

                const auto& rt_cell =
                    rt_cell_cache.cell(cell.slab_id, cell.slab_cell_id);
                const auto& rhs_cell =
                    rhs_state_cache.cell(cell.slab_id, cell.slab_cell_id);

                double local_flux = 0.0;
                double local_residual = 0.0;

                for (int qp_id = 0;
                     qp_id < RTCellCache::n_quadrature_points_v;
                     ++qp_id)
                {
                    const auto& rt_qp =
                        rt_cell.points[static_cast<std::size_t>(qp_id)];
                    const auto& rhs_qp =
                        rhs_cell.points[static_cast<std::size_t>(qp_id)];

                    FluxEvaluation flux_evaluation;
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set_->cell_patch_membership(
                                cell.slab_id,
                                cell.slab_cell_id,
                                membership_index);
                        const auto& flux_function =
                            flux_functions_[
                                static_cast<std::size_t>(
                                    membership.patch_id)];

                        for (int local_dof_id = 0;
                             local_dof_id < FluxSpaceType::local_dofs_v;
                             ++local_dof_id)
                        {
                            const double coefficient =
                                flux_function.local_coefficient(
                                    membership.patch_cell_index,
                                    local_dof_id);
                            if (coefficient == 0.0)
                                continue;

                            const auto& phi =
                                rt_qp.rt_basis_values[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                            flux_evaluation.sigma[0] +=
                                coefficient * phi[0];
                            flux_evaluation.sigma[1] +=
                                coefficient * phi[1];
                            flux_evaluation.div_sigma +=
                                coefficient *
                                rt_qp.rt_basis_divergences[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                        }
                    }

                    const auto& sigma_q = flux_evaluation.sigma;
                    const auto& grad_theta =
                        rhs_qp.grad_theta_tilde;
                    const auto& M_q = rhs_qp.diffusion_tensor;

                    double flux_term =
                        coefficients::inverse_diffusion_dot(
                            M_q,
                            sigma_q,
                            sigma_q) +
                        2.0 *
                            (sigma_q[0] * grad_theta[0] +
                             sigma_q[1] * grad_theta[1]) +
                        coefficients::diffusion_energy_dot(
                            M_q,
                            grad_theta,
                            grad_theta);
                    flux_term = detail::clamp_small_negative(flux_term);

                    const double residual =
                        rhs_qp.ell_value -
                        rhs_qp.u_time_derivative -
                        flux_evaluation.div_sigma;

                    local_flux +=
                        detail::checked_nonnegative_contribution(
                            flux_term,
                            rt_qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d unified streaming diagnostics flux");
                    local_residual +=
                        detail::checked_nonnegative_contribution(
                            residual * residual,
                            rt_qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d unified streaming diagnostics residual");
                }

                detail::add_to_map(
                    result.by_source_cell_flux,
                    rhs_cell.source_cell_id,
                    local_flux);
                detail::add_to_map(
                    result.by_source_cell_residual,
                    rhs_cell.source_cell_id,
                    local_residual);
            }
        }

        template<int QSpace, int QTime, class RTCellCache, class RHSStateCache>
        [[nodiscard]] FluxDiagnostics
        compute_fused_flux_diagnostics_from_local_error_state_(
            const RTCellCache& rt_cell_cache,
            const RHSStateCache& rhs_state_cache,
            const finite_element::detail::TimingRecorder& timing,
            int requested_threads,
            FluxDiagnosticsRuntimeStats* runtime_stats = nullptr) const
        {
            using Clock = std::chrono::steady_clock;
            const auto elapsed_seconds = [](const Clock::time_point start)
            {
                return std::chrono::duration<double>(
                           Clock::now() - start)
                    .count();
            };

            const auto total_start = Clock::now();
            const auto active_slab_cells = collect_active_slab_cell_refs_();
            const int n_cells =
                static_cast<int>(active_slab_cells.size());

            std::vector<int> source_cell_ids(
                static_cast<std::size_t>(n_cells),
                -1);
            std::vector<double> flux_contributions(
                static_cast<std::size_t>(n_cells),
                0.0);
            std::vector<double> residual_contributions(
                static_cast<std::size_t>(n_cells),
                0.0);

            const auto loop_start = Clock::now();

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
            const int n_threads = requested_threads > 1
                                      ? requested_threads
                                      : 1;
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
#endif
            for (int cell_index = 0; cell_index < n_cells; ++cell_index)
            {
                const auto cell =
                    active_slab_cells[static_cast<std::size_t>(cell_index)];
                const int membership_count =
                    patch_set_->cell_patch_count(
                        cell.slab_id,
                        cell.slab_cell_id);
                if (membership_count <= 0)
                {
                    throw std::runtime_error(
                        "TimeSlabEquilibratedFluxReconstruction2plus1d: fused flux diagnostics encountered a slab cell without vertex-patch memberships.");
                }

                const auto& rt_cell =
                    rt_cell_cache.cell(cell.slab_id, cell.slab_cell_id);
                const auto& rhs_cell =
                    rhs_state_cache.cell(cell.slab_id, cell.slab_cell_id);

                source_cell_ids[static_cast<std::size_t>(cell_index)] =
                    rhs_cell.source_cell_id;

                double local_flux = 0.0;
                double local_residual = 0.0;

                for (int qp_id = 0;
                     qp_id < RTCellCache::n_quadrature_points_v;
                     ++qp_id)
                {
                    const auto& rt_qp =
                        rt_cell.points[static_cast<std::size_t>(qp_id)];
                    const auto& rhs_qp =
                        rhs_cell.points[static_cast<std::size_t>(qp_id)];

                    FluxEvaluation flux_evaluation;
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set_->cell_patch_membership(
                                cell.slab_id,
                                cell.slab_cell_id,
                                membership_index);
                        const auto& flux_function =
                            flux_functions_[
                                static_cast<std::size_t>(
                                    membership.patch_id)];

                        for (int local_dof_id = 0;
                             local_dof_id < FluxSpaceType::local_dofs_v;
                             ++local_dof_id)
                        {
                            const double coefficient =
                                flux_function.local_coefficient(
                                    membership.patch_cell_index,
                                    local_dof_id);
                            if (coefficient == 0.0)
                                continue;

                            const auto& phi =
                                rt_qp.rt_basis_values[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                            flux_evaluation.sigma[0] +=
                                coefficient * phi[0];
                            flux_evaluation.sigma[1] +=
                                coefficient * phi[1];
                            flux_evaluation.div_sigma +=
                                coefficient *
                                rt_qp.rt_basis_divergences[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                        }
                    }

                    const auto& sigma_q = flux_evaluation.sigma;
                    const auto& grad_theta =
                        rhs_qp.grad_theta_tilde;
                    const auto& M_q = rhs_qp.diffusion_tensor;

                    double flux_term =
                        coefficients::inverse_diffusion_dot(
                            M_q,
                            sigma_q,
                            sigma_q) +
                        2.0 *
                            (sigma_q[0] * grad_theta[0] +
                             sigma_q[1] * grad_theta[1]) +
                        coefficients::diffusion_energy_dot(
                            M_q,
                            grad_theta,
                            grad_theta);
                    flux_term = detail::clamp_small_negative(flux_term);

                    const double residual =
                        rhs_qp.ell_value -
                        rhs_qp.u_time_derivative -
                        flux_evaluation.div_sigma;

                    local_flux +=
                        detail::checked_nonnegative_contribution(
                            flux_term,
                            rt_qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d fused flux diagnostics flux");
                    local_residual +=
                        detail::checked_nonnegative_contribution(
                            residual * residual,
                            rt_qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d fused flux diagnostics residual");
                }

                flux_contributions[static_cast<std::size_t>(cell_index)] =
                    local_flux;
                residual_contributions[
                    static_cast<std::size_t>(cell_index)] =
                    local_residual;
            }

            const double slab_cell_loop_seconds =
                elapsed_seconds(loop_start);

            const auto map_start = Clock::now();
            FluxDiagnostics result;
            result.by_source_cell_flux.reserve(
                static_cast<std::size_t>(n_cells));
            result.by_source_cell_residual.reserve(
                static_cast<std::size_t>(n_cells));
            for (int cell_index = 0; cell_index < n_cells; ++cell_index)
            {
                const int source_cell_id =
                    source_cell_ids[
                        static_cast<std::size_t>(cell_index)];
                detail::add_to_map(
                    result.by_source_cell_flux,
                    source_cell_id,
                    flux_contributions[
                        static_cast<std::size_t>(cell_index)]);
                detail::add_to_map(
                    result.by_source_cell_residual,
                    source_cell_id,
                    residual_contributions[
                        static_cast<std::size_t>(cell_index)]);
            }
            const double map_accumulation_seconds =
                elapsed_seconds(map_start);

            detail::require_nonnegative_cellwise_map(
                result.by_source_cell_flux,
                "TimeSlabEquilibratedFluxReconstruction2plus1d fused flux diagnostics flux aggregation");
            detail::require_nonnegative_cellwise_map(
                result.by_source_cell_residual,
                "TimeSlabEquilibratedFluxReconstruction2plus1d fused flux diagnostics residual aggregation");

            const double total_seconds = elapsed_seconds(total_start);
            timing.add("time_slab.flux_diagnostics.fused_used", 1.0);
            timing.add(
                "time_slab.flux_diagnostics.fused_total",
                total_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused",
                total_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused_slab_cell_loop",
                slab_cell_loop_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused_sigma_divergence_evaluation",
                slab_cell_loop_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused_gradient_evaluation_reused",
                0.0);
            timing.add(
                "time_slab.flux_diagnostics.fused_diffusion_tensor_evaluation_reused",
                0.0);
            timing.add(
                "time_slab.flux_diagnostics.fused_map_accumulation",
                map_accumulation_seconds);

            if (runtime_stats != nullptr)
            {
                runtime_stats->seconds = total_seconds;
                runtime_stats->qpoints =
                    static_cast<double>(n_cells) *
                    static_cast<double>(RTCellCache::n_quadrature_points_v);
                runtime_stats->reused_cell_state =
                    static_cast<double>(n_cells);
                runtime_stats->extra_cell_state_rebuilds = 0.0;
            }

            return result;
        }

        template<class QpointStateCache>
        void accumulate_fused_flux_diagnostics_from_qpoint_state_cells_(
            const std::vector<ActiveSlabCellRef>& active_slab_cells,
            const QpointStateCache& qpoint_state_cache,
            FluxDiagnostics& result) const
        {
            for (const auto& cell : active_slab_cells)
            {
                const int membership_count =
                    patch_set_->cell_patch_count(
                        cell.slab_id,
                        cell.slab_cell_id);
                if (membership_count <= 0)
                {
                    throw std::runtime_error(
                        "TimeSlabEquilibratedFluxReconstruction2plus1d: qpoint diagnostics encountered a slab cell without vertex-patch memberships.");
                }

                const auto& state_cell =
                    qpoint_state_cache.cell(cell.slab_id, cell.slab_cell_id);

                double local_flux = 0.0;
                double local_residual = 0.0;

                for (int qp_id = 0;
                     qp_id < QpointStateCache::n_quadrature_points_v;
                     ++qp_id)
                {
                    const auto& qp =
                        state_cell.points[static_cast<std::size_t>(qp_id)];

                    FluxEvaluation flux_evaluation;
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set_->cell_patch_membership(
                                cell.slab_id,
                                cell.slab_cell_id,
                                membership_index);
                        const auto& flux_function =
                            flux_functions_[
                                static_cast<std::size_t>(
                                    membership.patch_id)];

                        for (int local_dof_id = 0;
                             local_dof_id < FluxSpaceType::local_dofs_v;
                             ++local_dof_id)
                        {
                            const double coefficient =
                                flux_function.local_coefficient(
                                    membership.patch_cell_index,
                                    local_dof_id);
                            if (coefficient == 0.0)
                                continue;

                            const auto& phi =
                                qp.rt_basis_values[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                            flux_evaluation.sigma[0] +=
                                coefficient * phi[0];
                            flux_evaluation.sigma[1] +=
                                coefficient * phi[1];
                            flux_evaluation.div_sigma +=
                                coefficient *
                                qp.rt_basis_divergences[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                        }
                    }

                    const auto& sigma_q = flux_evaluation.sigma;
                    const auto& grad_theta = qp.grad_theta_tilde;
                    const auto& M_q = qp.diffusion_tensor;

                    double flux_term =
                        coefficients::inverse_diffusion_dot(
                            M_q,
                            sigma_q,
                            sigma_q) +
                        2.0 *
                            (sigma_q[0] * grad_theta[0] +
                             sigma_q[1] * grad_theta[1]) +
                        coefficients::diffusion_energy_dot(
                            M_q,
                            grad_theta,
                            grad_theta);
                    flux_term = detail::clamp_small_negative(flux_term);

                    const double residual =
                        qp.ell_value -
                        qp.u_time_derivative -
                        flux_evaluation.div_sigma;

                    local_flux +=
                        detail::checked_nonnegative_contribution(
                            flux_term,
                            qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d qpoint diagnostics flux");
                    local_residual +=
                        detail::checked_nonnegative_contribution(
                            residual * residual,
                            qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d qpoint diagnostics residual");
                }

                detail::add_to_map(
                    result.by_source_cell_flux,
                    state_cell.source_cell_id,
                    local_flux);
                detail::add_to_map(
                    result.by_source_cell_residual,
                    state_cell.source_cell_id,
                    local_residual);
            }
        }

        template<class QpointStateCache>
        [[nodiscard]] std::pair<double, double>
        compute_streaming_flux_diagnostics_contribution_from_compact_state_(
            const typename QpointStateCache::FluxDiagnosticCellState2D&
                state_cell,
            double& qpoint_eval_seconds) const
        {
            using OperatorDiffusionMode =
                typename QpointStateCache::OperatorDiffusionMode;
            if (state_cell.diffusion_mode == OperatorDiffusionMode::variable)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: compact streaming flux diagnostics require identity or constant diffusion.");
            }

            const int membership_count =
                patch_set_->cell_patch_count(
                    state_cell.slab_id,
                    state_cell.slab_cell_id);
            if (membership_count <= 0)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: compact streaming diagnostics encountered a slab cell without vertex-patch memberships.");
            }

            const auto& first_membership =
                patch_set_->cell_patch_membership(
                    state_cell.slab_id,
                    state_cell.slab_cell_id,
                    0);
            const auto& first_flux_space =
                flux_spaces_[static_cast<std::size_t>(
                    first_membership.patch_id)];
            const auto cell_map =
                first_flux_space.physical_map_for_patch_cell(
                    first_membership.patch_cell_index);

            const auto eval_start = std::chrono::steady_clock::now();
            double local_flux = 0.0;
            double local_residual = 0.0;
            const auto& reference_tables = QpointStateCache::reference_tables();

            for (int qp_id = 0;
                 qp_id < QpointStateCache::n_quadrature_points_v;
                 ++qp_id)
            {
                const auto& qp =
                    state_cell.points[static_cast<std::size_t>(qp_id)];

                FluxEvaluation flux_evaluation;
                const int spatial_qp_id =
                    reference_tables.spatial_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const int time_qp_id =
                    reference_tables.time_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const auto& x_ref =
                    reference_tables.spatial_reference_points[
                        static_cast<std::size_t>(spatial_qp_id)];
                const double t_ref =
                    reference_tables.time_reference_points[
                        static_cast<std::size_t>(time_qp_id)];
                const auto spatial_values =
                    FluxSpaceType::PiolaBasis::eval_all(cell_map, x_ref);
                const auto spatial_divergences =
                    FluxSpaceType::PiolaBasis::div_all(cell_map, x_ref);
                typename FluxSpaceType::TimeValues time_values{};
                FluxSpaceType::evaluate_time_basis(t_ref, time_values);
                typename FluxSpaceType::LocalValues rt_values{};
                typename FluxSpaceType::LocalDivergences rt_divergences{};
                for (int spatial_local_dof = 0;
                     spatial_local_dof <
                         FluxSpaceType::spatial_local_dofs_v;
                     ++spatial_local_dof)
                {
                    const auto& spatial_value =
                        spatial_values[
                            static_cast<std::size_t>(spatial_local_dof)];
                    const double spatial_divergence =
                        spatial_divergences[
                            static_cast<std::size_t>(spatial_local_dof)];
                    for (int time_dof = 0;
                         time_dof < FluxSpaceType::n_time_dofs_v;
                         ++time_dof)
                    {
                        const int local_id =
                            spatial_local_dof *
                                FluxSpaceType::n_time_dofs_v +
                            time_dof;
                        const double time_value =
                            time_values[static_cast<std::size_t>(
                                time_dof)];
                        rt_values[static_cast<std::size_t>(local_id)] =
                            typename FluxSpaceType::VectorValue{
                                spatial_value[0] * time_value,
                                spatial_value[1] * time_value};
                        rt_divergences[
                            static_cast<std::size_t>(local_id)] =
                            spatial_divergence * time_value;
                    }
                }
                for (int membership_index = 0;
                     membership_index < membership_count;
                     ++membership_index)
                {
                    const auto& membership =
                        patch_set_->cell_patch_membership(
                            state_cell.slab_id,
                            state_cell.slab_cell_id,
                            membership_index);
                    const int patch_id = membership.patch_id;
                    const auto& flux_function =
                        flux_functions_[static_cast<std::size_t>(patch_id)];

                    for (int local_dof_id = 0;
                         local_dof_id < FluxSpaceType::local_dofs_v;
                         ++local_dof_id)
                    {
                        const double coefficient =
                            flux_function.local_coefficient(
                                membership.patch_cell_index,
                                local_dof_id);
                        if (coefficient == 0.0)
                            continue;

                        const auto& phi =
                            rt_values[static_cast<std::size_t>(
                                local_dof_id)];
                        flux_evaluation.sigma[0] += coefficient * phi[0];
                        flux_evaluation.sigma[1] += coefficient * phi[1];
                        flux_evaluation.div_sigma +=
                            coefficient *
                            rt_divergences[static_cast<std::size_t>(
                                local_dof_id)];
                    }
                }

                const auto& sigma_q = flux_evaluation.sigma;
                const auto& grad_theta = qp.grad_theta_tilde;
                const coefficients::DiffusionVector<2> grad_theta_vec{
                    grad_theta[0],
                    grad_theta[1]};
                double flux_term =
                    detail::flux_mismatch_energy<2>(
                        state_cell.diffusion_tensor,
                        sigma_q,
                        grad_theta_vec);
                flux_term = detail::clamp_small_negative(flux_term);

                const double residual =
                    qp.ell_value -
                    qp.u_time_derivative -
                    flux_evaluation.div_sigma;

                local_flux +=
                    detail::checked_nonnegative_contribution(
                        flux_term,
                        qp.jacobian_weight,
                        "TimeSlabEquilibratedFluxReconstruction2plus1d compact streaming diagnostics flux");
                local_residual +=
                    detail::checked_nonnegative_contribution(
                        residual * residual,
                        qp.jacobian_weight,
                        "TimeSlabEquilibratedFluxReconstruction2plus1d compact streaming diagnostics residual");
            }
            qpoint_eval_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - eval_start)
                    .count();

            return {local_flux, local_residual};
        }

        template<int QSpace, int QTime, class QpointStateCache>
        [[nodiscard]] FluxDiagnostics
        compute_fused_flux_diagnostics_from_qpoint_state_(
            const QpointStateCache& qpoint_state_cache,
            const finite_element::detail::TimingRecorder& timing,
            int requested_threads,
            FluxDiagnosticsRuntimeStats* runtime_stats = nullptr) const
        {
            static_cast<void>(QSpace);
            static_cast<void>(QTime);
            using Clock = std::chrono::steady_clock;
            const auto elapsed_seconds = [](const Clock::time_point start)
            {
                return std::chrono::duration<double>(
                           Clock::now() - start)
                    .count();
            };

            const auto total_start = Clock::now();
            const auto active_slab_cells = collect_active_slab_cell_refs_();
            const int n_cells =
                static_cast<int>(active_slab_cells.size());

            std::vector<int> source_cell_ids(
                static_cast<std::size_t>(n_cells),
                -1);
            std::vector<double> flux_contributions(
                static_cast<std::size_t>(n_cells),
                0.0);
            std::vector<double> residual_contributions(
                static_cast<std::size_t>(n_cells),
                0.0);

            const auto loop_start = Clock::now();

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
            const int n_threads = requested_threads > 1
                                      ? requested_threads
                                      : 1;
#pragma omp parallel for schedule(static) if(n_threads > 1) num_threads(n_threads)
#endif
            for (int cell_index = 0; cell_index < n_cells; ++cell_index)
            {
                const auto cell =
                    active_slab_cells[static_cast<std::size_t>(cell_index)];
                const int membership_count =
                    patch_set_->cell_patch_count(
                        cell.slab_id,
                        cell.slab_cell_id);
                if (membership_count <= 0)
                {
                    throw std::runtime_error(
                        "TimeSlabEquilibratedFluxReconstruction2plus1d: qpoint fused flux diagnostics encountered a slab cell without vertex-patch memberships.");
                }

                const auto& state_cell =
                    qpoint_state_cache.cell(cell.slab_id, cell.slab_cell_id);

                source_cell_ids[static_cast<std::size_t>(cell_index)] =
                    state_cell.source_cell_id;

                double local_flux = 0.0;
                double local_residual = 0.0;

                for (int qp_id = 0;
                     qp_id < QpointStateCache::n_quadrature_points_v;
                     ++qp_id)
                {
                    const auto& qp =
                        state_cell.points[static_cast<std::size_t>(qp_id)];

                    FluxEvaluation flux_evaluation;
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set_->cell_patch_membership(
                                cell.slab_id,
                                cell.slab_cell_id,
                                membership_index);
                        const auto& flux_function =
                            flux_functions_[
                                static_cast<std::size_t>(
                                    membership.patch_id)];

                        for (int local_dof_id = 0;
                             local_dof_id < FluxSpaceType::local_dofs_v;
                             ++local_dof_id)
                        {
                            const double coefficient =
                                flux_function.local_coefficient(
                                    membership.patch_cell_index,
                                    local_dof_id);
                            if (coefficient == 0.0)
                                continue;

                            const auto& phi =
                                qp.rt_basis_values[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                            flux_evaluation.sigma[0] +=
                                coefficient * phi[0];
                            flux_evaluation.sigma[1] +=
                                coefficient * phi[1];
                            flux_evaluation.div_sigma +=
                                coefficient *
                                qp.rt_basis_divergences[
                                    static_cast<std::size_t>(
                                        local_dof_id)];
                        }
                    }

                    const auto& sigma_q = flux_evaluation.sigma;
                    const auto& grad_theta = qp.grad_theta_tilde;
                    const auto& M_q = qp.diffusion_tensor;

                    double flux_term =
                        coefficients::inverse_diffusion_dot(
                            M_q,
                            sigma_q,
                            sigma_q) +
                        2.0 *
                            (sigma_q[0] * grad_theta[0] +
                             sigma_q[1] * grad_theta[1]) +
                        coefficients::diffusion_energy_dot(
                            M_q,
                            grad_theta,
                            grad_theta);
                    flux_term = detail::clamp_small_negative(flux_term);

                    const double residual =
                        qp.ell_value -
                        qp.u_time_derivative -
                        flux_evaluation.div_sigma;

                    local_flux +=
                        detail::checked_nonnegative_contribution(
                            flux_term,
                            qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d qpoint fused flux diagnostics flux");
                    local_residual +=
                        detail::checked_nonnegative_contribution(
                            residual * residual,
                            qp.jacobian_weight,
                            "TimeSlabEquilibratedFluxReconstruction2plus1d qpoint fused flux diagnostics residual");
                }

                flux_contributions[static_cast<std::size_t>(cell_index)] =
                    local_flux;
                residual_contributions[
                    static_cast<std::size_t>(cell_index)] =
                    local_residual;
            }

            const double slab_cell_loop_seconds =
                elapsed_seconds(loop_start);

            const auto map_start = Clock::now();
            FluxDiagnostics result;
            result.by_source_cell_flux.reserve(
                static_cast<std::size_t>(n_cells));
            result.by_source_cell_residual.reserve(
                static_cast<std::size_t>(n_cells));
            for (int cell_index = 0; cell_index < n_cells; ++cell_index)
            {
                const int source_cell_id =
                    source_cell_ids[
                        static_cast<std::size_t>(cell_index)];
                detail::add_to_map(
                    result.by_source_cell_flux,
                    source_cell_id,
                    flux_contributions[
                        static_cast<std::size_t>(cell_index)]);
                detail::add_to_map(
                    result.by_source_cell_residual,
                    source_cell_id,
                    residual_contributions[
                        static_cast<std::size_t>(cell_index)]);
            }
            const double map_accumulation_seconds =
                elapsed_seconds(map_start);

            detail::require_nonnegative_cellwise_map(
                result.by_source_cell_flux,
                "TimeSlabEquilibratedFluxReconstruction2plus1d qpoint fused flux diagnostics flux aggregation");
            detail::require_nonnegative_cellwise_map(
                result.by_source_cell_residual,
                "TimeSlabEquilibratedFluxReconstruction2plus1d qpoint fused flux diagnostics residual aggregation");

            const double total_seconds = elapsed_seconds(total_start);
            timing.add("time_slab.flux_diagnostics.fused_used", 1.0);
            timing.add(
                "time_slab.flux_diagnostics.fused_total",
                total_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused",
                total_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused_slab_cell_loop",
                slab_cell_loop_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused_sigma_divergence_evaluation",
                slab_cell_loop_seconds);
            timing.add(
                "time_slab.flux_diagnostics.fused_gradient_evaluation_reused",
                0.0);
            timing.add(
                "time_slab.flux_diagnostics.fused_diffusion_tensor_evaluation_reused",
                0.0);
            timing.add(
                "time_slab.flux_diagnostics.fused_map_accumulation",
                map_accumulation_seconds);

            if (runtime_stats != nullptr)
            {
                runtime_stats->seconds = total_seconds;
                runtime_stats->qpoints =
                    static_cast<double>(n_cells) *
                    static_cast<double>(
                        QpointStateCache::n_quadrature_points_v);
                runtime_stats->reused_cell_state =
                    static_cast<double>(n_cells);
                runtime_stats->extra_cell_state_rebuilds = 0.0;
            }

            return result;
        }

        [[nodiscard]] std::vector<std::vector<int>>
        build_cell_color_classes_(
            const std::vector<ActiveSlabCellRef>& active_slab_cells) const
        {
            const int patch_count = n_patches();
            std::vector<std::vector<int>> cell_color_classes;
            std::vector<int> patch_color_marker(
                static_cast<std::size_t>(patch_count),
                -1);

            for (int cell_index = 0;
                 cell_index < static_cast<int>(active_slab_cells.size());
                 ++cell_index)
            {
                const auto cell =
                    active_slab_cells[static_cast<std::size_t>(cell_index)];
                const int membership_count =
                    patch_set_->cell_patch_count(
                        cell.slab_id,
                        cell.slab_cell_id);
                int color = 0;
                for (;; ++color)
                {
                    bool conflict = false;
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set_->cell_patch_membership(
                                cell.slab_id,
                                cell.slab_cell_id,
                                membership_index);
                        if (patch_color_marker[
                                static_cast<std::size_t>(
                                    membership.patch_id)] == color)
                        {
                            conflict = true;
                            break;
                        }
                    }

                    if (!conflict)
                        break;
                }

                if (color >= static_cast<int>(cell_color_classes.size()))
                {
                    cell_color_classes.resize(
                        static_cast<std::size_t>(color + 1));
                }
                cell_color_classes[static_cast<std::size_t>(color)]
                    .push_back(cell_index);

                for (int membership_index = 0;
                     membership_index < membership_count;
                     ++membership_index)
                {
                    const auto& membership =
                        patch_set_->cell_patch_membership(
                            cell.slab_id,
                            cell.slab_cell_id,
                            membership_index);
                    patch_color_marker[
                        static_cast<std::size_t>(membership.patch_id)] =
                        color;
                }
            }

            return cell_color_classes;
        }

        [[nodiscard]] static std::size_t dense_matrix_bytes_(
            const DenseMatrixType& matrix) noexcept
        {
            return matrix.estimated_memory_bytes();
        }

        [[nodiscard]] static std::size_t dense_vector_bytes_(
            const VectorType& vector) noexcept
        {
            return static_cast<std::size_t>(vector.size()) * sizeof(double);
        }

        [[nodiscard]] std::size_t
        estimate_scalar_reduction_basis_bytes_() const noexcept
        {
            return 0;
        }

        [[nodiscard]] std::size_t scalar_reduction_basis_peak_bytes_()
            const noexcept
        {
            if (!false)
                return 0;

            std::size_t bytes = 0;
            for (const auto& scalar : scalar_spaces_)
            {
                if (!scalar.has_mean_zero_constraint())
                    continue;

                const int n_u = scalar.n_dofs();
                const int n_u_reduced =
                    finite_element::assembly::error_system::
                        reduced_scalar_dimension_2d(scalar, n_u);
                bytes =
                    std::max(
                        bytes,
                        static_cast<std::size_t>(n_u) *
                            static_cast<std::size_t>(n_u_reduced) *
                            sizeof(double));
            }
            return bytes;
        }

        [[nodiscard]] std::size_t
        estimate_concurrent_scalar_reduction_basis_bytes_(int n_threads)
            const noexcept
        {
            return static_cast<std::size_t>(std::max(1, n_threads)) *
                   scalar_reduction_basis_peak_bytes_();
        }

        [[nodiscard]] std::size_t operator_data_matrix_bytes_(
            const typename LocalOperatorCache::OperatorData& data)
            const noexcept
        {
            return dense_matrix_bytes_(data.A) +
                   dense_matrix_bytes_(data.B) +
                   dense_matrix_bytes_(data.C) +
                   dense_matrix_bytes_(data.explicit_constraint_matrix);
        }

        [[nodiscard]] std::size_t operator_data_factor_bytes_(
            const typename LocalOperatorCache::OperatorData& data)
            const noexcept
        {
            if (!data.has_explicit_constraint_factorization())
                return 0;

            const auto n =
                static_cast<std::size_t>(data.explicit_system_size);
            return n * n * sizeof(double) + n * sizeof(int);
        }

        [[nodiscard]] std::size_t operator_cache_matrix_bytes_()
            const noexcept
        {
            if (local_operator_cache_ == nullptr)
                return 0;

            std::size_t bytes =
                local_operator_cache_->operators.size() *
                (sizeof(typename LocalOperatorCache::OperatorData) +
                 sizeof(std::size_t) + 3 * sizeof(void*));
            for (const auto& entry : local_operator_cache_->operators)
                bytes += operator_data_matrix_bytes_(entry.second);
            return bytes;
        }

        [[nodiscard]] std::size_t factor_cache_bytes_() const noexcept
        {
            if (local_operator_cache_ == nullptr)
                return 0;

            std::size_t bytes = 0;
            for (const auto& entry : local_operator_cache_->operators)
                bytes += operator_data_factor_bytes_(entry.second);
            return bytes;
        }

        template<class DenseBlockVector>
        [[nodiscard]] static std::size_t dense_blocks_bytes_(
            const DenseBlockVector& dense_blocks) noexcept
        {
            std::size_t bytes =
                dense_blocks.capacity() *
                sizeof(typename DenseBlockVector::value_type);
            for (const auto& block : dense_blocks)
                bytes += block.estimated_memory_bytes();
            return bytes;
        }

        template<class DenseBlockVector>
        [[nodiscard]] static std::size_t dense_block_peak_patch_bytes_(
            const DenseBlockVector& dense_blocks) noexcept
        {
            std::size_t bytes = 0;
            for (const auto& block : dense_blocks)
            {
                bytes =
                    std::max(bytes, block.estimated_memory_bytes());
            }
            return bytes;
        }

        template<class TablesVector>
        [[nodiscard]] static std::size_t optional_tables_bytes_(
            const TablesVector& tables) noexcept
        {
            std::size_t bytes =
                tables.capacity() * sizeof(typename TablesVector::value_type);
            for (const auto& table : tables)
            {
                if (table.has_value())
                    bytes += table->estimated_memory_bytes();
            }
            return bytes;
        }

        template<class Tables>
        [[nodiscard]] std::size_t estimate_tables_peak_bytes_from_spaces_()
            const noexcept
        {
            std::size_t bytes =
                flux_spaces_.capacity() * sizeof(Tables);
            for (const auto& flux_space : flux_spaces_)
            {
                bytes += static_cast<std::size_t>(
                             flux_space.n_patch_cells()) *
                         sizeof(typename Tables::CellData);
            }
            return bytes;
        }

        template<class SolutionVector>
        [[nodiscard]] static std::size_t patch_solutions_bytes_(
            const SolutionVector& patch_solutions) noexcept
        {
            std::size_t bytes =
                patch_solutions.capacity() *
                sizeof(typename SolutionVector::value_type);
            for (const auto& solution : patch_solutions)
            {
                bytes +=
                    static_cast<std::size_t>(solution.lambda.size()) *
                    sizeof(double);
                bytes +=
                    static_cast<std::size_t>(solution.u.size()) *
                    sizeof(double);
            }
            return bytes;
        }

        [[nodiscard]] std::size_t sequential_patch_solution_peak_bytes_()
            const noexcept
        {
            std::size_t bytes = 0;
            const int patch_count = n_patches();
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                const auto& flux =
                    flux_spaces_[static_cast<std::size_t>(patch_id)];
                const auto& scalar =
                    scalar_spaces_[static_cast<std::size_t>(patch_id)];
                bytes = std::max(
                    bytes,
                    (static_cast<std::size_t>(flux.n_dofs()) +
                     static_cast<std::size_t>(scalar.n_dofs())) *
                        sizeof(double));
            }
            return bytes;
        }

        [[nodiscard]] std::size_t estimate_per_thread_context_bytes_(
            int n_threads) const noexcept
        {
            if (n_threads <= 0)
                return 0;

            return static_cast<std::size_t>(n_threads) *
                   (sizeof(finite_element::detail::CellGeometryCache<XSpace>) +
                    sizeof(finite_element::assembly::detail::
                               SourceActiveAncestorCache<XSpace>) +
                    static_cast<std::size_t>(slab_space_ref().n_slabs()) *
                        sizeof(finite_element::detail::CellGeometryCache<
                               LocalSlabSpaceType>));
        }

        [[nodiscard]] int patch_system_size_(int patch_id) const
        {
            const auto& flux =
                flux_spaces_[static_cast<std::size_t>(patch_id)];
            const auto& scalar =
                scalar_spaces_[static_cast<std::size_t>(patch_id)];
            const int n_lambda = flux.n_dofs();
            const int n_u = scalar.n_dofs();
            if (false &&
                scalar.has_mean_zero_constraint())
            {
                return n_lambda +
                       finite_element::assembly::error_system::
                           reduced_scalar_dimension_2d(scalar, n_u);
            }

            const int n_constraints =
                scalar.has_mean_zero_constraint()
                    ? scalar.n_mean_zero_constraints()
                    : 0;
            return n_lambda + n_u + n_constraints;
        }

        [[nodiscard]] int patch_explicit_constraint_count_(
            int patch_id) const
        {
            const auto& scalar =
                scalar_spaces_[static_cast<std::size_t>(patch_id)];
            return scalar.has_mean_zero_constraint()
                       ? scalar.n_mean_zero_constraints()
                       : 0;
        }

        struct PatchSolveBatchStats
        {
            double patch_solve_groups = 0.0;
            double largest_group_size = 0.0;
            double batched_patch_systems = 0.0;
            double workspace_patch_systems = 0.0;
            double fallback_patch_systems = 0.0;
            double factorization_seconds = 0.0;
            double solve_apply_seconds = 0.0;
            double workspace_allocation_seconds = 0.0;
            std::size_t dense_solver_workspace_bytes = 0;
        };

        struct PatchSolveGroup
        {
            int system_size = 0;
            int constraint_count = 0;
            int patch_type = 0;
            int solver_mode = 0;
            std::vector<int> local_patch_ids{};
        };

        [[nodiscard]] std::vector<PatchSolveGroup>
        patch_solve_groups_(
            int patch_begin,
            int patch_end) const
        {
            std::vector<PatchSolveGroup> groups;
            const int patch_count = std::max(0, patch_end - patch_begin);
            if (patch_count == 0)
                return groups;

            groups.reserve(static_cast<std::size_t>(patch_count));
            for (int patch_id = patch_begin; patch_id < patch_end; ++patch_id)
            {
                const int system_size = patch_system_size_(patch_id);
                const int constraint_count =
                    patch_explicit_constraint_count_(patch_id);
                const int patch_type = constraint_count == 0 ? 0 : 1;
                const int solver_mode = 0;

                auto it =
                    std::find_if(
                        groups.begin(),
                        groups.end(),
                        [&](const PatchSolveGroup& group)
                        {
                            return group.system_size == system_size &&
                                   group.constraint_count ==
                                       constraint_count &&
                                   group.patch_type == patch_type &&
                                   group.solver_mode == solver_mode;
                        });
                if (it == groups.end())
                {
                    PatchSolveGroup group;
                    group.system_size = system_size;
                    group.constraint_count = constraint_count;
                    group.patch_type = patch_type;
                    group.solver_mode = solver_mode;
                    group.local_patch_ids.push_back(patch_id - patch_begin);
                    groups.push_back(std::move(group));
                }
                else
                {
                    it->local_patch_ids.push_back(patch_id - patch_begin);
                }
            }

            std::sort(
                groups.begin(),
                groups.end(),
                [](const PatchSolveGroup& a, const PatchSolveGroup& b)
                {
                    if (a.system_size != b.system_size)
                        return a.system_size < b.system_size;
                    if (a.constraint_count != b.constraint_count)
                        return a.constraint_count < b.constraint_count;
                    if (a.patch_type != b.patch_type)
                        return a.patch_type < b.patch_type;
                    return a.solver_mode < b.solver_mode;
                });

            return groups;
        }

        [[nodiscard]] std::vector<int> patch_solve_order_(
            int patch_begin,
            int patch_end,
            int selected_threads) const
        {
            const int patch_count = std::max(0, patch_end - patch_begin);
            std::vector<int> order;
            order.reserve(static_cast<std::size_t>(patch_count));

            if (!false ||
                false ||
                local_operator_cache_ != nullptr)
            {
                for (int local_patch_id = 0;
                     local_patch_id < patch_count;
                     ++local_patch_id)
                {
                    order.push_back(local_patch_id);
                }
                return order;
            }

            const int n_ranges =
                std::max(1, std::min(selected_threads, patch_count));
            for (int range_id = 0; range_id < n_ranges; ++range_id)
            {
                const int local_begin =
                    static_cast<int>(
                        (static_cast<long long>(range_id) * patch_count) /
                        n_ranges);
                const int local_end =
                    static_cast<int>(
                        (static_cast<long long>(range_id + 1) *
                         patch_count) /
                        n_ranges);
                const auto groups =
                    patch_solve_groups_(
                        patch_begin + local_begin,
                        patch_begin + local_end);
                for (const auto& group : groups)
                {
                    for (const int local_patch_id :
                         group.local_patch_ids)
                    {
                        order.push_back(local_begin + local_patch_id);
                    }
                }
            }
            return order;
        }

        [[nodiscard]] PatchSolveBatchStats patch_solve_batch_stats_(
            int patch_begin,
            int patch_end,
            int selected_threads) const
        {
            PatchSolveBatchStats stats;
            const int patch_count = std::max(0, patch_end - patch_begin);
            if (patch_count == 0)
                return stats;

            if (!use_local_error_patch_solve_workspace_() ||
                local_operator_cache_ != nullptr)
            {
                stats.fallback_patch_systems =
                    static_cast<double>(patch_count);
                return stats;
            }

            const auto groups =
                patch_solve_groups_(patch_begin, patch_end);
            int max_system_size = 0;
            stats.patch_solve_groups =
                static_cast<double>(groups.size());
            for (const auto& group : groups)
            {
                max_system_size =
                    std::max(max_system_size, group.system_size);
                stats.largest_group_size =
                    std::max(
                        stats.largest_group_size,
                        static_cast<double>(
                            group.local_patch_ids.size()));
            }
            stats.batched_patch_systems = static_cast<double>(patch_count);
            stats.workspace_patch_systems = static_cast<double>(patch_count);

            const auto n =
                static_cast<std::size_t>(std::max(0, max_system_size));
            const auto per_workspace_bytes =
                (2 * n * n + 3 * n) * sizeof(double);
            stats.dense_solver_workspace_bytes =
                static_cast<std::size_t>(std::max(1, selected_threads)) *
                per_workspace_bytes;
            return stats;
        }

        static void add_patch_solve_batch_stats_(
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& timing_stats,
            const PatchSolveBatchStats& stats) noexcept
        {
            timing_stats.patch_solve_groups += stats.patch_solve_groups;
            timing_stats.patch_solve_largest_group_size =
                std::max(
                    timing_stats.patch_solve_largest_group_size,
                    stats.largest_group_size);
            timing_stats.batched_patch_systems +=
                stats.batched_patch_systems;
            timing_stats.workspace_patch_systems +=
                stats.workspace_patch_systems;
            timing_stats.fallback_patch_systems +=
                stats.fallback_patch_systems;
            timing_stats.patch_solve_factorization_seconds +=
                stats.factorization_seconds;
            timing_stats.patch_solve_apply_seconds +=
                stats.solve_apply_seconds;
            timing_stats.patch_solve_workspace_allocation_seconds +=
                stats.workspace_allocation_seconds;
            timing_stats.dense_solver_workspace_bytes =
                std::max(
                    timing_stats.dense_solver_workspace_bytes,
                    static_cast<double>(
                        stats.dense_solver_workspace_bytes));
        }

        [[nodiscard]] bool use_local_error_patch_solve_workspace_()
            const noexcept
        {
            // Reusing dense local solve workspaces avoids repeated allocation,
            // but the p=1 profiles showed it is not profitable for small patch
            // systems. Keep the optimized path to the high-order cases that
            // currently dominate local-error solve time.
            return local_error_reuse_patch_solve_workspace_ &&
                   FluxSpaceType::p_space_v >= 3;
        }

        [[nodiscard]] double patch_solver_mode_code_() const noexcept
        {
            if (local_error_patch_solver_ == "reduced_scalar_dense")
                return 1.0;
            if (local_error_patch_solver_ == "auto")
                return 2.0;
            return 0.0;
        }

        template<class PatchScalarSpaceType>
        [[nodiscard]] static int current_dense_patch_system_dimension_(
            const finite_element::assembly::error_system::
                DenseLocalErrorBlocks& blocks,
            const PatchScalarSpaceType& scalar_space) noexcept
        {
            return blocks.n_lambda + blocks.n_u +
                   (scalar_space.has_mean_zero_constraint()
                        ? scalar_space.n_mean_zero_constraints()
                        : 0);
        }

        template<class PatchScalarSpaceType>
        [[nodiscard]] static int reduced_scalar_patch_system_dimension_(
            const finite_element::assembly::error_system::
                DenseLocalErrorBlocks& blocks,
            const PatchScalarSpaceType& scalar_space) noexcept
        {
            return blocks.n_lambda +
                   finite_element::assembly::error_system::
                       reduced_scalar_dimension_2d(
                           scalar_space,
                           blocks.n_u);
        }

        template<class PatchScalarSpaceType>
        [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
        solve_dense_local_error_patch_with_selected_solver_(
            finite_element::assembly::error_system::DenseLocalErrorBlocks&
                blocks,
            const PatchScalarSpaceType& scalar_space,
            finite_element::assembly::error_system::
                DenseLocalErrorExplicitSolveWorkspace2D<Backend>* workspace,
            const typename LocalOperatorCache::OperatorData*
                factor_cache_entry,
            double zero_tol,
            double* transform_seconds,
            double* factorization_seconds,
            double* solve_apply_seconds,
            double* current_dense_count,
            double* reduced_scalar_dense_count,
            double* reduced_fallback_count,
            double* reduced_residual_fail_count,
            double* current_dimension_sum,
            double* current_dimension_count,
            double* reduced_dimension_sum,
            double* reduced_dimension_count) const
        {
            const int current_dimension =
                current_dense_patch_system_dimension_(blocks, scalar_space);
            const int reduced_dimension =
                reduced_scalar_patch_system_dimension_(blocks, scalar_space);
            const bool requested_reduced =
                local_error_patch_solver_ == "reduced_scalar_dense";
            const bool reduced_supported =
                scalar_space.has_mean_zero_constraint() &&
                reduced_dimension > 0 &&
                reduced_dimension < current_dimension;

            if (requested_reduced && reduced_supported)
            {
                if (reduced_scalar_dense_count != nullptr)
                    *reduced_scalar_dense_count += 1.0;
                if (reduced_dimension_sum != nullptr)
                    *reduced_dimension_sum +=
                        static_cast<double>(reduced_dimension);
                if (reduced_dimension_count != nullptr)
                    *reduced_dimension_count += 1.0;

                try
                {
                    return finite_element::assembly::error_system::
                        solve_dense_local_error_blocks_with_reduced_scalar_basis_2d<
                            Backend>(
                            blocks,
                            scalar_space,
                            nullptr,
                            zero_tol,
                            transform_seconds,
                            factorization_seconds,
                            solve_apply_seconds);
                }
                catch (const std::exception&)
                {
                    if (reduced_residual_fail_count != nullptr)
                        *reduced_residual_fail_count += 1.0;
                    if (reduced_fallback_count != nullptr)
                        *reduced_fallback_count += 1.0;
                }
            }
            else if (requested_reduced)
            {
                if (reduced_fallback_count != nullptr)
                    *reduced_fallback_count += 1.0;
            }

            if (current_dense_count != nullptr)
                *current_dense_count += 1.0;
            if (current_dimension_sum != nullptr)
                *current_dimension_sum += static_cast<double>(current_dimension);
            if (current_dimension_count != nullptr)
                *current_dimension_count += 1.0;

            if (factor_cache_entry != nullptr)
            {
                return solve_dense_local_error_blocks_with_cached_factor_(
                    blocks,
                    scalar_space,
                    factor_cache_entry,
                    zero_tol);
            }
            if (workspace != nullptr && use_local_error_patch_solve_workspace_())
            {
                return finite_element::assembly::error_system::
                    solve_dense_local_error_blocks_with_scalar_constraints_workspace_2d<
                        Backend>(
                        blocks,
                        scalar_space,
                        *workspace,
                        zero_tol,
                        factorization_seconds,
                        solve_apply_seconds);
            }
            return finite_element::assembly::error_system::
                solve_dense_local_error_blocks_with_scalar_constraints_2d<
                    Backend>(
                    blocks,
                    scalar_space,
                    zero_tol,
                    factorization_seconds,
                    solve_apply_seconds);
        }

        [[nodiscard]] static constexpr int
        default_streaming_patch_tile_size_() noexcept
        {
            if constexpr (FETraits::p_space_v >= 3)
                return 256;
            else
                return 512;
        }

        [[nodiscard]] static constexpr int
        default_streaming_cell_chunk_size_() noexcept
        {
            if constexpr (FETraits::p_space_v >= 3)
                return 256;
            else
                return 512;
        }

        [[nodiscard]] int effective_streaming_patch_tile_size_() const noexcept
        {
            return local_error_patch_tile_size_ > 0
                       ? local_error_patch_tile_size_
                       : default_streaming_patch_tile_size_();
        }

        [[nodiscard]] int effective_streaming_cell_chunk_size_() const noexcept
        {
            return local_error_cell_chunk_size_ > 0
                       ? local_error_cell_chunk_size_
                       : default_streaming_cell_chunk_size_();
        }

        [[nodiscard]] std::size_t dense_solve_peak_temporary_bytes_()
            const noexcept
        {
            std::size_t bytes = 0;
            const int patch_count = n_patches();
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                const auto system_size =
                    static_cast<std::size_t>(patch_system_size_(patch_id));
                const std::size_t patch_bytes =
                    2 * system_size * system_size * sizeof(double) +
                    3 * system_size * sizeof(double);
                bytes = std::max(bytes, patch_bytes);
            }
            return bytes;
        }

        [[nodiscard]] std::size_t dense_block_bytes_for_patch_(
            int patch_id) const
        {
            const auto& flux =
                flux_spaces_[static_cast<std::size_t>(patch_id)];
            const auto& scalar =
                scalar_spaces_[static_cast<std::size_t>(patch_id)];
            const auto n_lambda =
                static_cast<std::size_t>(flux.n_dofs());
            const auto n_u =
                static_cast<std::size_t>(scalar.n_dofs());
            return (n_lambda * n_lambda + n_u * n_lambda + n_u * n_u +
                    n_lambda + n_u) *
                   sizeof(double);
        }

        [[nodiscard]] std::size_t dense_blocks_bytes_for_patch_range_(
            int patch_begin,
            int patch_end) const
        {
            std::size_t bytes = 0;
            for (int patch_id = patch_begin; patch_id < patch_end; ++patch_id)
                bytes += dense_block_bytes_for_patch_(patch_id);
            return bytes;
        }

        struct LocalErrorThreadPolicyDecision
        {
            int selected_threads = 1;
            int candidate_threads = 1;
            int configured_max_threads = 0;
            int hardware_threads = 1;
            bool memory_limited = false;
            bool nested_parallel_disabled = false;
            std::size_t memory_budget_bytes = 0;
            std::size_t base_live_bytes = 0;
            std::size_t per_thread_temporary_bytes = 0;
        };

        [[nodiscard]] LocalErrorThreadPolicyDecision
        select_local_error_thread_policy_(
            int n_items,
            std::size_t base_live_bytes) const
        {
            LocalErrorThreadPolicyDecision decision;
            decision.configured_max_threads = local_error_max_threads_;
            decision.base_live_bytes = base_live_bytes;
            decision.per_thread_temporary_bytes =
                dense_solve_peak_temporary_bytes_() +
                estimate_per_thread_context_bytes_(1) +
                scalar_reduction_basis_peak_bytes_();
            decision.memory_budget_bytes =
                local_error_memory_budget_mb_ > 0.0
                    ? static_cast<std::size_t>(
                          local_error_memory_budget_mb_ * 1024.0 * 1024.0)
                    : 0u;

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
            decision.hardware_threads = core::max_openmp_threads();
            if (omp_in_parallel())
            {
                decision.nested_parallel_disabled = true;
                return decision;
            }

            const int automatic_cap = 4;
            const int configured_cap =
                local_error_max_threads_ > 0
                    ? local_error_max_threads_
                    : automatic_cap;
            decision.candidate_threads =
                std::max(
                    1,
                    std::min(
                        {configured_cap,
                         decision.hardware_threads,
                         finite_element::assembly::detail::
                             recommended_openmp_threads_for_patch_solves(
                                 n_items)}));
            decision.selected_threads = decision.candidate_threads;

            if (decision.memory_budget_bytes > 0 &&
                decision.per_thread_temporary_bytes > 0)
            {
                while (decision.selected_threads > 1)
                {
                    const auto estimated_bytes =
                        decision.base_live_bytes +
                        static_cast<std::size_t>(
                            decision.selected_threads) *
                            decision.per_thread_temporary_bytes;
                    if (estimated_bytes <= decision.memory_budget_bytes)
                        break;
                    --decision.selected_threads;
                    decision.memory_limited = true;
                }

                if (!decision.memory_limited)
                {
                    const auto estimated_bytes =
                        decision.base_live_bytes +
                        static_cast<std::size_t>(
                            decision.selected_threads) *
                            decision.per_thread_temporary_bytes;
                    decision.memory_limited =
                        estimated_bytes > decision.memory_budget_bytes;
                }
            }
#else
            static_cast<void>(n_items);
#endif

            return decision;
        }

        void record_local_error_thread_policy_(
            const finite_element::detail::TimingRecorder& timing,
            const LocalErrorThreadPolicyDecision& decision) const
        {
            timing.add(
                "time_slab.local_error_solves.thread_policy.selected_threads.count",
                static_cast<double>(decision.selected_threads));
            timing.add(
                "time_slab.local_error_solves.thread_policy.candidate_threads.count",
                static_cast<double>(decision.candidate_threads));
            timing.add(
                "time_slab.local_error_solves.thread_policy.configured_max_threads.count",
                static_cast<double>(decision.configured_max_threads));
            timing.add(
                "time_slab.local_error_solves.thread_policy.hardware_threads.count",
                static_cast<double>(decision.hardware_threads));
            timing.add(
                "time_slab.local_error_solves.thread_policy.memory_limited",
                decision.memory_limited ? 1.0 : 0.0);
            timing.add(
                "time_slab.local_error_solves.thread_policy.nested_parallel_disabled",
                decision.nested_parallel_disabled ? 1.0 : 0.0);
            timing.add(
                "time_slab.local_error_solves.thread_policy.memory_budget_bytes",
                static_cast<double>(decision.memory_budget_bytes));
            timing.add(
                "time_slab.local_error_solves.thread_policy.base_live_bytes",
                static_cast<double>(decision.base_live_bytes));
            timing.add(
                "time_slab.local_error_solves.thread_policy.per_thread_temporary_bytes",
                static_cast<double>(decision.per_thread_temporary_bytes));
        }

        static void add_local_error_thread_policy_stats_(
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats,
            const LocalErrorThreadPolicyDecision& decision) noexcept
        {
            stats.hardware_threads =
                std::max(
                    stats.hardware_threads,
                    static_cast<double>(decision.hardware_threads));
            stats.configured_max_threads =
                std::max(
                    stats.configured_max_threads,
                    static_cast<double>(decision.configured_max_threads));
            stats.candidate_threads =
                std::max(
                    stats.candidate_threads,
                    static_cast<double>(decision.candidate_threads));
            stats.selected_threads =
                std::max(
                    stats.selected_threads,
                    static_cast<double>(decision.selected_threads));
            stats.memory_limited += decision.memory_limited ? 1.0 : 0.0;
            stats.nested_parallel_disabled +=
                decision.nested_parallel_disabled ? 1.0 : 0.0;
        }

        static void record_local_error_reuse_summary_(
            const finite_element::detail::TimingRecorder& timing,
            int requested_patch_cells,
            int unique_slab_cells)
        {
            const int duplicate_patch_cells =
                requested_patch_cells - unique_slab_cells;
            const double average_patch_memberships =
                unique_slab_cells > 0
                    ? static_cast<double>(requested_patch_cells) /
                          static_cast<double>(unique_slab_cells)
                    : 0.0;

            timing.add(
                "time_slab.local_error_solves.reuse.requested_patch_cells.count",
                static_cast<double>(requested_patch_cells));
            timing.add(
                "time_slab.local_error_solves.reuse.unique_slab_cells.count",
                static_cast<double>(unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.reuse.duplicate_patch_cells.count",
                static_cast<double>(duplicate_patch_cells));
            timing.add(
                "time_slab.local_error_solves.average_patch_memberships_per_slab_cell",
                average_patch_memberships);
        }

        template<class ActiveSlabCellVector>
        [[nodiscard]] int local_error_patch_cell_membership_count_(
            const ActiveSlabCellVector& active_slab_cells) const
        {
            int membership_count_total = 0;
            for (const auto& cell : active_slab_cells)
            {
                membership_count_total +=
                    patch_set_->cell_patch_count(
                        cell.slab_id,
                        cell.slab_cell_id);
            }
            return membership_count_total;
        }

        template<class ActiveSlabCellVector>
        void record_local_error_global_reuse_summary_(
            const finite_element::detail::TimingRecorder& timing,
            const ActiveSlabCellVector& active_slab_cells) const
        {
            const int global_unique_cells =
                static_cast<int>(active_slab_cells.size());
            const int global_patch_cells =
                local_error_patch_cell_membership_count_(
                    active_slab_cells);
            const int global_duplicate_patch_cells =
                global_patch_cells - global_unique_cells;
            const double global_average =
                global_unique_cells > 0
                    ? static_cast<double>(global_patch_cells) /
                          static_cast<double>(global_unique_cells)
                    : 0.0;

            timing.add(
                "time_slab.local_error_solves.reuse.global_patch_cell_memberships.count",
                static_cast<double>(global_patch_cells));
            timing.add(
                "time_slab.local_error_solves.reuse.global_unique_slab_cells.count",
                static_cast<double>(global_unique_cells));
            timing.add(
                "time_slab.local_error_solves.reuse.global_duplicate_patch_cells.count",
                static_cast<double>(global_duplicate_patch_cells));
            timing.add(
                "time_slab.local_error_solves.reuse.global_average_patch_memberships_per_slab_cell",
                global_average);
        }

        template<class ActiveSlabCellVector>
        [[nodiscard]] std::optional<
            finite_element::assembly::error_system::
                SharedLocalErrorContext2D<XSpace, SlabSpaceType>>
        build_shared_context_if_requested_(
            const ActiveSlabCellVector& active_slab_cells,
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& timing_stats) const
        {
            if (local_error_context_storage_ != "shared_immutable" &&
                local_error_context_storage_ != "shared_immutable_shadow")
            {
                return std::nullopt;
            }

            using SharedContext =
                finite_element::assembly::error_system::
                    SharedLocalErrorContext2D<XSpace, SlabSpaceType>;

            timing_stats.shared_context_shadow_enabled = 1.0;
            const auto shared_context_begin =
                std::chrono::steady_clock::now();
            auto shared_context =
                SharedContext::build(
                    *x_space_,
                    slab_space_ref(),
                    active_slab_cells);
            timing_stats.shared_context_build_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() -
                    shared_context_begin)
                    .count();
            timing_stats.shared_context_memory_mb =
                static_cast<double>(
                    shared_context.estimated_memory_bytes()) /
                (1024.0 * 1024.0);
            timing_stats.shared_context_x_geometry_count =
                static_cast<double>(shared_context.x_geometry_count());
            timing_stats.shared_context_slab_geometry_count =
                static_cast<double>(shared_context.slab_geometry_count());
            timing_stats.shared_context_ancestor_count =
                static_cast<double>(shared_context.ancestor_count());
            timing_stats.shared_context_active_slab_cells =
                static_cast<double>(shared_context.active_slab_cell_count());

            const bool validate_shared_context =
                local_error_context_storage_ == "shared_immutable_shadow" ||
                shared_context_validation_ == "sample" ||
                shared_context_validation_ == "full_debug";
            timing_stats.shared_context_validation_enabled =
                validate_shared_context ? 1.0 : 0.0;
            if (validate_shared_context)
            {
                const auto validation_begin =
                    std::chrono::steady_clock::now();
                finite_element::detail::CellGeometryCache<XSpace>
                    comparison_x_geometry_cache(*x_space_);
                finite_element::assembly::detail::
                    SourceActiveAncestorCache<XSpace>
                        comparison_ancestor_cache(*x_space_);
                auto comparison_slab_geometry_caches =
                    make_slab_geometry_caches_();
                timing_stats
                    .shared_context_comparison_mutable_caches_constructed +=
                    1.0;
                const auto comparison =
                    shared_context.compare_sample(
                        comparison_x_geometry_cache,
                        comparison_ancestor_cache,
                        comparison_slab_geometry_caches);
                timing_stats.shared_context_sample_geometry_max_abs_diff =
                    comparison.sample_geometry_max_abs_diff;
                timing_stats.shared_context_sample_slab_geometry_max_abs_diff =
                    comparison.sample_slab_geometry_max_abs_diff;
                timing_stats.shared_context_sample_ancestor_mismatch_count =
                    comparison.sample_ancestor_mismatch_count;
                timing_stats.shared_context_sample_count =
                    comparison.sample_count;
                timing_stats.shared_context_validation_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - validation_begin)
                        .count();
            }

            return shared_context;
        }

        template<class RTCellCacheType, class ABElementCacheType, class RHSStateCacheType>
        static void add_local_error_cache_qpoint_counters_(
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats,
            const RTCellCacheType& rt_cell_cache,
            const ABElementCacheType& ab_element_cache,
            const RHSStateCacheType& rhs_state_cache,
            double skipped_ab_build_requests = 0.0)
        {
            constexpr double qpoints =
                static_cast<double>(
                    RTCellCacheType::n_quadrature_points_v);
            const double ab_build_requests =
                std::max(
                    0.0,
                    static_cast<double>(ab_element_cache.n_build_requests()) -
                        skipped_ab_build_requests);

            stats.rt_basis_qpoint_fills +=
                static_cast<double>(rt_cell_cache.n_build_requests()) *
                qpoints;
            stats.ab_diffusion_tensor_evaluations +=
                ab_build_requests * qpoints;
            stats.ab_scalar_basis_qpoint_fills +=
                ab_build_requests * qpoints;
            stats.rhs_diffusion_tensor_evaluations +=
                static_cast<double>(rhs_state_cache.n_build_requests()) *
                qpoints;
            stats.rhs_lambda_gradient_evaluations +=
                static_cast<double>(rhs_state_cache.n_build_requests()) *
                qpoints;
            stats.rhs_u_gradient_evaluations +=
                static_cast<double>(rhs_state_cache.n_build_requests()) *
                qpoints;
        }

        template<class QpointStateCacheType>
        static void add_local_error_qpoint_state_counters_(
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats,
            const QpointStateCacheType& qpoint_state_cache)
        {
            stats.rt_basis_qpoint_fills +=
                static_cast<double>(
                    qpoint_state_cache.rt_basis_qpoint_fills());
            stats.ab_diffusion_tensor_evaluations +=
                static_cast<double>(
                    qpoint_state_cache.diffusion_tensor_evaluations());
            stats.ab_scalar_basis_qpoint_fills +=
                static_cast<double>(
                    qpoint_state_cache.scalar_basis_qpoint_fills());
            // The unified qpoint state evaluates M once and reuses it for both
            // AB and RHS work. Keep the old RHS counter at zero so the legacy
            // AB+RHS sum still reports the actual number of tensor evaluations.
            stats.rhs_lambda_gradient_evaluations +=
                static_cast<double>(
                    qpoint_state_cache.lambda_gradient_evaluations());
            stats.rhs_u_gradient_evaluations +=
                static_cast<double>(
                    qpoint_state_cache.u_gradient_evaluations());
            stats.qpoint_state_scalar_basis_qpoint_fills +=
                static_cast<double>(
                    qpoint_state_cache.scalar_basis_qpoint_fills());
            stats.qpoint_state_partition_of_unity_qpoint_fills +=
                static_cast<double>(
                    qpoint_state_cache.partition_of_unity_qpoint_fills());
            stats.qpoint_state_patch_equivalent_scalar_basis_qpoint_fills +=
                static_cast<double>(
                    qpoint_state_cache
                        .patch_equivalent_scalar_basis_qpoint_fills());
            stats.qpoint_state_scalar_basis_qpoint_fills_avoided +=
                static_cast<double>(
                    qpoint_state_cache.scalar_basis_qpoint_fills_avoided());
            stats.qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills +=
                static_cast<double>(
                    qpoint_state_cache
                        .patch_equivalent_partition_of_unity_qpoint_fills());
            stats.qpoint_state_partition_of_unity_qpoint_fills_avoided +=
                static_cast<double>(
                    qpoint_state_cache
                        .partition_of_unity_qpoint_fills_avoided());
        }

        template<class QpointAuditStats>
        static void add_local_error_qpoint_state_audit_stats_(
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats,
            const QpointAuditStats& audit)
        {
            const double sample_scale = audit.qpoint_sample_scale();
            stats.state_prepare_total_seconds +=
                audit.state_prepare_total_seconds;
            stats.state_prepare_unique_count_seconds +=
                audit.state_prepare_unique_count_seconds;
            stats.state_prepare_set_allocation_seconds +=
                audit.state_prepare_set_allocation_seconds;
            stats.state_prepare_map_index_build_seconds +=
                audit.state_prepare_map_index_build_seconds;
            stats.state_prepare_ordinal_map_build_seconds +=
                audit.state_prepare_ordinal_map_build_seconds;
            stats.state_prepare_cell_vector_allocation_seconds +=
                audit.state_prepare_cell_vector_allocation_seconds;
            stats.state_prepare_request_collection_seconds +=
                audit.state_prepare_request_collection_seconds;
            stats.state_prepare_debug_duplicate_request_count +=
                audit.state_prepare_debug_duplicate_request_count;
            stats.state_prepare_memory_mb =
                std::max(stats.state_prepare_memory_mb,
                         audit.state_prepare_memory_mb);
            stats.state_index_mode =
                std::max(stats.state_index_mode, audit.state_index_mode);
            stats.state_index_flat_lookup_count +=
                audit.state_index_flat_lookup_count;
            stats.state_index_map_lookup_count +=
                audit.state_index_map_lookup_count;
            stats.state_index_fallback_hash_lookup_count +=
                audit.state_index_fallback_hash_lookup_count;
            stats.state_fill_total_seconds += audit.state_fill_total_seconds;
            stats.state_fill_active_ancestor_lookup_seconds +=
                audit.state_fill_active_ancestor_lookup_seconds;
            stats.state_fill_geometry_lookup_seconds +=
                audit.state_fill_geometry_lookup_seconds;
            stats.state_fill_time_basis_seconds +=
                audit.state_fill_time_basis_seconds;
            stats.state_fill_qpoints_processed +=
                audit.state_fill_qpoints_processed;
            stats.state_fill_sampled_qpoints +=
                audit.state_fill_sampled_qpoints;
            stats.coefficient_fast_path_enabled =
                std::max(stats.coefficient_fast_path_enabled,
                         audit.coefficient_fast_path_enabled);
            stats.coefficient_fast_path_identity_diffusion_cells +=
                audit.coefficient_fast_path_identity_diffusion_cells;
            stats.coefficient_fast_path_constant_diffusion_cells +=
                audit.coefficient_fast_path_constant_diffusion_cells;
            stats.coefficient_fast_path_zero_load_cells +=
                audit.coefficient_fast_path_zero_load_cells;
            stats.coefficient_fast_path_generic_cells +=
                audit.coefficient_fast_path_generic_cells;
            stats.operator_builder_mode =
                std::max(stats.operator_builder_mode,
                         audit.operator_builder_mode);
            stats.local_A_identity_reference_fast_path_count +=
                audit.local_A_identity_reference_fast_path_count;
            stats.local_A_constant_reference_fast_path_count +=
                audit.local_A_constant_reference_fast_path_count;
            stats.local_A_variable_generic_path_count +=
                audit.local_A_variable_generic_path_count;
            stats.local_B_reference_fast_path_count +=
                audit.local_B_reference_fast_path_count;
            stats.local_A_build_seconds += audit.local_A_build_seconds;
            stats.local_B_build_seconds += audit.local_B_build_seconds;
            stats.local_A_debug_max_abs_diff = std::max(
                stats.local_A_debug_max_abs_diff,
                audit.local_A_debug_max_abs_diff);
            stats.local_A_debug_rel_frobenius_diff = std::max(
                stats.local_A_debug_rel_frobenius_diff,
                audit.local_A_debug_rel_frobenius_diff);
            stats.local_B_debug_max_abs_diff = std::max(
                stats.local_B_debug_max_abs_diff,
                audit.local_B_debug_max_abs_diff);
            stats.local_B_debug_rel_frobenius_diff = std::max(
                stats.local_B_debug_rel_frobenius_diff,
                audit.local_B_debug_rel_frobenius_diff);
            stats.compact_state_shadow_enabled = std::max(
                stats.compact_state_shadow_enabled,
                audit.compact_state_shadow_enabled);
            stats.compact_state_shadow_sample_count +=
                audit.compact_state_shadow_sample_count;
            stats.compact_state_reference_rt_basis_max_abs_diff = std::max(
                stats.compact_state_reference_rt_basis_max_abs_diff,
                audit.compact_state_reference_rt_basis_max_abs_diff);
            stats.compact_state_reference_scalar_basis_max_abs_diff = std::max(
                stats.compact_state_reference_scalar_basis_max_abs_diff,
                audit.compact_state_reference_scalar_basis_max_abs_diff);
            stats.compact_state_reference_partition_value_max_abs_diff =
                std::max(
                    stats.compact_state_reference_partition_value_max_abs_diff,
                    audit.compact_state_reference_partition_value_max_abs_diff);
            stats.compact_state_reference_partition_gradient_max_abs_diff =
                std::max(
                    stats.compact_state_reference_partition_gradient_max_abs_diff,
                    audit.compact_state_reference_partition_gradient_max_abs_diff);
            stats.compact_state_local_A_max_abs_diff = std::max(
                stats.compact_state_local_A_max_abs_diff,
                audit.compact_state_local_A_max_abs_diff);
            stats.compact_state_local_B_max_abs_diff = std::max(
                stats.compact_state_local_B_max_abs_diff,
                audit.compact_state_local_B_max_abs_diff);
            stats.compact_state_rhs_f_max_abs_diff = std::max(
                stats.compact_state_rhs_f_max_abs_diff,
                audit.compact_state_rhs_f_max_abs_diff);
            stats.compact_state_rhs_g_max_abs_diff = std::max(
                stats.compact_state_rhs_g_max_abs_diff,
                audit.compact_state_rhs_g_max_abs_diff);
            stats.compact_state_grad_theta_max_abs_diff = std::max(
                stats.compact_state_grad_theta_max_abs_diff,
                audit.compact_state_grad_theta_max_abs_diff);
            stats.compact_state_u_time_derivative_max_abs_diff = std::max(
                stats.compact_state_u_time_derivative_max_abs_diff,
                audit.compact_state_u_time_derivative_max_abs_diff);
            stats.old_cell_data_bytes_per_cell = std::max(
                stats.old_cell_data_bytes_per_cell,
                audit.old_cell_data_bytes_per_cell);
            stats.operator_state_bytes_per_cell = std::max(
                stats.operator_state_bytes_per_cell,
                audit.operator_state_bytes_per_cell);
            stats.rhs_state_bytes_per_cell = std::max(
                stats.rhs_state_bytes_per_cell,
                audit.rhs_state_bytes_per_cell);
            stats.flux_diagnostic_state_bytes_per_cell = std::max(
                stats.flux_diagnostic_state_bytes_per_cell,
                audit.flux_diagnostic_state_bytes_per_cell);
            stats.reference_table_memory_mb = std::max(
                stats.reference_table_memory_mb,
                audit.reference_table_memory_mb);
            stats.monolithic_cell_data_constructed_count +=
                audit.monolithic_cell_data_constructed_count;
            stats.compact_operator_state_constructed_count +=
                audit.compact_operator_state_constructed_count;
            stats.compact_rhs_state_constructed_count +=
                audit.compact_rhs_state_constructed_count;
            stats.monolithic_debug_path_used_count +=
                audit.monolithic_debug_path_used_count;
            stats.thread_context_construction_count +=
                audit.thread_context_construction_count;
            stats.thread_context_construction_seconds +=
                audit.thread_context_construction_seconds;
            stats.geometry_cache_construction_seconds +=
                audit.geometry_cache_construction_seconds;
            stats.ancestor_cache_construction_seconds +=
                audit.ancestor_cache_construction_seconds;
            stats.slab_geometry_cache_construction_seconds +=
                audit.slab_geometry_cache_construction_seconds;

            stats.state_fill_affine_map_seconds +=
                audit.state_fill_affine_map_seconds * sample_scale;
            stats.state_fill_spatial_rt_basis_seconds +=
                audit.state_fill_spatial_rt_basis_seconds * sample_scale;
            stats.state_fill_scalar_basis_seconds +=
                audit.state_fill_scalar_basis_seconds * sample_scale;
            stats.state_fill_partition_of_unity_seconds +=
                audit.state_fill_partition_of_unity_seconds * sample_scale;
            stats.state_fill_lambda_gradient_seconds +=
                audit.state_fill_lambda_gradient_seconds * sample_scale;
            stats.state_fill_u_gradient_seconds +=
                audit.state_fill_u_gradient_seconds * sample_scale;
            stats.state_fill_load_evaluation_seconds +=
                audit.state_fill_load_evaluation_seconds * sample_scale;
            stats.state_fill_diffusion_evaluation_seconds +=
                audit.state_fill_diffusion_evaluation_seconds * sample_scale;
            stats.state_fill_diffusion_inverse_seconds +=
                audit.state_fill_diffusion_inverse_seconds * sample_scale;
            stats.state_fill_local_A_assembly_seconds +=
                audit.state_fill_local_A_assembly_seconds * sample_scale;
            stats.state_fill_local_B_assembly_seconds +=
                audit.state_fill_local_B_assembly_seconds * sample_scale;
        }

        template<class TableType>
        static void add_local_table_qpoint_counters_(
            finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats,
            const TableType& table)
        {
            stats.local_table_construction_patches += 1.0;
            stats.local_table_construction_patch_cells +=
                static_cast<double>(table.constructed_patch_cells());
            stats.local_table_scalar_basis_qpoint_fills +=
                static_cast<double>(table.scalar_basis_qpoint_fills());
            stats.local_table_partition_of_unity_qpoint_fills +=
                static_cast<double>(
                    table.partition_of_unity_qpoint_fills());
            stats.local_table_owned_rt_basis_qpoint_fills +=
                static_cast<double>(table.owned_rt_basis_qpoint_fills());
        }

        struct PatchDimensionCounters
        {
            int max_n_lambda = 0;
            int max_n_u = 0;
            int max_system_size = 0;
        };

        [[nodiscard]] PatchDimensionCounters patch_dimension_counters_()
            const
        {
            PatchDimensionCounters counters;
            const int patch_count = n_patches();
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                const auto& flux =
                    flux_spaces_[static_cast<std::size_t>(patch_id)];
                const auto& scalar =
                    scalar_spaces_[static_cast<std::size_t>(patch_id)];
                counters.max_n_lambda =
                    std::max(counters.max_n_lambda, flux.n_dofs());
                counters.max_n_u = std::max(counters.max_n_u, scalar.n_dofs());
                counters.max_system_size =
                    std::max(counters.max_system_size,
                             patch_system_size_(patch_id));
            }
            return counters;
        }

        template<class DenseBlockVector>
        void record_local_error_memory_counters_(
            const finite_element::detail::TimingRecorder& timing,
            const DenseBlockVector& dense_blocks,
            std::size_t local_error_tables_bytes,
            std::size_t unified_cell_state_bytes,
            std::size_t patch_solutions_bytes,
            int n_threads) const
        {
            const auto dense_blocks_bytes =
                dense_blocks_bytes_(dense_blocks);
            const auto dense_peak_patch_bytes =
                dense_block_peak_patch_bytes_(dense_blocks);
            const auto operator_cache_bytes = operator_cache_matrix_bytes_();
            const auto factor_cache_bytes = factor_cache_bytes_();
            const auto per_thread_context_bytes =
                estimate_per_thread_context_bytes_(n_threads);
            const auto dense_solve_peak_temporary_bytes =
                dense_solve_peak_temporary_bytes_();
            const auto reduction_basis_bytes =
                estimate_scalar_reduction_basis_bytes_() +
                estimate_concurrent_scalar_reduction_basis_bytes_(n_threads);
            const auto concurrent_solve_temporary_bytes =
                static_cast<std::size_t>(std::max(1, n_threads)) *
                dense_solve_peak_temporary_bytes;
            const auto dimensions = patch_dimension_counters_();
            const auto estimated_total_live_bytes =
                dense_blocks_bytes + patch_solutions_bytes +
                local_error_tables_bytes + unified_cell_state_bytes +
                reduction_basis_bytes + operator_cache_bytes +
                factor_cache_bytes + per_thread_context_bytes +
                concurrent_solve_temporary_bytes;

            timing.add(
                "time_slab.local_error_solves.memory.dense_blocks_bytes",
                static_cast<double>(dense_blocks_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.dense_block_peak_patch_bytes",
                static_cast<double>(dense_peak_patch_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.patch_solutions_bytes",
                static_cast<double>(patch_solutions_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.local_error_tables_bytes",
                static_cast<double>(local_error_tables_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.unified_cell_state_bytes",
                static_cast<double>(unified_cell_state_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.reduction_basis_bytes",
                static_cast<double>(reduction_basis_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.operator_cache_bytes",
                static_cast<double>(operator_cache_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.factor_cache_bytes",
                static_cast<double>(factor_cache_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.per_thread_context_bytes",
                static_cast<double>(per_thread_context_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.dense_solve_peak_temporary_bytes",
                static_cast<double>(dense_solve_peak_temporary_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.concurrent_dense_solve_temporary_bytes",
                static_cast<double>(concurrent_solve_temporary_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.estimated_total_live_bytes",
                static_cast<double>(estimated_total_live_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.process_rss_sample_bytes",
                static_cast<double>(
                    finite_element::detail::current_process_rss_bytes()));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_n_lambda.count",
                static_cast<double>(dimensions.max_n_lambda));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_n_u.count",
                static_cast<double>(dimensions.max_n_u));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_system_size.count",
                static_cast<double>(dimensions.max_system_size));
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_all_patches_tiled_chunked_explicit_(
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            double zero_tol,
            const finite_element::detail::TimingRecorder& timing)
        {
            const int patch_count = n_patches();
            const int tile_size =
                std::max(1, effective_streaming_patch_tile_size_());
            const int tile_count =
                (patch_count + tile_size - 1) / tile_size;
            const int cell_chunk_size =
                std::max(1, effective_streaming_cell_chunk_size_());

            last_solve_all_patches_used_openmp_ = false;

            using CellBuildRequest =
                finite_element::assembly::detail::
                    LocalErrorPatchCellBuildRequest2D;
            using RTCellCache =
                finite_element::assembly::detail::
                    LocalErrorRTCellQuadratureCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType>;
            using ABElementCache =
                finite_element::assembly::error_system::
                    LocalABElementCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using RHSStateCache =
                finite_element::assembly::error_system::
                    LocalRHSStateCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType>;
            using QpointStateCache =
                finite_element::assembly::error_system::
                    LocalErrorQpointStateCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using Tables =
                finite_element::assembly::detail::
                    LocalErrorQuadratureTables2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using DenseLocalErrorBlocks =
                finite_element::assembly::error_system::
                    DenseLocalErrorBlocks;
            finite_element::assembly::error_system::LocalErrorProblemTimingStats
                timing_stats;
            timing_stats.unused_patch_cells_by_local_patch_removed = 1.0;
            const bool use_compact_split_state =
                local_error_cell_state_representation_ == "compact_split";
            const auto unified_streaming_total_begin =
                std::chrono::steady_clock::now();

            std::vector<ActiveSlabCellRef> active_slab_cells;
            {
                finite_element::assembly::error_system::
                    LocalErrorProblemScopedTiming timer(
                        &timing_stats.active_slab_cell_collection_seconds);
                active_slab_cells = collect_active_slab_cell_refs_();
            }
            std::vector<std::vector<int>> cell_color_classes;
            {
                finite_element::assembly::error_system::
                    LocalErrorProblemScopedTiming timer(
                        &timing_stats.cell_coloring_seconds);
                cell_color_classes =
                    build_cell_color_classes_(active_slab_cells);
            }
            auto active_cell_key =
                [](int slab_id, int slab_cell_id) noexcept -> std::uint64_t
            {
                return (static_cast<std::uint64_t>(
                            static_cast<std::uint32_t>(slab_id))
                        << 32u) |
                       static_cast<std::uint32_t>(slab_cell_id);
            };
            std::unordered_map<std::uint64_t, int> active_cell_index;
            active_cell_index.reserve(active_slab_cells.size());
            for (int cell_index = 0;
                 cell_index < static_cast<int>(active_slab_cells.size());
                 ++cell_index)
            {
                const auto cell =
                    active_slab_cells[static_cast<std::size_t>(cell_index)];
                active_cell_index.emplace(
                    active_cell_key(cell.slab_id, cell.slab_cell_id),
                    cell_index);
            }
            std::vector<int> state_build_counts(active_slab_cells.size(), 0);
            int global_patch_cell_memberships = 0;
            double total_tiles_per_cell = 0.0;
            int max_tiles_per_cell = 0;
            int cells_spanning_multiple_tiles = 0;

            auto shared_context_storage =
                build_shared_context_if_requested_(
                active_slab_cells,
                timing_stats);
            const bool use_shared_context_for_state =
                local_error_context_storage_ == "shared_immutable" &&
                shared_context_storage.has_value();
            const bool use_flat_state_index =
                use_shared_context_for_state &&
                local_error_state_index_mode_ == "flat";
            const bool debug_check_flat_unique_ordinals =
                local_error_context_storage_ == "shared_immutable_shadow" ||
                shared_context_validation_ != "off";
            timing_stats.state_index_mode = use_flat_state_index ? 1.0 : 0.0;
            const auto shared_rhs_context =
                finite_element::assembly::error_system::
                    LocalErrorProblemContext<XSpace, SlabSpaceType>{
                        x_space_,
                        slab_space_ptr_,
                        nullptr,
                        nullptr,
                        nullptr,
                        use_shared_context_for_state
                            ? &*shared_context_storage
                            : nullptr};
            std::optional<finite_element::detail::CellGeometryCache<XSpace>>
                rhs_x_geometry_cache;
            std::optional<
                finite_element::assembly::detail::
                    SourceActiveAncestorCache<XSpace>>
                rhs_ancestor_cache;
            std::optional<
                std::vector<
                    finite_element::detail::
                        CellGeometryCache<LocalSlabSpaceType>>>
                rhs_slab_geometry_caches;
            std::optional<
                finite_element::assembly::error_system::
                    LocalErrorProblemContext<XSpace, SlabSpaceType>>
                rhs_context_storage;
            if (use_shared_context_for_state)
            {
                timing_stats.shared_rhs_context_used_count += 1.0;
            }
            else
            {
                const auto mutable_context_begin =
                    std::chrono::steady_clock::now();
                rhs_x_geometry_cache.emplace(*x_space_);
                rhs_ancestor_cache.emplace(*x_space_);
                rhs_slab_geometry_caches.emplace(
                    make_slab_geometry_caches_());
                rhs_context_storage.emplace(
                    finite_element::assembly::error_system::
                        LocalErrorProblemContext<XSpace, SlabSpaceType>{
                            x_space_,
                            slab_space_ptr_,
                            &*rhs_x_geometry_cache,
                            &*rhs_slab_geometry_caches,
                            &*rhs_ancestor_cache});
                timing_stats.mutable_rhs_context_constructed_count += 1.0;
                timing_stats.mutable_rhs_context_construction_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        mutable_context_begin)
                        .count();
            }

            auto optional_tables_bytes =
                [](const std::vector<std::optional<Tables>>& tables) noexcept
            {
                std::size_t bytes =
                    tables.capacity() * sizeof(std::optional<Tables>);
                for (const auto& table : tables)
                {
                    if (table.has_value())
                        bytes += table->estimated_memory_bytes();
                }
                return bytes;
            };

            std::size_t peak_dense_blocks_bytes = 0;
            std::size_t peak_dense_patch_bytes = 0;
            std::size_t peak_tables_bytes = 0;
            std::size_t peak_cell_state_bytes = 0;
            std::size_t peak_tile_workspace_bytes = 0;
            std::size_t peak_process_rss_bytes =
                finite_element::detail::current_process_rss_bytes();
            int peak_selected_threads = 1;
            int total_chunk_count = 0;
            int total_requested_patch_cells = 0;
            int total_unique_slab_cells = 0;
            int total_cell_state_rebuilds = 0;
            int peak_chunk_unique_slab_cells = 0;
            double total_rt_basis_qpoint_fills = 0.0;
            double total_ab_diffusion_tensor_evaluations = 0.0;
            double total_ab_scalar_basis_qpoint_fills = 0.0;
            double total_rhs_diffusion_tensor_evaluations = 0.0;
            double total_rhs_lambda_gradient_evaluations = 0.0;
            double total_rhs_u_gradient_evaluations = 0.0;
            double total_qpoint_state_scalar_basis_qpoint_fills = 0.0;
            double total_qpoint_state_partition_of_unity_qpoint_fills = 0.0;
            double total_qpoint_state_patch_equivalent_scalar_basis_qpoint_fills =
                0.0;
            double total_qpoint_state_scalar_basis_qpoint_fills_avoided = 0.0;
            double
                total_qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills =
                    0.0;
            double
                total_qpoint_state_partition_of_unity_qpoint_fills_avoided =
                    0.0;
            FluxDiagnostics unified_streaming_diagnostics;
            std::vector<char> unified_diagnostics_cell_done(
                active_slab_cells.size(),
                0);
            std::vector<int> unified_diagnostics_source_cell_id(
                active_slab_cells.size(),
                -1);
            std::vector<double> unified_diagnostics_flux_contribution(
                active_slab_cells.size(),
                0.0);
            std::vector<double> unified_diagnostics_residual_contribution(
                active_slab_cells.size(),
                0.0);
            int unified_diagnostics_finalized_cells = 0;
            double unified_diagnostics_state_seconds = 0.0;
            double unified_diagnostics_eval_seconds = 0.0;
            double unified_diagnostics_qpoint_eval_seconds = 0.0;
            double unified_diagnostics_accumulation_seconds = 0.0;
            double unified_diagnostics_extra_cell_state_rebuilds = 0.0;
            double unified_diagnostics_reused_cell_state = 0.0;
            double unified_diagnostics_built_diagnostic_state_count = 0.0;
            double unified_diagnostics_missing_cells = 0.0;
            double unified_diagnostics_duplicate_cells = 0.0;
            int cell_state_cache_hits = 0;
            int cell_state_cache_misses = 0;
            int cell_state_cache_evictions = 0;
            int cell_state_cache_stale_state_detected_count = 0;
            std::size_t peak_cell_state_cache_bytes = 0;
            constexpr std::size_t base_cell_state_cache_limit_bytes =
                8u * 1024u * 1024u;
            const std::size_t configured_cell_state_cache_limit_bytes =
                local_error_memory_budget_mb_ > 0.0
                    ? static_cast<std::size_t>(
                          0.01 * local_error_memory_budget_mb_ *
                          1024.0 * 1024.0)
                    : base_cell_state_cache_limit_bytes;
            const std::size_t cell_state_cache_limit_bytes =
                std::min(
                    base_cell_state_cache_limit_bytes,
                    configured_cell_state_cache_limit_bytes);
            const std::size_t estimated_full_cell_state_cache_bytes =
                active_slab_cells.size() *
                ((use_compact_split_state
                      ? sizeof(typename QpointStateCache::CompactCellData)
                      : sizeof(typename QpointStateCache::CellData)) +
                 sizeof(std::pair<const std::pair<int, int>, int>) +
                 3u * sizeof(void*));
            const bool old_use_solve_cell_state_cache =
                estimated_full_cell_state_cache_bytes <=
                cell_state_cache_limit_bytes;
            const bool streaming_diffusion_supported =
                local_error_coefficient_fast_path_ &&
                (coefficients::is_identity_diffusion_function<
                     GT::dim_space_v>(M) ||
                 coefficients::constant_diffusion_tensor_if_available<
                     GT::dim_space_v>(M)
                     .has_value());
            const bool compact_streaming_flux_supported =
                fused_error_and_flux_diagnostics_ && use_compact_split_state &&
                streaming_diffusion_supported;
            const double flux_diagnostics_mode_code =
                local_error_flux_diagnostics_mode_ == "standalone"
                    ? 0.0
                    : (local_error_flux_diagnostics_mode_ ==
                               "streaming_reuse"
                           ? 1.0
                           : 2.0);
            int flux_diagnostics_fallback_reason_code = 0;
            if (local_error_flux_diagnostics_mode_ == "streaming_reuse" &&
                !compact_streaming_flux_supported)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: --local-error-flux-diagnostics-mode streaming_reuse requires fused diagnostics, compact_split cell state, and identity/constant diffusion fast-path metadata.");
            }
            if (local_error_flux_diagnostics_mode_ == "auto" &&
                !compact_streaming_flux_supported)
            {
                flux_diagnostics_fallback_reason_code =
                    !fused_error_and_flux_diagnostics_
                        ? 1
                        : (!use_compact_split_state
                               ? 2
                               : (!streaming_diffusion_supported ? 4 : 0));
            }
            if (local_error_flux_diagnostics_mode_ == "standalone")
                flux_diagnostics_fallback_reason_code = 3;
            const bool use_unified_streaming_flux_diagnostics =
                compact_streaming_flux_supported &&
                local_error_flux_diagnostics_mode_ != "standalone";
            const bool use_persistent_worker_contexts =
                !use_shared_context_for_state &&
                local_error_context_storage_ ==
                    "persistent_per_thread_debug" &&
                (local_error_worker_context_mode_ ==
                     "persistent_all_p_debug" ||
                 (local_error_worker_context_mode_ != "per_chunk_debug" &&
                  FETraits::p_space_v == 1));
            timing_stats.worker_context_mode =
                use_shared_context_for_state
                    ? 2.0
                    : (use_persistent_worker_contexts ? 1.0 : 0.0);
            std::vector<std::unique_ptr<LocalErrorWorkerContext2D>>
                persistent_worker_contexts;
            auto ensure_persistent_worker_contexts =
                [&](int count)
            {
                if (!use_persistent_worker_contexts)
                    return;
                count = std::max(1, count);
                while (static_cast<int>(persistent_worker_contexts.size()) <
                       count)
                {
                    const auto context_begin =
                        std::chrono::steady_clock::now();
                    const auto geometry_begin =
                        std::chrono::steady_clock::now();
                    auto x_geometry_cache =
                        std::make_unique<
                            finite_element::detail::
                                CellGeometryCache<XSpace>>(
                            *x_space_);
                    timing_stats.geometry_cache_construction_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            geometry_begin)
                            .count();

                    const auto ancestor_begin =
                        std::chrono::steady_clock::now();
                    auto ancestor_cache =
                        std::make_unique<
                            finite_element::assembly::detail::
                                SourceActiveAncestorCache<XSpace>>(
                            *x_space_);
                    timing_stats.ancestor_cache_construction_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            ancestor_begin)
                            .count();

                    const auto slab_geometry_begin =
                        std::chrono::steady_clock::now();
                    auto slab_geometry_caches = make_slab_geometry_caches_();
                    timing_stats.slab_geometry_cache_construction_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            slab_geometry_begin)
                            .count();

                    persistent_worker_contexts.push_back(
                        std::make_unique<LocalErrorWorkerContext2D>(
                            *x_space_,
                            slab_space_ref(),
                            std::move(x_geometry_cache),
                            std::move(ancestor_cache),
                            std::move(slab_geometry_caches)));
                    timing_stats.thread_context_construction_seconds +=
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            context_begin)
                            .count();
                    timing_stats.thread_context_construction_count += 1.0;
                }
            };

            struct MembershipPlanEntry
            {
                int local_patch_id = -1;
                int global_patch_id = -1;
                int patch_cell_index = -1;
            };

            struct CellPlan
            {
                int active_slab_cell_ordinal = -1;
                int slab_id = -1;
                int slab_cell_id = -1;
                int representative_patch_id = -1;
                int representative_patch_cell_index = -1;
                int first_membership = 0;
                int membership_count = 0;
                int source_membership_count = 0;
                int max_patch_id = -1;
            };

            struct LocalErrorTilePlan2D
            {
                int tile_begin = 0;
                int tile_end = 0;
                std::vector<int> active_slab_cell_ordinals{};
                std::vector<CellPlan> cells{};
                std::vector<MembershipPlanEntry> memberships{};
            };

            auto tile_plan_memory_bytes =
                [](const std::vector<LocalErrorTilePlan2D>& plans) noexcept
            {
                std::size_t bytes =
                    plans.capacity() * sizeof(LocalErrorTilePlan2D);
                for (const auto& plan : plans)
                {
                    bytes +=
                        plan.active_slab_cell_ordinals.capacity() *
                        sizeof(int);
                    bytes +=
                        plan.cells.capacity() * sizeof(CellPlan);
                    bytes +=
                        plan.memberships.capacity() *
                        sizeof(MembershipPlanEntry);
                }
                return bytes;
            };

            std::vector<LocalErrorTilePlan2D> tile_plans(
                static_cast<std::size_t>(tile_count));
            {
                const auto tile_plan_begin =
                    std::chrono::steady_clock::now();
                for (int tile_id = 0; tile_id < tile_count; ++tile_id)
                {
                    auto& plan =
                        tile_plans[static_cast<std::size_t>(tile_id)];
                    plan.tile_begin = tile_id * tile_size;
                    plan.tile_end =
                        std::min(patch_count, plan.tile_begin + tile_size);
                }

                std::vector<int> tile_cell_marker(
                    static_cast<std::size_t>(std::max(1, tile_count)),
                    -1);
                std::vector<int> tile_cell_plan_index(
                    static_cast<std::size_t>(std::max(1, tile_count)),
                    -1);
                std::vector<int> touched_tile_ids;
                touched_tile_ids.reserve(8);
                double source_memberships_for_tile_cells = 0.0;

                for (const auto& color_cells : cell_color_classes)
                {
                    for (const int cell_index : color_cells)
                    {
                        const auto cell =
                            active_slab_cells[
                                static_cast<std::size_t>(cell_index)];
                        const int membership_count =
                            patch_set_->cell_patch_count(
                                cell.slab_id,
                                cell.slab_cell_id);
                        timing_stats.patch_membership_lookup_count += 1.0;
                        global_patch_cell_memberships += membership_count;

                        int representative_patch_id = -1;
                        int representative_patch_cell_index = -1;
                        int max_patch_id = -1;
                        int tiles_for_cell = 0;
                        touched_tile_ids.clear();

                        for (int membership_index = 0;
                             membership_index < membership_count;
                             ++membership_index)
                        {
                            const auto& membership =
                                patch_set_->cell_patch_membership(
                                    cell.slab_id,
                                    cell.slab_cell_id,
                                    membership_index);
                            timing_stats.patch_membership_lookup_count += 1.0;
                            const int patch_id = membership.patch_id;
                            if (representative_patch_id < 0)
                            {
                                representative_patch_id = patch_id;
                                representative_patch_cell_index =
                                    membership.patch_cell_index;
                            }
                            max_patch_id = std::max(max_patch_id, patch_id);
                            if (patch_id < 0 || patch_id >= patch_count)
                                continue;

                            const int tile_id = patch_id / tile_size;
                            auto& plan =
                                tile_plans[
                                    static_cast<std::size_t>(tile_id)];
                            if (tile_cell_marker[
                                    static_cast<std::size_t>(tile_id)] !=
                                cell_index)
                            {
                                tile_cell_marker[
                                    static_cast<std::size_t>(tile_id)] =
                                    cell_index;
                                tile_cell_plan_index[
                                    static_cast<std::size_t>(tile_id)] =
                                    static_cast<int>(plan.cells.size());
                                touched_tile_ids.push_back(tile_id);
                                ++tiles_for_cell;
                                source_memberships_for_tile_cells +=
                                    static_cast<double>(membership_count);
                                plan.active_slab_cell_ordinals.push_back(
                                    cell_index);
                                plan.cells.push_back(
                                    CellPlan{
                                        cell_index,
                                        cell.slab_id,
                                        cell.slab_cell_id,
                                        representative_patch_id,
                                        representative_patch_cell_index,
                                        static_cast<int>(
                                            plan.memberships.size()),
                                        0,
                                        membership_count,
                                        max_patch_id});
                            }

                            const int cell_plan_index =
                                tile_cell_plan_index[
                                    static_cast<std::size_t>(tile_id)];
                            plan.memberships.push_back(
                                MembershipPlanEntry{
                                    patch_id - plan.tile_begin,
                                    patch_id,
                                    membership.patch_cell_index});
                            ++plan.cells[
                                  static_cast<std::size_t>(cell_plan_index)]
                                  .membership_count;
                        }

                        for (const int tile_id : touched_tile_ids)
                        {
                            auto& plan =
                                tile_plans[
                                    static_cast<std::size_t>(tile_id)];
                            auto& plan_cell =
                                plan.cells[
                                    static_cast<std::size_t>(
                                        tile_cell_plan_index[
                                            static_cast<std::size_t>(
                                                tile_id)])];
                            plan_cell.representative_patch_id =
                                representative_patch_id;
                            plan_cell.representative_patch_cell_index =
                                representative_patch_cell_index;
                            plan_cell.source_membership_count =
                                membership_count;
                            plan_cell.max_patch_id = max_patch_id;
                        }

                        total_tiles_per_cell +=
                            static_cast<double>(tiles_for_cell);
                        max_tiles_per_cell =
                            std::max(max_tiles_per_cell, tiles_for_cell);
                        if (tiles_for_cell > 1)
                            ++cells_spanning_multiple_tiles;
                    }
                }

                double tile_plan_cells = 0.0;
                double tile_plan_memberships = 0.0;
                for (const auto& plan : tile_plans)
                {
                    tile_plan_cells +=
                        static_cast<double>(plan.cells.size());
                    tile_plan_memberships +=
                        static_cast<double>(plan.memberships.size());
                }

                const double elapsed =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        tile_plan_begin)
                        .count();
                timing_stats.tile_cell_order_construction_seconds += elapsed;
                timing_stats.tile_plan_build_seconds += elapsed;
                timing_stats.membership_plan_construction_seconds += elapsed;
                timing_stats.tile_plan_memory_mb =
                    static_cast<double>(tile_plan_memory_bytes(tile_plans)) /
                    (1024.0 * 1024.0);
                timing_stats.tile_plan_cells += tile_plan_cells;
                timing_stats.tile_plan_memberships += tile_plan_memberships;
                timing_stats.tile_plan_tile_count =
                    static_cast<double>(tile_count);
                timing_stats.tile_plan_membership_scans_avoided +=
                    5.0 * source_memberships_for_tile_cells;
            }

            struct StateLifetimePlanMetrics
            {
                int state_build_requests = 0;
                int cells_used_once = 0;
                int cells_used_multiple_tiles = 0;
                int cells_reused_across_non_adjacent_tiles = 0;
                int max_build_requests_per_cell = 0;
                int max_incident_patches_per_cell = 0;
                int lifetime_window_peak_live_cells = 0;
                double mean_tiles_per_cell = 0.0;
                double mean_incident_patches_per_cell = 0.0;
            };

            struct CacheSimulationResult
            {
                double hits = 0.0;
                double misses = 0.0;
                double evictions = 0.0;
                double hit_rate = 0.0;
                double constructions_avoided = 0.0;
                double memory_bytes = 0.0;
                double expected_cell_state_saved_seconds = 0.0;
            };

            std::vector<int> state_access_ordinals;
            state_access_ordinals.reserve(
                static_cast<std::size_t>(
                    std::max(0.0, timing_stats.tile_plan_cells)));
            std::vector<int> lifetime_first_tile(
                active_slab_cells.size(),
                -1);
            std::vector<int> lifetime_last_tile(
                active_slab_cells.size(),
                -1);
            std::vector<int> lifetime_incident_tile_count(
                active_slab_cells.size(),
                0);
            std::vector<int> lifetime_incident_patch_count(
                active_slab_cells.size(),
                0);
            std::vector<int> lifetime_state_build_requests(
                active_slab_cells.size(),
                0);
            std::vector<int> lifetime_previous_tile(
                active_slab_cells.size(),
                -1);
            std::vector<char> lifetime_non_adjacent_reuse(
                active_slab_cells.size(),
                0);

            for (int tile_id = 0; tile_id < tile_count; ++tile_id)
            {
                const auto& tile_plan =
                    tile_plans[static_cast<std::size_t>(tile_id)];
                for (const auto& plan_cell : tile_plan.cells)
                {
                    const int ordinal =
                        plan_cell.active_slab_cell_ordinal;
                    if (ordinal < 0 ||
                        ordinal >=
                            static_cast<int>(active_slab_cells.size()))
                        continue;
                    const auto ordinal_index =
                        static_cast<std::size_t>(ordinal);
                    state_access_ordinals.push_back(ordinal);
                    if (lifetime_first_tile[ordinal_index] < 0)
                        lifetime_first_tile[ordinal_index] = tile_id;
                    if (lifetime_previous_tile[ordinal_index] >= 0 &&
                        tile_id > lifetime_previous_tile[ordinal_index] + 1)
                        lifetime_non_adjacent_reuse[ordinal_index] = 1;
                    lifetime_previous_tile[ordinal_index] = tile_id;
                    lifetime_last_tile[ordinal_index] = tile_id;
                    ++lifetime_incident_tile_count[ordinal_index];
                    ++lifetime_state_build_requests[ordinal_index];
                    lifetime_incident_patch_count[ordinal_index] =
                        std::max(
                            lifetime_incident_patch_count[ordinal_index],
                            plan_cell.source_membership_count);
                }
            }

            StateLifetimePlanMetrics lifetime_metrics;
            lifetime_metrics.state_build_requests =
                static_cast<int>(state_access_ordinals.size());
            double lifetime_tile_total = 0.0;
            double lifetime_patch_total = 0.0;
            std::vector<int> lifetime_events(
                static_cast<std::size_t>(std::max(1, tile_count + 1)),
                0);
            for (std::size_t ordinal = 0;
                 ordinal < active_slab_cells.size();
                 ++ordinal)
            {
                const int requests =
                    lifetime_state_build_requests[ordinal];
                if (requests == 1)
                    ++lifetime_metrics.cells_used_once;
                else if (requests > 1)
                    ++lifetime_metrics.cells_used_multiple_tiles;
                lifetime_metrics.max_build_requests_per_cell =
                    std::max(
                        lifetime_metrics.max_build_requests_per_cell,
                        requests);
                lifetime_metrics.max_incident_patches_per_cell =
                    std::max(
                        lifetime_metrics.max_incident_patches_per_cell,
                        lifetime_incident_patch_count[ordinal]);
                lifetime_tile_total +=
                    static_cast<double>(
                        lifetime_incident_tile_count[ordinal]);
                lifetime_patch_total +=
                    static_cast<double>(
                        lifetime_incident_patch_count[ordinal]);
                if (lifetime_non_adjacent_reuse[ordinal])
                    ++lifetime_metrics
                          .cells_reused_across_non_adjacent_tiles;
                const int first_tile = lifetime_first_tile[ordinal];
                const int last_tile = lifetime_last_tile[ordinal];
                if (first_tile >= 0 && last_tile >= first_tile)
                {
                    ++lifetime_events[
                        static_cast<std::size_t>(first_tile)];
                    if (last_tile + 1 < static_cast<int>(
                            lifetime_events.size()))
                        --lifetime_events[
                            static_cast<std::size_t>(last_tile + 1)];
                }
            }
            int lifetime_live_cells = 0;
            for (const int event : lifetime_events)
            {
                lifetime_live_cells += event;
                lifetime_metrics.lifetime_window_peak_live_cells =
                    std::max(
                        lifetime_metrics.lifetime_window_peak_live_cells,
                        lifetime_live_cells);
            }
            if (!active_slab_cells.empty())
            {
                lifetime_metrics.mean_tiles_per_cell =
                    lifetime_tile_total /
                    static_cast<double>(active_slab_cells.size());
                lifetime_metrics.mean_incident_patches_per_cell =
                    lifetime_patch_total /
                    static_cast<double>(active_slab_cells.size());
            }

            auto make_cache_result =
                [&](double hits,
                    double misses,
                    double evictions,
                    double memory_bytes) -> CacheSimulationResult
            {
                CacheSimulationResult result;
                result.hits = hits;
                result.misses = misses;
                result.evictions = evictions;
                const double total = hits + misses;
                result.hit_rate = total > 0.0 ? hits / total : 0.0;
                result.constructions_avoided = hits;
                result.memory_bytes = memory_bytes;
                return result;
            };

            auto simulate_lru_cache =
                [&](std::size_t budget_bytes) -> CacheSimulationResult
            {
                using HeapEntry = std::pair<int, int>;
                const std::size_t cell_bytes =
                    use_compact_split_state
                        ? sizeof(typename QpointStateCache::CompactCellData)
                        : sizeof(typename QpointStateCache::CellData);
                const int capacity =
                    cell_bytes == 0
                        ? 0
                        : static_cast<int>(budget_bytes / cell_bytes);
                if (capacity <= 0)
                    return make_cache_result(
                        0.0,
                        static_cast<double>(state_access_ordinals.size()),
                        0.0,
                        0.0);

                std::vector<int> last_seen(
                    active_slab_cells.size(),
                    -1);
                std::vector<char> in_cache(
                    active_slab_cells.size(),
                    0);
                std::priority_queue<
                    HeapEntry,
                    std::vector<HeapEntry>,
                    std::greater<HeapEntry>>
                    heap;
                int current_size = 0;
                int peak_size = 0;
                double hits = 0.0;
                double misses = 0.0;
                double evictions = 0.0;

                for (int access_id = 0;
                     access_id <
                         static_cast<int>(state_access_ordinals.size());
                     ++access_id)
                {
                    const int ordinal =
                        state_access_ordinals[
                            static_cast<std::size_t>(access_id)];
                    if (ordinal < 0 ||
                        ordinal >=
                            static_cast<int>(active_slab_cells.size()))
                        continue;
                    const auto ordinal_index =
                        static_cast<std::size_t>(ordinal);
                    if (in_cache[ordinal_index])
                    {
                        ++hits;
                        last_seen[ordinal_index] = access_id;
                        heap.push(HeapEntry{access_id, ordinal});
                        continue;
                    }

                    ++misses;
                    while (current_size >= capacity && !heap.empty())
                    {
                        const auto [old_access_id, old_ordinal] =
                            heap.top();
                        heap.pop();
                        if (old_ordinal < 0 ||
                            old_ordinal >= static_cast<int>(
                                active_slab_cells.size()))
                            continue;
                        const auto old_index =
                            static_cast<std::size_t>(old_ordinal);
                        if (!in_cache[old_index] ||
                            last_seen[old_index] != old_access_id)
                            continue;
                        in_cache[old_index] = 0;
                        --current_size;
                        ++evictions;
                        break;
                    }
                    in_cache[ordinal_index] = 1;
                    last_seen[ordinal_index] = access_id;
                    heap.push(HeapEntry{access_id, ordinal});
                    ++current_size;
                    peak_size = std::max(peak_size, current_size);
                }

                return make_cache_result(
                    hits,
                    misses,
                    evictions,
                    static_cast<double>(peak_size) *
                        static_cast<double>(cell_bytes));
            };

            const std::size_t cell_data_bytes =
                use_compact_split_state
                    ? sizeof(typename QpointStateCache::CompactCellData)
                    : sizeof(typename QpointStateCache::CellData);
            const double state_access_count =
                static_cast<double>(state_access_ordinals.size());
            const double active_state_count =
                static_cast<double>(active_slab_cells.size());
            CacheSimulationResult cache_sim_off =
                make_cache_result(0.0, state_access_count, 0.0, 0.0);
            CacheSimulationResult cache_sim_tile_local = cache_sim_off;
            CacheSimulationResult cache_sim_lru_64 =
                simulate_lru_cache(64ull * 1024ull * 1024ull);
            CacheSimulationResult cache_sim_lru_256 =
                simulate_lru_cache(256ull * 1024ull * 1024ull);
            CacheSimulationResult cache_sim_lru_1024 =
                simulate_lru_cache(1024ull * 1024ull * 1024ull);
            CacheSimulationResult cache_sim_lru_4096 =
                simulate_lru_cache(4096ull * 1024ull * 1024ull);
            CacheSimulationResult cache_sim_lru_8192 =
                simulate_lru_cache(8192ull * 1024ull * 1024ull);
            CacheSimulationResult cache_sim_lifetime_window =
                make_cache_result(
                    std::max(0.0, state_access_count - active_state_count),
                    active_state_count,
                    active_state_count,
                    static_cast<double>(
                        lifetime_metrics.lifetime_window_peak_live_cells) *
                        static_cast<double>(cell_data_bytes));
            CacheSimulationResult cache_sim_full_if_fits =
                make_cache_result(
                    std::max(0.0, state_access_count - active_state_count),
                    active_state_count,
                    0.0,
                    active_state_count *
                        static_cast<double>(cell_data_bytes));

            std::vector<std::vector<int>> ordinals_expiring_after_tile(
                static_cast<std::size_t>(std::max(1, tile_count)));
            for (std::size_t ordinal = 0;
                 ordinal < lifetime_last_tile.size();
                 ++ordinal)
            {
                const int last_tile = lifetime_last_tile[ordinal];
                if (last_tile >= 0 && last_tile < tile_count)
                    ordinals_expiring_after_tile[
                        static_cast<std::size_t>(last_tile)]
                        .push_back(static_cast<int>(ordinal));
            }

            struct SolveCellStateReuseCache
            {
                using CellData = typename QpointStateCache::CellData;
                using CompactCellData =
                    typename QpointStateCache::CompactCellData;
                using HeapEntry = std::pair<int, int>;

                std::vector<std::unique_ptr<CellData>> cells{};
                std::vector<std::unique_ptr<CompactCellData>> compact_cells{};
                std::vector<int> last_access{};
                std::priority_queue<
                    HeapEntry,
                    std::vector<HeapEntry>,
                    std::greater<HeapEntry>>
                    lru_heap{};
                std::string mode = "off";
                int mode_code = 0;
                bool enabled = false;
                bool bounded_lru = false;
                bool lifetime_window = false;
                std::size_t budget_bytes = 0;
                std::size_t cell_bytes = sizeof(CellData);
                std::size_t current_bytes = 0;
                std::size_t peak_bytes = 0;
                int access_counter = 0;
                int entries = 0;
                int peak_entries = 0;
                int evictions = 0;
                bool compact_split = false;

                SolveCellStateReuseCache(
                    std::size_t ordinal_count,
                    std::string cache_mode,
                    std::size_t cache_budget_bytes,
                    std::size_t estimated_full_bytes,
                    bool flat_state_index_enabled,
                    bool compact_split_state)
                    : cells(ordinal_count),
                      compact_cells(ordinal_count),
                      last_access(ordinal_count, -1),
                      mode(std::move(cache_mode)),
                      budget_bytes(cache_budget_bytes),
                      compact_split(compact_split_state)
                {
                    cell_bytes =
                        compact_split ? sizeof(CompactCellData)
                                      : sizeof(CellData);
                    if (mode == "tile")
                    {
                        mode_code = 1;
                    }
                    else if (mode == "bounded_lru")
                        mode_code = 2;
                    else if (mode == "lifetime_window")
                        mode_code = 3;
                    else if (mode == "full_if_fits")
                        mode_code = 4;
                    else
                        mode_code = 0;

                    if (!flat_state_index_enabled)
                        return;
                    if (mode == "tile")
                    {
                        enabled = true;
                    }
                    else if (mode == "bounded_lru")
                    {
                        bounded_lru = true;
                        enabled =
                            budget_bytes >= cell_bytes && cell_bytes > 0u;
                    }
                    else if (mode == "lifetime_window")
                    {
                        lifetime_window = true;
                        enabled = true;
                    }
                    else if (mode == "full_if_fits")
                    {
                        enabled =
                            budget_bytes > 0u &&
                            estimated_full_bytes <= budget_bytes;
                    }
                }

                [[nodiscard]] bool has(int ordinal) const noexcept
                {
                    return enabled && ordinal >= 0 &&
                           static_cast<std::size_t>(ordinal) <
                               last_access.size() &&
                           (compact_split
                                ? compact_cells[
                                      static_cast<std::size_t>(ordinal)] !=
                                      nullptr
                                : cells[static_cast<std::size_t>(ordinal)] !=
                                      nullptr);
                }

                [[nodiscard]] const CellData& cell(int ordinal) const
                {
                    if (!has(ordinal))
                        throw std::runtime_error(
                            "SolveCellStateReuseCache: cell not cached.");
                    return *cells[static_cast<std::size_t>(ordinal)];
                }

                [[nodiscard]] const CompactCellData& compact_cell(
                    int ordinal) const
                {
                    if (!has(ordinal) || !compact_split)
                        throw std::runtime_error(
                            "SolveCellStateReuseCache: compact cell not cached.");
                    return *compact_cells[static_cast<std::size_t>(ordinal)];
                }

                void touch(int ordinal)
                {
                    if (!enabled || ordinal < 0 ||
                        static_cast<std::size_t>(ordinal) >=
                            last_access.size())
                        return;
                    const int access_id = ++access_counter;
                    last_access[static_cast<std::size_t>(ordinal)] =
                        access_id;
                    if (bounded_lru)
                        lru_heap.push(HeapEntry{access_id, ordinal});
                }

                void store(CellData&& cell_data)
                {
                    if (!enabled || compact_split)
                        return;
                    const int ordinal = cell_data.active_slab_cell_ordinal;
                    if (ordinal < 0 ||
                        static_cast<std::size_t>(ordinal) >= cells.size())
                        return;
                    const auto ordinal_index =
                        static_cast<std::size_t>(ordinal);
                    if (cells[ordinal_index] == nullptr)
                    {
                        cells[ordinal_index] =
                            std::make_unique<CellData>(std::move(cell_data));
                        current_bytes += cell_bytes;
                        ++entries;
                        peak_entries = std::max(peak_entries, entries);
                    }
                    else
                    {
                        *cells[ordinal_index] = std::move(cell_data);
                    }
                    touch(ordinal);
                    evict_lru_if_needed();
                    peak_bytes = std::max(peak_bytes, current_bytes);
                }

                void store(CompactCellData&& cell_data)
                {
                    if (!enabled || !compact_split)
                        return;
                    const int ordinal =
                        cell_data.operator_state.active_slab_cell_ordinal;
                    if (ordinal < 0 ||
                        static_cast<std::size_t>(ordinal) >=
                            compact_cells.size())
                        return;
                    const auto ordinal_index =
                        static_cast<std::size_t>(ordinal);
                    if (compact_cells[ordinal_index] == nullptr)
                    {
                        compact_cells[ordinal_index] =
                            std::make_unique<CompactCellData>(
                                std::move(cell_data));
                        current_bytes += cell_bytes;
                        ++entries;
                        peak_entries = std::max(peak_entries, entries);
                    }
                    else
                    {
                        *compact_cells[ordinal_index] = std::move(cell_data);
                    }
                    touch(ordinal);
                    evict_lru_if_needed();
                    peak_bytes = std::max(peak_bytes, current_bytes);
                }

                void evict_expired(const std::vector<int>& ordinals)
                {
                    if (!enabled || !lifetime_window)
                        return;
                    for (const int ordinal : ordinals)
                        evict(ordinal);
                }

                void evict(int ordinal)
                {
                    if (ordinal < 0 ||
                        static_cast<std::size_t>(ordinal) >= cells.size())
                        return;
                    if (compact_split)
                    {
                        auto& slot =
                            compact_cells[static_cast<std::size_t>(ordinal)];
                        if (slot == nullptr)
                            return;
                        slot.reset();
                    }
                    else
                    {
                        auto& slot = cells[static_cast<std::size_t>(ordinal)];
                        if (slot == nullptr)
                            return;
                        slot.reset();
                    }
                    current_bytes =
                        current_bytes >= cell_bytes
                            ? current_bytes - cell_bytes
                            : 0u;
                    --entries;
                    ++evictions;
                }

                void evict_lru_if_needed()
                {
                    if (!bounded_lru || budget_bytes == 0u)
                        return;
                    while (current_bytes > budget_bytes && !lru_heap.empty())
                    {
                        const auto [access_id, ordinal] = lru_heap.top();
                        lru_heap.pop();
                        if (ordinal < 0 ||
                            static_cast<std::size_t>(ordinal) >=
                                cells.size())
                            continue;
                        const bool empty_slot =
                            compact_split
                                ? compact_cells[
                                      static_cast<std::size_t>(ordinal)] ==
                                      nullptr
                                : cells[static_cast<std::size_t>(ordinal)] ==
                                      nullptr;
                        if (empty_slot ||
                            last_access[static_cast<std::size_t>(ordinal)] !=
                                access_id)
                            continue;
                        evict(ordinal);
                    }
                }

                [[nodiscard]] std::size_t memory_bytes() const noexcept
                {
                    return current_bytes +
                           cells.capacity() *
                               sizeof(std::unique_ptr<CellData>) +
                           compact_cells.capacity() *
                               sizeof(std::unique_ptr<CompactCellData>) +
                           last_access.capacity() * sizeof(int);
                }
            };

            const std::size_t requested_cache_budget_bytes =
                local_error_cell_state_cache_budget_mb_ > 0.0
                    ? static_cast<std::size_t>(
                          local_error_cell_state_cache_budget_mb_ *
                          1024.0 * 1024.0)
                    : 0u;
            SolveCellStateReuseCache solve_cell_state_cache(
                active_slab_cells.size(),
                local_error_cell_state_cache_mode_,
                requested_cache_budget_bytes,
                estimated_full_cell_state_cache_bytes,
                use_flat_state_index,
                use_compact_split_state);

            for (int tile_id = 0; tile_id < tile_count; ++tile_id)
            {
                const auto& tile_plan =
                    tile_plans[static_cast<std::size_t>(tile_id)];
                const int tile_begin = tile_plan.tile_begin;
                const int tile_end = tile_plan.tile_end;
                const int current_tile_size = tile_end - tile_begin;

                std::vector<DenseLocalErrorBlocks> dense_blocks;
                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming allocation_timer(
                            &timing_stats
                                 .tile_dense_block_allocation_seconds);
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming timer(
                            &timing_stats.assemble_C_seconds);
                    dense_blocks.reserve(
                        static_cast<std::size_t>(current_tile_size));
                    for (int patch_id = tile_begin;
                         patch_id < tile_end;
                         ++patch_id)
                    {
                        dense_blocks.emplace_back(
                            flux_spaces_[static_cast<std::size_t>(patch_id)]
                                .n_dofs(),
                            scalar_spaces_[static_cast<std::size_t>(patch_id)]
                                .n_dofs());
                    }
                }

                peak_dense_blocks_bytes =
                    std::max(peak_dense_blocks_bytes,
                             dense_blocks_bytes_(dense_blocks));
                peak_dense_patch_bytes =
                    std::max(peak_dense_patch_bytes,
                             dense_block_peak_patch_bytes_(dense_blocks));
                peak_tile_workspace_bytes =
                    std::max(
                        peak_tile_workspace_bytes,
                        dense_blocks_bytes_(dense_blocks));
                std::unordered_map<
                    int,
                    typename QpointStateCache::FluxDiagnosticCellState2D>
                    tile_flux_diagnostic_states;
                if (use_unified_streaming_flux_diagnostics)
                    tile_flux_diagnostic_states.reserve(
                        tile_plan.cells.size());

                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming tile_assembly_timer(
                            &timing_stats.streaming_tile_assembly_seconds);
                    for (int chunk_begin = 0;
                         chunk_begin <
                             static_cast<int>(tile_plan.cells.size());
                         chunk_begin += cell_chunk_size)
                    {
                        const int chunk_end =
                            std::min(
                                static_cast<int>(tile_plan.cells.size()),
                                chunk_begin + cell_chunk_size);
                        ++total_chunk_count;

                        std::vector<CellBuildRequest> cell_requests;
                        int chunk_patch_cell_memberships = 0;

                        {
                            finite_element::assembly::error_system::
                                LocalErrorProblemScopedTiming timer(
                                    &timing_stats
                                         .cell_requests_construction_seconds);
                            cell_requests.reserve(
                                static_cast<std::size_t>(
                                    chunk_end - chunk_begin));
                            for (int order_index = chunk_begin;
                                 order_index < chunk_end;
                                 ++order_index)
                            {
                                const auto& plan_cell =
                                    tile_plan.cells[
                                        static_cast<std::size_t>(order_index)];
                                cell_requests.push_back(
                                    CellBuildRequest{
                                        plan_cell.representative_patch_id,
                                        plan_cell
                                            .representative_patch_cell_index,
                                        plan_cell.active_slab_cell_ordinal});
                                chunk_patch_cell_memberships +=
                                    plan_cell.membership_count;
                            }
                        }

                        int chunk_cell_state_cache_hits = 0;
                        int chunk_cell_state_cache_misses = 0;
                        std::vector<CellBuildRequest> uncached_cell_requests;
                        uncached_cell_requests.reserve(cell_requests.size());
                        std::vector<char> cell_request_cache_hit(
                            cell_requests.size(),
                            0);
                        std::vector<int> cell_request_build_index(
                            cell_requests.size(),
                            -1);
                        QpointStateCache chunk_qpoint_state_cache;
                        chunk_qpoint_state_cache
                            .set_cell_state_representation(
                                local_error_cell_state_representation_);
                        typename QpointStateCache::AuditStats chunk_audit_stats;
                        {
                            finite_element::assembly::error_system::
                                LocalErrorProblemScopedTiming state_timer(
                                    &timing_stats
                                         .chunk_cell_state_construction_seconds);
                            finite_element::assembly::error_system::
                                LocalErrorProblemScopedTiming
                                    streaming_state_timer(
                                        &timing_stats
                                             .streaming_chunk_state_construction_seconds);
                            auto unified_timer =
                                timing.scoped(
                                    "time_slab.local_error_solves.unified_cell_state_construction");

                            if (solve_cell_state_cache.enabled)
                            {
                                for (int request_index = 0;
                                     request_index <
                                         static_cast<int>(
                                             cell_requests.size());
                                     ++request_index)
                                {
                                    const auto& request =
                                        cell_requests[
                                            static_cast<std::size_t>(
                                                request_index)];
                                    const bool cache_hit =
                                        use_flat_state_index &&
                                        solve_cell_state_cache.has(
                                            request
                                                .active_slab_cell_ordinal);
                                    if (use_flat_state_index)
                                        timing_stats
                                            .state_index_flat_lookup_count +=
                                            1.0;
                                    else
                                        timing_stats
                                            .state_index_map_lookup_count +=
                                            1.0;
                                    if (cache_hit)
                                    {
                                        ++chunk_cell_state_cache_hits;
                                        cell_request_cache_hit[
                                            static_cast<std::size_t>(
                                                request_index)] = 1;
                                        solve_cell_state_cache.touch(
                                            request
                                                .active_slab_cell_ordinal);
                                    }
                                    else
                                    {
                                        cell_request_build_index[
                                            static_cast<std::size_t>(
                                                request_index)] =
                                            static_cast<int>(
                                                uncached_cell_requests.size());
                                        uncached_cell_requests.push_back(
                                            request);
                                        ++chunk_cell_state_cache_misses;
                                    }
                                }
                            }
                            else
                            {
                                chunk_cell_state_cache_misses =
                                    static_cast<int>(cell_requests.size());
                                for (int request_index = 0;
                                     request_index <
                                         static_cast<int>(
                                             cell_requests.size());
                                     ++request_index)
                                {
                                    cell_request_build_index[
                                        static_cast<std::size_t>(
                                            request_index)] = request_index;
                                }
                            }

                            const auto& build_cell_requests =
                                solve_cell_state_cache.enabled
                                    ? uncached_cell_requests
                                    : cell_requests;

                            if (!build_cell_requests.empty())
                            {
                                if (use_flat_state_index)
                                {
                                    chunk_qpoint_state_cache
                                        .prepare_from_patch_cells_flat(
                                            flux_spaces_,
                                            scalar_spaces_,
                                            build_cell_requests,
                                            *shared_context_storage,
                                            &chunk_audit_stats,
                                            debug_check_flat_unique_ordinals);
                                }
                                else
                                {
                                    chunk_qpoint_state_cache
                                        .prepare_from_patch_cells(
                                            flux_spaces_,
                                            scalar_spaces_,
                                            build_cell_requests,
                                            &chunk_audit_stats);
                                }
                                for (const auto& request : build_cell_requests)
                                {
                                    if (use_flat_state_index &&
                                        request.active_slab_cell_ordinal >= 0)
                                    {
                                        ++state_build_counts[
                                            static_cast<std::size_t>(
                                                request
                                                    .active_slab_cell_ordinal)];
                                    }
                                    else
                                    {
                                        const auto& flux_space =
                                            flux_spaces_[
                                                static_cast<std::size_t>(
                                                    request.patch_id)];
                                        const auto& patch_cell =
                                            flux_space.patch().cell(
                                                request.patch_cell_index);
                                        const auto it =
                                            active_cell_index.find(
                                                active_cell_key(
                                                    flux_space.patch().slab_id,
                                                    patch_cell.slab_cell_id));
                                        if (it != active_cell_index.end())
                                        {
                                            ++state_build_counts[
                                                static_cast<std::size_t>(
                                                    it->second)];
                                        }
                                    }
                                }
                            }

                            const auto state_thread_policy =
                                select_local_error_thread_policy_(
                                    static_cast<int>(
                                        build_cell_requests.size()),
                                    chunk_qpoint_state_cache
                                        .estimated_memory_bytes());
                            add_local_error_thread_policy_stats_(
                                timing_stats,
                                state_thread_policy);
                            peak_selected_threads =
                                std::max(
                                    peak_selected_threads,
                                    state_thread_policy.selected_threads);
                            if (state_thread_policy.selected_threads > 1)
                                last_solve_all_patches_used_openmp_ = true;

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
                            if (!build_cell_requests.empty() &&
                                state_thread_policy.selected_threads > 1)
                            {
                                std::exception_ptr state_error;
                                std::vector<typename QpointStateCache::AuditStats>
                                    thread_audit_stats(
                                        static_cast<std::size_t>(
                                            state_thread_policy
                                                .selected_threads));
                                ensure_persistent_worker_contexts(
                                    state_thread_policy.selected_threads);
                                timing_stats.parallel_region_count += 1.0;
#pragma omp parallel num_threads(state_thread_policy.selected_threads)
                                {
                                    int thread_id = 0;
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
                                    thread_id = omp_get_thread_num();
#endif
                                    auto& thread_audit =
                                        thread_audit_stats[
                                            static_cast<std::size_t>(
                                                thread_id)];
                                    std::unique_ptr<
                                        finite_element::detail::
                                            CellGeometryCache<XSpace>>
                                        thread_x_geometry_cache;
                                    std::unique_ptr<
                                        finite_element::assembly::detail::
                                            SourceActiveAncestorCache<XSpace>>
                                        thread_ancestor_cache;
                                    std::vector<
                                        finite_element::detail::
                                            CellGeometryCache<
                                                LocalSlabSpaceType>>
                                        thread_slab_geometry_caches;
                                    finite_element::assembly::error_system::
                                        LocalErrorProblemContext<
                                            XSpace,
                                            SlabSpaceType>
                                        thread_rhs_context{};
                                    const finite_element::assembly::
                                        error_system::
                                            LocalErrorProblemContext<
                                                XSpace,
                                                SlabSpaceType>*
                                        active_rhs_context = nullptr;
                                    if (use_shared_context_for_state)
                                    {
                                        active_rhs_context =
                                            &shared_rhs_context;
                                    }
                                    else if (use_persistent_worker_contexts)
                                    {
                                        active_rhs_context =
                                            &persistent_worker_contexts[
                                                 static_cast<std::size_t>(
                                                     thread_id)]
                                                 ->problem_context;
                                    }
                                    else
                                    {
                                        const auto context_begin =
                                            std::chrono::steady_clock::now();
                                        const auto geometry_begin =
                                            std::chrono::steady_clock::now();
                                        thread_x_geometry_cache =
                                            std::make_unique<
                                                finite_element::detail::
                                                    CellGeometryCache<XSpace>>(
                                                *x_space_);
                                        thread_audit
                                            .geometry_cache_construction_seconds +=
                                            std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                geometry_begin)
                                                .count();
                                        const auto ancestor_begin =
                                            std::chrono::steady_clock::now();
                                        thread_ancestor_cache =
                                            std::make_unique<
                                                finite_element::assembly::
                                                    detail::
                                                        SourceActiveAncestorCache<
                                                            XSpace>>(
                                                *x_space_);
                                        thread_audit
                                            .ancestor_cache_construction_seconds +=
                                            std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                ancestor_begin)
                                                .count();
                                        const auto slab_geometry_begin =
                                            std::chrono::steady_clock::now();
                                        thread_slab_geometry_caches =
                                            make_slab_geometry_caches_();
                                        thread_audit
                                            .slab_geometry_cache_construction_seconds +=
                                            std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                slab_geometry_begin)
                                                .count();
                                        thread_rhs_context =
                                            finite_element::assembly::
                                                error_system::
                                                    LocalErrorProblemContext<
                                                        XSpace,
                                                        SlabSpaceType>{
                                                        x_space_,
                                                        slab_space_ptr_,
                                                        thread_x_geometry_cache
                                                            .get(),
                                                        &thread_slab_geometry_caches,
                                                        thread_ancestor_cache
                                                            .get()};
                                        active_rhs_context =
                                            &thread_rhs_context;
                                        thread_audit
                                            .thread_context_construction_seconds +=
                                            std::chrono::duration<double>(
                                                std::chrono::steady_clock::now() -
                                                context_begin)
                                                .count();
                                        thread_audit
                                            .thread_context_construction_count +=
                                            1.0;
                                    }

#pragma omp for schedule(dynamic, 8)
                                    for (int miss_index = 0;
                                         miss_index <
                                            static_cast<int>(
                                                 build_cell_requests.size());
                                         ++miss_index)
                                    {
                                        try
                                        {
                                            chunk_qpoint_state_cache
                                                .fill_build_request(
                                                    miss_index,
                                                    flux_spaces_,
                                                    scalar_spaces_,
                                                    *active_rhs_context,
                                                    lambda_tilde,
                                                    u_delta,
                                                    ell,
                                                    M,
                                                    &thread_audit,
                                                    local_error_coefficient_fast_path_,
                                                    local_error_compact_state_shadow_);
                                        }
                                        catch (...)
                                        {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                                            {
                                                if (!state_error)
                                                    state_error =
                                                        std::current_exception();
                                            }
                                        }
                                    }
                                }
                                finite_element::assembly::detail::
                                    rethrow_parallel_exception(
                                        state_error);
                                for (const auto& thread_audit :
                                     thread_audit_stats)
                                    chunk_audit_stats.add(thread_audit);
                            }
                            else
#endif
                            {
                                ensure_persistent_worker_contexts(1);
                                const auto* active_rhs_context =
                                    use_shared_context_for_state
                                        ? &shared_rhs_context
                                        : (use_persistent_worker_contexts
                                               ? &persistent_worker_contexts
                                                     .front()
                                                     ->problem_context
                                               : &*rhs_context_storage);
                                for (int request_id = 0;
                                     request_id <
                                         chunk_qpoint_state_cache
                                             .n_build_requests();
                                     ++request_id)
                                {
                                    chunk_qpoint_state_cache.fill_build_request(
                                        request_id,
                                        flux_spaces_,
                                        scalar_spaces_,
                                        *active_rhs_context,
                                        lambda_tilde,
                                        u_delta,
                                        ell,
                                        M,
                                        &chunk_audit_stats,
                                        local_error_coefficient_fast_path_,
                                        local_error_compact_state_shadow_);
                                }
                            }

                            const std::size_t cell_state_bytes =
                                chunk_qpoint_state_cache.estimated_memory_bytes();
                            peak_cell_state_bytes =
                                std::max(
                                    peak_cell_state_bytes,
                                    cell_state_bytes);
                            add_local_error_qpoint_state_audit_stats_(
                                timing_stats,
                                chunk_audit_stats);
                        }

                        cell_state_cache_hits += chunk_cell_state_cache_hits;
                        cell_state_cache_misses +=
                            chunk_cell_state_cache_misses;
                        total_requested_patch_cells +=
                            chunk_patch_cell_memberships;
                        total_unique_slab_cells +=
                            static_cast<int>(cell_requests.size());
                        total_cell_state_rebuilds +=
                            chunk_cell_state_cache_misses;
                        peak_cell_state_cache_bytes =
                            std::max(
                                peak_cell_state_cache_bytes,
                                solve_cell_state_cache
                                    .memory_bytes());
                        total_rt_basis_qpoint_fills +=
                            static_cast<double>(
                                chunk_cell_state_cache_misses) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_ab_diffusion_tensor_evaluations +=
                            static_cast<double>(
                                chunk_cell_state_cache_misses) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_ab_scalar_basis_qpoint_fills +=
                            static_cast<double>(
                                chunk_cell_state_cache_misses) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_qpoint_state_scalar_basis_qpoint_fills +=
                            static_cast<double>(
                                chunk_cell_state_cache_misses) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_qpoint_state_partition_of_unity_qpoint_fills +=
                            static_cast<double>(
                                chunk_cell_state_cache_misses) *
                            3.0 *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_qpoint_state_patch_equivalent_scalar_basis_qpoint_fills +=
                            static_cast<double>(
                                chunk_patch_cell_memberships) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_qpoint_state_scalar_basis_qpoint_fills_avoided +=
                            std::max(
                                0.0,
                                (static_cast<double>(
                                     chunk_patch_cell_memberships) -
                                 static_cast<double>(
                                     chunk_cell_state_cache_misses)) *
                                    static_cast<double>(
                                        QpointStateCache::
                                            n_quadrature_points_v));
                        total_qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills +=
                            static_cast<double>(
                                chunk_patch_cell_memberships) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_qpoint_state_partition_of_unity_qpoint_fills_avoided +=
                            std::max(
                                0.0,
                                (static_cast<double>(
                                     chunk_patch_cell_memberships) -
                                 3.0 * static_cast<double>(
                                           chunk_cell_state_cache_misses)) *
                                    static_cast<double>(
                                        QpointStateCache::
                                            n_quadrature_points_v));
                        total_rhs_lambda_gradient_evaluations +=
                            static_cast<double>(
                                chunk_cell_state_cache_misses) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        total_rhs_u_gradient_evaluations +=
                            static_cast<double>(
                                chunk_cell_state_cache_misses) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v);
                        peak_chunk_unique_slab_cells =
                            std::max(
                                peak_chunk_unique_slab_cells,
                                static_cast<int>(cell_requests.size()));

                        auto monolithic_state_cell_for_order =
                            [&](int order_index)
                                -> const typename QpointStateCache::CellData&
                        {
                            const auto& plan_cell =
                                tile_plan.cells[
                                    static_cast<std::size_t>(order_index)];
                            const int local_request_index =
                                order_index - chunk_begin;
                            if (solve_cell_state_cache.enabled &&
                                local_request_index >= 0 &&
                                local_request_index <
                                    static_cast<int>(
                                        cell_request_cache_hit.size()) &&
                                cell_request_cache_hit[
                                    static_cast<std::size_t>(
                                        local_request_index)])
                            {
                                return solve_cell_state_cache.cell(
                                    plan_cell.active_slab_cell_ordinal);
                            }

                            const int chunk_cache_index =
                                local_request_index >= 0 &&
                                        local_request_index <
                                            static_cast<int>(
                                                cell_request_build_index
                                                    .size())
                                    ? cell_request_build_index[
                                          static_cast<std::size_t>(
                                              local_request_index)]
                                    : -1;
                            if (use_flat_state_index)
                            {
                                return chunk_qpoint_state_cache.cell_by_index(
                                    chunk_cache_index);
                            }
                            return chunk_qpoint_state_cache.cell(
                                plan_cell.slab_id,
                                plan_cell.slab_cell_id);
                        };
                        auto compact_state_cell_for_order =
                            [&](int order_index)
                                -> const typename QpointStateCache::
                                    CompactCellData&
                        {
                            const auto& plan_cell =
                                tile_plan.cells[
                                    static_cast<std::size_t>(order_index)];
                            const int local_request_index =
                                order_index - chunk_begin;
                            if (solve_cell_state_cache.enabled &&
                                local_request_index >= 0 &&
                                local_request_index <
                                    static_cast<int>(
                                        cell_request_cache_hit.size()) &&
                                cell_request_cache_hit[
                                    static_cast<std::size_t>(
                                        local_request_index)])
                            {
                                return solve_cell_state_cache.compact_cell(
                                    plan_cell.active_slab_cell_ordinal);
                            }

                            const int chunk_cache_index =
                                local_request_index >= 0 &&
                                        local_request_index <
                                            static_cast<int>(
                                                cell_request_build_index
                                                    .size())
                                    ? cell_request_build_index[
                                          static_cast<std::size_t>(
                                              local_request_index)]
                                    : -1;
                            if (use_flat_state_index)
                            {
                                return chunk_qpoint_state_cache
                                    .compact_cell_by_index(chunk_cache_index);
                            }
                            return chunk_qpoint_state_cache
                                .compact_cell_by_ordinal(
                                    plan_cell.active_slab_cell_ordinal);
                        };

                        {
                            peak_tile_workspace_bytes =
                                std::max(
                                    peak_tile_workspace_bytes,
                                    dense_blocks_bytes_(dense_blocks));

                            {
                                finite_element::assembly::error_system::
                                    LocalErrorProblemScopedTiming timer(
                                        &timing_stats.assemble_A_seconds);
                                for (int order_index = chunk_begin;
                                     order_index < chunk_end;
                                     ++order_index)
                                {
                                    const auto& plan_cell =
                                        tile_plan.cells[
                                            static_cast<std::size_t>(
                                                order_index)];
                                    if (use_flat_state_index)
                                        timing_stats
                                            .state_index_flat_lookup_count +=
                                            1.0;
                                    else
                                        timing_stats
                                            .state_index_map_lookup_count +=
                                            1.0;
                                    const auto& local_A =
                                        use_compact_split_state
                                            ? compact_state_cell_for_order(
                                                  order_index)
                                                  .operator_state.local_A
                                            : finite_element::assembly::
                                                  error_system::
                                                      local_rt_mass_matrix_from_qpoint_state_2d<
                                                          FluxSpaceType>(
                                                          monolithic_state_cell_for_order(
                                                              order_index));
                                    for (int membership_index =
                                             plan_cell.first_membership;
                                         membership_index <
                                             plan_cell.first_membership +
                                                 plan_cell.membership_count;
                                         ++membership_index)
                                    {
                                        const auto& membership =
                                            tile_plan.memberships[
                                                static_cast<std::size_t>(
                                                    membership_index)];
                                        const int patch_id =
                                            membership.global_patch_id;

                                        finite_element::assembly::error_system::
                                            scatter_rt_local_matrix_dense_2d(
                                                dense_blocks[
                                                    static_cast<std::size_t>(
                                                        membership
                                                            .local_patch_id)]
                                                    .A,
                                                local_A,
                                                flux_spaces_[
                                                    static_cast<std::size_t>(
                                                        patch_id)],
                                                membership.patch_cell_index,
                                                zero_tol);
                                    }
                                }
                            }

                            {
                                finite_element::assembly::error_system::
                                    LocalErrorProblemScopedTiming timer(
                                        &timing_stats.assemble_B_seconds);
                                for (int order_index = chunk_begin;
                                     order_index < chunk_end;
                                     ++order_index)
                                {
                                    const auto& plan_cell =
                                        tile_plan.cells[
                                            static_cast<std::size_t>(
                                                order_index)];
                                    if (use_flat_state_index)
                                        timing_stats
                                            .state_index_flat_lookup_count +=
                                            1.0;
                                    else
                                        timing_stats
                                            .state_index_map_lookup_count +=
                                            1.0;
                                    const auto& local_B =
                                        use_compact_split_state
                                            ? compact_state_cell_for_order(
                                                  order_index)
                                                  .operator_state.local_B
                                            : finite_element::assembly::
                                                  error_system::
                                                      local_divergence_matrix_from_qpoint_state_2d<
                                                          ScalarSpaceType,
                                                          FluxSpaceType>(
                                                          monolithic_state_cell_for_order(
                                                              order_index));
                                    for (int membership_index =
                                             plan_cell.first_membership;
                                         membership_index <
                                             plan_cell.first_membership +
                                                 plan_cell.membership_count;
                                         ++membership_index)
                                    {
                                        const auto& membership =
                                            tile_plan.memberships[
                                                static_cast<std::size_t>(
                                                    membership_index)];
                                        const int patch_id =
                                            membership.global_patch_id;

                                        finite_element::assembly::error_system::
                                            scatter_divergence_local_matrix_dense_2d(
                                                dense_blocks[
                                                    static_cast<std::size_t>(
                                                        membership
                                                            .local_patch_id)]
                                                    .B,
                                                local_B,
                                                scalar_spaces_[
                                                    static_cast<std::size_t>(
                                                        patch_id)],
                                                flux_spaces_[
                                                    static_cast<std::size_t>(
                                                        patch_id)],
                                                membership.patch_cell_index,
                                                zero_tol);
                                    }
                                }
                            }

                            {
                                finite_element::assembly::error_system::
                                    LocalErrorProblemScopedTiming timer(
                                        &timing_stats.assemble_f_seconds);
                                for (int order_index = chunk_begin;
                                     order_index < chunk_end;
                                     ++order_index)
                                {
                                    const auto& plan_cell =
                                        tile_plan.cells[
                                            static_cast<std::size_t>(
                                                order_index)];
                                    if (use_flat_state_index)
                                        timing_stats
                                            .state_index_flat_lookup_count +=
                                            1.0;
                                    else
                                        timing_stats
                                            .state_index_map_lookup_count +=
                                            1.0;
                                    for (int membership_index =
                                             plan_cell.first_membership;
                                         membership_index <
                                             plan_cell.first_membership +
                                                 plan_cell.membership_count;
                                         ++membership_index)
                                    {
                                        const auto& membership =
                                            tile_plan.memberships[
                                                static_cast<std::size_t>(
                                                    membership_index)];
                                        const int patch_id =
                                            membership.global_patch_id;

                                        const auto& flux_space =
                                            flux_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)];
                                        la::local::FixedLocalVector<
                                            FluxSpaceType::local_dofs_v>
                                            local_f;
                                        finite_element::assembly::detail::
                                            zero_local_vector(local_f);
                                        if (use_compact_split_state)
                                        {
                                            QpointStateCache::
                                                accumulate_patch_flux_rhs_from_compact_state(
                                                    local_f,
                                                    flux_space,
                                                    compact_state_cell_for_order(
                                                        order_index),
                                                    membership
                                                        .patch_cell_index);
                                        }
                                        else
                                        {
                                            finite_element::assembly::error_system::
                                                accumulate_patch_flux_rhs_from_qpoint_state_2d(
                                                    local_f,
                                                    flux_space,
                                                    monolithic_state_cell_for_order(
                                                        order_index),
                                                    membership
                                                        .patch_cell_index);
                                        }
                                        finite_element::assembly::error_system::
                                            scatter_rt_local_vector_dense_2d(
                                                dense_blocks[
                                                    static_cast<std::size_t>(
                                                        membership
                                                            .local_patch_id)]
                                                    .f,
                                                local_f,
                                                flux_space,
                                                membership.patch_cell_index);
                                    }
                                }
                            }

                            {
                                finite_element::assembly::error_system::
                                    LocalErrorProblemScopedTiming timer(
                                        &timing_stats.assemble_g_seconds);
                                for (int order_index = chunk_begin;
                                     order_index < chunk_end;
                                     ++order_index)
                                {
                                    const auto& plan_cell =
                                        tile_plan.cells[
                                            static_cast<std::size_t>(
                                                order_index)];
                                    if (use_flat_state_index)
                                        timing_stats
                                            .state_index_flat_lookup_count +=
                                            1.0;
                                    else
                                        timing_stats
                                            .state_index_map_lookup_count +=
                                            1.0;
                                    for (int membership_index =
                                             plan_cell.first_membership;
                                         membership_index <
                                             plan_cell.first_membership +
                                                 plan_cell.membership_count;
                                         ++membership_index)
                                    {
                                        const auto& membership =
                                            tile_plan.memberships[
                                                static_cast<std::size_t>(
                                                    membership_index)];
                                        const int patch_id =
                                            membership.global_patch_id;

                                        const auto& scalar_space =
                                            scalar_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)];
                                        la::local::FixedLocalVector<
                                            ScalarSpaceType::local_dofs_v>
                                            local_g;
                                        finite_element::assembly::detail::
                                            zero_local_vector(local_g);
                                        if (use_compact_split_state)
                                        {
                                            const auto& flux_space =
                                                flux_spaces_[
                                                    static_cast<std::size_t>(
                                                        patch_id)];
                                            QpointStateCache::
                                                accumulate_patch_scalar_rhs_from_compact_state(
                                                    local_g,
                                                    flux_space,
                                                    scalar_space,
                                                    compact_state_cell_for_order(
                                                        order_index),
                                                    membership
                                                        .patch_cell_index);
                                        }
                                        else
                                        {
                                            finite_element::assembly::error_system::
                                                accumulate_patch_scalar_rhs_from_qpoint_state_2d(
                                                    local_g,
                                                    scalar_space,
                                                    monolithic_state_cell_for_order(
                                                        order_index),
                                                    membership
                                                        .patch_cell_index);
                                        }
                                        finite_element::assembly::error_system::
                                            scatter_scalar_local_vector_dense_2d(
                                                dense_blocks[
                                                    static_cast<std::size_t>(
                                                        membership
                                                            .local_patch_id)]
                                                    .g,
                                                local_g,
                                                scalar_space,
                                                membership.patch_cell_index);
                                    }
                                }
                            }
                        }

                        if (use_unified_streaming_flux_diagnostics)
                        {
                            const auto diagnostic_state_begin =
                                std::chrono::steady_clock::now();
                            for (int order_index = chunk_begin;
                                 order_index < chunk_end;
                                 ++order_index)
                            {
                                const auto& plan_cell =
                                    tile_plan.cells[
                                        static_cast<std::size_t>(
                                            order_index)];
                                const bool have_representative =
                                    plan_cell.representative_patch_id >= 0 &&
                                    plan_cell.representative_patch_cell_index >=
                                        0;
                                const bool all_memberships_solved =
                                    plan_cell.max_patch_id < tile_end;
                                if (!have_representative ||
                                    !all_memberships_solved)
                                    continue;
                                const int ordinal =
                                    plan_cell.active_slab_cell_ordinal;
                                if (ordinal < 0 ||
                                    ordinal >=
                                        static_cast<int>(
                                            unified_diagnostics_cell_done
                                                .size()) ||
                                    unified_diagnostics_cell_done[
                                        static_cast<std::size_t>(
                                            ordinal)] != 0)
                                    continue;

                                auto diagnostic_state =
                                    QpointStateCache::
                                        flux_diagnostic_state_from_compact_state(
                                            compact_state_cell_for_order(
                                                order_index));
                                const auto [it, inserted] =
                                    tile_flux_diagnostic_states.emplace(
                                        ordinal,
                                        std::move(diagnostic_state));
                                if (inserted)
                                {
                                    unified_diagnostics_reused_cell_state +=
                                        1.0;
                                    unified_diagnostics_built_diagnostic_state_count +=
                                        1.0;
                                }
                                else
                                {
                                    it->second =
                                        QpointStateCache::
                                            flux_diagnostic_state_from_compact_state(
                                                compact_state_cell_for_order(
                                                    order_index));
                                    unified_diagnostics_duplicate_cells += 1.0;
                                }
                            }
                            unified_diagnostics_state_seconds +=
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() -
                                    diagnostic_state_begin)
                                    .count();
                        }

                        if (solve_cell_state_cache.enabled)
                        {
                            for (int request_id = 0;
                                 request_id <
                                     chunk_qpoint_state_cache
                                         .n_build_requests();
                                 ++request_id)
                            {
                                if (use_compact_split_state)
                                {
                                    solve_cell_state_cache.store(
                                        std::move(
                                            chunk_qpoint_state_cache
                                                .mutable_compact_cell_by_index(
                                                    request_id)));
                                }
                                else
                                {
                                    solve_cell_state_cache.store(
                                        std::move(
                                            chunk_qpoint_state_cache
                                                .mutable_cell_by_index(
                                                    request_id)));
                                }
                            }
                            peak_cell_state_cache_bytes =
                                std::max(
                                    peak_cell_state_cache_bytes,
                                    solve_cell_state_cache.memory_bytes());
                        }

                        peak_process_rss_bytes =
                            std::max(
                                peak_process_rss_bytes,
                                finite_element::detail::
                                    current_process_rss_bytes());
                    }
                }

                {
                    const auto thread_policy =
                        select_local_error_thread_policy_(
                            current_tile_size,
                            dense_blocks_bytes_(dense_blocks) +
                                peak_cell_state_bytes + peak_tables_bytes +
                                estimate_scalar_reduction_basis_bytes_());
                    record_local_error_thread_policy_(timing, thread_policy);
                    add_local_error_thread_policy_stats_(
                        timing_stats,
                        thread_policy);
                    peak_selected_threads =
                        std::max(
                            peak_selected_threads,
                            thread_policy.selected_threads);
                    if (thread_policy.selected_threads > 1)
                        last_solve_all_patches_used_openmp_ = true;

                    auto batch_stats =
                        patch_solve_batch_stats_(
                            tile_begin,
                            tile_end,
                            thread_policy.selected_threads);
                    using ExplicitSolveWorkspace =
                        finite_element::assembly::error_system::
                            DenseLocalErrorExplicitSolveWorkspace2D<Backend>;
                    const auto solve_workspace_alloc_start =
                        std::chrono::steady_clock::now();
                    std::vector<ExplicitSolveWorkspace> solve_workspaces(
                        static_cast<std::size_t>(
                            std::max(1, thread_policy.selected_threads)));
                    batch_stats.workspace_allocation_seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            solve_workspace_alloc_start)
                            .count();
                    std::vector<double> reduced_transform_seconds_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> factorization_seconds_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> solve_apply_seconds_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> current_dense_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_scalar_dense_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_fallback_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_residual_fail_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> current_dimension_sum_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> current_dimension_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_dimension_sum_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_dimension_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> writeback_seconds_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    const auto solve_one_patch =
                        [&](int local_patch_id)
                        {
                            const int patch_id = tile_begin + local_patch_id;
                            double local_transform_seconds = 0.0;
                            double local_factorization_seconds = 0.0;
                            double local_solve_apply_seconds = 0.0;
                            double local_current_dense_count = 0.0;
                            double local_reduced_scalar_dense_count = 0.0;
                            double local_reduced_fallback_count = 0.0;
                            double local_reduced_residual_fail_count = 0.0;
                            double local_current_dimension_sum = 0.0;
                            double local_current_dimension_count = 0.0;
                            double local_reduced_dimension_sum = 0.0;
                            double local_reduced_dimension_count = 0.0;
                            auto split =
                                [&]() {
                                    ExplicitSolveWorkspace* workspace =
                                        nullptr;
                                    if (use_local_error_patch_solve_workspace_())
                                    {
                                        int workspace_id = 0;
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
                                        if (thread_policy.selected_threads > 1)
                                            workspace_id = omp_get_thread_num();
#endif
                                        workspace =
                                            &solve_workspaces[
                                                static_cast<std::size_t>(
                                                    workspace_id)];
                                    }
                                    return
                                        solve_dense_local_error_patch_with_selected_solver_(
                                            dense_blocks[
                                                static_cast<std::size_t>(
                                                    local_patch_id)],
                                            scalar_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)],
                                            workspace,
                                            nullptr,
                                            zero_tol,
                                            &local_transform_seconds,
                                            &local_factorization_seconds,
                                            &local_solve_apply_seconds,
                                            &local_current_dense_count,
                                            &local_reduced_scalar_dense_count,
                                            &local_reduced_fallback_count,
                                            &local_reduced_residual_fail_count,
                                            &local_current_dimension_sum,
                                            &local_current_dimension_count,
                                            &local_reduced_dimension_sum,
                                            &local_reduced_dimension_count);
                                }();
                            reduced_transform_seconds_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_transform_seconds;
                            factorization_seconds_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_factorization_seconds;
                            solve_apply_seconds_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_solve_apply_seconds;
                            current_dense_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_current_dense_count;
                            reduced_scalar_dense_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_scalar_dense_count;
                            reduced_fallback_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_fallback_count;
                            reduced_residual_fail_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_residual_fail_count;
                            current_dimension_sum_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_current_dimension_sum;
                            current_dimension_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_current_dimension_count;
                            reduced_dimension_sum_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_dimension_sum;
                            reduced_dimension_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_dimension_count;

                            const auto writeback_begin =
                                std::chrono::steady_clock::now();
                            flux_functions_[
                                static_cast<std::size_t>(patch_id)]
                                .update_coefficients(split.lambda);
                            scalar_functions_[
                                static_cast<std::size_t>(patch_id)]
                                .update_coefficients(split.u);
                            writeback_seconds_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() -
                                    writeback_begin)
                                    .count();
                        };
                    const auto patch_solve_order =
                        patch_solve_order_(
                            tile_begin,
                            tile_end,
                            thread_policy.selected_threads);

                    {
                        finite_element::assembly::error_system::
                            LocalErrorProblemScopedTiming timer(
                                &timing_stats.solve_patch_systems_seconds);
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
                        if (thread_policy.selected_threads > 1)
                        {
                            std::exception_ptr error;
                            timing_stats.parallel_region_count += 1.0;
#pragma omp parallel for num_threads(thread_policy.selected_threads) schedule(static)
                            for (int solve_index = 0;
                                 solve_index <
                                     static_cast<int>(
                                         patch_solve_order.size());
                                 ++solve_index)
                            {
                                try
                                {
                                    solve_one_patch(
                                        patch_solve_order[
                                            static_cast<std::size_t>(
                                                solve_index)]);
                                }
                                catch (...)
                                {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                                    {
                                        if (!error)
                                            error = std::current_exception();
                                    }
                                }
                            }
                            finite_element::assembly::detail::
                                rethrow_parallel_exception(error);
                        }
                        else
#endif
                        {
                            for (const int local_patch_id :
                                 patch_solve_order)
                            {
                                solve_one_patch(local_patch_id);
                            }
                        }
                    }
                    for (const double seconds :
                         reduced_transform_seconds_by_patch)
                    {
                        timing_stats.reduced_basis_transform_seconds +=
                            seconds;
                    }
                    for (const double seconds :
                         factorization_seconds_by_patch)
                    {
                        batch_stats.factorization_seconds += seconds;
                    }
                    for (const double seconds : solve_apply_seconds_by_patch)
                    {
                        batch_stats.solve_apply_seconds += seconds;
                    }
                    for (const double count : current_dense_count_by_patch)
                        timing_stats.patch_solver_current_dense_count += count;
                    for (const double count : reduced_scalar_dense_count_by_patch)
                        timing_stats.patch_solver_reduced_scalar_dense_count +=
                            count;
                    for (const double count : reduced_fallback_count_by_patch)
                        timing_stats.patch_solver_reduced_fallback_count +=
                            count;
                    for (const double count : reduced_residual_fail_count_by_patch)
                        timing_stats.patch_solver_reduced_residual_fail_count +=
                            count;
                    for (const double value : current_dimension_sum_by_patch)
                        timing_stats.patch_solver_current_dimension_sum +=
                            value;
                    for (const double value : current_dimension_count_by_patch)
                        timing_stats.patch_solver_current_dimension_count +=
                            value;
                    for (const double value : reduced_dimension_sum_by_patch)
                        timing_stats.patch_solver_reduced_dimension_sum +=
                            value;
                    for (const double value : reduced_dimension_count_by_patch)
                        timing_stats.patch_solver_reduced_dimension_count +=
                            value;
                    timing_stats.patch_solver_mode =
                        patch_solver_mode_code_();
                    for (const double seconds : writeback_seconds_by_patch)
                    {
                        timing_stats.coefficient_writeback_seconds +=
                            seconds;
                    }
                    if (batch_stats.dense_solver_workspace_bytes == 0)
                    {
                        std::size_t workspace_bytes = 0;
                        for (const auto& workspace : solve_workspaces)
                            workspace_bytes += workspace.estimated_memory_bytes();
                        batch_stats.dense_solver_workspace_bytes =
                            workspace_bytes;
                    }
                    peak_tile_workspace_bytes =
                        std::max(
                            peak_tile_workspace_bytes,
                            dense_blocks_bytes_(dense_blocks) +
                                static_cast<std::size_t>(
                                    batch_stats.dense_solver_workspace_bytes));
                    add_patch_solve_batch_stats_(
                        timing_stats,
                        batch_stats);
                }

                if (use_unified_streaming_flux_diagnostics)
                {
                    for (const auto& plan_cell : tile_plan.cells)
                    {
                        const int ordinal =
                            plan_cell.active_slab_cell_ordinal;
                        if (ordinal < 0 ||
                            ordinal >=
                                static_cast<int>(
                                    unified_diagnostics_cell_done.size()))
                        {
                            ++unified_diagnostics_missing_cells;
                            continue;
                        }
                        if (unified_diagnostics_cell_done[
                                static_cast<std::size_t>(ordinal)] != 0)
                        {
                            ++unified_diagnostics_duplicate_cells;
                            continue;
                        }

                        const bool have_representative =
                            plan_cell.representative_patch_id >= 0 &&
                            plan_cell.representative_patch_cell_index >= 0;
                        const bool all_memberships_solved =
                            plan_cell.max_patch_id < tile_end;

                        if (!have_representative || !all_memberships_solved)
                            continue;

                        const auto state_it =
                            tile_flux_diagnostic_states.find(ordinal);
                        if (state_it == tile_flux_diagnostic_states.end())
                        {
                            ++unified_diagnostics_missing_cells;
                            continue;
                        }

                        const auto eval_start =
                            std::chrono::steady_clock::now();
                        const auto [local_flux, local_residual] =
                            compute_streaming_flux_diagnostics_contribution_from_compact_state_<
                            QpointStateCache>(
                            state_it->second,
                            unified_diagnostics_qpoint_eval_seconds);
                        unified_diagnostics_eval_seconds +=
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now() -
                                eval_start)
                                .count();
                        const auto ordinal_index =
                            static_cast<std::size_t>(ordinal);
                        unified_diagnostics_source_cell_id[ordinal_index] =
                            state_it->second.source_cell_id;
                        unified_diagnostics_flux_contribution[ordinal_index] =
                            local_flux;
                        unified_diagnostics_residual_contribution[
                            ordinal_index] = local_residual;
                        unified_diagnostics_cell_done[
                            ordinal_index] = 1;
                        ++unified_diagnostics_finalized_cells;
                    }
                }

                if (solve_cell_state_cache.mode_code == 1)
                {
                    for (const auto& plan_cell : tile_plan.cells)
                        solve_cell_state_cache.evict(
                            plan_cell.active_slab_cell_ordinal);
                }
                else
                {
                    solve_cell_state_cache.evict_expired(
                        ordinals_expiring_after_tile[
                            static_cast<std::size_t>(tile_id)]);
                }
                peak_cell_state_cache_bytes =
                    std::max(
                        peak_cell_state_cache_bytes,
                        solve_cell_state_cache.memory_bytes());

                peak_process_rss_bytes =
                    std::max(
                        peak_process_rss_bytes,
                        finite_element::detail::current_process_rss_bytes());
            }

            if (use_unified_streaming_flux_diagnostics)
            {
                int unfinalized_cells = 0;
                for (const char done : unified_diagnostics_cell_done)
                {
                    if (done == 0)
                        ++unfinalized_cells;
                }
                unified_diagnostics_missing_cells +=
                    static_cast<double>(unfinalized_cells);
                if (unified_diagnostics_finalized_cells !=
                    static_cast<int>(active_slab_cells.size()))
                {
                    throw std::runtime_error(
                        "TimeSlabEquilibratedFluxReconstruction2plus1d: unified streaming diagnostics did not finalize every active slab cell.");
                }
                const auto accumulation_start =
                    std::chrono::steady_clock::now();
                for (std::size_t ordinal = 0;
                     ordinal < unified_diagnostics_cell_done.size();
                     ++ordinal)
                {
                    const int source_cell_id =
                        unified_diagnostics_source_cell_id[ordinal];
                    if (source_cell_id < 0)
                    {
                        throw std::runtime_error(
                            "TimeSlabEquilibratedFluxReconstruction2plus1d: unified streaming diagnostics finalized a cell without a source-cell id.");
                    }
                    detail::add_to_map(
                        unified_streaming_diagnostics.by_source_cell_flux,
                        source_cell_id,
                        unified_diagnostics_flux_contribution[ordinal]);
                    detail::add_to_map(
                        unified_streaming_diagnostics.by_source_cell_residual,
                        source_cell_id,
                        unified_diagnostics_residual_contribution[ordinal]);
                }
                unified_diagnostics_accumulation_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        accumulation_start)
                        .count();
                detail::require_nonnegative_cellwise_map(
                    unified_streaming_diagnostics.by_source_cell_flux,
                    "TimeSlabEquilibratedFluxReconstruction2plus1d unified streaming flux aggregation");
                detail::require_nonnegative_cellwise_map(
                    unified_streaming_diagnostics.by_source_cell_residual,
                    "TimeSlabEquilibratedFluxReconstruction2plus1d unified streaming residual aggregation");
                last_fused_flux_diagnostics_ =
                    std::move(unified_streaming_diagnostics);
                last_fused_flux_diagnostics_runtime_stats_ =
                    FluxDiagnosticsRuntimeStats{
                        unified_diagnostics_state_seconds +
                            unified_diagnostics_eval_seconds,
                        static_cast<double>(
                            unified_diagnostics_finalized_cells) *
                            static_cast<double>(
                                QpointStateCache::n_quadrature_points_v),
                        unified_diagnostics_reused_cell_state,
                        unified_diagnostics_extra_cell_state_rebuilds};
            }

            const auto dense_solve_peak_temporary_bytes =
                dense_solve_peak_temporary_bytes_();
            const auto reduction_basis_bytes =
                estimate_scalar_reduction_basis_bytes_() +
                estimate_concurrent_scalar_reduction_basis_bytes_(
                    peak_selected_threads);
            const auto patch_solutions_bytes =
                static_cast<std::size_t>(peak_selected_threads) *
                sequential_patch_solution_peak_bytes_();
            const auto per_thread_context_bytes =
                estimate_per_thread_context_bytes_(peak_selected_threads);
            const auto concurrent_solve_temporary_bytes =
                static_cast<std::size_t>(peak_selected_threads) *
                dense_solve_peak_temporary_bytes;
            const auto dimensions = patch_dimension_counters_();
            cell_state_cache_evictions = solve_cell_state_cache.evictions;
            const auto estimated_total_live_bytes =
                peak_dense_blocks_bytes + patch_solutions_bytes +
                peak_tables_bytes + peak_cell_state_bytes +
                peak_cell_state_cache_bytes +
                reduction_basis_bytes + per_thread_context_bytes +
                concurrent_solve_temporary_bytes;
            int distinct_state_constructed_cells = 0;
            int max_state_builds_per_cell = 0;
            for (const int count : state_build_counts)
            {
                if (count > 0)
                    ++distinct_state_constructed_cells;
                max_state_builds_per_cell =
                    std::max(max_state_builds_per_cell, count);
            }
            const double current_solve_global_unique_slab_cells =
                static_cast<double>(active_slab_cells.size());
            const double current_solve_state_build_requests =
                static_cast<double>(total_unique_slab_cells);
            const double current_solve_actual_state_constructions =
                static_cast<double>(total_cell_state_rebuilds);
            const double current_solve_cross_tile_rebuilds =
                current_solve_actual_state_constructions >=
                        current_solve_global_unique_slab_cells
                    ? current_solve_actual_state_constructions -
                          current_solve_global_unique_slab_cells
                    : 0.0;
            const double current_solve_builds_per_global_cell_mean =
                current_solve_global_unique_slab_cells > 0.0
                    ? current_solve_actual_state_constructions /
                          current_solve_global_unique_slab_cells
                    : 0.0;
            const double current_solve_max_builds_per_cell =
                static_cast<double>(max_state_builds_per_cell);
            int current_solve_invariant_failures = 0;
            if (current_solve_global_unique_slab_cells <= 0.0)
                ++current_solve_invariant_failures;
            if (current_solve_actual_state_constructions <
                current_solve_global_unique_slab_cells)
                ++current_solve_invariant_failures;
            if (std::abs(
                    current_solve_cross_tile_rebuilds -
                    (current_solve_actual_state_constructions -
                     current_solve_global_unique_slab_cells)) > 0.5)
                ++current_solve_invariant_failures;
            if (current_solve_global_unique_slab_cells > 0.0)
            {
                const double expected_mean =
                    current_solve_actual_state_constructions /
                    current_solve_global_unique_slab_cells;
                if (std::abs(
                        current_solve_builds_per_global_cell_mean -
                        expected_mean) > 1.0e-12)
                    ++current_solve_invariant_failures;
            }
            if (current_solve_max_builds_per_cell + 1.0e-12 <
                current_solve_builds_per_global_cell_mean)
                ++current_solve_invariant_failures;
            const double current_solve_time_per_state_seconds =
                current_solve_actual_state_constructions > 0.0
                    ? timing_stats.chunk_cell_state_construction_seconds /
                          current_solve_actual_state_constructions
                    : 0.0;
            auto finalize_cache_simulation =
                [&](CacheSimulationResult& result)
            {
                result.expected_cell_state_saved_seconds =
                    result.constructions_avoided *
                    current_solve_time_per_state_seconds;
            };
            finalize_cache_simulation(cache_sim_off);
            finalize_cache_simulation(cache_sim_tile_local);
            finalize_cache_simulation(cache_sim_lru_64);
            finalize_cache_simulation(cache_sim_lru_256);
            finalize_cache_simulation(cache_sim_lru_1024);
            finalize_cache_simulation(cache_sim_lru_4096);
            finalize_cache_simulation(cache_sim_lru_8192);
            finalize_cache_simulation(cache_sim_lifetime_window);
            finalize_cache_simulation(cache_sim_full_if_fits);
            timing_stats.patch_count =
                static_cast<double>(patch_count);
            timing_stats.tile_size =
                static_cast<double>(tile_size);
            timing_stats.tile_count =
                static_cast<double>(tile_count);
            timing_stats.cell_chunk_size =
                static_cast<double>(cell_chunk_size);
            timing_stats.chunk_count =
                static_cast<double>(total_chunk_count);
            timing_stats.tile_plan_chunk_count =
                static_cast<double>(total_chunk_count);
            timing_stats.active_slab_cells =
                static_cast<double>(active_slab_cells.size());
            timing_stats.global_unique_slab_cells =
                current_solve_global_unique_slab_cells;
            timing_stats.total_state_build_requests =
                current_solve_state_build_requests;
            timing_stats.actual_state_constructions =
                current_solve_actual_state_constructions;
            timing_stats.distinct_state_constructed_cells =
                static_cast<double>(distinct_state_constructed_cells);
            timing_stats.cross_tile_state_rebuilds =
                current_solve_cross_tile_rebuilds;
            timing_stats.state_builds_per_global_cell_mean =
                current_solve_builds_per_global_cell_mean;
            timing_stats.state_builds_per_global_cell_max =
                current_solve_max_builds_per_cell;
            timing_stats.cell_patch_memberships =
                static_cast<double>(global_patch_cell_memberships);
            timing_stats.tiles_per_cell_mean =
                active_slab_cells.empty()
                    ? 0.0
                    : total_tiles_per_cell /
                          static_cast<double>(active_slab_cells.size());
            timing_stats.tiles_per_cell_max =
                static_cast<double>(max_tiles_per_cell);
            timing_stats.cells_spanning_multiple_tiles =
                static_cast<double>(cells_spanning_multiple_tiles);
            timing_stats.state_bytes_per_cell =
                static_cast<double>(
                    use_compact_split_state
                        ? sizeof(typename QpointStateCache::CompactCellData)
                        : sizeof(typename QpointStateCache::CellData));
            timing_stats.estimated_full_cache_bytes =
                static_cast<double>(estimated_full_cell_state_cache_bytes);
            const double compact_bytes_per_cell =
                timing_stats.operator_state_bytes_per_cell +
                timing_stats.rhs_state_bytes_per_cell +
                timing_stats.flux_diagnostic_state_bytes_per_cell;
            if (compact_bytes_per_cell > 0.0)
            {
                timing_stats.estimated_compact_full_cache_gib =
                    current_solve_global_unique_slab_cells *
                    compact_bytes_per_cell /
                    (1024.0 * 1024.0 * 1024.0);
                timing_stats.estimated_lifetime_window_cache_mb =
                    static_cast<double>(
                        lifetime_metrics.lifetime_window_peak_live_cells) *
                    compact_bytes_per_cell /
                    (1024.0 * 1024.0);
            }
            timing_stats.configured_cache_limit_bytes =
                static_cast<double>(cell_state_cache_limit_bytes);
            timing_stats.cell_state_cache_mode =
                static_cast<double>(solve_cell_state_cache.mode_code);
            timing_stats.cell_state_cache_budget_mb =
                local_error_cell_state_cache_budget_mb_;
            timing_stats.cell_state_cache_entries =
                static_cast<double>(solve_cell_state_cache.peak_entries);
            timing_stats.cell_state_cache_memory_mb =
                static_cast<double>(peak_cell_state_cache_bytes) /
                (1024.0 * 1024.0);
            timing_stats.cell_state_cache_hits =
                static_cast<double>(cell_state_cache_hits);
            timing_stats.cell_state_cache_misses =
                static_cast<double>(cell_state_cache_misses);
            timing_stats.cell_state_cache_evictions =
                static_cast<double>(cell_state_cache_evictions);
            timing_stats.cell_state_cache_hit_rate =
                cell_state_cache_hits + cell_state_cache_misses > 0
                    ? static_cast<double>(cell_state_cache_hits) /
                          static_cast<double>(
                              cell_state_cache_hits +
                              cell_state_cache_misses)
                    : 0.0;
            timing_stats.cell_state_cache_cross_tile_rebuilds_avoided =
                static_cast<double>(cell_state_cache_hits);
            timing_stats.cell_state_cache_stale_state_detected_count =
                static_cast<double>(
                    cell_state_cache_stale_state_detected_count);
            timing_stats.flux_diagnostics_mode =
                flux_diagnostics_mode_code;
            timing_stats.flux_diagnostics_streaming_reuse_used =
                use_unified_streaming_flux_diagnostics ? 1.0 : 0.0;
            timing_stats.flux_diagnostics_standalone_used =
                use_unified_streaming_flux_diagnostics ? 0.0 : 1.0;
            timing_stats.flux_diagnostics_fallback_reason_code =
                static_cast<double>(
                    flux_diagnostics_fallback_reason_code);
            timing_stats.flux_diagnostics_reused_rhs_state_count =
                unified_diagnostics_reused_cell_state;
            timing_stats.flux_diagnostics_built_diagnostic_state_count =
                unified_diagnostics_built_diagnostic_state_count;
            timing_stats.flux_diagnostics_rebuilt_state_count =
                unified_diagnostics_extra_cell_state_rebuilds;
            timing_stats
                .flux_diagnostics_monolithic_cell_data_constructed_count = 0.0;
            timing_stats.flux_diagnostics_finalized_active_slab_cells =
                static_cast<double>(unified_diagnostics_finalized_cells);
            timing_stats.flux_diagnostics_missing_active_slab_cells =
                unified_diagnostics_missing_cells;
            timing_stats.flux_diagnostics_duplicate_active_slab_cells =
                unified_diagnostics_duplicate_cells;
            timing_stats.flux_diagnostics_cells_visited =
                static_cast<double>(unified_diagnostics_finalized_cells);
            timing_stats.flux_diagnostics_qpoints_visited =
                static_cast<double>(unified_diagnostics_finalized_cells) *
                static_cast<double>(QpointStateCache::n_quadrature_points_v);
            timing_stats.flux_diagnostics_streaming_state_build_seconds =
                unified_diagnostics_state_seconds;
            timing_stats.flux_diagnostics_streaming_qpoint_eval_seconds =
                unified_diagnostics_qpoint_eval_seconds;
            timing_stats.flux_diagnostics_streaming_accumulation_seconds =
                unified_diagnostics_accumulation_seconds;
            timing_stats.flux_diagnostics_streaming_seconds =
                unified_diagnostics_state_seconds +
                unified_diagnostics_eval_seconds;
            timing_stats.flux_diagnostics_standalone_seconds = 0.0;
            timing_stats.worker_context_memory_mb =
                static_cast<double>(per_thread_context_bytes) /
                (1024.0 * 1024.0);

            timing_stats.rt_basis_qpoint_fills +=
                total_rt_basis_qpoint_fills;
            timing_stats.ab_diffusion_tensor_evaluations +=
                total_ab_diffusion_tensor_evaluations;
            timing_stats.ab_scalar_basis_qpoint_fills +=
                total_ab_scalar_basis_qpoint_fills;
            timing_stats.rhs_diffusion_tensor_evaluations +=
                total_rhs_diffusion_tensor_evaluations;
            timing_stats.rhs_lambda_gradient_evaluations +=
                total_rhs_lambda_gradient_evaluations;
            timing_stats.rhs_u_gradient_evaluations +=
                total_rhs_u_gradient_evaluations;
            timing_stats.qpoint_state_scalar_basis_qpoint_fills +=
                total_qpoint_state_scalar_basis_qpoint_fills;
            timing_stats.qpoint_state_partition_of_unity_qpoint_fills +=
                total_qpoint_state_partition_of_unity_qpoint_fills;
            timing_stats.qpoint_state_patch_equivalent_scalar_basis_qpoint_fills +=
                total_qpoint_state_patch_equivalent_scalar_basis_qpoint_fills;
            timing_stats.qpoint_state_scalar_basis_qpoint_fills_avoided +=
                total_qpoint_state_scalar_basis_qpoint_fills_avoided;
            timing_stats.qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills +=
                total_qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills;
            timing_stats.qpoint_state_partition_of_unity_qpoint_fills_avoided +=
                total_qpoint_state_partition_of_unity_qpoint_fills_avoided;

            timing.add(
                "time_slab.local_error_solves.streaming_used",
                1.0);
            timing.add(
                "time_slab.local_error_solves.unified_streaming_used",
                use_unified_streaming_flux_diagnostics ? 1.0 : 0.0);
            timing.add(
                "time_slab.local_error_solves.unified_streaming_used.count",
                use_unified_streaming_flux_diagnostics ? 1.0 : 0.0);
            timing.add(
                "flux_diagnostics.mode",
                timing_stats.flux_diagnostics_mode);
            timing.add(
                "flux_diagnostics.streaming_reuse_used",
                timing_stats.flux_diagnostics_streaming_reuse_used);
            timing.add(
                "flux_diagnostics.standalone_used",
                timing_stats.flux_diagnostics_standalone_used);
            timing.add(
                "flux_diagnostics.fallback_reason_code",
                timing_stats.flux_diagnostics_fallback_reason_code);
            timing.add(
                "flux_diagnostics.reused_rhs_state_count",
                timing_stats.flux_diagnostics_reused_rhs_state_count);
            timing.add(
                "flux_diagnostics.built_diagnostic_state_count",
                timing_stats.flux_diagnostics_built_diagnostic_state_count);
            timing.add(
                "flux_diagnostics.rebuilt_state_count",
                timing_stats.flux_diagnostics_rebuilt_state_count);
            timing.add(
                "flux_diagnostics.monolithic_cell_data_constructed_count",
                timing_stats
                    .flux_diagnostics_monolithic_cell_data_constructed_count);
            timing.add(
                "flux_diagnostics.finalized_active_slab_cells",
                timing_stats.flux_diagnostics_finalized_active_slab_cells);
            timing.add(
                "flux_diagnostics.missing_active_slab_cells",
                timing_stats.flux_diagnostics_missing_active_slab_cells);
            timing.add(
                "flux_diagnostics.duplicate_active_slab_cells",
                timing_stats.flux_diagnostics_duplicate_active_slab_cells);
            timing.add(
                "flux_diagnostics.cells_visited",
                timing_stats.flux_diagnostics_cells_visited);
            timing.add(
                "flux_diagnostics.qpoints_visited",
                timing_stats.flux_diagnostics_qpoints_visited);
            timing.add(
                "flux_diagnostics.streaming_wall",
                timing_stats.flux_diagnostics_streaming_seconds);
            timing.add(
                "flux_diagnostics.streaming_state_build_wall",
                timing_stats.flux_diagnostics_streaming_state_build_seconds);
            timing.add(
                "flux_diagnostics.streaming_qpoint_eval_wall",
                timing_stats.flux_diagnostics_streaming_qpoint_eval_seconds);
            timing.add(
                "flux_diagnostics.streaming_accumulation_wall",
                timing_stats.flux_diagnostics_streaming_accumulation_seconds);
            timing.add(
                "flux_diagnostics.standalone_wall",
                timing_stats.flux_diagnostics_standalone_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_tiling_used",
                1.0);
            timing.add(
                "time_slab.local_error_solves.cell_state_chunking_used",
                1.0);
            timing.add(
                "time_slab.local_error_solves.patch_tile_size.count",
                static_cast<double>(tile_size));
            timing.add("local_error.tile_size", timing_stats.tile_size);
            timing.add(
                "time_slab.local_error_solves.patch_tile_count.count",
                static_cast<double>(tile_count));
            timing.add("local_error.tile_count", timing_stats.tile_count);
            timing.add(
                "time_slab.local_error_solves.streaming_tile_count",
                static_cast<double>(tile_count));
            timing.add(
                "time_slab.local_error_solves.streaming_tile_count.count",
                static_cast<double>(tile_count));
            timing.add(
                "time_slab.local_error_solves.cell_chunk_size.count",
                static_cast<double>(cell_chunk_size));
            timing.add("local_error.chunk_size", timing_stats.cell_chunk_size);
            timing.add(
                "time_slab.local_error_solves.cell_chunk_count.count",
                static_cast<double>(total_chunk_count));
            timing.add("local_error.chunk_count", timing_stats.chunk_count);
            timing.add(
                "time_slab.local_error_solves.chunk_count.count",
                static_cast<double>(total_chunk_count));
            timing.add(
                "time_slab.local_error_solves.streaming_cell_chunk_count",
                static_cast<double>(total_chunk_count));
            timing.add(
                "time_slab.local_error_solves.streaming_total_cell_state_rebuilds",
                static_cast<double>(total_cell_state_rebuilds));
            timing.add(
                "time_slab.local_error_solves.streaming_total_cell_state_rebuilds.count",
                static_cast<double>(total_cell_state_rebuilds));
            timing.add(
                "time_slab.local_error_solves.cell_states_constructed.count",
                static_cast<double>(total_cell_state_rebuilds));
            timing.add(
                "local_error.n_cell_states_constructed",
                static_cast<double>(total_cell_state_rebuilds));
            timing.add(
                "local_error.patch_count",
                timing_stats.patch_count);
            timing.add(
                "local_error.parallel_region_count",
                timing_stats.parallel_region_count);
            timing.add(
                "local_error.active_slab_cells",
                timing_stats.active_slab_cells);
            timing.add(
                "local_error.global_unique_slab_cells",
                timing_stats.global_unique_slab_cells);
            timing.add(
                "local_error.total_state_build_requests",
                timing_stats.total_state_build_requests);
            timing.add(
                "local_error.actual_state_constructions",
                timing_stats.actual_state_constructions);
            timing.add(
                "local_error.distinct_state_constructed_cells",
                timing_stats.distinct_state_constructed_cells);
            timing.add(
                "local_error.cross_tile_state_rebuilds",
                timing_stats.cross_tile_state_rebuilds);
            timing.add(
                "local_error.state_builds_per_global_cell_mean",
                timing_stats.state_builds_per_global_cell_mean);
            timing.add(
                "local_error.state_builds_per_global_cell_max",
                timing_stats.state_builds_per_global_cell_max);
            timing.add(
                "local_error.cell_patch_memberships",
                timing_stats.cell_patch_memberships);
            timing.add(
                "local_error.tiles_per_cell_mean",
                timing_stats.tiles_per_cell_mean);
            timing.add(
                "local_error.tiles_per_cell_max",
                timing_stats.tiles_per_cell_max);
            timing.add(
                "local_error.cells_spanning_multiple_tiles",
                timing_stats.cells_spanning_multiple_tiles);
            timing.add(
                "local_error.current_solve.global_unique_slab_cells",
                current_solve_global_unique_slab_cells);
            timing.add(
                "local_error.current_solve.state_build_requests",
                current_solve_state_build_requests);
            timing.add(
                "local_error.current_solve.actual_state_constructions",
                current_solve_actual_state_constructions);
            timing.add(
                "local_error.current_solve.cross_tile_rebuilds",
                current_solve_cross_tile_rebuilds);
            timing.add(
                "local_error.current_solve.builds_per_global_cell_mean",
                current_solve_builds_per_global_cell_mean);
            timing.add(
                "local_error.current_solve.max_builds_per_cell",
                current_solve_max_builds_per_cell);
            timing.add(
                "local_error.current_solve.cells_used_once",
                static_cast<double>(lifetime_metrics.cells_used_once));
            timing.add(
                "local_error.current_solve.cells_used_multiple_tiles",
                static_cast<double>(
                    lifetime_metrics.cells_used_multiple_tiles));
            timing.add(
                "local_error.current_solve.cells_spanning_multiple_tiles",
                timing_stats.cells_spanning_multiple_tiles);
            timing.add(
                "local_error.current_solve.mean_tiles_per_cell",
                lifetime_metrics.mean_tiles_per_cell);
            timing.add(
                "local_error.current_solve.max_tiles_per_cell",
                timing_stats.tiles_per_cell_max);
            timing.add(
                "local_error.current_solve.cell_state_bytes_per_cell",
                static_cast<double>(
                    use_compact_split_state
                        ? sizeof(typename QpointStateCache::CompactCellData)
                        : sizeof(typename QpointStateCache::CellData)));
            timing.add(
                "local_error.current_solve.estimated_full_cache_bytes",
                static_cast<double>(estimated_full_cell_state_cache_bytes));
            timing.add(
                "local_error.current_solve.invariant_failure_count",
                static_cast<double>(current_solve_invariant_failures));
            timing.add(
                "local_error.current_solve.state_time_per_construction",
                current_solve_time_per_state_seconds);
            timing.add(
                "local_error.current_solve.cells_reused_non_adjacent_tiles",
                static_cast<double>(
                    lifetime_metrics
                        .cells_reused_across_non_adjacent_tiles));
            timing.add(
                "local_error.current_solve.mean_incident_patches_per_cell",
                lifetime_metrics.mean_incident_patches_per_cell);
            timing.add(
                "local_error.current_solve.max_incident_patches_per_cell",
                static_cast<double>(
                    lifetime_metrics.max_incident_patches_per_cell));
            timing.add(
                "local_error.current_solve.lifetime_window_peak_live_cells",
                static_cast<double>(
                    lifetime_metrics.lifetime_window_peak_live_cells));
            timing.add(
                "local_error.cumulative.local_error_solve_count",
                1.0);
            timing.add(
                "local_error.cumulative.global_unique_slab_cell_appearances",
                current_solve_global_unique_slab_cells);
            timing.add(
                "local_error.cumulative.state_build_requests",
                current_solve_state_build_requests);
            timing.add(
                "local_error.cumulative.actual_state_constructions",
                current_solve_actual_state_constructions);
            timing.add(
                "local_error.cumulative.cross_tile_rebuilds",
                current_solve_cross_tile_rebuilds);
            timing.add(
                "local_error.cumulative.max_builds_per_cell_over_all_solves",
                current_solve_max_builds_per_cell);
            timing.add(
                "local_error.cumulative.invariant_failure_count",
                static_cast<double>(current_solve_invariant_failures));
            timing.add(
                "time_slab.local_error_solves.repeated_cell_state_constructions.count",
                timing_stats.cross_tile_state_rebuilds);
            timing.add(
                "local_error.repeated_cell_state_constructions",
                timing_stats.cross_tile_state_rebuilds);
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_hits.count",
                static_cast<double>(cell_state_cache_hits));
            timing.add(
                "local_error.n_cell_state_cache_hits",
                static_cast<double>(cell_state_cache_hits));
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_misses.count",
                static_cast<double>(cell_state_cache_misses));
            timing.add(
                "local_error.n_cell_state_cache_misses",
                static_cast<double>(cell_state_cache_misses));
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_entries.count",
                static_cast<double>(solve_cell_state_cache.peak_entries));
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_enabled.count",
                solve_cell_state_cache.enabled ? 1.0 : 0.0);
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_disabled_by_memory.count",
                solve_cell_state_cache.enabled ? 0.0 : 1.0);
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_memory_mb",
                static_cast<double>(peak_cell_state_cache_bytes) /
                    (1024.0 * 1024.0));
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_estimated_full_bytes",
                static_cast<double>(estimated_full_cell_state_cache_bytes));
            timing.add(
                "time_slab.local_error_solves.cell_state_cache_limit_bytes",
                static_cast<double>(cell_state_cache_limit_bytes));
            timing.add(
                "local_error.old_cache.full_cache_estimate_bytes",
                static_cast<double>(estimated_full_cell_state_cache_bytes));
            timing.add(
                "local_error.old_cache.configured_limit_bytes",
                static_cast<double>(
                    configured_cell_state_cache_limit_bytes));
            timing.add(
                "local_error.old_cache.effective_limit_bytes",
                static_cast<double>(cell_state_cache_limit_bytes));
            timing.add(
                "local_error.old_cache.enabled",
                old_use_solve_cell_state_cache ? 1.0 : 0.0);
            timing.add(
                "local_error.old_cache.disabled_reason",
                old_use_solve_cell_state_cache ? 0.0 : 1.0);
            timing.add(
                "local_error.old_cache.all_or_nothing_policy",
                1.0);
            timing.add(
                "local_error.old_cache.hard_cap_bytes",
                static_cast<double>(base_cell_state_cache_limit_bytes));

            auto record_cache_simulation =
                [&](const std::string& name,
                    const CacheSimulationResult& result)
            {
                const std::string prefix =
                    "local_error.cache_sim." + name + ".";
                timing.add(prefix + "hits", result.hits);
                timing.add(prefix + "misses", result.misses);
                timing.add(prefix + "evictions", result.evictions);
                timing.add(prefix + "hit_rate", result.hit_rate);
                timing.add(
                    prefix + "state_constructions_avoided",
                    result.constructions_avoided);
                timing.add(prefix + "memory_used_bytes", result.memory_bytes);
                timing.add(
                    prefix + "memory_used_mb",
                    result.memory_bytes / (1024.0 * 1024.0));
                timing.add(
                    prefix + "expected_cell_state_saved_wall",
                    result.expected_cell_state_saved_seconds);
                timing.add(
                    prefix + "expected_total_saved_wall",
                    result.expected_cell_state_saved_seconds);
            };
            record_cache_simulation("off", cache_sim_off);
            record_cache_simulation("tile_local", cache_sim_tile_local);
            record_cache_simulation("lru_64mib", cache_sim_lru_64);
            record_cache_simulation("lru_256mib", cache_sim_lru_256);
            record_cache_simulation("lru_1024mib", cache_sim_lru_1024);
            record_cache_simulation("lru_4096mib", cache_sim_lru_4096);
            record_cache_simulation("lru_8192mib", cache_sim_lru_8192);
            record_cache_simulation(
                "lifetime_window",
                cache_sim_lifetime_window);
            record_cache_simulation(
                "full_if_fits",
                cache_sim_full_if_fits);
            timing.add(
                "time_slab.local_error_solves.cell_chunk_peak_cells.count",
                static_cast<double>(peak_chunk_unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.lock_free_colored_cell_assembly",
                0.0);
            timing.add(
                "time_slab.local_error_solves.lock_free_cell_colors.count",
                static_cast<double>(cell_color_classes.size()));
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.requested_patch_cells.count",
                static_cast<double>(total_requested_patch_cells));
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.unique_slab_cells.count",
                static_cast<double>(total_unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.duplicate_patch_cells.count",
                static_cast<double>(
                    total_requested_patch_cells - total_unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.requested_patch_cells.count",
                static_cast<double>(total_requested_patch_cells));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.unique_slab_cells.count",
                static_cast<double>(total_unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.duplicate_patch_cells.count",
                static_cast<double>(
                    total_requested_patch_cells - total_unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.requested_patch_cells.count",
                static_cast<double>(total_requested_patch_cells));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.unique_slab_cells.count",
                static_cast<double>(total_unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.duplicate_patch_cells.count",
                static_cast<double>(
                    total_requested_patch_cells - total_unique_slab_cells));
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.diffusion_tensor_evaluations.count",
                total_ab_diffusion_tensor_evaluations);
            record_local_error_reuse_summary_(
                timing,
                total_requested_patch_cells,
                total_unique_slab_cells);
            record_local_error_global_reuse_summary_(
                timing,
                active_slab_cells);
            timing.add(
                "time_slab.local_error_solves.memory.dense_blocks_bytes",
                static_cast<double>(peak_dense_blocks_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.dense_block_peak_patch_bytes",
                static_cast<double>(peak_dense_patch_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.patch_solutions_bytes",
                static_cast<double>(patch_solutions_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.local_error_tables_bytes",
                static_cast<double>(peak_tables_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.unified_cell_state_bytes",
                static_cast<double>(peak_cell_state_bytes));
            timing.add(
                "time_slab.local_error_solves.streaming_chunk_state_bytes",
                static_cast<double>(peak_cell_state_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.streaming_chunk_state_bytes",
                static_cast<double>(peak_cell_state_bytes));
            timing.add(
                "time_slab.local_error_solves.streaming_peak_tile_workspace_bytes",
                static_cast<double>(peak_tile_workspace_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.streaming_peak_tile_workspace_bytes",
                static_cast<double>(peak_tile_workspace_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.reduction_basis_bytes",
                static_cast<double>(reduction_basis_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.operator_cache_bytes",
                0.0);
            timing.add(
                "time_slab.local_error_solves.memory.factor_cache_bytes",
                0.0);
            timing.add(
                "time_slab.local_error_solves.memory.per_thread_context_bytes",
                static_cast<double>(per_thread_context_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.dense_solve_peak_temporary_bytes",
                static_cast<double>(dense_solve_peak_temporary_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.concurrent_dense_solve_temporary_bytes",
                static_cast<double>(concurrent_solve_temporary_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.estimated_total_live_bytes",
                static_cast<double>(estimated_total_live_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.process_rss_sample_bytes",
                static_cast<double>(peak_process_rss_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_n_lambda.count",
                static_cast<double>(dimensions.max_n_lambda));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_n_u.count",
                static_cast<double>(dimensions.max_n_u));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_system_size.count",
                static_cast<double>(dimensions.max_system_size));
            if (use_unified_streaming_flux_diagnostics)
            {
                const double unified_streaming_total_seconds =
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now() -
                        unified_streaming_total_begin)
                        .count();
                const double unified_membership_plan_seconds =
                    timing_stats.active_slab_cell_collection_seconds +
                    timing_stats.cell_coloring_seconds +
                    timing_stats.tile_cell_order_construction_seconds +
                    timing_stats.cell_requests_construction_seconds +
                    timing_stats
                        .patch_cells_by_local_patch_construction_seconds;
                const double unified_flux_diagnostics_seconds =
                    unified_diagnostics_state_seconds +
                    unified_diagnostics_eval_seconds;

                timing.add(
                    "time_slab.estimator.unified_streaming.total",
                    unified_streaming_total_seconds);
                timing.add(
                    "estimator.streaming.total",
                    unified_streaming_total_seconds);
                timing.add(
                    "time_slab.estimator.unified_streaming.membership_plan",
                    unified_membership_plan_seconds);
                timing.add(
                    "estimator.streaming.membership_plan",
                    unified_membership_plan_seconds);
                timing.add(
                    "time_slab.estimator.unified_streaming.tile_assembly",
                    timing_stats.streaming_tile_assembly_seconds);
                timing.add(
                    "estimator.streaming.tile_assembly",
                    timing_stats.streaming_tile_assembly_seconds);
                timing.add(
                    "estimator.streaming.chunk_state",
                    timing_stats.streaming_chunk_state_construction_seconds);
                timing.add(
                    "time_slab.estimator.unified_streaming.patch_solves",
                    timing_stats.solve_patch_systems_seconds);
                timing.add(
                    "estimator.streaming.patch_solves",
                    timing_stats.solve_patch_systems_seconds);
                timing.add(
                    "time_slab.estimator.unified_streaming.coefficient_writeback",
                    timing_stats.coefficient_writeback_seconds);
                timing.add(
                    "estimator.streaming.coefficient_writeback",
                    timing_stats.coefficient_writeback_seconds);
                timing.add(
                    "time_slab.estimator.unified_streaming.fused_flux_diagnostics",
                    unified_flux_diagnostics_seconds);
                timing.add(
                    "estimator.streaming.fused_flux_diagnostics",
                    unified_flux_diagnostics_seconds);
                timing.add(
                    "time_slab.estimator.unified_streaming.peak_tile_bytes",
                    static_cast<double>(peak_tile_workspace_bytes));
                timing.add(
                    "estimator.streaming.peak_tile_bytes",
                    static_cast<double>(peak_tile_workspace_bytes));
                timing.add(
                    "time_slab.estimator.unified_streaming.peak_chunk_bytes",
                    static_cast<double>(peak_cell_state_bytes));
                timing.add(
                    "estimator.streaming.peak_chunk_bytes",
                    static_cast<double>(peak_cell_state_bytes));
                timing.add("time_slab.flux_diagnostics.fused_used", 1.0);
                timing.add(
                    "time_slab.flux_diagnostics.unified_streaming_used",
                    1.0);
                timing.add(
                    "time_slab.flux_diagnostics.unified_streaming_cells.count",
                    static_cast<double>(
                        unified_diagnostics_finalized_cells));
                timing.add(
                    "time_slab.flux_diagnostics.unified_streaming_state_construction",
                    unified_diagnostics_state_seconds);
                timing.add(
                    "time_slab.flux_diagnostics.unified_streaming_accumulation",
                    unified_diagnostics_eval_seconds);
                timing.add(
                    "time_slab.flux_diagnostics.unified_streaming_qpoint_eval",
                    unified_diagnostics_qpoint_eval_seconds);
                timing.add(
                    "time_slab.flux_diagnostics.unified_streaming_map_accumulation",
                    unified_diagnostics_accumulation_seconds);
                timing.add(
                    "time_slab.flux_diagnostics.fused_total",
                    unified_diagnostics_state_seconds +
                        unified_diagnostics_eval_seconds);
                timing.add(
                    "time_slab.flux_diagnostics.fused",
                    unified_diagnostics_state_seconds +
                        unified_diagnostics_eval_seconds);
            }

            if (false)
                record_reduced_mean_zero_dimension_counts_(timing);
            record_local_error_timing_(timing, timing_stats);
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_all_patches_tiled_explicit_(
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            double zero_tol,
            const finite_element::detail::TimingRecorder& timing)
        {
            solve_all_patches_tiled_chunked_explicit_<
                QSpace,
                QTime>(
                lambda_tilde,
                ell,
                u_delta,
                M,
                zero_tol,
                timing);
            return;

            const int patch_count = n_patches();
            const int tile_size =
                std::max(1, local_error_patch_tile_size_);
            const int tile_count =
                (patch_count + tile_size - 1) / tile_size;

            last_solve_all_patches_used_openmp_ = false;

            using RTCellCache =
                finite_element::assembly::detail::
                    LocalErrorRTCellQuadratureCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType>;
            using ABElementCache =
                finite_element::assembly::error_system::
                    LocalABElementCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using RHSStateCache =
                finite_element::assembly::error_system::
                    LocalRHSStateCache2D<
                        QSpace,
                        QTime,
                        FluxSpaceType>;
            using Tables =
                finite_element::assembly::detail::
                    LocalErrorQuadratureTables2D<
                        QSpace,
                        QTime,
                        FluxSpaceType,
                        ScalarSpaceType>;
            using DenseLocalErrorBlocks =
                finite_element::assembly::error_system::
                    DenseLocalErrorBlocks;

            finite_element::assembly::error_system::LocalErrorProblemTimingStats
                timing_stats;

            const auto active_slab_cells =
                collect_active_slab_cell_refs_();
            const auto cell_color_classes =
                build_cell_color_classes_(active_slab_cells);
            auto shared_context_storage =
                build_shared_context_if_requested_(
                    active_slab_cells,
                    timing_stats);
            const bool use_shared_context_for_state =
                local_error_context_storage_ == "shared_immutable" &&
                shared_context_storage.has_value();

            std::optional<RTCellCache> rt_cell_cache;
            {
                auto timer =
                    timing.scoped(
                        "time_slab.local_error_solves.rt_cell_cache_construction");
                rt_cell_cache.emplace(flux_spaces_);
            }
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.requested_patch_cells.count",
                static_cast<double>(rt_cell_cache->requested_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.unique_slab_cells.count",
                static_cast<double>(rt_cell_cache->unique_slab_cells()));
            timing.add(
                "time_slab.local_error_solves.rt_cell_cache.duplicate_patch_cells.count",
                static_cast<double>(rt_cell_cache->duplicate_patch_cells()));

            std::optional<ABElementCache> ab_element_cache;
            {
                auto timer =
                    timing.scoped(
                        "time_slab.local_error_solves.ab_element_cache_construction");
                ab_element_cache.emplace(
                    flux_spaces_,
                    scalar_spaces_,
                    *rt_cell_cache,
                    M);
            }
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.requested_patch_cells.count",
                static_cast<double>(ab_element_cache->requested_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.unique_slab_cells.count",
                static_cast<double>(ab_element_cache->unique_slab_cells()));
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.duplicate_patch_cells.count",
                static_cast<double>(ab_element_cache->duplicate_patch_cells()));

            finite_element::detail::CellGeometryCache<XSpace>
                rhs_x_geometry_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>
                rhs_ancestor_cache(*x_space_);
            auto rhs_slab_geometry_caches = make_slab_geometry_caches_();
            const auto rhs_context =
                finite_element::assembly::error_system::
                    LocalErrorProblemContext<XSpace, SlabSpaceType>{
                        x_space_,
                        slab_space_ptr_,
                        &rhs_x_geometry_cache,
                        &rhs_slab_geometry_caches,
                        &rhs_ancestor_cache,
                        nullptr};
            const auto shared_rhs_context =
                finite_element::assembly::error_system::
                    LocalErrorProblemContext<XSpace, SlabSpaceType>{
                        x_space_,
                        slab_space_ptr_,
                        nullptr,
                        nullptr,
                        nullptr,
                        use_shared_context_for_state
                            ? &*shared_context_storage
                            : nullptr};
            const auto& active_rhs_context =
                use_shared_context_for_state ? shared_rhs_context : rhs_context;

            std::optional<RHSStateCache> rhs_state_cache;
            {
                auto timer =
                    timing.scoped(
                    "time_slab.local_error_solves.rhs_state_cache_construction");
                rhs_state_cache.emplace(
                    flux_spaces_,
                    *rt_cell_cache,
                    active_rhs_context,
                    lambda_tilde,
                    u_delta,
                    ell,
                    M);
            }
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.requested_patch_cells.count",
                static_cast<double>(rhs_state_cache->requested_patch_cells()));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.unique_slab_cells.count",
                static_cast<double>(rhs_state_cache->unique_slab_cells()));
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.duplicate_patch_cells.count",
                static_cast<double>(rhs_state_cache->duplicate_patch_cells()));
            record_local_error_reuse_summary_(
                timing,
                rt_cell_cache->requested_patch_cells(),
                rt_cell_cache->unique_slab_cells());
            add_local_error_cache_qpoint_counters_(
                timing_stats,
                *rt_cell_cache,
                *ab_element_cache,
                *rhs_state_cache);

            const std::size_t unified_cell_state_bytes =
                rt_cell_cache->estimated_memory_bytes() +
                ab_element_cache->estimated_memory_bytes() +
                rhs_state_cache->estimated_memory_bytes();
            std::size_t peak_dense_blocks_bytes = 0;
            std::size_t peak_dense_patch_bytes = 0;
            std::size_t peak_tables_bytes = 0;
            std::size_t peak_process_rss_bytes =
                finite_element::detail::current_process_rss_bytes();
            int peak_selected_threads = 1;

            auto tables_bytes =
                [](const std::vector<Tables>& tables) noexcept
            {
                std::size_t bytes =
                    tables.capacity() * sizeof(Tables);
                for (const auto& table : tables)
                    bytes += table.estimated_memory_bytes();
                return bytes;
            };

            for (int tile_begin = 0;
                 tile_begin < patch_count;
                 tile_begin += tile_size)
            {
                const int tile_end =
                    std::min(patch_count, tile_begin + tile_size);
                const int current_tile_size = tile_end - tile_begin;

                std::vector<DenseLocalErrorBlocks> dense_blocks;
                std::vector<Tables> tables;

                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming timer(
                            &timing_stats.assemble_C_seconds);
                    dense_blocks.reserve(
                        static_cast<std::size_t>(current_tile_size));
                    for (int patch_id = tile_begin;
                         patch_id < tile_end;
                         ++patch_id)
                    {
                        dense_blocks.emplace_back(
                            flux_spaces_[static_cast<std::size_t>(patch_id)]
                                .n_dofs(),
                            scalar_spaces_[static_cast<std::size_t>(patch_id)]
                                .n_dofs());
                    }
                }

                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming timer(
                            &timing_stats
                                 .quadrature_table_construction_seconds);
                    tables.reserve(
                        static_cast<std::size_t>(current_tile_size));
                    for (int patch_id = tile_begin;
                         patch_id < tile_end;
                         ++patch_id)
                    {
                        tables.emplace_back(
                            flux_spaces_[static_cast<std::size_t>(patch_id)],
                            scalar_spaces_[static_cast<std::size_t>(patch_id)],
                            *rt_cell_cache);
                        add_local_table_qpoint_counters_(
                            timing_stats,
                            tables.back());
                    }
                }

                peak_dense_blocks_bytes =
                    std::max(peak_dense_blocks_bytes,
                             dense_blocks_bytes_(dense_blocks));
                peak_dense_patch_bytes =
                    std::max(peak_dense_patch_bytes,
                             dense_block_peak_patch_bytes_(dense_blocks));
                peak_tables_bytes =
                    std::max(peak_tables_bytes,
                             tables_bytes(tables));

                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming timer(
                            &timing_stats.assemble_A_seconds);
                    for (const auto& color_cells : cell_color_classes)
                    {
                        for (const int color_cell_index : color_cells)
                        {
                            const auto cell =
                                active_slab_cells[
                                    static_cast<std::size_t>(
                                        color_cell_index)];
                            const auto& local_A =
                                ab_element_cache
                                    ->cell(cell.slab_id, cell.slab_cell_id)
                                    .A;
                            const int membership_count =
                                patch_set_->cell_patch_count(
                                    cell.slab_id,
                                    cell.slab_cell_id);
                            for (int membership_index = 0;
                                 membership_index < membership_count;
                                 ++membership_index)
                            {
                                const auto& membership =
                                    patch_set_->cell_patch_membership(
                                        cell.slab_id,
                                        cell.slab_cell_id,
                                        membership_index);
                                const int patch_id = membership.patch_id;
                                if (patch_id < tile_begin ||
                                    patch_id >= tile_end)
                                {
                                    continue;
                                }

                                finite_element::assembly::error_system::
                                    scatter_rt_local_matrix_dense_2d(
                                        dense_blocks[
                                            static_cast<std::size_t>(
                                                patch_id - tile_begin)]
                                            .A,
                                        local_A,
                                        flux_spaces_[
                                            static_cast<std::size_t>(
                                                patch_id)],
                                        membership.patch_cell_index,
                                        zero_tol);
                            }
                        }
                    }
                }

                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming timer(
                            &timing_stats.assemble_B_seconds);
                    for (const auto& color_cells : cell_color_classes)
                    {
                        for (const int color_cell_index : color_cells)
                        {
                            const auto cell =
                                active_slab_cells[
                                    static_cast<std::size_t>(
                                        color_cell_index)];
                            const auto& local_B =
                                ab_element_cache
                                    ->cell(cell.slab_id, cell.slab_cell_id)
                                    .B;
                            const int membership_count =
                                patch_set_->cell_patch_count(
                                    cell.slab_id,
                                    cell.slab_cell_id);
                            for (int membership_index = 0;
                                 membership_index < membership_count;
                                 ++membership_index)
                            {
                                const auto& membership =
                                    patch_set_->cell_patch_membership(
                                        cell.slab_id,
                                        cell.slab_cell_id,
                                        membership_index);
                                const int patch_id = membership.patch_id;
                                if (patch_id < tile_begin ||
                                    patch_id >= tile_end)
                                {
                                    continue;
                                }

                                finite_element::assembly::error_system::
                                    scatter_divergence_local_matrix_dense_2d(
                                        dense_blocks[
                                            static_cast<std::size_t>(
                                                patch_id - tile_begin)]
                                            .B,
                                        local_B,
                                        scalar_spaces_[
                                            static_cast<std::size_t>(
                                                patch_id)],
                                        flux_spaces_[
                                            static_cast<std::size_t>(
                                                patch_id)],
                                        membership.patch_cell_index,
                                        zero_tol);
                            }
                        }
                    }
                }

                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming timer(
                            &timing_stats.assemble_f_seconds);
                    for (const auto& color_cells : cell_color_classes)
                    {
                        for (const int color_cell_index : color_cells)
                        {
                            const auto cell =
                                active_slab_cells[
                                    static_cast<std::size_t>(
                                        color_cell_index)];
                            const int membership_count =
                                patch_set_->cell_patch_count(
                                    cell.slab_id,
                                    cell.slab_cell_id);
                            for (int membership_index = 0;
                                 membership_index < membership_count;
                                 ++membership_index)
                            {
                                const auto& membership =
                                    patch_set_->cell_patch_membership(
                                        cell.slab_id,
                                        cell.slab_cell_id,
                                        membership_index);
                                const int patch_id = membership.patch_id;
                                if (patch_id < tile_begin ||
                                    patch_id >= tile_end)
                                {
                                    continue;
                                }

                                const int local_patch_id =
                                    patch_id - tile_begin;
                                const auto& flux_space =
                                    flux_spaces_[
                                        static_cast<std::size_t>(patch_id)];
                                la::local::FixedLocalVector<
                                    FluxSpaceType::local_dofs_v> local_f;
                                finite_element::assembly::detail::
                                    zero_local_vector(local_f);
                                finite_element::assembly::error_system::
                                    accumulate_patch_flux_rhs_on_cell_time_2d_from_rhs_cache<
                                        QSpace,
                                        QTime>(
                                        local_f,
                                        flux_space,
                                        tables[
                                            static_cast<std::size_t>(
                                                local_patch_id)],
                                        membership.patch_cell_index,
                                        cell.slab_id,
                                        *rhs_state_cache);
                                finite_element::assembly::error_system::
                                    scatter_rt_local_vector_dense_2d(
                                        dense_blocks[
                                            static_cast<std::size_t>(
                                                local_patch_id)]
                                            .f,
                                        local_f,
                                        flux_space,
                                        membership.patch_cell_index);
                            }
                        }
                    }
                }

                {
                    finite_element::assembly::error_system::
                        LocalErrorProblemScopedTiming timer(
                            &timing_stats.assemble_g_seconds);
                    for (const auto& color_cells : cell_color_classes)
                    {
                        for (const int color_cell_index : color_cells)
                        {
                            const auto cell =
                                active_slab_cells[
                                    static_cast<std::size_t>(
                                        color_cell_index)];
                            const int membership_count =
                                patch_set_->cell_patch_count(
                                    cell.slab_id,
                                    cell.slab_cell_id);
                            for (int membership_index = 0;
                                 membership_index < membership_count;
                                 ++membership_index)
                            {
                                const auto& membership =
                                    patch_set_->cell_patch_membership(
                                        cell.slab_id,
                                        cell.slab_cell_id,
                                        membership_index);
                                const int patch_id = membership.patch_id;
                                if (patch_id < tile_begin ||
                                    patch_id >= tile_end)
                                {
                                    continue;
                                }

                                const int local_patch_id =
                                    patch_id - tile_begin;
                                const auto& scalar_space =
                                    scalar_spaces_[
                                        static_cast<std::size_t>(patch_id)];
                                la::local::FixedLocalVector<
                                    ScalarSpaceType::local_dofs_v> local_g;
                                finite_element::assembly::detail::
                                    zero_local_vector(local_g);
                                finite_element::assembly::error_system::
                                    accumulate_patch_scalar_rhs_on_cell_time_2d_from_rhs_cache<
                                        QSpace,
                                        QTime>(
                                        local_g,
                                        scalar_space,
                                        tables[
                                            static_cast<std::size_t>(
                                                local_patch_id)],
                                        membership.patch_cell_index,
                                        cell.slab_id,
                                        *rhs_state_cache);
                                finite_element::assembly::error_system::
                                    scatter_scalar_local_vector_dense_2d(
                                        dense_blocks[
                                            static_cast<std::size_t>(
                                                local_patch_id)]
                                            .g,
                                        local_g,
                                        scalar_space,
                                        membership.patch_cell_index);
                            }
                        }
                    }
                }

                {
                    const auto thread_policy =
                        select_local_error_thread_policy_(
                            current_tile_size,
                            dense_blocks_bytes_(dense_blocks) +
                                tables_bytes(tables) +
                                unified_cell_state_bytes +
                                estimate_scalar_reduction_basis_bytes_());
                    record_local_error_thread_policy_(timing, thread_policy);
                    peak_selected_threads =
                        std::max(
                            peak_selected_threads,
                            thread_policy.selected_threads);
                    if (thread_policy.selected_threads > 1)
                        last_solve_all_patches_used_openmp_ = true;

                    auto batch_stats =
                        patch_solve_batch_stats_(
                            tile_begin,
                            tile_end,
                            thread_policy.selected_threads);
                    using ExplicitSolveWorkspace =
                        finite_element::assembly::error_system::
                            DenseLocalErrorExplicitSolveWorkspace2D<Backend>;
                    const auto solve_workspace_alloc_start =
                        std::chrono::steady_clock::now();
                    std::vector<ExplicitSolveWorkspace> solve_workspaces(
                        static_cast<std::size_t>(
                            std::max(1, thread_policy.selected_threads)));
                    batch_stats.workspace_allocation_seconds =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() -
                            solve_workspace_alloc_start)
                            .count();
                    std::vector<double> reduced_transform_seconds_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> factorization_seconds_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> solve_apply_seconds_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> current_dense_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_scalar_dense_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_fallback_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_residual_fail_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> current_dimension_sum_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> current_dimension_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_dimension_sum_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    std::vector<double> reduced_dimension_count_by_patch(
                        static_cast<std::size_t>(current_tile_size),
                        0.0);
                    const auto solve_one_patch =
                        [&](int local_patch_id)
                        {
                            const int patch_id = tile_begin + local_patch_id;
                            double local_transform_seconds = 0.0;
                            double local_factorization_seconds = 0.0;
                            double local_solve_apply_seconds = 0.0;
                            double local_current_dense_count = 0.0;
                            double local_reduced_scalar_dense_count = 0.0;
                            double local_reduced_fallback_count = 0.0;
                            double local_reduced_residual_fail_count = 0.0;
                            double local_current_dimension_sum = 0.0;
                            double local_current_dimension_count = 0.0;
                            double local_reduced_dimension_sum = 0.0;
                            double local_reduced_dimension_count = 0.0;
                            auto split =
                                [&]() {
                                    ExplicitSolveWorkspace* workspace =
                                        nullptr;
                                    if (use_local_error_patch_solve_workspace_())
                                    {
                                        int workspace_id = 0;
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
                                        if (thread_policy.selected_threads > 1)
                                            workspace_id = omp_get_thread_num();
#endif
                                        workspace =
                                            &solve_workspaces[
                                                static_cast<std::size_t>(
                                                    workspace_id)];
                                    }
                                    return
                                        solve_dense_local_error_patch_with_selected_solver_(
                                            dense_blocks[
                                                static_cast<std::size_t>(
                                                    local_patch_id)],
                                            scalar_spaces_[
                                                static_cast<std::size_t>(
                                                    patch_id)],
                                            workspace,
                                            nullptr,
                                            zero_tol,
                                            &local_transform_seconds,
                                            &local_factorization_seconds,
                                            &local_solve_apply_seconds,
                                            &local_current_dense_count,
                                            &local_reduced_scalar_dense_count,
                                            &local_reduced_fallback_count,
                                            &local_reduced_residual_fail_count,
                                            &local_current_dimension_sum,
                                            &local_current_dimension_count,
                                            &local_reduced_dimension_sum,
                                            &local_reduced_dimension_count);
                                }();
                            reduced_transform_seconds_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_transform_seconds;
                            factorization_seconds_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_factorization_seconds;
                            solve_apply_seconds_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_solve_apply_seconds;
                            current_dense_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_current_dense_count;
                            reduced_scalar_dense_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_scalar_dense_count;
                            reduced_fallback_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_fallback_count;
                            reduced_residual_fail_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_residual_fail_count;
                            current_dimension_sum_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_current_dimension_sum;
                            current_dimension_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_current_dimension_count;
                            reduced_dimension_sum_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_dimension_sum;
                            reduced_dimension_count_by_patch[
                                static_cast<std::size_t>(local_patch_id)] =
                                local_reduced_dimension_count;

                            flux_functions_[
                                static_cast<std::size_t>(patch_id)]
                                .update_coefficients(split.lambda);
                            scalar_functions_[
                                static_cast<std::size_t>(patch_id)]
                                .update_coefficients(split.u);
                        };
                    const auto patch_solve_order =
                        patch_solve_order_(
                            tile_begin,
                            tile_end,
                            thread_policy.selected_threads);

                    {
                        finite_element::assembly::error_system::
                            LocalErrorProblemScopedTiming timer(
                                &timing_stats.solve_patch_systems_seconds);
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
                        if (thread_policy.selected_threads > 1)
                        {
                            std::exception_ptr error;
#pragma omp parallel for num_threads(thread_policy.selected_threads) schedule(static)
                            for (int solve_index = 0;
                                 solve_index <
                                     static_cast<int>(
                                         patch_solve_order.size());
                                 ++solve_index)
                            {
                                try
                                {
                                    solve_one_patch(
                                        patch_solve_order[
                                            static_cast<std::size_t>(
                                                solve_index)]);
                                }
                                catch (...)
                                {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                                    {
                                        if (!error)
                                            error = std::current_exception();
                                    }
                                }
                            }
                            finite_element::assembly::detail::
                                rethrow_parallel_exception(error);
                        }
                        else
#endif
                        {
                            for (const int local_patch_id :
                                 patch_solve_order)
                            {
                                solve_one_patch(local_patch_id);
                            }
                        }
                    }
                    for (const double seconds :
                         reduced_transform_seconds_by_patch)
                    {
                        timing_stats.reduced_basis_transform_seconds +=
                            seconds;
                    }
                    for (const double seconds :
                         factorization_seconds_by_patch)
                    {
                        batch_stats.factorization_seconds += seconds;
                    }
                    for (const double seconds : solve_apply_seconds_by_patch)
                    {
                        batch_stats.solve_apply_seconds += seconds;
                    }
                    for (const double count : current_dense_count_by_patch)
                        timing_stats.patch_solver_current_dense_count += count;
                    for (const double count : reduced_scalar_dense_count_by_patch)
                        timing_stats.patch_solver_reduced_scalar_dense_count +=
                            count;
                    for (const double count : reduced_fallback_count_by_patch)
                        timing_stats.patch_solver_reduced_fallback_count +=
                            count;
                    for (const double count : reduced_residual_fail_count_by_patch)
                        timing_stats.patch_solver_reduced_residual_fail_count +=
                            count;
                    for (const double value : current_dimension_sum_by_patch)
                        timing_stats.patch_solver_current_dimension_sum +=
                            value;
                    for (const double value : current_dimension_count_by_patch)
                        timing_stats.patch_solver_current_dimension_count +=
                            value;
                    for (const double value : reduced_dimension_sum_by_patch)
                        timing_stats.patch_solver_reduced_dimension_sum +=
                            value;
                    for (const double value : reduced_dimension_count_by_patch)
                        timing_stats.patch_solver_reduced_dimension_count +=
                            value;
                    timing_stats.patch_solver_mode =
                        patch_solver_mode_code_();
                    if (batch_stats.dense_solver_workspace_bytes == 0)
                    {
                        std::size_t workspace_bytes = 0;
                        for (const auto& workspace : solve_workspaces)
                            workspace_bytes += workspace.estimated_memory_bytes();
                        batch_stats.dense_solver_workspace_bytes =
                            workspace_bytes;
                    }
                    add_patch_solve_batch_stats_(
                        timing_stats,
                        batch_stats);
                }

                peak_process_rss_bytes =
                    std::max(
                        peak_process_rss_bytes,
                        finite_element::detail::current_process_rss_bytes());
            }

            const auto dense_solve_peak_temporary_bytes =
                dense_solve_peak_temporary_bytes_();
            const auto reduction_basis_bytes =
                estimate_scalar_reduction_basis_bytes_() +
                estimate_concurrent_scalar_reduction_basis_bytes_(
                    peak_selected_threads);
            const auto patch_solutions_bytes =
                static_cast<std::size_t>(peak_selected_threads) *
                sequential_patch_solution_peak_bytes_();
            const auto per_thread_context_bytes =
                estimate_per_thread_context_bytes_(peak_selected_threads);
            const auto concurrent_solve_temporary_bytes =
                static_cast<std::size_t>(peak_selected_threads) *
                dense_solve_peak_temporary_bytes;
            const auto dimensions = patch_dimension_counters_();
            const auto estimated_total_live_bytes =
                peak_dense_blocks_bytes + patch_solutions_bytes +
                peak_tables_bytes + unified_cell_state_bytes +
                reduction_basis_bytes + per_thread_context_bytes +
                concurrent_solve_temporary_bytes;

            timing.add(
                "time_slab.local_error_solves.patch_tiling_used",
                1.0);
            timing.add(
                "time_slab.local_error_solves.cell_state_chunking_used",
                0.0);
            timing.add(
                "time_slab.local_error_solves.patch_tile_size.count",
                static_cast<double>(tile_size));
            timing.add(
                "time_slab.local_error_solves.patch_tile_count.count",
                static_cast<double>(tile_count));
            timing.add(
                "time_slab.local_error_solves.streaming_tile_count.count",
                static_cast<double>(tile_count));
            timing.add(
                "time_slab.local_error_solves.chunk_count.count",
                0.0);
            timing.add(
                "time_slab.local_error_solves.cell_states_constructed.count",
                0.0);
            timing.add("local_error.n_cell_states_constructed", 0.0);
            timing.add(
                "time_slab.local_error_solves.repeated_cell_state_constructions.count",
                0.0);
            timing.add("local_error.n_cell_state_cache_hits", 0.0);
            timing.add("local_error.n_cell_state_cache_misses", 0.0);
            timing.add(
                "time_slab.local_error_solves.lock_free_colored_cell_assembly",
                0.0);
            timing.add(
                "time_slab.local_error_solves.lock_free_cell_colors.count",
                static_cast<double>(cell_color_classes.size()));
            timing.add(
                "time_slab.local_error_solves.memory.dense_blocks_bytes",
                static_cast<double>(peak_dense_blocks_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.dense_block_peak_patch_bytes",
                static_cast<double>(peak_dense_patch_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.patch_solutions_bytes",
                static_cast<double>(patch_solutions_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.local_error_tables_bytes",
                static_cast<double>(peak_tables_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.unified_cell_state_bytes",
                static_cast<double>(unified_cell_state_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.reduction_basis_bytes",
                static_cast<double>(reduction_basis_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.operator_cache_bytes",
                0.0);
            timing.add(
                "time_slab.local_error_solves.memory.factor_cache_bytes",
                0.0);
            timing.add(
                "time_slab.local_error_solves.memory.per_thread_context_bytes",
                static_cast<double>(per_thread_context_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.dense_solve_peak_temporary_bytes",
                static_cast<double>(dense_solve_peak_temporary_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.concurrent_dense_solve_temporary_bytes",
                static_cast<double>(concurrent_solve_temporary_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.estimated_total_live_bytes",
                static_cast<double>(estimated_total_live_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.process_rss_sample_bytes",
                static_cast<double>(peak_process_rss_bytes));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_n_lambda.count",
                static_cast<double>(dimensions.max_n_lambda));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_n_u.count",
                static_cast<double>(dimensions.max_n_u));
            timing.add(
                "time_slab.local_error_solves.memory.max_patch_system_size.count",
                static_cast<double>(dimensions.max_system_size));

            if (false)
                record_reduced_mean_zero_dimension_counts_(timing);
            record_local_error_timing_(timing, timing_stats);
        }

        template<int QSpace, int QTime, class MFunction>
        [[nodiscard]] std::size_t local_operator_cache_key_(
            int patch_id) const
        {
            const auto& patch =
                patch_set_->patch(patch_id);
            const auto& flux =
                flux_spaces_[static_cast<std::size_t>(patch_id)];
            const auto& scalar =
                scalar_spaces_[static_cast<std::size_t>(patch_id)];

            std::size_t key = 0;
            hash_combine_(key, 0x2d1d0f3u);
            hash_combine_(key, FluxSpaceType::p_space_v);
            hash_combine_(key, FluxSpaceType::p_time_v);
            hash_combine_(key, QSpace);
            hash_combine_(key, QTime);
            hash_combine_(key, typeid(MFunction).hash_code());
            hash_combine_(key, patch.is_boundary() ? 1 : 0);
            hash_combine_(key, patch.t_begin);
            hash_combine_(key, patch.t_end);
            hash_combine_(key, patch.n_cells);

            if (patch.n_cells > 0)
            {
                hash_combine_(
                    key,
                    patch.cell(0).source_spatial_vertex_id);
            }

            for (int patch_cell_index = 0;
                 patch_cell_index < patch.n_cells;
                 ++patch_cell_index)
            {
                const auto& cell = patch.cell(patch_cell_index);
                hash_combine_(key, patch_cell_index);
                hash_combine_(key, cell.source_cell_id);
                hash_combine_(key, cell.local_vertex_index);
                hash_combine_(key, cell.source_local_vertex_index);
                hash_combine_(key, cell.source_spatial_vertex_id);
                for (const int vertex_id : cell.source_spatial_vertex_ids)
                    hash_combine_(key, vertex_id);
            }

            const auto& topology = flux.topology();
            hash_combine_(key, topology.n_edges());
            for (int edge_id = 0; edge_id < topology.n_edges(); ++edge_id)
            {
                const auto& edge = topology.edge(edge_id);
                hash_combine_(key, edge_id);
                hash_combine_(key, edge.is_internal() ? 1 : 0);
                hash_combine_(key, edge.on_physical_boundary ? 1 : 0);
                hash_combine_(key, edge.n_incident_cells());
                for (const auto& incident : edge.incident_cells)
                {
                    hash_combine_(key, incident.patch_cell_index);
                    hash_combine_(key, incident.local_face_id);
                    hash_combine_(key, incident.orientation_sign);
                }
            }

            for (const int edge_id : flux.spatial_space().free_edge_ids())
            {
                hash_combine_(key, 0x71eeu);
                hash_combine_(key, edge_id);
            }
            for (const int edge_id :
                 flux.spatial_space().constrained_edge_ids())
            {
                hash_combine_(key, 0xc075u);
                hash_combine_(key, edge_id);
            }
            hash_combine_(key, flux.n_patch_cells());
            hash_combine_(key, flux.n_dofs());
            hash_combine_(key, scalar.n_patch_cells());
            hash_combine_(key, scalar.n_dofs());

            return key;
        }

        template<int QSpace, int QTime, class MFunction>
        void initialize_dense_blocks_and_load_operator_cache_(
            std::vector<finite_element::assembly::error_system::
                            DenseLocalErrorBlocks>& dense_blocks,
            std::vector<std::size_t>& operator_cache_keys,
            std::vector<char>& operator_cache_hits,
            double& operator_cache_hit_count,
            double& operator_cache_miss_count) const
        {
            const int patch_count = n_patches();
            dense_blocks.clear();
            dense_blocks.reserve(static_cast<std::size_t>(patch_count));
            operator_cache_keys.assign(
                static_cast<std::size_t>(patch_count),
                0u);
            operator_cache_hits.assign(
                static_cast<std::size_t>(patch_count),
                0);
            operator_cache_hit_count = 0.0;
            operator_cache_miss_count = 0.0;

            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                const int n_lambda =
                    flux_spaces_[static_cast<std::size_t>(patch_id)].n_dofs();
                const int n_u =
                    scalar_spaces_[static_cast<std::size_t>(patch_id)].n_dofs();
                dense_blocks.emplace_back(n_lambda, n_u);

                if (local_operator_cache_ == nullptr)
                    continue;

                auto key =
                    local_operator_cache_key_<QSpace, QTime, MFunction>(
                        patch_id);
                auto it = local_operator_cache_->operators.find(key);
                if (it != local_operator_cache_->operators.end() &&
                    it->second.n_lambda == n_lambda &&
                    it->second.n_u == n_u)
                {
                    auto& block =
                        dense_blocks[static_cast<std::size_t>(patch_id)];
                    block.A = it->second.A;
                    block.B = it->second.B;
                    block.C = it->second.C;
                    block.f.resize(n_lambda);
                    block.f.set_zero();
                    block.g.resize(n_u);
                    block.g.set_zero();
                    operator_cache_hits[
                        static_cast<std::size_t>(patch_id)] = 1;
                    operator_cache_hit_count += 1.0;
                }
                else
                {
                    operator_cache_miss_count += 1.0;
                }

                operator_cache_keys[static_cast<std::size_t>(patch_id)] =
                    std::move(key);
            }
        }

        template<class ABElementCacheType>
        [[nodiscard]] std::vector<char> local_operator_ab_requests_needed_(
            const ABElementCacheType& ab_element_cache,
            const std::vector<char>& operator_cache_hits) const
        {
            std::vector<char> needed(
                static_cast<std::size_t>(
                    ab_element_cache.n_build_requests()),
                0);

            if (operator_cache_hits.empty())
            {
                std::fill(needed.begin(), needed.end(), 1);
                return needed;
            }

            for (int request_id = 0;
                 request_id < ab_element_cache.n_build_requests();
                 ++request_id)
            {
                const int slab_id =
                    ab_element_cache.build_request_slab_id(request_id);
                const int slab_cell_id =
                    ab_element_cache.build_request_slab_cell_id(request_id);
                const int membership_count =
                    patch_set_->cell_patch_count(slab_id, slab_cell_id);
                for (int membership_index = 0;
                     membership_index < membership_count;
                     ++membership_index)
                {
                    const auto& membership =
                        patch_set_->cell_patch_membership(
                            slab_id,
                            slab_cell_id,
                            membership_index);
                    if (operator_cache_hits[
                            static_cast<std::size_t>(
                                membership.patch_id)] == 0)
                    {
                        needed[static_cast<std::size_t>(request_id)] = 1;
                        break;
                    }
                }
            }

            return needed;
        }

        static double count_false_(const std::vector<char>& values)
        {
            double count = 0.0;
            for (const char value : values)
            {
                if (value == 0)
                    count += 1.0;
            }
            return count;
        }

        void store_operator_cache_misses_(
            const std::vector<finite_element::assembly::error_system::
                                  DenseLocalErrorBlocks>& dense_blocks,
            const std::vector<std::size_t>& operator_cache_keys,
            const std::vector<char>& operator_cache_hits)
        {
            if (local_operator_cache_ == nullptr)
                return;

            const int patch_count = static_cast<int>(dense_blocks.size());
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                if (operator_cache_hits[
                        static_cast<std::size_t>(patch_id)] != 0)
                {
                    continue;
                }

                const auto key =
                    operator_cache_keys[static_cast<std::size_t>(patch_id)];

                const auto& block =
                    dense_blocks[static_cast<std::size_t>(patch_id)];
                typename LocalOperatorCache::OperatorData data;
                data.n_lambda = block.n_lambda;
                data.n_u = block.n_u;
                data.A = block.A;
                data.B = block.B;
                data.C = block.C;
                local_operator_cache_->operators[key] = std::move(data);
            }
        }

        [[nodiscard]] bool operator_data_matches_patch_(
            const typename LocalOperatorCache::OperatorData& data,
            int patch_id) const
        {
            const int n_lambda =
                flux_spaces_[static_cast<std::size_t>(patch_id)].n_dofs();
            const int n_u =
                scalar_spaces_[static_cast<std::size_t>(patch_id)].n_dofs();
            return data.n_lambda == n_lambda && data.n_u == n_u;
        }

        [[nodiscard]] bool operator_data_has_reusable_factorization_(
            const typename LocalOperatorCache::OperatorData& data,
            int patch_id) const
        {
            if (!operator_data_matches_patch_(data, patch_id) ||
                !data.has_explicit_constraint_factorization())
            {
                return false;
            }

            const auto& scalar =
                scalar_spaces_[static_cast<std::size_t>(patch_id)];
            const int n_constraints =
                scalar.has_mean_zero_constraint()
                    ? scalar.n_mean_zero_constraints()
                    : 0;
            return data.n_constraints == n_constraints &&
                   data.explicit_system_size ==
                       data.n_lambda + data.n_u + data.n_constraints;
        }

        void load_operator_factor_cache_entries_(
            const std::vector<std::size_t>& operator_cache_keys,
            std::vector<const typename LocalOperatorCache::OperatorData*>&
                entries) const
        {
            const int patch_count = n_patches();
            entries.assign(static_cast<std::size_t>(patch_count), nullptr);
            if (local_operator_cache_ == nullptr ||
                operator_cache_keys.size() !=
                    static_cast<std::size_t>(patch_count))
            {
                return;
            }

            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                auto it = local_operator_cache_->operators.find(
                    operator_cache_keys[static_cast<std::size_t>(patch_id)]);
                if (it != local_operator_cache_->operators.end() &&
                    operator_data_has_reusable_factorization_(
                        it->second,
                        patch_id))
                {
                    entries[static_cast<std::size_t>(patch_id)] =
                        &it->second;
                }
            }
        }

        void initialize_operator_factor_cache_status_(
            const std::vector<std::size_t>& operator_cache_keys,
            std::vector<const typename LocalOperatorCache::OperatorData*>&
                entries,
            std::vector<char>& factor_cache_hits,
            double& factor_cache_hit_count,
            double& factor_cache_miss_count) const
        {
            load_operator_factor_cache_entries_(
                operator_cache_keys,
                entries);

            const int patch_count = n_patches();
            factor_cache_hits.assign(
                static_cast<std::size_t>(patch_count),
                0);
            factor_cache_hit_count = 0.0;
            factor_cache_miss_count = 0.0;
            if (local_operator_cache_ == nullptr)
                return;

            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                if (entries[static_cast<std::size_t>(patch_id)] != nullptr)
                {
                    factor_cache_hits[
                        static_cast<std::size_t>(patch_id)] = 1;
                    factor_cache_hit_count += 1.0;
                }
                else
                {
                    factor_cache_miss_count += 1.0;
                }
            }
        }

        template<class PatchScalarSpaceType>
        [[nodiscard]] bool build_operator_factor_cache_entry_(
            const finite_element::assembly::error_system::
                DenseLocalErrorBlocks& block,
            const PatchScalarSpaceType& scalar_space,
            double zero_tol,
            typename LocalOperatorCache::OperatorData& data,
            double& factorization_seconds) const
        {
            using finite_element::assembly::error_system::
                build_dense_local_error_scalar_constraint_matrix_2d;
            using finite_element::assembly::error_system::
                build_dense_local_error_scalar_constraint_rhs_2d;
            using finite_element::assembly::error_system::
                dense_local_error_constraint_residual_is_acceptable_2d;

            auto system =
                build_dense_local_error_scalar_constraint_matrix_2d(
                    block,
                    scalar_space,
                    zero_tol);
            const auto rhs =
                build_dense_local_error_scalar_constraint_rhs_2d(
                    block,
                    system);

            auto solver = std::make_shared<DenseSolverType>();
            la::concepts::SolverOptions solver_options;
            solver_options.dense_factorization =
                la::concepts::DenseFactorizationType::PartialPivotDenseLU;
            const auto start = std::chrono::steady_clock::now();
            solver->compute(system.matrix, solver_options);
            factorization_seconds += std::chrono::duration<double>(
                                         std::chrono::steady_clock::now() -
                                         start)
                                         .count();

            const VectorType candidate = solver->solve(rhs);
            if (!dense_local_error_constraint_residual_is_acceptable_2d(
                    system.matrix,
                    rhs,
                    candidate))
            {
                return false;
            }

            data.n_lambda = block.n_lambda;
            data.n_u = block.n_u;
            data.n_constraints = system.n_constraints;
            data.explicit_system_size = system.system_size();
            data.A = block.A;
            data.B = block.B;
            data.C = block.C;
            data.explicit_constraint_matrix = std::move(system.matrix);
            data.explicit_constraint_solver = std::move(solver);
            return true;
        }

        void store_operator_factor_cache_entries_(
            const std::vector<typename LocalOperatorCache::OperatorData>&
                pending_entries,
            const std::vector<char>& pending_ready,
            const std::vector<std::size_t>& operator_cache_keys)
        {
            if (local_operator_cache_ == nullptr)
                return;

            const int patch_count =
                static_cast<int>(pending_entries.size());
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                if (pending_ready[
                        static_cast<std::size_t>(patch_id)] == 0)
                {
                    continue;
                }

                local_operator_cache_->operators[
                    operator_cache_keys[
                        static_cast<std::size_t>(patch_id)]] =
                    pending_entries[
                        static_cast<std::size_t>(patch_id)];
            }
        }

        template<class PatchScalarSpaceType>
        [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
        solve_dense_local_error_blocks_with_cached_factor_(
            const finite_element::assembly::error_system::
                DenseLocalErrorBlocks& block,
            const PatchScalarSpaceType& scalar_space,
            const typename LocalOperatorCache::OperatorData* data,
            double zero_tol) const
        {
            if (data == nullptr ||
                !data->has_explicit_constraint_factorization())
            {
                return finite_element::assembly::error_system::
                    solve_dense_local_error_blocks_with_scalar_constraints_2d<
                        Backend>(
                        block,
                        scalar_space,
                        zero_tol);
            }

            finite_element::assembly::error_system::
                DenseLocalErrorConstraintMatrix2D system;
            system.n_lambda = data->n_lambda;
            system.n_u = data->n_u;
            system.n_constraints = data->n_constraints;
            const auto rhs =
                finite_element::assembly::error_system::
                    build_dense_local_error_scalar_constraint_rhs_2d(
                        block,
                        system);
            const VectorType solution =
                data->explicit_constraint_solver->solve(rhs);
            if (!finite_element::assembly::error_system::
                    dense_local_error_constraint_residual_is_acceptable_2d(
                        data->explicit_constraint_matrix,
                        rhs,
                        solution))
            {
                return finite_element::assembly::error_system::
                    solve_dense_local_error_blocks_with_scalar_constraints_2d<
                        Backend>(
                        block,
                        scalar_space,
                        zero_tol);
            }

            return finite_element::assembly::error_system::
                split_dense_local_error_scalar_constraint_solution_2d<
                    Backend>(
                    solution,
                    data->n_lambda,
                    data->n_u);
        }

        template<
            int QSpace,
            int QTime,
            class ThetaFunctionType,
            class EllFunction,
            class XFunctionType,
            class MFunction>
        void solve_patch_sparse_reference_impl_(
            int patch_id,
            const ThetaFunctionType& lambda_tilde,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            const MFunction& M,
            finite_element::detail::CellGeometryCache<XSpace>& x_geometry_cache,
            std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>&
                slab_geometry_caches,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            double zero_tol,
            const finite_element::assembly::detail::
                LocalErrorRTCellQuadratureCache2D<
                    QSpace,
                    QTime,
                    FluxSpaceType>* rt_cell_cache = nullptr,
            finite_element::assembly::error_system::LocalErrorProblemTimingStats*
                timing_stats = nullptr)
        {
            ensure_patch_index_(patch_id);

            const auto context =
                finite_element::assembly::error_system::LocalErrorProblemContext<
                    XSpace,
                    SlabSpaceType>{
                        x_space_,
                        slab_space_ptr_,
                        &x_geometry_cache,
                        &slab_geometry_caches,
                        &ancestor_cache
                    };

            const auto& patch_flux_space = flux_space(patch_id);
            const auto& patch_scalar_space = scalar_space(patch_id);
            auto blocks =
                rt_cell_cache != nullptr
                    ? finite_element::assembly::error_system::
                          assemble_local_error_problem<
                              QSpace,
                              QTime,
                              Backend>(
                              patch_flux_space,
                              patch_scalar_space,
                              context,
                              lambda_tilde,
                              u_delta,
                              ell,
                              M,
                              *rt_cell_cache,
                              zero_tol,
                              timing_stats)
                    : finite_element::assembly::error_system::
                          assemble_local_error_problem<
                              QSpace,
                              QTime,
                              Backend>(
                              patch_flux_space,
                              patch_scalar_space,
                              context,
                              lambda_tilde,
                              u_delta,
                              ell,
                              M,
                              zero_tol,
                              timing_stats);

            const auto split =
                solve_patch_blocks_with_scalar_constraints_(
                    blocks,
                    patch_scalar_space,
                    solver,
                    options,
                    zero_tol,
                    timing_stats);

            flux_functions_[static_cast<std::size_t>(patch_id)]
                .update_coefficients(split.lambda);
            scalar_functions_[static_cast<std::size_t>(patch_id)]
                .update_coefficients(split.u);
        }

        template<class Blocks>
        [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
        solve_patch_blocks_with_scalar_constraints_(
            const Blocks& blocks,
            const ScalarSpaceType& scalar_space,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            double zero_tol,
            finite_element::assembly::error_system::LocalErrorProblemTimingStats*
                timing_stats = nullptr) const
        {
            if (!scalar_space.has_mean_zero_constraint())
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    solve_timer(
                        timing_stats != nullptr
                            ? &timing_stats->solve_patch_systems_seconds
                            : nullptr);
                return la::saddle::solve_and_split<Backend>(blocks, solver, options);
            }

            const int n_lambda = blocks.n_lambda;
            const int n_u = blocks.n_u;
            const int n_constraints = scalar_space.n_mean_zero_constraints();
            const int n_total = n_lambda + n_u + n_constraints;

            typename Backend::SparseMatrix matrix;
            typename Backend::Vector rhs(n_total);
            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    compose_timer(
                        timing_stats != nullptr
                            ? &timing_stats->compose_with_constraints_seconds
                            : nullptr);

                typename Backend::SparseBuilder builder;
                builder.reserve(
                    la::block::estimate_nnz(blocks.A) +
                    2 * la::block::estimate_nnz(blocks.B) +
                    la::block::estimate_nnz(blocks.C) +
                    2 * static_cast<std::size_t>(n_constraints) *
                        static_cast<std::size_t>(n_u));

                la::block::add_matrix_block(builder, blocks.A, 0, 0);
                la::block::add_matrix_block_transpose(builder, blocks.B, 0, n_lambda);
                la::block::add_matrix_block(builder, blocks.B, n_lambda, 0);
                la::block::add_matrix_block(builder, blocks.C, n_lambda, n_lambda);

                for (int constraint_id = 0;
                     constraint_id < n_constraints;
                     ++constraint_id)
                {
                    const auto& row =
                        scalar_space.mean_zero_constraint_row(constraint_id);
                    if (row.size() != static_cast<std::size_t>(n_u))
                    {
                        throw std::runtime_error(
                            "TimeSlabEquilibratedFluxReconstruction2plus1d: scalar constraint row size mismatch.");
                    }

                    const int constraint_col = n_lambda + n_u + constraint_id;
                    const int constraint_row = n_lambda + n_u + constraint_id;
                    for (int scalar_dof = 0; scalar_dof < n_u; ++scalar_dof)
                    {
                        const double value =
                            row[static_cast<std::size_t>(scalar_dof)];
                        if (std::abs(value) <= zero_tol)
                            continue;

                        builder.add(n_lambda + scalar_dof, constraint_col, value);
                        builder.add(constraint_row, n_lambda + scalar_dof, value);
                    }
                }

                matrix.resize(n_total, n_total);
                matrix.set_from_builder(builder);

                rhs.set_zero();
                for (int i = 0; i < n_lambda; ++i)
                    rhs[i] = blocks.f[i];
                for (int i = 0; i < n_u; ++i)
                    rhs[n_lambda + i] = blocks.g[i];
            }

            la::linear::LinearSystem<Backend> system;
            system.matrix = std::move(matrix);
            system.rhs = std::move(rhs);
            system.solution = typename Backend::Vector(n_total);
            system.solution.set_zero();

            {
                finite_element::assembly::error_system::LocalErrorProblemScopedTiming
                    solve_timer(
                        timing_stats != nullptr
                            ? &timing_stats->solve_patch_systems_seconds
                            : nullptr);
                la::linear::solve_in_place(system, solver, options);
            }

            la::saddle::SaddlePointSolution<Backend> split;
            split.lambda.resize(n_lambda);
            split.u.resize(n_u);
            for (int i = 0; i < n_lambda; ++i)
                split.lambda[i] = system.solution[i];
            for (int i = 0; i < n_u; ++i)
                split.u[i] = system.solution[n_lambda + i];

            return split;
        }

        static void record_local_error_timing_(
            const finite_element::detail::TimingRecorder& timing,
            const finite_element::assembly::error_system::
                LocalErrorProblemTimingStats& stats)
        {
            detail::record_local_error_v4_timing_aliases(timing, stats);

            timing.add(
                "time_slab.local_error_solves.patch_matrix_assembly",
                stats.assemble_A_seconds + stats.assemble_B_seconds +
                    stats.assemble_C_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_rhs_assembly",
                stats.assemble_f_seconds + stats.assemble_g_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_solve",
                stats.solve_patch_systems_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_A",
                stats.assemble_A_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_B",
                stats.assemble_B_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_C",
                stats.assemble_C_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_f",
                stats.assemble_f_seconds);
            timing.add(
                "time_slab.local_error_solves.assemble_g",
                stats.assemble_g_seconds);
            timing.add(
                "time_slab.local_error_solves.quadrature_table_construction",
                stats.quadrature_table_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.compose_with_constraints",
                stats.compose_with_constraints_seconds);
            timing.add(
                "time_slab.local_error_solves.reduced_basis_transform",
                stats.reduced_basis_transform_seconds);
            timing.add(
                "time_slab.local_error_solves.solve_patch_systems",
                stats.solve_patch_systems_seconds);
            timing.add(
                "time_slab.local_error_solves.coefficient_writeback",
                stats.coefficient_writeback_seconds);
            timing.add(
                "time_slab.local_error_solves.solve_patch_systems.factorization_time",
                stats.patch_solve_factorization_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_factorization",
                stats.patch_solve_factorization_seconds);
            timing.add(
                "time_slab.local_error_solves.solve_patch_systems.solve_apply_time",
                stats.patch_solve_apply_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_solve_apply",
                stats.patch_solve_apply_seconds);
            timing.add(
                "time_slab.local_error_solves.patch_solve_groups.count",
                stats.patch_solve_groups);
            timing.add(
                "time_slab.local_error_solves.patch_solve_largest_group_size.count",
                stats.patch_solve_largest_group_size);
            timing.add(
                "time_slab.local_error_solves.batched_patch_systems.count",
                stats.batched_patch_systems);
            timing.add(
                "time_slab.local_error_solves.workspace_patch_systems.count",
                stats.workspace_patch_systems);
            timing.add(
                "time_slab.local_error_solves.legacy_patch_systems.count",
                stats.fallback_patch_systems);
            timing.add(
                "time_slab.local_error_solves.memory.dense_solver_workspace_bytes",
                stats.dense_solver_workspace_bytes);
            timing.add(
                "shared_context.shadow_enabled",
                stats.shared_context_shadow_enabled);
            timing.add(
                "shared_context.build_wall",
                stats.shared_context_build_seconds);
            timing.add(
                "shared_context.memory_mb",
                stats.shared_context_memory_mb);
            timing.add(
                "shared_context.x_geometry_count",
                stats.shared_context_x_geometry_count);
            timing.add(
                "shared_context.slab_geometry_count",
                stats.shared_context_slab_geometry_count);
            timing.add(
                "shared_context.ancestor_count",
                stats.shared_context_ancestor_count);
            timing.add(
                "shared_context.active_slab_cells",
                stats.shared_context_active_slab_cells);
            timing.add(
                "shared_context.sample_geometry_max_abs_diff",
                stats.shared_context_sample_geometry_max_abs_diff);
            timing.add(
                "shared_context.sample_slab_geometry_max_abs_diff",
                stats.shared_context_sample_slab_geometry_max_abs_diff);
            timing.add(
                "shared_context.sample_ancestor_mismatch_count",
                stats.shared_context_sample_ancestor_mismatch_count);
            timing.add(
                "shared_context.sample_count",
                stats.shared_context_sample_count);
            timing.add(
                "shared_context.validation_enabled",
                stats.shared_context_validation_enabled);
            timing.add(
                "shared_context.validation_wall",
                stats.shared_context_validation_seconds);
            timing.add(
                "shared_context.comparison_mutable_caches_constructed",
                stats.shared_context_comparison_mutable_caches_constructed);
            timing.add(
                "local_error.mutable_rhs_context_constructed_count",
                stats.mutable_rhs_context_constructed_count);
            timing.add(
                "local_error.mutable_rhs_context_construction_wall",
                stats.mutable_rhs_context_construction_seconds);
            timing.add(
                "local_error.shared_rhs_context_used_count",
                stats.shared_rhs_context_used_count);
            timing.add(
                "time_slab.local_error_solves.active_slab_cell_collection",
                stats.active_slab_cell_collection_seconds);
            timing.add(
                "time_slab.local_error_solves.cell_coloring",
                stats.cell_coloring_seconds);
            timing.add(
                "time_slab.local_error_solves.tile_cell_order_construction",
                stats.tile_cell_order_construction_seconds);
            timing.add(
                "local_error.tile_cell_order_construction_wall",
                stats.tile_cell_order_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.cell_requests_construction",
                stats.cell_requests_construction_seconds);
            timing.add(
                "local_error.cell_requests_construction_wall",
                stats.cell_requests_construction_seconds);
            timing.add(
                "local_error.membership_plan_construction_wall",
                stats.membership_plan_construction_seconds);
            timing.add(
                "local_error.patch_membership_lookup_count",
                stats.patch_membership_lookup_count);
            timing.add(
                "local_error.patch_membership_lookup_count_after_plan",
                stats.patch_membership_lookup_count_after_plan);
            timing.add(
                "local_error.tile_plan.build_wall",
                stats.tile_plan_build_seconds);
            timing.add(
                "local_error.tile_plan.memory_mb",
                stats.tile_plan_memory_mb);
            timing.add(
                "local_error.tile_plan.cells",
                stats.tile_plan_cells);
            timing.add(
                "local_error.tile_plan.memberships",
                stats.tile_plan_memberships);
            timing.add(
                "local_error.tile_plan.tile_count",
                stats.tile_plan_tile_count);
            timing.add(
                "local_error.tile_plan.chunk_count",
                stats.tile_plan_chunk_count);
            timing.add(
                "local_error.tile_plan.membership_scans_avoided",
                stats.tile_plan_membership_scans_avoided);
            timing.add(
                "time_slab.local_error_solves.patch_cells_by_local_patch_construction",
                stats.patch_cells_by_local_patch_construction_seconds);
            timing.add(
                "local_error.unused_patch_cells_by_local_patch_removed",
                stats.unused_patch_cells_by_local_patch_removed);
            timing.add(
                "time_slab.local_error_solves.chunk_table_construction",
                stats.chunk_table_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.chunk_cell_state_construction",
                stats.chunk_cell_state_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.streaming_chunk_state_construction",
                stats.streaming_chunk_state_construction_seconds);
            timing.add(
                "time_slab.local_error_solves.streaming_tile_assembly",
                stats.streaming_tile_assembly_seconds);
            timing.add(
                "time_slab.local_error_solves.tile_dense_block_allocation",
                stats.tile_dense_block_allocation_seconds);
            timing.add(
                "time_slab.local_error_solves.rt_basis_qpoint_fills.count",
                stats.rt_basis_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.scalar_basis_qpoint_fills.count",
                stats.ab_scalar_basis_qpoint_fills +
                    stats.local_table_scalar_basis_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.partition_of_unity_qpoint_fills.count",
                stats.local_table_partition_of_unity_qpoint_fills +
                    stats.qpoint_state_partition_of_unity_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.diffusion_tensor_evaluations.count",
                stats.ab_diffusion_tensor_evaluations);
            timing.add(
                "time_slab.local_error_solves.ab_element_cache.scalar_basis_qpoint_fills.count",
                stats.ab_scalar_basis_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.diffusion_tensor_evaluations.count",
                stats.rhs_diffusion_tensor_evaluations);
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.lambda_gradient_evaluations.count",
                stats.rhs_lambda_gradient_evaluations);
            timing.add(
                "time_slab.local_error_solves.rhs_state_cache.u_gradient_evaluations.count",
                stats.rhs_u_gradient_evaluations);
            timing.add(
                "time_slab.local_error_solves.local_table_construction.patch_count",
                stats.local_table_construction_patches);
            timing.add(
                "time_slab.local_error_solves.local_table_construction.patch_cell_count",
                stats.local_table_construction_patch_cells);
            timing.add(
                "time_slab.local_error_solves.local_tables.scalar_basis_qpoint_fills.count",
                stats.local_table_scalar_basis_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.local_tables.partition_of_unity_qpoint_fills.count",
                stats.local_table_partition_of_unity_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.local_tables.owned_rt_basis_qpoint_fills.count",
                stats.local_table_owned_rt_basis_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.scalar_basis_qpoint_fills.count",
                stats.qpoint_state_scalar_basis_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.partition_of_unity_qpoint_fills.count",
                stats.qpoint_state_partition_of_unity_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.patch_equivalent_scalar_basis_qpoint_fills.count",
                stats.qpoint_state_patch_equivalent_scalar_basis_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.scalar_basis_qpoint_fills_avoided.count",
                stats.qpoint_state_scalar_basis_qpoint_fills_avoided);
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.patch_equivalent_partition_of_unity_qpoint_fills.count",
                stats.qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills);
            timing.add(
                "time_slab.local_error_solves.qpoint_state_cache.partition_of_unity_qpoint_fills_avoided.count",
                stats.qpoint_state_partition_of_unity_qpoint_fills_avoided);
        }

        void build_local_spaces_()
        {
            if (!patch_set_.has_value())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d::build_local_spaces_: patch set not initialized.");
            }

            flux_spaces_.clear();
            scalar_spaces_.clear();
            flux_functions_.clear();
            scalar_functions_.clear();

            const int patch_count = patch_set_->n_patches();

            flux_spaces_.reserve(static_cast<std::size_t>(patch_count));
            scalar_spaces_.reserve(static_cast<std::size_t>(patch_count));
            for (const auto& patch : patch_set_->patches())
            {
                flux_spaces_.emplace_back(patch, slab_space_ref());
                scalar_spaces_.emplace_back(patch, slab_space_ref());
            }

            flux_functions_.reserve(static_cast<std::size_t>(patch_count));
            scalar_functions_.reserve(static_cast<std::size_t>(patch_count));
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                flux_functions_.emplace_back(
                    flux_spaces_[static_cast<std::size_t>(patch_id)]);
                scalar_functions_.emplace_back(
                    scalar_spaces_[static_cast<std::size_t>(patch_id)]);
            }
        }

        void reset_local_state_()
        {
            initialized_ = false;
            patch_set_.reset();
            flux_spaces_.clear();
            scalar_spaces_.clear();
            flux_functions_.clear();
            scalar_functions_.clear();
            last_solve_all_patches_used_openmp_ = false;
            last_fused_flux_diagnostics_.reset();
            last_fused_flux_diagnostics_runtime_stats_ = {};
        }

        [[nodiscard]] std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>
        make_slab_geometry_caches_() const
        {
            std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>> caches;
            const int slab_count = slab_space_ref().n_slabs();
            caches.reserve(static_cast<std::size_t>(slab_count));
            for (int slab_id = 0; slab_id < slab_count; ++slab_id)
            {
                caches.emplace_back(slab_space_ref().slab(slab_id).fespace_ref());
            }
            return caches;
        }

        void ensure_initialized_() const
        {
            if (!initialized_)
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: reconstruction not initialized.");
            }
        }

        void ensure_patch_index_(int patch_id) const
        {
            ensure_initialized_();
            if (patch_id < 0 || patch_id >= patch_set_->n_patches())
            {
                throw std::runtime_error(
                    "TimeSlabEquilibratedFluxReconstruction2plus1d: patch id out of range.");
            }
        }

        const SourceYSpace* source_y_space_ = nullptr;
        const XSpace* x_space_ = nullptr;
        const SlabSpaceType* slab_space_ptr_ = nullptr;
        std::optional<SlabSpaceType> owned_slab_space_{};
        std::optional<PatchSetType> patch_set_{};

        std::vector<FluxSpaceType> flux_spaces_{};
        std::vector<ScalarSpaceType> scalar_spaces_{};
        std::vector<FluxFunctionType> flux_functions_{};
        std::vector<ScalarFunctionType> scalar_functions_{};

        bool initialized_ = false;
        bool last_solve_all_patches_used_openmp_ = false;
        bool fused_error_and_flux_diagnostics_ = true;
        bool local_error_reuse_patch_solve_workspace_ = true;
        std::optional<FluxDiagnostics> last_fused_flux_diagnostics_{};
        FluxDiagnosticsRuntimeStats last_fused_flux_diagnostics_runtime_stats_{};
        int local_error_patch_tile_size_ = 0;
        int local_error_cell_chunk_size_ = 0;
        int local_error_max_threads_ = 0;
        double local_error_memory_budget_mb_ = 0.0;
        std::string local_error_worker_context_mode_ = "persistent";
        std::string local_error_context_storage_ = "shared_immutable";
        std::string local_error_state_index_mode_ = "flat";
        std::string local_error_cell_state_cache_mode_ = "off";
        double local_error_cell_state_cache_budget_mb_ = 1024.0;
        bool local_error_coefficient_fast_path_ = true;
        bool local_error_compact_state_shadow_ = false;
        std::string local_error_cell_state_representation_ = "compact_split";
        std::string local_error_flux_diagnostics_mode_ = "auto";
        std::string local_error_patch_solver_ = "current_dense";
        std::string shared_context_validation_ = "off";
        static constexpr LocalOperatorCache* local_operator_cache_ = nullptr;
    };

}

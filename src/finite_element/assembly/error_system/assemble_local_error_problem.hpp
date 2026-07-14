#pragma once

#include <algorithm>
#include <chrono>
#include <optional>
#include <utility>

#include "../../../linear_algebra/system/saddle_point_system.hpp"

#include "../detail/local_error_quadrature_tables_2d.hpp"

#include "mat_A.hpp"
#include "mat_B.hpp"
#include "mat_C.hpp"
#include "vec_f.hpp"
#include "vec_g.hpp"

namespace finite_element::assembly::error_system
{
    struct LocalErrorProblemTimingStats
    {
        double assemble_A_seconds = 0.0;
        double assemble_B_seconds = 0.0;
        double assemble_C_seconds = 0.0;
        double assemble_f_seconds = 0.0;
        double assemble_g_seconds = 0.0;
        double quadrature_table_construction_seconds = 0.0;
        double compose_with_constraints_seconds = 0.0;
        double reduced_basis_transform_seconds = 0.0;
        double solve_patch_systems_seconds = 0.0;
        double coefficient_writeback_seconds = 0.0;
        double patch_solve_factorization_seconds = 0.0;
        double patch_solve_apply_seconds = 0.0;
        double patch_solve_workspace_allocation_seconds = 0.0;
        double patch_solve_groups = 0.0;
        double patch_solve_largest_group_size = 0.0;
        double patch_solver_mode = 0.0;
        double patch_solver_current_dense_count = 0.0;
        double patch_solver_reduced_scalar_dense_count = 0.0;
        double patch_solver_reduced_fallback_count = 0.0;
        double patch_solver_reduced_residual_fail_count = 0.0;
        double patch_solver_current_dimension_sum = 0.0;
        double patch_solver_current_dimension_count = 0.0;
        double patch_solver_reduced_dimension_sum = 0.0;
        double patch_solver_reduced_dimension_count = 0.0;
        double patch_solver_reduced_dimension_ratio = 0.0;
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
        double batched_patch_systems = 0.0;
        double workspace_patch_systems = 0.0;
        double fallback_patch_systems = 0.0;
        double dense_solver_workspace_bytes = 0.0;
        double active_slab_cell_collection_seconds = 0.0;
        double cell_coloring_seconds = 0.0;
        double tile_cell_order_construction_seconds = 0.0;
        double cell_requests_construction_seconds = 0.0;
        double patch_cells_by_local_patch_construction_seconds = 0.0;
        double unused_patch_cells_by_local_patch_removed = 0.0;
        double tile_plan_build_seconds = 0.0;
        double tile_plan_memory_mb = 0.0;
        double tile_plan_cells = 0.0;
        double tile_plan_memberships = 0.0;
        double tile_plan_tile_count = 0.0;
        double tile_plan_chunk_count = 0.0;
        double tile_plan_membership_scans_avoided = 0.0;
        double membership_plan_construction_seconds = 0.0;
        double patch_membership_lookup_count = 0.0;
        double patch_membership_lookup_count_after_plan = 0.0;
        double chunk_table_construction_seconds = 0.0;
        double chunk_cell_state_construction_seconds = 0.0;
        double streaming_chunk_state_construction_seconds = 0.0;
        double streaming_tile_assembly_seconds = 0.0;
        double tile_dense_block_allocation_seconds = 0.0;
        double rt_basis_qpoint_fills = 0.0;
        double ab_diffusion_tensor_evaluations = 0.0;
        double ab_scalar_basis_qpoint_fills = 0.0;
        double rhs_diffusion_tensor_evaluations = 0.0;
        double rhs_lambda_gradient_evaluations = 0.0;
        double rhs_u_gradient_evaluations = 0.0;
        double local_table_construction_patches = 0.0;
        double local_table_construction_patch_cells = 0.0;
        double local_table_scalar_basis_qpoint_fills = 0.0;
        double local_table_partition_of_unity_qpoint_fills = 0.0;
        double local_table_owned_rt_basis_qpoint_fills = 0.0;
        double qpoint_state_scalar_basis_qpoint_fills = 0.0;
        double qpoint_state_partition_of_unity_qpoint_fills = 0.0;
        double qpoint_state_patch_equivalent_scalar_basis_qpoint_fills = 0.0;
        double qpoint_state_scalar_basis_qpoint_fills_avoided = 0.0;
        double qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills =
            0.0;
        double qpoint_state_partition_of_unity_qpoint_fills_avoided = 0.0;
        double patch_count = 0.0;
        double tile_size = 0.0;
        double tile_count = 0.0;
        double cell_chunk_size = 0.0;
        double chunk_count = 0.0;
        double parallel_region_count = 0.0;
        double active_slab_cells = 0.0;
        double global_unique_slab_cells = 0.0;
        double total_state_build_requests = 0.0;
        double actual_state_constructions = 0.0;
        double distinct_state_constructed_cells = 0.0;
        double cross_tile_state_rebuilds = 0.0;
        double state_builds_per_global_cell_mean = 0.0;
        double state_builds_per_global_cell_max = 0.0;
        double cell_patch_memberships = 0.0;
        double tiles_per_cell_mean = 0.0;
        double tiles_per_cell_max = 0.0;
        double cells_spanning_multiple_tiles = 0.0;
        double hardware_threads = 0.0;
        double configured_max_threads = 0.0;
        double candidate_threads = 0.0;
        double selected_threads = 0.0;
        double memory_limited = 0.0;
        double nested_parallel_disabled = 0.0;
        double worker_context_mode = 1.0;
        double worker_context_memory_mb = 0.0;
        double thread_context_construction_count = 0.0;
        double thread_context_construction_seconds = 0.0;
        double geometry_cache_construction_seconds = 0.0;
        double ancestor_cache_construction_seconds = 0.0;
        double slab_geometry_cache_construction_seconds = 0.0;
        double shared_context_shadow_enabled = 0.0;
        double shared_context_build_seconds = 0.0;
        double shared_context_memory_mb = 0.0;
        double shared_context_x_geometry_count = 0.0;
        double shared_context_slab_geometry_count = 0.0;
        double shared_context_ancestor_count = 0.0;
        double shared_context_active_slab_cells = 0.0;
        double shared_context_sample_geometry_max_abs_diff = 0.0;
        double shared_context_sample_slab_geometry_max_abs_diff = 0.0;
        double shared_context_sample_ancestor_mismatch_count = 0.0;
        double shared_context_sample_count = 0.0;
        double shared_context_validation_enabled = 0.0;
        double shared_context_validation_seconds = 0.0;
        double shared_context_comparison_mutable_caches_constructed = 0.0;
        double mutable_rhs_context_constructed_count = 0.0;
        double mutable_rhs_context_construction_seconds = 0.0;
        double shared_rhs_context_used_count = 0.0;
        double state_prepare_total_seconds = 0.0;
        double state_prepare_unique_count_seconds = 0.0;
        double state_prepare_set_allocation_seconds = 0.0;
        double state_prepare_map_index_build_seconds = 0.0;
        double state_prepare_ordinal_map_build_seconds = 0.0;
        double state_prepare_cell_vector_allocation_seconds = 0.0;
        double state_prepare_request_collection_seconds = 0.0;
        double state_prepare_debug_duplicate_request_count = 0.0;
        double state_prepare_memory_mb = 0.0;
        double state_index_mode = 0.0;
        double state_index_flat_lookup_count = 0.0;
        double state_index_map_lookup_count = 0.0;
        double state_index_fallback_hash_lookup_count = 0.0;
        double state_fill_total_seconds = 0.0;
        double state_fill_active_ancestor_lookup_seconds = 0.0;
        double state_fill_geometry_lookup_seconds = 0.0;
        double state_fill_affine_map_seconds = 0.0;
        double state_fill_time_basis_seconds = 0.0;
        double state_fill_spatial_rt_basis_seconds = 0.0;
        double state_fill_scalar_basis_seconds = 0.0;
        double state_fill_partition_of_unity_seconds = 0.0;
        double state_fill_lambda_gradient_seconds = 0.0;
        double state_fill_u_gradient_seconds = 0.0;
        double state_fill_load_evaluation_seconds = 0.0;
        double state_fill_diffusion_evaluation_seconds = 0.0;
        double state_fill_diffusion_inverse_seconds = 0.0;
        double state_fill_local_A_assembly_seconds = 0.0;
        double state_fill_local_B_assembly_seconds = 0.0;
        double state_fill_qpoints_processed = 0.0;
        double state_fill_sampled_qpoints = 0.0;
        double coefficient_fast_path_enabled = 0.0;
        double coefficient_fast_path_identity_diffusion_cells = 0.0;
        double coefficient_fast_path_constant_diffusion_cells = 0.0;
        double coefficient_fast_path_zero_load_cells = 0.0;
        double coefficient_fast_path_generic_cells = 0.0;
        double operator_builder_mode = 0.0;
        double local_A_identity_reference_fast_path_count = 0.0;
        double local_A_constant_reference_fast_path_count = 0.0;
        double local_A_variable_generic_path_count = 0.0;
        double local_B_reference_fast_path_count = 0.0;
        double local_A_build_seconds = 0.0;
        double local_B_build_seconds = 0.0;
        double local_A_debug_max_abs_diff = 0.0;
        double local_A_debug_rel_frobenius_diff = 0.0;
        double local_B_debug_max_abs_diff = 0.0;
        double local_B_debug_rel_frobenius_diff = 0.0;
        double compact_state_shadow_enabled = 0.0;
        double compact_state_shadow_sample_count = 0.0;
        double compact_state_reference_rt_basis_max_abs_diff = 0.0;
        double compact_state_reference_scalar_basis_max_abs_diff = 0.0;
        double compact_state_reference_partition_value_max_abs_diff = 0.0;
        double compact_state_reference_partition_gradient_max_abs_diff = 0.0;
        double compact_state_local_A_max_abs_diff = 0.0;
        double compact_state_local_B_max_abs_diff = 0.0;
        double compact_state_rhs_f_max_abs_diff = 0.0;
        double compact_state_rhs_g_max_abs_diff = 0.0;
        double compact_state_grad_theta_max_abs_diff = 0.0;
        double compact_state_u_time_derivative_max_abs_diff = 0.0;
        double old_cell_data_bytes_per_cell = 0.0;
        double operator_state_bytes_per_cell = 0.0;
        double rhs_state_bytes_per_cell = 0.0;
        double flux_diagnostic_state_bytes_per_cell = 0.0;
        double reference_table_memory_mb = 0.0;
        double monolithic_cell_data_constructed_count = 0.0;
        double compact_operator_state_constructed_count = 0.0;
        double compact_rhs_state_constructed_count = 0.0;
        double monolithic_debug_path_used_count = 0.0;
        double estimated_compact_full_cache_gib = 0.0;
        double estimated_lifetime_window_cache_mb = 0.0;
        double state_bytes_per_cell = 0.0;
        double estimated_full_cache_bytes = 0.0;
        double configured_cache_limit_bytes = 0.0;
        double cell_state_cache_mode = 0.0;
        double cell_state_cache_budget_mb = 0.0;
        double cell_state_cache_entries = 0.0;
        double cell_state_cache_memory_mb = 0.0;
        double cell_state_cache_hits = 0.0;
        double cell_state_cache_misses = 0.0;
        double cell_state_cache_evictions = 0.0;
        double cell_state_cache_hit_rate = 0.0;
        double cell_state_cache_cross_tile_rebuilds_avoided = 0.0;
        double cell_state_cache_stale_state_detected_count = 0.0;
        double flux_diagnostics_mode = 0.0;
        double flux_diagnostics_streaming_reuse_used = 0.0;
        double flux_diagnostics_standalone_used = 0.0;
        double flux_diagnostics_fallback_reason_code = 0.0;
        double flux_diagnostics_reused_rhs_state_count = 0.0;
        double flux_diagnostics_built_diagnostic_state_count = 0.0;
        double flux_diagnostics_rebuilt_state_count = 0.0;
        double flux_diagnostics_monolithic_cell_data_constructed_count = 0.0;
        double flux_diagnostics_finalized_active_slab_cells = 0.0;
        double flux_diagnostics_missing_active_slab_cells = 0.0;
        double flux_diagnostics_duplicate_active_slab_cells = 0.0;
        double flux_diagnostics_cells_visited = 0.0;
        double flux_diagnostics_qpoints_visited = 0.0;
        double flux_diagnostics_streaming_seconds = 0.0;
        double flux_diagnostics_streaming_state_build_seconds = 0.0;
        double flux_diagnostics_streaming_qpoint_eval_seconds = 0.0;
        double flux_diagnostics_streaming_accumulation_seconds = 0.0;
        double flux_diagnostics_standalone_seconds = 0.0;

        void add(const LocalErrorProblemTimingStats& other) noexcept
        {
            assemble_A_seconds += other.assemble_A_seconds;
            assemble_B_seconds += other.assemble_B_seconds;
            assemble_C_seconds += other.assemble_C_seconds;
            assemble_f_seconds += other.assemble_f_seconds;
            assemble_g_seconds += other.assemble_g_seconds;
            quadrature_table_construction_seconds +=
                other.quadrature_table_construction_seconds;
            compose_with_constraints_seconds +=
                other.compose_with_constraints_seconds;
            reduced_basis_transform_seconds +=
                other.reduced_basis_transform_seconds;
            solve_patch_systems_seconds += other.solve_patch_systems_seconds;
            coefficient_writeback_seconds +=
                other.coefficient_writeback_seconds;
            patch_solve_factorization_seconds +=
                other.patch_solve_factorization_seconds;
            patch_solve_apply_seconds += other.patch_solve_apply_seconds;
            patch_solve_workspace_allocation_seconds +=
                other.patch_solve_workspace_allocation_seconds;
            patch_solve_groups += other.patch_solve_groups;
            patch_solve_largest_group_size =
                std::max(
                    patch_solve_largest_group_size,
                    other.patch_solve_largest_group_size);
            patch_solver_mode =
                std::max(patch_solver_mode, other.patch_solver_mode);
            patch_solver_current_dense_count +=
                other.patch_solver_current_dense_count;
            patch_solver_reduced_scalar_dense_count +=
                other.patch_solver_reduced_scalar_dense_count;
            patch_solver_reduced_fallback_count +=
                other.patch_solver_reduced_fallback_count;
            patch_solver_reduced_residual_fail_count +=
                other.patch_solver_reduced_residual_fail_count;
            patch_solver_current_dimension_sum +=
                other.patch_solver_current_dimension_sum;
            patch_solver_current_dimension_count +=
                other.patch_solver_current_dimension_count;
            patch_solver_reduced_dimension_sum +=
                other.patch_solver_reduced_dimension_sum;
            patch_solver_reduced_dimension_count +=
                other.patch_solver_reduced_dimension_count;
            patch_solver_reduced_dimension_ratio = std::max(
                patch_solver_reduced_dimension_ratio,
                other.patch_solver_reduced_dimension_ratio);
            patch_dofs_distinct_sizes = std::max(
                patch_dofs_distinct_sizes,
                other.patch_dofs_distinct_sizes);
            patch_dofs_mode_size =
                std::max(patch_dofs_mode_size, other.patch_dofs_mode_size);
            patch_dofs_mode_count =
                std::max(patch_dofs_mode_count, other.patch_dofs_mode_count);
            flux_dofs_distinct_sizes = std::max(
                flux_dofs_distinct_sizes,
                other.flux_dofs_distinct_sizes);
            flux_dofs_mode_size =
                std::max(flux_dofs_mode_size, other.flux_dofs_mode_size);
            flux_dofs_mode_count =
                std::max(flux_dofs_mode_count, other.flux_dofs_mode_count);
            scalar_dofs_distinct_sizes = std::max(
                scalar_dofs_distinct_sizes,
                other.scalar_dofs_distinct_sizes);
            scalar_dofs_mode_size =
                std::max(scalar_dofs_mode_size, other.scalar_dofs_mode_size);
            scalar_dofs_mode_count = std::max(
                scalar_dofs_mode_count,
                other.scalar_dofs_mode_count);
            reduced_scalar_basis_candidate_count +=
                other.reduced_scalar_basis_candidate_count;
            batched_patch_systems += other.batched_patch_systems;
            workspace_patch_systems += other.workspace_patch_systems;
            fallback_patch_systems += other.fallback_patch_systems;
            dense_solver_workspace_bytes =
                std::max(
                    dense_solver_workspace_bytes,
                    other.dense_solver_workspace_bytes);
            active_slab_cell_collection_seconds +=
                other.active_slab_cell_collection_seconds;
            cell_coloring_seconds += other.cell_coloring_seconds;
            tile_cell_order_construction_seconds +=
                other.tile_cell_order_construction_seconds;
            cell_requests_construction_seconds +=
                other.cell_requests_construction_seconds;
            patch_cells_by_local_patch_construction_seconds +=
                other.patch_cells_by_local_patch_construction_seconds;
            unused_patch_cells_by_local_patch_removed = std::max(
                unused_patch_cells_by_local_patch_removed,
                other.unused_patch_cells_by_local_patch_removed);
            tile_plan_build_seconds += other.tile_plan_build_seconds;
            tile_plan_memory_mb =
                std::max(tile_plan_memory_mb, other.tile_plan_memory_mb);
            tile_plan_cells += other.tile_plan_cells;
            tile_plan_memberships += other.tile_plan_memberships;
            tile_plan_tile_count = std::max(
                tile_plan_tile_count,
                other.tile_plan_tile_count);
            tile_plan_chunk_count += other.tile_plan_chunk_count;
            tile_plan_membership_scans_avoided +=
                other.tile_plan_membership_scans_avoided;
            membership_plan_construction_seconds +=
                other.membership_plan_construction_seconds;
            patch_membership_lookup_count +=
                other.patch_membership_lookup_count;
            patch_membership_lookup_count_after_plan +=
                other.patch_membership_lookup_count_after_plan;
            chunk_table_construction_seconds +=
                other.chunk_table_construction_seconds;
            chunk_cell_state_construction_seconds +=
                other.chunk_cell_state_construction_seconds;
            streaming_chunk_state_construction_seconds +=
                other.streaming_chunk_state_construction_seconds;
            streaming_tile_assembly_seconds +=
                other.streaming_tile_assembly_seconds;
            tile_dense_block_allocation_seconds +=
                other.tile_dense_block_allocation_seconds;
            rt_basis_qpoint_fills += other.rt_basis_qpoint_fills;
            ab_diffusion_tensor_evaluations +=
                other.ab_diffusion_tensor_evaluations;
            ab_scalar_basis_qpoint_fills +=
                other.ab_scalar_basis_qpoint_fills;
            rhs_diffusion_tensor_evaluations +=
                other.rhs_diffusion_tensor_evaluations;
            rhs_lambda_gradient_evaluations +=
                other.rhs_lambda_gradient_evaluations;
            rhs_u_gradient_evaluations +=
                other.rhs_u_gradient_evaluations;
            local_table_construction_patches +=
                other.local_table_construction_patches;
            local_table_construction_patch_cells +=
                other.local_table_construction_patch_cells;
            local_table_scalar_basis_qpoint_fills +=
                other.local_table_scalar_basis_qpoint_fills;
            local_table_partition_of_unity_qpoint_fills +=
                other.local_table_partition_of_unity_qpoint_fills;
            local_table_owned_rt_basis_qpoint_fills +=
                other.local_table_owned_rt_basis_qpoint_fills;
            qpoint_state_scalar_basis_qpoint_fills +=
                other.qpoint_state_scalar_basis_qpoint_fills;
            qpoint_state_partition_of_unity_qpoint_fills +=
                other.qpoint_state_partition_of_unity_qpoint_fills;
            qpoint_state_patch_equivalent_scalar_basis_qpoint_fills +=
                other.qpoint_state_patch_equivalent_scalar_basis_qpoint_fills;
            qpoint_state_scalar_basis_qpoint_fills_avoided +=
                other.qpoint_state_scalar_basis_qpoint_fills_avoided;
            qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills +=
                other
                    .qpoint_state_patch_equivalent_partition_of_unity_qpoint_fills;
            qpoint_state_partition_of_unity_qpoint_fills_avoided +=
                other.qpoint_state_partition_of_unity_qpoint_fills_avoided;
            patch_count = std::max(patch_count, other.patch_count);
            tile_size = std::max(tile_size, other.tile_size);
            tile_count = std::max(tile_count, other.tile_count);
            cell_chunk_size = std::max(cell_chunk_size, other.cell_chunk_size);
            chunk_count += other.chunk_count;
            parallel_region_count += other.parallel_region_count;
            active_slab_cells = std::max(
                active_slab_cells,
                other.active_slab_cells);
            global_unique_slab_cells = std::max(
                global_unique_slab_cells,
                other.global_unique_slab_cells);
            total_state_build_requests += other.total_state_build_requests;
            actual_state_constructions += other.actual_state_constructions;
            distinct_state_constructed_cells = std::max(
                distinct_state_constructed_cells,
                other.distinct_state_constructed_cells);
            cross_tile_state_rebuilds += other.cross_tile_state_rebuilds;
            state_builds_per_global_cell_mean = std::max(
                state_builds_per_global_cell_mean,
                other.state_builds_per_global_cell_mean);
            state_builds_per_global_cell_max = std::max(
                state_builds_per_global_cell_max,
                other.state_builds_per_global_cell_max);
            cell_patch_memberships += other.cell_patch_memberships;
            tiles_per_cell_mean = std::max(
                tiles_per_cell_mean,
                other.tiles_per_cell_mean);
            tiles_per_cell_max = std::max(
                tiles_per_cell_max,
                other.tiles_per_cell_max);
            cells_spanning_multiple_tiles += other.cells_spanning_multiple_tiles;
            hardware_threads = std::max(
                hardware_threads,
                other.hardware_threads);
            configured_max_threads = std::max(
                configured_max_threads,
                other.configured_max_threads);
            candidate_threads = std::max(
                candidate_threads,
                other.candidate_threads);
            selected_threads = std::max(
                selected_threads,
                other.selected_threads);
            memory_limited += other.memory_limited;
            nested_parallel_disabled += other.nested_parallel_disabled;
            worker_context_mode = std::max(
                worker_context_mode,
                other.worker_context_mode);
            worker_context_memory_mb = std::max(
                worker_context_memory_mb,
                other.worker_context_memory_mb);
            thread_context_construction_count +=
                other.thread_context_construction_count;
            thread_context_construction_seconds +=
                other.thread_context_construction_seconds;
            geometry_cache_construction_seconds +=
                other.geometry_cache_construction_seconds;
            ancestor_cache_construction_seconds +=
                other.ancestor_cache_construction_seconds;
            slab_geometry_cache_construction_seconds +=
                other.slab_geometry_cache_construction_seconds;
            shared_context_shadow_enabled = std::max(
                shared_context_shadow_enabled,
                other.shared_context_shadow_enabled);
            shared_context_build_seconds +=
                other.shared_context_build_seconds;
            shared_context_memory_mb = std::max(
                shared_context_memory_mb,
                other.shared_context_memory_mb);
            shared_context_x_geometry_count = std::max(
                shared_context_x_geometry_count,
                other.shared_context_x_geometry_count);
            shared_context_slab_geometry_count = std::max(
                shared_context_slab_geometry_count,
                other.shared_context_slab_geometry_count);
            shared_context_ancestor_count = std::max(
                shared_context_ancestor_count,
                other.shared_context_ancestor_count);
            shared_context_active_slab_cells = std::max(
                shared_context_active_slab_cells,
                other.shared_context_active_slab_cells);
            shared_context_sample_geometry_max_abs_diff = std::max(
                shared_context_sample_geometry_max_abs_diff,
                other.shared_context_sample_geometry_max_abs_diff);
            shared_context_sample_slab_geometry_max_abs_diff = std::max(
                shared_context_sample_slab_geometry_max_abs_diff,
                other.shared_context_sample_slab_geometry_max_abs_diff);
            shared_context_sample_ancestor_mismatch_count +=
                other.shared_context_sample_ancestor_mismatch_count;
            shared_context_sample_count += other.shared_context_sample_count;
            shared_context_validation_enabled = std::max(
                shared_context_validation_enabled,
                other.shared_context_validation_enabled);
            shared_context_validation_seconds +=
                other.shared_context_validation_seconds;
            shared_context_comparison_mutable_caches_constructed +=
                other.shared_context_comparison_mutable_caches_constructed;
            mutable_rhs_context_constructed_count +=
                other.mutable_rhs_context_constructed_count;
            mutable_rhs_context_construction_seconds +=
                other.mutable_rhs_context_construction_seconds;
            shared_rhs_context_used_count += other.shared_rhs_context_used_count;
            state_prepare_total_seconds += other.state_prepare_total_seconds;
            state_prepare_unique_count_seconds +=
                other.state_prepare_unique_count_seconds;
            state_prepare_set_allocation_seconds +=
                other.state_prepare_set_allocation_seconds;
            state_prepare_map_index_build_seconds +=
                other.state_prepare_map_index_build_seconds;
            state_prepare_ordinal_map_build_seconds +=
                other.state_prepare_ordinal_map_build_seconds;
            state_prepare_cell_vector_allocation_seconds +=
                other.state_prepare_cell_vector_allocation_seconds;
            state_prepare_request_collection_seconds +=
                other.state_prepare_request_collection_seconds;
            state_prepare_debug_duplicate_request_count +=
                other.state_prepare_debug_duplicate_request_count;
            state_prepare_memory_mb =
                std::max(state_prepare_memory_mb, other.state_prepare_memory_mb);
            state_index_mode = std::max(state_index_mode, other.state_index_mode);
            state_index_flat_lookup_count += other.state_index_flat_lookup_count;
            state_index_map_lookup_count += other.state_index_map_lookup_count;
            state_index_fallback_hash_lookup_count +=
                other.state_index_fallback_hash_lookup_count;
            state_fill_total_seconds += other.state_fill_total_seconds;
            state_fill_active_ancestor_lookup_seconds +=
                other.state_fill_active_ancestor_lookup_seconds;
            state_fill_geometry_lookup_seconds +=
                other.state_fill_geometry_lookup_seconds;
            state_fill_affine_map_seconds +=
                other.state_fill_affine_map_seconds;
            state_fill_time_basis_seconds += other.state_fill_time_basis_seconds;
            state_fill_spatial_rt_basis_seconds +=
                other.state_fill_spatial_rt_basis_seconds;
            state_fill_scalar_basis_seconds +=
                other.state_fill_scalar_basis_seconds;
            state_fill_partition_of_unity_seconds +=
                other.state_fill_partition_of_unity_seconds;
            state_fill_lambda_gradient_seconds +=
                other.state_fill_lambda_gradient_seconds;
            state_fill_u_gradient_seconds +=
                other.state_fill_u_gradient_seconds;
            state_fill_load_evaluation_seconds +=
                other.state_fill_load_evaluation_seconds;
            state_fill_diffusion_evaluation_seconds +=
                other.state_fill_diffusion_evaluation_seconds;
            state_fill_diffusion_inverse_seconds +=
                other.state_fill_diffusion_inverse_seconds;
            state_fill_local_A_assembly_seconds +=
                other.state_fill_local_A_assembly_seconds;
            state_fill_local_B_assembly_seconds +=
                other.state_fill_local_B_assembly_seconds;
            state_fill_qpoints_processed += other.state_fill_qpoints_processed;
            state_fill_sampled_qpoints += other.state_fill_sampled_qpoints;
            coefficient_fast_path_enabled =
                std::max(coefficient_fast_path_enabled,
                         other.coefficient_fast_path_enabled);
            coefficient_fast_path_identity_diffusion_cells +=
                other.coefficient_fast_path_identity_diffusion_cells;
            coefficient_fast_path_constant_diffusion_cells +=
                other.coefficient_fast_path_constant_diffusion_cells;
            coefficient_fast_path_zero_load_cells +=
                other.coefficient_fast_path_zero_load_cells;
            coefficient_fast_path_generic_cells +=
                other.coefficient_fast_path_generic_cells;
            operator_builder_mode =
                std::max(operator_builder_mode, other.operator_builder_mode);
            local_A_identity_reference_fast_path_count +=
                other.local_A_identity_reference_fast_path_count;
            local_A_constant_reference_fast_path_count +=
                other.local_A_constant_reference_fast_path_count;
            local_A_variable_generic_path_count +=
                other.local_A_variable_generic_path_count;
            local_B_reference_fast_path_count +=
                other.local_B_reference_fast_path_count;
            local_A_build_seconds += other.local_A_build_seconds;
            local_B_build_seconds += other.local_B_build_seconds;
            local_A_debug_max_abs_diff = std::max(
                local_A_debug_max_abs_diff,
                other.local_A_debug_max_abs_diff);
            local_A_debug_rel_frobenius_diff = std::max(
                local_A_debug_rel_frobenius_diff,
                other.local_A_debug_rel_frobenius_diff);
            local_B_debug_max_abs_diff = std::max(
                local_B_debug_max_abs_diff,
                other.local_B_debug_max_abs_diff);
            local_B_debug_rel_frobenius_diff = std::max(
                local_B_debug_rel_frobenius_diff,
                other.local_B_debug_rel_frobenius_diff);
            compact_state_shadow_enabled = std::max(
                compact_state_shadow_enabled,
                other.compact_state_shadow_enabled);
            compact_state_shadow_sample_count +=
                other.compact_state_shadow_sample_count;
            compact_state_reference_rt_basis_max_abs_diff = std::max(
                compact_state_reference_rt_basis_max_abs_diff,
                other.compact_state_reference_rt_basis_max_abs_diff);
            compact_state_reference_scalar_basis_max_abs_diff = std::max(
                compact_state_reference_scalar_basis_max_abs_diff,
                other.compact_state_reference_scalar_basis_max_abs_diff);
            compact_state_reference_partition_value_max_abs_diff = std::max(
                compact_state_reference_partition_value_max_abs_diff,
                other.compact_state_reference_partition_value_max_abs_diff);
            compact_state_reference_partition_gradient_max_abs_diff = std::max(
                compact_state_reference_partition_gradient_max_abs_diff,
                other.compact_state_reference_partition_gradient_max_abs_diff);
            compact_state_local_A_max_abs_diff = std::max(
                compact_state_local_A_max_abs_diff,
                other.compact_state_local_A_max_abs_diff);
            compact_state_local_B_max_abs_diff = std::max(
                compact_state_local_B_max_abs_diff,
                other.compact_state_local_B_max_abs_diff);
            compact_state_rhs_f_max_abs_diff = std::max(
                compact_state_rhs_f_max_abs_diff,
                other.compact_state_rhs_f_max_abs_diff);
            compact_state_rhs_g_max_abs_diff = std::max(
                compact_state_rhs_g_max_abs_diff,
                other.compact_state_rhs_g_max_abs_diff);
            compact_state_grad_theta_max_abs_diff = std::max(
                compact_state_grad_theta_max_abs_diff,
                other.compact_state_grad_theta_max_abs_diff);
            compact_state_u_time_derivative_max_abs_diff = std::max(
                compact_state_u_time_derivative_max_abs_diff,
                other.compact_state_u_time_derivative_max_abs_diff);
            old_cell_data_bytes_per_cell = std::max(
                old_cell_data_bytes_per_cell,
                other.old_cell_data_bytes_per_cell);
            operator_state_bytes_per_cell = std::max(
                operator_state_bytes_per_cell,
                other.operator_state_bytes_per_cell);
            rhs_state_bytes_per_cell = std::max(
                rhs_state_bytes_per_cell,
                other.rhs_state_bytes_per_cell);
            flux_diagnostic_state_bytes_per_cell = std::max(
                flux_diagnostic_state_bytes_per_cell,
                other.flux_diagnostic_state_bytes_per_cell);
            reference_table_memory_mb = std::max(
                reference_table_memory_mb,
                other.reference_table_memory_mb);
            monolithic_cell_data_constructed_count +=
                other.monolithic_cell_data_constructed_count;
            compact_operator_state_constructed_count +=
                other.compact_operator_state_constructed_count;
            compact_rhs_state_constructed_count +=
                other.compact_rhs_state_constructed_count;
            monolithic_debug_path_used_count +=
                other.monolithic_debug_path_used_count;
            estimated_compact_full_cache_gib = std::max(
                estimated_compact_full_cache_gib,
                other.estimated_compact_full_cache_gib);
            estimated_lifetime_window_cache_mb = std::max(
                estimated_lifetime_window_cache_mb,
                other.estimated_lifetime_window_cache_mb);
            state_bytes_per_cell = std::max(
                state_bytes_per_cell,
                other.state_bytes_per_cell);
            estimated_full_cache_bytes = std::max(
                estimated_full_cache_bytes,
                other.estimated_full_cache_bytes);
            configured_cache_limit_bytes = std::max(
                configured_cache_limit_bytes,
                other.configured_cache_limit_bytes);
            cell_state_cache_mode = std::max(
                cell_state_cache_mode,
                other.cell_state_cache_mode);
            cell_state_cache_budget_mb = std::max(
                cell_state_cache_budget_mb,
                other.cell_state_cache_budget_mb);
            cell_state_cache_entries = std::max(
                cell_state_cache_entries,
                other.cell_state_cache_entries);
            cell_state_cache_memory_mb = std::max(
                cell_state_cache_memory_mb,
                other.cell_state_cache_memory_mb);
            cell_state_cache_hits += other.cell_state_cache_hits;
            cell_state_cache_misses += other.cell_state_cache_misses;
            cell_state_cache_evictions += other.cell_state_cache_evictions;
            cell_state_cache_hit_rate = std::max(
                cell_state_cache_hit_rate,
                other.cell_state_cache_hit_rate);
            cell_state_cache_cross_tile_rebuilds_avoided +=
                other.cell_state_cache_cross_tile_rebuilds_avoided;
            cell_state_cache_stale_state_detected_count +=
                other.cell_state_cache_stale_state_detected_count;
            flux_diagnostics_mode =
                std::max(flux_diagnostics_mode, other.flux_diagnostics_mode);
            flux_diagnostics_streaming_reuse_used = std::max(
                flux_diagnostics_streaming_reuse_used,
                other.flux_diagnostics_streaming_reuse_used);
            flux_diagnostics_standalone_used = std::max(
                flux_diagnostics_standalone_used,
                other.flux_diagnostics_standalone_used);
            flux_diagnostics_fallback_reason_code = std::max(
                flux_diagnostics_fallback_reason_code,
                other.flux_diagnostics_fallback_reason_code);
            flux_diagnostics_reused_rhs_state_count +=
                other.flux_diagnostics_reused_rhs_state_count;
            flux_diagnostics_built_diagnostic_state_count +=
                other.flux_diagnostics_built_diagnostic_state_count;
            flux_diagnostics_rebuilt_state_count +=
                other.flux_diagnostics_rebuilt_state_count;
            flux_diagnostics_monolithic_cell_data_constructed_count +=
                other.flux_diagnostics_monolithic_cell_data_constructed_count;
            flux_diagnostics_finalized_active_slab_cells +=
                other.flux_diagnostics_finalized_active_slab_cells;
            flux_diagnostics_missing_active_slab_cells +=
                other.flux_diagnostics_missing_active_slab_cells;
            flux_diagnostics_duplicate_active_slab_cells +=
                other.flux_diagnostics_duplicate_active_slab_cells;
            flux_diagnostics_cells_visited +=
                other.flux_diagnostics_cells_visited;
            flux_diagnostics_qpoints_visited +=
                other.flux_diagnostics_qpoints_visited;
            flux_diagnostics_streaming_seconds +=
                other.flux_diagnostics_streaming_seconds;
            flux_diagnostics_streaming_state_build_seconds +=
                other.flux_diagnostics_streaming_state_build_seconds;
            flux_diagnostics_streaming_qpoint_eval_seconds +=
                other.flux_diagnostics_streaming_qpoint_eval_seconds;
            flux_diagnostics_streaming_accumulation_seconds +=
                other.flux_diagnostics_streaming_accumulation_seconds;
            flux_diagnostics_standalone_seconds +=
                other.flux_diagnostics_standalone_seconds;
        }
    };

    class LocalErrorProblemScopedTiming
    {
    public:
        explicit LocalErrorProblemScopedTiming(double* accumulator) noexcept
            : accumulator_(accumulator)
        {
            if (accumulator_ != nullptr)
                start_ = Clock::now();
        }

        LocalErrorProblemScopedTiming(const LocalErrorProblemScopedTiming&) = delete;
        LocalErrorProblemScopedTiming& operator=(
            const LocalErrorProblemScopedTiming&) = delete;

        ~LocalErrorProblemScopedTiming() noexcept
        {
            if (accumulator_ == nullptr)
                return;

            const auto end = Clock::now();
            *accumulator_ +=
                std::chrono::duration<double>(end - start_).count();
        }

    private:
        using Clock = std::chrono::steady_clock;

        double* accumulator_ = nullptr;
        Clock::time_point start_{};
    };

    template<
        int QSpace,
        int QTime,
        class Backend,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    [[nodiscard]] la::saddle::SaddlePointBlocks<Backend>
    assemble_local_error_problem(
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M,
        double zero_tol = 1.0e-15,
        LocalErrorProblemTimingStats* timing_stats = nullptr)
    {
        using SparseMatrix = typename Backend::SparseMatrix;
        using Vector       = typename Backend::Vector;

        SparseMatrix A;
        SparseMatrix B;
        SparseMatrix C;
        Vector f;
        Vector g;

        if constexpr (
            PatchFluxSpaceType::GT::dim_space_v == 2 &&
            requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; })
        {
            using Tables =
                finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                    QSpace,
                    QTime,
                    PatchFluxSpaceType,
                    PatchScalarSpaceType>;

            std::optional<Tables> tables;
            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr
                        ? &timing_stats->quadrature_table_construction_seconds
                        : nullptr);
                tables.emplace(flux_space, scalar_space);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_A_seconds : nullptr);
                assemble_mat_A<QSpace, QTime, Backend>(
                    A,
                    flux_space,
                    scalar_space,
                    *tables,
                    M,
                    zero_tol);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_B_seconds : nullptr);
                assemble_mat_B<QSpace, QTime, Backend>(
                    B,
                    flux_space,
                    scalar_space,
                    *tables,
                    zero_tol);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_C_seconds : nullptr);
                assemble_mat_C<Backend>(
                    C,
                    scalar_space);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_f_seconds : nullptr);
                assemble_vec_f<QSpace, QTime>(
                    f,
                    flux_space,
                    scalar_space,
                    *tables,
                    context,
                    lambda_tilde,
                    u_delta);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_g_seconds : nullptr);
                assemble_vec_g<QSpace, QTime>(
                    g,
                    scalar_space,
                    flux_space,
                    *tables,
                    context,
                    lambda_tilde,
                    u_delta,
                    ell,
                    M);
            }
        }
        else
        {
            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_A_seconds : nullptr);
                assemble_mat_A<QSpace, QTime, Backend>(
                    A,
                    flux_space,
                    M,
                    zero_tol);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_B_seconds : nullptr);
                assemble_mat_B<QSpace, QTime, Backend>(
                    B,
                    flux_space,
                    scalar_space,
                    zero_tol);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_C_seconds : nullptr);
                assemble_mat_C<Backend>(
                    C,
                    scalar_space);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_f_seconds : nullptr);
                assemble_vec_f<QSpace, QTime>(
                    f,
                    flux_space,
                    scalar_space,
                    context,
                    lambda_tilde,
                    u_delta);
            }

            {
                LocalErrorProblemScopedTiming timer(
                    timing_stats != nullptr ? &timing_stats->assemble_g_seconds : nullptr);
                assemble_vec_g<QSpace, QTime>(
                    g,
                    scalar_space,
                    flux_space,
                    context,
                    lambda_tilde,
                    u_delta,
                    ell,
                    M);
            }
        }

        return la::saddle::make_saddle_point_blocks<Backend>(
            std::move(A),
            std::move(B),
            std::move(C),
            std::move(f),
            std::move(g));
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    [[nodiscard]] la::saddle::SaddlePointBlocks<Backend>
    assemble_local_error_problem(
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M,
        const finite_element::assembly::detail::
            LocalErrorRTCellQuadratureCache2D<
                QSpace,
                QTime,
                PatchFluxSpaceType>& rt_cell_cache,
        double zero_tol = 1.0e-15,
        LocalErrorProblemTimingStats* timing_stats = nullptr)
    {
        using SparseMatrix = typename Backend::SparseMatrix;
        using Vector       = typename Backend::Vector;
        using Tables =
            finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                QSpace,
                QTime,
                PatchFluxSpaceType,
                PatchScalarSpaceType>;

        SparseMatrix A;
        SparseMatrix B;
        SparseMatrix C;
        Vector f;
        Vector g;

        std::optional<Tables> tables;
        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->quadrature_table_construction_seconds
                    : nullptr);
            tables.emplace(flux_space, scalar_space, rt_cell_cache);
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr ? &timing_stats->assemble_A_seconds : nullptr);
            assemble_mat_A<QSpace, QTime, Backend>(
                A,
                flux_space,
                scalar_space,
                *tables,
                M,
                zero_tol);
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr ? &timing_stats->assemble_B_seconds : nullptr);
            assemble_mat_B<QSpace, QTime, Backend>(
                B,
                flux_space,
                scalar_space,
                *tables,
                zero_tol);
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr ? &timing_stats->assemble_C_seconds : nullptr);
            assemble_mat_C<Backend>(
                C,
                scalar_space);
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr ? &timing_stats->assemble_f_seconds : nullptr);
            assemble_vec_f<QSpace, QTime>(
                f,
                flux_space,
                scalar_space,
                *tables,
                context,
                lambda_tilde,
                u_delta);
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr ? &timing_stats->assemble_g_seconds : nullptr);
            assemble_vec_g<QSpace, QTime>(
                g,
                scalar_space,
                flux_space,
                *tables,
                context,
                lambda_tilde,
                u_delta,
                ell,
                M);
        }

        return la::saddle::make_saddle_point_blocks<Backend>(
            std::move(A),
            std::move(B),
            std::move(C),
            std::move(f),
            std::move(g));
    }

    template<
        class Backend,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType>
    [[nodiscard]] la::saddle::SaddlePointBlocks<Backend>
    assemble_local_error_problem(
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15)
    {
        using SparseMatrix = typename Backend::SparseMatrix;
        using Vector       = typename Backend::Vector;

        SparseMatrix A;
        SparseMatrix B;
        SparseMatrix C;
        Vector f;
        Vector g;

        assemble_mat_A_2d<Backend>(
            A,
            flux_space,
            zero_tol);

        assemble_mat_B_2d<Backend>(
            B,
            scalar_space,
            flux_space,
            zero_tol);

        assemble_mat_C_2d<Backend>(
            C,
            scalar_space);

        assemble_vec_f_2d(
            f,
            flux_space,
            zero_tol);

        assemble_vec_g_2d(
            g,
            scalar_space,
            zero_tol);

        return la::saddle::make_saddle_point_blocks<Backend>(
            std::move(A),
            std::move(B),
            std::move(C),
            std::move(f),
            std::move(g));
    }
}

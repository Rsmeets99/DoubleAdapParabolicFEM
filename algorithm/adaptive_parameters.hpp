#pragma once

#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "finite_element/system/solve_main_system.hpp"
#include "finite_element/time_slabs/time_slab_overlay.hpp"
#include "timing.hpp"

namespace adaptive_algorithm
{
    enum class DofLimitTarget
    {
        YTrueDofs,
        XTrueDofs
    };

    [[nodiscard]] inline std::string_view
    dof_limit_target_name(DofLimitTarget target) noexcept
    {
        switch (target)
        {
        case DofLimitTarget::YTrueDofs:
            return "y";
        case DofLimitTarget::XTrueDofs:
            return "x";
        }

        return "unknown";
    }

    struct AdaptiveOutputSettings
    {
        bool print_iteration_tables = true;
        bool save_refinement_history = false;
        bool save_estimator_components = false;
        bool save_mesh_statistics = true;
        bool export_history = false;
        bool save_iteration_snapshots = false;
        bool save_snapshot_dofs = false;

        std::filesystem::path output_directory{};
        std::string summary_filename = "adaptive_summary.txt";
        std::string outer_history_filename = "outer_history.csv";
        std::string inner_history_filename = "inner_history.csv";
        std::string timing_history_filename = "timing_history.csv";
        std::string partial_outer_history_filename = "partial_outer_history.csv";
        std::string partial_inner_history_filename = "partial_inner_history.csv";
        std::string partial_timing_history_filename = "partial_timing_history.csv";
        std::string partial_timing_breakdown_filename = "partial_timing_breakdown.csv";
        std::string refinement_history_filename = "refinement_history.txt";
        std::string snapshot_directory = "snapshots";

        std::ostream* stream = &std::cout;
    };

    struct AdaptiveParameters
    {
        double rho = 0.5;

        int max_outer_iterations = 10;
        int max_inner_iterations = 10;

        double doerfler_theta_y = 0.5;
        double doerfler_theta_x = 0.5;
        bool uniform_x_refinement = false;
        bool uniform_y_refinement = false;
        bool force_accept_inner_with_effective_rho = false;
        bool compute_g_estimator = false;
        bool compute_g_estimator_on_empty_y_marking_stop = false;
        bool compute_g_estimator_every_inner_iteration = false;
        std::string g_solver = "same_as_main";
        double g_solver_tolerance = 0.0;
        double g_solver_memory_limit_mb = 0.0;

        int polynomial_degree = -1;

        double zero_tol = 1.0e-15;
        double divergence_residual_l2_tolerance = 1.0e-8;
        double eta_squared_stop = 0.0;
        double inner_estimator_squared_stop = 0.0;
        // Soft wall-time budget in seconds. A value <= 0 disables the budget.
        // The driver does not interrupt an active solve; it stops before
        // starting the next adaptive iteration after the current one finishes.
        double max_wall_time_seconds = 0.0;
        int max_y_true_dofs = 500000;
        int max_x_true_dofs = 0;
        DofLimitTarget max_dofs_target = DofLimitTarget::YTrueDofs;
        // Whole-process adaptive-run memory guard in MiB.  A value <= 0
        // disables the guard.  This is separate from the direct-solver guard:
        // it is checked before major adaptive phases and against Linux
        // MemAvailable so WSL can stop cleanly before the OOM killer.
        double memory_limit_mb = 0.0;
        double memory_reserve_mb = 2048.0;
        double memory_guard_safety_factor = 1.15;
        double memory_guard_near_cap_fraction = 0.85;

        bool check_divergence_residual = true;
        bool stop_on_empty_y_marking = true;
        bool local_time_slab_closure = false;
        bool use_adaptive_initial_guess = false;
        bool solve_main_system_correction = false;
        bool fused_error_and_flux_diagnostics = true;
        bool local_error_reuse_patch_solve_workspace = true;
        bool deterministic_estimator_reductions = true;
        double doerfler_near_tie_tolerance = 0.0;
        bool refinement_edge_query_cache = true;
        int refinement_batch_target_split_cells = 32;
        std::string post_flush_closure_mode = "presplit_neighbour";
        bool post_flush_affected_containment_only = false;
        bool refinement_full_conformity_check = false;
        std::string refinement_main_closure_query_mode =
            "exact_and_ancestor";
        int local_error_patch_tile_size = 0;
        int local_error_cell_chunk_size = 0;
        int local_error_max_threads = 0;
        double local_error_memory_budget_mb = 0.0;
        std::string local_error_worker_context_mode = "persistent";
        std::string local_error_context_storage = "shared_immutable";
        std::string local_error_state_index_mode = "flat";
        std::string local_error_cell_state_cache_mode = "off";
        double local_error_cell_state_cache_budget_mb = 1024.0;
        std::string local_error_cell_state_representation = "compact_split";
        std::string local_error_flux_diagnostics_mode = "auto";
        std::string local_error_patch_solver = "current_dense";
        bool local_error_coefficient_fast_path = true;
        bool local_error_compact_state_shadow = false;
        std::string shared_context_validation = "off";
        std::string slab_reconstruction_operator_mode = "auto";
        int main_assembly_max_threads = 4;
        int slab_reconstruction_max_threads = 4;
        double main_assembly_memory_budget_mb = 0.0;
        double slab_reconstruction_memory_budget_mb = 0.0;
        int main_two_pass_numeric_fill_max_threads = 0;
        double main_two_pass_numeric_fill_memory_budget_mb = 0.0;
        finite_element::time_slabs::TimeSlabBackend time_slab_backend =
            finite_element::time_slabs::TimeSlabBackend::CopiedMesh;
        bool allow_copied_time_slab_estimator_fallback = true;
        bool virtual_backend_diagnostics = false;
        finite_element::system::MainSystemExportOptions
            main_system_export{};

        TimingOptions timing{};
        AdaptiveOutputSettings output{};
    };
}

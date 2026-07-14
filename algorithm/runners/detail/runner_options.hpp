#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>

#ifndef ADAPPARABOLICFEM_HAVE_MKL_PARDISO
#define ADAPPARABOLICFEM_HAVE_MKL_PARDISO 0
#endif

namespace adaptive_algorithm::runners::detail
{
    [[nodiscard]] inline std::string normalize_output_profile_choice(
        std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string normalize_uniform_refinement_mode_choice(
        std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string normalize_main_solver_diagnostics_choice(
        std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string normalize_post_flush_closure_mode_choice(
        std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_refinement_main_closure_query_mode_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_local_error_worker_context_mode_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_local_error_context_storage_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_local_error_state_index_mode_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_local_error_cell_state_cache_mode_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_local_error_cell_state_representation_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_local_error_flux_diagnostics_mode_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_local_error_patch_solver_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_slab_reconstruction_operator_mode_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string
    normalize_shared_context_validation_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string effective_uniform_refinement_mode(
        bool uniform_x_refinement,
        bool uniform_y_refinement)
    {
        if (uniform_x_refinement && uniform_y_refinement)
            return "uniform_xy";
        if (uniform_x_refinement)
            return "uniform_x";
        if (uniform_y_refinement)
            return "uniform_y";
        return "adaptive";
    }

    [[nodiscard]] inline double detect_available_memory_mb()
    {
#if defined(__linux__)
        std::ifstream in("/proc/meminfo");
        std::string key;
        unsigned long long value_kb = 0;
        std::string unit;
        double mem_total_mb = 0.0;

        while (in >> key >> value_kb >> unit)
        {
            if (key == "MemAvailable:")
                return static_cast<double>(value_kb) / 1024.0;
            if (key == "MemTotal:")
                mem_total_mb = static_cast<double>(value_kb) / 1024.0;
        }

        return mem_total_mb;
#else
        return 0.0;
#endif
    }

    [[nodiscard]] inline double default_main_solver_memory_limit_mb()
    {
        return 10000.0;
    }

    struct RunnerOptions
    {
        std::optional<std::filesystem::path> config_file{};
        std::string example_name = "smooth_initial";
        std::optional<int> dimension{1};
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
        std::string uniform_refinement_mode = "adaptive";
        int polynomial_degree = 2;
        double zero_tol = 1.0e-15;
        double divergence_residual_l2_tolerance = 1.0e-8;
        double eta_squared_stop = 0.0;
        double inner_estimator_squared_stop = 0.0;
        double max_wall_time_seconds = 0.0;
        int max_y_true_dofs = 500000;
        int max_x_true_dofs = 0;
        std::string max_dofs_target = "y";
        double memory_limit_mb = 0.0;
        bool memory_limit_explicit = false;
        double memory_reserve_mb = 2048.0;
        double memory_guard_safety_factor = 1.15;
        double memory_guard_near_cap_fraction = 0.85;
        bool increased_accuracy = false;
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        std::string main_solver = "pardiso_ldlt";
#else
        std::string main_solver = "sparse_lu";
#endif
        std::string main_solver_pardiso_memory_mode = "in_core";
        int main_solver_max_iterations = 1000;
        double main_solver_tolerance = 1.0e-10;
        double main_solver_symmetry_tolerance = 1.0e-12;
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        bool main_solver_direct_residual_retry = true;
#else
        bool main_solver_direct_residual_retry = false;
#endif
        double main_solver_direct_residual_retry_tolerance = 1.0e-10;
        std::string main_solver_diagnostics = "summary";
        double main_solver_memory_limit_mb =
            default_main_solver_memory_limit_mb();
        bool main_solver_ooc_auto_switch = false;
        double main_solver_ooc_switch_threshold = 0.85;
        bool main_solver_ooc_switch_to_lu = true;
        bool main_solver_reuse_symbolic_analysis = true;

        bool quiet = false;
        bool list_examples = false;
        bool show_help = false;
        bool check_divergence_residual = true;
        bool stop_on_empty_y_marking = true;
        bool local_time_slab_closure = false;
        bool use_adaptive_initial_guess = false;
        bool use_adaptive_initial_guess_explicit = false;
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
        bool local_error_worker_context_mode_explicit = false;
        bool local_error_context_storage_explicit = false;
        int main_assembly_max_threads = 4;
        int slab_reconstruction_max_threads = 4;
        double main_assembly_memory_budget_mb = 0.0;
        double slab_reconstruction_memory_budget_mb = 0.0;
        int main_two_pass_numeric_fill_max_threads = 0;
        double main_two_pass_numeric_fill_memory_budget_mb = 0.0;
        std::string time_slab_backend = "copied_mesh";
        bool allow_copied_time_slab_estimator_fallback = true;
        bool virtual_backend_diagnostics = false;
        bool solver_diagnostics_enabled = false;
        bool solver_diagnostics_export_matrix_market = false;
        bool solver_diagnostics_export_rhs = false;
        bool solver_diagnostics_export_solution = false;
        int solver_diagnostics_max_export_dofs = 20000;

        std::string output_profile = "benchmark";
        bool save_heavy_diagnostics = false;
        bool export_history = true;
        bool save_estimator_components = true;
        bool save_refinement_history = true;
        bool save_mesh_statistics = true;
        bool save_iteration_snapshots = false;
        bool save_snapshot_dofs = false;
        bool enable_timing_breakdown = false;

        std::filesystem::path output_directory{};
        std::string timing_history_filename = "timing_history.csv";
        std::string timing_detail_level = "summary";
    };

    struct RunnerOptionOverrides
    {
        std::optional<std::filesystem::path> config_file{};
        std::optional<std::string> example_name{};
        std::optional<int> dimension{};
        std::optional<double> rho{};
        std::optional<int> max_outer_iterations{};
        std::optional<int> max_inner_iterations{};
        std::optional<double> doerfler_theta_y{};
        std::optional<double> doerfler_theta_x{};
        std::optional<bool> uniform_x_refinement{};
        std::optional<bool> uniform_y_refinement{};
        std::optional<bool> force_accept_inner_with_effective_rho{};
        std::optional<bool> compute_g_estimator{};
        std::optional<bool> compute_g_estimator_on_empty_y_marking_stop{};
        std::optional<bool> compute_g_estimator_every_inner_iteration{};
        std::optional<std::string> g_solver{};
        std::optional<double> g_solver_tolerance{};
        std::optional<double> g_solver_memory_limit_mb{};
        std::optional<std::string> uniform_refinement_mode{};
        std::optional<int> polynomial_degree{};
        std::optional<double> zero_tol{};
        std::optional<double> divergence_residual_l2_tolerance{};
        std::optional<double> eta_squared_stop{};
        std::optional<double> inner_estimator_squared_stop{};
        std::optional<double> max_wall_time_seconds{};
        std::optional<int> max_y_true_dofs{};
        std::optional<int> max_x_true_dofs{};
        std::optional<std::string> max_dofs_target{};
        std::optional<double> memory_limit_mb{};
        std::optional<double> memory_reserve_mb{};
        std::optional<double> memory_guard_safety_factor{};
        std::optional<double> memory_guard_near_cap_fraction{};
        std::optional<bool> increased_accuracy{};
        std::optional<std::string> main_solver{};
        std::optional<std::string> main_solver_pardiso_memory_mode{};
        std::optional<int> main_solver_max_iterations{};
        std::optional<double> main_solver_tolerance{};
        std::optional<double> main_solver_symmetry_tolerance{};
        std::optional<bool> main_solver_direct_residual_retry{};
        std::optional<double> main_solver_direct_residual_retry_tolerance{};
        std::optional<std::string> main_solver_diagnostics{};
        std::optional<double> main_solver_memory_limit_mb{};
        std::optional<bool> main_solver_ooc_auto_switch{};
        std::optional<double> main_solver_ooc_switch_threshold{};
        std::optional<bool> main_solver_ooc_switch_to_lu{};
        std::optional<bool> main_solver_reuse_symbolic_analysis{};

        std::optional<bool> quiet{};
        std::optional<bool> list_examples{};
        std::optional<bool> show_help{};
        std::optional<bool> check_divergence_residual{};
        std::optional<bool> stop_on_empty_y_marking{};
        std::optional<bool> local_time_slab_closure{};
        std::optional<bool> use_adaptive_initial_guess{};
        std::optional<bool> solve_main_system_correction{};
        std::optional<bool> fused_error_and_flux_diagnostics{};
        std::optional<bool> local_error_reuse_patch_solve_workspace{};
        std::optional<bool> deterministic_estimator_reductions{};
        std::optional<double> doerfler_near_tie_tolerance{};
        std::optional<bool> refinement_edge_query_cache{};
        std::optional<int> refinement_batch_target_split_cells{};
        std::optional<std::string> post_flush_closure_mode{};
        std::optional<bool> post_flush_affected_containment_only{};
        std::optional<bool> refinement_full_conformity_check{};
        std::optional<std::string> refinement_main_closure_query_mode{};
        std::optional<int> local_error_patch_tile_size{};
        std::optional<int> local_error_cell_chunk_size{};
        std::optional<int> local_error_max_threads{};
        std::optional<double> local_error_memory_budget_mb{};
        std::optional<std::string> local_error_worker_context_mode{};
        std::optional<std::string> local_error_context_storage{};
        std::optional<std::string> local_error_state_index_mode{};
        std::optional<std::string> local_error_cell_state_cache_mode{};
        std::optional<double> local_error_cell_state_cache_budget_mb{};
        std::optional<std::string> local_error_cell_state_representation{};
        std::optional<std::string> local_error_flux_diagnostics_mode{};
        std::optional<std::string> local_error_patch_solver{};
        std::optional<bool> local_error_coefficient_fast_path{};
        std::optional<bool> local_error_compact_state_shadow{};
        std::optional<std::string> shared_context_validation{};
        std::optional<std::string> slab_reconstruction_operator_mode{};
        std::optional<int> main_assembly_max_threads{};
        std::optional<int> slab_reconstruction_max_threads{};
        std::optional<double> main_assembly_memory_budget_mb{};
        std::optional<double> slab_reconstruction_memory_budget_mb{};
        std::optional<int> main_two_pass_numeric_fill_max_threads{};
        std::optional<double> main_two_pass_numeric_fill_memory_budget_mb{};
        std::optional<std::string> time_slab_backend{};
        std::optional<bool> allow_copied_time_slab_estimator_fallback{};
        std::optional<bool> virtual_backend_diagnostics{};
        std::optional<bool> solver_diagnostics_enabled{};
        std::optional<bool> solver_diagnostics_export_matrix_market{};
        std::optional<bool> solver_diagnostics_export_rhs{};
        std::optional<bool> solver_diagnostics_export_solution{};
        std::optional<int> solver_diagnostics_max_export_dofs{};

        std::optional<std::string> output_profile{};
        std::optional<bool> save_heavy_diagnostics{};
        std::optional<bool> export_history{};
        std::optional<bool> save_estimator_components{};
        std::optional<bool> save_refinement_history{};
        std::optional<bool> save_mesh_statistics{};
        std::optional<bool> save_iteration_snapshots{};
        std::optional<bool> save_snapshot_dofs{};
        std::optional<bool> enable_timing_breakdown{};

        std::optional<std::filesystem::path> output_directory{};
        std::optional<std::string> timing_history_filename{};
        std::optional<std::string> timing_detail_level{};
    };
}

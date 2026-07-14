#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "finite_element/fespace/functions.hpp"
#include "finite_element/time_slabs/time_slab_error_indicators.hpp"
#include "linear_algebra/concepts/solver.hpp"
#include "timing.hpp"

namespace adaptive_algorithm
{
    [[nodiscard]] inline const char* solver_type_name(
        la::concepts::SolverType solver) noexcept
    {
        switch (solver)
        {
        case la::concepts::SolverType::SparseLU:
            return "sparse_lu";
        case la::concepts::SolverType::PardisoLU:
            return "pardiso_lu";
        case la::concepts::SolverType::PardisoLDLT:
            return "pardiso_ldlt";
        case la::concepts::SolverType::PardisoLDLTAuto:
            return "pardiso_ldlt_auto";
        case la::concepts::SolverType::MINRES:
            return "minres_parabolic_graph_norm";
        }

        return "unknown";
    }

    struct MainSolveDiagnosticsRecord
    {
        bool available = false;
        la::concepts::SolverType selected_solver =
            la::concepts::default_direct_solver_type();
        la::concepts::SolverType effective_solver =
            la::concepts::default_direct_solver_type();

        int n = 0;
        int matrix_rows = 0;
        int matrix_cols = 0;
        std::size_t nnz_matrix = 0;
        std::size_t matrix_nnz = 0;
        std::size_t nnz_factors = 0;
        std::optional<std::size_t> factor_nnz{};
        double fill_ratio = 0.0;
        double symbolic_analysis_seconds = 0.0;
        bool symbolic_analysis_reused = false;
        std::size_t symbolic_pattern_cache_hits = 0;
        std::size_t symbolic_pattern_cache_misses = 0;
        double numeric_factorization_seconds = 0.0;
        double backsolve_seconds = 0.0;
        std::optional<double> estimated_factor_memory_bytes{};
        std::string solver_status = "MISSING";
        double symbolic_memory = 0.0;
        double numerical_factor_memory = 0.0;
        double estimated_in_core_peak_memory = 0.0;
        double out_of_core_minimum_memory = 0.0;
        double process_rss_before_factorization = 0.0;
        double process_rss_after_factorization = 0.0;
        double process_rss_after_solve = 0.0;
        double memory_guard_estimated_extra_memory = 0.0;
        double direct_memory_limit = 0.0;
        double memory_guard_estimated_peak_memory = 0.0;
        bool memory_guard_triggered = false;
        bool out_of_core_auto_switch_attempted = false;
        std::string out_of_core_auto_switch_solver{};
        std::string effective_pardiso_memory_mode{};
        std::string pardiso_ldlt_robustness_profile{};
        long long pardiso_iparm_7 = 0;
        long long pardiso_iparm_9 = 0;
        long long pardiso_iparm_10 = 0;
        long long pardiso_perturbed_pivots = 0;
        long long pardiso_positive_eigenvalues = 0;
        long long pardiso_negative_eigenvalues = 0;
        int iteration_count = 0;
        double final_residual = 0.0;
        double preconditioner_setup_seconds = 0.0;
        double setup_seconds = 0.0;
        double solve_seconds = 0.0;
        double linear_residual_absolute = 0.0;
        double linear_residual_relative = 0.0;
        double initial_guess_norm = 0.0;
        double initial_residual_absolute = 0.0;
        double initial_residual_relative = 0.0;
        bool backend_converged = false;
        double backend_reported_error = 0.0;
        bool convergence_accepted_by_true_residual = false;
        int residual_check_batches = 0;
        double final_true_residual = 0.0;
        bool true_residual_stopping_used = false;
        double matrix_norm = 0.0;
        double matrix_symmetry_difference_norm = 0.0;
        double matrix_relative_asymmetry = 0.0;
        bool residual_retry_attempted = false;
        std::string residual_retry_solver{};
        double residual_before_retry = 0.0;
        double residual_after_retry = 0.0;
        int residual_correction_steps = 0;
        double residual_before_correction = 0.0;
        double residual_after_correction = 0.0;
    };

    struct ActiveGenerationStats
    {
        int min_generation = -1;
        int max_generation = -1;
        int distinct_generations = 0;
    };

    struct AdaptiveRefinementStatistics
    {
        std::optional<int> refined_cells{};
        std::optional<int> closure_cells{};
        std::optional<int> edge_interval_queries{};
        std::optional<int> edge_interval_records_visited{};
        std::optional<int> full_active_scans{};
    };

    [[nodiscard]] inline MainSolveDiagnosticsRecord make_main_solve_diagnostics_record(
        const la::concepts::SolverDiagnostics& diagnostics,
        double setup_seconds,
        double solve_seconds)
    {
        MainSolveDiagnosticsRecord record;
        record.available = diagnostics.rows > 0 || diagnostics.cols > 0;
        record.selected_solver = diagnostics.requested_solver;
        record.effective_solver = diagnostics.effective_solver;
        record.n = diagnostics.direct_stats.n.value_or(diagnostics.rows);
        record.matrix_rows = diagnostics.rows;
        record.matrix_cols = diagnostics.cols;
        record.nnz_matrix =
            diagnostics.direct_stats.nnz_matrix.value_or(diagnostics.nnz_matrix);
        record.matrix_nnz = diagnostics.nnz_matrix;
        record.nnz_factors = diagnostics.direct_stats.nnz_factors.value_or(0);
        if (diagnostics.direct_stats.nnz_factors.has_value())
            record.factor_nnz = *diagnostics.direct_stats.nnz_factors;
        record.fill_ratio = diagnostics.direct_stats.fill_ratio.value_or(0.0);
        record.symbolic_analysis_seconds =
            diagnostics.direct_stats.symbolic_analysis_seconds.value_or(0.0);
        record.symbolic_analysis_reused =
            diagnostics.direct_stats.symbolic_analysis_reused;
        record.symbolic_pattern_cache_hits =
            diagnostics.direct_stats.symbolic_pattern_cache_hits.value_or(0);
        record.symbolic_pattern_cache_misses =
            diagnostics.direct_stats.symbolic_pattern_cache_misses.value_or(0);
        record.numeric_factorization_seconds =
            diagnostics.direct_stats.numeric_factorization_seconds.value_or(
                setup_seconds);
        record.backsolve_seconds =
            diagnostics.direct_stats.backsolve_seconds.value_or(solve_seconds);
        record.symbolic_memory =
            diagnostics.direct_stats.symbolic_memory.value_or(0.0);
        record.numerical_factor_memory =
            diagnostics.direct_stats.numerical_factor_memory.value_or(0.0);
        record.estimated_in_core_peak_memory =
            diagnostics.direct_stats.estimated_in_core_peak_memory.value_or(0.0);
        if (diagnostics.direct_stats.estimated_in_core_peak_memory.has_value())
        {
            record.estimated_factor_memory_bytes =
                *diagnostics.direct_stats.estimated_in_core_peak_memory *
                1024.0;
        }
        record.out_of_core_minimum_memory =
            diagnostics.direct_stats.out_of_core_minimum_memory.value_or(0.0);
        record.process_rss_before_factorization =
            diagnostics.direct_stats.process_rss_before_factorization.value_or(
                0.0);
        record.process_rss_after_factorization =
            diagnostics.direct_stats.process_rss_after_factorization.value_or(
                0.0);
        record.process_rss_after_solve =
            diagnostics.direct_stats.process_rss_after_solve.value_or(0.0);
        record.memory_guard_estimated_extra_memory =
            diagnostics.direct_stats.memory_guard_estimated_extra_memory
                .value_or(0.0);
        record.direct_memory_limit =
            diagnostics.direct_stats.memory_limit.value_or(0.0);
        record.memory_guard_estimated_peak_memory =
            diagnostics.direct_stats.memory_guard_estimated_peak_memory
                .value_or(0.0);
        record.memory_guard_triggered =
            diagnostics.direct_stats.memory_guard_triggered;
        record.out_of_core_auto_switch_attempted =
            diagnostics.direct_stats.pardiso_out_of_core_auto_switch_attempted;
        if (diagnostics.direct_stats
                .pardiso_out_of_core_auto_switch_solver.has_value())
        {
            record.out_of_core_auto_switch_solver =
                solver_type_name(
                    *diagnostics.direct_stats
                         .pardiso_out_of_core_auto_switch_solver);
        }
        if (diagnostics.direct_stats.effective_pardiso_memory_mode.has_value())
        {
            record.effective_pardiso_memory_mode =
                std::string(
                    la::concepts::pardiso_memory_mode_name(
                        *diagnostics.direct_stats
                             .effective_pardiso_memory_mode));
        }
        if (diagnostics.direct_stats.pardiso_ldlt_robustness_profile.has_value())
        {
            record.pardiso_ldlt_robustness_profile =
                std::string(
                    la::concepts::pardiso_ldlt_robustness_profile_name(
                        *diagnostics.direct_stats
                             .pardiso_ldlt_robustness_profile));
        }
        record.pardiso_iparm_7 =
            diagnostics.direct_stats
                .pardiso_iterative_refinement_steps.value_or(0);
        record.pardiso_iparm_9 =
            diagnostics.direct_stats.pardiso_pivot_perturbation.value_or(0);
        record.pardiso_iparm_10 =
            diagnostics.direct_stats.pardiso_scaling.value_or(0);
        record.pardiso_perturbed_pivots =
            diagnostics.direct_stats.pardiso_perturbed_pivots.value_or(0);
        record.pardiso_positive_eigenvalues =
            diagnostics.direct_stats.pardiso_positive_eigenvalues.value_or(0);
        record.pardiso_negative_eigenvalues =
            diagnostics.direct_stats.pardiso_negative_eigenvalues.value_or(0);
        record.iteration_count =
            diagnostics.iterative_stats.iterations.value_or(0);
        record.final_residual =
            diagnostics.iterative_stats.final_error.value_or(0.0);
        record.preconditioner_setup_seconds =
            diagnostics.preconditioner_setup_seconds;
        record.setup_seconds = setup_seconds;
        record.solve_seconds = solve_seconds;
        record.linear_residual_absolute =
            diagnostics.linear_residual_absolute.value_or(0.0);
        record.linear_residual_relative =
            diagnostics.linear_residual_relative.value_or(0.0);
        record.initial_guess_norm =
            diagnostics.initial_guess_norm.value_or(0.0);
        record.initial_residual_absolute =
            diagnostics.initial_residual_absolute.value_or(0.0);
        record.initial_residual_relative =
            diagnostics.initial_residual_relative.value_or(0.0);
        record.backend_converged =
            diagnostics.iterative_stats.backend_converged.value_or(
                diagnostics.iterative_stats.converged.value_or(false));
        record.backend_reported_error =
            diagnostics.iterative_stats.backend_reported_error.value_or(
                diagnostics.iterative_stats.final_error.value_or(0.0));
        record.convergence_accepted_by_true_residual =
            diagnostics.iterative_stats
                .convergence_accepted_by_true_residual.value_or(false);
        record.residual_check_batches =
            diagnostics.iterative_stats.residual_check_batches.value_or(0);
        record.final_true_residual =
            diagnostics.iterative_stats.final_true_residual.value_or(
                diagnostics.linear_residual_relative.value_or(0.0));
        record.true_residual_stopping_used =
            diagnostics.iterative_stats.true_residual_stopping_used.value_or(
                false);
        record.matrix_norm =
            diagnostics.matrix_norm.value_or(0.0);
        record.matrix_symmetry_difference_norm =
            diagnostics.matrix_symmetry_difference_norm.value_or(0.0);
        record.matrix_relative_asymmetry =
            diagnostics.matrix_relative_asymmetry.value_or(0.0);
        record.residual_retry_attempted =
            diagnostics.residual_retry_attempted;
        if (diagnostics.residual_retry_solver.has_value())
        {
            record.residual_retry_solver =
                solver_type_name(*diagnostics.residual_retry_solver);
        }
        record.residual_before_retry =
            diagnostics.residual_before_retry.value_or(0.0);
        record.residual_after_retry =
            diagnostics.residual_after_retry.value_or(0.0);
        record.residual_correction_steps =
            diagnostics.direct_residual_correction_steps;
        record.residual_before_correction =
            diagnostics.residual_before_correction.value_or(0.0);
        record.residual_after_correction =
            diagnostics.residual_after_correction.value_or(0.0);
        record.solver_status =
            diagnostics.validation_rejection_reason.has_value()
                ? "rejected"
                : "success";
        return record;
    }

    template<typename CellIdType = int>
    struct XMarkingIndicatorComponents
    {
        finite_element::time_slabs::CellwiseSquaredError<CellIdType>
            lambda_y_squared_by_y_cell{};
        finite_element::time_slabs::CellwiseSquaredError<CellIdType>
            lambda_y_squared_by_x_cell{};
        finite_element::time_slabs::CellwiseSquaredError<CellIdType>
            initial_trace_squared_by_x_cell{};
        finite_element::time_slabs::CellwiseSquaredError<CellIdType>
            eta_squared_by_x_cell{};

        [[nodiscard]] double lambda_y_squared_total() const
        {
            return lambda_y_squared_by_y_cell.total();
        }

        [[nodiscard]] double initial_trace_squared_total() const
        {
            return initial_trace_squared_by_x_cell.total();
        }

        [[nodiscard]] double eta_squared_total() const
        {
            return eta_squared_by_x_cell.total();
        }
    };

    template<typename CellIdType = int>
    struct AdaptiveYIterationRecord
    {
        int inner_iteration = 0;

        int n_y_active_cells_before = 0;
        int n_y_active_cells_after = 0;
        int n_y_true_dofs_before = 0;
        int n_y_true_dofs_after = 0;
        ActiveGenerationStats y_generation_before{};
        ActiveGenerationStats y_generation_after{};
        int n_slabs = 0;
        int n_patches = 0;
        std::optional<int> copied_slab_cells{};
        std::optional<int> virtual_slab_cells{};
        std::string time_slab_backend = "MISSING";
        std::string estimator_backend = "MISSING";
        std::string time_slab_backend_requested = "MISSING";
        std::string time_slab_backend_effective = "MISSING";
        bool virtual_overlay_constructed = false;
        bool tilde_y_space_constructed = false;
        bool copied_estimator_fallback_enabled = false;
        bool estimator_uses_copied_fallback = false;
        bool copied_estimator_fallback_used = false;
        std::optional<int> estimator_fallback_copied_slab_cells{};
        int copied_fallback_component_count = 0;
        std::string copied_fallback_components{};
        bool strict_virtual_estimator = false;
        std::string strict_virtual_estimator_status = "MISSING";
        std::optional<int> copied_slab_cells_constructed_total{};
        std::optional<int> copied_slab_cells_constructed_for_fallback{};
        std::optional<int> source_mesh_mutation_count{};

        double lambda_y_squared = 0.0;
        double initial_trace_squared = 0.0;
        double eta_squared = 0.0;

        double y_estimator_squared = 0.0;
        double y_flux_squared = 0.0;
        double y_reconstruction_squared = 0.0;
        double divergence_residual_squared = 0.0;
        double divergence_residual_l2 = 0.0;
        double y_estimator_threshold_squared = 0.0;
        double configured_rho = 0.0;
        std::optional<double> effective_rho{};
        bool effective_rho_available = false;
        std::string effective_rho_reason = "not_computed";
        double y_estimator_threshold_configured_rho_squared = 0.0;
        std::optional<double> y_estimator_threshold_effective_rho_squared{};
        double posteriori_factor_configured_rho = 0.0;
        std::optional<double> posteriori_factor_effective_rho{};
        double posteriori_estimator_configured_rho_squared = 0.0;
        std::optional<double> posteriori_estimator_effective_rho_squared{};
        std::optional<double> posteriori_improvement_factor{};
        bool force_accept_inner_with_effective_rho = false;
        bool configured_rho_ignored_for_inner_acceptance = false;
        bool effective_rho_acceptance_used = false;
        std::string effective_rho_acceptance_reason = "not_used";
        bool g_estimator_enabled = false;
        bool g_estimator_computed = false;
        std::string g_estimator_skipped_reason = "not_requested";
        bool g_space_constructed = false;
        std::optional<int> g_true_dofs{};
        int g_solve_count = 0;
        std::string g_solver_status = "not_requested";
        std::optional<double> g_assembly_seconds{};
        std::optional<double> g_solve_seconds{};
        std::optional<double> g_solver_residual{};
        std::optional<double> g_solver_relative_residual{};
        std::optional<double> g_rhs_inf_norm{};
        std::optional<double> g_lambda_inf_norm{};
        bool g_lambda_difference_available = false;
        std::optional<double> g_lambda_difference_squared{};
        std::optional<double> g_lambda_difference{};
        double iteration_seconds = 0.0;
        double elapsed_seconds = 0.0;
        MainSolveDiagnosticsRecord main_solve{};

        bool stopping_criterion_satisfied = false;
        bool refined_y = false;
        bool uniform_y_refinement = false;
        std::string y_refinement_mode = "adaptive";
        int y_refinement_target_cells = 0;
        bool local_time_slab_closure_applied = false;
        int local_time_slab_closure_marked_split_cells = 0;
        int local_time_slab_closure_temporal_waves = 0;
        int local_time_slab_closure_temporally_refined_cells = 0;
        AdaptiveRefinementStatistics refinement_statistics{};

        std::vector<CellIdType> marked_y_cells{};
        finite_element::time_slabs::CellwiseTimeSlabEstimatorError<CellIdType>
            estimator{};
    };

    template<typename CellIdType = int>
    struct AdaptiveOuterIterationRecord
    {
        int outer_iteration = 0;

        int n_x_active_cells_before = 0;
        int n_x_active_cells_after = 0;
        int n_x_true_dofs_before = 0;
        int n_x_true_dofs_after = 0;
        ActiveGenerationStats x_generation_before{};
        ActiveGenerationStats x_generation_after{};

        int n_y_active_cells_before = 0;
        int n_y_active_cells_after = 0;
        int n_y_true_dofs_before = 0;
        int n_y_true_dofs_after = 0;
        ActiveGenerationStats y_generation_before{};
        ActiveGenerationStats y_generation_after{};

        std::vector<AdaptiveYIterationRecord<CellIdType>> inner_iterations{};

        bool y_converged = false;
        bool stopped_on_empty_y_marking = false;
        bool refined_x = false;
        bool uniform_x_refinement = false;
        bool uniform_y_refinement = false;
        std::string x_refinement_mode = "adaptive";
        bool x_marking_empty = false;
        AdaptiveRefinementStatistics refinement_statistics{};

        double lambda_y_squared = 0.0;
        double initial_trace_squared = 0.0;
        double eta_squared = 0.0;
        double y_estimator_squared = 0.0;
        double y_estimator_threshold_squared = 0.0;
        double configured_rho = 0.0;
        std::optional<double> effective_rho{};
        bool effective_rho_available = false;
        std::string effective_rho_reason = "not_computed";
        double y_estimator_threshold_configured_rho_squared = 0.0;
        std::optional<double> y_estimator_threshold_effective_rho_squared{};
        double posteriori_factor_configured_rho = 0.0;
        std::optional<double> posteriori_factor_effective_rho{};
        double posteriori_estimator_configured_rho_squared = 0.0;
        std::optional<double> posteriori_estimator_effective_rho_squared{};
        std::optional<double> posteriori_improvement_factor{};
        bool force_accept_inner_with_effective_rho = false;
        bool configured_rho_ignored_for_inner_acceptance = false;
        bool effective_rho_acceptance_used = false;
        std::string effective_rho_acceptance_reason = "not_used";
        bool g_estimator_enabled = false;
        bool g_estimator_computed = false;
        std::string g_estimator_skipped_reason = "not_requested";
        bool g_space_constructed = false;
        std::optional<int> g_true_dofs{};
        int g_solve_count = 0;
        std::string g_solver_status = "not_requested";
        std::optional<double> g_assembly_seconds{};
        std::optional<double> g_solve_seconds{};
        std::optional<double> g_solver_residual{};
        std::optional<double> g_solver_relative_residual{};
        std::optional<double> g_rhs_inf_norm{};
        std::optional<double> g_lambda_inf_norm{};
        bool g_lambda_difference_available = false;
        std::optional<double> g_lambda_difference_squared{};
        std::optional<double> g_lambda_difference{};
        double iteration_seconds = 0.0;
        double elapsed_seconds = 0.0;

        std::vector<CellIdType> marked_x_cells{};
        finite_element::time_slabs::CellwiseTimeSlabEstimatorError<CellIdType>
            final_y_estimator{};
        XMarkingIndicatorComponents<CellIdType> x_indicator_components{};

        [[nodiscard]] int n_inner_iterations() const noexcept
        {
            return static_cast<int>(inner_iterations.size());
        }

        [[nodiscard]] const AdaptiveYIterationRecord<CellIdType>& last_inner_iteration() const
        {
            if (inner_iterations.empty())
            {
                throw std::runtime_error(
                    "AdaptiveOuterIterationRecord::last_inner_iteration: no inner iterations available.");
            }

            return inner_iterations.back();
        }
    };

    template<class Backend, class XSpaceType, class YSpaceType>
    struct AdaptiveResult
    {
        using Vector = typename Backend::Vector;
        using XFunction = finite_element::Function<XSpaceType, Vector>;
        using YFunction = finite_element::Function<YSpaceType, Vector>;

        std::string problem_name{};
        std::string problem_description{};
        int spatial_dimension = XSpaceType::GT::dim_space_v;
        int polynomial_degree = XSpaceType::FETraitsType::p_space_v;

        bool timing_enabled = false;
        std::string timing_detail_level = "summary";
        std::vector<TimingRecord> timing_records{};
        std::vector<AdaptiveOuterIterationRecord<int>> outer_iterations{};

        bool compute_g_estimator = false;
        bool compute_g_estimator_on_empty_y_marking_stop = false;
        bool compute_g_estimator_every_inner_iteration = false;
        std::string g_solver = "same_as_main";
        double g_solver_tolerance = 0.0;
        double g_solver_memory_limit_mb = 0.0;

        bool converged = false;
        bool terminated_early = false;
        std::string termination_reason{};

        std::optional<YFunction> final_lambda_delta{};
        std::optional<XFunction> final_u_delta{};
        std::optional<
            finite_element::time_slabs::CellwiseTimeSlabEstimatorError<int>>
            final_y_estimator{};
        std::optional<XMarkingIndicatorComponents<int>> final_x_indicators{};

        [[nodiscard]] int n_outer_iterations() const noexcept
        {
            return static_cast<int>(outer_iterations.size());
        }

        [[nodiscard]] const AdaptiveOuterIterationRecord<int>& last_outer_iteration() const
        {
            if (outer_iterations.empty())
            {
                throw std::runtime_error(
                    "AdaptiveResult::last_outer_iteration: no outer iterations available.");
            }

            return outer_iterations.back();
        }
    };
}

#pragma once

#include <cmath>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

#include "../adaptive_parameters.hpp"
#include "../adaptive_result.hpp"

namespace adaptive_algorithm
{
    namespace detail
    {
        inline void ensure_output_directory(
            const std::filesystem::path& output_directory)
        {
            if (output_directory.empty())
            {
                throw std::runtime_error(
                    "adaptive_algorithm output: output_directory must not be empty.");
            }

            std::filesystem::create_directories(output_directory);
        }

        [[nodiscard]] inline std::string escape_summary_field(
            const std::string_view value)
        {
            std::string escaped;
            escaped.reserve(value.size());

            bool needs_quotes = value.empty();

            for (const char ch : value)
            {
                switch (ch)
                {
                case '"':
                    escaped += "\"\"";
                    needs_quotes = true;
                    break;
                case '\n':
                    escaped += "\\n";
                    needs_quotes = true;
                    break;
                case '\r':
                    escaped += "\\r";
                    needs_quotes = true;
                    break;
                case ',':
                    escaped += ch;
                    needs_quotes = true;
                    break;
                default:
                    escaped += ch;
                    break;
                }
            }

            if (!value.empty() &&
                (value.front() == ' ' || value.front() == '\t' ||
                 value.back() == ' ' || value.back() == '\t'))
            {
                needs_quotes = true;
            }

            if (!needs_quotes)
                return escaped;

            return '"' + escaped + '"';
        }

        inline void write_optional_int(
            std::ofstream& out,
            const std::optional<int>& value)
        {
            if (value.has_value())
                out << *value;
            else
                out << "MISSING";
        }

        template<class T>
        inline void write_optional_value(
            std::ofstream& out,
            const std::optional<T>& value)
        {
            if (value.has_value())
                out << *value;
            else
                out << "MISSING";
        }

        [[nodiscard]] inline bool csv_needs_header(
            const std::filesystem::path& path)
        {
            return !std::filesystem::exists(path) ||
                std::filesystem::file_size(path) == 0;
        }
    }

    template<class Backend, class XSpaceType, class YSpaceType>
    void write_summary_text(
        const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result,
        const AdaptiveOutputSettings& settings)
    {
        if (!settings.export_history || settings.output_directory.empty())
            return;

        detail::ensure_output_directory(settings.output_directory);

        std::ofstream out(settings.output_directory / settings.summary_filename);
        if (!out)
        {
            throw std::runtime_error(
                "write_summary_text: failed to open summary output file.");
        }

        out
            << "problem_name," << detail::escape_summary_field(result.problem_name) << '\n'
            << "problem_description," << detail::escape_summary_field(result.problem_description) << '\n'
            << "spatial_dimension," << result.spatial_dimension << '\n'
            << "polynomial_degree," << result.polynomial_degree << '\n'
            << "outer_iterations," << result.n_outer_iterations() << '\n'
            << "converged," << (result.converged ? 1 : 0) << '\n'
            << "terminated_early," << (result.terminated_early ? 1 : 0) << '\n'
            << "termination_reason," << detail::escape_summary_field(result.termination_reason) << '\n';
        out << "compute_g_estimator,"
            << (result.compute_g_estimator ? 1 : 0) << '\n';
        out << "compute_g_estimator_on_empty_y_marking_stop,"
            << (result.compute_g_estimator_on_empty_y_marking_stop ? 1 : 0)
            << '\n';
        out << "compute_g_estimator_every_inner_iteration,"
            << (result.compute_g_estimator_every_inner_iteration ? 1 : 0)
            << '\n';
        out << "g_solver,"
            << detail::escape_summary_field(result.g_solver) << '\n';
        out << "g_solver_tolerance,"
            << result.g_solver_tolerance << '\n';
        out << "g_solver_memory_limit_mb,"
            << result.g_solver_memory_limit_mb << '\n';

        if (result.n_outer_iterations() == 0)
            return;

        const auto& last = result.last_outer_iteration();
        out << "final_eta_squared," << last.eta_squared << '\n';
        out << "final_eta," << std::sqrt(last.eta_squared) << '\n';
        out << "final_y_estimator_squared," << last.y_estimator_squared << '\n';
        out << "final_y_estimator," << std::sqrt(last.y_estimator_squared) << '\n';
        out << "final_y_threshold_squared," << last.y_estimator_threshold_squared << '\n';
        out << "final_y_threshold," << std::sqrt(last.y_estimator_threshold_squared) << '\n';
        out << "final_configured_rho," << last.configured_rho << '\n';
        out << "final_effective_rho,";
        detail::write_optional_value(out, last.effective_rho);
        out << '\n';
        out << "final_effective_rho_available,"
            << (last.effective_rho_available ? 1 : 0) << '\n';
        out << "final_effective_rho_reason,"
            << detail::escape_summary_field(last.effective_rho_reason) << '\n';
        out << "final_y_threshold_configured_rho_squared,"
            << last.y_estimator_threshold_configured_rho_squared << '\n';
        out << "final_y_threshold_effective_rho_squared,";
        detail::write_optional_value(
            out,
            last.y_estimator_threshold_effective_rho_squared);
        out << '\n';
        out << "final_posteriori_factor_configured_rho,"
            << last.posteriori_factor_configured_rho << '\n';
        out << "final_posteriori_factor_effective_rho,";
        detail::write_optional_value(
            out,
            last.posteriori_factor_effective_rho);
        out << '\n';
        out << "final_posteriori_estimator_configured_rho_squared,"
            << last.posteriori_estimator_configured_rho_squared << '\n';
        out << "final_posteriori_estimator_effective_rho_squared,";
        detail::write_optional_value(
            out,
            last.posteriori_estimator_effective_rho_squared);
        out << '\n';
        out << "final_posteriori_improvement_factor,";
        detail::write_optional_value(
            out,
            last.posteriori_improvement_factor);
        out << '\n';
        out << "final_force_accept_inner_with_effective_rho,"
            << (last.force_accept_inner_with_effective_rho ? 1 : 0)
            << '\n';
        out << "final_configured_rho_ignored_for_inner_acceptance,"
            << (last.configured_rho_ignored_for_inner_acceptance ? 1 : 0)
            << '\n';
        out << "final_effective_rho_acceptance_used,"
            << (last.effective_rho_acceptance_used ? 1 : 0) << '\n';
        out << "final_effective_rho_acceptance_reason,"
            << detail::escape_summary_field(
                   last.effective_rho_acceptance_reason)
            << '\n';
        out << "final_g_estimator_enabled,"
            << (last.g_estimator_enabled ? 1 : 0) << '\n';
        out << "final_g_estimator_computed,"
            << (last.g_estimator_computed ? 1 : 0) << '\n';
        out << "final_g_estimator_skipped_reason,"
            << detail::escape_summary_field(last.g_estimator_skipped_reason)
            << '\n';
        out << "final_g_space_constructed,"
            << (last.g_space_constructed ? 1 : 0) << '\n';
        out << "final_g_true_dofs,";
        detail::write_optional_int(out, last.g_true_dofs);
        out << '\n';
        out << "final_g_solve_count," << last.g_solve_count << '\n';
        out << "final_g_solver_status,"
            << detail::escape_summary_field(last.g_solver_status) << '\n';
        out << "final_g_assembly_seconds,";
        detail::write_optional_value(out, last.g_assembly_seconds);
        out << '\n';
        out << "final_g_solve_seconds,";
        detail::write_optional_value(out, last.g_solve_seconds);
        out << '\n';
        out << "final_g_solver_residual,";
        detail::write_optional_value(out, last.g_solver_residual);
        out << '\n';
        out << "final_g_solver_relative_residual,";
        detail::write_optional_value(out, last.g_solver_relative_residual);
        out << '\n';
        out << "final_g_rhs_inf_norm,";
        detail::write_optional_value(out, last.g_rhs_inf_norm);
        out << '\n';
        out << "final_g_lambda_inf_norm,";
        detail::write_optional_value(out, last.g_lambda_inf_norm);
        out << '\n';
        out << "final_g_lambda_difference_available,"
            << (last.g_lambda_difference_available ? 1 : 0) << '\n';
        out << "final_g_lambda_difference_squared,";
        detail::write_optional_value(out, last.g_lambda_difference_squared);
        out << '\n';
        out << "final_g_lambda_difference,";
        detail::write_optional_value(out, last.g_lambda_difference);
        out << '\n';

        if (settings.save_mesh_statistics)
        {
            out << "final_x_active_cells," << last.n_x_active_cells_before << '\n';
            out << "final_x_true_dofs," << last.n_x_true_dofs_before << '\n';
            out << "final_y_active_cells," << last.n_y_active_cells_after << '\n';
            out << "final_y_true_dofs," << last.n_y_true_dofs_after << '\n';

            if (last.n_x_active_cells_after != last.n_x_active_cells_before ||
                last.n_x_true_dofs_after != last.n_x_true_dofs_before)
            {
                out << "final_refined_x_active_cells,"
                    << last.n_x_active_cells_after << '\n';
                out << "final_refined_x_true_dofs,"
                    << last.n_x_true_dofs_after << '\n';
            }
        }

        if (settings.save_estimator_components)
        {
            out << "final_lambda_y_squared," << last.lambda_y_squared << '\n';
            out << "final_lambda_y," << std::sqrt(last.lambda_y_squared) << '\n';
            out << "final_initial_trace_squared," << last.initial_trace_squared << '\n';
            out << "final_initial_trace," << std::sqrt(last.initial_trace_squared) << '\n';
            out << "final_y_flux_squared," << last.final_y_estimator.equilibrated_flux_y_squared.total() << '\n';
            out << "final_y_flux," << std::sqrt(last.final_y_estimator.equilibrated_flux_y_squared.total()) << '\n';
            out << "final_y_reconstruction_squared," << last.final_y_estimator.reconstruction_y_squared.total() << '\n';
            out << "final_y_reconstruction," << std::sqrt(last.final_y_estimator.reconstruction_y_squared.total()) << '\n';
            out << "final_divergence_residual_squared," << last.final_y_estimator.divergence_residual_total() << '\n';
            out << "final_divergence_residual," << std::sqrt(last.final_y_estimator.divergence_residual_total()) << '\n';
        }
    }

    template<typename OuterRecord>
    void append_partial_outer_history_csv(
        const OuterRecord& record,
        const AdaptiveOutputSettings& settings,
        std::string_view history_status)
    {
        if (!settings.export_history || settings.output_directory.empty())
            return;

        detail::ensure_output_directory(settings.output_directory);
        const auto path =
            settings.output_directory / settings.partial_outer_history_filename;
        const bool needs_header = detail::csv_needs_header(path);

        std::ofstream out(path, std::ios::app);
        if (!out)
        {
            throw std::runtime_error(
                "append_partial_outer_history_csv: failed to open partial outer-history output file.");
        }

        if (needs_header)
        {
            out
                << "history_status"
                << ",outer_iteration"
                << ",x_active_cells_before"
                << ",x_active_cells_after"
                << ",x_true_dofs_before"
                << ",x_true_dofs_after"
                << ",x_generation_min_before"
                << ",x_generation_max_before"
                << ",x_generation_distinct_before"
                << ",x_generation_min_after"
                << ",x_generation_max_after"
                << ",x_generation_distinct_after"
                << ",y_active_cells_before"
                << ",y_active_cells_after"
                << ",y_true_dofs_before"
                << ",y_true_dofs_after"
                << ",y_generation_min_before"
                << ",y_generation_max_before"
                << ",y_generation_distinct_before"
                << ",y_generation_min_after"
                << ",y_generation_max_after"
                << ",y_generation_distinct_after"
                << ",iteration_seconds"
                << ",elapsed_seconds"
                << ",inner_iterations"
                << ",y_converged"
                << ",stopped_on_empty_y_marking"
                << ",refined_x"
                << ",uniform_x_refinement"
                << ",x_refinement_mode"
                << ",uniform_y_refinement"
                << ",x_marking_empty"
                << ",lambda_y_squared"
                << ",eta_squared"
                << ",y_estimator_squared"
                << ",y_estimator_threshold_squared"
                << ",configured_rho"
                << ",effective_rho"
                << ",effective_rho_available"
                << ",effective_rho_reason"
                << ",y_estimator_threshold_configured_rho_squared"
                << ",y_estimator_threshold_effective_rho_squared"
                << ",posteriori_factor_configured_rho"
                << ",posteriori_factor_effective_rho"
                << ",posteriori_estimator_configured_rho_squared"
                << ",posteriori_estimator_effective_rho_squared"
                << ",posteriori_improvement_factor"
                << ",force_accept_inner_with_effective_rho"
                << ",configured_rho_ignored_for_inner_acceptance"
                << ",effective_rho_acceptance_used"
                << ",effective_rho_acceptance_reason"
                << ",g_estimator_enabled"
                << ",g_estimator_computed"
                << ",g_estimator_skipped_reason"
                << ",g_space_constructed"
                << ",g_true_dofs"
                << ",g_solve_count"
                << ",g_solver_status"
                << ",g_assembly_seconds"
                << ",g_solve_seconds"
                << ",g_solver_residual"
                << ",g_solver_relative_residual"
                << ",g_rhs_inf_norm"
                << ",g_lambda_inf_norm"
                << ",g_lambda_difference_available"
                << ",g_lambda_difference_squared"
                << ",g_lambda_difference"
                << ",marked_x_cells"
                << ",x_marked_cells"
                << ",x_refined_cells"
                << ",x_closure_cells"
                << ",x_min_generation"
                << ",x_max_generation"
                << ",x_distinct_generations"
                << ",y_min_generation"
                << ",y_max_generation"
                << ",y_distinct_generations"
                << ",edge_interval_queries"
                << ",edge_interval_records_visited"
                << ",2d_edge_interval_queries"
                << ",2d_edge_interval_records_visited"
                << ",full_active_scans"
                << ",x_2d_edge_interval_queries"
                << ",x_2d_edge_interval_records_visited"
                << ",x_full_active_scans\n";
        }

        out
            << detail::escape_summary_field(history_status)
            << ',' << record.outer_iteration
            << ',' << record.n_x_active_cells_before
            << ',' << record.n_x_active_cells_after
            << ',' << record.n_x_true_dofs_before
            << ',' << record.n_x_true_dofs_after
            << ',' << record.x_generation_before.min_generation
            << ',' << record.x_generation_before.max_generation
            << ',' << record.x_generation_before.distinct_generations
            << ',' << record.x_generation_after.min_generation
            << ',' << record.x_generation_after.max_generation
            << ',' << record.x_generation_after.distinct_generations
            << ',' << record.n_y_active_cells_before
            << ',' << record.n_y_active_cells_after
            << ',' << record.n_y_true_dofs_before
            << ',' << record.n_y_true_dofs_after
            << ',' << record.y_generation_before.min_generation
            << ',' << record.y_generation_before.max_generation
            << ',' << record.y_generation_before.distinct_generations
            << ',' << record.y_generation_after.min_generation
            << ',' << record.y_generation_after.max_generation
            << ',' << record.y_generation_after.distinct_generations
            << ',' << record.iteration_seconds
            << ',' << record.elapsed_seconds
            << ',' << record.n_inner_iterations()
            << ',' << (record.y_converged ? 1 : 0)
            << ',' << (record.stopped_on_empty_y_marking ? 1 : 0)
            << ',' << (record.refined_x ? 1 : 0)
            << ',' << (record.uniform_x_refinement ? 1 : 0)
            << ',' << detail::escape_summary_field(record.x_refinement_mode)
            << ',' << (record.uniform_y_refinement ? 1 : 0)
            << ',' << (record.x_marking_empty ? 1 : 0)
            << ',' << record.lambda_y_squared
            << ',' << record.eta_squared
            << ',' << record.y_estimator_squared
            << ',' << record.y_estimator_threshold_squared
            << ',' << record.configured_rho
            << ',';
        detail::write_optional_value(out, record.effective_rho);
        out
            << ',' << (record.effective_rho_available ? 1 : 0)
            << ',' << detail::escape_summary_field(record.effective_rho_reason)
            << ',' << record.y_estimator_threshold_configured_rho_squared
            << ',';
        detail::write_optional_value(
            out,
            record.y_estimator_threshold_effective_rho_squared);
        out
            << ',' << record.posteriori_factor_configured_rho
            << ',';
        detail::write_optional_value(out, record.posteriori_factor_effective_rho);
        out
            << ',' << record.posteriori_estimator_configured_rho_squared
            << ',';
        detail::write_optional_value(
            out,
            record.posteriori_estimator_effective_rho_squared);
        out << ',';
        detail::write_optional_value(
            out,
            record.posteriori_improvement_factor);
        out
            << ',' << (record.force_accept_inner_with_effective_rho ? 1 : 0)
            << ',' << (record.configured_rho_ignored_for_inner_acceptance ? 1 : 0)
            << ',' << (record.effective_rho_acceptance_used ? 1 : 0)
            << ',' << detail::escape_summary_field(
                record.effective_rho_acceptance_reason)
            << ',' << (record.g_estimator_enabled ? 1 : 0)
            << ',' << (record.g_estimator_computed ? 1 : 0)
            << ',' << detail::escape_summary_field(
                record.g_estimator_skipped_reason)
            << ',' << (record.g_space_constructed ? 1 : 0)
            << ',';
        detail::write_optional_int(out, record.g_true_dofs);
        out
            << ',' << record.g_solve_count
            << ',' << detail::escape_summary_field(record.g_solver_status)
            << ',';
        detail::write_optional_value(out, record.g_assembly_seconds);
        out << ',';
        detail::write_optional_value(out, record.g_solve_seconds);
        out << ',';
        detail::write_optional_value(out, record.g_solver_residual);
        out << ',';
        detail::write_optional_value(out, record.g_solver_relative_residual);
        out << ',';
        detail::write_optional_value(out, record.g_rhs_inf_norm);
        out << ',';
        detail::write_optional_value(out, record.g_lambda_inf_norm);
        out
            << ',' << (record.g_lambda_difference_available ? 1 : 0)
            << ',';
        detail::write_optional_value(
            out,
            record.g_lambda_difference_squared);
        out << ',';
        detail::write_optional_value(out, record.g_lambda_difference);
        out
            << ',' << record.marked_x_cells.size()
            << ',' << record.marked_x_cells.size();
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.refined_cells);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.closure_cells);
        out
            << ',' << record.x_generation_after.min_generation
            << ',' << record.x_generation_after.max_generation
            << ',' << record.x_generation_after.distinct_generations
            << ',' << record.y_generation_after.min_generation
            << ',' << record.y_generation_after.max_generation
            << ',' << record.y_generation_after.distinct_generations
            << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_queries);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_records_visited);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_queries);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_records_visited);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.full_active_scans);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_queries);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_records_visited);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.full_active_scans);
        out << '\n';
        out.flush();
    }

    template<typename InnerRecord>
    void append_partial_inner_history_csv(
        int outer_iteration,
        const InnerRecord& record,
        const AdaptiveOutputSettings& settings,
        std::string_view history_status)
    {
        if (!settings.export_history ||
            settings.output_directory.empty())
        {
            return;
        }

        detail::ensure_output_directory(settings.output_directory);
        const auto path =
            settings.output_directory / settings.partial_inner_history_filename;
        const bool needs_header = detail::csv_needs_header(path);

        std::ofstream out(path, std::ios::app);
        if (!out)
        {
            throw std::runtime_error(
                "append_partial_inner_history_csv: failed to open partial inner-history output file.");
        }

        if (needs_header)
        {
            out
                << "history_status"
                << ",outer_iteration"
                << ",inner_iteration"
                << ",y_active_cells_before"
                << ",y_active_cells_after"
                << ",y_true_dofs_before"
                << ",y_true_dofs_after"
                << ",y_generation_min_before"
                << ",y_generation_max_before"
                << ",y_generation_distinct_before"
                << ",y_generation_min_after"
                << ",y_generation_max_after"
                << ",y_generation_distinct_after"
                << ",iteration_seconds"
                << ",elapsed_seconds"
                << ",n_slabs"
                << ",n_patches"
                << ",copied_slab_cells"
                << ",virtual_slab_cells"
                << ",time_slab_backend"
                << ",estimator_backend"
                << ",estimator_uses_copied_fallback"
                << ",estimator_fallback_copied_slab_cells"
                << ",time_slab_backend_requested"
                << ",time_slab_backend_effective"
                << ",virtual_overlay_constructed"
                << ",tilde_y_space_constructed"
                << ",copied_estimator_fallback_enabled"
                << ",copied_estimator_fallback_used"
                << ",copied_fallback_component_count"
                << ",copied_fallback_components"
                << ",strict_virtual_estimator"
                << ",strict_virtual_estimator_status"
                << ",copied_slab_cells_constructed_total"
                << ",copied_slab_cells_constructed_for_fallback"
                << ",source_mesh_mutation_count"
                << ",lambda_y_squared"
                << ",eta_squared"
                << ",y_estimator_squared"
                << ",y_estimator_threshold_squared"
                << ",configured_rho"
                << ",effective_rho"
                << ",effective_rho_available"
                << ",effective_rho_reason"
                << ",y_estimator_threshold_configured_rho_squared"
                << ",y_estimator_threshold_effective_rho_squared"
                << ",posteriori_factor_configured_rho"
                << ",posteriori_factor_effective_rho"
                << ",posteriori_estimator_configured_rho_squared"
                << ",posteriori_estimator_effective_rho_squared"
                << ",posteriori_improvement_factor"
                << ",force_accept_inner_with_effective_rho"
                << ",configured_rho_ignored_for_inner_acceptance"
                << ",effective_rho_acceptance_used"
                << ",effective_rho_acceptance_reason"
                << ",g_estimator_enabled"
                << ",g_estimator_computed"
                << ",g_estimator_skipped_reason"
                << ",g_space_constructed"
                << ",g_true_dofs"
                << ",g_solve_count"
                << ",g_solver_status"
                << ",g_assembly_seconds"
                << ",g_solve_seconds"
                << ",g_solver_residual"
                << ",g_solver_relative_residual"
                << ",g_rhs_inf_norm"
                << ",g_lambda_inf_norm"
                << ",g_lambda_difference_available"
                << ",g_lambda_difference_squared"
                << ",g_lambda_difference"
                << ",stopping_criterion_satisfied"
                << ",refined_y"
                << ",uniform_y_refinement"
                << ",y_refinement_mode"
                << ",y_refinement_target_cells"
                << ",marked_y_cells"
                << ",y_marked_cells"
                << ",y_refined_cells"
                << ",y_closure_cells"
                << ",y_min_generation"
                << ",y_max_generation"
                << ",y_distinct_generations"
                << ",edge_interval_queries"
                << ",edge_interval_records_visited"
                << ",2d_edge_interval_queries"
                << ",2d_edge_interval_records_visited"
                << ",full_active_scans"
                << ",y_2d_edge_interval_queries"
                << ",y_2d_edge_interval_records_visited"
                << ",y_full_active_scans"
                << ",local_time_slab_closure_applied"
                << ",local_time_slab_closure_marked_split_cells"
                << ",local_time_slab_closure_temporal_waves"
                << ",local_time_slab_closure_temporally_refined_cells"
                << ",main_solve_selected_solver"
                << ",main_solve_effective_solver"
                << ",main_solve_solver_status"
                << ",main_solve_matrix_rows"
                << ",main_solve_matrix_cols"
                << ",main_solve_matrix_nnz"
                << ",main_solve_n"
                << ",main_solve_nnz_matrix"
                << ",main_solve_nnz_factors"
                << ",main_solve_factor_nnz"
                << ",main_solve_fill_ratio"
                << ",main_solve_symbolic_analysis_seconds"
                << ",main_solve_symbolic_analysis_reused"
                << ",main_solve_symbolic_pattern_cache_hits"
                << ",main_solve_symbolic_pattern_cache_misses"
                << ",main_solve_numeric_factorization_seconds"
                << ",main_solve_backsolve_seconds"
                << ",main_solve_estimated_factor_memory_bytes"
                << ",main_solve_preconditioner_setup_seconds"
                << ",main_solve_setup_seconds"
                << ",main_solve_solve_seconds\n";
        }

        const auto& main_solve = record.main_solve;
        out
            << detail::escape_summary_field(history_status)
            << ',' << outer_iteration
            << ',' << record.inner_iteration
            << ',' << record.n_y_active_cells_before
            << ',' << record.n_y_active_cells_after
            << ',' << record.n_y_true_dofs_before
            << ',' << record.n_y_true_dofs_after
            << ',' << record.y_generation_before.min_generation
            << ',' << record.y_generation_before.max_generation
            << ',' << record.y_generation_before.distinct_generations
            << ',' << record.y_generation_after.min_generation
            << ',' << record.y_generation_after.max_generation
            << ',' << record.y_generation_after.distinct_generations
            << ',' << record.iteration_seconds
            << ',' << record.elapsed_seconds
            << ',' << record.n_slabs
            << ',' << record.n_patches
            << ',';
        detail::write_optional_int(out, record.copied_slab_cells);
        out << ',';
        detail::write_optional_int(out, record.virtual_slab_cells);
        out
            << ',' << detail::escape_summary_field(record.time_slab_backend)
            << ',' << detail::escape_summary_field(record.estimator_backend)
            << ',' << (record.estimator_uses_copied_fallback ? 1 : 0)
            << ',';
        detail::write_optional_int(
            out,
            record.estimator_fallback_copied_slab_cells);
        out
            << ',' << detail::escape_summary_field(
                record.time_slab_backend_requested)
            << ',' << detail::escape_summary_field(
                record.time_slab_backend_effective)
            << ',' << (record.virtual_overlay_constructed ? 1 : 0)
            << ',' << (record.tilde_y_space_constructed ? 1 : 0)
            << ',' << (record.copied_estimator_fallback_enabled ? 1 : 0)
            << ',' << (record.copied_estimator_fallback_used ? 1 : 0)
            << ',' << record.copied_fallback_component_count
            << ',' << detail::escape_summary_field(
                record.copied_fallback_components)
            << ',' << (record.strict_virtual_estimator ? 1 : 0)
            << ',' << detail::escape_summary_field(
                record.strict_virtual_estimator_status)
            << ',';
        detail::write_optional_int(
            out,
            record.copied_slab_cells_constructed_total);
        out << ',';
        detail::write_optional_int(
            out,
            record.copied_slab_cells_constructed_for_fallback);
        out << ',';
        detail::write_optional_int(
            out,
            record.source_mesh_mutation_count);
        out
            << ',' << record.lambda_y_squared
            << ',' << record.eta_squared
            << ',' << record.y_estimator_squared
            << ',' << record.y_estimator_threshold_squared
            << ',' << record.configured_rho
            << ',';
        detail::write_optional_value(out, record.effective_rho);
        out
            << ',' << (record.effective_rho_available ? 1 : 0)
            << ',' << detail::escape_summary_field(record.effective_rho_reason)
            << ',' << record.y_estimator_threshold_configured_rho_squared
            << ',';
        detail::write_optional_value(
            out,
            record.y_estimator_threshold_effective_rho_squared);
        out
            << ',' << record.posteriori_factor_configured_rho
            << ',';
        detail::write_optional_value(out, record.posteriori_factor_effective_rho);
        out
            << ',' << record.posteriori_estimator_configured_rho_squared
            << ',';
        detail::write_optional_value(
            out,
            record.posteriori_estimator_effective_rho_squared);
        out << ',';
        detail::write_optional_value(
            out,
            record.posteriori_improvement_factor);
        out
            << ',' << (record.force_accept_inner_with_effective_rho ? 1 : 0)
            << ',' << (record.configured_rho_ignored_for_inner_acceptance ? 1 : 0)
            << ',' << (record.effective_rho_acceptance_used ? 1 : 0)
            << ',' << detail::escape_summary_field(
                record.effective_rho_acceptance_reason)
            << ',' << (record.g_estimator_enabled ? 1 : 0)
            << ',' << (record.g_estimator_computed ? 1 : 0)
            << ',' << detail::escape_summary_field(
                record.g_estimator_skipped_reason)
            << ',' << (record.g_space_constructed ? 1 : 0)
            << ',';
        detail::write_optional_int(out, record.g_true_dofs);
        out
            << ',' << record.g_solve_count
            << ',' << detail::escape_summary_field(record.g_solver_status)
            << ',';
        detail::write_optional_value(out, record.g_assembly_seconds);
        out << ',';
        detail::write_optional_value(out, record.g_solve_seconds);
        out << ',';
        detail::write_optional_value(out, record.g_solver_residual);
        out << ',';
        detail::write_optional_value(out, record.g_solver_relative_residual);
        out << ',';
        detail::write_optional_value(out, record.g_rhs_inf_norm);
        out << ',';
        detail::write_optional_value(out, record.g_lambda_inf_norm);
        out
            << ',' << (record.g_lambda_difference_available ? 1 : 0)
            << ',';
        detail::write_optional_value(
            out,
            record.g_lambda_difference_squared);
        out << ',';
        detail::write_optional_value(out, record.g_lambda_difference);
        out
            << ',' << (record.stopping_criterion_satisfied ? 1 : 0)
            << ',' << (record.refined_y ? 1 : 0)
            << ',' << (record.uniform_y_refinement ? 1 : 0)
            << ',' << detail::escape_summary_field(record.y_refinement_mode)
            << ',' << record.y_refinement_target_cells
            << ',' << record.marked_y_cells.size()
            << ',' << record.marked_y_cells.size();
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.refined_cells);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.closure_cells);
        out
            << ',' << record.y_generation_after.min_generation
            << ',' << record.y_generation_after.max_generation
            << ',' << record.y_generation_after.distinct_generations
            << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_queries);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_records_visited);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_queries);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_records_visited);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.full_active_scans);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_queries);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.edge_interval_records_visited);
        out << ',';
        detail::write_optional_int(
            out,
            record.refinement_statistics.full_active_scans);
        out
            << ',' << (record.local_time_slab_closure_applied ? 1 : 0)
            << ',' << record.local_time_slab_closure_marked_split_cells
            << ',' << record.local_time_slab_closure_temporal_waves
            << ',' << record.local_time_slab_closure_temporally_refined_cells
            << ',' << (main_solve.available
                ? solver_type_name(main_solve.selected_solver)
                : "")
            << ',' << (main_solve.available
                ? solver_type_name(main_solve.effective_solver)
                : "")
            << ',' << detail::escape_summary_field(main_solve.solver_status)
            << ',' << main_solve.matrix_rows
            << ',' << main_solve.matrix_cols
            << ',' << main_solve.matrix_nnz
            << ',' << main_solve.n
            << ',' << main_solve.nnz_matrix
            << ',' << main_solve.nnz_factors
            << ',';
        detail::write_optional_value(out, main_solve.factor_nnz);
        out
            << ',' << main_solve.fill_ratio
            << ',' << main_solve.symbolic_analysis_seconds
            << ',' << (main_solve.symbolic_analysis_reused ? 1 : 0)
            << ',' << main_solve.symbolic_pattern_cache_hits
            << ',' << main_solve.symbolic_pattern_cache_misses
            << ',' << main_solve.numeric_factorization_seconds
            << ',' << main_solve.backsolve_seconds
            << ',';
        detail::write_optional_value(
            out,
            main_solve.estimated_factor_memory_bytes);
        out
            << ',' << main_solve.preconditioner_setup_seconds
            << ',' << main_solve.setup_seconds
            << ',' << main_solve.solve_seconds
            << '\n';
        out.flush();
    }

    inline void append_partial_timing_history_csv(
        const AdaptiveOutputSettings& settings,
        int outer_iteration,
        int inner_iteration,
        std::string_view history_status,
        double elapsed_seconds,
        int x_active_cells,
        int y_active_cells,
        int x_true_dofs,
        int y_true_dofs)
    {
        if (!settings.export_history || settings.output_directory.empty())
            return;

        detail::ensure_output_directory(settings.output_directory);
        const auto path =
            settings.output_directory / settings.partial_timing_history_filename;
        const bool needs_header = detail::csv_needs_header(path);

        std::ofstream out(path, std::ios::app);
        if (!out)
        {
            throw std::runtime_error(
                "append_partial_timing_history_csv: failed to open partial timing-history output file.");
        }

        if (needs_header)
        {
            out
                << "history_status"
                << ",outer_iteration"
                << ",inner_iteration"
                << ",elapsed_seconds"
                << ",x_active_cells"
                << ",y_active_cells"
                << ",x_true_dofs"
                << ",y_true_dofs\n";
        }

        out
            << detail::escape_summary_field(history_status)
            << ',' << outer_iteration
            << ',' << inner_iteration
            << ',' << elapsed_seconds
            << ',' << x_active_cells
            << ',' << y_active_cells
            << ',' << x_true_dofs
            << ',' << y_true_dofs
            << '\n';
        out.flush();
    }

    inline void append_partial_timing_breakdown_csv(
        const AdaptiveOutputSettings& settings,
        int outer_iteration,
        int inner_iteration,
        std::string_view history_status,
        double elapsed_seconds,
        int x_active_cells,
        int y_active_cells,
        int x_true_dofs,
        int y_true_dofs,
        const std::vector<TimingRecord>& records)
    {
        if (!settings.export_history ||
            settings.output_directory.empty() ||
            settings.partial_timing_breakdown_filename.empty() ||
            records.empty())
        {
            return;
        }

        detail::ensure_output_directory(settings.output_directory);
        const auto path =
            settings.output_directory / settings.partial_timing_breakdown_filename;
        const bool needs_header = detail::csv_needs_header(path);

        std::ofstream out(path, std::ios::app);
        if (!out)
        {
            throw std::runtime_error(
                "append_partial_timing_breakdown_csv: failed to open partial timing-breakdown output file.");
        }

        if (needs_header)
        {
            out
                << "history_status"
                << ",outer_iteration"
                << ",inner_iteration"
                << ",elapsed_seconds"
                << ",phase"
                << ",metric_kind"
                << ",total_seconds"
                << ",last_seconds"
                << ",call_count"
                << ",x_active_cells"
                << ",y_active_cells"
                << ",x_true_dofs"
                << ",y_true_dofs\n";
        }

        for (const auto& record : records)
        {
            out
                << detail::escape_summary_field(history_status)
                << ',' << outer_iteration
                << ',' << inner_iteration
                << ',' << elapsed_seconds
                << ',' << detail::escape_summary_field(record.phase)
                << ',' << metric_kind_name(record.metric_kind)
                << ',' << record.total_seconds
                << ',' << record.last_seconds
                << ',' << record.call_count
                << ',' << x_active_cells
                << ',' << y_active_cells
                << ',' << x_true_dofs
                << ',' << y_true_dofs
                << '\n';
        }
        out.flush();
    }

    template<class Backend, class XSpaceType, class YSpaceType>
    void write_outer_history_csv(
        const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result,
        const AdaptiveOutputSettings& settings)
    {
        if (!settings.export_history || settings.output_directory.empty())
            return;

        detail::ensure_output_directory(settings.output_directory);

        std::ofstream out(settings.output_directory / settings.outer_history_filename);
        if (!out)
        {
            throw std::runtime_error(
                "write_outer_history_csv: failed to open outer-history output file.");
        }

        out
            << "outer_iteration"
            << ",x_active_cells_before"
            << ",x_active_cells_after"
            << ",x_true_dofs_before"
            << ",x_true_dofs_after"
            << ",x_generation_min_before"
            << ",x_generation_max_before"
            << ",x_generation_distinct_before"
            << ",x_generation_min_after"
            << ",x_generation_max_after"
            << ",x_generation_distinct_after"
            << ",y_active_cells_before"
            << ",y_active_cells_after"
            << ",y_true_dofs_before"
            << ",y_true_dofs_after"
            << ",y_generation_min_before"
            << ",y_generation_max_before"
            << ",y_generation_distinct_before"
            << ",y_generation_min_after"
            << ",y_generation_max_after"
            << ",y_generation_distinct_after"
            << ",iteration_seconds"
            << ",elapsed_seconds"
            << ",inner_iterations"
            << ",y_converged"
            << ",stopped_on_empty_y_marking"
            << ",refined_x"
            << ",uniform_x_refinement"
            << ",x_refinement_mode"
            << ",uniform_y_refinement"
            << ",x_marking_empty"
            << ",lambda_y_squared"
            << ",lambda_y"
            << ",initial_trace_squared"
            << ",initial_trace"
            << ",eta_squared"
            << ",eta"
            << ",y_estimator_squared"
            << ",y_estimator"
            << ",y_estimator_threshold_squared"
            << ",y_estimator_threshold"
            << ",configured_rho"
            << ",effective_rho"
            << ",effective_rho_available"
            << ",effective_rho_reason"
            << ",y_estimator_threshold_configured_rho_squared"
            << ",y_estimator_threshold_effective_rho_squared"
            << ",posteriori_factor_configured_rho"
            << ",posteriori_factor_effective_rho"
            << ",posteriori_estimator_configured_rho_squared"
            << ",posteriori_estimator_effective_rho_squared"
            << ",posteriori_improvement_factor"
            << ",force_accept_inner_with_effective_rho"
            << ",configured_rho_ignored_for_inner_acceptance"
            << ",effective_rho_acceptance_used"
            << ",effective_rho_acceptance_reason"
            << ",g_estimator_enabled"
            << ",g_estimator_computed"
            << ",g_estimator_skipped_reason"
            << ",g_space_constructed"
            << ",g_true_dofs"
            << ",g_solve_count"
            << ",g_solver_status"
            << ",g_assembly_seconds"
            << ",g_solve_seconds"
            << ",g_solver_residual"
            << ",g_solver_relative_residual"
            << ",g_rhs_inf_norm"
            << ",g_lambda_inf_norm"
            << ",g_lambda_difference_available"
            << ",g_lambda_difference_squared"
            << ",g_lambda_difference"
            << ",y_flux_squared"
            << ",y_flux"
            << ",y_reconstruction_squared"
            << ",y_reconstruction"
            << ",divergence_residual_squared"
            << ",divergence_residual"
            << ",divergence_residual_l2"
            << ",marked_x_cells"
            << ",x_marked_cells"
            << ",x_refined_cells"
            << ",x_closure_cells"
            << ",x_min_generation"
            << ",x_max_generation"
            << ",x_distinct_generations"
            << ",y_min_generation"
            << ",y_max_generation"
            << ",y_distinct_generations"
            << ",edge_interval_queries"
            << ",edge_interval_records_visited"
            << ",2d_edge_interval_queries"
            << ",2d_edge_interval_records_visited"
            << ",full_active_scans"
            << ",x_2d_edge_interval_queries"
            << ",x_2d_edge_interval_records_visited"
            << ",x_full_active_scans"
            << ",history_status\n";

        for (const auto& record : result.outer_iterations)
        {
            const double divergence_residual_l2 =
                record.inner_iterations.empty()
                    ? 0.0
                    : record.last_inner_iteration().divergence_residual_l2;

            out
                << record.outer_iteration
                << ',' << record.n_x_active_cells_before
                << ',' << record.n_x_active_cells_after
                << ',' << record.n_x_true_dofs_before
                << ',' << record.n_x_true_dofs_after
                << ',' << record.x_generation_before.min_generation
                << ',' << record.x_generation_before.max_generation
                << ',' << record.x_generation_before.distinct_generations
                << ',' << record.x_generation_after.min_generation
                << ',' << record.x_generation_after.max_generation
                << ',' << record.x_generation_after.distinct_generations
                << ',' << record.n_y_active_cells_before
                << ',' << record.n_y_active_cells_after
                << ',' << record.n_y_true_dofs_before
                << ',' << record.n_y_true_dofs_after
                << ',' << record.y_generation_before.min_generation
                << ',' << record.y_generation_before.max_generation
                << ',' << record.y_generation_before.distinct_generations
                << ',' << record.y_generation_after.min_generation
                << ',' << record.y_generation_after.max_generation
                << ',' << record.y_generation_after.distinct_generations
                << ',' << record.iteration_seconds
                << ',' << record.elapsed_seconds
                << ',' << record.n_inner_iterations()
                << ',' << (record.y_converged ? 1 : 0)
                << ',' << (record.stopped_on_empty_y_marking ? 1 : 0)
                << ',' << (record.refined_x ? 1 : 0)
                << ',' << (record.uniform_x_refinement ? 1 : 0)
                << ',' << detail::escape_summary_field(record.x_refinement_mode)
                << ',' << (record.uniform_y_refinement ? 1 : 0)
                << ',' << (record.x_marking_empty ? 1 : 0)
                << ',' << record.lambda_y_squared
                << ',' << std::sqrt(record.lambda_y_squared)
                << ',' << record.initial_trace_squared
                << ',' << std::sqrt(record.initial_trace_squared)
                << ',' << record.eta_squared
                << ',' << std::sqrt(record.eta_squared)
                << ',' << record.y_estimator_squared
                << ',' << std::sqrt(record.y_estimator_squared)
                << ',' << record.y_estimator_threshold_squared
                << ',' << std::sqrt(record.y_estimator_threshold_squared)
                << ',' << record.configured_rho
                << ',';
            detail::write_optional_value(out, record.effective_rho);
            out
                << ',' << (record.effective_rho_available ? 1 : 0)
                << ',' << detail::escape_summary_field(record.effective_rho_reason)
                << ',' << record.y_estimator_threshold_configured_rho_squared
                << ',';
            detail::write_optional_value(
                out,
                record.y_estimator_threshold_effective_rho_squared);
            out
                << ',' << record.posteriori_factor_configured_rho
                << ',';
            detail::write_optional_value(
                out,
                record.posteriori_factor_effective_rho);
            out
                << ','
                << record.posteriori_estimator_configured_rho_squared
                << ',';
            detail::write_optional_value(
                out,
                record.posteriori_estimator_effective_rho_squared);
            out << ',';
            detail::write_optional_value(
                out,
                record.posteriori_improvement_factor);
            out
                << ',' << (record.force_accept_inner_with_effective_rho ? 1 : 0)
                << ',' << (record.configured_rho_ignored_for_inner_acceptance ? 1 : 0)
                << ',' << (record.effective_rho_acceptance_used ? 1 : 0)
                << ',' << detail::escape_summary_field(
                    record.effective_rho_acceptance_reason)
                << ',' << (record.g_estimator_enabled ? 1 : 0)
                << ',' << (record.g_estimator_computed ? 1 : 0)
                << ',' << detail::escape_summary_field(
                    record.g_estimator_skipped_reason)
                << ',' << (record.g_space_constructed ? 1 : 0)
                << ',';
            detail::write_optional_int(out, record.g_true_dofs);
            out
                << ',' << record.g_solve_count
                << ',' << detail::escape_summary_field(record.g_solver_status)
                << ',';
            detail::write_optional_value(out, record.g_assembly_seconds);
            out << ',';
            detail::write_optional_value(out, record.g_solve_seconds);
            out << ',';
            detail::write_optional_value(out, record.g_solver_residual);
            out << ',';
            detail::write_optional_value(out, record.g_solver_relative_residual);
            out << ',';
            detail::write_optional_value(out, record.g_rhs_inf_norm);
            out << ',';
            detail::write_optional_value(out, record.g_lambda_inf_norm);
            out
                << ',' << (record.g_lambda_difference_available ? 1 : 0)
                << ',';
            detail::write_optional_value(
                out,
                record.g_lambda_difference_squared);
            out << ',';
            detail::write_optional_value(out, record.g_lambda_difference);
            out
                << ',' << record.final_y_estimator.equilibrated_flux_y_squared.total()
                << ',' << std::sqrt(record.final_y_estimator.equilibrated_flux_y_squared.total())
                << ',' << record.final_y_estimator.reconstruction_y_squared.total()
                << ',' << std::sqrt(record.final_y_estimator.reconstruction_y_squared.total())
                << ',' << record.final_y_estimator.divergence_residual_total()
                << ',' << std::sqrt(record.final_y_estimator.divergence_residual_total())
                << ',' << divergence_residual_l2
                << ',' << record.marked_x_cells.size()
                << ',' << record.marked_x_cells.size();
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.refined_cells);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.closure_cells);
            out
                << ',' << record.x_generation_after.min_generation
                << ',' << record.x_generation_after.max_generation
                << ',' << record.x_generation_after.distinct_generations
                << ',' << record.y_generation_after.min_generation
                << ',' << record.y_generation_after.max_generation
                << ',' << record.y_generation_after.distinct_generations
                << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.edge_interval_queries);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.edge_interval_records_visited);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.edge_interval_queries);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.edge_interval_records_visited);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.full_active_scans);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.edge_interval_queries);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.edge_interval_records_visited);
            out << ',';
            detail::write_optional_int(
                out,
                record.refinement_statistics.full_active_scans);
            out << ",completed\n";
        }
    }

    template<class Backend, class XSpaceType, class YSpaceType>
    void write_inner_history_csv(
        const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result,
        const AdaptiveOutputSettings& settings)
    {
        if (!settings.export_history ||
            settings.output_directory.empty())
            return;

        detail::ensure_output_directory(settings.output_directory);

        std::ofstream out(settings.output_directory / settings.inner_history_filename);
        if (!out)
        {
            throw std::runtime_error(
                "write_inner_history_csv: failed to open inner-history output file.");
        }

        out
            << "outer_iteration"
            << ",inner_iteration"
            << ",y_active_cells_before"
            << ",y_active_cells_after"
            << ",y_true_dofs_before"
            << ",y_true_dofs_after"
            << ",y_generation_min_before"
            << ",y_generation_max_before"
            << ",y_generation_distinct_before"
            << ",y_generation_min_after"
            << ",y_generation_max_after"
            << ",y_generation_distinct_after"
            << ",iteration_seconds"
            << ",elapsed_seconds"
            << ",n_slabs"
            << ",n_patches"
            << ",copied_slab_cells"
            << ",virtual_slab_cells"
            << ",time_slab_backend"
            << ",estimator_backend"
            << ",estimator_uses_copied_fallback"
            << ",estimator_fallback_copied_slab_cells"
            << ",time_slab_backend_requested"
            << ",time_slab_backend_effective"
            << ",virtual_overlay_constructed"
            << ",tilde_y_space_constructed"
            << ",copied_estimator_fallback_enabled"
            << ",copied_estimator_fallback_used"
            << ",copied_fallback_component_count"
            << ",copied_fallback_components"
            << ",strict_virtual_estimator"
            << ",strict_virtual_estimator_status"
            << ",copied_slab_cells_constructed_total"
            << ",copied_slab_cells_constructed_for_fallback"
            << ",source_mesh_mutation_count"
            << ",lambda_y_squared"
            << ",lambda_y"
            << ",initial_trace_squared"
            << ",initial_trace"
            << ",eta_squared"
            << ",eta"
            << ",y_estimator_squared"
            << ",y_estimator"
            << ",y_flux_squared"
            << ",y_flux"
            << ",y_reconstruction_squared"
            << ",y_reconstruction"
            << ",divergence_residual_squared"
            << ",divergence_residual"
            << ",divergence_residual_l2"
            << ",y_estimator_threshold_squared"
            << ",y_estimator_threshold"
            << ",configured_rho"
            << ",effective_rho"
            << ",effective_rho_available"
            << ",effective_rho_reason"
            << ",y_estimator_threshold_configured_rho_squared"
            << ",y_estimator_threshold_effective_rho_squared"
            << ",posteriori_factor_configured_rho"
            << ",posteriori_factor_effective_rho"
            << ",posteriori_estimator_configured_rho_squared"
            << ",posteriori_estimator_effective_rho_squared"
            << ",posteriori_improvement_factor"
            << ",force_accept_inner_with_effective_rho"
            << ",configured_rho_ignored_for_inner_acceptance"
            << ",effective_rho_acceptance_used"
            << ",effective_rho_acceptance_reason"
            << ",g_estimator_enabled"
            << ",g_estimator_computed"
            << ",g_estimator_skipped_reason"
            << ",g_space_constructed"
            << ",g_true_dofs"
            << ",g_solve_count"
            << ",g_solver_status"
            << ",g_assembly_seconds"
            << ",g_solve_seconds"
            << ",g_solver_residual"
            << ",g_solver_relative_residual"
            << ",g_rhs_inf_norm"
            << ",g_lambda_inf_norm"
            << ",g_lambda_difference_available"
            << ",g_lambda_difference_squared"
            << ",g_lambda_difference"
            << ",stopping_criterion_satisfied"
            << ",refined_y"
            << ",uniform_y_refinement"
            << ",y_refinement_mode"
            << ",y_refinement_target_cells"
            << ",marked_y_cells"
            << ",y_marked_cells"
            << ",y_refined_cells"
            << ",y_closure_cells"
            << ",y_min_generation"
            << ",y_max_generation"
            << ",y_distinct_generations"
            << ",edge_interval_queries"
            << ",edge_interval_records_visited"
            << ",2d_edge_interval_queries"
            << ",2d_edge_interval_records_visited"
            << ",full_active_scans"
            << ",y_2d_edge_interval_queries"
            << ",y_2d_edge_interval_records_visited"
            << ",y_full_active_scans"
            << ",local_time_slab_closure_applied"
            << ",local_time_slab_closure_marked_split_cells"
            << ",local_time_slab_closure_temporal_waves"
            << ",local_time_slab_closure_temporally_refined_cells"
            << ",main_solve_selected_solver"
            << ",main_solve_effective_solver"
            << ",main_solve_solver_status"
            << ",main_solve_matrix_rows"
            << ",main_solve_matrix_cols"
            << ",main_solve_matrix_nnz"
            << ",main_solve_n"
            << ",main_solve_nnz_matrix"
            << ",main_solve_nnz_factors"
            << ",main_solve_factor_nnz"
                << ",main_solve_fill_ratio"
                << ",main_solve_symbolic_analysis_seconds"
                << ",main_solve_symbolic_analysis_reused"
                << ",main_solve_symbolic_pattern_cache_hits"
                << ",main_solve_symbolic_pattern_cache_misses"
                << ",main_solve_numeric_factorization_seconds"
            << ",main_solve_backsolve_seconds"
            << ",main_solve_estimated_factor_memory_bytes"
            << ",main_solve_symbolic_memory"
            << ",main_solve_numerical_factor_memory"
            << ",main_solve_estimated_in_core_peak_memory"
            << ",main_solve_out_of_core_minimum_memory"
            << ",main_solve_process_rss_before_factorization"
            << ",main_solve_process_rss_after_factorization"
            << ",main_solve_process_rss_after_solve"
            << ",main_solve_memory_guard_estimated_extra_memory"
            << ",main_solve_direct_memory_limit"
            << ",main_solve_memory_guard_estimated_peak_memory"
            << ",main_solve_memory_guard_triggered"
            << ",main_solve_ooc_auto_switch_attempted"
            << ",main_solve_ooc_auto_switch_solver"
            << ",main_solve_effective_pardiso_memory_mode"
            << ",main_solve_iteration_count"
            << ",main_solve_final_residual"
            << ",linear_residual_absolute"
            << ",linear_residual_relative"
            << ",main_solve_initial_guess_norm"
            << ",main_solve_initial_residual_absolute"
            << ",main_solve_initial_residual_relative"
            << ",main_solve_backend_converged"
            << ",main_solve_backend_reported_error"
            << ",main_solve_accepted_by_true_residual"
            << ",main_solve_residual_check_batches"
            << ",main_solve_final_true_residual"
            << ",main_solve_true_residual_stopping_used"
            << ",main_solve_matrix_norm"
            << ",main_solve_symmetry_difference_norm"
            << ",main_solve_relative_asymmetry"
            << ",main_solve_residual_retry_attempted"
            << ",main_solve_residual_retry_solver"
            << ",main_solve_residual_before_retry"
            << ",main_solve_residual_after_retry"
            << ",main_solve_residual_correction_steps"
            << ",main_solve_residual_before_correction"
            << ",main_solve_residual_after_correction"
            << ",main_solve_preconditioner_setup_seconds"
            << ",main_solve_setup_seconds"
            << ",main_solve_solve_seconds"
            << ",history_status\n";

        for (const auto& outer_record : result.outer_iterations)
        {
            for (const auto& inner_record : outer_record.inner_iterations)
            {
                const auto& main_solve = inner_record.main_solve;
                out
                    << outer_record.outer_iteration
                    << ',' << inner_record.inner_iteration
                    << ',' << inner_record.n_y_active_cells_before
                    << ',' << inner_record.n_y_active_cells_after
                    << ',' << inner_record.n_y_true_dofs_before
                    << ',' << inner_record.n_y_true_dofs_after
                    << ',' << inner_record.y_generation_before.min_generation
                    << ',' << inner_record.y_generation_before.max_generation
                    << ',' << inner_record.y_generation_before.distinct_generations
                    << ',' << inner_record.y_generation_after.min_generation
                    << ',' << inner_record.y_generation_after.max_generation
                    << ',' << inner_record.y_generation_after.distinct_generations
                    << ',' << inner_record.iteration_seconds
                    << ',' << inner_record.elapsed_seconds
                    << ',' << inner_record.n_slabs
                    << ',' << inner_record.n_patches
                    << ',';
                detail::write_optional_int(
                    out,
                    inner_record.copied_slab_cells);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.virtual_slab_cells);
                out
                    << ',' << detail::escape_summary_field(
                        inner_record.time_slab_backend)
                    << ',' << detail::escape_summary_field(
                        inner_record.estimator_backend)
                    << ','
                    << (inner_record.estimator_uses_copied_fallback ? 1 : 0)
                    << ',';
                detail::write_optional_int(
                    out,
                    inner_record.estimator_fallback_copied_slab_cells);
                out
                    << ',' << detail::escape_summary_field(
                        inner_record.time_slab_backend_requested)
                    << ',' << detail::escape_summary_field(
                        inner_record.time_slab_backend_effective)
                    << ',' << (inner_record.virtual_overlay_constructed ? 1 : 0)
                    << ',' << (inner_record.tilde_y_space_constructed ? 1 : 0)
                    << ',' << (inner_record.copied_estimator_fallback_enabled ? 1 : 0)
                    << ',' << (inner_record.copied_estimator_fallback_used ? 1 : 0)
                    << ',' << inner_record.copied_fallback_component_count
                    << ',' << detail::escape_summary_field(
                        inner_record.copied_fallback_components)
                    << ',' << (inner_record.strict_virtual_estimator ? 1 : 0)
                    << ',' << detail::escape_summary_field(
                        inner_record.strict_virtual_estimator_status)
                    << ',';
                detail::write_optional_int(
                    out,
                    inner_record.copied_slab_cells_constructed_total);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.copied_slab_cells_constructed_for_fallback);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.source_mesh_mutation_count);
                out
                    << ',' << inner_record.lambda_y_squared
                    << ',' << std::sqrt(inner_record.lambda_y_squared)
                    << ',' << inner_record.initial_trace_squared
                    << ',' << std::sqrt(inner_record.initial_trace_squared)
                    << ',' << inner_record.eta_squared
                    << ',' << std::sqrt(inner_record.eta_squared)
                    << ',' << inner_record.y_estimator_squared
                    << ',' << std::sqrt(inner_record.y_estimator_squared)
                    << ',' << inner_record.y_flux_squared
                    << ',' << std::sqrt(inner_record.y_flux_squared)
                    << ',' << inner_record.y_reconstruction_squared
                    << ',' << std::sqrt(inner_record.y_reconstruction_squared)
                    << ',' << inner_record.divergence_residual_squared
                    << ',' << std::sqrt(inner_record.divergence_residual_squared)
                    << ',' << inner_record.divergence_residual_l2
                    << ',' << inner_record.y_estimator_threshold_squared
                    << ',' << std::sqrt(inner_record.y_estimator_threshold_squared)
                    << ',' << inner_record.configured_rho
                    << ',';
                detail::write_optional_value(out, inner_record.effective_rho);
                out
                    << ',' << (inner_record.effective_rho_available ? 1 : 0)
                    << ',' << detail::escape_summary_field(
                        inner_record.effective_rho_reason)
                    << ','
                    << inner_record.y_estimator_threshold_configured_rho_squared
                    << ',';
                detail::write_optional_value(
                    out,
                    inner_record.y_estimator_threshold_effective_rho_squared);
                out
                    << ',' << inner_record.posteriori_factor_configured_rho
                    << ',';
                detail::write_optional_value(
                    out,
                    inner_record.posteriori_factor_effective_rho);
                out
                    << ','
                    << inner_record.posteriori_estimator_configured_rho_squared
                    << ',';
                detail::write_optional_value(
                    out,
                    inner_record.posteriori_estimator_effective_rho_squared);
                out << ',';
                detail::write_optional_value(
                    out,
                    inner_record.posteriori_improvement_factor);
                out
                    << ',' << (inner_record.force_accept_inner_with_effective_rho ? 1 : 0)
                    << ',' << (inner_record.configured_rho_ignored_for_inner_acceptance ? 1 : 0)
                    << ',' << (inner_record.effective_rho_acceptance_used ? 1 : 0)
                    << ',' << detail::escape_summary_field(
                        inner_record.effective_rho_acceptance_reason)
                    << ',' << (inner_record.g_estimator_enabled ? 1 : 0)
                    << ',' << (inner_record.g_estimator_computed ? 1 : 0)
                    << ',' << detail::escape_summary_field(
                        inner_record.g_estimator_skipped_reason)
                    << ',' << (inner_record.g_space_constructed ? 1 : 0)
                    << ',';
                detail::write_optional_int(out, inner_record.g_true_dofs);
                out
                    << ',' << inner_record.g_solve_count
                    << ',' << detail::escape_summary_field(
                        inner_record.g_solver_status)
                    << ',';
                detail::write_optional_value(
                    out,
                    inner_record.g_assembly_seconds);
                out << ',';
                detail::write_optional_value(out, inner_record.g_solve_seconds);
                out << ',';
                detail::write_optional_value(
                    out,
                    inner_record.g_solver_residual);
                out << ',';
                detail::write_optional_value(
                    out,
                    inner_record.g_solver_relative_residual);
                out << ',';
                detail::write_optional_value(out, inner_record.g_rhs_inf_norm);
                out << ',';
                detail::write_optional_value(
                    out,
                    inner_record.g_lambda_inf_norm);
                out
                    << ',' << (inner_record.g_lambda_difference_available ? 1 : 0)
                    << ',';
                detail::write_optional_value(
                    out,
                    inner_record.g_lambda_difference_squared);
                out << ',';
                detail::write_optional_value(
                    out,
                    inner_record.g_lambda_difference);
                out
                    << ',' << (inner_record.stopping_criterion_satisfied ? 1 : 0)
                    << ',' << (inner_record.refined_y ? 1 : 0)
                    << ',' << (inner_record.uniform_y_refinement ? 1 : 0)
                    << ',' << detail::escape_summary_field(
                        inner_record.y_refinement_mode)
                    << ',' << inner_record.y_refinement_target_cells
                    << ',' << inner_record.marked_y_cells.size()
                    << ',' << inner_record.marked_y_cells.size();
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics.refined_cells);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics.closure_cells);
                out
                    << ',' << inner_record.y_generation_after.min_generation
                    << ',' << inner_record.y_generation_after.max_generation
                    << ',' << inner_record.y_generation_after.distinct_generations
                    << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics.edge_interval_queries);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics
                        .edge_interval_records_visited);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics.edge_interval_queries);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics
                        .edge_interval_records_visited);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics.full_active_scans);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics.edge_interval_queries);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics
                        .edge_interval_records_visited);
                out << ',';
                detail::write_optional_int(
                    out,
                    inner_record.refinement_statistics.full_active_scans);
                out
                    << ',' << (inner_record.local_time_slab_closure_applied ? 1 : 0)
                    << ',' << inner_record.local_time_slab_closure_marked_split_cells
                    << ',' << inner_record.local_time_slab_closure_temporal_waves
                    << ',' << inner_record.local_time_slab_closure_temporally_refined_cells
                    << ',' << (main_solve.available
                        ? solver_type_name(main_solve.selected_solver)
                        : "")
                    << ',' << (main_solve.available
                        ? solver_type_name(main_solve.effective_solver)
                        : "")
                    << ',' << detail::escape_summary_field(
                        main_solve.solver_status)
                    << ',' << main_solve.matrix_rows
                    << ',' << main_solve.matrix_cols
                    << ',' << main_solve.matrix_nnz
                    << ',' << main_solve.n
                    << ',' << main_solve.nnz_matrix
                    << ',' << main_solve.nnz_factors
                    << ',';
                detail::write_optional_value(out, main_solve.factor_nnz);
                out
                    << ',' << main_solve.fill_ratio
                    << ',' << main_solve.symbolic_analysis_seconds
                    << ',' << (main_solve.symbolic_analysis_reused ? 1 : 0)
                    << ',' << main_solve.symbolic_pattern_cache_hits
                    << ',' << main_solve.symbolic_pattern_cache_misses
                    << ',' << main_solve.numeric_factorization_seconds
                    << ',' << main_solve.backsolve_seconds
                    << ',';
                detail::write_optional_value(
                    out,
                    main_solve.estimated_factor_memory_bytes);
                out
                    << ',' << main_solve.symbolic_memory
                    << ',' << main_solve.numerical_factor_memory
                    << ',' << main_solve.estimated_in_core_peak_memory
                    << ',' << main_solve.out_of_core_minimum_memory
                    << ',' << main_solve.process_rss_before_factorization
                    << ',' << main_solve.process_rss_after_factorization
                    << ',' << main_solve.process_rss_after_solve
                    << ',' << main_solve.memory_guard_estimated_extra_memory
                    << ',' << main_solve.direct_memory_limit
                    << ',' << main_solve.memory_guard_estimated_peak_memory
                    << ',' << (main_solve.memory_guard_triggered ? 1 : 0)
                    << ',' << (main_solve.out_of_core_auto_switch_attempted ? 1 : 0)
                    << ',' << main_solve.out_of_core_auto_switch_solver
                    << ',' << main_solve.effective_pardiso_memory_mode
                    << ',' << main_solve.iteration_count
                    << ',' << main_solve.final_residual
                    << ',' << main_solve.linear_residual_absolute
                    << ',' << main_solve.linear_residual_relative
                    << ',' << main_solve.initial_guess_norm
                    << ',' << main_solve.initial_residual_absolute
                    << ',' << main_solve.initial_residual_relative
                    << ',' << (main_solve.backend_converged ? 1 : 0)
                    << ',' << main_solve.backend_reported_error
                    << ',' << (main_solve.convergence_accepted_by_true_residual
                        ? 1
                        : 0)
                    << ',' << main_solve.residual_check_batches
                    << ',' << main_solve.final_true_residual
                    << ',' << (main_solve.true_residual_stopping_used ? 1 : 0)
                    << ',' << main_solve.matrix_norm
                    << ',' << main_solve.matrix_symmetry_difference_norm
                    << ',' << main_solve.matrix_relative_asymmetry
                    << ',' << (main_solve.residual_retry_attempted ? 1 : 0)
                    << ',' << main_solve.residual_retry_solver
                    << ',' << main_solve.residual_before_retry
                    << ',' << main_solve.residual_after_retry
                    << ',' << main_solve.residual_correction_steps
                    << ',' << main_solve.residual_before_correction
                    << ',' << main_solve.residual_after_correction
                    << ',' << main_solve.preconditioner_setup_seconds
                    << ',' << main_solve.setup_seconds
                    << ',' << main_solve.solve_seconds
                    << ",completed\n";
            }
        }
    }

    template<class Backend, class XSpaceType, class YSpaceType>
    void write_timing_history_csv(
        const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result,
        const AdaptiveOutputSettings& settings)
    {
        if (!settings.export_history ||
            !result.timing_enabled ||
            settings.output_directory.empty())
        {
            return;
        }

        detail::ensure_output_directory(settings.output_directory);

        std::ofstream out(settings.output_directory / settings.timing_history_filename);
        if (!out)
        {
            throw std::runtime_error(
                "write_timing_history_csv: failed to open timing-history output file.");
        }

        int final_x_active_cells = 0;
        int final_y_active_cells = 0;
        int final_x_true_dofs = 0;
        int final_y_true_dofs = 0;
        int final_n_slabs = 0;
        int final_n_patches = 0;

        if (!result.outer_iterations.empty())
        {
            const auto& outer = result.outer_iterations.back();
            final_x_active_cells = outer.n_x_active_cells_before;
            final_y_active_cells = outer.n_y_active_cells_after;
            final_x_true_dofs = outer.n_x_true_dofs_before;
            final_y_true_dofs = outer.n_y_true_dofs_after;

            if (!outer.inner_iterations.empty())
            {
                const auto& inner = outer.inner_iterations.back();
                final_n_slabs = inner.n_slabs;
                final_n_patches = inner.n_patches;
            }
        }

        out
            << "outer_iteration"
            << ",inner_iteration"
            << ",phase"
            << ",metric_kind"
            << ",total_seconds"
            << ",last_seconds"
            << ",call_count"
            << ",x_active_cells"
            << ",y_active_cells"
            << ",x_true_dofs"
            << ",y_true_dofs"
            << ",n_slabs"
            << ",n_patches\n";

        for (const auto& record : result.timing_records)
        {
            out
                << -1
                << ',' << -1
                << ',' << detail::escape_summary_field(record.phase)
                << ',' << metric_kind_name(record.metric_kind)
                << ',' << record.total_seconds
                << ',' << record.last_seconds
                << ',' << record.call_count
                << ',' << final_x_active_cells
                << ',' << final_y_active_cells
                << ',' << final_x_true_dofs
                << ',' << final_y_true_dofs
                << ',' << final_n_slabs
                << ',' << final_n_patches
                << '\n';
        }
    }

    template<class Backend, class XSpaceType, class YSpaceType>
    void write_refinement_history_text(
        const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result,
        const AdaptiveOutputSettings& settings)
    {
        if (!settings.export_history ||
            !settings.save_refinement_history ||
            settings.output_directory.empty())
            return;

        detail::ensure_output_directory(settings.output_directory);

        std::ofstream out(settings.output_directory / settings.refinement_history_filename);
        if (!out)
        {
            throw std::runtime_error(
                "write_refinement_history_text: failed to open refinement-history output file.");
        }

        for (const auto& outer_record : result.outer_iterations)
        {
            out << "outer " << outer_record.outer_iteration << '\n';

            for (const auto& inner_record : outer_record.inner_iterations)
            {
                out << "  inner " << inner_record.inner_iteration << " marked_y:";
                for (const int cell_id : inner_record.marked_y_cells)
                    out << ' ' << cell_id;
                out << '\n';
            }

            out << "  marked_x:";
            for (const int cell_id : outer_record.marked_x_cells)
                out << ' ' << cell_id;
            out << "\n\n";
        }
    }

    template<class Backend, class XSpaceType, class YSpaceType>
    void write_history_outputs(
        const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result,
        const AdaptiveOutputSettings& settings)
    {
        write_summary_text(result, settings);
        write_outer_history_csv(result, settings);
        write_inner_history_csv(result, settings);
        write_timing_history_csv(result, settings);
        write_refinement_history_text(result, settings);
    }
}

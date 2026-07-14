#pragma once

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "runner_option_specs.hpp"
#include "runner_options.hpp"
#include "runner_usage_text.hpp"

namespace adaptive_algorithm::runners::detail
{
    [[nodiscard]] inline std::string normalize_key(std::string key)
    {
        std::transform(
            key.begin(),
            key.end(),
            key.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return key;
    }

    [[nodiscard]] inline std::string trim_copy(std::string text)
    {
        const auto is_space =
            [](unsigned char c) noexcept
            {
                return std::isspace(c) != 0;
            };

        text.erase(
            text.begin(),
            std::find_if(
                text.begin(),
                text.end(),
                [is_space](unsigned char c) { return !is_space(c); }));
        text.erase(
            std::find_if(
                text.rbegin(),
                text.rend(),
                [is_space](unsigned char c) { return !is_space(c); })
                .base(),
            text.end());

        return text;
    }

    [[nodiscard]] inline std::string strip_inline_comment(std::string text)
    {
        bool in_single_quotes = false;
        bool in_double_quotes = false;

        for (std::size_t i = 0; i < text.size(); ++i)
        {
            const char c = text[i];
            if (c == '\'' && !in_double_quotes)
            {
                in_single_quotes = !in_single_quotes;
            }
            else if (c == '"' && !in_single_quotes)
            {
                in_double_quotes = !in_double_quotes;
            }
            else if (c == '#' && !in_single_quotes && !in_double_quotes)
            {
                return trim_copy(text.substr(0, i));
            }
        }

        return trim_copy(text);
    }

    [[nodiscard]] inline std::string strip_matching_quotes(std::string text)
    {
        if (text.size() >= 2)
        {
            const char first = text.front();
            const char last = text.back();
            if ((first == '"' && last == '"') ||
                (first == '\'' && last == '\''))
            {
                return text.substr(1, text.size() - 2);
            }
        }

        return text;
    }

    [[nodiscard]] inline std::string lowercase_copy(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline bool parse_bool_text(
        const std::string& text,
        const char* argv0,
        const std::string& option)
    {
        const std::string lowered = normalize_key(text);

        if (lowered == "1" ||
            lowered == "true" ||
            lowered == "yes" ||
            lowered == "on")
        {
            return true;
        }

        if (lowered == "0" ||
            lowered == "false" ||
            lowered == "no" ||
            lowered == "off")
        {
            return false;
        }

        throw std::runtime_error(
            usage_text(argv0) +
            "\nFailed to parse boolean value for option '" + option +
            "' from value '" + text +
            "'. Expected one of: true, false, yes, no, on, off, 1, 0.");
    }

    template<class T>
    [[nodiscard]] T parse_value(
        const char* argv0,
        const std::string& option,
        const char* text);

    template<>
    [[nodiscard]] inline int parse_value<int>(
        const char* argv0,
        const std::string& option,
        const char* text)
    {
        try
        {
            const std::string value(text);
            std::size_t parsed_length = 0;
            const int parsed_value = std::stoi(value, &parsed_length);
            if (parsed_length != value.size())
                throw std::invalid_argument("trailing characters");

            return parsed_value;
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                usage_text(argv0) +
                "\nFailed to parse integer for option '" + option +
                "' from value '" + std::string(text) +
                "'. Expected an integer.");
        }
    }

    template<>
    [[nodiscard]] inline double parse_value<double>(
        const char* argv0,
        const std::string& option,
        const char* text)
    {
        try
        {
            const std::string value(text);
            std::size_t parsed_length = 0;
            const double parsed_value = std::stod(value, &parsed_length);
            if (parsed_length != value.size())
                throw std::invalid_argument("trailing characters");

            return parsed_value;
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                usage_text(argv0) +
                "\nFailed to parse floating-point value for option '" + option +
                "' from value '" + std::string(text) +
                "'. Expected a number.");
        }
    }

    [[nodiscard]] inline const char* require_next_argument(
        int argc,
        char** argv,
        int& i)
    {
        if (i + 1 >= argc)
        {
            throw std::runtime_error(
                usage_text(argv[0]) +
                "\nMissing value after option '" + std::string(argv[i]) + "'.");
        }

        ++i;
        return argv[i];
    }

    inline void assign_string_override(
        RunnerOptionOverrides& options,
        RunnerOptionId id,
        std::string value)
    {
        switch (id)
        {
        case RunnerOptionId::config_file:
            options.config_file = std::filesystem::path(std::move(value));
            return;
        case RunnerOptionId::example_name:
            options.example_name = std::move(value);
            return;
        case RunnerOptionId::main_solver:
            options.main_solver = std::move(value);
            return;
        case RunnerOptionId::g_solver:
            options.g_solver = std::move(value);
            return;
        case RunnerOptionId::main_solver_pardiso_memory_mode:
            options.main_solver_pardiso_memory_mode = std::move(value);
            return;
        case RunnerOptionId::main_solver_diagnostics:
            options.main_solver_diagnostics =
                normalize_main_solver_diagnostics_choice(std::move(value));
            return;
        case RunnerOptionId::output_directory:
            options.output_directory = std::filesystem::path(std::move(value));
            return;
        case RunnerOptionId::output_profile:
            options.output_profile =
                normalize_output_profile_choice(std::move(value));
            return;
        case RunnerOptionId::uniform_refinement_mode:
            options.uniform_refinement_mode =
                normalize_uniform_refinement_mode_choice(std::move(value));
            return;
        case RunnerOptionId::max_dofs_target:
            options.max_dofs_target = normalize_key(std::move(value));
            return;
        case RunnerOptionId::timing_history_filename:
            options.timing_history_filename = std::move(value);
            return;
        case RunnerOptionId::timing_detail_level:
            options.timing_detail_level = std::move(value);
            return;
        case RunnerOptionId::time_slab_backend:
            options.time_slab_backend = std::move(value);
            return;
        case RunnerOptionId::post_flush_closure_mode:
            options.post_flush_closure_mode =
                normalize_post_flush_closure_mode_choice(std::move(value));
            return;
        case RunnerOptionId::refinement_main_closure_query_mode:
            options.refinement_main_closure_query_mode =
                normalize_refinement_main_closure_query_mode_choice(
                    std::move(value));
            return;
        case RunnerOptionId::local_error_worker_context_mode:
            options.local_error_worker_context_mode =
                normalize_local_error_worker_context_mode_choice(
                    std::move(value));
            return;
        case RunnerOptionId::local_error_context_storage:
            options.local_error_context_storage =
                normalize_local_error_context_storage_choice(
                    std::move(value));
            return;
        case RunnerOptionId::local_error_state_index_mode:
            options.local_error_state_index_mode =
                normalize_local_error_state_index_mode_choice(
                    std::move(value));
            return;
        case RunnerOptionId::local_error_cell_state_cache_mode:
            options.local_error_cell_state_cache_mode =
                normalize_local_error_cell_state_cache_mode_choice(
                    std::move(value));
            return;
        case RunnerOptionId::local_error_cell_state_representation:
            options.local_error_cell_state_representation =
                normalize_local_error_cell_state_representation_choice(
                    std::move(value));
            return;
        case RunnerOptionId::local_error_flux_diagnostics_mode:
            options.local_error_flux_diagnostics_mode =
                normalize_local_error_flux_diagnostics_mode_choice(
                    std::move(value));
            return;
        case RunnerOptionId::local_error_patch_solver:
            options.local_error_patch_solver =
                normalize_local_error_patch_solver_choice(std::move(value));
            return;
        case RunnerOptionId::shared_context_validation:
            options.shared_context_validation =
                normalize_shared_context_validation_choice(std::move(value));
            return;
        case RunnerOptionId::slab_reconstruction_operator_mode:
            options.slab_reconstruction_operator_mode =
                normalize_slab_reconstruction_operator_mode_choice(
                    std::move(value));
            return;
        default:
            throw std::logic_error("runner_parser: unexpected string option id.");
        }
    }

    inline void assign_int_override(
        RunnerOptionOverrides& options,
        RunnerOptionId id,
        int value)
    {
        switch (id)
        {
        case RunnerOptionId::max_outer_iterations:
            options.max_outer_iterations = value;
            return;
        case RunnerOptionId::max_inner_iterations:
            options.max_inner_iterations = value;
            return;
        case RunnerOptionId::dimension:
            options.dimension = value;
            return;
        case RunnerOptionId::polynomial_degree:
            options.polynomial_degree = value;
            return;
        case RunnerOptionId::max_y_true_dofs:
            options.max_y_true_dofs = value;
            return;
        case RunnerOptionId::max_x_true_dofs:
            options.max_x_true_dofs = value;
            return;
        case RunnerOptionId::main_solver_max_iterations:
            options.main_solver_max_iterations = value;
            return;
        case RunnerOptionId::refinement_batch_target_split_cells:
            options.refinement_batch_target_split_cells = value;
            return;
        case RunnerOptionId::local_error_patch_tile_size:
            options.local_error_patch_tile_size = value;
            return;
        case RunnerOptionId::local_error_cell_chunk_size:
            options.local_error_cell_chunk_size = value;
            return;
        case RunnerOptionId::local_error_max_threads:
            options.local_error_max_threads = value;
            return;
        case RunnerOptionId::main_assembly_max_threads:
            options.main_assembly_max_threads = value;
            return;
        case RunnerOptionId::slab_reconstruction_max_threads:
            options.slab_reconstruction_max_threads = value;
            return;
        case RunnerOptionId::main_two_pass_numeric_fill_max_threads:
            options.main_two_pass_numeric_fill_max_threads = value;
            return;
        case RunnerOptionId::solver_diagnostics_max_export_dofs:
            options.solver_diagnostics_max_export_dofs = value;
            return;
        default:
            throw std::logic_error("runner_parser: unexpected integer option id.");
        }
    }

    inline void assign_double_override(
        RunnerOptionOverrides& options,
        RunnerOptionId id,
        double value)
    {
        switch (id)
        {
        case RunnerOptionId::rho:
            options.rho = value;
            return;
        case RunnerOptionId::doerfler_theta_x:
            options.doerfler_theta_x = value;
            return;
        case RunnerOptionId::doerfler_theta_y:
            options.doerfler_theta_y = value;
            return;
        case RunnerOptionId::zero_tol:
            options.zero_tol = value;
            return;
        case RunnerOptionId::divergence_residual_l2_tolerance:
            options.divergence_residual_l2_tolerance = value;
            return;
        case RunnerOptionId::eta_squared_stop:
            options.eta_squared_stop = value;
            return;
        case RunnerOptionId::inner_estimator_squared_stop:
            options.inner_estimator_squared_stop = value;
            return;
        case RunnerOptionId::max_wall_time_seconds:
            options.max_wall_time_seconds = value;
            return;
        case RunnerOptionId::memory_limit_mb:
            options.memory_limit_mb = value;
            return;
        case RunnerOptionId::memory_reserve_mb:
            options.memory_reserve_mb = value;
            return;
        case RunnerOptionId::memory_guard_safety_factor:
            options.memory_guard_safety_factor = value;
            return;
        case RunnerOptionId::memory_guard_near_cap_fraction:
            options.memory_guard_near_cap_fraction = value;
            return;
        case RunnerOptionId::main_solver_tolerance:
            options.main_solver_tolerance = value;
            return;
        case RunnerOptionId::g_solver_tolerance:
            options.g_solver_tolerance = value;
            return;
        case RunnerOptionId::main_solver_symmetry_tolerance:
            options.main_solver_symmetry_tolerance = value;
            return;
        case RunnerOptionId::main_solver_direct_residual_retry_tolerance:
            options.main_solver_direct_residual_retry_tolerance = value;
            return;
        case RunnerOptionId::main_solver_memory_limit_mb:
            options.main_solver_memory_limit_mb = value;
            return;
        case RunnerOptionId::g_solver_memory_limit_mb:
            options.g_solver_memory_limit_mb = value;
            return;
        case RunnerOptionId::main_solver_ooc_switch_threshold:
            options.main_solver_ooc_switch_threshold = value;
            return;
        case RunnerOptionId::local_error_memory_budget_mb:
            options.local_error_memory_budget_mb = value;
            return;
        case RunnerOptionId::local_error_cell_state_cache_budget_mb:
            options.local_error_cell_state_cache_budget_mb = value;
            return;
        case RunnerOptionId::doerfler_near_tie_tolerance:
            options.doerfler_near_tie_tolerance = value;
            return;
        case RunnerOptionId::main_assembly_memory_budget_mb:
            options.main_assembly_memory_budget_mb = value;
            return;
        case RunnerOptionId::slab_reconstruction_memory_budget_mb:
            options.slab_reconstruction_memory_budget_mb = value;
            return;
        case RunnerOptionId::main_two_pass_numeric_fill_memory_budget_mb:
            options.main_two_pass_numeric_fill_memory_budget_mb = value;
            return;
        default:
            throw std::logic_error("runner_parser: unexpected floating-point option id.");
        }
    }

    inline void assign_bool_override(
        RunnerOptionOverrides& options,
        RunnerOptionId id,
        bool value)
    {
        switch (id)
        {
        case RunnerOptionId::quiet:
            options.quiet = value;
            return;
        case RunnerOptionId::list_examples:
            options.list_examples = value;
            return;
        case RunnerOptionId::show_help:
            options.show_help = value;
            return;
        case RunnerOptionId::check_divergence_residual:
            options.check_divergence_residual = value;
            return;
        case RunnerOptionId::stop_on_empty_y_marking:
            options.stop_on_empty_y_marking = value;
            return;
        case RunnerOptionId::uniform_x_refinement:
            options.uniform_x_refinement = value;
            return;
        case RunnerOptionId::uniform_y_refinement:
            options.uniform_y_refinement = value;
            return;
        case RunnerOptionId::force_accept_inner_with_effective_rho:
            options.force_accept_inner_with_effective_rho = value;
            return;
        case RunnerOptionId::compute_g_estimator:
            options.compute_g_estimator = value;
            return;
        case RunnerOptionId::compute_g_estimator_on_empty_y_marking_stop:
            options.compute_g_estimator_on_empty_y_marking_stop = value;
            return;
        case RunnerOptionId::compute_g_estimator_every_inner_iteration:
            options.compute_g_estimator_every_inner_iteration = value;
            return;
        case RunnerOptionId::local_time_slab_closure:
            options.local_time_slab_closure = value;
            return;
        case RunnerOptionId::use_adaptive_initial_guess:
            options.use_adaptive_initial_guess = value;
            return;
        case RunnerOptionId::solve_main_system_correction:
            options.solve_main_system_correction = value;
            return;
        case RunnerOptionId::fused_error_and_flux_diagnostics:
            options.fused_error_and_flux_diagnostics = value;
            return;
        case RunnerOptionId::local_error_reuse_patch_solve_workspace:
            options.local_error_reuse_patch_solve_workspace = value;
            return;
        case RunnerOptionId::local_error_coefficient_fast_path:
            options.local_error_coefficient_fast_path = value;
            return;
        case RunnerOptionId::local_error_compact_state_shadow:
            options.local_error_compact_state_shadow = value;
            return;
        case RunnerOptionId::deterministic_estimator_reductions:
            options.deterministic_estimator_reductions = value;
            return;
        case RunnerOptionId::refinement_edge_query_cache:
            options.refinement_edge_query_cache = value;
            return;
        case RunnerOptionId::post_flush_affected_containment_only:
            options.post_flush_affected_containment_only = value;
            return;
        case RunnerOptionId::refinement_full_conformity_check:
            options.refinement_full_conformity_check = value;
            return;
        case RunnerOptionId::allow_copied_time_slab_estimator_fallback:
            options.allow_copied_time_slab_estimator_fallback = value;
            return;
        case RunnerOptionId::virtual_backend_diagnostics:
            options.virtual_backend_diagnostics = value;
            return;
        case RunnerOptionId::main_solver_direct_residual_retry:
            options.main_solver_direct_residual_retry = value;
            return;
        case RunnerOptionId::increased_accuracy:
            options.increased_accuracy = value;
            return;
        case RunnerOptionId::main_solver_ooc_auto_switch:
            options.main_solver_ooc_auto_switch = value;
            return;
        case RunnerOptionId::main_solver_ooc_switch_to_lu:
            options.main_solver_ooc_switch_to_lu = value;
            return;
        case RunnerOptionId::main_solver_reuse_symbolic_analysis:
            options.main_solver_reuse_symbolic_analysis = value;
            return;
        case RunnerOptionId::export_history:
            options.export_history = value;
            return;
        case RunnerOptionId::save_estimator_components:
            options.save_estimator_components = value;
            return;
        case RunnerOptionId::save_refinement_history:
            options.save_refinement_history = value;
            return;
        case RunnerOptionId::save_mesh_statistics:
            options.save_mesh_statistics = value;
            return;
        case RunnerOptionId::save_iteration_snapshots:
            options.save_iteration_snapshots = value;
            return;
        case RunnerOptionId::save_snapshot_dofs:
            options.save_snapshot_dofs = value;
            return;
        case RunnerOptionId::enable_timing_breakdown:
            options.enable_timing_breakdown = value;
            return;
        case RunnerOptionId::solver_diagnostics_enabled:
            options.solver_diagnostics_enabled = value;
            return;
        case RunnerOptionId::solver_diagnostics_export_matrix_market:
            options.solver_diagnostics_export_matrix_market = value;
            return;
        case RunnerOptionId::solver_diagnostics_export_rhs:
            options.solver_diagnostics_export_rhs = value;
            return;
        case RunnerOptionId::solver_diagnostics_export_solution:
            options.solver_diagnostics_export_solution = value;
            return;
        case RunnerOptionId::save_heavy_diagnostics:
            options.save_heavy_diagnostics = value;
            return;
        default:
            throw std::logic_error("runner_parser: unexpected boolean option id.");
        }
    }

    inline void apply_typed_override(
        RunnerOptionOverrides& options,
        RunnerOptionId id,
        RunnerOptionValueType value_type,
        const std::string& value,
        const char* argv0,
        const std::string& option_name)
    {
        switch (value_type)
        {
        case RunnerOptionValueType::string_value:
            assign_string_override(options, id, value);
            return;
        case RunnerOptionValueType::int_value:
            assign_int_override(
                options,
                id,
                parse_value<int>(argv0, option_name, value.c_str()));
            return;
        case RunnerOptionValueType::double_value:
            assign_double_override(
                options,
                id,
                parse_value<double>(argv0, option_name, value.c_str()));
            return;
        case RunnerOptionValueType::bool_value:
            assign_bool_override(
                options,
                id,
                parse_bool_text(value, argv0, option_name));
            return;
        case RunnerOptionValueType::bool_flag:
            throw std::logic_error("runner_parser: boolean flags do not accept text values.");
        }

        throw std::logic_error("runner_parser: unsupported option value type.");
    }

    inline void disable_heavy_solver_exports(RunnerOptions& options)
    {
        options.solver_diagnostics_export_matrix_market = false;
        options.solver_diagnostics_export_rhs = false;
        options.solver_diagnostics_export_solution = false;
    }

    inline void apply_output_profile_defaults(
        RunnerOptions& options,
        const std::string& profile)
    {
        const std::string normalized = normalize_output_profile_choice(profile);
        options.output_profile = normalized;

        if (normalized == "minimal")
        {
            options.export_history = true;
            options.save_mesh_statistics = true;
            options.save_estimator_components = false;
            options.save_refinement_history = false;
            options.save_iteration_snapshots = false;
            options.save_snapshot_dofs = false;
            options.enable_timing_breakdown = false;
            options.timing_detail_level = "summary";
            options.main_solver_diagnostics = "summary";
            disable_heavy_solver_exports(options);
            return;
        }

        if (normalized == "production")
        {
            options.export_history = true;
            options.save_mesh_statistics = true;
            options.save_estimator_components = false;
            options.save_refinement_history = false;
            options.save_iteration_snapshots = false;
            options.save_snapshot_dofs = false;
            options.enable_timing_breakdown = true;
            options.timing_detail_level = "summary";
            options.main_solver_diagnostics = "summary";
            disable_heavy_solver_exports(options);
            return;
        }

        if (normalized == "benchmark")
        {
            options.export_history = true;
            options.save_mesh_statistics = true;
            options.save_estimator_components = true;
            options.save_refinement_history = true;
            options.save_iteration_snapshots = false;
            options.save_snapshot_dofs = false;
            options.enable_timing_breakdown = false;
            options.timing_detail_level = "summary";
            options.main_solver_diagnostics = "summary";
            disable_heavy_solver_exports(options);
            return;
        }

        if (normalized == "debug")
        {
            options.export_history = true;
            options.save_mesh_statistics = true;
            options.save_estimator_components = true;
            options.save_refinement_history = true;
            options.save_iteration_snapshots = true;
            options.save_snapshot_dofs = false;
            options.enable_timing_breakdown = true;
            options.timing_detail_level = "detailed";
            options.main_solver_diagnostics = "detailed";
            return;
        }

        throw std::runtime_error(
            "Unsupported output_profile '" + profile +
            "'. Supported values are: minimal, production, benchmark, debug.");
    }

    inline void apply_heavy_diagnostics_defaults(RunnerOptions& options)
    {
        options.save_heavy_diagnostics = true;
        options.export_history = true;
        options.save_mesh_statistics = true;
        options.save_estimator_components = true;
        options.save_refinement_history = true;
        options.save_iteration_snapshots = true;
        options.enable_timing_breakdown = true;
        options.timing_detail_level = "detailed";
        options.solver_diagnostics_enabled = true;
        options.main_solver_diagnostics = "detailed";
    }

    inline void apply_uniform_refinement_mode_defaults(
        RunnerOptions& options,
        const std::string& mode)
    {
        const std::string normalized =
            normalize_uniform_refinement_mode_choice(mode);
        options.uniform_refinement_mode = normalized;

        if (normalized == "adaptive")
        {
            options.uniform_x_refinement = false;
            options.uniform_y_refinement = false;
            return;
        }

        if (normalized == "uniform_x")
        {
            options.uniform_x_refinement = true;
            options.uniform_y_refinement = false;
            return;
        }

        if (normalized == "uniform_y")
        {
            options.uniform_x_refinement = false;
            options.uniform_y_refinement = true;
            return;
        }

        if (normalized == "uniform_xy")
        {
            options.uniform_x_refinement = true;
            options.uniform_y_refinement = true;
            return;
        }

        throw std::runtime_error(
            "Unsupported uniform_refinement_mode '" + mode +
            "'. Supported values are: adaptive, uniform_x, uniform_y, uniform_xy.");
    }

    inline void apply_overrides(
        RunnerOptions& options,
        const RunnerOptionOverrides& overrides)
    {
        if (overrides.uniform_refinement_mode.has_value())
        {
            apply_uniform_refinement_mode_defaults(
                options,
                *overrides.uniform_refinement_mode);
        }
        if (overrides.output_profile.has_value())
        {
            apply_output_profile_defaults(options, *overrides.output_profile);
        }
        if (overrides.save_heavy_diagnostics.has_value())
        {
            options.save_heavy_diagnostics =
                *overrides.save_heavy_diagnostics;
            if (*overrides.save_heavy_diagnostics)
                apply_heavy_diagnostics_defaults(options);
        }

        if (overrides.config_file.has_value())
            options.config_file = *overrides.config_file;
        if (overrides.example_name.has_value())
            options.example_name = *overrides.example_name;
        if (overrides.dimension.has_value())
            options.dimension = *overrides.dimension;
        if (overrides.rho.has_value())
            options.rho = *overrides.rho;
        if (overrides.max_outer_iterations.has_value())
            options.max_outer_iterations = *overrides.max_outer_iterations;
        if (overrides.max_inner_iterations.has_value())
            options.max_inner_iterations = *overrides.max_inner_iterations;
        if (overrides.doerfler_theta_y.has_value())
            options.doerfler_theta_y = *overrides.doerfler_theta_y;
        if (overrides.doerfler_theta_x.has_value())
            options.doerfler_theta_x = *overrides.doerfler_theta_x;
        if (overrides.uniform_x_refinement.has_value())
            options.uniform_x_refinement = *overrides.uniform_x_refinement;
        if (overrides.uniform_y_refinement.has_value())
            options.uniform_y_refinement = *overrides.uniform_y_refinement;
        if (overrides.force_accept_inner_with_effective_rho.has_value())
        {
            options.force_accept_inner_with_effective_rho =
                *overrides.force_accept_inner_with_effective_rho;
        }
        if (overrides.compute_g_estimator.has_value())
            options.compute_g_estimator = *overrides.compute_g_estimator;
        if (overrides.compute_g_estimator_on_empty_y_marking_stop.has_value())
        {
            options.compute_g_estimator_on_empty_y_marking_stop =
                *overrides.compute_g_estimator_on_empty_y_marking_stop;
        }
        if (overrides.compute_g_estimator_every_inner_iteration.has_value())
        {
            options.compute_g_estimator_every_inner_iteration =
                *overrides.compute_g_estimator_every_inner_iteration;
        }
        if (overrides.g_solver.has_value())
            options.g_solver = *overrides.g_solver;
        if (overrides.g_solver_tolerance.has_value())
            options.g_solver_tolerance = *overrides.g_solver_tolerance;
        if (overrides.g_solver_memory_limit_mb.has_value())
            options.g_solver_memory_limit_mb =
                *overrides.g_solver_memory_limit_mb;
        options.uniform_refinement_mode =
            effective_uniform_refinement_mode(
                options.uniform_x_refinement,
                options.uniform_y_refinement);
        if (overrides.polynomial_degree.has_value())
            options.polynomial_degree = *overrides.polynomial_degree;
        if (overrides.max_y_true_dofs.has_value())
            options.max_y_true_dofs = *overrides.max_y_true_dofs;
        if (overrides.max_x_true_dofs.has_value())
            options.max_x_true_dofs = *overrides.max_x_true_dofs;
        if (overrides.max_dofs_target.has_value())
            options.max_dofs_target = *overrides.max_dofs_target;
        if (overrides.increased_accuracy.has_value())
            options.increased_accuracy = *overrides.increased_accuracy;
        if (overrides.main_solver.has_value())
            options.main_solver = *overrides.main_solver;
        if (overrides.memory_limit_mb.has_value())
        {
            options.memory_limit_mb = *overrides.memory_limit_mb;
            options.memory_limit_explicit = true;
        }
        if (overrides.memory_reserve_mb.has_value())
            options.memory_reserve_mb = *overrides.memory_reserve_mb;
        if (overrides.memory_guard_safety_factor.has_value())
        {
            options.memory_guard_safety_factor =
                *overrides.memory_guard_safety_factor;
        }
        if (overrides.memory_guard_near_cap_fraction.has_value())
        {
            options.memory_guard_near_cap_fraction =
                *overrides.memory_guard_near_cap_fraction;
        }
        if (overrides.main_solver_pardiso_memory_mode.has_value())
        {
            options.main_solver_pardiso_memory_mode =
                *overrides.main_solver_pardiso_memory_mode;
        }
        if (overrides.main_solver_max_iterations.has_value())
        {
            options.main_solver_max_iterations =
                *overrides.main_solver_max_iterations;
        }
        if (overrides.main_solver_tolerance.has_value())
        {
            options.main_solver_tolerance =
                *overrides.main_solver_tolerance;
        }
        if (overrides.main_solver_symmetry_tolerance.has_value())
        {
            options.main_solver_symmetry_tolerance =
                *overrides.main_solver_symmetry_tolerance;
        }
        if (overrides.main_solver_direct_residual_retry.has_value())
        {
            options.main_solver_direct_residual_retry =
                *overrides.main_solver_direct_residual_retry;
        }
        if (overrides.main_solver_direct_residual_retry_tolerance.has_value())
        {
            options.main_solver_direct_residual_retry_tolerance =
                *overrides.main_solver_direct_residual_retry_tolerance;
        }
        if (overrides.main_solver_diagnostics.has_value())
        {
            options.main_solver_diagnostics =
                *overrides.main_solver_diagnostics;
        }
        if (overrides.main_solver_memory_limit_mb.has_value())
        {
            options.main_solver_memory_limit_mb =
                *overrides.main_solver_memory_limit_mb;
        }
        if (overrides.main_solver_ooc_auto_switch.has_value())
        {
            options.main_solver_ooc_auto_switch =
                *overrides.main_solver_ooc_auto_switch;
        }
        if (overrides.main_solver_ooc_switch_threshold.has_value())
        {
            options.main_solver_ooc_switch_threshold =
                *overrides.main_solver_ooc_switch_threshold;
        }
        if (overrides.main_solver_ooc_switch_to_lu.has_value())
        {
            options.main_solver_ooc_switch_to_lu =
                *overrides.main_solver_ooc_switch_to_lu;
        }
        if (overrides.main_solver_reuse_symbolic_analysis.has_value())
        {
            options.main_solver_reuse_symbolic_analysis =
                *overrides.main_solver_reuse_symbolic_analysis;
        }
        if (overrides.zero_tol.has_value())
            options.zero_tol = *overrides.zero_tol;
        if (overrides.divergence_residual_l2_tolerance.has_value())
        {
            options.divergence_residual_l2_tolerance =
                *overrides.divergence_residual_l2_tolerance;
        }
        if (overrides.eta_squared_stop.has_value())
            options.eta_squared_stop = *overrides.eta_squared_stop;
        if (overrides.inner_estimator_squared_stop.has_value())
        {
            options.inner_estimator_squared_stop =
                *overrides.inner_estimator_squared_stop;
        }
        if (overrides.max_wall_time_seconds.has_value())
            options.max_wall_time_seconds = *overrides.max_wall_time_seconds;

        if (overrides.quiet.has_value())
            options.quiet = *overrides.quiet;
        if (overrides.list_examples.has_value())
            options.list_examples = *overrides.list_examples;
        if (overrides.show_help.has_value())
            options.show_help = *overrides.show_help;
        if (overrides.check_divergence_residual.has_value())
        {
            options.check_divergence_residual =
                *overrides.check_divergence_residual;
        }
        if (overrides.stop_on_empty_y_marking.has_value())
        {
            options.stop_on_empty_y_marking =
                *overrides.stop_on_empty_y_marking;
        }
        if (overrides.local_time_slab_closure.has_value())
        {
            options.local_time_slab_closure =
                *overrides.local_time_slab_closure;
        }
        if (overrides.use_adaptive_initial_guess.has_value())
        {
            options.use_adaptive_initial_guess =
                *overrides.use_adaptive_initial_guess;
            options.use_adaptive_initial_guess_explicit = true;
        }
        if (overrides.solve_main_system_correction.has_value())
        {
            options.solve_main_system_correction =
                *overrides.solve_main_system_correction;
        }
        if (overrides.fused_error_and_flux_diagnostics.has_value())
        {
            options.fused_error_and_flux_diagnostics =
                *overrides.fused_error_and_flux_diagnostics;
        }
        if (overrides.local_error_reuse_patch_solve_workspace.has_value())
        {
            options.local_error_reuse_patch_solve_workspace =
                *overrides.local_error_reuse_patch_solve_workspace;
        }
        if (overrides.deterministic_estimator_reductions.has_value())
        {
            options.deterministic_estimator_reductions =
                *overrides.deterministic_estimator_reductions;
        }
        if (overrides.doerfler_near_tie_tolerance.has_value())
        {
            options.doerfler_near_tie_tolerance =
                *overrides.doerfler_near_tie_tolerance;
        }
        if (overrides.refinement_edge_query_cache.has_value())
        {
            options.refinement_edge_query_cache =
                *overrides.refinement_edge_query_cache;
        }
        if (overrides.refinement_batch_target_split_cells.has_value())
        {
            options.refinement_batch_target_split_cells =
                *overrides.refinement_batch_target_split_cells;
        }
        if (overrides.post_flush_closure_mode.has_value())
        {
            options.post_flush_closure_mode =
                normalize_post_flush_closure_mode_choice(
                    *overrides.post_flush_closure_mode);
        }
        if (overrides.post_flush_affected_containment_only.has_value())
        {
            options.post_flush_affected_containment_only =
                *overrides.post_flush_affected_containment_only;
        }
        if (overrides.refinement_full_conformity_check.has_value())
        {
            options.refinement_full_conformity_check =
                *overrides.refinement_full_conformity_check;
        }
        if (overrides.refinement_main_closure_query_mode.has_value())
        {
            options.refinement_main_closure_query_mode =
                normalize_refinement_main_closure_query_mode_choice(
                    *overrides.refinement_main_closure_query_mode);
        }
        if (overrides.local_error_patch_tile_size.has_value())
        {
            options.local_error_patch_tile_size =
                *overrides.local_error_patch_tile_size;
        }
        if (overrides.local_error_cell_chunk_size.has_value())
        {
            options.local_error_cell_chunk_size =
                *overrides.local_error_cell_chunk_size;
        }
        if (overrides.local_error_max_threads.has_value())
        {
            options.local_error_max_threads =
                *overrides.local_error_max_threads;
        }
        if (overrides.local_error_memory_budget_mb.has_value())
        {
            options.local_error_memory_budget_mb =
                *overrides.local_error_memory_budget_mb;
        }
        if (overrides.local_error_worker_context_mode.has_value())
        {
            // Deprecated compatibility alias. local_error_context_storage is
            // authoritative; when only the old worker option is provided, map
            // it to the equivalent debug storage mode.
            options.local_error_worker_context_mode =
                normalize_local_error_worker_context_mode_choice(
                    *overrides.local_error_worker_context_mode);
            options.local_error_worker_context_mode_explicit = true;
            if (!overrides.local_error_context_storage.has_value())
            {
                options.local_error_context_storage =
                    options.local_error_worker_context_mode ==
                            "per_chunk_debug"
                        ? "per_chunk_debug"
                        : "persistent_per_thread_debug";
            }
        }
        if (overrides.local_error_context_storage.has_value())
        {
            options.local_error_context_storage =
                normalize_local_error_context_storage_choice(
                    *overrides.local_error_context_storage);
            options.local_error_context_storage_explicit = true;
        }
        if (overrides.local_error_state_index_mode.has_value())
        {
            options.local_error_state_index_mode =
                normalize_local_error_state_index_mode_choice(
                    *overrides.local_error_state_index_mode);
        }
        if (overrides.local_error_cell_state_cache_mode.has_value())
        {
            options.local_error_cell_state_cache_mode =
                normalize_local_error_cell_state_cache_mode_choice(
                    *overrides.local_error_cell_state_cache_mode);
        }
        if (overrides.local_error_cell_state_cache_budget_mb.has_value())
        {
            options.local_error_cell_state_cache_budget_mb =
                *overrides.local_error_cell_state_cache_budget_mb;
        }
        if (overrides.local_error_cell_state_representation.has_value())
        {
            options.local_error_cell_state_representation =
                normalize_local_error_cell_state_representation_choice(
                    *overrides.local_error_cell_state_representation);
        }
        if (overrides.local_error_flux_diagnostics_mode.has_value())
        {
            options.local_error_flux_diagnostics_mode =
                normalize_local_error_flux_diagnostics_mode_choice(
                    *overrides.local_error_flux_diagnostics_mode);
        }
        if (overrides.local_error_patch_solver.has_value())
        {
            options.local_error_patch_solver =
                normalize_local_error_patch_solver_choice(
                    *overrides.local_error_patch_solver);
        }
        if (overrides.local_error_coefficient_fast_path.has_value())
        {
            options.local_error_coefficient_fast_path =
                *overrides.local_error_coefficient_fast_path;
        }
        if (overrides.local_error_compact_state_shadow.has_value())
        {
            options.local_error_compact_state_shadow =
                *overrides.local_error_compact_state_shadow;
        }
        if (overrides.shared_context_validation.has_value())
        {
            options.shared_context_validation =
                normalize_shared_context_validation_choice(
                    *overrides.shared_context_validation);
        }
        if (overrides.slab_reconstruction_operator_mode.has_value())
        {
            options.slab_reconstruction_operator_mode =
                normalize_slab_reconstruction_operator_mode_choice(
                    *overrides.slab_reconstruction_operator_mode);
        }
        if (overrides.main_assembly_max_threads.has_value())
        {
            options.main_assembly_max_threads =
                *overrides.main_assembly_max_threads;
        }
        if (overrides.slab_reconstruction_max_threads.has_value())
        {
            options.slab_reconstruction_max_threads =
                *overrides.slab_reconstruction_max_threads;
        }
        if (overrides.main_assembly_memory_budget_mb.has_value())
        {
            options.main_assembly_memory_budget_mb =
                *overrides.main_assembly_memory_budget_mb;
        }
        if (overrides.slab_reconstruction_memory_budget_mb.has_value())
        {
            options.slab_reconstruction_memory_budget_mb =
                *overrides.slab_reconstruction_memory_budget_mb;
        }
        if (overrides.main_two_pass_numeric_fill_max_threads.has_value())
        {
            options.main_two_pass_numeric_fill_max_threads =
                *overrides.main_two_pass_numeric_fill_max_threads;
        }
        if (overrides.main_two_pass_numeric_fill_memory_budget_mb.has_value())
        {
            options.main_two_pass_numeric_fill_memory_budget_mb =
                *overrides.main_two_pass_numeric_fill_memory_budget_mb;
        }
        if (overrides.time_slab_backend.has_value())
            options.time_slab_backend = *overrides.time_slab_backend;
        if (overrides.allow_copied_time_slab_estimator_fallback.has_value())
        {
            options.allow_copied_time_slab_estimator_fallback =
                *overrides.allow_copied_time_slab_estimator_fallback;
        }
        if (overrides.virtual_backend_diagnostics.has_value())
        {
            options.virtual_backend_diagnostics =
                *overrides.virtual_backend_diagnostics;
        }
        if (overrides.solver_diagnostics_enabled.has_value())
        {
            options.solver_diagnostics_enabled =
                *overrides.solver_diagnostics_enabled;
        }
        if (overrides.solver_diagnostics_export_matrix_market.has_value())
        {
            options.solver_diagnostics_export_matrix_market =
                *overrides.solver_diagnostics_export_matrix_market;
        }
        if (overrides.solver_diagnostics_export_rhs.has_value())
        {
            options.solver_diagnostics_export_rhs =
                *overrides.solver_diagnostics_export_rhs;
        }
        if (overrides.solver_diagnostics_export_solution.has_value())
        {
            options.solver_diagnostics_export_solution =
                *overrides.solver_diagnostics_export_solution;
        }
        if (overrides.solver_diagnostics_max_export_dofs.has_value())
        {
            options.solver_diagnostics_max_export_dofs =
                *overrides.solver_diagnostics_max_export_dofs;
        }
        if (overrides.output_profile.has_value())
        {
            options.output_profile =
                normalize_output_profile_choice(*overrides.output_profile);
        }
        if (overrides.save_heavy_diagnostics.has_value())
        {
            options.save_heavy_diagnostics =
                *overrides.save_heavy_diagnostics;
        }
        if (overrides.export_history.has_value())
            options.export_history = *overrides.export_history;
        if (overrides.save_estimator_components.has_value())
            options.save_estimator_components =
                *overrides.save_estimator_components;
        if (overrides.save_refinement_history.has_value())
            options.save_refinement_history =
                *overrides.save_refinement_history;
        if (overrides.save_mesh_statistics.has_value())
            options.save_mesh_statistics =
                *overrides.save_mesh_statistics;
        if (overrides.save_iteration_snapshots.has_value())
            options.save_iteration_snapshots =
                *overrides.save_iteration_snapshots;
        if (overrides.save_snapshot_dofs.has_value())
            options.save_snapshot_dofs = *overrides.save_snapshot_dofs;
        if (overrides.enable_timing_breakdown.has_value())
            options.enable_timing_breakdown =
                *overrides.enable_timing_breakdown;

        if (overrides.output_directory.has_value())
            options.output_directory = *overrides.output_directory;
        if (overrides.timing_history_filename.has_value())
        {
            options.timing_history_filename =
                *overrides.timing_history_filename;
        }
        if (overrides.timing_detail_level.has_value())
            options.timing_detail_level = *overrides.timing_detail_level;

        if (options.save_snapshot_dofs)
            options.save_iteration_snapshots = true;
    }
}

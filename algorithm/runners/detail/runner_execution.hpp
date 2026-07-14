#pragma once

#include <algorithm>
#include <array>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "../../adaptive_parameters.hpp"
#include "../../example_registry.hpp"
#include "../../output/output_paths.hpp"

#include "linear_algebra/concepts/solver.hpp"

#include "runner_cases.hpp"
#include "runner_options.hpp"
#include "runner_parser.hpp"

#ifndef ADAPPARABOLICFEM_HAVE_MKL_PARDISO
#define ADAPPARABOLICFEM_HAVE_MKL_PARDISO 0
#endif

namespace adaptive_algorithm::runners::detail
{
    struct MainSolverChoice
    {
        std::string_view name{};
        la::concepts::SolverType solver = la::concepts::SolverType::SparseLU;
        la::concepts::PreconditionerType preconditioner =
            la::concepts::PreconditionerType::None;
    };

    struct PardisoMemoryModeChoice
    {
        std::string_view name{};
        la::concepts::PardisoMemoryMode mode =
            la::concepts::PardisoMemoryMode::InCore;
    };

    inline constexpr auto main_solver_choices = std::to_array<MainSolverChoice>({
        {"sparse_lu",
         la::concepts::SolverType::SparseLU,
         la::concepts::PreconditionerType::None},
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        {"pardiso_lu",
         la::concepts::SolverType::PardisoLU,
         la::concepts::PreconditionerType::None},
        {"pardiso_ldlt",
         la::concepts::SolverType::PardisoLDLT,
         la::concepts::PreconditionerType::None},
        {"pardiso_ldlt_auto",
         la::concepts::SolverType::PardisoLDLTAuto,
         la::concepts::PreconditionerType::None},
#endif
        {"minres_parabolic_graph_norm",
         la::concepts::SolverType::MINRES,
         la::concepts::PreconditionerType::ParabolicGraphNorm},
    });

    inline constexpr auto pardiso_memory_mode_choices =
        std::to_array<PardisoMemoryModeChoice>({
            {"in_core", la::concepts::PardisoMemoryMode::InCore},
            {"auto", la::concepts::PardisoMemoryMode::Auto},
            {"out_of_core", la::concepts::PardisoMemoryMode::OutOfCore},
        });

    [[nodiscard]] inline std::string normalize_solver_choice(std::string text)
    {
        std::transform(
            text.begin(),
            text.end(),
            text.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == '+' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });

        return text;
    }

    [[nodiscard]] inline std::string normalize_pardiso_memory_mode_choice(
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

    [[nodiscard]] inline std::string normalize_timing_detail_level_choice(
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

    [[nodiscard]] inline bool main_solver_diagnostics_mode_is_supported(
        const std::string& value)
    {
        const std::string normalized =
            normalize_main_solver_diagnostics_choice(value);
        return normalized == "off" ||
            normalized == "summary" ||
            normalized == "detailed";
    }

    [[nodiscard]] inline std::string
    supported_main_solver_diagnostics_modes_message()
    {
        return "off, summary, detailed";
    }

    [[nodiscard]] inline la::concepts::SolverDiagnosticsMode
    parse_main_solver_diagnostics_mode(const std::string& value)
    {
        const std::string normalized =
            normalize_main_solver_diagnostics_choice(value);
        if (normalized == "off")
            return la::concepts::SolverDiagnosticsMode::Off;
        if (normalized == "summary")
            return la::concepts::SolverDiagnosticsMode::Summary;
        if (normalized == "detailed")
            return la::concepts::SolverDiagnosticsMode::Detailed;

        throw std::runtime_error(
            "unsupported main_solver_diagnostics='" + normalized +
            "'. Supported values are: " +
            supported_main_solver_diagnostics_modes_message() + ".");
    }

    [[nodiscard]] inline DofLimitTarget parse_dof_limit_target_choice(
        std::string value)
    {
        std::transform(
            value.begin(),
            value.end(),
            value.begin(),
            [](unsigned char c) -> char
            {
                if (c == '-' || c == ' ')
                    return '_';

                return static_cast<char>(std::tolower(c));
            });
        const std::string& normalized = value;
        if (normalized == "y" || normalized == "y_true_dofs" ||
            normalized == "y_dofs")
        {
            return DofLimitTarget::YTrueDofs;
        }
        if (normalized == "x" || normalized == "x_true_dofs" ||
            normalized == "x_dofs")
        {
            return DofLimitTarget::XTrueDofs;
        }

        throw std::runtime_error(
            "unsupported max_dofs_target='" + normalized +
            "'. Supported values are: y, x.");
    }

    [[nodiscard]] inline std::string supported_main_solver_choices_message()
    {
        std::string message;
        for (std::size_t i = 0; i < main_solver_choices.size(); ++i)
        {
            if (i > 0)
                message += ", ";

            message += main_solver_choices[i].name;
        }

#if !ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        for (const std::string_view name :
             {"pardiso_lu", "pardiso_ldlt", "pardiso_ldlt_auto"})
        {
            if (!message.empty())
                message += ", ";

            message += name;
        }
#endif

        return message;
    }

    [[nodiscard]] inline std::string supported_pardiso_memory_mode_choices_message()
    {
        std::string message;
        for (std::size_t i = 0; i < pardiso_memory_mode_choices.size(); ++i)
        {
            if (i > 0)
                message += ", ";

            message += pardiso_memory_mode_choices[i].name;
        }

        return message;
    }

    [[nodiscard]] inline const MainSolverChoice* find_main_solver_choice(
        const std::string& value)
    {
        const std::string normalized = normalize_solver_choice(value);
        const auto it =
            std::find_if(
                main_solver_choices.begin(),
                main_solver_choices.end(),
                [&normalized](const MainSolverChoice& choice)
                {
                    return choice.name == normalized;
                });

        return it == main_solver_choices.end() ? nullptr : &*it;
    }

    [[nodiscard]] inline const PardisoMemoryModeChoice*
    find_pardiso_memory_mode_choice(const std::string& value)
    {
        const std::string normalized =
            normalize_pardiso_memory_mode_choice(value);
        const auto it =
            std::find_if(
                pardiso_memory_mode_choices.begin(),
                pardiso_memory_mode_choices.end(),
                [&normalized](const PardisoMemoryModeChoice& choice)
                {
                    return choice.name == normalized;
                });

        return it == pardiso_memory_mode_choices.end() ? nullptr : &*it;
    }

    [[nodiscard]] inline bool requests_pardiso_lu(
        const std::string& value)
    {
        return normalize_solver_choice(value) == "pardiso_lu";
    }

    [[nodiscard]] inline bool requests_pardiso_ldlt(
        const std::string& value)
    {
        return normalize_solver_choice(value) == "pardiso_ldlt";
    }

    [[nodiscard]] inline bool requests_pardiso_ldlt_auto(
        const std::string& value)
    {
        return normalize_solver_choice(value) == "pardiso_ldlt_auto";
    }

    [[nodiscard]] inline bool requests_mkl_pardiso_solver(
        const std::string& value)
    {
        return requests_pardiso_lu(value) ||
            requests_pardiso_ldlt(value) ||
            requests_pardiso_ldlt_auto(value);
    }

    [[nodiscard]] inline bool main_solver_falls_back_to_sparse_lu(
        const std::string& value)
    {
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        static_cast<void>(value);
        return false;
#else
        return requests_mkl_pardiso_solver(value);
#endif
    }

    [[nodiscard]] inline bool solver_type_is_iterative(
        la::concepts::SolverType solver) noexcept
    {
        return solver == la::concepts::SolverType::MINRES;
    }

    [[nodiscard]] inline bool main_solver_uses_adaptive_initial_guess_by_default(
        const std::string& value)
    {
        if (main_solver_falls_back_to_sparse_lu(value))
            return false;

        const MainSolverChoice* choice = find_main_solver_choice(value);
        return choice != nullptr && solver_type_is_iterative(choice->solver);
    }

    [[nodiscard]] inline bool resolved_use_adaptive_initial_guess(
        const RunnerOptions& options)
    {
        if (options.use_adaptive_initial_guess_explicit)
            return options.use_adaptive_initial_guess;

        return main_solver_uses_adaptive_initial_guess_by_default(
            options.main_solver);
    }

    [[nodiscard]] inline int default_local_error_tile_size_for_degree(
        int p) noexcept
    {
        return p >= 3 ? 256 : 512;
    }

    [[nodiscard]] inline int default_local_error_cell_chunk_size_for_degree(
        int p) noexcept
    {
        return p >= 3 ? 256 : 512;
    }

    [[nodiscard]] inline int default_two_pass_numeric_fill_threads_for_degree(
        int p) noexcept
    {
        (void)p;
        return 4;
    }

    [[nodiscard]] inline std::string resolved_main_solver_name(
        const std::string& value)
    {
        if (main_solver_falls_back_to_sparse_lu(value))
            return "sparse_lu";

        return normalize_solver_choice(value);
    }

    [[nodiscard]] inline bool is_same_as_main_solver_choice(
        const std::string& value)
    {
        const std::string normalized = normalize_solver_choice(value);
        return normalized == "same_as_main" ||
            normalized == "main" ||
            normalized == "inherit_main";
    }

    [[nodiscard]] inline std::string effective_g_solver_choice(
        const RunnerOptions& options)
    {
        return is_same_as_main_solver_choice(options.g_solver)
            ? options.main_solver
            : options.g_solver;
    }

    [[nodiscard]] inline std::string resolved_g_solver_name(
        const RunnerOptions& options)
    {
        return resolved_main_solver_name(effective_g_solver_choice(options));
    }

    [[nodiscard]] inline int resolved_spatial_dimension(
        const RunnerOptions& options)
    {
        if (options.dimension.has_value())
        {
            if (adaptive_algorithm::examples::
                    example_is_registered_for_dimension(
                        options.example_name,
                        *options.dimension))
            {
                return *options.dimension;
            }

            throw std::runtime_error(
                "Requested dimension " + std::to_string(*options.dimension) +
                " does not match example '" + options.example_name + "'.");
        }

        return adaptive_algorithm::examples::example_spatial_dimension(
            options.example_name);
    }

    inline void print_available_examples()
    {
        std::cout << "Available examples:\n";
        for (const auto& descriptor :
             adaptive_algorithm::examples::available_examples())
        {
            std::cout << "  " << descriptor.name << " ("
                      << adaptive_algorithm::examples::dimension_label(
                             descriptor.dim_space)
                      << ")\n";
        }
    }

    inline void validate_runner_options(
        const RunnerOptions& options,
        const char* argv0)
    {
        const auto throw_invalid_value =
            [argv0](const std::string& option, const std::string& message)
            {
                throw std::runtime_error(
                    usage_text(argv0) +
                    "\nInvalid value for option '" + option + "': " + message);
            };

        const auto available_examples =
            adaptive_algorithm::examples::available_example_names();
        if (std::find(
                available_examples.begin(),
                available_examples.end(),
                options.example_name) == available_examples.end())
        {
            throw std::runtime_error(
                usage_text(argv0) +
                "\nUnknown example '" + options.example_name + "'. " +
                adaptive_algorithm::examples::available_example_names_message());
        }

        if (options.dimension.has_value())
        {
            if (*options.dimension != 1 && *options.dimension != 2)
            {
                throw_invalid_value(
                    "--dimension",
                    "expected 1 or 2.");
            }

            if (!adaptive_algorithm::examples::
                    example_is_registered_for_dimension(
                        options.example_name,
                        *options.dimension))
            {
                throw_invalid_value(
                    "--dimension",
                    "example '" + options.example_name +
                        "' is not registered for dimension " +
                        std::to_string(*options.dimension) + ".");
            }
        }
        else
        {
            try
            {
                (void)adaptive_algorithm::examples::example_spatial_dimension(
                    options.example_name);
            }
            catch (const std::runtime_error& e)
            {
                throw std::runtime_error(
                    usage_text(argv0) + "\n" + e.what());
            }
        }

        if (!(options.rho > 0.0))
            throw_invalid_value("--rho", "expected a positive number.");

        if (options.max_outer_iterations < 0)
        {
            throw_invalid_value(
                "--max-outer",
                "expected a nonnegative integer.");
        }

        if (options.max_outer_iterations > 0 &&
            options.max_inner_iterations <= 0)
        {
            throw_invalid_value(
                "--max-inner",
                "expected a positive integer when outer iterations are enabled.");
        }

        if (!(options.doerfler_theta_x > 0.0 &&
              options.doerfler_theta_x <= 1.0))
        {
            throw_invalid_value(
                "--theta-x",
                "expected a number in the interval (0, 1].");
        }

        if (!(options.doerfler_theta_y > 0.0 &&
              options.doerfler_theta_y <= 1.0))
        {
            throw_invalid_value(
                "--theta-y",
                "expected a number in the interval (0, 1].");
        }

        if (options.zero_tol < 0.0)
            throw_invalid_value("--zero-tol", "expected a nonnegative number.");

        if (options.divergence_residual_l2_tolerance < 0.0)
        {
            throw_invalid_value(
                "--divergence-tol",
                "expected a nonnegative number.");
        }

        if (options.eta_squared_stop < 0.0)
            throw_invalid_value("--eta-stop", "expected a nonnegative number.");

        if (options.inner_estimator_squared_stop < 0.0)
        {
            throw_invalid_value(
                "--inner-estimator-stop",
                "expected a nonnegative number.");
        }

        if (options.max_y_true_dofs < 0)
        {
            throw_invalid_value(
                "--max-y-dofs",
                "expected a nonnegative integer.");
        }

        if (options.max_x_true_dofs < 0)
        {
            throw_invalid_value(
                "--max-x-dofs",
                "expected a nonnegative integer.");
        }

        try
        {
            static_cast<void>(
                parse_dof_limit_target_choice(options.max_dofs_target));
        }
        catch (const std::exception& e)
        {
            throw_invalid_value("--max-dofs-target", e.what());
        }

        const MainSolverChoice* main_solver_choice =
            find_main_solver_choice(options.main_solver);
        if (main_solver_choice == nullptr &&
            !main_solver_falls_back_to_sparse_lu(options.main_solver))
        {
            throw_invalid_value(
                "--main-solver",
                "supported values are: " +
                    supported_main_solver_choices_message() + '.');
        }

        if (main_solver_choice != nullptr)
        {
            const auto validation =
                la::concepts::validate_solver_preconditioner(
                    main_solver_choice->solver,
                    main_solver_choice->preconditioner);
            if (!validation.accepted)
            {
                throw_invalid_value(
                    "--main-solver",
                    validation.rejection_reason);
            }
        }

        const std::string effective_g_solver =
            effective_g_solver_choice(options);
        const MainSolverChoice* g_solver_choice =
            find_main_solver_choice(effective_g_solver);
        if (g_solver_choice == nullptr &&
            !main_solver_falls_back_to_sparse_lu(effective_g_solver))
        {
            throw_invalid_value(
                "--g-solver",
                "supported values are: same_as_main, " +
                    supported_main_solver_choices_message() + '.');
        }

        if (g_solver_choice != nullptr)
        {
            const auto validation =
                la::concepts::validate_solver_preconditioner(
                    g_solver_choice->solver,
                    g_solver_choice->preconditioner);
            if (!validation.accepted)
            {
                throw_invalid_value(
                    "--g-solver",
                    validation.rejection_reason);
            }
        }

        if (find_pardiso_memory_mode_choice(
                options.main_solver_pardiso_memory_mode) == nullptr)
        {
            throw_invalid_value(
                "--main-solver-pardiso-memory-mode",
                "supported values are: " +
                    supported_pardiso_memory_mode_choices_message() + '.');
        }

        if (options.main_solver_max_iterations <= 0)
        {
            throw_invalid_value(
                "--main-solver-max-iterations",
                "expected a positive integer.");
        }

        if (!(options.main_solver_tolerance > 0.0))
        {
            throw_invalid_value(
                "--main-solver-tolerance",
                "expected a positive number.");
        }

        if (options.g_solver_tolerance < 0.0)
        {
            throw_invalid_value(
                "--g-solver-tolerance",
                "expected a nonnegative number; 0 inherits the main-solver tolerance.");
        }

        if (options.main_solver_symmetry_tolerance < 0.0)
        {
            throw_invalid_value(
                "--main-solver-symmetry-tolerance",
                "expected a nonnegative number.");
        }

        if (!main_solver_diagnostics_mode_is_supported(
                options.main_solver_diagnostics))
        {
            throw_invalid_value(
                "--main-solver-diagnostics",
                "supported values are: " +
                    supported_main_solver_diagnostics_modes_message() + ".");
        }

        if (!(options.main_solver_direct_residual_retry_tolerance > 0.0))
        {
            throw_invalid_value(
                "--main-solver-direct-residual-retry-tolerance",
                "expected a positive number.");
        }

        if (options.main_solver_memory_limit_mb < 0.0)
        {
            throw_invalid_value(
                "--main-solver-memory-limit-mb",
                "expected a nonnegative number.");
        }

        if (options.g_solver_memory_limit_mb < 0.0)
        {
            throw_invalid_value(
                "--g-solver-memory-limit-mb",
                "expected a nonnegative number; 0 inherits the main-solver memory limit.");
        }

        if (!(options.main_solver_ooc_switch_threshold > 0.0) ||
            options.main_solver_ooc_switch_threshold > 1.0)
        {
            throw_invalid_value(
                "--main-solver-ooc-switch-threshold",
                "expected a number in the interval (0, 1].");
        }

        if (options.max_wall_time_seconds < 0.0)
        {
            throw_invalid_value(
                "--max-wall-time-seconds",
                "expected a nonnegative number.");
        }

        if (options.memory_limit_mb < 0.0)
        {
            throw_invalid_value(
                "--memory-limit-mb",
                "expected a nonnegative number.");
        }

        if (options.memory_reserve_mb < 0.0)
        {
            throw_invalid_value(
                "--memory-reserve-mb",
                "expected a nonnegative number.");
        }

        if (!(options.memory_guard_safety_factor >= 1.0))
        {
            throw_invalid_value(
                "--memory-guard-safety-factor",
                "expected a number >= 1.");
        }

        if (!(options.memory_guard_near_cap_fraction > 0.0) ||
            options.memory_guard_near_cap_fraction > 1.0)
        {
            throw_invalid_value(
                "--memory-guard-near-cap-fraction",
                "expected a number in the interval (0, 1].");
        }

        if (options.local_error_patch_tile_size < 0)
        {
            throw_invalid_value(
                "--local-error-patch-tile-size",
                "expected a nonnegative integer.");
        }

        if (options.refinement_batch_target_split_cells <= 0)
        {
            throw_invalid_value(
                "--refinement-batch-target-split-cells",
                "expected a positive integer.");
        }

        const std::string post_flush_closure_mode =
            normalize_post_flush_closure_mode_choice(
                options.post_flush_closure_mode);
        if (post_flush_closure_mode != "off_debug" &&
            post_flush_closure_mode != "split_edges_only_debug" &&
            post_flush_closure_mode != "split_and_inherited_edges" &&
            post_flush_closure_mode != "affected_edges" &&
            post_flush_closure_mode != "presplit_neighbour" &&
            post_flush_closure_mode != "all_faces_debug")
        {
            throw_invalid_value(
                "--post-flush-closure-mode",
                "expected off_debug, split_edges_only_debug, split_and_inherited_edges, affected_edges, presplit_neighbour, or all_faces_debug.");
        }

        const std::string refinement_main_closure_query_mode =
            normalize_refinement_main_closure_query_mode_choice(
                options.refinement_main_closure_query_mode);
        if (refinement_main_closure_query_mode != "exact_and_ancestor" &&
            refinement_main_closure_query_mode !=
                "exact_ancestor_plus_containment" &&
            refinement_main_closure_query_mode != "old_bidirectional_debug")
        {
            throw_invalid_value(
                "--refinement-main-closure-query-mode",
                "expected exact_and_ancestor, exact_ancestor_plus_containment, or old_bidirectional_debug.");
        }

        if (options.local_error_cell_chunk_size < 0)
        {
            throw_invalid_value(
                "--local-error-cell-chunk-size",
                "expected a nonnegative integer.");
        }

        if (options.local_error_max_threads < 0)
        {
            throw_invalid_value(
                "--local-error-max-threads",
                "expected a nonnegative integer.");
        }

        if (options.main_assembly_max_threads < 0)
        {
            throw_invalid_value(
                "--main-assembly-max-threads",
                "expected a nonnegative integer.");
        }

        if (options.slab_reconstruction_max_threads < 0)
        {
            throw_invalid_value(
                "--slab-reconstruction-max-threads",
                "expected a nonnegative integer.");
        }

        if (options.main_two_pass_numeric_fill_max_threads < 0)
        {
            throw_invalid_value(
                "--main-two-pass-numeric-fill-max-threads",
                "expected a nonnegative integer.");
        }

        if (options.local_error_memory_budget_mb < 0.0)
        {
            throw_invalid_value(
                "--local-error-memory-budget-mb",
                "expected a nonnegative number.");
        }
        if (options.local_error_cell_state_cache_budget_mb < 0.0)
        {
            throw_invalid_value(
                "--local-error-cell-state-cache-budget-mb",
                "expected a nonnegative number.");
        }
        if (options.doerfler_near_tie_tolerance < 0.0)
        {
            throw_invalid_value(
                "--doerfler-near-tie-tolerance",
                "expected a nonnegative number.");
        }
        const std::string local_error_worker_context_mode =
            normalize_local_error_worker_context_mode_choice(
                options.local_error_worker_context_mode);
        if (local_error_worker_context_mode != "persistent" &&
            local_error_worker_context_mode != "per_chunk_debug" &&
            local_error_worker_context_mode != "persistent_all_p_debug")
        {
            throw_invalid_value(
                "--local-error-worker-context-mode",
                "expected persistent, per_chunk_debug, or persistent_all_p_debug.");
        }
        const std::string local_error_context_storage =
            normalize_local_error_context_storage_choice(
                options.local_error_context_storage);
        if (local_error_context_storage != "per_chunk_debug" &&
            local_error_context_storage != "persistent_per_thread_debug" &&
            local_error_context_storage != "shared_immutable" &&
            local_error_context_storage != "shared_immutable_shadow")
        {
            throw_invalid_value(
                "--local-error-context-storage",
                "expected per_chunk_debug, persistent_per_thread_debug, shared_immutable, or shared_immutable_shadow.");
        }
        if (options.local_error_worker_context_mode_explicit &&
            options.local_error_context_storage_explicit)
        {
            const bool compatible =
                (local_error_context_storage == "per_chunk_debug" &&
                 local_error_worker_context_mode == "per_chunk_debug") ||
                (local_error_context_storage ==
                     "persistent_per_thread_debug" &&
                 (local_error_worker_context_mode == "persistent" ||
                  local_error_worker_context_mode ==
                      "persistent_all_p_debug"));
            if (!compatible)
            {
                throw_invalid_value(
                    "--local-error-worker-context-mode / --local-error-context-storage",
                    "local_error_context_storage is authoritative. Do not combine the deprecated worker-context option with shared_immutable or shared_immutable_shadow, and only combine it with matching debug storage modes.");
            }
        }
        const std::string local_error_state_index_mode =
            normalize_local_error_state_index_mode_choice(
                options.local_error_state_index_mode);
        if (local_error_state_index_mode != "flat" &&
            local_error_state_index_mode != "map_debug")
        {
            throw_invalid_value(
                "--local-error-state-index-mode",
                "expected flat or map_debug.");
        }
        const std::string local_error_cell_state_cache_mode =
            normalize_local_error_cell_state_cache_mode_choice(
                options.local_error_cell_state_cache_mode);
        if (local_error_cell_state_cache_mode != "off" &&
            local_error_cell_state_cache_mode != "tile" &&
            local_error_cell_state_cache_mode != "bounded_lru" &&
            local_error_cell_state_cache_mode != "lifetime_window" &&
            local_error_cell_state_cache_mode != "full_if_fits")
        {
            throw_invalid_value(
                "--local-error-cell-state-cache-mode",
                "expected off, tile, bounded_lru, lifetime_window, or full_if_fits.");
        }
        const std::string local_error_cell_state_representation =
            normalize_local_error_cell_state_representation_choice(
                options.local_error_cell_state_representation);
        if (local_error_cell_state_representation != "compact_split" &&
            local_error_cell_state_representation != "monolithic_debug")
        {
            throw_invalid_value(
                "--local-error-cell-state-representation",
                "expected compact_split or monolithic_debug.");
        }
        const std::string local_error_flux_diagnostics_mode =
            normalize_local_error_flux_diagnostics_mode_choice(
                options.local_error_flux_diagnostics_mode);
        if (local_error_flux_diagnostics_mode != "auto" &&
            local_error_flux_diagnostics_mode != "streaming_reuse" &&
            local_error_flux_diagnostics_mode != "standalone")
        {
            throw_invalid_value(
                "--local-error-flux-diagnostics-mode",
                "expected auto, streaming_reuse, or standalone.");
        }
        const std::string local_error_patch_solver =
            normalize_local_error_patch_solver_choice(
                options.local_error_patch_solver);
        if (local_error_patch_solver != "current_dense" &&
            local_error_patch_solver != "reduced_scalar_dense" &&
            local_error_patch_solver != "auto")
        {
            throw_invalid_value(
                "--local-error-patch-solver",
                "expected current_dense, reduced_scalar_dense, or auto.");
        }

        const std::string slab_reconstruction_operator_mode =
            normalize_slab_reconstruction_operator_mode_choice(
                options.slab_reconstruction_operator_mode);
        if (slab_reconstruction_operator_mode != "auto" &&
            slab_reconstruction_operator_mode !=
                "identity_zero_load_fast_path" &&
            slab_reconstruction_operator_mode !=
                "constant_diffusion_fast_path" &&
            slab_reconstruction_operator_mode != "generic_variable_path")
        {
            throw_invalid_value(
                "--slab-reconstruction-operator-mode",
                "expected auto, identity_zero_load_fast_path, "
                "constant_diffusion_fast_path, or generic_variable_path.");
        }
        const std::string shared_context_validation =
            normalize_shared_context_validation_choice(
                options.shared_context_validation);
        if (shared_context_validation != "off" &&
            shared_context_validation != "sample" &&
            shared_context_validation != "full_debug")
        {
            throw_invalid_value(
                "--shared-context-validation",
                "expected off, sample, or full_debug.");
        }

        if (options.main_assembly_memory_budget_mb < 0.0)
        {
            throw_invalid_value(
                "--main-assembly-memory-budget-mb",
                "expected a nonnegative number.");
        }

        if (options.slab_reconstruction_memory_budget_mb < 0.0)
        {
            throw_invalid_value(
                "--slab-reconstruction-memory-budget-mb",
                "expected a nonnegative number.");
        }

        if (options.main_two_pass_numeric_fill_memory_budget_mb < 0.0)
        {
            throw_invalid_value(
                "--main-two-pass-numeric-fill-memory-budget-mb",
                "expected a nonnegative number.");
        }

        if (options.solver_diagnostics_max_export_dofs < 0)
        {
            throw_invalid_value(
                "--solver-diagnostics-max-export-dofs",
                "expected a nonnegative integer.");
        }

        const std::string output_profile =
            normalize_output_profile_choice(options.output_profile);
        if (output_profile != "minimal" &&
            output_profile != "production" &&
            output_profile != "benchmark" &&
            output_profile != "debug")
        {
            throw_invalid_value(
                "--output-profile",
                "supported values are: minimal, production, benchmark, debug.");
        }

        const std::string uniform_refinement_mode =
            normalize_uniform_refinement_mode_choice(
                options.uniform_refinement_mode);
        if (uniform_refinement_mode != "adaptive" &&
            uniform_refinement_mode != "uniform_x" &&
            uniform_refinement_mode != "uniform_y" &&
            uniform_refinement_mode != "uniform_xy")
        {
            throw_invalid_value(
                "--uniform-refinement-mode",
                "supported values are: adaptive, uniform_x, uniform_y, uniform_xy.");
        }

        const std::string timing_detail_level =
            normalize_timing_detail_level_choice(options.timing_detail_level);
        if (timing_detail_level != "summary" &&
            timing_detail_level != "detailed")
        {
            throw_invalid_value(
                "--timing-detail-level",
                "supported values are: summary, detailed.");
        }

        if (options.enable_timing_breakdown &&
            options.timing_history_filename.empty())
        {
            throw_invalid_value(
                "--timing-history-filename",
                "expected a nonempty filename when timing is enabled.");
        }

        try
        {
            (void)finite_element::time_slabs::parse_time_slab_backend(
                options.time_slab_backend);
        }
        catch (const std::exception& exc)
        {
            throw_invalid_value("--time-slab-backend", exc.what());
        }

    }

    [[nodiscard]] inline double effective_g_solver_tolerance(
        const RunnerOptions& options);

    [[nodiscard]] inline double effective_g_direct_solver_memory_limit_mb(
        const RunnerOptions& options);

    template<int P, class ExampleType>
    [[nodiscard]] inline AdaptiveParameters make_adaptive_parameters(
        const RunnerOptions& options,
        const ExampleType& example)
    {
        AdaptiveParameters parameters;
        parameters.rho = options.rho;
        parameters.max_outer_iterations = options.max_outer_iterations;
        parameters.max_inner_iterations = options.max_inner_iterations;
        parameters.doerfler_theta_y = options.doerfler_theta_y;
        parameters.doerfler_theta_x = options.doerfler_theta_x;
        parameters.uniform_x_refinement = options.uniform_x_refinement;
        parameters.uniform_y_refinement = options.uniform_y_refinement;
        parameters.force_accept_inner_with_effective_rho =
            options.force_accept_inner_with_effective_rho;
        parameters.compute_g_estimator = options.compute_g_estimator;
        parameters.compute_g_estimator_on_empty_y_marking_stop =
            options.compute_g_estimator_on_empty_y_marking_stop;
        parameters.compute_g_estimator_every_inner_iteration =
            options.compute_g_estimator_every_inner_iteration;
        parameters.g_solver = resolved_g_solver_name(options);
        parameters.g_solver_tolerance = effective_g_solver_tolerance(options);
        parameters.g_solver_memory_limit_mb =
            effective_g_direct_solver_memory_limit_mb(options);
        parameters.polynomial_degree = P;
        parameters.zero_tol = options.zero_tol;
        parameters.divergence_residual_l2_tolerance =
            options.divergence_residual_l2_tolerance;
        parameters.eta_squared_stop = options.eta_squared_stop;
        parameters.inner_estimator_squared_stop =
            options.inner_estimator_squared_stop;
        parameters.max_wall_time_seconds = options.max_wall_time_seconds;
        parameters.max_y_true_dofs = options.max_y_true_dofs;
        parameters.max_x_true_dofs = options.max_x_true_dofs;
        parameters.max_dofs_target =
            parse_dof_limit_target_choice(options.max_dofs_target);
        parameters.memory_limit_mb =
            options.memory_limit_explicit
                ? options.memory_limit_mb
                : options.main_solver_memory_limit_mb;
        parameters.memory_reserve_mb = options.memory_reserve_mb;
        parameters.memory_guard_safety_factor =
            options.memory_guard_safety_factor;
        parameters.memory_guard_near_cap_fraction =
            options.memory_guard_near_cap_fraction;
        parameters.check_divergence_residual =
            options.check_divergence_residual;
        parameters.stop_on_empty_y_marking =
            options.stop_on_empty_y_marking;
        parameters.local_time_slab_closure =
            options.local_time_slab_closure;
        parameters.use_adaptive_initial_guess =
            resolved_use_adaptive_initial_guess(options);
        parameters.solve_main_system_correction =
            options.solve_main_system_correction;
        parameters.fused_error_and_flux_diagnostics =
            options.fused_error_and_flux_diagnostics;
        parameters.local_error_reuse_patch_solve_workspace =
            options.local_error_reuse_patch_solve_workspace;
        parameters.deterministic_estimator_reductions =
            options.deterministic_estimator_reductions;
        parameters.doerfler_near_tie_tolerance =
            options.doerfler_near_tie_tolerance;
        parameters.refinement_edge_query_cache =
            options.refinement_edge_query_cache;
        parameters.refinement_batch_target_split_cells =
            options.refinement_batch_target_split_cells;
        parameters.post_flush_closure_mode =
            normalize_post_flush_closure_mode_choice(
                options.post_flush_closure_mode);
        parameters.post_flush_affected_containment_only =
            options.post_flush_affected_containment_only;
        parameters.refinement_full_conformity_check =
            options.refinement_full_conformity_check;
        parameters.refinement_main_closure_query_mode =
            normalize_refinement_main_closure_query_mode_choice(
                options.refinement_main_closure_query_mode);
        parameters.local_error_patch_tile_size =
            options.local_error_patch_tile_size > 0
                ? options.local_error_patch_tile_size
                : default_local_error_tile_size_for_degree(P);
        parameters.local_error_cell_chunk_size =
            options.local_error_cell_chunk_size > 0
                ? options.local_error_cell_chunk_size
                : default_local_error_cell_chunk_size_for_degree(P);
        parameters.local_error_max_threads =
            options.local_error_max_threads;
        parameters.local_error_memory_budget_mb =
            options.local_error_memory_budget_mb;
        parameters.local_error_worker_context_mode =
            normalize_local_error_worker_context_mode_choice(
                options.local_error_worker_context_mode);
        parameters.local_error_context_storage =
            normalize_local_error_context_storage_choice(
                options.local_error_context_storage);
        parameters.local_error_state_index_mode =
            normalize_local_error_state_index_mode_choice(
                options.local_error_state_index_mode);
        parameters.local_error_cell_state_cache_mode =
            normalize_local_error_cell_state_cache_mode_choice(
                options.local_error_cell_state_cache_mode);
        parameters.local_error_cell_state_cache_budget_mb =
            options.local_error_cell_state_cache_budget_mb;
        parameters.local_error_cell_state_representation =
            normalize_local_error_cell_state_representation_choice(
                options.local_error_cell_state_representation);
        parameters.local_error_flux_diagnostics_mode =
            normalize_local_error_flux_diagnostics_mode_choice(
                options.local_error_flux_diagnostics_mode);
        parameters.local_error_patch_solver =
            normalize_local_error_patch_solver_choice(
                options.local_error_patch_solver);
        parameters.local_error_coefficient_fast_path =
            options.local_error_coefficient_fast_path;
        parameters.local_error_compact_state_shadow =
            options.local_error_compact_state_shadow;
        parameters.shared_context_validation =
            normalize_shared_context_validation_choice(
                options.shared_context_validation);
        parameters.slab_reconstruction_operator_mode =
            normalize_slab_reconstruction_operator_mode_choice(
                options.slab_reconstruction_operator_mode);
        parameters.main_assembly_max_threads =
            options.main_assembly_max_threads;
        parameters.slab_reconstruction_max_threads =
            options.slab_reconstruction_max_threads;
        parameters.main_assembly_memory_budget_mb =
            options.main_assembly_memory_budget_mb;
        parameters.slab_reconstruction_memory_budget_mb =
            options.slab_reconstruction_memory_budget_mb;
        parameters.main_two_pass_numeric_fill_max_threads =
            options.main_two_pass_numeric_fill_max_threads > 0
                ? options.main_two_pass_numeric_fill_max_threads
                : default_two_pass_numeric_fill_threads_for_degree(P);
        parameters.main_two_pass_numeric_fill_memory_budget_mb =
            options.main_two_pass_numeric_fill_memory_budget_mb;
        parameters.time_slab_backend =
            finite_element::time_slabs::parse_time_slab_backend(
                options.time_slab_backend);
        parameters.allow_copied_time_slab_estimator_fallback =
            options.allow_copied_time_slab_estimator_fallback;
        parameters.virtual_backend_diagnostics =
            options.virtual_backend_diagnostics;
        parameters.main_system_export.enabled =
            options.solver_diagnostics_enabled;
        parameters.main_system_export.export_matrix_market =
            options.solver_diagnostics_export_matrix_market;
        parameters.main_system_export.export_rhs =
            options.solver_diagnostics_export_rhs;
        parameters.main_system_export.export_solution =
            options.solver_diagnostics_export_solution;
        parameters.main_system_export.max_export_dofs =
            options.solver_diagnostics_max_export_dofs;
        if (parameters.memory_limit_mb > 0.0)
        {
            if (parameters.local_error_memory_budget_mb <= 0.0)
                parameters.local_error_memory_budget_mb =
                    std::max(256.0, 0.25 * parameters.memory_limit_mb);
            if (parameters.main_assembly_memory_budget_mb <= 0.0)
                parameters.main_assembly_memory_budget_mb =
                    std::max(256.0, 0.20 * parameters.memory_limit_mb);
            if (parameters.slab_reconstruction_memory_budget_mb <= 0.0)
                parameters.slab_reconstruction_memory_budget_mb =
                    std::max(256.0, 0.25 * parameters.memory_limit_mb);
            if (parameters.main_two_pass_numeric_fill_memory_budget_mb <= 0.0)
                parameters.main_two_pass_numeric_fill_memory_budget_mb =
                    std::max(256.0, 0.20 * parameters.memory_limit_mb);
        }
        parameters.timing.enabled = options.enable_timing_breakdown;
        parameters.timing.detail_level =
            normalize_timing_detail_level_choice(options.timing_detail_level);
        parameters.output.print_iteration_tables = !options.quiet;
        parameters.output.export_history = options.export_history;
        parameters.output.save_estimator_components =
            options.save_estimator_components;
        parameters.output.save_refinement_history =
            options.save_refinement_history;
        parameters.output.save_mesh_statistics =
            options.save_mesh_statistics;
        parameters.output.save_iteration_snapshots =
            options.save_iteration_snapshots;
        parameters.output.save_snapshot_dofs =
            options.save_snapshot_dofs;
        parameters.output.timing_history_filename =
            options.timing_history_filename;
        parameters.output.output_directory =
            options.output_directory.empty()
                ? adaptive_algorithm::output::ensure_algorithm_output_dir(
                    example.effective_output_directory_name())
                : options.output_directory;

        return parameters;
    }

    [[nodiscard]] inline int quadrature_degree_boost(
        const RunnerOptions& options)
    {
        return options.increased_accuracy ? 2 : 0;
    }

    [[nodiscard]] inline double solver_accuracy_tolerance_factor(
        const RunnerOptions& options)
    {
        return options.increased_accuracy ? 0.1 : 1.0;
    }

    [[nodiscard]] inline double effective_main_solver_tolerance(
        const RunnerOptions& options)
    {
        return options.main_solver_tolerance *
            solver_accuracy_tolerance_factor(options);
    }

    [[nodiscard]] inline double
    effective_main_solver_direct_residual_retry_tolerance(
        const RunnerOptions& options)
    {
        return options.main_solver_direct_residual_retry_tolerance *
            solver_accuracy_tolerance_factor(options);
    }

    [[nodiscard]] inline double effective_g_solver_tolerance(
        const RunnerOptions& options)
    {
        const double requested =
            options.g_solver_tolerance > 0.0
                ? options.g_solver_tolerance
                : options.main_solver_tolerance;
        return requested * solver_accuracy_tolerance_factor(options);
    }

    [[nodiscard]] inline double effective_memory_limit_mb(
        const RunnerOptions& options)
    {
        return options.memory_limit_explicit
            ? options.memory_limit_mb
            : options.main_solver_memory_limit_mb;
    }

    [[nodiscard]] inline double effective_direct_solver_memory_limit_mb(
        const RunnerOptions& options)
    {
        const double global_limit = effective_memory_limit_mb(options);
        const double direct_limit = options.main_solver_memory_limit_mb;
        if (global_limit > 0.0 && direct_limit > 0.0)
            return std::min(global_limit, direct_limit);
        if (global_limit > 0.0)
            return global_limit;
        return direct_limit;
    }

    [[nodiscard]] inline double effective_g_direct_solver_memory_limit_mb(
        const RunnerOptions& options)
    {
        if (options.g_solver_memory_limit_mb > 0.0)
        {
            const double global_limit = effective_memory_limit_mb(options);
            const double g_limit = options.g_solver_memory_limit_mb;
            if (global_limit > 0.0)
                return std::min(global_limit, g_limit);
            return g_limit;
        }

        return effective_direct_solver_memory_limit_mb(options);
    }

    [[nodiscard]] inline la::concepts::SolverOptions make_main_solver_options(
        const RunnerOptions& options)
    {
        const auto* memory_mode_choice =
            find_pardiso_memory_mode_choice(
                options.main_solver_pardiso_memory_mode);
        if (memory_mode_choice == nullptr)
        {
            throw std::runtime_error(
                "Unsupported main_solver_pardiso_memory_mode '" +
                options.main_solver_pardiso_memory_mode +
                "'. Supported values are: " +
                supported_pardiso_memory_mode_choices_message() + '.');
        }

        if (main_solver_falls_back_to_sparse_lu(options.main_solver))
        {
            const std::string requested_solver =
                normalize_solver_choice(options.main_solver);
            std::cerr
                << "Warning: main_solver='" << requested_solver << "' was requested, but this build "
                   "does not include MKL Pardiso support. Falling back to sparse_lu.\n";
            auto solver_options = la::concepts::make_sparse_lu_solver_options();
            solver_options.tolerance = effective_main_solver_tolerance(options);
            solver_options.pardiso_memory_mode = memory_mode_choice->mode;
            solver_options.symmetry_tolerance =
                options.main_solver_symmetry_tolerance;
            solver_options.direct_residual_retry_mode =
                la::concepts::DirectResidualRetryMode::Disabled;
            solver_options.direct_residual_retry_tolerance =
                effective_main_solver_direct_residual_retry_tolerance(
                    options);
            solver_options.diagnostics_mode =
                parse_main_solver_diagnostics_mode(
                    options.main_solver_diagnostics);
            solver_options.direct_memory_limit_mb =
                effective_direct_solver_memory_limit_mb(options);
            solver_options.direct_memory_reserve_mb =
                options.memory_reserve_mb;
            solver_options.direct_memory_safety_factor =
                options.memory_guard_safety_factor;
            solver_options.pardiso_out_of_core_auto_switch =
                options.main_solver_ooc_auto_switch;
            solver_options.pardiso_out_of_core_switch_threshold =
                options.main_solver_ooc_switch_threshold;
            solver_options.pardiso_out_of_core_switch_to_lu =
                options.main_solver_ooc_switch_to_lu;
            solver_options.reuse_symbolic_analysis_when_pattern_unchanged =
                options.main_solver_reuse_symbolic_analysis;
            return solver_options;
        }

        la::concepts::SolverOptions solver_options;
        const auto* choice = find_main_solver_choice(options.main_solver);
        if (choice == nullptr)
        {
            throw std::runtime_error(
                "Unsupported main_solver '" + options.main_solver +
                "'. Supported values are: " +
                supported_main_solver_choices_message() + '.');
        }

        solver_options.solver = choice->solver;
        solver_options.preconditioner = choice->preconditioner;
        solver_options.pardiso_memory_mode = memory_mode_choice->mode;
        solver_options.max_iterations = options.main_solver_max_iterations;
        solver_options.tolerance = effective_main_solver_tolerance(options);
        solver_options.symmetry_tolerance =
            options.main_solver_symmetry_tolerance;
        solver_options.direct_residual_retry_mode =
            options.main_solver_direct_residual_retry
                ? la::concepts::DirectResidualRetryMode::
                      RetryWithSaferDirectSolver
                : la::concepts::DirectResidualRetryMode::Disabled;
        solver_options.direct_residual_retry_tolerance =
            effective_main_solver_direct_residual_retry_tolerance(options);
        solver_options.diagnostics_mode =
            parse_main_solver_diagnostics_mode(
                options.main_solver_diagnostics);
        solver_options.direct_memory_limit_mb =
            effective_direct_solver_memory_limit_mb(options);
        solver_options.direct_memory_reserve_mb =
            options.memory_reserve_mb;
        solver_options.direct_memory_safety_factor =
            options.memory_guard_safety_factor;
        solver_options.pardiso_out_of_core_auto_switch =
            options.main_solver_ooc_auto_switch;
        solver_options.pardiso_out_of_core_switch_threshold =
            options.main_solver_ooc_switch_threshold;
        solver_options.pardiso_out_of_core_switch_to_lu =
            options.main_solver_ooc_switch_to_lu;
        solver_options.reuse_symbolic_analysis_when_pattern_unchanged =
            options.main_solver_reuse_symbolic_analysis;

        if (const auto warning =
                la::concepts::pardiso_ldlt_memory_mode_warning(
                    solver_options.solver,
                    solver_options.pardiso_memory_mode);
            warning.has_value())
        {
            std::cerr << "Warning: " << *warning << '\n';
        }

        const auto validation =
            la::concepts::validate_solver_preconditioner(
                solver_options.solver,
                solver_options.preconditioner);
        if (!validation.accepted)
        {
            throw std::runtime_error(
                "Unsupported main_solver '" + options.main_solver +
                "'. " + validation.rejection_reason);
        }

        return solver_options;
    }

    [[nodiscard]] inline la::concepts::SolverOptions make_g_solver_options(
        const RunnerOptions& options)
    {
        RunnerOptions g_options = options;
        g_options.main_solver = effective_g_solver_choice(options);
        g_options.main_solver_tolerance =
            options.g_solver_tolerance > 0.0
                ? options.g_solver_tolerance
                : options.main_solver_tolerance;
        g_options.main_solver_memory_limit_mb =
            options.g_solver_memory_limit_mb > 0.0
                ? options.g_solver_memory_limit_mb
                : options.main_solver_memory_limit_mb;

        return make_main_solver_options(g_options);
    }

    [[nodiscard]] inline la::concepts::SolverOptions make_local_solver_options()
    {
        return la::concepts::make_sparse_lu_solver_options();
    }

    [[nodiscard]] inline const char* yaml_bool(bool value)
    {
        return value ? "true" : "false";
    }

    [[nodiscard]] inline std::string yaml_quote(std::string_view text)
    {
        std::string quoted = "\"";
        for (const char c : text)
        {
            switch (c)
            {
            case '\\':
                quoted += "\\\\";
                break;
            case '"':
                quoted += "\\\"";
                break;
            case '\n':
                quoted += "\\n";
                break;
            case '\r':
                quoted += "\\r";
                break;
            case '\t':
                quoted += "\\t";
                break;
            default:
                quoted += c;
                break;
            }
        }

        quoted += '"';
        return quoted;
    }

    inline void write_run_parameters_file(
        const RunnerOptions& options,
        const AdaptiveParameters& parameters,
        int q_space,
        int q_time)
    {
        if (!parameters.output.export_history ||
            parameters.output.output_directory.empty())
        {
            return;
        }

        std::filesystem::create_directories(parameters.output.output_directory);

        const auto output_path =
            parameters.output.output_directory / "run_parameters.yml";
        std::ofstream out(output_path);
        if (!out)
        {
            throw std::runtime_error(
                "write_run_parameters_file: failed to open '" +
                output_path.string() + "'.");
        }

        const std::string requested_main_solver =
            normalize_solver_choice(options.main_solver);
        const std::string effective_main_solver =
            resolved_main_solver_name(options.main_solver);
        const std::string pardiso_memory_mode =
            normalize_pardiso_memory_mode_choice(
                options.main_solver_pardiso_memory_mode);

        out << std::setprecision(17);
        out << "# Effective flat runner configuration written by "
               "run_adaptive_algorithm.\n";
        out << "# Re-run with: run_adaptive_algorithm --config "
            << output_path.filename().string() << "\n";
        if (options.config_file.has_value())
        {
            out << "# source_config_file: "
                << yaml_quote(options.config_file->string()) << "\n";
        }
        if (requested_main_solver != effective_main_solver)
        {
            out << "# requested_main_solver: "
                << yaml_quote(requested_main_solver) << "\n";
        }
        out << "\n";

        out << "example: " << yaml_quote(options.example_name) << "\n";
        out << "dimension: " << resolved_spatial_dimension(options) << "\n";
        out << "rho: " << parameters.rho << "\n";
        out << "max_outer_iterations: "
            << parameters.max_outer_iterations << "\n";
        out << "max_inner_iterations: "
            << parameters.max_inner_iterations << "\n";
        out << "doerfler_theta_x: "
            << parameters.doerfler_theta_x << "\n";
        out << "doerfler_theta_y: "
            << parameters.doerfler_theta_y << "\n";
        out << "uniform_x_refinement: "
            << yaml_bool(parameters.uniform_x_refinement) << "\n";
        out << "uniform_y_refinement: "
            << yaml_bool(parameters.uniform_y_refinement) << "\n";
        out << "force_accept_inner_with_effective_rho: "
            << yaml_bool(
                   parameters.force_accept_inner_with_effective_rho)
            << "\n";
        out << "effective_rho_only_inner_acceptance: "
            << yaml_bool(
                   parameters.force_accept_inner_with_effective_rho)
            << "\n";
        out << "compute_g_estimator: "
            << yaml_bool(parameters.compute_g_estimator) << "\n";
        out << "compute_g_estimator_on_empty_y_marking_stop: "
            << yaml_bool(
                   parameters.compute_g_estimator_on_empty_y_marking_stop)
            << "\n";
        out << "compute_g_estimator_every_inner_iteration: "
            << yaml_bool(
                   parameters.compute_g_estimator_every_inner_iteration)
            << "\n";
        out << "g_solver: "
            << yaml_quote(options.g_solver) << "\n";
        out << "# effective_g_solver: "
            << yaml_quote(parameters.g_solver) << "\n";
        out << "g_solver_tolerance: "
            << options.g_solver_tolerance << "\n";
        out << "# effective_g_solver_tolerance: "
            << parameters.g_solver_tolerance << "\n";
        out << "g_solver_memory_limit_mb: "
            << options.g_solver_memory_limit_mb << "\n";
        out << "# effective_g_solver_memory_limit_mb: "
            << parameters.g_solver_memory_limit_mb << "\n";
        out << "uniform_refinement_mode: "
            << yaml_quote(
                   effective_uniform_refinement_mode(
                       parameters.uniform_x_refinement,
                       parameters.uniform_y_refinement))
            << "\n";
        out << "polynomial_degree: "
            << parameters.polynomial_degree << "\n";
        out << "zero_tol: " << parameters.zero_tol << "\n";
        out << "divergence_residual_l2_tolerance: "
            << parameters.divergence_residual_l2_tolerance << "\n";
        out << "eta_squared_stop: "
            << parameters.eta_squared_stop << "\n";
        out << "inner_estimator_squared_stop: "
            << parameters.inner_estimator_squared_stop << "\n";
        out << "max_wall_time_seconds: "
            << parameters.max_wall_time_seconds << "\n";
        out << "max_y_true_dofs: "
            << parameters.max_y_true_dofs << "\n";
        out << "max_x_true_dofs: "
            << parameters.max_x_true_dofs << "\n";
        out << "max_dofs_target: "
            << dof_limit_target_name(parameters.max_dofs_target) << "\n";
        out << "memory_limit_mb: "
            << parameters.memory_limit_mb << "\n";
        out << "# memory_limit_source: "
            << (options.memory_limit_explicit
                    ? "explicit"
                    : "main_solver_memory_limit_mb")
            << "\n";
        out << "memory_reserve_mb: "
            << parameters.memory_reserve_mb << "\n";
        out << "memory_guard_safety_factor: "
            << parameters.memory_guard_safety_factor << "\n";
        out << "memory_guard_near_cap_fraction: "
            << parameters.memory_guard_near_cap_fraction << "\n";
        out << "increased_accuracy: "
            << yaml_bool(options.increased_accuracy) << "\n";
        out << "# quadrature_degree_boost: "
            << quadrature_degree_boost(options) << "\n";
        out << "# effective_q_space: " << q_space << "\n";
        out << "# effective_q_time: " << q_time << "\n";
        out << "main_solver: "
            << yaml_quote(effective_main_solver) << "\n";
        out << "main_solver_pardiso_memory_mode: "
            << yaml_quote(pardiso_memory_mode) << "\n";
        out << "main_solver_max_iterations: "
            << options.main_solver_max_iterations << "\n";
        out << "main_solver_tolerance: "
            << options.main_solver_tolerance << "\n";
        out << "# effective_main_solver_tolerance: "
            << effective_main_solver_tolerance(options) << "\n";
        out << "main_solver_symmetry_tolerance: "
            << options.main_solver_symmetry_tolerance << "\n";
        out << "main_solver_direct_residual_retry: "
            << yaml_bool(options.main_solver_direct_residual_retry) << "\n";
        out << "main_solver_direct_residual_retry_tolerance: "
            << options.main_solver_direct_residual_retry_tolerance << "\n";
        out << "# effective_main_solver_direct_residual_retry_tolerance: "
            << effective_main_solver_direct_residual_retry_tolerance(options)
            << "\n";
        out << "main_solver_diagnostics: "
            << yaml_quote(
                   normalize_main_solver_diagnostics_choice(
                       options.main_solver_diagnostics))
            << "\n";
        out << "main_solver_memory_limit_mb: "
            << options.main_solver_memory_limit_mb << "\n";
        out << "# effective_direct_solver_memory_limit_mb: "
            << effective_direct_solver_memory_limit_mb(options) << "\n";
        out << "main_solver_ooc_auto_switch: "
            << yaml_bool(options.main_solver_ooc_auto_switch) << "\n";
        out << "main_solver_ooc_switch_threshold: "
            << options.main_solver_ooc_switch_threshold << "\n";
        out << "main_solver_ooc_switch_to_lu: "
            << yaml_bool(options.main_solver_ooc_switch_to_lu) << "\n";
        out << "main_solver_reuse_symbolic_analysis: "
            << yaml_bool(options.main_solver_reuse_symbolic_analysis) << "\n";
        out << "output: "
            << yaml_quote(parameters.output.output_directory.string()) << "\n";
        out << "quiet: " << yaml_bool(options.quiet) << "\n";
        out << "check_divergence_residual: "
            << yaml_bool(parameters.check_divergence_residual) << "\n";
        out << "stop_on_empty_y_marking: "
            << yaml_bool(parameters.stop_on_empty_y_marking) << "\n";
        out << "local_time_slab_closure: "
            << yaml_bool(parameters.local_time_slab_closure) << "\n";
        out << "use_adaptive_initial_guess: "
            << yaml_bool(parameters.use_adaptive_initial_guess) << "\n";
        out << "solve_main_system_correction: "
            << yaml_bool(parameters.solve_main_system_correction) << "\n";
        out << "fused_error_and_flux_diagnostics: "
            << yaml_bool(parameters.fused_error_and_flux_diagnostics) << "\n";
        out << "local_error_reuse_patch_solve_workspace: "
            << yaml_bool(parameters.local_error_reuse_patch_solve_workspace)
            << "\n";
        out << "refinement_edge_query_cache: "
            << yaml_bool(parameters.refinement_edge_query_cache) << "\n";
        out << "refinement_batch_target_split_cells: "
            << parameters.refinement_batch_target_split_cells << "\n";
        out << "post_flush_closure_mode: "
            << yaml_quote(parameters.post_flush_closure_mode) << "\n";
        out << "post_flush_affected_containment_only: "
            << yaml_bool(parameters.post_flush_affected_containment_only)
            << "\n";
        out << "refinement_full_conformity_check: "
            << yaml_bool(parameters.refinement_full_conformity_check)
            << "\n";
        out << "refinement_main_closure_query_mode: "
            << yaml_quote(parameters.refinement_main_closure_query_mode)
            << "\n";
        out << "local_error_patch_tile_size: "
            << parameters.local_error_patch_tile_size << "\n";
        out << "local_error_cell_chunk_size: "
            << parameters.local_error_cell_chunk_size << "\n";
        out << "local_error_max_threads: "
            << parameters.local_error_max_threads << "\n";
        out << "local_error_memory_budget_mb: "
            << parameters.local_error_memory_budget_mb << "\n";
        out << "local_error_worker_context_mode: "
            << yaml_quote(parameters.local_error_worker_context_mode)
            << "\n";
        out << "local_error_context_storage: "
            << yaml_quote(parameters.local_error_context_storage)
            << "\n";
        out << "effective_local_error_context_storage: "
            << yaml_quote(parameters.local_error_context_storage)
            << "\n";
        out << "local_error_state_index_mode: "
            << yaml_quote(parameters.local_error_state_index_mode)
            << "\n";
        out << "effective_local_error_state_index_mode: "
            << yaml_quote(parameters.local_error_state_index_mode)
            << "\n";
        out << "local_error_cell_state_cache_mode: "
            << yaml_quote(parameters.local_error_cell_state_cache_mode)
            << "\n";
        out << "effective_local_error_cell_state_cache_mode: "
            << yaml_quote(parameters.local_error_cell_state_cache_mode)
            << "\n";
        out << "local_error_cell_state_cache_budget_mb: "
            << parameters.local_error_cell_state_cache_budget_mb << "\n";
        out << "local_error_cell_state_representation: "
            << yaml_quote(parameters.local_error_cell_state_representation)
            << "\n";
        out << "effective_local_error_cell_state_representation: "
            << yaml_quote(parameters.local_error_cell_state_representation)
            << "\n";
        out << "local_error_flux_diagnostics_mode: "
            << yaml_quote(parameters.local_error_flux_diagnostics_mode)
            << "\n";
        out << "effective_local_error_flux_diagnostics_mode: "
            << yaml_quote(parameters.local_error_flux_diagnostics_mode)
            << "\n";
        out << "local_error_patch_solver: "
            << yaml_quote(parameters.local_error_patch_solver)
            << "\n";
        out << "effective_local_error_patch_solver: "
            << yaml_quote(parameters.local_error_patch_solver)
            << "\n";
        out << "local_error_coefficient_fast_path: "
            << yaml_bool(parameters.local_error_coefficient_fast_path)
            << "\n";
        out << "local_error_compact_state_shadow: "
            << yaml_bool(parameters.local_error_compact_state_shadow)
            << "\n";
        out << "shared_context_validation: "
            << yaml_quote(parameters.shared_context_validation)
            << "\n";
        out << "slab_reconstruction_operator_mode: "
            << yaml_quote(parameters.slab_reconstruction_operator_mode)
            << "\n";
        out << "effective_slab_reconstruction_operator_mode: "
            << yaml_quote(parameters.slab_reconstruction_operator_mode)
            << "\n";
        out << "deterministic_estimator_reductions: "
            << yaml_bool(parameters.deterministic_estimator_reductions)
            << "\n";
        out << "doerfler_near_tie_tolerance: "
            << parameters.doerfler_near_tie_tolerance << "\n";
        out << "main_assembly_max_threads: "
            << parameters.main_assembly_max_threads << "\n";
        out << "slab_reconstruction_max_threads: "
            << parameters.slab_reconstruction_max_threads << "\n";
        out << "main_assembly_memory_budget_mb: "
            << parameters.main_assembly_memory_budget_mb << "\n";
        out << "slab_reconstruction_memory_budget_mb: "
            << parameters.slab_reconstruction_memory_budget_mb << "\n";
        out << "main_two_pass_numeric_fill_max_threads: "
            << parameters.main_two_pass_numeric_fill_max_threads << "\n";
        out << "main_two_pass_numeric_fill_memory_budget_mb: "
            << parameters.main_two_pass_numeric_fill_memory_budget_mb << "\n";
        out << "time_slab_backend: "
            << yaml_quote(
                   finite_element::time_slabs::time_slab_backend_name(
                       parameters.time_slab_backend))
            << "\n";
        out << "allow_copied_time_slab_estimator_fallback: "
            << yaml_bool(
                   parameters.allow_copied_time_slab_estimator_fallback)
            << "\n";
        out << "virtual_backend_diagnostics: "
            << yaml_bool(parameters.virtual_backend_diagnostics) << "\n";
        out << "solver_diagnostics_enabled: "
            << yaml_bool(parameters.main_system_export.enabled) << "\n";
        out << "solver_diagnostics_export_matrix_market: "
            << yaml_bool(
                   parameters.main_system_export.export_matrix_market)
            << "\n";
        out << "solver_diagnostics_export_rhs: "
            << yaml_bool(parameters.main_system_export.export_rhs)
            << "\n";
        out << "solver_diagnostics_export_solution: "
            << yaml_bool(parameters.main_system_export.export_solution)
            << "\n";
        out << "solver_diagnostics_max_export_dofs: "
            << parameters.main_system_export.max_export_dofs << "\n";
        out << "output_profile: "
            << yaml_quote(normalize_output_profile_choice(options.output_profile))
            << "\n";
        out << "save_heavy_diagnostics: "
            << yaml_bool(options.save_heavy_diagnostics) << "\n";
        out << "export_history: "
            << yaml_bool(parameters.output.export_history) << "\n";
        out << "save_estimator_components: "
            << yaml_bool(parameters.output.save_estimator_components) << "\n";
        out << "save_refinement_history: "
            << yaml_bool(parameters.output.save_refinement_history) << "\n";
        out << "save_mesh_statistics: "
            << yaml_bool(parameters.output.save_mesh_statistics) << "\n";
        out << "save_iteration_snapshots: "
            << yaml_bool(parameters.output.save_iteration_snapshots) << "\n";
        out << "save_snapshot_dofs: "
            << yaml_bool(parameters.output.save_snapshot_dofs) << "\n";
        out << "enable_timing_breakdown: "
            << yaml_bool(parameters.timing.enabled) << "\n";
        out << "timing_history_filename: "
            << yaml_quote(parameters.output.timing_history_filename) << "\n";
        out << "timing_detail_level: "
            << yaml_quote(parameters.timing.detail_level) << "\n";
    }

    inline int dispatch_by_polynomial_degree(
        const RunnerOptions& options,
        const char* argv0)
    {
        const int dim_space = resolved_spatial_dimension(options);
        const int quadrature_boost = quadrature_degree_boost(options);

        switch (options.polynomial_degree)
        {
        case 1:
            if (dim_space == 1)
                return run_degree_1_dim_1(options, quadrature_boost);
            if (dim_space == 2)
                return run_degree_1_dim_2(options, quadrature_boost);
            break;
        case 2:
            if (dim_space == 1)
                return run_degree_2_dim_1(options, quadrature_boost);
            if (dim_space == 2)
                return run_degree_2_dim_2(options, quadrature_boost);
            break;
        case 3:
            if (dim_space == 1)
                return run_degree_3_dim_1(options, quadrature_boost);
            if (dim_space == 2)
                return run_degree_3_dim_2(options, quadrature_boost);
            break;
        case 4:
            if (dim_space == 1)
                return run_degree_4_dim_1(options, quadrature_boost);
            if (dim_space == 2)
                return run_degree_4_dim_2(options, quadrature_boost);
            break;
        default:
            throw std::runtime_error(
                usage_text(argv0) +
                "\nUnsupported polynomial degree. Supported values are 1, 2, 3, and 4.");
        }

        throw std::runtime_error(
            "Unsupported example spatial dimension " +
            std::to_string(dim_space) + ".");
    }

    inline int run_from_command_line(int argc, char** argv)
    {
        const auto command_line_overrides =
            parse_command_line_overrides(argc, argv);

        if (command_line_overrides.show_help.value_or(false))
        {
            std::cout << usage_text(argv[0]);
            return 0;
        }

        if (command_line_overrides.list_examples.value_or(false))
        {
            print_available_examples();
            return 0;
        }

        RunnerOptions options;
        if (command_line_overrides.config_file.has_value())
        {
            const auto file_overrides =
                parse_config_file(*command_line_overrides.config_file, argv[0]);
            apply_overrides(options, file_overrides);
        }

        apply_overrides(options, command_line_overrides);

        if (options.show_help)
        {
            std::cout << usage_text(argv[0]);
            return 0;
        }

        if (options.list_examples)
        {
            print_available_examples();
            return 0;
        }

        validate_runner_options(options, argv[0]);
        return dispatch_by_polynomial_degree(options, argv[0]);
    }
}

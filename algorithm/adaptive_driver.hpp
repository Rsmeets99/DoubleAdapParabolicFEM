#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "adaptive_parameters.hpp"
#include "adaptive_result.hpp"
#include "detail/indicator_utilities.hpp"
#include "output/history_writer.hpp"
#include "output/iteration_snapshot_writer.hpp"
#include "output/logger.hpp"

#include "finite_element/detail/space_time_capabilities.hpp"
#include "finite_element/detail/memory_usage.hpp"
#include "finite_element/assembly/main_system/mat_A.hpp"
#include "finite_element/assembly/main_system/mat_B.hpp"
#include "finite_element/assembly/main_system/vec_f.hpp"
#include "finite_element/fespace/fe_traits.hpp"
#include "finite_element/fespace/prolongation.hpp"
#include "finite_element/system/solve_main_system.hpp"
#include "finite_element/time_slabs/time_slab_adaptive_loop.hpp"
#include "finite_element/time_slabs/time_slab_adaptive_refinement.hpp"

#include "linear_algebra/concepts/solver.hpp"
#include "linear_algebra/operations/sparse_matrix_ops.hpp"
#include "linear_algebra/operations/vector_ops.hpp"
#include "mesh/refinement/refinement_type.hpp"

namespace adaptive_algorithm
{
    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class YSpaceType,
        class ProblemDataType>
    class AdaptiveDriver
    {
    public:
        using Vector = typename Backend::Vector;
        using Solver = typename Backend::Solver;
        using XFunction = finite_element::Function<XSpaceType, Vector>;
        using YFunction = finite_element::Function<YSpaceType, Vector>;
        using ResultType = AdaptiveResult<Backend, XSpaceType, YSpaceType>;
        using OuterRecord = AdaptiveOuterIterationRecord<int>;
        using InnerRecord = AdaptiveYIterationRecord<int>;
        using Clock = std::chrono::steady_clock;
        using YIterationState =
            finite_element::time_slabs::AdaptiveTestSpaceIterationState<
                Backend,
                XSpaceType,
                YSpaceType>;
        using YIterationParameters =
            finite_element::time_slabs::AdaptiveTestSpaceIterationParameters<
                Backend,
                XSpaceType,
                YSpaceType>;
        using YIterationSolutionSnapshot =
            finite_element::time_slabs::AdaptiveTestSpaceSolutionSnapshot<
                Backend,
                XSpaceType,
                YSpaceType>;
        using MainTwoPassAssemblyCache =
            finite_element::assembly::TwoPassFullSaddleAssemblyCache2D<
                Backend,
                XSpaceType,
                YSpaceType>;
        AdaptiveDriver(
            XSpaceType& x_space,
            YSpaceType& y_space,
            const ProblemDataType& problem,
            Solver& solver,
            la::concepts::SolverOptions main_solver_options = {},
            AdaptiveParameters parameters = {},
            std::optional<la::concepts::SolverOptions> g_solver_options =
                std::nullopt)
            : x_space_(x_space),
              y_space_(y_space),
              problem_(problem),
              solver_(solver),
              main_solver_options_(std::move(main_solver_options)),
              g_solver_options_(
                  g_solver_options.has_value()
                      ? std::move(*g_solver_options)
                      : main_solver_options_),
              local_solver_options_(la::concepts::make_sparse_lu_solver_options()),
              parameters_(std::move(parameters)),
              timing_(parameters_.timing)
        {}

        [[nodiscard]] ResultType run()
        {
            validate_configuration_();
            timing_.set_enabled(parameters_.timing.enabled);
            timing_.reset();
            [[maybe_unused]] const SpaceTimingHookScope space_timing_hooks(*this);

            ResultType result;
            result.problem_name = problem_.name;
            result.problem_description = problem_.description;
            result.spatial_dimension = XSpaceType::GT::dim_space_v;
            result.polynomial_degree = XSpaceType::FETraitsType::p_space_v;
            result.compute_g_estimator = parameters_.compute_g_estimator;
            result.compute_g_estimator_on_empty_y_marking_stop =
                parameters_.compute_g_estimator_on_empty_y_marking_stop;
            result.compute_g_estimator_every_inner_iteration =
                parameters_.compute_g_estimator_every_inner_iteration;
            result.g_solver = parameters_.g_solver;
            result.g_solver_tolerance = parameters_.g_solver_tolerance;
            result.g_solver_memory_limit_mb =
                parameters_.g_solver_memory_limit_mb;
            AlgorithmLogger logger(parameters_.output);
            record_output_profile_checks_();
            std::optional<OuterRecord> pending_outer_record{};
            std::optional<Clock::time_point> pending_outer_start{};
            const auto run_start = Clock::now();
            active_run_start_ = run_start;

            try
            {
                if (parameters_.max_outer_iterations == 0)
                {
                    result.terminated_early = true;
                    result.termination_reason = "max_outer_iterations == 0";
                    finalize_output_(result, logger);
                    return result;
                }

                std::optional<YIterationSolutionSnapshot> previous_solution_snapshot{};

                for (int outer_iteration = 0;
                     outer_iteration < parameters_.max_outer_iterations;
                     ++outer_iteration)
                {
                    validate_current_spaces_();

                    pending_outer_record.emplace();
                    pending_outer_start = Clock::now();
                    auto& outer_record = *pending_outer_record;
                    outer_record.outer_iteration = outer_iteration;
                    outer_record.uniform_x_refinement =
                        parameters_.uniform_x_refinement;
                    outer_record.uniform_y_refinement =
                        parameters_.uniform_y_refinement;
                    outer_record.x_refinement_mode =
                        parameters_.uniform_x_refinement
                            ? "uniform"
                            : "adaptive";
                    update_outer_counts_before_(outer_record);
                    log_outer_iteration_start_(logger, outer_record);
                    append_partial_outer_history_(
                        outer_record,
                        *pending_outer_start,
                        run_start,
                        "started");

                    throw_if_memory_budget_exceeded_(
                        "outer_iteration_start");

                    if (x_true_dof_limit_exceeded_(outer_record.n_x_true_dofs_before))
                    {
                        update_current_outer_counts_after_(outer_record);
                        stamp_outer_timing_(
                            outer_record,
                            *pending_outer_start,
                            run_start);
                        append_partial_outer_history_(
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            "failed");
                        result.terminated_early = true;
                        result.termination_reason =
                            make_x_true_dof_limit_reason_(
                                outer_record.n_x_true_dofs_before,
                                "before starting the inner Y loop");
                        log_outer_iteration_summary_(logger, outer_record);
                        result.outer_iterations.push_back(std::move(outer_record));
                        pending_outer_record.reset();
                        finalize_output_(result, logger);
                        return result;
                    }

                    if (y_true_dof_limit_exceeded_(outer_record.n_y_true_dofs_before))
                    {
                        update_current_outer_counts_after_(outer_record);
                        stamp_outer_timing_(
                            outer_record,
                            *pending_outer_start,
                            run_start);
                        append_partial_outer_history_(
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            "failed");
                        result.terminated_early = true;
                        result.termination_reason =
                            make_y_true_dof_limit_reason_(
                                outer_record.n_y_true_dofs_before,
                                "before starting the inner Y loop");
                        log_outer_iteration_summary_(logger, outer_record);
                        result.outer_iterations.push_back(std::move(outer_record));
                        pending_outer_record.reset();
                        finalize_output_(result, logger);
                        return result;
                    }

                    std::optional<YIterationState> last_state{};
                    std::optional<XMarkingIndicatorComponents<int>> last_x_indicators{};
                    bool time_budget_reached_during_inner_loop = false;

                    for (int inner_iteration = 0;
                         inner_iteration < parameters_.max_inner_iterations;
                         ++inner_iteration)
                    {
                        if (!outer_record.inner_iterations.empty() &&
                            wall_time_budget_reached_(run_start))
                        {
                            time_budget_reached_during_inner_loop = true;
                            break;
                        }

                        const auto inner_start = Clock::now();
                        const auto before_main_solve_bookkeeping_start =
                            inner_start;

                        // For fixed X^delta, every inner iteration resolves the mixed
                        // problem on the current Y^delta before deciding whether the
                        // posteriori control of ||lambda_delta - lambda^delta||_Y is
                        // already sharp enough to move on to X-marking.
                        InnerRecord inner_record;
                        inner_record.inner_iteration = inner_iteration;
                        inner_record.uniform_y_refinement =
                            parameters_.uniform_y_refinement;
                        inner_record.force_accept_inner_with_effective_rho =
                            parameters_.force_accept_inner_with_effective_rho;
                        inner_record
                            .configured_rho_ignored_for_inner_acceptance =
                            parameters_.force_accept_inner_with_effective_rho;
                        inner_record.effective_rho_acceptance_reason =
                            parameters_.force_accept_inner_with_effective_rho
                                ? "not_at_max_inner_iterations"
                                : "not_used";
                        inner_record.g_estimator_enabled =
                            parameters_.compute_g_estimator;
                        inner_record.g_estimator_skipped_reason =
                            parameters_.compute_g_estimator
                                ? "inner_iteration_not_accepted"
                                : "not_requested";
                        inner_record.g_solver_status =
                            parameters_.compute_g_estimator
                                ? "inner_iteration_not_accepted"
                                : "not_requested";
                        inner_record.y_refinement_mode =
                            parameters_.uniform_y_refinement
                                ? "uniform"
                                : "adaptive";
                        inner_record.n_y_active_cells_before =
                            static_cast<int>(y_space_.active_cells().size());
                        inner_record.n_y_true_dofs_before =
                            y_space_.dof_handler_ref().n_true_dofs();
                        inner_record.y_generation_before =
                            active_generation_stats_(y_space_);
                        inner_record.refinement_statistics =
                            zero_refinement_statistics_();
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "start_inner_y_loop");

                        throw_if_memory_budget_exceeded_(
                            "inner_solve_start");

                        auto y_iteration_parameters =
                            make_y_iteration_parameters_();
                        inner_record.time_slab_backend_requested =
                            finite_element::time_slabs::time_slab_backend_name(
                                y_iteration_parameters.time_slab_backend);
                        inner_record.copied_estimator_fallback_enabled =
                            false;
                        inner_record.strict_virtual_estimator = false;
                        inner_record.strict_virtual_estimator_status =
                            "not_requested";
                        configure_main_system_export_(
                            y_iteration_parameters.main_system_export,
                            outer_iteration,
                            inner_iteration);
                        auto main_solver_options =
                            memory_adjusted_main_solver_options_();
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "before_main_solve");
                        record_since_(
                            "adaptive_driver.before_main_solve_bookkeeping_wall",
                            before_main_solve_bookkeeping_start);

                        auto state = [&]()
                        {
                            auto timer =
                                timing_.scoped("adaptive.inner_iteration_core");
                            return finite_element::time_slabs::
                                compute_adaptive_test_space_iteration<
                                    QSpace,
                                    QTime,
                                    Backend>(
                                        y_space_,
                                        x_space_,
                                        problem_,
                                        solver_,
                                        main_solver_options,
                                        y_iteration_parameters,
                                        local_solver_options_,
                                        previous_solution_snapshot.has_value()
                                            ? &*previous_solution_snapshot
                                            : nullptr);
                        }();

                        const auto after_main_solve_bookkeeping_start =
                            Clock::now();
                        bool after_main_solve_bookkeeping_recorded = false;
                        const auto record_after_main_solve_bookkeeping =
                            [&]()
                            {
                                if (after_main_solve_bookkeeping_recorded)
                                    return;
                                record_since_(
                                    "adaptive_driver."
                                    "after_main_solve_bookkeeping_wall",
                                    after_main_solve_bookkeeping_start);
                                after_main_solve_bookkeeping_recorded = true;
                            };

                        record_memory_sample_("inner_solve_finished");

                        auto x_indicators = [&]()
                        {
                            auto timer =
                                timing_.scoped("x.marking_indicator_components");
                            return detail::compute_x_marking_indicator_components<
                                QSpace,
                                QTime>(
                                    x_space_,
                                    state.lambda_delta,
                                    state.u_delta,
                                    problem_.u0,
                                    problem_.M);
                        }();

                        inner_record.n_slabs = state.n_slabs();
                        inner_record.n_patches = state.n_patches();
                        inner_record.copied_slab_cells =
                            state.copied_slab_cells();
                        inner_record.virtual_slab_cells =
                            state.virtual_slab_cells();
                        inner_record.time_slab_backend =
                            state.time_slab_backend();
                        inner_record.estimator_backend =
                            state.estimator_backend();
                        inner_record.time_slab_backend_requested =
                            finite_element::time_slabs::time_slab_backend_name(
                                y_iteration_parameters.time_slab_backend);
                        inner_record.time_slab_backend_effective =
                            state.time_slab_backend_effective();
                        inner_record.virtual_overlay_constructed =
                            state.virtual_overlay_constructed();
                        inner_record.tilde_y_space_constructed =
                            state.tilde_y_space_constructed();
                        inner_record.copied_estimator_fallback_enabled =
                            false;
                        inner_record.estimator_uses_copied_fallback =
                            state.estimator_uses_copied_fallback();
                        inner_record.copied_estimator_fallback_used =
                            state.estimator_uses_copied_fallback();
                        inner_record.estimator_fallback_copied_slab_cells =
                            state.estimator_fallback_copied_slab_cells();
                        inner_record.copied_fallback_component_count =
                            state.copied_fallback_component_count();
                        inner_record.copied_fallback_components =
                            state.copied_fallback_components();
                        inner_record.strict_virtual_estimator =
                            false;
                        inner_record.strict_virtual_estimator_status =
                            "not_requested";
                        inner_record.copied_slab_cells_constructed_total =
                            state.copied_slab_cells_constructed_total();
                        inner_record
                            .copied_slab_cells_constructed_for_fallback =
                            state.estimator_fallback_copied_slab_cells();
                        inner_record.source_mesh_mutation_count =
                            state.source_mesh_mutation_count;
                        inner_record.lambda_y_squared =
                            x_indicators.lambda_y_squared_total();
                        inner_record.initial_trace_squared =
                            x_indicators.initial_trace_squared_total();
                        inner_record.eta_squared =
                            x_indicators.eta_squared_total();
                        inner_record.y_estimator_squared = state.estimator.total();
                        inner_record.y_flux_squared =
                            state.estimator.equilibrated_flux_y_squared.total();
                        inner_record.y_reconstruction_squared =
                            state.estimator.reconstruction_y_squared.total();
                        inner_record.divergence_residual_squared =
                            state.estimator.divergence_residual_total();
                        inner_record.divergence_residual_l2 =
                            state.divergence_residual_l2;
                        inner_record.y_estimator_threshold_squared =
                            detail::compute_y_estimator_threshold_squared(
                                parameters_.rho,
                                inner_record.lambda_y_squared,
                                inner_record.initial_trace_squared);
                        inner_record.configured_rho = parameters_.rho;
                        inner_record
                            .y_estimator_threshold_configured_rho_squared =
                            inner_record.y_estimator_threshold_squared;
                        inner_record.main_solve =
                            make_main_solve_diagnostics_record(
                                state.main_solve_diagnostics,
                                state.main_solve_setup_seconds,
                                state.main_solve_solve_seconds);
                        const bool configured_rho_acceptance_satisfied =
                            inner_record.y_estimator_squared <=
                            inner_record.y_estimator_threshold_squared;
                        const bool explicit_inner_stop_satisfied =
                            parameters_.inner_estimator_squared_stop > 0.0 &&
                             inner_record.y_estimator_squared <=
                                 parameters_.inner_estimator_squared_stop;
                        inner_record.stopping_criterion_satisfied =
                            (!parameters_
                                  .force_accept_inner_with_effective_rho &&
                             configured_rho_acceptance_satisfied) ||
                            explicit_inner_stop_satisfied;
                        const bool force_accept_on_last_inner =
                            parameters_.force_accept_inner_with_effective_rho &&
                            !inner_record.stopping_criterion_satisfied &&
                            inner_iteration + 1 >=
                                parameters_.max_inner_iterations;
                        if (force_accept_on_last_inner)
                        {
                            inner_record.stopping_criterion_satisfied = true;
                            inner_record.effective_rho_acceptance_used = true;
                            inner_record.effective_rho_acceptance_reason =
                                "accepted_by_forced_effective_rho_at_max_inner";
                        }
                        populate_effective_rho_(inner_record);
                        inner_record.marked_y_cells = state.marked_source_cells;
                        inner_record.estimator = state.estimator;
                        inner_record.n_y_active_cells_after =
                            inner_record.n_y_active_cells_before;
                        inner_record.n_y_true_dofs_after =
                            inner_record.n_y_true_dofs_before;
                        inner_record.y_generation_after =
                            inner_record.y_generation_before;
                        if (parameters_.compute_g_estimator &&
                            parameters_
                                .compute_g_estimator_every_inner_iteration)
                        {
                            compute_g_estimator_for_accepted_inner_iteration_(
                                inner_record,
                                state.lambda_delta,
                                state.u_delta,
                                true);
                        }
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "after_x_solve");
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "after_y_solve");
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "after_main_solve");
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "after_time_slab_reconstruction");
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "after_y_estimator_marking");

                        {
                            auto timer =
                                timing_.scoped("output.inner_iteration_snapshot");
                            measure_output_history_phase_(
                                "adaptive_driver.snapshot_output_checks",
                                [&]()
                                {
                                    output::write_inner_iteration_snapshot(
                                        outer_iteration,
                                        inner_iteration,
                                        x_space_,
                                        y_space_,
                                        x_indicators,
                                        state,
                                        parameters_.output);
                                });
                        }

                        last_state.emplace(std::move(state));
                        last_x_indicators.emplace(std::move(x_indicators));

                        if (inner_record.stopping_criterion_satisfied)
                        {
                            outer_record.y_converged = true;
                            if (!inner_record.g_estimator_computed)
                            {
                                compute_g_estimator_for_accepted_inner_iteration_(
                                    inner_record,
                                    last_state->lambda_delta,
                                    last_state->u_delta);
                            }
                            stamp_inner_timing_(inner_record, inner_start, run_start);
                            append_partial_inner_history_(
                                outer_iteration,
                                inner_record,
                                inner_start,
                                run_start,
                                "completed");
                            push_inner_record_(
                                outer_record,
                                std::move(inner_record));
                            log_inner_iteration_(
                                logger,
                                outer_iteration,
                                outer_record.last_inner_iteration());
                            record_after_main_solve_bookkeeping();
                            break;
                        }

                        const bool stop_on_empty_y_marking =
                            parameters_.stop_on_empty_y_marking &&
                            !parameters_.uniform_y_refinement &&
                            inner_record.marked_y_cells.empty();

                        if (stop_on_empty_y_marking)
                        {
                            outer_record.stopped_on_empty_y_marking = true;
                            if (parameters_.compute_g_estimator &&
                                parameters_
                                    .compute_g_estimator_on_empty_y_marking_stop)
                            {
                                if (!inner_record.g_estimator_computed)
                                {
                                    compute_g_estimator_for_accepted_inner_iteration_(
                                        inner_record,
                                        last_state->lambda_delta,
                                        last_state->u_delta,
                                        true);
                                }
                            }
                            else if (parameters_.compute_g_estimator)
                            {
                                inner_record.g_estimator_skipped_reason =
                                    "stopped_on_empty_y_marking";
                                inner_record.g_solver_status =
                                    "stopped_on_empty_y_marking";
                            }
                            stamp_inner_timing_(inner_record, inner_start, run_start);
                            append_partial_inner_history_(
                                outer_iteration,
                                inner_record,
                                inner_start,
                                run_start,
                                "completed");
                            push_inner_record_(
                                outer_record,
                                std::move(inner_record));
                            log_inner_iteration_(
                                logger,
                                outer_iteration,
                                outer_record.last_inner_iteration());
                            record_after_main_solve_bookkeeping();
                            break;
                        }

                        const bool can_refine_y =
                            inner_iteration + 1 < parameters_.max_inner_iterations &&
                            (parameters_.uniform_y_refinement ||
                             !inner_record.marked_y_cells.empty());

                        if (can_refine_y)
                        {
                            const auto y_refinement_metrics_before =
                                refinement_metric_snapshot_("y");
                            record_memory_sample_("before_y_refinement");
                            if (auto reason =
                                    memory_budget_reason_("Y refinement");
                                reason.has_value())
                            {
                                stamp_inner_timing_(
                                    inner_record,
                                    inner_start,
                                    run_start);
                                append_partial_inner_history_(
                                    outer_iteration,
                                    inner_record,
                                    inner_start,
                                    run_start,
                                    "failed");
                                push_inner_record_(
                                    outer_record,
                                    std::move(inner_record));
                                log_inner_iteration_(
                                    logger,
                                    outer_iteration,
                                    outer_record.last_inner_iteration());
                                record_after_main_solve_bookkeeping();
                                finalize_after_exception_(
                                    result,
                                    pending_outer_record,
                                    pending_outer_start,
                                    logger,
                                    *reason,
                                    run_start);
                                return result;
                            }

                            if (wall_time_budget_reached_(run_start))
                            {
                                stamp_inner_timing_(
                                    inner_record,
                                    inner_start,
                                    run_start);
                                append_partial_inner_history_(
                                    outer_iteration,
                                    inner_record,
                                    inner_start,
                                    run_start,
                                    "interrupted");
                                push_inner_record_(
                                    outer_record,
                                    std::move(inner_record));
                                log_inner_iteration_(
                                    logger,
                                    outer_iteration,
                                    outer_record.last_inner_iteration());
                                record_after_main_solve_bookkeeping();
                                finish_current_outer_with_time_budget_(
                                    result,
                                    outer_record,
                                    *pending_outer_start,
                                    run_start,
                                    logger,
                                    *last_state,
                                    *last_x_indicators);
                                pending_outer_record.reset();
                                pending_outer_start.reset();
                                return result;
                            }

                            record_after_main_solve_bookkeeping();
                            record_previous_solution_snapshot_(
                                previous_solution_snapshot,
                                last_state->lambda_delta,
                                last_state->u_delta);

                            const auto y_refinement_total_start = Clock::now();
                            {
                                auto timer = timing_.scoped("y.refinement");
                                const auto closure_mode =
                                    parameters_.local_time_slab_closure
                                        ? finite_element::time_slabs::
                                              LocalTimeSlabClosureMode::
                                                  marked_split_cells
                                        : finite_element::time_slabs::
                                              LocalTimeSlabClosureMode::
                                                  disabled;
                                if (parameters_.uniform_y_refinement)
                                {
                                    const auto active_y_cells =
                                        y_space_.active_cells();
                                    inner_record.y_refinement_target_cells =
                                        static_cast<int>(active_y_cells.size());
                                    refine_space_uniform_isotropically_(
                                        y_space_,
                                        active_y_cells,
                                        "y");
                                    finite_element::time_slabs::
                                        require_trial_space_embedded_in_test_space(
                                            x_space_,
                                            y_space_);
                                }
                                else
                                {
                                    inner_record.y_refinement_target_cells =
                                        static_cast<int>(
                                            inner_record.marked_y_cells.size());
                                    const auto closure_stats =
                                        finite_element::time_slabs::
                                            refine_test_space_from_marked_source_cells(
                                                y_space_,
                                                x_space_,
                                                inner_record.marked_y_cells,
                                                closure_mode);
                                    inner_record.local_time_slab_closure_applied =
                                        parameters_.local_time_slab_closure &&
                                        closure_stats.marked_split_cells > 0;
                                    inner_record
                                        .local_time_slab_closure_marked_split_cells =
                                            closure_stats.marked_split_cells;
                                    inner_record
                                        .local_time_slab_closure_temporal_waves =
                                            closure_stats.temporal_refinement_waves;
                                    inner_record
                                        .local_time_slab_closure_temporally_refined_cells =
                                            closure_stats.temporally_refined_cells;
                                }
                            }

                            inner_record.refined_y = true;
                            inner_record.refinement_statistics =
                                refinement_statistics_since_(
                                    "y",
                                    y_refinement_metrics_before);
                            inner_record.n_y_active_cells_after =
                                static_cast<int>(y_space_.active_cells().size());
                            inner_record.n_y_true_dofs_after =
                                y_space_.dof_handler_ref().n_true_dofs();
                            inner_record.y_generation_after =
                                active_generation_stats_(y_space_);
                            record_since_(
                                "adaptive_driver.y_refinement_total_wall",
                                y_refinement_total_start);

                            validate_current_spaces_();
                            append_partial_inner_history_(
                                outer_iteration,
                                inner_record,
                                inner_start,
                                run_start,
                                "after_y_refinement");

                            if (y_true_dof_limit_exceeded_(
                                    inner_record.n_y_true_dofs_after))
                            {
                                stamp_inner_timing_(inner_record, inner_start, run_start);
                                append_partial_inner_history_(
                                    outer_iteration,
                                    inner_record,
                                    inner_start,
                                    run_start,
                                    "interrupted");
                                push_inner_record_(
                                    outer_record,
                                    std::move(inner_record));
                                log_inner_iteration_(
                                    logger,
                                    outer_iteration,
                                    outer_record.last_inner_iteration());
                                record_after_main_solve_bookkeeping();
                                populate_outer_record_from_last_inner_(outer_record);
                                outer_record.x_indicator_components = *last_x_indicators;
                                update_outer_x_counts_after_(outer_record);
                                stamp_outer_timing_(
                                    outer_record,
                                    *pending_outer_start,
                                    run_start);
                                append_partial_outer_history_(
                                    outer_record,
                                    *pending_outer_start,
                                    run_start,
                                    "interrupted");
                                result.terminated_early = true;
                                result.termination_reason =
                                    make_y_true_dof_limit_reason_(
                                        outer_record.last_inner_iteration().n_y_true_dofs_after,
                                        "after refining Y^delta");
                                log_outer_iteration_summary_(
                                    logger,
                                    outer_record);
                                result.outer_iterations.push_back(std::move(outer_record));
                                pending_outer_record.reset();
                                set_final_state_(result, *last_state);
                                result.final_y_estimator =
                                    result.outer_iterations.back().final_y_estimator;
                                result.final_x_indicators =
                                    result.outer_iterations.back().x_indicator_components;
                                finalize_output_(result, logger);
                                return result;
                            }

                            if (wall_time_budget_reached_(run_start))
                            {
                                stamp_inner_timing_(
                                    inner_record,
                                    inner_start,
                                    run_start);
                                push_inner_record_(
                                    outer_record,
                                    std::move(inner_record));
                                log_inner_iteration_(
                                    logger,
                                    outer_iteration,
                                    outer_record.last_inner_iteration());
                                record_after_main_solve_bookkeeping();
                                finish_current_outer_with_time_budget_(
                                    result,
                                    outer_record,
                                    *pending_outer_start,
                                    run_start,
                                    logger,
                                    *last_state,
                                    *last_x_indicators);
                                pending_outer_record.reset();
                                pending_outer_start.reset();
                                return result;
                            }
                        }

                        record_after_main_solve_bookkeeping();
                        stamp_inner_timing_(inner_record, inner_start, run_start);
                        append_partial_inner_history_(
                            outer_iteration,
                            inner_record,
                            inner_start,
                            run_start,
                            "completed");
                        push_inner_record_(
                            outer_record,
                            std::move(inner_record));
                        log_inner_iteration_(
                            logger,
                            outer_iteration,
                            outer_record.last_inner_iteration());
                    }

                    const auto outer_post_inner_bookkeeping_start = Clock::now();
                    bool outer_post_inner_bookkeeping_recorded = false;
                    const auto record_outer_post_inner_bookkeeping =
                        [&]()
                        {
                            if (outer_post_inner_bookkeeping_recorded)
                                return;
                            record_since_(
                                "adaptive_driver."
                                "outer_post_inner_bookkeeping_wall",
                                outer_post_inner_bookkeeping_start);
                            outer_post_inner_bookkeeping_recorded = true;
                        };

                    if (outer_record.inner_iterations.empty() || !last_state.has_value())
                    {
                        update_current_outer_counts_after_(outer_record);
                        stamp_outer_timing_(
                            outer_record,
                            *pending_outer_start,
                            run_start);
                        append_partial_outer_history_(
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            "failed");
                        result.terminated_early = true;
                        result.termination_reason =
                            "max_inner_iterations == 0 prevented the Y-refinement loop from running";
                        log_outer_iteration_summary_(logger, outer_record);
                        record_outer_post_inner_bookkeeping();
                        result.outer_iterations.push_back(std::move(outer_record));
                        pending_outer_record.reset();
                        finalize_output_(result, logger);
                        return result;
                    }

                    if ((time_budget_reached_during_inner_loop ||
                         wall_time_budget_reached_(run_start)) &&
                        last_x_indicators.has_value())
                    {
                        record_outer_post_inner_bookkeeping();
                        finish_current_outer_with_time_budget_(
                            result,
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            logger,
                            *last_state,
                            *last_x_indicators);
                        pending_outer_record.reset();
                        pending_outer_start.reset();
                        return result;
                    }

                    populate_outer_record_from_last_inner_(outer_record);
                    outer_record.x_indicator_components = *last_x_indicators;

                    if (!outer_record.y_converged)
                    {
                        update_outer_x_counts_after_(outer_record);
                        stamp_outer_timing_(
                            outer_record,
                            *pending_outer_start,
                            run_start);
                        append_partial_outer_history_(
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            "completed");
                        result.terminated_early = true;
                        result.termination_reason =
                            outer_record.stopped_on_empty_y_marking
                                ? "inner Y loop stopped with an empty Dörfler marking before satisfying the stopping criterion"
                                : "inner Y loop reached max_inner_iterations before satisfying the stopping criterion";
                        log_outer_iteration_summary_(logger, outer_record);
                        record_outer_post_inner_bookkeeping();
                        result.outer_iterations.push_back(std::move(outer_record));
                        pending_outer_record.reset();
                        set_final_state_(result, *last_state);
                        result.final_y_estimator = result.outer_iterations.back().final_y_estimator;
                        result.final_x_indicators = result.outer_iterations.back().x_indicator_components;
                        finalize_output_(result, logger);
                        return result;
                    }

                    if (parameters_.eta_squared_stop > 0.0 &&
                        outer_record.eta_squared <= parameters_.eta_squared_stop)
                    {
                        update_outer_x_counts_after_(outer_record);
                        stamp_outer_timing_(
                            outer_record,
                            *pending_outer_start,
                            run_start);
                        append_partial_outer_history_(
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            "completed");
                        result.converged = true;
                        result.termination_reason =
                            "eta_squared_stop reached after satisfying the Y-loop stopping criterion";
                        log_outer_iteration_summary_(logger, outer_record);
                        record_outer_post_inner_bookkeeping();
                        result.outer_iterations.push_back(std::move(outer_record));
                        pending_outer_record.reset();
                        set_final_state_(result, *last_state);
                        result.final_y_estimator = result.outer_iterations.back().final_y_estimator;
                        result.final_x_indicators = result.outer_iterations.back().x_indicator_components;
                        finalize_output_(result, logger);
                        return result;
                    }

                    {
                        auto timer = timing_.scoped("x.marking");
                        if (parameters_.uniform_x_refinement)
                        {
                            outer_record.marked_x_cells =
                                x_space_.active_cells();
                        }
                        else if (parameters_.deterministic_estimator_reductions)
                        {
                            finite_element::time_slabs::
                                DoerflerMarkingDiagnostics diagnostics;
                            outer_record.marked_x_cells =
                                outer_record.x_indicator_components
                                    .eta_squared_by_x_cell
                                    .doerfler_marking_deterministic(
                                        parameters_.doerfler_theta_x,
                                        parameters_.doerfler_near_tie_tolerance,
                                        &diagnostics,
                                        make_timing_recorder_());
                            timing_.add(
                                "x_doerfler.cutoff_value",
                                diagnostics.cutoff_value);
                            timing_.add(
                                "x_doerfler.cutoff_index",
                                static_cast<double>(
                                    diagnostics.cutoff_index));
                            timing_.add(
                                "x_doerfler.cutoff_margin",
                                diagnostics.cutoff_margin);
                            timing_.add(
                                "x_doerfler.near_tie_count",
                                static_cast<double>(
                                    diagnostics.near_tie_count));
                        }
                        else
                        {
                            outer_record.marked_x_cells =
                                outer_record.x_indicator_components
                                    .eta_squared_by_x_cell
                                    .doerfler_marking(
                                        parameters_.doerfler_theta_x);
                        }
                    }
                    const std::uint64_t x_marked_hash =
                        finite_element::time_slabs::detail::
                            hash_cell_id_vector(
                                outer_record.marked_x_cells);
                    timing_.add(
                        "x_marked_set_checksum_low32",
                        static_cast<double>(
                            static_cast<std::uint32_t>(
                                x_marked_hash & 0xffffffffULL)));
                    timing_.add(
                        "x_marked_set_checksum_high32",
                        static_cast<double>(
                            static_cast<std::uint32_t>(
                                x_marked_hash >> 32U)));
                    timing_.add(
                        "x_marked_count",
                        static_cast<double>(
                            outer_record.marked_x_cells.size()));
                    outer_record.x_marking_empty = outer_record.marked_x_cells.empty();
                    append_partial_outer_history_(
                        outer_record,
                        *pending_outer_start,
                        run_start,
                        "after_x_estimator_marking");

                    {
                        auto timer =
                            timing_.scoped("output.outer_iteration_snapshot");
                        measure_output_history_phase_(
                            "adaptive_driver.snapshot_output_checks",
                            [&]()
                            {
                                output::write_outer_iteration_snapshot(
                                    outer_iteration,
                                    x_space_,
                                    outer_record.marked_x_cells,
                                    parameters_.output);
                            });
                    }

                    const bool can_refine_x =
                        outer_iteration + 1 < parameters_.max_outer_iterations &&
                        !outer_record.x_marking_empty;

                    if (can_refine_x)
                    {
                        record_memory_sample_("before_x_refinement");
                        if (auto reason =
                                memory_budget_reason_(
                                    "X refinement",
                                    estimated_space_storage_extra_after_x_refine_(
                                        outer_record));
                            reason.has_value())
                        {
                            finish_current_outer_with_memory_budget_(
                                result,
                                outer_record,
                                *pending_outer_start,
                                run_start,
                                logger,
                                *last_state,
                                *last_x_indicators,
                                *reason);
                            pending_outer_record.reset();
                            pending_outer_start.reset();
                            return result;
                        }

                        if (wall_time_budget_reached_(run_start))
                        {
                            record_outer_post_inner_bookkeeping();
                            finish_current_outer_with_time_budget_(
                                result,
                                outer_record,
                                *pending_outer_start,
                                run_start,
                                logger,
                                *last_state,
                                *last_x_indicators);
                            pending_outer_record.reset();
                            pending_outer_start.reset();
                            return result;
                        }

                        record_outer_post_inner_bookkeeping();
                        record_previous_solution_snapshot_(
                            previous_solution_snapshot,
                            last_state->lambda_delta,
                            last_state->u_delta);

                        const auto x_refinement_total_start = Clock::now();
                        {
                            auto timer = timing_.scoped("x.refinement");
                            const auto x_refinement_metrics_before =
                                refinement_metric_snapshot_("x");
                            if (parameters_.uniform_x_refinement)
                                refine_space_uniform_isotropically_(
                                    x_space_,
                                    outer_record.marked_x_cells,
                                    "x");
                            else
                                x_space_.refine(outer_record.marked_x_cells);
                            outer_record.refinement_statistics =
                                refinement_statistics_since_(
                                    "x",
                                    x_refinement_metrics_before);
                        }
                        outer_record.refined_x = true;

                        record_memory_sample_("after_x_refinement");
                        throw_if_memory_budget_exceeded_(
                            "y_initialization_after_x_refinement");

                        y_space_.initialize(x_space_.active_cells());
                        record_memory_sample_("after_y_initialization");
                        validate_current_spaces_();
                        update_outer_x_counts_after_(outer_record);
                        record_since_(
                            "adaptive_driver.x_refinement_total_wall",
                            x_refinement_total_start);
                        append_partial_outer_history_(
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            "after_x_refinement");

                        if (x_true_dof_limit_exceeded_(
                                outer_record.n_x_true_dofs_after))
                        {
                            stamp_outer_timing_(
                                outer_record,
                                *pending_outer_start,
                                run_start);
                            append_partial_outer_history_(
                                outer_record,
                                *pending_outer_start,
                                run_start,
                                "interrupted");
                            result.terminated_early = true;
                            result.termination_reason =
                                make_x_true_dof_limit_reason_(
                                    outer_record.n_x_true_dofs_after,
                                    "after refining X^delta");
                            log_outer_iteration_summary_(
                                logger,
                                outer_record);
                            record_outer_post_inner_bookkeeping();
                            result.outer_iterations.push_back(std::move(outer_record));
                            pending_outer_record.reset();
                            pending_outer_start.reset();
                            set_final_state_(result, *last_state);
                            result.final_y_estimator =
                                result.outer_iterations.back().final_y_estimator;
                            result.final_x_indicators =
                                result.outer_iterations.back().x_indicator_components;
                            finalize_output_(result, logger);
                            return result;
                        }
                    }

                    update_outer_x_counts_after_(outer_record);
                    stamp_outer_timing_(
                        outer_record,
                        *pending_outer_start,
                        run_start);
                    append_partial_outer_history_(
                        outer_record,
                        *pending_outer_start,
                        run_start,
                        "completed");
                    log_outer_iteration_summary_(logger, outer_record);
                    record_outer_post_inner_bookkeeping();
                    result.outer_iterations.push_back(std::move(outer_record));
                    pending_outer_record.reset();
                    pending_outer_start.reset();

                    if (can_refine_x &&
                        wall_time_budget_reached_(run_start))
                    {
                        set_final_state_(result, *last_state);
                        result.final_y_estimator =
                            result.outer_iterations.back().final_y_estimator;
                        result.final_x_indicators =
                            result.outer_iterations.back().x_indicator_components;
                        result.converged = false;
                        result.terminated_early = true;
                        result.termination_reason =
                            make_wall_time_budget_reason_(run_start);
                        finalize_output_(result, logger);
                        return result;
                    }

                    if (can_refine_x)
                        continue;

                    set_final_state_(result, *last_state);
                    result.final_y_estimator = result.outer_iterations.back().final_y_estimator;
                    result.final_x_indicators = result.outer_iterations.back().x_indicator_components;

                    if (result.outer_iterations.back().x_marking_empty)
                    {
                        result.converged = true;
                        result.termination_reason =
                            "X-marking became empty after the Y-loop stopping criterion was satisfied";
                    }
                    else
                    {
                        result.terminated_early = true;
                        result.termination_reason =
                            "maximum number of outer iterations reached";
                    }

                    finalize_output_(result, logger);
                    return result;
                }

                finalize_output_(result, logger);
                return result;
            }
            catch (const la::concepts::DirectSolverMemoryLimitExceeded& e)
            {
                finalize_after_exception_(
                    result,
                    pending_outer_record,
                    pending_outer_start,
                    logger,
                    e.what(),
                    run_start);
                return result;
            }
            catch (const MemoryBudgetExceeded& e)
            {
                finalize_after_exception_(
                    result,
                    pending_outer_record,
                    pending_outer_start,
                    logger,
                    e.what(),
                    run_start);
                return result;
            }
            catch (const std::exception& e)
            {
                finalize_after_exception_(
                    result,
                    pending_outer_record,
                    pending_outer_start,
                    logger,
                    e.what(),
                    run_start);
                throw;
            }
        }

    private:
        struct CellCoordinateExtents
        {
            std::array<double, XSpaceType::GT::dim_space_v> spatial{};
            double temporal = 0.0;
        };

        class MemoryBudgetExceeded : public std::runtime_error
        {
        public:
            explicit MemoryBudgetExceeded(const std::string& message)
                : std::runtime_error(message)
            {}
        };

        [[nodiscard]] static std::size_t bytes_from_mib_(double mib) noexcept
        {
            return mib > 0.0
                ? static_cast<std::size_t>(mib * 1024.0 * 1024.0)
                : std::size_t{0};
        }

        [[nodiscard]] static double mib_from_bytes_(std::size_t bytes) noexcept
        {
            return static_cast<double>(bytes) / (1024.0 * 1024.0);
        }

        [[nodiscard]] bool memory_guard_enabled_() const noexcept
        {
            return parameters_.memory_limit_mb > 0.0;
        }

        [[nodiscard]] double memory_guard_safety_factor_() const noexcept
        {
            return std::max(1.0, parameters_.memory_guard_safety_factor);
        }

        [[nodiscard]] std::size_t current_process_rss_bytes_() const noexcept
        {
            return finite_element::detail::current_process_rss_bytes();
        }

        [[nodiscard]] std::size_t available_system_memory_bytes_() const noexcept
        {
            return finite_element::detail::available_system_memory_bytes();
        }

        void record_memory_sample_(std::string_view phase)
        {
            if (!parameters_.timing.enabled)
                return;

            const auto sample_start = Clock::now();
            std::string prefix("memory_guard.");
            prefix.append(phase.data(), phase.size());
            timing_.add(
                prefix + ".process_rss_bytes",
                static_cast<double>(
                    finite_element::detail::current_process_rss_bytes()));
            timing_.add(
                prefix + ".process_peak_rss_bytes",
                static_cast<double>(
                    finite_element::detail::peak_process_rss_bytes()));
            timing_.add(
                prefix + ".mem_available_bytes",
                static_cast<double>(
                    finite_element::detail::available_system_memory_bytes()));
            record_bookkeeping_seconds_(
                "adaptive_driver.memory_sampling",
                sample_start);
        }

        [[nodiscard]] std::optional<std::string> memory_budget_reason_(
            std::string_view phase,
            std::size_t estimated_extra_bytes = 0) const
        {
            if (!memory_guard_enabled_())
                return std::nullopt;

            const std::size_t limit_bytes =
                bytes_from_mib_(parameters_.memory_limit_mb);
            const std::size_t reserve_bytes =
                bytes_from_mib_(parameters_.memory_reserve_mb);
            const std::size_t rss_bytes = current_process_rss_bytes_();
            const std::size_t available_bytes = available_system_memory_bytes_();
            const std::size_t adjusted_extra =
                static_cast<std::size_t>(
                    static_cast<double>(estimated_extra_bytes) *
                    memory_guard_safety_factor_());

            const auto make_message =
                [&](std::string_view trigger)
                {
                    std::ostringstream message;
                    message
                        << "memory_budget_reached " << trigger
                        << " before " << std::string(phase)
                        << ": process RSS=" << mib_from_bytes_(rss_bytes)
                        << " MiB";
                    if (adjusted_extra > 0)
                    {
                        message
                            << ", estimated extra="
                            << mib_from_bytes_(adjusted_extra)
                            << " MiB";
                    }
                    message
                        << ", cap=" << parameters_.memory_limit_mb
                        << " MiB, MemAvailable="
                        << mib_from_bytes_(available_bytes)
                        << " MiB, reserve="
                        << parameters_.memory_reserve_mb << " MiB";
                    return message.str();
                };

            if (limit_bytes > 0 && rss_bytes + adjusted_extra > limit_bytes)
                return make_message("process cap");

            if (available_bytes > 0 &&
                available_bytes <= reserve_bytes + adjusted_extra)
            {
                return make_message("available-memory reserve");
            }

            return std::nullopt;
        }

        void throw_if_memory_budget_exceeded_(
            std::string_view phase,
            std::size_t estimated_extra_bytes = 0)
        {
            record_memory_sample_(phase);
            if (auto reason =
                    memory_budget_reason_(phase, estimated_extra_bytes);
                reason.has_value())
            {
                throw MemoryBudgetExceeded(*reason);
            }
        }

        [[nodiscard]] double memory_cap_fraction_() const noexcept
        {
            if (!memory_guard_enabled_())
                return 0.0;
            const auto limit_bytes = bytes_from_mib_(parameters_.memory_limit_mb);
            if (limit_bytes == 0)
                return 0.0;
            return static_cast<double>(current_process_rss_bytes_()) /
                static_cast<double>(limit_bytes);
        }

        [[nodiscard]] bool near_memory_cap_() const noexcept
        {
            if (!memory_guard_enabled_())
                return false;

            const double threshold =
                std::clamp(
                    parameters_.memory_guard_near_cap_fraction,
                    0.0,
                    1.0);
            if (threshold > 0.0 && memory_cap_fraction_() >= threshold)
                return true;

            const auto available = available_system_memory_bytes_();
            const auto reserve = bytes_from_mib_(parameters_.memory_reserve_mb);
            return available > 0 && reserve > 0 &&
                available <=
                    static_cast<std::size_t>(
                        static_cast<double>(reserve) * 1.5);
        }

        [[nodiscard]] std::size_t estimated_space_storage_extra_after_x_refine_(
            const OuterRecord& outer_record) const
        {
            if (outer_record.marked_x_cells.empty())
                return 0;

            const double active =
                static_cast<double>(
                    std::max<std::size_t>(
                        std::size_t{1},
                        x_space_.active_cells().size()));
            const double marked =
                static_cast<double>(outer_record.marked_x_cells.size());
            const double marked_fraction = std::clamp(marked / active, 0.0, 1.0);
            const double local_growth =
                XSpaceType::GT::dim_space_v == 2 ? 4.0 : 4.0;
            const double growth =
                parameters_.uniform_x_refinement
                    ? local_growth
                    : 1.0 + marked_fraction * (local_growth - 1.0);

            const std::size_t x_dof_bytes =
                x_space_.dof_handler_ref().estimated_memory_bytes();
            const std::size_t y_dof_bytes =
                y_space_.dof_handler_ref().estimated_memory_bytes();
            const std::size_t cell_bytes =
                (x_space_.active_cells().size() +
                 y_space_.active_cells().size()) *
                std::size_t{768};
            const double current_structural_bytes =
                static_cast<double>(x_dof_bytes + y_dof_bytes + cell_bytes);

            if (!(growth > 1.0) || !(current_structural_bytes > 0.0))
                return 0;

            return static_cast<std::size_t>(
                current_structural_bytes * (growth - 1.0));
        }

        template<class SpaceType>
        [[nodiscard]] CellCoordinateExtents cell_coordinate_extents_(
            const SpaceType& space,
            const int cell_id) const
        {
            CellCoordinateExtents extents;
            const auto& mesh = space.mesh_ref();
            const auto& cell = mesh.cell(cell_id);
            const auto& spatial_vertices = mesh.spatial_vertices();
            const auto& temporal_vertices = mesh.temporal_vertices();

            for (int dim = 0; dim < SpaceType::GT::dim_space_v; ++dim)
            {
                double min_value = spatial_vertices[
                    static_cast<std::size_t>(cell.spatial_vertex_ids[0])][dim];
                double max_value = min_value;
                for (const int vertex_id : cell.spatial_vertex_ids)
                {
                    const double value =
                        spatial_vertices[static_cast<std::size_t>(vertex_id)][dim];
                    min_value = std::min(min_value, value);
                    max_value = std::max(max_value, value);
                }
                extents.spatial[static_cast<std::size_t>(dim)] =
                    max_value - min_value;
            }

            double min_time = temporal_vertices[
                static_cast<std::size_t>(cell.temporal_vertex_ids[0])][0];
            double max_time = min_time;
            for (const int vertex_id : cell.temporal_vertex_ids)
            {
                const double value =
                    temporal_vertices[static_cast<std::size_t>(vertex_id)][0];
                min_time = std::min(min_time, value);
                max_time = std::max(max_time, value);
            }
            extents.temporal = max_time - min_time;

            return extents;
        }

        [[nodiscard]] static bool extent_strictly_smaller_(
            const double child_extent,
            const double parent_extent) noexcept
        {
            const double tolerance =
                64.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, parent_extent);
            return child_extent < parent_extent - tolerance;
        }

        template<class SpaceType>
        [[nodiscard]] int active_uniform_parent_ancestor_(
            const SpaceType& space,
            int cell_id,
            const std::unordered_set<int>& uniform_parent_ids) const
        {
            const auto& mesh = space.mesh_ref();
            while (cell_id >= 0)
            {
                if (uniform_parent_ids.find(cell_id) != uniform_parent_ids.end())
                    return cell_id;
                cell_id = mesh.cell(cell_id).parent_id;
            }
            return -1;
        }

        template<class SpaceType>
        [[nodiscard]] std::vector<int> collect_uniform_spatial_shrink_failures_(
            const SpaceType& space,
            const std::unordered_set<int>& uniform_parent_ids,
            const std::unordered_map<int, CellCoordinateExtents>& parent_extents,
            std::string_view space_label)
            const
        {
            std::vector<int> failures;

            for (const int cell_id : space.active_cells())
            {
                const int parent_id =
                    active_uniform_parent_ancestor_(
                        space,
                        cell_id,
                        uniform_parent_ids);
                if (parent_id < 0)
                    throw std::runtime_error(
                        "uniform " + std::string(space_label) +
                        "-refinement: active cell is not a descendant of the original uniform partition.");

                const auto parent_it = parent_extents.find(parent_id);
                if (parent_it == parent_extents.end())
                    throw std::runtime_error(
                        "uniform " + std::string(space_label) +
                        "-refinement: missing original-cell extent data.");

                const auto child_extents =
                    cell_coordinate_extents_(space, cell_id);
                const auto& original_extents = parent_it->second;

                bool spatial_is_smaller = true;
                for (int dim = 0; dim < SpaceType::GT::dim_space_v; ++dim)
                {
                    spatial_is_smaller =
                        spatial_is_smaller &&
                        extent_strictly_smaller_(
                            child_extents.spatial[static_cast<std::size_t>(dim)],
                            original_extents.spatial[static_cast<std::size_t>(dim)]);
                }

                if (!spatial_is_smaller)
                    failures.push_back(cell_id);
            }

            return failures;
        }

        template<class SpaceType>
        [[nodiscard]] std::vector<int> collect_uniform_active_descendants_(
            const SpaceType& space,
            const std::unordered_set<int>& uniform_parent_ids) const
        {
            std::vector<int> descendants;
            descendants.reserve(space.active_cells().size());
            for (const int cell_id : space.active_cells())
            {
                if (active_uniform_parent_ancestor_(
                        space,
                        cell_id,
                        uniform_parent_ids) >= 0)
                {
                    descendants.push_back(cell_id);
                }
            }
            return descendants;
        }

        template<class SpaceType>
        void refine_uniform_cells_directly_(
            SpaceType& space,
            const std::vector<int>& cell_ids,
            mesh::RefinementType refinement_type,
            std::string_view space_label)
        {
            if (cell_ids.empty())
                return;

            if constexpr (SpaceType::GT::dim_space_v == 2 &&
                          SpaceType::GT::dim_time_v == 1)
            {
                auto& mesh = space.unsafe_mesh_ref();
                std::unordered_set<int> add_ids;
                std::unordered_set<int> remove_ids;
                std::vector<int> changed_cells;
                add_ids.reserve(cell_ids.size() * 2U);
                remove_ids.reserve(cell_ids.size());
                changed_cells.reserve(cell_ids.size() * 3U);

                const auto record_count =
                    [&](std::string_view name, std::size_t value)
                    {
                        const double count = static_cast<double>(value);
                        space.record_timing_metric(name, count);
                        std::string count_name(name);
                        count_name += ".count";
                        space.record_timing_metric(count_name, count);
                    };

                record_count(
                    "refinement.initially_marked_active_cells",
                    cell_ids.size());
                record_count("refinement.marked_cells", cell_ids.size());

                for (const int cell_id : cell_ids)
                {
                    if (!space.is_active_cell(cell_id))
                        continue;
                    if (!mesh.valid_cell_id(cell_id))
                        throw std::runtime_error(
                            "uniform " + std::string(space_label) +
                            "-refinement: active cell id out of range.");
                    if (!mesh.cell_is_storage_leaf(cell_id))
                        throw std::runtime_error(
                            "uniform " + std::string(space_label) +
                            "-refinement: active cell is not a storage leaf.");

                    mesh.refine_storage_leaf_without_closure(
                        cell_id,
                        refinement_type);

                    const auto& refined = mesh.cell(cell_id);
                    if (refined.children.empty())
                        throw std::runtime_error(
                            "uniform " + std::string(space_label) +
                            "-refinement: direct tensor-grid split produced no children.");

                    remove_ids.insert(cell_id);
                    changed_cells.push_back(cell_id);
                    for (const int child_id : refined.children)
                    {
                        if (!mesh.valid_cell_id(child_id) ||
                            !mesh.cell_is_storage_leaf(child_id))
                        {
                            throw std::runtime_error(
                                "uniform " + std::string(space_label) +
                                "-refinement: direct tensor-grid split produced an invalid child.");
                        }
                        add_ids.insert(child_id);
                        changed_cells.push_back(child_id);
                    }
                }

                record_count("refinement.queue_pops", cell_ids.size());
                record_count(
                    "refinement.unique_pending_cells_seen",
                    cell_ids.size());
                record_count("refinement.repeated_pending_cell_pops", 0U);
                record_count("refinement.requeued_due_to_blockers", 0U);
                record_count("refinement.blockers_found", 0U);
                record_count("refinement.blockers_already_seen", 0U);
                record_count(
                    "refinement.closure_decision_cache_possible_count",
                    0U);
                record_count("refinement.blocker_cells", 0U);
                record_count("refinement.closure_cells", 0U);
                record_count(
                    "refinement.actually_split_active_cells",
                    remove_ids.size());
                record_count(
                    "refinement.actually_split_cells",
                    remove_ids.size());
                record_count("refinement.grading_forced_cells", 0U);
                record_count("refinement.edge_interval_index_queries", 0U);
                record_count("refinement.edge_interval_records_visited", 0U);
                record_count("refinement.full_active_scans", 0U);

                if (add_ids.empty() && remove_ids.empty())
                    return;

                space.time_phase(
                    "refinement.active_partition_update_total",
                    [&]()
                    {
                        space.unsafe_update_active_cells(
                            add_ids,
                            remove_ids);
                    });

                std::sort(changed_cells.begin(), changed_cells.end());
                changed_cells.erase(
                    std::unique(
                        changed_cells.begin(),
                        changed_cells.end()),
                    changed_cells.end());

                space.time_phase(
                    "fespace.refinement.rebuild",
                    [&]()
                    {
                        space.rebuild_incremental_after_refinement(
                            changed_cells);
                    });
            }
            else
            {
                space.refine(cell_ids, refinement_type);
            }
        }

        template<class SpaceType>
        void refine_space_uniform_isotropically_(
            SpaceType& space,
            const std::vector<int>& seed_cells,
            std::string_view space_label)
        {
            if (seed_cells.empty())
                return;

            std::unordered_set<int> uniform_parent_ids;
            uniform_parent_ids.reserve(seed_cells.size());
            std::unordered_map<int, CellCoordinateExtents> parent_extents;
            parent_extents.reserve(seed_cells.size());
            for (const int cell_id : seed_cells)
            {
                if (!space.is_active_cell(cell_id))
                    continue;
                uniform_parent_ids.insert(cell_id);
                parent_extents.emplace(
                    cell_id,
                    cell_coordinate_extents_(space, cell_id));
            }

            if (uniform_parent_ids.empty())
                return;

            constexpr int max_uniform_isotropic_spatial_waves = 32;
            for (int spatial_wave = 0;
                 spatial_wave < max_uniform_isotropic_spatial_waves;
                 ++spatial_wave)
            {
                auto spatial_failures =
                    collect_uniform_spatial_shrink_failures_(
                        space,
                        uniform_parent_ids,
                        parent_extents,
                        space_label);
                if (spatial_failures.empty())
                {
                    timing_.add(
                        std::string(space_label) +
                            ".uniform_isotropic_refinement.spatial_waves.count",
                        static_cast<double>(spatial_wave));
                    auto temporal_targets =
                        collect_uniform_active_descendants_(
                            space,
                            uniform_parent_ids);
                    if (!temporal_targets.empty())
                    {
                        refine_uniform_cells_directly_(
                            space,
                            temporal_targets,
                            mesh::RefinementType::temporal,
                            space_label);
                        timing_.add(
                            std::string(space_label) +
                                ".uniform_isotropic_refinement.temporal_waves.count",
                            1.0);
                    }
                    return;
                }

                auto spatial_targets =
                    collect_uniform_active_descendants_(
                        space,
                        uniform_parent_ids);
                if (spatial_targets.empty())
                    throw std::runtime_error(
                        "uniform " + std::string(space_label) +
                        "-refinement: no active descendants available for spatial refinement.");
                refine_uniform_cells_directly_(
                    space,
                    spatial_targets,
                    mesh::RefinementType::spatial,
                    space_label);
            }

            throw std::runtime_error(
                "uniform " + std::string(space_label) +
                "-refinement: failed to obtain strictly smaller spatial cell extents after 32 refinement waves.");
        }

        class SpaceTimingHookScope
        {
        public:
            explicit SpaceTimingHookScope(AdaptiveDriver& driver)
                : driver_(&driver)
            {
                driver_->install_space_timing_hooks_();
            }

            SpaceTimingHookScope(const SpaceTimingHookScope&) = delete;
            SpaceTimingHookScope& operator=(const SpaceTimingHookScope&) = delete;

            SpaceTimingHookScope(SpaceTimingHookScope&& other) noexcept
                : driver_(std::exchange(other.driver_, nullptr))
            {}

            SpaceTimingHookScope& operator=(SpaceTimingHookScope&& other) noexcept
            {
                if (this == &other)
                    return *this;

                clear_noexcept_();
                driver_ = std::exchange(other.driver_, nullptr);
                return *this;
            }

            ~SpaceTimingHookScope() noexcept
            {
                clear_noexcept_();
            }

        private:
            void clear_noexcept_() noexcept
            {
                if (driver_ == nullptr)
                    return;

                try
                {
                    driver_->clear_space_timing_hooks_();
                }
                catch (...)
                {
                }

                driver_ = nullptr;
            }

            AdaptiveDriver* driver_ = nullptr;
        };

        void validate_configuration_() const
        {
            finite_element::detail::require_supported_time_slab_estimator_capability<
                typename XSpaceType::GT>();
            finite_element::detail::require_supported_time_slab_estimator_capability<
                typename YSpaceType::GT>();
            static_assert(XSpaceType::FETraitsType::p_space_v ==
                              XSpaceType::FETraitsType::p_time_v,
                          "AdaptiveDriver requires matching trial-space polynomial degrees in space and time.");
            static_assert(YSpaceType::FETraitsType::p_space_v ==
                              YSpaceType::FETraitsType::p_time_v,
                          "AdaptiveDriver requires matching test-space polynomial degrees in space and time.");
            static_assert(XSpaceType::FETraitsType::p_space_v ==
                              YSpaceType::FETraitsType::p_space_v,
                          "AdaptiveDriver requires matching polynomial degrees in X and Y.");

            if (!(parameters_.rho > 0.0))
            {
                throw std::runtime_error(
                    "AdaptiveDriver: rho must be positive.");
            }

            if (parameters_.max_outer_iterations < 0)
            {
                throw std::runtime_error(
                    "AdaptiveDriver: max_outer_iterations must be nonnegative.");
            }

            if (parameters_.max_inner_iterations <= 0 &&
                parameters_.max_outer_iterations > 0)
            {
                throw std::runtime_error(
                    "AdaptiveDriver: max_inner_iterations must be positive when outer iterations are requested.");
            }

            if (!(parameters_.doerfler_theta_y > 0.0 &&
                  parameters_.doerfler_theta_y <= 1.0))
            {
                throw std::runtime_error(
                    "AdaptiveDriver: doerfler_theta_y must lie in (0, 1].");
            }

            if (!(parameters_.doerfler_theta_x > 0.0 &&
                  parameters_.doerfler_theta_x <= 1.0))
            {
                throw std::runtime_error(
                    "AdaptiveDriver: doerfler_theta_x must lie in (0, 1].");
            }

            if (parameters_.polynomial_degree >= 0 &&
                parameters_.polynomial_degree != XSpaceType::FETraitsType::p_space_v)
            {
                throw std::runtime_error(
                    "AdaptiveDriver: polynomial_degree does not match the compile-time FE traits.");
            }

            if (parameters_.max_y_true_dofs < 0)
            {
                throw std::runtime_error(
                    "AdaptiveDriver: max_y_true_dofs must be nonnegative.");
            }

            if (parameters_.max_x_true_dofs < 0)
            {
                throw std::runtime_error(
                    "AdaptiveDriver: max_x_true_dofs must be nonnegative.");
            }

            if (parameters_.max_wall_time_seconds < 0.0)
            {
                throw std::runtime_error(
                    "AdaptiveDriver: max_wall_time_seconds must be nonnegative.");
            }

            validate_current_spaces_();
        }

        void validate_current_spaces_() const
        {
            finite_element::time_slabs::require_trial_space_embedded_in_test_space(
                x_space_,
                y_space_);
        }

        [[nodiscard]] bool y_true_dof_limit_exceeded_(int n_true_dofs) const noexcept
        {
            return parameters_.max_y_true_dofs > 0 &&
                n_true_dofs > parameters_.max_y_true_dofs;
        }

        [[nodiscard]] bool x_true_dof_limit_exceeded_(int n_true_dofs) const noexcept
        {
            return parameters_.max_x_true_dofs > 0 &&
                n_true_dofs > parameters_.max_x_true_dofs;
        }

        [[nodiscard]] std::string make_y_true_dof_limit_reason_(
            int n_true_dofs,
            std::string_view context) const
        {
            return
                "Y^delta true DoF count " + std::to_string(n_true_dofs) +
                " exceeded max_y_true_dofs=" +
                std::to_string(parameters_.max_y_true_dofs) + ' ' +
                std::string(context);
        }

        [[nodiscard]] std::string make_x_true_dof_limit_reason_(
            int n_true_dofs,
            std::string_view context) const
        {
            return
                "X^delta true DoF count " + std::to_string(n_true_dofs) +
                " exceeded max_x_true_dofs=" +
                std::to_string(parameters_.max_x_true_dofs) + ' ' +
                std::string(context);
        }

        [[nodiscard]] double elapsed_wall_seconds_(
            const Clock::time_point& run_start) const
        {
            return std::chrono::duration<double>(
                Clock::now() - run_start).count();
        }

        [[nodiscard]] bool wall_time_budget_reached_(
            const Clock::time_point& run_start) const
        {
            return parameters_.max_wall_time_seconds > 0.0 &&
                elapsed_wall_seconds_(run_start) >=
                    parameters_.max_wall_time_seconds;
        }

        [[nodiscard]] std::string make_wall_time_budget_reason_(
            const Clock::time_point& run_start) const
        {
            return
                "time_budget_reached: elapsed wall time " +
                std::to_string(elapsed_wall_seconds_(run_start)) +
                "s reached max_wall_time_seconds=" +
                std::to_string(parameters_.max_wall_time_seconds);
        }

        void record_bookkeeping_seconds_(
            std::string_view phase,
            const Clock::time_point& start) const
        {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            timing_.add(phase, seconds);
            timing_.add("adaptive_driver.bookkeeping.total", seconds);
        }

        void record_since_(
            std::string_view phase,
            const Clock::time_point& start) const
        {
            timing_.add(
                phase,
                std::chrono::duration<double>(Clock::now() - start).count());
        }

        void record_history_seconds_(
            std::string_view phase,
            const Clock::time_point& start) const
        {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            timing_.add(phase, seconds);
            timing_.add("adaptive_driver.history_writing.total", seconds);
            timing_.add("adaptive_driver.output_and_history_total_wall", seconds);
            timing_.add("adaptive_driver.bookkeeping.total", seconds);
        }

        template<class Fn>
        void measure_bookkeeping_phase_(
            std::string_view phase,
            Fn&& fn) const
        {
            const auto start = Clock::now();
            std::forward<Fn>(fn)();
            record_bookkeeping_seconds_(phase, start);
        }

        template<class Fn>
        void measure_history_phase_(
            std::string_view phase,
            Fn&& fn) const
        {
            const auto start = Clock::now();
            std::forward<Fn>(fn)();
            record_history_seconds_(phase, start);
        }

        template<class Fn>
        void measure_output_history_phase_(
            std::string_view phase,
            Fn&& fn) const
        {
            const auto start = Clock::now();
            std::forward<Fn>(fn)();
            const double seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            timing_.add(phase, seconds);
            timing_.add("adaptive_driver.output_and_history_total_wall", seconds);
            timing_.add("adaptive_driver.bookkeeping.total", seconds);
        }

        void record_active_run_total_wall_() const
        {
            if (!active_run_start_.has_value())
                return;

            timing_.add(
                "adaptive_driver.total_wall",
                std::chrono::duration<double>(
                    Clock::now() - *active_run_start_)
                    .count());
        }

        void record_previous_solution_snapshot_(
            std::optional<YIterationSolutionSnapshot>& snapshot,
            const YFunction& lambda_delta,
            const XFunction& u_delta) const
        {
            const auto snapshot_start = Clock::now();
            snapshot.emplace(
                y_space_,
                x_space_,
                lambda_delta,
                u_delta);
            record_since_(
                "adaptive_driver.previous_solution_snapshot_wall",
                snapshot_start);
            timing_.add("adaptive_driver.previous_solution_snapshot_count", 1.0);
        }

        void record_output_profile_checks_() const
        {
            measure_bookkeeping_phase_(
                "adaptive_driver.output_profile_checks",
                [&]()
                {
                    [[maybe_unused]] const bool export_history =
                        parameters_.output.export_history;
                    [[maybe_unused]] const bool save_components =
                        parameters_.output.save_estimator_components;
                    [[maybe_unused]] const bool save_refinement =
                        parameters_.output.save_refinement_history;
                    [[maybe_unused]] const bool save_snapshots =
                        parameters_.output.save_iteration_snapshots;
                    [[maybe_unused]] const bool save_snapshot_dofs =
                        parameters_.output.save_snapshot_dofs;
                    [[maybe_unused]] const bool save_mesh_statistics =
                        parameters_.output.save_mesh_statistics;
                });
        }

        template<typename CellIdType>
        void log_outer_iteration_start_(
            AlgorithmLogger& logger,
            const AdaptiveOuterIterationRecord<CellIdType>& record) const
        {
            measure_bookkeeping_phase_(
                "adaptive_driver.logger",
                [&]() { logger.print_outer_iteration_start(record); });
        }

        template<typename CellIdType>
        void log_inner_iteration_(
            AlgorithmLogger& logger,
            int outer_iteration,
            const AdaptiveYIterationRecord<CellIdType>& record) const
        {
            measure_bookkeeping_phase_(
                "adaptive_driver.logger",
                [&]()
                {
                    logger.print_inner_iteration(
                        outer_iteration,
                        record);
                });
        }

        template<typename CellIdType>
        void log_outer_iteration_summary_(
            AlgorithmLogger& logger,
            const AdaptiveOuterIterationRecord<CellIdType>& record) const
        {
            measure_bookkeeping_phase_(
                "adaptive_driver.logger",
                [&]() { logger.print_outer_iteration_summary(record); });
        }

        void log_completion_(
            AlgorithmLogger& logger,
            const ResultType& result) const
        {
            measure_bookkeeping_phase_(
                "adaptive_driver.logger",
                [&]() { logger.print_completion(result); });
        }

        [[nodiscard]] YIterationParameters make_y_iteration_parameters_()
        {
            YIterationParameters parameters;
            parameters.doerfler_theta = parameters_.doerfler_theta_y;
            parameters.zero_tol = parameters_.zero_tol;
            parameters.divergence_residual_l2_tolerance =
                parameters_.divergence_residual_l2_tolerance;
            parameters.check_divergence_residual =
                parameters_.check_divergence_residual;
            parameters.use_adaptive_initial_guess =
                parameters_.use_adaptive_initial_guess;
            parameters.solve_main_system_correction =
                parameters_.solve_main_system_correction;
            parameters.fused_error_and_flux_diagnostics =
                parameters_.fused_error_and_flux_diagnostics;
            parameters.local_error_reuse_patch_solve_workspace =
                parameters_.local_error_reuse_patch_solve_workspace;
            parameters.deterministic_estimator_reductions =
                parameters_.deterministic_estimator_reductions;
            parameters.doerfler_near_tie_tolerance =
                parameters_.doerfler_near_tie_tolerance;
            parameters.local_error_patch_tile_size =
                parameters_.local_error_patch_tile_size;
            parameters.local_error_cell_chunk_size =
                parameters_.local_error_cell_chunk_size;
            parameters.local_error_max_threads =
                parameters_.local_error_max_threads;
            parameters.local_error_memory_budget_mb =
                parameters_.local_error_memory_budget_mb;
            parameters.local_error_worker_context_mode =
                parameters_.local_error_worker_context_mode;
            parameters.local_error_context_storage =
                parameters_.local_error_context_storage;
            parameters.local_error_state_index_mode =
                parameters_.local_error_state_index_mode;
            parameters.local_error_cell_state_cache_mode =
                parameters_.local_error_cell_state_cache_mode;
            parameters.local_error_cell_state_cache_budget_mb =
                parameters_.local_error_cell_state_cache_budget_mb;
            parameters.local_error_cell_state_representation =
                parameters_.local_error_cell_state_representation;
            parameters.local_error_flux_diagnostics_mode =
                parameters_.local_error_flux_diagnostics_mode;
            parameters.local_error_patch_solver =
                parameters_.local_error_patch_solver;
            parameters.local_error_coefficient_fast_path =
                parameters_.local_error_coefficient_fast_path;
            parameters.local_error_compact_state_shadow =
                parameters_.local_error_compact_state_shadow;
            parameters.shared_context_validation =
                parameters_.shared_context_validation;
            parameters.slab_reconstruction_operator_mode =
                parameters_.slab_reconstruction_operator_mode;
            parameters.main_assembly_max_threads =
                parameters_.main_assembly_max_threads;
            parameters.slab_reconstruction_max_threads =
                parameters_.slab_reconstruction_max_threads;
            parameters.main_assembly_memory_budget_mb =
                parameters_.main_assembly_memory_budget_mb;
            parameters.slab_reconstruction_memory_budget_mb =
                parameters_.slab_reconstruction_memory_budget_mb;
            parameters.main_two_pass_numeric_fill_max_threads =
                parameters_.main_two_pass_numeric_fill_max_threads;
            parameters.main_two_pass_numeric_fill_memory_budget_mb =
                parameters_
                    .main_two_pass_numeric_fill_memory_budget_mb;
            parameters.time_slab_backend = parameters_.time_slab_backend;
            parameters.allow_copied_time_slab_estimator_fallback =
                parameters_
                    .allow_copied_time_slab_estimator_fallback;
            parameters.virtual_backend_diagnostics =
                parameters_.virtual_backend_diagnostics;
            parameters.main_system_export =
                parameters_.main_system_export;
            parameters.main_two_pass_assembly_cache =
                &main_two_pass_assembly_cache_;
            parameters.timing = make_timing_recorder_();

            if (near_memory_cap_())
            {
                timing_.add("memory_guard.near_cap_degradation", 1.0);
                parameters.local_error_max_threads = 1;
                parameters.main_assembly_max_threads = 1;
                parameters.slab_reconstruction_max_threads = 1;
                parameters.main_two_pass_numeric_fill_max_threads = 1;
            }

            return parameters;
        }

        void configure_main_system_export_(
            finite_element::system::MainSystemExportOptions& export_options,
            int outer_iteration,
            int inner_iteration) const
        {
            if (!export_options.enabled)
                return;

            if (export_options.output_directory.empty())
            {
                export_options.output_directory =
                    parameters_.output.output_directory /
                    "solver_diagnostics";
            }

            export_options.prefix =
                "main_system_outer" + std::to_string(outer_iteration) +
                "_inner" + std::to_string(inner_iteration);
        }

        [[nodiscard]] la::concepts::SolverOptions
        memory_adjusted_main_solver_options_()
        {
            auto options = main_solver_options_;
            options.direct_memory_reserve_mb = parameters_.memory_reserve_mb;
            options.direct_memory_safety_factor =
                parameters_.memory_guard_safety_factor;
            if (parameters_.memory_limit_mb > 0.0)
            {
                options.direct_memory_limit_mb =
                    options.direct_memory_limit_mb > 0.0
                        ? std::min(
                              options.direct_memory_limit_mb,
                              parameters_.memory_limit_mb)
                        : parameters_.memory_limit_mb;
            }

            if (near_memory_cap_())
            {
                timing_.add("memory_guard.near_cap_ooc_switch_enabled", 1.0);
                options.pardiso_out_of_core_auto_switch = true;
                options.pardiso_out_of_core_switch_threshold =
                    std::min(
                        options.pardiso_out_of_core_switch_threshold,
                        0.75);
            }

            return options;
        }

        [[nodiscard]] la::concepts::SolverOptions
        memory_adjusted_g_solver_options_()
        {
            auto options = g_solver_options_;
            options.direct_memory_reserve_mb = parameters_.memory_reserve_mb;
            options.direct_memory_safety_factor =
                parameters_.memory_guard_safety_factor;
            if (parameters_.memory_limit_mb > 0.0)
            {
                options.direct_memory_limit_mb =
                    options.direct_memory_limit_mb > 0.0
                        ? std::min(
                              options.direct_memory_limit_mb,
                              parameters_.memory_limit_mb)
                        : parameters_.memory_limit_mb;
            }

            if (near_memory_cap_())
            {
                timing_.add("memory_guard.near_cap_g_ooc_switch_enabled", 1.0);
                options.pardiso_out_of_core_auto_switch = true;
                options.pardiso_out_of_core_switch_threshold =
                    std::min(
                        options.pardiso_out_of_core_switch_threshold,
                        0.75);
            }

            return options;
        }

        template<class SpaceType>
        [[nodiscard]] ActiveGenerationStats
        active_generation_stats_(const SpaceType& space) const
        {
            const auto stats_start = Clock::now();
            ActiveGenerationStats stats;
            const auto& active_cells = space.active_cells();
            if (active_cells.empty())
            {
                record_bookkeeping_seconds_(
                    "adaptive_driver.generation_stats",
                    stats_start);
                return stats;
            }

            stats.min_generation = std::numeric_limits<int>::max();
            stats.max_generation = std::numeric_limits<int>::min();
            std::unordered_set<int> generations;
            generations.reserve(active_cells.size());
            const auto& mesh = space.mesh_ref();
            for (const int cell_id : active_cells)
            {
                const int generation = mesh.cell(cell_id).generation;
                stats.min_generation =
                    std::min(stats.min_generation, generation);
                stats.max_generation =
                    std::max(stats.max_generation, generation);
                generations.insert(generation);
            }
            stats.distinct_generations =
                static_cast<int>(generations.size());
            record_bookkeeping_seconds_(
                "adaptive_driver.generation_stats",
                stats_start);
            return stats;
        }

        struct RefinementMetricSnapshot
        {
            std::optional<double> refined_cells{};
            std::optional<double> grading_forced_cells{};
            std::optional<double> blocker_cells{};
            std::optional<double> edge_interval_queries{};
            std::optional<double> edge_interval_records_visited{};
            std::optional<double> full_active_scans{};
        };

        [[nodiscard]] std::optional<double>
        timing_total_(const std::string& phase) const
        {
            const auto record = timing_.find(phase);
            if (!record.has_value())
                return std::nullopt;
            return record->total_seconds;
        }

        [[nodiscard]] std::optional<double>
        timing_total_(std::string_view space_name, std::string_view phase) const
        {
            return timing_total_(
                make_space_timing_phase_name_(space_name, phase));
        }

        [[nodiscard]] RefinementMetricSnapshot
        refinement_metric_snapshot_(std::string_view space_name) const
        {
            RefinementMetricSnapshot snapshot;
            snapshot.refined_cells =
                timing_total_(space_name, "refinement.actually_split_active_cells");
            snapshot.grading_forced_cells =
                timing_total_(space_name, "refinement.grading_forced_cells");
            snapshot.blocker_cells =
                timing_total_(space_name, "refinement.blocker_cells");
            snapshot.edge_interval_queries =
                timing_total_(space_name, "refinement.edge_interval_index_queries");
            snapshot.edge_interval_records_visited =
                timing_total_(
                    space_name,
                    "refinement.edge_interval_records_visited");
            snapshot.full_active_scans =
                timing_total_(space_name, "refinement.full_active_scans");
            if (!snapshot.full_active_scans.has_value())
            {
                snapshot.full_active_scans =
                    timing_total_(
                        space_name,
                        "refinement.full_active_scans_in_normal_path");
            }
            return snapshot;
        }

        [[nodiscard]] static std::optional<int>
        nonnegative_count_delta_(
            const std::optional<double> before,
            const std::optional<double> after)
        {
            if (!after.has_value())
                return std::nullopt;

            const double delta = *after - before.value_or(0.0);
            if (delta < -0.5)
                return std::nullopt;

            return static_cast<int>(
                std::llround(std::max(0.0, delta)));
        }

        [[nodiscard]] static std::optional<int>
        optional_count_sum_(
            const std::optional<int> a,
            const std::optional<int> b)
        {
            if (!a.has_value())
                return b;
            if (!b.has_value())
                return a;
            return *a + *b;
        }

        [[nodiscard]] AdaptiveRefinementStatistics
        refinement_statistics_since_(
            std::string_view space_name,
            const RefinementMetricSnapshot& before) const
        {
            const auto after = refinement_metric_snapshot_(space_name);
            AdaptiveRefinementStatistics stats;
            stats.refined_cells =
                nonnegative_count_delta_(
                    before.refined_cells,
                    after.refined_cells);
            const auto grading_forced =
                nonnegative_count_delta_(
                    before.grading_forced_cells,
                    after.grading_forced_cells);
            const auto blockers =
                nonnegative_count_delta_(
                    before.blocker_cells,
                    after.blocker_cells);
            stats.closure_cells = optional_count_sum_(grading_forced, blockers);
            stats.edge_interval_queries =
                nonnegative_count_delta_(
                    before.edge_interval_queries,
                    after.edge_interval_queries);
            stats.edge_interval_records_visited =
                nonnegative_count_delta_(
                    before.edge_interval_records_visited,
                    after.edge_interval_records_visited);
            stats.full_active_scans =
                nonnegative_count_delta_(
                    before.full_active_scans,
                    after.full_active_scans);
            return stats;
        }

        [[nodiscard]] static AdaptiveRefinementStatistics
        zero_refinement_statistics_()
        {
            AdaptiveRefinementStatistics stats;
            stats.refined_cells = 0;
            stats.closure_cells = 0;
            stats.edge_interval_queries = 0;
            stats.edge_interval_records_visited = 0;
            stats.full_active_scans = 0;
            return stats;
        }

        void update_outer_counts_before_(OuterRecord& record) const
        {
            record.n_x_active_cells_before =
                static_cast<int>(x_space_.active_cells().size());
            record.n_x_true_dofs_before =
                x_space_.dof_handler_ref().n_true_dofs();
            record.x_generation_before =
                active_generation_stats_(x_space_);
            record.n_y_active_cells_before =
                static_cast<int>(y_space_.active_cells().size());
            record.n_y_true_dofs_before =
                y_space_.dof_handler_ref().n_true_dofs();
            record.y_generation_before =
                active_generation_stats_(y_space_);
            record.refinement_statistics = zero_refinement_statistics_();
        }

        void update_current_outer_counts_after_(OuterRecord& record) const
        {
            record.n_x_active_cells_after =
                static_cast<int>(x_space_.active_cells().size());
            record.n_x_true_dofs_after =
                x_space_.dof_handler_ref().n_true_dofs();
            record.x_generation_after =
                active_generation_stats_(x_space_);
            record.n_y_active_cells_after =
                static_cast<int>(y_space_.active_cells().size());
            record.n_y_true_dofs_after =
                y_space_.dof_handler_ref().n_true_dofs();
            record.y_generation_after =
                active_generation_stats_(y_space_);
        }

        void update_outer_x_counts_after_(OuterRecord& record) const
        {
            record.n_x_active_cells_after =
                static_cast<int>(x_space_.active_cells().size());
            record.n_x_true_dofs_after =
                x_space_.dof_handler_ref().n_true_dofs();
            record.x_generation_after =
                active_generation_stats_(x_space_);
        }

        void populate_outer_record_from_last_inner_(OuterRecord& record) const
        {
            const auto& last_inner = record.last_inner_iteration();
            record.lambda_y_squared = last_inner.lambda_y_squared;
            record.initial_trace_squared = last_inner.initial_trace_squared;
            record.eta_squared = last_inner.eta_squared;
            record.y_estimator_squared = last_inner.y_estimator_squared;
            record.y_estimator_threshold_squared =
                last_inner.y_estimator_threshold_squared;
            record.configured_rho = last_inner.configured_rho;
            record.effective_rho = last_inner.effective_rho;
            record.effective_rho_available =
                last_inner.effective_rho_available;
            record.effective_rho_reason =
                last_inner.effective_rho_reason;
            record.y_estimator_threshold_configured_rho_squared =
                last_inner.y_estimator_threshold_configured_rho_squared;
            record.y_estimator_threshold_effective_rho_squared =
                last_inner.y_estimator_threshold_effective_rho_squared;
            record.posteriori_factor_configured_rho =
                last_inner.posteriori_factor_configured_rho;
            record.posteriori_factor_effective_rho =
                last_inner.posteriori_factor_effective_rho;
            record.posteriori_estimator_configured_rho_squared =
                last_inner.posteriori_estimator_configured_rho_squared;
            record.posteriori_estimator_effective_rho_squared =
                last_inner.posteriori_estimator_effective_rho_squared;
            record.posteriori_improvement_factor =
                last_inner.posteriori_improvement_factor;
            record.force_accept_inner_with_effective_rho =
                last_inner.force_accept_inner_with_effective_rho;
            record.configured_rho_ignored_for_inner_acceptance =
                last_inner.configured_rho_ignored_for_inner_acceptance;
            record.effective_rho_acceptance_used =
                last_inner.effective_rho_acceptance_used;
            record.effective_rho_acceptance_reason =
                last_inner.effective_rho_acceptance_reason;
            record.g_estimator_enabled = last_inner.g_estimator_enabled;
            record.g_estimator_computed = last_inner.g_estimator_computed;
            record.g_estimator_skipped_reason =
                last_inner.g_estimator_skipped_reason;
            record.g_space_constructed = last_inner.g_space_constructed;
            record.g_true_dofs = last_inner.g_true_dofs;
            record.g_solve_count = last_inner.g_solve_count;
            record.g_solver_status = last_inner.g_solver_status;
            record.g_assembly_seconds = last_inner.g_assembly_seconds;
            record.g_solve_seconds = last_inner.g_solve_seconds;
            record.g_solver_residual = last_inner.g_solver_residual;
            record.g_solver_relative_residual =
                last_inner.g_solver_relative_residual;
            record.g_rhs_inf_norm = last_inner.g_rhs_inf_norm;
            record.g_lambda_inf_norm = last_inner.g_lambda_inf_norm;
            record.g_lambda_difference_available =
                last_inner.g_lambda_difference_available;
            record.g_lambda_difference_squared =
                last_inner.g_lambda_difference_squared;
            record.g_lambda_difference =
                last_inner.g_lambda_difference;
            record.final_y_estimator = last_inner.estimator;
            record.n_y_active_cells_after = last_inner.n_y_active_cells_after;
            record.n_y_true_dofs_after = last_inner.n_y_true_dofs_after;
            record.y_generation_after = last_inner.y_generation_after;
        }

        void record_final_inner_g_estimator_counters_(
            const InnerRecord& record) const
        {
            if (!parameters_.compute_g_estimator)
            {
                timing_.add("g_estimator.skipped_disabled_count", 1.0);
                return;
            }

            if (record.g_estimator_computed)
            {
                if (record.stopping_criterion_satisfied &&
                    !record.refined_y)
                {
                    timing_.add(
                        "g_estimator.computed_on_accepted_inner_count",
                        1.0);
                }
                else if (!record.stopping_criterion_satisfied &&
                         record.refined_y)
                {
                    timing_.add(
                        "g_estimator.computed_on_rejected_inner_count",
                        1.0);
                }
                else if (!record.stopping_criterion_satisfied)
                {
                    timing_.add(
                        "g_estimator.computed_on_nonaccepted_stop_count",
                        1.0);
                }
                return;
            }

            if (record.g_estimator_skipped_reason ==
                "inner_iteration_not_accepted")
            {
                timing_.add(
                    "g_estimator.skipped_rejected_inner_count",
                    1.0);
            }
        }

        void push_inner_record_(
            OuterRecord& outer_record,
            InnerRecord&& inner_record) const
        {
            record_final_inner_g_estimator_counters_(inner_record);
            outer_record.inner_iterations.push_back(std::move(inner_record));
        }

        void compute_g_estimator_for_accepted_inner_iteration_(
            InnerRecord& record,
            const YFunction& lambda_delta,
            const XFunction& u_delta,
            bool allow_empty_y_marking_stop = false)
        {
            if (!parameters_.compute_g_estimator)
                return;

            if (record.g_estimator_computed)
                return;

            if (!record.stopping_criterion_satisfied &&
                !allow_empty_y_marking_stop)
                throw std::logic_error(
                    "AdaptiveDriver: attempted to solve G^delta for a "
                    "non-accepted inner iteration.");

            auto g_total_timer =
                timing_.scoped("g_estimator.total_seconds");

            using GeomTraits = typename YSpaceType::GT;
            using YFETraits = typename YSpaceType::FETraitsType;
            using GFETraits = finite_element::FiniteElementTraits<
                GeomTraits,
                YFETraits::p_space_v + 1,
                YFETraits::p_time_v + 1>;
            using GSpaceType = finite_element::FESpace<
                GeomTraits,
                GFETraits,
                finite_element::SpaceOnlyPolicy>;
            using GFunction =
                finite_element::Function<GSpaceType, Vector>;

            finite_element::FESpaceInitializationOptions options;
            options.search_index =
                finite_element::SearchIndexBuildMode::Disabled;
            options.build_refinement_indices = false;
            options.build_edge_interval_index = false;

            GSpaceType g_space(y_space_.unsafe_mesh_ref());
            {
                auto timer = timing_.scoped("g_delta.space_construction");
                auto g_timer =
                    timing_.scoped("g_estimator.space_construction_seconds");
                g_space.initialize(y_space_.active_cells(), options);
            }

            record.g_space_constructed = true;
            record.g_true_dofs =
                g_space.dof_handler_ref().n_true_dofs();

            typename Backend::SparseMatrix A_g;
            typename Backend::SparseMatrix B_xg;
            Vector f_g;
            Vector rhs_g;
            Vector lambda_g_true;

            const auto assemble_start = Clock::now();
            {
                auto timer = timing_.scoped("g_delta.assembly");
                finite_element::assembly::detail::AssemblySpaceCache<
                    GSpaceType>
                    g_cache(g_space);
                finite_element::assembly::detail::AssemblySpaceCache<
                    XSpaceType>
                    x_cache(x_space_);
                finite_element::assembly::detail::ActiveAncestorCache<
                    XSpaceType>
                    ancestor_cache(x_space_);

                {
                    auto g_timer =
                        timing_.scoped("g_estimator.assembly_A_seconds");
                    finite_element::assembly::assemble_mat_A<
                        QSpace + 1,
                        QTime + 1,
                        Backend>(
                            A_g,
                            g_space,
                            problem_.M,
                            g_cache,
                            parameters_.zero_tol);
                }

                {
                    auto g_timer =
                        timing_.scoped("g_estimator.assembly_rhs_seconds");
                    finite_element::assembly::assemble_vec_f<
                        QSpace + 1,
                        QTime + 1>(
                            f_g,
                            g_space,
                            problem_.ell,
                            g_cache,
                            parameters_.zero_tol);

                    // Match the first block of the main saddle system:
                    // A_G lambda_G = f_G - B(X,G)^T u_delta, with full
                    // B = B_dt + B_A.
                    finite_element::assembly::assemble_mat_B<
                        QSpace + 1,
                        QTime + 1,
                        Backend>(
                            B_xg,
                            x_space_,
                            g_space,
                            x_cache,
                            g_cache,
                            ancestor_cache,
                            problem_.M,
                            parameters_.zero_tol);

                    const auto mixed_u_g =
                        la::ops::transpose_matvec(
                            B_xg,
                            u_delta.true_coefficients());
                    rhs_g = la::ops::subtract(f_g, mixed_u_g);
                }
            }
            const auto assemble_end = Clock::now();
            record.g_assembly_seconds =
                std::chrono::duration<double>(
                    assemble_end - assemble_start)
                    .count();

            const auto solve_start = Clock::now();
            {
                auto timer = timing_.scoped("g_delta.solve");
                auto g_timer = timing_.scoped("g_estimator.solve_seconds");
                auto solver_options = memory_adjusted_g_solver_options_();
                solver_.compute(A_g, solver_options);
                lambda_g_true.resize(rhs_g.size());
                solver_.solve(rhs_g, lambda_g_true);
            }
            const auto solve_end = Clock::now();
            record.g_solve_seconds =
                std::chrono::duration<double>(
                    solve_end - solve_start)
                    .count();

            double lambda_difference_sq_raw = 0.0;
            double lambda_difference_coeff_sq = 0.0;
            {
                auto g_norm_timer =
                    timing_.scoped("g_estimator.norm_seconds");
                GFunction lambda_G(g_space);
                {
                    auto timer = timing_.scoped("g_delta.solution_update");
                    lambda_G.update_from_true_solution(lambda_g_true);
                }

                Vector lambda_delta_in_g_true;
                {
                    auto timer = timing_.scoped("g_delta.prolongation");
                    lambda_delta_in_g_true =
                        finite_element::fespace::
                            prolong_true_coefficients_nodal(
                                lambda_delta,
                                g_space);
                }

                Vector lambda_difference_true;
                {
                    auto timer =
                        timing_.scoped(
                            "g_delta.lambda_difference_coefficients");
                    lambda_difference_true =
                        la::ops::subtract(
                            lambda_G.true_coefficients(),
                            lambda_delta_in_g_true);
                    lambda_difference_coeff_sq =
                        la::ops::dot(
                            lambda_difference_true,
                            lambda_difference_true);
                }

                {
                    auto timer =
                        timing_.scoped(
                            "g_delta.lambda_difference_energy_norm");
                    lambda_difference_sq_raw =
                        la::ops::quadratic_form<
                            typename Backend::SparseMatrix,
                            typename Backend::SparseBuilder>(
                                A_g,
                                lambda_difference_true);
                }
            }
            const double negative_tolerance =
                1000.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, lambda_difference_coeff_sq);
            if (lambda_difference_sq_raw < -negative_tolerance)
            {
                std::ostringstream msg;
                msg
                    << "AdaptiveDriver: computed a negative G^delta "
                    << "lambda-difference energy norm squared ("
                    << lambda_difference_sq_raw << ").";
                throw std::runtime_error(msg.str());
            }
            const double lambda_difference_squared =
                std::max(0.0, lambda_difference_sq_raw);

            {
                auto timer = timing_.scoped("g_delta.diagnostics");
                const auto diagnostics = solver_.last_diagnostics();
                record.g_solver_residual =
                    diagnostics.linear_residual_absolute;
                record.g_solver_relative_residual =
                    diagnostics.linear_residual_relative;
                record.g_rhs_inf_norm = la::ops::inf_norm(rhs_g);
                record.g_lambda_inf_norm = la::ops::inf_norm(lambda_g_true);
                record.g_lambda_difference_available = true;
                record.g_lambda_difference_squared = lambda_difference_squared;
                record.g_lambda_difference =
                    std::sqrt(lambda_difference_squared);
                record.g_solve_count = 1;
                record.g_solver_status = "success";
                record.g_estimator_computed = true;
                record.g_estimator_skipped_reason = "not_skipped";
            }

            timing_.add("g_estimator.compute_count", 1.0);
            timing_.add("g_delta.solve_count.count", 1.0);
            timing_.add(
                "g_delta.assembly_seconds",
                *record.g_assembly_seconds);
            timing_.add("g_delta.solve_seconds", *record.g_solve_seconds);
            if (record.g_solver_residual.has_value())
                timing_.add(
                    "g_delta.solver_residual",
                    *record.g_solver_residual);
            if (record.g_solver_relative_residual.has_value())
                timing_.add(
                    "g_delta.solver_relative_residual",
                    *record.g_solver_relative_residual);
            timing_.add(
                "g_delta.lambda_difference_squared",
                lambda_difference_squared);
            timing_.add(
                "g_delta.lambda_difference",
                std::sqrt(lambda_difference_squared));
        }

        void populate_effective_rho_(InnerRecord& record) const
        {
            record.configured_rho = parameters_.rho;
            record.y_estimator_threshold_configured_rho_squared =
                record.y_estimator_threshold_squared;
            record.posteriori_factor_configured_rho =
                detail::posteriori_factor(parameters_.rho);
            record.posteriori_estimator_configured_rho_squared =
                record.posteriori_factor_configured_rho * record.eta_squared;

            detail::EffectiveRhoComputation computation;
            {
                auto timer = timing_.scoped("effective_rho.compute_seconds");
                computation =
                    record.configured_rho_ignored_for_inner_acceptance
                        ? detail::compute_required_effective_rho(
                              record.y_estimator_squared,
                              record.lambda_y_squared,
                              record.initial_trace_squared)
                        : detail::compute_effective_rho(
                              record.y_estimator_squared,
                              record.lambda_y_squared,
                              record.initial_trace_squared,
                              parameters_.rho);
            }
            timing_.add("effective_rho.compute_count", 1.0);

            record.effective_rho = computation.rho;
            record.effective_rho_available = computation.available;
            record.effective_rho_reason = computation.reason;
            record.y_estimator_threshold_effective_rho_squared =
                computation.threshold_squared;

            if (record.effective_rho_acceptance_used &&
                !record.effective_rho_available)
            {
                throw std::runtime_error(
                    "AdaptiveDriver: force_accept_inner_with_effective_rho "
                    "could not compute a finite effective rho (" +
                    record.effective_rho_reason + ").");
            }

            if (record.effective_rho_available &&
                record.effective_rho.has_value())
            {
                {
                    auto timer =
                        timing_.scoped(
                            "posteriori_effective_rho.compute_seconds");
                    record.posteriori_factor_effective_rho =
                        detail::posteriori_factor(*record.effective_rho);
                    record.posteriori_estimator_effective_rho_squared =
                        *record.posteriori_factor_effective_rho *
                        record.eta_squared;
                    if (*record.posteriori_estimator_effective_rho_squared >
                        0.0)
                    {
                        record.posteriori_improvement_factor =
                            record
                                .posteriori_estimator_configured_rho_squared /
                            *record
                                 .posteriori_estimator_effective_rho_squared;
                    }
                    else
                    {
                        record.posteriori_improvement_factor.reset();
                    }
                }
            }
            else
            {
                record.posteriori_factor_effective_rho.reset();
                record.posteriori_estimator_effective_rho_squared.reset();
                record.posteriori_improvement_factor.reset();
            }
        }

        void stamp_inner_timing_(
            InnerRecord& record,
            const Clock::time_point& iteration_start,
            const Clock::time_point& run_start) const
        {
            const auto now = Clock::now();
            record.iteration_seconds =
                std::chrono::duration<double>(now - iteration_start).count();
            record.elapsed_seconds =
                std::chrono::duration<double>(now - run_start).count();
        }

        void stamp_outer_timing_(
            OuterRecord& record,
            const Clock::time_point& iteration_start,
            const Clock::time_point& run_start) const
        {
            const auto now = Clock::now();
            record.iteration_seconds =
                std::chrono::duration<double>(now - iteration_start).count();
            record.elapsed_seconds =
                std::chrono::duration<double>(now - run_start).count();
        }

        void append_partial_timing_history_(
            int outer_iteration,
            int inner_iteration,
            const Clock::time_point& run_start,
            std::string_view history_status) const
        {
            const double elapsed_seconds = elapsed_wall_seconds_(run_start);
            const int x_active_cells =
                static_cast<int>(x_space_.active_cells().size());
            const int y_active_cells =
                static_cast<int>(y_space_.active_cells().size());
            const int x_true_dofs =
                x_space_.dof_handler_ref().n_true_dofs();
            const int y_true_dofs =
                y_space_.dof_handler_ref().n_true_dofs();

            measure_history_phase_(
                "adaptive_driver.partial_timing_history_writing",
                [&]()
                {
                    append_partial_timing_history_csv(
                        parameters_.output,
                        outer_iteration,
                        inner_iteration,
                        history_status,
                        elapsed_seconds,
                        x_active_cells,
                        y_active_cells,
                        x_true_dofs,
                        y_true_dofs);
                    const auto timing_records =
                        timing_.records_snapshot();
                    append_partial_timing_breakdown_csv(
                        parameters_.output,
                        outer_iteration,
                        inner_iteration,
                        history_status,
                        elapsed_seconds,
                        x_active_cells,
                        y_active_cells,
                        x_true_dofs,
                        y_true_dofs,
                        timing_records);
                });
        }

        void append_partial_outer_history_(
            OuterRecord& record,
            const Clock::time_point& iteration_start,
            const Clock::time_point& run_start,
            std::string_view history_status) const
        {
            stamp_outer_timing_(record, iteration_start, run_start);
            if (is_terminal_history_status_(history_status))
            {
                timing_.add(
                    "adaptive_driver.outer_iteration.total_wall",
                    record.iteration_seconds);
            }
            measure_history_phase_(
                "adaptive_driver.partial_outer_history_writing",
                [&]()
                {
                    append_partial_outer_history_csv(
                        record,
                        parameters_.output,
                        history_status);
                });
            append_partial_timing_history_(
                record.outer_iteration,
                -1,
                run_start,
                history_status);
        }

        void append_partial_inner_history_(
            int outer_iteration,
            InnerRecord& record,
            const Clock::time_point& iteration_start,
            const Clock::time_point& run_start,
            std::string_view history_status) const
        {
            stamp_inner_timing_(record, iteration_start, run_start);
            if (is_terminal_history_status_(history_status))
            {
                timing_.add(
                    "adaptive_driver.inner_iteration.total_wall",
                    record.iteration_seconds);
            }
            measure_history_phase_(
                "adaptive_driver.partial_inner_history_writing",
                [&]()
                {
                    append_partial_inner_history_csv(
                        outer_iteration,
                        record,
                        parameters_.output,
                        history_status);
                });
            append_partial_timing_history_(
                outer_iteration,
                record.inner_iteration,
                run_start,
                history_status);
        }

        void set_final_state_(
            ResultType& result,
            const YIterationState& state) const
        {
            result.final_lambda_delta.emplace(state.lambda_delta);
            result.final_u_delta.emplace(state.u_delta);
        }

        void finish_current_outer_with_time_budget_(
            ResultType& result,
            OuterRecord& outer_record,
            const Clock::time_point& outer_start,
            const Clock::time_point& run_start,
            AlgorithmLogger& logger,
            const YIterationState& last_state,
            const XMarkingIndicatorComponents<int>& last_x_indicators)
        {
            populate_outer_record_from_last_inner_(outer_record);
            outer_record.x_indicator_components = last_x_indicators;
            update_outer_x_counts_after_(outer_record);
            stamp_outer_timing_(outer_record, outer_start, run_start);

            result.converged = false;
            result.terminated_early = true;
            result.termination_reason =
                make_wall_time_budget_reason_(run_start);
            log_outer_iteration_summary_(logger, outer_record);
            append_partial_outer_history_(
                outer_record,
                outer_start,
                run_start,
                "interrupted");
            result.outer_iterations.push_back(std::move(outer_record));
            set_final_state_(result, last_state);
            result.final_y_estimator =
                result.outer_iterations.back().final_y_estimator;
            result.final_x_indicators =
                result.outer_iterations.back().x_indicator_components;
            finalize_output_(result, logger);
        }

        void finish_current_outer_with_memory_budget_(
            ResultType& result,
            OuterRecord& outer_record,
            const Clock::time_point& outer_start,
            const Clock::time_point& run_start,
            AlgorithmLogger& logger,
            const YIterationState& last_state,
            const XMarkingIndicatorComponents<int>& last_x_indicators,
            const std::string& reason)
        {
            populate_outer_record_from_last_inner_(outer_record);
            outer_record.x_indicator_components = last_x_indicators;
            update_outer_x_counts_after_(outer_record);
            stamp_outer_timing_(outer_record, outer_start, run_start);

            result.converged = false;
            result.terminated_early = true;
            result.termination_reason = reason;
            log_outer_iteration_summary_(logger, outer_record);
            append_partial_outer_history_(
                outer_record,
                outer_start,
                run_start,
                "failed");
            result.outer_iterations.push_back(std::move(outer_record));
            set_final_state_(result, last_state);
            result.final_y_estimator =
                result.outer_iterations.back().final_y_estimator;
            result.final_x_indicators =
                result.outer_iterations.back().x_indicator_components;
            finalize_output_(result, logger);
        }

        void install_space_timing_hooks_()
        {
            x_space_.set_refinement_edge_query_cache_enabled(
                parameters_.refinement_edge_query_cache);
            y_space_.set_refinement_edge_query_cache_enabled(
                parameters_.refinement_edge_query_cache);
            x_space_.set_refinement_batch_target_split_cells(
                static_cast<std::size_t>(
                    std::max(1, parameters_.refinement_batch_target_split_cells)));
            y_space_.set_refinement_batch_target_split_cells(
                static_cast<std::size_t>(
                    std::max(1, parameters_.refinement_batch_target_split_cells)));
            auto post_flush_closure_mode =
                finite_element::FESpacePostFlushClosureMode2D::AffectedEdges;
            if (parameters_.post_flush_closure_mode == "off_debug")
            {
                post_flush_closure_mode =
                    finite_element::FESpacePostFlushClosureMode2D::OffDebug;
            }
            else if (
                parameters_.post_flush_closure_mode ==
                "split_edges_only_debug")
            {
                post_flush_closure_mode =
                    finite_element::FESpacePostFlushClosureMode2D::
                        SplitEdgesOnlyDebug;
            }
            else if (
                parameters_.post_flush_closure_mode ==
                "split_and_inherited_edges")
            {
                post_flush_closure_mode =
                    finite_element::FESpacePostFlushClosureMode2D::
                        SplitAndInheritedEdges;
            }
            else if (
                parameters_.post_flush_closure_mode == "presplit_neighbour")
            {
                post_flush_closure_mode =
                    finite_element::FESpacePostFlushClosureMode2D::
                        PreSplitNeighbour;
            }
            else if (parameters_.post_flush_closure_mode == "all_faces_debug")
            {
                post_flush_closure_mode =
                    finite_element::FESpacePostFlushClosureMode2D::
                        AllFacesDebug;
            }
            x_space_.set_post_flush_closure_mode_2d(
                post_flush_closure_mode);
            y_space_.set_post_flush_closure_mode_2d(
                post_flush_closure_mode);
            x_space_.set_post_flush_affected_containment_only(
                parameters_.post_flush_affected_containment_only);
            y_space_.set_post_flush_affected_containment_only(
                parameters_.post_flush_affected_containment_only);
            x_space_.set_full_conformity_check_after_refinement(
                parameters_.refinement_full_conformity_check);
            y_space_.set_full_conformity_check_after_refinement(
                parameters_.refinement_full_conformity_check);
            auto main_closure_query_mode =
                finite_element::FESpaceMainClosureQueryMode2D::
                    ExactAndAncestor;
            if (parameters_.refinement_main_closure_query_mode ==
                "exact_ancestor_plus_containment")
            {
                main_closure_query_mode =
                    finite_element::FESpaceMainClosureQueryMode2D::
                        ExactAncestorPlusContainment;
            }
            else if (
                parameters_.refinement_main_closure_query_mode ==
                "old_bidirectional_debug")
            {
                main_closure_query_mode =
                    finite_element::FESpaceMainClosureQueryMode2D::
                        OldBidirectionalDebug;
            }
            x_space_.set_main_closure_query_mode_2d(
                main_closure_query_mode);
            y_space_.set_main_closure_query_mode_2d(
                main_closure_query_mode);

            if (!parameters_.timing.enabled)
            {
                clear_space_timing_hooks_();
                return;
            }

            const auto fespace_diagnostic_level =
                parameters_.timing.detail_level == "detailed"
                    ? finite_element::FESpaceDiagnosticLevel::Detailed
                    : finite_element::FESpaceDiagnosticLevel::Summary;
            x_space_.set_diagnostic_level(fespace_diagnostic_level);
            y_space_.set_diagnostic_level(fespace_diagnostic_level);

            x_space_.set_timing_callback(
                [this](std::string_view phase, double seconds)
                {
                    record_space_timing_("x", phase, seconds);
                });
            y_space_.set_timing_callback(
                [this](std::string_view phase, double seconds)
                {
                    record_space_timing_("y", phase, seconds);
                });
        }

        void clear_space_timing_hooks_()
        {
            x_space_.clear_timing_callback();
            y_space_.clear_timing_callback();
            x_space_.set_diagnostic_level(
                finite_element::FESpaceDiagnosticLevel::None);
            y_space_.set_diagnostic_level(
                finite_element::FESpaceDiagnosticLevel::None);
        }

        void record_space_timing_(
            std::string_view space_name,
            std::string_view phase,
            double seconds)
        {
            timing_.add(make_space_timing_phase_name_(space_name, phase), seconds);
        }

        [[nodiscard]] finite_element::detail::TimingRecorder make_timing_recorder_()
        {
            if (!parameters_.timing.enabled)
                return {};

            return finite_element::detail::TimingRecorder(
                [this](std::string_view phase, double seconds)
                {
                    timing_.add(phase, seconds);
                });
        }

        [[nodiscard]] static std::string make_space_timing_phase_name_(
            std::string_view space_name,
            std::string_view phase)
        {
            std::string name;
            name.reserve(space_name.size() + 1 + phase.size());
            name.append(space_name.data(), space_name.size());
            name.push_back('.');
            name.append(phase.data(), phase.size());
            return name;
        }

        [[nodiscard]] static bool is_terminal_history_status_(
            std::string_view status) noexcept
        {
            return status == "completed" ||
                status == "failed" ||
                status == "interrupted";
        }

        void finalize_output_(
            ResultType& result,
            AlgorithmLogger& logger)
        {
            const auto final_output_start = Clock::now();
            result.timing_enabled = parameters_.timing.enabled;
            result.timing_detail_level = parameters_.timing.detail_level;
            log_completion_(logger, result);

            measure_history_phase_(
                "adaptive_driver.final_history_writing",
                [&]()
                {
                    auto timer = timing_.scoped("output.summary_text");
                    write_summary_text(result, parameters_.output);
                });
            measure_history_phase_(
                "adaptive_driver.final_history_writing",
                [&]()
                {
                    auto timer = timing_.scoped("output.outer_history_csv");
                    write_outer_history_csv(result, parameters_.output);
                });
            measure_history_phase_(
                "adaptive_driver.final_history_writing",
                [&]()
                {
                    auto timer = timing_.scoped("output.inner_history_csv");
                    write_inner_history_csv(result, parameters_.output);
                });
            measure_history_phase_(
                "adaptive_driver.final_history_writing",
                [&]()
                {
                    auto timer =
                        timing_.scoped("output.refinement_history_text");
                    write_refinement_history_text(result, parameters_.output);
                });

            if (parameters_.timing.enabled)
            {
                result.timing_records = timing_.records_snapshot();

                const auto timing_write_start = Clock::now();
                write_timing_history_csv(result, parameters_.output);
                const auto timing_write_seconds =
                    std::chrono::duration<double>(
                        Clock::now() - timing_write_start)
                        .count();
                timing_.add("output.timing_history_csv", timing_write_seconds);
                timing_.add(
                    "adaptive_driver.final_history_writing",
                    timing_write_seconds);
                timing_.add(
                    "adaptive_driver.history_writing.total",
                    timing_write_seconds);
                timing_.add(
                    "adaptive_driver.bookkeeping.total",
                    timing_write_seconds);
                timing_.add(
                    "adaptive_driver.output_and_history_total_wall",
                    timing_write_seconds);

                record_since_(
                    "adaptive_driver.final_output_wall",
                    final_output_start);
                record_active_run_total_wall_();
                result.timing_records = timing_.records_snapshot();
                write_timing_history_csv(result, parameters_.output);
            }
            else
            {
                result.timing_records = {};
            }
        }

        void finalize_after_exception_(
            ResultType& result,
            std::optional<OuterRecord>& pending_outer_record,
            std::optional<Clock::time_point>& pending_outer_start,
            AlgorithmLogger& logger,
            const std::string& reason,
            const Clock::time_point& run_start)
        {
            if (pending_outer_record.has_value())
            {
                auto& outer_record = *pending_outer_record;
                if (!outer_record.inner_iterations.empty())
                {
                    populate_outer_record_from_last_inner_(outer_record);
                    update_outer_x_counts_after_(outer_record);
                    if (pending_outer_start.has_value())
                    {
                        stamp_outer_timing_(
                            outer_record,
                            *pending_outer_start,
                            run_start);
                        append_partial_outer_history_(
                            outer_record,
                            *pending_outer_start,
                            run_start,
                            "failed");
                    }
                    result.outer_iterations.push_back(std::move(outer_record));
                }
                else if (pending_outer_start.has_value())
                {
                    update_current_outer_counts_after_(outer_record);
                    append_partial_outer_history_(
                        outer_record,
                        *pending_outer_start,
                        run_start,
                        "failed");
                }
                pending_outer_record.reset();
                pending_outer_start.reset();
            }

            result.converged = false;
            result.terminated_early = true;
            result.termination_reason = reason;

            try
            {
                finalize_output_(result, logger);
            }
            catch (...)
            {
            }
        }

        XSpaceType& x_space_;
        YSpaceType& y_space_;
        const ProblemDataType& problem_;
        Solver& solver_;
        la::concepts::SolverOptions main_solver_options_{};
        la::concepts::SolverOptions g_solver_options_{};
        la::concepts::SolverOptions local_solver_options_{};
        AdaptiveParameters parameters_{};
        mutable TimingCollector timing_{};
        MainTwoPassAssemblyCache main_two_pass_assembly_cache_{};
        mutable std::optional<Clock::time_point> active_run_start_{};
    };

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class YSpaceType,
        class ProblemDataType>
    [[nodiscard]] AdaptiveResult<Backend, XSpaceType, YSpaceType>
    run_adaptive_driver(
        XSpaceType& x_space,
        YSpaceType& y_space,
        const ProblemDataType& problem,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& main_solver_options = {},
        AdaptiveParameters parameters = {},
        std::optional<la::concepts::SolverOptions> g_solver_options =
            std::nullopt)
    {
        AdaptiveDriver<QSpace, QTime, Backend, XSpaceType, YSpaceType, ProblemDataType>
            driver(
                x_space,
                y_space,
                problem,
                solver,
                main_solver_options,
                std::move(parameters),
                std::move(g_solver_options));

        return driver.run();
    }
}

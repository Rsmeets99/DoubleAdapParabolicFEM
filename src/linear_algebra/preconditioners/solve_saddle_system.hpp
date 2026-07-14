#pragma once

#include <optional>
#include <stdexcept>

#include "linear_algebra/concepts/solver.hpp"
#include "linear_algebra/preconditioners/schur_approximations.hpp"
#include "linear_algebra/preconditioners/solve_failure.hpp"
#include "linear_algebra/system/saddle_point_system.hpp"

namespace la::preconditioners
{
    [[nodiscard]] inline bool is_parabolic_graph_norm_minres_solve_requested(
        const la::concepts::SolverOptions& options) noexcept
    {
        return options.solver == la::concepts::SolverType::MINRES &&
               options.preconditioner ==
                    la::concepts::PreconditionerType::ParabolicGraphNorm;
    }

    [[nodiscard]] inline bool is_saddle_preconditioned_solve_requested(
        const la::concepts::SolverOptions& options) noexcept
    {
        return is_parabolic_graph_norm_minres_solve_requested(options);
    }

    template<class Backend>
    struct SaddlePreconditionerContext
    {
        const la::preconditioners::ParabolicGraphNormApproximation<Backend>*
            graph_norm = nullptr;
    };

    template<class Backend>
    struct SaddlePreconditionedSolveResult
    {
        typename Backend::Vector solution{};
        la::concepts::SolverDiagnostics diagnostics{};
        double setup_seconds = 0.0;
        double solve_seconds = 0.0;
    };

    template<class Backend>
    struct SaddlePreconditionedSolveDispatcher
    {
        template<
            class Solver,
            class Matrix,
            class Vector,
            class InitialGuess>
        [[nodiscard]] static std::optional<
            SaddlePreconditionedSolveResult<Backend>>
        solve(
            Solver& solver,
            const Matrix& K,
            const Vector& rhs,
            const la::saddle::SaddlePointBlocks<Backend>& blocks,
            const la::concepts::SolverOptions& options,
            const InitialGuess& initial_guess,
            const SaddlePreconditionerContext<Backend>& context)
        {
            static_cast<void>(solver);
            static_cast<void>(K);
            static_cast<void>(rhs);
            static_cast<void>(blocks);
            static_cast<void>(initial_guess);
            static_cast<void>(context);

            const auto validation =
                la::concepts::validate_solver_preconditioner(
                    options.solver,
                    options.preconditioner);
            if (!validation.accepted)
                throw std::invalid_argument(validation.rejection_reason);

            if (is_saddle_preconditioned_solve_requested(options))
            {
                throw std::invalid_argument(
                    "Saddle-block preconditioned solves are not implemented for this backend.");
            }

            return std::nullopt;
        }
    };

    template<
        class Backend,
        class Solver,
        class Matrix,
        class Vector,
        class InitialGuess>
    [[nodiscard]] inline std::optional<
        SaddlePreconditionedSolveResult<Backend>>
    solve_saddle_system_with_preconditioner(
        Solver& solver,
        const Matrix& K,
        const Vector& rhs,
        const la::saddle::SaddlePointBlocks<Backend>& blocks,
        const la::concepts::SolverOptions& options,
        const InitialGuess& initial_guess,
        const SaddlePreconditionerContext<Backend>& context)
    {
        return SaddlePreconditionedSolveDispatcher<Backend>::solve(
            solver,
            K,
            rhs,
            blocks,
            options,
            initial_guess,
            context);
    }
}

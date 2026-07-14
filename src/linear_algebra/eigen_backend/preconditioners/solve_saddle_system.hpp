#pragma once

#include <optional>
#include <stdexcept>
#include <vector>

#include "parabolic_graph_norm_minres.hpp"
#include "linear_algebra/eigen_backend/backend_types.hpp"
#include "linear_algebra/operations/linalg_ops.hpp"
#include "linear_algebra/preconditioners/solve_saddle_system.hpp"
#include "linear_algebra/system/saddle_point_system.hpp"

namespace la::preconditioners
{
    namespace detail
    {
        [[nodiscard]] inline SaddlePreconditionedSolveResult<la::eigen::Backend>
        make_result_from_minres(
            la::eigen::preconditioners::ParabolicGraphNormMinresResult&&
                minres_result)
        {
            SaddlePreconditionedSolveResult<la::eigen::Backend> result;
            result.solution = std::move(minres_result.solution);
            result.diagnostics = std::move(minres_result.diagnostics);
            result.setup_seconds = minres_result.setup_seconds;
            result.solve_seconds = minres_result.solve_seconds;
            return result;
        }

    }

    template<>
    struct SaddlePreconditionedSolveDispatcher<la::eigen::Backend>
    {
        template<class InitialGuess>
        [[nodiscard]] static std::optional<
            SaddlePreconditionedSolveResult<la::eigen::Backend>>
        solve(
            la::eigen::Solver& solver,
            const la::eigen::SparseMatrix& K,
            const la::eigen::Vector& rhs,
            const la::saddle::SaddlePointBlocks<la::eigen::Backend>& blocks,
            const la::concepts::SolverOptions& options,
            const InitialGuess& initial_guess,
            const SaddlePreconditionerContext<la::eigen::Backend>& context)
        {
            static_cast<void>(solver);

            const auto validation =
                la::concepts::validate_solver_preconditioner(
                    options.solver,
                    options.preconditioner);
            if (!validation.accepted)
                throw std::invalid_argument(validation.rejection_reason);

            if (!is_saddle_preconditioned_solve_requested(options))
                return std::nullopt;

            blocks.validate();

            if (K.rows() != K.cols() || K.rows() != rhs.size())
            {
                throw std::runtime_error(
                    "solve_saddle_system_with_preconditioner: system dimensions are inconsistent.");
            }
            if (initial_guess.has_value() &&
                initial_guess.full_vector->size() != rhs.size())
            {
                throw std::runtime_error(
                    "solve_saddle_system_with_preconditioner: initial guess size does not match the full saddle system.");
            }

            const la::eigen::Vector* initial_guess_ptr =
                initial_guess.has_value() ? initial_guess.full_vector : nullptr;
            la::eigen::Vector residual;
            const la::eigen::Vector* rhs_for_solve = &rhs;
            if (initial_guess.has_value() &&
                initial_guess.solve_correction_equation)
            {
                residual =
                    la::ops::subtract(
                        rhs,
                        la::ops::matvec(K, *initial_guess.full_vector));
                rhs_for_solve = &residual;
                initial_guess_ptr = nullptr;
            }

            SaddlePreconditionedSolveResult<la::eigen::Backend> result;

            if (is_parabolic_graph_norm_minres_solve_requested(options))
            {
                const auto* graph_norm = context.graph_norm;

                if (graph_norm == nullptr)
                {
                    throw std::runtime_error(
                        "solve_saddle_system_with_preconditioner: ParabolicGraphNorm requires graph-norm block context from assembly.");
                }

                auto minres_result =
                    la::eigen::preconditioners::solve_minres_with_parabolic_graph_norm(
                        K,
                        blocks,
                        *graph_norm,
                        *rhs_for_solve,
                        options,
                        initial_guess_ptr);
                result =
                    detail::make_result_from_minres(std::move(minres_result));
            }

            if (initial_guess.has_value() &&
                initial_guess.solve_correction_equation)
            {
                result.solution =
                    la::ops::add(*initial_guess.full_vector, result.solution);
            }

            return result;
        }
    };
}

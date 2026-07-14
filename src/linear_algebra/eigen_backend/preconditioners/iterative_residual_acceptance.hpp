#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <Eigen/Dense>

#include "linear_algebra/concepts/solver.hpp"
#include "linear_algebra/eigen_backend/backend_types.hpp"

namespace la::eigen::preconditioners
{
    struct MonolithicResidualDiagnostics
    {
        double absolute = 0.0;
        double relative = 0.0;
    };

    struct ResidualCheckedIterativeSolveSummary
    {
        int iterations = 0;
        int batches = 0;
        double backend_reported_error =
            std::numeric_limits<double>::infinity();
        bool backend_converged = false;
        bool true_residual_stopping_used = false;
        MonolithicResidualDiagnostics residual{};
    };

    [[nodiscard]] inline MonolithicResidualDiagnostics
    compute_monolithic_residual_diagnostics(
        const la::eigen::SparseMatrix& K,
        const la::eigen::Vector& rhs,
        const la::eigen::Vector& solution)
    {
        if (K.rows() != K.cols() || K.rows() != rhs.size() ||
            solution.size() != rhs.size())
        {
            throw std::runtime_error(
                "compute_monolithic_residual_diagnostics: inconsistent system dimensions.");
        }

        const Eigen::VectorXd residual =
            K.native() * solution.native() - rhs.native();
        const double residual_norm = residual.norm();
        const double rhs_norm = rhs.native().norm();
        // With a zero RHS there is no meaningful relative scale. Use one so
        // the reported relative residual equals the absolute residual.
        const double denominator = rhs_norm > 0.0 ? rhs_norm : 1.0;

        return {residual_norm, residual_norm / denominator};
    }

    inline void record_iterative_backend_status(
        la::concepts::SolverDiagnostics& diagnostics,
        int iterations,
        double backend_reported_error,
        bool backend_converged)
    {
        diagnostics.iterative_stats.iterations = iterations;
        diagnostics.iterative_stats.final_error = backend_reported_error;
        diagnostics.iterative_stats.converged = backend_converged;
        diagnostics.iterative_stats.backend_converged = backend_converged;
        diagnostics.iterative_stats.backend_reported_error =
            backend_reported_error;
        diagnostics.iterative_stats.convergence_accepted_by_true_residual =
            false;
    }

    [[nodiscard]] inline bool record_true_residual_acceptance(
        la::concepts::SolverDiagnostics& diagnostics,
        const la::eigen::SparseMatrix& K,
        const la::eigen::Vector& rhs,
        const la::eigen::Vector& solution,
        double tolerance)
    {
        const auto residual =
            compute_monolithic_residual_diagnostics(K, rhs, solution);
        diagnostics.linear_residual_absolute = residual.absolute;
        diagnostics.linear_residual_relative = residual.relative;
        diagnostics.iterative_stats.final_true_residual = residual.relative;

        const bool accepted_by_true_residual =
            std::isfinite(residual.relative) &&
            residual.relative <= tolerance;

        diagnostics.iterative_stats.convergence_accepted_by_true_residual =
            accepted_by_true_residual;
        diagnostics.iterative_stats.true_residual_stopping_used = true;
        diagnostics.iterative_stats.converged = accepted_by_true_residual;
        diagnostics.iterative_stats.final_error = residual.relative;

        return diagnostics.iterative_stats.converged.value_or(false);
    }

    inline void record_initial_guess_convergence(
        la::concepts::SolverDiagnostics& diagnostics,
        double relative_residual)
    {
        diagnostics.iterative_stats.iterations = 0;
        diagnostics.iterative_stats.final_error = relative_residual;
        diagnostics.iterative_stats.converged = true;
        diagnostics.iterative_stats.backend_converged = true;
        diagnostics.iterative_stats.backend_reported_error =
            relative_residual;
        diagnostics.iterative_stats.convergence_accepted_by_true_residual =
            true;
        diagnostics.iterative_stats.residual_check_batches = 0;
        diagnostics.iterative_stats.final_true_residual = relative_residual;
        diagnostics.iterative_stats.true_residual_stopping_used = true;
        diagnostics.linear_residual_relative = relative_residual;
    }

    template<class IterativeSolver>
    [[nodiscard]] inline ResidualCheckedIterativeSolveSummary
    solve_with_true_residual_batches(
        IterativeSolver& solver,
        const la::eigen::SparseMatrix& K,
        const la::eigen::Vector& rhs,
        const Eigen::VectorXd& initial_guess,
        int max_iterations,
        int requested_batch_size,
        double tolerance,
        la::eigen::Vector& solution,
        la::concepts::SolverDiagnostics& diagnostics)
    {
        if (initial_guess.size() != rhs.size())
        {
            throw std::runtime_error(
                "solve_with_true_residual_batches: initial guess size mismatch.");
        }

        const int iteration_limit = std::max(1, max_iterations);
        const int batch_size =
            std::min(
                iteration_limit,
                std::max(1, requested_batch_size));
        const double target_tolerance = std::max(0.0, tolerance);
        const double minimum_backend_tolerance =
            std::max(
                std::numeric_limits<double>::epsilon(),
                target_tolerance *
                    std::sqrt(std::numeric_limits<double>::epsilon()));

        Eigen::VectorXd current = initial_guess;
        ResidualCheckedIterativeSolveSummary summary;
        double backend_tolerance = target_tolerance;

        while (summary.iterations < iteration_limit)
        {
            const int remaining = iteration_limit - summary.iterations;
            const int current_batch_size = std::min(batch_size, remaining);
            solver.setMaxIterations(current_batch_size);
            solver.setTolerance(backend_tolerance);

            solution.native() = solver.solveWithGuess(rhs.native(), current);

            ++summary.batches;
            summary.iterations +=
                std::max(0, static_cast<int>(solver.iterations()));
            summary.backend_reported_error = solver.error();
            summary.backend_converged = solver.info() == Eigen::Success;

            record_iterative_backend_status(
                diagnostics,
                summary.iterations,
                summary.backend_reported_error,
                summary.backend_converged);
            const bool accepted_by_true_residual =
                record_true_residual_acceptance(
                    diagnostics,
                    K,
                    rhs,
                    solution,
                    target_tolerance);

            summary.residual.absolute =
                diagnostics.linear_residual_absolute.value_or(
                    std::numeric_limits<double>::infinity());
            summary.residual.relative =
                diagnostics.linear_residual_relative.value_or(
                    std::numeric_limits<double>::infinity());
            summary.true_residual_stopping_used =
                diagnostics.iterative_stats
                    .true_residual_stopping_used.value_or(false);
            diagnostics.iterative_stats.residual_check_batches =
                summary.batches;
            diagnostics.iterative_stats.final_true_residual =
                summary.residual.relative;
            diagnostics.iterative_stats.true_residual_stopping_used =
                summary.true_residual_stopping_used;

            if (accepted_by_true_residual)
                return summary;

            if (!std::isfinite(summary.residual.relative) ||
                !std::isfinite(summary.backend_reported_error) ||
                solver.iterations() <= 0)
            {
                return summary;
            }

            current = solution.native();
            if (summary.backend_converged)
            {
                backend_tolerance =
                    std::max(
                        minimum_backend_tolerance,
                        backend_tolerance * 0.1);
            }
        }

        return summary;
    }
}

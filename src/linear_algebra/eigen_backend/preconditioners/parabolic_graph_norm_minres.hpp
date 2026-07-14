#pragma once

#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <unsupported/Eigen/IterativeSolvers>

#include "linear_algebra/concepts/solver.hpp"
#include "linear_algebra/eigen_backend/backend_types.hpp"
#include "linear_algebra/eigen_backend/operations.hpp"
#include "linear_algebra/eigen_backend/preconditioners/iterative_residual_acceptance.hpp"
#include "linear_algebra/preconditioners/schur_approximations.hpp"
#include "linear_algebra/preconditioners/solve_failure.hpp"
#include "linear_algebra/system/saddle_point_system.hpp"

namespace la::eigen::preconditioners
{
    using ParabolicGraphNormApproximation =
        la::preconditioners::ParabolicGraphNormApproximation<
            la::eigen::Backend>;
    using ParabolicGraphNormPreconditionerDiagnostics =
        la::preconditioners::ParabolicGraphNormPreconditionerDiagnostics;

    [[nodiscard]] inline ParabolicGraphNormApproximation
    parabolic_graph_norm_approximation(
        const la::eigen::SparseMatrix& A_y,
        const la::eigen::SparseMatrix& B_dt,
        const la::eigen::SparseMatrix& C_signed,
        const la::eigen::SparseMatrix& A_x,
        double diagonal_tolerance = 1.0e-14)
    {
        return la::preconditioners::parabolic_graph_norm_approximation<
            la::eigen::Backend>(
                A_y,
                B_dt,
                C_signed,
                A_x,
                diagonal_tolerance);
    }

    class ParabolicGraphNormBlockPreconditioner
    {
    public:
        using MatrixType = la::eigen::SparseMatrix::native_type;
        using VectorType = Eigen::VectorXd;
        using Scalar = double;

        ParabolicGraphNormBlockPreconditioner() = default;

        void compute_from_graph_norm(
            const la::eigen::SparseMatrix& A_y,
            const ParabolicGraphNormApproximation& graph_norm)
        {
            const std::chrono::steady_clock::time_point setup_start =
                std::chrono::steady_clock::now();

            diagnostics_ = ParabolicGraphNormPreconditionerDiagnostics{};
            info_ = Eigen::NumericalIssue;

            if (A_y.rows() != A_y.cols())
                throw std::runtime_error(
                    "ParabolicGraphNormBlockPreconditioner: A_y must be square.");
            if (graph_norm.hhat.rows() != graph_norm.hhat.cols())
                throw std::runtime_error(
                    "ParabolicGraphNormBlockPreconditioner: Hhat must be square.");

            n_lambda_ = A_y.rows();
            n_u_ = graph_norm.hhat.rows();
            hhat_ = graph_norm.hhat;

            A_y_solver_.compute(A_y.native());
            if (A_y_solver_.info() != Eigen::Success)
            {
                throw std::runtime_error(
                    "ParabolicGraphNormBlockPreconditioner: A_y is not SPD for the internal block factorization.");
            }

            hhat_solver_.compute(hhat_.native());
            if (hhat_solver_.info() != Eigen::Success)
            {
                throw std::runtime_error(
                    "ParabolicGraphNormBlockPreconditioner: Hhat is not SPD for the internal block factorization.");
            }

            const std::chrono::steady_clock::time_point setup_end =
                std::chrono::steady_clock::now();

            diagnostics_.is_spd = true;
            diagnostics_.n_lambda = n_lambda_;
            diagnostics_.n_u = n_u_;
            diagnostics_.hhat_rows = hhat_.rows();
            diagnostics_.hhat_cols = hhat_.cols();
            diagnostics_.nnz_hhat =
                static_cast<int>(hhat_.native().nonZeros());
            diagnostics_.setup_seconds =
                std::chrono::duration<double>(setup_end - setup_start).count() +
                graph_norm.diagnostics.setup_seconds;
            diagnostics_.approximation = graph_norm.diagnostics;
            diagnostics_.c_signed_nnz =
                graph_norm.diagnostics.c_signed_nnz;
            diagnostics_.b_or_bdt_nnz_used =
                graph_norm.diagnostics.b_or_bdt_nnz_used;
            diagnostics_.approximate_fill_ratio =
                graph_norm.diagnostics.approximate_fill_ratio;
            info_ = Eigen::Success;
        }

        template<class MatrixDerived>
        ParabolicGraphNormBlockPreconditioner& analyzePattern(
            const MatrixDerived&)
        {
            return *this;
        }

        template<class MatrixDerived>
        ParabolicGraphNormBlockPreconditioner& factorize(
            const MatrixDerived&)
        {
            return *this;
        }

        template<class MatrixDerived>
        ParabolicGraphNormBlockPreconditioner& compute(
            const MatrixDerived&)
        {
            return *this;
        }

        template<class Rhs>
        [[nodiscard]] VectorType solve(const Rhs& rhs) const
        {
            if (info_ != Eigen::Success)
            {
                throw std::runtime_error(
                    "ParabolicGraphNormBlockPreconditioner::solve called before successful setup.");
            }
            if (rhs.rows() != n_lambda_ + n_u_)
            {
                throw std::runtime_error(
                    "ParabolicGraphNormBlockPreconditioner::solve: RHS size mismatch.");
            }

            VectorType z(rhs.rows());
            z.segment(0, n_lambda_) =
                A_y_solver_.solve(rhs.segment(0, n_lambda_));
            z.segment(n_lambda_, n_u_) =
                hhat_solver_.solve(rhs.segment(n_lambda_, n_u_));

            return z;
        }

        [[nodiscard]] Eigen::ComputationInfo info() const noexcept
        {
            return info_;
        }

        [[nodiscard]] const ParabolicGraphNormPreconditionerDiagnostics&
        diagnostics() const noexcept
        {
            return diagnostics_;
        }

        [[nodiscard]] const la::eigen::SparseMatrix& Hhat() const noexcept
        {
            return hhat_;
        }

    private:
        int n_lambda_ = 0;
        int n_u_ = 0;
        la::eigen::SparseMatrix hhat_{};
        Eigen::SimplicialLLT<MatrixType> A_y_solver_{};
        Eigen::SimplicialLLT<MatrixType> hhat_solver_{};
        Eigen::ComputationInfo info_ = Eigen::NumericalIssue;
        ParabolicGraphNormPreconditionerDiagnostics diagnostics_{};
    };

    struct ParabolicGraphNormMinresResult
    {
        la::eigen::Vector solution;
        la::concepts::SolverDiagnostics diagnostics{};
        double setup_seconds = 0.0;
        double solve_seconds = 0.0;
        ParabolicGraphNormPreconditionerDiagnostics
            preconditioner_diagnostics{};
    };

    [[nodiscard]] inline std::string format_parabolic_graph_norm_minres_failure(
        int iterations,
        double backend_reported_error,
        double true_relative_residual)
    {
        std::ostringstream message;
        message << std::scientific << std::setprecision(12)
                << "MINRES(ParabolicGraphNorm) solve failed. iterations="
                << iterations
                << ", error=" << backend_reported_error
                << ", true_relative_residual=" << true_relative_residual;
        return message.str();
    }

    [[nodiscard]] inline bool graph_norm_initial_guess_satisfies_tolerance(
        const la::eigen::SparseMatrix& K,
        const la::eigen::Vector& rhs,
        const la::eigen::Vector& initial_guess,
        double tolerance,
        la::eigen::Vector& solution,
        la::concepts::SolverDiagnostics& diagnostics)
    {
        const Eigen::VectorXd residual =
            rhs.native() - K.native() * initial_guess.native();
        const double rhs_norm = rhs.native().norm();
        const double residual_norm = residual.norm();
        const double relative_error =
            rhs_norm > 0.0 ? residual_norm / rhs_norm : residual_norm;

        if (relative_error > tolerance)
            return false;

        solution.native() = initial_guess.native();
        diagnostics.linear_residual_absolute = residual_norm;
        record_initial_guess_convergence(diagnostics, relative_error);
        return true;
    }

    [[nodiscard]] inline ParabolicGraphNormMinresResult
    solve_minres_with_parabolic_graph_norm(
        const la::eigen::SparseMatrix& K,
        const la::saddle::SaddlePointBlocks<la::eigen::Backend>& blocks,
        const ParabolicGraphNormApproximation& graph_norm,
        const la::eigen::Vector& rhs,
        const la::concepts::SolverOptions& options,
        const la::eigen::Vector* initial_guess = nullptr)
    {
        using MinresSolver = Eigen::MINRES<
            la::eigen::SparseMatrix::native_type,
            Eigen::Lower,
            ParabolicGraphNormBlockPreconditioner>;

        if (options.solver != la::concepts::SolverType::MINRES ||
            options.preconditioner !=
                la::concepts::PreconditionerType::ParabolicGraphNorm)
        {
            throw std::runtime_error(
                "solve_minres_with_parabolic_graph_norm requires MINRES with ParabolicGraphNorm preconditioner.");
        }
        if (K.rows() != K.cols() || K.rows() != rhs.size())
        {
            throw std::runtime_error(
                "solve_minres_with_parabolic_graph_norm: system dimensions are inconsistent.");
        }
        if (initial_guess != nullptr && initial_guess->size() != rhs.size())
        {
            throw std::runtime_error(
                "solve_minres_with_parabolic_graph_norm: initial guess size mismatch.");
        }

        ParabolicGraphNormMinresResult result;
        result.solution.resize(rhs.size());
        result.solution.set_zero();

        result.diagnostics.requested_solver = options.solver;
        result.diagnostics.effective_solver = options.solver;
        result.diagnostics.preconditioner = options.preconditioner;
        result.diagnostics.rows = K.rows();
        result.diagnostics.cols = K.cols();
        result.diagnostics.nnz_matrix =
            static_cast<std::size_t>(K.native().nonZeros());
        result.diagnostics.direct = false;
        result.diagnostics.iterative = true;

        MinresSolver minres;
        minres.setMaxIterations(options.max_iterations);
        minres.setTolerance(options.tolerance);

        const std::chrono::steady_clock::time_point setup_start =
            std::chrono::steady_clock::now();
        minres.preconditioner().compute_from_graph_norm(
            blocks.A,
            graph_norm);
        minres.compute(K.native());
        const std::chrono::steady_clock::time_point setup_end =
            std::chrono::steady_clock::now();

        if (minres.info() != Eigen::Success)
        {
            throw std::runtime_error(
                "MINRES(ParabolicGraphNorm) initialization failed.");
        }

        result.preconditioner_diagnostics =
            minres.preconditioner().diagnostics();
        result.diagnostics.preconditioner_setup_seconds =
            result.preconditioner_diagnostics.setup_seconds;
        result.diagnostics.preconditioner_stats.setup_seconds =
            result.preconditioner_diagnostics.setup_seconds;
        result.diagnostics.preconditioner_stats.assumes_spd = true;
        result.diagnostics.preconditioner_stats.assumes_symmetric = true;
        result.diagnostics.preconditioner_stats.is_spd =
            result.preconditioner_diagnostics.is_spd;
        result.diagnostics.preconditioner_stats.is_symmetric =
            result.preconditioner_diagnostics.is_spd;
        result.setup_seconds =
            std::chrono::duration<double>(setup_end - setup_start).count() +
            graph_norm.diagnostics.setup_seconds;

        const std::chrono::steady_clock::time_point solve_start =
            std::chrono::steady_clock::now();
        if (initial_guess != nullptr &&
            graph_norm_initial_guess_satisfies_tolerance(
                K,
                rhs,
                *initial_guess,
                options.tolerance,
                result.solution,
                result.diagnostics))
        {
            const std::chrono::steady_clock::time_point solve_end =
                std::chrono::steady_clock::now();
            result.solve_seconds =
                std::chrono::duration<double>(solve_end - solve_start).count();
            return result;
        }

        const Eigen::VectorXd start_guess =
            initial_guess != nullptr
            ? initial_guess->native()
            : Eigen::VectorXd::Zero(rhs.size());
        const int batch_size =
            options.use_true_residual_batching
            ? options.true_residual_check_interval
            : options.max_iterations;
        const ResidualCheckedIterativeSolveSummary solve_summary =
            solve_with_true_residual_batches(
                minres,
                K,
                rhs,
                start_guess,
                options.max_iterations,
                batch_size,
                options.tolerance,
                result.solution,
                result.diagnostics);
        const std::chrono::steady_clock::time_point solve_end =
            std::chrono::steady_clock::now();

        const bool converged =
            result.diagnostics.iterative_stats.converged.value_or(false);
        result.solve_seconds =
            std::chrono::duration<double>(solve_end - solve_start).count();

        if (!converged)
        {
            throw la::preconditioners::PreconditionedSolveFailure(
                format_parabolic_graph_norm_minres_failure(
                    solve_summary.iterations,
                    solve_summary.backend_reported_error,
                    result.diagnostics.linear_residual_relative.value_or(
                        -1.0)),
                result.diagnostics);
        }

        return result;
    }
}

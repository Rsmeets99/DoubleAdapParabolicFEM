#pragma once

#include <algorithm>
#include <memory>
#include <cmath>
#include <stdexcept>

#include <Eigen/Dense>
#include <Eigen/LU>

#include "dense_matrix.hpp"
#include "linear_algebra/concepts/solver.hpp"
#include "linear_algebra/operations/linalg_ops.hpp"

namespace la::eigen
{
    class DenseDirectSolver
    {
    public:
        using Matrix = la::eigen::DenseMatrix;
        using Vector = la::eigen::Vector;
        using Options = la::concepts::SolverOptions;
        using Diagnostics = la::concepts::SolverDiagnostics;

        void compute(const Matrix& A)
        {
            compute(A, Options{});
        }

        void compute(const Matrix& A, const Options& options)
        {
            if (A.rows() != A.cols())
            {
                throw std::runtime_error(
                    "DenseDirectSolver::compute requires a square matrix.");
            }
            if (A.rows() == 0)
            {
                throw std::runtime_error(
                    "DenseDirectSolver::compute refuses an empty system.");
            }

            computed_ = false;
            matrix_rows_ = A.rows();

            diagnostics_ = Diagnostics{};
            diagnostics_.rows = A.rows();
            diagnostics_.cols = A.cols();
            diagnostics_.requested_solver = options.solver;
            diagnostics_.effective_solver = options.solver;
            diagnostics_.preconditioner = options.preconditioner;
            diagnostics_.nnz_matrix =
                static_cast<std::size_t>(A.rows()) *
                static_cast<std::size_t>(A.cols());
            diagnostics_.direct = true;
            diagnostics_.iterative = false;
            diagnostics_.direct_stats.n = A.rows();
            diagnostics_.direct_stats.nnz_matrix = diagnostics_.nnz_matrix;
            diagnostics_.matrix_norm = A.native().norm();

            switch (options.dense_factorization)
            {
                case la::concepts::DenseFactorizationType::PartialPivotDenseLU:
                    full_piv_lu_.reset();
                    if (!partial_piv_lu_)
                    {
                        partial_piv_lu_ =
                            std::make_unique<
                                Eigen::PartialPivLU<Matrix::native_type>>(
                                A.native());
                    }
                    else
                    {
                        partial_piv_lu_->compute(A.native());
                    }
                    computed_ = true;
                    break;

                case la::concepts::DenseFactorizationType::RankRevealingDenseLU:
                    partial_piv_lu_.reset();
                    if (!full_piv_lu_)
                    {
                        full_piv_lu_ =
                            std::make_unique<
                                Eigen::FullPivLU<Matrix::native_type>>(
                                A.native());
                    }
                    else
                    {
                        full_piv_lu_->compute(A.native());
                    }
                    full_piv_lu_->setThreshold(
                        options.dense_rank_revealing_threshold);
                    if (!full_piv_lu_->isInvertible())
                    {
                        computed_ = false;
                        partial_piv_lu_.reset();
                        full_piv_lu_.reset();
                        throw std::runtime_error(
                            "DenseDirectSolver::compute found a singular matrix.");
                    }
                    computed_ = true;
                    break;
            }
        }

        [[nodiscard]] Vector solve(const Vector& b) const
        {
            Vector x;
            solve(b, x);
            return x;
        }

        void solve(const Vector& b, Vector& x) const
        {
            if (!computed_)
            {
                throw std::runtime_error(
                    "DenseDirectSolver::solve called before compute.");
            }
            if (b.size() != matrix_rows_)
            {
                throw std::runtime_error(
                    "DenseDirectSolver::solve RHS dimension mismatch.");
            }

            x.resize(matrix_rows_);
            if (partial_piv_lu_)
            {
                x.native() = partial_piv_lu_->solve(b.native());
                return;
            }
            if (full_piv_lu_)
            {
                x.native() = full_piv_lu_->solve(b.native());
                return;
            }

            throw std::runtime_error(
                "DenseDirectSolver::solve has no active factorization.");
        }

        [[nodiscard]] const Diagnostics& last_diagnostics() const noexcept
        {
            return diagnostics_;
        }

    private:
        int matrix_rows_ = 0;
        bool computed_ = false;
        mutable Diagnostics diagnostics_{};
        std::unique_ptr<Eigen::PartialPivLU<DenseMatrix::native_type>>
            partial_piv_lu_{};
        std::unique_ptr<Eigen::FullPivLU<DenseMatrix::native_type>>
            full_piv_lu_{};
    };
}

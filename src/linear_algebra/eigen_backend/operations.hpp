#pragma once

#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

#include <Eigen/Core>
#include <Eigen/SparseCore>

#include "vector.hpp"
#include "sparse_matrix.hpp"
#include "sparse_builder.hpp"

#include "../operations/scalar_ops.hpp"
#include "../operations/dense_matrix_ops.hpp"
#include "../operations/sparse_matrix_ops.hpp"
#include "../preconditioners/schur_approximations.hpp"

namespace la::ops
{
    namespace detail
    {
        struct EigenSchurBackend
        {
            using SparseMatrix = la::eigen::SparseMatrix;
            using SparseBuilder = la::eigen::SparseBuilder;
        };
    }

    inline double inf_norm(const la::eigen::Vector& x)
    {
        return x.native().lpNorm<Eigen::Infinity>();
    }

    inline la::eigen::Vector subtract(
        const la::eigen::Vector& a,
        const la::eigen::Vector& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("la::ops::subtract(eigen): size mismatch.");

        la::eigen::Vector out(a.size());
        out.native() = a.native() - b.native();
        return out;
    }

    inline la::eigen::Vector add(
        const la::eigen::Vector& a,
        const la::eigen::Vector& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("la::ops::add(eigen): size mismatch.");

        la::eigen::Vector out(a.size());
        out.native() = a.native() + b.native();
        return out;
    }

    inline double dot(
        const la::eigen::Vector& a,
        const la::eigen::Vector& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("la::ops::dot(eigen): size mismatch.");

        return a.native().dot(b.native());
    }

    inline void scale_in_place(la::eigen::Vector& x, double alpha)
    {
        x.native() *= alpha;
    }

    inline la::eigen::Vector scaled(const la::eigen::Vector& x, double alpha)
    {
        la::eigen::Vector out(x.size());
        out.native() = alpha * x.native();
        return out;
    }

    inline void axpy(double alpha, const la::eigen::Vector& x, la::eigen::Vector& y)
    {
        if (x.size() != y.size())
            throw std::runtime_error("la::ops::axpy(eigen): size mismatch.");

        y.native() += alpha * x.native();
    }

    inline la::eigen::Vector matvec(
        const la::eigen::SparseMatrix& A,
        const la::eigen::Vector& x)
    {
        if (A.cols() != x.size())
            throw std::runtime_error("la::ops::matvec(eigen): dimension mismatch.");

        la::eigen::Vector y(A.rows());
        y.native() = A.native() * x.native();
        return y;
    }

    inline la::eigen::Vector transpose_matvec(
        const la::eigen::SparseMatrix& A,
        const la::eigen::Vector& x)
    {
        if (A.rows() != x.size())
            throw std::runtime_error("la::ops::transpose_matvec(eigen): dimension mismatch.");

        la::eigen::Vector y(A.cols());
        y.native() = A.native().transpose() * x.native();
        return y;
    }

    inline la::eigen::SparseMatrix diagonal_schur_approximation(
        const la::eigen::SparseMatrix& A_y,
        const la::eigen::SparseMatrix& lower_left_B,
        const la::eigen::SparseMatrix& C_signed,
        double diagonal_tolerance = 1.0e-14)
    {
        return la::preconditioners::diagonal_schur_approximation<
            detail::EigenSchurBackend>(
                A_y,
                lower_left_B,
                C_signed,
                diagonal_tolerance);
    }

    inline double quadratic_form(
        const la::eigen::SparseMatrix& A,
        const la::eigen::Vector& x)
    {
        if (A.rows() != A.cols())
            throw std::runtime_error("la::ops::quadratic_form(eigen): matrix must be square.");

        if (A.cols() != x.size())
            throw std::runtime_error("la::ops::quadratic_form(eigen): dimension mismatch.");

        return x.native().dot(A.native() * x.native());
    }

    inline bool is_symmetric(
        const la::eigen::SparseMatrix& A,
        double tol = 1.0e-10)
    {
        if (A.rows() != A.cols())
            return false;

        bool symmetric = true;

        A.for_each_nonzero(
            [&](int i, int j, double aij)
            {
                if (!symmetric)
                    return;

                if (!approx(aij, A.coeff(j, i), tol))
                    symmetric = false;
            });

        return symmetric;
    }

    inline SparseSymmetryDiagnostics relative_symmetry_diagnostics(
        const la::eigen::SparseMatrix& A)
    {
        SparseSymmetryDiagnostics diagnostics;
        diagnostics.is_square = (A.rows() == A.cols());

        if (A.rows() == 0 || A.cols() == 0)
        {
            diagnostics.matrix_norm = 0.0;
            diagnostics.difference_norm =
                diagnostics.is_square
                    ? 0.0
                    : std::numeric_limits<double>::infinity();
            diagnostics.relative_asymmetry =
                diagnostics.is_square
                    ? 0.0
                    : std::numeric_limits<double>::infinity();
            return diagnostics;
        }

        diagnostics.matrix_norm = A.native().norm();

        if (!diagnostics.is_square)
        {
            diagnostics.difference_norm =
                std::numeric_limits<double>::infinity();
            diagnostics.relative_asymmetry =
                std::numeric_limits<double>::infinity();
            return diagnostics;
        }

        la::eigen::SparseMatrix::native_type transpose = A.native().transpose();
        la::eigen::SparseMatrix::native_type difference =
            A.native() - transpose;
        diagnostics.difference_norm = difference.norm();
        diagnostics.relative_asymmetry =
            diagnostics.matrix_norm > 0.0
                ? diagnostics.difference_norm / diagnostics.matrix_norm
                : 0.0;

        return diagnostics;
    }

    inline SparseStorageDiagnostics sparse_storage_diagnostics(
        const la::eigen::SparseMatrix& A,
        double diagonal_tolerance = 0.0)
    {
        auto diagnostics =
            la::ops::sparse_storage_diagnostics<la::eigen::SparseMatrix>(
                A,
                diagonal_tolerance);
        diagnostics.is_compressed = A.native().isCompressed();
        return diagnostics;
    }

    inline void print_matrix_sparse(
        const la::eigen::SparseMatrix& A,
        std::ostream& os = std::cout)
    {
        os << "Eigen SparseMatrix (" << A.rows() << " x " << A.cols() << ")\n";

        int nnz = 0;
        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                os << "(" << i << ", " << j << ") = " << value << '\n';
                ++nnz;
            });

        os << "nnz = " << nnz << '\n';
    }

    inline void print_matrix(
        const la::eigen::SparseMatrix& A,
        std::ostream& os = std::cout)
    {
        os << "Eigen SparseMatrix (" << A.rows() << " x " << A.cols() << ")\n";
        os << A.native() << '\n';
        os << "nnz = " << A.native().nonZeros() << '\n';
    }

    inline void print_vector(
        const la::eigen::Vector& x,
        std::ostream& os = std::cout)
    {
        os << "Eigen Vector (size = " << x.size() << ")\n";
        os << x.native() << '\n';
    }

    inline void print_vector_compact(
        const la::eigen::Vector& x,
        std::ostream& os = std::cout,
        int width = 12)
    {
        os << "Eigen Vector (size = " << x.size() << ") [";
        for (int i = 0; i < x.size(); ++i)
        {
            if (i > 0)
                os << ' ';
            os << std::setw(width) << x[i];
        }
        os << " ]\n";
    }
}

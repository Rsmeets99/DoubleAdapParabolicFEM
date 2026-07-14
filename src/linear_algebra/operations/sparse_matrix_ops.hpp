#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_set>

#include "../concepts/vector.hpp"
#include "../concepts/sparse_builder.hpp"
#include "../concepts/sparse_matrix.hpp"
#include "scalar_ops.hpp"

namespace la::ops
{
    struct SparseSymmetryDiagnostics
    {
        double matrix_norm = 0.0;
        double difference_norm = 0.0;
        double relative_asymmetry = 0.0;
        bool is_square = false;

        [[nodiscard]] bool is_symmetric(double tol) const noexcept
        {
            return is_square && relative_asymmetry <= tol;
        }
    };

    struct SparseStorageDiagnostics
    {
        int rows = 0;
        int cols = 0;
        std::size_t stored_nonzeros = 0;
        std::size_t duplicate_positions = 0;
        std::size_t lower_triangle_nonzeros = 0;
        std::size_t upper_triangle_nonzeros = 0;
        int diagonal_entries_present = 0;
        int diagonal_entries_missing = 0;
        bool all_diagonal_entries_present = false;
        bool is_square = false;
        std::optional<bool> is_compressed{};
        double matrix_norm = 0.0;
        double triangle_difference_norm = 0.0;
        double relative_triangle_difference = 0.0;
        double max_triangle_entry_difference = 0.0;

        [[nodiscard]] bool has_duplicate_positions() const noexcept
        {
            return duplicate_positions > 0;
        }

        [[nodiscard]] bool has_inconsistent_triangle_entries(
            double tol) const noexcept
        {
            return is_square && relative_triangle_difference > tol;
        }
    };

    template<class SparseMatrix>
    requires requires(const SparseMatrix& A)
             {
                 { A.rows() } -> std::convertible_to<int>;
                 { A.cols() } -> std::convertible_to<int>;
                 { A.coeff(0, 0) } -> std::convertible_to<double>;
                 A.for_each_nonzero(
                     [](int, int, double) {});
             }
    SparseSymmetryDiagnostics relative_symmetry_diagnostics(
        const SparseMatrix& A)
    {
        SparseSymmetryDiagnostics diagnostics;
        diagnostics.is_square = (A.rows() == A.cols());

        double matrix_norm_squared = 0.0;
        std::unordered_set<std::int64_t> stored_positions;
        const auto encode_position =
            [n_cols = A.cols()](int i, int j) -> std::int64_t
            {
                return static_cast<std::int64_t>(i) *
                    static_cast<std::int64_t>(n_cols) +
                    static_cast<std::int64_t>(j);
            };

        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                matrix_norm_squared += value * value;
                if (diagnostics.is_square)
                    stored_positions.insert(encode_position(i, j));
            });

        diagnostics.matrix_norm = std::sqrt(matrix_norm_squared);

        if (!diagnostics.is_square)
        {
            diagnostics.difference_norm =
                std::numeric_limits<double>::infinity();
            diagnostics.relative_asymmetry =
                std::numeric_limits<double>::infinity();
            return diagnostics;
        }

        double difference_norm_squared = 0.0;
        A.for_each_nonzero(
            [&](int i, int j, double aij)
            {
                const double diff = aij - A.coeff(j, i);
                difference_norm_squared += diff * diff;

                if (i != j &&
                    stored_positions.find(encode_position(j, i)) ==
                        stored_positions.end())
                {
                    difference_norm_squared += diff * diff;
                }
            });

        diagnostics.difference_norm = std::sqrt(difference_norm_squared);
        diagnostics.relative_asymmetry =
            diagnostics.matrix_norm > 0.0
                ? diagnostics.difference_norm / diagnostics.matrix_norm
                : 0.0;

        return diagnostics;
    }

    template<class SparseMatrix>
    requires requires(const SparseMatrix& A)
             {
                 { A.rows() } -> std::convertible_to<int>;
                 { A.cols() } -> std::convertible_to<int>;
                 { A.coeff(0, 0) } -> std::convertible_to<double>;
                 A.for_each_nonzero(
                     [](int, int, double) {});
             }
    SparseStorageDiagnostics sparse_storage_diagnostics(
        const SparseMatrix& A,
        double diagonal_tolerance = 0.0)
    {
        (void)diagonal_tolerance;
        SparseStorageDiagnostics diagnostics;
        diagnostics.rows = A.rows();
        diagnostics.cols = A.cols();
        diagnostics.is_square = (A.rows() == A.cols());

        const int diagonal_size = std::min(A.rows(), A.cols());
        std::unordered_set<int> diagonal_positions;
        std::unordered_set<std::int64_t> stored_positions;
        const auto encode_position =
            [n_cols = A.cols()](int i, int j) -> std::int64_t
            {
                return static_cast<std::int64_t>(i) *
                    static_cast<std::int64_t>(n_cols) +
                    static_cast<std::int64_t>(j);
            };

        A.for_each_nonzero(
            [&](int i, int j, double)
            {
                ++diagnostics.stored_nonzeros;

                const auto [_, inserted] =
                    stored_positions.insert(encode_position(i, j));
                if (!inserted)
                    ++diagnostics.duplicate_positions;

                if (diagnostics.is_square)
                {
                    if (i > j)
                        ++diagnostics.lower_triangle_nonzeros;
                    else if (i < j)
                        ++diagnostics.upper_triangle_nonzeros;

                    if (i == j)
                        diagonal_positions.insert(i);
                }
            });

        diagnostics.diagonal_entries_present =
            static_cast<int>(diagonal_positions.size());
        diagnostics.diagonal_entries_missing =
            diagonal_size - diagnostics.diagonal_entries_present;
        diagnostics.all_diagonal_entries_present =
            diagnostics.diagonal_entries_missing == 0;

        const auto symmetry = relative_symmetry_diagnostics(A);
        diagnostics.matrix_norm = symmetry.matrix_norm;
        diagnostics.triangle_difference_norm = symmetry.difference_norm;
        diagnostics.relative_triangle_difference = symmetry.relative_asymmetry;

        if (diagnostics.is_square)
        {
            A.for_each_nonzero(
                [&](int i, int j, double aij)
                {
                    const double difference = std::abs(aij - A.coeff(j, i));
                    diagnostics.max_triangle_entry_difference =
                        std::max(
                            diagnostics.max_triangle_entry_difference,
                            difference);
                });
        }

        return diagnostics;
    }

    template<class SparseMatrix, class VectorLike>
    requires la::concepts::VectorLike<VectorLike> &&
             requires(const SparseMatrix& A)
             {
                 { A.rows() } -> std::convertible_to<int>;
                 { A.cols() } -> std::convertible_to<int>;
                 A.for_each_nonzero(
                     [](int, int, double) {});
             }
    VectorLike matvec(const SparseMatrix& A, const VectorLike& x)
    {
        if (A.cols() != x.size())
            throw std::runtime_error("la::ops::matvec: dimension mismatch.");

        VectorLike y(A.rows());
        y.set_zero();

        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                y[i] += value * x[j];
            });

        return y;
    }

    template<class SparseMatrix, class VectorLike>
    requires la::concepts::VectorLike<VectorLike> &&
             requires(const SparseMatrix& A)
             {
                 { A.rows() } -> std::convertible_to<int>;
                 { A.cols() } -> std::convertible_to<int>;
                 A.for_each_nonzero(
                     [](int, int, double) {});
             }
    VectorLike transpose_matvec(const SparseMatrix& A, const VectorLike& x)
    {
        if (A.rows() != x.size())
            throw std::runtime_error("la::ops::transpose_matvec: dimension mismatch.");

        VectorLike y(A.cols());
        y.set_zero();

        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                y[j] += value * x[i];
            });

        return y;
    }

    template<class SparseMatrix, class SparseBuilder, class VectorLike>
    requires la::concepts::SparseMatrixLike<SparseMatrix, SparseBuilder> &&
             la::concepts::VectorLike<VectorLike>
    double quadratic_form(const SparseMatrix& A, const VectorLike& x)
    {
        if (A.rows() != A.cols())
            throw std::runtime_error("la::ops::quadratic_form: matrix must be square.");

        if (A.cols() != x.size())
            throw std::runtime_error("la::ops::quadratic_form: dimension mismatch.");

        double value = 0.0;
        A.for_each_nonzero(
            [&](int i, int j, double aij)
            {
                value += x[i] * aij * x[j];
            });

        return value;
    }

    template<class SparseMatrix, class SparseBuilder>
    requires la::concepts::SparseMatrixLike<SparseMatrix, SparseBuilder>
    SparseMatrix add_sparse_matrices(
        const SparseMatrix& A,
        const SparseMatrix& B,
        double scale_A = 1.0,
        double scale_B = 1.0,
        double zero_tol = 0.0)
    {
        if (A.rows() != B.rows() || A.cols() != B.cols())
            throw std::runtime_error(
                "la::ops::add_sparse_matrices: dimension mismatch.");

        SparseBuilder builder;
        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                const double scaled = scale_A * value;
                if (std::abs(scaled) > zero_tol)
                    builder.add(i, j, scaled);
            });
        B.for_each_nonzero(
            [&](int i, int j, double value)
            {
                const double scaled = scale_B * value;
                if (std::abs(scaled) > zero_tol)
                    builder.add(i, j, scaled);
            });

        SparseMatrix out(A.rows(), A.cols());
        out.set_from_builder(builder);
        return out;
    }

    template<class SparseMatrix, class SparseBuilder>
    requires la::concepts::SparseMatrixLike<SparseMatrix, SparseBuilder>
    bool is_symmetric(const SparseMatrix& A, double tol = 1.0e-10)
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

    template<class SparseMatrix, class SparseBuilder>
    requires la::concepts::SparseMatrixLike<SparseMatrix, SparseBuilder>
    void print_matrix_sparse(const SparseMatrix& A, std::ostream& os = std::cout)
    {
        os << "SparseMatrix (" << A.rows() << " x " << A.cols() << ")\n";

        int nnz = 0;
        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                os << "(" << i << ", " << j << ") = " << value << '\n';
                ++nnz;
            });

        os << "nnz = " << nnz << '\n';
    }

    template<class SparseMatrix, class SparseBuilder>
    requires la::concepts::SparseMatrixLike<SparseMatrix, SparseBuilder>
    void print_matrix(const SparseMatrix& A, std::ostream& os = std::cout, int width = 12)
    {
        os << "SparseMatrix (" << A.rows() << " x " << A.cols() << ")\n";

        int nnz = 0;
        A.for_each_nonzero(
            [&](int, int, double)
            {
                ++nnz;
            });

        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
                os << std::setw(width) << A.coeff(i, j) << ' ';
            os << '\n';
        }

        os << "nnz = " << nnz << '\n';
    }
}

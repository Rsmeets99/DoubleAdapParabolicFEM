#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "linear_algebra/operations/sparse_matrix_ops.hpp"
#include "linear_algebra/preconditioners/diagnostics.hpp"

namespace la::preconditioners
{
    template<class Backend>
    struct DiagonalSchurApproximation
    {
        typename Backend::SparseMatrix hhat{};
        ParabolicGraphNormApproximationDiagnostics diagnostics{};
    };

    template<class Backend>
    struct ParabolicGraphNormApproximation
    {
        typename Backend::SparseMatrix hhat{};
        ParabolicGraphNormApproximationDiagnostics diagnostics{};
    };

    namespace detail
    {
        template<class SparseMatrix>
        [[nodiscard]] inline std::size_t count_sparse_nonzeros_(
            const SparseMatrix& matrix,
            double zero_tol)
        {
            std::size_t nnz = 0;
            matrix.for_each_nonzero(
                [&](int, int, double value)
                {
                    if (std::abs(value) > zero_tol)
                        ++nnz;
                });
            return nnz;
        }

        template<class SparseMatrix>
        inline void finalize_sparse_schur_diagnostics_(
            ParabolicGraphNormApproximationDiagnostics& diagnostics,
            const SparseMatrix& hhat,
            const SparseMatrix& C_signed,
            std::size_t b_or_bdt_nnz_used,
            std::size_t sparse_builder_entries,
            double zero_tol)
        {
            diagnostics.hhat_rows = hhat.rows();
            diagnostics.hhat_cols = hhat.cols();
            diagnostics.hhat_nnz = count_sparse_nonzeros_(hhat, zero_tol);
            diagnostics.c_signed_nnz =
                count_sparse_nonzeros_(C_signed, zero_tol);
            if (diagnostics.b_or_bdt_nnz_used == 0)
                diagnostics.b_or_bdt_nnz_used = b_or_bdt_nnz_used;
            diagnostics.sparse_builder_entries = sparse_builder_entries;

            const std::size_t input_nnz =
                diagnostics.c_signed_nnz +
                diagnostics.b_or_bdt_nnz_used;
            diagnostics.approximate_fill_ratio =
                input_nnz > 0
                    ? static_cast<double>(diagnostics.hhat_nnz) /
                          static_cast<double>(input_nnz)
                    : (diagnostics.hhat_nnz > 0
                           ? std::numeric_limits<double>::infinity()
                           : 0.0);
        }
    }

    template<class Backend>
    [[nodiscard]] DiagonalSchurApproximation<Backend>
    diagonal_schur_approximation_with_diagnostics(
        const typename Backend::SparseMatrix& A_y,
        const typename Backend::SparseMatrix& lower_left_B,
        const typename Backend::SparseMatrix& C_signed,
        double diagonal_tolerance = 1.0e-14,
        double zero_tol = 0.0)
    {
        using SparseMatrix = typename Backend::SparseMatrix;
        using SparseBuilder = typename Backend::SparseBuilder;

        const std::chrono::steady_clock::time_point setup_start =
            std::chrono::steady_clock::now();
        ParabolicGraphNormApproximationDiagnostics diagnostics;

        if (A_y.rows() != A_y.cols())
            throw std::runtime_error(
                "diagonal_schur_approximation: A_y must be square.");
        if (C_signed.rows() != C_signed.cols())
            throw std::runtime_error(
                "diagonal_schur_approximation: C_signed must be square.");
        if (lower_left_B.cols() != A_y.rows())
            throw std::runtime_error(
                "diagonal_schur_approximation: B.cols() must equal A_y.rows().");
        if (lower_left_B.rows() != C_signed.rows())
            throw std::runtime_error(
                "diagonal_schur_approximation: B.rows() must equal C_signed.rows().");

        std::vector<double> inverse_diagonal(
            static_cast<std::size_t>(A_y.rows()),
            0.0);
        for (int i = 0; i < A_y.rows(); ++i)
        {
            const double diagonal = A_y.coeff(i, i);
            if (!(diagonal > diagonal_tolerance))
            {
                throw std::runtime_error(
                    "diagonal_schur_approximation: A_y diagonal must be positive.");
            }
            inverse_diagonal[static_cast<std::size_t>(i)] =
                1.0 / diagonal;
        }

        std::vector<std::vector<std::pair<int, double>>> B_by_column(
            static_cast<std::size_t>(A_y.rows()));
        lower_left_B.for_each_nonzero(
            [&](int row, int col, double value)
            {
                if (row < 0 || row >= lower_left_B.rows() ||
                    col < 0 || col >= lower_left_B.cols())
                {
                    throw std::runtime_error(
                        "diagonal_schur_approximation: B nonzero is out of range.");
                }
                if (std::abs(value) > zero_tol)
                {
                    B_by_column[static_cast<std::size_t>(col)].push_back(
                        {row, value});
                    ++diagnostics.b_or_bdt_nnz_used;
                }
            });

        std::size_t c_nnz = 0;
        C_signed.for_each_nonzero(
            [&](int, int, double value)
            {
                if (std::abs(value) > zero_tol)
                    ++c_nnz;
            });

        std::size_t schur_nnz_upper_bound = 0;
        for (const auto& column : B_by_column)
            schur_nnz_upper_bound += column.size() * column.size();

        SparseBuilder builder;
        builder.reserve(c_nnz + schur_nnz_upper_bound);
        std::size_t sparse_builder_entries = 0;

        // blocks.C is signed negative trace. The positive Schur approximation
        // starts with C_pos = -C_signed.
        C_signed.for_each_nonzero(
            [&](int i, int j, double value)
            {
                if (std::abs(value) > zero_tol)
                {
                    builder.add(i, j, -value);
                    ++sparse_builder_entries;
                }
            });

        for (int lambda_col = 0; lambda_col < A_y.rows(); ++lambda_col)
        {
            const auto& column =
                B_by_column[static_cast<std::size_t>(lambda_col)];
            if (column.empty())
                continue;

            const double scale =
                inverse_diagonal[static_cast<std::size_t>(lambda_col)];
            for (const auto& [row_i, value_i] : column)
            {
                for (const auto& [row_j, value_j] : column)
                {
                    const double contribution = scale * value_i * value_j;
                    if (std::abs(contribution) > zero_tol)
                    {
                        builder.add(row_i, row_j, contribution);
                        ++sparse_builder_entries;
                    }
                }
            }
        }

        SparseMatrix hhat(C_signed.rows(), C_signed.cols());
        hhat.set_from_builder(builder);

        detail::finalize_sparse_schur_diagnostics_(
            diagnostics,
            hhat,
            C_signed,
            diagnostics.b_or_bdt_nnz_used,
            sparse_builder_entries,
            zero_tol);

        const std::chrono::steady_clock::time_point setup_end =
            std::chrono::steady_clock::now();
        diagnostics.setup_seconds =
            std::chrono::duration<double>(setup_end - setup_start).count();

        DiagonalSchurApproximation<Backend> out;
        out.hhat = std::move(hhat);
        out.diagnostics = diagnostics;
        return out;
    }

    template<class Backend>
    [[nodiscard]] typename Backend::SparseMatrix diagonal_schur_approximation(
        const typename Backend::SparseMatrix& A_y,
        const typename Backend::SparseMatrix& lower_left_B,
        const typename Backend::SparseMatrix& C_signed,
        double diagonal_tolerance = 1.0e-14,
        double zero_tol = 0.0)
    {
        return diagonal_schur_approximation_with_diagnostics<Backend>(
                   A_y,
                   lower_left_B,
                   C_signed,
                   diagonal_tolerance,
                   zero_tol)
            .hhat;
    }

    template<class Backend>
    [[nodiscard]] ParabolicGraphNormApproximation<Backend>
    parabolic_graph_norm_approximation(
        const typename Backend::SparseMatrix& A_y,
        const typename Backend::SparseMatrix& B_dt,
        const typename Backend::SparseMatrix& C_signed,
        const typename Backend::SparseMatrix& A_x,
        double diagonal_tolerance = 1.0e-14,
        double zero_tol = 0.0)
    {
        using SparseMatrix = typename Backend::SparseMatrix;
        using SparseBuilder = typename Backend::SparseBuilder;

        if (A_x.rows() != A_x.cols())
            throw std::runtime_error(
                "parabolic_graph_norm_approximation: A_x must be square.");
        if (C_signed.rows() != A_x.rows())
            throw std::runtime_error(
                "parabolic_graph_norm_approximation: C_signed size must match A_x.");
        if (B_dt.rows() != A_x.rows())
            throw std::runtime_error(
                "parabolic_graph_norm_approximation: B_dt rows must match A_x.");

        const std::chrono::steady_clock::time_point setup_start =
            std::chrono::steady_clock::now();

        ParabolicGraphNormApproximation<Backend> out;
        DiagonalSchurApproximation<Backend> hpar =
            diagonal_schur_approximation_with_diagnostics<Backend>(
                A_y,
                B_dt,
                C_signed,
                diagonal_tolerance,
                zero_tol);
        out.hhat =
            la::ops::add_sparse_matrices<SparseMatrix, SparseBuilder>(
                hpar.hhat,
                A_x,
                1.0,
                1.0,
                zero_tol);
        out.diagnostics = hpar.diagnostics;
        detail::finalize_sparse_schur_diagnostics_(
            out.diagnostics,
            out.hhat,
            C_signed,
            hpar.diagnostics.b_or_bdt_nnz_used,
            hpar.diagnostics.sparse_builder_entries +
                detail::count_sparse_nonzeros_(A_x, zero_tol),
            zero_tol);

        const std::chrono::steady_clock::time_point setup_end =
            std::chrono::steady_clock::now();
        out.diagnostics.setup_seconds =
            std::chrono::duration<double>(setup_end - setup_start).count();
        return out;
    }
}

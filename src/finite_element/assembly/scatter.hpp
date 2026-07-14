#pragma once

#include <cmath>
#include <stdexcept>

#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/vector.hpp"
#include "linear_algebra/assembly/local_objects.hpp"
#include "constrained_dofs.hpp"

namespace finite_element::assembly
{
    template<class PatternBuilder>
    void scatter_matrix_pattern(
        PatternBuilder& builder,
        const CellRestriction& test_restriction,
        const CellRestriction& trial_restriction,
        int row_offset = 0,
        int col_offset = 0)
    {
        for (int i = 0; i < test_restriction.size(); ++i)
        {
            const auto& I = test_restriction[i];
            for (int j = 0; j < trial_restriction.size(); ++j)
            {
                const auto& J = trial_restriction[j];
                for (const auto& wi : I)
                {
                    if (wi.true_dof < 0)
                        continue;

                    for (const auto& wj : J)
                    {
                        if (wj.true_dof < 0)
                            continue;

                        builder.add_pattern(
                            row_offset + wi.true_dof,
                            col_offset + wj.true_dof);
                    }
                }
            }
        }
    }

    template<class PatternBuilder>
    void scatter_matrix_pattern_transpose(
        PatternBuilder& builder,
        const CellRestriction& test_restriction,
        const CellRestriction& trial_restriction,
        int row_offset = 0,
        int col_offset = 0)
    {
        scatter_matrix_pattern(
            builder,
            trial_restriction,
            test_restriction,
            row_offset,
            col_offset);
    }

    // Scatter the cell matrix by the cell restrictions:
    //     A += C_test,c^T A_c C_trial,c.
    template<class SparseBuilder, class LocalMatrixType>
    requires la::concepts::SparseBuilderLike<SparseBuilder>
    void scatter_matrix(
        SparseBuilder& builder,
        const LocalMatrixType& local,
        const CellRestriction& test_restriction,
        const CellRestriction& trial_restriction,
        double zero_tol = 1e-15)
    {
        if (local.rows != test_restriction.size())
        {
            throw std::runtime_error(
                "scatter_matrix: local.rows != test restriction size.");
        }

        if (local.cols != trial_restriction.size())
        {
            throw std::runtime_error(
                "scatter_matrix: local.cols != trial restriction size.");
        }

        for (int i = 0; i < local.rows; ++i)
        {
            const auto& I = test_restriction[i];

            for (int j = 0; j < local.cols; ++j)
            {
                const double a_ij = local(i, j);
                if (std::abs(a_ij) <= zero_tol)
                    continue;

                const auto& J = trial_restriction[j];

                for (const auto& wi : I)
                {
                    if (wi.true_dof < 0)
                        continue;

                    for (const auto& wj : J)
                    {
                        if (wj.true_dof < 0)
                            continue;

                        builder.add(
                            wi.true_dof,
                            wj.true_dof,
                            wi.weight * wj.weight * a_ij);
                    }
                }
            }
        }
    }

    template<class MatrixAccumulator, class LocalMatrixType>
    void scatter_matrix_offset(
        MatrixAccumulator& matrix,
        const LocalMatrixType& local,
        const CellRestriction& test_restriction,
        const CellRestriction& trial_restriction,
        int row_offset,
        int col_offset,
        double zero_tol = 1e-15)
    {
        if (local.rows != test_restriction.size())
        {
            throw std::runtime_error(
                "scatter_matrix_offset: local.rows != test restriction size.");
        }

        if (local.cols != trial_restriction.size())
        {
            throw std::runtime_error(
                "scatter_matrix_offset: local.cols != trial restriction size.");
        }

        for (int i = 0; i < local.rows; ++i)
        {
            const auto& I = test_restriction[i];

            for (int j = 0; j < local.cols; ++j)
            {
                const double a_ij = local(i, j);
                if (std::abs(a_ij) <= zero_tol)
                    continue;

                const auto& J = trial_restriction[j];

                for (const auto& wi : I)
                {
                    if (wi.true_dof < 0)
                        continue;

                    for (const auto& wj : J)
                    {
                        if (wj.true_dof < 0)
                            continue;

                        matrix.add(
                            row_offset + wi.true_dof,
                            col_offset + wj.true_dof,
                            wi.weight * wj.weight * a_ij);
                    }
                }
            }
        }
    }

    template<class MatrixAccumulator, class LocalMatrixType>
    void scatter_matrix_transpose_offset(
        MatrixAccumulator& matrix,
        const LocalMatrixType& local,
        const CellRestriction& test_restriction,
        const CellRestriction& trial_restriction,
        int row_offset,
        int col_offset,
        double zero_tol = 1e-15)
    {
        if (local.rows != test_restriction.size())
        {
            throw std::runtime_error(
                "scatter_matrix_transpose_offset: local.rows != test restriction size.");
        }

        if (local.cols != trial_restriction.size())
        {
            throw std::runtime_error(
                "scatter_matrix_transpose_offset: local.cols != trial restriction size.");
        }

        for (int i = 0; i < local.rows; ++i)
        {
            const auto& I = test_restriction[i];

            for (int j = 0; j < local.cols; ++j)
            {
                const double a_ij = local(i, j);
                if (std::abs(a_ij) <= zero_tol)
                    continue;

                const auto& J = trial_restriction[j];

                for (const auto& wi : I)
                {
                    if (wi.true_dof < 0)
                        continue;

                    for (const auto& wj : J)
                    {
                        if (wj.true_dof < 0)
                            continue;

                        matrix.add(
                            row_offset + wj.true_dof,
                            col_offset + wi.true_dof,
                            wi.weight * wj.weight * a_ij);
                    }
                }
            }
        }
    }

    template<class VectorLike, class LocalVectorType>
    requires la::concepts::VectorLike<VectorLike>
    void scatter_vector(
        VectorLike& builder,
        const LocalVectorType& local,
        const CellRestriction& test_restriction,
        double zero_tol = 1e-15)
    {
        if (local.size != test_restriction.size())
        {
            throw std::runtime_error(
                "scatter_vector: local.size != test restriction size.");
        }

        for (int i = 0; i < local.size; ++i)
        {
            const double f_i = local[i];
            if (std::abs(f_i) <= zero_tol)
                continue;

            // f += C_test,c^T f_c.
            for (const auto& wi : test_restriction[i])
            {
                if (wi.true_dof < 0)
                    continue;

                builder.add(wi.true_dof, wi.weight * f_i);
            }
        }
    }
}

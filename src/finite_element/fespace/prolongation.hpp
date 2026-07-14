#pragma once

#include <cmath>
#include <stdexcept>
#include <vector>

#include "../assembly/detail/active_cell_locator.hpp"
#include "dofs/physical_dof_coords.hpp"
#include "functions.hpp"

namespace finite_element::fespace
{
    template<class SourceFunctionType, class TargetSpaceType>
    [[nodiscard]] typename SourceFunctionType::Vector
    prolong_true_coefficients_nodal(
        const SourceFunctionType& source_function,
        const TargetSpaceType& target_space)
    {
        using SourceSpaceType = typename SourceFunctionType::SpaceType;
        using Vector = typename SourceFunctionType::Vector;

        const SourceSpaceType& source_space = source_function.fespace();

        if (&source_space.mesh_ref() != &target_space.mesh_ref())
        {
            throw std::runtime_error(
                "prolong_true_coefficients_nodal: source and target spaces must share the same mesh.");
        }

        const auto& target_dofs = target_space.dof_handler_ref();

        Vector target_true(target_dofs.n_true_dofs());
        target_true.set_zero();

        std::vector<char> assigned(
            static_cast<std::size_t>(target_dofs.n_true_dofs()),
            0);

        finite_element::assembly::detail::ActiveAncestorCache<SourceSpaceType>
            ancestor_cache(source_space);

        for (const int target_cell_id : target_space.active_cells())
        {
            const int source_cell_id =
                finite_element::assembly::detail::find_active_ancestor_cell(
                    ancestor_cache,
                    source_space,
                    target_space,
                    target_cell_id);

            const auto& target_cell_dofs =
                target_dofs.cell_dofs(target_cell_id);

            for (int local = 0; local < TargetSpaceType::FETraitsType::dofs_per_cell; ++local)
            {
                const int target_gid = target_cell_dofs[local];
                if (target_gid < 0)
                    continue;

                const auto& target_dof = target_dofs.dof(target_gid);
                if (target_dof.is_constrained)
                    continue;

                const int target_true_id = target_dof.true_dof_id;
                if (target_true_id < 0 ||
                    target_true_id >= target_dofs.n_true_dofs())
                {
                    throw std::runtime_error(
                        "prolong_true_coefficients_nodal: target true DoF id is invalid.");
                }

                if (assigned[static_cast<std::size_t>(target_true_id)] != 0)
                    continue;

                const auto point =
                    finite_element::fespace::physical_dof_coord(
                        target_space,
                        target_cell_id,
                        local);

                target_true[target_true_id] =
                    source_function.value_on_cell(source_cell_id, point);
                assigned[static_cast<std::size_t>(target_true_id)] = 1;
            }
        }

        for (int true_id = 0; true_id < target_dofs.n_true_dofs(); ++true_id)
        {
            if (assigned[static_cast<std::size_t>(true_id)] == 0)
            {
                throw std::runtime_error(
                    "prolong_true_coefficients_nodal: failed to assign a target true DoF.");
            }
        }

        return target_true;
    }

    template<class SourceFunctionType, class TargetSpaceType>
    [[nodiscard]] finite_element::Function<
        TargetSpaceType,
        typename SourceFunctionType::Vector>
    prolong_nodal_function(
        const SourceFunctionType& source_function,
        const TargetSpaceType& target_space)
    {
        auto target_true =
            prolong_true_coefficients_nodal(source_function, target_space);

        return finite_element::Function<
            TargetSpaceType,
            typename SourceFunctionType::Vector>(
                target_space,
                target_true);
    }

    template<class Backend, class SourceSpaceType, class TargetSpaceType>
    [[nodiscard]] typename Backend::SparseMatrix
    nodal_true_dof_prolongation_matrix(
        const SourceSpaceType& source_space,
        const TargetSpaceType& target_space,
        double zero_tol = 1.0e-15)
    {
        using Vector = typename Backend::Vector;
        using SparseBuilder = typename Backend::SparseBuilder;
        using SparseMatrix = typename Backend::SparseMatrix;
        using SourceFunction =
            finite_element::Function<SourceSpaceType, Vector>;

        if (&source_space.mesh_ref() != &target_space.mesh_ref())
        {
            throw std::runtime_error(
                "nodal_true_dof_prolongation_matrix: source and target spaces must share the same mesh.");
        }

        const int n_source =
            source_space.dof_handler_ref().n_true_dofs();
        const int n_target =
            target_space.dof_handler_ref().n_true_dofs();

        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(n_target) *
            static_cast<std::size_t>(SourceSpaceType::FETraitsType::dofs_per_cell));

        for (int col = 0; col < n_source; ++col)
        {
            Vector source_true(n_source);
            source_true.set_zero();
            source_true[col] = 1.0;

            SourceFunction source_function(source_space, source_true);
            Vector target_true =
                prolong_true_coefficients_nodal(
                    source_function,
                    target_space);

            for (int row = 0; row < n_target; ++row)
            {
                const double value = target_true[row];
                if (std::abs(value) > zero_tol)
                    builder.add(row, col, value);
            }
        }

        SparseMatrix prolongation(n_target, n_source);
        prolongation.set_from_builder(builder);
        prolongation.compress();
        return prolongation;
    }

    template<class Backend>
    [[nodiscard]] typename Backend::SparseMatrix
    block_diagonal_prolongation_matrix(
        const typename Backend::SparseMatrix& P_y,
        const typename Backend::SparseMatrix& P_x)
    {
        using SparseBuilder = typename Backend::SparseBuilder;
        using SparseMatrix = typename Backend::SparseMatrix;

        SparseBuilder builder;
        std::size_t reserve_count = 0;
        P_y.for_each_nonzero(
            [&](int, int, double)
            {
                ++reserve_count;
            });
        P_x.for_each_nonzero(
            [&](int, int, double)
            {
                ++reserve_count;
            });
        builder.reserve(reserve_count);

        P_y.for_each_nonzero(
            [&](int i, int j, double value)
            {
                builder.add(i, j, value);
            });

        P_x.for_each_nonzero(
            [&](int i, int j, double value)
            {
                builder.add(
                    P_y.rows() + i,
                    P_y.cols() + j,
                    value);
            });

        SparseMatrix P(
            P_y.rows() + P_x.rows(),
            P_y.cols() + P_x.cols());
        P.set_from_builder(builder);
        P.compress();
        return P;
    }
}

#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"

namespace finite_element::assembly::error_system
{
    template<class Backend, class RTFluxSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_rt_local_prolongation_matrix(
        typename Backend::SparseMatrix& R,
        const RTFluxSpaceType& space,
        int patch_cell_index)
    {
        typename Backend::SparseBuilder builder;
        builder.reserve(static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v));

        const auto& map = space.cell_dof_map(patch_cell_index);
        for (int local_dof_id = 0;
             local_dof_id < RTFluxSpaceType::local_dofs_v;
             ++local_dof_id)
        {
            const auto& entry =
                map[static_cast<std::size_t>(local_dof_id)];
            if (entry.patch_dof_id < 0)
                continue;

            builder.add(
                local_dof_id,
                entry.patch_dof_id,
                static_cast<double>(entry.orientation_sign));
        }

        R.resize(RTFluxSpaceType::local_dofs_v, space.n_dofs());
        R.set_from_builder(builder);
    }

    template<class Backend, class ScalarSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_scalar_local_prolongation_matrix(
        typename Backend::SparseMatrix& S,
        const ScalarSpaceType& space,
        int patch_cell_index)
    {
        typename Backend::SparseBuilder builder;
        builder.reserve(static_cast<std::size_t>(ScalarSpaceType::local_dofs_v));

        for (int local_dof_id = 0;
             local_dof_id < ScalarSpaceType::local_dofs_v;
             ++local_dof_id)
        {
            const int patch_dof_id =
                space.local_to_patch_dof(patch_cell_index, local_dof_id);
            builder.add(local_dof_id, patch_dof_id, 1.0);
        }

        S.resize(ScalarSpaceType::local_dofs_v, space.n_dofs());
        S.set_from_builder(builder);
    }

    template<class SparseBuilder, class RTFluxSpaceType>
    requires la::concepts::SparseBuilderLike<SparseBuilder>
    void scatter_rt_local_matrix_2d(
        SparseBuilder& builder,
        const la::local::LocalMatrix& local,
        const RTFluxSpaceType& space,
        int patch_cell_index,
        double zero_tol = 1.0e-15)
    {
        if (local.rows != RTFluxSpaceType::local_dofs_v ||
            local.cols != RTFluxSpaceType::local_dofs_v)
        {
            throw std::runtime_error(
                "scatter_rt_local_matrix_2d: local matrix has unexpected dimensions.");
        }

        const auto& map = space.cell_dof_map(patch_cell_index);
        for (int i = 0; i < RTFluxSpaceType::local_dofs_v; ++i)
        {
            const auto& I = map[static_cast<std::size_t>(i)];
            if (I.patch_dof_id < 0)
                continue;

            for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
            {
                const auto& J = map[static_cast<std::size_t>(j)];
                if (J.patch_dof_id < 0)
                    continue;

                const double value =
                    static_cast<double>(I.orientation_sign) *
                    static_cast<double>(J.orientation_sign) *
                    local(i, j);
                if (std::abs(value) > zero_tol)
                    builder.add(I.patch_dof_id, J.patch_dof_id, value);
            }
        }
    }

    template<class SparseBuilder, class ScalarSpaceType>
    requires la::concepts::SparseBuilderLike<SparseBuilder>
    void scatter_scalar_local_matrix_2d(
        SparseBuilder& builder,
        const la::local::LocalMatrix& local,
        const ScalarSpaceType& space,
        int patch_cell_index,
        double zero_tol = 1.0e-15)
    {
        if (local.rows != ScalarSpaceType::local_dofs_v ||
            local.cols != ScalarSpaceType::local_dofs_v)
        {
            throw std::runtime_error(
                "scatter_scalar_local_matrix_2d: local matrix has unexpected dimensions.");
        }

        for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
        {
            const int I = space.local_to_patch_dof(patch_cell_index, i);
            for (int j = 0; j < ScalarSpaceType::local_dofs_v; ++j)
            {
                const int J = space.local_to_patch_dof(patch_cell_index, j);
                const double value = local(i, j);
                if (std::abs(value) > zero_tol)
                    builder.add(I, J, value);
            }
        }
    }

    template<class SparseBuilder, class ScalarSpaceType, class RTFluxSpaceType>
    requires la::concepts::SparseBuilderLike<SparseBuilder>
    void scatter_divergence_local_matrix_2d(
        SparseBuilder& builder,
        const la::local::LocalMatrix& local,
        const ScalarSpaceType& scalar_space,
        const RTFluxSpaceType& rt_space,
        int patch_cell_index,
        double zero_tol = 1.0e-15)
    {
        if (local.rows != ScalarSpaceType::local_dofs_v ||
            local.cols != RTFluxSpaceType::local_dofs_v)
        {
            throw std::runtime_error(
                "scatter_divergence_local_matrix_2d: local matrix has unexpected dimensions.");
        }

        const auto& rt_map = rt_space.cell_dof_map(patch_cell_index);
        for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
        {
            const int I = scalar_space.local_to_patch_dof(patch_cell_index, i);
            for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
            {
                const auto& J = rt_map[static_cast<std::size_t>(j)];
                if (J.patch_dof_id < 0)
                    continue;

                const double value =
                    static_cast<double>(J.orientation_sign) * local(i, j);
                if (std::abs(value) > zero_tol)
                    builder.add(I, J.patch_dof_id, value);
            }
        }
    }
}

#pragma once

#include <cmath>

#include "../detail/local_error_bilinear_forms.hpp"

#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"
#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::error_system
{
    template<int QSpace, int QTime, class Backend, class PatchFluxSpaceType, class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (PatchFluxSpaceType::GT::dim_space_v == 1)
    void assemble_mat_A(
        typename Backend::SparseMatrix& A,
        const PatchFluxSpaceType& flux_space,
        const MFunction& M,
        double zero_tol = 1.0e-15)
    {
        using SparseBuilder = typename Backend::SparseBuilder;
        using PatchScalarSpaceType = error_fespace::PatchScalarSpace<
            typename PatchFluxSpaceType::Patch,
            PatchFluxSpaceType::p_space_v,
            PatchFluxSpaceType::p_time_v>;

        la::local::LocalMatrix local(flux_space.n_dofs(), flux_space.n_dofs());
        finite_element::assembly::detail::zero_local_matrix(local);
        const PatchScalarSpaceType scalar_space(flux_space.patch());
        const finite_element::assembly::detail::LocalErrorQuadratureTables1D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType> tables(flux_space, scalar_space);

        for (int patch_cell_index = 0;
             patch_cell_index < flux_space.patch().n_cells;
             ++patch_cell_index)
        {
            finite_element::assembly::detail::accumulate_patch_flux_mass_matrix_on_cell<
                QSpace,
                QTime>(
                    local,
                    flux_space,
                    tables,
                    patch_cell_index,
                    M);
        }

        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(flux_space.n_dofs()) *
            static_cast<std::size_t>(flux_space.n_dofs()));

        for (int i = 0; i < flux_space.n_dofs(); ++i)
        {
            for (int j = 0; j < flux_space.n_dofs(); ++j)
            {
                const double value = local(i, j);
                if (std::abs(value) > zero_tol)
                    builder.add(i, j, value);
            }
        }

        A.resize(flux_space.n_dofs(), flux_space.n_dofs());
        A.set_from_builder(builder);
    }
}

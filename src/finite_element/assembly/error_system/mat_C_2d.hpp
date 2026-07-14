#pragma once

#include <cstddef>

#include "patch_assembly_maps_2d.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"
#include "quadrature/reference_triangle_duffy.hpp"

namespace finite_element::assembly::error_system
{
    template<class PatchScalarSpaceType>
    [[nodiscard]] la::local::LocalMatrix assemble_local_scalar_mass_matrix_2d(
        const PatchScalarSpaceType& scalar_space,
        int patch_cell_index)
    {
        static_assert(PatchScalarSpaceType::p_space_v <= 10,
                      "Scalar local mass quadrature currently expects p<=10.");

        la::local::LocalMatrix local(
            PatchScalarSpaceType::local_dofs_v,
            PatchScalarSpaceType::local_dofs_v);

        const double jac = scalar_space.spatial_jacobian_measure(patch_cell_index);
        quadrature::reference::for_each_reference_triangle_duffy_point<
            2 * PatchScalarSpaceType::p_space_v>(
            [&](const double x, const double y, const double weight)
            {
            typename PatchScalarSpaceType::LocalValues values{};
            scalar_space.evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                typename PatchScalarSpaceType::ReferencePoint{x, y},
                values);

            const double dmu = weight * jac;
            for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
            {
                for (int j = 0; j < PatchScalarSpaceType::local_dofs_v; ++j)
                {
                    local(i, j) +=
                        values[static_cast<std::size_t>(i)] *
                        values[static_cast<std::size_t>(j)] *
                        dmu;
                }
            }
            });

        return local;
    }

    template<class Backend, class PatchScalarSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_scalar_mass_matrix(
        typename Backend::SparseMatrix& C,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15)
    {
        using SparseBuilder = typename Backend::SparseBuilder;

        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(scalar_space.n_patch_cells()) *
            static_cast<std::size_t>(PatchScalarSpaceType::local_dofs_v) *
            static_cast<std::size_t>(PatchScalarSpaceType::local_dofs_v));

        for (int patch_cell_index = 0;
             patch_cell_index < scalar_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_scalar_mass_matrix_2d(
                    scalar_space,
                    patch_cell_index);
            scatter_scalar_local_matrix_2d(
                builder,
                local,
                scalar_space,
                patch_cell_index,
                zero_tol);
        }

        C.resize(scalar_space.n_dofs(), scalar_space.n_dofs());
        C.set_from_builder(builder);
    }

    template<class Backend, class PatchScalarSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_C_2d(
        typename Backend::SparseMatrix& C,
        const PatchScalarSpaceType& scalar_space)
    {
        typename Backend::SparseBuilder builder;
        builder.reserve(0);
        C.resize(scalar_space.n_dofs(), scalar_space.n_dofs());
        C.set_from_builder(builder);
    }
}

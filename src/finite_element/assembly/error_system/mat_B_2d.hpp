#pragma once

#include <cstddef>
#include <stdexcept>

#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "patch_assembly_maps_2d.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"
#include "quadrature/reference_triangle_duffy.hpp"

namespace finite_element::assembly::error_system
{
    template<int QSpace, int QTime, class PatchScalarSpaceType, class RTFluxSpaceType>
    [[nodiscard]] la::local::LocalMatrix assemble_local_divergence_coupling_matrix_time_2d(
        const PatchScalarSpaceType& scalar_space,
        const RTFluxSpaceType& flux_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            RTFluxSpaceType,
            PatchScalarSpaceType>& tables,
        int patch_cell_index)
    {
        la::local::LocalMatrix local(
            PatchScalarSpaceType::local_dofs_v,
            RTFluxSpaceType::local_dofs_v);

        const auto& cell_data = tables.cell(patch_cell_index);
        for (const auto& qp : cell_data.points)
        {
            const double dmu = qp.jacobian_weight;
            const auto& rt_basis_divergences = qp.rt_basis_divergences();
            for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
            {
                const double weighted_q_i =
                    qp.scalar_basis_values[static_cast<std::size_t>(i)] *
                    dmu;
                for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
                {
                    local(i, j) +=
                        weighted_q_i *
                        rt_basis_divergences[
                            static_cast<std::size_t>(j)];
                }
            }
        }

        (void)scalar_space;
        (void)flux_space;
        return local;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class RTFluxSpaceType,
        class PatchScalarSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (RTFluxSpaceType::GT::dim_space_v == 2) &&
             requires { RTFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_mat_B(
        typename Backend::SparseMatrix& B,
        const RTFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            RTFluxSpaceType,
            PatchScalarSpaceType>& tables,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells() ||
            tables.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_mat_B<2D time-dependent>: patch cell count mismatch.");
        }

        using SparseBuilder = typename Backend::SparseBuilder;

        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(scalar_space.n_patch_cells()) *
            static_cast<std::size_t>(PatchScalarSpaceType::local_dofs_v) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v));

        for (int patch_cell_index = 0;
             patch_cell_index < scalar_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_divergence_coupling_matrix_time_2d<QSpace, QTime>(
                    scalar_space,
                    flux_space,
                    tables,
                    patch_cell_index);
            scatter_divergence_local_matrix_2d(
                builder,
                local,
                scalar_space,
                flux_space,
                patch_cell_index,
                zero_tol);
        }

        B.resize(scalar_space.n_dofs(), flux_space.n_dofs());
        B.set_from_builder(builder);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class RTFluxSpaceType,
        class PatchScalarSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (RTFluxSpaceType::GT::dim_space_v == 2) &&
             requires { RTFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_mat_B(
        typename Backend::SparseMatrix& B,
        const RTFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15)
    {
        using Tables =
            finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                QSpace,
                QTime,
                RTFluxSpaceType,
                PatchScalarSpaceType>;

        const Tables tables(flux_space, scalar_space);
        assemble_mat_B<QSpace, QTime, Backend>(
            B,
            flux_space,
            scalar_space,
            tables,
            zero_tol);
    }

    template<class PatchScalarSpaceType, class RTFluxSpaceType>
    [[nodiscard]] la::local::LocalMatrix assemble_local_divergence_coupling_matrix_2d(
        const PatchScalarSpaceType& scalar_space,
        const RTFluxSpaceType& flux_space,
        int patch_cell_index)
    {
        static_assert(PatchScalarSpaceType::p_space_v + RTFluxSpaceType::p_space_v <= 22,
                      "Divergence-coupling quadrature currently expects p_scalar+p_rt<=22.");

        la::local::LocalMatrix local(
            PatchScalarSpaceType::local_dofs_v,
            RTFluxSpaceType::local_dofs_v);

        const auto map =
            flux_space.physical_map_for_patch_cell(patch_cell_index);
        const double jac =
            RTFluxSpaceType::PiolaBasis::jacobian_measure(map);

        quadrature::reference::for_each_reference_triangle_duffy_point<
            PatchScalarSpaceType::p_space_v + RTFluxSpaceType::p_space_v>(
            [&](const double x, const double y, const double weight)
            {
            typename PatchScalarSpaceType::LocalValues scalar_values{};
            scalar_space.evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                typename PatchScalarSpaceType::ReferencePoint{x, y},
                scalar_values);

            typename RTFluxSpaceType::LocalDivergences flux_divergences{};
            flux_space.evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                typename RTFluxSpaceType::ReferencePoint{x, y},
                flux_divergences);

            const double dmu = weight * jac;
            for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
            {
                const double weighted_scalar =
                    scalar_values[static_cast<std::size_t>(i)] * dmu;
                for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
                {
                    local(i, j) +=
                        weighted_scalar *
                        flux_divergences[static_cast<std::size_t>(j)];
                }
            }
            });

        return local;
    }

    template<class Backend, class PatchScalarSpaceType, class RTFluxSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_2d(
        typename Backend::SparseMatrix& B,
        const PatchScalarSpaceType& scalar_space,
        const RTFluxSpaceType& flux_space,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_mat_B_2d: patch cell count mismatch.");
        }

        using SparseBuilder = typename Backend::SparseBuilder;

        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(scalar_space.n_patch_cells()) *
            static_cast<std::size_t>(PatchScalarSpaceType::local_dofs_v) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v));

        for (int patch_cell_index = 0;
             patch_cell_index < scalar_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_divergence_coupling_matrix_2d(
                    scalar_space,
                    flux_space,
                    patch_cell_index);
            scatter_divergence_local_matrix_2d(
                builder,
                local,
                scalar_space,
                flux_space,
                patch_cell_index,
                zero_tol);
        }

        B.resize(scalar_space.n_dofs(), flux_space.n_dofs());
        B.set_from_builder(builder);
    }

    template<class Backend, class PatchScalarSpaceType, class RTFluxSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_divergence_coupling_matrix(
        typename Backend::SparseMatrix& B,
        const PatchScalarSpaceType& scalar_space,
        const RTFluxSpaceType& flux_space,
        double zero_tol = 1.0e-15)
    {
        assemble_mat_B_2d<Backend>(
            B,
            scalar_space,
            flux_space,
            zero_tol);
    }
}

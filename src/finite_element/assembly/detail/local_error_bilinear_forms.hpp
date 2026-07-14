#pragma once

#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../error_fespace/patch_flux_space_1d.hpp"
#include "../../error_fespace/patch_scalar_space.hpp"
#include "../../time_slabs/time_slab_edge_patch_builder.hpp"
#include "local_error_quadrature_tables_1d.hpp"
#include "zero_local.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::detail
{
    template<int QSpace, int QTime, class PatchFluxSpaceType, class MFunction>
    void accumulate_patch_flux_mass_matrix_on_cell(
        la::local::LocalMatrix& local,
        const PatchFluxSpaceType& flux_space,
        const LocalErrorQuadratureTables1D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            error_fespace::PatchScalarSpace<
                typename PatchFluxSpaceType::Patch,
                PatchFluxSpaceType::p_space_v,
                PatchFluxSpaceType::p_time_v>>& tables,
        int patch_cell_index,
        const MFunction& M)
    {
        const auto& patch = flux_space.patch();
        const double jac = patch.cell_jacobian_measure(patch_cell_index);

        for (int qx = 0; qx < QSpace; ++qx)
        {
            const double x_ref = tables.space_rule.points[qx][0];
            const double w_x   = tables.space_rule.weights[qx];

            for (int qt = 0; qt < QTime; ++qt)
            {
                const double t_ref = tables.time_rule.points[qt][0];
                const double w_t   = tables.time_rule.weights[qt];
                typename PatchFluxSpaceType::BasisValues flux_basis_values;
                tables.fill_flux_basis_values(
                    flux_space,
                    patch_cell_index,
                    qx,
                    qt,
                    flux_basis_values);

                const auto p = patch.map_to_physical(
                    patch_cell_index,
                    x_ref,
                    t_ref);
                const auto M_q =
                    coefficients::evaluate_diffusion_tensor<1>(M, p);
                const double dmu = jac * w_x * w_t;

                for (int i = 0; i < flux_space.n_dofs(); ++i)
                {
                    for (int j = 0; j < flux_space.n_dofs(); ++j)
                    {
                        const coefficients::DiffusionVector<1> sigma_i{
                            flux_basis_values[static_cast<std::size_t>(i)]
                        };
                        const coefficients::DiffusionVector<1> sigma_j{
                            flux_basis_values[static_cast<std::size_t>(j)]
                        };
                        local(i, j) +=
                            coefficients::inverse_diffusion_dot(
                                M_q,
                                sigma_i,
                                sigma_j) *
                            dmu;
                    }
                }
            }
        }
    }

    template<int QSpace, int QTime, class PatchFluxSpaceType, class PatchScalarSpaceType>
    void accumulate_patch_divergence_matrix_on_cell(
        la::local::LocalMatrix& local,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const LocalErrorQuadratureTables1D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        int patch_cell_index)
    {
        const auto& patch = flux_space.patch();
        const double jac = patch.cell_jacobian_measure(patch_cell_index);

        for (int qx = 0; qx < QSpace; ++qx)
        {
            const double w_x = tables.space_rule.weights[qx];

            for (int qt = 0; qt < QTime; ++qt)
            {
                const double w_t = tables.time_rule.weights[qt];
                typename PatchFluxSpaceType::BasisValues flux_basis_divergences;
                typename PatchScalarSpaceType::BasisValues scalar_basis_values;
                tables.fill_flux_basis_divergences(
                    flux_space,
                    patch_cell_index,
                    qx,
                    qt,
                    flux_basis_divergences);
                tables.fill_scalar_basis_values(
                    scalar_space,
                    patch_cell_index,
                    qx,
                    qt,
                    scalar_basis_values);

                const double dmu = jac * w_x * w_t;

                for (int i = 0; i < scalar_space.n_dofs(); ++i)
                {
                    const double weighted_scalar_i =
                        scalar_basis_values[static_cast<std::size_t>(i)] * dmu;

                    for (int j = 0; j < flux_space.n_dofs(); ++j)
                    {
                        local(i, j) +=
                            weighted_scalar_i *
                            flux_basis_divergences[static_cast<std::size_t>(j)];
                    }
                }
            }
        }
    }
}

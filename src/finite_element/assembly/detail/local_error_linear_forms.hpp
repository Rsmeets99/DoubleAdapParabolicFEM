#pragma once

#include <array>
#include <tuple>

#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../error_fespace/patch_flux_space_1d.hpp"
#include "../../error_fespace/patch_scalar_space.hpp"
#include "../../time_slabs/time_slab_edge_patch_builder.hpp"
#include "local_error_quadrature_tables_1d.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::detail
{
    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class SlabGeometryData,
        class XGeometryData>
    void accumulate_patch_flux_rhs_on_cell(
        la::local::LocalVector& local,
        const PatchFluxSpaceType& flux_space,
        const LocalErrorQuadratureTables1D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        int patch_cell_index,
        int slab_id,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        int x_cell_id,
        const SlabGeometryData& slab_geom,
        const XGeometryData& x_geom)
    {
        const auto& patch = flux_space.patch();
        const int slab_cell_id = patch.cell(patch_cell_index).slab_cell_id;
        const auto& cell_tables = tables.cells[static_cast<std::size_t>(patch_cell_index)];
        const double jac = patch.cell_jacobian_measure(patch_cell_index);

        for (int qx = 0; qx < QSpace; ++qx)
        {
            const double x_ref = tables.space_rule.points[qx][0];
            const double w_x   = tables.space_rule.weights[qx];
            const double psi_q =
                cell_tables.partition_of_unity_values[static_cast<std::size_t>(qx)];

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
                const auto grad_lambda_tilde =
                    lambda_tilde.gradient_on_cell(slab_id, slab_cell_id, p, slab_geom);
                const auto grad_u =
                    u_delta.gradient_on_cell(x_cell_id, p, x_geom);
                // The patch problems are written in terms of
                // \widetilde{\theta}^\delta = u^\delta + \widetilde{\lambda}^\delta,
                // so the spatial gradient is assembled from both pieces here.
                const double grad_theta_tilde_x =
                    grad_lambda_tilde[0] + grad_u[0];

                const double rhs =
                    -psi_q * grad_theta_tilde_x;
                const double rhs_dmu = rhs * jac * w_x * w_t;

                for (int i = 0; i < flux_space.n_dofs(); ++i)
                {
                    local[i] +=
                        rhs_dmu *
                        flux_basis_values[static_cast<std::size_t>(i)];
                }
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchScalarSpaceType,
        class PatchFluxSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class SlabGeometryData,
        class XGeometryData,
        class EllFunction,
        class MFunction>
    void accumulate_patch_scalar_rhs_on_cell(
        la::local::LocalVector& local,
        const PatchScalarSpaceType& scalar_space,
        const LocalErrorQuadratureTables1D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        int patch_cell_index,
        int slab_id,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        int x_cell_id,
        const SlabGeometryData& slab_geom,
        const XGeometryData& x_geom,
        const EllFunction& ell,
        const MFunction& M)
    {
        const auto& patch = scalar_space.patch();
        const int slab_cell_id = patch.cell(patch_cell_index).slab_cell_id;
        const auto& cell_tables = tables.cells[static_cast<std::size_t>(patch_cell_index)];
        const double jac = patch.cell_jacobian_measure(patch_cell_index);

        for (int qx = 0; qx < QSpace; ++qx)
        {
            const double x_ref = tables.space_rule.points[qx][0];
            const double w_x   = tables.space_rule.weights[qx];
            const double psi_dx =
                cell_tables.partition_of_unity_dx[static_cast<std::size_t>(qx)];

            for (int qt = 0; qt < QTime; ++qt)
            {
                const double t_ref = tables.time_rule.points[qt][0];
                const double w_t   = tables.time_rule.weights[qt];
                typename PatchScalarSpaceType::BasisValues scalar_basis_values;
                tables.fill_scalar_basis_values(
                    scalar_space,
                    patch_cell_index,
                    qx,
                    qt,
                    scalar_basis_values);

                const auto p = patch.map_to_physical(
                    patch_cell_index,
                    x_ref,
                    t_ref);

                const auto grad_lambda_tilde =
                    lambda_tilde.gradient_on_cell(slab_id, slab_cell_id, p, slab_geom);
                const auto grad_u =
                    u_delta.gradient_on_cell(x_cell_id, p, x_geom);
                constexpr int time_component =
                    static_cast<int>(std::tuple_size_v<decltype(grad_u)>) - 1;
                // Keep the time derivative lookup generic so the 1+1D code does
                // not hard-code an index that would later block higher-dimensional
                // space extensions.
                const double grad_theta_tilde_x =
                    grad_lambda_tilde[0] + grad_u[0];
                const auto M_q =
                    coefficients::evaluate_diffusion_tensor<1>(M, p);
                const coefficients::DiffusionVector<1> grad_psi{psi_dx};
                const coefficients::DiffusionVector<1> grad_theta{
                    grad_theta_tilde_x
                };

                const double rhs =
                    cell_tables.partition_of_unity_values[static_cast<std::size_t>(qx)] *
                        (static_cast<double>(ell(p)) - grad_u[time_component]) -
                    coefficients::diffusion_energy_dot(
                        M_q,
                        grad_psi,
                        grad_theta);

                const double rhs_dmu = rhs * jac * w_x * w_t;

                for (int i = 0; i < scalar_space.n_dofs(); ++i)
                {
                    local[i] +=
                        rhs_dmu *
                        scalar_basis_values[static_cast<std::size_t>(i)];
                }
            }
        }
    }
}

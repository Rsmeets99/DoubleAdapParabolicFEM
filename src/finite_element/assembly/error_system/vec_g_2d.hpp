#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "../detail/active_cell_locator_time_slab.hpp"
#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "vec_f_1d.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/vector.hpp"
#include "quadrature/reference_triangle_duffy.hpp"

namespace finite_element::assembly::error_system
{
    template<
        int QSpace,
        int QTime,
        class LocalVectorType,
        class PatchScalarSpaceType,
        class PatchFluxSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class SlabGeometryData,
        class XGeometryData,
        class EllFunction,
        class MFunction>
    void accumulate_patch_scalar_rhs_on_cell_time_2d(
        LocalVectorType& local,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
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
        const auto& cell_data = tables.cell(patch_cell_index);

        for (const auto& qp : cell_data.points)
        {
            const auto grad_lambda_tilde =
                lambda_tilde.gradient_on_cell(
                    slab_id,
                    slab_cell_id,
                    qp.physical_point,
                    slab_geom);
            const auto grad_u =
                u_delta.gradient_on_cell(
                    x_cell_id,
                    qp.physical_point,
                    x_geom);

            const std::array<double, 2> grad_theta_tilde{
                grad_lambda_tilde[0] + grad_u[0],
                grad_lambda_tilde[1] + grad_u[1]
            };

            constexpr int time_component = PatchScalarSpaceType::GT::dim_space_v;
            const double psi_q = qp.partition_of_unity_value;
            const double grad_psi_M_grad_theta =
                coefficients::diffusion_energy_dot(
                    coefficients::evaluate_diffusion_tensor<
                        PatchScalarSpaceType::GT::dim_space_v>(
                            M,
                            qp.physical_point),
                    qp.partition_of_unity_gradient,
                    grad_theta_tilde);
            const double rhs =
                psi_q *
                    (static_cast<double>(ell(qp.physical_point)) -
                     grad_u[time_component]) -
                grad_psi_M_grad_theta;
            const double rhs_dmu = rhs * qp.jacobian_weight;

            for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
            {
                local[i] +=
                    rhs_dmu *
                    qp.scalar_basis_values[static_cast<std::size_t>(i)];
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class LocalVectorType,
        class PatchScalarSpaceType,
        class PatchFluxSpaceType,
        class RHSStateCacheType>
    void accumulate_patch_scalar_rhs_on_cell_time_2d_from_rhs_cache(
        LocalVectorType& local,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        int patch_cell_index,
        int slab_id,
        const RHSStateCacheType& rhs_state_cache)
    {
        const auto& patch = scalar_space.patch();
        const int slab_cell_id = patch.cell(patch_cell_index).slab_cell_id;
        const auto& cell_data = tables.cell(patch_cell_index);
        const auto& rhs_cell =
            rhs_state_cache.cell(slab_id, slab_cell_id);

        for (int qp_id = 0;
             qp_id <
                 finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                     QSpace,
                     QTime,
                     PatchFluxSpaceType,
                     PatchScalarSpaceType>::n_quadrature_points_v;
             ++qp_id)
        {
            const auto& qp =
                cell_data.points[static_cast<std::size_t>(qp_id)];
            const auto& rhs_qp =
                rhs_cell.points[static_cast<std::size_t>(qp_id)];

            const double psi_q = qp.partition_of_unity_value;
            const double grad_psi_M_grad_theta =
                qp.partition_of_unity_gradient[0] *
                    rhs_qp.M_grad_theta_tilde[0] +
                qp.partition_of_unity_gradient[1] *
                    rhs_qp.M_grad_theta_tilde[1];
            const double rhs =
                psi_q *
                    (rhs_qp.ell_value - rhs_qp.u_time_derivative) -
                grad_psi_M_grad_theta;
            const double rhs_dmu = rhs * qp.jacobian_weight;

            for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
            {
                local[i] +=
                    rhs_dmu *
                    qp.scalar_basis_values[static_cast<std::size_t>(i)];
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class PatchScalarSpaceType,
        class PatchFluxSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    requires la::concepts::VectorLike<VectorLike>
        && (PatchScalarSpaceType::GT::dim_space_v == 2)
        && requires { PatchScalarSpaceType::p_time_v; PatchFluxSpaceType::p_time_v; }
    void assemble_vec_g(
        VectorLike& g,
        const PatchScalarSpaceType& scalar_space,
        const PatchFluxSpaceType& flux_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        g.resize(scalar_space.n_dofs());
        g.set_zero();

        la::local::LocalVector global(scalar_space.n_dofs());
        finite_element::assembly::detail::zero_local_vector(global);

        const auto& patch = scalar_space.patch();

        for (int patch_cell_index = 0;
             patch_cell_index < patch.n_cells;
             ++patch_cell_index)
        {
            la::local::FixedLocalVector<PatchScalarSpaceType::local_dofs_v>
                local;
            finite_element::assembly::detail::zero_local_vector(local);

            const auto source_cell_id = patch.cell(patch_cell_index).source_cell_id;
            const int x_cell_id =
                finite_element::assembly::detail::find_active_ancestor_cell_from_source_cell(
                    *context.x_ancestor_cache,
                    *context.x_space,
                    source_cell_id);

            const int slab_id = patch.slab_id;
            const int slab_cell_id = patch.cell(patch_cell_index).slab_cell_id;
            const auto& slab_geom =
                (*context.slab_geometry_caches)[static_cast<std::size_t>(slab_id)]
                    .geometry(slab_cell_id);
            const auto& x_geom =
                context.x_geometry_cache->geometry(x_cell_id);

            accumulate_patch_scalar_rhs_on_cell_time_2d<QSpace, QTime>(
                local,
                scalar_space,
                tables,
                patch_cell_index,
                slab_id,
                lambda_tilde,
                u_delta,
                x_cell_id,
                slab_geom,
                x_geom,
                ell,
                M);

            for (int local_dof_id = 0;
                 local_dof_id < PatchScalarSpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const int patch_dof_id =
                    scalar_space.local_to_patch_dof(
                        patch_cell_index,
                        local_dof_id);
                global[patch_dof_id] += local[local_dof_id];
            }
        }

        for (int i = 0; i < scalar_space.n_dofs(); ++i)
            g[i] = global[i];
    }

    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class PatchScalarSpaceType,
        class PatchFluxSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    requires la::concepts::VectorLike<VectorLike>
        && (PatchScalarSpaceType::GT::dim_space_v == 2)
        && requires { PatchScalarSpaceType::p_time_v; PatchFluxSpaceType::p_time_v; }
    void assemble_vec_g(
        VectorLike& g,
        const PatchScalarSpaceType& scalar_space,
        const PatchFluxSpaceType& flux_space,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType> tables(flux_space, scalar_space);

        assemble_vec_g<QSpace, QTime>(
            g,
            scalar_space,
            flux_space,
            tables,
            context,
            lambda_tilde,
            u_delta,
            ell,
            M);
    }

    template<class PatchScalarSpaceType>
    [[nodiscard]] la::local::LocalVector assemble_local_scalar_integral_row_2d(
        const PatchScalarSpaceType& scalar_space,
        int patch_cell_index)
    {
        static_assert(PatchScalarSpaceType::p_space_v <= 22,
                      "Scalar integral quadrature currently expects p<=22.");

        la::local::LocalVector local(PatchScalarSpaceType::local_dofs_v);

        const double jac = scalar_space.spatial_jacobian_measure(patch_cell_index);
        quadrature::reference::for_each_reference_triangle_duffy_point<
            PatchScalarSpaceType::p_space_v>(
            [&](const double x, const double y, const double weight)
            {
            typename PatchScalarSpaceType::LocalValues values{};
            scalar_space.evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                typename PatchScalarSpaceType::ReferencePoint{x, y},
                values);

            const double dmu = weight * jac;
            for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
                local[i] += values[static_cast<std::size_t>(i)] * dmu;
            });

        return local;
    }

    template<class VectorLike, class PatchScalarSpaceType>
    requires la::concepts::VectorLike<VectorLike>
    void assemble_scalar_integral_row(
        VectorLike& row,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15)
    {
        row.resize(scalar_space.n_dofs());
        row.set_zero();

        for (int patch_cell_index = 0;
             patch_cell_index < scalar_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_scalar_integral_row_2d(
                    scalar_space,
                    patch_cell_index);

            for (int local_dof_id = 0;
                 local_dof_id < PatchScalarSpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const double value = local[local_dof_id];
                if (std::abs(value) <= zero_tol)
                    continue;

                const int patch_dof_id =
                    scalar_space.local_to_patch_dof(
                        patch_cell_index,
                        local_dof_id);
                row.add(patch_dof_id, value);
            }
        }
    }

    template<class VectorLike, class PatchScalarSpaceType>
    requires la::concepts::VectorLike<VectorLike>
    void assemble_vec_g_2d(
        VectorLike& g,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15)
    {
        assemble_scalar_integral_row(
            g,
            scalar_space,
            zero_tol);
    }
}

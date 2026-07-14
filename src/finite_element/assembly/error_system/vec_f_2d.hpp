#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "../detail/active_cell_locator_time_slab.hpp"
#include "../detail/local_error_quadrature_tables_2d.hpp"
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
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class SlabGeometryData,
        class XGeometryData>
    void accumulate_patch_flux_rhs_on_cell_time_2d(
        LocalVectorType& local,
        const PatchFluxSpaceType& flux_space,
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
        const XGeometryData& x_geom)
    {
        const auto& patch = flux_space.patch();
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

            const double psi_q = qp.partition_of_unity_value;
            const double dmu = qp.jacobian_weight;
            const auto& rt_basis_values = qp.rt_basis_values();

            for (int i = 0; i < PatchFluxSpaceType::local_dofs_v; ++i)
            {
                const auto& sigma_i =
                    rt_basis_values[static_cast<std::size_t>(i)];
                local[i] +=
                    -psi_q *
                    (grad_theta_tilde[0] * sigma_i[0] +
                     grad_theta_tilde[1] * sigma_i[1]) *
                    dmu;
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class LocalVectorType,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class RHSStateCacheType>
    void accumulate_patch_flux_rhs_on_cell_time_2d_from_rhs_cache(
        LocalVectorType& local,
        const PatchFluxSpaceType& flux_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        int patch_cell_index,
        int slab_id,
        const RHSStateCacheType& rhs_state_cache)
    {
        const auto& patch = flux_space.patch();
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
            const double dmu = qp.jacobian_weight;
            const auto& rt_basis_values = qp.rt_basis_values();

            for (int i = 0; i < PatchFluxSpaceType::local_dofs_v; ++i)
            {
                const auto& sigma_i =
                    rt_basis_values[static_cast<std::size_t>(i)];
                local[i] +=
                    -psi_q *
                    (rhs_qp.grad_theta_tilde[0] * sigma_i[0] +
                     rhs_qp.grad_theta_tilde[1] * sigma_i[1]) *
                    dmu;
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class XSpaceType,
        class SlabSpaceType>
    requires la::concepts::VectorLike<VectorLike>
        && (PatchFluxSpaceType::GT::dim_space_v == 2)
        && requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_vec_f(
        VectorLike& f,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta)
    {
        f.resize(flux_space.n_dofs());
        f.set_zero();

        la::local::LocalVector global(flux_space.n_dofs());
        finite_element::assembly::detail::zero_local_vector(global);

        const auto& patch = flux_space.patch();

        for (int patch_cell_index = 0;
             patch_cell_index < patch.n_cells;
             ++patch_cell_index)
        {
            la::local::FixedLocalVector<PatchFluxSpaceType::local_dofs_v>
                local;
            finite_element::assembly::detail::zero_local_vector(local);

            const int slab_id = patch.slab_id;
            const int slab_cell_id = patch.cell(patch_cell_index).slab_cell_id;
            const int x_cell_id =
                finite_element::assembly::detail::find_active_ancestor_cell_from_source_cell(
                    *context.x_ancestor_cache,
                    *context.x_space,
                    patch.cell(patch_cell_index).source_cell_id);
            const auto& slab_geom =
                (*context.slab_geometry_caches)[static_cast<std::size_t>(slab_id)]
                    .geometry(slab_cell_id);
            const auto& x_geom =
                context.x_geometry_cache->geometry(x_cell_id);

            accumulate_patch_flux_rhs_on_cell_time_2d<QSpace, QTime>(
                local,
                flux_space,
                tables,
                patch_cell_index,
                slab_id,
                lambda_tilde,
                u_delta,
                x_cell_id,
                slab_geom,
                x_geom);

            const auto& map = flux_space.cell_dof_map(patch_cell_index);
            for (int local_dof_id = 0;
                 local_dof_id < PatchFluxSpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const auto& entry =
                    map[static_cast<std::size_t>(local_dof_id)];
                if (entry.patch_dof_id < 0)
                    continue;

                global[entry.patch_dof_id] +=
                    static_cast<double>(entry.orientation_sign) *
                    local[local_dof_id];
            }
        }

        for (int i = 0; i < flux_space.n_dofs(); ++i)
            f[i] = global[i];
    }

    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class XSpaceType,
        class SlabSpaceType>
    requires la::concepts::VectorLike<VectorLike>
        && (PatchFluxSpaceType::GT::dim_space_v == 2)
        && requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_vec_f(
        VectorLike& f,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta)
    {
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType> tables(flux_space, scalar_space);

        assemble_vec_f<QSpace, QTime>(
            f,
            flux_space,
            scalar_space,
            tables,
            context,
            lambda_tilde,
            u_delta);
    }

    template<class RTFluxSpaceType>
    [[nodiscard]] la::local::LocalVector assemble_local_rt_divergence_integral_row_2d(
        const RTFluxSpaceType& flux_space,
        int patch_cell_index)
    {
        static_assert(RTFluxSpaceType::p_space_v <= 10,
                      "RT divergence integral quadrature currently expects p<=10.");

        la::local::LocalVector local(RTFluxSpaceType::local_dofs_v);

        const auto map =
            flux_space.physical_map_for_patch_cell(patch_cell_index);
        const double jac =
            RTFluxSpaceType::PiolaBasis::jacobian_measure(map);

        quadrature::reference::for_each_reference_triangle_duffy_point<
            RTFluxSpaceType::p_space_v>(
            [&](const double x, const double y, const double weight)
            {
            typename RTFluxSpaceType::LocalDivergences divergences{};
            flux_space.evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                typename RTFluxSpaceType::ReferencePoint{x, y},
                divergences);

            const double dmu = weight * jac;
            for (int i = 0; i < RTFluxSpaceType::local_dofs_v; ++i)
                local[i] += divergences[static_cast<std::size_t>(i)] * dmu;
            });

        return local;
    }

    template<class VectorLike, class RTFluxSpaceType>
    requires la::concepts::VectorLike<VectorLike>
    void assemble_rt_divergence_integral_row(
        VectorLike& row,
        const RTFluxSpaceType& flux_space,
        double zero_tol = 1.0e-15)
    {
        row.resize(flux_space.n_dofs());
        row.set_zero();

        for (int patch_cell_index = 0;
             patch_cell_index < flux_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_rt_divergence_integral_row_2d(
                    flux_space,
                    patch_cell_index);
            const auto& map = flux_space.cell_dof_map(patch_cell_index);

            for (int local_dof_id = 0;
                 local_dof_id < RTFluxSpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const auto& entry =
                    map[static_cast<std::size_t>(local_dof_id)];
                if (entry.patch_dof_id < 0)
                    continue;

                const double value =
                    static_cast<double>(entry.orientation_sign) *
                    local[local_dof_id];
                if (std::abs(value) > zero_tol)
                    row.add(entry.patch_dof_id, value);
            }
        }
    }

    template<class VectorLike, class RTFluxSpaceType>
    requires la::concepts::VectorLike<VectorLike>
    void assemble_vec_f_2d(
        VectorLike& f,
        const RTFluxSpaceType& flux_space,
        double zero_tol = 1.0e-15)
    {
        assemble_rt_divergence_integral_row(
            f,
            flux_space,
            zero_tol);
    }
}

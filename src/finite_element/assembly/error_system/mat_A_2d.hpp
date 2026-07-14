#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../error_fespace/patch_scalar_space_time_2d.hpp"
#include "patch_assembly_maps_2d.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"
#include "quadrature/reference_triangle_duffy.hpp"

namespace finite_element::assembly::error_system
{
    namespace detail
    {
        using DiffusionTensor2D = coefficients::DiffusionTensor<2>;

        [[nodiscard]] inline DiffusionTensor2D inverse_diffusion_tensor_2d(
            const DiffusionTensor2D& M_value) noexcept
        {
            const double a = M_value[0][0];
            const double b = M_value[0][1];
            const double c = M_value[1][1];
            const double det = a * c - b * b;

            return DiffusionTensor2D{
                std::array<double, 2>{c / det, -b / det},
                std::array<double, 2>{-b / det, a / det}
            };
        }

        template<class VectorValue>
        [[nodiscard]] VectorValue apply_inverse_tensor_2d(
            const DiffusionTensor2D& M_inverse,
            const VectorValue& value,
            const double scale) noexcept
        {
            return VectorValue{
                scale * (M_inverse[0][0] * value[0] +
                         M_inverse[0][1] * value[1]),
                scale * (M_inverse[1][0] * value[0] +
                         M_inverse[1][1] * value[1])
            };
        }

        template<
            int QSpace,
            int QTime,
            class RTFluxSpaceType,
            class PatchScalarSpaceType,
            class MFunction>
        class LocalRTMassQuadratureData2D
        {
        public:
            using BaseTables =
                finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                    QSpace,
                    QTime,
                    RTFluxSpaceType,
                    PatchScalarSpaceType>;
            using VectorValue = typename RTFluxSpaceType::VectorValue;
            using RTBasisValues = typename RTFluxSpaceType::LocalValues;

            static constexpr int n_quadrature_points_v =
                BaseTables::n_quadrature_points_v;

            struct QuadraturePointData
            {
                double jacobian_weight = 0.0;
                DiffusionTensor2D inverse_diffusion_tensor{};
                RTBasisValues weighted_inverse_rt_basis_values{};
            };

            struct CellData
            {
                int patch_cell_index = -1;
                std::array<QuadraturePointData, n_quadrature_points_v> points{};
            };

            LocalRTMassQuadratureData2D(
                const BaseTables& tables,
                const MFunction& M)
            {
                cells_.resize(static_cast<std::size_t>(tables.n_patch_cells()));
                for (int patch_cell_index = 0;
                     patch_cell_index < tables.n_patch_cells();
                     ++patch_cell_index)
                {
                    fill_cell_(
                        tables.cell(patch_cell_index),
                        M,
                        cells_[static_cast<std::size_t>(patch_cell_index)]);
                }
            }

            [[nodiscard]] int n_patch_cells() const noexcept
            {
                return static_cast<int>(cells_.size());
            }

            [[nodiscard]] const CellData& cell(const int patch_cell_index) const
            {
                if (patch_cell_index < 0 ||
                    patch_cell_index >= static_cast<int>(cells_.size()))
                {
                    throw std::runtime_error(
                        "LocalRTMassQuadratureData2D: patch cell index out of range.");
                }

                return cells_[static_cast<std::size_t>(patch_cell_index)];
            }

        private:
            std::vector<CellData> cells_{};

            static void fill_cell_(
                const typename BaseTables::CellData& source_cell,
                const MFunction& M,
                CellData& cell_data)
            {
                cell_data.patch_cell_index = source_cell.patch_cell_index;

                for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
                {
                    const auto& source_qp =
                        source_cell.points[static_cast<std::size_t>(qp_id)];
                    auto& qp =
                        cell_data.points[static_cast<std::size_t>(qp_id)];

                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<
                            RTFluxSpaceType::GT::dim_space_v>(
                                M,
                                source_qp.physical_point);
                    qp.jacobian_weight = source_qp.jacobian_weight;
                    qp.inverse_diffusion_tensor =
                        inverse_diffusion_tensor_2d(M_q);
                    const auto& rt_basis_values =
                        source_qp.rt_basis_values();

                    for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
                    {
                        qp.weighted_inverse_rt_basis_values[
                            static_cast<std::size_t>(j)] =
                            apply_inverse_tensor_2d(
                                qp.inverse_diffusion_tensor,
                                rt_basis_values[
                                    static_cast<std::size_t>(j)],
                                qp.jacobian_weight);
                    }
                }
            }
        };
    }

    template<
        int QSpace,
        int QTime,
        class RTFluxSpaceType,
        class PatchScalarSpaceType,
        class MFunction>
    [[nodiscard]] la::local::LocalMatrix
    assemble_local_rt_mass_matrix_time_2d_reference(
        const RTFluxSpaceType& flux_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            RTFluxSpaceType,
            PatchScalarSpaceType>& tables,
        int patch_cell_index,
        const MFunction& M)
    {
        static_cast<void>(flux_space);

        la::local::LocalMatrix local(
            RTFluxSpaceType::local_dofs_v,
            RTFluxSpaceType::local_dofs_v);

        const auto& cell_data = tables.cell(patch_cell_index);
        for (const auto& qp : cell_data.points)
        {
            const auto& rt_basis_values = qp.rt_basis_values();
            const auto M_q =
                coefficients::evaluate_diffusion_tensor<
                    RTFluxSpaceType::GT::dim_space_v>(
                        M,
                        qp.physical_point);
            const double dmu = qp.jacobian_weight;

            for (int i = 0; i < RTFluxSpaceType::local_dofs_v; ++i)
            {
                const auto& sigma_i =
                    rt_basis_values[static_cast<std::size_t>(i)];
                for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
                {
                    const auto& sigma_j =
                        rt_basis_values[static_cast<std::size_t>(j)];
                    local(i, j) +=
                        coefficients::inverse_diffusion_dot(
                            M_q,
                            sigma_i,
                            sigma_j) *
                        dmu;
                }
            }
        }

        return local;
    }

    template<
        int QSpace,
        int QTime,
        class RTFluxSpaceType,
        class PatchScalarSpaceType,
        class MFunction>
    [[nodiscard]] la::local::LocalMatrix assemble_local_rt_mass_matrix_time_2d(
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            RTFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const detail::LocalRTMassQuadratureData2D<
            QSpace,
            QTime,
            RTFluxSpaceType,
            PatchScalarSpaceType,
            MFunction>& mass_data,
        int patch_cell_index)
    {
        la::local::LocalMatrix local(
            RTFluxSpaceType::local_dofs_v,
            RTFluxSpaceType::local_dofs_v);

        const auto& cell_data = tables.cell(patch_cell_index);
        const auto& mass_cell_data = mass_data.cell(patch_cell_index);
        for (int qp_id = 0;
             qp_id < detail::LocalRTMassQuadratureData2D<
                         QSpace,
                         QTime,
                         RTFluxSpaceType,
                         PatchScalarSpaceType,
                         MFunction>::n_quadrature_points_v;
             ++qp_id)
        {
            const auto& qp =
                cell_data.points[static_cast<std::size_t>(qp_id)];
            const auto& mass_qp =
                mass_cell_data.points[static_cast<std::size_t>(qp_id)];
            const auto& rt_basis_values = qp.rt_basis_values();

            for (int i = 0; i < RTFluxSpaceType::local_dofs_v; ++i)
            {
                const auto& sigma_i =
                    rt_basis_values[static_cast<std::size_t>(i)];
                for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
                {
                    const auto& M_inv_sigma_j_dmu =
                        mass_qp.weighted_inverse_rt_basis_values[
                            static_cast<std::size_t>(j)];
                    local(i, j) +=
                        sigma_i[0] * M_inv_sigma_j_dmu[0] +
                        sigma_i[1] * M_inv_sigma_j_dmu[1];
                }
            }
        }

        return local;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class RTFluxSpaceType,
        class PatchScalarSpaceType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (RTFluxSpaceType::GT::dim_space_v == 2) &&
             requires { RTFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_mat_A_time_2d_reference(
        typename Backend::SparseMatrix& A,
        const RTFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            RTFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const MFunction& M,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells() ||
            tables.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_mat_A_time_2d_reference: patch cell count mismatch.");
        }

        using SparseBuilder = typename Backend::SparseBuilder;

        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(flux_space.n_patch_cells()) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v));

        for (int patch_cell_index = 0;
             patch_cell_index < flux_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_rt_mass_matrix_time_2d_reference<QSpace, QTime>(
                    flux_space,
                    tables,
                    patch_cell_index,
                    M);
            scatter_rt_local_matrix_2d(
                builder,
                local,
                flux_space,
                patch_cell_index,
                zero_tol);
        }

        A.resize(flux_space.n_dofs(), flux_space.n_dofs());
        A.set_from_builder(builder);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class RTFluxSpaceType,
        class PatchScalarSpaceType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (RTFluxSpaceType::GT::dim_space_v == 2) &&
             requires { RTFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_mat_A(
        typename Backend::SparseMatrix& A,
        const RTFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            RTFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const MFunction& M,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells() ||
            tables.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_mat_A<2D time-dependent>: patch cell count mismatch.");
        }

        using SparseBuilder = typename Backend::SparseBuilder;
        using MassData =
            detail::LocalRTMassQuadratureData2D<
                QSpace,
                QTime,
                RTFluxSpaceType,
                PatchScalarSpaceType,
                MFunction>;

        const MassData mass_data(tables, M);
        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(flux_space.n_patch_cells()) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v));

        for (int patch_cell_index = 0;
             patch_cell_index < flux_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_rt_mass_matrix_time_2d<QSpace, QTime>(
                    tables,
                    mass_data,
                    patch_cell_index);
            scatter_rt_local_matrix_2d(
                builder,
                local,
                flux_space,
                patch_cell_index,
                zero_tol);
        }

        A.resize(flux_space.n_dofs(), flux_space.n_dofs());
        A.set_from_builder(builder);
    }

    template<int QSpace, int QTime, class Backend, class RTFluxSpaceType, class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (RTFluxSpaceType::GT::dim_space_v == 2) &&
             requires { RTFluxSpaceType::p_time_v; }
    void assemble_mat_A(
        typename Backend::SparseMatrix& A,
        const RTFluxSpaceType& flux_space,
        const MFunction& M,
        double zero_tol = 1.0e-15)
    {
        using PatchScalarSpaceType =
            error_fespace::PatchScalarSpaceTime2D<
                typename RTFluxSpaceType::Patch,
                RTFluxSpaceType::p_space_v,
                RTFluxSpaceType::p_time_v>;
        using Tables =
            finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                QSpace,
                QTime,
                RTFluxSpaceType,
                PatchScalarSpaceType>;

        const PatchScalarSpaceType scalar_space(
            flux_space.patch(),
            flux_space.slab_space());
        const Tables tables(flux_space, scalar_space);

        assemble_mat_A<QSpace, QTime, Backend>(
            A,
            flux_space,
            scalar_space,
            tables,
            M,
            zero_tol);
    }

    template<class RTFluxSpaceType>
    [[nodiscard]] la::local::LocalMatrix assemble_local_rt_mass_matrix_2d(
        const RTFluxSpaceType& flux_space,
        int patch_cell_index)
    {
        static_assert(RTFluxSpaceType::p_space_v <= 10,
                      "RT local mass quadrature currently expects p<=10.");

        la::local::LocalMatrix local(
            RTFluxSpaceType::local_dofs_v,
            RTFluxSpaceType::local_dofs_v);

        const auto map =
            flux_space.physical_map_for_patch_cell(patch_cell_index);
        const double jac =
            RTFluxSpaceType::PiolaBasis::jacobian_measure(map);

        quadrature::reference::for_each_reference_triangle_duffy_point<
            2 * RTFluxSpaceType::p_space_v + 2>(
            [&](const double x, const double y, const double weight)
            {
            typename RTFluxSpaceType::LocalValues values{};
            flux_space.evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                typename RTFluxSpaceType::ReferencePoint{x, y},
                values);

            const double dmu = weight * jac;
            for (int i = 0; i < RTFluxSpaceType::local_dofs_v; ++i)
            {
                const auto& vi = values[static_cast<std::size_t>(i)];
                for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
                {
                    const auto& vj = values[static_cast<std::size_t>(j)];
                    local(i, j) +=
                        (vi[0] * vj[0] + vi[1] * vj[1]) * dmu;
                }
            }
            });

        return local;
    }

    template<class Backend, class RTFluxSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_A_2d(
        typename Backend::SparseMatrix& A,
        const RTFluxSpaceType& flux_space,
        double zero_tol = 1.0e-15)
    {
        using SparseBuilder = typename Backend::SparseBuilder;

        SparseBuilder builder;
        builder.reserve(
            static_cast<std::size_t>(flux_space.n_patch_cells()) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v) *
            static_cast<std::size_t>(RTFluxSpaceType::local_dofs_v));

        for (int patch_cell_index = 0;
             patch_cell_index < flux_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local =
                assemble_local_rt_mass_matrix_2d(
                    flux_space,
                    patch_cell_index);
            scatter_rt_local_matrix_2d(
                builder,
                local,
                flux_space,
                patch_cell_index,
                zero_tol);
        }

        A.resize(flux_space.n_dofs(), flux_space.n_dofs());
        A.set_from_builder(builder);
    }

    template<class Backend, class RTFluxSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_rt_mass_matrix(
        typename Backend::SparseMatrix& A,
        const RTFluxSpaceType& flux_space,
        double zero_tol = 1.0e-15)
    {
        assemble_mat_A_2d<Backend>(A, flux_space, zero_tol);
    }
}

#pragma once

#include <array>
#include <numeric>
#include <tuple>

#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../geometry/cell_geometry.hpp"
#include "zero_local.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::detail
{
    enum class MixedBilinearFormPart
    {
        TimeDerivative,
        SpatialDiffusion,
        Full
    };

    // -------------------------------------------------------------------------
    // Assemble local mixed matrix on a fine Y-cell:
    //
    //   B_ij^K = \int_K [ (partial_t psi_i) phi_j
    //                   + grad_x psi_i^T M grad_x phi_j ] dx dt
    //
    // psi_i: X-space basis, evaluated at runtime on containing X-cell
    // phi_j: Y-space basis, taken from pretabulated Y tables on the Y-cell rule
    // -------------------------------------------------------------------------
    template<
        MixedBilinearFormPart Part,
        class XBasis,
        class YTables,
        class XGeometry,
        class YGeometry,
        class MFunction,
        class LocalMatrixType>
    void assemble_local_mixed_space_time_matrix_part(
        LocalMatrixType& local,
        const typename XGeometry::Data& x_geom,
        const typename YGeometry::Data& y_geom,
        const MFunction& M)
    {
        constexpr int x_dofs_per_cell = XBasis::N;
        constexpr int y_dofs_per_cell = YTables::n_basis;
        constexpr int n_q             = YTables::n_cell_q;
        constexpr int dim_space =
            static_cast<int>(
                std::tuple_size_v<typename XGeometry::SpatialGradient>);

        static_assert(
            dim_space ==
            static_cast<int>(
                std::tuple_size_v<typename YGeometry::SpatialGradient>),
            "Mixed local assembly requires matching spatial gradient dimensions.");

        if (local.rows != x_dofs_per_cell || local.cols != y_dofs_per_cell)
            local.resize(x_dofs_per_cell, y_dofs_per_cell);
        else
            zero_local_matrix(local);

        const double jac_y = YGeometry::jacobian_measure(y_geom);

        for (int q = 0; q < n_q; ++q)
        {
            const auto& xi_y          = YTables::cell_rule.points[q];
            const double w_q          = YTables::cell_rule.weights[q];
            const auto& phi_vals      = YTables::values_on_cell_qp(q);
            const auto& phi_grads_ref = YTables::gradients_on_cell_qp(q);

            const auto x_q  = YGeometry::map_to_physical(y_geom, xi_y);
            const auto xi_x = XGeometry::physical_to_reference(x_geom, x_q);
            const auto M_q =
                [&]()
                {
                    if constexpr (
                        Part == MixedBilinearFormPart::SpatialDiffusion ||
                        Part == MixedBilinearFormPart::Full)
                    {
                        return coefficients::evaluate_diffusion_tensor<
                            static_cast<std::size_t>(dim_space)>(
                                M,
                                x_q);
                    }
                    else
                    {
                        return coefficients::identity_diffusion_tensor<
                            static_cast<std::size_t>(dim_space)>();
                    }
                }();

            const auto psi_grads_ref = XBasis::grad_all(xi_x);

            const double dmu = jac_y * w_q;
            std::array<double, x_dofs_per_cell> dt_psi_on_q{};
            std::array<typename XGeometry::SpatialGradient, x_dofs_per_cell> gradx_psi_on_q{};
            std::array<typename YGeometry::SpatialGradient, y_dofs_per_cell> gradx_phi_on_q{};
            std::array<typename YGeometry::SpatialGradient, y_dofs_per_cell> M_gradx_phi_on_q{};

            for (int i = 0; i < x_dofs_per_cell; ++i)
            {
                dt_psi_on_q[static_cast<std::size_t>(i)] =
                    XGeometry::time_derivative(x_geom, psi_grads_ref[i]);
                gradx_psi_on_q[static_cast<std::size_t>(i)] =
                    XGeometry::spatial_gradient(x_geom, psi_grads_ref[i]);
            }

            for (int j = 0; j < y_dofs_per_cell; ++j)
            {
                gradx_phi_on_q[static_cast<std::size_t>(j)] =
                    YGeometry::spatial_gradient(y_geom, phi_grads_ref[j]);
                if constexpr (
                    Part == MixedBilinearFormPart::SpatialDiffusion ||
                    Part == MixedBilinearFormPart::Full)
                {
                    M_gradx_phi_on_q[static_cast<std::size_t>(j)] =
                        coefficients::apply_validated_M<
                            static_cast<std::size_t>(dim_space)>(
                            M_q,
                            gradx_phi_on_q[static_cast<std::size_t>(j)]);
                }
            }

            for (int i = 0; i < x_dofs_per_cell; ++i)
            {
                const double dt_psi_i = dt_psi_on_q[static_cast<std::size_t>(i)];
                const auto& gradx_psi_i = gradx_psi_on_q[static_cast<std::size_t>(i)];

                for (int j = 0; j < y_dofs_per_cell; ++j)
                {
                    double value = 0.0;
                    if constexpr (
                        Part == MixedBilinearFormPart::TimeDerivative ||
                        Part == MixedBilinearFormPart::Full)
                    {
                        value += dt_psi_i * phi_vals[j];
                    }
                    if constexpr (
                        Part == MixedBilinearFormPart::SpatialDiffusion ||
                        Part == MixedBilinearFormPart::Full)
                    {
                        value += coefficients::dot<
                            static_cast<std::size_t>(dim_space)>(
                            gradx_psi_i,
                            M_gradx_phi_on_q[static_cast<std::size_t>(j)]);
                    }

                    local(i, j) += value * dmu;
                }
            }
        }
    }

    template<
        class XBasis,
        class YTables,
        class XGeometry,
        class YGeometry,
        class MFunction,
        class LocalMatrixType>
    void assemble_local_mixed_time_derivative_matrix(
        LocalMatrixType& local,
        const typename XGeometry::Data& x_geom,
        const typename YGeometry::Data& y_geom,
        const MFunction& M)
    {
        assemble_local_mixed_space_time_matrix_part<
            MixedBilinearFormPart::TimeDerivative,
            XBasis,
            YTables,
            XGeometry,
            YGeometry>(
                local,
                x_geom,
                y_geom,
                M);
    }

    template<
        class XBasis,
        class YTables,
        class XGeometry,
        class YGeometry,
        class LocalMatrixType>
    void assemble_local_mixed_time_derivative_matrix(
        LocalMatrixType& local,
        const typename XGeometry::Data& x_geom,
        const typename YGeometry::Data& y_geom)
    {
        constexpr auto dim_space =
            std::tuple_size_v<typename XGeometry::SpatialGradient>;
        assemble_local_mixed_time_derivative_matrix<
            XBasis,
            YTables,
            XGeometry,
            YGeometry>(
                local,
                x_geom,
                y_geom,
                coefficients::IdentityDiffusion<dim_space>{});
    }

    template<
        class XBasis,
        class YTables,
        class XGeometry,
        class YGeometry,
        class MFunction,
        class LocalMatrixType>
    void assemble_local_mixed_spatial_diffusion_matrix(
        LocalMatrixType& local,
        const typename XGeometry::Data& x_geom,
        const typename YGeometry::Data& y_geom,
        const MFunction& M)
    {
        assemble_local_mixed_space_time_matrix_part<
            MixedBilinearFormPart::SpatialDiffusion,
            XBasis,
            YTables,
            XGeometry,
            YGeometry>(
                local,
                x_geom,
                y_geom,
                M);
    }

    template<
        class XBasis,
        class YTables,
        class XGeometry,
        class YGeometry,
        class LocalMatrixType>
    void assemble_local_mixed_spatial_diffusion_matrix(
        LocalMatrixType& local,
        const typename XGeometry::Data& x_geom,
        const typename YGeometry::Data& y_geom)
    {
        constexpr auto dim_space =
            std::tuple_size_v<typename XGeometry::SpatialGradient>;
        assemble_local_mixed_spatial_diffusion_matrix<
            XBasis,
            YTables,
            XGeometry,
            YGeometry>(
                local,
                x_geom,
                y_geom,
                coefficients::IdentityDiffusion<dim_space>{});
    }

    template<
        class XBasis,
        class YTables,
        class XGeometry,
        class YGeometry,
        class MFunction,
        class LocalMatrixType>
    void assemble_local_mixed_space_time_matrix(
        LocalMatrixType& local,
        const typename XGeometry::Data& x_geom,
        const typename YGeometry::Data& y_geom,
        const MFunction& M)
    {
        assemble_local_mixed_space_time_matrix_part<
            MixedBilinearFormPart::Full,
            XBasis,
            YTables,
            XGeometry,
            YGeometry>(
                local,
                x_geom,
                y_geom,
                M);
    }

    template<
        class XBasis,
        class YTables,
        class XGeometry,
        class YGeometry,
        class LocalMatrixType>
    void assemble_local_mixed_space_time_matrix(
        LocalMatrixType& local,
        const typename XGeometry::Data& x_geom,
        const typename YGeometry::Data& y_geom)
    {
        constexpr auto dim_space =
            std::tuple_size_v<typename XGeometry::SpatialGradient>;
        assemble_local_mixed_space_time_matrix<
            XBasis,
            YTables,
            XGeometry,
            YGeometry>(
                local,
                x_geom,
                y_geom,
                coefficients::IdentityDiffusion<dim_space>{});
    }
}

#pragma once

#include <array>
#include <tuple>

#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../geometry/cell_geometry.hpp"
#include "zero_local.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::detail
{
    // -------------------------------------------------------------------------
    // Assemble the local cell matrix for
    //
    //   a_K(u,v) = \int_K coeff(x) (grad_x u)·(grad_x v) dx dt
    // -------------------------------------------------------------------------
    template<
        class Tables,
        class Geometry,
        class CoefficientFunction,
        class LocalMatrixType>
    void assemble_local_spatial_diffusion_matrix(
        LocalMatrixType& local,
        const typename Geometry::Data& geom,
        const CoefficientFunction& coeff)
    {
        constexpr int dofs_per_cell = Tables::n_basis;
        constexpr int n_q           = Tables::n_cell_q;
        constexpr int dim_space =
            static_cast<int>(
                std::tuple_size_v<typename Geometry::SpatialGradient>);

        if (local.rows != dofs_per_cell || local.cols != dofs_per_cell)
            local.resize(dofs_per_cell, dofs_per_cell);
        else
            zero_local_matrix(local);

        const double jac = Geometry::jacobian_measure(geom);

        for (int q = 0; q < n_q; ++q)
        {
            const auto& xi_q  = Tables::cell_rule.points[q];
            const double w_q  = Tables::cell_rule.weights[q];
            const auto& grads = Tables::gradients_on_cell_qp(q);

            const auto x_q  = Geometry::map_to_physical(geom, xi_q);
            const auto M_q =
                coefficients::evaluate_diffusion_tensor<
                    dim_space>(
                        coeff,
                        x_q);
            const double dmu = jac * w_q;
            std::array<typename Geometry::SpatialGradient, dofs_per_cell> spatial_grads_on_q{};
            std::array<typename Geometry::SpatialGradient, dofs_per_cell> M_spatial_grads_on_q{};

            for (int i = 0; i < dofs_per_cell; ++i)
            {
                spatial_grads_on_q[static_cast<std::size_t>(i)] =
                    Geometry::spatial_gradient(geom, grads[i]);
                M_spatial_grads_on_q[static_cast<std::size_t>(i)] =
                    coefficients::apply_validated_M<dim_space>(
                        M_q,
                        spatial_grads_on_q[static_cast<std::size_t>(i)]);
            }

            for (int i = 0; i < dofs_per_cell; ++i)
            {
                const auto& grad_i = spatial_grads_on_q[static_cast<std::size_t>(i)];
                for (int j = 0; j < dofs_per_cell; ++j)
                {
                    local(i, j) +=
                        coefficients::dot<dim_space>(
                            grad_i,
                            M_spatial_grads_on_q[static_cast<std::size_t>(j)]) *
                        dmu;
                }
            }
        }
    }
}

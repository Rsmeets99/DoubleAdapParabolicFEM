#pragma once

#include "trace_geometry_utils.hpp"
#include "zero_local.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::detail
{
    // -------------------------------------------------------------------------
    // Assemble the local initial-trace load vector
    //
    //   g_K(i) = scale * \int_{K \cap {t=0}} u0(x) psi_i(x,0) dx
    //
    // using the pretabulated bottom-face values from the basis tables.
    // -------------------------------------------------------------------------
    template<
        class Tables,
        class Geometry,
        class InitialValueFunction,
        class LocalVectorType>
    void assemble_local_initial_trace_rhs_vector(
        LocalVectorType& local,
        const typename Geometry::Data& geom,
        const InitialValueFunction& u0,
        double scale = 1.0)
    {
        constexpr int dofs_per_cell = Tables::n_basis;
        constexpr int n_q           = Tables::n_bottom_q;

        if (local.size != dofs_per_cell)
            local.resize(dofs_per_cell);
        else
            zero_local_vector(local);

        const double trace_jac = initial_trace_measure<Geometry>(geom);

        for (int q = 0; q < n_q; ++q)
        {
            const auto& xi_bottom = Tables::bottom_rule.points[q];
            const double w_q      = Tables::bottom_rule.weights[q];
            const auto& psi_vals  = Tables::values_on_bottom_qp(q);

            const auto x_q_st   = map_bottom_qp_to_physical<Geometry>(geom, xi_bottom);
            const auto x_q      = spatial_argument_from_space_time_point<Geometry>(x_q_st);
            const double u0_q   = static_cast<double>(u0(x_q));
            const double dgamma = scale * trace_jac * w_q;

            for (int i = 0; i < dofs_per_cell; ++i)
                local[i] += u0_q * psi_vals[i] * dgamma;
        }
    }
}

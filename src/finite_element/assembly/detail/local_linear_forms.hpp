#pragma once

#include "../../geometry/cell_geometry.hpp"
#include "zero_local.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::detail
{
    // -------------------------------------------------------------------------
    // Assemble the local cell load vector
    //
    //   f_K(i) = \int_K rhs(x) phi_i(x) dx dt
    //
    // using compile-time basis/quadrature tables.
    // -------------------------------------------------------------------------
    template<class Tables, class Geometry, class RhsFunction, class LocalVectorType>
    void assemble_local_volume_rhs_vector(
        LocalVectorType& local,
        const typename Geometry::Data& geom,
        const RhsFunction& rhs)
    {
        constexpr int dofs_per_cell = Tables::n_basis;
        constexpr int n_q           = Tables::n_cell_q;

        if (local.size != dofs_per_cell)
            local.resize(dofs_per_cell);
        else
            zero_local_vector(local);

        const double jac = Geometry::jacobian_measure(geom);

        for (int q = 0; q < n_q; ++q)
        {
            const auto& xi_q   = Tables::cell_rule.points[q];
            const double w_q   = Tables::cell_rule.weights[q];
            const auto& values = Tables::values_on_cell_qp(q);

            const auto x_q   = Geometry::map_to_physical(geom, xi_q);
            const double f_q = static_cast<double>(rhs(x_q));
            const double dmu = jac * w_q;

            for (int i = 0; i < dofs_per_cell; ++i)
                local[i] += f_q * values[i] * dmu;
        }
    }
}

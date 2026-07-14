#pragma once

#include "trace_geometry_utils.hpp"
#include "zero_local.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::detail
{
    // -------------------------------------------------------------------------
    // Assemble the signed local initial-trace matrix used as the saddle
    // system's lower-right block C.
    //
    //   c_K(u,v) = -\int_{K \cap {t=0}} u(x,0) v(x,0) dx
    //
    // Thus the assembled block is already C_signed = -gamma_0^T gamma_0.
    // Later positive Schur-complement formulas should use C_pos = -C_signed.
    // -------------------------------------------------------------------------
    template<class Tables, class Geometry, class LocalMatrixType>
    void assemble_local_initial_trace_mass_matrix(
        LocalMatrixType& local,
        const typename Geometry::Data& geom)
    {
        constexpr int dofs_per_cell = Tables::n_basis;
        constexpr int n_q           = Tables::n_bottom_q;

        if (local.rows != dofs_per_cell || local.cols != dofs_per_cell)
            local.resize(dofs_per_cell, dofs_per_cell);
        else
            zero_local_matrix(local);

        const double trace_jac = initial_trace_measure<Geometry>(geom);

        for (int q = 0; q < n_q; ++q)
        {
            const double w_q     = Tables::bottom_rule.weights[q];
            const auto& psi_vals = Tables::values_on_bottom_qp(q);
            const double dgamma  = trace_jac * w_q;

            for (int i = 0; i < dofs_per_cell; ++i)
            {
                for (int j = 0; j < dofs_per_cell; ++j)
                {
                    local(i, j) -= psi_vals[i] * psi_vals[j] * dgamma;
                }
            }
        }
    }
}

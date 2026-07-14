#pragma once

#include "../detail/active_cell_locator.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../../system/main_preconditioner_context.hpp"
#include "mat_A.hpp"
#include "mat_B.hpp"

#include "../../../linear_algebra/preconditioners/schur_approximations.hpp"

namespace finite_element::assembly
{
    template<
        int QSpace,
        int QTime,
        class Backend,
        class YSpaceType,
        class XSpaceType,
        class MFunction>
    [[nodiscard]] finite_element::system::ParabolicGraphNormPreconditionerContext<
        Backend>
    assemble_parabolic_graph_norm_approximation(
        const YSpaceType& y_space,
        const XSpaceType& x_space,
        const MFunction& M,
        const typename Backend::SparseMatrix& A_y,
        const typename Backend::SparseMatrix& C_signed,
        double zero_tol = 1.0e-15,
        double diagonal_tolerance = 1.0e-14)
    {
        typename Backend::SparseMatrix B_dt;
        typename Backend::SparseMatrix A_x;

        detail::AssemblySpaceCache<XSpaceType> x_cache(x_space);
        detail::AssemblySpaceCache<YSpaceType> y_cache(y_space);
        detail::ActiveAncestorCache<XSpaceType> ancestor_cache(x_space);

        finite_element::assembly::assemble_mat_B_dt<QSpace, QTime, Backend>(
            B_dt,
            x_space,
            y_space,
            x_cache,
            y_cache,
            ancestor_cache,
            zero_tol);

        finite_element::assembly::assemble_mat_A<QSpace, QTime, Backend>(
            A_x,
            x_space,
            M,
            x_cache,
            zero_tol);

        finite_element::system::ParabolicGraphNormPreconditionerContext<
            Backend>
            context;
        context.approximation =
            la::preconditioners::parabolic_graph_norm_approximation<Backend>(
                A_y,
                B_dt,
                C_signed,
                A_x,
                diagonal_tolerance);
        return context;
    }
}

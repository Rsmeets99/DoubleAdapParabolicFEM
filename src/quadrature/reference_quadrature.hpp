#pragma once

#include "gauss_legendre_1d.hpp"
#include "dunavant_triangle.hpp"
#include "reference_triangle_duffy.hpp"
#include "tensor_product_quadrature.hpp"

namespace quadrature::reference
{
    // Reference 1D interval quadrature on [0,1]
    template<int Order>
    constexpr auto make_reference_interval_quadrature()
    {
        return quadrature::gauss_legendre::gauss_legendre_rule_1d<Order>;
    }

    // Reference triangle quadrature on conv{(0,0),(1,0),(0,1)}
    template<int Degree>
    constexpr auto make_reference_triangle_quadrature()
    {
        return quadrature::reference::duffy_triangle_rule<Degree>;
    }

    // Reference 1+1D quadrilateral = [0,1] x [0,1]
    template<int OrderX, int OrderT>
    constexpr auto make_reference_quadrilateral_space_time_quadrature()
    {
        return quadrature::tensor_product::make_quadrilateral_space_time_rule(
            quadrature::gauss_legendre::gauss_legendre_rule_1d<OrderX>,
            quadrature::gauss_legendre::gauss_legendre_rule_1d<OrderT>
        );
    }

    // Reference 2+1D triangular prism = T_ref x [0,1]
    template<int DegreeXY, int OrderT>
    constexpr auto make_reference_triangular_prism_space_time_quadrature()
    {
        return quadrature::tensor_product::make_triangular_prism_space_time_rule(
            quadrature::reference::duffy_triangle_rule<DegreeXY>,
            quadrature::gauss_legendre::gauss_legendre_rule_1d<OrderT>
        );
    }
}

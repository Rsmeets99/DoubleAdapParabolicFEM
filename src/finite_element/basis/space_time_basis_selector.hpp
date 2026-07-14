#pragma once

#include "functions/quadrilateral_lagrange.hpp"
#include "functions/triangular_prism_lagrange.hpp"

namespace finite_element::basis
{
    template<typename GeomTraits, typename FETraits>
    struct SpaceTimeBasisSelector;

    template<typename GeomTraits, typename FETraits>
        requires (GeomTraits::dim_space_v == 1)
    struct SpaceTimeBasisSelector<GeomTraits, FETraits>
    {
        using type = functions::QuadrilateralLagrangeBasis<
            FETraits::p_space_v,
            FETraits::p_time_v,
            typename FETraits::SpatialNodes,
            typename FETraits::TemporalNodes>;
    };

    template<typename GeomTraits, typename FETraits>
        requires (GeomTraits::dim_space_v == 2)
    struct SpaceTimeBasisSelector<GeomTraits, FETraits>
    {
        using type = functions::TriangularPrismLagrangeBasis<
            FETraits::p_space_v,
            FETraits::p_time_v,
            typename FETraits::SpatialNodes,
            typename FETraits::TemporalNodes>;
    };

    template<typename GeomTraits, typename FETraits>
    using SpaceTimeBasis =
        typename SpaceTimeBasisSelector<GeomTraits, FETraits>::type;
}
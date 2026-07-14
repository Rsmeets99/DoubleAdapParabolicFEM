#pragma once

#include "../../tables/quadrilateral_basis_tables.hpp"
#include "../../tables/triangular_prism_basis_tables.hpp"

namespace finite_element::assembly::detail
{
    template<typename GeomTraits, typename FETraits, int QSpace, int QTime>
    struct SpaceTimeBasisTablesSelector
    {
        static_assert(
            GeomTraits::dim_space_v == 1 || GeomTraits::dim_space_v == 2,
            "SpaceTimeBasisTablesSelector only supports dim_space_v = 1 or 2.");

        using type = std::conditional_t<
            (GeomTraits::dim_space_v == 1),
            finite_element::tables::QuadrilateralBasisTables<
                FETraits::p_space_v,
                FETraits::p_time_v,
                QSpace,
                QTime,
                typename FETraits::SpatialNodes,
                typename FETraits::TemporalNodes>,
            finite_element::tables::TriangularPrismBasisTables<
                FETraits::p_space_v,
                FETraits::p_time_v,
                QSpace,
                QTime,
                typename FETraits::SpatialNodes,
                typename FETraits::TemporalNodes>>;
    };

    template<typename GeomTraits, typename FETraits, int QSpace, int QTime>
    using SpaceTimeBasisTables =
        typename SpaceTimeBasisTablesSelector<GeomTraits, FETraits, QSpace, QTime>::type;
}
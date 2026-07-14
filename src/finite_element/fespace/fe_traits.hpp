#pragma once

#include "../nodes/default_node_sets.hpp"

namespace finite_element
{
    template<typename GeomTraits,
             int p_space,
             int p_time,
             typename SpatialNodes_ = nodes::DefaultSpatialNodesT<GeomTraits, p_space>,
             typename TemporalNodes_ = nodes::DefaultTemporalNodesT<p_time>>
    struct FiniteElementTraits
    {
        static constexpr int p_space_v = p_space;
        static constexpr int p_time_v  = p_time;

        using SpatialNodes  = SpatialNodes_;
        using TemporalNodes = TemporalNodes_;

        static constexpr int total_temporal_dofs = p_time + 1;
        static constexpr int total_spatial_dofs =
            (GeomTraits::dim_space_v == 1
                ? (p_space + 1)
                : static_cast<int>((p_space + 1) * (p_space + 2) / 2));

        static constexpr int dofs_per_cell = total_spatial_dofs * total_temporal_dofs;

        static constexpr int dofs_per_spatial_face =
            (GeomTraits::dim_space_v == 1
                ? total_temporal_dofs
                : (p_space + 1) * total_temporal_dofs);

        static constexpr int dofs_per_temporal_face = total_spatial_dofs;

        static constexpr int dofs_per_interior =
            (GeomTraits::dim_space_v == 1
                ? (p_space - 1) * (p_time - 1)
                : static_cast<int>((p_space - 1) * (p_space - 2) / 2) * (p_time - 1));
    };
}

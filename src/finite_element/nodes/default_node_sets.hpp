#pragma once

#include "../../mesh/mesh_traits.hpp"
#include "segment_equidistant.hpp"
#include "triangle_equidistant.hpp"
#include "triangle_warp_blend.hpp"

namespace finite_element::nodes
{
    template<typename GeomTraits, int P>
    struct DefaultSpatialNodes;

    template<int P>
    struct DefaultSpatialNodes<mesh::MeshTraits<1>, P>
    {
        using type = EquidistantNodes<P>;
    };

    template<int P>
    struct DefaultSpatialNodes<mesh::MeshTraits<2>, P>
    {
        using type = EquidistantTriangleNodes<P>;
    };

    template<int P>
    struct DefaultTemporalNodes
    {
        using type = EquidistantNodes<P>;
    };

    template<typename GeomTraits, int P>
    using DefaultSpatialNodesT = typename DefaultSpatialNodes<GeomTraits, P>::type;

    template<int P>
    using DefaultTemporalNodesT = typename DefaultTemporalNodes<P>::type;
}
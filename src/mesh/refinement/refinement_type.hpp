#pragma once

namespace mesh
{
    enum class RefinementType
    {
        none,
        spatial,
        // Auxiliary/legacy request: split only the time interval. Normal
        // physical FE refinement should pass `none` and use the
        // generation-based split policy instead.
        temporal,
        spacetime
    };
}

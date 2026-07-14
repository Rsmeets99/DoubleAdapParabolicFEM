#pragma once

#include "refinement_type.hpp"

namespace mesh::refinement
{
    template<typename GeomTraits>
    [[nodiscard]] inline RefinementType next_split_type(int generation)
    {
        constexpr int d = GeomTraits::dim_space_v;

        static_assert(d >= 1, "next_split_type requires dim_space_v >= 1.");

        if constexpr (d == 1)
        {
            // Since generation mod 1 == 0 == d-1, every split is space-time.
            return RefinementType::spacetime;
        }
        else
        {
            const int r = generation % d;
            if (r == d - 1)
                return RefinementType::spacetime;
            return RefinementType::spatial;
        }
    }
}
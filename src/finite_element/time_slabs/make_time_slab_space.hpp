#pragma once

#include "../detail/timing.hpp"
#include "time_slab_builder.hpp"

namespace finite_element::time_slabs
{
    template<typename GeomTraits, typename FETraits>
    [[nodiscard]] inline TimeSlabSpace<GeomTraits, FETraits>
    make_copied_time_slab_space(
        const FESpace<GeomTraits, FETraits, finite_element::SpaceOnlyPolicy>&
            source_space,
        double time_tol = 0.0,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        return TimeSlabBuilder<GeomTraits, FETraits>::build(
            source_space,
            time_tol,
            timing);
    }

    template<typename GeomTraits, typename FETraits>
    [[nodiscard]] inline TimeSlabSpace<GeomTraits, FETraits>
    make_time_slab_space(
        const FESpace<GeomTraits, FETraits, finite_element::SpaceOnlyPolicy>&
            source_space,
        double time_tol = 0.0)
    {
        return make_copied_time_slab_space<GeomTraits, FETraits>(
            source_space,
            time_tol);
    }
}

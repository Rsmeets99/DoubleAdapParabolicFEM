#pragma once

namespace finite_element::detail
{
    template<class GeomTraits>
    inline constexpr bool is_currently_supported_1plus1d_space_time_v =
        GeomTraits::dim_space_v == 1 && GeomTraits::dim_time_v == 1;

    template<class GeomTraits>
    inline constexpr bool is_currently_supported_2plus1d_space_time_v =
        GeomTraits::dim_space_v == 2 && GeomTraits::dim_time_v == 1;

    template<class GeomTraits>
    inline constexpr bool is_supported_time_slab_estimator_space_time_v =
        is_currently_supported_1plus1d_space_time_v<GeomTraits> ||
        is_currently_supported_2plus1d_space_time_v<GeomTraits>;

    // Centralize the current dimensionality contract in one place so the
    // eventual 2+1D lift only needs one semantic update instead of several
    // scattered ad hoc static_assert messages.
    template<class GeomTraits>
    consteval void require_current_1plus1d_space_time_capability()
    {
        static_assert(
            is_currently_supported_1plus1d_space_time_v<GeomTraits>,
            "This component currently supports only 1+1D space-time discretizations.");
    }

    template<class GeomTraits>
    consteval void require_1plus1d_time_slab_estimator_capability()
    {
        static_assert(
            is_currently_supported_1plus1d_space_time_v<GeomTraits>,
            "The time-slab estimator and equilibrated-flux reconstruction currently support only 1+1D space-time discretizations.");
    }

    template<class GeomTraits>
    consteval void require_supported_time_slab_estimator_capability()
    {
        static_assert(
            is_supported_time_slab_estimator_space_time_v<GeomTraits>,
            "The time-slab estimator and equilibrated-flux reconstruction currently support only 1+1D and 2+1D space-time discretizations.");
    }
}

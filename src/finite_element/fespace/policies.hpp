#pragma once

namespace finite_element
{
    // FE-space policies describe continuity and grading behavior of the finite-element
    // space built on top of a mesh. They do not prescribe whether the underlying mesh
    // uses spatial, temporal, or space-time splits.

    struct SpaceTimePolicy
    {
        static constexpr bool continuous_in_space = true;
        static constexpr bool continuous_in_time  = true;

        // Enforce one-level grading on the active FE partition in space and time.
        static constexpr bool enforce_spatial_grading_d1 = true;
        static constexpr bool enforce_temporal_grading   = true;
    };

    struct SpaceOnlyPolicy
    {
        static constexpr bool continuous_in_space = true;
        // "SpaceOnly" refers to the FE continuity choice: the space is discontinuous in
        // time, but grading in time can still be enforced on the active partition.
        static constexpr bool continuous_in_time  = false;

        static constexpr bool enforce_spatial_grading_d1 = true;
        static constexpr bool enforce_temporal_grading   = true;
    };
}

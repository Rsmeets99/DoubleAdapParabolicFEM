#pragma once

#include <algorithm>
#include <vector>

namespace finite_element::time_slabs::detail
{
    enum class SlabEndpointMode
    {
        containing_point,
        left_of_right_endpoint,
    };

    [[nodiscard]] inline int slab_index_from_time(
        const std::vector<double>& slab_times,
        double t,
        SlabEndpointMode mode)
    {
        if (slab_times.size() < 2)
            return -1;

        auto it = std::lower_bound(slab_times.begin(), slab_times.end(), t);
        if (it == slab_times.end())
        {
            if (t == slab_times.back())
                return static_cast<int>(slab_times.size()) - 2;
            return -1;
        }

        const int idx = static_cast<int>(std::distance(slab_times.begin(), it));

        if (*it == t)
        {
            if (mode == SlabEndpointMode::left_of_right_endpoint)
            {
                if (idx == 0)
                    return 0;
                return idx - 1;
            }

            if (idx == static_cast<int>(slab_times.size()) - 1)
                return idx - 1;
            return idx;
        }

        if (idx == 0)
            return -1;

        return idx - 1;
    }
}

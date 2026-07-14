#pragma once

#include <cstddef>

#include "../../../core/hash.hpp"

namespace finite_element::fespace::detail
{
    struct LocalDoFRef
    {
        int cell_id = -1;
        int local_index = -1;

        bool operator==(const LocalDoFRef&) const noexcept = default;
    };

    struct LocalDoFRefHash
    {
        std::size_t operator()(const LocalDoFRef& r) const noexcept
        {
            std::size_t seed = 0;
            core::hash_combine(seed, r.cell_id);
            core::hash_combine(seed, r.local_index);
            return seed;
        }
    };
}

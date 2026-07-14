#pragma once

#include <concepts>
#include <cstddef>
#include <utility>

namespace la::concepts
{
    template<class Builder>
    concept SparseBuilderLike =
        requires(Builder b, int i, int j, double v, std::size_t n)
        {
            Builder{};
            b.reserve(n);
            b.add(i, j, v);
            b.merge_from(b);
            b.merge_from(Builder{});
            b.clear();
        };
}

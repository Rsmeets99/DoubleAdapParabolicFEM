#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include "hash.hpp"

namespace core
{
    template<int dim>
    struct CoordKey
    {
        std::array<long long, dim> values{};

        bool operator==(const CoordKey&) const noexcept = default;
    };

    template<int dim>
    [[nodiscard]] inline CoordKey<dim> make_coord_key(
        const std::array<double, dim>& x,
        double scale = 1e12)
    {
        CoordKey<dim> key{};
        for (int i = 0; i < dim; ++i)
            key.values[i] = static_cast<long long>(std::llround(scale * x[i]));
        return key;
    }
}

namespace std
{
    template<int dim>
    struct hash<core::CoordKey<dim>>
    {
        std::size_t operator()(const core::CoordKey<dim>& key) const noexcept
        {
            std::size_t seed = 0;
            for (const auto v : key.values)
                core::hash_combine(seed, v);
            return seed;
        }
    };
}
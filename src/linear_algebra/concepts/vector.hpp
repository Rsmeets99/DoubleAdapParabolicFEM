#pragma once

#include <concepts>

namespace la::concepts
{
    template<class Vector>
    concept VectorLike =
        requires(Vector x, const Vector cx, int n, int i, double v)
        {
            Vector{};
            Vector(n);

            x.resize(n);
            x.set_zero();
            x.add(i, v);

            { cx.size() } -> std::convertible_to<int>;
            { x[i] } -> std::same_as<double&>;
            { cx[i] } -> std::convertible_to<double>;
        };
}
#pragma once

namespace la::ops
{
    inline constexpr double abs_val(double x) noexcept
    {
        return x < 0.0 ? -x : x;
    }

    inline constexpr double max_val(double a, double b) noexcept
    {
        return a > b ? a : b;
    }

    inline constexpr double pow_n(double x, int n) noexcept
    {
        double r = 1.0;
        for (int i = 0; i < n; ++i)
            r *= x;
        return r;
    }

    inline constexpr bool approx(double a, double b, double tol = 1.0e-10) noexcept
    {
        const double scale = max_val(1.0, max_val(abs_val(a), abs_val(b)));
        return abs_val(a - b) <= tol * scale;
    }
}
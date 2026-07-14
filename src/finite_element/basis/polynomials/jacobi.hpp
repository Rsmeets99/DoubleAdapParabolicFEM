#pragma once

#include <array>
#include <cstddef>

namespace finite_element::basis::polynomials
{
    template<int MaxN>
    struct JacobiPolynomials
    {
        static constexpr std::size_t n_values = static_cast<std::size_t>(MaxN + 1);

        using Array = std::array<double, n_values>;

        static constexpr std::size_t index(int n) noexcept
        {
            return static_cast<std::size_t>(n);
        }

        static constexpr void family(
            int nmax,
            double alpha,
            double beta,
            double x,
            Array& out)
        {
            out[0U] = 1.0;

            if (nmax == 0)
                return;

            out[1U] = 0.5 * ((alpha - beta) + (alpha + beta + 2.0) * x);

            for (int n = 2; n <= nmax; ++n)
            {
                const double nn = static_cast<double>(n);

                const double a1 = 2.0 * nn * (nn + alpha + beta) * (2.0 * nn + alpha + beta - 2.0);
                const double a2 = (2.0 * nn + alpha + beta - 1.0);
                const double a3 = (2.0 * nn + alpha + beta) * (2.0 * nn + alpha + beta - 2.0);
                const double a4 = alpha * alpha - beta * beta;
                const double a5 = 2.0 * (nn + alpha - 1.0) * (nn + beta - 1.0) * (2.0 * nn + alpha + beta);

                out[index(n)] = ((a2 * (a3 * x + a4)) * out[index(n - 1)] - a5 * out[index(n - 2)]) / a1;
            }
        }

        static constexpr void derivative_family(
            int nmax,
            double alpha,
            double beta,
            double x,
            Array& out)
        {
            out[0U] = 0.0;
            if (nmax == 0)
                return;

            Array shifted{};
            family(nmax - 1, alpha + 1.0, beta + 1.0, x, shifted);

            for (int n = 1; n <= nmax; ++n)
                out[index(n)] = 0.5 * (n + alpha + beta + 1.0) * shifted[index(n - 1)];
        }

        static constexpr double eval(int n, double alpha, double beta, double x)
        {
            Array vals{};
            family(n, alpha, beta, x, vals);
            return vals[index(n)];
        }

        static constexpr double eval_derivative(int n, double alpha, double beta, double x)
        {
            Array vals{};
            derivative_family(n, alpha, beta, x, vals);
            return vals[index(n)];
        }
    };
}

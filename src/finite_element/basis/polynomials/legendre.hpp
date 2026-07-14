#pragma once

#include <array>
#include <cstddef>
#include <cmath>

#include "linear_algebra/dense/constexpr_linalg.hpp"
#include "jacobi.hpp"

namespace finite_element::basis::polynomials
{
    template<int MaxN>
    struct LegendrePolynomials
    {
        using Array = std::array<double, MaxN + 1>;

        static constexpr void family(double x, Array& out, int nmax = MaxN)
        {
            JacobiPolynomials<MaxN>::family(nmax, 0.0, 0.0, x, out);
        }

        static constexpr void derivative_family(double x, Array& out, int nmax = MaxN)
        {
            JacobiPolynomials<MaxN>::derivative_family(nmax, 0.0, 0.0, x, out);
        }

        static constexpr double eval(int n, double x)
        {
            return JacobiPolynomials<MaxN>::eval(n, 0.0, 0.0, x);
        }

        static constexpr double eval_derivative(int n, double x)
        {
            return JacobiPolynomials<MaxN>::eval_derivative(n, 0.0, 0.0, x);
        }

        static constexpr double eval_second_derivative_from_ode(int n, double x)
        {
            // Legendre ODE:
            // (1-x^2) P'' - 2x P' + n(n+1)P = 0
            const double P  = eval(n, x);
            const double dP = eval_derivative(n, x);
            return (2.0 * x * dP - n * (n + 1.0) * P) / (1.0 - x * x);
        }

        template<int N>
        static constexpr std::array<double, N> gauss_lobatto_nodes()
        {
            static_assert(N >= 2, "Need at least 2 Gauss-Lobatto nodes");

            constexpr int Degree = N - 1;
            std::array<double, N> r{};

            r[0]        = -1.0;
            r[N - 1]    =  1.0;

            if constexpr (N == 2)
            {
                return r;
            }
            else
            {
                for (int i = 1; i < N - 1; ++i)
                {
                    double x = -std::cos(3.141592653589793238462643383279502884
                                        * static_cast<double>(i) / static_cast<double>(Degree));

                    for (int it = 0; it < 50; ++it)
                    {
                        const double dP  = eval_derivative(Degree, x);
                        const double ddP = eval_second_derivative_from_ode(Degree, x);

                        const double dx = -dP / ddP;
                        x += dx;

                        if (cdla::cabs(dx) < 1e-15)
                            break;
                    }

                    r[i] = x;
                }

                return r;
            }
        }
    };
}
#pragma once

#include <array>
#include <cstddef>
#include <limits>

#include "linear_algebra/dense/constexpr_linalg.hpp"
#include "jacobi.hpp"

namespace finite_element::basis::polynomials
{
    template<int P>
    struct DubinerBasis
    {
        static_assert(P >= 0, "DubinerBasis requires P >= 0");

        static constexpr int N = (P + 1) * (P + 2) / 2;
        static constexpr std::size_t n_modes = static_cast<std::size_t>(N);
        static constexpr std::size_t n_order_values = static_cast<std::size_t>(P + 1);

        using Values = cdla::Vec<N>;
        using Grads  = std::array<std::array<double, 2>, n_modes>;

        static constexpr std::size_t array_index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        static constexpr int index(int p, int q)
        {
            return p * (P + 1) - (p * (p - 1)) / 2 + q;
        }

        static constexpr Values eval_all(double x, double y)
        {
            Values out{};

            const double omy = 1.0 - y;
            const double eps = 64.0 * std::numeric_limits<double>::epsilon();

            const double a = (cdla::cabs(omy) <= eps) ? -1.0 : (2.0 * x / omy - 1.0);
            const double b = 2.0 * y - 1.0;

            typename JacobiPolynomials<P>::Array Pa{};
            JacobiPolynomials<P>::family(P, 0.0, 0.0, a, Pa);

            std::array<double, n_order_values> omy_pow{};
            omy_pow[0] = 1.0;
            for (int p = 1; p <= P; ++p)
                omy_pow[array_index(p)] = omy_pow[array_index(p - 1)] * omy;

            int id = 0;
            for (int p = 0; p <= P; ++p)
            {
                typename JacobiPolynomials<P>::Array Q{};
                JacobiPolynomials<P>::family(P - p, 2.0 * p + 1.0, 0.0, b, Q);

                const double pref = Pa[array_index(p)] * omy_pow[array_index(p)];
                for (int q = 0; q <= P - p; ++q)
                    out[array_index(id++)] = pref * Q[array_index(q)];
            }

            return out;
        }

        static constexpr Grads grad_all(double x, double y)
        {
            Grads out{};

            const double eps = 64.0 * std::numeric_limits<double>::epsilon();
            const double yy  = (cdla::cabs(1.0 - y) <= eps) ? (1.0 - 1e-14) : y;
            const double omy = 1.0 - yy;

            const double a = 2.0 * x / omy - 1.0;
            const double b = 2.0 * yy - 1.0;

            const double da_dx = 2.0 / omy;
            const double da_dy = 2.0 * x / (omy * omy);
            const double db_dy = 2.0;

            typename JacobiPolynomials<P>::Array Pa{}, dPa{};
            JacobiPolynomials<P>::family(P, 0.0, 0.0, a, Pa);
            JacobiPolynomials<P>::derivative_family(P, 0.0, 0.0, a, dPa);

            std::array<double, n_order_values> omy_pow{};
            omy_pow[0] = 1.0;
            for (int p = 1; p <= P; ++p)
                omy_pow[array_index(p)] = omy_pow[array_index(p - 1)] * omy;

            int id = 0;
            for (int p = 0; p <= P; ++p)
            {
                typename JacobiPolynomials<P>::Array Q{}, dQ{};
                JacobiPolynomials<P>::family(P - p, 2.0 * p + 1.0, 0.0, b, Q);
                JacobiPolynomials<P>::derivative_family(P - p, 2.0 * p + 1.0, 0.0, b, dQ);

                const double fac    = omy_pow[array_index(p)];
                const double fac_dy = (p == 0) ? 0.0 : -static_cast<double>(p) * omy_pow[array_index(p - 1)];

                for (int q = 0; q <= P - p; ++q)
                {
                    const std::size_t id_index = array_index(id);
                    const std::size_t p_index = array_index(p);
                    const std::size_t q_index = array_index(q);

                    out[id_index][0U] =
                        dPa[p_index] * da_dx * fac * Q[q_index];

                    out[id_index][1U] =
                        dPa[p_index] * da_dy * fac * Q[q_index]
                        + Pa[p_index] * fac_dy * Q[q_index]
                        + Pa[p_index] * fac * dQ[q_index] * db_dy;

                    ++id;
                }
            }

            return out;
        }
    };
}

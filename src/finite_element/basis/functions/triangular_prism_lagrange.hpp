#pragma once

#include <array>
#include <cstddef>

#include "../polynomials/triangular_lagrange.hpp"
#include "../polynomials/segment_lagrange.hpp"

namespace finite_element::basis::functions
{
    template<int P, int Q, typename SpatialNodes, typename TemporalNodes>
    struct TriangularPrismLagrangeBasis
    {
        static_assert(P >= 0, "TriangularPrismLagrangeBasis requires P >= 0.");
        static_assert(Q >= 0, "TriangularPrismLagrangeBasis requires Q >= 0.");

        static constexpr int N_tri  = basis::polynomials::TriangularLagrangeBasis<P, SpatialNodes>::N;
        static constexpr int N_time = Q + 1;
        static constexpr int N      = N_tri * N_time;
        static constexpr std::size_t n_values = static_cast<std::size_t>(N);

        using SpaceTimePoint = std::array<double, 3>;
        using SpatialPoint   = std::array<double, 2>;
        using Values         = std::array<double, n_values>;
        using Grads          = std::array<std::array<double, 3>, n_values>;

        static constexpr std::size_t array_index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        static constexpr int index(int i_space, int i_time)
        {
            return i_space * N_time + i_time;
        }

        static constexpr Values eval_all(double x, double y, double t)
        {
            const SpatialPoint sp{x, y};

            const auto L_tri  = basis::polynomials::TriangularLagrangeBasis<P, SpatialNodes>::eval_all(sp);
            const auto L_time = basis::polynomials::SegmentLagrangeBasis<Q, TemporalNodes>::eval_all(t);

            Values vals{};
            for (int i = 0; i < N_tri; ++i)
            {
                for (int j = 0; j < N_time; ++j)
                {
                    vals[array_index(index(i, j))] =
                        L_tri[array_index(i)] * L_time[array_index(j)];
                }
            }

            return vals;
        }

        static constexpr Values eval_all(const SpaceTimePoint& pt)
        {
            return eval_all(pt[0U], pt[1U], pt[2U]);
        }

        static constexpr Grads grad_all(double x, double y, double t)
        {
            const SpatialPoint sp{x, y};

            const auto G_tri   = basis::polynomials::TriangularLagrangeBasis<P, SpatialNodes>::grad_all(sp);
            const auto L_tri   = basis::polynomials::TriangularLagrangeBasis<P, SpatialNodes>::eval_all(sp);
            const auto L_time  = basis::polynomials::SegmentLagrangeBasis<Q, TemporalNodes>::eval_all(t);
            const auto dL_time = basis::polynomials::SegmentLagrangeBasis<Q, TemporalNodes>::grad_all(t);

            Grads grads{};
            for (int i = 0; i < N_tri; ++i)
            {
                const double Li_tri = L_tri[array_index(i)];
                const double Gx     = G_tri[array_index(i)][0U];
                const double Gy     = G_tri[array_index(i)][1U];

                for (int j = 0; j < N_time; ++j)
                {
                    const int k = index(i, j);
                    grads[array_index(k)][0U] = Gx * L_time[array_index(j)];
                    grads[array_index(k)][1U] = Gy * L_time[array_index(j)];
                    grads[array_index(k)][2U] = Li_tri * dL_time[array_index(j)];
                }
            }

            return grads;
        }

        static constexpr Grads grad_all(const SpaceTimePoint& pt)
        {
            return grad_all(pt[0U], pt[1U], pt[2U]);
        }
    };
}

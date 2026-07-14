#pragma once
#include <array>
#include <cstddef>
#include "../polynomials/segment_lagrange.hpp"

// Quadrilateral 1+1D Lagrange basis using 1D SegmentLagrangeBasis in each direction.
namespace finite_element::basis::functions
{
    template<int P, int Q, typename SpatialNodes, typename TemporalNodes>
    struct QuadrilateralLagrangeBasis
    {
        static_assert(P >= 1, "QuadrilateralLagrangeBasis requires P >= 1.");
        static_assert(Q >= 1, "QuadrilateralLagrangeBasis requires Q >= 1.");
        
        static constexpr int N_space = P + 1;
        static constexpr int N_time = Q + 1;
        static constexpr int N = N_space * N_time;
        static constexpr std::size_t n_space = static_cast<std::size_t>(N_space);
        static constexpr std::size_t n_time = static_cast<std::size_t>(N_time);
        static constexpr std::size_t n_values = static_cast<std::size_t>(N);
        using SpaceTimePoint = std::array<double, 2>;

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        //--------------------------------------------------------
        // Evaluate all basis functions at (x,t)
        //--------------------------------------------------------
        static constexpr std::array<double, n_values> eval_all(const SpaceTimePoint& pt)
        {
            auto L_space = basis::polynomials::SegmentLagrangeBasis<P, SpatialNodes>::eval_all(pt[0U]);
            auto L_time  = basis::polynomials::SegmentLagrangeBasis<Q, TemporalNodes>::eval_all(pt[1U]);

            std::array<double, n_values> vals{};
            std::size_t idx = 0;
            for (std::size_t i = 0; i < n_space; ++i)
                for (std::size_t j = 0; j < n_time; ++j)
                    vals[idx++] = L_space[i] * L_time[j];

            return vals;
        }

        //--------------------------------------------------------
        // Evaluate gradient (dx, dt) of all basis functions
        //--------------------------------------------------------
        static constexpr std::array<std::array<double, 2>, n_values> grad_all(const SpaceTimePoint& pt)
        {
            auto L_space  = basis::polynomials::SegmentLagrangeBasis<P, SpatialNodes>::eval_all(pt[0U]);
            auto dL_space = basis::polynomials::SegmentLagrangeBasis<P, SpatialNodes>::grad_all(pt[0U]);

            auto L_time   = basis::polynomials::SegmentLagrangeBasis<Q, TemporalNodes>::eval_all(pt[1U]);
            auto dL_time  = basis::polynomials::SegmentLagrangeBasis<Q, TemporalNodes>::grad_all(pt[1U]);

            std::array<std::array<double, 2>, n_values> grads{};
            std::size_t idx = 0;
            for (std::size_t i = 0; i < n_space; ++i)
            {
                double Li  = L_space[i];
                double dLi = dL_space[i];

                for (std::size_t j = 0; j < n_time; ++j)
                {
                    grads[idx][0U] = dLi * L_time[j];
                    grads[idx][1U] = Li  * dL_time[j];
                    ++idx;
                }
            }

            return grads;
        }
    };
}

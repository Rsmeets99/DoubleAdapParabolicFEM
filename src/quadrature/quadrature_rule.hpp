#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace quadrature::rule
{
    template<int Dim, int N>
    struct QuadratureRule
    {
        static_assert(Dim >= 1, "QuadratureRule requires Dim >= 1.");
        static_assert(N >= 1,   "QuadratureRule requires N >= 1.");

        static constexpr int dim      = Dim;
        static constexpr int n_points = N;
        static constexpr std::size_t dim_size = static_cast<std::size_t>(Dim);
        static constexpr std::size_t n_points_size = static_cast<std::size_t>(N);

        std::array<std::array<double, dim_size>, n_points_size> points{};
        std::array<double, n_points_size>                        weights{};

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        constexpr int size() const noexcept
        {
            return N;
        }

        constexpr const std::array<double, dim_size>& point(int i) const noexcept
        {
            return points[index(i)];
        }

        constexpr const double& weight(int i) const noexcept
        {
            return weights[index(i)];
        }
    };

    template<int Dim, int N, class Function>
    constexpr double integrate(const QuadratureRule<Dim, N>& rule, Function&& f)
    {
        double value = 0.0;

        for (std::size_t i = 0; i < QuadratureRule<Dim, N>::n_points_size; ++i)
            value += rule.weights[i] * std::forward<Function>(f)(rule.points[i]);

        return value;
    }
}

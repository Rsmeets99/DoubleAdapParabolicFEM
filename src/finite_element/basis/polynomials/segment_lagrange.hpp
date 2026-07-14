#pragma once

#include <array>
#include <cstddef>
#include <limits>

namespace finite_element::basis::polynomials
{
    template<int Degree, typename Nodes>
    struct SegmentLagrangeBasis
    {
        static_assert(Degree >= 1, "SegmentLagrangeBasis requires Degree >= 1");
        static_assert(Nodes::N == Degree + 1,
                    "SegmentLagrangeBasis: Nodes::N must equal Degree + 1.");

        static constexpr int N = Degree + 1;
        static constexpr std::size_t n_values = static_cast<std::size_t>(N);

        using Point  = typename Nodes::Point;
        using Values = std::array<double, n_values>;
        using Grads  = std::array<double, n_values>;

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        static constexpr double absval(double x)
        {
            return x < 0.0 ? -x : x;
        }

        static constexpr double maxval(double a, double b)
        {
            return a > b ? a : b;
        }

        static constexpr std::array<double, n_values> extract_abscissae()
        {
            std::array<double, n_values> x{};
            for (std::size_t i = 0; i < n_values; ++i)
                x[i] = Nodes::points[i][0U];
            return x;
        }

        // denom[i] = prod_{j!=i} (x_i - x_j)
        static constexpr std::array<double, n_values> generate_denominators(
            const std::array<double, n_values>& node_abscissae)
        {
            std::array<double, n_values> d{};
            for (std::size_t i = 0; i < n_values; ++i)
            {
                double prod = 1.0;
                for (std::size_t j = 0; j < n_values; ++j)
                {
                    if (i != j)
                        prod *= (node_abscissae[i] - node_abscissae[j]);
                }
                d[i] = prod;
            }
            return d;
        }

        // barycentric weights w_i = 1 / denom_i
        static constexpr std::array<double, n_values> generate_weights(
            const std::array<double, n_values>& denominators)
        {
            std::array<double, n_values> w{};
            for (std::size_t i = 0; i < n_values; ++i)
                w[i] = 1.0 / denominators[i];
            return w;
        }

        static constexpr std::array<double, n_values> nodes = extract_abscissae();
        static constexpr std::array<double, n_values> denom  = generate_denominators(nodes);
        static constexpr std::array<double, n_values> weight = generate_weights(denom);

        struct EvalData
        {
            std::array<double, n_values> inv_dx{}; // 1 / (xi - x_i)
            std::array<double, n_values> t{};      // w_i / (xi - x_i)
            double s1 = 0.0;                // sum_i t_i
            double s2 = 0.0;                // sum_i t_i / (xi - x_i)
            int hit = -1;                   // node index if xi is (numerically) at a node
        };

        static constexpr EvalData prepare_eval(double xi)
        {
            EvalData data{};

            // Scale-aware tolerance for nodal detection
            const double eps = 32.0 * std::numeric_limits<double>::epsilon();

            for (std::size_t i = 0; i < n_values; ++i)
            {
                const double dx = xi - nodes[i];
                const double scale = maxval(1.0, absval(xi) + absval(nodes[i]));
                if (absval(dx) <= eps * scale)
                {
                    data.hit = static_cast<int>(i);
                    return data;
                }
            }

            for (std::size_t i = 0; i < n_values; ++i)
            {
                const double dx  = xi - nodes[i];
                const double inv = 1.0 / dx;
                const double ti  = weight[i] * inv;

                data.inv_dx[i] = inv;
                data.t[i]      = ti;
                data.s1       += ti;
                data.s2       += ti * inv;
            }

            return data;
        }

        static constexpr Values eval_all(double xi)
        {
            const EvalData data = prepare_eval(xi);
            Values L{};

            if (data.hit >= 0)
            {
                L[index(data.hit)] = 1.0;
                return L;
            }

            const double inv_s1 = 1.0 / data.s1;
            for (std::size_t i = 0; i < n_values; ++i)
                L[i] = data.t[i] * inv_s1;

            return L;
        }

        static constexpr Grads grad_all(double xi)
        {
            const EvalData data = prepare_eval(xi);
            Grads dL{};

            // Exact/near-node evaluation: use nodal differentiation formula
            if (data.hit >= 0)
            {
                const std::size_t k = index(data.hit);

                double row_sum = 0.0;
                for (std::size_t i = 0; i < n_values; ++i)
                {
                    if (i == k)
                        continue;

                    // l_i'(x_k) = w_i / ( w_k * (x_k - x_i) )
                    dL[i] = weight[i] / (weight[k] * (nodes[k] - nodes[i]));
                    row_sum += dL[i];
                }

                // Enforce sum_i l_i'(x_k) = 0
                dL[k] = -row_sum;
                return dL;
            }

            // Non-nodal evaluation:
            // l_i' = l_i * (s2/s1 - 1/(xi - x_i))
            const double c = data.s2 / data.s1;
            const double inv_s1 = 1.0 / data.s1;

            for (std::size_t i = 0; i < n_values; ++i)
            {
                const double Li = data.t[i] * inv_s1;
                dL[i] = Li * (c - data.inv_dx[i]);
            }

            return dL;
        }

        static constexpr double eval(int i, double xi)
        {
            return eval_all(xi)[index(i)];
        }

        static constexpr double grad(int i, double xi)
        {
            return grad_all(xi)[index(i)];
        }
    };
}

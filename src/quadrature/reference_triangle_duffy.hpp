#pragma once

#include "gauss_legendre_1d.hpp"
#include "quadrature_rule.hpp"

namespace quadrature::reference
{
    namespace detail
    {
        template<int Degree>
        struct DuffyTriangleRuleTraits
        {
            static_assert(Degree >= 0,
                          "DuffyTriangleRuleTraits requires Degree >= 0.");

            // Duffy map: (u,v) -> (u, (1-u)v), jacobian = 1-u.
            // A total-degree Degree triangle polynomial becomes degree Degree+1
            // in u after the jacobian and degree Degree in v.
            static constexpr int order = (Degree + 3) / 2;
            static constexpr int n_points = order * order;
        };
    }

    template<int Degree>
    constexpr auto duffy_triangle()
    {
        static_assert(Degree >= 0,
                      "duffy_triangle requires Degree >= 0.");

        constexpr int order = detail::DuffyTriangleRuleTraits<Degree>::order;
        static_assert(order >= 1 && order <= 12,
                      "duffy_triangle currently supports exactness Degree <= 22.");

        constexpr auto interval_rule =
            quadrature::gauss_legendre::gauss_legendre_rule_1d<order>;

        quadrature::rule::QuadratureRule<
            2,
            detail::DuffyTriangleRuleTraits<Degree>::n_points> rule{};

        int k = 0;
        for (int i = 0; i < interval_rule.size(); ++i)
        {
            const double u = interval_rule.point(i)[0];
            const double one_minus_u = 1.0 - u;

            for (int j = 0; j < interval_rule.size(); ++j)
            {
                const double v = interval_rule.point(j)[0];
                rule.points[static_cast<std::size_t>(k)] = {
                    u,
                    one_minus_u * v
                };
                rule.weights[static_cast<std::size_t>(k)] =
                    interval_rule.weight(i) *
                    interval_rule.weight(j) *
                    one_minus_u;
                ++k;
            }
        }

        return rule;
    }

    template<int Degree>
    inline constexpr auto duffy_triangle_rule = duffy_triangle<Degree>();

    template<int Degree, class Function>
    void for_each_reference_triangle_duffy_point(Function&& function)
    {
        static_assert(Degree >= 0,
                      "for_each_reference_triangle_duffy_point requires Degree >= 0.");

        constexpr int order = detail::DuffyTriangleRuleTraits<Degree>::order;
        static_assert(order >= 1 && order <= 12,
                      "integrate_reference_triangle_duffy currently supports exactness Degree <= 22.");

        constexpr auto rule =
            quadrature::gauss_legendre::gauss_legendre_rule_1d<order>;

        for (int i = 0; i < rule.size(); ++i)
        {
            const double u = rule.point(i)[0];
            const double one_minus_u = 1.0 - u;

            for (int j = 0; j < rule.size(); ++j)
            {
                const double v = rule.point(j)[0];
                function(
                    u,
                    one_minus_u * v,
                    rule.weight(i) * rule.weight(j) * one_minus_u);
            }
        }
    }

    template<int Degree, class Integrand>
    [[nodiscard]] double integrate_reference_triangle_duffy(
        Integrand&& integrand)
    {
        double result = 0.0;
        for_each_reference_triangle_duffy_point<Degree>(
            [&](const double x, const double y, const double weight)
            {
                result += weight * integrand(x, y);
            });

        return result;
    }
}

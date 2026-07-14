#pragma once

#include <array>
#include <cstddef>

#include "quadrature_rule.hpp"

namespace quadrature::gauss_legendre
{
    template<int N>
    using GaussLegendreRawRule1D = std::array<std::array<double, 2>, static_cast<std::size_t>(N)>;

    inline constexpr GaussLegendreRawRule1D<1> gauss_legendre_raw_1 = {{
        {{ 0.0, 2.0 }}
    }};

    inline constexpr GaussLegendreRawRule1D<2> gauss_legendre_raw_2 = {{
        {{ -0.5773502691896257, 1.0 }},
        {{  0.5773502691896257, 1.0 }}
    }};

    inline constexpr GaussLegendreRawRule1D<3> gauss_legendre_raw_3 = {{
        {{ -0.7745966692414834, 0.5555555555555556 }},
        {{  0.0,                0.8888888888888888 }},
        {{  0.7745966692414834, 0.5555555555555556 }}
    }};

    inline constexpr GaussLegendreRawRule1D<4> gauss_legendre_raw_4 = {{
        {{ -0.8611363115940526, 0.3478548451374538 }},
        {{ -0.3399810435848563, 0.6521451548625461 }},
        {{  0.3399810435848563, 0.6521451548625461 }},
        {{  0.8611363115940526, 0.3478548451374538 }}
    }};

    inline constexpr GaussLegendreRawRule1D<5> gauss_legendre_raw_5 = {{
        {{ -0.9061798459386640, 0.2369268850561891 }},
        {{ -0.5384693101056831, 0.4786286704993665 }},
        {{  0.0,                0.5688888888888889 }},
        {{  0.5384693101056831, 0.4786286704993665 }},
        {{  0.9061798459386640, 0.2369268850561891 }}
    }};

    inline constexpr GaussLegendreRawRule1D<6> gauss_legendre_raw_6 = {{
        {{ -0.9324695142031521, 0.1713244923791704 }},
        {{ -0.6612093864662645, 0.3607615730481386 }},
        {{ -0.2386191860831969, 0.4679139345726910 }},
        {{  0.2386191860831969, 0.4679139345726910 }},
        {{  0.6612093864662645, 0.3607615730481386 }},
        {{  0.9324695142031521, 0.1713244923791704 }}
    }};

    inline constexpr GaussLegendreRawRule1D<7> gauss_legendre_raw_7 = {{
        {{ -0.9491079123427585, 0.1294849661688697 }},
        {{ -0.7415311855993945, 0.2797053914892766 }},
        {{ -0.4058451513773972, 0.3818300505051189 }},
        {{  0.0,                0.4179591836734694 }},
        {{  0.4058451513773972, 0.3818300505051189 }},
        {{  0.7415311855993945, 0.2797053914892766 }},
        {{  0.9491079123427585, 0.1294849661688697 }}
    }};

    inline constexpr GaussLegendreRawRule1D<8> gauss_legendre_raw_8 = {{
        {{ -0.9602898564975363, 0.1012285362903763 }},
        {{ -0.7966664774136267, 0.2223810344533745 }},
        {{ -0.5255324099163290, 0.3137066458778873 }},
        {{ -0.1834346424956498, 0.3626837833783620 }},
        {{  0.1834346424956498, 0.3626837833783620 }},
        {{  0.5255324099163290, 0.3137066458778873 }},
        {{  0.7966664774136267, 0.2223810344533745 }},
        {{  0.9602898564975363, 0.1012285362903763 }}
    }};

    inline constexpr GaussLegendreRawRule1D<9> gauss_legendre_raw_9 = {{
        {{ -0.9681602395076261, 0.0812743883615744 }},
        {{ -0.8360311073266358, 0.1806481606948574 }},
        {{ -0.6133714327005904, 0.2606106964029354 }},
        {{ -0.3242534234038089, 0.3123470770400029 }},
        {{  0.0,                0.3302393550012598 }},
        {{  0.3242534234038089, 0.3123470770400029 }},
        {{  0.6133714327005904, 0.2606106964029354 }},
        {{  0.8360311073266358, 0.1806481606948574 }},
        {{  0.9681602395076261, 0.0812743883615744 }}
    }};

    inline constexpr GaussLegendreRawRule1D<10> gauss_legendre_raw_10 = {{
        {{ -0.9739065285171717, 0.0666713443086881 }},
        {{ -0.8650633666889845, 0.1494513491505806 }},
        {{ -0.6794095682990244, 0.2190863625159820 }},
        {{ -0.4333953941292472, 0.2692667193099963 }},
        {{ -0.1488743389816312, 0.2955242247147529 }},
        {{  0.1488743389816312, 0.2955242247147529 }},
        {{  0.4333953941292472, 0.2692667193099963 }},
        {{  0.6794095682990244, 0.2190863625159820 }},
        {{  0.8650633666889845, 0.1494513491505806 }},
        {{  0.9739065285171717, 0.0666713443086881 }}
    }};

    inline constexpr GaussLegendreRawRule1D<11> gauss_legendre_raw_11 = {{
        {{ -0.9782286581460570, 0.0556685671161737 }},
        {{ -0.8870625997680953, 0.1255803694649046 }},
        {{ -0.7301520055740494, 0.1862902109277343 }},
        {{ -0.5190961292068118, 0.2331937645919905 }},
        {{ -0.2695431559523450, 0.2628045445102467 }},
        {{  0.0,                0.2729250867779006 }},
        {{  0.2695431559523450, 0.2628045445102467 }},
        {{  0.5190961292068118, 0.2331937645919905 }},
        {{  0.7301520055740494, 0.1862902109277343 }},
        {{  0.8870625997680953, 0.1255803694649046 }},
        {{  0.9782286581460570, 0.0556685671161737 }}
    }};

    inline constexpr GaussLegendreRawRule1D<12> gauss_legendre_raw_12 = {{
        {{ -0.9815606342467192, 0.0471753363865118 }},
        {{ -0.9041172563704749, 0.1069393259953184 }},
        {{ -0.7699026741943047, 0.1600783285433462 }},
        {{ -0.5873179542866175, 0.2031674267230659 }},
        {{ -0.3678314989981802, 0.2334925365383548 }},
        {{ -0.1252334085114689, 0.2491470458134028 }},
        {{  0.1252334085114689, 0.2491470458134028 }},
        {{  0.3678314989981802, 0.2334925365383548 }},
        {{  0.5873179542866175, 0.2031674267230659 }},
        {{  0.7699026741943047, 0.1600783285433462 }},
        {{  0.9041172563704749, 0.1069393259953184 }},
        {{  0.9815606342467192, 0.0471753363865118 }}
    }};

    template<int Order>
    constexpr const auto& gauss_legendre_raw_rule_1d()
    {
        static_assert(Order >= 1 && Order <= 12,
                    "gauss_legendre_raw_rule_1d supports orders 1 through 12.");

        if constexpr (Order == 1) return gauss_legendre_raw_1;
        else if constexpr (Order == 2) return gauss_legendre_raw_2;
        else if constexpr (Order == 3) return gauss_legendre_raw_3;
        else if constexpr (Order == 4) return gauss_legendre_raw_4;
        else if constexpr (Order == 5) return gauss_legendre_raw_5;
        else if constexpr (Order == 6) return gauss_legendre_raw_6;
        else if constexpr (Order == 7) return gauss_legendre_raw_7;
        else if constexpr (Order == 8) return gauss_legendre_raw_8;
        else if constexpr (Order == 9) return gauss_legendre_raw_9;
        else if constexpr (Order == 10) return gauss_legendre_raw_10;
        else if constexpr (Order == 11) return gauss_legendre_raw_11;
        else                            return gauss_legendre_raw_12;
    }

    template<int Order>
    constexpr quadrature::rule::QuadratureRule<1, Order> gauss_legendre_1d()
    {
        static_assert(Order >= 1 && Order <= 12,
                    "gauss_legendre_1d supports orders 1 through 12.");

        constexpr auto& raw = gauss_legendre_raw_rule_1d<Order>();

        quadrature::rule::QuadratureRule<1, Order> rule{};

        for (std::size_t i = 0; i < static_cast<std::size_t>(Order); ++i)
        {
            const double xi = raw[i][0U];
            const double w  = raw[i][1U];

            // map [-1,1] -> [0,1]
            rule.points[i][0U] = 0.5 * (xi + 1.0);
            rule.weights[i]   = 0.5 * w;
        }

        return rule;
    }

    template<int Order>
    inline constexpr auto gauss_legendre_rule_1d = gauss_legendre_1d<Order>();

}

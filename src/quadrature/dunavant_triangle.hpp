#pragma once

#include <cstddef>

#include "quadrature_rule.hpp"

namespace quadrature::dunavant
{

    template<int Degree>
    struct DunavantTraits;

    template<> struct DunavantTraits<1>  { static constexpr int n_points = 1;  };
    template<> struct DunavantTraits<2>  { static constexpr int n_points = 3;  };
    template<> struct DunavantTraits<3>  { static constexpr int n_points = 4;  };
    template<> struct DunavantTraits<4>  { static constexpr int n_points = 6;  };
    template<> struct DunavantTraits<5>  { static constexpr int n_points = 7;  };
    template<> struct DunavantTraits<6>  { static constexpr int n_points = 12; };
    template<> struct DunavantTraits<7>  { static constexpr int n_points = 13; };
    template<> struct DunavantTraits<8>  { static constexpr int n_points = 16; };
    template<> struct DunavantTraits<9>  { static constexpr int n_points = 19; };
    template<> struct DunavantTraits<10> { static constexpr int n_points = 25; };

    template<int N>
    constexpr void add_centroid(quadrature::rule::QuadratureRule<2, N>& rule,
                                std::size_t&          k,
                                double                weight) noexcept
    {
        rule.points[k]  = { 1.0 / 3.0, 1.0 / 3.0 };
        rule.weights[k] = weight;
        ++k;
    }

    template<int N>
    constexpr void add_orbit_aab(quadrature::rule::QuadratureRule<2, N>& rule,
                                std::size_t&          k,
                                double                a,
                                double                b,
                                double                weight) noexcept
    {
        rule.points[k]  = { a, b };
        rule.weights[k] = weight;
        ++k;

        rule.points[k]  = { b, a };
        rule.weights[k] = weight;
        ++k;

        rule.points[k]  = { a, a };
        rule.weights[k] = weight;
        ++k;
    }

    template<int N>
    constexpr void add_orbit_abc(quadrature::rule::QuadratureRule<2, N>& rule,
                                std::size_t&          k,
                                double                a,
                                double                b,
                                double                c,
                                double                weight) noexcept
    {
        rule.points[k]  = { b, c };
        rule.weights[k] = weight;
        ++k;

        rule.points[k]  = { c, b };
        rule.weights[k] = weight;
        ++k;

        rule.points[k]  = { a, c };
        rule.weights[k] = weight;
        ++k;

        rule.points[k]  = { c, a };
        rule.weights[k] = weight;
        ++k;

        rule.points[k]  = { a, b };
        rule.weights[k] = weight;
        ++k;

        rule.points[k]  = { b, a };
        rule.weights[k] = weight;
        ++k;
    }

    template<int Degree>
    constexpr auto dunavant_triangle()
    {
        static_assert(Degree >= 1 && Degree <= 10,
                    "dunavant_triangle supports exactness degrees 1 through 10.");

        constexpr int N = DunavantTraits<Degree>::n_points;
        quadrature::rule::QuadratureRule<2, N> rule{};

        std::size_t k = 0;

        if constexpr (Degree == 1)
        {
            add_centroid(rule, k, 0.5);
        }
        else if constexpr (Degree == 2)
        {
            add_orbit_aab(rule, k,
                                1.0 / 6.0,
                                2.0 / 3.0,
                                1.0 / 6.0);
        }
        else if constexpr (Degree == 3)
        {
            add_centroid(rule, k, -9.0 / 32.0);

            add_orbit_aab(rule, k,
                                0.2,
                                0.6,
                                25.0 / 96.0);
        }
        else if constexpr (Degree == 4)
        {
            add_orbit_aab(rule, k,
                                0.445948490915965,
                                0.108103018168070,
                                0.1116907948390055);

            add_orbit_aab(rule, k,
                                0.091576213509771,
                                0.816847572980459,
                                0.0549758718276610);
        }
        else if constexpr (Degree == 5)
        {
            add_centroid(rule, k, 0.1125000000000000);

            add_orbit_aab(rule, k,
                                0.470142064105115,
                                0.059715871789770,
                                0.0661970763942530);

            add_orbit_aab(rule, k,
                                0.101286507323456,
                                0.797426985353087,
                                0.0629695902724135);
        }
        else if constexpr (Degree == 6)
        {
            add_orbit_aab(rule, k,
                                0.249286745170910,
                                0.501426509658179,
                                0.0583931378631895);

            add_orbit_aab(rule, k,
                                0.063089014491502,
                                0.873821971016996,
                                0.0254224531851035);

            add_orbit_abc(rule, k,
                                0.310352451033785,
                                0.636502499121399,
                                0.053145049844816,
                                0.0414255378091870);
        }
        else if constexpr (Degree == 7)
        {
            add_centroid(rule, k, -0.0747850222338410);

            add_orbit_aab(rule, k,
                                0.260345966079040,
                                0.479308067841920,
                                0.0878076287166040);

            add_orbit_aab(rule, k,
                                0.065130102902216,
                                0.869739794195568,
                                0.0266736178044190);

            add_orbit_abc(rule, k,
                                0.312865496004874,
                                0.638444188569809,
                                0.048690315425316,
                                0.0385568804451280);
        }
        else if constexpr (Degree == 8)
        {
            add_centroid(rule, k, 0.0721578038388940);

            add_orbit_aab(rule, k,
                                0.459292588292723,
                                0.081414823414554,
                                0.0475458171336420);

            add_orbit_aab(rule, k,
                                0.170569307751760,
                                0.658861384496480,
                                0.0516086852673590);

            add_orbit_aab(rule, k,
                                0.050547228317031,
                                0.898905543365938,
                                0.0162292488115990);

            add_orbit_abc(rule, k,
                                0.263112829634638,
                                0.728492392955404,
                                0.008394777409958,
                                0.0136151570872170);
        }
        else if constexpr (Degree == 9)
        {
            add_centroid(rule, k, 0.0485678981414000);

            add_orbit_aab(rule, k,
                                0.489682519198738,
                                0.020634961602525,
                                0.0156673501135700);

            add_orbit_aab(rule, k,
                                0.437089591492937,
                                0.125820817014127,
                                0.0389137705023870);

            add_orbit_aab(rule, k,
                                0.188203535619033,
                                0.623592928761935,
                                0.0398238694636050);

            add_orbit_aab(rule, k,
                                0.044729513394453,
                                0.910540973211095,
                                0.0127888378293490);

            add_orbit_abc(rule, k,
                                0.221962989160766,
                                0.741198598784498,
                                0.036838412054736,
                                0.0216417696886450);
        }
        else if constexpr (Degree == 10)
        {
            add_centroid(rule, k, 0.0454089951913770);

            add_orbit_aab(rule, k,
                                0.485577633383657,
                                0.028844733232685,
                                0.0183629788782330);

            add_orbit_aab(rule, k,
                                0.109481575485037,
                                0.781036849029926,
                                0.0226605297177640);

            add_orbit_abc(rule, k,
                                0.141707219414880,
                                0.307939838764121,
                                0.550352941820999,
                                0.0363789584227100);

            add_orbit_abc(rule, k,
                                0.025003534762686,
                                0.246672560639903,
                                0.728323904597411,
                                0.0141636212655280);

            add_orbit_abc(rule, k,
                                0.009540815400299,
                                0.066803251012200,
                                0.923655933587500,
                                0.0047108334818670);
        }

        return rule;
    }

    template<int Degree>
    inline constexpr auto dunavant_triangle_rule = dunavant_triangle<Degree>();

}

#pragma once

#include "quadrature_rule.hpp"

// Generic space x time tensor product.
// time is always 1D and appended as the last coordinate.
namespace quadrature::tensor_product
{
    template<int DimSpace, int NSpace, int NTime>
    constexpr quadrature::rule::QuadratureRule<DimSpace + 1, NSpace * NTime>
    make_space_time_tensor_product_rule(const quadrature::rule::QuadratureRule<DimSpace, NSpace>& space_rule,
                                        const quadrature::rule::QuadratureRule<1, NTime>&         time_rule) noexcept
    {
        quadrature::rule::QuadratureRule<DimSpace + 1, NSpace * NTime> out{};

        constexpr std::size_t n_space = static_cast<std::size_t>(NSpace);
        constexpr std::size_t n_time = static_cast<std::size_t>(NTime);
        constexpr std::size_t dim_space = static_cast<std::size_t>(DimSpace);

        for (std::size_t is = 0; is < n_space; ++is)
        {
            for (std::size_t it = 0; it < n_time; ++it)
            {
                const std::size_t k = is * n_time + it;

                for (std::size_t d = 0; d < dim_space; ++d)
                    out.points[k][d] = space_rule.points[is][d];

                out.points[k][dim_space] = time_rule.points[it][0U];
                out.weights[k]          = space_rule.weights[is] * time_rule.weights[it];
            }
        }

        return out;
    }

    // 1+1D: segment x time interval
    template<int NX, int NT>
    constexpr quadrature::rule::QuadratureRule<2, NX * NT>
    make_quadrilateral_space_time_rule(const quadrature::rule::QuadratureRule<1, NX>& space_rule,
                                    const quadrature::rule::QuadratureRule<1, NT>& time_rule) noexcept
    {
        return make_space_time_tensor_product_rule(space_rule, time_rule);
    }

    // 2+1D: triangle x time interval
    template<int NXY, int NT>
    constexpr quadrature::rule::QuadratureRule<3, NXY * NT>
    make_triangular_prism_space_time_rule(const quadrature::rule::QuadratureRule<2, NXY>& space_rule,
                                        const quadrature::rule::QuadratureRule<1, NT>&  time_rule) noexcept
    {
        return make_space_time_tensor_product_rule(space_rule, time_rule);
    }
}

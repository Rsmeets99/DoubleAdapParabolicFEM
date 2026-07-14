#pragma once

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include "../../coefficients/diffusion_coefficient.hpp"

namespace finite_element::time_slabs::detail
{
    template<int DimSpace, typename GradientType>
    [[nodiscard]] double spatial_component_sq_norm(const GradientType& grad) noexcept
    {
        double value = 0.0;
        for (int d = 0; d < DimSpace; ++d)
            value += grad[d] * grad[d];
        return value;
    }

    template<int DimSpace, typename GradientTypeA, typename GradientTypeB>
    [[nodiscard]] double spatial_difference_sq_norm(
        const GradientTypeA& a,
        const GradientTypeB& b) noexcept
    {
        double value = 0.0;
        for (int d = 0; d < DimSpace; ++d)
        {
            const double diff = a[d] - b[d];
            value += diff * diff;
        }
        return value;
    }

    [[nodiscard]] inline double clamp_small_negative(double value) noexcept
    {
        if (value < 0.0 && value > -1.0e-12)
            return 0.0;
        return value;
    }

    template<std::size_t DimSpace>
    [[nodiscard]] double flux_mismatch_energy(
        const coefficients::DiffusionTensor<DimSpace>& M_value,
        const coefficients::DiffusionVector<DimSpace>& sigma,
        const coefficients::DiffusionVector<DimSpace>& grad_theta)
    {
        const auto M_grad_theta =
            coefficients::apply_M<DimSpace>(M_value, grad_theta);

        coefficients::DiffusionVector<DimSpace> mismatch{};
        for (std::size_t d = 0; d < DimSpace; ++d)
            mismatch[d] = sigma[d] + M_grad_theta[d];

        return coefficients::inverse_diffusion_dot(
            M_value,
            mismatch,
            mismatch);
    }

    inline void require_nonnegative_squared_value(
        double value,
        std::string_view context)
    {
        if (!std::isfinite(value) || value < -1.0e-14)
        {
            throw std::runtime_error(
                std::string(context) +
                ": squared quantity is negative or non-finite.");
        }
    }

    [[nodiscard]] inline double checked_positive_measure_factor(
        double jacobian_measure,
        double quadrature_weight,
        std::string_view context)
    {
        if (!std::isfinite(jacobian_measure) || !(jacobian_measure > 0.0))
        {
            throw std::runtime_error(
                std::string(context) +
                ": non-positive or non-finite Jacobian measure.");
        }

        if (!std::isfinite(quadrature_weight) || !(quadrature_weight > 0.0))
        {
            throw std::runtime_error(
                std::string(context) +
                ": non-positive or non-finite quadrature weight.");
        }

        const double factor = jacobian_measure * quadrature_weight;
        if (!std::isfinite(factor) || !(factor > 0.0))
        {
            throw std::runtime_error(
                std::string(context) +
                ": non-positive or non-finite quadrature measure factor.");
        }

        return factor;
    }

    [[nodiscard]] inline double checked_nonnegative_contribution(
        double squared_value,
        double measure_factor,
        std::string_view context)
    {
        require_nonnegative_squared_value(squared_value, context);

        const double contribution = squared_value * measure_factor;
        if (!std::isfinite(contribution) || contribution < -1.0e-14)
        {
            throw std::runtime_error(
                std::string(context) +
                ": local squared contribution is negative or non-finite.");
        }

        return clamp_small_negative(contribution);
    }

    template<class MapType>
    void require_nonnegative_cellwise_map(
        const MapType& values,
        std::string_view context)
    {
        for (const auto& [cell_id, value] : values)
        {
            static_cast<void>(cell_id);
            require_nonnegative_squared_value(value, context);
        }
    }

    struct UnitCoefficient
    {
        template<typename PointType>
        [[nodiscard]] auto operator()(const PointType&) const noexcept
        {
            constexpr std::size_t dim_space =
                std::tuple_size_v<PointType> - 1u;
            if constexpr (dim_space == 1u)
            {
                return 1.0;
            }
            else
            {
                return coefficients::identity_diffusion_tensor<dim_space>();
            }
        }
    };
}

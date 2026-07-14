#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <optional>
#include <type_traits>
#include <tuple>

namespace finite_element::coefficients
{
    /*
     * Diffusion coefficient contract
     * ------------------------------
     *
     * The public assembly code represents all evaluated coefficients as a
     * DiffusionTensor<DimSpace>. The callback type itself is intentionally
     * dimension-specific:
     *
     * - dim_space == 1: M(point) must return a positive scalar convertible to
     *   double. evaluate_diffusion_tensor converts it to the 1x1 tensor [M].
     * - dim_space >= 2: M(point) must return a DimSpace x DimSpace tensor-like
     *   value, normally std::array<std::array<double, DimSpace>, DimSpace>.
     *   Scalar callbacks are rejected at compile time.
     *
     * Tensor values are assumed finite, symmetric, and positive definite at
     * every quadrature point. The hot assembly and estimator kernels do not
     * validate this at runtime; choosing a uniformly elliptic coefficient is the
     * responsibility of the example/application:
     *
     *     alpha |xi|^2 <= xi^T M(x,t) xi <= beta |xi|^2
     *
     * for constants 0 < alpha <= beta < infinity.
     */

    // Local coefficient/vector value used for gradients and fluxes in
    // diffusion-related pointwise operations.
    template<std::size_t DimSpace>
    using DiffusionVector =
        std::array<double, DimSpace>;

    // Local diffusion tensor value after evaluating M at a quadrature point.
    template<std::size_t DimSpace>
    using DiffusionTensor =
        std::array<
            std::array<double, DimSpace>,
            DimSpace>;

    template<std::size_t DimSpace>
    [[nodiscard]] constexpr DiffusionTensor<DimSpace>
    identity_diffusion_tensor() noexcept
    {
        static_assert(DimSpace >= 1u,
                      "Diffusion dimension must be positive.");

        DiffusionTensor<DimSpace> tensor{};
        for (std::size_t d = 0; d < DimSpace; ++d)
            tensor[d][d] = 1.0;
        return tensor;
    }

    template<std::size_t DimSpace>
    struct IdentityDiffusion
    {
        template<class Point>
        [[nodiscard]] constexpr auto operator()(const Point&) const noexcept
        {
            static_assert(DimSpace >= 1u,
                          "Diffusion dimension must be positive.");

            if constexpr (DimSpace == 1u)
                return 1.0;
            else
                return identity_diffusion_tensor<DimSpace>();
        }
    };

    template<std::size_t DimSpace>
    struct ConstantDiffusion
    {
        DiffusionTensor<DimSpace> value =
            identity_diffusion_tensor<DimSpace>();

        template<class Point>
        [[nodiscard]] constexpr auto operator()(const Point&) const noexcept
        {
            if constexpr (DimSpace == 1u)
                return value[0][0];
            else
                return value;
        }
    };

    struct ZeroLoad
    {
        template<class Point>
        [[nodiscard]] constexpr double operator()(const Point&) const noexcept
        {
            return 0.0;
        }
    };

    namespace detail
    {
        template<class>
        inline constexpr bool dependent_false_v = false;

        template<std::size_t DimSpace>
        [[nodiscard]] constexpr DiffusionTensor<DimSpace>
        scalar_to_tensor(double value) noexcept
        {
            static_assert(DimSpace >= 1u,
                          "Diffusion dimension must be positive.");

            DiffusionTensor<DimSpace> tensor{};
            for (std::size_t d = 0; d < DimSpace; ++d)
                tensor[d][d] = value;
            return tensor;
        }

        template<class Value>
        using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<Value>>;

        template<class Function, class Target>
        concept has_std_function_target =
            requires(const Function& function)
            {
                {
                    function.template target<Target>()
                } -> std::same_as<const Target*>;
            };

        template<class Value>
        using tensor_row_t =
            remove_cvref_t<decltype(std::declval<const Value&>()[std::size_t{0}])>;

        template<class Value, std::size_t DimSpace>
        concept TensorLike =
            requires(const Value& value)
            {
                typename std::tuple_size<remove_cvref_t<Value>>::type;
                typename std::tuple_size<tensor_row_t<Value>>::type;
                {
                    value[std::size_t{0}][std::size_t{0}]
                } -> std::convertible_to<double>;
            } &&
            (std::tuple_size_v<remove_cvref_t<Value>> == DimSpace) &&
            (std::tuple_size_v<tensor_row_t<Value>> == DimSpace);

        template<std::size_t DimSpace>
        [[nodiscard]] DiffusionTensor<DimSpace> cholesky_factor_unchecked(
            const DiffusionTensor<DimSpace>& tensor)
        {
            static_assert(DimSpace >= 1u,
                          "Diffusion dimension must be positive.");

            DiffusionTensor<DimSpace> factor{};
            for (std::size_t i = 0; i < DimSpace; ++i)
            {
                for (std::size_t j = 0; j <= i; ++j)
                {
                    double value = tensor[i][j];
                    for (std::size_t k = 0; k < j; ++k)
                        value -= factor[i][k] * factor[j][k];

                    if (i == j)
                    {
                        factor[i][j] = std::sqrt(value);
                    }
                    else
                    {
                        factor[i][j] = value / factor[j][j];
                    }
                }
            }
            return factor;
        }
    }

    template<std::size_t DimSpace, class Function>
    [[nodiscard]] bool is_identity_diffusion_function(
        const Function& function) noexcept
    {
        using FunctionType = detail::remove_cvref_t<Function>;
        using IdentityType = IdentityDiffusion<DimSpace>;
        if constexpr (std::same_as<FunctionType, IdentityType>)
        {
            return true;
        }
        else if constexpr (detail::has_std_function_target<
                               FunctionType,
                               IdentityType>)
        {
            return function.template target<IdentityType>() != nullptr;
        }
        else
        {
            return false;
        }
    }

    template<std::size_t DimSpace, class Function>
    [[nodiscard]] std::optional<DiffusionTensor<DimSpace>>
    constant_diffusion_tensor_if_available(const Function& function)
    {
        using FunctionType = detail::remove_cvref_t<Function>;
        using ConstantType = ConstantDiffusion<DimSpace>;
        using IdentityType = IdentityDiffusion<DimSpace>;
        if constexpr (std::same_as<FunctionType, IdentityType>)
        {
            return identity_diffusion_tensor<DimSpace>();
        }
        else if constexpr (std::same_as<FunctionType, ConstantType>)
        {
            return function.value;
        }
        else
        {
            if constexpr (detail::has_std_function_target<
                              FunctionType,
                              IdentityType>)
            {
                if (const auto* identity =
                        function.template target<IdentityType>())
                {
                    static_cast<void>(identity);
                    return identity_diffusion_tensor<DimSpace>();
                }
            }
            if constexpr (detail::has_std_function_target<
                              FunctionType,
                              ConstantType>)
            {
                if (const auto* constant =
                        function.template target<ConstantType>())
                {
                    return constant->value;
                }
            }
            return std::nullopt;
        }
    }

    template<class Function>
    [[nodiscard]] bool is_zero_load_function(
        const Function& function) noexcept
    {
        using FunctionType = detail::remove_cvref_t<Function>;
        if constexpr (std::same_as<FunctionType, ZeroLoad>)
        {
            return true;
        }
        else if constexpr (detail::has_std_function_target<
                               FunctionType,
                               ZeroLoad>)
        {
            return function.template target<ZeroLoad>() != nullptr;
        }
        else
        {
            return false;
        }
    }

    template<class MFunction, class Point>
    concept is_scalar_diffusion =
        requires(const MFunction& M, const Point& point)
        {
            { std::invoke(M, point) } -> std::convertible_to<double>;
        };

    template<class MFunction, class Point>
    inline constexpr bool is_scalar_diffusion_v =
        is_scalar_diffusion<MFunction, Point>;

    // Compile-time callback contract for a coefficient evaluated at Point.
    // This checks the returned shape only. Runtime validity of tensor values is
    // intentionally not checked in the hot assembly/estimator path.
    template<class MFunction, std::size_t DimSpace, class Point>
    concept is_valid_diffusion_coefficient =
        requires(const MFunction& M, const Point& point)
        {
            typename std::invoke_result_t<const MFunction&, const Point&>;
            requires
                ((DimSpace == 1u &&
                  std::convertible_to<
                      std::invoke_result_t<const MFunction&, const Point&>,
                      double>) ||
                 detail::TensorLike<
                     std::invoke_result_t<const MFunction&, const Point&>,
                     DimSpace>);
        };

    // Evaluate M(point) and enforce only the dimension/shape contract. The
    // returned tensor is consumed by assembly and estimator code without runtime
    // validity checks.
    template<class MFunction, std::size_t DimSpace, class Point>
    [[nodiscard]] DiffusionTensor<DimSpace> evaluate_diffusion_tensor(
        const MFunction& M,
        const Point& point)
    {
        static_assert(DimSpace >= 1u,
                      "Diffusion dimension must be positive.");

        using RawValue =
            std::invoke_result_t<const MFunction&, const Point&>;

        if constexpr (std::convertible_to<RawValue, double>)
        {
            static_assert(
                DimSpace == 1u,
                "Scalar diffusion coefficients are only accepted when "
                "dim_space == 1. For dim_space >= 2, return a symmetric "
                "positive definite dim_space x dim_space tensor.");
            return detail::scalar_to_tensor<DimSpace>(
                static_cast<double>(std::invoke(M, point)));
        }
        else if constexpr (detail::TensorLike<RawValue, DimSpace>)
        {
            const auto raw = std::invoke(M, point);
            DiffusionTensor<DimSpace> tensor{};
            for (std::size_t i = 0; i < DimSpace; ++i)
            {
                for (std::size_t j = 0; j < DimSpace; ++j)
                {
                    tensor[i][j] = static_cast<double>(raw[i][j]);
                }
            }
            return tensor;
        }
        else
        {
            static_assert(
                detail::dependent_false_v<RawValue>,
                "Diffusion coefficient must return either a scalar or a "
                "DimSpace x DimSpace tensor with operator[] access.");
        }
    }

    template<std::size_t DimSpace, class MFunction, class Point>
    [[nodiscard]] DiffusionTensor<DimSpace> evaluate_diffusion_tensor(
        const MFunction& M,
        const Point& point)
    {
        return evaluate_diffusion_tensor<MFunction, DimSpace>(M, point);
    }

    template<std::size_t DimSpace>
    [[nodiscard]] double dot(
        const DiffusionVector<DimSpace>& a,
        const DiffusionVector<DimSpace>& b) noexcept
    {
        double value = 0.0;
        for (std::size_t d = 0; d < DimSpace; ++d)
            value += a[d] * b[d];
        return value;
    }

    template<std::size_t DimSpace>
    [[nodiscard]] DiffusionVector<DimSpace> apply_M(
        const DiffusionTensor<DimSpace>& M_value,
        const DiffusionVector<DimSpace>& grad) noexcept
    {
        DiffusionVector<DimSpace> result{};
        for (std::size_t i = 0; i < DimSpace; ++i)
        {
            for (std::size_t j = 0; j < DimSpace; ++j)
                result[i] += M_value[i][j] * grad[j];
        }
        return result;
    }

    // Backward-compatible name used by assembly kernels. It is now identical to
    // apply_M because coefficient validation has been removed from hot paths.
    template<std::size_t DimSpace>
    [[nodiscard]] DiffusionVector<DimSpace> apply_validated_M(
        const DiffusionTensor<DimSpace>& M_value,
        const DiffusionVector<DimSpace>& grad) noexcept
    {
        return apply_M<DimSpace>(M_value, grad);
    }

    template<std::size_t DimSpace>
    [[nodiscard]] DiffusionVector<DimSpace> apply_M_inverse(
        const DiffusionTensor<DimSpace>& M_value,
        const DiffusionVector<DimSpace>& vector) noexcept
    {
        if constexpr (DimSpace == 1u)
        {
            return DiffusionVector<DimSpace>{
                vector[0] / M_value[0][0]
            };
        }
        else if constexpr (DimSpace == 2u)
        {
            const double a = M_value[0][0];
            const double b = M_value[0][1];
            const double c = M_value[1][1];
            const double det = a * c - b * b;

            return DiffusionVector<DimSpace>{
                (c * vector[0] - b * vector[1]) / det,
                (-b * vector[0] + a * vector[1]) / det
            };
        }
        else
        {
            const auto L = detail::cholesky_factor_unchecked(M_value);

            DiffusionVector<DimSpace> y{};
            for (std::size_t i = 0; i < DimSpace; ++i)
            {
                double value = vector[i];
                for (std::size_t k = 0; k < i; ++k)
                    value -= L[i][k] * y[k];
                y[i] = value / L[i][i];
            }

            DiffusionVector<DimSpace> x{};
            for (std::size_t i = DimSpace; i-- > 0u;)
            {
                double value = y[i];
                for (std::size_t k = i + 1; k < DimSpace; ++k)
                    value -= L[k][i] * x[k];
                x[i] = value / L[i][i];
            }
            return x;
        }
    }

    // Bilinear energy product grad_a^T M grad_b.
    template<std::size_t DimSpace>
    [[nodiscard]] double diffusion_energy_dot(
        const DiffusionTensor<DimSpace>& M_value,
        const DiffusionVector<DimSpace>& grad_a,
        const DiffusionVector<DimSpace>& grad_b)
    {
        return dot(grad_a, apply_M(M_value, grad_b));
    }

    // Bilinear inverse-flux product sigma_a^T M^{-1} sigma_b.
    template<std::size_t DimSpace>
    [[nodiscard]] double inverse_diffusion_dot(
        const DiffusionTensor<DimSpace>& M_value,
        const DiffusionVector<DimSpace>& sigma_a,
        const DiffusionVector<DimSpace>& sigma_b)
    {
        return dot(sigma_a, apply_M_inverse(M_value, sigma_b));
    }
}

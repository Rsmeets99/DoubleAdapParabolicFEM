#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include "../adaptive_result.hpp"

#include "finite_element/assembly/detail/active_cell_locator.hpp"
#include "finite_element/assembly/detail/space_time_basis_tables.hpp"
#include "finite_element/assembly/detail/trace_geometry_utils.hpp"
#include "finite_element/coefficients/diffusion_coefficient.hpp"
#include "finite_element/detail/cell_geometry_cache.hpp"
#include "finite_element/geometry/cell_geometry.hpp"

namespace adaptive_algorithm::detail
{
    template<typename CellIdType = int>
    void add_to_cellwise_error(
        finite_element::time_slabs::CellwiseSquaredError<CellIdType>& error,
        CellIdType cell_id,
        double value)
    {
        auto it = error.by_source_cell.find(cell_id);
        if (it == error.by_source_cell.end())
            error.by_source_cell.emplace(cell_id, value);
        else
            it->second += value;
    }

    template<typename CellIdType = int>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<CellIdType>
    sum_cellwise_errors(
        const finite_element::time_slabs::CellwiseSquaredError<CellIdType>& a,
        const finite_element::time_slabs::CellwiseSquaredError<CellIdType>& b)
    {
        auto out = a;

        for (const auto& [cell_id, value] : b.by_source_cell)
            add_to_cellwise_error(out, cell_id, value);

        return out;
    }

    template<typename GradientType>
    [[nodiscard]] double spatial_sq_norm(const GradientType& gradient) noexcept
    {
        constexpr int dim =
            static_cast<int>(std::tuple_size_v<GradientType>) - 1;

        double value = 0.0;
        for (int d = 0; d < dim; ++d)
            value += gradient[d] * gradient[d];

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
            const double diff = a[static_cast<std::size_t>(d)] -
                b[static_cast<std::size_t>(d)];
            value += diff * diff;
        }
        return value;
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
                return finite_element::coefficients::
                    identity_diffusion_tensor<dim_space>();
            }
        }
    };

    template<int DimSpace, class GradientType>
    [[nodiscard]] finite_element::coefficients::DiffusionVector<DimSpace>
    spatial_gradient_vector(const GradientType& gradient) noexcept
    {
        finite_element::coefficients::DiffusionVector<DimSpace> out{};
        for (int d = 0; d < DimSpace; ++d)
            out[static_cast<std::size_t>(d)] =
                gradient[static_cast<std::size_t>(d)];
        return out;
    }

    template<
        int QSpace,
        int QTime,
        class FunctionType,
        class CoefficientFunction>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<int>
    compute_spatial_energy_squared_by_active_cell(
        const FunctionType& function,
        const CoefficientFunction& coefficient)
    {
        using SpaceType = typename FunctionType::SpaceType;
        using GT = typename SpaceType::GT;
        using FETraits = typename SpaceType::FETraitsType;
        using Tables =
            finite_element::assembly::detail::SpaceTimeBasisTables<
                GT,
                FETraits,
                QSpace,
                QTime>;
        using Geometry =
            finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;

        finite_element::time_slabs::CellwiseSquaredError<int> result;

        const auto& space = function.fespace();
        finite_element::detail::CellGeometryCache<SpaceType> geometry_cache(space);

        result.by_source_cell.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
        {
            const auto& geom = geometry_cache.geometry(cell_id);
            const double jac = Geometry::jacobian_measure(geom);

            double local_value = 0.0;
            for (int q = 0; q < Tables::n_cell_q; ++q)
            {
                const auto& xi_q = Tables::cell_rule.points[q];
                const double w_q = Tables::cell_rule.weights[q];
                const auto p_q = Geometry::map_to_physical(geom, xi_q);
                const auto gradient =
                    function.gradient_on_cell(cell_id, p_q, geom);
                const auto spatial_gradient =
                    spatial_gradient_vector<GT::dim_space_v>(gradient);
                const auto M_q =
                    finite_element::coefficients::evaluate_diffusion_tensor<
                        GT::dim_space_v>(
                            coefficient,
                            p_q);

                local_value +=
                    finite_element::coefficients::diffusion_energy_dot(
                        M_q,
                        spatial_gradient,
                        spatial_gradient) *
                    jac * w_q;
            }

            result.by_source_cell.emplace(cell_id, local_value);
        }

        return result;
    }

    template<
        int QSpace,
        int QTime,
        class FunctionType,
        class ExactValueFunction>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<int>
    compute_exact_value_error_squared_by_active_cell(
        const FunctionType& function,
        const ExactValueFunction& exact_value)
    {
        using SpaceType = typename FunctionType::SpaceType;
        using GT = typename SpaceType::GT;
        using FETraits = typename SpaceType::FETraitsType;
        using Tables =
            finite_element::assembly::detail::SpaceTimeBasisTables<
                GT,
                FETraits,
                QSpace,
                QTime>;
        using Geometry =
            finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;

        finite_element::time_slabs::CellwiseSquaredError<int> result;

        const auto& space = function.fespace();
        finite_element::detail::CellGeometryCache<SpaceType> geometry_cache(space);

        result.by_source_cell.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
        {
            const auto& geom = geometry_cache.geometry(cell_id);
            const double jac = Geometry::jacobian_measure(geom);

            double local_value = 0.0;
            for (int q = 0; q < Tables::n_cell_q; ++q)
            {
                const auto& xi_q = Tables::cell_rule.points[q];
                const double w_q = Tables::cell_rule.weights[q];
                const auto p_q = Geometry::map_to_physical(geom, xi_q);

                const double diff =
                    function.value_on_cell(cell_id, p_q, geom) -
                    static_cast<double>(exact_value(p_q));

                local_value += diff * diff * jac * w_q;
            }

            result.by_source_cell.emplace(cell_id, local_value);
        }

        return result;
    }

    template<
        int QSpace,
        int QTime,
        class FunctionType,
        class ExactValueFunction>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<int>
    compute_exact_l2_error_squared_by_active_cell(
        const FunctionType& function,
        const ExactValueFunction& exact_value)
    {
        return compute_exact_value_error_squared_by_active_cell<QSpace, QTime>(
            function,
            exact_value);
    }

    template<
        int QSpace,
        int QTime,
        class FunctionType,
        class ExactSpatialGradientFunction,
        class CoefficientFunction>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<int>
    compute_exact_spatial_energy_error_squared_by_active_cell(
        const FunctionType& function,
        const ExactSpatialGradientFunction& exact_spatial_gradient,
        const CoefficientFunction& coefficient)
    {
        using SpaceType = typename FunctionType::SpaceType;
        using GT = typename SpaceType::GT;
        using FETraits = typename SpaceType::FETraitsType;
        using Tables =
            finite_element::assembly::detail::SpaceTimeBasisTables<
                GT,
                FETraits,
                QSpace,
                QTime>;
        using Geometry =
            finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;

        finite_element::time_slabs::CellwiseSquaredError<int> result;

        const auto& space = function.fespace();
        finite_element::detail::CellGeometryCache<SpaceType> geometry_cache(space);

        result.by_source_cell.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
        {
            const auto& geom = geometry_cache.geometry(cell_id);
            const double jac = Geometry::jacobian_measure(geom);

            double local_value = 0.0;
            for (int q = 0; q < Tables::n_cell_q; ++q)
            {
                const auto& xi_q = Tables::cell_rule.points[q];
                const double w_q = Tables::cell_rule.weights[q];
                const auto p_q = Geometry::map_to_physical(geom, xi_q);

                const auto approximate_gradient =
                    function.gradient_on_cell(cell_id, p_q, geom);
                const auto exact_gradient =
                    exact_spatial_gradient(p_q);
                finite_element::coefficients::DiffusionVector<
                    GT::dim_space_v> gradient_diff{};
                for (int d = 0; d < GT::dim_space_v; ++d)
                {
                    gradient_diff[static_cast<std::size_t>(d)] =
                        approximate_gradient[static_cast<std::size_t>(d)] -
                        exact_gradient[static_cast<std::size_t>(d)];
                }
                const auto M_q =
                    finite_element::coefficients::evaluate_diffusion_tensor<
                        GT::dim_space_v>(
                            coefficient,
                            p_q);

                local_value +=
                    finite_element::coefficients::diffusion_energy_dot(
                        M_q,
                        gradient_diff,
                        gradient_diff) *
                    jac * w_q;
            }

            result.by_source_cell.emplace(cell_id, local_value);
        }

        return result;
    }

    template<
        int QSpace,
        int QTime,
        class FunctionType,
        class ExactSpatialGradientFunction>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<int>
    compute_exact_spatial_gradient_error_squared_by_active_cell(
        const FunctionType& function,
        const ExactSpatialGradientFunction& exact_spatial_gradient)
    {
        return compute_exact_spatial_energy_error_squared_by_active_cell<QSpace, QTime>(
            function,
            exact_spatial_gradient,
            UnitCoefficient{});
    }

    template<
        int QSpace,
        int QTime,
        class XFunctionType,
        class InitialValueFunction>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<int>
    compute_initial_trace_mismatch_squared_by_x_cell(
        const XFunctionType& u_delta,
        const InitialValueFunction& u0)
    {
        using SpaceType = typename XFunctionType::SpaceType;
        using GT = typename SpaceType::GT;
        using FETraits = typename SpaceType::FETraitsType;
        using Tables =
            finite_element::assembly::detail::SpaceTimeBasisTables<
                GT,
                FETraits,
                QSpace,
                QTime>;
        using Geometry =
            finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;

        finite_element::time_slabs::CellwiseSquaredError<int> result;

        const auto& space = u_delta.fespace();
        const auto& mesh = space.mesh_ref();
        finite_element::detail::CellGeometryCache<SpaceType> geometry_cache(space);

        result.by_source_cell.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
        {
            const auto& cell = mesh.cell(cell_id);
            if (!cell.temporal_boundary[0])
                continue;

            const auto& geom = geometry_cache.geometry(cell_id);
            const double trace_measure =
                finite_element::assembly::detail::initial_trace_measure<Geometry>(geom);

            double local_value = 0.0;
            for (int q = 0; q < Tables::n_bottom_q; ++q)
            {
                const auto& xi_bottom = Tables::bottom_rule.points[q];
                const double w_q = Tables::bottom_rule.weights[q];
                const auto p_q =
                    finite_element::assembly::detail::map_bottom_qp_to_physical<Geometry>(
                        geom,
                        xi_bottom);
                const auto x_q =
                    finite_element::assembly::detail::spatial_argument_from_space_time_point<Geometry>(
                        p_q);
                const double diff =
                    static_cast<double>(u0(x_q)) -
                    u_delta.value_on_cell(cell_id, p_q, geom);

                local_value += diff * diff * trace_measure * w_q;
            }

            result.by_source_cell.emplace(cell_id, local_value);
        }

        return result;
    }

    template<class XSpaceType, class FineSpaceType>
    [[nodiscard]] finite_element::time_slabs::CellwiseSquaredError<int>
    aggregate_to_x_active_cells(
        const XSpaceType& x_space,
        const FineSpaceType& fine_space,
        const finite_element::time_slabs::CellwiseSquaredError<int>& fine_error)
    {
        if (&x_space.mesh_ref() != &fine_space.mesh_ref())
        {
            throw std::runtime_error(
                "aggregate_to_x_active_cells: X-space and fine space must share the same mesh.");
        }

        finite_element::time_slabs::CellwiseSquaredError<int> result;
        result.by_source_cell.reserve(x_space.active_cells().size());

        finite_element::assembly::detail::ActiveAncestorCache<XSpaceType> ancestor_cache(x_space);

        for (const auto& [fine_cell_id, value] : fine_error.by_source_cell)
        {
            // X-marking happens on active X-cells, but the lambda^delta term is
            // assembled naturally on the finer Y-mesh. Push each Y-cell
            // contribution to the first active X-ancestor so that the localized
            // outer indicator matches the theory without flattening the tree.
            const int x_cell_id =
                finite_element::assembly::detail::find_active_ancestor_cell(
                    ancestor_cache,
                    x_space,
                    fine_space,
                    fine_cell_id);

            add_to_cellwise_error(result, x_cell_id, value);
        }

        return result;
    }

    template<
        int QSpace,
        int QTime,
        class XSpaceType,
        class YFunctionType,
        class XFunctionType,
        class InitialValueFunction,
        class CoefficientFunction>
    [[nodiscard]] XMarkingIndicatorComponents<int>
    compute_x_marking_indicator_components(
        const XSpaceType& x_space,
        const YFunctionType& lambda_delta,
        const XFunctionType& u_delta,
        const InitialValueFunction& u0,
        const CoefficientFunction& coefficient)
    {
        const auto& y_space = lambda_delta.fespace();

        XMarkingIndicatorComponents<int> result;
        result.lambda_y_squared_by_y_cell =
            compute_spatial_energy_squared_by_active_cell<QSpace, QTime>(
                lambda_delta,
                coefficient);
        result.lambda_y_squared_by_x_cell =
            aggregate_to_x_active_cells(
                x_space,
                y_space,
                result.lambda_y_squared_by_y_cell);
        result.initial_trace_squared_by_x_cell =
            compute_initial_trace_mismatch_squared_by_x_cell<QSpace, QTime>(
                u_delta,
                u0);
        result.eta_squared_by_x_cell =
            sum_cellwise_errors(
                result.lambda_y_squared_by_x_cell,
                result.initial_trace_squared_by_x_cell);

        return result;
    }

    [[nodiscard]] inline double compute_y_estimator_threshold_squared(
        double rho,
        double lambda_y_squared,
        double initial_trace_squared)
    {
        if (!(rho > 0.0))
        {
            throw std::runtime_error(
                "compute_y_estimator_threshold_squared: rho must be positive.");
        }

        const double shifted_rho = rho + 0.5;
        const double initial_trace_factor =
            1.0 + 1.0 / (shifted_rho * shifted_rho);

        return rho * rho *
            (lambda_y_squared + initial_trace_factor * initial_trace_squared);
    }

    [[nodiscard]] inline double posteriori_factor(double rho)
    {
        if (!(rho >= 0.0) || !std::isfinite(rho))
        {
            throw std::runtime_error(
                "posteriori_factor: rho must be finite and nonnegative.");
        }

        const double shifted_rho = rho + 0.5;
        return 1.0 + rho * rho *
            (1.0 + 1.0 / (shifted_rho * shifted_rho));
    }

    struct EffectiveRhoComputation
    {
        bool available = false;
        std::optional<double> rho{};
        std::optional<double> threshold_squared{};
        std::string reason = "not_computed";
    };

    [[nodiscard]] inline EffectiveRhoComputation compute_effective_rho(
        double estimator_squared,
        double lambda_y_squared,
        double initial_trace_squared,
        double rho_upper_bound)
    {
        if (!(rho_upper_bound > 0.0))
        {
            throw std::runtime_error(
                "compute_effective_rho: rho_upper_bound must be positive.");
        }

        EffectiveRhoComputation result;

        if (!std::isfinite(estimator_squared) ||
            !std::isfinite(lambda_y_squared) ||
            !std::isfinite(initial_trace_squared))
        {
            result.reason = "nonfinite_input";
            return result;
        }

        if (lambda_y_squared < 0.0 || initial_trace_squared < 0.0)
        {
            result.reason = "negative_reference_terms";
            return result;
        }

        if (estimator_squared <= 0.0)
        {
            result.available = true;
            result.rho = 0.0;
            result.threshold_squared = 0.0;
            result.reason = "zero_estimator";
            return result;
        }

        if (!(lambda_y_squared > 0.0) && !(initial_trace_squared > 0.0))
        {
            result.reason = "missing_positive_reference_terms";
            return result;
        }

        const auto threshold_for =
            [lambda_y_squared, initial_trace_squared](double rho)
            {
                return compute_y_estimator_threshold_squared(
                    rho,
                    lambda_y_squared,
                    initial_trace_squared);
            };

        const double configured_threshold =
            threshold_for(rho_upper_bound);
        if (!std::isfinite(configured_threshold))
        {
            result.reason = "nonfinite_configured_threshold";
            return result;
        }

        const double tolerance =
            128.0 * std::numeric_limits<double>::epsilon() *
            std::max(
                1.0,
                std::max(
                    std::abs(estimator_squared),
                    std::abs(configured_threshold)));

        if (estimator_squared > configured_threshold + tolerance)
        {
            result.reason = "estimator_exceeds_configured_threshold";
            return result;
        }

        if (estimator_squared >= configured_threshold - tolerance)
        {
            result.available = true;
            result.rho = rho_upper_bound;
            result.threshold_squared = configured_threshold;
            result.reason = "configured_rho_is_effective";
            return result;
        }

        double lo = 0.0;
        double hi = rho_upper_bound;
        for (int iteration = 0; iteration < 120; ++iteration)
        {
            const double mid = 0.5 * (lo + hi);
            const double mid_threshold = threshold_for(mid);
            if (mid_threshold < estimator_squared)
                lo = mid;
            else
                hi = mid;
        }

        const double rho_eff = hi;
        result.available = true;
        result.rho = rho_eff;
        result.threshold_squared = threshold_for(rho_eff);
        result.reason = "computed";
        return result;
    }

    [[nodiscard]] inline EffectiveRhoComputation
    compute_required_effective_rho(
        double estimator_squared,
        double lambda_y_squared,
        double initial_trace_squared)
    {
        EffectiveRhoComputation result;

        if (!std::isfinite(estimator_squared) ||
            !std::isfinite(lambda_y_squared) ||
            !std::isfinite(initial_trace_squared))
        {
            result.reason = "nonfinite_input";
            return result;
        }

        if (lambda_y_squared < 0.0 || initial_trace_squared < 0.0)
        {
            result.reason = "negative_reference_terms";
            return result;
        }

        if (estimator_squared <= 0.0)
        {
            result.available = true;
            result.rho = 0.0;
            result.threshold_squared = 0.0;
            result.reason = "zero_estimator";
            return result;
        }

        if (!(lambda_y_squared > 0.0) && !(initial_trace_squared > 0.0))
        {
            result.reason = "missing_positive_reference_terms";
            return result;
        }

        const auto threshold_for =
            [lambda_y_squared, initial_trace_squared](double rho)
            {
                return compute_y_estimator_threshold_squared(
                    rho,
                    lambda_y_squared,
                    initial_trace_squared);
            };

        double lo = 0.0;
        double hi = 1.0;
        double hi_threshold = threshold_for(hi);
        for (int expansion = 0;
             hi_threshold < estimator_squared && expansion < 128;
             ++expansion)
        {
            lo = hi;
            hi *= 2.0;
            hi_threshold = threshold_for(hi);
            if (!std::isfinite(hi) || !std::isfinite(hi_threshold))
            {
                result.reason = "nonfinite_required_threshold";
                return result;
            }
        }

        if (hi_threshold < estimator_squared)
        {
            result.reason = "unable_to_bracket_required_rho";
            return result;
        }

        const double tolerance =
            128.0 * std::numeric_limits<double>::epsilon() *
            std::max(
                1.0,
                std::max(
                    std::abs(estimator_squared),
                    std::abs(hi_threshold)));

        if (estimator_squared >= hi_threshold - tolerance)
        {
            result.available = true;
            result.rho = hi;
            result.threshold_squared = hi_threshold;
            result.reason = "computed_required";
            return result;
        }

        for (int iteration = 0; iteration < 160; ++iteration)
        {
            const double mid = 0.5 * (lo + hi);
            const double mid_threshold = threshold_for(mid);
            if (mid_threshold < estimator_squared)
                lo = mid;
            else
                hi = mid;
        }

        const double rho_eff = hi;
        result.available = true;
        result.rho = rho_eff;
        result.threshold_squared = threshold_for(rho_eff);
        result.reason = "computed_required";
        return result;
    }
}

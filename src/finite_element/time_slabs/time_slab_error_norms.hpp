#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "../assembly/detail/active_cell_locator_time_slab.hpp"
#include "../coefficients/diffusion_coefficient.hpp"
#include "../assembly/detail/space_time_basis_tables.hpp"
#include "../detail/cell_geometry_cache.hpp"
#include "../detail/timing.hpp"
#include "../geometry/cell_geometry.hpp"
#include "detail/time_slab_error_indicator_detail.hpp"
#include "time_slab_cellwise_errors.hpp"
#include "time_slab_function_view.hpp"

namespace finite_element::time_slabs
{
    struct ReconstructionYCellCrossTermDiagnostics
    {
        int source_cell_id = -1;
        double error_squared = 0.0;
        double source_norm = 0.0;
        double slab_norm = 0.0;
        double cross_term = 0.0;
        int source_slice_count = 0;
        std::vector<int> slab_ids{};
    };

    struct ReconstructionYCrossTermDiagnostics
    {
        CellwiseSquaredError<int> error_squared{};
        std::unordered_map<int, double> source_norm_by_source_cell{};
        std::unordered_map<int, double> slab_norm_by_source_cell{};
        std::unordered_map<int, double> cross_term_by_source_cell{};
        std::unordered_map<int, int> source_slice_count{};
        std::unordered_map<int, std::vector<int>> slab_ids_by_source_cell{};

        [[nodiscard]] double total_source_norm() const
        {
            return sum_map_(source_norm_by_source_cell);
        }

        [[nodiscard]] double total_slab_norm() const
        {
            return sum_map_(slab_norm_by_source_cell);
        }

        [[nodiscard]] double total_cross_term() const
        {
            return sum_map_(cross_term_by_source_cell);
        }

        [[nodiscard]] double total_error_from_cross_terms() const
        {
            const double source_norm = total_source_norm();
            const double slab_norm = total_slab_norm();
            const double cross_term = total_cross_term();
            double value = source_norm + slab_norm - 2.0 * cross_term;
            const double scale =
                std::max(
                    1.0,
                    std::abs(source_norm) + std::abs(slab_norm) +
                        2.0 * std::abs(cross_term));
            const double cancellation_tol = 1.0e-12 * scale;
            if (value < 0.0 && value > -cancellation_tol)
                value = 0.0;
            return value;
        }

        [[nodiscard]] std::vector<ReconstructionYCellCrossTermDiagnostics>
        top_cells(std::size_t k) const
        {
            const auto entries = error_squared.top_k(k);
            std::vector<ReconstructionYCellCrossTermDiagnostics> result;
            result.reserve(entries.size());

            for (const auto& [source_cell_id, value] : entries)
            {
                ReconstructionYCellCrossTermDiagnostics cell;
                cell.source_cell_id = source_cell_id;
                cell.error_squared = value;
                cell.source_norm =
                    find_or_zero_(source_norm_by_source_cell, source_cell_id);
                cell.slab_norm =
                    find_or_zero_(slab_norm_by_source_cell, source_cell_id);
                cell.cross_term =
                    find_or_zero_(cross_term_by_source_cell, source_cell_id);

                const auto count_it =
                    source_slice_count.find(source_cell_id);
                if (count_it != source_slice_count.end())
                    cell.source_slice_count = count_it->second;

                const auto slab_it =
                    slab_ids_by_source_cell.find(source_cell_id);
                if (slab_it != slab_ids_by_source_cell.end())
                    cell.slab_ids = slab_it->second;

                result.push_back(std::move(cell));
            }

            return result;
        }

    private:
        template<class MapType>
        [[nodiscard]] static double sum_map_(const MapType& map)
        {
            double sum = 0.0;
            for (const auto& [key, value] : map)
            {
                static_cast<void>(key);
                sum += value;
            }
            return sum;
        }

        template<class MapType>
        [[nodiscard]] static double find_or_zero_(
            const MapType& map,
            const int key)
        {
            const auto it = map.find(key);
            return it == map.end() ? 0.0 : it->second;
        }
    };

    namespace detail
    {
        struct ReconstructionYNormViewProfile
        {
            double total = 0.0;
            double slab_cell_iteration = 0.0;
            double slab_cell_view_construction = 0.0;
            double geometry_construction = 0.0;
            double source_geometry_lookup = 0.0;
            double tilde_geometry_lookup = 0.0;
            double source_gradient_eval = 0.0;
            double tilde_gradient_eval = 0.0;
            double diffusion_eval = 0.0;
            double accumulation = 0.0;
            double map_conversion = 0.0;
            double slab_cell_views_allocated = 0.0;
            double qpoints_visited = 0.0;
            double source_cells_touched = 0.0;
            double slab_cells_touched = 0.0;
            double geometry_cache_hits = 0.0;
            double geometry_cache_misses = 0.0;
            double virtual_geometry_constructed = 0.0;
            double copied_geometry_cache_lookups = 0.0;
            double fast_kernel_used = 0.0;
            double accumulator_type = 0.0;
            double accumulator_type_compact_vector = 0.0;

            void record(
                const finite_element::detail::TimingRecorder& timing) const
            {
                timing.add("reconstruction_y_norm.total", total);
                timing.add(
                    "reconstruction_y_norm.slab_cell_iteration",
                    slab_cell_iteration);
                timing.add(
                    "reconstruction_y_norm.slab_cell_view_construction",
                    slab_cell_view_construction);
                timing.add(
                    "reconstruction_y_norm.geometry_construction",
                    geometry_construction);
                timing.add(
                    "reconstruction_y_norm.source_geometry_lookup",
                    source_geometry_lookup);
                timing.add(
                    "reconstruction_y_norm.tilde_geometry_lookup",
                    tilde_geometry_lookup);
                timing.add(
                    "reconstruction_y_norm.source_gradient_eval",
                    source_gradient_eval);
                timing.add(
                    "reconstruction_y_norm.tilde_gradient_eval",
                    tilde_gradient_eval);
                timing.add(
                    "reconstruction_y_norm.diffusion_eval",
                    diffusion_eval);
                timing.add(
                    "reconstruction_y_norm.accumulation",
                    accumulation);
                timing.add(
                    "reconstruction_y_norm.map_conversion",
                    map_conversion);
                timing.add(
                    "reconstruction_y_norm.slab_cell_views_allocated",
                    slab_cell_views_allocated);
                timing.add(
                    "reconstruction_y_norm.qpoints_visited",
                    qpoints_visited);
                timing.add(
                    "reconstruction_y_norm.source_cells_touched",
                    source_cells_touched);
                timing.add(
                    "reconstruction_y_norm.slab_cells_touched",
                    slab_cells_touched);
                timing.add(
                    "reconstruction_y_norm.geometry_cache_hits",
                    geometry_cache_hits);
                timing.add(
                    "reconstruction_y_norm.geometry_cache_misses",
                    geometry_cache_misses);
                timing.add(
                    "reconstruction_y_norm.virtual_geometry_constructed",
                    virtual_geometry_constructed);
                timing.add(
                    "reconstruction_y_norm.copied_geometry_cache_lookups",
                    copied_geometry_cache_lookups);
                timing.add(
                    "reconstruction_y_norm.fast_kernel_used",
                    fast_kernel_used);
                timing.add(
                    "reconstruction_y_norm.accumulator_type",
                    accumulator_type);
                timing.add(
                    "reconstruction_y_norm.accumulator_type."
                    "compact_vector.count",
                    accumulator_type_compact_vector);
            }
        };

        inline constexpr double compact_vector_accumulator_type_code = 1.0;

        using ReconstructionYNormClock = std::chrono::steady_clock;

        [[nodiscard]] inline double reconstruction_y_norm_elapsed_seconds(
            const ReconstructionYNormClock::time_point begin)
        {
            return std::chrono::duration<double>(
                ReconstructionYNormClock::now() - begin)
                .count();
        }

        [[nodiscard]] inline double reconstruction_error_from_cross_terms(
            double source_norm,
            double slab_norm,
            double cross_term)
        {
            double value = source_norm + slab_norm - 2.0 * cross_term;
            const double scale =
                std::max(
                    1.0,
                    std::abs(source_norm) + std::abs(slab_norm) +
                        2.0 * std::abs(cross_term));
            const double cancellation_tol = 1.0e-12 * scale;
            if (value < 0.0 && value > -cancellation_tol)
                value = 0.0;
            return value;
        }

        template<
            int QSpace,
            int QTime,
            class SourceFunctionType,
            class ReconstructedFunctionType,
            class CoefficientFunction>
        [[nodiscard]] CellwiseSquaredError<int>
        compute_spatial_energy_error_squared_impl(
            const SourceFunctionType& lambda_delta,
            const ReconstructedFunctionType& lambda_tilde,
            const CoefficientFunction& coeff)
        {
            using SourceSpaceType = typename SourceFunctionType::SpaceType;
            using GT              = typename SourceSpaceType::GT;
            using FETraits        = typename SourceSpaceType::FETraitsType;

            using SlabSpaceType   = typename ReconstructedFunctionType::SlabSpaceType;
            using SlabType        = typename SlabSpaceType::SlabType;
            using LocalSpaceType  = typename SlabType::SpaceType;

            using Tables   =
                finite_element::assembly::detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
            using Geometry =
                finite_element::geometry::CellGeometry<LocalSpaceType, GT::dim_space_v>;

            CellwiseSquaredError<int> result;

            const auto& slab_space = lambda_tilde.slab_space();
            const auto& source_space = lambda_delta.fespace();

            result.by_source_cell.reserve(source_space.active_cells().size());

            finite_element::detail::CellGeometryCache<SourceSpaceType> source_geometry_cache(source_space);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab          = slab_space.slab(slab_id);
                const auto& slab_fespace  = slab.fespace_ref();
                const auto& slab_function = lambda_tilde.slab_function(slab_id);
                finite_element::detail::CellGeometryCache<LocalSpaceType> slab_geometry_cache(slab_fespace);

                for (const int slab_cell_id : slab.active_cells())
                {
                    const int source_cell_id = slab.source_cell_id(slab_cell_id);
                    const auto& source_geom  = source_geometry_cache.geometry(source_cell_id);
                    const auto& geom         = slab_geometry_cache.geometry(slab_cell_id);
                    const double jac         = Geometry::jacobian_measure(geom);

                    double local_contribution = 0.0;

                    for (int q = 0; q < Tables::n_cell_q; ++q)
                    {
                        const auto& xi_q = Tables::cell_rule.points[q];
                        const double w_q = Tables::cell_rule.weights[q];
                        const auto x_q   = Geometry::map_to_physical(geom, xi_q);
                        const double measure_factor =
                            checked_positive_measure_factor(
                                jac,
                                w_q,
                                "compute_spatial_energy_error_squared_impl");

                        const auto grad_source =
                            lambda_delta.gradient_on_cell(source_cell_id, x_q, source_geom);
                        const auto grad_tilde =
                            slab_function.gradient_on_cell(slab_cell_id, x_q, geom);

                        coefficients::DiffusionVector<GT::dim_space_v> grad_diff{};
                        for (int d = 0; d < GT::dim_space_v; ++d)
                        {
                            grad_diff[static_cast<std::size_t>(d)] =
                                grad_source[static_cast<std::size_t>(d)] -
                                grad_tilde[static_cast<std::size_t>(d)];
                        }
                        const auto M_q =
                            coefficients::evaluate_diffusion_tensor<
                                GT::dim_space_v>(
                                    coeff,
                                    x_q);

                        local_contribution +=
                            checked_nonnegative_contribution(
                                coefficients::diffusion_energy_dot(
                                    M_q,
                                    grad_diff,
                                    grad_diff),
                                measure_factor,
                                "compute_spatial_energy_error_squared_impl");
                    }

                    add_to_map(result.by_source_cell, source_cell_id, local_contribution);
                }
            }

            require_nonnegative_cellwise_map(
                result.by_source_cell,
                "compute_spatial_energy_error_squared_impl aggregation");

            return result;
        }

        template<
            int QSpace,
            int QTime,
            class SourceFunctionType,
            TimeSlabFunctionView FunctionViewType,
            class CoefficientFunction>
        [[nodiscard]] CellwiseSquaredError<int>
        compute_Y_error_squared_by_source_cell_fast_view(
            const SourceFunctionType& lambda_delta,
            const FunctionViewType& lambda_tilde,
            const CoefficientFunction& coeff,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            const auto total_begin = ReconstructionYNormClock::now();
            ReconstructionYNormViewProfile profile;
            profile.fast_kernel_used = 1.0;
            profile.accumulator_type =
                compact_vector_accumulator_type_code;
            profile.accumulator_type_compact_vector = 1.0;

            using SourceSpaceType = typename SourceFunctionType::SpaceType;
            using GT              = typename SourceSpaceType::GT;
            using FETraits        = typename SourceSpaceType::FETraitsType;

            using Tables =
                finite_element::assembly::detail::SpaceTimeBasisTables<
                    GT,
                    FETraits,
                    QSpace,
                    QTime>;
            using SourceGeometry =
                finite_element::geometry::CellGeometry<
                    SourceSpaceType,
                    GT::dim_space_v>;
            using SlabGeometry = typename FunctionViewType::Geometry;

            const auto& source_space = lambda_delta.fespace();
            CellwiseActiveAccumulator<int> result_accumulator(
                source_space.active_cells());

            finite_element::detail::CellGeometryCache<SourceSpaceType>
                source_geometry_cache(source_space);
            auto tilde_geometry_cache = lambda_tilde.make_geometry_cache();

            for (int slab_id = 0; slab_id < lambda_tilde.n_slabs(); ++slab_id)
            {
                lambda_tilde.for_each_slab_cell(
                    slab_id,
                    [&](const auto& slab_cell)
                {
                    const int source_cell_id =
                        lambda_tilde.source_cell_id(slab_cell);
                    ++profile.slab_cells_touched;
                    profile.qpoints_visited +=
                        static_cast<double>(Tables::n_cell_q);

                    const auto& source_geom =
                        source_geometry_cache.geometry(source_cell_id);
                    const auto& slab_geom =
                        lambda_tilde.geometry(
                            tilde_geometry_cache,
                            slab_cell,
                            source_geom);
                    const double jac =
                        SlabGeometry::jacobian_measure(slab_geom);

                    double local_contribution = 0.0;

                    for (int q = 0; q < Tables::n_cell_q; ++q)
                    {
                        const auto& xi_q = Tables::cell_rule.points[q];
                        const double w_q = Tables::cell_rule.weights[q];
                        const auto x_q =
                            SlabGeometry::map_to_physical(
                                slab_geom,
                                xi_q);
                        const double measure_factor =
                            checked_positive_measure_factor(
                                jac,
                                w_q,
                                "compute_time_slab_function_view_spatial_"
                                "energy_error_squared_impl");

                        const auto grad_source =
                            lambda_delta.gradient_on_cell(
                                source_cell_id,
                                x_q,
                                source_geom);
                        const auto grad_tilde =
                            lambda_tilde.gradient_on_cell(
                                slab_cell,
                                x_q,
                                slab_geom);

                        coefficients::DiffusionVector<GT::dim_space_v>
                            grad_diff{};
                        for (int d = 0; d < GT::dim_space_v; ++d)
                        {
                            grad_diff[static_cast<std::size_t>(d)] =
                                grad_source[static_cast<std::size_t>(d)] -
                                grad_tilde[static_cast<std::size_t>(d)];
                        }
                        const auto M_q =
                            coefficients::evaluate_diffusion_tensor<
                                GT::dim_space_v>(
                                    coeff,
                                    x_q);
                        const double density =
                            coefficients::diffusion_energy_dot(
                                M_q,
                                grad_diff,
                                grad_diff);

                        local_contribution +=
                            checked_nonnegative_contribution(
                                density,
                                measure_factor,
                                "compute_time_slab_function_view_spatial_"
                                "energy_error_squared_impl");
                    }

                    result_accumulator.add(source_cell_id, local_contribution);
                });
            }

            const auto map_conversion_begin =
                ReconstructionYNormClock::now();
            auto result = result_accumulator.to_cellwise_squared_error();
            profile.map_conversion +=
                reconstruction_y_norm_elapsed_seconds(map_conversion_begin);

            require_nonnegative_cellwise_map(
                result.by_source_cell,
                "compute_time_slab_function_view_spatial_energy_error_"
                "squared_impl aggregation");

            profile.source_cells_touched =
                static_cast<double>(result_accumulator.touched_cell_count());
            profile.geometry_cache_hits =
                static_cast<double>(
                    source_geometry_cache.cache_hit_count() +
                    tilde_geometry_cache.cache_hit_count());
            profile.geometry_cache_misses =
                static_cast<double>(
                    source_geometry_cache.cache_miss_count() +
                    tilde_geometry_cache.cache_miss_count());
            profile.virtual_geometry_constructed =
                static_cast<double>(
                    tilde_geometry_cache.virtual_geometry_constructed_count());
            profile.copied_geometry_cache_lookups =
                static_cast<double>(
                    tilde_geometry_cache
                        .copied_geometry_cache_lookup_count());
            profile.total =
                reconstruction_y_norm_elapsed_seconds(total_begin);
            profile.record(timing);

            return result;
        }

        template<
            int QSpace,
            int QTime,
            class SourceFunctionType,
            TimeSlabFunctionView FunctionViewType,
            class CoefficientFunction>
        [[nodiscard]] CellwiseSquaredError<int>
        compute_time_slab_function_view_spatial_energy_error_squared_impl(
            const SourceFunctionType& lambda_delta,
            const FunctionViewType& lambda_tilde,
            const CoefficientFunction& coeff,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            return compute_Y_error_squared_by_source_cell_fast_view<
                QSpace,
                QTime>(
                lambda_delta,
                lambda_tilde,
                coeff,
                timing);
        }

        template<
            int QSpace,
            int QTime,
            class SourceFunctionType,
            class ReconstructedFunctionType,
            class CoefficientFunction>
        [[nodiscard]] ReconstructionYCrossTermDiagnostics
        compute_spatial_energy_error_cross_terms_impl(
            const SourceFunctionType& lambda_delta,
            const ReconstructedFunctionType& lambda_tilde,
            const CoefficientFunction& coeff)
        {
            using SourceSpaceType = typename SourceFunctionType::SpaceType;
            using GT              = typename SourceSpaceType::GT;
            using FETraits        = typename SourceSpaceType::FETraitsType;

            using SlabSpaceType   = typename ReconstructedFunctionType::SlabSpaceType;
            using SlabType        = typename SlabSpaceType::SlabType;
            using LocalSpaceType  = typename SlabType::SpaceType;

            using Tables   =
                finite_element::assembly::detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
            using Geometry =
                finite_element::geometry::CellGeometry<LocalSpaceType, GT::dim_space_v>;

            ReconstructionYCrossTermDiagnostics diagnostics;

            const auto& slab_space = lambda_tilde.slab_space();
            const auto& source_space = lambda_delta.fespace();

            diagnostics.error_squared.by_source_cell.reserve(
                source_space.active_cells().size());
            diagnostics.source_norm_by_source_cell.reserve(
                source_space.active_cells().size());
            diagnostics.slab_norm_by_source_cell.reserve(
                source_space.active_cells().size());
            diagnostics.cross_term_by_source_cell.reserve(
                source_space.active_cells().size());
            diagnostics.source_slice_count.reserve(
                source_space.active_cells().size());
            diagnostics.slab_ids_by_source_cell.reserve(
                source_space.active_cells().size());

            finite_element::detail::CellGeometryCache<SourceSpaceType>
                source_geometry_cache(source_space);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab          = slab_space.slab(slab_id);
                const auto& slab_fespace  = slab.fespace_ref();
                const auto& slab_function = lambda_tilde.slab_function(slab_id);
                finite_element::detail::CellGeometryCache<LocalSpaceType>
                    slab_geometry_cache(slab_fespace);

                for (const int slab_cell_id : slab.active_cells())
                {
                    const int source_cell_id = slab.source_cell_id(slab_cell_id);
                    const auto& source_geom =
                        source_geometry_cache.geometry(source_cell_id);
                    const auto& geom =
                        slab_geometry_cache.geometry(slab_cell_id);
                    const double jac = Geometry::jacobian_measure(geom);

                    double local_source_norm = 0.0;
                    double local_slab_norm = 0.0;
                    double local_cross_term = 0.0;

                    for (int q = 0; q < Tables::n_cell_q; ++q)
                    {
                        const auto& xi_q = Tables::cell_rule.points[q];
                        const double w_q = Tables::cell_rule.weights[q];
                        const auto x_q   = Geometry::map_to_physical(geom, xi_q);
                        const double measure_factor =
                            checked_positive_measure_factor(
                                jac,
                                w_q,
                                "compute_spatial_energy_error_cross_terms_impl");

                        const auto grad_source =
                            lambda_delta.gradient_on_cell(
                                source_cell_id,
                                x_q,
                                source_geom);
                        const auto grad_tilde =
                            slab_function.gradient_on_cell(
                                slab_cell_id,
                                x_q,
                                geom);
                        coefficients::DiffusionVector<GT::dim_space_v>
                            grad_source_spatial{};
                        coefficients::DiffusionVector<GT::dim_space_v>
                            grad_tilde_spatial{};
                        for (int d = 0; d < GT::dim_space_v; ++d)
                        {
                            grad_source_spatial[
                                static_cast<std::size_t>(d)] =
                                grad_source[static_cast<std::size_t>(d)];
                            grad_tilde_spatial[
                                static_cast<std::size_t>(d)] =
                                grad_tilde[static_cast<std::size_t>(d)];
                        }
                        const auto M_q =
                            coefficients::evaluate_diffusion_tensor<
                                GT::dim_space_v>(
                                    coeff,
                                    x_q);

                        const double source_density =
                            coefficients::diffusion_energy_dot(
                                M_q,
                                grad_source_spatial,
                                grad_source_spatial);
                        const double slab_density =
                            coefficients::diffusion_energy_dot(
                                M_q,
                                grad_tilde_spatial,
                                grad_tilde_spatial);
                        const double cross_density =
                            coefficients::diffusion_energy_dot(
                                M_q,
                                grad_source_spatial,
                                grad_tilde_spatial);

                        local_source_norm +=
                            checked_nonnegative_contribution(
                                source_density,
                                measure_factor,
                                "compute_spatial_energy_error_cross_terms_impl source");
                        local_slab_norm +=
                            checked_nonnegative_contribution(
                                slab_density,
                                measure_factor,
                                "compute_spatial_energy_error_cross_terms_impl slab");

                        const double local_cross =
                            cross_density * measure_factor;
                        if (!std::isfinite(local_cross))
                        {
                            throw std::runtime_error(
                                "compute_spatial_energy_error_cross_terms_impl: "
                                "cross contribution is non-finite.");
                        }
                        local_cross_term += local_cross;
                    }

                    detail::add_to_map(
                        diagnostics.source_norm_by_source_cell,
                        source_cell_id,
                        local_source_norm);
                    detail::add_to_map(
                        diagnostics.slab_norm_by_source_cell,
                        source_cell_id,
                        local_slab_norm);
                    detail::add_to_map(
                        diagnostics.cross_term_by_source_cell,
                        source_cell_id,
                        local_cross_term);
                    ++diagnostics.source_slice_count[source_cell_id];

                    auto& slab_ids =
                        diagnostics.slab_ids_by_source_cell[source_cell_id];
                    if (std::find(slab_ids.begin(), slab_ids.end(), slab_id) ==
                        slab_ids.end())
                    {
                        slab_ids.push_back(slab_id);
                    }
                }
            }

            for (const auto& [source_cell_id, source_norm] :
                 diagnostics.source_norm_by_source_cell)
            {
                const double slab_norm =
                    diagnostics.slab_norm_by_source_cell.at(source_cell_id);
                const double cross_term =
                    diagnostics.cross_term_by_source_cell.at(source_cell_id);
                const double error =
                    reconstruction_error_from_cross_terms(
                        source_norm,
                        slab_norm,
                        cross_term);
                diagnostics.error_squared.by_source_cell.emplace(
                    source_cell_id,
                    error);
            }

            require_nonnegative_cellwise_map(
                diagnostics.error_squared.by_source_cell,
                "compute_spatial_energy_error_cross_terms_impl aggregation");

            for (auto& [source_cell_id, slab_ids] :
                 diagnostics.slab_ids_by_source_cell)
            {
                static_cast<void>(source_cell_id);
                std::sort(slab_ids.begin(), slab_ids.end());
            }

            return diagnostics;
        }

        template<
            int QSpace,
            int QTime,
            class SourceFunctionType,
            class ReconstructedFunctionType>
        [[nodiscard]] CellwiseSquaredError<int>
        compute_l2_error_squared_impl(
            const SourceFunctionType& lambda_delta,
            const ReconstructedFunctionType& lambda_tilde)
        {
            using SourceSpaceType = typename SourceFunctionType::SpaceType;
            using GT              = typename SourceSpaceType::GT;
            using FETraits        = typename SourceSpaceType::FETraitsType;

            using SlabSpaceType   = typename ReconstructedFunctionType::SlabSpaceType;
            using SlabType        = typename SlabSpaceType::SlabType;
            using LocalSpaceType  = typename SlabType::SpaceType;

            using Tables   =
                finite_element::assembly::detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
            using Geometry =
                finite_element::geometry::CellGeometry<LocalSpaceType, GT::dim_space_v>;

            CellwiseSquaredError<int> result;

            const auto& slab_space = lambda_tilde.slab_space();
            const auto& source_space = lambda_delta.fespace();

            result.by_source_cell.reserve(source_space.active_cells().size());

            finite_element::detail::CellGeometryCache<SourceSpaceType> source_geometry_cache(source_space);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab          = slab_space.slab(slab_id);
                const auto& slab_fespace  = slab.fespace_ref();
                const auto& slab_function = lambda_tilde.slab_function(slab_id);
                finite_element::detail::CellGeometryCache<LocalSpaceType> slab_geometry_cache(slab_fespace);

                for (const int slab_cell_id : slab.active_cells())
                {
                    const int source_cell_id = slab.source_cell_id(slab_cell_id);
                    const auto& source_geom  = source_geometry_cache.geometry(source_cell_id);
                    const auto& geom         = slab_geometry_cache.geometry(slab_cell_id);
                    const double jac         = Geometry::jacobian_measure(geom);

                    double local_contribution = 0.0;

                    for (int q = 0; q < Tables::n_cell_q; ++q)
                    {
                        const auto& xi_q = Tables::cell_rule.points[q];
                        const double w_q = Tables::cell_rule.weights[q];
                        const auto x_q   = Geometry::map_to_physical(geom, xi_q);
                        const double measure_factor =
                            checked_positive_measure_factor(
                                jac,
                                w_q,
                                "compute_l2_error_squared_impl");

                        const double value_source =
                            lambda_delta.value_on_cell(source_cell_id, x_q, source_geom);
                        const double value_tilde =
                            slab_function.value_on_cell(slab_cell_id, x_q, geom);

                        const double diff = value_source - value_tilde;

                        local_contribution +=
                            checked_nonnegative_contribution(
                                diff * diff,
                                measure_factor,
                                "compute_l2_error_squared_impl");
                    }

                    add_to_map(result.by_source_cell, source_cell_id, local_contribution);
                }
            }

            require_nonnegative_cellwise_map(
                result.by_source_cell,
                "compute_l2_error_squared_impl aggregation");

            return result;
        }

        [[nodiscard]] inline bool reconstruction_y_trace_requested()
        {
            const char* value = std::getenv("ADAPPFEM_RECONSTRUCTION_Y_TRACE");
            if (value == nullptr || value[0] == '\0')
                return false;

            const std::string_view text(value);
            return text != "0" && text != "false" && text != "off";
        }

        [[nodiscard]] inline bool reconstruction_y_cross_trace_requested()
        {
            const char* value =
                std::getenv("ADAPPFEM_RECONSTRUCTION_Y_CROSS_TRACE");
            if (value == nullptr || value[0] == '\0')
                return false;

            const std::string_view text(value);
            return text != "0" && text != "false" && text != "off";
        }

        [[nodiscard]] inline int reconstruction_y_trace_limit()
        {
            const char* value =
                std::getenv("ADAPPFEM_RECONSTRUCTION_Y_TRACE_LIMIT");
            if (value == nullptr || value[0] == '\0')
                return 64;

            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value)
                return 64;
            if (parsed <= 0)
                return 0;
            if (parsed > 1000000)
                return 1000000;

            return static_cast<int>(parsed);
        }

        [[nodiscard]] inline int reconstruction_y_cross_trace_top_k()
        {
            const char* value =
                std::getenv("ADAPPFEM_RECONSTRUCTION_Y_CROSS_TRACE_TOP_K");
            if (value == nullptr || value[0] == '\0')
                return 10;

            char* end = nullptr;
            const long parsed = std::strtol(value, &end, 10);
            if (end == value)
                return 10;
            if (parsed <= 0)
                return 0;
            if (parsed > 1000000)
                return 1000000;

            return static_cast<int>(parsed);
        }

        template<int Dim, class VectorType>
        void write_fixed_vector(std::ostream& out, const VectorType& vector)
        {
            out << '(';
            for (int d = 0; d < Dim; ++d)
            {
                if (d > 0)
                    out << ',';
                out << vector[static_cast<std::size_t>(d)];
            }
            out << ')';
        }

        template<int DimSpace>
        void write_diffusion_tensor(
            std::ostream& out,
            const coefficients::DiffusionTensor<DimSpace>& tensor)
        {
            out << '[';
            for (int i = 0; i < DimSpace; ++i)
            {
                if (i > 0)
                    out << ';';
                for (int j = 0; j < DimSpace; ++j)
                {
                    if (j > 0)
                        out << ',';
                    out << tensor[static_cast<std::size_t>(i)]
                                 [static_cast<std::size_t>(j)];
                }
            }
            out << ']';
        }

        inline void write_int_vector(
            std::ostream& out,
            const std::vector<int>& values)
        {
            out << '[';
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (i > 0)
                    out << ',';
                out << values[i];
            }
            out << ']';
        }
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class SlabSpaceType,
        class VectorType,
        class MFunction>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_Y_error_squared_by_source_cell(
        const SourceFunctionType& lambda_delta,
        const TimeSlabFunction<SlabSpaceType, VectorType>& lambda_tilde,
        const MFunction& M,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        const auto view = make_copied_time_slab_function_view(lambda_tilde);
        return detail::compute_Y_error_squared_by_source_cell_fast_view<
            QSpace,
            QTime>(
            lambda_delta,
            view,
            M,
            timing);
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType,
        class MFunction>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_Y_error_squared_by_source_cell_copied_legacy(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde,
        const MFunction& M)
    {
        return detail::compute_spatial_energy_error_squared_impl<
            QSpace,
            QTime>(
            lambda_delta,
            lambda_tilde,
            M);
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        TimeSlabFunctionView FunctionViewType,
        class MFunction>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_time_slab_function_view_Y_error_squared_by_source_cell(
        const SourceFunctionType& lambda_delta,
        const FunctionViewType& lambda_tilde,
        const MFunction& M,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        return detail::compute_Y_error_squared_by_source_cell_fast_view<
            QSpace,
            QTime>(
            lambda_delta,
            lambda_tilde,
            M,
            timing);
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType,
        class MFunction>
    [[nodiscard]] ReconstructionYCrossTermDiagnostics
    compute_Y_error_cross_terms_by_source_cell(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde,
        const MFunction& M)
    {
        return detail::compute_spatial_energy_error_cross_terms_impl<
            QSpace,
            QTime>(
            lambda_delta,
            lambda_tilde,
            M);
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType,
        class MFunction>
    void trace_Y_error_cross_terms_if_requested(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde,
        const MFunction& M,
        const CellwiseSquaredError<int>& quadrature_error,
        std::ostream& out = std::cerr)
    {
        if (!detail::reconstruction_y_cross_trace_requested())
            return;

        const auto diagnostics =
            compute_Y_error_cross_terms_by_source_cell<QSpace, QTime>(
                lambda_delta,
                lambda_tilde,
                M);

        double max_abs_cell_diff = 0.0;
        for (const auto& [source_cell_id, value] :
             quadrature_error.by_source_cell)
        {
            const auto it =
                diagnostics.error_squared.by_source_cell.find(source_cell_id);
            const double cross_value =
                it == diagnostics.error_squared.by_source_cell.end()
                    ? 0.0
                    : it->second;
            max_abs_cell_diff =
                std::max(max_abs_cell_diff, std::abs(value - cross_value));
        }

        const int top_k = detail::reconstruction_y_cross_trace_top_k();
        out
            << "ADAPPFEM_RECONSTRUCTION_Y_CROSS_TRACE begin "
            << "q_space=" << QSpace
            << " q_time=" << QTime
            << " total_quadrature_error=" << quadrature_error.total()
            << " total_cross_error=" << diagnostics.error_squared.total()
            << " total_source_norm=" << diagnostics.total_source_norm()
            << " total_slab_norm=" << diagnostics.total_slab_norm()
            << " total_cross_term=" << diagnostics.total_cross_term()
            << " total_error_from_cross_terms="
            << diagnostics.total_error_from_cross_terms()
            << " max_abs_cell_diff=" << max_abs_cell_diff
            << " top_k=" << top_k
            << '\n';

        for (const auto& cell :
             diagnostics.top_cells(static_cast<std::size_t>(top_k)))
        {
            out
                << "ADAPPFEM_RECONSTRUCTION_Y_CROSS_TRACE top_cell "
                << "source_cell=" << cell.source_cell_id
                << " error_squared=" << cell.error_squared
                << " source_norm=" << cell.source_norm
                << " slab_norm=" << cell.slab_norm
                << " cross_term=" << cell.cross_term
                << " source_slice_count=" << cell.source_slice_count
                << " slab_ids=";
            detail::write_int_vector(out, cell.slab_ids);
            out << '\n';
        }

        out << "ADAPPFEM_RECONSTRUCTION_Y_CROSS_TRACE end\n";
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType,
        class MFunction,
        class XSpaceType>
    void trace_Y_error_contributions_if_requested(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde,
        const MFunction& M,
        const XSpaceType& x_space,
        std::ostream& out = std::cerr)
    {
        if (!detail::reconstruction_y_trace_requested())
            return;

        using SourceSpaceType = typename SourceFunctionType::SpaceType;
        using GT              = typename SourceSpaceType::GT;
        using FETraits        = typename SourceSpaceType::FETraitsType;

        using SlabSpaceType   = typename ReconstructedFunctionType::SlabSpaceType;
        using SlabType        = typename SlabSpaceType::SlabType;
        using LocalSpaceType  = typename SlabType::SpaceType;

        using Tables   =
            finite_element::assembly::detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
        using Geometry =
            finite_element::geometry::CellGeometry<LocalSpaceType, GT::dim_space_v>;

        const auto& slab_space = lambda_tilde.slab_space();
        const auto& source_space = lambda_delta.fespace();

        finite_element::detail::CellGeometryCache<SourceSpaceType>
            source_geometry_cache(source_space);
        finite_element::assembly::detail::SourceActiveAncestorCache<XSpaceType>
            x_ancestor_cache(x_space);

        const int limit = detail::reconstruction_y_trace_limit();
        int emitted = 0;
        int total = 0;
        bool reported_limit = false;

        out
            << "ADAPPFEM_RECONSTRUCTION_Y_TRACE begin "
            << "q_space=" << QSpace
            << " q_time=" << QTime
            << " max_records=" << limit
            << '\n';

        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
        {
            const auto& slab          = slab_space.slab(slab_id);
            const auto& slab_fespace  = slab.fespace_ref();
            const auto& slab_function = lambda_tilde.slab_function(slab_id);
            finite_element::detail::CellGeometryCache<LocalSpaceType>
                slab_geometry_cache(slab_fespace);

            for (const int slab_cell_id : slab.active_cells())
            {
                const int source_cell_id = slab.source_cell_id(slab_cell_id);
                const int x_ancestor_cell_id =
                    finite_element::assembly::detail::
                        find_active_ancestor_cell_from_source_cell(
                            x_ancestor_cache,
                            x_space,
                            source_cell_id);

                const auto& source_geom =
                    source_geometry_cache.geometry(source_cell_id);
                const auto& geom =
                    slab_geometry_cache.geometry(slab_cell_id);
                const double jac = Geometry::jacobian_measure(geom);

                for (int q = 0; q < Tables::n_cell_q; ++q)
                {
                    const auto& xi_q = Tables::cell_rule.points[q];
                    const double w_q = Tables::cell_rule.weights[q];
                    const auto x_q   = Geometry::map_to_physical(geom, xi_q);
                    const double measure_factor =
                        detail::checked_positive_measure_factor(
                            jac,
                            w_q,
                            "trace_Y_error_contributions_if_requested");

                    const auto grad_source =
                        lambda_delta.gradient_on_cell(
                            source_cell_id,
                            x_q,
                            source_geom);
                    const auto grad_tilde =
                        slab_function.gradient_on_cell(
                            slab_cell_id,
                            x_q,
                            geom);

                    coefficients::DiffusionVector<GT::dim_space_v> grad_diff{};
                    for (int d = 0; d < GT::dim_space_v; ++d)
                    {
                        grad_diff[static_cast<std::size_t>(d)] =
                            grad_source[static_cast<std::size_t>(d)] -
                            grad_tilde[static_cast<std::size_t>(d)];
                    }

                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<
                            GT::dim_space_v>(
                                M,
                                x_q);
                    const double energy_density =
                        coefficients::diffusion_energy_dot(
                            M_q,
                            grad_diff,
                            grad_diff);
                    const double local_contribution =
                        detail::checked_nonnegative_contribution(
                            energy_density,
                            measure_factor,
                            "trace_Y_error_contributions_if_requested");

                    if (emitted < limit)
                    {
                        out
                            << "ADAPPFEM_RECONSTRUCTION_Y_TRACE "
                            << "source_cell=" << source_cell_id
                            << " slab=" << slab_id
                            << " slab_cell=" << slab_cell_id
                            << " x_ancestor_cell=" << x_ancestor_cell_id
                            << " q=" << q
                            << " x=";
                        detail::write_fixed_vector<GT::dim_v>(out, x_q);
                        out << " grad_lambda_delta=";
                        detail::write_fixed_vector<GT::dim_v>(out, grad_source);
                        out << " grad_lambda_tilde=";
                        detail::write_fixed_vector<GT::dim_v>(out, grad_tilde);
                        out << " M=";
                        detail::write_diffusion_tensor<GT::dim_space_v>(out, M_q);
                        out
                            << " energy_density=" << energy_density
                            << " measure_factor=" << measure_factor
                            << " local_contribution=" << local_contribution
                            << '\n';
                        ++emitted;
                    }
                    else if (!reported_limit)
                    {
                        out
                            << "ADAPPFEM_RECONSTRUCTION_Y_TRACE "
                            << "record limit reached; set "
                            << "ADAPPFEM_RECONSTRUCTION_Y_TRACE_LIMIT to emit more"
                            << '\n';
                        reported_limit = true;
                    }

                    ++total;
                }
            }
        }

        out
            << "ADAPPFEM_RECONSTRUCTION_Y_TRACE end "
            << "emitted=" << emitted
            << " total_records=" << total
            << '\n';
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_H10_error_squared_by_source_cell(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde)
    {
        return detail::compute_spatial_energy_error_squared_impl<QSpace, QTime>(
            lambda_delta,
            lambda_tilde,
            detail::UnitCoefficient{});
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_L2_error_squared_by_source_cell(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde)
    {
        return detail::compute_l2_error_squared_impl<QSpace, QTime>(
            lambda_delta,
            lambda_tilde);
    }
}

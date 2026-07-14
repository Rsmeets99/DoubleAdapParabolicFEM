#pragma once

#include <algorithm>
#include <chrono>
#include <vector>

#include "../detail/timing.hpp"
#include "detail/time_slab_error_indicator_detail.hpp"
#include "time_slab_cellwise_errors.hpp"

namespace finite_element::time_slabs
{
    namespace detail
    {
        template<typename CellIdType, class... MapTypes>
        [[nodiscard]] std::vector<CellIdType>
        sorted_union_cell_ids_from_maps(const MapTypes&... maps)
        {
            std::vector<CellIdType> ids;
            ids.reserve((maps.size() + ... + 0U));
            auto append = [&ids](const auto& map)
            {
                for (const auto& [cell_id, value] : map)
                {
                    static_cast<void>(value);
                    ids.push_back(cell_id);
                }
            };
            (append(maps), ...);

            std::sort(ids.begin(), ids.end());
            ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
            return ids;
        }

        template<class MapType, typename CellIdType>
        [[nodiscard]] double map_value_or_zero(
            const MapType& map,
            const CellIdType cell_id)
        {
            const auto it = map.find(cell_id);
            if (it == map.end())
                return 0.0;
            return it->second;
        }
    }

    template<typename CellIdType = int>
    [[nodiscard]] CellwiseTimeSlabEstimatorError<CellIdType>
    combine_time_slab_estimator_squared_errors(
        const CellwiseEquilibratedFluxError<CellIdType>& equilibrated_flux_error,
        const CellwiseSquaredError<CellIdType>& reconstruction_y_error)
    {
        CellwiseTimeSlabEstimatorError<CellIdType> out;
        out.equilibrated_flux_y_squared.by_source_cell =
            equilibrated_flux_error.by_source_cell_flux;
        out.reconstruction_y_squared = reconstruction_y_error;
        out.divergence_residual_squared.by_source_cell =
            equilibrated_flux_error.by_source_cell_residual;

        auto estimator_accumulator =
            detail::CellwiseActiveAccumulator<CellIdType>::from_map_keys(
                out.equilibrated_flux_y_squared.by_source_cell,
                out.reconstruction_y_squared.by_source_cell);
        estimator_accumulator.add_map(
            out.equilibrated_flux_y_squared.by_source_cell);
        estimator_accumulator.add_map(
            out.reconstruction_y_squared.by_source_cell);
        out.estimator_squared =
            estimator_accumulator.to_cellwise_squared_error();

        detail::require_nonnegative_cellwise_map(
            out.equilibrated_flux_y_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors flux");
        detail::require_nonnegative_cellwise_map(
            out.reconstruction_y_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors reconstruction");
        detail::require_nonnegative_cellwise_map(
            out.divergence_residual_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors divergence residual");
        detail::require_nonnegative_cellwise_map(
            out.estimator_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors estimator");

        return out;
    }

    template<typename CellIdType = int>
    [[nodiscard]] CellwiseTimeSlabEstimatorError<CellIdType>
    combine_time_slab_estimator_squared_errors_deterministic(
        const CellwiseEquilibratedFluxError<CellIdType>& equilibrated_flux_error,
        const CellwiseSquaredError<CellIdType>& reconstruction_y_error,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        using Clock = std::chrono::steady_clock;

        timing.add("deterministic_estimator_reductions_enabled", 1.0);
        timing.add("estimator_reduction.vector_path_used", 1.0);
        timing.add("estimator_reduction.unordered_map_path_used", 0.0);
        timing.add("estimator_reduction.thread_local_accumulators", 0.0);

        CellwiseTimeSlabEstimatorError<CellIdType> out;
        const auto flux_copy_start = Clock::now();
        out.equilibrated_flux_y_squared.by_source_cell =
            equilibrated_flux_error.by_source_cell_flux;
        const auto flux_copy_end = Clock::now();
        const auto reconstruction_copy_start = Clock::now();
        out.reconstruction_y_squared = reconstruction_y_error;
        const auto reconstruction_copy_end = Clock::now();
        const auto residual_copy_start = Clock::now();
        out.divergence_residual_squared.by_source_cell =
            equilibrated_flux_error.by_source_cell_residual;
        const auto residual_copy_end = Clock::now();

        timing.add(
            "estimator_reduction.flux_map_merge_wall",
            std::chrono::duration<double>(flux_copy_end - flux_copy_start)
                .count());
        timing.add(
            "estimator_reduction.reconstruction_map_merge_wall",
            std::chrono::duration<double>(
                reconstruction_copy_end - reconstruction_copy_start)
                .count());
        timing.add(
            "estimator_reduction.residual_map_merge_wall",
            std::chrono::duration<double>(residual_copy_end - residual_copy_start)
                .count());
        timing.add(
            "estimator_reduction.cells_with_flux_component.count",
            static_cast<double>(
                out.equilibrated_flux_y_squared.by_source_cell.size()));
        timing.add(
            "estimator_reduction.cells_with_reconstruction_component.count",
            static_cast<double>(
                out.reconstruction_y_squared.by_source_cell.size()));
        timing.add(
            "estimator_reduction.cells_with_residual_component.count",
            static_cast<double>(
                out.divergence_residual_squared.by_source_cell.size()));

        const auto ids_collect_start = Clock::now();
        std::vector<CellIdType> estimator_ids;
        estimator_ids.reserve(
            out.equilibrated_flux_y_squared.by_source_cell.size() +
            out.reconstruction_y_squared.by_source_cell.size());
        for (const auto& [cell_id, value] :
             out.equilibrated_flux_y_squared.by_source_cell)
        {
            static_cast<void>(value);
            estimator_ids.push_back(cell_id);
        }
        for (const auto& [cell_id, value] :
             out.reconstruction_y_squared.by_source_cell)
        {
            static_cast<void>(value);
            estimator_ids.push_back(cell_id);
        }
        const auto ids_collect_end = Clock::now();
        const std::size_t sorted_entries = estimator_ids.size();
        const auto ids_sort_start = Clock::now();
        std::sort(estimator_ids.begin(), estimator_ids.end());
        estimator_ids.erase(
            std::unique(estimator_ids.begin(), estimator_ids.end()),
            estimator_ids.end());
        const auto ids_sort_end = Clock::now();
        timing.add(
            "estimator_reduction.map_to_vector_conversion_wall",
            std::chrono::duration<double>(
                ids_collect_end - ids_collect_start)
                .count());
        timing.add(
            "estimator_reduction.source_cell_sort_wall",
            std::chrono::duration<double>(ids_sort_end - ids_sort_start)
                .count());
        timing.add("estimator_reduction.sort_calls.count", 1.0);
        timing.add(
            "estimator_reduction.sorted_entries.count",
            static_cast<double>(sorted_entries));
        timing.add(
            "estimator_reduction.source_cells_touched.count",
            static_cast<double>(estimator_ids.size()));

        const auto estimator_insert_start = Clock::now();
        out.estimator_squared.by_source_cell.reserve(estimator_ids.size());

        for (const CellIdType cell_id : estimator_ids)
        {
            const double value =
                detail::map_value_or_zero(
                    out.equilibrated_flux_y_squared.by_source_cell,
                    cell_id) +
                detail::map_value_or_zero(
                    out.reconstruction_y_squared.by_source_cell,
                    cell_id);
            out.estimator_squared.by_source_cell.emplace(cell_id, value);
        }
        const auto estimator_insert_end = Clock::now();
        timing.add(
            "estimator_reduction.vector_to_map_conversion_wall",
            std::chrono::duration<double>(
                estimator_insert_end - estimator_insert_start)
                .count());
        timing.add(
            "estimator_reduction.unordered_map_insertions.count",
            static_cast<double>(estimator_ids.size()));
        timing.add(
            "estimator_reduction.allocation_proxy.count",
            4.0);
        timing.add(
            "estimator_reduction.vector_capacity.count",
            static_cast<double>(sorted_entries));

        timing.add(
            "estimator_reduction.merge_wall",
            std::chrono::duration<double>(
                estimator_insert_end - flux_copy_start)
                .count());
        timing.add(
            "estimator_reduction.active_source_cells",
            static_cast<double>(estimator_ids.size()));
        timing.add(
            "estimator_reduction.max_thread_local_vector_size",
            static_cast<double>(estimator_ids.size()));

        const auto total_start = Clock::now();
        (void)out.estimator_squared.deterministic_total();
        const auto total_end = Clock::now();
        timing.add(
            "estimator_reduction.total_sum_wall",
            std::chrono::duration<double>(total_end - total_start).count());

        const auto nonnegative_start = Clock::now();
        detail::require_nonnegative_cellwise_map(
            out.equilibrated_flux_y_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors_deterministic flux");
        detail::require_nonnegative_cellwise_map(
            out.reconstruction_y_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors_deterministic reconstruction");
        detail::require_nonnegative_cellwise_map(
            out.divergence_residual_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors_deterministic divergence residual");
        detail::require_nonnegative_cellwise_map(
            out.estimator_squared.by_source_cell,
            "combine_time_slab_estimator_squared_errors_deterministic estimator");
        const auto nonnegative_end = Clock::now();
        timing.add(
            "estimator_reduction.nonnegative_check_wall",
            std::chrono::duration<double>(
                nonnegative_end - nonnegative_start)
                .count());

        return out;
    }

    inline void record_unordered_estimator_reduction_path(
        const finite_element::detail::TimingRecorder& timing)
    {
        timing.add("deterministic_estimator_reductions_enabled", 0.0);
        timing.add("estimator_reduction.vector_path_used", 0.0);
        timing.add("estimator_reduction.unordered_map_path_used", 1.0);
        timing.add("estimator_reduction.thread_local_accumulators", 0.0);
    }
}

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../detail/timing.hpp"

namespace finite_element::time_slabs
{
    struct DoerflerMarkingDiagnostics
    {
        double total = 0.0;
        double target = 0.0;
        double cutoff_value = 0.0;
        double cutoff_margin = 0.0;
        double max_indicator = 0.0;
        std::size_t cutoff_index = 0;
        std::size_t near_tie_count = 0;
    };

    template<typename CellIdType = int>
    struct CellwiseSquaredError
    {
        std::unordered_map<CellIdType, double> by_source_cell{};

        [[nodiscard]] double total() const
        {
            double sum = 0.0;
            for (const auto& [cell_id, value] : by_source_cell)
                sum += value;
            return sum;
        }

        [[nodiscard]] double deterministic_total() const
        {
            const auto entries = sorted_by_cell_id();
            return deterministic_sum_entries_(entries);
        }

        void clear()
        {
            by_source_cell.clear();
        }

        [[nodiscard]] std::vector<std::pair<CellIdType, double>>
        sorted_descending() const
        {
            std::vector<std::pair<CellIdType, double>> entries(
                by_source_cell.begin(),
                by_source_cell.end());

            std::sort(entries.begin(), entries.end(),
                    [](const auto& a, const auto& b)
                    {
                        if (a.second != b.second)
                            return a.second > b.second;

                        return a.first < b.first;
                    });

            return entries;
        }

        [[nodiscard]] std::vector<std::pair<CellIdType, double>>
        sorted_by_cell_id() const
        {
            std::vector<std::pair<CellIdType, double>> entries(
                by_source_cell.begin(),
                by_source_cell.end());

            std::sort(entries.begin(), entries.end(),
                    [](const auto& a, const auto& b)
                    {
                        return a.first < b.first;
                    });

            return entries;
        }

        [[nodiscard]] std::vector<std::pair<CellIdType, double>>
        top_k(std::size_t k) const
        {
            auto entries = sorted_descending();

            if (k < entries.size())
                entries.resize(k);

            return entries;
        }

        [[nodiscard]] std::vector<std::pair<CellIdType, double>>
        above_threshold(double threshold) const
        {
            std::vector<std::pair<CellIdType, double>> result;
            result.reserve(by_source_cell.size());

            for (const auto& [cell_id, value] : by_source_cell)
            {
                if (value >= threshold)
                    result.emplace_back(cell_id, value);
            }

            std::sort(result.begin(), result.end(),
                    [](const auto& a, const auto& b)
                    {
                        if (a.second != b.second)
                            return a.second > b.second;

                        return a.first < b.first;
                    });

            return result;
        }

        [[nodiscard]] std::vector<CellIdType>
        doerfler_marking(double theta) const
        {
            if (theta <= 0.0 || theta > 1.0)
                throw std::runtime_error(
                    "CellwiseSquaredError::doerfler_marking: theta must be in (0,1].");

            const double total_error = total();
            if (!(total_error > 0.0))
                return {};

            const double target = theta * total_error;

            auto entries = sorted_descending();

            std::vector<CellIdType> marked;
            marked.reserve(entries.size());

            double cumulative = 0.0;

            for (const auto& [cell_id, value] : entries)
            {
                marked.push_back(cell_id);
                cumulative += value;

                if (cumulative >= target)
                    break;
            }

            return marked;
        }

        [[nodiscard]] std::vector<CellIdType>
        doerfler_marking_deterministic(
            double theta,
            double near_tie_tolerance = 0.0,
            DoerflerMarkingDiagnostics* diagnostics = nullptr,
            const finite_element::detail::TimingRecorder& timing = {}) const
        {
            using Clock = std::chrono::steady_clock;
            if (theta <= 0.0 || theta > 1.0)
                throw std::runtime_error(
                    "CellwiseSquaredError::doerfler_marking_deterministic: theta must be in (0,1].");
            if (near_tie_tolerance < 0.0)
                throw std::runtime_error(
                    "CellwiseSquaredError::doerfler_marking_deterministic: near-tie tolerance must be nonnegative.");

            const auto by_id_start = Clock::now();
            const auto by_id_entries = sorted_by_cell_id();
            const auto by_id_end = Clock::now();
            timing.add(
                "doerfler_marking.by_id_sort_wall",
                std::chrono::duration<double>(by_id_end - by_id_start)
                    .count());
            timing.add(
                "doerfler_marking.input_entries.count",
                static_cast<double>(by_id_entries.size()));
            timing.add(
                "estimator_reduction.dorfler_input_build_wall",
                std::chrono::duration<double>(by_id_end - by_id_start)
                    .count());

            const auto total_start = Clock::now();
            const double total_error =
                deterministic_sum_entries_(by_id_entries);
            const auto total_end = Clock::now();
            timing.add(
                "doerfler_marking.total_sum_wall",
                std::chrono::duration<double>(total_end - total_start)
                    .count());
            DoerflerMarkingDiagnostics local_diagnostics;
            local_diagnostics.total = total_error;
            local_diagnostics.target = theta * total_error;
            local_diagnostics.cutoff_index = by_id_entries.size();

            if (!(total_error > 0.0))
            {
                if (diagnostics)
                    *diagnostics = local_diagnostics;
                return {};
            }

            const auto descending_start = Clock::now();
            auto entries = sorted_descending();
            const auto descending_end = Clock::now();
            timing.add(
                "doerfler_marking.descending_sort_wall",
                std::chrono::duration<double>(
                    descending_end - descending_start)
                    .count());
            local_diagnostics.max_indicator =
                entries.empty() ? 0.0 : entries.front().second;

            std::vector<CellIdType> marked;
            marked.reserve(entries.size());

            double cumulative = 0.0;
            double compensation = 0.0;

            const auto accumulation_start = Clock::now();
            for (std::size_t i = 0; i < entries.size(); ++i)
            {
                marked.push_back(entries[i].first);
                neumaier_add_(cumulative, compensation, entries[i].second);

                if (cumulative + compensation >= local_diagnostics.target)
                {
                    local_diagnostics.cutoff_index = i;
                    local_diagnostics.cutoff_value = entries[i].second;
                    if (i + 1U < entries.size())
                    {
                        local_diagnostics.cutoff_margin =
                            entries[i].second - entries[i + 1U].second;
                    }
                    break;
                }
            }
            const auto accumulation_end = Clock::now();
            timing.add(
                "doerfler_marking.marked_accumulation_wall",
                std::chrono::duration<double>(
                    accumulation_end - accumulation_start)
                    .count());
            timing.add(
                "doerfler_marking.marked_entries.count",
                static_cast<double>(marked.size()));

            if (local_diagnostics.cutoff_index < entries.size())
            {
                const auto near_tie_start = Clock::now();
                const double cutoff = local_diagnostics.cutoff_value;
                const double tolerance = near_tie_tolerance;
                std::size_t near_ties = 0;
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    if (i == local_diagnostics.cutoff_index)
                        continue;
                    if (std::abs(entries[i].second - cutoff) <= tolerance)
                        ++near_ties;
                }
                local_diagnostics.near_tie_count = near_ties;
                const auto near_tie_end = Clock::now();
                timing.add(
                    "doerfler_marking.near_tie_scan_wall",
                    std::chrono::duration<double>(
                        near_tie_end - near_tie_start)
                        .count());
            }
            else
            {
                timing.add("doerfler_marking.near_tie_scan_wall", 0.0);
            }

            if (diagnostics)
                *diagnostics = local_diagnostics;

            return marked;
        }

        [[nodiscard]] std::vector<CellIdType>
        sorted_cell_ids() const
        {
            auto entries = sorted_descending();

            std::vector<CellIdType> ids;
            ids.reserve(entries.size());

            for (const auto& [cell_id, _] : entries)
                ids.push_back(cell_id);

            return ids;
        }

    private:
        static void neumaier_add_(
            double& sum,
            double& compensation,
            const double value) noexcept
        {
            const double tentative = sum + value;
            if (std::abs(sum) >= std::abs(value))
                compensation += (sum - tentative) + value;
            else
                compensation += (value - tentative) + sum;
            sum = tentative;
        }

        [[nodiscard]] static double deterministic_sum_entries_(
            const std::vector<std::pair<CellIdType, double>>& entries) noexcept
        {
            double sum = 0.0;
            double compensation = 0.0;
            for (const auto& [cell_id, value] : entries)
            {
                static_cast<void>(cell_id);
                neumaier_add_(sum, compensation, value);
            }
            return sum + compensation;
        }
    };

    template<typename CellIdType = int>
    struct CellwiseEquilibratedFluxError
    {
        std::unordered_map<CellIdType, double> by_source_cell_flux{};
        std::unordered_map<CellIdType, double> by_source_cell_residual{};

        [[nodiscard]] double total_flux() const
        {
            double sum = 0.0;
            for (const auto& [cell_id, value] : by_source_cell_flux)
                sum += value;
            return sum;
        }

        [[nodiscard]] double total_residual() const
        {
            double sum = 0.0;
            for (const auto& [cell_id, value] : by_source_cell_residual)
                sum += value;
            return sum;
        }

        [[nodiscard]] double deterministic_total_flux() const
        {
            return equilibrated_flux_part().deterministic_total();
        }

        [[nodiscard]] double deterministic_total_residual() const
        {
            return divergence_residual_part().deterministic_total();
        }

        [[nodiscard]] double deterministic_total() const
        {
            return deterministic_total_flux() + deterministic_total_residual();
        }

        [[nodiscard]] double total() const
        {
            return total_flux() + total_residual();
        }

        void clear()
        {
            by_source_cell_flux.clear();
            by_source_cell_residual.clear();
        }

        [[nodiscard]] CellwiseSquaredError<CellIdType> equilibrated_flux_part() const
        {
            CellwiseSquaredError<CellIdType> out;
            out.by_source_cell = by_source_cell_flux;
            return out;
        }

        [[nodiscard]] CellwiseSquaredError<CellIdType> divergence_residual_part() const
        {
            CellwiseSquaredError<CellIdType> out;
            out.by_source_cell = by_source_cell_residual;
            return out;
        }

        [[nodiscard]] CellwiseSquaredError<CellIdType> combined() const
        {
            CellwiseSquaredError<CellIdType> total_error;
            total_error.by_source_cell = by_source_cell_flux;

            for (const auto& [cell_id, value] : by_source_cell_residual)
            {
                auto it = total_error.by_source_cell.find(cell_id);
                if (it == total_error.by_source_cell.end())
                    total_error.by_source_cell.emplace(cell_id, value);
                else
                    it->second += value;
            }

            return total_error;
        }

        [[nodiscard]] std::vector<std::pair<CellIdType, double>>
        sorted_descending() const
        {
            return combined().sorted_descending();
        }

        [[nodiscard]] std::vector<CellIdType>
        doerfler_marking(double theta) const
        {
            return combined().doerfler_marking(theta);
        }

        [[nodiscard]] std::vector<CellIdType>
        doerfler_marking_deterministic(
            double theta,
            double near_tie_tolerance = 0.0,
            DoerflerMarkingDiagnostics* diagnostics = nullptr,
            const finite_element::detail::TimingRecorder& timing = {}) const
        {
            return combined().doerfler_marking_deterministic(
                theta,
                near_tie_tolerance,
                diagnostics,
                timing);
        }
    };

    template<typename CellIdType = int>
    struct CellwiseTimeSlabEstimatorError
    {
        CellwiseSquaredError<CellIdType> equilibrated_flux_y_squared{};
        CellwiseSquaredError<CellIdType> reconstruction_y_squared{};
        CellwiseSquaredError<CellIdType> divergence_residual_squared{};
        CellwiseSquaredError<CellIdType> estimator_squared{};

        [[nodiscard]] double total() const
        {
            return estimator_squared.total();
        }

        [[nodiscard]] double deterministic_total() const
        {
            return estimator_squared.deterministic_total();
        }

        [[nodiscard]] double y_upper_bound_total() const
        {
            return equilibrated_flux_y_squared.total() + reconstruction_y_squared.total();
        }

        [[nodiscard]] double deterministic_y_upper_bound_total() const
        {
            return equilibrated_flux_y_squared.deterministic_total() +
                   reconstruction_y_squared.deterministic_total();
        }

        [[nodiscard]] double divergence_residual_total() const
        {
            return divergence_residual_squared.total();
        }

        [[nodiscard]] double deterministic_divergence_residual_total() const
        {
            return divergence_residual_squared.deterministic_total();
        }

        [[nodiscard]] std::vector<std::pair<CellIdType, double>>
        sorted_descending() const
        {
            return estimator_squared.sorted_descending();
        }

        [[nodiscard]] std::vector<CellIdType>
        doerfler_marking(double theta) const
        {
            return estimator_squared.doerfler_marking(theta);
        }

        [[nodiscard]] std::vector<CellIdType>
        doerfler_marking_deterministic(
            double theta,
            double near_tie_tolerance = 0.0,
            DoerflerMarkingDiagnostics* diagnostics = nullptr,
            const finite_element::detail::TimingRecorder& timing = {}) const
        {
            return estimator_squared.doerfler_marking_deterministic(
                theta,
                near_tie_tolerance,
                diagnostics,
                timing);
        }
    };

    namespace detail
    {
        template<typename CellIdType = int>
        class CellwiseActiveAccumulator
        {
        public:
            static_assert(
                std::is_integral_v<CellIdType>,
                "CellwiseActiveAccumulator requires an integral cell id type.");

            CellwiseActiveAccumulator() = default;

            template<class CellRange>
            explicit CellwiseActiveAccumulator(const CellRange& active_cells)
            {
                reset(active_cells);
            }

            template<class CellRange>
            void reset(const CellRange& active_cells)
            {
                active_cell_ids_.clear();
                values_.clear();
                touched_.clear();
                index_by_cell_id_.clear();
                total_ = 0.0;
                touched_cell_count_ = 0;

                CellIdType max_cell_id{};
                bool have_cells = false;
                for (const auto cell_id : active_cells)
                {
                    check_nonnegative_(cell_id);
                    if (!have_cells || cell_id > max_cell_id)
                        max_cell_id = cell_id;
                    have_cells = true;
                }

                if (have_cells)
                {
                    index_by_cell_id_.assign(
                        checked_index_(max_cell_id) + 1U,
                        invalid_index());
                }

                for (const auto cell_id : active_cells)
                    add_active_cell(cell_id);
            }

            template<class... MapTypes>
            [[nodiscard]] static CellwiseActiveAccumulator from_map_keys(
                const MapTypes&... maps)
            {
                CellwiseActiveAccumulator accumulator;
                (accumulator.add_map_keys_(maps), ...);
                return accumulator;
            }

            void add_active_cell(const CellIdType cell_id)
            {
                check_nonnegative_(cell_id);
                const std::size_t direct_index = checked_index_(cell_id);
                if (direct_index >= index_by_cell_id_.size())
                {
                    index_by_cell_id_.resize(
                        direct_index + 1U,
                        invalid_index());
                }

                if (index_by_cell_id_[direct_index] != invalid_index())
                    return;

                index_by_cell_id_[direct_index] = active_cell_ids_.size();
                active_cell_ids_.push_back(cell_id);
                values_.push_back(0.0);
                touched_.push_back(0U);
            }

            [[nodiscard]] bool contains(const CellIdType cell_id) const
            {
                if (!is_nonnegative_(cell_id))
                    return false;
                const std::size_t direct_index =
                    static_cast<std::size_t>(cell_id);
                return direct_index < index_by_cell_id_.size() &&
                       index_by_cell_id_[direct_index] != invalid_index();
            }

            [[nodiscard]] std::size_t compact_index(
                const CellIdType cell_id) const
            {
                if (!contains(cell_id))
                    throw std::runtime_error(
                        "CellwiseActiveAccumulator::compact_index: source "
                        "cell id is not active.");
                return index_by_cell_id_[static_cast<std::size_t>(cell_id)];
            }

            void add(const CellIdType cell_id, const double value)
            {
                const std::size_t index = compact_index(cell_id);
                values_[index] += value;
                total_ += value;
                if (touched_[index] == 0U)
                {
                    touched_[index] = 1U;
                    ++touched_cell_count_;
                }
            }

            template<class MapType>
            void add_map(const MapType& map)
            {
                for (const auto& [cell_id, value] : map)
                    add(cell_id, value);
            }

            void add_accumulator(const CellwiseActiveAccumulator& other)
            {
                for (std::size_t i = 0; i < other.values_.size(); ++i)
                {
                    if (other.touched_[i] == 0U)
                        continue;
                    add(other.active_cell_ids_[i], other.values_[i]);
                }
            }

            [[nodiscard]] double total() const noexcept
            {
                return total_;
            }

            [[nodiscard]] std::size_t active_cell_count() const noexcept
            {
                return active_cell_ids_.size();
            }

            [[nodiscard]] std::size_t touched_cell_count() const noexcept
            {
                return touched_cell_count_;
            }

            [[nodiscard]] CellwiseSquaredError<CellIdType>
            to_cellwise_squared_error(
                const bool include_untouched_cells = false) const
            {
                CellwiseSquaredError<CellIdType> out;
                out.by_source_cell.reserve(active_cell_ids_.size());

                for (std::size_t i = 0; i < active_cell_ids_.size(); ++i)
                {
                    if (!include_untouched_cells && touched_[i] == 0U)
                        continue;
                    out.by_source_cell.emplace(active_cell_ids_[i], values_[i]);
                }

                return out;
            }

        private:
            [[nodiscard]] static constexpr std::size_t invalid_index() noexcept
            {
                return std::numeric_limits<std::size_t>::max();
            }

            [[nodiscard]] static bool is_nonnegative_(
                const CellIdType cell_id) noexcept
            {
                if constexpr (std::is_signed_v<CellIdType>)
                    return cell_id >= 0;
                else
                    return true;
            }

            static void check_nonnegative_(const CellIdType cell_id)
            {
                if (!is_nonnegative_(cell_id))
                    throw std::runtime_error(
                        "CellwiseActiveAccumulator: negative source cell id.");
            }

            [[nodiscard]] static std::size_t checked_index_(
                const CellIdType cell_id)
            {
                check_nonnegative_(cell_id);
                return static_cast<std::size_t>(cell_id);
            }

            template<class MapType>
            void add_map_keys_(const MapType& map)
            {
                for (const auto& [cell_id, value] : map)
                {
                    static_cast<void>(value);
                    add_active_cell(cell_id);
                }
            }

            std::vector<CellIdType> active_cell_ids_{};
            std::vector<double> values_{};
            std::vector<unsigned char> touched_{};
            std::vector<std::size_t> index_by_cell_id_{};
            double total_ = 0.0;
            std::size_t touched_cell_count_ = 0;
        };

        template<typename MapType>
        void add_to_map(MapType& map, int cell_id, double value)
        {
            auto it = map.find(cell_id);
            if (it == map.end())
                map.emplace(cell_id, value);
            else
                it->second += value;
        }

        template<typename CellIdType>
        [[nodiscard]] CellwiseSquaredError<CellIdType>
        sum_cellwise_squared_errors(
            const CellwiseSquaredError<CellIdType>& a,
            const CellwiseSquaredError<CellIdType>& b)
        {
            CellwiseSquaredError<CellIdType> out;
            out.by_source_cell = a.by_source_cell;

            for (const auto& [cell_id, value] : b.by_source_cell)
                add_to_map(out.by_source_cell, cell_id, value);

            return out;
        }

        template<typename CellIdType>
        [[nodiscard]] CellwiseSquaredError<CellIdType>
        scale_cellwise_squared_error(
            const CellwiseSquaredError<CellIdType>& error,
            double factor)
        {
            CellwiseSquaredError<CellIdType> out;
            out.by_source_cell.reserve(error.by_source_cell.size());

            for (const auto& [cell_id, value] : error.by_source_cell)
                out.by_source_cell.emplace(cell_id, factor * value);

            return out;
        }
    }
}

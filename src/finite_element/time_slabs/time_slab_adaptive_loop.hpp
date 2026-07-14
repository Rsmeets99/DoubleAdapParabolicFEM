#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "../fespace/functions.hpp"
#include "../assembly/detail/assembly_diagnostics.hpp"
#include "../detail/space_time_capabilities.hpp"
#include "../detail/timing.hpp"
#include "../system/solve_main_system.hpp"

#include "time_slab_adaptive_refinement.hpp"
#include "time_slab_equilibrated_flux_reconstruction.hpp"
#include "time_slab_error_indicators.hpp"
#include "time_slab_estimator_input_checks.hpp"
#include "time_slab_reconstruction.hpp"

namespace finite_element::time_slabs
{
    namespace detail
    {
        [[nodiscard]] inline std::uint64_t mix_hash_u64(
            std::uint64_t seed,
            std::uint64_t value) noexcept
        {
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
            value = value ^ (value >> 31U);
            return seed ^ (value + 0x9e3779b97f4a7c15ULL +
                           (seed << 6U) + (seed >> 2U));
        }

        [[nodiscard]] inline std::uint64_t hash_double_bits(
            const double value) noexcept
        {
            std::uint64_t bits = 0;
            std::memcpy(&bits, &value, sizeof(double));
            return bits;
        }

        [[nodiscard]] inline std::uint64_t hash_cell_id_vector(
            std::vector<int> ids)
        {
            std::sort(ids.begin(), ids.end());
            std::uint64_t hash = 1469598103934665603ULL;
            for (const int id : ids)
                hash = mix_hash_u64(hash, static_cast<std::uint64_t>(id));
            hash = mix_hash_u64(hash, ids.size());
            return hash;
        }

        [[nodiscard]] inline std::uint64_t hash_cellwise_squared_error(
            const CellwiseSquaredError<int>& error)
        {
            std::vector<std::pair<int, double>> entries(
                error.by_source_cell.begin(),
                error.by_source_cell.end());
            std::sort(
                entries.begin(),
                entries.end(),
                [](const auto& a, const auto& b)
                {
                    return a.first < b.first;
                });

            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto& [cell_id, value] : entries)
            {
                hash = mix_hash_u64(hash, static_cast<std::uint64_t>(cell_id));
                hash = mix_hash_u64(hash, hash_double_bits(value));
            }
            hash = mix_hash_u64(hash, entries.size());
            return hash;
        }

        template<class MapType>
        [[nodiscard]] inline std::uint64_t hash_cellwise_source_map(
            const MapType& values)
        {
            std::vector<std::pair<int, double>> entries(
                values.begin(),
                values.end());
            std::sort(
                entries.begin(),
                entries.end(),
                [](const auto& a, const auto& b)
                {
                    return a.first < b.first;
                });

            std::uint64_t hash = 1469598103934665603ULL;
            for (const auto& [cell_id, value] : entries)
            {
                hash = mix_hash_u64(hash, static_cast<std::uint64_t>(cell_id));
                hash = mix_hash_u64(hash, hash_double_bits(value));
            }
            hash = mix_hash_u64(hash, entries.size());
            return hash;
        }

        inline void add_hash_metric(
            const finite_element::detail::TimingRecorder& timing,
            std::string_view prefix,
            std::uint64_t hash);

        template<class FluxDiagnosticsType>
        inline void record_flux_diagnostics_checksums(
            const finite_element::detail::TimingRecorder& timing,
            const FluxDiagnosticsType& diagnostics)
        {
            const std::uint64_t flux_hash =
                hash_cellwise_source_map(diagnostics.by_source_cell_flux);
            const std::uint64_t residual_hash =
                hash_cellwise_source_map(diagnostics.by_source_cell_residual);

            add_hash_metric(
                timing,
                "flux_diagnostics.flux_map_checksum",
                flux_hash);
            add_hash_metric(
                timing,
                "flux_diagnostics.residual_map_checksum",
                residual_hash);
            add_hash_metric(
                timing,
                "local_error.flux_map_checksum",
                flux_hash);
            add_hash_metric(
                timing,
                "local_error.residual_map_checksum",
                residual_hash);
        }

        inline void add_hash_metric(
            const finite_element::detail::TimingRecorder& timing,
            const std::string_view prefix,
            const std::uint64_t hash)
        {
            const auto low = static_cast<std::uint32_t>(hash & 0xffffffffULL);
            const auto high = static_cast<std::uint32_t>(hash >> 32U);
            timing.add(std::string(prefix) + "_low32", static_cast<double>(low));
            timing.add(
                std::string(prefix) + "_high32",
                static_cast<double>(high));
        }

        inline void record_local_error_marking_diagnostics(
            const finite_element::detail::TimingRecorder& timing,
            const CellwiseTimeSlabEstimatorError<int>& estimator,
            const CellwiseSquaredError<int>& reconstruction_y_squared,
            const std::vector<int>& marked_source_cells,
            const double theta,
            const bool deterministic_reductions,
            const double near_tie_tolerance,
            const DoerflerMarkingDiagnostics* deterministic_diagnostics =
                nullptr)
        {
            timing.add(
                "deterministic_estimator_reductions_enabled",
                deterministic_reductions ? 1.0 : 0.0);

            const std::uint64_t estimator_hash =
                hash_cellwise_squared_error(estimator.estimator_squared);
            add_hash_metric(
                timing,
                "local_error.estimator_map_checksum",
                estimator_hash);
            add_hash_metric(timing, "estimator_map_checksum", estimator_hash);

            const std::uint64_t reconstruction_hash =
                hash_cellwise_squared_error(reconstruction_y_squared);
            add_hash_metric(
                timing,
                "local_error.reconstruction_map_checksum",
                reconstruction_hash);

            const std::uint64_t marked_hash =
                hash_cell_id_vector(marked_source_cells);
            add_hash_metric(
                timing,
                "local_error.marked_y_set_checksum",
                marked_hash);
            add_hash_metric(timing, "y_marked_set_checksum", marked_hash);
            timing.add(
                "local_error.marked_y_count",
                static_cast<double>(marked_source_cells.size()));
            timing.add(
                "y_marked_count",
                static_cast<double>(marked_source_cells.size()));

            DoerflerMarkingDiagnostics diagnostics;
            if (deterministic_diagnostics)
            {
                diagnostics = *deterministic_diagnostics;
            }
            else
            {
                const auto entries =
                    estimator.estimator_squared.sorted_descending();
                const double total = estimator.estimator_squared.total();
                const double target = theta * total;
                double cumulative = 0.0;
                double threshold = 0.0;
                double margin = 0.0;
                double max_indicator =
                    entries.empty() ? 0.0 : entries.front().second;
                std::size_t cutoff_index = entries.size();
                for (std::size_t i = 0; i < entries.size(); ++i)
                {
                    cumulative += entries[i].second;
                    if (cumulative >= target)
                    {
                        cutoff_index = i;
                        threshold = entries[i].second;
                        if (i + 1U < entries.size())
                            margin =
                                entries[i].second - entries[i + 1U].second;
                        break;
                    }
                }
                diagnostics.total = total;
                diagnostics.target = target;
                diagnostics.cutoff_value = threshold;
                diagnostics.cutoff_margin = margin;
                diagnostics.max_indicator = max_indicator;
                diagnostics.cutoff_index = cutoff_index;
                if (cutoff_index < entries.size())
                {
                    const double cutoff = threshold;
                    for (std::size_t i = 0; i < entries.size(); ++i)
                    {
                        if (i == cutoff_index)
                            continue;
                        if (std::abs(entries[i].second - cutoff) <=
                            near_tie_tolerance)
                        {
                            ++diagnostics.near_tie_count;
                        }
                    }
                }
            }

            timing.add(
                "local_error.max_indicator_value",
                diagnostics.max_indicator);
            timing.add(
                "local_error.doerfler_threshold_value",
                diagnostics.cutoff_value);
            timing.add(
                "local_error.doerfler_cutoff_margin",
                diagnostics.cutoff_margin);
            timing.add(
                "local_error.doerfler_cutoff_index",
                diagnostics.cutoff_index ==
                        estimator.estimator_squared.by_source_cell.size()
                    ? 0.0
                    : static_cast<double>(diagnostics.cutoff_index));
            timing.add("doerfler.cutoff_value", diagnostics.cutoff_value);
            timing.add(
                "doerfler.cutoff_index",
                diagnostics.cutoff_index ==
                        estimator.estimator_squared.by_source_cell.size()
                    ? 0.0
                    : static_cast<double>(diagnostics.cutoff_index));
            timing.add("doerfler.cutoff_margin", diagnostics.cutoff_margin);
            timing.add(
                "doerfler.near_tie_count",
                static_cast<double>(diagnostics.near_tie_count));
            timing.add(
                "doerfler.near_tie_tolerance",
                near_tie_tolerance);
        }

        template<int DimSpace, class Backend, class XSpaceType, class YSpaceType>
        struct AdaptiveFluxReconstructionSelector
        {
            static_assert(
                DimSpace == 1 || DimSpace == 2,
                "Adaptive test-space flux reconstruction supports only dim_space = 1 or 2.");
        };

        template<class Backend, class XSpaceType, class YSpaceType>
        struct AdaptiveFluxReconstructionSelector<1, Backend, XSpaceType, YSpaceType>
        {
            using type =
                TimeSlabEquilibratedFluxReconstruction1plus1d<
                    Backend,
                    XSpaceType,
                    YSpaceType>;
        };

        template<class Backend, class XSpaceType, class YSpaceType>
        struct AdaptiveFluxReconstructionSelector<2, Backend, XSpaceType, YSpaceType>
        {
            using type =
                TimeSlabEquilibratedFluxReconstruction2plus1d<
                    Backend,
                    XSpaceType,
                    YSpaceType>;
        };

        template<class Backend, class XSpaceType, class YSpaceType>
        using AdaptiveFluxReconstructionType =
            typename AdaptiveFluxReconstructionSelector<
                XSpaceType::GT::dim_space_v,
                Backend,
                XSpaceType,
                YSpaceType>::type;

        [[nodiscard]] inline double interval_union_seconds(
            std::vector<std::pair<double, double>> intervals)
        {
            intervals.erase(
                std::remove_if(
                    intervals.begin(),
                    intervals.end(),
                    [](const auto& interval)
                    {
                        return interval.second <= interval.first;
                    }),
                intervals.end());

            if (intervals.empty())
                return 0.0;

            std::sort(intervals.begin(), intervals.end());
            double begin = intervals.front().first;
            double end = intervals.front().second;
            double total = 0.0;

            for (std::size_t k = 1; k < intervals.size(); ++k)
            {
                const auto& interval = intervals[k];
                if (interval.first <= end)
                {
                    end = std::max(end, interval.second);
                    continue;
                }

                total += end - begin;
                begin = interval.first;
                end = interval.second;
            }

            total += end - begin;
            return total;
        }

        template<
            int QSpace,
            int QTime,
            class Backend,
            class XSpaceType,
            class YSpaceType,
            class ReconstructedFunctionType,
            class XFunctionType,
            class EllFunction,
            class MFunction>
        [[nodiscard]] CellwiseEquilibratedFluxError<int>
        compute_adaptive_flux_diagnostics(
            const AdaptiveFluxReconstructionType<Backend, XSpaceType, YSpaceType>& reconstruction,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            using GT = typename XSpaceType::GT;

            static_assert(
                XSpaceType::GT::dim_space_v == YSpaceType::GT::dim_space_v &&
                XSpaceType::GT::dim_time_v == YSpaceType::GT::dim_time_v,
                "Adaptive flux diagnostics require matching X/Y geometry dimensions.");

            if constexpr (GT::dim_space_v == 1)
            {
                return compute_equilibrated_flux_error_squared_by_source_cell_1plus1d<
                    QSpace,
                    QTime>(
                        reconstruction,
                        lambda_tilde,
                        u_delta,
                        ell,
                        M);
            }
            else if constexpr (GT::dim_space_v == 2)
            {
                return compute_equilibrated_flux_error_squared_by_source_cell_2plus1d<
                    QSpace,
                    QTime>(
                        reconstruction,
                        lambda_tilde,
                        u_delta,
                        ell,
                        M,
                        timing);
            }
            else
            {
                finite_element::detail::
                    require_supported_time_slab_estimator_capability<GT>();
            }
        }
    }

    template<class Backend, class XSpaceType, class YSpaceType>
    struct AdaptiveTestSpaceIterationParameters
    {
        double doerfler_theta = 0.5;
        double zero_tol = 1.0e-15;
        double divergence_residual_l2_tolerance = 1.0e-8;
        bool check_divergence_residual = true;
        bool use_adaptive_initial_guess = false;
        bool solve_main_system_correction = false;
        bool fused_error_and_flux_diagnostics = true;
        bool local_error_reuse_patch_solve_workspace = true;
        bool deterministic_estimator_reductions = true;
        double doerfler_near_tie_tolerance = 0.0;
        int local_error_patch_tile_size = 0;
        int local_error_cell_chunk_size = 0;
        int local_error_max_threads = 0;
        double local_error_memory_budget_mb = 0.0;
        std::string local_error_worker_context_mode = "persistent";
        std::string local_error_context_storage = "shared_immutable";
        std::string local_error_state_index_mode = "flat";
        std::string local_error_cell_state_cache_mode = "off";
        double local_error_cell_state_cache_budget_mb = 1024.0;
        std::string local_error_cell_state_representation = "compact_split";
        std::string local_error_flux_diagnostics_mode = "auto";
        std::string local_error_patch_solver = "current_dense";
        bool local_error_coefficient_fast_path = true;
        bool local_error_compact_state_shadow = false;
        std::string shared_context_validation = "off";
        std::string slab_reconstruction_operator_mode = "auto";
        int main_assembly_max_threads = 4;
        int slab_reconstruction_max_threads = 4;
        double main_assembly_memory_budget_mb = 0.0;
        double slab_reconstruction_memory_budget_mb = 0.0;
        int main_two_pass_numeric_fill_max_threads = 0;
        double main_two_pass_numeric_fill_memory_budget_mb = 0.0;
        finite_element::assembly::TwoPassFullSaddleAssemblyCache2D<
            Backend,
            XSpaceType,
            YSpaceType>* main_two_pass_assembly_cache = nullptr;
        TimeSlabBackend time_slab_backend = TimeSlabBackend::CopiedMesh;
        bool allow_copied_time_slab_estimator_fallback = true;
        bool virtual_backend_diagnostics = false;
        finite_element::system::MainSystemExportOptions
            main_system_export{};
        finite_element::detail::TimingRecorder timing{};
    };

    template<class Backend, class XSpaceType, class YSpaceType>
    struct AdaptiveTestSpaceIterationState
    {
        using Vector = typename Backend::Vector;
        using GT = typename YSpaceType::GT;
        using FETraits = typename YSpaceType::FETraitsType;
        using YFunction = finite_element::Function<YSpaceType, Vector>;
        using XFunction = finite_element::Function<XSpaceType, Vector>;
        using ReconstructionType =
            TimeSlabReconstruction<Backend, XSpaceType, YSpaceType>;
        using FluxReconstructionType =
            detail::AdaptiveFluxReconstructionType<Backend, XSpaceType, YSpaceType>;

        YFunction lambda_delta;
        XFunction u_delta;
        std::optional<ReconstructionType> slab_reconstruction{};
        std::optional<FluxReconstructionType> flux_reconstruction{};
        CellwiseEquilibratedFluxError<int> flux_diagnostics{};
        CellwiseSquaredError<int> reconstruction_y_squared{};
        CellwiseTimeSlabEstimatorError<int> estimator{};
        std::vector<int> marked_source_cells{};
        double divergence_residual_l2 = 0.0;
        la::concepts::SolverDiagnostics main_solve_diagnostics{};
        double main_solve_setup_seconds = 0.0;
        double main_solve_solve_seconds = 0.0;
        int source_mesh_mutation_count = 0;

        AdaptiveTestSpaceIterationState(
            const YSpaceType& y_space,
            const XSpaceType& x_space)
            : lambda_delta(y_space),
              u_delta(x_space),
              source_y_space_(&y_space),
              source_x_space_(&x_space)
        {}

        [[nodiscard]] const YSpaceType& source_y_space() const
        {
            if (source_y_space_ == nullptr)
                throw std::runtime_error(
                    "AdaptiveTestSpaceIterationState: source Y-space pointer is null.");
            return *source_y_space_;
        }

        [[nodiscard]] const XSpaceType& source_x_space() const
        {
            if (source_x_space_ == nullptr)
                throw std::runtime_error(
                    "AdaptiveTestSpaceIterationState: source X-space pointer is null.");
            return *source_x_space_;
        }

        [[nodiscard]] const auto& lambda_tilde() const
        {
            ensure_reconstruction_initialized_();
            return slab_reconstruction->reconstructed_function();
        }

        [[nodiscard]] int n_slabs() const
        {
            ensure_reconstruction_initialized_();
            return slab_reconstruction->slab_space_ref().n_slabs();
        }

        [[nodiscard]] int copied_slab_cells() const
        {
            ensure_reconstruction_initialized_();
            return slab_reconstruction->slab_space_ref().n_slab_cells();
        }

        [[nodiscard]] int virtual_slab_cells() const
        {
            return 0;
        }

        [[nodiscard]] int selected_backend_true_dofs() const
        {
            ensure_reconstruction_initialized_();
            return slab_reconstruction->slab_space_ref().n_true_dofs();
        }

        [[nodiscard]] const char* time_slab_backend() const noexcept
        {
            return time_slab_backend_name(TimeSlabBackend::CopiedMesh);
        }

        [[nodiscard]] const char* estimator_backend() const noexcept
        {
            return time_slab_backend_name(TimeSlabBackend::CopiedMesh);
        }

        [[nodiscard]] const char* time_slab_backend_effective() const noexcept
        {
            return time_slab_backend();
        }

        [[nodiscard]] bool estimator_uses_copied_fallback() const noexcept
        {
            return false;
        }

        [[nodiscard]] bool virtual_overlay_constructed() const noexcept
        {
            return false;
        }

        [[nodiscard]] bool tilde_y_space_constructed() const noexcept
        {
            return false;
        }

        [[nodiscard]] int copied_fallback_component_count() const noexcept
        {
            return 0;
        }

        [[nodiscard]] const char* copied_fallback_components() const noexcept
        {
            return "";
        }

        [[nodiscard]] int estimator_fallback_copied_slab_cells() const
        {
            return 0;
        }

        [[nodiscard]] int copied_slab_cells_constructed_total() const
        {
            return copied_slab_cells();
        }

        [[nodiscard]] int n_patches() const
        {
            ensure_flux_reconstruction_initialized_();
            return flux_reconstruction->n_patches();
        }

    private:
        void ensure_reconstruction_initialized_() const
        {
            if (!slab_reconstruction.has_value())
            {
                throw std::runtime_error(
                    "AdaptiveTestSpaceIterationState: slab reconstruction not initialized.");
            }
        }

        void ensure_flux_reconstruction_initialized_() const
        {
            if (!flux_reconstruction.has_value())
            {
                throw std::runtime_error(
                    "AdaptiveTestSpaceIterationState: flux reconstruction not initialized.");
            }
        }

        const YSpaceType* source_y_space_ = nullptr;
        const XSpaceType* source_x_space_ = nullptr;
    };

    template<class Backend, class XSpaceType, class YSpaceType>
    struct AdaptiveTestSpaceSolutionSnapshot
    {
        using Vector = typename Backend::Vector;
        using YFunction = finite_element::Function<YSpaceType, Vector>;
        using XFunction = finite_element::Function<XSpaceType, Vector>;

        YSpaceType y_space;
        XSpaceType x_space;
        YFunction lambda_delta;
        XFunction u_delta;

        AdaptiveTestSpaceSolutionSnapshot(
            const YSpaceType& source_y_space,
            const XSpaceType& source_x_space,
            const YFunction& source_lambda_delta,
            const XFunction& source_u_delta)
            : y_space(source_y_space),
              x_space(source_x_space),
              lambda_delta(y_space, source_lambda_delta.true_coefficients()),
              u_delta(x_space, source_u_delta.true_coefficients())
        {}
    };

    template<typename CellIdType = int>
    struct AdaptiveTestSpaceIterationSummary
    {
        CellwiseTimeSlabEstimatorError<CellIdType> estimator{};
        std::vector<CellIdType> marked_source_cells{};
        double divergence_residual_l2 = 0.0;
        int n_slabs = 0;
        int n_patches = 0;
        int n_y_active_cells_before = 0;
        int n_y_active_cells_after = 0;
        la::concepts::SolverDiagnostics main_solve_diagnostics{};
        double main_solve_setup_seconds = 0.0;
        double main_solve_solve_seconds = 0.0;
        bool refined = false;
    };

    template<typename CellIdType = int>
    struct AdaptiveTestSpaceLoopResult
    {
        std::vector<AdaptiveTestSpaceIterationSummary<CellIdType>> iterations{};

        [[nodiscard]] int n_iterations() const noexcept
        {
            return static_cast<int>(iterations.size());
        }

        [[nodiscard]] const AdaptiveTestSpaceIterationSummary<CellIdType>& last_iteration() const
        {
            if (iterations.empty())
            {
                throw std::runtime_error(
                    "AdaptiveTestSpaceLoopResult::last_iteration: no iterations available.");
            }

            return iterations.back();
        }
    };

    template<class Backend, class XSpaceType, class YSpaceType>
    struct AdaptiveTestSpaceLoopOptions
        : AdaptiveTestSpaceIterationParameters<Backend, XSpaceType, YSpaceType>
    {
        int max_iterations = 1;
        double estimator_squared_stop = 0.0;
        bool stop_on_empty_marking = true;
    };

    template<class Backend, class XSpaceType, class YSpaceType>
    [[nodiscard]] AdaptiveTestSpaceIterationSummary<int>
    make_adaptive_test_space_iteration_summary(
        const AdaptiveTestSpaceIterationState<Backend, XSpaceType, YSpaceType>& state,
        int n_y_active_cells_before,
        int n_y_active_cells_after,
        bool refined)
    {
        AdaptiveTestSpaceIterationSummary<int> summary;
        summary.estimator = state.estimator;
        summary.marked_source_cells = state.marked_source_cells;
        summary.divergence_residual_l2 = state.divergence_residual_l2;
        summary.n_slabs = state.n_slabs();
        summary.n_patches = state.n_patches();
        summary.n_y_active_cells_before = n_y_active_cells_before;
        summary.n_y_active_cells_after = n_y_active_cells_after;
        summary.main_solve_diagnostics = state.main_solve_diagnostics;
        summary.main_solve_setup_seconds = state.main_solve_setup_seconds;
        summary.main_solve_solve_seconds = state.main_solve_solve_seconds;
        summary.refined = refined;
        return summary;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class YSpaceType,
        class ExampleType>
    [[nodiscard]] AdaptiveTestSpaceIterationState<Backend, XSpaceType, YSpaceType>
    compute_adaptive_test_space_iteration(
        const YSpaceType& y_space,
        const XSpaceType& x_space,
        const ExampleType& example,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& main_solver_options = {},
        const AdaptiveTestSpaceIterationParameters<Backend, XSpaceType, YSpaceType>& parameters = {},
        const la::concepts::SolverOptions& local_solver_options =
            la::concepts::make_sparse_lu_solver_options(),
        const AdaptiveTestSpaceSolutionSnapshot<Backend, XSpaceType, YSpaceType>*
            initial_guess_snapshot = nullptr)
    {
        finite_element::detail::require_supported_time_slab_estimator_capability<
            typename XSpaceType::GT>();
        finite_element::detail::require_supported_time_slab_estimator_capability<
            typename YSpaceType::GT>();
        require_trial_space_embedded_in_test_space(x_space, y_space);
        auto iteration_timer =
            parameters.timing.scoped("time_slab.adaptive_iteration_total");
        auto iteration_total_wall_timer =
            parameters.timing.scoped(
                "time_slab.compute_adaptive_test_space_iteration.total_wall");

        AdaptiveTestSpaceIterationState<Backend, XSpaceType, YSpaceType> state(
            y_space,
            x_space);
        parameters.timing.add(
            "time_slab.virtual_backend_diagnostics_enabled.count",
            0.0);
        parameters.timing.add(
            "time_slab.virtual_backend_diagnostics_ran.count",
            0.0);

        {
            auto timer = parameters.timing.scoped("time_slab.main_solve_total");
            auto main_system_total_timer =
                parameters.timing.scoped("main_system.total_wall");
            using Clock = std::chrono::steady_clock;
            const auto initial_guess_preparation_start = Clock::now();
            std::optional<typename Backend::Vector> initial_guess_vector;
            finite_element::system::MainSystemInitialGuess<Backend>
                initial_guess;
            finite_element::system::MainSystemTwoLevelHierarchy<
                YSpaceType,
                XSpaceType> two_level_hierarchy;
            const finite_element::system::MainSystemTwoLevelHierarchy<
                YSpaceType,
                XSpaceType>* two_level_hierarchy_ptr = nullptr;

            if (initial_guess_snapshot != nullptr)
            {
                two_level_hierarchy.coarse_y_space =
                    &initial_guess_snapshot->y_space;
                two_level_hierarchy.coarse_x_space =
                    &initial_guess_snapshot->x_space;
                two_level_hierarchy_ptr = &two_level_hierarchy;

                if (parameters.use_adaptive_initial_guess ||
                    parameters.solve_main_system_correction)
                {
                    initial_guess_vector =
                        finite_element::system::make_saddle_initial_guess_vector<Backend>(
                            initial_guess_snapshot->lambda_delta,
                            initial_guess_snapshot->u_delta,
                            y_space,
                            x_space);
                    initial_guess.full_vector = &*initial_guess_vector;
                    initial_guess.solve_correction_equation =
                        parameters.solve_main_system_correction;
                }
            }
            parameters.timing.add(
                "main_system.initial_guess_preparation_wall",
                std::chrono::duration<double>(
                    Clock::now() - initial_guess_preparation_start)
                    .count());

            const auto main_solve =
                finite_element::system::assemble_and_solve_into_functions<
                    QSpace,
                    QTime,
                    Backend>(
                        state.lambda_delta,
                        state.u_delta,
                        y_space,
                        x_space,
                        example,
                        solver,
                        main_solver_options,
                        parameters.zero_tol,
                        -1.0,
                        initial_guess,
                        two_level_hierarchy_ptr,
                        parameters.timing,
                        parameters.main_assembly_max_threads,
                        parameters.main_assembly_memory_budget_mb,
                        parameters.main_two_pass_numeric_fill_max_threads,
                        parameters
                            .main_two_pass_numeric_fill_memory_budget_mb,
                        parameters.main_system_export,
                        parameters.main_two_pass_assembly_cache);
            state.main_solve_diagnostics = main_solve.solver_diagnostics;
            state.main_solve_setup_seconds = main_solve.solver_setup_seconds;
            state.main_solve_solve_seconds = main_solve.solver_solve_seconds;
        }

        {
            auto timer = parameters.timing.scoped("time_slab.space_construction");
            auto total_wall_timer =
                parameters.timing.scoped(
                    "time_slab.space_construction.total_wall");
            parameters.timing.add(
                "time_slab.estimator_copied_backend_fallback.count",
                0.0);
            parameters.timing.add(
                "time_slab.copied_estimator_fallback_used.count",
                0.0);
            parameters.timing.add(
                "time_slab.copied_fallback_component_count.count",
                0.0);
            parameters.timing.add(
                "time_slab.virtual_backend.unsupported_full_components.count",
                0.0);
            parameters.timing.add(
                "time_slab.source_mesh_cells_before.count",
                static_cast<double>(y_space.mesh_ref().n_cells()));
            parameters.timing.add(
                "time_slab.source_mesh_cells_after.count",
                static_cast<double>(y_space.mesh_ref().n_cells()));
            parameters.timing.add(
                "time_slab.source_mesh_cell_mutation.count",
                0.0);
            parameters.timing.add(
                "time_slab.source_mesh_mutation_count.count",
                0.0);

            state.slab_reconstruction.emplace(y_space, x_space);
            state.slab_reconstruction->set_slab_reconstruction_max_threads(
                parameters.slab_reconstruction_max_threads);
            state.slab_reconstruction
                ->set_slab_reconstruction_memory_budget_mb(
                    parameters.slab_reconstruction_memory_budget_mb);
            state.slab_reconstruction
                ->set_slab_reconstruction_operator_mode(
                    parameters.slab_reconstruction_operator_mode);
            state.slab_reconstruction->initialize(0.0, parameters.timing);
            parameters.timing.add(
                "time_slab.selected_backend.copied_slab_cells.count",
                static_cast<double>(state.copied_slab_cells()));
            parameters.timing.add(
                "time_slab.selected_backend.virtual_slab_cells.count",
                static_cast<double>(state.virtual_slab_cells()));
            parameters.timing.add(
                "time_slab.selected_backend.tilde_y_true_dofs.count",
                static_cast<double>(state.selected_backend_true_dofs()));
            parameters.timing.add(
                "time_slab.estimator_fallback.copied_slab_cells.count",
                static_cast<double>(
                    state.estimator_fallback_copied_slab_cells()));
            parameters.timing.add(
                "time_slab.estimator_fallback.used.count",
                state.estimator_uses_copied_fallback() ? 1.0 : 0.0);
            parameters.timing.add(
                "time_slab.copied_slab_cells_constructed_total.count",
                static_cast<double>(
                    state.copied_slab_cells_constructed_total()));
            parameters.timing.add(
                "time_slab.copied_slab_cells_constructed_for_fallback.count",
                static_cast<double>(
                    state.estimator_fallback_copied_slab_cells()));
            parameters.timing.add(
                "time_slab.virtual_overlay_constructed.count",
                state.virtual_overlay_constructed() ? 1.0 : 0.0);
            parameters.timing.add(
                "time_slab.tilde_y_space_constructed.count",
                state.tilde_y_space_constructed() ? 1.0 : 0.0);
        }

        {
            using Clock = std::chrono::steady_clock;
            const auto slab_reconstruction_wall_begin = Clock::now();
            state.slab_reconstruction->template solve_all_slabs<QSpace, QTime>(
                example.M,
                example.ell,
                state.u_delta,
                solver,
                local_solver_options,
                parameters.zero_tol);
            const auto slab_reconstruction_wall_end = Clock::now();
            const double slab_reconstruction_wall_seconds =
                std::chrono::duration<double>(
                    slab_reconstruction_wall_end -
                    slab_reconstruction_wall_begin).count();
            parameters.timing.add(
                "time_slab.slab_reconstruction",
                slab_reconstruction_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.total_wall",
                slab_reconstruction_wall_seconds);
            parameters.timing.add(
                state.slab_reconstruction->last_solve_all_slabs_used_openmp()
                    ? "time_slab.slab_reconstruction.openmp_parallel_used"
                    : "time_slab.slab_reconstruction.serial_used",
                0.0);
            parameters.timing.add(
                "time_slab.slab_reconstruction.thread_policy."
                "configured_max_threads.count",
                static_cast<double>(
                    state.slab_reconstruction
                        ->last_configured_max_threads()));
            parameters.timing.add(
                "time_slab.slab_reconstruction.thread_policy."
                "effective_max_threads.count",
                static_cast<double>(
                    state.slab_reconstruction
                        ->last_effective_max_threads()));
            parameters.timing.add(
                "time_slab.slab_reconstruction.thread_policy."
                "selected_threads.count",
                static_cast<double>(
                    state.slab_reconstruction
                        ->last_selected_threads()));
            parameters.timing.add(
                "time_slab.slab_reconstruction.thread_policy."
                "memory_budget_bytes",
                state.slab_reconstruction
                    ->last_memory_budget_bytes());
            parameters.timing.add(
                "time_slab.slab_reconstruction.thread_policy."
                "estimated_per_thread_cache_bytes",
                state.slab_reconstruction
                    ->last_estimated_per_thread_cache_bytes());

            double assemble_A_seconds = 0.0;
            double direct_rhs_seconds = 0.0;
            double slab_solve_seconds = 0.0;
            double slab_solver_setup_seconds = 0.0;
            double slab_solver_apply_seconds = 0.0;
            double slab_solver_symbolic_seconds = 0.0;
            double slab_solver_numeric_seconds = 0.0;
            double slab_solver_backsolve_seconds = 0.0;
            double shared_x_cache_seconds = 0.0;
            double source_ancestor_cache_seconds = 0.0;
            double slab_y_cache_seconds = 0.0;
            double f_assembly_wall_seconds = 0.0;
            double bt_u_assembly_wall_seconds = 0.0;
            double rhs_subtract_wall_seconds = 0.0;
            double function_update_wall_seconds = 0.0;
            double geometry_cache_wall_seconds = 0.0;
            double cell_restriction_wall_seconds = 0.0;
            double active_ancestor_lookup_wall_seconds = 0.0;
            double transfer_or_trace_wall_seconds = 0.0;
            double pattern_build_seconds = 0.0;
            double numeric_fill_seconds = 0.0;
            double matrix_allocation_seconds = 0.0;
            double matrix_copy_seconds = 0.0;
            double reference_table_build_seconds = 0.0;
            double residual_check_seconds = 0.0;
            double source_permutation_check_seconds = 0.0;
            double max_reconstruction_cache_bytes = 0.0;
            std::size_t qpoints_visited = 0;
            std::size_t slab_cells_visited = 0;
            std::size_t geometry_cache_hits = 0;
            std::size_t geometry_cache_misses = 0;
            std::size_t reconstruction_pattern_candidates = 0;
            std::size_t reconstruction_pattern_entries = 0;
            std::size_t reconstruction_pattern_duplicate_entries = 0;
            std::size_t reconstruction_pattern_candidate_bytes = 0;
            std::size_t reconstruction_pattern_bytes = 0;
            std::size_t reconstruction_numeric_matrix_bytes = 0;
            std::size_t reconstruction_triplet_bytes_avoided = 0;
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                assemble_A_diagnostics;
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                rhs_diagnostics;
            int slab_operator_mode_identity_zero_load_count = 0;
            int slab_operator_mode_constant_diffusion_count = 0;
            int slab_operator_mode_generic_variable_count = 0;
            std::size_t slab_diffusion_evaluations = 0;
            std::size_t slab_load_evaluations = 0;
            int slab_local_A_fast_path_count = 0;
            int slab_local_A_generic_count = 0;
            int slab_rhs_zero_load_fast_path_count = 0;
            double slab_local_A_debug_max_abs_diff = 0.0;
            double slab_rhs_debug_max_abs_diff = 0.0;
            std::vector<std::pair<double, double>> assemble_A_intervals;
            std::vector<std::pair<double, double>> rhs_intervals;
            std::vector<std::pair<double, double>> solve_intervals;
            int reconstruction_A_cache_hits = 0;
            int reconstruction_A_cache_misses = 0;
            int reconstruction_factor_cache_hits = 0;
            int reconstruction_factor_cache_misses = 0;
            int reused_symbolic_count = 0;
            int reused_sparsity_pattern_count = 0;
            int reused_operator_structure_count = 0;
            int residual_check_count = 0;
            int source_permutation_check_count = 0;
            double max_residual_inf_norm = 0.0;
            double max_source_slab_permutation_error = -1.0;
            int total_matrix_rows = 0;
            int total_rhs_size = 0;
            int total_source_cell_refs = 0;
            int max_matrix_rows = 0;
            int max_rhs_size = 0;

            for (const auto& diagnostics :
                 state.slab_reconstruction->last_slab_diagnostics())
            {
                assemble_A_seconds += diagnostics.assemble_A_seconds;
                direct_rhs_seconds += diagnostics.assemble_rhs_seconds;
                slab_solve_seconds += diagnostics.solve_seconds;
                slab_solver_setup_seconds += diagnostics.solver_setup_seconds;
                slab_solver_apply_seconds += diagnostics.solver_apply_seconds;
                slab_solver_symbolic_seconds +=
                    diagnostics.solver_symbolic_seconds;
                slab_solver_numeric_seconds += diagnostics.solver_numeric_seconds;
                slab_solver_backsolve_seconds +=
                    diagnostics.solver_backsolve_seconds;
                shared_x_cache_seconds +=
                    diagnostics.shared_x_cache_construction_seconds;
                source_ancestor_cache_seconds +=
                    diagnostics.source_ancestor_cache_construction_seconds;
                slab_y_cache_seconds +=
                    diagnostics.slab_y_cache_construction_seconds;
                f_assembly_wall_seconds +=
                    diagnostics.f_assembly_wall_seconds;
                bt_u_assembly_wall_seconds +=
                    diagnostics.bt_u_assembly_wall_seconds;
                rhs_subtract_wall_seconds +=
                    diagnostics.rhs_subtract_wall_seconds;
                function_update_wall_seconds +=
                    diagnostics.function_update_wall_seconds;
                geometry_cache_wall_seconds +=
                    diagnostics.geometry_cache_wall_seconds;
                cell_restriction_wall_seconds +=
                    diagnostics.cell_restriction_wall_seconds;
                active_ancestor_lookup_wall_seconds +=
                    diagnostics.active_ancestor_lookup_wall_seconds;
                transfer_or_trace_wall_seconds +=
                    diagnostics.transfer_or_trace_wall_seconds;
                pattern_build_seconds += diagnostics.pattern_build_seconds;
                numeric_fill_seconds += diagnostics.numeric_fill_seconds;
                matrix_allocation_seconds +=
                    diagnostics.matrix_allocation_seconds;
                matrix_copy_seconds += diagnostics.matrix_copy_seconds;
                reference_table_build_seconds +=
                    diagnostics.reference_table_build_seconds;
                residual_check_seconds += diagnostics.residual_check_seconds;
                source_permutation_check_seconds +=
                    diagnostics.source_permutation_check_seconds;
                qpoints_visited += diagnostics.qpoints_visited;
                slab_cells_visited += diagnostics.slab_cells_visited;
                geometry_cache_hits += diagnostics.geometry_cache_hits;
                geometry_cache_misses += diagnostics.geometry_cache_misses;
                reconstruction_pattern_candidates +=
                    diagnostics.pattern_candidates;
                reconstruction_pattern_entries += diagnostics.pattern_entries;
                reconstruction_pattern_duplicate_entries +=
                    diagnostics.pattern_duplicate_entries;
                reconstruction_pattern_candidate_bytes +=
                    diagnostics.pattern_candidate_bytes;
                reconstruction_pattern_bytes += diagnostics.pattern_bytes;
                reconstruction_numeric_matrix_bytes +=
                    diagnostics.numeric_matrix_bytes;
                reconstruction_triplet_bytes_avoided +=
                    diagnostics.triplet_bytes_avoided;
                assemble_A_diagnostics.add(
                    diagnostics.assemble_A_diagnostics);
                rhs_diagnostics.add(diagnostics.rhs_diagnostics);
                slab_operator_mode_identity_zero_load_count +=
                    diagnostics.operator_mode_identity_zero_load_count;
                slab_operator_mode_constant_diffusion_count +=
                    diagnostics.operator_mode_constant_diffusion_count;
                slab_operator_mode_generic_variable_count +=
                    diagnostics.operator_mode_generic_variable_count;
                slab_diffusion_evaluations +=
                    diagnostics.diffusion_evaluations;
                slab_load_evaluations += diagnostics.load_evaluations;
                slab_local_A_fast_path_count +=
                    diagnostics.local_A_fast_path_count;
                slab_local_A_generic_count +=
                    diagnostics.local_A_generic_count;
                slab_rhs_zero_load_fast_path_count +=
                    diagnostics.rhs_zero_load_fast_path_count;
                slab_local_A_debug_max_abs_diff =
                    std::max(
                        slab_local_A_debug_max_abs_diff,
                        diagnostics.local_A_debug_max_abs_diff);
                slab_rhs_debug_max_abs_diff =
                    std::max(
                        slab_rhs_debug_max_abs_diff,
                        diagnostics.rhs_debug_max_abs_diff);
                assemble_A_intervals.emplace_back(
                    diagnostics.assemble_A_wall_begin_seconds,
                    diagnostics.assemble_A_wall_end_seconds);
                rhs_intervals.emplace_back(
                    diagnostics.assemble_rhs_wall_begin_seconds,
                    diagnostics.assemble_rhs_wall_end_seconds);
                solve_intervals.emplace_back(
                    diagnostics.solve_wall_begin_seconds,
                    diagnostics.solve_wall_end_seconds);
                max_reconstruction_cache_bytes =
                    std::max(
                        max_reconstruction_cache_bytes,
                        diagnostics.reconstruction_cache_bytes);
                reconstruction_A_cache_hits +=
                    diagnostics.reconstruction_A_cache_hits;
                reconstruction_A_cache_misses +=
                    diagnostics.reconstruction_A_cache_misses;
                reconstruction_factor_cache_hits +=
                    diagnostics.reconstruction_factor_cache_hits;
                reconstruction_factor_cache_misses +=
                    diagnostics.reconstruction_factor_cache_misses;
                reused_symbolic_count += diagnostics.reused_symbolic_count;
                reused_sparsity_pattern_count +=
                    diagnostics.reused_sparsity_pattern_count;
                reused_operator_structure_count +=
                    diagnostics.reused_operator_structure_count;
                residual_check_count += diagnostics.residual_check_count;
                source_permutation_check_count +=
                    diagnostics.source_permutation_check_count;
                max_residual_inf_norm =
                    std::max(
                        max_residual_inf_norm,
                        diagnostics.residual_inf_norm);
                max_source_slab_permutation_error =
                    std::max(
                        max_source_slab_permutation_error,
                        diagnostics.source_slab_permutation_error);
                total_matrix_rows += diagnostics.matrix_rows;
                total_rhs_size += diagnostics.rhs_size;
                max_matrix_rows =
                    std::max(max_matrix_rows, diagnostics.matrix_rows);
                max_rhs_size = std::max(max_rhs_size, diagnostics.rhs_size);
                total_source_cell_refs +=
                    static_cast<int>(diagnostics.source_cell_ids.size());
            }

            parameters.timing.add(
                "time_slab.slab_reconstruction.assemble_A",
                assemble_A_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.direct_rhs_assembly",
                direct_rhs_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.slab_solves",
                slab_solve_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.solver_setup_wall",
                slab_solver_setup_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.solver_solve_wall",
                slab_solver_apply_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.solver_apply_wall",
                slab_solver_apply_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.solver_symbolic_wall",
                slab_solver_symbolic_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.solver_numeric_wall",
                slab_solver_numeric_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.solver_backsolve_wall",
                slab_solver_backsolve_seconds);
            const double assemble_A_union_wall =
                detail::interval_union_seconds(std::move(assemble_A_intervals));
            const double rhs_union_wall =
                detail::interval_union_seconds(std::move(rhs_intervals));
            const double solve_union_wall =
                detail::interval_union_seconds(std::move(solve_intervals));
            parameters.timing.add(
                "time_slab.slab_reconstruction.assemble_A_wall",
                assemble_A_union_wall);
            parameters.timing.add(
                "time_slab.slab_reconstruction.direct_rhs_assembly_wall",
                rhs_union_wall);
            parameters.timing.add(
                "time_slab.slab_reconstruction.slab_solves_wall",
                solve_union_wall);
            parameters.timing.add(
                "time_slab.slab_reconstruction.f_assembly_wall",
                f_assembly_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.BT_u_assembly_wall",
                bt_u_assembly_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.rhs_subtract_wall",
                rhs_subtract_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.function_update_wall",
                function_update_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.geometry_cache_wall",
                geometry_cache_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.cell_restriction_wall",
                cell_restriction_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.active_ancestor_lookup_wall",
                active_ancestor_lookup_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.transfer_or_trace_wall",
                transfer_or_trace_wall_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.qpoints_visited",
                static_cast<double>(qpoints_visited));
            parameters.timing.add(
                "time_slab.slab_reconstruction.slab_cells_visited",
                static_cast<double>(slab_cells_visited));
            parameters.timing.add(
                "time_slab.slab_reconstruction.geometry_cache_hits",
                static_cast<double>(geometry_cache_hits));
            parameters.timing.add(
                "time_slab.slab_reconstruction.geometry_cache_misses",
                static_cast<double>(geometry_cache_misses));
            finite_element::assembly::detail::record_assembly_diagnostics(
                parameters.timing,
                "time_slab.slab_reconstruction.assemble_A",
                assemble_A_diagnostics);
            finite_element::assembly::detail::record_assembly_diagnostics(
                parameters.timing,
                "time_slab.slab_reconstruction.direct_rhs_assembly",
                rhs_diagnostics);
            parameters.timing.add(
                "time_slab.slab_reconstruction.operator_mode."
                "identity_zero_load_count",
                static_cast<double>(
                    slab_operator_mode_identity_zero_load_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.operator_mode."
                "constant_diffusion_count",
                static_cast<double>(
                    slab_operator_mode_constant_diffusion_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.operator_mode."
                "generic_variable_count",
                static_cast<double>(
                    slab_operator_mode_generic_variable_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.diffusion_evaluations",
                static_cast<double>(slab_diffusion_evaluations));
            parameters.timing.add(
                "time_slab.slab_reconstruction.load_evaluations",
                static_cast<double>(slab_load_evaluations));
            parameters.timing.add(
                "time_slab.slab_reconstruction.local_A_fast_path_count",
                static_cast<double>(slab_local_A_fast_path_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.local_A_generic_count",
                static_cast<double>(slab_local_A_generic_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "rhs_zero_load_fast_path_count",
                static_cast<double>(slab_rhs_zero_load_fast_path_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.local_A_debug_max_abs_diff",
                slab_local_A_debug_max_abs_diff);
            parameters.timing.add(
                "time_slab.slab_reconstruction.rhs_debug_max_abs_diff",
                slab_rhs_debug_max_abs_diff);
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "shared_x_cache_construction",
                shared_x_cache_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "source_ancestor_cache_construction",
                source_ancestor_cache_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.slab_y_cache_construction",
                slab_y_cache_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.pattern_build",
                pattern_build_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.numeric_fill",
                numeric_fill_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.matrix_allocation_wall",
                matrix_allocation_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.matrix_copy_wall",
                matrix_copy_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.reference_table_build_wall",
                reference_table_build_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.residual_check_wall",
                residual_check_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "source_permutation_check_wall",
                source_permutation_check_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction."
                "shared_cache_build",
                shared_x_cache_seconds + source_ancestor_cache_seconds +
                    slab_y_cache_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction.pattern_build",
                pattern_build_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction.numeric_fill",
                numeric_fill_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction.direct_rhs",
                direct_rhs_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction.slab_solves",
                slab_solve_seconds);
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "reconstruction_A_cache_hits.count",
                static_cast<double>(reconstruction_A_cache_hits));
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "reconstruction_A_cache_misses.count",
                static_cast<double>(reconstruction_A_cache_misses));
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "reconstruction_factor_cache_hits.count",
                static_cast<double>(reconstruction_factor_cache_hits));
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "reconstruction_factor_cache_misses.count",
                static_cast<double>(reconstruction_factor_cache_misses));
            parameters.timing.add(
                "time_slab.slab_reconstruction.reused_symbolic_count",
                static_cast<double>(reused_symbolic_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "reused_sparsity_pattern_count",
                static_cast<double>(reused_sparsity_pattern_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "reused_operator_structure_count",
                static_cast<double>(reused_operator_structure_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.residual_check_count",
                static_cast<double>(residual_check_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "source_permutation_check_count",
                static_cast<double>(source_permutation_check_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction_cache_bytes",
                max_reconstruction_cache_bytes);
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction.cache_hits.count",
                static_cast<double>(
                    reconstruction_A_cache_hits +
                    reconstruction_factor_cache_hits));
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction.cache_misses.count",
                static_cast<double>(
                    reconstruction_A_cache_misses +
                    reconstruction_factor_cache_misses));
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction.cache_bytes",
                max_reconstruction_cache_bytes);
            parameters.timing.add(
                "time_slab.slab_reconstruction.pattern_candidates.count",
                static_cast<double>(reconstruction_pattern_candidates));
            parameters.timing.add(
                "time_slab.slab_reconstruction.pattern_unique_entries.count",
                static_cast<double>(reconstruction_pattern_entries));
            parameters.timing.add(
                "time_slab.slab_reconstruction.pattern_duplicate_entries.count",
                static_cast<double>(
                    reconstruction_pattern_duplicate_entries));
            parameters.timing.add(
                "time_slab.slab_reconstruction.pattern_candidate_bytes",
                static_cast<double>(
                    reconstruction_pattern_candidate_bytes));
            parameters.timing.add(
                "time_slab.slab_reconstruction.pattern_bytes",
                static_cast<double>(reconstruction_pattern_bytes));
            parameters.timing.add(
                "time_slab.slab_reconstruction.numeric_matrix_bytes",
                static_cast<double>(
                    reconstruction_numeric_matrix_bytes));
            parameters.timing.add(
                "time_slab.slab_reconstruction.triplet_bytes_avoided",
                static_cast<double>(
                    reconstruction_triplet_bytes_avoided));
            parameters.timing.add(
                "time_slab.slab_reconstruction.slab_count.count",
                static_cast<double>(
                    state.slab_reconstruction->last_slab_diagnostics().size()));
            parameters.timing.add(
                "time_slab.slab_reconstruction.matrix_rows.count",
                static_cast<double>(total_matrix_rows));
            const auto slab_diagnostic_count =
                state.slab_reconstruction->last_slab_diagnostics().size();
            parameters.timing.add(
                "time_slab.slab_reconstruction.mean_slab_true_dofs.count",
                slab_diagnostic_count == 0
                    ? 0.0
                    : static_cast<double>(total_matrix_rows) /
                          static_cast<double>(slab_diagnostic_count));
            parameters.timing.add(
                "time_slab.slab_reconstruction.max_slab_true_dofs.count",
                static_cast<double>(max_matrix_rows));
            parameters.timing.add(
                "time_slab.slab_reconstruction.reconstruction_unknown_count.count",
                static_cast<double>(total_matrix_rows));
            parameters.timing.add(
                "time_slab.slab_reconstruction.rhs_size.count",
                static_cast<double>(total_rhs_size));
            parameters.timing.add(
                "time_slab.slab_reconstruction.max_rhs_size.count",
                static_cast<double>(max_rhs_size));
            parameters.timing.add(
                "time_slab.slab_reconstruction.source_cell_refs.count",
                static_cast<double>(total_source_cell_refs));
            parameters.timing.add(
                "time_slab.slab_reconstruction.max_residual_inf_norm.count",
                max_residual_inf_norm);
            parameters.timing.add(
                "time_slab.slab_reconstruction."
                "max_source_slab_permutation_error.count",
                max_source_slab_permutation_error);
            const double f2_slab_count =
                static_cast<double>(slab_diagnostic_count);
            const double f2_mean_slab_true_dofs =
                slab_diagnostic_count == 0
                    ? 0.0
                    : static_cast<double>(total_matrix_rows) /
                          static_cast<double>(slab_diagnostic_count);
            parameters.timing.add(
                "slab_reconstruction.total_wall",
                slab_reconstruction_wall_seconds);
            parameters.timing.add(
                "slab_reconstruction.operator_assembly_wall",
                assemble_A_union_wall);
            parameters.timing.add(
                "slab_reconstruction.rhs_assembly_wall",
                rhs_union_wall);
            parameters.timing.add(
                "slab_reconstruction.solver_setup_wall",
                slab_solver_setup_seconds);
            parameters.timing.add(
                "slab_reconstruction.symbolic_factorization_wall",
                slab_solver_symbolic_seconds);
            parameters.timing.add(
                "slab_reconstruction.numeric_factorization_wall",
                slab_solver_numeric_seconds);
            parameters.timing.add(
                "slab_reconstruction.solve_wall",
                slab_solver_apply_seconds);
            parameters.timing.add(
                "slab_reconstruction.function_update_wall",
                function_update_wall_seconds);
            parameters.timing.add(
                "slab_reconstruction.sparsity_pattern_build_wall",
                pattern_build_seconds);
            parameters.timing.add(
                "slab_reconstruction.matrix_allocation_wall",
                matrix_allocation_seconds);
            parameters.timing.add(
                "slab_reconstruction.matrix_copy_wall",
                matrix_copy_seconds);
            parameters.timing.add(
                "slab_reconstruction.reference_table_build_wall",
                reference_table_build_seconds);
            parameters.timing.add(
                "slab_reconstruction.slabs",
                f2_slab_count);
            parameters.timing.add(
                "slab_reconstruction.total_slab_true_dofs_processed",
                static_cast<double>(total_matrix_rows));
            parameters.timing.add(
                "slab_reconstruction.mean_slab_true_dofs",
                f2_mean_slab_true_dofs);
            parameters.timing.add(
                "slab_reconstruction.max_slab_true_dofs",
                static_cast<double>(max_matrix_rows));
            parameters.timing.add(
                "slab_reconstruction.reused_symbolic_count",
                static_cast<double>(reused_symbolic_count));
            parameters.timing.add(
                "slab_reconstruction.reused_sparsity_pattern_count",
                static_cast<double>(reused_sparsity_pattern_count));
            parameters.timing.add(
                "slab_reconstruction.reused_operator_structure_count",
                static_cast<double>(reused_operator_structure_count));
            parameters.timing.add(
                "slab_reconstruction.operator_mode.identity_zero_load_count",
                static_cast<double>(
                    slab_operator_mode_identity_zero_load_count));
            parameters.timing.add(
                "slab_reconstruction.operator_mode.constant_diffusion_count",
                static_cast<double>(
                    slab_operator_mode_constant_diffusion_count));
            parameters.timing.add(
                "slab_reconstruction.operator_mode.generic_variable_count",
                static_cast<double>(
                    slab_operator_mode_generic_variable_count));
            parameters.timing.add(
                "slab_reconstruction.diffusion_evaluations",
                static_cast<double>(slab_diffusion_evaluations));
            parameters.timing.add(
                "slab_reconstruction.load_evaluations",
                static_cast<double>(slab_load_evaluations));
            parameters.timing.add(
                "slab_reconstruction.local_A_fast_path_count",
                static_cast<double>(slab_local_A_fast_path_count));
            parameters.timing.add(
                "slab_reconstruction.local_A_generic_count",
                static_cast<double>(slab_local_A_generic_count));
            parameters.timing.add(
                "slab_reconstruction.rhs_zero_load_fast_path_count",
                static_cast<double>(slab_rhs_zero_load_fast_path_count));
            parameters.timing.add(
                "slab_reconstruction.local_A_debug_max_abs_diff",
                slab_local_A_debug_max_abs_diff);
            parameters.timing.add(
                "slab_reconstruction.rhs_debug_max_abs_diff",
                slab_rhs_debug_max_abs_diff);
        }

        {
            {
                trace_time_slab_estimator_inputs_if_requested<QSpace, QTime>(
                    state.lambda_delta,
                    state.lambda_tilde(),
                    state.u_delta);
            }

            {
                auto timer =
                    parameters.timing.scoped(
                        "time_slab.local_error_space_construction");
                auto total_wall_timer =
                    parameters.timing.scoped(
                        "time_slab.local_error_space_construction.total_wall");
                state.flux_reconstruction.emplace(
                    state.slab_reconstruction->slab_space_ref(),
                    x_space);
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_patch_tile_size(
                                          parameters.local_error_patch_tile_size);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_patch_tile_size(
                            parameters.local_error_patch_tile_size);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_cell_chunk_size(
                                          parameters.local_error_cell_chunk_size);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_cell_chunk_size(
                            parameters.local_error_cell_chunk_size);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_max_threads(
                                          parameters.local_error_max_threads);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_max_threads(
                            parameters.local_error_max_threads);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_memory_budget_mb(
                                          parameters
                                              .local_error_memory_budget_mb);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_memory_budget_mb(
                            parameters.local_error_memory_budget_mb);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_worker_context_mode(
                                          parameters
                                              .local_error_worker_context_mode);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_worker_context_mode(
                            parameters.local_error_worker_context_mode);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_context_storage(
                                          parameters
                                              .local_error_context_storage);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_context_storage(
                            parameters.local_error_context_storage);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_state_index_mode(
                                          parameters
                                              .local_error_state_index_mode);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_state_index_mode(
                            parameters.local_error_state_index_mode);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_cell_state_cache_mode(
                                          parameters
                                              .local_error_cell_state_cache_mode);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_cell_state_cache_mode(
                            parameters.local_error_cell_state_cache_mode);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_cell_state_cache_budget_mb(
                                          parameters
                                              .local_error_cell_state_cache_budget_mb);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_cell_state_cache_budget_mb(
                            parameters
                                .local_error_cell_state_cache_budget_mb);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_coefficient_fast_path(
                                          parameters
                                              .local_error_coefficient_fast_path);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_coefficient_fast_path(
                            parameters.local_error_coefficient_fast_path);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_cell_state_representation(
                                          parameters
                                              .local_error_cell_state_representation);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_cell_state_representation(
                            parameters.local_error_cell_state_representation);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_flux_diagnostics_mode(
                                          parameters
                                              .local_error_flux_diagnostics_mode);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_flux_diagnostics_mode(
                            parameters.local_error_flux_diagnostics_mode);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_patch_solver(
                                          parameters.local_error_patch_solver);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_patch_solver(
                            parameters.local_error_patch_solver);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_compact_state_shadow(
                                          parameters
                                              .local_error_compact_state_shadow);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_compact_state_shadow(
                            parameters.local_error_compact_state_shadow);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_shared_context_validation(
                                          parameters
                                              .shared_context_validation);
                              })
                {
                    state.flux_reconstruction
                        ->set_shared_context_validation(
                            parameters.shared_context_validation);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_fused_error_and_flux_diagnostics(
                                          parameters
                                              .fused_error_and_flux_diagnostics);
                              })
                {
                    state.flux_reconstruction
                        ->set_fused_error_and_flux_diagnostics(
                            parameters.fused_error_and_flux_diagnostics);
                }
                if constexpr (requires
                              {
                                  state.flux_reconstruction
                                      ->set_local_error_reuse_patch_solve_workspace(
                                          parameters
                                              .local_error_reuse_patch_solve_workspace);
                              })
                {
                    state.flux_reconstruction
                        ->set_local_error_reuse_patch_solve_workspace(
                            parameters.local_error_reuse_patch_solve_workspace);
                }
                state.flux_reconstruction->initialize();
            }

            {
                auto timer =
                    parameters.timing.scoped("time_slab.local_error_solves");
                auto total_wall_timer =
                    parameters.timing.scoped(
                        "time_slab.local_error_solves.total_wall");
                parameters.timing.add(
                    "time_slab.local_error_solves.patch_count.count",
                    static_cast<double>(
                        state.flux_reconstruction->n_patches()));
                state.flux_reconstruction->template solve_all_patches<
                    QSpace,
                    QTime>(
                    state.lambda_tilde(),
                    example.ell,
                    state.u_delta,
                    example.M,
                    solver,
                    local_solver_options,
                    parameters.zero_tol,
                    parameters.timing);
                parameters.timing.add(
                    state.flux_reconstruction
                            ->last_solve_all_patches_used_openmp()
                        ? "time_slab.local_error_solves.openmp_parallel_used"
                        : "time_slab.local_error_solves.serial_used",
                    0.0);
                parameters.timing.add(
                    "time_slab.local_error_solves.openmp_parallel_used.count",
                    state.flux_reconstruction
                            ->last_solve_all_patches_used_openmp()
                        ? 1.0
                        : 0.0);
                parameters.timing.add(
                    "time_slab.local_error_solves.serial_used.count",
                    state.flux_reconstruction
                            ->last_solve_all_patches_used_openmp()
                        ? 0.0
                        : 1.0);
            }

            {
                auto timer =
                    parameters.timing.scoped(
                        "time_slab.estimator_combination");
                auto total_wall_timer =
                    parameters.timing.scoped(
                        "time_slab.estimator_combination.total_wall");
                {
                    auto sub_timer =
                        parameters.timing.scoped("time_slab.flux_diagnostics");
                    auto sub_total_wall_timer =
                        parameters.timing.scoped(
                            "time_slab.flux_diagnostics.total_wall");
                    if constexpr (XSpaceType::GT::dim_space_v == 2)
                    {
                        if (state.flux_reconstruction
                                ->has_fused_flux_diagnostics())
                        {
                            parameters.timing.add(
                                "time_slab.flux_diagnostics.fused_cache_hit",
                                0.0);
                            const auto fused_stats =
                                state.flux_reconstruction
                                    ->fused_flux_diagnostics_runtime_stats();
                            parameters.timing.add(
                                "time_slab.flux_diagnostics.fused_qpoints.count",
                                fused_stats.qpoints);
                            parameters.timing.add(
                                "time_slab.flux_diagnostics.reused_cell_state.count",
                                fused_stats.reused_cell_state);
                            parameters.timing.add(
                                "time_slab.flux_diagnostics."
                                "extra_cell_state_rebuilds.count",
                                fused_stats.extra_cell_state_rebuilds);
                            state.flux_diagnostics =
                                state.flux_reconstruction
                                    ->fused_flux_diagnostics();
                        }
                        else
                        {
                            parameters.timing.add(
                                "time_slab.flux_diagnostics.standalone_fallback",
                                0.0);
                            const auto standalone_start =
                                std::chrono::steady_clock::now();
                            state.flux_diagnostics =
                                detail::compute_adaptive_flux_diagnostics<
                                    QSpace,
                                    QTime,
                                    Backend,
                                    XSpaceType,
                                    YSpaceType>(
                                        *state.flux_reconstruction,
                                        state.lambda_tilde(),
                                        state.u_delta,
                                        example.ell,
                                        example.M,
                                        parameters.timing);
                            parameters.timing.add(
                                "time_slab.flux_diagnostics.standalone",
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() -
                                    standalone_start)
                                    .count());
                            parameters.timing.add(
                                "flux_diagnostics.standalone_wall",
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now() -
                                    standalone_start)
                                    .count());
                            parameters.timing.add(
                                "time_slab.flux_diagnostics.fused_qpoints.count",
                                0.0);
                            parameters.timing.add(
                                "time_slab.flux_diagnostics."
                                "reused_cell_state.count",
                                0.0);
                            parameters.timing.add(
                                "time_slab.flux_diagnostics."
                                "extra_cell_state_rebuilds.count",
                                0.0);
                        }
                    }
                    else
                    {
                        state.flux_diagnostics =
                            detail::compute_adaptive_flux_diagnostics<
                                QSpace,
                                QTime,
                                Backend,
                                XSpaceType,
                                YSpaceType>(
                                    *state.flux_reconstruction,
                                    state.lambda_tilde(),
                                    state.u_delta,
                                    example.ell,
                                    example.M,
                                    parameters.timing);
                    }
                }
                detail::record_flux_diagnostics_checksums(
                    parameters.timing,
                    state.flux_diagnostics);
                {
                    auto sub_timer =
                        parameters.timing.scoped(
                            "time_slab.reconstruction_y_norm");
                    auto sub_total_wall_timer =
                        parameters.timing.scoped(
                            "time_slab.reconstruction_y_norm.total_wall");
                    state.reconstruction_y_squared =
                        compute_Y_error_squared_by_source_cell<
                            QSpace,
                            QTime>(
                                state.lambda_delta,
                                state.lambda_tilde(),
                                example.M,
                                parameters.timing);
                    parameters.timing.add(
                        "time_slab.virtual_component.used.count",
                        0.0);
                    trace_Y_error_contributions_if_requested<
                        QSpace,
                        QTime>(
                            state.lambda_delta,
                            state.lambda_tilde(),
                            example.M,
                            x_space);
                    trace_Y_error_cross_terms_if_requested<
                        QSpace,
                        QTime>(
                            state.lambda_delta,
                            state.lambda_tilde(),
                            example.M,
                            state.reconstruction_y_squared);
                }

                    {
                        auto sub_timer =
                            parameters.timing.scoped("time_slab.map_combination");
                        if (parameters.deterministic_estimator_reductions)
                        {
                            state.estimator =
                                combine_time_slab_estimator_squared_errors_deterministic(
                                    state.flux_diagnostics,
                                    state.reconstruction_y_squared,
                                    parameters.timing);
                        }
                        else
                        {
                            record_unordered_estimator_reduction_path(
                                parameters.timing);
                            state.estimator =
                                combine_time_slab_estimator_squared_errors(
                                    state.flux_diagnostics,
                                    state.reconstruction_y_squared);
                        }

                    const double divergence_residual_squared =
                        state.estimator.divergence_residual_total();
                    detail::require_nonnegative_squared_value(
                        divergence_residual_squared,
                        "compute_adaptive_test_space_iteration divergence residual total");
                    state.divergence_residual_l2 =
                        std::sqrt(detail::clamp_small_negative(
                            divergence_residual_squared));
                }
            }
        }
        if (parameters.check_divergence_residual &&
            state.divergence_residual_l2 > parameters.divergence_residual_l2_tolerance)
        {
            std::ostringstream message;
            message
                << "compute_adaptive_test_space_iteration: divergence residual L2 norm "
                << state.divergence_residual_l2
                << " exceeds tolerance "
                << parameters.divergence_residual_l2_tolerance
                << '.';
            throw std::runtime_error(message.str());
        }

        {
            auto timer = parameters.timing.scoped("time_slab.marking");
            auto total_wall_timer =
                parameters.timing.scoped("time_slab.marking.total_wall");
            DoerflerMarkingDiagnostics diagnostics;
            if (parameters.deterministic_estimator_reductions)
            {
                state.marked_source_cells =
                    state.estimator.doerfler_marking_deterministic(
                        parameters.doerfler_theta,
                        parameters.doerfler_near_tie_tolerance,
                        &diagnostics,
                        parameters.timing);
            }
            else
            {
                state.marked_source_cells =
                    doerfler_mark_test_space_entities(
                        state.estimator,
                        parameters.doerfler_theta);
            }
            detail::record_local_error_marking_diagnostics(
                parameters.timing,
                state.estimator,
                state.reconstruction_y_squared,
                state.marked_source_cells,
                parameters.doerfler_theta,
                parameters.deterministic_estimator_reductions,
                parameters.doerfler_near_tie_tolerance,
                parameters.deterministic_estimator_reductions
                    ? &diagnostics
                    : nullptr);
        }

        return state;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class YSpaceType,
        class ExampleType>
    [[nodiscard]] AdaptiveTestSpaceLoopResult<int>
    run_adaptive_test_space_loop(
        YSpaceType& y_space,
        const XSpaceType& x_space,
        const ExampleType& example,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& main_solver_options = {},
        const AdaptiveTestSpaceLoopOptions<Backend, XSpaceType, YSpaceType>& loop_options = {},
        const la::concepts::SolverOptions& local_solver_options =
            la::concepts::make_sparse_lu_solver_options())
    {
        if (loop_options.max_iterations < 0)
        {
            throw std::runtime_error(
                "run_adaptive_test_space_loop: max_iterations must be nonnegative.");
        }

        AdaptiveTestSpaceLoopResult<int> result;

        AdaptiveTestSpaceIterationParameters<Backend, XSpaceType, YSpaceType>
            iteration_parameters;
        iteration_parameters.doerfler_theta = loop_options.doerfler_theta;
        iteration_parameters.zero_tol = loop_options.zero_tol;
        iteration_parameters.divergence_residual_l2_tolerance =
            loop_options.divergence_residual_l2_tolerance;
        iteration_parameters.check_divergence_residual =
            loop_options.check_divergence_residual;
        iteration_parameters.use_adaptive_initial_guess =
            loop_options.use_adaptive_initial_guess;
        iteration_parameters.solve_main_system_correction =
            loop_options.solve_main_system_correction;
        iteration_parameters.fused_error_and_flux_diagnostics =
            loop_options.fused_error_and_flux_diagnostics;
        iteration_parameters.local_error_reuse_patch_solve_workspace =
            loop_options.local_error_reuse_patch_solve_workspace;
        iteration_parameters.deterministic_estimator_reductions =
            loop_options.deterministic_estimator_reductions;
        iteration_parameters.doerfler_near_tie_tolerance =
            loop_options.doerfler_near_tie_tolerance;
        iteration_parameters.local_error_patch_tile_size =
            loop_options.local_error_patch_tile_size > 0
                ? loop_options.local_error_patch_tile_size
                : (YSpaceType::FETraitsType::p_space_v >= 3 ? 256 : 512);
        iteration_parameters.local_error_cell_chunk_size =
            loop_options.local_error_cell_chunk_size > 0
                ? loop_options.local_error_cell_chunk_size
                : (YSpaceType::FETraitsType::p_space_v >= 3 ? 256 : 512);
        iteration_parameters.local_error_max_threads =
            loop_options.local_error_max_threads;
        iteration_parameters.local_error_memory_budget_mb =
            loop_options.local_error_memory_budget_mb;
        iteration_parameters.local_error_worker_context_mode =
            loop_options.local_error_worker_context_mode;
        iteration_parameters.local_error_context_storage =
            loop_options.local_error_context_storage;
        iteration_parameters.local_error_state_index_mode =
            loop_options.local_error_state_index_mode;
        iteration_parameters.local_error_cell_state_cache_mode =
            loop_options.local_error_cell_state_cache_mode;
        iteration_parameters.local_error_cell_state_cache_budget_mb =
            loop_options.local_error_cell_state_cache_budget_mb;
        iteration_parameters.local_error_cell_state_representation =
            loop_options.local_error_cell_state_representation;
        iteration_parameters.local_error_flux_diagnostics_mode =
            loop_options.local_error_flux_diagnostics_mode;
        iteration_parameters.local_error_patch_solver =
            loop_options.local_error_patch_solver;
        iteration_parameters.local_error_coefficient_fast_path =
            loop_options.local_error_coefficient_fast_path;
        iteration_parameters.local_error_compact_state_shadow =
            loop_options.local_error_compact_state_shadow;
        iteration_parameters.shared_context_validation =
            loop_options.shared_context_validation;
        iteration_parameters.slab_reconstruction_operator_mode =
            loop_options.slab_reconstruction_operator_mode;
        iteration_parameters.main_assembly_max_threads =
            loop_options.main_assembly_max_threads;
        iteration_parameters.slab_reconstruction_max_threads =
            loop_options.slab_reconstruction_max_threads;
        iteration_parameters.main_assembly_memory_budget_mb =
            loop_options.main_assembly_memory_budget_mb;
        iteration_parameters.slab_reconstruction_memory_budget_mb =
            loop_options.slab_reconstruction_memory_budget_mb;
        iteration_parameters.main_two_pass_numeric_fill_max_threads =
            loop_options.main_two_pass_numeric_fill_max_threads > 0
                ? loop_options.main_two_pass_numeric_fill_max_threads
                : 4;
        iteration_parameters.main_two_pass_numeric_fill_memory_budget_mb =
            loop_options.main_two_pass_numeric_fill_memory_budget_mb;
        iteration_parameters.time_slab_backend =
            loop_options.time_slab_backend;
        iteration_parameters.main_system_export =
            loop_options.main_system_export;
        iteration_parameters.timing = loop_options.timing;

        using SolutionSnapshot =
            AdaptiveTestSpaceSolutionSnapshot<Backend, XSpaceType, YSpaceType>;
        std::optional<SolutionSnapshot> previous_solution_snapshot{};
        finite_element::assembly::TwoPassFullSaddleAssemblyCache2D<
            Backend,
            XSpaceType,
            YSpaceType> main_two_pass_assembly_cache;
        iteration_parameters.main_two_pass_assembly_cache =
            &main_two_pass_assembly_cache;
        for (int iter = 0; iter < loop_options.max_iterations; ++iter)
        {
            const int n_y_active_cells_before =
                static_cast<int>(y_space.active_cells().size());

            const auto state =
                compute_adaptive_test_space_iteration<
                    QSpace,
                    QTime,
                    Backend>(
                        y_space,
                        x_space,
                        example,
                        solver,
                        main_solver_options,
                        iteration_parameters,
                        local_solver_options,
                        previous_solution_snapshot.has_value()
                            ? &*previous_solution_snapshot
                            : nullptr);

            const bool stop_for_small_estimator =
                state.estimator.total() <= loop_options.estimator_squared_stop;
            const bool stop_for_empty_marking =
                loop_options.stop_on_empty_marking &&
                state.marked_source_cells.empty();
            const bool can_refine_again =
                iter + 1 < loop_options.max_iterations &&
                !stop_for_small_estimator &&
                !stop_for_empty_marking &&
                !state.marked_source_cells.empty();

            int n_y_active_cells_after = n_y_active_cells_before;
            if (can_refine_again)
            {
                previous_solution_snapshot.emplace(
                    y_space,
                    x_space,
                    state.lambda_delta,
                    state.u_delta);

                refine_test_space_from_marked_source_cells(
                    y_space,
                    x_space,
                    state.marked_source_cells);
                n_y_active_cells_after =
                    static_cast<int>(y_space.active_cells().size());
            }

            result.iterations.push_back(
                make_adaptive_test_space_iteration_summary(
                    state,
                    n_y_active_cells_before,
                    n_y_active_cells_after,
                    can_refine_again));

            if (stop_for_small_estimator || stop_for_empty_marking)
                break;
        }

        return result;
    }
}

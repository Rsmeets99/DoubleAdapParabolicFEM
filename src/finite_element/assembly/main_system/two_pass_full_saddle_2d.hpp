#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <utility>
#include <vector>

#include "core/openmp.hpp"

#include "../constrained_dofs.hpp"
#include "../scatter.hpp"
#include "../detail/active_cell_locator.hpp"
#include "../detail/assembly_diagnostics.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../detail/space_time_basis_tables.hpp"
#include "../detail/trace_geometry_utils.hpp"
#include "../detail/zero_local.hpp"
#include "../../basis/space_time_basis_selector.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../geometry/cell_geometry.hpp"
#include "../../detail/timing.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/sparse_pattern_builder.hpp"
#include "linear_algebra/concepts/vector.hpp"
#include "linear_algebra/system/linear_system.hpp"

#ifndef ADAPPARABOLICFEM_ENABLE_SAME_CELL_PRETABULATED_X_GRADS_2D
#define ADAPPARABOLICFEM_ENABLE_SAME_CELL_PRETABULATED_X_GRADS_2D 1
#endif

namespace finite_element::assembly
{
    struct TwoPassFullSaddleAssemblyDiagnostics2D
    {
        std::size_t y_active_cells = 0;
        std::size_t x_initial_trace_cells = 0;
        std::size_t pattern_candidates = 0;
        std::size_t pattern_entries = 0;
        std::size_t pattern_duplicate_entries = 0;
        std::size_t numeric_nonzeros = 0;
        std::size_t nnz_A_y = 0;
        std::size_t nnz_B = 0;
        std::size_t nnz_B_transpose = 0;
        std::size_t nnz_C_signed = 0;
        std::size_t pattern_candidate_bytes = 0;
        std::size_t pattern_bytes = 0;
        std::size_t compact_pattern_bytes = 0;
        std::size_t pattern_bytes_saved = 0;
        std::size_t numeric_matrix_bytes = 0;
        std::size_t slot_map_bytes = 0;
        std::size_t triplet_bytes_avoided = 0;
        std::size_t estimated_assembly_peak_bytes = 0;
        int numeric_fill_threads = 1;
        std::size_t numeric_fill_thread_buffer_bytes = 0;
        int numeric_fill_conflict_strategy = 0;
        bool numeric_fill_fallback_to_serial = true;
        bool pattern_reused = false;
        bool slot_maps_reused = false;
        std::size_t cell_restrictions_reused = 0;
        std::size_t geometry_cache_hits = 0;
        std::size_t geometry_cache_misses = 0;
        bool assembly_cache_enabled = false;
        bool assembly_cache_key_reused = false;
        bool slot_maps_enabled = false;
        std::size_t coefficient_fast_path_identity_count = 0;
        std::size_t coefficient_fast_path_zero_load_count = 0;
        std::size_t coefficient_fast_path_generic_count = 0;
        std::size_t diffusion_evaluations = 0;
        std::size_t load_evaluations = 0;
    };

    inline void atomic_add_double(
        std::atomic<double>& accumulator,
        const double value) noexcept
    {
        double current = accumulator.load(std::memory_order_relaxed);
        while (!accumulator.compare_exchange_weak(
            current,
            current + value,
            std::memory_order_relaxed,
            std::memory_order_relaxed))
        {
        }
    }

    template<class VectorLike, class LocalVectorType>
    requires la::concepts::VectorLike<VectorLike>
    void scatter_vector_offset(
        VectorLike& vector,
        const LocalVectorType& local,
        const LocalDofExpansion& test_dofs,
        int offset,
        double zero_tol = 1e-15)
    {
        if (local.size != test_dofs.size())
        {
            throw std::runtime_error(
                "scatter_vector_offset: local.size != test_dofs.size().");
        }

        for (int i = 0; i < local.size; ++i)
        {
            const double f_i = local[i];
            if (std::abs(f_i) <= zero_tol)
                continue;

            for (const auto& wi : test_dofs[i])
            {
                if (wi.true_dof < 0)
                    continue;

                vector.add(offset + wi.true_dof, wi.weight * f_i);
            }
        }
    }

    inline void append_restriction_true_dofs(
        std::vector<int>& out,
        const CellRestriction& restriction,
        int offset)
    {
        for (const auto& local_entries : restriction.rows)
        {
            for (const auto& entry : local_entries)
            {
                if (entry.true_dof >= 0)
                    out.push_back(offset + entry.true_dof);
            }
        }
    }

    inline void sort_unique_dofs(std::vector<int>& dofs)
    {
        std::sort(dofs.begin(), dofs.end());
        dofs.erase(std::unique(dofs.begin(), dofs.end()), dofs.end());
    }

    [[nodiscard]] inline std::vector<std::vector<int>>
    color_items_by_true_dofs(
        const std::vector<std::vector<int>>& item_dofs,
        int n_total_dofs)
    {
        std::vector<std::vector<int>> colors;
        std::vector<std::vector<int>> colors_by_dof(
            static_cast<std::size_t>(n_total_dofs));
        std::vector<int> blocked_colors;

        for (int item = 0; item < static_cast<int>(item_dofs.size()); ++item)
        {
            blocked_colors.clear();
            for (const int dof : item_dofs[static_cast<std::size_t>(item)])
            {
                if (dof < 0 || dof >= n_total_dofs)
                    continue;
                const auto& dof_colors =
                    colors_by_dof[static_cast<std::size_t>(dof)];
                blocked_colors.insert(
                    blocked_colors.end(),
                    dof_colors.begin(),
                    dof_colors.end());
            }

            std::sort(blocked_colors.begin(), blocked_colors.end());
            blocked_colors.erase(
                std::unique(blocked_colors.begin(), blocked_colors.end()),
                blocked_colors.end());

            int color = 0;
            while (std::binary_search(
                       blocked_colors.begin(),
                       blocked_colors.end(),
                       color))
            {
                ++color;
            }

            if (color == static_cast<int>(colors.size()))
                colors.emplace_back();
            colors[static_cast<std::size_t>(color)].push_back(item);

            for (const int dof : item_dofs[static_cast<std::size_t>(item)])
            {
                if (dof < 0 || dof >= n_total_dofs)
                    continue;
                colors_by_dof[static_cast<std::size_t>(dof)].push_back(color);
            }
        }

        return colors;
    }

    struct WeightedSparseSlot
    {
        std::size_t slot = 0;
        double weight = 0.0;
    };

    struct MatrixScatterSlotMap
    {
        int rows = 0;
        int cols = 0;
        std::vector<std::size_t> offsets{};
        std::vector<WeightedSparseSlot> entries{};

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return offsets.capacity() * sizeof(std::size_t) +
                   entries.capacity() * sizeof(WeightedSparseSlot);
        }
    };

    [[nodiscard]] inline std::size_t estimate_matrix_scatter_slot_map_bytes(
        const LocalDofExpansion& test_dofs,
        const LocalDofExpansion& trial_dofs)
    {
        const auto rows = static_cast<std::size_t>(test_dofs.size());
        const auto cols = static_cast<std::size_t>(trial_dofs.size());
        std::size_t entries = 0;
        for (int i = 0; i < test_dofs.size(); ++i)
        {
            const auto& I = test_dofs[i];
            for (int j = 0; j < trial_dofs.size(); ++j)
            {
                const auto& J = trial_dofs[j];
                entries += I.size() * J.size();
            }
        }
        return (rows * cols + 1U) * sizeof(std::size_t) +
               entries * sizeof(WeightedSparseSlot);
    }

    struct YCellSaddleSlotMaps
    {
        MatrixScatterSlotMap A{};
        MatrixScatterSlotMap B{};
        MatrixScatterSlotMap Bt{};

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return A.estimated_memory_bytes() +
                   B.estimated_memory_bytes() +
                   Bt.estimated_memory_bytes();
        }
    };

    struct XTraceSlotMaps
    {
        MatrixScatterSlotMap C{};

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return C.estimated_memory_bytes();
        }
    };

    struct AssemblySpaceVersionKey2D
    {
        const void* space_ptr = nullptr;
        const void* mesh_ptr = nullptr;
        std::uint64_t mesh_storage_version = 0;
        std::uint64_t active_version = 0;
        std::uint64_t adjacency_version = 0;
        std::uint64_t dof_distribution_version = 0;
        int n_true_dofs = 0;
        std::size_t n_active_cells = 0;

        bool operator==(const AssemblySpaceVersionKey2D&) const noexcept =
            default;
    };

    template<class SpaceType>
    [[nodiscard]] AssemblySpaceVersionKey2D make_assembly_space_version_key_2d(
        const SpaceType& space)
    {
        AssemblySpaceVersionKey2D key;
        key.space_ptr = &space;
        key.mesh_ptr = &space.mesh_ref();
        if constexpr (requires { space.mesh_ref().storage_version(); })
            key.mesh_storage_version = space.mesh_ref().storage_version();
        if constexpr (requires { space.active_version(); })
            key.active_version = space.active_version();
        if constexpr (requires { space.adjacency_version(); })
            key.adjacency_version = space.adjacency_version();
        if constexpr (requires { space.dof_distribution_version(); })
            key.dof_distribution_version = space.dof_distribution_version();
        key.n_true_dofs = space.dof_handler_ref().n_true_dofs();
        key.n_active_cells = space.active_cells().size();
        return key;
    }

    struct TwoPassFullSaddleAssemblyCacheKey2D
    {
        AssemblySpaceVersionKey2D x_space{};
        AssemblySpaceVersionKey2D y_space{};
        int n_lambda = 0;
        int n_u = 0;
        int n_total = 0;

        bool operator==(const TwoPassFullSaddleAssemblyCacheKey2D&) const
            noexcept = default;
    };

    template<class XSpaceType, class YSpaceType>
    [[nodiscard]] TwoPassFullSaddleAssemblyCacheKey2D
    make_two_pass_full_saddle_cache_key_2d(
        const XSpaceType& x_space,
        const YSpaceType& y_space)
    {
        TwoPassFullSaddleAssemblyCacheKey2D key;
        key.x_space = make_assembly_space_version_key_2d(x_space);
        key.y_space = make_assembly_space_version_key_2d(y_space);
        key.n_lambda = y_space.dof_handler_ref().n_true_dofs();
        key.n_u = x_space.dof_handler_ref().n_true_dofs();
        key.n_total = key.n_lambda + key.n_u;
        return key;
    }

    template<class Backend, class XFESpaceType, class YFESpaceType>
    struct TwoPassFullSaddleAssemblyCache2D
    {
        using PatternBuilder = typename Backend::SparsePatternBuilder;

        TwoPassFullSaddleAssemblyCacheKey2D key{};
        bool valid = false;

        bool pattern_valid = false;
        PatternBuilder pattern_builder{};

        bool slot_map_estimate_valid = false;
        bool slot_maps_valid = false;
        bool use_scatter_slot_maps = false;
        std::size_t estimated_scatter_slot_map_bytes = 0;
        std::size_t scatter_slot_map_bytes = 0;

        std::vector<int> x_cell_ids{};
        std::vector<int> x_trace_cells{};
        std::vector<YCellSaddleSlotMaps> y_slot_maps{};
        std::vector<XTraceSlotMaps> x_trace_slot_maps{};

        std::size_t pattern_reuse_count = 0;
        std::size_t slot_map_reuse_count = 0;

        void reset_for_key(const TwoPassFullSaddleAssemblyCacheKey2D& new_key)
        {
            key = new_key;
            valid = true;
            pattern_valid = false;
            pattern_builder = PatternBuilder{};
            slot_map_estimate_valid = false;
            slot_maps_valid = false;
            use_scatter_slot_maps = false;
            estimated_scatter_slot_map_bytes = 0;
            scatter_slot_map_bytes = 0;
            x_cell_ids.clear();
            x_trace_cells.clear();
            std::vector<YCellSaddleSlotMaps>().swap(y_slot_maps);
            std::vector<XTraceSlotMaps>().swap(x_trace_slot_maps);
        }

        [[nodiscard]] bool matches(
            const TwoPassFullSaddleAssemblyCacheKey2D& query_key) const
            noexcept
        {
            return valid && key == query_key;
        }
    };

    template<class PatternBuilder>
    [[nodiscard]] MatrixScatterSlotMap build_matrix_scatter_slot_map(
        const PatternBuilder& builder,
        const LocalDofExpansion& test_dofs,
        const LocalDofExpansion& trial_dofs,
        int row_offset,
        int col_offset,
        bool transpose = false)
    {
        MatrixScatterSlotMap map;
        map.rows = test_dofs.size();
        map.cols = trial_dofs.size();
        map.offsets.assign(
            static_cast<std::size_t>(map.rows * map.cols) + 1U,
            0U);

        for (int i = 0; i < map.rows; ++i)
        {
            const auto& I = test_dofs[i];
            for (int j = 0; j < map.cols; ++j)
            {
                const auto& J = trial_dofs[j];
                const auto local_entry =
                    static_cast<std::size_t>(i * map.cols + j);
                map.offsets[local_entry] = map.entries.size();

                for (const auto& wi : I)
                {
                    if (wi.true_dof < 0)
                        continue;

                    for (const auto& wj : J)
                    {
                        if (wj.true_dof < 0)
                            continue;

                        const int row =
                            transpose
                                ? row_offset + wj.true_dof
                                : row_offset + wi.true_dof;
                        const int col =
                            transpose
                                ? col_offset + wi.true_dof
                                : col_offset + wj.true_dof;
                        map.entries.push_back(
                            WeightedSparseSlot{
                                builder.slot(row, col),
                                wi.weight * wj.weight});
                    }
                }
            }
        }
        map.offsets.back() = map.entries.size();
        return map;
    }

    template<class PatternBuilder, class LocalMatrixType>
    void scatter_matrix_by_slots(
        PatternBuilder& builder,
        const LocalMatrixType& local,
        const MatrixScatterSlotMap& slots,
        double zero_tol = 1e-15)
    {
        if (local.rows != slots.rows || local.cols != slots.cols)
        {
            throw std::runtime_error(
                "scatter_matrix_by_slots: local dimensions do not match slot map.");
        }

        for (int i = 0; i < local.rows; ++i)
        {
            for (int j = 0; j < local.cols; ++j)
            {
                const double a_ij = local(i, j);
                if (std::abs(a_ij) <= zero_tol)
                    continue;

                const auto local_entry =
                    static_cast<std::size_t>(i * local.cols + j);
                const std::size_t begin = slots.offsets[local_entry];
                const std::size_t end = slots.offsets[local_entry + 1U];
                for (std::size_t k = begin; k < end; ++k)
                {
                    const auto& entry = slots.entries[k];
                    builder.add_to_slot(entry.slot, entry.weight * a_ij);
                }
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType,
        class MFunction,
        class EllFunction,
        class InitialValueFunction>
    requires la::concepts::SparsePatternBuilderLike<
                 typename Backend::SparsePatternBuilder,
                 typename Backend::SparseMatrix> &&
             la::concepts::VectorLike<typename Backend::Vector>
    [[nodiscard]] la::linear::LinearSystem<Backend>
    assemble_main_full_saddle_two_pass_2d(
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        const EllFunction& ell,
        const InitialValueFunction& u0,
        double zero_tol,
        double g_scale,
        const finite_element::detail::TimingRecorder& timing,
        TwoPassFullSaddleAssemblyDiagnostics2D* diagnostics = nullptr,
        int numeric_fill_max_threads = 1,
        double numeric_fill_memory_budget_mb = 0.0,
        TwoPassFullSaddleAssemblyCache2D<Backend, XFESpaceType, YFESpaceType>*
            assembly_cache = nullptr)
    {
        using PatternBuilder = typename Backend::SparsePatternBuilder;
        using Vector = typename Backend::Vector;
        using XBasis =
            finite_element::basis::SpaceTimeBasis<
                typename XFESpaceType::GT,
                typename XFESpaceType::FETraitsType>;
        using YTables =
            detail::SpaceTimeBasisTables<
                typename YFESpaceType::GT,
                typename YFESpaceType::FETraitsType,
                QSpace,
                QTime>;
        using XTables =
            detail::SpaceTimeBasisTables<
                typename XFESpaceType::GT,
                typename XFESpaceType::FETraitsType,
                QSpace,
                QTime>;
        using XGeometry =
            finite_element::geometry::CellGeometry<XFESpaceType, 2>;
        using YGeometry =
            finite_element::geometry::CellGeometry<YFESpaceType, 2>;

        static_assert(
            XFESpaceType::GT::dim_space_v == 2 &&
            YFESpaceType::GT::dim_space_v == 2,
            "assemble_main_full_saddle_two_pass_2d is only for 2+1D.");

        auto total_alias_timer =
            timing.scoped(
                "main_system.direct_full_saddle_assembly.full_saddle_assembly_total");
        auto assembly_total_timer = timing.scoped("assembly.total_seconds");

        constexpr int x_dofs_per_cell =
            XFESpaceType::FETraitsType::dofs_per_cell;
        constexpr int y_dofs_per_cell =
            YFESpaceType::FETraitsType::dofs_per_cell;
        constexpr int n_cell_q = YTables::n_cell_q;
        constexpr int n_bottom_q = XTables::n_bottom_q;

        const auto& x_dof_handler = x_space.dof_handler_ref();
        const auto& y_dof_handler = y_space.dof_handler_ref();
        const auto& x_mesh = x_space.mesh_ref();
        const auto& y_active_cells = y_space.active_cells();
        const auto& x_active_cells = x_space.active_cells();
        const int n_lambda = y_dof_handler.n_true_dofs();
        const int n_u = x_dof_handler.n_true_dofs();
        const int n_total = n_lambda + n_u;
        const int n_y_active = static_cast<int>(y_active_cells.size());

        const auto assembly_cache_key =
            make_two_pass_full_saddle_cache_key_2d(x_space, y_space);
        const bool assembly_cache_enabled = assembly_cache != nullptr;
        const bool assembly_cache_key_reused =
            assembly_cache_enabled &&
            assembly_cache->matches(assembly_cache_key);
        if (assembly_cache != nullptr &&
            !assembly_cache_key_reused)
        {
            assembly_cache->reset_for_key(assembly_cache_key);
        }

        const auto x_restriction_hits_before =
            x_cache.cell_restriction_cache_hit_count();
        const auto y_restriction_hits_before =
            y_cache.cell_restriction_cache_hit_count();
        const auto x_geometry_hits_before =
            x_cache.geometry_cache_hit_count();
        const auto y_geometry_hits_before =
            y_cache.geometry_cache_hit_count();
        const auto x_geometry_misses_before =
            x_cache.geometry_cache_miss_count();
        const auto y_geometry_misses_before =
            y_cache.geometry_cache_miss_count();

        using Clock = std::chrono::steady_clock;
        const bool collect_timing = timing.enabled();
        double geometry_cache_seconds = 0.0;
        double cell_restriction_seconds = 0.0;
        double local_kernel_seconds = 0.0;
        double scatter_seconds = 0.0;
        double rhs_seconds = 0.0;
        std::atomic<double> assemble_A_y_diagnostic_seconds{0.0};
        std::atomic<double> assemble_B_diagnostic_seconds{0.0};
        std::atomic<double> assemble_C_signed_diagnostic_seconds{0.0};
        std::atomic<double> assemble_f_diagnostic_seconds{0.0};
        std::atomic<double> assemble_g_diagnostic_seconds{0.0};
        std::atomic<std::size_t> same_cell_pretabulated_x_gradient_cells{0u};
        std::atomic<std::size_t> coefficient_fast_path_identity_count{0u};
        std::atomic<std::size_t> coefficient_fast_path_zero_load_count{0u};
        std::atomic<std::size_t> coefficient_fast_path_generic_count{0u};
        std::atomic<std::size_t> diffusion_evaluations{0u};
        std::atomic<std::size_t> load_evaluations{0u};

        const bool identity_diffusion_fast_path =
            coefficients::is_identity_diffusion_function<2>(M);
        const auto constant_diffusion_tensor =
            coefficients::constant_diffusion_tensor_if_available<2>(M);
        const bool constant_diffusion_fast_path =
            constant_diffusion_tensor.has_value();
        const bool zero_load_fast_path =
            coefficients::is_zero_load_function(ell);
        const auto constant_M =
            constant_diffusion_fast_path
                ? *constant_diffusion_tensor
                : coefficients::identity_diffusion_tensor<2>();

        const auto add_elapsed =
            [](double& accumulator, Clock::time_point start)
            {
                accumulator +=
                    std::chrono::duration<double>(Clock::now() - start)
                        .count();
            };
        const auto add_atomic_elapsed =
            [](std::atomic<double>& accumulator, Clock::time_point start)
            {
                atomic_add_double(
                    accumulator,
                    std::chrono::duration<double>(Clock::now() - start)
                        .count());
            };

        std::vector<int> x_cell_ids;
        if (assembly_cache != nullptr &&
            assembly_cache->matches(assembly_cache_key) &&
            assembly_cache->x_cell_ids.size() ==
                static_cast<std::size_t>(n_y_active))
        {
            x_cell_ids = assembly_cache->x_cell_ids;
        }
        else
        {
            x_cell_ids.assign(static_cast<std::size_t>(n_y_active), -1);
            for (int item_index = 0; item_index < n_y_active; ++item_index)
            {
                const int y_cell_id =
                    y_active_cells[static_cast<std::size_t>(item_index)];
                const int x_cell_id =
                    detail::find_active_ancestor_cell(
                        ancestor_cache,
                        x_space,
                        y_space,
                        y_cell_id);
                x_cell_ids[static_cast<std::size_t>(item_index)] = x_cell_id;
            }
        }

        std::vector<int> x_trace_cells;
        if (assembly_cache != nullptr &&
            assembly_cache->matches(assembly_cache_key) &&
            !assembly_cache->x_trace_cells.empty())
        {
            x_trace_cells = assembly_cache->x_trace_cells;
        }
        else
        {
            x_trace_cells.reserve(x_active_cells.size());
            for (const int x_cell_id : x_active_cells)
            {
                if (x_mesh.cell(x_cell_id).temporal_boundary[0])
                    x_trace_cells.push_back(x_cell_id);
            }
        }

        if (assembly_cache != nullptr &&
            assembly_cache->matches(assembly_cache_key))
        {
            assembly_cache->x_cell_ids = x_cell_ids;
            assembly_cache->x_trace_cells = x_trace_cells;
        }

        for (int item_index = 0; item_index < n_y_active; ++item_index)
        {
            const int y_cell_id =
                y_active_cells[static_cast<std::size_t>(item_index)];
                const int x_cell_id = x_cell_ids[static_cast<std::size_t>(item_index)];
                if (identity_diffusion_fast_path)
                {
                    coefficient_fast_path_identity_count.fetch_add(
                        1u,
                        std::memory_order_relaxed);
                }
                else if (!constant_diffusion_fast_path)
                {
                    coefficient_fast_path_generic_count.fetch_add(
                        1u,
                        std::memory_order_relaxed);
                }
                if (zero_load_fast_path)
                {
                    coefficient_fast_path_zero_load_count.fetch_add(
                        1u,
                        std::memory_order_relaxed);
                }

                auto cache_start = Clock::now();
            (void)x_cache.dof_expansion(x_cell_id);
            add_elapsed(cell_restriction_seconds, cache_start);
            cache_start = Clock::now();
            (void)x_cache.geometry(x_cell_id);
            add_elapsed(geometry_cache_seconds, cache_start);
            cache_start = Clock::now();
            (void)y_cache.dof_expansion(y_cell_id);
            add_elapsed(cell_restriction_seconds, cache_start);
            cache_start = Clock::now();
            (void)y_cache.geometry(y_cell_id);
            add_elapsed(geometry_cache_seconds, cache_start);
        }
        for (const int x_cell_id : x_active_cells)
        {
            auto cache_start = Clock::now();
            (void)x_cache.dof_expansion(x_cell_id);
            add_elapsed(cell_restriction_seconds, cache_start);
            cache_start = Clock::now();
            (void)x_cache.geometry(x_cell_id);
            add_elapsed(geometry_cache_seconds, cache_start);
        }

        const std::size_t reserve_pattern =
            static_cast<std::size_t>(n_y_active) *
                (static_cast<std::size_t>(y_dofs_per_cell) *
                     static_cast<std::size_t>(y_dofs_per_cell) +
                 2u * static_cast<std::size_t>(x_dofs_per_cell) *
                     static_cast<std::size_t>(y_dofs_per_cell)) *
                4u +
            static_cast<std::size_t>(x_active_cells.size()) *
                static_cast<std::size_t>(x_dofs_per_cell) *
                static_cast<std::size_t>(x_dofs_per_cell) * 4u;

        PatternBuilder builder;
        bool pattern_reused = false;
        if (assembly_cache != nullptr &&
            assembly_cache->matches(assembly_cache_key) &&
            assembly_cache->pattern_valid)
        {
            builder = assembly_cache->pattern_builder;
            builder.zero_values();
            pattern_reused = true;
            ++assembly_cache->pattern_reuse_count;
        }
        else
        {
            builder.resize(n_total, n_total);
            builder.reserve_pattern(reserve_pattern);

            {
                auto timer =
                    timing.scoped("main_system.direct_full_saddle_assembly.pattern_build");
                auto requested_alias_timer =
                    timing.scoped("main_system.pattern_build_wall");
                auto alias_timer =
                    timing.scoped("assembly.pattern_build_seconds");
                for (int item_index = 0; item_index < n_y_active; ++item_index)
                {
                    const int y_cell_id =
                        y_active_cells[static_cast<std::size_t>(item_index)];
                    const int x_cell_id =
                        x_cell_ids[static_cast<std::size_t>(item_index)];

                    const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                    const auto& y_expanded = y_cache.dof_expansion(y_cell_id);

                    scatter_matrix_pattern(
                        builder,
                        y_expanded,
                        y_expanded,
                        0,
                        0);
                    scatter_matrix_pattern(
                        builder,
                        x_expanded,
                        y_expanded,
                        n_lambda,
                        0);
                    scatter_matrix_pattern_transpose(
                        builder,
                        x_expanded,
                        y_expanded,
                        0,
                        n_lambda);
                }

                for (const int x_cell_id : x_trace_cells)
                {
                    const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                    scatter_matrix_pattern(
                        builder,
                        x_expanded,
                        x_expanded,
                        n_lambda,
                        n_lambda);
                }
            }

            {
                auto timer =
                    timing.scoped("main_system.direct_full_saddle_assembly.pattern_finalization");
                auto alias_timer =
                    timing.scoped("assembly.pattern_finalize_seconds");
                builder.finalize_pattern();
                builder.zero_values();
            }

            if (assembly_cache != nullptr &&
                assembly_cache->matches(assembly_cache_key))
            {
                assembly_cache->pattern_builder = builder;
                assembly_cache->pattern_valid = true;
                assembly_cache->slot_maps_valid = false;
            }
        }
        if (pattern_reused)
            timing.add("main_system.pattern_build_wall", 0.0);

        Vector rhs(n_total);
        rhs.set_zero();

        std::vector<YCellSaddleSlotMaps> y_slot_maps_storage(
            static_cast<std::size_t>(n_y_active));
        std::vector<XTraceSlotMaps> x_trace_slot_maps_storage(
            x_trace_cells.size());
        const std::vector<YCellSaddleSlotMaps>* y_slot_maps =
            &y_slot_maps_storage;
        const std::vector<XTraceSlotMaps>* x_trace_slot_maps =
            &x_trace_slot_maps_storage;
        std::size_t scatter_slot_map_bytes = 0;
        std::size_t estimated_scatter_slot_map_bytes = 0;
        if (numeric_fill_memory_budget_mb > 0.0)
        {
            if (assembly_cache != nullptr &&
                assembly_cache->matches(assembly_cache_key) &&
                assembly_cache->slot_map_estimate_valid)
            {
                estimated_scatter_slot_map_bytes =
                    assembly_cache->estimated_scatter_slot_map_bytes;
            }
            else
            {
                for (int item_index = 0; item_index < n_y_active; ++item_index)
                {
                    const int y_cell_id =
                        y_active_cells[static_cast<std::size_t>(item_index)];
                    const int x_cell_id =
                        x_cell_ids[static_cast<std::size_t>(item_index)];
                    const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                    const auto& y_expanded = y_cache.dof_expansion(y_cell_id);
                    estimated_scatter_slot_map_bytes +=
                        estimate_matrix_scatter_slot_map_bytes(
                            y_expanded,
                            y_expanded);
                    estimated_scatter_slot_map_bytes +=
                        estimate_matrix_scatter_slot_map_bytes(
                            x_expanded,
                            y_expanded);
                    estimated_scatter_slot_map_bytes +=
                        estimate_matrix_scatter_slot_map_bytes(
                            x_expanded,
                            y_expanded);
                }
                for (const int x_cell_id : x_trace_cells)
                {
                    const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                    estimated_scatter_slot_map_bytes +=
                        estimate_matrix_scatter_slot_map_bytes(
                            x_expanded,
                            x_expanded);
                }

                if (assembly_cache != nullptr &&
                    assembly_cache->matches(assembly_cache_key))
                {
                    assembly_cache->estimated_scatter_slot_map_bytes =
                        estimated_scatter_slot_map_bytes;
                    assembly_cache->slot_map_estimate_valid = true;
                }
            }
        }

        const bool use_scatter_slot_maps =
            numeric_fill_memory_budget_mb > 0.0 &&
            static_cast<double>(estimated_scatter_slot_map_bytes) <=
                numeric_fill_memory_budget_mb * 1024.0 * 1024.0;

        bool slot_maps_reused = false;
        if (use_scatter_slot_maps &&
            assembly_cache != nullptr &&
            assembly_cache->matches(assembly_cache_key) &&
            assembly_cache->slot_maps_valid &&
            assembly_cache->use_scatter_slot_maps &&
            assembly_cache->y_slot_maps.size() ==
                static_cast<std::size_t>(n_y_active) &&
            assembly_cache->x_trace_slot_maps.size() == x_trace_cells.size())
        {
            y_slot_maps = &assembly_cache->y_slot_maps;
            x_trace_slot_maps = &assembly_cache->x_trace_slot_maps;
            scatter_slot_map_bytes = assembly_cache->scatter_slot_map_bytes;
            slot_maps_reused = true;
            ++assembly_cache->slot_map_reuse_count;
        }
        else if (use_scatter_slot_maps)
        {
            auto timer =
                timing.scoped("main_system.direct_full_saddle_assembly.slot_lookup_build");
            auto requested_alias_timer =
                timing.scoped("main_system.slot_lookup_wall");
            auto alias_timer = timing.scoped("assembly.slot_map_build_seconds");
            for (int item_index = 0; item_index < n_y_active; ++item_index)
            {
                const int y_cell_id =
                    y_active_cells[static_cast<std::size_t>(item_index)];
                const int x_cell_id =
                    x_cell_ids[static_cast<std::size_t>(item_index)];

                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                const auto& y_expanded = y_cache.dof_expansion(y_cell_id);
                auto& maps =
                    y_slot_maps_storage[static_cast<std::size_t>(item_index)];
                maps.A = build_matrix_scatter_slot_map(
                    builder,
                    y_expanded,
                    y_expanded,
                    0,
                    0);
                maps.B = build_matrix_scatter_slot_map(
                    builder,
                    x_expanded,
                    y_expanded,
                    n_lambda,
                    0);
                maps.Bt = build_matrix_scatter_slot_map(
                    builder,
                    x_expanded,
                    y_expanded,
                    0,
                    n_lambda,
                    true);
                scatter_slot_map_bytes += maps.estimated_memory_bytes();
            }

            for (
                int item = 0;
                item < static_cast<int>(x_trace_cells.size());
                ++item)
            {
                const int x_cell_id =
                    x_trace_cells[static_cast<std::size_t>(item)];
                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                auto& maps =
                    x_trace_slot_maps_storage[static_cast<std::size_t>(item)];
                maps.C = build_matrix_scatter_slot_map(
                    builder,
                    x_expanded,
                    x_expanded,
                    n_lambda,
                    n_lambda);
                scatter_slot_map_bytes += maps.estimated_memory_bytes();
            }

            if (assembly_cache != nullptr &&
                assembly_cache->matches(assembly_cache_key) &&
                pattern_reused)
            {
                assembly_cache->y_slot_maps = std::move(y_slot_maps_storage);
                assembly_cache->x_trace_slot_maps =
                    std::move(x_trace_slot_maps_storage);
                assembly_cache->scatter_slot_map_bytes = scatter_slot_map_bytes;
                assembly_cache->use_scatter_slot_maps = true;
                assembly_cache->slot_maps_valid = true;
                y_slot_maps = &assembly_cache->y_slot_maps;
                x_trace_slot_maps = &assembly_cache->x_trace_slot_maps;
            }
            else if (assembly_cache != nullptr &&
                     assembly_cache->matches(assembly_cache_key))
            {
                assembly_cache->slot_maps_valid = false;
                assembly_cache->use_scatter_slot_maps = false;
            }
        }
        if (!use_scatter_slot_maps || slot_maps_reused)
            timing.add("main_system.slot_lookup_wall", 0.0);

        const std::size_t y_numeric_thread_buffer_bytes =
            (static_cast<std::size_t>(y_dofs_per_cell) *
                 static_cast<std::size_t>(y_dofs_per_cell) +
             static_cast<std::size_t>(x_dofs_per_cell) *
                 static_cast<std::size_t>(y_dofs_per_cell) +
             static_cast<std::size_t>(y_dofs_per_cell)) *
            sizeof(double);
        const std::size_t x_trace_numeric_thread_buffer_bytes =
            (static_cast<std::size_t>(x_dofs_per_cell) *
                 static_cast<std::size_t>(x_dofs_per_cell) +
             static_cast<std::size_t>(x_dofs_per_cell)) *
            sizeof(double);
        const std::size_t numeric_thread_workspace_bytes =
            std::max(
                y_numeric_thread_buffer_bytes,
                x_trace_numeric_thread_buffer_bytes);
        int selected_numeric_threads =
            numeric_fill_max_threads > 0
                ? numeric_fill_max_threads
                : core::max_openmp_threads();
        selected_numeric_threads =
            std::max(
                1,
                std::min(
                    selected_numeric_threads,
                    core::max_openmp_threads()));
        if (numeric_fill_memory_budget_mb > 0.0)
        {
            const double budget_bytes =
                numeric_fill_memory_budget_mb * 1024.0 * 1024.0;
            while (
                selected_numeric_threads > 1 &&
                static_cast<double>(
                    static_cast<std::size_t>(selected_numeric_threads) *
                    numeric_thread_workspace_bytes) > budget_bytes)
            {
                --selected_numeric_threads;
            }
        }
        const std::size_t selected_numeric_thread_buffer_bytes =
            static_cast<std::size_t>(selected_numeric_threads) *
            numeric_thread_workspace_bytes;
        const bool record_numeric_subphase_timings =
            selected_numeric_threads <= 1;

        {
            auto timer =
                timing.scoped("main_system.direct_full_saddle_assembly.numeric_fill");
            auto requested_alias_timer =
                timing.scoped("main_system.numeric_fill_wall");
            auto alias_timer = timing.scoped("assembly.numeric_fill_seconds");

            const auto fill_y_item =
                [&](int item_index,
                    auto& local_A,
                    auto& local_B,
                    auto& local_f)
            {
                const int y_cell_id =
                    y_active_cells[static_cast<std::size_t>(item_index)];
                const int x_cell_id =
                    x_cell_ids[static_cast<std::size_t>(item_index)];

                auto cache_start = Clock::now();
                const auto& x_geom = x_cache.geometry(x_cell_id);
                if (record_numeric_subphase_timings)
                    add_elapsed(geometry_cache_seconds, cache_start);
                cache_start = Clock::now();
                const auto& y_geom = y_cache.geometry(y_cell_id);
                if (record_numeric_subphase_timings)
                    add_elapsed(geometry_cache_seconds, cache_start);
                cache_start = Clock::now();
                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                if (record_numeric_subphase_timings)
                    add_elapsed(cell_restriction_seconds, cache_start);
                cache_start = Clock::now();
                const auto& y_expanded = y_cache.dof_expansion(y_cell_id);
                if (record_numeric_subphase_timings)
                    add_elapsed(cell_restriction_seconds, cache_start);
                const bool same_cell_pretabulated_x_grads =
                    ADAPPARABOLICFEM_ENABLE_SAME_CELL_PRETABULATED_X_GRADS_2D &&
                    x_cell_id == y_cell_id;
                if (same_cell_pretabulated_x_grads)
                {
                    same_cell_pretabulated_x_gradient_cells.fetch_add(
                        1u,
                        std::memory_order_relaxed);
                }

                const auto kernel_start = Clock::now();
                detail::zero_local_matrix(local_A);
                detail::zero_local_matrix(local_B);
                detail::zero_local_vector(local_f);

                const double jac_y = YGeometry::jacobian_measure(y_geom);
                for (int q = 0; q < n_cell_q; ++q)
                {
                    const auto& xi_y = YTables::cell_rule.points[q];
                    const double dmu = jac_y * YTables::cell_rule.weights[q];
                    const auto& phi_vals = YTables::values_on_cell_qp(q);
                    const auto& phi_grads_ref =
                        YTables::gradients_on_cell_qp(q);

                    const auto x_q = YGeometry::map_to_physical(y_geom, xi_y);
                    const double ell_q =
                        zero_load_fast_path ? 0.0 : static_cast<double>(ell(x_q));
                    if (!zero_load_fast_path)
                    {
                        load_evaluations.fetch_add(
                            1u,
                            std::memory_order_relaxed);
                    }

                    std::array<
                        typename YGeometry::SpatialGradient,
                        y_dofs_per_cell> gradx_phi_on_q{};
                    std::array<
                        typename YGeometry::SpatialGradient,
                        y_dofs_per_cell> M_gradx_phi_on_q{};
                    auto M_q = constant_M;
                    if (!identity_diffusion_fast_path &&
                        !constant_diffusion_fast_path)
                    {
                        M_q =
                            coefficients::evaluate_diffusion_tensor<
                                MFunction,
                                2>(
                                M,
                                x_q);
                        diffusion_evaluations.fetch_add(
                            1u,
                            std::memory_order_relaxed);
                    }
                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        gradx_phi_on_q[static_cast<std::size_t>(j)] =
                            YGeometry::spatial_gradient(
                                y_geom,
                                phi_grads_ref[j]);
                        if (identity_diffusion_fast_path)
                        {
                            M_gradx_phi_on_q[static_cast<std::size_t>(j)] =
                                gradx_phi_on_q[static_cast<std::size_t>(j)];
                        }
                        else if (constant_diffusion_fast_path)
                        {
                            M_gradx_phi_on_q[static_cast<std::size_t>(j)] =
                                coefficients::apply_validated_M<2>(
                                    constant_M,
                                    gradx_phi_on_q[
                                        static_cast<std::size_t>(j)]);
                        }
                        else
                        {
                            M_gradx_phi_on_q[static_cast<std::size_t>(j)] =
                                coefficients::apply_validated_M<2>(
                                    M_q,
                                    gradx_phi_on_q[
                                        static_cast<std::size_t>(j)]);
                        }
                    }

                    std::array<double, x_dofs_per_cell> dt_psi_on_q{};
                    std::array<
                        typename XGeometry::SpatialGradient,
                        x_dofs_per_cell> gradx_psi_on_q{};
                    const auto fill_x_derivatives =
                        [&](const auto& psi_grads_ref)
                        {
                            for (int i = 0; i < x_dofs_per_cell; ++i)
                            {
                                dt_psi_on_q[static_cast<std::size_t>(i)] =
                                    XGeometry::time_derivative(
                                        x_geom,
                                        psi_grads_ref[i]);
                                gradx_psi_on_q[static_cast<std::size_t>(i)] =
                                    XGeometry::spatial_gradient(
                                        x_geom,
                                        psi_grads_ref[i]);
                            }
                        };
                    if (same_cell_pretabulated_x_grads)
                    {
                        fill_x_derivatives(XTables::gradients_on_cell_qp(q));
                    }
                    else
                    {
                        const auto xi_x =
                            XGeometry::physical_to_reference(x_geom, x_q);
                        const auto psi_grads_ref = XBasis::grad_all(xi_x);
                        fill_x_derivatives(psi_grads_ref);
                    }

                    const auto local_A_f_start = Clock::now();
                    for (int i = 0; i < y_dofs_per_cell; ++i)
                    {
                        const auto& grad_i =
                            gradx_phi_on_q[static_cast<std::size_t>(i)];
                        local_f[i] += ell_q * phi_vals[i] * dmu;
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local_A(i, j) +=
                                coefficients::dot<2>(
                                    grad_i,
                                    M_gradx_phi_on_q[
                                        static_cast<std::size_t>(j)]) *
                                dmu;
                        }
                    }
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_A_y_diagnostic_seconds,
                            local_A_f_start);

                    const auto local_B_start = Clock::now();
                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        const double dt_psi_i =
                            dt_psi_on_q[static_cast<std::size_t>(i)];
                        const auto& gradx_psi_i =
                            gradx_psi_on_q[static_cast<std::size_t>(i)];
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local_B(i, j) +=
                                (dt_psi_i * phi_vals[j] +
                                 coefficients::dot<2>(
                                     gradx_psi_i,
                                     M_gradx_phi_on_q[
                                         static_cast<std::size_t>(j)])) *
                                dmu;
                        }
                    }
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_B_diagnostic_seconds,
                            local_B_start);
                }
                if (record_numeric_subphase_timings)
                    add_elapsed(local_kernel_seconds, kernel_start);

                const auto scatter_start = Clock::now();
                if (use_scatter_slot_maps)
                {
                    const auto& slots =
                        (*y_slot_maps)[static_cast<std::size_t>(item_index)];
                    const auto scatter_A_start = Clock::now();
                    scatter_matrix_by_slots(
                        builder,
                        local_A,
                        slots.A,
                        zero_tol);
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_A_y_diagnostic_seconds,
                            scatter_A_start);
                    const auto scatter_B_start = Clock::now();
                    scatter_matrix_by_slots(
                        builder,
                        local_B,
                        slots.B,
                        zero_tol);
                    scatter_matrix_by_slots(
                        builder,
                        local_B,
                        slots.Bt,
                        zero_tol);
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_B_diagnostic_seconds,
                            scatter_B_start);
                }
                else
                {
                    const auto scatter_A_start = Clock::now();
                    scatter_matrix_offset(
                        builder,
                        local_A,
                        y_expanded,
                        y_expanded,
                        0,
                        0,
                        zero_tol);
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_A_y_diagnostic_seconds,
                            scatter_A_start);
                    const auto scatter_B_start = Clock::now();
                    scatter_matrix_offset(
                        builder,
                        local_B,
                        x_expanded,
                        y_expanded,
                        n_lambda,
                        0,
                        zero_tol);
                    scatter_matrix_transpose_offset(
                        builder,
                        local_B,
                        x_expanded,
                        y_expanded,
                        0,
                        n_lambda,
                        zero_tol);
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_B_diagnostic_seconds,
                            scatter_B_start);
                }
                if (record_numeric_subphase_timings)
                    add_elapsed(scatter_seconds, scatter_start);
                const auto rhs_start = Clock::now();
                scatter_vector_offset(
                    rhs,
                    local_f,
                    y_expanded,
                    0,
                    zero_tol);
                if (collect_timing)
                    add_atomic_elapsed(assemble_f_diagnostic_seconds, rhs_start);
                if (record_numeric_subphase_timings)
                    add_elapsed(rhs_seconds, rhs_start);
            };

            const auto fill_x_trace_cell =
                [&](int trace_item,
                    auto& local_C,
                    auto& local_g)
            {
                const int x_cell_id =
                    x_trace_cells[static_cast<std::size_t>(trace_item)];
                auto cache_start = Clock::now();
                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                if (record_numeric_subphase_timings)
                    add_elapsed(cell_restriction_seconds, cache_start);
                cache_start = Clock::now();
                const auto& x_geom = x_cache.geometry(x_cell_id);
                if (record_numeric_subphase_timings)
                    add_elapsed(geometry_cache_seconds, cache_start);
                const double trace_jac =
                    detail::initial_trace_measure<XGeometry>(x_geom);

                const auto kernel_start = Clock::now();
                detail::zero_local_matrix(local_C);
                detail::zero_local_vector(local_g);

                for (int q = 0; q < n_bottom_q; ++q)
                {
                    const auto& xi_bottom = XTables::bottom_rule.points[q];
                    const double dgamma =
                        trace_jac * XTables::bottom_rule.weights[q];
                    const auto& values = XTables::values_on_bottom_qp(q);
                    const auto x_q_st =
                        detail::map_bottom_qp_to_physical<XGeometry>(
                            x_geom,
                            xi_bottom);
                    const auto x_q =
                        detail::spatial_argument_from_space_time_point<
                            XGeometry>(x_q_st);
                    const double u0_q = static_cast<double>(u0(x_q));

                    const auto local_C_g_start = Clock::now();
                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        local_g[i] += g_scale * u0_q * values[i] * dgamma;
                        for (int j = 0; j < x_dofs_per_cell; ++j)
                            local_C(i, j) -=
                                values[i] * values[j] * dgamma;
                    }
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_C_signed_diagnostic_seconds,
                            local_C_g_start);
                }
                if (record_numeric_subphase_timings)
                    add_elapsed(local_kernel_seconds, kernel_start);

                const auto scatter_start = Clock::now();
                if (use_scatter_slot_maps)
                {
                    const auto& slots =
                        (*x_trace_slot_maps)[
                            static_cast<std::size_t>(trace_item)];
                    const auto scatter_C_start = Clock::now();
                    scatter_matrix_by_slots(
                        builder,
                        local_C,
                        slots.C,
                        zero_tol);
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_C_signed_diagnostic_seconds,
                            scatter_C_start);
                }
                else
                {
                    const auto scatter_C_start = Clock::now();
                    scatter_matrix_offset(
                        builder,
                        local_C,
                        x_expanded,
                        x_expanded,
                        n_lambda,
                        n_lambda,
                        zero_tol);
                    if (collect_timing)
                        add_atomic_elapsed(
                            assemble_C_signed_diagnostic_seconds,
                            scatter_C_start);
                }
                if (record_numeric_subphase_timings)
                    add_elapsed(scatter_seconds, scatter_start);
                const auto rhs_start = Clock::now();
                scatter_vector_offset(
                    rhs,
                    local_g,
                    x_expanded,
                    n_lambda,
                    zero_tol);
                if (collect_timing)
                    add_atomic_elapsed(assemble_g_diagnostic_seconds, rhs_start);
                if (record_numeric_subphase_timings)
                    add_elapsed(rhs_seconds, rhs_start);
            };

            if (selected_numeric_threads <= 1)
            {
                auto serial_timer =
                    timing.scoped(
                        "main_system.direct_full_saddle_assembly.numeric_fill_serial");

                la::local::FixedLocalMatrix<
                    y_dofs_per_cell,
                    y_dofs_per_cell> local_A;
                la::local::FixedLocalMatrix<
                    x_dofs_per_cell,
                    y_dofs_per_cell> local_B;
                la::local::FixedLocalVector<y_dofs_per_cell> local_f;
                for (int item_index = 0; item_index < n_y_active; ++item_index)
                    fill_y_item(item_index, local_A, local_B, local_f);

                la::local::FixedLocalMatrix<
                    x_dofs_per_cell,
                    x_dofs_per_cell> local_C;
                la::local::FixedLocalVector<x_dofs_per_cell> local_g;
                for (
                    int item = 0;
                    item < static_cast<int>(x_trace_cells.size());
                    ++item)
                {
                    fill_x_trace_cell(item, local_C, local_g);
                }
            }
            else
            {
                auto parallel_timer =
                    timing.scoped(
                        "main_system.direct_full_saddle_assembly.numeric_fill_parallel");

                std::vector<std::vector<int>> y_item_dofs(
                    static_cast<std::size_t>(n_y_active));
                for (int item_index = 0; item_index < n_y_active; ++item_index)
                {
                    const int y_cell_id =
                        y_active_cells[static_cast<std::size_t>(item_index)];
                    const int x_cell_id =
                        x_cell_ids[static_cast<std::size_t>(item_index)];
                    auto& dofs =
                        y_item_dofs[static_cast<std::size_t>(item_index)];
                    append_restriction_true_dofs(
                        dofs,
                        y_cache.dof_expansion(y_cell_id),
                        0);
                    append_restriction_true_dofs(
                        dofs,
                        x_cache.dof_expansion(x_cell_id),
                        n_lambda);
                    sort_unique_dofs(dofs);
                }
                const auto y_color_classes =
                    color_items_by_true_dofs(y_item_dofs, n_total);

                std::vector<std::vector<int>> x_trace_dofs(
                    x_trace_cells.size());
                for (
                    int item = 0;
                    item < static_cast<int>(x_trace_cells.size());
                    ++item)
                {
                    auto& dofs =
                        x_trace_dofs[static_cast<std::size_t>(item)];
                    append_restriction_true_dofs(
                        dofs,
                        x_cache.dof_expansion(
                            x_trace_cells[static_cast<std::size_t>(item)]),
                        n_lambda);
                    sort_unique_dofs(dofs);
                }
                const auto x_trace_color_classes =
                    color_items_by_true_dofs(x_trace_dofs, n_total);

                std::exception_ptr parallel_exception;
                std::atomic<bool> parallel_failed{false};

#pragma omp parallel num_threads(selected_numeric_threads)
                {
                    la::local::FixedLocalMatrix<
                        y_dofs_per_cell,
                        y_dofs_per_cell> local_A;
                    la::local::FixedLocalMatrix<
                        x_dofs_per_cell,
                        y_dofs_per_cell> local_B;
                    la::local::FixedLocalVector<y_dofs_per_cell> local_f;

                    for (const auto& color_items : y_color_classes)
                    {

#pragma omp for schedule(static)
                        for (
                            int k = 0;
                            k < static_cast<int>(color_items.size());
                            ++k)
                        {
                            if (parallel_failed.load(std::memory_order_relaxed))
                                continue;

                            try
                            {
                                fill_y_item(
                                    color_items[
                                        static_cast<std::size_t>(k)],
                                    local_A,
                                    local_B,
                                    local_f);
                            }
                            catch (...)
                            {
#pragma omp critical(apf_two_pass_numeric_fill_exception)
                                {
                                    if (!parallel_exception)
                                        parallel_exception =
                                            std::current_exception();
                                    parallel_failed.store(
                                        true,
                                        std::memory_order_relaxed);
                                }
                            }
                        }
                    }
                }
                if (parallel_exception)
                    std::rethrow_exception(parallel_exception);

#pragma omp parallel num_threads(selected_numeric_threads)
                {
                    la::local::FixedLocalMatrix<
                        x_dofs_per_cell,
                        x_dofs_per_cell> local_C;
                    la::local::FixedLocalVector<x_dofs_per_cell> local_g;

                    for (const auto& color_items : x_trace_color_classes)
                    {

#pragma omp for schedule(static)
                        for (
                            int k = 0;
                            k < static_cast<int>(color_items.size());
                            ++k)
                        {
                            if (parallel_failed.load(std::memory_order_relaxed))
                                continue;

                            try
                            {
                                fill_x_trace_cell(
                                    color_items[static_cast<std::size_t>(k)],
                                    local_C,
                                    local_g);
                            }
                            catch (...)
                            {
#pragma omp critical(apf_two_pass_numeric_fill_exception)
                                {
                                    if (!parallel_exception)
                                        parallel_exception =
                                            std::current_exception();
                                    parallel_failed.store(
                                        true,
                                        std::memory_order_relaxed);
                                }
                            }
                        }
                    }
                }
                if (parallel_exception)
                    std::rethrow_exception(parallel_exception);
            }
        }

        timing.add("assembly.geometry_cache_seconds", geometry_cache_seconds);
        timing.add(
            "assembly.cell_restriction_seconds",
            cell_restriction_seconds);
        timing.add("assembly.local_kernel_seconds", local_kernel_seconds);
        timing.add("assembly.scatter_seconds", scatter_seconds);
        timing.add("assembly.rhs_seconds", rhs_seconds);
        const double assemble_A_y_seconds =
            assemble_A_y_diagnostic_seconds.load(std::memory_order_relaxed);
        const double assemble_B_seconds =
            assemble_B_diagnostic_seconds.load(std::memory_order_relaxed);
        const double assemble_C_signed_seconds =
            assemble_C_signed_diagnostic_seconds.load(std::memory_order_relaxed);
        const double assemble_f_seconds =
            assemble_f_diagnostic_seconds.load(std::memory_order_relaxed);
        const double assemble_g_seconds =
            assemble_g_diagnostic_seconds.load(std::memory_order_relaxed);
        timing.add("main_system.assemble_A_y", assemble_A_y_seconds);
        timing.add("main_system.assemble_A_y_wall", assemble_A_y_seconds);
        timing.add("main_system.assemble_B", assemble_B_seconds);
        timing.add("main_system.assemble_B_wall", assemble_B_seconds);
        timing.add("main_system.assemble_C_signed", assemble_C_signed_seconds);
        timing.add(
            "main_system.assemble_C_signed_wall",
            assemble_C_signed_seconds);
        timing.add("main_system.assemble_f", assemble_f_seconds);
        timing.add("main_system.assemble_f_wall", assemble_f_seconds);
        timing.add("main_system.assemble_g", assemble_g_seconds);
        timing.add("main_system.assemble_g_wall", assemble_g_seconds);
        timing.add(
            "main_system.direct_full_saddle_assembly.assemble_A_y_diagnostic_work",
            assemble_A_y_seconds);
        timing.add(
            "main_system.direct_full_saddle_assembly.assemble_B_diagnostic_work",
            assemble_B_seconds);
        timing.add(
            "main_system.direct_full_saddle_assembly.assemble_C_signed_diagnostic_work",
            assemble_C_signed_seconds);
        timing.add(
            "main_system.direct_full_saddle_assembly.assemble_f_diagnostic_work",
            assemble_f_seconds);
        timing.add(
            "main_system.direct_full_saddle_assembly.assemble_g_diagnostic_work",
            assemble_g_seconds);
        timing.add("main_system.constraint_application_wall", 0.0);
        timing.add("main_system.constrained_dofs.count", 0.0);
        timing.add("main_system.rhs_solves.count", 1.0);
        timing.add("main_system.solver_reordering_strategy_code.count", 0.0);
        timing.add(
            "assembly.same_cell_pretabulated_x_gradient_cells.count",
            static_cast<double>(
                same_cell_pretabulated_x_gradient_cells.load(
                    std::memory_order_relaxed)));
        timing.add(
            "main_system.coefficient_fast_path.identity_count",
            static_cast<double>(
                coefficient_fast_path_identity_count.load(
                    std::memory_order_relaxed)));
        timing.add(
            "main_system.coefficient_fast_path.zero_load_count",
            static_cast<double>(
                coefficient_fast_path_zero_load_count.load(
                    std::memory_order_relaxed)));
        timing.add(
            "main_system.coefficient_fast_path.generic_count",
            static_cast<double>(
                coefficient_fast_path_generic_count.load(
                    std::memory_order_relaxed)));
        timing.add(
            "main_system.diffusion_evaluations",
            static_cast<double>(
                diffusion_evaluations.load(std::memory_order_relaxed)));
        timing.add(
            "main_system.load_evaluations",
            static_cast<double>(
                load_evaluations.load(std::memory_order_relaxed)));

        const std::size_t cell_restrictions_reused =
            (x_cache.cell_restriction_cache_hit_count() -
             x_restriction_hits_before) +
            (y_cache.cell_restriction_cache_hit_count() -
             y_restriction_hits_before);
        const std::size_t geometry_cache_hits =
            (x_cache.geometry_cache_hit_count() - x_geometry_hits_before) +
            (y_cache.geometry_cache_hit_count() - y_geometry_hits_before);
        const std::size_t geometry_cache_misses =
            (x_cache.geometry_cache_miss_count() - x_geometry_misses_before) +
            (y_cache.geometry_cache_miss_count() - y_geometry_misses_before);

        const std::size_t finalized_pattern_entries = builder.pattern_entries();
        const std::size_t finalized_pattern_candidates =
            [&]() -> std::size_t
            {
                if constexpr (requires { builder.pattern_candidate_count(); })
                    return builder.pattern_candidate_count();
                else
                    return finalized_pattern_entries;
            }();
        const std::size_t finalized_pattern_duplicates =
            [&]() -> std::size_t
            {
                if constexpr (requires { builder.pattern_duplicate_count(); })
                    return builder.pattern_duplicate_count();
                else
                    return finalized_pattern_candidates >
                                   finalized_pattern_entries
                               ? finalized_pattern_candidates -
                                     finalized_pattern_entries
                               : 0u;
            }();
        const std::size_t finalized_pattern_candidate_bytes =
            [&]() -> std::size_t
            {
                if constexpr (requires { builder.pattern_candidate_bytes(); })
                    return builder.pattern_candidate_bytes();
                else
                    return finalized_pattern_candidates *
                           sizeof(typename Backend::index_type);
            }();
        const std::size_t finalized_pattern_bytes = builder.pattern_bytes();
        const std::size_t finalized_compact_pattern_bytes =
            [&]() -> std::size_t
            {
                if constexpr (requires { builder.compact_pattern_bytes(); })
                    return builder.compact_pattern_bytes();
                else
                    return finalized_pattern_bytes;
            }();
        const std::size_t finalized_pattern_bytes_saved =
            [&]() -> std::size_t
            {
                if constexpr (requires { builder.pattern_bytes_saved(); })
                    return builder.pattern_bytes_saved();
                else
                    return finalized_pattern_candidate_bytes >
                                   finalized_compact_pattern_bytes
                               ? finalized_pattern_candidate_bytes -
                                     finalized_compact_pattern_bytes
                               : 0u;
            }();
        const std::size_t finalized_slot_map_bytes =
            builder.slot_map_bytes() + scatter_slot_map_bytes;
        auto matrix = builder.release_matrix();
        matrix.prune(1.0, zero_tol);

        la::linear::LinearSystem<Backend> out;
        out.matrix = std::move(matrix);
        out.rhs = std::move(rhs);
        out.solution = Vector(n_total);
        out.solution.set_zero();

        if (diagnostics != nullptr || timing.enabled())
        {
            std::size_t nnz_A_y = 0;
            std::size_t nnz_B = 0;
            std::size_t nnz_B_transpose = 0;
            std::size_t nnz_C_signed = 0;
            out.matrix.for_each_nonzero(
                [&](const int row, const int col, double)
                {
                    if (row < n_lambda && col < n_lambda)
                        ++nnz_A_y;
                    else if (row >= n_lambda && col < n_lambda)
                        ++nnz_B;
                    else if (row < n_lambda && col >= n_lambda)
                        ++nnz_B_transpose;
                    else
                        ++nnz_C_signed;
                });

            TwoPassFullSaddleAssemblyDiagnostics2D local_diagnostics;
            local_diagnostics.y_active_cells =
                static_cast<std::size_t>(n_y_active);
            for (const int x_cell_id : x_active_cells)
            {
                if (x_mesh.cell(x_cell_id).temporal_boundary[0])
                    ++local_diagnostics.x_initial_trace_cells;
            }
            local_diagnostics.pattern_candidates =
                finalized_pattern_candidates;
            local_diagnostics.pattern_entries = finalized_pattern_entries;
            local_diagnostics.pattern_duplicate_entries =
                finalized_pattern_duplicates;
            local_diagnostics.numeric_nonzeros =
                nnz_A_y + nnz_B + nnz_B_transpose + nnz_C_signed;
            local_diagnostics.nnz_A_y = nnz_A_y;
            local_diagnostics.nnz_B = nnz_B;
            local_diagnostics.nnz_B_transpose = nnz_B_transpose;
            local_diagnostics.nnz_C_signed = nnz_C_signed;
            local_diagnostics.pattern_candidate_bytes =
                finalized_pattern_candidate_bytes;
            local_diagnostics.pattern_bytes = finalized_pattern_bytes;
            local_diagnostics.compact_pattern_bytes =
                finalized_compact_pattern_bytes;
            local_diagnostics.pattern_bytes_saved =
                finalized_pattern_bytes_saved;
            local_diagnostics.numeric_matrix_bytes =
                detail::estimate_compressed_sparse_matrix_bytes(
                    out.matrix.rows(),
                    out.matrix.cols(),
                    local_diagnostics.numeric_nonzeros);
            local_diagnostics.slot_map_bytes = finalized_slot_map_bytes;
            local_diagnostics.triplet_bytes_avoided =
                local_diagnostics.pattern_entries *
                detail::estimated_triplet_bytes<typename Backend::SparseBuilder>();
            local_diagnostics.estimated_assembly_peak_bytes =
                local_diagnostics.pattern_bytes +
                local_diagnostics.numeric_matrix_bytes +
                local_diagnostics.slot_map_bytes;
            local_diagnostics.numeric_fill_threads =
                selected_numeric_threads;
            local_diagnostics.numeric_fill_thread_buffer_bytes =
                selected_numeric_thread_buffer_bytes;
            local_diagnostics.numeric_fill_conflict_strategy =
                selected_numeric_threads > 1 ? 1 : 0;
            local_diagnostics.numeric_fill_fallback_to_serial =
                selected_numeric_threads <= 1;
            local_diagnostics.pattern_reused = pattern_reused;
            local_diagnostics.slot_maps_reused = slot_maps_reused;
            local_diagnostics.assembly_cache_enabled = assembly_cache_enabled;
            local_diagnostics.assembly_cache_key_reused =
                assembly_cache_key_reused;
            local_diagnostics.slot_maps_enabled = use_scatter_slot_maps;
            local_diagnostics.cell_restrictions_reused =
                cell_restrictions_reused;
            local_diagnostics.geometry_cache_hits = geometry_cache_hits;
            local_diagnostics.geometry_cache_misses = geometry_cache_misses;
            local_diagnostics.coefficient_fast_path_identity_count =
                coefficient_fast_path_identity_count.load(
                    std::memory_order_relaxed);
            local_diagnostics.coefficient_fast_path_zero_load_count =
                coefficient_fast_path_zero_load_count.load(
                    std::memory_order_relaxed);
            local_diagnostics.coefficient_fast_path_generic_count =
                coefficient_fast_path_generic_count.load(
                    std::memory_order_relaxed);
            local_diagnostics.diffusion_evaluations =
                diffusion_evaluations.load(std::memory_order_relaxed);
            local_diagnostics.load_evaluations =
                load_evaluations.load(std::memory_order_relaxed);

            if (diagnostics != nullptr)
                *diagnostics = local_diagnostics;

            if (timing.enabled())
            {
                timing.add(
                    "main_system.direct_full_saddle_assembly.pattern_candidates.count",
                    static_cast<double>(
                        local_diagnostics.pattern_candidates));
                timing.add(
                    "main_system.direct_full_saddle_assembly.pattern_unique_entries.count",
                    static_cast<double>(
                        local_diagnostics.pattern_entries));
                const double duplicate_ratio =
                    local_diagnostics.pattern_entries == 0
                        ? 0.0
                        : static_cast<double>(
                              local_diagnostics.pattern_candidates) /
                              static_cast<double>(
                                  local_diagnostics.pattern_entries);
                timing.add(
                    "main_system.direct_full_saddle_assembly.pattern_duplicate_ratio",
                    duplicate_ratio);
                timing.add(
                    "main_system.sparse_memory.pattern_candidate_bytes",
                    static_cast<double>(
                        local_diagnostics.pattern_candidate_bytes));
                timing.add(
                    "main_system.sparse_memory.pattern_bytes",
                    static_cast<double>(local_diagnostics.pattern_bytes));
                timing.add(
                    "main_system.sparse_memory.compact_pattern_bytes",
                    static_cast<double>(
                        local_diagnostics.compact_pattern_bytes));
                timing.add(
                    "main_system.sparse_memory.pattern_bytes_saved",
                    static_cast<double>(
                        local_diagnostics.pattern_bytes_saved));
                timing.add(
                    "main_system.sparse_memory.numeric_matrix_bytes",
                    static_cast<double>(
                        local_diagnostics.numeric_matrix_bytes));
                timing.add(
                    "main_system.sparse_memory.slot_map_bytes",
                    static_cast<double>(local_diagnostics.slot_map_bytes));
                timing.add(
                    "main_system.sparse_memory.slot_map_estimated_bytes",
                    static_cast<double>(estimated_scatter_slot_map_bytes));
                timing.add(
                    "main_system.direct_full_saddle_assembly.slot_maps_enabled",
                    use_scatter_slot_maps ? 1.0 : 0.0);
                timing.add(
                    "main_system.assembly_cache_enabled",
                    local_diagnostics.assembly_cache_enabled ? 1.0 : 0.0);
                timing.add(
                    "main_system.assembly_cache_hit_count",
                    local_diagnostics.assembly_cache_key_reused ? 1.0 : 0.0);
                timing.add(
                    "main_system.assembly_cache_miss_count",
                    local_diagnostics.assembly_cache_enabled &&
                            !local_diagnostics.assembly_cache_key_reused
                        ? 1.0
                        : 0.0);
                timing.add(
                    "main_system.pattern_cache_hit_count",
                    local_diagnostics.pattern_reused ? 1.0 : 0.0);
                timing.add(
                    "main_system.pattern_cache_miss_count",
                    local_diagnostics.assembly_cache_enabled &&
                            !local_diagnostics.pattern_reused
                        ? 1.0
                        : 0.0);
                timing.add(
                    "main_system.slot_map_cache_hit_count",
                    local_diagnostics.slot_maps_reused ? 1.0 : 0.0);
                timing.add(
                    "main_system.slot_map_cache_miss_count",
                    local_diagnostics.assembly_cache_enabled &&
                            use_scatter_slot_maps &&
                            !local_diagnostics.slot_maps_reused
                        ? 1.0
                        : 0.0);
                timing.add(
                    "main_system.slot_maps_enabled",
                    local_diagnostics.slot_maps_enabled ? 1.0 : 0.0);
                timing.add(
                    "main_system.slot_map_memory_mb",
                    static_cast<double>(local_diagnostics.slot_map_bytes) /
                        (1024.0 * 1024.0));
                timing.add(
                    "main_system.sparse_memory.triplet_bytes_avoided",
                    static_cast<double>(
                        local_diagnostics.triplet_bytes_avoided));
                timing.add(
                    "main_system.sparse_memory.estimated_assembly_peak_bytes",
                    static_cast<double>(
                        local_diagnostics.estimated_assembly_peak_bytes));
                timing.add(
                    "main_system.sparse_memory.full_saddle_nnz_estimate.count",
                    static_cast<double>(
                        local_diagnostics.numeric_nonzeros));
                timing.add(
                    "main_system.nnz_A_y.count",
                    static_cast<double>(local_diagnostics.nnz_A_y));
                timing.add(
                    "main_system.nnz_B.count",
                    static_cast<double>(local_diagnostics.nnz_B));
                timing.add(
                    "main_system.nnz_B_transpose.count",
                    static_cast<double>(
                        local_diagnostics.nnz_B_transpose));
                timing.add(
                    "main_system.nnz_C_signed.count",
                    static_cast<double>(local_diagnostics.nnz_C_signed));
                timing.add(
                    "main_system.direct_full_saddle_assembly.nnz_A_y.count",
                    static_cast<double>(local_diagnostics.nnz_A_y));
                timing.add(
                    "main_system.direct_full_saddle_assembly.nnz_B.count",
                    static_cast<double>(local_diagnostics.nnz_B));
                timing.add(
                    "main_system.direct_full_saddle_assembly.nnz_C_signed.count",
                    static_cast<double>(local_diagnostics.nnz_C_signed));
                timing.add(
                    "main_system.sparse_memory.full_saddle_matrix_bytes",
                    static_cast<double>(
                        local_diagnostics.numeric_matrix_bytes));
                timing.add(
                    "main_system.sparse_memory.two_pass_direct_full_saddle",
                    1.0);
                timing.add(
                    "main_system.direct_full_saddle_assembly.numeric_fill_threads.count",
                    static_cast<double>(
                        local_diagnostics.numeric_fill_threads));
                timing.add(
                    "main_system.direct_full_saddle_assembly.numeric_fill_thread_buffer_bytes",
                    static_cast<double>(
                        local_diagnostics
                            .numeric_fill_thread_buffer_bytes));
                timing.add(
                    "main_system.direct_full_saddle_assembly.numeric_fill_conflict_strategy",
                    static_cast<double>(
                        local_diagnostics
                            .numeric_fill_conflict_strategy));
                timing.add(
                    "main_system.direct_full_saddle_assembly.numeric_fill_fallback_to_serial",
                    local_diagnostics.numeric_fill_fallback_to_serial
                        ? 1.0
                        : 0.0);
                timing.add(
                    "assembly.pattern_reused",
                    local_diagnostics.pattern_reused ? 1.0 : 0.0);
                timing.add(
                    "assembly.pattern_reused.count",
                    local_diagnostics.pattern_reused ? 1.0 : 0.0);
                timing.add(
                    "assembly.slot_maps_reused",
                    local_diagnostics.slot_maps_reused ? 1.0 : 0.0);
                timing.add(
                    "assembly.slot_maps_reused.count",
                    local_diagnostics.slot_maps_reused ? 1.0 : 0.0);
                timing.add(
                    "assembly.cell_restrictions_reused",
                    static_cast<double>(
                        local_diagnostics.cell_restrictions_reused));
                timing.add(
                    "assembly.cell_restrictions_reused.count",
                    static_cast<double>(
                        local_diagnostics.cell_restrictions_reused));
                timing.add(
                    "assembly.geometry_cache_hits",
                    static_cast<double>(
                        local_diagnostics.geometry_cache_hits));
                timing.add(
                    "assembly.geometry_cache_hits.count",
                    static_cast<double>(
                        local_diagnostics.geometry_cache_hits));
                timing.add(
                    "assembly.geometry_cache_misses",
                    static_cast<double>(
                        local_diagnostics.geometry_cache_misses));
                timing.add(
                    "assembly.geometry_cache_misses.count",
                    static_cast<double>(
                        local_diagnostics.geometry_cache_misses));
            }
        }

        return out;
    }
}

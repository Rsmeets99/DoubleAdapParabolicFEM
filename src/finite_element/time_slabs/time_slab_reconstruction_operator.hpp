#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../linear_algebra/assembly/local_objects.hpp"
#include "../../linear_algebra/concepts/solver.hpp"
#include "../../linear_algebra/operations/sparse_matrix_ops.hpp"
#include "../../linear_algebra/operations/vector_ops.hpp"
#include "../assembly/scatter.hpp"
#include "../assembly/detail/active_cell_locator_time_slab.hpp"
#include "../assembly/detail/assembly_diagnostics.hpp"
#include "../assembly/detail/assembly_space_cache.hpp"
#include "../assembly/detail/local_mixed_bilinear_forms.hpp"
#include "../assembly/detail/space_time_basis_tables.hpp"
#include "../assembly/detail/zero_local.hpp"
#include "../assembly/main_system/mat_A.hpp"
#include "../assembly/main_system/vec_f.hpp"
#include "../basis/space_time_basis_selector.hpp"
#include "../coefficients/diffusion_coefficient.hpp"
#include "time_slab_space.hpp"

#ifndef APF_FORCE_REFERENCE_SLAB_RECONSTRUCTION_2D
#define APF_FORCE_REFERENCE_SLAB_RECONSTRUCTION_2D 0
#endif

namespace finite_element::time_slabs
{
    template<class SpaceType>
    inline constexpr bool use_fixed_degree_slab_reconstruction_kernel_2d_v =
        SpaceType::GT::dim_space_v == 2 &&
        SpaceType::FETraitsType::p_space_v >= 1 &&
        SpaceType::FETraitsType::p_space_v <= 4 &&
        SpaceType::FETraitsType::p_time_v >= 1 &&
        SpaceType::FETraitsType::p_time_v <= 4 &&
        APF_FORCE_REFERENCE_SLAB_RECONSTRUCTION_2D == 0;

    template<class Backend, class XSpaceType, class SlabSpaceType>
    class TimeSlabReconstructionOperator
    {
    public:
        using VectorType       = typename Backend::Vector;
        using SparseMatrixType = typename Backend::SparseMatrix;
        using XSpace           = XSpaceType;
        using SlabSpace        = SlabSpaceType;
        using GT               = typename SlabSpace::GT;
        using FETraits         = typename SlabSpace::FETraitsType;
        using SlabType         = typename SlabSpace::SlabType;
        using LocalSlabSpace   = typename SlabType::SpaceType;

        struct SlabDiagnostics
        {
            int slab_id = -1;
            std::vector<int> source_cell_ids{};
            int matrix_rows = 0;
            int matrix_cols = 0;
            int rhs_size = 0;
            double rhs_inf_norm = 0.0;
            double solution_inf_norm = 0.0;
            double residual_inf_norm = 0.0;
            double assemble_A_seconds = 0.0;
            double assemble_rhs_seconds = 0.0;
            double solve_seconds = 0.0;
            double solver_setup_seconds = 0.0;
            double solver_apply_seconds = 0.0;
            double solver_symbolic_seconds = 0.0;
            double solver_numeric_seconds = 0.0;
            double solver_backsolve_seconds = 0.0;
            double shared_x_cache_construction_seconds = 0.0;
            double source_ancestor_cache_construction_seconds = 0.0;
            double slab_y_cache_construction_seconds = 0.0;
            double reconstruction_cache_bytes = 0.0;
            double pattern_build_seconds = 0.0;
            double numeric_fill_seconds = 0.0;
            double matrix_allocation_seconds = 0.0;
            double matrix_copy_seconds = 0.0;
            double reference_table_build_seconds = 0.0;
            double residual_check_seconds = 0.0;
            double source_permutation_check_seconds = 0.0;
            std::size_t pattern_candidates = 0;
            std::size_t pattern_entries = 0;
            std::size_t pattern_duplicate_entries = 0;
            std::size_t pattern_candidate_bytes = 0;
            std::size_t pattern_bytes = 0;
            std::size_t numeric_matrix_bytes = 0;
            std::size_t triplet_bytes_avoided = 0;
            double assemble_A_wall_begin_seconds = 0.0;
            double assemble_A_wall_end_seconds = 0.0;
            double assemble_rhs_wall_begin_seconds = 0.0;
            double assemble_rhs_wall_end_seconds = 0.0;
            double solve_wall_begin_seconds = 0.0;
            double solve_wall_end_seconds = 0.0;
            double f_assembly_wall_seconds = 0.0;
            double bt_u_assembly_wall_seconds = 0.0;
            double rhs_subtract_wall_seconds = 0.0;
            double function_update_wall_seconds = 0.0;
            double geometry_cache_wall_seconds = 0.0;
            double cell_restriction_wall_seconds = 0.0;
            double active_ancestor_lookup_wall_seconds = 0.0;
            double transfer_or_trace_wall_seconds = 0.0;
            std::size_t qpoints_visited = 0;
            std::size_t slab_cells_visited = 0;
            std::size_t geometry_cache_hits = 0;
            std::size_t geometry_cache_misses = 0;
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                assemble_A_diagnostics{};
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                rhs_diagnostics{};
            int operator_mode_identity_zero_load_count = 0;
            int operator_mode_constant_diffusion_count = 0;
            int operator_mode_generic_variable_count = 0;
            std::size_t diffusion_evaluations = 0;
            std::size_t load_evaluations = 0;
            int local_A_fast_path_count = 0;
            int local_A_generic_count = 0;
            int rhs_zero_load_fast_path_count = 0;
            double local_A_debug_max_abs_diff = 0.0;
            double rhs_debug_max_abs_diff = 0.0;
            int reconstruction_A_cache_hits = 0;
            int reconstruction_A_cache_misses = 0;
            int reconstruction_factor_cache_hits = 0;
            int reconstruction_factor_cache_misses = 0;
            int reused_symbolic_count = 0;
            int reused_sparsity_pattern_count = 0;
            int reused_operator_structure_count = 0;
            int residual_check_count = 0;
            int source_permutation_check_count = 0;
            // Negative means the source space was not already in slab format.
            double source_slab_permutation_error = -1.0;
        };

        TimeSlabReconstructionOperator(
            const XSpace& x_space,
            const SlabSpace& slab_space)
            : x_space_(&x_space),
              slab_space_(&slab_space)
        {}

        void set_slab_reconstruction_operator_mode(std::string mode)
        {
            if (mode != "auto" &&
                mode != "identity_zero_load_fast_path" &&
                mode != "constant_diffusion_fast_path" &&
                mode != "generic_variable_path")
            {
                throw std::runtime_error(
                    "TimeSlabReconstructionOperator: unsupported slab "
                    "reconstruction operator mode '" +
                    mode +
                    "'. Expected auto, identity_zero_load_fast_path, "
                    "constant_diffusion_fast_path, or generic_variable_path.");
            }

            slab_reconstruction_operator_mode_ = std::move(mode);
        }

        [[nodiscard]] const std::string&
        slab_reconstruction_operator_mode() const noexcept
        {
            return slab_reconstruction_operator_mode_;
        }

        template<int QSpace, int QTime, class MFunction>
        void assemble_mat_A(
            SparseMatrixType& A_slab,
            int slab_id,
            const MFunction& M,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            double zero_tol = 1.0e-15,
            finite_element::assembly::detail::AssemblyKernelDiagnostics*
                diagnostics = nullptr) const
        {
            constexpr bool use_fixed_degree_2d_kernel =
                use_fixed_degree_slab_reconstruction_kernel_2d_v<
                    LocalSlabSpace>;

            if constexpr (use_fixed_degree_2d_kernel)
            {
                assemble_mat_A_fast_2d_<QSpace, QTime>(
                    A_slab,
                    slab_id,
                    M,
                    y_cache,
                    zero_tol,
                    diagnostics);
            }
            else
            {
                finite_element::assembly::assemble_mat_A<
                    QSpace,
                    QTime,
                    Backend>(
                        A_slab,
                        slab_space_->slab(slab_id).fespace_ref(),
                        M,
                        y_cache,
                        zero_tol,
                        diagnostics);
            }
        }

        template<int QSpace, int QTime, class XFunctionType, class MFunction>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_vec_BT_u_direct(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>&
                ancestor_cache,
            const MFunction& M,
            double zero_tol = 1.0e-15,
            finite_element::assembly::detail::AssemblyKernelDiagnostics*
                diagnostics = nullptr,
            SlabDiagnostics* slab_diagnostics = nullptr) const
        {
            using XBasis =
                finite_element::basis::SpaceTimeBasis<
                    typename XSpace::GT,
                    typename XSpace::FETraitsType>;
            using YTables =
                finite_element::assembly::detail::SpaceTimeBasisTables<
                    typename LocalSlabSpace::GT,
                    typename LocalSlabSpace::FETraitsType,
                    QSpace,
                    QTime>;
            using XGeometry =
                finite_element::geometry::CellGeometry<
                    XSpace,
                    XSpace::GT::dim_space_v>;
            using YGeometry =
                finite_element::geometry::CellGeometry<
                    LocalSlabSpace,
                    LocalSlabSpace::GT::dim_space_v>;
            using XCache =
                finite_element::assembly::detail::AssemblySpaceCache<XSpace>;
            using YCache =
                finite_element::assembly::detail::
                    AssemblySpaceCache<LocalSlabSpace>;
            using XGeometryData = typename XCache::GeometryData;
            using YGeometryData = typename YCache::GeometryData;

            static_assert(
                XSpace::GT::dim_space_v == LocalSlabSpace::GT::dim_space_v,
                "TimeSlabReconstructionOperator requires matching dimensions.");

            constexpr int x_dofs_per_cell =
                XSpace::FETraitsType::dofs_per_cell;
            constexpr int y_dofs_per_cell =
                LocalSlabSpace::FETraitsType::dofs_per_cell;

            const auto& slab = slab_space_->slab(slab_id);
            const auto& y_space = slab.fespace_ref();
            const auto& y_dof_handler = y_space.dof_handler_ref();
            const auto& u_true = u_delta.true_coefficients();
            double active_ancestor_lookup_seconds = 0.0;
            double geometry_cache_seconds = 0.0;
            double cell_restriction_seconds = 0.0;
            double transfer_or_trace_seconds = 0.0;
            std::size_t slab_cell_visits = 0;

            bt_u_slab.resize(y_dof_handler.n_true_dofs());
            bt_u_slab.set_zero();

            la::local::FixedLocalMatrix<
                x_dofs_per_cell,
                y_dofs_per_cell> local;
            std::array<double, static_cast<std::size_t>(x_dofs_per_cell)>
                x_coefficients{};

            for (const int slab_cell_id : slab.active_cells())
            {
                ++slab_cell_visits;
                const int source_y_cell_id = slab.source_cell_id(slab_cell_id);
                int x_cell_id = -1;
                if (slab_diagnostics != nullptr)
                {
                    const auto t0 = Clock::now();
                    x_cell_id =
                        finite_element::assembly::detail::
                            find_active_ancestor_cell_from_source_cell(
                                ancestor_cache,
                                *x_space_,
                                source_y_cell_id);
                    const auto t1 = Clock::now();
                    active_ancestor_lookup_seconds += seconds_(t0, t1);
                }
                else
                {
                    x_cell_id =
                        finite_element::assembly::detail::
                            find_active_ancestor_cell_from_source_cell(
                                ancestor_cache,
                                *x_space_,
                                source_y_cell_id);
                }

                const XGeometryData* x_geom_ptr = nullptr;
                const YGeometryData* y_geom_ptr = nullptr;
                if (slab_diagnostics != nullptr)
                {
                    const auto t0 = Clock::now();
                    x_geom_ptr = &x_cache.geometry(x_cell_id);
                    y_geom_ptr = &y_cache.geometry(slab_cell_id);
                    const auto t1 = Clock::now();
                    geometry_cache_seconds += seconds_(t0, t1);
                }
                else
                {
                    x_geom_ptr = &x_cache.geometry(x_cell_id);
                    y_geom_ptr = &y_cache.geometry(slab_cell_id);
                }
                const auto& x_geom = *x_geom_ptr;
                const auto& y_geom = *y_geom_ptr;

                const finite_element::assembly::LocalDofExpansion*
                    x_expanded_ptr = nullptr;
                const finite_element::assembly::LocalDofExpansion*
                    y_expanded_ptr = nullptr;
                if (slab_diagnostics != nullptr)
                {
                    const auto t0 = Clock::now();
                    x_expanded_ptr = &x_cache.dof_expansion(x_cell_id);
                    y_expanded_ptr = &y_cache.dof_expansion(slab_cell_id);
                    const auto t1 = Clock::now();
                    cell_restriction_seconds += seconds_(t0, t1);
                }
                else
                {
                    x_expanded_ptr = &x_cache.dof_expansion(x_cell_id);
                    y_expanded_ptr = &y_cache.dof_expansion(slab_cell_id);
                }
                const auto& x_expanded = *x_expanded_ptr;
                const auto& y_expanded = *y_expanded_ptr;

                if (slab_diagnostics != nullptr)
                {
                    const auto t0 = Clock::now();
                    finite_element::assembly::detail::
                        assemble_local_mixed_space_time_matrix_part<
                            finite_element::assembly::detail::
                                MixedBilinearFormPart::Full,
                            XBasis,
                            YTables,
                            XGeometry,
                            YGeometry>(
                                local,
                                x_geom,
                                y_geom,
                                M);
                    const auto t1 = Clock::now();
                    transfer_or_trace_seconds += seconds_(t0, t1);
                }
                else
                {
                    finite_element::assembly::detail::
                        assemble_local_mixed_space_time_matrix_part<
                            finite_element::assembly::detail::
                                MixedBilinearFormPart::Full,
                            XBasis,
                            YTables,
                            XGeometry,
                            YGeometry>(
                                local,
                                x_geom,
                                y_geom,
                                M);
                }

                if (slab_diagnostics != nullptr)
                {
                    const auto t0 = Clock::now();
                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        double value = 0.0;
                        for (const auto& wi : x_expanded[i])
                        {
                            if (wi.true_dof >= 0)
                                value += wi.weight * u_true[wi.true_dof];
                        }
                        x_coefficients[static_cast<std::size_t>(i)] = value;
                    }
                    const auto t1 = Clock::now();
                    transfer_or_trace_seconds += seconds_(t0, t1);
                }
                else
                {
                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        double value = 0.0;
                        for (const auto& wi : x_expanded[i])
                        {
                            if (wi.true_dof >= 0)
                                value += wi.weight * u_true[wi.true_dof];
                        }
                        x_coefficients[static_cast<std::size_t>(i)] = value;
                    }
                }

                for (int j = 0; j < y_dofs_per_cell; ++j)
                {
                    double local_action = 0.0;
                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        local_action +=
                            local(i, j) *
                            x_coefficients[static_cast<std::size_t>(i)];
                    }

                    if (std::abs(local_action) <= zero_tol)
                        continue;

                    for (const auto& wj : y_expanded[j])
                    {
                        if (wj.true_dof >= 0)
                            bt_u_slab.add(
                                wj.true_dof,
                                wj.weight * local_action);
                    }
                }
            }

            if (diagnostics != nullptr)
            {
                diagnostics->active_cells += slab.active_cells().size();
                diagnostics->quadrature_points +=
                    slab.active_cells().size() *
                    static_cast<std::size_t>(YTables::n_cell_q);
                diagnostics->scalar_basis_evaluations +=
                    slab.active_cells().size() *
                    static_cast<std::size_t>(YTables::n_cell_q) *
                    static_cast<std::size_t>(y_dofs_per_cell);
                diagnostics->gradient_evaluations +=
                    slab.active_cells().size() *
                    static_cast<std::size_t>(YTables::n_cell_q) *
                    static_cast<std::size_t>(
                        x_dofs_per_cell + y_dofs_per_cell);
                diagnostics->diffusion_tensor_evaluations +=
                    slab.active_cells().size() *
                    static_cast<std::size_t>(YTables::n_cell_q);
            }

            if (slab_diagnostics != nullptr)
            {
                slab_diagnostics->active_ancestor_lookup_wall_seconds +=
                    active_ancestor_lookup_seconds;
                slab_diagnostics->geometry_cache_wall_seconds +=
                    geometry_cache_seconds;
                slab_diagnostics->cell_restriction_wall_seconds +=
                    cell_restriction_seconds;
                slab_diagnostics->transfer_or_trace_wall_seconds +=
                    transfer_or_trace_seconds;
                slab_diagnostics->slab_cells_visited += slab_cell_visits;
                slab_diagnostics->qpoints_visited +=
                    slab_cell_visits *
                    static_cast<std::size_t>(YTables::n_cell_q);
            }
        }

        template<int QSpace, int QTime, class XFunctionType>
        void assemble_vec_BT_u_direct(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>&
                ancestor_cache,
            double zero_tol = 1.0e-15) const
        {
            assemble_vec_BT_u_direct<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                x_cache,
                y_cache,
                ancestor_cache,
                coefficients::IdentityDiffusion<GT::dim_space_v>{},
                zero_tol,
                nullptr);
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_rhs(
            VectorType& rhs_slab,
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>&
                ancestor_cache,
            double zero_tol = 1.0e-15,
            finite_element::assembly::detail::AssemblyKernelDiagnostics*
                diagnostics = nullptr,
            SlabDiagnostics* slab_diagnostics = nullptr) const
        {
            constexpr bool use_fixed_degree_2d_kernel =
                use_fixed_degree_slab_reconstruction_kernel_2d_v<
                    LocalSlabSpace>;

            if constexpr (use_fixed_degree_2d_kernel)
            {
                assemble_rhs_fast_2d_<QSpace, QTime>(
                    rhs_slab,
                    slab_id,
                    M,
                    ell,
                    u_delta,
                    y_cache,
                    x_cache,
                    ancestor_cache,
                    zero_tol,
                    diagnostics);
            }
            else
            {
                VectorType f_slab;
                VectorType bt_u_slab;
                finite_element::assembly::detail::AssemblyKernelDiagnostics
                    f_diagnostics;
                finite_element::assembly::detail::AssemblyKernelDiagnostics
                    bt_u_diagnostics;

                const auto t_f0 = Clock::now();
                finite_element::assembly::assemble_vec_f<QSpace, QTime>(
                    f_slab,
                    slab_space_->slab(slab_id).fespace_ref(),
                    ell,
                    y_cache,
                    zero_tol,
                    &f_diagnostics);
                const auto t_f1 = Clock::now();

                const auto t_bt0 = Clock::now();
                assemble_vec_BT_u_direct<QSpace, QTime>(
                    bt_u_slab,
                    slab_id,
                    u_delta,
                    x_cache,
                    y_cache,
                    ancestor_cache,
                    M,
                    zero_tol,
                    &bt_u_diagnostics,
                    slab_diagnostics);
                const auto t_bt1 = Clock::now();

                if (diagnostics != nullptr)
                {
                    diagnostics->add(f_diagnostics);
                    diagnostics->add(bt_u_diagnostics);
                }

                if (f_slab.size() != bt_u_slab.size())
                {
                    throw std::runtime_error(
                        "TimeSlabReconstructionOperator::assemble_rhs: size mismatch.");
                }

                const auto t_subtract0 = Clock::now();
                rhs_slab.resize(f_slab.size());
                for (int i = 0; i < rhs_slab.size(); ++i)
                    rhs_slab[i] = f_slab[i] - bt_u_slab[i];
                const auto t_subtract1 = Clock::now();

                if (slab_diagnostics != nullptr)
                {
                    slab_diagnostics->f_assembly_wall_seconds +=
                        seconds_(t_f0, t_f1);
                    slab_diagnostics->bt_u_assembly_wall_seconds +=
                        seconds_(t_bt0, t_bt1);
                    slab_diagnostics->rhs_subtract_wall_seconds +=
                        seconds_(t_subtract0, t_subtract1);
                    slab_diagnostics->slab_cells_visited +=
                        f_diagnostics.active_cells;
                    slab_diagnostics->qpoints_visited +=
                        f_diagnostics.quadrature_points;
                }
            }
        }

        template<int QSpace, int QTime, class EllFunction, class XFunctionType>
        void assemble_rhs(
            VectorType& rhs_slab,
            int slab_id,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>&
                ancestor_cache,
            double zero_tol = 1.0e-15) const
        {
            assemble_rhs<QSpace, QTime>(
                rhs_slab,
                slab_id,
                coefficients::IdentityDiffusion<GT::dim_space_v>{},
                ell,
                u_delta,
                y_cache,
                x_cache,
                ancestor_cache,
                zero_tol,
                nullptr);
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType,
            class SolverType,
            class LocalFunctionType>
        SlabDiagnostics solve_slab(
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            LocalFunctionType& out_function,
            double zero_tol = 1.0e-15) const
        {
            const auto t_x0 = Clock::now();
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>
                x_cache(*x_space_);
            const auto t_x1 = Clock::now();

            const auto t_ancestor0 = Clock::now();
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>
                ancestor_cache(*x_space_);
            const auto t_ancestor1 = Clock::now();

            const std::size_t shared_cache_bytes =
                x_cache.estimated_live_bytes() +
                ancestor_cache.estimated_live_bytes();

            return solve_slab_with_shared_caches<QSpace, QTime>(
                slab_id,
                M,
                ell,
                u_delta,
                solver,
                options,
                out_function,
                x_cache,
                ancestor_cache,
                seconds_(t_x0, t_x1),
                seconds_(t_ancestor0, t_ancestor1),
                shared_cache_bytes,
                zero_tol);
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType,
            class SolverType,
            class LocalFunctionType>
        SlabDiagnostics solve_slab_with_shared_caches(
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            LocalFunctionType& out_function,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>&
                x_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>&
                ancestor_cache,
            double shared_x_cache_construction_seconds,
            double source_ancestor_cache_construction_seconds,
            std::size_t shared_cache_bytes,
            double zero_tol = 1.0e-15) const
        {
            SlabDiagnostics diagnostics;
            diagnostics.slab_id = slab_id;
            diagnostics.source_cell_ids = source_cells_for_slab_(slab_id);
            diagnostics.shared_x_cache_construction_seconds =
                shared_x_cache_construction_seconds;
            diagnostics.source_ancestor_cache_construction_seconds =
                source_ancestor_cache_construction_seconds;
            diagnostics.reconstruction_A_cache_misses = 1;
            const std::size_t x_geometry_hits_before =
                x_cache.geometry_cache_hit_count();
            const std::size_t x_geometry_misses_before =
                x_cache.geometry_cache_miss_count();

            const auto& y_slab_space = slab_space_->slab(slab_id).fespace_ref();
            const auto t_y_cache0 = Clock::now();
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpace>
                y_cache(y_slab_space);
            const auto t_y_cache1 = Clock::now();
            diagnostics.slab_y_cache_construction_seconds =
                seconds_(t_y_cache0, t_y_cache1);
            diagnostics.reconstruction_cache_bytes =
                static_cast<double>(
                    shared_cache_bytes + y_cache.estimated_live_bytes());

            SparseMatrixType A_slab;
            VectorType rhs_slab;
            VectorType lambda_slab;

            constexpr bool use_shared_state_2d_kernel =
                use_fixed_degree_slab_reconstruction_kernel_2d_v<
                    LocalSlabSpace>;

            Clock::time_point t_A0{};
            Clock::time_point t_A1{};
            Clock::time_point t_rhs0{};
            Clock::time_point t_rhs1{};
            if constexpr (use_shared_state_2d_kernel)
            {
                t_A0 = Clock::now();
                diagnostics.assemble_A_wall_begin_seconds =
                    absolute_seconds_(t_A0);
                assemble_mat_A_and_rhs_fast_2d_shared_state_<
                    QSpace,
                    QTime>(
                    A_slab,
                    rhs_slab,
                    slab_id,
                    M,
                    ell,
                    u_delta,
                    y_cache,
                    x_cache,
                    ancestor_cache,
                    zero_tol,
                    &diagnostics.assemble_A_diagnostics,
                    &diagnostics.rhs_diagnostics,
                    &diagnostics);
                t_A1 = Clock::now();
                t_rhs0 = t_A1;
                t_rhs1 = t_A1;
                diagnostics.assemble_A_wall_end_seconds =
                    absolute_seconds_(t_A1);
                diagnostics.assemble_rhs_wall_begin_seconds =
                    absolute_seconds_(t_rhs0);
                diagnostics.assemble_rhs_wall_end_seconds =
                    absolute_seconds_(t_rhs1);
            }
            else
            {
                t_A0 = Clock::now();
                diagnostics.assemble_A_wall_begin_seconds =
                    absolute_seconds_(t_A0);
                assemble_mat_A<QSpace, QTime>(
                    A_slab,
                    slab_id,
                    M,
                    y_cache,
                    zero_tol,
                    &diagnostics.assemble_A_diagnostics);
                t_A1 = Clock::now();
                diagnostics.assemble_A_wall_end_seconds =
                    absolute_seconds_(t_A1);

                t_rhs0 = Clock::now();
                diagnostics.assemble_rhs_wall_begin_seconds =
                    absolute_seconds_(t_rhs0);
                assemble_rhs<QSpace, QTime>(
                    rhs_slab,
                    slab_id,
                    M,
                    ell,
                    u_delta,
                    y_cache,
                    x_cache,
                    ancestor_cache,
                    zero_tol,
                    &diagnostics.rhs_diagnostics,
                    &diagnostics);
                t_rhs1 = Clock::now();
                diagnostics.assemble_rhs_wall_end_seconds =
                    absolute_seconds_(t_rhs1);
            }

            const auto t_solve0 = Clock::now();
            diagnostics.solve_wall_begin_seconds =
                absolute_seconds_(t_solve0);
            const auto t_setup0 = Clock::now();
            solver.compute(A_slab, options);
            const auto t_setup1 = Clock::now();
            lambda_slab.resize(rhs_slab.size());
            const auto t_apply0 = Clock::now();
            solver.solve(rhs_slab, lambda_slab);
            const auto t_apply1 = Clock::now();
            const auto t_solve1 = Clock::now();
            diagnostics.solve_wall_end_seconds =
                absolute_seconds_(t_solve1);
            const auto& solver_diagnostics = solver.last_diagnostics();
            diagnostics.solver_setup_seconds = seconds_(t_setup0, t_setup1);
            diagnostics.solver_apply_seconds = seconds_(t_apply0, t_apply1);
            diagnostics.solver_symbolic_seconds =
                solver_diagnostics.direct_stats.symbolic_analysis_seconds
                    .value_or(0.0);
            diagnostics.solver_numeric_seconds =
                solver_diagnostics.direct_stats.numeric_factorization_seconds
                    .value_or(0.0);
            diagnostics.solver_backsolve_seconds =
                solver_diagnostics.direct_stats.backsolve_seconds.value_or(0.0);
            const bool symbolic_reused =
                solver_diagnostics.direct_stats.symbolic_analysis_reused;
            diagnostics.reused_symbolic_count = symbolic_reused ? 1 : 0;
            diagnostics.reconstruction_factor_cache_hits =
                symbolic_reused ? 1 : 0;
            diagnostics.reconstruction_factor_cache_misses =
                symbolic_reused ? 0 : 1;

            const auto t_update0 = Clock::now();
            out_function.update_from_true_solution(lambda_slab);
            const auto t_update1 = Clock::now();

            diagnostics.matrix_rows = A_slab.rows();
            diagnostics.matrix_cols = A_slab.cols();
            diagnostics.rhs_size = rhs_slab.size();
            diagnostics.rhs_inf_norm = la::ops::inf_norm(rhs_slab);
            diagnostics.solution_inf_norm = la::ops::inf_norm(lambda_slab);
            diagnostics.assemble_A_seconds = seconds_(t_A0, t_A1);
            diagnostics.assemble_rhs_seconds = seconds_(t_rhs0, t_rhs1);
            diagnostics.solve_seconds = seconds_(t_solve0, t_solve1);
            diagnostics.function_update_wall_seconds =
                seconds_(t_update0, t_update1);
            diagnostics.geometry_cache_hits =
                (x_cache.geometry_cache_hit_count() -
                 x_geometry_hits_before) +
                y_cache.geometry_cache_hit_count();
            diagnostics.geometry_cache_misses =
                (x_cache.geometry_cache_miss_count() -
                 x_geometry_misses_before) +
                y_cache.geometry_cache_miss_count();
            if (options.diagnostics_mode ==
                la::concepts::SolverDiagnosticsMode::Detailed)
            {
                const auto t_residual0 = Clock::now();
                diagnostics.residual_inf_norm =
                    residual_inf_norm_(A_slab, lambda_slab, rhs_slab);
                const auto t_residual1 = Clock::now();
                diagnostics.residual_check_seconds =
                    seconds_(t_residual0, t_residual1);
                diagnostics.residual_check_count = 1;

                const auto t_permutation0 = Clock::now();
                diagnostics.source_slab_permutation_error =
                    source_slab_permutation_error_();
                const auto t_permutation1 = Clock::now();
                diagnostics.source_permutation_check_seconds =
                    seconds_(t_permutation0, t_permutation1);
                diagnostics.source_permutation_check_count = 1;
            }
            return diagnostics;
        }

    private:
        using Clock = std::chrono::steady_clock;

        [[nodiscard]] static double seconds_(
            const Clock::time_point& begin,
            const Clock::time_point& end)
        {
            return std::chrono::duration<double>(end - begin).count();
        }

        [[nodiscard]] static double absolute_seconds_(
            const Clock::time_point& value)
        {
            return std::chrono::duration<double>(
                value.time_since_epoch()).count();
        }

        [[nodiscard]] std::vector<int> source_cells_for_slab_(int slab_id) const
        {
            std::set<int> source_cells;
            const auto& slab = slab_space_->slab(slab_id);
            for (const int slab_cell_id : slab.active_cells())
                source_cells.insert(slab.source_cell_id(slab_cell_id));
            return std::vector<int>(source_cells.begin(), source_cells.end());
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType>
        void assemble_mat_A_and_rhs_fast_2d_shared_state_(
            SparseMatrixType& A_slab,
            VectorType& rhs_slab,
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>&
                x_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>&
                ancestor_cache,
            double zero_tol,
            finite_element::assembly::detail::AssemblyKernelDiagnostics*
                A_diagnostics,
            finite_element::assembly::detail::AssemblyKernelDiagnostics*
                rhs_diagnostics,
            SlabDiagnostics* slab_diagnostics = nullptr) const
        {
            using PatternBuilder = typename Backend::SparsePatternBuilder;
            using XBasis =
                finite_element::basis::SpaceTimeBasis<
                    typename XSpace::GT,
                    typename XSpace::FETraitsType>;
            using YTables =
                finite_element::assembly::detail::SpaceTimeBasisTables<
                    typename LocalSlabSpace::GT,
                    typename LocalSlabSpace::FETraitsType,
                    QSpace,
                    QTime>;
            using XGeometry =
                finite_element::geometry::CellGeometry<XSpace, 2>;
            using YGeometry =
                finite_element::geometry::CellGeometry<LocalSlabSpace, 2>;

            static_assert(
                XSpace::GT::dim_space_v == 2 &&
                    LocalSlabSpace::GT::dim_space_v == 2,
                "The shared slab reconstruction kernel is only for 2+1D.");

            constexpr int x_dofs_per_cell =
                XSpace::FETraitsType::dofs_per_cell;
            constexpr int y_dofs_per_cell =
                LocalSlabSpace::FETraitsType::dofs_per_cell;
            constexpr int n_q = YTables::n_cell_q;

            const auto& slab = slab_space_->slab(slab_id);
            const auto& y_space = slab.fespace_ref();
            const auto& dof_handler = y_space.dof_handler_ref();
            const auto& active_cells = slab.active_cells();
            const int n_active_cells =
                static_cast<int>(active_cells.size());
            const auto& u_true = u_delta.true_coefficients();

            const std::size_t reserve_entries =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(y_dofs_per_cell) *
                static_cast<std::size_t>(y_dofs_per_cell) * 4u;

            PatternBuilder builder;
            builder.resize(
                dof_handler.n_true_dofs(),
                dof_handler.n_true_dofs());
            builder.reserve_pattern(reserve_entries);

            const auto t_pattern0 = Clock::now();
            for (const int slab_cell_id : active_cells)
            {
                const auto& y_expanded = y_cache.dof_expansion(slab_cell_id);
                finite_element::assembly::scatter_matrix_pattern(
                    builder,
                    y_expanded,
                    y_expanded);
            }
            const auto t_matrix_allocation0 = Clock::now();
            builder.finalize_pattern();
            const auto t_matrix_allocation1 = Clock::now();
            builder.zero_values();
            const auto t_pattern1 = Clock::now();

            rhs_slab.resize(dof_handler.n_true_dofs());
            rhs_slab.set_zero();

            la::local::FixedLocalMatrix<
                y_dofs_per_cell,
                y_dofs_per_cell> local_A;
            la::local::FixedLocalVector<y_dofs_per_cell> local_rhs;
            std::array<double, x_dofs_per_cell> x_coefficients{};
            std::array<
                typename YGeometry::SpatialGradient,
                y_dofs_per_cell> grad_phi{};
            std::array<
                typename YGeometry::SpatialGradient,
                y_dofs_per_cell> M_grad_phi{};

            enum class CoefficientMode
            {
                IdentityZeroLoad,
                ConstantDiffusion,
                GenericVariable
            };

            const bool identity_diffusion_available =
                coefficients::is_identity_diffusion_function<2>(M);
            const auto constant_diffusion_tensor =
                coefficients::constant_diffusion_tensor_if_available<2>(M);
            const bool zero_load_available =
                coefficients::is_zero_load_function(ell);
            const std::string& requested_mode =
                slab_reconstruction_operator_mode_;
            CoefficientMode coefficient_mode =
                CoefficientMode::GenericVariable;

            if (requested_mode == "generic_variable_path")
            {
                coefficient_mode = CoefficientMode::GenericVariable;
            }
            else if (requested_mode == "identity_zero_load_fast_path")
            {
                if (!identity_diffusion_available || !zero_load_available)
                {
                    throw std::runtime_error(
                        "slab_reconstruction_operator_mode="
                        "identity_zero_load_fast_path requires identity "
                        "diffusion and zero load.");
                }
                coefficient_mode = CoefficientMode::IdentityZeroLoad;
            }
            else if (requested_mode == "constant_diffusion_fast_path")
            {
                if (!constant_diffusion_tensor.has_value())
                {
                    throw std::runtime_error(
                        "slab_reconstruction_operator_mode="
                        "constant_diffusion_fast_path requires a constant "
                        "or identity diffusion descriptor.");
                }
                coefficient_mode = CoefficientMode::ConstantDiffusion;
            }
            else if (requested_mode == "auto")
            {
                if (identity_diffusion_available && zero_load_available)
                {
                    coefficient_mode = CoefficientMode::IdentityZeroLoad;
                }
                else if (constant_diffusion_tensor.has_value())
                {
                    coefficient_mode = CoefficientMode::ConstantDiffusion;
                }
            }
            else
            {
                throw std::runtime_error(
                    "TimeSlabReconstructionOperator: unsupported slab "
                    "reconstruction operator mode '" +
                    requested_mode + "'.");
            }

            const auto fast_diffusion_tensor =
                coefficient_mode == CoefficientMode::IdentityZeroLoad
                    ? coefficients::identity_diffusion_tensor<2>()
                    : (constant_diffusion_tensor.has_value()
                           ? *constant_diffusion_tensor
                           : coefficients::identity_diffusion_tensor<2>());
            const bool skip_load_evaluation =
                zero_load_available &&
                requested_mode != "generic_variable_path";
            const bool debug_compare_sample =
                requested_mode != "auto" &&
                coefficient_mode != CoefficientMode::GenericVariable;

            la::local::FixedLocalMatrix<
                y_dofs_per_cell,
                y_dofs_per_cell> local_A_generic_debug;
            la::local::FixedLocalVector<y_dofs_per_cell>
                local_rhs_generic_debug;
            std::array<
                typename YGeometry::SpatialGradient,
                y_dofs_per_cell> M_grad_phi_generic_debug{};
            bool debug_sample_taken = false;
            double local_A_debug_max_abs_diff = 0.0;
            double rhs_debug_max_abs_diff = 0.0;

            const auto t_numeric0 = Clock::now();
            for (const int slab_cell_id : active_cells)
            {
                const int source_y_cell_id =
                    slab.source_cell_id(slab_cell_id);
                const int x_cell_id =
                    finite_element::assembly::detail::
                        find_active_ancestor_cell_from_source_cell(
                            ancestor_cache,
                            *x_space_,
                            source_y_cell_id);

                const auto& x_geom = x_cache.geometry(x_cell_id);
                const auto& y_geom = y_cache.geometry(slab_cell_id);
                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                const auto& y_expanded = y_cache.dof_expansion(slab_cell_id);
                const double jac_y = YGeometry::jacobian_measure(y_geom);

                for (int i = 0; i < x_dofs_per_cell; ++i)
                {
                    double value = 0.0;
                    for (const auto& wi : x_expanded[i])
                    {
                        if (wi.true_dof >= 0)
                            value += wi.weight * u_true[wi.true_dof];
                    }
                    x_coefficients[static_cast<std::size_t>(i)] = value;
                }

                finite_element::assembly::detail::zero_local_matrix(local_A);
                finite_element::assembly::detail::zero_local_vector(local_rhs);

                for (int q = 0; q < n_q; ++q)
                {
                    const auto& xi_y = YTables::cell_rule.points[q];
                    const double dmu = jac_y * YTables::cell_rule.weights[q];
                    const auto& phi_vals = YTables::values_on_cell_qp(q);
                    const auto& phi_grads_ref =
                        YTables::gradients_on_cell_qp(q);

                    const auto x_q =
                        YGeometry::map_to_physical(y_geom, xi_y);
                    const auto xi_x =
                        XGeometry::physical_to_reference(x_geom, x_q);
                    const double ell_q =
                        skip_load_evaluation
                            ? 0.0
                            : static_cast<double>(ell(x_q));

                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        grad_phi[static_cast<std::size_t>(j)] =
                            YGeometry::spatial_gradient(
                                y_geom,
                                phi_grads_ref[j]);
                    }

                    if (coefficient_mode ==
                        CoefficientMode::GenericVariable)
                    {
                        const auto M_q =
                            coefficients::evaluate_diffusion_tensor<
                                MFunction,
                                2>(
                                M,
                                x_q);
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            M_grad_phi[static_cast<std::size_t>(j)] =
                                coefficients::apply_validated_M<2>(
                                    M_q,
                                    grad_phi[static_cast<std::size_t>(j)]);
                        }
                    }
                    else if (coefficient_mode ==
                             CoefficientMode::IdentityZeroLoad)
                    {
                        M_grad_phi = grad_phi;
                    }
                    else
                    {
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            M_grad_phi[static_cast<std::size_t>(j)] =
                                coefficients::apply_validated_M<2>(
                                    fast_diffusion_tensor,
                                    grad_phi[static_cast<std::size_t>(j)]);
                        }
                    }

                    for (int i = 0; i < y_dofs_per_cell; ++i)
                    {
                        const auto& grad_i =
                            grad_phi[static_cast<std::size_t>(i)];
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local_A(i, j) +=
                                coefficients::dot<2>(
                                    grad_i,
                                    M_grad_phi[
                                        static_cast<std::size_t>(j)]) *
                                dmu;
                        }
                    }

                    const auto psi_grads_ref = XBasis::grad_all(xi_x);
                    double dt_u = 0.0;
                    typename XGeometry::SpatialGradient grad_u{};
                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        const double coefficient =
                            x_coefficients[static_cast<std::size_t>(i)];
                        if (coefficient == 0.0)
                            continue;

                        dt_u +=
                            coefficient *
                            XGeometry::time_derivative(
                                x_geom,
                                psi_grads_ref[i]);
                        const auto grad_psi =
                            XGeometry::spatial_gradient(
                                x_geom,
                                psi_grads_ref[i]);
                        grad_u[0] += coefficient * grad_psi[0];
                        grad_u[1] += coefficient * grad_psi[1];
                    }

                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        const double value =
                            ell_q * phi_vals[j] -
                            dt_u * phi_vals[j] -
                            coefficients::dot<2>(
                                grad_u,
                                M_grad_phi[static_cast<std::size_t>(j)]);
                        local_rhs[j] += value * dmu;
                    }
                }

                if (debug_compare_sample && !debug_sample_taken)
                {
                    finite_element::assembly::detail::zero_local_matrix(
                        local_A_generic_debug);
                    finite_element::assembly::detail::zero_local_vector(
                        local_rhs_generic_debug);

                    for (int q = 0; q < n_q; ++q)
                    {
                        const auto& xi_y = YTables::cell_rule.points[q];
                        const double dmu =
                            jac_y * YTables::cell_rule.weights[q];
                        const auto& phi_vals =
                            YTables::values_on_cell_qp(q);
                        const auto& phi_grads_ref =
                            YTables::gradients_on_cell_qp(q);

                        const auto x_q =
                            YGeometry::map_to_physical(y_geom, xi_y);
                        const auto xi_x =
                            XGeometry::physical_to_reference(x_geom, x_q);
                        const auto M_q =
                            coefficients::evaluate_diffusion_tensor<
                                MFunction,
                                2>(
                                M,
                                x_q);
                        const double ell_q =
                            static_cast<double>(ell(x_q));

                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            grad_phi[static_cast<std::size_t>(j)] =
                                YGeometry::spatial_gradient(
                                    y_geom,
                                    phi_grads_ref[j]);
                            M_grad_phi_generic_debug[
                                static_cast<std::size_t>(j)] =
                                coefficients::apply_validated_M<2>(
                                    M_q,
                                    grad_phi[static_cast<std::size_t>(j)]);
                        }

                        for (int i = 0; i < y_dofs_per_cell; ++i)
                        {
                            const auto& grad_i =
                                grad_phi[static_cast<std::size_t>(i)];
                            for (int j = 0; j < y_dofs_per_cell; ++j)
                            {
                                local_A_generic_debug(i, j) +=
                                    coefficients::dot<2>(
                                        grad_i,
                                        M_grad_phi_generic_debug[
                                            static_cast<std::size_t>(j)]) *
                                    dmu;
                            }
                        }

                        const auto psi_grads_ref = XBasis::grad_all(xi_x);
                        double dt_u = 0.0;
                        typename XGeometry::SpatialGradient grad_u{};
                        for (int i = 0; i < x_dofs_per_cell; ++i)
                        {
                            const double coefficient =
                                x_coefficients[static_cast<std::size_t>(i)];
                            if (coefficient == 0.0)
                                continue;

                            dt_u +=
                                coefficient *
                                XGeometry::time_derivative(
                                    x_geom,
                                    psi_grads_ref[i]);
                            const auto grad_psi =
                                XGeometry::spatial_gradient(
                                    x_geom,
                                    psi_grads_ref[i]);
                            grad_u[0] += coefficient * grad_psi[0];
                            grad_u[1] += coefficient * grad_psi[1];
                        }

                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            const double value =
                                ell_q * phi_vals[j] -
                                dt_u * phi_vals[j] -
                                coefficients::dot<2>(
                                    grad_u,
                                    M_grad_phi_generic_debug[
                                        static_cast<std::size_t>(j)]);
                            local_rhs_generic_debug[j] += value * dmu;
                        }
                    }

                    for (int i = 0; i < y_dofs_per_cell; ++i)
                    {
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local_A_debug_max_abs_diff =
                                std::max(
                                    local_A_debug_max_abs_diff,
                                    std::abs(
                                        local_A(i, j) -
                                        local_A_generic_debug(i, j)));
                        }
                        rhs_debug_max_abs_diff =
                            std::max(
                                rhs_debug_max_abs_diff,
                                std::abs(
                                    local_rhs[i] -
                                    local_rhs_generic_debug[i]));
                    }
                    debug_sample_taken = true;
                }

                finite_element::assembly::scatter_matrix_offset(
                    builder,
                    local_A,
                    y_expanded,
                    y_expanded,
                    0,
                    0,
                    zero_tol);
                finite_element::assembly::scatter_vector(
                    rhs_slab,
                    local_rhs,
                    y_expanded,
                    zero_tol);
            }
            const auto t_numeric1 = Clock::now();

            const std::size_t finalized_pattern_entries =
                builder.pattern_entries();
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
            const std::size_t finalized_pattern_bytes =
                builder.pattern_bytes();
            const std::size_t finalized_numeric_matrix_bytes =
                builder.numeric_matrix_bytes();

            if (A_diagnostics != nullptr)
            {
                A_diagnostics->active_cells =
                    static_cast<std::size_t>(n_active_cells);
                A_diagnostics->quadrature_points =
                    static_cast<std::size_t>(n_active_cells) *
                    static_cast<std::size_t>(n_q);
                A_diagnostics->gradient_evaluations =
                    A_diagnostics->quadrature_points *
                    static_cast<std::size_t>(y_dofs_per_cell);
                A_diagnostics->diffusion_tensor_evaluations =
                    coefficient_mode == CoefficientMode::GenericVariable
                        ? A_diagnostics->quadrature_points
                        : 0u;
                A_diagnostics->sparse_triplets_emitted = 0;
                A_diagnostics->peak_triplet_bytes = 0;
            }

            if (rhs_diagnostics != nullptr)
            {
                rhs_diagnostics->active_cells =
                    static_cast<std::size_t>(n_active_cells);
                rhs_diagnostics->quadrature_points =
                    static_cast<std::size_t>(n_active_cells) *
                    static_cast<std::size_t>(n_q);
                rhs_diagnostics->scalar_basis_evaluations =
                    rhs_diagnostics->quadrature_points *
                    static_cast<std::size_t>(y_dofs_per_cell);
                rhs_diagnostics->gradient_evaluations =
                    rhs_diagnostics->quadrature_points *
                    static_cast<std::size_t>(
                        x_dofs_per_cell + y_dofs_per_cell);
                rhs_diagnostics->source_evaluations =
                    skip_load_evaluation
                        ? 0u
                        : rhs_diagnostics->quadrature_points;
                // The shared-state kernel evaluates M once per quadrature
                // point while assembling A, then reuses M grad(phi_j) for RHS.
                rhs_diagnostics->diffusion_tensor_evaluations = 0;
            }

            const auto t_matrix_copy0 = Clock::now();
            A_slab = builder.release_matrix();
            A_slab.prune(1.0, zero_tol);
            const auto t_matrix_copy1 = Clock::now();

            if (A_diagnostics != nullptr)
            {
                A_diagnostics->final_matrix_nonzeros =
                    finite_element::assembly::detail::count_nonzeros(A_slab);
                A_diagnostics->peak_sparse_matrix_bytes =
                    finite_element::assembly::detail::
                        estimate_compressed_sparse_matrix_bytes(
                            A_slab.rows(),
                            A_slab.cols(),
                            A_diagnostics->final_matrix_nonzeros);
            }

            if (slab_diagnostics != nullptr)
            {
                slab_diagnostics->pattern_build_seconds =
                    seconds_(t_pattern0, t_pattern1);
                slab_diagnostics->numeric_fill_seconds =
                    seconds_(t_numeric0, t_numeric1);
                slab_diagnostics->matrix_allocation_seconds =
                    seconds_(t_matrix_allocation0, t_matrix_allocation1);
                slab_diagnostics->matrix_copy_seconds =
                    seconds_(t_matrix_copy0, t_matrix_copy1);
                slab_diagnostics->reference_table_build_seconds = 0.0;
                if (coefficient_mode ==
                    CoefficientMode::IdentityZeroLoad)
                {
                    slab_diagnostics
                        ->operator_mode_identity_zero_load_count += 1;
                }
                else if (coefficient_mode ==
                         CoefficientMode::ConstantDiffusion)
                {
                    slab_diagnostics
                        ->operator_mode_constant_diffusion_count += 1;
                }
                else
                {
                    slab_diagnostics
                        ->operator_mode_generic_variable_count += 1;
                }
                const std::size_t total_qpoints =
                    static_cast<std::size_t>(n_active_cells) *
                    static_cast<std::size_t>(n_q);
                slab_diagnostics->diffusion_evaluations +=
                    coefficient_mode == CoefficientMode::GenericVariable
                        ? total_qpoints
                        : 0u;
                slab_diagnostics->load_evaluations +=
                    skip_load_evaluation ? 0u : total_qpoints;
                if (coefficient_mode ==
                    CoefficientMode::GenericVariable)
                {
                    slab_diagnostics->local_A_generic_count +=
                        n_active_cells;
                }
                else
                {
                    slab_diagnostics->local_A_fast_path_count +=
                        n_active_cells;
                }
                if (skip_load_evaluation)
                {
                    slab_diagnostics->rhs_zero_load_fast_path_count +=
                        n_active_cells;
                }
                slab_diagnostics->local_A_debug_max_abs_diff =
                    std::max(
                        slab_diagnostics->local_A_debug_max_abs_diff,
                        local_A_debug_max_abs_diff);
                slab_diagnostics->rhs_debug_max_abs_diff =
                    std::max(
                        slab_diagnostics->rhs_debug_max_abs_diff,
                        rhs_debug_max_abs_diff);
                slab_diagnostics->pattern_candidates =
                    finalized_pattern_candidates;
                slab_diagnostics->pattern_entries =
                    finalized_pattern_entries;
                slab_diagnostics->pattern_duplicate_entries =
                    finalized_pattern_duplicates;
                slab_diagnostics->pattern_candidate_bytes =
                    finalized_pattern_candidate_bytes;
                slab_diagnostics->pattern_bytes = finalized_pattern_bytes;
                slab_diagnostics->numeric_matrix_bytes =
                    finalized_numeric_matrix_bytes;
                slab_diagnostics->triplet_bytes_avoided =
                    finalized_pattern_candidates *
                    finite_element::assembly::detail::
                        estimated_triplet_bytes<typename Backend::SparseBuilder>();
                slab_diagnostics->reconstruction_cache_bytes +=
                    static_cast<double>(
                        finalized_pattern_bytes +
                        finalized_numeric_matrix_bytes);
            }
        }

        template<int QSpace, int QTime, class MFunction>
        void assemble_mat_A_fast_2d_(
            SparseMatrixType& A_slab,
            int slab_id,
            const MFunction& M,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            double zero_tol,
            finite_element::assembly::detail::AssemblyKernelDiagnostics*
                diagnostics) const
        {
            using SparseBuilder = typename Backend::SparseBuilder;
            using YTables =
                finite_element::assembly::detail::SpaceTimeBasisTables<
                    typename LocalSlabSpace::GT,
                    typename LocalSlabSpace::FETraitsType,
                    QSpace,
                    QTime>;
            using YGeometry =
                finite_element::geometry::CellGeometry<LocalSlabSpace, 2>;

            constexpr int y_dofs_per_cell =
                LocalSlabSpace::FETraitsType::dofs_per_cell;
            constexpr int n_q = YTables::n_cell_q;

            const auto& slab = slab_space_->slab(slab_id);
            const auto& y_space = slab.fespace_ref();
            const auto& dof_handler = y_space.dof_handler_ref();
            const auto& active_cells = slab.active_cells();
            const int n_active_cells =
                static_cast<int>(active_cells.size());
            const std::size_t reserve_entries =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(y_dofs_per_cell) *
                static_cast<std::size_t>(y_dofs_per_cell) * 4u;

            SparseBuilder builder;
            builder.reserve(reserve_entries);
            la::local::FixedLocalMatrix<
                y_dofs_per_cell,
                y_dofs_per_cell> local_A;
            std::array<
                typename YGeometry::SpatialGradient,
                y_dofs_per_cell> grad_phi{};
            std::array<
                typename YGeometry::SpatialGradient,
                y_dofs_per_cell> M_grad_phi{};

            for (const int slab_cell_id : active_cells)
            {
                const auto& y_geom = y_cache.geometry(slab_cell_id);
                const auto& y_expanded = y_cache.dof_expansion(slab_cell_id);
                const double jac_y = YGeometry::jacobian_measure(y_geom);

                finite_element::assembly::detail::zero_local_matrix(local_A);

                for (int q = 0; q < n_q; ++q)
                {
                    const auto& xi_y = YTables::cell_rule.points[q];
                    const double dmu = jac_y * YTables::cell_rule.weights[q];
                    const auto& phi_grads_ref =
                        YTables::gradients_on_cell_qp(q);
                    const auto x_q =
                        YGeometry::map_to_physical(y_geom, xi_y);
                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<
                            MFunction,
                            2>(
                            M,
                            x_q);

                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        grad_phi[static_cast<std::size_t>(j)] =
                            YGeometry::spatial_gradient(
                                y_geom,
                                phi_grads_ref[j]);
                        M_grad_phi[static_cast<std::size_t>(j)] =
                            coefficients::apply_validated_M<2>(
                                M_q,
                                grad_phi[static_cast<std::size_t>(j)]);
                    }

                    for (int i = 0; i < y_dofs_per_cell; ++i)
                    {
                        const auto& grad_i =
                            grad_phi[static_cast<std::size_t>(i)];
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local_A(i, j) +=
                                coefficients::dot<2>(
                                    grad_i,
                                    M_grad_phi[
                                        static_cast<std::size_t>(j)]) *
                                dmu;
                        }
                    }
                }

                finite_element::assembly::scatter_matrix(
                    builder,
                    local_A,
                    y_expanded,
                    y_expanded,
                    zero_tol);
            }

            if (diagnostics != nullptr)
            {
                diagnostics->active_cells =
                    static_cast<std::size_t>(n_active_cells);
                diagnostics->quadrature_points =
                    static_cast<std::size_t>(n_active_cells) *
                    static_cast<std::size_t>(n_q);
                diagnostics->gradient_evaluations =
                    diagnostics->quadrature_points *
                    static_cast<std::size_t>(y_dofs_per_cell);
                diagnostics->diffusion_tensor_evaluations =
                    diagnostics->quadrature_points;
                diagnostics->sparse_triplets_emitted = builder.size();
                diagnostics->peak_triplet_bytes =
                    builder.size() *
                    finite_element::assembly::detail::
                        estimated_triplet_bytes<SparseBuilder>();
            }

            A_slab.resize(
                dof_handler.n_true_dofs(),
                dof_handler.n_true_dofs());
            A_slab.set_from_builder(builder);

            if (diagnostics != nullptr)
            {
                diagnostics->final_matrix_nonzeros =
                    finite_element::assembly::detail::count_nonzeros(A_slab);
                diagnostics->peak_sparse_matrix_bytes =
                    finite_element::assembly::detail::
                        estimate_compressed_sparse_matrix_bytes(
                            A_slab.rows(),
                            A_slab.cols(),
                            diagnostics->final_matrix_nonzeros);
            }
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType>
        void assemble_rhs_fast_2d_(
            VectorType& rhs_slab,
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpace>& y_cache,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>&
                x_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>&
                ancestor_cache,
            double zero_tol,
            finite_element::assembly::detail::AssemblyKernelDiagnostics*
                diagnostics) const
        {
            using XBasis =
                finite_element::basis::SpaceTimeBasis<
                    typename XSpace::GT,
                    typename XSpace::FETraitsType>;
            using YTables =
                finite_element::assembly::detail::SpaceTimeBasisTables<
                    typename LocalSlabSpace::GT,
                    typename LocalSlabSpace::FETraitsType,
                    QSpace,
                    QTime>;
            using XGeometry =
                finite_element::geometry::CellGeometry<XSpace, 2>;
            using YGeometry =
                finite_element::geometry::CellGeometry<LocalSlabSpace, 2>;

            constexpr int x_dofs_per_cell =
                XSpace::FETraitsType::dofs_per_cell;
            constexpr int y_dofs_per_cell =
                LocalSlabSpace::FETraitsType::dofs_per_cell;
            constexpr int n_q = YTables::n_cell_q;

            const auto& slab = slab_space_->slab(slab_id);
            const auto& y_space = slab.fespace_ref();
            const auto& y_dof_handler = y_space.dof_handler_ref();
            const auto& u_true = u_delta.true_coefficients();
            const auto& active_cells = slab.active_cells();
            const int n_active_cells =
                static_cast<int>(active_cells.size());

            rhs_slab.resize(y_dof_handler.n_true_dofs());
            rhs_slab.set_zero();

            la::local::FixedLocalVector<y_dofs_per_cell> local_rhs;
            std::array<double, x_dofs_per_cell> x_coefficients{};
            std::array<
                typename YGeometry::SpatialGradient,
                y_dofs_per_cell> grad_phi{};
            std::array<
                typename YGeometry::SpatialGradient,
                y_dofs_per_cell> M_grad_phi{};

            for (const int slab_cell_id : active_cells)
            {
                const int source_y_cell_id =
                    slab.source_cell_id(slab_cell_id);
                const int x_cell_id =
                    finite_element::assembly::detail::
                        find_active_ancestor_cell_from_source_cell(
                            ancestor_cache,
                            *x_space_,
                            source_y_cell_id);

                const auto& x_geom = x_cache.geometry(x_cell_id);
                const auto& y_geom = y_cache.geometry(slab_cell_id);
                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                const auto& y_expanded = y_cache.dof_expansion(slab_cell_id);
                const double jac_y = YGeometry::jacobian_measure(y_geom);

                for (int i = 0; i < x_dofs_per_cell; ++i)
                {
                    double value = 0.0;
                    for (const auto& wi : x_expanded[i])
                    {
                        if (wi.true_dof >= 0)
                            value += wi.weight * u_true[wi.true_dof];
                    }
                    x_coefficients[static_cast<std::size_t>(i)] = value;
                }

                finite_element::assembly::detail::zero_local_vector(local_rhs);

                for (int q = 0; q < n_q; ++q)
                {
                    const auto& xi_y = YTables::cell_rule.points[q];
                    const double dmu = jac_y * YTables::cell_rule.weights[q];
                    const auto& phi_vals = YTables::values_on_cell_qp(q);
                    const auto& phi_grads_ref =
                        YTables::gradients_on_cell_qp(q);

                    const auto x_q =
                        YGeometry::map_to_physical(y_geom, xi_y);
                    const auto xi_x =
                        XGeometry::physical_to_reference(x_geom, x_q);
                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<
                            MFunction,
                            2>(
                            M,
                            x_q);
                    const double ell_q = static_cast<double>(ell(x_q));

                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        grad_phi[static_cast<std::size_t>(j)] =
                            YGeometry::spatial_gradient(
                                y_geom,
                                phi_grads_ref[j]);
                        M_grad_phi[static_cast<std::size_t>(j)] =
                            coefficients::apply_validated_M<2>(
                                M_q,
                                grad_phi[static_cast<std::size_t>(j)]);
                    }

                    const auto psi_grads_ref = XBasis::grad_all(xi_x);
                    double dt_u = 0.0;
                    typename XGeometry::SpatialGradient grad_u{};
                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        const double coefficient =
                            x_coefficients[static_cast<std::size_t>(i)];
                        if (coefficient == 0.0)
                            continue;

                        dt_u +=
                            coefficient *
                            XGeometry::time_derivative(
                                x_geom,
                                psi_grads_ref[i]);
                        const auto grad_psi =
                            XGeometry::spatial_gradient(
                                x_geom,
                                psi_grads_ref[i]);
                        grad_u[0] += coefficient * grad_psi[0];
                        grad_u[1] += coefficient * grad_psi[1];
                    }

                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        const double value =
                            ell_q * phi_vals[j] -
                            dt_u * phi_vals[j] -
                            coefficients::dot<2>(
                                grad_u,
                                M_grad_phi[static_cast<std::size_t>(j)]);
                        local_rhs[j] += value * dmu;
                    }
                }

                finite_element::assembly::scatter_vector(
                    rhs_slab,
                    local_rhs,
                    y_expanded,
                    zero_tol);
            }

            if (diagnostics != nullptr)
            {
                diagnostics->active_cells =
                    static_cast<std::size_t>(n_active_cells);
                diagnostics->quadrature_points =
                    static_cast<std::size_t>(n_active_cells) *
                    static_cast<std::size_t>(n_q);
                diagnostics->scalar_basis_evaluations =
                    diagnostics->quadrature_points *
                    static_cast<std::size_t>(y_dofs_per_cell);
                diagnostics->gradient_evaluations =
                    diagnostics->quadrature_points *
                    static_cast<std::size_t>(
                        x_dofs_per_cell + y_dofs_per_cell);
                diagnostics->diffusion_tensor_evaluations =
                    diagnostics->quadrature_points;
                diagnostics->source_evaluations =
                    diagnostics->quadrature_points;
            }
        }

        [[nodiscard]] static double residual_inf_norm_(
            const SparseMatrixType& A,
            const VectorType& x,
            const VectorType& rhs)
        {
            auto residual = la::ops::matvec(A, x);
            if (residual.size() != rhs.size())
                throw std::runtime_error(
                    "TimeSlabReconstructionOperator: residual size mismatch.");
            for (int i = 0; i < residual.size(); ++i)
                residual[i] -= rhs[i];
            return la::ops::inf_norm(residual);
        }

        [[nodiscard]] double source_slab_permutation_error_() const
        {
            try
            {
                const auto permutation =
                    slab_space_->already_slabbed_true_dof_permutation();
                return permutation.is_bijection() ? 0.0 : 1.0;
            }
            catch (...)
            {
                return -1.0;
            }
        }

        const XSpace* x_space_ = nullptr;
        const SlabSpace* slab_space_ = nullptr;
        std::string slab_reconstruction_operator_mode_ = "auto";
    };
}

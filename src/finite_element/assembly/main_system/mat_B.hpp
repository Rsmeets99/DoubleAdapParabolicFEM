#pragma once

#include <concepts>
#include <cstddef>
#include <exception>
#include <vector>

#include "../constrained_dofs.hpp"
#include "../scatter.hpp"

#include "../detail/openmp_assembly.hpp"
#include "../detail/assembly_diagnostics.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../../basis/space_time_basis_selector.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../geometry/cell_geometry.hpp"
#include "../detail/active_cell_locator.hpp"
#include "../detail/local_mixed_bilinear_forms.hpp"
#include "../detail/space_time_basis_tables.hpp"

#include "../../../linear_algebra/assembly/local_objects.hpp"
#include "../../../linear_algebra/concepts/sparse_builder.hpp"
#include "../../../linear_algebra/concepts/sparse_matrix.hpp"

namespace finite_element::assembly
{
    template<
        detail::MixedBilinearFormPart Part,
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_contribution_(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        using XSpaceType    = XFESpaceType;
        using YSpaceType    = YFESpaceType;
        using XGT           = typename XSpaceType::GT;
        using YGT           = typename YSpaceType::GT;
        using XFETraits     = typename XSpaceType::FETraitsType;
        using YFETraits     = typename YSpaceType::FETraitsType;
        using SparseBuilder = typename Backend::SparseBuilder;

        static_assert(XGT::dim_space_v == YGT::dim_space_v,
                      "assemble_mat_B: X and Y must have the same spatial dimension.");

        using XBasis    = finite_element::basis::SpaceTimeBasis<XGT, XFETraits>;
        using YTables   = detail::SpaceTimeBasisTables<YGT, YFETraits, QSpace, QTime>;
        using XGeometry = finite_element::geometry::CellGeometry<XSpaceType, XGT::dim_space_v>;
        using YGeometry = finite_element::geometry::CellGeometry<YSpaceType, YGT::dim_space_v>;

        constexpr int x_dofs_per_cell = XFETraits::dofs_per_cell;
        constexpr int y_dofs_per_cell = YFETraits::dofs_per_cell;
        constexpr int n_q             = YTables::n_cell_q;

        const auto& x_dof_handler  = x_space.dof_handler_ref();
        const auto& y_dof_handler  = y_space.dof_handler_ref();
        const auto& y_active_cells = y_space.active_cells();

        const int n_active_cells = static_cast<int>(y_active_cells.size());
        const std::size_t reserve_entries =
            static_cast<std::size_t>(n_active_cells) *
            static_cast<std::size_t>(x_dofs_per_cell) *
            static_cast<std::size_t>(y_dofs_per_cell) * 4u;

        SparseBuilder builder;

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            std::vector<int> x_cell_ids(static_cast<std::size_t>(n_active_cells), -1);

            for (int item_index = 0; item_index < n_active_cells; ++item_index)
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

                (void)x_cache.dof_expansion(x_cell_id);
                (void)x_cache.geometry(x_cell_id);
                (void)y_cache.dof_expansion(y_cell_id);
                (void)y_cache.geometry(y_cell_id);
            }

            auto builders =
                detail::make_thread_local_sparse_builders<SparseBuilder>(
                    detail::openmp_thread_count(),
                    reserve_entries);
            std::exception_ptr error;

#pragma omp parallel
            {
                try
                {
                    const int thread_id = omp_get_thread_num();
                    auto& local_builder = builders[static_cast<std::size_t>(thread_id)];
                    la::local::FixedLocalMatrix<
                        x_dofs_per_cell,
                        y_dofs_per_cell> local;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int y_cell_id =
                            y_active_cells[static_cast<std::size_t>(item_index)];
                        const int x_cell_id =
                            x_cell_ids[static_cast<std::size_t>(item_index)];

                        const auto& x_geom = x_cache.geometry(x_cell_id);
                        const auto& y_geom = y_cache.geometry(y_cell_id);

                        const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                        const auto& y_expanded = y_cache.dof_expansion(y_cell_id);

                        detail::assemble_local_mixed_space_time_matrix_part<
                            Part,
                            XBasis,
                            YTables,
                            XGeometry,
                            YGeometry>(
                                local,
                                x_geom,
                                y_geom,
                                M);

                        scatter_matrix(local_builder, local, x_expanded, y_expanded, zero_tol);
                    }
                }
                catch (...)
                {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }

            detail::rethrow_parallel_exception(error);
            builder = detail::merge_sparse_builders(builders);
        }
        else
#endif
        {
            builder.reserve(reserve_entries);
            la::local::FixedLocalMatrix<
                x_dofs_per_cell,
                y_dofs_per_cell> local;

            for (const int y_cell_id : y_active_cells)
            {
                const int x_cell_id =
                    detail::find_active_ancestor_cell(
                        ancestor_cache,
                        x_space,
                        y_space,
                        y_cell_id);

                const auto& x_geom = x_cache.geometry(x_cell_id);
                const auto& y_geom = y_cache.geometry(y_cell_id);

                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                const auto& y_expanded = y_cache.dof_expansion(y_cell_id);

                detail::assemble_local_mixed_space_time_matrix_part<
                    Part,
                    XBasis,
                    YTables,
                    XGeometry,
                    YGeometry>(
                        local,
                        x_geom,
                        y_geom,
                        M);

                scatter_matrix(builder, local, x_expanded, y_expanded, zero_tol);
            }
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
            if constexpr (
                Part == detail::MixedBilinearFormPart::SpatialDiffusion ||
                Part == detail::MixedBilinearFormPart::Full)
            {
                diagnostics->diffusion_tensor_evaluations =
                    diagnostics->quadrature_points;
            }
            diagnostics->sparse_triplets_emitted = builder.size();
            diagnostics->peak_triplet_bytes =
                builder.size() *
                detail::estimated_triplet_bytes<SparseBuilder>();
        }

        B.resize(x_dof_handler.n_true_dofs(), y_dof_handler.n_true_dofs());
        B.set_from_builder(builder);

        if (diagnostics != nullptr)
        {
            diagnostics->final_matrix_nonzeros = detail::count_nonzeros(B);
            diagnostics->peak_sparse_matrix_bytes =
                detail::estimate_compressed_sparse_matrix_bytes(
                    B.rows(),
                    B.cols(),
                    diagnostics->final_matrix_nonzeros);
        }
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_dt(
        typename Backend::SparseMatrix& B_dt,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_contribution_<
            detail::MixedBilinearFormPart::TimeDerivative,
            QSpace,
            QTime,
            Backend>(
                B_dt,
                x_space,
                y_space,
                x_cache,
                y_cache,
                ancestor_cache,
                coefficients::IdentityDiffusion<XGT::dim_space_v>{},
                zero_tol,
                nullptr);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B_A(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        assemble_mat_B_contribution_<
            detail::MixedBilinearFormPart::SpatialDiffusion,
            QSpace,
            QTime,
            Backend>(
                B_A,
                x_space,
                y_space,
                x_cache,
                y_cache,
                ancestor_cache,
                M,
                zero_tol,
                diagnostics);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_A(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_A<QSpace, QTime, Backend>(
            B_A,
            x_space,
            y_space,
            x_cache,
            y_cache,
            ancestor_cache,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol,
            diagnostics);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        // Assemble B = B_dt + B_A in one pass so existing callers do not pay
        // to materialize both large split matrices unless they request them.
        assemble_mat_B_contribution_<
            detail::MixedBilinearFormPart::Full,
            QSpace,
            QTime,
            Backend>(
                B,
                x_space,
                y_space,
                x_cache,
                y_cache,
                ancestor_cache,
                M,
                zero_tol,
                diagnostics);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B<QSpace, QTime, Backend>(
            B,
            x_space,
            y_space,
            x_cache,
            y_cache,
            ancestor_cache,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol,
            diagnostics);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_dt(
        typename Backend::SparseMatrix& B_dt,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        double zero_tol = 1e-15)
    {
        detail::AssemblySpaceCache<XFESpaceType> x_cache(x_space);
        detail::AssemblySpaceCache<YFESpaceType> y_cache(y_space);
        detail::ActiveAncestorCache<XFESpaceType> ancestor_cache(x_space);

        assemble_mat_B_dt<QSpace, QTime, Backend>(
            B_dt,
            x_space,
            y_space,
            x_cache,
            y_cache,
            ancestor_cache,
            zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B_A(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        const MFunction& M,
        double zero_tol = 1e-15)
    {
        detail::AssemblySpaceCache<XFESpaceType> x_cache(x_space);
        detail::AssemblySpaceCache<YFESpaceType> y_cache(y_space);
        detail::ActiveAncestorCache<XFESpaceType> ancestor_cache(x_space);

        assemble_mat_B_A<QSpace, QTime, Backend>(
            B_A,
            x_space,
            y_space,
            x_cache,
            y_cache,
            ancestor_cache,
            M,
            zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_A(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_A<QSpace, QTime, Backend>(
            B_A,
            x_space,
            y_space,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        const MFunction& M,
        double zero_tol = 1e-15)
    {
        detail::AssemblySpaceCache<XFESpaceType> x_cache(x_space);
        detail::AssemblySpaceCache<YFESpaceType> y_cache(y_space);
        detail::ActiveAncestorCache<XFESpaceType> ancestor_cache(x_space);

        assemble_mat_B<QSpace, QTime, Backend>(
            B,
            x_space,
            y_space,
            x_cache,
            y_cache,
            ancestor_cache,
            M,
            zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B<QSpace, QTime, Backend>(
            B,
            x_space,
            y_space,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol);
    }
}

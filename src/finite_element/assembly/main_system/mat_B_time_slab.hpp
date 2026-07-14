#pragma once

#include <concepts>
#include <cstddef>
#include <exception>
#include <vector>

#include "../constrained_dofs.hpp"
#include "../scatter.hpp"

#include "../detail/openmp_assembly.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../../basis/space_time_basis_selector.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../geometry/cell_geometry.hpp"
#include "../detail/active_cell_locator_time_slab.hpp"
#include "../detail/local_mixed_bilinear_forms.hpp"
#include "../detail/space_time_basis_tables.hpp"

#include "../../../linear_algebra/assembly/local_objects.hpp"
#include "../../../linear_algebra/concepts/sparse_builder.hpp"
#include "../../../linear_algebra/concepts/sparse_matrix.hpp"

namespace finite_element::assembly
{
    // -------------------------------------------------------------------------
    // Assemble B on a time slab:
    //
    //   B_ij = \int_{slab cells} [ (partial_t psi_i) phi_j
    //                           + grad_x psi_i^T M grad_x phi_j ] dx dt
    //
    // where:
    //   - psi_i belongs to X^delta on the active ancestor X-cell of the source
    //     Y-cell from which the slab cell was sliced,
    //   - phi_j belongs to the slab-local FE space on the synthetic slab cell.
    //
    // Matrix dimensions:
    //   rows = n_true_dofs(X^delta)
    //   cols = n_true_dofs(slabs)
    // -------------------------------------------------------------------------
    template<
        detail::MixedBilinearFormPart Part,
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class TimeSlabType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_on_time_slab_contribution_(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType>& y_cache,
        detail::SourceActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        double zero_tol = 1e-15)
    {
        using XSpaceType    = XFESpaceType;
        using YSpaceType    = typename TimeSlabType::SpaceType;
        using XGT           = typename XSpaceType::GT;
        using YGT           = typename YSpaceType::GT;
        using XFETraits     = typename XSpaceType::FETraitsType;
        using YFETraits     = typename YSpaceType::FETraitsType;
        using SparseBuilder = typename Backend::SparseBuilder;

        static_assert(XGT::dim_space_v == YGT::dim_space_v,
                      "assemble_mat_B_on_time_slab: X and slab space must have the same spatial dimension.");

        using XBasis    = finite_element::basis::SpaceTimeBasis<XGT, XFETraits>;
        using YTables   = detail::SpaceTimeBasisTables<YGT, YFETraits, QSpace, QTime>;
        using XGeometry = finite_element::geometry::CellGeometry<XSpaceType, XGT::dim_space_v>;
        using YGeometry = finite_element::geometry::CellGeometry<YSpaceType, YGT::dim_space_v>;

        constexpr int x_dofs_per_cell = XFETraits::dofs_per_cell;
        constexpr int y_dofs_per_cell = YFETraits::dofs_per_cell;

        const auto& y_space       = slab.fespace_ref();
        const auto& x_dof_handler = x_space.dof_handler_ref();
        const auto& y_dof_handler = y_space.dof_handler_ref();

        const auto& slab_active_cells = slab.active_cells();
        const int n_active_cells = static_cast<int>(slab_active_cells.size());
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
                const int slab_cell_id =
                    slab_active_cells[static_cast<std::size_t>(item_index)];
                const int source_y_cell_id = slab.source_cell_id(slab_cell_id);
                const int x_cell_id =
                    detail::find_active_ancestor_cell_from_source_cell(
                        ancestor_cache,
                        x_space,
                        source_y_cell_id);

                x_cell_ids[static_cast<std::size_t>(item_index)] = x_cell_id;

                (void)x_cache.dof_expansion(x_cell_id);
                (void)x_cache.geometry(x_cell_id);
                (void)y_cache.dof_expansion(slab_cell_id);
                (void)y_cache.geometry(slab_cell_id);
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
                        const int slab_cell_id =
                            slab_active_cells[static_cast<std::size_t>(item_index)];
                        const int x_cell_id =
                            x_cell_ids[static_cast<std::size_t>(item_index)];

                        const auto& x_geom = x_cache.geometry(x_cell_id);
                        const auto& y_geom = y_cache.geometry(slab_cell_id);

                        const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                        const auto& y_expanded = y_cache.dof_expansion(slab_cell_id);

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

            for (const int slab_cell_id : slab_active_cells)
            {
                const int source_y_cell_id = slab.source_cell_id(slab_cell_id);

                const int x_cell_id =
                    detail::find_active_ancestor_cell_from_source_cell(
                        ancestor_cache,
                        x_space,
                        source_y_cell_id);

                const auto& x_geom = x_cache.geometry(x_cell_id);
                const auto& y_geom = y_cache.geometry(slab_cell_id);

                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                const auto& y_expanded = y_cache.dof_expansion(slab_cell_id);

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

        B.resize(x_dof_handler.n_true_dofs(), y_dof_handler.n_true_dofs());
        B.set_from_builder(builder);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class TimeSlabType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_dt_on_time_slab(
        typename Backend::SparseMatrix& B_dt,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType>& y_cache,
        detail::SourceActiveAncestorCache<XFESpaceType>& ancestor_cache,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_on_time_slab_contribution_<
            detail::MixedBilinearFormPart::TimeDerivative,
            QSpace,
            QTime,
            Backend>(
                B_dt,
                x_space,
                slab,
                x_cache,
                y_cache,
                ancestor_cache,
                coefficients::IdentityDiffusion<XGT::dim_space_v>{},
                zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class TimeSlabType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B_A_on_time_slab(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType>& y_cache,
        detail::SourceActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        double zero_tol = 1e-15)
    {
        assemble_mat_B_on_time_slab_contribution_<
            detail::MixedBilinearFormPart::SpatialDiffusion,
            QSpace,
            QTime,
            Backend>(
                B_A,
                x_space,
                slab,
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
        class TimeSlabType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_A_on_time_slab(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType>& y_cache,
        detail::SourceActiveAncestorCache<XFESpaceType>& ancestor_cache,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_A_on_time_slab<QSpace, QTime, Backend>(
            B_A,
            x_space,
            slab,
            x_cache,
            y_cache,
            ancestor_cache,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class TimeSlabType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B_on_time_slab(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType>& y_cache,
        detail::SourceActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        double zero_tol = 1e-15)
    {
        // Assemble B = B_dt + B_A in one pass so existing slab callers do not
        // materialize both large split matrices unless they request them.
        assemble_mat_B_on_time_slab_contribution_<
            detail::MixedBilinearFormPart::Full,
            QSpace,
            QTime,
            Backend>(
                B,
                x_space,
                slab,
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
        class TimeSlabType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_on_time_slab(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType>& y_cache,
        detail::SourceActiveAncestorCache<XFESpaceType>& ancestor_cache,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_on_time_slab<QSpace, QTime, Backend>(
            B,
            x_space,
            slab,
            x_cache,
            y_cache,
            ancestor_cache,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class TimeSlabType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_dt_on_time_slab(
        typename Backend::SparseMatrix& B_dt,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        double zero_tol = 1e-15)
    {
        const auto& y_space = slab.fespace_ref();
        detail::AssemblySpaceCache<XFESpaceType> x_cache(x_space);
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType> y_cache(y_space);
        detail::SourceActiveAncestorCache<XFESpaceType> ancestor_cache(x_space);

        assemble_mat_B_dt_on_time_slab<QSpace, QTime, Backend>(
            B_dt,
            x_space,
            slab,
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
        class TimeSlabType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B_A_on_time_slab(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        const MFunction& M,
        double zero_tol = 1e-15)
    {
        const auto& y_space = slab.fespace_ref();
        detail::AssemblySpaceCache<XFESpaceType> x_cache(x_space);
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType> y_cache(y_space);
        detail::SourceActiveAncestorCache<XFESpaceType> ancestor_cache(x_space);

        assemble_mat_B_A_on_time_slab<QSpace, QTime, Backend>(
            B_A,
            x_space,
            slab,
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
        class TimeSlabType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_A_on_time_slab(
        typename Backend::SparseMatrix& B_A,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_A_on_time_slab<QSpace, QTime, Backend>(
            B_A,
            x_space,
            slab,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class TimeSlabType,
        class MFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             (!std::convertible_to<MFunction, double>)
    void assemble_mat_B_on_time_slab(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        const MFunction& M,
        double zero_tol = 1e-15)
    {
        const auto& y_space = slab.fespace_ref();
        detail::AssemblySpaceCache<XFESpaceType> x_cache(x_space);
        detail::AssemblySpaceCache<typename TimeSlabType::SpaceType> y_cache(y_space);
        detail::SourceActiveAncestorCache<XFESpaceType> ancestor_cache(x_space);

        assemble_mat_B_on_time_slab<QSpace, QTime, Backend>(
            B,
            x_space,
            slab,
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
        class TimeSlabType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_on_time_slab(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const TimeSlabType& slab,
        double zero_tol = 1e-15)
    {
        using XGT = typename XFESpaceType::GT;
        assemble_mat_B_on_time_slab<QSpace, QTime, Backend>(
            B,
            x_space,
            slab,
            coefficients::IdentityDiffusion<XGT::dim_space_v>{},
            zero_tol);
    }
}

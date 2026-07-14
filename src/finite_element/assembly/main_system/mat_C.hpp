#pragma once

#include <cstddef>
#include <exception>

#include "../constrained_dofs.hpp"
#include "../scatter.hpp"

#include "../detail/openmp_assembly.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../../geometry/cell_geometry.hpp"
#include "../detail/local_trace_bilinear_forms.hpp"
#include "../detail/space_time_basis_tables.hpp"

#include "../../../linear_algebra/assembly/local_objects.hpp"
#include "../../../linear_algebra/concepts/sparse_builder.hpp"
#include "../../../linear_algebra/concepts/sparse_matrix.hpp"

namespace finite_element::assembly
{
    template<
        int QSpace,
        int QTime,
        class Backend,
        class FESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_C(
        typename Backend::SparseMatrix& C,
        const FESpaceType& space,
        detail::AssemblySpaceCache<FESpaceType>& cache,
        double zero_tol = 1e-15)
    {
        using SpaceType     = FESpaceType;
        using GT            = typename SpaceType::GT;
        using FETraits      = typename SpaceType::FETraitsType;
        using SparseBuilder = typename Backend::SparseBuilder;

        using Tables   = detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
        using Geometry = finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;

        constexpr int dofs_per_cell = FETraits::dofs_per_cell;

        const auto& mesh         = space.mesh_ref();
        const auto& dof_handler  = space.dof_handler_ref();
        const auto& active_cells = space.active_cells();

        const int n_active_cells = static_cast<int>(active_cells.size());
        const std::size_t reserve_entries =
            static_cast<std::size_t>(n_active_cells) *
            static_cast<std::size_t>(dofs_per_cell) *
            static_cast<std::size_t>(dofs_per_cell) * 4u;

        SparseBuilder builder;

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            for (const int cell_id : active_cells)
            {
                (void)cache.dof_expansion(cell_id);
                (void)cache.geometry(cell_id);
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
                        dofs_per_cell,
                        dofs_per_cell> local;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int cell_id = active_cells[static_cast<std::size_t>(item_index)];
                        const auto& cell = mesh.cell(cell_id);

                        if (!cell.temporal_boundary[0])
                            continue;

                        const auto& expanded = cache.dof_expansion(cell_id);
                        const auto& geom     = cache.geometry(cell_id);

                        detail::assemble_local_initial_trace_mass_matrix<
                            Tables,
                            Geometry>(
                                local,
                                geom);

                        scatter_matrix(local_builder, local, expanded, expanded, zero_tol);
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
                dofs_per_cell,
                dofs_per_cell> local;

            for (const int cell_id : active_cells)
            {
                const auto& cell = mesh.cell(cell_id);

                if (!cell.temporal_boundary[0])
                    continue;

                const auto& expanded = cache.dof_expansion(cell_id);
                const auto& geom     = cache.geometry(cell_id);

                detail::assemble_local_initial_trace_mass_matrix<
                    Tables,
                    Geometry>(
                        local,
                        geom);

                scatter_matrix(builder, local, expanded, expanded, zero_tol);
            }
        }

        C.resize(dof_handler.n_true_dofs(), dof_handler.n_true_dofs());
        C.set_from_builder(builder);
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class FESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_C(
        typename Backend::SparseMatrix& C,
        const FESpaceType& space,
        double zero_tol = 1e-15)
    {
        detail::AssemblySpaceCache<FESpaceType> cache(space);
        assemble_mat_C<QSpace, QTime, Backend>(C, space, cache, zero_tol);
    }
}

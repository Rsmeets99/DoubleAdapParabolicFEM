#pragma once

#include <exception>

#include "../constrained_dofs.hpp"
#include "../scatter.hpp"

#include "../detail/openmp_assembly.hpp"
#include "../detail/assembly_diagnostics.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../../geometry/cell_geometry.hpp"
#include "../detail/local_linear_forms.hpp"
#include "../detail/space_time_basis_tables.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/vector.hpp"

namespace finite_element::assembly
{
    // -------------------------------------------------------------------------
    // Assemble vector f:
    //
    //   f_i = \int_\Sigma ell(x,t) phi_i(x,t) dx dt
    //
    // where phi_i are Y-space basis functions.
    //
    // Assembly is done into the true-DoF system.
    // -------------------------------------------------------------------------
    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class FESpaceType,
        class EllFunction>
    requires la::concepts::VectorLike<VectorLike>
    void assemble_vec_f(
        VectorLike& f,
        const FESpaceType& space,
        const EllFunction& ell,
        detail::AssemblySpaceCache<FESpaceType>& cache,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        using SpaceType = FESpaceType;
        using GT        = typename SpaceType::GT;
        using FETraits  = typename SpaceType::FETraitsType;

        using Tables   = detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
        using Geometry = finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;

        constexpr int dofs_per_cell = FETraits::dofs_per_cell;
        constexpr int n_q           = Tables::n_cell_q;

        const auto& dof_handler  = space.dof_handler_ref();
        const auto& active_cells = space.active_cells();

        f.resize(dof_handler.n_true_dofs());
        f.set_zero();

        [[maybe_unused]] const int n_active_cells =
            static_cast<int>(active_cells.size());

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            for (const int cell_id : active_cells)
            {
                (void)cache.dof_expansion(cell_id);
                (void)cache.geometry(cell_id);
            }

            auto thread_local_vectors =
                detail::make_thread_local_vectors<VectorLike>(
                    detail::openmp_thread_count(),
                    f.size());
            std::exception_ptr error;

#pragma omp parallel
            {
                try
                {
                    const int thread_id = omp_get_thread_num();
                    auto& local_rhs =
                        thread_local_vectors[static_cast<std::size_t>(thread_id)];
                    la::local::FixedLocalVector<dofs_per_cell> local;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int cell_id = active_cells[static_cast<std::size_t>(item_index)];
                        const auto& expanded = cache.dof_expansion(cell_id);
                        const auto& geom     = cache.geometry(cell_id);

                        detail::assemble_local_volume_rhs_vector<
                            Tables,
                            Geometry>(
                                local,
                                geom,
                                ell);

                        scatter_vector(local_rhs, local, expanded, zero_tol);
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
            detail::reduce_thread_local_vectors(f, thread_local_vectors);
        }
        else
#endif
        {
            la::local::FixedLocalVector<dofs_per_cell> local;

            for (const int cell_id : active_cells)
            {
                const auto& expanded = cache.dof_expansion(cell_id);
                const auto& geom     = cache.geometry(cell_id);

                detail::assemble_local_volume_rhs_vector<
                    Tables,
                    Geometry>(
                        local,
                        geom,
                        ell);

                scatter_vector(f, local, expanded, zero_tol);
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
                static_cast<std::size_t>(dofs_per_cell);
            diagnostics->source_evaluations = diagnostics->quadrature_points;
        }
    }

    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class FESpaceType,
        class EllFunction>
    requires la::concepts::VectorLike<VectorLike>
    void assemble_vec_f(
        VectorLike& f,
        const FESpaceType& space,
        const EllFunction& ell,
        double zero_tol = 1e-15)
    {
        detail::AssemblySpaceCache<FESpaceType> cache(space);
        assemble_vec_f<QSpace, QTime>(f, space, ell, cache, zero_tol, nullptr);
    }
}

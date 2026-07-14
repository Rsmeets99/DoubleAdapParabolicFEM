#pragma once

#include "../detail/local_error_linear_forms.hpp"

#include "../../detail/cell_geometry_cache.hpp"
#include "../detail/active_cell_locator_time_slab.hpp"

#include "vec_f_1d.hpp"

#include "linear_algebra/concepts/vector.hpp"
#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::error_system
{
    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class PatchScalarSpaceType,
        class PatchFluxSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    requires la::concepts::VectorLike<VectorLike>
        && (PatchScalarSpaceType::GT::dim_space_v == 1)
    void assemble_vec_g(
        VectorLike& g,
        const PatchScalarSpaceType& scalar_space,
        const PatchFluxSpaceType& flux_space,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        g.resize(scalar_space.n_dofs());
        g.set_zero();

        la::local::LocalVector local(scalar_space.n_dofs());
        finite_element::assembly::detail::zero_local_vector(local);

        const auto& patch = scalar_space.patch();
        const finite_element::assembly::detail::LocalErrorQuadratureTables1D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType> tables(flux_space, scalar_space);

        for (int patch_cell_index = 0; patch_cell_index < patch.n_cells; ++patch_cell_index)
        {
            const auto source_cell_id = patch.cell(patch_cell_index).source_cell_id;
            const int slab_id = patch.slab_id;
            const int slab_cell_id = patch.cell(patch_cell_index).slab_cell_id;

            if (context.shared_context_1d != nullptr)
            {
                const int slab_ordinal =
                    context.shared_context_1d->slab_cell_ordinal(
                        slab_id,
                        slab_cell_id);
                const int x_cell_id =
                    context.shared_context_1d
                        ->active_x_cell_for_source_cell(source_cell_id);
                const auto& slab_geom =
                    context.shared_context_1d->slab_geometry(slab_ordinal);
                const auto& x_geom =
                    context.shared_context_1d->x_geometry(x_cell_id);

                finite_element::assembly::detail::accumulate_patch_scalar_rhs_on_cell<
                    QSpace,
                    QTime>(
                        local,
                        scalar_space,
                        tables,
                        patch_cell_index,
                        slab_id,
                        lambda_tilde,
                        u_delta,
                        x_cell_id,
                        slab_geom,
                        x_geom,
                        ell,
                        M);
                continue;
            }

            const int x_cell_id =
                finite_element::assembly::detail::find_active_ancestor_cell_from_source_cell(
                    *context.x_ancestor_cache,
                    *context.x_space,
                    source_cell_id);
            const auto& slab_geom =
                (*context.slab_geometry_caches)[static_cast<std::size_t>(slab_id)]
                    .geometry(slab_cell_id);
            const auto& x_geom =
                context.x_geometry_cache->geometry(x_cell_id);

            finite_element::assembly::detail::accumulate_patch_scalar_rhs_on_cell<
                QSpace,
                QTime>(
                    local,
                    scalar_space,
                    tables,
                    patch_cell_index,
                    slab_id,
                    lambda_tilde,
                    u_delta,
                    x_cell_id,
                    slab_geom,
                    x_geom,
                    ell,
                    M);
        }

        for (int i = 0; i < scalar_space.n_dofs(); ++i)
            g[i] = local[i];
    }
}

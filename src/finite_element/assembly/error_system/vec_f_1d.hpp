#pragma once

#include <vector>

#include "../detail/local_error_linear_forms.hpp"
#include "../detail/zero_local.hpp"

#include "../../detail/cell_geometry_cache.hpp"
#include "../detail/active_cell_locator_time_slab.hpp"

#include "shared_local_error_context_1d.hpp"

#include "linear_algebra/concepts/vector.hpp"
#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::error_system
{
    template<class XSpaceType, class SlabSpaceType>
    class SharedLocalErrorContext1D;

    template<class XSpaceType, class SlabSpaceType>
    class SharedLocalErrorContext2D;

    template<class XSpaceType, class SlabSpaceType>
    struct LocalErrorProblemContext
    {
        using LocalSlabSpaceType = typename SlabSpaceType::SlabType::SpaceType;

        const XSpaceType* x_space = nullptr;
        const SlabSpaceType* slab_space = nullptr;
        finite_element::detail::CellGeometryCache<XSpaceType>* x_geometry_cache = nullptr;
        std::vector<finite_element::detail::CellGeometryCache<LocalSlabSpaceType>>*
            slab_geometry_caches = nullptr;
        finite_element::assembly::detail::SourceActiveAncestorCache<XSpaceType>* x_ancestor_cache = nullptr;
        const SharedLocalErrorContext2D<XSpaceType, SlabSpaceType>*
            shared_context = nullptr;
        const SharedLocalErrorContext1D<XSpaceType, SlabSpaceType>*
            shared_context_1d = nullptr;
    };

    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class XSpaceType,
        class SlabSpaceType>
    requires la::concepts::VectorLike<VectorLike>
        && (PatchFluxSpaceType::GT::dim_space_v == 1)
    void assemble_vec_f(
        VectorLike& f,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta)
    {
        f.resize(flux_space.n_dofs());
        f.set_zero();

        la::local::LocalVector local(flux_space.n_dofs());
        finite_element::assembly::detail::zero_local_vector(local);

        const auto& patch = flux_space.patch();
        const finite_element::assembly::detail::LocalErrorQuadratureTables1D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType> tables(flux_space, scalar_space);

        for (int patch_cell_index = 0; patch_cell_index < patch.n_cells; ++patch_cell_index)
        {
            const int slab_id = patch.slab_id;
            const int slab_cell_id = patch.cell(patch_cell_index).slab_cell_id;
            const int source_cell_id =
                patch.cell(patch_cell_index).source_cell_id;

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

                finite_element::assembly::detail::accumulate_patch_flux_rhs_on_cell<
                    QSpace,
                    QTime>(
                        local,
                        flux_space,
                        tables,
                        patch_cell_index,
                        slab_id,
                        lambda_tilde,
                        u_delta,
                        x_cell_id,
                        slab_geom,
                        x_geom);
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

            finite_element::assembly::detail::accumulate_patch_flux_rhs_on_cell<
                QSpace,
                QTime>(
                    local,
                    flux_space,
                    tables,
                    patch_cell_index,
                    slab_id,
                    lambda_tilde,
                    u_delta,
                    x_cell_id,
                    slab_geom,
                    x_geom);
        }

        for (int i = 0; i < flux_space.n_dofs(); ++i)
            f[i] = local[i];
    }
}

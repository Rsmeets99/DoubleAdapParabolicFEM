#pragma once

#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../assembly/constrained_dofs.hpp"
#include "../assembly/detail/assembly_space_cache.hpp"
#include "slab_cell_view.hpp"
#include "time_slab_space.hpp"

namespace finite_element::time_slabs
{
    struct SlabBasisMetadata
    {
        int dim_space = 0;
        int p_space = 0;
        int p_time = 0;
        int local_dof_count = 0;
        int spatial_dof_count = 0;
        int temporal_dof_count = 0;
        int scalar_components = 1;
        int vector_components = 0;
        bool scalar_valued = true;
    };

    template<typename FETraits>
    [[nodiscard]] constexpr SlabBasisMetadata slab_basis_metadata(
        const int dim_space)
    {
        return SlabBasisMetadata{
            dim_space,
            FETraits::p_space_v,
            FETraits::p_time_v,
            FETraits::dofs_per_cell,
            FETraits::total_spatial_dofs,
            FETraits::total_temporal_dofs,
            1,
            0,
            true};
    }

    template<typename GeomTraits, typename FETraits>
    class CopiedSlabDofView
    {
    public:
        using GT = GeomTraits;
        using FETraitsType = FETraits;
        using SlabSpaceType = TimeSlabSpace<GeomTraits, FETraits>;
        using SlabCellViewType = SlabCellView<GeomTraits>;
        using SlabType = typename SlabSpaceType::SlabType;
        using SlabFESpaceType = typename SlabType::SpaceType;
        using CacheType =
            finite_element::assembly::detail::AssemblySpaceCache<
                SlabFESpaceType>;

        static constexpr int local_dofs_per_cell = FETraits::dofs_per_cell;

        explicit CopiedSlabDofView(const SlabSpaceType& slab_space)
            : slab_space_(&slab_space),
              caches_(static_cast<std::size_t>(slab_space.n_slabs()))
        {}

        [[nodiscard]] TimeSlabBackend backend() const noexcept
        {
            return TimeSlabBackend::CopiedMesh;
        }

        [[nodiscard]] const char* backend_name() const noexcept
        {
            return time_slab_backend_name(backend());
        }

        [[nodiscard]] int n_slabs() const
        {
            return slab_space_ref_().n_slabs();
        }

        [[nodiscard]] int n_true_dofs() const
        {
            return slab_space_ref_().n_true_dofs();
        }

        [[nodiscard]] int n_true_dofs_on_slab(int slab_id) const
        {
            return slab_space_ref_()
                .slab(slab_id)
                .fespace_ref()
                .dof_handler_ref()
                .n_true_dofs();
        }

        [[nodiscard]] int local_dof_count() const noexcept
        {
            return local_dofs_per_cell;
        }

        [[nodiscard]] SlabBasisMetadata basis_metadata() const noexcept
        {
            return slab_basis_metadata<FETraits>(GeomTraits::dim_space_v);
        }

        [[nodiscard]] int polynomial_degree_space() const noexcept
        {
            return FETraits::p_space_v;
        }

        [[nodiscard]] int polynomial_degree_time() const noexcept
        {
            return FETraits::p_time_v;
        }

        [[nodiscard]] finite_element::assembly::CellRestriction
        cell_restriction(const SlabCellViewType& slab_cell) const
        {
            require_compatible_cell_(slab_cell);
            const int copied_cell_id =
                *slab_cell.copied_slab_local_cell_id();
            return cache_for_slab_(slab_cell.slab_id())
                .cell_restriction(copied_cell_id);
        }

        [[nodiscard]] bool local_row_is_boundary_eliminated(
            const SlabCellViewType& slab_cell,
            int local_dof) const
        {
            check_local_dof_(local_dof);
            return cell_restriction(slab_cell)
                .rows[static_cast<std::size_t>(local_dof)]
                .empty();
        }

    private:
        [[nodiscard]] const SlabSpaceType& slab_space_ref_() const
        {
            if (slab_space_ == nullptr)
                throw std::runtime_error(
                    "CopiedSlabDofView: slab space pointer is null.");
            return *slab_space_;
        }

        [[nodiscard]] CacheType& cache_for_slab_(int slab_id) const
        {
            if (slab_id < 0 || slab_id >= slab_space_ref_().n_slabs())
                throw std::runtime_error(
                    "CopiedSlabDofView::cache_for_slab_: invalid slab id.");

            auto& cache = caches_[static_cast<std::size_t>(slab_id)];
            if (!cache)
            {
                cache = std::make_unique<CacheType>(
                    slab_space_ref_().slab(slab_id).fespace_ref());
            }
            return *cache;
        }

        void require_compatible_cell_(
            const SlabCellViewType& slab_cell) const
        {
            if (slab_cell.backend() != TimeSlabBackend::CopiedMesh ||
                !slab_cell.copied_slab_local_cell_id().has_value())
            {
                throw std::runtime_error(
                    "CopiedSlabDofView::cell_restriction: expected copied "
                    "slab cell view.");
            }
        }

        static void check_local_dof_(int local_dof)
        {
            if (local_dof < 0 || local_dof >= local_dofs_per_cell)
            {
                throw std::runtime_error(
                    "CopiedSlabDofView: local DoF index out of range.");
            }
        }

        const SlabSpaceType* slab_space_ = nullptr;
        mutable std::vector<std::unique_ptr<CacheType>> caches_{};
    };

    template<typename GeomTraits, typename FETraits>
    [[nodiscard]] inline CopiedSlabDofView<GeomTraits, FETraits>
    make_copied_slab_dof_view(
        const TimeSlabSpace<GeomTraits, FETraits>& slab_space)
    {
        return CopiedSlabDofView<GeomTraits, FETraits>(slab_space);
    }

}

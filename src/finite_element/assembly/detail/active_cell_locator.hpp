#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <stdexcept>

namespace finite_element::assembly::detail
{
    template<class CoarseFESpaceType>
    class ActiveAncestorCache
    {
    public:
        explicit ActiveAncestorCache(const CoarseFESpaceType& coarse_space)
            : coarse_space_(&coarse_space),
              cached_ancestors_(coarse_space.mesh_ref().n_cells(), unknown_ancestor_),
              cached_storage_version_(current_storage_version_()),
              cached_active_version_(current_active_version_())
        {}

        [[nodiscard]] int find(int fine_cell_id)
        {
            ensure_current_();
            if (fine_cell_id < 0 ||
                static_cast<std::size_t>(fine_cell_id) >= cached_ancestors_.size())
            {
                throw std::runtime_error(
                    "ActiveAncestorCache::find: invalid fine cell id.");
            }

            const int cached = cached_ancestors_[static_cast<std::size_t>(fine_cell_id)];
            if (cached != unknown_ancestor_)
            {
                if (cached < 0)
                    throw std::runtime_error(
                        "ActiveAncestorCache::find: no active ancestor found in coarse_space.");
                return cached;
            }

            const auto& mesh = coarse_space_->mesh_ref();

            std::vector<int> visited;
            int cell_id = fine_cell_id;
            int result  = no_active_ancestor_;

            while (cell_id >= 0)
            {
                if (static_cast<std::size_t>(cell_id) >= cached_ancestors_.size())
                {
                    throw std::runtime_error(
                        "ActiveAncestorCache::find: invalid ancestor cell id.");
                }

                const int ancestor = cached_ancestors_[static_cast<std::size_t>(cell_id)];
                if (ancestor != unknown_ancestor_)
                {
                    result = ancestor;
                    break;
                }

                visited.push_back(cell_id);

                if (coarse_space_->is_active_cell(cell_id))
                {
                    result = cell_id;
                    break;
                }

                cell_id = mesh.cell(cell_id).parent_id;
            }

            for (const int visited_cell_id : visited)
                cached_ancestors_[static_cast<std::size_t>(visited_cell_id)] = result;

            if (result < 0)
            {
                throw std::runtime_error(
                    "ActiveAncestorCache::find: no active ancestor found in coarse_space.");
            }

            return result;
        }

    private:
        static constexpr int unknown_ancestor_   = -2;
        static constexpr int no_active_ancestor_ = -1;

        [[nodiscard]] std::uint64_t current_storage_version_() const noexcept
        {
            if constexpr (requires { coarse_space_->mesh_ref().storage_version(); })
                return coarse_space_->mesh_ref().storage_version();
            else
                return 0;
        }

        [[nodiscard]] std::uint64_t current_active_version_() const noexcept
        {
            if constexpr (requires { coarse_space_->active_version(); })
                return coarse_space_->active_version();
            else
                return 0;
        }

        void ensure_current_()
        {
            const auto storage_version = current_storage_version_();
            const auto active_version = current_active_version_();
            const auto needed =
                static_cast<std::size_t>(coarse_space_->mesh_ref().n_cells());
            if (cached_storage_version_ != storage_version ||
                cached_active_version_ != active_version)
            {
                cached_ancestors_.assign(needed, unknown_ancestor_);
                cached_storage_version_ = storage_version;
                cached_active_version_ = active_version;
                return;
            }

            if (cached_ancestors_.size() < needed)
                cached_ancestors_.resize(needed, unknown_ancestor_);
        }

        const CoarseFESpaceType* coarse_space_ = nullptr;
        std::vector<int> cached_ancestors_{};
        std::uint64_t cached_storage_version_ = 0;
        std::uint64_t cached_active_version_ = 0;
    };

    // -------------------------------------------------------------------------
    // Find the active ancestor in coarse_space that contains fine_cell_id.
    //
    // Assumptions:
    //   1. coarse_space and fine_space share the same underlying mesh object
    //   2. every active fine cell lies inside an active coarse ancestor cell
    // -------------------------------------------------------------------------
    template<class CoarseFESpaceType, class FineFESpaceType>
    int find_active_ancestor_cell(
        const CoarseFESpaceType& coarse_space,
        const FineFESpaceType& fine_space,
        int fine_cell_id)
    {
        const auto* coarse_mesh_ptr = &coarse_space.mesh_ref();
        const auto* fine_mesh_ptr   = &fine_space.mesh_ref();

        if (coarse_mesh_ptr != fine_mesh_ptr)
        {
            throw std::runtime_error(
                "find_active_ancestor_cell: coarse_space and fine_space do not share the same mesh.");
        }

        const auto& mesh = coarse_space.mesh_ref();

        ActiveAncestorCache<CoarseFESpaceType> cache(coarse_space);
        return cache.find(fine_cell_id);
    }

    template<class CoarseFESpaceType, class FineFESpaceType>
    int find_active_ancestor_cell(
        ActiveAncestorCache<CoarseFESpaceType>& cache,
        const CoarseFESpaceType& coarse_space,
        const FineFESpaceType& fine_space,
        int fine_cell_id)
    {
        const auto* coarse_mesh_ptr = &coarse_space.mesh_ref();
        const auto* fine_mesh_ptr   = &fine_space.mesh_ref();

        if (coarse_mesh_ptr != fine_mesh_ptr)
        {
            throw std::runtime_error(
                "find_active_ancestor_cell: coarse_space and fine_space do not share the same mesh.");
        }

        return cache.find(fine_cell_id);
    }
}

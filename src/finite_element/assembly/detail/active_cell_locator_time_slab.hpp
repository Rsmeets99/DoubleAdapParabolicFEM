#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace finite_element::assembly::detail
{
    template<class XFESpaceType>
    class SourceActiveAncestorCache
    {
    public:
        explicit SourceActiveAncestorCache(const XFESpaceType& x_space)
            : x_space_(&x_space),
              cached_ancestors_(x_space.mesh_ref().n_cells(), unknown_ancestor_)
        {}

        [[nodiscard]] int find(int source_cell_id)
        {
            if (source_cell_id < 0 ||
                static_cast<std::size_t>(source_cell_id) >= cached_ancestors_.size())
            {
                throw std::runtime_error(
                    "SourceActiveAncestorCache::find: invalid source cell id.");
            }

            const int cached = cached_ancestors_[static_cast<std::size_t>(source_cell_id)];
            if (cached != unknown_ancestor_)
            {
                if (cached < 0)
                    throw std::runtime_error(
                        "SourceActiveAncestorCache::find: no active ancestor found in X-space.");
                return cached;
            }

            const auto& mesh = x_space_->mesh_ref();

            std::vector<int> visited;
            int cell_id = source_cell_id;
            int result  = no_active_ancestor_;

            while (cell_id >= 0)
            {
                if (static_cast<std::size_t>(cell_id) >= cached_ancestors_.size())
                {
                    throw std::runtime_error(
                        "SourceActiveAncestorCache::find: invalid ancestor cell id.");
                }

                const int ancestor = cached_ancestors_[static_cast<std::size_t>(cell_id)];
                if (ancestor != unknown_ancestor_)
                {
                    result = ancestor;
                    break;
                }

                visited.push_back(cell_id);

                if (x_space_->is_active_cell(cell_id))
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
                    "SourceActiveAncestorCache::find: no active ancestor found in X-space.");
            }

            return result;
        }

        [[nodiscard]] std::size_t estimated_live_bytes() const noexcept
        {
            return sizeof(*this) +
                   cached_ancestors_.capacity() * sizeof(int);
        }

    private:
        static constexpr int unknown_ancestor_   = -2;
        static constexpr int no_active_ancestor_ = -1;

        const XFESpaceType* x_space_ = nullptr;
        std::vector<int> cached_ancestors_{};
    };

    // -------------------------------------------------------------------------
    // Given a source cell id from the original Y-mesh hierarchy, find the active
    // ancestor cell in X^delta.
    //
    // This is the correct lookup for time-slab reconstruction:
    //
    //   slab-local synthetic cell
    //       -> source cell in original Y^delta
    //       -> active ancestor cell in X^delta
    //
    // Assumption:
    //   - source_cell_id belongs to the same underlying mesh hierarchy as x_space
    // -------------------------------------------------------------------------
    template<class XFESpaceType>
    [[nodiscard]] int find_active_ancestor_cell_from_source_cell(
        const XFESpaceType& x_space,
        int source_cell_id)
    {
        if (source_cell_id < 0)
            throw std::runtime_error(
                "find_active_ancestor_cell_from_source_cell: negative source cell id.");

        SourceActiveAncestorCache<XFESpaceType> cache(x_space);
        return cache.find(source_cell_id);
    }

    template<class XFESpaceType>
    [[nodiscard]] int find_active_ancestor_cell_from_source_cell(
        SourceActiveAncestorCache<XFESpaceType>& cache,
        const XFESpaceType&,
        int source_cell_id)
    {
        return cache.find(source_cell_id);
    }
}

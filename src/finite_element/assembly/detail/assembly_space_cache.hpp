#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../detail/cell_geometry_cache.hpp"
#include "../constrained_dofs.hpp"

namespace finite_element::assembly::detail
{
    template<class FESpaceType>
    class AssemblySpaceCache
    {
    public:
        using SpaceType    = FESpaceType;
        using GeometryData =
            typename finite_element::detail::CellGeometryCache<SpaceType>::GeometryData;

        explicit AssemblySpaceCache(const SpaceType& space)
            : space_(&space),
              geometry_cache_(space),
              cell_restrictions_(space.mesh_ref().n_cells())
        {}

        [[nodiscard]] const GeometryData& geometry(int cell_id)
        {
            return geometry_cache_.geometry(cell_id);
        }

        [[nodiscard]] const CellRestriction& cell_restriction(int cell_id) const
        {
            ensure_cell_restrictions_current_();
            check_cell_id_(cell_id);

            auto& entry = cell_restrictions_[static_cast<std::size_t>(cell_id)];
            if (!entry.has_value())
            {
                entry.emplace(build_cell_restriction(*space_, cell_id));
                ++cell_restriction_builds_;
            }
            else
            {
                ++cell_restriction_hits_;
            }

            return *entry;
        }

        [[nodiscard]] const LocalDofExpansion& dof_expansion(int cell_id) const
        {
            return cell_restriction(cell_id);
        }

        void clear()
        {
            geometry_cache_.clear();
            for (auto& entry : cell_restrictions_)
                entry.reset();
            cached_dof_distribution_version_ =
                current_dof_distribution_version_();
        }

        [[nodiscard]] std::size_t cell_restriction_build_count() const noexcept
        {
            return cell_restriction_builds_;
        }

        [[nodiscard]] std::size_t cell_restriction_cache_hit_count() const noexcept
        {
            return cell_restriction_hits_;
        }

        [[nodiscard]] std::size_t geometry_cache_hit_count() const noexcept
        {
            return geometry_cache_.cache_hit_count();
        }

        [[nodiscard]] std::size_t geometry_cache_miss_count() const noexcept
        {
            return geometry_cache_.cache_miss_count();
        }

        [[nodiscard]] std::size_t cached_cell_restriction_count() const
        {
            ensure_cell_restrictions_current_();
            std::size_t count = 0;
            for (const auto& entry : cell_restrictions_)
            {
                if (entry.has_value())
                    ++count;
            }
            return count;
        }

        [[nodiscard]] std::size_t estimated_live_bytes() const
        {
            ensure_cell_restrictions_current_();
            return sizeof(*this) +
                   geometry_cache_.estimated_live_bytes() +
                   cell_restrictions_.capacity() *
                       sizeof(std::optional<CellRestriction>);
        }

    private:
        [[nodiscard]] std::uint64_t current_dof_distribution_version_() const noexcept
        {
            if constexpr (requires { space_->dof_distribution_version(); })
                return space_->dof_distribution_version();
            else
                return 0;
        }

        void ensure_cell_restrictions_current_() const
        {
            const auto version = current_dof_distribution_version_();
            if (cached_dof_distribution_version_ != version)
            {
                for (auto& entry : cell_restrictions_)
                    entry.reset();
                cached_dof_distribution_version_ = version;
            }

            const auto needed =
                static_cast<std::size_t>(space_->mesh_ref().n_cells());
            if (cell_restrictions_.size() < needed)
                cell_restrictions_.resize(needed);
        }

        void check_cell_id_(int cell_id) const
        {
            if (cell_id < 0 ||
                static_cast<std::size_t>(cell_id) >= cell_restrictions_.size())
            {
                throw std::runtime_error("AssemblySpaceCache: invalid cell id.");
            }
        }

        const SpaceType* space_ = nullptr;
        finite_element::detail::CellGeometryCache<SpaceType> geometry_cache_;
        mutable std::vector<std::optional<CellRestriction>> cell_restrictions_{};
        mutable std::uint64_t cached_dof_distribution_version_ = 0;
        mutable std::size_t cell_restriction_builds_ = 0;
        mutable std::size_t cell_restriction_hits_ = 0;
    };
}

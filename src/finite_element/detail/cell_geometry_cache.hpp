#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../geometry/cell_geometry.hpp"

namespace finite_element::detail
{
    template<class FESpaceType>
    class CellGeometryCache
    {
    public:
        using SpaceType    = FESpaceType;
        using GT           = typename SpaceType::GT;
        using Geometry     = finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;
        using GeometryData = typename Geometry::Data;

        explicit CellGeometryCache(const SpaceType& space)
            : space_(&space),
              geometries_(space.mesh_ref().n_cells())
        {}

        [[nodiscard]] const GeometryData& geometry(int cell_id)
        {
            ensure_current_();
            check_cell_id_(cell_id);

            auto& entry = geometries_[static_cast<std::size_t>(cell_id)];
            if (!entry.has_value())
            {
                entry.emplace(Geometry::make(*space_, cell_id));
                ++misses_;
            }
            else
            {
                ++hits_;
            }

            return *entry;
        }

        void clear()
        {
            for (auto& entry : geometries_)
                entry.reset();
            cached_storage_version_ = current_storage_version_();
        }

        [[nodiscard]] std::size_t cache_hit_count() const noexcept
        {
            return hits_;
        }

        [[nodiscard]] std::size_t cache_miss_count() const noexcept
        {
            return misses_;
        }

        [[nodiscard]] std::size_t estimated_live_bytes() const noexcept
        {
            return sizeof(*this) +
                   geometries_.capacity() *
                       sizeof(std::optional<GeometryData>);
        }

    private:
        [[nodiscard]] std::uint64_t current_storage_version_() const noexcept
        {
            if constexpr (requires { space_->mesh_ref().storage_version(); })
                return space_->mesh_ref().storage_version();
            else
                return 0;
        }

        void ensure_current_()
        {
            const auto version = current_storage_version_();
            if (cached_storage_version_ != version)
            {
                for (auto& entry : geometries_)
                    entry.reset();
                cached_storage_version_ = version;
            }

            const auto needed =
                static_cast<std::size_t>(space_->mesh_ref().n_cells());
            if (geometries_.size() < needed)
                geometries_.resize(needed);
        }

        void check_cell_id_(int cell_id) const
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= geometries_.size())
                throw std::runtime_error("CellGeometryCache: invalid cell id.");
        }

        const SpaceType* space_ = nullptr;
        std::vector<std::optional<GeometryData>> geometries_{};
        std::uint64_t cached_storage_version_ = 0;
        std::size_t hits_ = 0;
        std::size_t misses_ = 0;
    };
}

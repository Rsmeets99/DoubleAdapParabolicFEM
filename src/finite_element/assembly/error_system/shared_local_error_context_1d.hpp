#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../../detail/cell_geometry_cache.hpp"
#include "../detail/active_cell_locator_time_slab.hpp"

namespace finite_element::assembly::error_system
{
    template<class XSpaceType, class SlabSpaceType>
    class SharedLocalErrorContext1D
    {
    public:
        using XSpace = XSpaceType;
        using SlabSpace = SlabSpaceType;
        using LocalSlabSpace = typename SlabSpace::SlabType::SpaceType;
        using XGeometry =
            finite_element::geometry::CellGeometry<
                XSpace,
                XSpace::GT::dim_space_v>;
        using SlabGeometry =
            finite_element::geometry::CellGeometry<
                LocalSlabSpace,
                LocalSlabSpace::GT::dim_space_v>;
        using XGeometryData = typename XGeometry::Data;
        using SlabGeometryData = typename SlabGeometry::Data;

        static_assert(
            XSpace::GT::dim_space_v == 1 &&
            LocalSlabSpace::GT::dim_space_v == 1,
            "SharedLocalErrorContext1D requires 1D space-time cells.");

        struct SlabCellMetadata
        {
            int slab_id = -1;
            int slab_cell_id = -1;
            int source_cell_id = -1;
            int active_x_cell_id = -1;
            int slab_time_begin_id = -1;
            int slab_time_end_id = -1;
            double slab_t_begin = 0.0;
            double slab_t_end = 0.0;
            double source_t_begin = 0.0;
            double source_t_end = 0.0;
        };

        struct ComparisonStats
        {
            double sample_geometry_max_abs_diff = 0.0;
            double sample_slab_geometry_max_abs_diff = 0.0;
            double sample_ancestor_mismatch_count = 0.0;
            double sample_count = 0.0;
        };

        SharedLocalErrorContext1D() = default;

        template<class ActiveSlabCellVector>
        static SharedLocalErrorContext1D build(
            const XSpace& x_space,
            const SlabSpace& slab_space,
            const ActiveSlabCellVector& active_slab_cells)
        {
            SharedLocalErrorContext1D result;
            result.x_space_ = &x_space;
            result.slab_space_ = &slab_space;
            result.source_to_active_x_cell_.assign(
                static_cast<std::size_t>(x_space.mesh_ref().n_cells()),
                unknown_cell_);
            result.x_geometry_by_cell_id_.resize(
                static_cast<std::size_t>(x_space.mesh_ref().n_cells()));
            result.slab_geometry_by_ordinal_.reserve(active_slab_cells.size());
            result.slab_cell_metadata_.reserve(active_slab_cells.size());
            result.slab_cell_ordinal_by_key_.reserve(active_slab_cells.size());

            finite_element::assembly::detail::
                SourceActiveAncestorCache<XSpace> ancestor_cache(x_space);

            for (const auto& cell : active_slab_cells)
            {
                const int slab_id = cell.slab_id;
                const int slab_cell_id = cell.slab_cell_id;
                const auto& slab = slab_space.slab(slab_id);
                const auto& info = slab.sliced_cell_info(slab_cell_id);
                const int source_cell_id = info.source_cell_id;
                const int active_x_cell_id =
                    ancestor_cache.find(source_cell_id);

                result.ensure_source_index_(source_cell_id);
                if (result.source_to_active_x_cell_[
                        static_cast<std::size_t>(source_cell_id)] ==
                    unknown_cell_)
                {
                    result.source_to_active_x_cell_[
                        static_cast<std::size_t>(source_cell_id)] =
                        active_x_cell_id;
                    ++result.ancestor_count_;
                }

                result.ensure_x_geometry_index_(active_x_cell_id);
                auto& x_geometry =
                    result.x_geometry_by_cell_id_[
                        static_cast<std::size_t>(active_x_cell_id)];
                if (!x_geometry.has_value())
                {
                    x_geometry.emplace(
                        XGeometry::make(x_space, active_x_cell_id));
                    ++result.x_geometry_count_;
                }

                const int ordinal =
                    static_cast<int>(result.slab_geometry_by_ordinal_.size());
                result.slab_geometry_by_ordinal_.push_back(
                    SlabGeometry::make(slab.fespace_ref(), slab_cell_id));
                result.slab_cell_metadata_.push_back(
                    SlabCellMetadata{
                        slab_id,
                        slab_cell_id,
                        source_cell_id,
                        active_x_cell_id,
                        info.slab_time_begin_id,
                        info.slab_time_end_id,
                        info.t_begin,
                        info.t_end,
                        info.source_t_begin,
                        info.source_t_end});
                result.slab_cell_ordinal_by_key_.emplace(
                    key_(slab_id, slab_cell_id),
                    ordinal);
            }

            return result;
        }

        [[nodiscard]] int active_slab_cell_count() const noexcept
        {
            return static_cast<int>(slab_cell_metadata_.size());
        }

        [[nodiscard]] int x_geometry_count() const noexcept
        {
            return x_geometry_count_;
        }

        [[nodiscard]] int slab_geometry_count() const noexcept
        {
            return static_cast<int>(slab_geometry_by_ordinal_.size());
        }

        [[nodiscard]] int ancestor_count() const noexcept
        {
            return ancestor_count_;
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return sizeof(*this) +
                   x_geometry_by_cell_id_.capacity() *
                       sizeof(std::optional<XGeometryData>) +
                   source_to_active_x_cell_.capacity() * sizeof(int) +
                   slab_geometry_by_ordinal_.capacity() *
                       sizeof(SlabGeometryData) +
                   slab_cell_metadata_.capacity() * sizeof(SlabCellMetadata) +
                   slab_cell_ordinal_by_key_.size() *
                       (sizeof(std::pair<const std::uint64_t, int>) +
                        3u * sizeof(void*));
        }

        [[nodiscard]] int slab_cell_ordinal(
            int slab_id,
            int slab_cell_id) const
        {
            const auto it =
                slab_cell_ordinal_by_key_.find(key_(slab_id, slab_cell_id));
            if (it == slab_cell_ordinal_by_key_.end())
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: slab cell not indexed.");
            return it->second;
        }

        [[nodiscard]] const SlabCellMetadata& slab_cell_metadata(
            int ordinal) const
        {
            if (ordinal < 0 ||
                static_cast<std::size_t>(ordinal) >=
                    slab_cell_metadata_.size())
            {
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: invalid slab-cell ordinal.");
            }
            return slab_cell_metadata_[static_cast<std::size_t>(ordinal)];
        }

        [[nodiscard]] const SlabGeometryData& slab_geometry(
            int ordinal) const
        {
            if (ordinal < 0 ||
                static_cast<std::size_t>(ordinal) >=
                    slab_geometry_by_ordinal_.size())
            {
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: invalid slab geometry ordinal.");
            }
            return slab_geometry_by_ordinal_[static_cast<std::size_t>(ordinal)];
        }

        [[nodiscard]] int active_x_cell_for_source_cell(
            int source_cell_id) const
        {
            if (source_cell_id < 0 ||
                static_cast<std::size_t>(source_cell_id) >=
                    source_to_active_x_cell_.size())
            {
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: invalid source cell id.");
            }
            const int cell_id =
                source_to_active_x_cell_[
                    static_cast<std::size_t>(source_cell_id)];
            if (cell_id == unknown_cell_)
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: source cell not indexed.");
            return cell_id;
        }

        [[nodiscard]] const XGeometryData& x_geometry(int cell_id) const
        {
            if (cell_id < 0 ||
                static_cast<std::size_t>(cell_id) >=
                    x_geometry_by_cell_id_.size())
            {
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: invalid X cell id.");
            }
            const auto& entry =
                x_geometry_by_cell_id_[static_cast<std::size_t>(cell_id)];
            if (!entry.has_value())
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: X geometry not indexed.");
            return *entry;
        }

        template<
            class MutableXGeometryCache,
            class MutableAncestorCache,
            class MutableSlabGeometryCaches>
        [[nodiscard]] ComparisonStats compare_sample(
            MutableXGeometryCache& x_geometry_cache,
            MutableAncestorCache& ancestor_cache,
            MutableSlabGeometryCaches& slab_geometry_caches,
            int max_samples = 2048) const
        {
            ComparisonStats stats;
            if (slab_cell_metadata_.empty())
                return stats;

            const int sample_limit = std::max(1, max_samples);
            const int stride = std::max(
                1,
                static_cast<int>(slab_cell_metadata_.size()) / sample_limit);

            for (int ordinal = 0;
                 ordinal < static_cast<int>(slab_cell_metadata_.size());
                 ordinal += stride)
            {
                const auto& meta =
                    slab_cell_metadata_[static_cast<std::size_t>(ordinal)];
                const int mutable_ancestor =
                    ancestor_cache.find(meta.source_cell_id);
                if (mutable_ancestor != meta.active_x_cell_id)
                    stats.sample_ancestor_mismatch_count += 1.0;

                const auto& mutable_x_geometry =
                    x_geometry_cache.geometry(mutable_ancestor);
                const auto& shared_x_geometry =
                    x_geometry(meta.active_x_cell_id);
                stats.sample_geometry_max_abs_diff =
                    std::max(
                        stats.sample_geometry_max_abs_diff,
                        geometry_max_abs_diff_(
                            mutable_x_geometry,
                            shared_x_geometry));

                const auto& mutable_slab_geometry =
                    slab_geometry_caches[
                        static_cast<std::size_t>(meta.slab_id)]
                        .geometry(meta.slab_cell_id);
                const auto& shared_slab_geometry = slab_geometry(ordinal);
                stats.sample_slab_geometry_max_abs_diff =
                    std::max(
                        stats.sample_slab_geometry_max_abs_diff,
                        geometry_max_abs_diff_(
                            mutable_slab_geometry,
                            shared_slab_geometry));

                stats.sample_count += 1.0;
            }

            return stats;
        }

    private:
        static constexpr int unknown_cell_ = -1;

        [[nodiscard]] static std::uint64_t key_(
            int slab_id,
            int slab_cell_id) noexcept
        {
            return (static_cast<std::uint64_t>(
                        static_cast<std::uint32_t>(slab_id))
                    << 32u) |
                   static_cast<std::uint32_t>(slab_cell_id);
        }

        void ensure_source_index_(int source_cell_id) const
        {
            if (source_cell_id < 0 ||
                static_cast<std::size_t>(source_cell_id) >=
                    source_to_active_x_cell_.size())
            {
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: source cell id out of range.");
            }
        }

        void ensure_x_geometry_index_(int cell_id) const
        {
            if (cell_id < 0 ||
                static_cast<std::size_t>(cell_id) >=
                    x_geometry_by_cell_id_.size())
            {
                throw std::runtime_error(
                    "SharedLocalErrorContext1D: X cell id out of range.");
            }
        }

        template<class A, class B>
        [[nodiscard]] static double geometry_max_abs_diff_(
            const A& a,
            const B& b) noexcept
        {
            double result = 0.0;
            auto update = [&](double lhs, double rhs) noexcept
            {
                result = std::max(result, std::abs(lhs - rhs));
            };

            update(a.map.space.a, b.map.space.a);
            update(a.map.space.b, b.map.space.b);
            update(a.map.time.a, b.map.time.a);
            update(a.map.time.b, b.map.time.b);
            update(a.x0, b.x0);
            update(a.hx, b.hx);
            update(a.t0, b.t0);
            update(a.ht, b.ht);
            update(a.inv_hx, b.inv_hx);
            update(a.inv_ht, b.inv_ht);

            return result;
        }

        const XSpace* x_space_ = nullptr;
        const SlabSpace* slab_space_ = nullptr;
        std::vector<std::optional<XGeometryData>> x_geometry_by_cell_id_{};
        std::vector<int> source_to_active_x_cell_{};
        std::vector<SlabGeometryData> slab_geometry_by_ordinal_{};
        std::vector<SlabCellMetadata> slab_cell_metadata_{};
        std::unordered_map<std::uint64_t, int> slab_cell_ordinal_by_key_{};
        int x_geometry_count_ = 0;
        int ancestor_count_ = 0;
    };
}

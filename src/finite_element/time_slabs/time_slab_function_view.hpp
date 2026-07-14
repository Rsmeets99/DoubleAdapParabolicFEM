#pragma once

#include <concepts>
#include <cstddef>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "../detail/cell_geometry_cache.hpp"
#include "../geometry/cell_geometry.hpp"
#include "slab_cell_view.hpp"
#include "slab_dof_view.hpp"
#include "time_slab_function.hpp"

namespace finite_element::time_slabs
{
    template<class View>
    concept TimeSlabFunctionView = requires(
        const View& view,
        const typename View::SlabCellViewType& slab_cell,
        const typename View::HotSlabCellRef& hot_slab_cell,
        const typename View::SpaceTimePoint& point,
        const typename View::GeometryData& geom,
        int slab_id,
        int true_dof,
        int local_dof)
    {
        typename View::HotSlabCellRef;
        { view.backend() } -> std::same_as<TimeSlabBackend>;
        { view.backend_name() } -> std::convertible_to<const char*>;
        { view.n_slabs() } -> std::convertible_to<int>;
        { view.n_true_dofs() } -> std::convertible_to<int>;
        { view.n_true_dofs_on_slab(slab_id) } -> std::convertible_to<int>;
        { view.slab_cell_views_on_slab(slab_id) }
            -> std::same_as<std::vector<typename View::SlabCellViewType>>;
        { view.slab_true_coefficient(slab_id, true_dof) }
            -> std::convertible_to<double>;
        { view.local_coefficient(slab_cell, local_dof) }
            -> std::convertible_to<double>;
        { view.geometry(slab_cell) } -> std::same_as<typename View::GeometryData>;
        { view.value_on_cell(slab_cell, point, geom) }
            -> std::convertible_to<double>;
        { view.gradient_on_cell(slab_cell, point, geom) }
            -> std::same_as<typename View::GradientType>;
        { view.source_cell_id(slab_cell) } -> std::convertible_to<int>;
        { view.geometry(hot_slab_cell) }
            -> std::same_as<typename View::GeometryData>;
        { view.gradient_on_cell(hot_slab_cell, point, geom) }
            -> std::same_as<typename View::GradientType>;
        { view.source_cell_id(hot_slab_cell) } -> std::convertible_to<int>;
        { view.make_geometry_cache() }
            -> std::same_as<typename View::GeometryCacheType>;
    };

    template<class TimeSlabFunctionType>
    class CopiedTimeSlabFunctionView
    {
    public:
        using FunctionType = TimeSlabFunctionType;
        using SlabSpaceType = typename FunctionType::SlabSpaceType;
        using GT = typename FunctionType::GT;
        using FETraitsType = typename FunctionType::FETraits;
        using Vector = typename FunctionType::Vector;
        using SlabCellViewType = SlabCellView<GT>;
        using LocalFunction = typename FunctionType::LocalFunction;
        using LocalSpaceType = typename FunctionType::LocalSpaceType;
        using Geometry = typename LocalFunction::Geometry;
        using GeometryData = typename LocalFunction::GeometryData;
        using SpaceTimePoint = typename FunctionType::SpaceTimePoint;
        using GradientType = typename FunctionType::GradientType;
        using DofViewType = CopiedSlabDofView<GT, FETraitsType>;

        struct HotSlabCellRef
        {
            TimeSlabBackend backend = TimeSlabBackend::CopiedMesh;
            int slab_id = -1;
            int slab_local_ordinal = -1;
            int source_cell_id = -1;
            int copied_slab_cell_id = -1;
            int slab_time_begin_id = -1;
            int slab_time_end_id = -1;
            double slab_t_begin = 0.0;
            double slab_t_end = 0.0;
            double source_t_begin = 0.0;
            double source_t_end = 0.0;
        };

        class GeometryCache
        {
        public:
            explicit GeometryCache(const FunctionType& function)
            {
                const auto& slab_space = function.slab_space();
                slab_geometry_caches_.reserve(
                    static_cast<std::size_t>(slab_space.n_slabs()));
                for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
                {
                    slab_geometry_caches_.emplace_back(
                        slab_space.slab(slab_id).fespace_ref());
                }
            }

            [[nodiscard]] const GeometryData& geometry(
                const HotSlabCellRef& slab_cell)
            {
                check_slab_cell_(slab_cell);
                ++copied_geometry_cache_lookups_;
                return slab_geometry_caches_[
                    static_cast<std::size_t>(slab_cell.slab_id)]
                    .geometry(slab_cell.copied_slab_cell_id);
            }

            [[nodiscard]] std::size_t cache_hit_count() const noexcept
            {
                std::size_t count = 0;
                for (const auto& cache : slab_geometry_caches_)
                    count += cache.cache_hit_count();
                return count;
            }

            [[nodiscard]] std::size_t cache_miss_count() const noexcept
            {
                std::size_t count = 0;
                for (const auto& cache : slab_geometry_caches_)
                    count += cache.cache_miss_count();
                return count;
            }

            [[nodiscard]] std::size_t
            copied_geometry_cache_lookup_count() const noexcept
            {
                return copied_geometry_cache_lookups_;
            }

            [[nodiscard]] std::size_t
            virtual_geometry_constructed_count() const noexcept
            {
                return 0;
            }

        private:
            void check_slab_cell_(const HotSlabCellRef& slab_cell) const
            {
                if (slab_cell.backend != TimeSlabBackend::CopiedMesh ||
                    slab_cell.slab_id < 0 ||
                    static_cast<std::size_t>(slab_cell.slab_id) >=
                        slab_geometry_caches_.size() ||
                    slab_cell.copied_slab_cell_id < 0)
                {
                    throw std::runtime_error(
                        "CopiedTimeSlabFunctionView::GeometryCache: invalid "
                        "copied hot slab-cell ref.");
                }
            }

            std::vector<
                finite_element::detail::CellGeometryCache<LocalSpaceType>>
                slab_geometry_caches_{};
            std::size_t copied_geometry_cache_lookups_ = 0;
        };

        using GeometryCacheType = GeometryCache;

        explicit CopiedTimeSlabFunctionView(const FunctionType& function)
            : function_(&function),
              dof_view_(function.slab_space())
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
            return function_ref_().n_slabs();
        }

        [[nodiscard]] int n_true_dofs() const
        {
            return function_ref_().slab_space().n_true_dofs();
        }

        [[nodiscard]] int n_true_dofs_on_slab(const int slab_id) const
        {
            return dof_view_.n_true_dofs_on_slab(slab_id);
        }

        [[nodiscard]] std::vector<SlabCellViewType> slab_cell_views_on_slab(
            const int slab_id) const
        {
            return copied_slab_cell_views_on_slab(
                function_ref_().slab_space(),
                slab_id);
        }

        template<class Callback>
        void for_each_slab_cell(
            const int slab_id,
            Callback&& callback) const
        {
            const auto& slab = function_ref_().slab_space().slab(slab_id);
            for (int local = 0; local < slab.n_active_cells(); ++local)
            {
                const auto& info = slab.sliced_cell_info(local);
                HotSlabCellRef ref;
                ref.backend = TimeSlabBackend::CopiedMesh;
                ref.slab_id = info.slab_id;
                ref.slab_local_ordinal = local;
                ref.source_cell_id = info.source_cell_id;
                ref.copied_slab_cell_id = info.slab_local_cell_id;
                ref.slab_time_begin_id = info.slab_time_begin_id;
                ref.slab_time_end_id = info.slab_time_end_id;
                ref.slab_t_begin = info.t_begin;
                ref.slab_t_end = info.t_end;
                ref.source_t_begin = info.source_t_begin;
                ref.source_t_end = info.source_t_end;
                callback(ref);
            }
        }

        [[nodiscard]] GeometryCacheType make_geometry_cache() const
        {
            return GeometryCacheType(function_ref_());
        }

        [[nodiscard]] double slab_true_coefficient(
            const int slab_id,
            const int slab_true_dof) const
        {
            const auto& coeffs =
                function_ref_().slab_function(slab_id).true_coefficients();
            if (slab_true_dof < 0 || slab_true_dof >= coeffs.size())
            {
                throw std::runtime_error(
                    "CopiedTimeSlabFunctionView::slab_true_coefficient: "
                    "true DoF index out of range.");
            }
            return coeffs[slab_true_dof];
        }

        [[nodiscard]] double local_coefficient(
            const SlabCellViewType& slab_cell,
            const int local_dof) const
        {
            const auto restriction = dof_view_.cell_restriction(slab_cell);
            if (local_dof < 0 || local_dof >= restriction.size())
            {
                throw std::runtime_error(
                    "CopiedTimeSlabFunctionView::local_coefficient: "
                    "local DoF index out of range.");
            }

            double value = 0.0;
            const auto& row =
                restriction.rows[static_cast<std::size_t>(local_dof)];
            for (const auto& entry : row)
            {
                value += entry.weight *
                         slab_true_coefficient(
                             slab_cell.slab_id(),
                             entry.true_dof);
            }
            return value;
        }

        [[nodiscard]] GeometryData geometry(
            const SlabCellViewType& slab_cell) const
        {
            const int copied_cell_id = require_copied_cell_id_(slab_cell);
            return Geometry::make(
                function_ref_()
                    .slab_space()
                    .slab(slab_cell.slab_id())
                    .fespace_ref(),
                copied_cell_id);
        }

        [[nodiscard]] GeometryData geometry(
            const HotSlabCellRef& slab_cell) const
        {
            if (slab_cell.backend != TimeSlabBackend::CopiedMesh ||
                slab_cell.copied_slab_cell_id < 0)
            {
                throw std::runtime_error(
                    "CopiedTimeSlabFunctionView::geometry: expected a "
                    "copied hot slab-cell ref.");
            }
            return Geometry::make(
                function_ref_()
                    .slab_space()
                    .slab(slab_cell.slab_id)
                    .fespace_ref(),
                slab_cell.copied_slab_cell_id);
        }

        template<class SourceGeometryData>
        [[nodiscard]] const GeometryData& geometry(
            GeometryCacheType& geometry_cache,
            const HotSlabCellRef& slab_cell,
            const SourceGeometryData&) const
        {
            return geometry_cache.geometry(slab_cell);
        }

        [[nodiscard]] double value_on_cell(
            const SlabCellViewType& slab_cell,
            const SpaceTimePoint& point) const
        {
            return function_ref_().value_on_cell(
                slab_cell.slab_id(),
                require_copied_cell_id_(slab_cell),
                point);
        }

        [[nodiscard]] double value_on_cell(
            const SlabCellViewType& slab_cell,
            const SpaceTimePoint& point,
            const GeometryData& geom) const
        {
            return function_ref_().value_on_cell(
                slab_cell.slab_id(),
                require_copied_cell_id_(slab_cell),
                point,
                geom);
        }

        [[nodiscard]] double value_on_cell(
            const HotSlabCellRef& slab_cell,
            const SpaceTimePoint& point,
            const GeometryData& geom) const
        {
            return function_ref_().value_on_cell(
                slab_cell.slab_id,
                slab_cell.copied_slab_cell_id,
                point,
                geom);
        }

        [[nodiscard]] GradientType gradient_on_cell(
            const SlabCellViewType& slab_cell,
            const SpaceTimePoint& point) const
        {
            return function_ref_().gradient_on_cell(
                slab_cell.slab_id(),
                require_copied_cell_id_(slab_cell),
                point);
        }

        [[nodiscard]] GradientType gradient_on_cell(
            const SlabCellViewType& slab_cell,
            const SpaceTimePoint& point,
            const GeometryData& geom) const
        {
            return function_ref_().gradient_on_cell(
                slab_cell.slab_id(),
                require_copied_cell_id_(slab_cell),
                point,
                geom);
        }

        [[nodiscard]] GradientType gradient_on_cell(
            const HotSlabCellRef& slab_cell,
            const SpaceTimePoint& point,
            const GeometryData& geom) const
        {
            return function_ref_().gradient_on_cell(
                slab_cell.slab_id,
                slab_cell.copied_slab_cell_id,
                point,
                geom);
        }

        [[nodiscard]] int source_cell_id(
            const SlabCellViewType& slab_cell) const noexcept
        {
            return slab_cell.source_y_cell_id();
        }

        [[nodiscard]] int source_cell_id(
            const HotSlabCellRef& slab_cell) const noexcept
        {
            return slab_cell.source_cell_id;
        }

        [[nodiscard]] const FunctionType& function() const
        {
            return function_ref_();
        }

        [[nodiscard]] const DofViewType& dof_view() const noexcept
        {
            return dof_view_;
        }

    private:
        [[nodiscard]] const FunctionType& function_ref_() const
        {
            if (function_ == nullptr)
            {
                throw std::runtime_error(
                    "CopiedTimeSlabFunctionView: function pointer is null.");
            }
            return *function_;
        }

        [[nodiscard]] static int require_copied_cell_id_(
            const SlabCellViewType& slab_cell)
        {
            if (slab_cell.backend() != TimeSlabBackend::CopiedMesh ||
                !slab_cell.copied_slab_local_cell_id().has_value())
            {
                throw std::runtime_error(
                    "CopiedTimeSlabFunctionView: expected a copied "
                    "SlabCellView.");
            }
            return *slab_cell.copied_slab_local_cell_id();
        }

        const FunctionType* function_ = nullptr;
        DofViewType dof_view_;
    };

    template<class FunctionType>
    [[nodiscard]] inline CopiedTimeSlabFunctionView<FunctionType>
    make_copied_time_slab_function_view(const FunctionType& function)
    {
        return CopiedTimeSlabFunctionView<FunctionType>(function);
    }

    template<TimeSlabFunctionView View, class Map>
    void add_time_slab_function_source_cell_contribution(
        const View& view,
        Map& contributions_by_source_cell,
        const typename View::SlabCellViewType& slab_cell,
        const double contribution)
    {
        contributions_by_source_cell[view.source_cell_id(slab_cell)] +=
            contribution;
    }
}

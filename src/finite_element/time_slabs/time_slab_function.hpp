#pragma once

#include <array>
#include <stdexcept>
#include <vector>

#include "../../linear_algebra/concepts/vector.hpp"
#include "../fespace/functions.hpp"
#include "time_slab_space.hpp"

namespace finite_element::time_slabs
{
    template<class TimeSlabSpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class TimeSlabFunction
    {
    public:
        using SlabSpaceType  = TimeSlabSpaceType;
        using Vector         = VectorType;

        using GT             = typename SlabSpaceType::GT;
        using FETraits       = typename SlabSpaceType::FETraitsType;
        using SlabType       = typename SlabSpaceType::SlabType;
        using LocalSpaceType = typename SlabType::SpaceType;
        using LocalFunction  = finite_element::Function<LocalSpaceType, Vector>;

        using SpaceTimePoint = typename SlabSpaceType::SpaceTimePoint;
        using SpatialPoint   = typename SlabSpaceType::SpatialPoint;
        using TemporalPoint  = typename SlabSpaceType::TemporalPoint;

        static constexpr int dim = GT::dim_v;
        using ValueType          = double;
        using GradientType       = std::array<double, dim>;

        explicit TimeSlabFunction(const SlabSpaceType& slab_space)
            : slab_space_(&slab_space)
        {
            slab_functions_.reserve(static_cast<std::size_t>(slab_space_->n_slabs()));
            for (int k = 0; k < slab_space_->n_slabs(); ++k)
                slab_functions_.emplace_back(slab_space_->slab(k).fespace_ref());
        }

        [[nodiscard]] const SlabSpaceType& slab_space() const noexcept
        {
            return *slab_space_;
        }

        [[nodiscard]] int n_slabs() const noexcept
        {
            return static_cast<int>(slab_functions_.size());
        }

        [[nodiscard]] LocalFunction& slab_function(int k)
        {
            check_slab_index_(k);
            return slab_functions_[static_cast<std::size_t>(k)];
        }

        [[nodiscard]] const LocalFunction& slab_function(int k) const
        {
            check_slab_index_(k);
            return slab_functions_[static_cast<std::size_t>(k)];
        }

        void set_zero()
        {
            for (auto& f : slab_functions_)
                f.set_zero();
        }

        [[nodiscard]] ValueType operator()(const SpaceTimePoint& p) const
        {
            const auto loc = slab_space_->find_active_cell(p);
            if (!loc.is_valid())
                throw std::runtime_error(
                    "TimeSlabFunction::operator(): point not found in any active slab cell.");

            return slab_function(loc.slab_id).value_on_cell(loc.cell_id, p);
        }

        [[nodiscard]] ValueType operator()(
            const SpaceTimePoint& p,
            typename SlabSpaceType::ActiveCellHint& hint) const
        {
            const auto loc = slab_space_->find_active_cell(p, hint);
            if (!loc.is_valid())
                throw std::runtime_error(
                    "TimeSlabFunction::operator()(hinted): point not found in any active slab cell.");

            return slab_function(loc.slab_id).value_on_cell(loc.cell_id, p);
        }

        [[nodiscard]] ValueType value_on_cell(
            int slab_id,
            int cell_id,
            const SpaceTimePoint& p) const
        {
            return slab_function(slab_id).value_on_cell(cell_id, p);
        }

        [[nodiscard]] ValueType value_on_cell(
            int slab_id,
            int cell_id,
            const SpaceTimePoint& p,
            const typename LocalFunction::GeometryData& geom) const
        {
            return slab_function(slab_id).value_on_cell(cell_id, p, geom);
        }

        [[nodiscard]] GradientType gradient_on_cell(
            int slab_id,
            int cell_id,
            const SpaceTimePoint& p) const
        {
            return slab_function(slab_id).gradient_on_cell(cell_id, p);
        }

        [[nodiscard]] GradientType gradient_on_cell(
            int slab_id,
            int cell_id,
            const SpaceTimePoint& p,
            const typename LocalFunction::GeometryData& geom) const
        {
            return slab_function(slab_id).gradient_on_cell(cell_id, p, geom);
        }

        [[nodiscard]] GradientType gradient(
            const SpaceTimePoint& p,
            typename SlabSpaceType::ActiveCellHint& hint) const
        {
            const auto loc = slab_space_->find_active_cell(p, hint);
            if (!loc.is_valid())
                throw std::runtime_error(
                    "TimeSlabFunction::gradient(hinted): point not found in any active slab cell.");

            return slab_function(loc.slab_id).gradient_on_cell(loc.cell_id, p);
        }

    private:
        void check_slab_index_(int k) const
        {
            if (k < 0 || k >= n_slabs())
                throw std::runtime_error(
                    "TimeSlabFunction::slab_function: slab index out of range.");
        }

        const SlabSpaceType* slab_space_ = nullptr;
        std::vector<LocalFunction> slab_functions_{};
    };
}

#pragma once

#include <stdexcept>

#include "../../linear_algebra/concepts/vector.hpp"
#include "patch_scalar_space.hpp"

namespace finite_element::error_fespace
{
    template<class PatchScalarSpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class PatchScalarFunction
    {
    public:
        using SpaceType = PatchScalarSpaceType;
        using Vector    = VectorType;
        using PatchType = typename SpaceType::Patch;

        explicit PatchScalarFunction(const SpaceType& space)
            : space_(&space),
              coefficients_(space.n_dofs())
        {
            set_zero();
        }

        [[nodiscard]] const SpaceType& space() const noexcept
        {
            return *space_;
        }

        [[nodiscard]] const Vector& coefficients() const noexcept
        {
            return coefficients_;
        }

        void set_zero()
        {
            coefficients_.resize(space_->n_dofs());
            coefficients_.set_zero();
        }

        void update_coefficients(const Vector& coefficients)
        {
            if (coefficients.size() != space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarFunction::update_coefficients: size mismatch.");
            }

            coefficients_.resize(coefficients.size());
            for (int i = 0; i < coefficients.size(); ++i)
                coefficients_[i] = coefficients[i];
        }

        [[nodiscard]] double value_on_cell(
            int patch_cell_index,
            double x_ref,
            double t_ref) const
        {
            typename SpaceType::BasisValues basis_values{};
            space_->evaluate_on_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                basis_values);

            double value = 0.0;
            for (int i = 0; i < space_->n_dofs(); ++i)
                value += coefficients_[i] * basis_values[static_cast<std::size_t>(i)];

            return value;
        }

        [[nodiscard]] double value_on_cell(
            int patch_cell_index,
            const typename PatchType::SpaceTimePoint& p) const
        {
            const auto& patch = space_->patch();
            return value_on_cell(
                patch_cell_index,
                patch.spatial_reference(patch_cell_index, p[0]),
                patch.time_reference(p[PatchType::GT::dim_space_v]));
        }

    private:
        const SpaceType* space_ = nullptr;
        Vector coefficients_{};
    };
}

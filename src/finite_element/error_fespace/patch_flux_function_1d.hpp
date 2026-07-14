#pragma once

#include <stdexcept>

#include "../../linear_algebra/concepts/vector.hpp"
#include "patch_flux_space_1d.hpp"

namespace finite_element::error_fespace
{
    template<class PatchFluxSpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class PatchFluxFunction1D
    {
    public:
        struct CellEvaluation
        {
            double value = 0.0;
            double divergence = 0.0;
        };

        using SpaceType = PatchFluxSpaceType;
        using Vector    = VectorType;
        using PatchType = typename SpaceType::Patch;

        explicit PatchFluxFunction1D(const SpaceType& space)
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
                    "PatchFluxFunction1D::update_coefficients: size mismatch.");
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
            return evaluate_on_cell(patch_cell_index, x_ref, t_ref).value;
        }

        [[nodiscard]] double divergence_on_cell(
            int patch_cell_index,
            double x_ref,
            double t_ref) const
        {
            return evaluate_on_cell(patch_cell_index, x_ref, t_ref).divergence;
        }

        [[nodiscard]] CellEvaluation evaluate_on_cell(
            int patch_cell_index,
            double x_ref,
            double t_ref) const
        {
            typename SpaceType::BasisValues basis_values{};
            typename SpaceType::BasisValues basis_divergences{};
            space_->evaluate_on_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                basis_values,
                basis_divergences);

            CellEvaluation evaluation;
            for (int i = 0; i < space_->n_dofs(); ++i)
            {
                const double coefficient = coefficients_[i];
                evaluation.value +=
                    coefficient *
                    basis_values[static_cast<std::size_t>(i)];
                evaluation.divergence +=
                    coefficient *
                    basis_divergences[static_cast<std::size_t>(i)];
            }

            return evaluation;
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

        [[nodiscard]] CellEvaluation evaluate_on_cell(
            int patch_cell_index,
            const typename PatchType::SpaceTimePoint& p) const
        {
            const auto& patch = space_->patch();
            return evaluate_on_cell(
                patch_cell_index,
                patch.spatial_reference(patch_cell_index, p[0]),
                patch.time_reference(p[PatchType::GT::dim_space_v]));
        }

        [[nodiscard]] double divergence_on_cell(
            int patch_cell_index,
            const typename PatchType::SpaceTimePoint& p) const
        {
            const auto& patch = space_->patch();
            return divergence_on_cell(
                patch_cell_index,
                patch.spatial_reference(patch_cell_index, p[0]),
                patch.time_reference(p[PatchType::GT::dim_space_v]));
        }

    private:
        const SpaceType* space_ = nullptr;
        Vector coefficients_{};
    };
}

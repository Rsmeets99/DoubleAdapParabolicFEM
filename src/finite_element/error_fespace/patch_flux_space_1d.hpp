#pragma once

#include <array>

#include "patch_dof_map.hpp"
#include "../time_slabs/time_slab_edge_patch_builder.hpp"

namespace finite_element::error_fespace
{
    template<class PatchType, int PSpace, int PTime>
    requires time_slabs::is_time_slab_edge_patch_v<PatchType>
    class PatchFluxSpace1D
    {
    public:
        using Patch = PatchType;
        using GT    = typename PatchType::GT;

        static_assert(GT::dim_space_v == 1,
                      "PatchFluxSpace1D requires a 1D spatial edge patch.");
        static_assert(GT::dim_time_v == 1,
                      "PatchFluxSpace1D requires dim_time_v == 1.");
        static_assert(PSpace >= 1,
                      "PatchFluxSpace1D requires PSpace >= 1.");

        static constexpr int p_space_v = PSpace;
        static constexpr int p_time_v  = PTime;

        static constexpr int max_spatial_dofs_v = 2 * PSpace + 1;
        static constexpr int n_time_dofs_v      = PTime + 1;
        static constexpr int max_dofs_v         =
            max_spatial_dofs_v * n_time_dofs_v;

        using SpatialValues = std::array<double, max_spatial_dofs_v>;
        using TimeValues    = std::array<double, n_time_dofs_v>;
        using BasisValues   = std::array<double, max_dofs_v>;

        explicit PatchFluxSpace1D(const PatchType& patch)
            : patch_(&patch),
              dof_map_(
                  1 + PSpace * patch.n_cells,
                  n_time_dofs_v)
        {}

        [[nodiscard]] const PatchType& patch() const noexcept
        {
            return *patch_;
        }

        [[nodiscard]] int n_spatial_dofs() const noexcept
        {
            return dof_map_.n_spatial_dofs();
        }

        [[nodiscard]] int n_time_dofs() const noexcept
        {
            return dof_map_.n_time_dofs();
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return dof_map_.n_dofs();
        }

        [[nodiscard]] const PatchDofMap& dof_map() const noexcept
        {
            return dof_map_;
        }

        static void evaluate_time_basis(double t_ref, TimeValues& values)
        {
            values = detail::shifted_legendre_family<PTime>(t_ref);
        }

        void evaluate_reference_spatial_basis_on_patch_cell(
            int patch_cell_index,
            double x_ref,
            SpatialValues& values,
            SpatialValues& derivatives_ref) const
        {
            detail::zero_array_prefix(values);
            detail::zero_array_prefix(derivatives_ref);

            const auto& patch_cell = patch().cell(patch_cell_index);

            values[0] = patch().partition_of_unity_value(patch_cell_index, x_ref);
            derivatives_ref[0] =
                patch().partition_of_unity_dx(patch_cell_index) * patch_cell.length();

            const auto legendre_values =
                detail::shifted_legendre_family<PSpace>(x_ref);
            const auto legendre_derivatives =
                detail::shifted_legendre_derivative_family<PSpace>(x_ref);

            const double bubble = x_ref * (1.0 - x_ref);
            const double bubble_derivative = 1.0 - 2.0 * x_ref;

            const int bubble_offset =
                patch().is_boundary()
                    ? 1
                    : (patch_cell.side == time_slabs::TimeSlabEdgePatchCellSide::left_of_vertex
                        ? 1
                        : 1 + PSpace);

            for (int degree = 0; degree < PSpace; ++degree)
            {
                const int basis_index = bubble_offset + degree;

                const double legendre =
                    legendre_values[static_cast<std::size_t>(degree)];
                const double d_legendre =
                    legendre_derivatives[static_cast<std::size_t>(degree)];

                values[static_cast<std::size_t>(basis_index)] =
                    bubble * legendre;

                derivatives_ref[static_cast<std::size_t>(basis_index)] =
                    bubble_derivative * legendre + bubble * d_legendre;
            }
        }

        void evaluate_on_cell(
            int patch_cell_index,
            double x_ref,
            double t_ref,
            BasisValues& values,
            BasisValues& divergences) const
        {
            detail::zero_array_prefix(values);
            detail::zero_array_prefix(divergences);

            SpatialValues spatial_values{};
            SpatialValues spatial_derivatives_ref{};
            evaluate_reference_spatial_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                spatial_values,
                spatial_derivatives_ref);

            TimeValues time_values{};
            evaluate_time_basis(t_ref, time_values);

            const double inv_h = 1.0 / patch().cell(patch_cell_index).length();

            for (int i_space = 0; i_space < n_spatial_dofs(); ++i_space)
            {
                for (int i_time = 0; i_time < n_time_dofs(); ++i_time)
                {
                    const int dof_id = dof_map_.index(i_space, i_time);
                    const auto idx = static_cast<std::size_t>(dof_id);

                    values[idx] =
                        spatial_values[static_cast<std::size_t>(i_space)] *
                        time_values[static_cast<std::size_t>(i_time)];

                    divergences[idx] =
                        spatial_derivatives_ref[static_cast<std::size_t>(i_space)] * inv_h *
                        time_values[static_cast<std::size_t>(i_time)];
                }
            }
        }

    private:

        const PatchType* patch_ = nullptr;
        PatchDofMap dof_map_{};
    };
}

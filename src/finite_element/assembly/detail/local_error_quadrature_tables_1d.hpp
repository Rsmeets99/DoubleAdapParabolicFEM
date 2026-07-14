#pragma once

#include <array>

#include "quadrature/gauss_legendre_1d.hpp"

namespace finite_element::assembly::detail
{
    template<int QSpace, int QTime, class PatchFluxSpaceType, class PatchScalarSpaceType>
    struct LocalErrorQuadratureTables1D
    {
        static constexpr auto space_rule =
            quadrature::gauss_legendre::gauss_legendre_rule_1d<QSpace>;
        static constexpr auto time_rule =
            quadrature::gauss_legendre::gauss_legendre_rule_1d<QTime>;

        using FluxSpatialValues = typename PatchFluxSpaceType::SpatialValues;
        using FluxTimeValues    = typename PatchFluxSpaceType::TimeValues;
        using ScalarSpatialValues = typename PatchScalarSpaceType::SpatialValues;
        using ScalarTimeValues    = typename PatchScalarSpaceType::TimeValues;

        struct CellData
        {
            std::array<FluxSpatialValues, QSpace> flux_spatial_values{};
            std::array<FluxSpatialValues, QSpace> flux_spatial_derivatives_ref{};
            std::array<ScalarSpatialValues, QSpace> scalar_spatial_values{};
            std::array<double, QSpace> partition_of_unity_values{};
            std::array<double, QSpace> partition_of_unity_dx{};
        };

        std::array<CellData, 2> cells{};
        std::array<FluxTimeValues, QTime> flux_time_values{};
        std::array<ScalarTimeValues, QTime> scalar_time_values{};

        explicit LocalErrorQuadratureTables1D(
            const PatchFluxSpaceType& flux_space,
            const PatchScalarSpaceType& scalar_space)
        {
            for (int qt = 0; qt < QTime; ++qt)
            {
                PatchFluxSpaceType::evaluate_time_basis(
                    time_rule.points[qt][0],
                    flux_time_values[static_cast<std::size_t>(qt)]);
                PatchScalarSpaceType::evaluate_time_basis(
                    time_rule.points[qt][0],
                    scalar_time_values[static_cast<std::size_t>(qt)]);
            }

            for (int patch_cell_index = 0;
                 patch_cell_index < flux_space.patch().n_cells;
                 ++patch_cell_index)
            {
                auto& cell_data = cells[static_cast<std::size_t>(patch_cell_index)];

                for (int qx = 0; qx < QSpace; ++qx)
                {
                    const double x_ref = space_rule.points[qx][0];

                    flux_space.evaluate_reference_spatial_basis_on_patch_cell(
                        patch_cell_index,
                        x_ref,
                        cell_data.flux_spatial_values[static_cast<std::size_t>(qx)],
                        cell_data.flux_spatial_derivatives_ref[static_cast<std::size_t>(qx)]);

                    scalar_space.evaluate_spatial_basis_on_patch_cell(
                        patch_cell_index,
                        x_ref,
                        cell_data.scalar_spatial_values[static_cast<std::size_t>(qx)]);

                    cell_data.partition_of_unity_values[static_cast<std::size_t>(qx)] =
                        flux_space.patch().partition_of_unity_value(patch_cell_index, x_ref);
                    cell_data.partition_of_unity_dx[static_cast<std::size_t>(qx)] =
                        flux_space.patch().partition_of_unity_dx(patch_cell_index);
                }
            }
        }

        void fill_flux_basis_values(
            const PatchFluxSpaceType& flux_space,
            int patch_cell_index,
            int qx,
            int qt,
            typename PatchFluxSpaceType::BasisValues& values) const
        {
            const auto& cell_data = cells[static_cast<std::size_t>(patch_cell_index)];
            const auto& spatial_values =
                cell_data.flux_spatial_values[static_cast<std::size_t>(qx)];
            const auto& time_values =
                flux_time_values[static_cast<std::size_t>(qt)];
            const int n_time_dofs = flux_space.n_time_dofs();

            for (int i_space = 0; i_space < flux_space.n_spatial_dofs(); ++i_space)
            {
                const double spatial_value =
                    spatial_values[static_cast<std::size_t>(i_space)];
                const int offset = i_space * n_time_dofs;

                for (int i_time = 0; i_time < n_time_dofs; ++i_time)
                {
                    values[static_cast<std::size_t>(offset + i_time)] =
                        spatial_value * time_values[static_cast<std::size_t>(i_time)];
                }
            }
        }

        void fill_flux_basis_divergences(
            const PatchFluxSpaceType& flux_space,
            int patch_cell_index,
            int qx,
            int qt,
            typename PatchFluxSpaceType::BasisValues& divergences) const
        {
            const auto& cell_data = cells[static_cast<std::size_t>(patch_cell_index)];
            const auto& spatial_derivatives_ref =
                cell_data.flux_spatial_derivatives_ref[static_cast<std::size_t>(qx)];
            const auto& time_values =
                flux_time_values[static_cast<std::size_t>(qt)];
            const double inv_h = 1.0 / flux_space.patch().cell(patch_cell_index).length();
            const int n_time_dofs = flux_space.n_time_dofs();

            for (int i_space = 0; i_space < flux_space.n_spatial_dofs(); ++i_space)
            {
                const double spatial_divergence =
                    spatial_derivatives_ref[static_cast<std::size_t>(i_space)] * inv_h;
                const int offset = i_space * n_time_dofs;

                for (int i_time = 0; i_time < n_time_dofs; ++i_time)
                {
                    divergences[static_cast<std::size_t>(offset + i_time)] =
                        spatial_divergence * time_values[static_cast<std::size_t>(i_time)];
                }
            }
        }

        void fill_scalar_basis_values(
            const PatchScalarSpaceType& scalar_space,
            int patch_cell_index,
            int qx,
            int qt,
            typename PatchScalarSpaceType::BasisValues& values) const
        {
            const auto& cell_data = cells[static_cast<std::size_t>(patch_cell_index)];
            const auto& spatial_values =
                cell_data.scalar_spatial_values[static_cast<std::size_t>(qx)];
            const auto& time_values =
                scalar_time_values[static_cast<std::size_t>(qt)];
            const int n_time_dofs = scalar_space.n_time_dofs();

            for (int i_space = 0; i_space < scalar_space.n_spatial_dofs(); ++i_space)
            {
                const double spatial_value =
                    spatial_values[static_cast<std::size_t>(i_space)];
                const int offset = i_space * n_time_dofs;

                for (int i_time = 0; i_time < n_time_dofs; ++i_time)
                {
                    values[static_cast<std::size_t>(offset + i_time)] =
                        spatial_value * time_values[static_cast<std::size_t>(i_time)];
                }
            }
        }
    };
}

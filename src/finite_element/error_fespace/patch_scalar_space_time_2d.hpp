#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "patch_dof_map.hpp"
#include "patch_scalar_space_2d.hpp"

namespace finite_element::error_fespace
{
    template<class PatchType, int PSpace, int PTime>
    requires time_slabs::is_time_slab_vertex_patch_v<PatchType>
    class PatchScalarSpaceTime2D
    {
    public:
        using Patch         = PatchType;
        using GT            = typename PatchType::GT;
        using FETraitsType  = typename PatchType::FETraitsType;
        using SlabSpaceType = time_slabs::TimeSlabSpace<GT, FETraitsType>;
        using SpatialSpace  = PatchScalarSpace2D<PatchType, PSpace>;

        using SpatialReferencePoint = typename SpatialSpace::ReferencePoint;
        using SpaceTimeReferencePoint = std::array<double, 3>;
        using TimeValues = std::array<double, PTime + 1>;

        static_assert(GT::dim_space_v == 2,
                      "PatchScalarSpaceTime2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "PatchScalarSpaceTime2D requires dim_time_v == 1.");
        static_assert(PSpace >= 1,
                      "PatchScalarSpaceTime2D requires PSpace >= 1.");
        static_assert(PTime >= 1,
                      "PatchScalarSpaceTime2D requires PTime >= 1.");

        static constexpr int p_space_v = PSpace;
        static constexpr int p_time_v = PTime;
        static constexpr int n_time_dofs_v = PTime + 1;
        static constexpr int spatial_local_dofs_v = SpatialSpace::local_dofs_v;
        static constexpr int local_dofs_v =
            spatial_local_dofs_v * n_time_dofs_v;

        using LocalValues = std::array<double, local_dofs_v>;
        using LocalDofMap = std::array<int, local_dofs_v>;

        PatchScalarSpaceTime2D(
            const PatchType& patch,
            const SlabSpaceType& slab_space)
            : spatial_space_(patch, slab_space)
        {
            build_cell_dof_maps_();
            build_mean_zero_constraint_rows_();
        }

        explicit PatchScalarSpaceTime2D(SpatialSpace spatial_space)
            : spatial_space_(std::move(spatial_space))
        {
            build_cell_dof_maps_();
            build_mean_zero_constraint_rows_();
        }

        [[nodiscard]] const SpatialSpace& spatial_space() const noexcept
        {
            return spatial_space_;
        }

        [[nodiscard]] SpatialSpace& spatial_space() noexcept
        {
            return spatial_space_;
        }

        [[nodiscard]] const PatchType& patch() const noexcept
        {
            return spatial_space_.patch();
        }

        [[nodiscard]] const SlabSpaceType& slab_space() const noexcept
        {
            return spatial_space_.slab_space();
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return spatial_space_.n_patch_cells();
        }

        [[nodiscard]] int n_time_dofs() const noexcept
        {
            return n_time_dofs_v;
        }

        [[nodiscard]] int n_spatial_broken_dofs() const noexcept
        {
            return spatial_space_.n_broken_dofs();
        }

        [[nodiscard]] int n_spatial_effective_dofs() const noexcept
        {
            return spatial_space_.effective_dimension();
        }

        [[nodiscard]] int n_local_dofs_per_cell() const noexcept
        {
            return local_dofs_v;
        }

        [[nodiscard]] int n_broken_dofs() const noexcept
        {
            return spatial_space_.n_broken_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return n_broken_dofs();
        }

        [[nodiscard]] int n_effective_dofs() const noexcept
        {
            return spatial_space_.effective_dimension() * n_time_dofs_v;
        }

        [[nodiscard]] int effective_dimension() const noexcept
        {
            return n_effective_dofs();
        }

        [[nodiscard]] bool has_mean_zero_constraint() const noexcept
        {
            return spatial_space_.has_mean_zero_constraint();
        }

        [[nodiscard]] int n_mean_zero_constraints() const noexcept
        {
            return has_mean_zero_constraint() ? n_time_dofs_v : 0;
        }

        [[nodiscard]] const std::vector<std::vector<double>>&
        mean_zero_constraint_rows() const noexcept
        {
            return mean_zero_constraint_rows_;
        }

        [[nodiscard]] const std::vector<double>& mean_zero_constraint_row(
            int time_dof_id) const
        {
            check_constraint_time_dof_index_(time_dof_id);
            return mean_zero_constraint_rows_[static_cast<std::size_t>(time_dof_id)];
        }

        [[nodiscard]] double mean_zero_constraint_value(
            const std::vector<double>& coefficients,
            int time_dof_id) const
        {
            const auto& row = mean_zero_constraint_row(time_dof_id);
            if (coefficients.size() != static_cast<std::size_t>(n_dofs()))
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D::mean_zero_constraint_value: coefficient size mismatch.");
            }

            double value = 0.0;
            for (int i = 0; i < n_dofs(); ++i)
            {
                value += row[static_cast<std::size_t>(i)] *
                         coefficients[static_cast<std::size_t>(i)];
            }
            return value;
        }

        [[nodiscard]] double time_length() const noexcept
        {
            return patch().time_length();
        }

        [[nodiscard]] double map_time_to_physical(double t_ref) const noexcept
        {
            return patch().t_begin + t_ref * time_length();
        }

        [[nodiscard]] double map_time_to_reference(double t) const noexcept
        {
            return (t - patch().t_begin) / time_length();
        }

        static void evaluate_time_basis(double t_ref, TimeValues& values)
        {
            values = detail::shifted_legendre_family<PTime>(t_ref);
        }

        [[nodiscard]] int patch_dof_index(
            int spatial_patch_dof_id,
            int time_dof_id) const
        {
            check_spatial_patch_dof_index_(spatial_patch_dof_id);
            check_time_dof_index_(time_dof_id);
            return spatial_patch_dof_id * n_time_dofs_v + time_dof_id;
        }

        [[nodiscard]] int spatial_patch_dof_id(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return patch_dof_id / n_time_dofs_v;
        }

        [[nodiscard]] int time_dof_id(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return patch_dof_id % n_time_dofs_v;
        }

        [[nodiscard]] int local_dof_index(
            int spatial_local_dof_id,
            int time_dof_id) const
        {
            check_spatial_local_dof_index_(spatial_local_dof_id);
            check_time_dof_index_(time_dof_id);
            return spatial_local_dof_id * n_time_dofs_v + time_dof_id;
        }

        [[nodiscard]] int spatial_local_dof_id(int local_dof_id) const
        {
            check_local_dof_index_(local_dof_id);
            return local_dof_id / n_time_dofs_v;
        }

        [[nodiscard]] int local_time_dof_id(int local_dof_id) const
        {
            check_local_dof_index_(local_dof_id);
            return local_dof_id % n_time_dofs_v;
        }

        [[nodiscard]] const std::vector<LocalDofMap>& cell_dof_maps() const noexcept
        {
            return cell_dof_maps_;
        }

        [[nodiscard]] const LocalDofMap& cell_dof_map(int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int local_dof_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_local_dof_index_(local_dof_id);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)]
                                 [static_cast<std::size_t>(local_dof_id)];
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int spatial_local_dof_id,
            int time_dof_id) const
        {
            return local_to_patch_dof(
                patch_cell_index,
                local_dof_index(spatial_local_dof_id, time_dof_id));
        }

        void evaluate_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            LocalValues& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.fill(0.0);

            typename SpatialSpace::LocalValues spatial_values{};
            spatial_space_.evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                spatial_values);

            TimeValues time_values{};
            evaluate_time_basis(t_ref, time_values);

            for (int spatial_local_dof = 0;
                 spatial_local_dof < spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                for (int time_dof = 0;
                     time_dof < n_time_dofs_v;
                     ++time_dof)
                {
                    const int local_id =
                        local_dof_index(spatial_local_dof, time_dof);
                    values[static_cast<std::size_t>(local_id)] =
                        spatial_values[static_cast<std::size_t>(spatial_local_dof)] *
                        time_values[static_cast<std::size_t>(time_dof)];
                }
            }
        }

        void evaluate_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpaceTimeReferencePoint& x_ref,
            LocalValues& values) const
        {
            evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                SpatialReferencePoint{x_ref[0], x_ref[1]},
                x_ref[2],
                values);
        }

        void evaluate_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            std::vector<double>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(static_cast<std::size_t>(n_dofs()), 0.0);

            LocalValues local_values{};
            evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);

            for (int local_dof_id = 0; local_dof_id < local_dofs_v; ++local_dof_id)
            {
                const int patch_dof_id =
                    local_to_patch_dof(patch_cell_index, local_dof_id);
                values[static_cast<std::size_t>(patch_dof_id)] =
                    local_values[static_cast<std::size_t>(local_dof_id)];
            }
        }

    private:
        SpatialSpace spatial_space_;
        std::vector<LocalDofMap> cell_dof_maps_{};
        std::vector<std::vector<double>> mean_zero_constraint_rows_{};

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D: patch cell index out of range.");
            }
        }

        static void check_time_dof_index_(int time_dof_id)
        {
            if (time_dof_id < 0 || time_dof_id >= n_time_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D: time DoF index out of range.");
            }
        }

        void check_constraint_time_dof_index_(int time_dof_id) const
        {
            if (!has_mean_zero_constraint())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D: boundary patches have no mean-zero constraints.");
            }

            check_time_dof_index_(time_dof_id);
        }

        static void check_spatial_local_dof_index_(int spatial_local_dof_id)
        {
            if (spatial_local_dof_id < 0 ||
                spatial_local_dof_id >= spatial_local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D: spatial local scalar DoF index out of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D: local scalar DoF index out of range.");
            }
        }

        void check_spatial_patch_dof_index_(int spatial_patch_dof_id) const
        {
            if (spatial_patch_dof_id < 0 ||
                spatial_patch_dof_id >= spatial_space_.n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D: spatial patch DoF index out of range.");
            }
        }

        void check_patch_dof_index_(int patch_dof_id) const
        {
            if (patch_dof_id < 0 || patch_dof_id >= n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2D: patch DoF index out of range.");
            }
        }

        void build_cell_dof_maps_()
        {
            cell_dof_maps_.resize(static_cast<std::size_t>(n_patch_cells()));

            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                auto& map =
                    cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];

                for (int spatial_local_dof = 0;
                     spatial_local_dof < spatial_local_dofs_v;
                     ++spatial_local_dof)
                {
                    const int spatial_patch_dof =
                        spatial_space_.local_to_patch_dof(
                            patch_cell_index,
                            spatial_local_dof);

                    for (int time_dof = 0;
                         time_dof < n_time_dofs_v;
                         ++time_dof)
                    {
                        const int local_id =
                            local_dof_index(spatial_local_dof, time_dof);
                        map[static_cast<std::size_t>(local_id)] =
                            patch_dof_index(spatial_patch_dof, time_dof);
                    }
                }
            }
        }

        void build_mean_zero_constraint_rows_()
        {
            mean_zero_constraint_rows_.clear();
            if (!has_mean_zero_constraint())
                return;

            mean_zero_constraint_rows_.assign(
                static_cast<std::size_t>(n_time_dofs_v),
                std::vector<double>(static_cast<std::size_t>(n_dofs()), 0.0));

            const auto& spatial_row =
                spatial_space_.mean_zero_constraint_row();

            for (int spatial_patch_dof = 0;
                 spatial_patch_dof < spatial_space_.n_dofs();
                 ++spatial_patch_dof)
            {
                for (int time_dof = 0;
                     time_dof < n_time_dofs_v;
                     ++time_dof)
                {
                    mean_zero_constraint_rows_[static_cast<std::size_t>(time_dof)]
                        [static_cast<std::size_t>(
                            patch_dof_index(spatial_patch_dof, time_dof))] =
                        spatial_row[static_cast<std::size_t>(spatial_patch_dof)];
                }
            }
        }
    };
}

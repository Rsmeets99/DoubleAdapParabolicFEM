#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "patch_dof_map.hpp"
#include "patch_rt_flux_space_2d.hpp"

namespace finite_element::error_fespace
{
    template<class PatchType, int PSpace, int PTime>
    requires time_slabs::is_time_slab_vertex_patch_v<PatchType>
    class PatchRTFluxSpaceTime2D
    {
    public:
        using Patch         = PatchType;
        using GT            = typename PatchType::GT;
        using FETraitsType  = typename PatchType::FETraitsType;
        using SlabSpaceType = time_slabs::TimeSlabSpace<GT, FETraitsType>;
        using SpatialSpace  = PatchRTFluxSpace2D<PatchType, PSpace>;

        using TopologyType  = typename SpatialSpace::TopologyType;
        using ReferenceBasis = typename SpatialSpace::ReferenceBasis;
        using PiolaBasis    = typename SpatialSpace::PiolaBasis;

        using SpatialReferencePoint = typename SpatialSpace::ReferencePoint;
        using SpaceTimeReferencePoint = std::array<double, 3>;
        using VectorValue = typename SpatialSpace::VectorValue;
        using TimeValues = std::array<double, PTime + 1>;

        static_assert(GT::dim_space_v == 2,
                      "PatchRTFluxSpaceTime2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "PatchRTFluxSpaceTime2D requires dim_time_v == 1.");
        static_assert(PSpace >= 1,
                      "PatchRTFluxSpaceTime2D requires PSpace >= 1.");
        static_assert(PTime >= 1,
                      "PatchRTFluxSpaceTime2D requires PTime >= 1.");

        static constexpr int p_space_v = PSpace;
        static constexpr int p_time_v = PTime;
        static constexpr int n_time_dofs_v = PTime + 1;
        static constexpr int spatial_local_dofs_v = SpatialSpace::local_dofs_v;
        static constexpr int local_dofs_v =
            spatial_local_dofs_v * n_time_dofs_v;
        static constexpr int spatial_local_edge_dofs_v =
            SpatialSpace::local_edge_dofs_v;
        static constexpr int spatial_local_interior_dofs_v =
            SpatialSpace::local_interior_dofs_v;
        static constexpr int local_edge_dofs_v =
            spatial_local_edge_dofs_v * n_time_dofs_v;
        static constexpr int local_interior_dofs_v =
            spatial_local_interior_dofs_v * n_time_dofs_v;
        static constexpr int edge_moments_per_edge_v =
            SpatialSpace::edge_moments_per_edge_v;
        static constexpr bool uses_dense_nullspace_v =
            SpatialSpace::uses_dense_nullspace_v;

        using LocalValues = std::array<VectorValue, local_dofs_v>;
        using LocalDivergences = std::array<double, local_dofs_v>;

        struct LocalDofMapEntry
        {
            int patch_dof_id = -1;
            int orientation_sign = 1;
            int spatial_patch_dof_id = -1;
            int spatial_local_dof_id = -1;
            int time_dof_id = -1;
            int edge_id = -1;
            int local_face_id = -1;
            bool is_edge_dof = false;
            bool is_constrained = false;
        };

        using LocalDofMap = std::array<LocalDofMapEntry, local_dofs_v>;

        PatchRTFluxSpaceTime2D(
            const PatchType& patch,
            const SlabSpaceType& slab_space)
            : spatial_space_(patch, slab_space)
        {
            build_cell_dof_maps_();
        }

        explicit PatchRTFluxSpaceTime2D(SpatialSpace spatial_space)
            : spatial_space_(std::move(spatial_space))
        {
            build_cell_dof_maps_();
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

        [[nodiscard]] const TopologyType& topology() const noexcept
        {
            return spatial_space_.topology();
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return spatial_space_.n_patch_cells();
        }

        [[nodiscard]] int n_time_dofs() const noexcept
        {
            return n_time_dofs_v;
        }

        [[nodiscard]] int n_spatial_dofs() const noexcept
        {
            return spatial_space_.n_dofs();
        }

        [[nodiscard]] int n_local_dofs_per_cell() const noexcept
        {
            return local_dofs_v;
        }

        [[nodiscard]] int n_local_edge_dofs_per_cell() const noexcept
        {
            return local_edge_dofs_v;
        }

        [[nodiscard]] int n_local_interior_dofs_per_cell() const noexcept
        {
            return local_interior_dofs_v;
        }

        [[nodiscard]] int n_free_patch_edges() const noexcept
        {
            return spatial_space_.n_free_patch_edges();
        }

        [[nodiscard]] int n_constrained_patch_edges() const noexcept
        {
            return spatial_space_.n_constrained_patch_edges();
        }

        [[nodiscard]] int n_free_edge_dofs() const noexcept
        {
            return spatial_space_.n_free_edge_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int n_cell_interior_dofs() const noexcept
        {
            return spatial_space_.n_cell_interior_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return spatial_space_.n_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int effective_dimension() const noexcept
        {
            return n_dofs();
        }

        [[nodiscard]] bool uses_dense_nullspace() const noexcept
        {
            return spatial_space_.uses_dense_nullspace();
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

        [[nodiscard]] int edge_patch_dof(
            int edge_id,
            int edge_moment_id,
            int time_dof_id) const
        {
            check_time_dof_index_(time_dof_id);
            const int spatial_patch_dof =
                spatial_space_.edge_patch_dof(edge_id, edge_moment_id);
            return spatial_patch_dof < 0
                ? -1
                : patch_dof_index(spatial_patch_dof, time_dof_id);
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

        [[nodiscard]] const LocalDofMapEntry& local_dof_entry(
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
            int local_dof_id) const
        {
            return local_dof_entry(patch_cell_index, local_dof_id).patch_dof_id;
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

        [[nodiscard]] int local_orientation_sign(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(
                patch_cell_index,
                local_dof_id).orientation_sign;
        }

        [[nodiscard]] bool local_dof_is_constrained(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(
                patch_cell_index,
                local_dof_id).is_constrained;
        }

        [[nodiscard]] int local_edge_parameter_orientation_sign(
            int patch_cell_index,
            int local_face_id,
            int edge_moment_id) const
        {
            return spatial_space_.local_edge_parameter_orientation_sign(
                patch_cell_index,
                local_face_id,
                edge_moment_id);
        }

        [[nodiscard]] typename PiolaBasis::AffineMap physical_map_for_patch_cell(
            int patch_cell_index) const
        {
            return spatial_space_.physical_map_for_patch_cell(patch_cell_index);
        }

        void evaluate_physical_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            LocalValues& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.fill(VectorValue{0.0, 0.0});

            typename SpatialSpace::LocalValues spatial_values{};
            spatial_space_.evaluate_physical_local_basis_on_patch_cell(
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
                    const auto& spatial_value =
                        spatial_values[static_cast<std::size_t>(spatial_local_dof)];
                    const double time_value =
                        time_values[static_cast<std::size_t>(time_dof)];
                    values[static_cast<std::size_t>(local_id)] =
                        VectorValue{
                            spatial_value[0] * time_value,
                            spatial_value[1] * time_value
                        };
                }
            }
        }

        void evaluate_physical_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpaceTimeReferencePoint& x_ref,
            LocalValues& values) const
        {
            evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                SpatialReferencePoint{x_ref[0], x_ref[1]},
                x_ref[2],
                values);
        }

        void evaluate_physical_local_divergences_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            LocalDivergences& divergences) const
        {
            check_patch_cell_index_(patch_cell_index);
            divergences.fill(0.0);

            typename SpatialSpace::LocalDivergences spatial_divergences{};
            spatial_space_.evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                x_ref,
                spatial_divergences);

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
                    divergences[static_cast<std::size_t>(local_id)] =
                        spatial_divergences[
                            static_cast<std::size_t>(spatial_local_dof)] *
                        time_values[static_cast<std::size_t>(time_dof)];
                }
            }
        }

        void evaluate_physical_local_divergences_on_patch_cell(
            int patch_cell_index,
            const SpaceTimeReferencePoint& x_ref,
            LocalDivergences& divergences) const
        {
            evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                SpatialReferencePoint{x_ref[0], x_ref[1]},
                x_ref[2],
                divergences);
        }

        void evaluate_physical_patch_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            std::vector<VectorValue>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(static_cast<std::size_t>(n_dofs()), VectorValue{0.0, 0.0});

            LocalValues local_values{};
            evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);

            const auto& map = cell_dof_map(patch_cell_index);
            for (int local_dof_id = 0; local_dof_id < local_dofs_v; ++local_dof_id)
            {
                const auto& entry =
                    map[static_cast<std::size_t>(local_dof_id)];
                if (entry.patch_dof_id < 0)
                    continue;

                auto& value =
                    values[static_cast<std::size_t>(entry.patch_dof_id)];
                const auto& local_value =
                    local_values[static_cast<std::size_t>(local_dof_id)];
                value[0] += static_cast<double>(entry.orientation_sign) *
                            local_value[0];
                value[1] += static_cast<double>(entry.orientation_sign) *
                            local_value[1];
            }
        }

    private:
        SpatialSpace spatial_space_;
        std::vector<LocalDofMap> cell_dof_maps_{};

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2D: patch cell index out of range.");
            }
        }

        static void check_time_dof_index_(int time_dof_id)
        {
            if (time_dof_id < 0 || time_dof_id >= n_time_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2D: time DoF index out of range.");
            }
        }

        static void check_spatial_local_dof_index_(int spatial_local_dof_id)
        {
            if (spatial_local_dof_id < 0 ||
                spatial_local_dof_id >= spatial_local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2D: spatial local RT DoF index out of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2D: local RT DoF index out of range.");
            }
        }

        void check_spatial_patch_dof_index_(int spatial_patch_dof_id) const
        {
            if (spatial_patch_dof_id < 0 ||
                spatial_patch_dof_id >= spatial_space_.n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2D: spatial patch DoF index out of range.");
            }
        }

        void check_patch_dof_index_(int patch_dof_id) const
        {
            if (patch_dof_id < 0 || patch_dof_id >= n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2D: patch DoF index out of range.");
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
                const auto& spatial_map =
                    spatial_space_.cell_dof_map(patch_cell_index);

                for (int spatial_local_dof = 0;
                     spatial_local_dof < spatial_local_dofs_v;
                     ++spatial_local_dof)
                {
                    const auto& spatial_entry =
                        spatial_map[static_cast<std::size_t>(spatial_local_dof)];

                    for (int time_dof = 0;
                         time_dof < n_time_dofs_v;
                         ++time_dof)
                    {
                        const int local_id =
                            local_dof_index(spatial_local_dof, time_dof);
                        auto& entry =
                            map[static_cast<std::size_t>(local_id)];

                        entry.spatial_patch_dof_id =
                            spatial_entry.patch_dof_id;
                        entry.patch_dof_id =
                            spatial_entry.patch_dof_id < 0
                                ? -1
                                : patch_dof_index(
                                    spatial_entry.patch_dof_id,
                                    time_dof);
                        entry.orientation_sign =
                            spatial_entry.orientation_sign;
                        entry.spatial_local_dof_id = spatial_local_dof;
                        entry.time_dof_id = time_dof;
                        entry.edge_id = spatial_entry.edge_id;
                        entry.local_face_id = spatial_entry.local_face_id;
                        entry.is_edge_dof = spatial_entry.is_edge_dof;
                        entry.is_constrained = spatial_entry.is_constrained;
                    }
                }
            }
        }
    };
}

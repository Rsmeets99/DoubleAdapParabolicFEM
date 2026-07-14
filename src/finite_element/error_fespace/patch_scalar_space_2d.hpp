#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "../basis/polynomials/dubiner_basis.hpp"
#include "../time_slabs/time_slab_vertex_patch_builder.hpp"
#include "quadrature/reference_quadrature.hpp"

namespace finite_element::error_fespace
{
    template<class PatchType, int PSpace>
    requires time_slabs::is_time_slab_vertex_patch_v<PatchType>
    class PatchScalarSpace2D
    {
    public:
        using Patch         = PatchType;
        using PatchGeometry = PatchType;
        using GT            = typename PatchType::GT;
        using FETraitsType  = typename PatchType::FETraitsType;
        using Types         = mesh::MeshTypes<GT>;
        using SlabSpaceType = time_slabs::TimeSlabSpace<GT, FETraitsType>;
        using MeshType      = mesh::Mesh<GT>;

        using ReferencePoint = std::array<double, 2>;

        static_assert(GT::dim_space_v == 2,
                      "PatchScalarSpace2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "PatchScalarSpace2D requires dim_time_v == 1.");
        static_assert(PSpace >= 1,
                      "PatchScalarSpace2D requires PSpace >= 1.");
        static_assert(PSpace <= 10,
                      "PatchScalarSpace2D currently supports PSpace <= 10 for exact constraint-row quadrature.");

        static constexpr int p_space_v = PSpace;
        static constexpr int local_dofs_v = (PSpace + 1) * (PSpace + 2) / 2;
        static constexpr int constraint_quadrature_degree_v = PSpace;

        using LocalValues = std::array<double, local_dofs_v>;
        using LocalDofMap = std::array<int, local_dofs_v>;
        using Basis       = basis::polynomials::DubinerBasis<PSpace>;

        PatchScalarSpace2D(
            const PatchType& patch,
            const SlabSpaceType& slab_space)
            : patch_(&patch),
              slab_space_(&slab_space)
        {
            validate_patch_slab_();
            build_cell_dof_maps_();
            build_mean_zero_constraint_row_();
        }

        [[nodiscard]] const PatchType& patch() const noexcept
        {
            return *patch_;
        }

        [[nodiscard]] const SlabSpaceType& slab_space() const noexcept
        {
            return *slab_space_;
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return patch_->n_cells;
        }

        [[nodiscard]] int n_local_dofs_per_cell() const noexcept
        {
            return local_dofs_v;
        }

        [[nodiscard]] int n_broken_dofs() const noexcept
        {
            return n_patch_cells() * local_dofs_v;
        }

        [[nodiscard]] int n_effective_dofs() const noexcept
        {
            return n_broken_dofs() - (has_mean_zero_constraint() ? 1 : 0);
        }

        [[nodiscard]] int effective_dimension() const noexcept
        {
            return n_effective_dofs();
        }

        // This space keeps the full broken coefficient vector. Interior-patch
        // mean-zero is represented by mean_zero_constraint_row().
        [[nodiscard]] int n_dofs() const noexcept
        {
            return n_broken_dofs();
        }

        [[nodiscard]] bool has_mean_zero_constraint() const noexcept
        {
            return !patch_->is_boundary();
        }

        [[nodiscard]] const std::vector<double>& mean_zero_constraint_row() const noexcept
        {
            return mean_zero_constraint_row_;
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

        static void evaluate_local_basis(
            const ReferencePoint& x_ref,
            LocalValues& values)
        {
            values = Basis::eval_all(x_ref[0], x_ref[1]);
        }

        void evaluate_local_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            LocalValues& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            evaluate_local_basis(x_ref, values);
        }

        void evaluate_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            std::vector<double>& values) const
        {
            check_patch_cell_index_(patch_cell_index);

            values.assign(static_cast<std::size_t>(n_broken_dofs()), 0.0);

            LocalValues local_values{};
            evaluate_local_basis(x_ref, local_values);

            for (int local_dof_id = 0; local_dof_id < local_dofs_v; ++local_dof_id)
            {
                const int patch_dof_id =
                    local_to_patch_dof(patch_cell_index, local_dof_id);
                values[static_cast<std::size_t>(patch_dof_id)] =
                    local_values[static_cast<std::size_t>(local_dof_id)];
            }
        }

        [[nodiscard]] double spatial_jacobian_measure(int patch_cell_index) const
        {
            const auto& mesh = slab_mesh_();
            const auto& patch_cell = patch_->cell(patch_cell_index);
            const auto& cell = mesh.cell(patch_cell.slab_cell_id);

            const auto& v0 =
                mesh.spatial_vertices()[static_cast<std::size_t>(cell.spatial_vertex_ids[0])];
            const auto& v1 =
                mesh.spatial_vertices()[static_cast<std::size_t>(cell.spatial_vertex_ids[1])];
            const auto& v2 =
                mesh.spatial_vertices()[static_cast<std::size_t>(cell.spatial_vertex_ids[2])];

            const double det =
                (v1[0] - v0[0]) * (v2[1] - v0[1]) -
                (v1[1] - v0[1]) * (v2[0] - v0[0]);
            return std::abs(det);
        }

        [[nodiscard]] double spatial_cell_measure(int patch_cell_index) const
        {
            return 0.5 * spatial_jacobian_measure(patch_cell_index);
        }

        [[nodiscard]] double patch_measure() const
        {
            double measure = 0.0;
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                measure += spatial_cell_measure(patch_cell_index);
            }
            return measure;
        }

        [[nodiscard]] double mean_zero_constraint_value(
            const std::vector<double>& broken_coefficients) const
        {
            if (!has_mean_zero_constraint())
            {
                throw std::runtime_error(
                    "PatchScalarSpace2D::mean_zero_constraint_value: boundary patches have no mean-zero constraint.");
            }

            if (broken_coefficients.size() !=
                static_cast<std::size_t>(n_broken_dofs()))
            {
                throw std::runtime_error(
                    "PatchScalarSpace2D::mean_zero_constraint_value: coefficient size mismatch.");
            }

            double value = 0.0;
            for (int i = 0; i < n_broken_dofs(); ++i)
            {
                value += mean_zero_constraint_row_[static_cast<std::size_t>(i)] *
                         broken_coefficients[static_cast<std::size_t>(i)];
            }
            return value;
        }

    private:
        void validate_patch_slab_() const
        {
            if (patch_->slab_id < 0 || patch_->slab_id >= slab_space_->n_slabs())
            {
                throw std::runtime_error(
                    "PatchScalarSpace2D: patch slab id is out of range for slab space.");
            }
        }

        [[nodiscard]] const MeshType& slab_mesh_() const
        {
            return slab_space_->slab(patch_->slab_id).mesh_ref();
        }

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchScalarSpace2D: patch cell index out of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpace2D: local scalar DoF index out of range.");
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

                for (int local_dof_id = 0;
                     local_dof_id < local_dofs_v;
                     ++local_dof_id)
                {
                    map[static_cast<std::size_t>(local_dof_id)] =
                        patch_cell_index * local_dofs_v + local_dof_id;
                }
            }
        }

        [[nodiscard]] double integrate_local_basis_(
            int patch_cell_index,
            int local_dof_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_local_dof_index_(local_dof_id);

            constexpr auto rule =
                quadrature::reference::make_reference_triangle_quadrature<
                    constraint_quadrature_degree_v>();

            const double jac = spatial_jacobian_measure(patch_cell_index);
            double integral = 0.0;

            for (int q = 0; q < rule.n_points; ++q)
            {
                const auto values =
                    Basis::eval_all(rule.points[static_cast<std::size_t>(q)][0],
                                    rule.points[static_cast<std::size_t>(q)][1]);
                integral +=
                    values[static_cast<std::size_t>(local_dof_id)] *
                    rule.weights[static_cast<std::size_t>(q)] *
                    jac;
            }

            return integral;
        }

        void build_mean_zero_constraint_row_()
        {
            mean_zero_constraint_row_.clear();
            if (!has_mean_zero_constraint())
                return;

            mean_zero_constraint_row_.assign(
                static_cast<std::size_t>(n_broken_dofs()),
                0.0);

            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                for (int local_dof_id = 0;
                     local_dof_id < local_dofs_v;
                     ++local_dof_id)
                {
                    const int patch_dof_id =
                        local_to_patch_dof(patch_cell_index, local_dof_id);
                    mean_zero_constraint_row_[
                        static_cast<std::size_t>(patch_dof_id)] =
                        integrate_local_basis_(patch_cell_index, local_dof_id);
                }
            }
        }

        const PatchType* patch_ = nullptr;
        const SlabSpaceType* slab_space_ = nullptr;

        std::vector<LocalDofMap> cell_dof_maps_{};
        std::vector<double> mean_zero_constraint_row_{};
    };
}

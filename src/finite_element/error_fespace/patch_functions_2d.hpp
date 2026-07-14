#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../basis/polynomials/legendre.hpp"
#include "../../linear_algebra/concepts/vector.hpp"
#include "patch_rt_flux_space_2d.hpp"
#include "patch_rt_flux_space_time_2d.hpp"
#include "patch_scalar_space_2d.hpp"
#include "patch_scalar_space_time_2d.hpp"
#include "quadrature/gauss_legendre_1d.hpp"
#include "quadrature/reference_quadrature.hpp"
#include "quadrature/reference_triangle_duffy.hpp"

namespace finite_element::error_fespace
{
    template<class ScalarSpaceType>
    class PatchScalarFunction2D
    {
    public:
        using SpaceType = ScalarSpaceType;
        using ReferencePoint = typename SpaceType::ReferencePoint;

        explicit PatchScalarFunction2D(const SpaceType& space)
            : space_(&space),
              coefficients_(static_cast<std::size_t>(space.n_dofs()), 0.0)
        {}

        PatchScalarFunction2D(
            const SpaceType& space,
            std::vector<double> coefficients)
            : space_(&space),
              coefficients_(std::move(coefficients))
        {
            validate_coefficient_size_();
        }

        [[nodiscard]] const SpaceType& space() const noexcept
        {
            return *space_;
        }

        [[nodiscard]] const std::vector<double>& coefficients() const noexcept
        {
            return coefficients_;
        }

        [[nodiscard]] std::vector<double>& coefficients() noexcept
        {
            return coefficients_;
        }

        [[nodiscard]] double coefficient(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return coefficients_[static_cast<std::size_t>(patch_dof_id)];
        }

        void set_coefficient(int patch_dof_id, double value)
        {
            check_patch_dof_index_(patch_dof_id);
            coefficients_[static_cast<std::size_t>(patch_dof_id)] = value;
        }

        [[nodiscard]] double local_coefficient(
            int patch_cell_index,
            int local_dof_id) const
        {
            const int patch_dof_id =
                space_->local_to_patch_dof(patch_cell_index, local_dof_id);
            check_patch_dof_index_(patch_dof_id);
            return coefficients_[static_cast<std::size_t>(patch_dof_id)];
        }

        [[nodiscard]] double value_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref) const
        {
            typename SpaceType::LocalValues local_values{};
            space_->evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                local_values);

            double value = 0.0;
            for (int local_dof_id = 0;
                 local_dof_id < SpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                value += local_coefficient(patch_cell_index, local_dof_id) *
                         local_values[static_cast<std::size_t>(local_dof_id)];
            }

            return value;
        }

    private:
        const SpaceType* space_ = nullptr;
        std::vector<double> coefficients_{};

        void validate_coefficient_size_() const
        {
            if (coefficients_.size() !=
                static_cast<std::size_t>(space_->n_dofs()))
            {
                throw std::runtime_error(
                    "PatchScalarFunction2D: coefficient size does not match scalar patch space.");
            }
        }

        void check_patch_dof_index_(int patch_dof_id) const
        {
            if (patch_dof_id < 0 || patch_dof_id >= space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarFunction2D: patch DoF id out of range.");
            }
        }
    };

    template<class RTFluxSpaceType>
    class PatchRTFluxFunction2D
    {
    public:
        using SpaceType = RTFluxSpaceType;
        using TopologyType = typename SpaceType::TopologyType;
        using IncidentCell = typename TopologyType::IncidentCell;
        using EdgeType = typename TopologyType::Edge;
        using PiolaBasis = typename SpaceType::PiolaBasis;
        using ReferencePoint = typename SpaceType::ReferencePoint;
        using VectorValue = typename SpaceType::VectorValue;

        static constexpr int p_space_v = SpaceType::p_space_v;
        static constexpr int edge_quadrature_order_v = p_space_v + 1;
        static_assert(edge_quadrature_order_v <= 12,
                      "PatchRTFluxFunction2D edge moments require p_space <= 11.");

        explicit PatchRTFluxFunction2D(const SpaceType& space)
            : space_(&space),
              coefficients_(static_cast<std::size_t>(space.n_dofs()), 0.0)
        {}

        PatchRTFluxFunction2D(
            const SpaceType& space,
            std::vector<double> coefficients)
            : space_(&space),
              coefficients_(std::move(coefficients))
        {
            validate_coefficient_size_();
        }

        [[nodiscard]] const SpaceType& space() const noexcept
        {
            return *space_;
        }

        [[nodiscard]] const std::vector<double>& coefficients() const noexcept
        {
            return coefficients_;
        }

        [[nodiscard]] std::vector<double>& coefficients() noexcept
        {
            return coefficients_;
        }

        [[nodiscard]] double coefficient(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return coefficients_[static_cast<std::size_t>(patch_dof_id)];
        }

        void set_coefficient(int patch_dof_id, double value)
        {
            check_patch_dof_index_(patch_dof_id);
            coefficients_[static_cast<std::size_t>(patch_dof_id)] = value;
        }

        [[nodiscard]] double local_coefficient(
            int patch_cell_index,
            int local_dof_id) const
        {
            const auto& entry =
                space_->local_dof_entry(patch_cell_index, local_dof_id);
            if (entry.patch_dof_id < 0)
                return 0.0;

            check_patch_dof_index_(entry.patch_dof_id);
            return static_cast<double>(entry.orientation_sign) *
                   coefficients_[static_cast<std::size_t>(entry.patch_dof_id)];
        }

        [[nodiscard]] VectorValue value_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref) const
        {
            typename SpaceType::LocalValues local_values{};
            space_->evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                local_values);

            VectorValue value{0.0, 0.0};
            for (int local_dof_id = 0;
                 local_dof_id < SpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const double c =
                    local_coefficient(patch_cell_index, local_dof_id);
                const auto& phi =
                    local_values[static_cast<std::size_t>(local_dof_id)];
                value[0] += c * phi[0];
                value[1] += c * phi[1];
            }

            return value;
        }

        [[nodiscard]] double divergence_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref) const
        {
            typename SpaceType::LocalDivergences local_divergences{};
            space_->evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                x_ref,
                local_divergences);

            double divergence = 0.0;
            for (int local_dof_id = 0;
                 local_dof_id < SpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                divergence +=
                    local_coefficient(patch_cell_index, local_dof_id) *
                    local_divergences[
                        static_cast<std::size_t>(local_dof_id)];
            }

            return divergence;
        }

        [[nodiscard]] double edge_normal_moment_on_patch_cell_face(
            int patch_cell_index,
            int local_face_id,
            int edge_moment_id) const
        {
            if (edge_moment_id < 0 || edge_moment_id > p_space_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxFunction2D: edge moment index out of range.");
            }

            const auto* edge =
                space_->topology().edge_for_patch_cell_face(
                    patch_cell_index,
                    local_face_id);
            if (edge == nullptr)
            {
                throw std::runtime_error(
                    "PatchRTFluxFunction2D: patch cell face has no topology edge.");
            }

            const auto* incident = incident_for_patch_cell_face_(
                *edge,
                patch_cell_index,
                local_face_id);
            if (incident == nullptr)
            {
                throw std::runtime_error(
                    "PatchRTFluxFunction2D: patch cell face has no incident entry.");
            }

            const VectorValue outward_normal{
                static_cast<double>(incident->orientation_sign) *
                    edge->canonical_normal[0],
                static_cast<double>(incident->orientation_sign) *
                    edge->canonical_normal[1]
            };

            constexpr auto rule =
                quadrature::gauss_legendre::
                    gauss_legendre_rule_1d<edge_quadrature_order_v>;

            double result = 0.0;
            for (int q = 0; q < rule.size(); ++q)
            {
                const double s = rule.point(q)[0];
                const auto x_ref = local_face_reference_point_(
                    local_face_id,
                    s);
                const auto value = value_on_patch_cell(
                    patch_cell_index,
                    x_ref);
                const double mu =
                    basis::polynomials::LegendrePolynomials<p_space_v>::eval(
                        edge_moment_id,
                        2.0 * s - 1.0) *
                    static_cast<double>(
                        space_->local_edge_parameter_orientation_sign(
                            patch_cell_index,
                            local_face_id,
                            edge_moment_id));

                result += rule.weight(q) *
                          (value[0] * outward_normal[0] +
                           value[1] * outward_normal[1]) *
                          mu *
                          edge->length;
            }

            return result;
        }

        [[nodiscard]] double divergence_integral_on_patch_cell(
            int patch_cell_index) const
        {
            const auto map = space_->physical_map_for_patch_cell(patch_cell_index);
            const double jac =
                PiolaBasis::jacobian_measure(map);

            double result = 0.0;
            result +=
                quadrature::reference::integrate_reference_triangle_duffy<
                    p_space_v>(
                    [&](const double x, const double y)
                    {
                        return divergence_on_patch_cell(
                            patch_cell_index,
                            ReferencePoint{x, y}) *
                            jac;
                    });

            return result;
        }

        [[nodiscard]] double divergence_integral() const
        {
            double result = 0.0;
            for (int patch_cell_index = 0;
                 patch_cell_index < space_->n_patch_cells();
                 ++patch_cell_index)
            {
                result += divergence_integral_on_patch_cell(patch_cell_index);
            }
            return result;
        }

    private:
        const SpaceType* space_ = nullptr;
        std::vector<double> coefficients_{};

        void validate_coefficient_size_() const
        {
            if (coefficients_.size() !=
                static_cast<std::size_t>(space_->n_dofs()))
            {
                throw std::runtime_error(
                    "PatchRTFluxFunction2D: coefficient size does not match RT patch space.");
            }
        }

        void check_patch_dof_index_(int patch_dof_id) const
        {
            if (patch_dof_id < 0 || patch_dof_id >= space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxFunction2D: patch DoF id out of range.");
            }
        }

        [[nodiscard]] static ReferencePoint local_face_reference_point_(
            int local_face_id,
            double s)
        {
            switch (local_face_id)
            {
                case 0:
                    return ReferencePoint{s, 0.0};
                case 1:
                    return ReferencePoint{1.0 - s, s};
                case 2:
                    return ReferencePoint{0.0, 1.0 - s};
                default:
                    throw std::runtime_error(
                        "PatchRTFluxFunction2D: local face id out of range.");
            }
        }

        [[nodiscard]] static const IncidentCell*
        incident_for_patch_cell_face_(
            const EdgeType& edge,
            int patch_cell_index,
            int local_face_id)
        {
            for (const auto& incident : edge.incident_cells)
            {
                if (incident.patch_cell_index == patch_cell_index &&
                    incident.local_face_id == local_face_id)
                {
                    return &incident;
                }
            }

            return nullptr;
        }
    };

    template<class ScalarSpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class PatchScalarFunctionTime2D
    {
    public:
        using SpaceType = ScalarSpaceType;
        using Vector    = VectorType;
        using PatchType = typename SpaceType::Patch;
        using SpatialReferencePoint = typename SpaceType::SpatialReferencePoint;
        using SpaceTimePoint = typename PatchType::Types::SpaceTimePoint;

        explicit PatchScalarFunctionTime2D(const SpaceType& space)
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
                    "PatchScalarFunctionTime2D::update_coefficients: size mismatch.");
            }

            coefficients_.resize(coefficients.size());
            for (int i = 0; i < coefficients.size(); ++i)
                coefficients_[i] = coefficients[i];
        }

        [[nodiscard]] double local_coefficient(
            int patch_cell_index,
            int local_dof_id) const
        {
            const int patch_dof_id =
                space_->local_to_patch_dof(patch_cell_index, local_dof_id);
            if (patch_dof_id < 0 || patch_dof_id >= space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarFunctionTime2D::local_coefficient: patch DoF id out of range.");
            }

            return coefficients_[patch_dof_id];
        }

        [[nodiscard]] double value_on_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref) const
        {
            typename SpaceType::LocalValues local_values{};
            space_->evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);

            double value = 0.0;
            for (int local_dof_id = 0;
                 local_dof_id < SpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                value +=
                    local_coefficient(patch_cell_index, local_dof_id) *
                    local_values[static_cast<std::size_t>(local_dof_id)];
            }

            return value;
        }

        [[nodiscard]] double value_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            const auto x_ref =
                physical_to_reference_(patch_cell_index, p);
            const double t_ref = space_->map_time_to_reference(p[2]);

            return value_on_cell(patch_cell_index, x_ref, t_ref);
        }

    private:
        const SpaceType* space_ = nullptr;
        Vector coefficients_{};

        [[nodiscard]] SpatialReferencePoint physical_to_reference_(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            const auto& patch_cell =
                space_->patch().cell(patch_cell_index);
            const auto& slab_mesh =
                space_->slab_space().slab(space_->patch().slab_id).mesh_ref();
            const auto& cell = slab_mesh.cell(patch_cell.slab_cell_id);

            const auto& v0 =
                slab_mesh.spatial_vertices()[static_cast<std::size_t>(
                    cell.spatial_vertex_ids[0])];
            const auto& v1 =
                slab_mesh.spatial_vertices()[static_cast<std::size_t>(
                    cell.spatial_vertex_ids[1])];
            const auto& v2 =
                slab_mesh.spatial_vertices()[static_cast<std::size_t>(
                    cell.spatial_vertex_ids[2])];

            const double J00 = v1[0] - v0[0];
            const double J01 = v2[0] - v0[0];
            const double J10 = v1[1] - v0[1];
            const double J11 = v2[1] - v0[1];
            const double det = J00 * J11 - J01 * J10;
            if (std::abs(det) <= 1.0e-15)
            {
                throw std::runtime_error(
                    "PatchScalarFunctionTime2D: degenerate patch cell.");
            }

            const double dx = p[0] - v0[0];
            const double dy = p[1] - v0[1];
            return SpatialReferencePoint{
                ( J11 * dx - J01 * dy) / det,
                (-J10 * dx + J00 * dy) / det
            };
        }
    };

    template<class RTFluxSpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class PatchRTFluxFunctionTime2D
    {
    public:
        struct CellEvaluation
        {
            typename RTFluxSpaceType::VectorValue value{0.0, 0.0};
            double divergence = 0.0;
        };

        using SpaceType = RTFluxSpaceType;
        using Vector    = VectorType;
        using PatchType = typename SpaceType::Patch;
        using SpatialReferencePoint = typename SpaceType::SpatialReferencePoint;
        using SpaceTimePoint = typename PatchType::Types::SpaceTimePoint;
        using VectorValue = typename SpaceType::VectorValue;

        explicit PatchRTFluxFunctionTime2D(const SpaceType& space)
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
                    "PatchRTFluxFunctionTime2D::update_coefficients: size mismatch.");
            }

            coefficients_.resize(coefficients.size());
            for (int i = 0; i < coefficients.size(); ++i)
                coefficients_[i] = coefficients[i];
        }

        [[nodiscard]] double local_coefficient(
            int patch_cell_index,
            int local_dof_id) const
        {
            const auto& entry =
                space_->local_dof_entry(patch_cell_index, local_dof_id);
            if (entry.patch_dof_id < 0)
                return 0.0;

            if (entry.patch_dof_id >= space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxFunctionTime2D::local_coefficient: patch DoF id out of range.");
            }

            return static_cast<double>(entry.orientation_sign) *
                   coefficients_[entry.patch_dof_id];
        }

        [[nodiscard]] CellEvaluation evaluate_on_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref) const
        {
            typename SpaceType::LocalValues local_values{};
            typename SpaceType::LocalDivergences local_divergences{};
            space_->evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);
            space_->evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_divergences);

            CellEvaluation evaluation;
            for (int local_dof_id = 0;
                 local_dof_id < SpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const double c =
                    local_coefficient(patch_cell_index, local_dof_id);
                const auto& phi =
                    local_values[static_cast<std::size_t>(local_dof_id)];
                evaluation.value[0] += c * phi[0];
                evaluation.value[1] += c * phi[1];
                evaluation.divergence +=
                    c * local_divergences[
                        static_cast<std::size_t>(local_dof_id)];
            }

            return evaluation;
        }

        [[nodiscard]] CellEvaluation evaluate_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            const auto map =
                space_->physical_map_for_patch_cell(patch_cell_index);
            const auto x_ref =
                SpaceType::PiolaBasis::physical_to_reference(
                    map,
                    typename SpaceType::SpatialReferencePoint{p[0], p[1]});
            const double t_ref = space_->map_time_to_reference(p[2]);

            return evaluate_on_cell(patch_cell_index, x_ref, t_ref);
        }

        [[nodiscard]] VectorValue value_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            return evaluate_on_cell(patch_cell_index, p).value;
        }

        [[nodiscard]] double divergence_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            return evaluate_on_cell(patch_cell_index, p).divergence;
        }

    private:
        const SpaceType* space_ = nullptr;
        Vector coefficients_{};
    };
}

#pragma once

#include <array>
#include <vector>

#include "finite_element/coefficients/diffusion_coefficient.hpp"
#include "finite_element/fespace/fe_traits.hpp"
#include "finite_element/fespace/fespace.hpp"
#include "finite_element/fespace/policies.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_traits.hpp"
#include "mesh/refinement/refinement_type.hpp"

namespace adaptive_algorithm::support::space_time_2d
{
    using GeomTraits = mesh::MeshTraits<2>;
    using MeshType = mesh::Mesh<GeomTraits>;
    using MeshTypes = mesh::MeshTypes<GeomTraits>;
    using SpatialPoint = typename MeshTypes::SpatialPoint;
    using TemporalPoint = typename MeshTypes::TemporalPoint;
    using SpaceTimePoint = typename MeshTypes::SpaceTimePoint;
    using SpatialSimplex = typename MeshTypes::SpatialSimplexPoints;
    using DiffusionTensor = finite_element::coefficients::DiffusionTensor<2>;
    using RefinementFETraits =
        finite_element::FiniteElementTraits<GeomTraits, 1, 1>;
    using RefinementSpace = finite_element::FESpace<
        GeomTraits,
        RefinementFETraits,
        finite_element::SpaceOnlyPolicy>;

    [[nodiscard]] inline DiffusionTensor identity_diffusion_tensor() noexcept
    {
        return DiffusionTensor{
            std::array<double, 2>{1.0, 0.0},
            std::array<double, 2>{0.0, 1.0}
        };
    }

    [[nodiscard]] inline DiffusionTensor anisotropic_diffusion_tensor() noexcept
    {
        return DiffusionTensor{
            std::array<double, 2>{2.0, 0.5},
            std::array<double, 2>{0.5, 3.0}
        };
    }

    [[nodiscard]] inline DiffusionTensor variable_diffusion_tensor(
        const SpaceTimePoint& p) noexcept
    {
        // On [0,1]^2 x [0,1], Gershgorin gives lambda_min >= 1.85 and
        // lambda_max <= 3.50, so this is a uniformly elliptic positive
        // definite tensor.
        const double x = p[0];
        const double y = p[1];
        const double t = p[2];
        const double coupling = 0.1 + 0.05 * x * y;

        return DiffusionTensor{
            std::array<double, 2>{2.0 + 0.2 * x + 0.1 * t, coupling},
            std::array<double, 2>{coupling, 3.0 + 0.3 * y + 0.05 * t}
        };
    }

    struct IdentityTensorM
    {
        [[nodiscard]] DiffusionTensor operator()(
            const SpaceTimePoint&) const noexcept
        {
            return identity_diffusion_tensor();
        }
    };

    struct AnisotropicTensorM
    {
        [[nodiscard]] DiffusionTensor operator()(
            const SpaceTimePoint&) const noexcept
        {
            return anisotropic_diffusion_tensor();
        }
    };

    struct VariableTensorM
    {
        [[nodiscard]] DiffusionTensor operator()(
            const SpaceTimePoint& p) const noexcept
        {
            return variable_diffusion_tensor(p);
        }
    };

    [[nodiscard]] inline MeshType make_unit_square_two_triangle_mesh()
    {
        const std::vector<SpatialSimplex> roots{
            SpatialSimplex{
                SpatialPoint{0.0, 0.0},
                SpatialPoint{1.0, 0.0},
                SpatialPoint{0.0, 1.0}
            },
            SpatialSimplex{
                SpatialPoint{1.0, 0.0},
                SpatialPoint{1.0, 1.0},
                SpatialPoint{0.0, 1.0}
            }
        };

        MeshType mesh;
        mesh.initialize_mesh(
            roots,
            TemporalPoint{0.0},
            TemporalPoint{1.0});
        mesh.cell(0).spatial_refinement_edge_local = {1, 2};
        mesh.cell(1).spatial_refinement_edge_local = {2, 0};
        return mesh;
    }

    inline void use_shared_diagonal_as_refinement_edge(MeshType& mesh)
    {
        mesh.cell(0).spatial_refinement_edge_local = {1, 2};
        mesh.cell(1).spatial_refinement_edge_local = {2, 0};
    }

    [[nodiscard]] inline MeshType make_unit_square_spatially_refined_mesh()
    {
        auto mesh = make_unit_square_two_triangle_mesh();
        use_shared_diagonal_as_refinement_edge(mesh);
        RefinementSpace space(mesh);
        space.initialize(mesh.leaf_cell_ids());
        space.refine({0}, mesh::RefinementType::spatial);
        return mesh;
    }

    [[nodiscard]] inline MeshType make_unit_square_temporally_refined_first_cell_mesh()
    {
        auto mesh = make_unit_square_two_triangle_mesh();
        RefinementSpace space(mesh);
        space.initialize(mesh.leaf_cell_ids());
        space.refine({0}, mesh::RefinementType::temporal);
        return mesh;
    }
}

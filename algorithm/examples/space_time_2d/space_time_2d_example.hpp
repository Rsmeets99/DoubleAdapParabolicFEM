#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "../../problem_data.hpp"
#include "../../support/space_time_1d.hpp"
#include "../../support/space_time_2d.hpp"

#include "mesh/mesh_traits.hpp"
#include "mesh/types.hpp"

namespace adaptive_algorithm::examples
{
    constexpr int dim_space_2d = 2;

    using SpaceTime2DGeomTraits = mesh::MeshTraits<dim_space_2d>;
    using SpaceTime2DMeshTypes = mesh::MeshTypes<SpaceTime2DGeomTraits>;
    using SpaceTime2DPoint = typename SpaceTime2DMeshTypes::SpaceTimePoint;
    using SpaceTime2DSpatialPoint = typename SpaceTime2DMeshTypes::SpatialPoint;
    using SpaceTime2DSpatialGradient =
        std::array<double, SpaceTime2DGeomTraits::dim_space_v>;
    using SpaceTime2DDiffusionTensor =
        finite_element::coefficients::DiffusionTensor<
            SpaceTime2DGeomTraits::dim_space_v>;

    struct SpaceTime2DExactData
    {
        std::function<double(const SpaceTime2DPoint&)> u{};
        std::function<double(const SpaceTime2DPoint&)> lambda{};
        std::function<double(const SpaceTime2DPoint&)> theta{};
        std::function<SpaceTime2DSpatialGradient(const SpaceTime2DPoint&)>
            grad_u_x{};
        std::function<SpaceTime2DSpatialGradient(const SpaceTime2DPoint&)>
            grad_lambda_x{};
        std::function<SpaceTime2DSpatialGradient(const SpaceTime2DPoint&)>
            grad_theta_x{};

        [[nodiscard]] bool has_any_data() const noexcept
        {
            return static_cast<bool>(u) ||
                static_cast<bool>(lambda) ||
                static_cast<bool>(theta) ||
                static_cast<bool>(grad_u_x) ||
                static_cast<bool>(grad_lambda_x) ||
                static_cast<bool>(grad_theta_x);
        }
    };

    enum class SpaceTime2DInitialMeshKind
    {
        unit_square_two_triangles,
        unit_square_spatially_refined,
        unit_square_temporally_refined_first_cell
    };

    struct SpaceTime2DMeshSetup
    {
        SpaceTime2DInitialMeshKind root_mesh =
            SpaceTime2DInitialMeshKind::unit_square_two_triangles;

        int x_uniform_levels = 0;

        std::vector<SpaceTime2DPoint> x_refinement_points{};
        int x_refinement_rounds = 0;

        std::vector<SpaceTime2DPoint> y_refinement_points{};
        int y_refinement_rounds = 0;
    };

    using SpaceTime2DDiffusionFunctionM =
        std::function<SpaceTime2DDiffusionTensor(const SpaceTime2DPoint&)>;
    using SpaceTime2DLoadFunctionEll =
        std::function<double(const SpaceTime2DPoint&)>;
    using SpaceTime2DInitialValueFunctionU0 =
        std::function<double(const SpaceTime2DSpatialPoint&)>;

    using SpaceTime2DProblemData =
        ProblemData<
            SpaceTime2DDiffusionFunctionM,
            SpaceTime2DLoadFunctionEll,
            SpaceTime2DInitialValueFunctionU0,
            SpaceTime2DExactData,
            std::monostate>;

    [[nodiscard]] inline SpaceTime2DProblemData make_space_time_2d_problem_data(
        SpaceTime2DDiffusionFunctionM M,
        SpaceTime2DLoadFunctionEll ell,
        SpaceTime2DInitialValueFunctionU0 u0,
        std::string name = {},
        std::string description = {},
        std::optional<SpaceTime2DExactData> exact = std::nullopt)
    {
        SpaceTime2DProblemData problem;
        problem.name = std::move(name);
        problem.description = std::move(description);
        problem.M = std::move(M);
        problem.ell = std::move(ell);
        problem.u0 = std::move(u0);
        problem.exact = std::move(exact);
        problem.auxiliary = std::nullopt;
        return problem;
    }

    struct SpaceTime2DExample
    {
        SpaceTime2DProblemData problem{};
        SpaceTime2DMeshSetup initial_mesh{};
        std::string label{};
        std::string output_directory_name{};

        [[nodiscard]] std::string effective_label() const
        {
            if (!label.empty())
                return label;

            return problem.name;
        }

        [[nodiscard]] std::string effective_output_directory_name() const
        {
            if (!output_directory_name.empty())
                return output_directory_name;

            return problem.name;
        }
    };

    [[nodiscard]] inline adaptive_algorithm::support::space_time_2d::MeshType
    make_initial_mesh_from_example(const SpaceTime2DExample& example)
    {
        using MeshKind = SpaceTime2DInitialMeshKind;

        switch (example.initial_mesh.root_mesh)
        {
        case MeshKind::unit_square_two_triangles:
            return adaptive_algorithm::support::space_time_2d::
                make_unit_square_two_triangle_mesh();
        case MeshKind::unit_square_spatially_refined:
            return adaptive_algorithm::support::space_time_2d::
                make_unit_square_spatially_refined_mesh();
        case MeshKind::unit_square_temporally_refined_first_cell:
            return adaptive_algorithm::support::space_time_2d::
                make_unit_square_temporally_refined_first_cell_mesh();
        }

        return adaptive_algorithm::support::space_time_2d::
            make_unit_square_two_triangle_mesh();
    }

    template<class XSpaceType, class YSpaceType>
    inline void initialize_spaces_from_example(
        XSpaceType& x_space,
        YSpaceType& y_space,
        const SpaceTime2DExample& example)
    {
        const auto& mesh = x_space.mesh_ref();

        x_space.initialize(mesh.leaf_cell_ids());
        support::refine_uniform_levels(
            x_space,
            example.initial_mesh.x_uniform_levels);
        support::refine_points_n_times(
            x_space,
            example.initial_mesh.x_refinement_points,
            example.initial_mesh.x_refinement_rounds,
            example.problem.name + ".x");

        y_space.initialize(x_space.active_cells());
        support::refine_points_n_times(
            y_space,
            example.initial_mesh.y_refinement_points,
            example.initial_mesh.y_refinement_rounds,
            example.problem.name + ".y");
    }
}

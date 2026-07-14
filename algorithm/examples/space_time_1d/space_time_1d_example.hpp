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

#include "mesh/mesh_traits.hpp"
#include "mesh/types.hpp"

namespace adaptive_algorithm::examples
{
    constexpr int dim_space = 1;

    using GeomTraits = mesh::MeshTraits<dim_space>;
    using MeshTypes = mesh::MeshTypes<GeomTraits>;
    using SpaceTimePoint = typename MeshTypes::SpaceTimePoint;
    using SpatialGradient = std::array<double, GeomTraits::dim_space_v>;

    struct SpaceTime1DExactData
    {
        std::function<double(const SpaceTimePoint&)> u{};
        std::function<double(const SpaceTimePoint&)> lambda{};
        std::function<double(const SpaceTimePoint&)> theta{};
        std::function<SpatialGradient(const SpaceTimePoint&)> grad_u_x{};
        std::function<SpatialGradient(const SpaceTimePoint&)> grad_lambda_x{};
        std::function<SpatialGradient(const SpaceTimePoint&)> grad_theta_x{};

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

    struct SpaceTime1DMeshSetup
    {
        int x_uniform_levels = 1;

        std::vector<SpaceTimePoint> x_refinement_points{};
        int x_refinement_rounds = 0;

        std::vector<SpaceTimePoint> y_refinement_points{};
        int y_refinement_rounds = 0;
    };

    using DiffusionFunctionM = std::function<double(const SpaceTimePoint&)>;
    using LoadFunctionEll = std::function<double(const SpaceTimePoint&)>;
    using InitialValueFunctionU0 = std::function<double(double)>;

    using SpaceTime1DProblemData =
        ProblemData<
            DiffusionFunctionM,
            LoadFunctionEll,
            InitialValueFunctionU0,
            SpaceTime1DExactData,
            std::monostate>;

    [[nodiscard]] inline SpaceTime1DProblemData make_space_time_1d_problem_data(
        DiffusionFunctionM M,
        LoadFunctionEll ell,
        InitialValueFunctionU0 u0,
        std::string name = {},
        std::string description = {},
        std::optional<SpaceTime1DExactData> exact = std::nullopt)
    {
        SpaceTime1DProblemData problem;
        problem.name = std::move(name);
        problem.description = std::move(description);
        problem.M = std::move(M);
        problem.ell = std::move(ell);
        problem.u0 = std::move(u0);
        problem.exact = std::move(exact);
        problem.auxiliary = std::nullopt;
        return problem;
    }

    struct SpaceTime1DExample
    {
        SpaceTime1DProblemData problem{};
        SpaceTime1DMeshSetup initial_mesh{};
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

    template<class XSpaceType, class YSpaceType>
    inline void initialize_spaces_from_example(
        XSpaceType& x_space,
        YSpaceType& y_space,
        const SpaceTime1DExample& example)
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

#pragma once

#include <cmath>

#include "space_time_1d_example.hpp"

namespace adaptive_algorithm::examples
{
    [[nodiscard]] inline SpaceTime1DExample
    smooth_initial_1d_example()
    {
        SpaceTime1DExample example;

        SpaceTime1DExactData exact;
        exact.u =
            [](const SpaceTimePoint& p) noexcept
            {
                constexpr double pi =
                    3.141592653589793238462643383279502884;
                return std::exp(-pi * pi * p[1]) *
                    std::sin(pi * p[0]);
            };
        exact.theta = exact.u;
        exact.grad_u_x =
            [](const SpaceTimePoint& p) noexcept -> SpatialGradient
            {
                constexpr double pi =
                    3.141592653589793238462643383279502884;
                return {
                    pi * std::exp(-pi * pi * p[1]) *
                        std::cos(pi * p[0])
                };
            };
        exact.grad_theta_x = exact.grad_u_x;

        example.problem = make_space_time_1d_problem_data(
            DiffusionFunctionM{
                [](const SpaceTimePoint&) noexcept
                {
                    return 1.0;
                }},
            LoadFunctionEll{
                [](const SpaceTimePoint&) noexcept
                {
                    return 0.0;
                }},
            InitialValueFunctionU0{
                [](double x) noexcept
                {
                    constexpr double pi =
                        3.141592653589793238462643383279502884;
                    return std::sin(pi * x);
                }},
            "smooth_initial",
            "1+1D compatible manufactured heat mode u(t,x) = exp(-pi^2 t) sin(pi x).",
            exact);

        example.initial_mesh.x_uniform_levels = 1;
        example.label = "1D compatible smooth heat mode";
        example.output_directory_name = "space_time_1d/smooth_initial";

        return example;
    }

    [[nodiscard]] inline SpaceTime1DExample
    non_matching_initial_1d_example()
    {
        SpaceTime1DExample example;
        example.problem = make_space_time_1d_problem_data(
            DiffusionFunctionM{
                [](const SpaceTimePoint&) noexcept
                {
                    return 1.0;
                }},
            LoadFunctionEll{
                [](const SpaceTimePoint&) noexcept
                {
                    return 2.0;
                }},
            InitialValueFunctionU0{
                [](double) noexcept
                {
                    return 1.0;
                }},
            "non_matching_initial",
            "1+1D constant diffusion and load with nonmatching initial datum u0 = 1.");

        example.initial_mesh.x_uniform_levels = 1;
        example.label = "1D nonmatching initial datum";
        example.output_directory_name = "space_time_1d/non_matching_initial";

        return example;
    }

    [[nodiscard]] inline SpaceTime1DExample
    boundary_singularity_1d_example()
    {
        SpaceTime1DExample example;
        example.problem = make_space_time_1d_problem_data(
            DiffusionFunctionM{
                [](const SpaceTimePoint&) noexcept
                {
                    return 1.0;
                }},
            LoadFunctionEll{
                [](const SpaceTimePoint&) noexcept
                {
                    return 0.0;
                }},
            InitialValueFunctionU0{
                [](double x) noexcept
                {
                    return std::sqrt(x) * (1.0 - x);
                }},
            "boundary_singularity",
            "1+1D constant diffusion with square-root boundary singularity in u0.");

        example.initial_mesh.x_uniform_levels = 1;
        example.label = "1D boundary singularity";
        example.output_directory_name = "space_time_1d/boundary_singularity";

        return example;
    }
}

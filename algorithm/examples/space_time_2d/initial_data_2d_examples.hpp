#pragma once

#include <cmath>

#include "space_time_2d_example.hpp"

namespace adaptive_algorithm::examples
{
    namespace initial_data_2d_detail
    {
        constexpr double pi =
            3.141592653589793238462643383279502884;
    }

    [[nodiscard]] inline SpaceTime2DExample
    smooth_initial_2d_example()
    {
        SpaceTime2DExample example;

        SpaceTime2DExactData exact;
        exact.u =
            [](const SpaceTime2DPoint& p) noexcept
            {
                const double factor =
                    std::exp(
                        -2.0 *
                        initial_data_2d_detail::pi *
                        initial_data_2d_detail::pi *
                        p[2]);
                return factor *
                    std::sin(initial_data_2d_detail::pi * p[0]) *
                    std::sin(initial_data_2d_detail::pi * p[1]);
            };
        exact.theta = exact.u;
        exact.grad_u_x =
            [](const SpaceTime2DPoint& p) noexcept
                -> SpaceTime2DSpatialGradient
            {
                const double factor =
                    std::exp(
                        -2.0 *
                        initial_data_2d_detail::pi *
                        initial_data_2d_detail::pi *
                        p[2]);
                return {
                    initial_data_2d_detail::pi *
                        factor *
                        std::cos(initial_data_2d_detail::pi * p[0]) *
                        std::sin(initial_data_2d_detail::pi * p[1]),
                    initial_data_2d_detail::pi *
                        factor *
                        std::sin(initial_data_2d_detail::pi * p[0]) *
                        std::cos(initial_data_2d_detail::pi * p[1])
                };
            };
        exact.grad_theta_x = exact.grad_u_x;

        example.problem = make_space_time_2d_problem_data(
            SpaceTime2DDiffusionFunctionM{
                finite_element::coefficients::IdentityDiffusion<2>{}},
            SpaceTime2DLoadFunctionEll{
                finite_element::coefficients::ZeroLoad{}},
            SpaceTime2DInitialValueFunctionU0{
                [](const SpaceTime2DSpatialPoint& x) noexcept
                {
                    return std::sin(initial_data_2d_detail::pi * x[0]) *
                        std::sin(initial_data_2d_detail::pi * x[1]);
                }},
            "smooth_initial",
            "2+1D compatible manufactured heat mode u(t,x,y) = exp(-2*pi^2 t) sin(pi x) sin(pi y).",
            exact);

        example.initial_mesh.root_mesh =
            SpaceTime2DInitialMeshKind::unit_square_two_triangles;
        example.initial_mesh.x_uniform_levels = 1;
        example.label = "2D compatible smooth heat mode";
        example.output_directory_name = "space_time_2d/smooth_initial";

        return example;
    }

    [[nodiscard]] inline SpaceTime2DExample
    non_matching_initial_2d_example()
    {
        SpaceTime2DExample example;
        example.problem = make_space_time_2d_problem_data(
            SpaceTime2DDiffusionFunctionM{
                finite_element::coefficients::IdentityDiffusion<2>{}},
            SpaceTime2DLoadFunctionEll{
                finite_element::coefficients::ZeroLoad{}},
            SpaceTime2DInitialValueFunctionU0{
                [](const SpaceTime2DSpatialPoint&) noexcept
                {
                    return 1.0;
                }},
            "non_matching_initial",
            "2+1D identity diffusion with homogeneous load and nonmatching initial datum u0 = 1.");

        example.initial_mesh.root_mesh =
            SpaceTime2DInitialMeshKind::unit_square_two_triangles;
        example.initial_mesh.x_uniform_levels = 1;
        example.label = "2D nonmatching initial datum";
        example.output_directory_name = "space_time_2d/non_matching_initial";

        return example;
    }

    [[nodiscard]] inline SpaceTime2DExample
    boundary_singularity_2d_example()
    {
        SpaceTime2DExample example;
        example.problem = make_space_time_2d_problem_data(
            SpaceTime2DDiffusionFunctionM{
                finite_element::coefficients::IdentityDiffusion<2>{}},
            SpaceTime2DLoadFunctionEll{
                finite_element::coefficients::ZeroLoad{}},
            SpaceTime2DInitialValueFunctionU0{
                [](const SpaceTime2DSpatialPoint& x) noexcept
                {
                    return std::pow(x[0], 0.75) *
                        (1.0 - x[0]) *
                        x[1] *
                        (1.0 - x[1]);
                }},
            "boundary_singularity",
            "2+1D identity diffusion with x^(3/4) boundary singularity in u0.");

        example.initial_mesh.root_mesh =
            SpaceTime2DInitialMeshKind::unit_square_two_triangles;
        example.initial_mesh.x_uniform_levels = 1;
        example.label = "2D boundary singularity";
        example.output_directory_name = "space_time_2d/boundary_singularity";

        return example;
    }
}

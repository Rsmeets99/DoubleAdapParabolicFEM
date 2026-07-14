#pragma once

#include <iostream>
#include <stdexcept>

#include "../../adaptive_driver.hpp"
#include "../../examples/space_time_1d/space_time_1d_example.hpp"
#include "../../examples/space_time_2d/space_time_2d_example.hpp"
#include "../../support/space_time_1d.hpp"
#include "../../support/space_time_2d.hpp"

#include "finite_element/fespace/fe_traits.hpp"
#include "finite_element/fespace/fespace.hpp"
#include "finite_element/fespace/policies.hpp"

#include "linear_algebra/eigen_backend/backend.hpp"

#include "mesh/mesh_traits.hpp"

#include "runner_execution.hpp"

namespace adaptive_algorithm::runners::detail
{
    template<int P, int DimSpace, int QuadratureBoost>
    int run_with_degree_for_dimension_with_quadrature(
        const RunnerOptions& options)
    {
        // In 1+1D QSpace is a Gauss order, hence exact through 2*QSpace-1.
        // In 2+1D QSpace is a triangle exactness degree. Match the 1D
        // effective exactness so the runner does not underintegrate 2D
        // polynomial products relative to the 1D path.
        constexpr int QSpace =
            ((DimSpace == 1) ? (2 * P + 1) : (4 * P + 1)) +
            QuadratureBoost;
        constexpr int QTime = 2 * P + 1 + QuadratureBoost;
        static_assert(QTime >= 1 && QTime <= 12);
        static_assert(
            (DimSpace == 1 && QSpace <= 12) ||
            (DimSpace == 2 && QSpace <= 22));

        using GeomTraits = mesh::MeshTraits<DimSpace>;
        using XFETraits = finite_element::FiniteElementTraits<GeomTraits, P, P>;
        using YFETraits = finite_element::FiniteElementTraits<GeomTraits, P, P>;
        using XSpaceType = finite_element::FESpace<
            GeomTraits,
            XFETraits,
            finite_element::SpaceTimePolicy>;
        using YSpaceType = finite_element::FESpace<
            GeomTraits,
            YFETraits,
            finite_element::SpaceOnlyPolicy>;
        using Backend = la::eigen::Backend;

        auto example =
            [&]()
            {
                if constexpr (DimSpace == 1)
                    return adaptive_algorithm::examples::make_1d_example(
                        options.example_name);
                else if constexpr (DimSpace == 2)
                    return adaptive_algorithm::examples::make_2d_example(
                        options.example_name);
            }();

        auto mesh =
            [&]()
            {
                if constexpr (DimSpace == 1)
                    return adaptive_algorithm::support::make_unit_mesh<
                        GeomTraits>();
                else if constexpr (DimSpace == 2)
                    return adaptive_algorithm::examples::
                        make_initial_mesh_from_example(example);
            }();

        XSpaceType x_space(mesh);
        YSpaceType y_space(mesh);
        adaptive_algorithm::examples::initialize_spaces_from_example(
            x_space,
            y_space,
            example);

        typename Backend::Solver solver;
        const auto solver_options = make_main_solver_options(options);
        const auto g_solver_options = make_g_solver_options(options);
        auto parameters = make_adaptive_parameters<P>(options, example);
        write_run_parameters_file(options, parameters, QSpace, QTime);

        const auto result =
            adaptive_algorithm::run_adaptive_driver<
                QSpace,
                QTime,
                Backend>(
                    x_space,
                    y_space,
                    example.problem,
                    solver,
                    solver_options,
                    parameters,
                    g_solver_options);

        std::cout
            << "example=" << example.problem.name
            << " dim_space=" << DimSpace
            << " p=" << P
            << " main_solver=" << resolved_main_solver_name(options.main_solver)
            << " g_solver=" << resolved_g_solver_name(options)
            << " outer_iterations=" << result.n_outer_iterations()
            << " converged=" << (result.converged ? "yes" : "no")
            << " terminated_early=" << (result.terminated_early ? "yes" : "no");

        if (!result.termination_reason.empty())
            std::cout << " reason=" << result.termination_reason;

        std::cout << '\n';

        if (parameters.output.export_history)
        {
            std::cout
                << "output_directory="
                << parameters.output.output_directory.string()
                << '\n';
        }

        // Non-convergence and handled caps are reported in the result summary.
        // They should not abort batch shells that intentionally run to a cap.
        return 0;
    }

    template<int P, int DimSpace>
    int run_degree_dimension_case(
        const RunnerOptions& options,
        const int quadrature_boost)
    {
        switch (quadrature_boost)
        {
        case 0:
            return run_with_degree_for_dimension_with_quadrature<
                P,
                DimSpace,
                0>(options);
        case 2:
            return run_with_degree_for_dimension_with_quadrature<
                P,
                DimSpace,
                2>(options);
        default:
            throw std::runtime_error(
                "Unsupported quadrature boost " +
                std::to_string(quadrature_boost) + ".");
        }
    }
}

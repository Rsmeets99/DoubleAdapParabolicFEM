#pragma once

#include <utility>

#include "../concepts/solver.hpp"
#include "linear_system.hpp"
#include "saddle_point_system.hpp"

namespace la::linear
{
    template<class Backend>
    void solve_in_place(
        LinearSystem<Backend>& system,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {})
    {
        solver.compute(system.matrix, options);
        solver.solve(system.rhs, system.solution);
    }

    template<class Backend>
    void solve_in_place_with_initial_guess(
        LinearSystem<Backend>& system,
        typename Backend::Solver& solver,
        const typename Backend::Vector& initial_guess,
        const la::concepts::SolverOptions& options = {})
    {
        solver.compute(system.matrix, options);
        solver.solve_with_initial_guess(
            system.rhs,
            initial_guess,
            system.solution);
    }

    template<class Backend>
    LinearSystem<Backend> solve(
        LinearSystem<Backend> system,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {})
    {
        solve_in_place(system, solver, options);
        return system;
    }

    template<class Backend>
    LinearSystem<Backend> solve_with_initial_guess(
        LinearSystem<Backend> system,
        typename Backend::Solver& solver,
        const typename Backend::Vector& initial_guess,
        const la::concepts::SolverOptions& options = {})
    {
        solve_in_place_with_initial_guess(
            system,
            solver,
            initial_guess,
            options);
        return system;
    }
}

namespace la::saddle
{
    template<class Backend>
    typename Backend::Vector solve_full_system(
        const SaddlePointBlocks<Backend>& blocks,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {})
    {
        auto system = blocks.make_full_system();
        la::linear::solve_in_place(system, solver, options);
        return system.solution;
    }

    template<class Backend>
    typename Backend::Vector solve_full_system(
        SaddlePointBlocks<Backend>&& blocks,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {})
    {
        auto system = std::move(blocks).make_full_system();
        la::linear::solve_in_place(system, solver, options);
        return system.solution;
    }

    template<class Backend>
    SaddlePointSolution<Backend> solve_and_split(
        const SaddlePointBlocks<Backend>& blocks,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {})
    {
        const auto full_solution = solve_full_system(blocks, solver, options);
        return split_saddle_point_solution<Backend>(
            full_solution,
            blocks.n_lambda,
            blocks.n_u);
    }

    template<class Backend>
    SaddlePointSolution<Backend> solve_and_split(
        SaddlePointBlocks<Backend>&& blocks,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {})
    {
        const int n_lambda = blocks.n_lambda;
        const int n_u = blocks.n_u;
        const auto full_solution = solve_full_system(std::move(blocks), solver, options);
        return split_saddle_point_solution<Backend>(
            full_solution,
            n_lambda,
            n_u);
    }
}

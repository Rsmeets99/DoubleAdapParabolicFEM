#pragma once

#include "../../linear_algebra/preconditioners/schur_approximations.hpp"
#include "../../linear_algebra/preconditioners/solve_saddle_system.hpp"

namespace finite_element::system
{
    template<class Backend>
    struct ParabolicGraphNormPreconditionerContext
    {
        // Uses the code convention C_pos = -C_signed internally.
        la::preconditioners::ParabolicGraphNormApproximation<Backend>
            approximation{};
    };

    template<class Backend>
    struct MainPreconditionerContext
    {
        const ParabolicGraphNormPreconditionerContext<Backend>* graph_norm =
            nullptr;

        [[nodiscard]] la::preconditioners::SaddlePreconditionerContext<Backend>
        to_linear_algebra_context() const noexcept
        {
            la::preconditioners::SaddlePreconditionerContext<Backend> out;
            out.graph_norm =
                graph_norm != nullptr ? &graph_norm->approximation : nullptr;
            return out;
        }
    };
}

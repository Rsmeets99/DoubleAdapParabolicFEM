#pragma once

#include <stdexcept>
#include <string>
#include <utility>

#include "linear_algebra/concepts/solver.hpp"

namespace la::preconditioners
{
    class PreconditionedSolveFailure : public std::runtime_error
    {
    public:
        PreconditionedSolveFailure(
            std::string message,
            la::concepts::SolverDiagnostics diagnostics)
            : std::runtime_error(std::move(message)),
              diagnostics_(std::move(diagnostics))
        {
        }

        [[nodiscard]] const la::concepts::SolverDiagnostics&
        diagnostics() const noexcept
        {
            return diagnostics_;
        }

    private:
        la::concepts::SolverDiagnostics diagnostics_{};
    };
}

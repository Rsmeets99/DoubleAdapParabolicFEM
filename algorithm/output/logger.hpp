#pragma once

#include <iomanip>
#include <ostream>
#include <sstream>
#include <string>

#include "../adaptive_parameters.hpp"
#include "../adaptive_result.hpp"

namespace adaptive_algorithm
{
    class AlgorithmLogger
    {
    public:
        explicit AlgorithmLogger(const AdaptiveOutputSettings& settings)
            : out_(settings.print_iteration_tables ? settings.stream : nullptr)
        {}

        [[nodiscard]] bool enabled() const noexcept
        {
            return out_ != nullptr;
        }

        template<typename CellIdType>
        void print_outer_iteration_start(
            const AdaptiveOuterIterationRecord<CellIdType>& record)
        {
            if (!enabled())
                return;

            *out_
                << "outer " << record.outer_iteration
                << "  X(cells=" << record.n_x_active_cells_before
                << ", dofs=" << record.n_x_true_dofs_before
                << ")  Y(cells=" << record.n_y_active_cells_before
                << ", dofs=" << record.n_y_true_dofs_before
                << ")\n";
        }

        template<typename CellIdType>
        void print_inner_iteration(
            int outer_iteration,
            const AdaptiveYIterationRecord<CellIdType>& record)
        {
            if (!enabled())
                return;

            *out_
                << "  inner " << outer_iteration << '.' << record.inner_iteration
                << "  t=" << format_seconds(record.elapsed_seconds)
                << "  dt=" << format_seconds(record.iteration_seconds)
                << "  est=" << std::setprecision(10) << record.y_estimator_squared
                << "  threshold=" << record.y_estimator_threshold_squared
                << "  flux=" << record.y_flux_squared
                << "  recon=" << record.y_reconstruction_squared
                << "  div=" << record.divergence_residual_squared
                << "  Y(cells=" << record.n_y_active_cells_before
                << "->" << record.n_y_active_cells_after
                << ", dofs=" << record.n_y_true_dofs_before
                << "->" << record.n_y_true_dofs_after
                << ")  marked=" << record.marked_y_cells.size();

            if (record.main_solve.available)
            {
                *out_
                    << "  main_solver="
                    << solver_type_name(record.main_solve.effective_solver)
                    << "  K(n=" << record.main_solve.n
                    << ", nnz=" << record.main_solve.nnz_matrix
                    << ")";
                *out_
                    << "  linear_dt="
                    << format_seconds(
                        record.main_solve.setup_seconds +
                        record.main_solve.solve_seconds);
            }

            if (record.stopping_criterion_satisfied)
                *out_ << "  stop=yes";
            else if (record.refined_y)
                *out_ << "  refined=yes";
            else
                *out_ << "  refined=no";

            *out_ << '\n';
        }

        template<typename CellIdType>
        void print_outer_iteration_summary(
            const AdaptiveOuterIterationRecord<CellIdType>& record)
        {
            if (!enabled())
                return;

            *out_
                << "  t=" << format_seconds(record.elapsed_seconds)
                << "  dt=" << format_seconds(record.iteration_seconds)
                << "  eta=" << std::setprecision(10) << record.eta_squared
                << "  lambda_Y=" << record.lambda_y_squared
                << "  trace=" << record.initial_trace_squared
                << "  X-marked=" << record.marked_x_cells.size()
                << "  X(cells=" << record.n_x_active_cells_before
                << "->" << record.n_x_active_cells_after
                << ", dofs=" << record.n_x_true_dofs_before
                << "->" << record.n_x_true_dofs_after
                << ")  Y(cells=" << record.n_y_active_cells_before
                << "->" << record.n_y_active_cells_after
                << ", dofs=" << record.n_y_true_dofs_before
                << "->" << record.n_y_true_dofs_after
                << ")";

            if (record.refined_x)
                *out_ << "  refined_X=yes";
            else
                *out_ << "  refined_X=no";

            *out_ << '\n';
        }

        template<class Backend, class XSpaceType, class YSpaceType>
        void print_completion(
            const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result)
        {
            if (!enabled())
                return;

            *out_
                << "completed outer iterations: " << result.n_outer_iterations()
                << "  elapsed=" << format_seconds(total_elapsed_seconds(result))
                << "  converged=" << (result.converged ? "yes" : "no");

            if (result.timing_enabled)
            {
                *out_
                    << "  timing_phases="
                    << result.timing_records.size();
            }

            if (!result.termination_reason.empty())
                *out_ << "  reason=" << result.termination_reason;

            *out_ << '\n';
        }

    private:
        [[nodiscard]] static std::string format_seconds(double seconds)
        {
            std::ostringstream out;
            out << std::fixed << std::setprecision(3) << seconds << 's';
            return out.str();
        }

        template<class Backend, class XSpaceType, class YSpaceType>
        [[nodiscard]] static double total_elapsed_seconds(
            const AdaptiveResult<Backend, XSpaceType, YSpaceType>& result) noexcept
        {
            if (result.outer_iterations.empty())
                return 0.0;

            return result.outer_iterations.back().elapsed_seconds;
        }

        std::ostream* out_ = nullptr;
    };
}

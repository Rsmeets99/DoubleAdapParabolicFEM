#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "../../core/openmp.hpp"
#include "../detail/timing.hpp"
#include "../fespace/functions.hpp"
#include "../fespace/prolongation.hpp"
#include "../assembly/main_system/parabolic_graph_norm.hpp"
#include "../assembly/main_system/two_pass_full_saddle_2d.hpp"
#include "assemble_saddle_problem.hpp"
#include "main_preconditioner_context.hpp"

#include "../../linear_algebra/operations/linalg_ops.hpp"
#include "../../linear_algebra/preconditioners/solve_saddle_system.hpp"
#include "../../linear_algebra/system/saddle_point_system.hpp"
#include "../../linear_algebra/system/solve.hpp"

#ifndef ADAPPARABOLICFEM_HAVE_MKL_PARDISO
#define ADAPPARABOLICFEM_HAVE_MKL_PARDISO 0
#endif

namespace finite_element::system
{
    struct MainSystemExportOptions
    {
        bool enabled = false;
        bool export_matrix_market = false;
        bool export_rhs = false;
        bool export_solution = false;
        int max_export_dofs = 20000;
        std::filesystem::path output_directory{};
        std::string prefix = "main_system";
    };

    namespace detail
    {
        template<class SparseMatrix>
        [[nodiscard]] std::size_t sparse_nnz(const SparseMatrix& A)
        {
            return la::block::estimate_nnz(A);
        }

        [[nodiscard]] inline std::size_t estimate_sparse_matrix_bytes(
            int rows,
            int cols,
            std::size_t nnz) noexcept
        {
            return finite_element::assembly::detail::
                estimate_compressed_sparse_matrix_bytes(rows, cols, nnz);
        }

        template<class Backend>
        [[nodiscard]] std::size_t estimate_triplet_bytes(std::size_t nnz) noexcept
        {
            return nnz *
                finite_element::assembly::detail::
                    estimated_triplet_bytes<typename Backend::SparseBuilder>();
        }

        [[nodiscard]] inline bool main_system_export_payload_allowed(
            const MainSystemExportOptions& options,
            int rows)
        {
            return options.enabled &&
                rows >= 0 &&
                options.max_export_dofs >= 0 &&
                rows <= options.max_export_dofs &&
                !options.output_directory.empty();
        }

        [[nodiscard]] inline std::filesystem::path main_system_export_path(
            const MainSystemExportOptions& options,
            const char* suffix)
        {
            return options.output_directory /
                (options.prefix + std::string(suffix));
        }

        template<class Backend>
        void write_matrix_market_coordinate(
            const typename Backend::SparseMatrix& matrix,
            const std::filesystem::path& path)
        {
            std::ofstream out(path);
            if (!out)
            {
                throw std::runtime_error(
                    "write_matrix_market_coordinate: failed to open '" +
                    path.string() + "'.");
            }

            out << std::setprecision(17);
            out << "%%MatrixMarket matrix coordinate real general\n";
            out << matrix.rows() << ' ' << matrix.cols() << ' '
                << sparse_nnz(matrix) << '\n';
            matrix.for_each_nonzero(
                [&](int row, int col, double value)
                {
                    // MatrixMarket uses 1-based indices.
                    out << (row + 1) << ' ' << (col + 1) << ' '
                        << value << '\n';
                });
        }

        template<class Vector>
        void write_matrix_market_vector(
            const Vector& vector,
            const std::filesystem::path& path)
        {
            std::ofstream out(path);
            if (!out)
            {
                throw std::runtime_error(
                    "write_matrix_market_vector: failed to open '" +
                    path.string() + "'.");
            }

            out << std::setprecision(17);
            out << "%%MatrixMarket matrix array real general\n";
            out << vector.size() << " 1\n";
            for (int i = 0; i < vector.size(); ++i)
                out << vector[i] << '\n';
        }

        [[nodiscard]] inline double kib_to_bytes(
            const std::optional<double>& value)
        {
            return value.has_value() ? *value * 1024.0 : 0.0;
        }

        template<class Backend>
        void write_main_system_export_metadata(
            const la::linear::LinearSystem<Backend>& system,
            const la::concepts::SolverDiagnostics& diagnostics,
            double setup_seconds,
            double solve_seconds,
            const MainSystemExportOptions& options,
            bool payload_allowed)
        {
            std::ofstream out(main_system_export_path(options, "_metadata.yml"));
            if (!out)
            {
                throw std::runtime_error(
                    "write_main_system_export_metadata: failed to open metadata file in '" +
                    options.output_directory.string() + "'.");
            }

            const auto& direct = diagnostics.direct_stats;
            out << std::setprecision(17);
            out << "solver_name: "
                << la::concepts::solver_type_name_for_validation(
                    diagnostics.effective_solver)
                << "\n";
            out << "requested_solver: "
                << la::concepts::solver_type_name_for_validation(
                    diagnostics.requested_solver)
                << "\n";
            out << "solver_status: success\n";
            out << "matrix_rows: " << system.matrix.rows() << "\n";
            out << "matrix_cols: " << system.matrix.cols() << "\n";
            out << "matrix_nnz: " << sparse_nnz(system.matrix) << "\n";
            out << "factor_nnz: ";
            if (direct.nnz_factors.has_value())
                out << *direct.nnz_factors;
            else
                out << "MISSING";
            out << "\n";
            out << "estimated_factor_memory_bytes: ";
            if (direct.estimated_in_core_peak_memory.has_value())
                out << kib_to_bytes(direct.estimated_in_core_peak_memory);
            else
                out << "MISSING";
            out << "\n";
            out << "symbolic_analysis_seconds: "
                << direct.symbolic_analysis_seconds.value_or(0.0) << "\n";
            out << "numeric_factorization_seconds: "
                << direct.numeric_factorization_seconds.value_or(0.0)
                << "\n";
            out << "backsolve_seconds: "
                << direct.backsolve_seconds.value_or(solve_seconds) << "\n";
            out << "setup_seconds: " << setup_seconds << "\n";
            out << "solve_seconds: " << solve_seconds << "\n";
            out << "residual_norm: "
                << diagnostics.linear_residual_absolute.value_or(0.0)
                << "\n";
            out << "relative_residual_norm: "
                << diagnostics.linear_residual_relative.value_or(0.0)
                << "\n";
            out << "export_payload_allowed: "
                << (payload_allowed ? "true" : "false") << "\n";
            out << "max_export_dofs: " << options.max_export_dofs << "\n";
            out << "matrix_market_exported: "
                << (payload_allowed && options.export_matrix_market
                        ? "true"
                        : "false")
                << "\n";
            out << "rhs_exported: "
                << (payload_allowed && options.export_rhs
                        ? "true"
                        : "false")
                << "\n";
            out << "solution_exported: "
                << (payload_allowed && options.export_solution
                        ? "true"
                        : "false")
                << "\n";
        }

        template<class Backend>
        void maybe_export_main_system(
            const la::linear::LinearSystem<Backend>& system,
            const la::concepts::SolverDiagnostics& diagnostics,
            double setup_seconds,
            double solve_seconds,
            const MainSystemExportOptions& options)
        {
            if (!options.enabled || options.output_directory.empty())
                return;

            std::filesystem::create_directories(options.output_directory);
            const bool payload_allowed =
                main_system_export_payload_allowed(options, system.matrix.rows());

            if (payload_allowed && options.export_matrix_market)
            {
                write_matrix_market_coordinate<Backend>(
                    system.matrix,
                    main_system_export_path(options, "_A.mtx"));
            }
            if (payload_allowed && options.export_rhs)
            {
                write_matrix_market_vector(
                    system.rhs,
                    main_system_export_path(options, "_rhs.mtx"));
            }
            if (payload_allowed && options.export_solution)
            {
                write_matrix_market_vector(
                    system.solution,
                    main_system_export_path(options, "_solution.mtx"));
            }

            write_main_system_export_metadata<Backend>(
                system,
                diagnostics,
                setup_seconds,
                solve_seconds,
                options,
                payload_allowed);
        }

        template<class Backend>
        void record_main_system_sparse_memory_diagnostics(
            const MainSystemBlocks<Backend>& blocks,
            const finite_element::detail::TimingRecorder& timing,
            bool memory_bounded_composition)
        {
            if (!timing.enabled())
                return;

            const auto& A = blocks.A_y();
            const auto& B = blocks.lower_left_B();
            const auto& C = blocks.C_signed();
            const std::size_t nnz_A = sparse_nnz(A);
            const std::size_t nnz_B = sparse_nnz(B);
            const std::size_t nnz_C = sparse_nnz(C);
            const std::size_t full_nnz_estimate =
                nnz_A + 2u * nnz_B + nnz_C;
            const int n_full = blocks.n_lambda() + blocks.n_u();

            const std::size_t bytes_A =
                estimate_sparse_matrix_bytes(A.rows(), A.cols(), nnz_A);
            const std::size_t bytes_B =
                estimate_sparse_matrix_bytes(B.rows(), B.cols(), nnz_B);
            const std::size_t bytes_C =
                estimate_sparse_matrix_bytes(C.rows(), C.cols(), nnz_C);
            const std::size_t block_bytes = bytes_A + bytes_B + bytes_C;
            const std::size_t full_matrix_bytes =
                estimate_sparse_matrix_bytes(n_full, n_full, full_nnz_estimate);
            const std::size_t full_triplet_bytes =
                estimate_triplet_bytes<Backend>(full_nnz_estimate);
            const std::size_t temporary_compose_bytes =
                memory_bounded_composition ? 0u : full_triplet_bytes;
            const std::size_t estimated_compose_peak_bytes =
                block_bytes + full_matrix_bytes + temporary_compose_bytes;

            timing.add(
                "main_system.sparse_memory.A_y_matrix_bytes",
                static_cast<double>(bytes_A));
            timing.add(
                "main_system.sparse_memory.B_matrix_bytes",
                static_cast<double>(bytes_B));
            timing.add(
                "main_system.sparse_memory.C_signed_matrix_bytes",
                static_cast<double>(bytes_C));
            timing.add(
                "main_system.sparse_memory.block_matrices_bytes",
                static_cast<double>(block_bytes));
            timing.add(
                "main_system.sparse_memory.full_saddle_nnz_estimate.count",
                static_cast<double>(full_nnz_estimate));
            timing.add(
                "main_system.sparse_memory.full_saddle_matrix_bytes",
                static_cast<double>(full_matrix_bytes));
            timing.add(
                "main_system.sparse_memory.full_saddle_triplet_bytes",
                static_cast<double>(full_triplet_bytes));
            timing.add(
                "main_system.sparse_memory.temporary_compose_bytes",
                static_cast<double>(temporary_compose_bytes));
            timing.add(
                "main_system.sparse_memory.estimated_compose_peak_bytes",
                static_cast<double>(estimated_compose_peak_bytes));
            timing.add(
                "main_system.sparse_memory.memory_bounded_composition",
                memory_bounded_composition ? 1.0 : 0.0);
        }

        inline void record_main_solver_memory_diagnostics(
            const la::concepts::SolverDiagnostics& diagnostics,
            const finite_element::detail::TimingRecorder& timing)
        {
            if (!timing.enabled())
                return;

            const auto& direct = diagnostics.direct_stats;
            const auto add_optional_count =
                [&](const char* phase, const auto& value)
                {
                    timing.add(
                        phase,
                        value.has_value()
                            ? static_cast<double>(*value)
                            : 0.0);
                };
            const auto add_optional_kib =
                [&](const char* phase, const std::optional<double>& value)
                {
                    timing.add(
                        phase,
                        value.has_value() ? *value * 1024.0 : 0.0);
                };

            add_optional_count(
                "main_system.sparse_memory.solver_matrix_nnz.count",
                direct.nnz_matrix);
            add_optional_count(
                "main_system.sparse_memory.solver_factor_nnz.count",
                direct.nnz_factors);
            add_optional_count(
                "main_system.factor_nnz.count",
                direct.nnz_factors);
            add_optional_count(
                "main_system.solver_object_construction_wall",
                direct.solver_object_construction_seconds);
            add_optional_count(
                "main_system.solver_symbolic_analysis_seconds",
                direct.symbolic_analysis_seconds);
            add_optional_count(
                "main_system.solver_symbolic_analysis_wall",
                direct.symbolic_analysis_seconds);
            timing.add(
                "main_system.solver_symbolic_analysis_reused.count",
                direct.symbolic_analysis_reused ? 1.0 : 0.0);
            add_optional_count(
                "main_system.solver_symbolic_pattern_cache_hits.count",
                direct.symbolic_pattern_cache_hits);
            add_optional_count(
                "main_system.solver_symbolic_pattern_cache_misses.count",
                direct.symbolic_pattern_cache_misses);
            add_optional_count(
                "main_system.solver_numeric_factorization_seconds",
                direct.numeric_factorization_seconds);
            add_optional_count(
                "main_system.solver_numeric_factorization_wall",
                direct.numeric_factorization_seconds);
            add_optional_count(
                "main_system.solver_backsolve_seconds",
                direct.backsolve_seconds);
            add_optional_count(
                "main_system.solver_backsolve_wall",
                direct.backsolve_seconds);
            timing.add("main_system.solver_symmetry_check_wall", 0.0);
            timing.add("main_system.solver_residual_check_wall", 0.0);
            timing.add("main_system.solver_diagnostics_wall", 0.0);
            timing.add("main_system.rhs_solves.count", 1.0);
            timing.add("main_system.solver_reordering_strategy_code.count", 0.0);
            add_optional_count(
                "solve.symbolic_analysis_seconds",
                direct.symbolic_analysis_seconds);
            timing.add(
                "solve.symbolic_analysis_reused.count",
                direct.symbolic_analysis_reused ? 1.0 : 0.0);
            add_optional_count(
                "solve.numeric_factorization_seconds",
                direct.numeric_factorization_seconds);
            add_optional_kib(
                "main_system.sparse_memory.pardiso_symbolic_memory_bytes",
                direct.symbolic_memory);
            add_optional_kib(
                "main_system.sparse_memory.pardiso_numerical_factor_memory_bytes",
                direct.numerical_factor_memory);
            add_optional_kib(
                "main_system.sparse_memory.pardiso_estimated_in_core_peak_memory_bytes",
                direct.estimated_in_core_peak_memory);
            add_optional_kib(
                "main_system.sparse_memory.pardiso_out_of_core_minimum_memory_bytes",
                direct.out_of_core_minimum_memory);
            add_optional_kib(
                "main_system.sparse_memory.process_rss_before_factorization_bytes",
                direct.process_rss_before_factorization);
            add_optional_kib(
                "main_system.sparse_memory.process_rss_after_factorization_bytes",
                direct.process_rss_after_factorization);
            add_optional_kib(
                "main_system.sparse_memory.process_rss_after_solve_bytes",
                direct.process_rss_after_solve);
            add_optional_kib(
                "main_system.sparse_memory.memory_guard_estimated_extra_bytes",
                direct.memory_guard_estimated_extra_memory);
            add_optional_kib(
                "main_system.sparse_memory.direct_memory_limit_bytes",
                direct.memory_limit);
            add_optional_kib(
                "main_system.sparse_memory.memory_guard_estimated_peak_bytes",
                direct.memory_guard_estimated_peak_memory);
            timing.add(
                "main_system.sparse_memory.memory_guard_triggered",
                direct.memory_guard_triggered ? 1.0 : 0.0);
            timing.add(
                "main_system.sparse_memory.pardiso_out_of_core_auto_switch_attempted",
                direct.pardiso_out_of_core_auto_switch_attempted ? 1.0 : 0.0);
        }

        template<class Backend>
        void record_main_system_matrix_counters(
            const la::linear::LinearSystem<Backend>& system,
            const finite_element::detail::TimingRecorder& timing,
            int matrix_copy_count,
            int matrix_compression_count)
        {
            if (!timing.enabled())
                return;

            timing.add(
                "main_system.matrix_rows.count",
                static_cast<double>(system.matrix.rows()));
            timing.add(
                "main_system.matrix_cols.count",
                static_cast<double>(system.matrix.cols()));
            timing.add(
                "main_system.matrix_nnz.count",
                static_cast<double>(sparse_nnz(system.matrix)));
            timing.add(
                "main_system.matrix_copy_count",
                static_cast<double>(matrix_copy_count));
            timing.add(
                "main_system.matrix_compression_count",
                static_cast<double>(matrix_compression_count));
            timing.add("main_system.matrix_finalize_wall", 0.0);
            timing.add("main_system.matrix_compression_wall", 0.0);
        }

        inline void record_solver_identity_counters(
            const la::concepts::SolverDiagnostics& diagnostics,
            const finite_element::detail::TimingRecorder& timing)
        {
            if (!timing.enabled())
                return;

            timing.add(
                "main_system.solver_status.success.count",
                1.0);
            timing.add(
                "main_system.solver_name." +
                    std::string(
                        la::concepts::solver_type_name_for_validation(
                            diagnostics.effective_solver)) +
                    ".count",
                1.0);
            timing.add(
                "main_system.requested_solver_name." +
                    std::string(
                        la::concepts::solver_type_name_for_validation(
                            diagnostics.requested_solver)) +
                    ".count",
                1.0);
            timing.add(
                "main_system.solver_direct.count",
                diagnostics.direct ? 1.0 : 0.0);
        }

        inline void record_solver_residual_counters(
            const la::concepts::SolverDiagnostics& diagnostics,
            const finite_element::detail::TimingRecorder& timing)
        {
            if (!timing.enabled())
                return;

            timing.add(
                "main_system.residual_norm",
                diagnostics.linear_residual_absolute.value_or(0.0));
            timing.add(
                "main_system.relative_residual_norm",
                diagnostics.linear_residual_relative.value_or(0.0));
        }

        template<class Backend>
        struct LinearResidualDiagnostics
        {
            double absolute = 0.0;
            double relative = 0.0;
        };

        [[nodiscard]] inline double sqrt_nonnegative_or_nan(
            double squared_norm) noexcept
        {
            if (!std::isfinite(squared_norm))
                return std::numeric_limits<double>::quiet_NaN();
            if (squared_norm < 0.0)
            {
                if (squared_norm > -1.0e-24)
                    return 0.0;
                return std::numeric_limits<double>::quiet_NaN();
            }
            return std::sqrt(squared_norm);
        }

        template<class Backend>
        [[nodiscard]] LinearResidualDiagnostics<Backend>
        compute_linear_residual_diagnostics(
            const typename Backend::SparseMatrix& K,
            const typename Backend::Vector& x,
            const typename Backend::Vector& b)
        {
            if (K.rows() == 0 && K.cols() == 0 &&
                x.size() == 0 && b.size() == 0)
            {
                return {0.0, 0.0};
            }

            const auto residual =
                la::ops::subtract(la::ops::matvec(K, x), b);
            const double residual_norm =
                sqrt_nonnegative_or_nan(
                    la::ops::dot(residual, residual));
            const double rhs_norm =
                sqrt_nonnegative_or_nan(la::ops::dot(b, b));
            // If b is the zero vector, a relative residual has no natural
            // scale.  Use 1 as a conservative denominator so the reported
            // relative value is the absolute residual and stays finite.
            const double denominator =
                std::isfinite(rhs_norm) && rhs_norm > 0.0 ? rhs_norm : 1.0;
            const double relative =
                std::isfinite(residual_norm)
                    ? residual_norm / denominator
                    : std::numeric_limits<double>::quiet_NaN();

            return {residual_norm, relative};
        }

        template<class Backend>
        [[nodiscard]] double vector_l2_norm(
            const typename Backend::Vector& vector)
        {
            if (vector.size() == 0)
                return 0.0;

            return sqrt_nonnegative_or_nan(
                la::ops::dot(vector, vector));
        }

        template<class Backend>
        void record_initial_guess_diagnostics(
            la::concepts::SolverDiagnostics& diagnostics,
            const typename Backend::SparseMatrix& K,
            const typename Backend::Vector& initial_guess,
            const typename Backend::Vector& b,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            auto timer =
                timing.scoped(
                    "main_system.correction_or_initial_guess_wall");
            const auto residual =
                compute_linear_residual_diagnostics<Backend>(
                    K,
                    initial_guess,
                    b);
            diagnostics.initial_guess_norm =
                vector_l2_norm<Backend>(initial_guess);
            diagnostics.initial_residual_absolute = residual.absolute;
            diagnostics.initial_residual_relative = residual.relative;
        }

        template<class Backend>
        void record_linear_residual_diagnostics(
            la::concepts::SolverDiagnostics& diagnostics,
            const typename Backend::SparseMatrix& K,
            const typename Backend::Vector& x,
            const typename Backend::Vector& b)
        {
            const auto residual =
                compute_linear_residual_diagnostics<Backend>(K, x, b);
            diagnostics.linear_residual_absolute = residual.absolute;
            diagnostics.linear_residual_relative = residual.relative;
        }

        template<class Backend>
        void record_matrix_symmetry_diagnostics(
            la::concepts::SolverDiagnostics& diagnostics,
            const typename Backend::SparseMatrix& K)
        {
            const auto symmetry =
                la::ops::relative_symmetry_diagnostics(K);
            diagnostics.matrix_norm = symmetry.matrix_norm;
            diagnostics.matrix_symmetry_difference_norm =
                symmetry.difference_norm;
            diagnostics.matrix_relative_asymmetry =
                symmetry.relative_asymmetry;
        }

        [[nodiscard]] inline la::concepts::SolverDiagnosticsMode
        diagnostics_mode_with_required_residual(
            const la::concepts::SolverOptions& options) noexcept
        {
            if (options.diagnostics_mode ==
                    la::concepts::SolverDiagnosticsMode::Off &&
                options.direct_residual_retry_mode !=
                    la::concepts::DirectResidualRetryMode::Disabled)
            {
                return la::concepts::SolverDiagnosticsMode::Summary;
            }

            return options.diagnostics_mode;
        }

        template<class Backend>
        void record_main_linear_system_diagnostics(
            la::concepts::SolverDiagnostics& diagnostics,
            const typename Backend::SparseMatrix& K,
            const typename Backend::Vector& x,
            const typename Backend::Vector& b,
            la::concepts::SolverDiagnosticsMode mode,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            if (mode == la::concepts::SolverDiagnosticsMode::Off)
            {
                if (timing.enabled())
                {
                    timing.add(
                        "main_system.solver_diagnostics.mode.off.count",
                        1.0);
                }
                return;
            }

            auto diagnostics_timer =
                timing.scoped("main_system.solver_diagnostics_wall");
            if (timing.enabled())
            {
                timing.add(
                    mode == la::concepts::SolverDiagnosticsMode::Detailed
                        ? "main_system.solver_diagnostics.mode.detailed.count"
                        : "main_system.solver_diagnostics.mode.summary.count",
                    1.0);
            }
            {
                auto timer =
                    timing.scoped(
                        "main_system.solver_residual_check_wall");
                record_linear_residual_diagnostics<Backend>(
                    diagnostics,
                    K,
                    x,
                    b);
            }
            if (mode == la::concepts::SolverDiagnosticsMode::Detailed)
            {
                auto timer =
                    timing.scoped(
                        "main_system.solver_symmetry_check_wall");
                record_matrix_symmetry_diagnostics<Backend>(
                    diagnostics,
                    K);
            }
            record_solver_residual_counters(diagnostics, timing);
        }

        [[nodiscard]] inline std::optional<la::concepts::SolverOptions>
        direct_failure_retry_options(
            const la::concepts::SolverOptions& original_options)
        {
            if (original_options.direct_residual_retry_mode ==
                la::concepts::DirectResidualRetryMode::Disabled)
            {
                return std::nullopt;
            }

            if (!la::concepts::solver_type_is_direct(original_options.solver))
                return std::nullopt;

            la::concepts::SolverOptions retry_options = original_options;
            retry_options.direct_residual_retry_mode =
                la::concepts::DirectResidualRetryMode::Disabled;
            retry_options.preconditioner =
                la::concepts::PreconditionerType::None;

#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
            if (original_options.solver ==
                    la::concepts::SolverType::PardisoLDLTAuto ||
                original_options.solver ==
                    la::concepts::SolverType::PardisoLDLT)
            {
                retry_options.solver = la::concepts::SolverType::PardisoLU;
                return retry_options;
            }
#else
            if (original_options.solver !=
                la::concepts::SolverType::SparseLU)
            {
                retry_options.solver = la::concepts::SolverType::SparseLU;
                return retry_options;
            }
#endif

            return std::nullopt;
        }

        [[nodiscard]] inline std::optional<la::concepts::SolverOptions>
        direct_residual_retry_options(
            const la::concepts::SolverOptions& original_options,
            const la::concepts::SolverDiagnostics& diagnostics)
        {
            if (diagnostics.effective_solver !=
                    la::concepts::SolverType::PardisoLDLT &&
                original_options.solver !=
                    la::concepts::SolverType::PardisoLDLTAuto &&
                original_options.solver !=
                    la::concepts::SolverType::PardisoLDLT)
            {
                return std::nullopt;
            }

            return direct_failure_retry_options(original_options);
        }

        [[nodiscard]] inline bool should_retry_direct_residual(
            const la::concepts::SolverOptions& options,
            const la::concepts::SolverDiagnostics& diagnostics)
        {
            if (options.direct_residual_retry_mode ==
                la::concepts::DirectResidualRetryMode::Disabled)
            {
                return false;
            }

            if (!(options.direct_residual_retry_tolerance > 0.0))
                return false;

            if (!diagnostics.direct)
                return false;

            if (!diagnostics.linear_residual_relative.has_value())
                return false;

            const double residual =
                *diagnostics.linear_residual_relative;
            return !std::isfinite(residual) ||
                residual > options.direct_residual_retry_tolerance;
        }

        template<class Backend>
        int apply_direct_residual_corrections(
            const typename Backend::SparseMatrix& K,
            const typename Backend::Vector& b,
            typename Backend::Vector& x,
            typename Backend::Solver& solver,
            la::concepts::SolverDiagnostics& diagnostics,
            la::concepts::SolverDiagnosticsMode diagnostics_mode,
            double tolerance,
            int max_steps = 3)
        {
            if (!diagnostics.direct || !(tolerance > 0.0) || max_steps <= 0)
                return 0;

            if (!diagnostics.linear_residual_relative.has_value())
                return 0;

            double best_residual = *diagnostics.linear_residual_relative;
            if (!std::isfinite(best_residual) || best_residual <= tolerance)
                return 0;

            int accepted_steps = 0;
            const double residual_before = best_residual;

            for (int step = 0; step < max_steps; ++step)
            {
                const auto residual =
                    la::ops::subtract(b, la::ops::matvec(K, x));
                typename Backend::Vector correction(x.size());

                try
                {
                    solver.solve(residual, correction);
                }
                catch (const std::exception&)
                {
                    break;
                }

                auto candidate = la::ops::add(x, correction);
                auto candidate_diagnostics = solver.last_diagnostics();
                record_main_linear_system_diagnostics<Backend>(
                    candidate_diagnostics,
                    K,
                    candidate,
                    b,
                    diagnostics_mode);
                candidate_diagnostics.requested_solver =
                    diagnostics.requested_solver;
                candidate_diagnostics.initial_guess_norm =
                    diagnostics.initial_guess_norm;
                candidate_diagnostics.initial_residual_absolute =
                    diagnostics.initial_residual_absolute;
                candidate_diagnostics.initial_residual_relative =
                    diagnostics.initial_residual_relative;
                candidate_diagnostics.residual_retry_attempted =
                    diagnostics.residual_retry_attempted;
                candidate_diagnostics.residual_retry_solver =
                    diagnostics.residual_retry_solver;
                candidate_diagnostics.residual_before_retry =
                    diagnostics.residual_before_retry;
                candidate_diagnostics.residual_after_retry =
                    diagnostics.residual_after_retry;
                candidate_diagnostics.direct_residual_correction_steps =
                    diagnostics.direct_residual_correction_steps;
                candidate_diagnostics.residual_before_correction =
                    diagnostics.residual_before_correction;
                candidate_diagnostics.residual_after_correction =
                    diagnostics.residual_after_correction;

                if (!candidate_diagnostics.linear_residual_relative.has_value())
                    break;

                const double candidate_residual =
                    *candidate_diagnostics.linear_residual_relative;
                if (!std::isfinite(candidate_residual) ||
                    candidate_residual >= best_residual)
                {
                    break;
                }

                x = std::move(candidate);
                diagnostics = std::move(candidate_diagnostics);
                best_residual = candidate_residual;
                ++accepted_steps;

                if (best_residual <= tolerance)
                    break;
            }

            if (accepted_steps > 0)
            {
                diagnostics.direct_residual_correction_steps +=
                    accepted_steps;
                if (!diagnostics.residual_before_correction.has_value())
                    diagnostics.residual_before_correction = residual_before;
                diagnostics.residual_after_correction = best_residual;
            }

            return accepted_steps;
        }

        [[nodiscard]] inline std::string format_scientific(double value)
        {
            std::ostringstream out;
            out << std::scientific << std::setprecision(16) << value;
            return out.str();
        }

        inline void throw_if_direct_residual_rejected(
            const char* context,
            const la::concepts::SolverOptions& options,
            const la::concepts::SolverDiagnostics& diagnostics)
        {
            if (options.direct_residual_retry_mode ==
                la::concepts::DirectResidualRetryMode::Disabled)
            {
                return;
            }

            if (!(options.direct_residual_retry_tolerance > 0.0))
                return;

            if (!diagnostics.direct)
                return;

            const bool residual_is_acceptable =
                diagnostics.linear_residual_relative.has_value() &&
                std::isfinite(*diagnostics.linear_residual_relative) &&
                *diagnostics.linear_residual_relative <=
                    options.direct_residual_retry_tolerance;
            if (residual_is_acceptable)
            {
                return;
            }

            const double reported_residual =
                diagnostics.linear_residual_relative.value_or(
                    std::numeric_limits<double>::quiet_NaN());
            std::string message = std::string(context) +
                ": rejecting direct main-system solve because true relative residual " +
                format_scientific(reported_residual) +
                " exceeds tolerance " +
                format_scientific(options.direct_residual_retry_tolerance) +
                ". requested_solver=" +
                std::string(
                    la::concepts::solver_type_name_for_validation(
                        diagnostics.requested_solver)) +
                ", effective_solver=" +
                std::string(
                    la::concepts::solver_type_name_for_validation(
                        diagnostics.effective_solver));

            if (diagnostics.residual_retry_attempted)
            {
                message += ", residual_before_retry=" +
                    format_scientific(
                        diagnostics.residual_before_retry.value_or(0.0)) +
                    ", residual_after_retry=" +
                    format_scientific(
                        diagnostics.residual_after_retry.value_or(
                            reported_residual));
                if (diagnostics.residual_retry_solver.has_value())
                {
                    message += ", retry_solver=" +
                        std::string(
                            la::concepts::solver_type_name_for_validation(
                                *diagnostics.residual_retry_solver));
                }
            }
            else
            {
                message +=
                    ", residual retry did not produce an accepted safer solve";
            }
            if (diagnostics.direct_residual_correction_steps > 0)
            {
                message += ", residual_correction_steps=" +
                    std::to_string(
                        diagnostics.direct_residual_correction_steps) +
                    ", residual_before_correction=" +
                    format_scientific(
                        diagnostics.residual_before_correction.value_or(0.0)) +
                    ", residual_after_correction=" +
                    format_scientific(
                        diagnostics.residual_after_correction.value_or(
                            reported_residual));
            }

            throw std::runtime_error(message);
        }
    }

    template<class Backend, class YSpaceType, class XSpaceType>
    struct SaddleProblemSolveResult
    {
        using Vector = typename Backend::Vector;

        Vector lambda_true;
        Vector u_true;
        la::concepts::SolverDiagnostics solver_diagnostics{};
        double solver_setup_seconds = 0.0;
        double solver_solve_seconds = 0.0;
    };

    template<class Backend>
    struct MainSystemInitialGuess
    {
        using Vector = typename Backend::Vector;

        // Full saddle vector in the code convention [lambda; u].
        const Vector* full_vector = nullptr;
        bool solve_correction_equation = false;

        [[nodiscard]] bool has_value() const noexcept
        {
            return full_vector != nullptr;
        }
    };

    template<class YSpaceType, class XSpaceType>
    struct MainSystemTwoLevelHierarchy
    {
        const YSpaceType* coarse_y_space = nullptr;
        const XSpaceType* coarse_x_space = nullptr;

        [[nodiscard]] bool has_value() const noexcept
        {
            return coarse_y_space != nullptr && coarse_x_space != nullptr;
        }
    };

    template<class Backend, class YSpaceType, class XSpaceType>
    [[nodiscard]] SaddleProblemSolveResult<Backend, YSpaceType, XSpaceType>
    solve_main_linear_system(
        la::linear::LinearSystem<Backend>&& system,
        int n_lambda,
        int n_u,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {},
        const MainSystemInitialGuess<Backend>& initial_guess = {},
        const finite_element::detail::TimingRecorder& timing = {},
        const MainSystemExportOptions& export_options = {})
    {
        if (n_lambda == 0 || n_u == 0)
        {
            throw std::runtime_error(
                "solve_main_linear_system: refusing to solve main system with "
                "zero true DoFs. Y/lambda true DoFs=" +
                std::to_string(n_lambda) + ", X/u true DoFs=" +
                std::to_string(n_u) + ".");
        }

        if (initial_guess.has_value() &&
            initial_guess.full_vector->size() != system.solution.size())
        {
            throw std::runtime_error(
                "solve_main_linear_system: initial guess size does not match the full saddle system.");
        }

        using Clock = std::chrono::steady_clock;
        detail::record_main_system_matrix_counters<Backend>(
            system,
            timing,
            0,
            0);
        timing.add("main_system.matrix_copy_or_conversion_wall", 0.0);

        auto solve_once =
            [&](const la::concepts::SolverOptions& solve_options,
                bool use_initial_guess)
            {
                const auto setup_start = Clock::now();
                solver.compute(system.matrix, solve_options);
                const auto setup_end = Clock::now();

                const auto solve_start = Clock::now();
                if (use_initial_guess && initial_guess.has_value() &&
                    initial_guess.solve_correction_equation)
                {
                    auto correction_timer =
                        timing.scoped(
                            "main_system.correction_or_initial_guess_wall");
                    auto residual =
                        la::ops::subtract(
                            system.rhs,
                            la::ops::matvec(
                                system.matrix,
                                *initial_guess.full_vector));
                    typename Backend::Vector correction(system.solution.size());
                    solver.solve(residual, correction);
                    system.solution =
                        la::ops::add(*initial_guess.full_vector, correction);
                }
                else if (use_initial_guess && initial_guess.has_value())
                {
                    auto correction_timer =
                        timing.scoped(
                            "main_system.correction_or_initial_guess_wall");
                    solver.solve_with_initial_guess(
                        system.rhs,
                        *initial_guess.full_vector,
                        system.solution);
                }
                else
                {
                    solver.solve(system.rhs, system.solution);
                }
                const auto solve_end = Clock::now();

                return std::pair{
                    std::chrono::duration<double>(
                        setup_end - setup_start)
                        .count(),
                    std::chrono::duration<double>(
                        solve_end - solve_start)
                        .count()};
            };

        double solver_setup_seconds = 0.0;
        double solver_solve_seconds = 0.0;
        la::concepts::SolverDiagnostics solver_diagnostics{};
        const auto post_solve_diagnostics_mode =
            detail::diagnostics_mode_with_required_residual(options);

        auto apply_residual_corrections =
            [&]()
            {
                const auto correction_start = Clock::now();
                const int steps =
                    detail::apply_direct_residual_corrections<Backend>(
                        system.matrix,
                        system.rhs,
                        system.solution,
                        solver,
                        solver_diagnostics,
                        post_solve_diagnostics_mode,
                        options.direct_residual_retry_tolerance);
                const auto correction_end = Clock::now();
                if (steps > 0)
                {
                    solver_solve_seconds +=
                        std::chrono::duration<double>(
                            correction_end - correction_start)
                            .count();
                }
                return steps;
            };

        const auto solver_total_start = Clock::now();
        try
        {
            const auto [setup_seconds, solve_seconds] =
                solve_once(options, true);
            solver_setup_seconds += setup_seconds;
            solver_solve_seconds += solve_seconds;

            solver_diagnostics = solver.last_diagnostics();
            if (initial_guess.has_value())
            {
                detail::record_initial_guess_diagnostics<Backend>(
                    solver_diagnostics,
                    system.matrix,
                    *initial_guess.full_vector,
                    system.rhs,
                    timing);
            }
            detail::record_main_linear_system_diagnostics<Backend>(
                solver_diagnostics,
                system.matrix,
                system.solution,
                system.rhs,
                post_solve_diagnostics_mode,
                timing);
        }
        catch (const std::exception&)
        {
            const auto retry_options =
                detail::direct_failure_retry_options(options);
            if (!retry_options.has_value())
                throw;

            const auto [setup_seconds, solve_seconds] =
                solve_once(*retry_options, false);
            solver_setup_seconds += setup_seconds;
            solver_solve_seconds += solve_seconds;

            solver_diagnostics = solver.last_diagnostics();
            detail::record_main_linear_system_diagnostics<Backend>(
                solver_diagnostics,
                system.matrix,
                system.solution,
                system.rhs,
                post_solve_diagnostics_mode,
                timing);
            solver_diagnostics.requested_solver = options.solver;
            solver_diagnostics.residual_retry_attempted = true;
            solver_diagnostics.residual_retry_solver =
                retry_options->solver;
            solver_diagnostics.residual_after_retry =
                solver_diagnostics.linear_residual_relative.value_or(0.0);
            if (detail::should_retry_direct_residual(
                    options,
                    solver_diagnostics))
            {
                (void)apply_residual_corrections();
                solver_diagnostics.residual_after_retry =
                    solver_diagnostics.linear_residual_relative.value_or(0.0);
            }
        }

        if (!solver_diagnostics.residual_retry_attempted &&
            detail::should_retry_direct_residual(options, solver_diagnostics))
        {
            (void)apply_residual_corrections();
        }

        if (!solver_diagnostics.residual_retry_attempted &&
            detail::should_retry_direct_residual(options, solver_diagnostics))
        {
            if (auto retry_options =
                    detail::direct_residual_retry_options(
                        options,
                        solver_diagnostics);
                retry_options.has_value())
            {
                const double residual_before =
                    solver_diagnostics.linear_residual_relative.value_or(0.0);

                const auto retry_setup_start = Clock::now();
                solver.compute(system.matrix, *retry_options);
                const auto retry_setup_end = Clock::now();

                const auto retry_solve_start = Clock::now();
                solver.solve(system.rhs, system.solution);
                const auto retry_solve_end = Clock::now();

                auto retry_diagnostics = solver.last_diagnostics();
                detail::record_main_linear_system_diagnostics<Backend>(
                    retry_diagnostics,
                    system.matrix,
                    system.solution,
                    system.rhs,
                    post_solve_diagnostics_mode,
                    timing);
                retry_diagnostics.requested_solver = options.solver;
                retry_diagnostics.residual_retry_attempted = true;
                retry_diagnostics.residual_retry_solver =
                    retry_options->solver;
                retry_diagnostics.residual_before_retry = residual_before;
                retry_diagnostics.residual_after_retry =
                    retry_diagnostics.linear_residual_relative.value_or(0.0);
                retry_diagnostics.initial_guess_norm =
                    solver_diagnostics.initial_guess_norm;
                retry_diagnostics.initial_residual_absolute =
                    solver_diagnostics.initial_residual_absolute;
                retry_diagnostics.initial_residual_relative =
                    solver_diagnostics.initial_residual_relative;
                retry_diagnostics.direct_residual_correction_steps =
                    solver_diagnostics.direct_residual_correction_steps;
                retry_diagnostics.residual_before_correction =
                    solver_diagnostics.residual_before_correction;
                retry_diagnostics.residual_after_correction =
                    solver_diagnostics.residual_after_correction;

                solver_diagnostics = std::move(retry_diagnostics);
                solver_setup_seconds +=
                    std::chrono::duration<double>(
                        retry_setup_end - retry_setup_start)
                        .count();
                solver_solve_seconds +=
                    std::chrono::duration<double>(
                        retry_solve_end - retry_solve_start)
                        .count();
                if (detail::should_retry_direct_residual(
                        options,
                        solver_diagnostics))
                {
                    (void)apply_residual_corrections();
                    solver_diagnostics.residual_after_retry =
                        solver_diagnostics.linear_residual_relative.value_or(
                            0.0);
                }
            }
        }

        detail::throw_if_direct_residual_rejected(
            "solve_main_linear_system",
            options,
            solver_diagnostics);
        timing.add(
            "main_system.solver_total_wall",
            std::chrono::duration<double>(
                Clock::now() - solver_total_start)
                .count());
        detail::record_solver_identity_counters(solver_diagnostics, timing);

        const auto postprocess_start = Clock::now();
        detail::record_main_solver_memory_diagnostics(
            solver_diagnostics,
            timing);
        detail::maybe_export_main_system<Backend>(
            system,
            solver_diagnostics,
            solver_setup_seconds,
            solver_solve_seconds,
            export_options);

        typename SaddleProblemSolveResult<
            Backend,
            YSpaceType,
            XSpaceType>::Vector lambda_true;
        typename SaddleProblemSolveResult<
            Backend,
            YSpaceType,
            XSpaceType>::Vector u_true;
        {
            auto timer = timing.scoped("main_system.solution_scatter_wall");
            auto split = la::saddle::split_saddle_point_solution<Backend>(
                system.solution,
                n_lambda,
                n_u);
            lambda_true = std::move(split.lambda);
            u_true = std::move(split.u);
        }

        SaddleProblemSolveResult<Backend, YSpaceType, XSpaceType> result;
        result.lambda_true = std::move(lambda_true);
        result.u_true = std::move(u_true);
        result.solver_diagnostics = std::move(solver_diagnostics);
        result.solver_setup_seconds = solver_setup_seconds;
        result.solver_solve_seconds = solver_solve_seconds;
        timing.add("main_system.solver_setup", result.solver_setup_seconds);
        timing.add("main_system.solver_solve", result.solver_solve_seconds);
        timing.add("solve.setup_seconds", result.solver_setup_seconds);
        timing.add(
            "solve.factorization_seconds",
            result.solver_setup_seconds);
        timing.add("solve.backsolve_seconds", result.solver_solve_seconds);
        timing.add(
            "main_system.postprocess_wall",
            std::chrono::duration<double>(
                Clock::now() - postprocess_start)
                .count());
        timing.add("main_system.cleanup_wall", 0.0);
        return result;
    }

    template<
        class Backend,
        class YSourceFunctionType,
        class XSourceFunctionType,
        class YTargetSpaceType,
        class XTargetSpaceType>
    [[nodiscard]] typename Backend::Vector
    make_saddle_initial_guess_vector(
        const YSourceFunctionType& lambda_coarse,
        const XSourceFunctionType& u_coarse,
        const YTargetSpaceType& y_target_space,
        const XTargetSpaceType& x_target_space)
    {
        auto lambda_target =
            finite_element::fespace::prolong_true_coefficients_nodal(
                lambda_coarse,
                y_target_space);
        auto u_target =
            finite_element::fespace::prolong_true_coefficients_nodal(
                u_coarse,
                x_target_space);

        typename Backend::Vector x0(lambda_target.size() + u_target.size());
        for (int i = 0; i < lambda_target.size(); ++i)
            x0[i] = lambda_target[i];
        for (int i = 0; i < u_target.size(); ++i)
            x0[lambda_target.size() + i] = u_target[i];

        return x0;
    }

    template<class Backend, class YSpaceType, class XSpaceType>
    [[nodiscard]] SaddleProblemSolveResult<Backend, YSpaceType, XSpaceType>
    solve_main_system_blocks(
        MainSystemBlocks<Backend>&& blocks,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {},
        const MainSystemInitialGuess<Backend>& initial_guess = {},
        const MainPreconditionerContext<Backend>&
            preconditioner_context =
                MainPreconditionerContext<Backend>{},
        const finite_element::detail::TimingRecorder& timing = {},
        const MainSystemExportOptions& export_options = {})
    {
        const int n_lambda = blocks.n_lambda();
        const int n_u = blocks.n_u();

        if (n_lambda == 0 || n_u == 0)
        {
            throw std::runtime_error(
                "solve_main_system_blocks: refusing to solve main system with "
                "zero true DoFs. Y/lambda true DoFs=" +
                std::to_string(n_lambda) + ", X/u true DoFs=" +
                std::to_string(n_u) +
                ". Refine the mesh or use a space with interior true DoFs.");
        }

        using Clock = std::chrono::steady_clock;
        const auto post_solve_diagnostics_mode =
            detail::diagnostics_mode_with_required_residual(options);

        if (la::preconditioners::is_saddle_preconditioned_solve_requested(
                options))
        {
            detail::record_main_system_sparse_memory_diagnostics(
                blocks,
                timing,
                false);
            auto system =
                [&blocks, &timing]()
                {
                    auto timer =
                        timing.scoped("main_system.compose_full_saddle_matrix");
                    auto conversion_timer =
                        timing.scoped(
                            "main_system.matrix_copy_or_conversion_wall");
                    return blocks.make_full_system(false);
                }();
            detail::record_main_system_matrix_counters<Backend>(
                system,
                timing,
                1,
                1);
            const auto linear_algebra_context =
                preconditioner_context.to_linear_algebra_context();
            auto preconditioned_result =
                [&]()
                {
                    const auto solver_total_start = Clock::now();
                    auto result =
                        la::preconditioners::
                            solve_saddle_system_with_preconditioner(
                                solver,
                                system.matrix,
                                system.rhs,
                                blocks.saddle_point_blocks(),
                                options,
                                initial_guess,
                                linear_algebra_context);
                    timing.add(
                        "main_system.solver_total_wall",
                        std::chrono::duration<double>(
                            Clock::now() - solver_total_start)
                            .count());
                    return result;
                }();

            if (preconditioned_result.has_value())
            {
                const auto postprocess_start = Clock::now();
                auto split = la::saddle::split_saddle_point_solution<Backend>(
                    preconditioned_result->solution,
                    n_lambda,
                    n_u);

                SaddleProblemSolveResult<Backend, YSpaceType, XSpaceType>
                    result;
                result.lambda_true = std::move(split.lambda);
                result.u_true = std::move(split.u);
                result.solver_diagnostics =
                    std::move(preconditioned_result->diagnostics);
                if (initial_guess.has_value())
                {
                    detail::record_initial_guess_diagnostics<Backend>(
                        result.solver_diagnostics,
                        system.matrix,
                        *initial_guess.full_vector,
                        system.rhs,
                        timing);
                }
                detail::record_main_linear_system_diagnostics<Backend>(
                    result.solver_diagnostics,
                    system.matrix,
                    preconditioned_result->solution,
                    system.rhs,
                    post_solve_diagnostics_mode,
                    timing);
                detail::record_main_solver_memory_diagnostics(
                    result.solver_diagnostics,
                    timing);
                detail::record_solver_identity_counters(
                    result.solver_diagnostics,
                    timing);
                result.solver_setup_seconds =
                    preconditioned_result->setup_seconds;
                result.solver_solve_seconds =
                    preconditioned_result->solve_seconds;
                detail::maybe_export_main_system<Backend>(
                    system,
                    result.solver_diagnostics,
                    result.solver_setup_seconds,
                    result.solver_solve_seconds,
                    export_options);
                timing.add(
                    "main_system.solver_setup",
                    result.solver_setup_seconds);
                timing.add(
                    "main_system.solver_solve",
                    result.solver_solve_seconds);
                timing.add(
                    "solve.setup_seconds",
                    result.solver_setup_seconds);
                timing.add(
                    "solve.factorization_seconds",
                    result.solver_setup_seconds);
                timing.add(
                    "solve.backsolve_seconds",
                    result.solver_solve_seconds);
                timing.add(
                    "main_system.postprocess_wall",
                    std::chrono::duration<double>(
                        Clock::now() - postprocess_start)
                        .count());
                return result;
            }
        }

        detail::record_main_system_sparse_memory_diagnostics(
            blocks,
            timing,
            false);
        auto system =
            [&blocks, &timing]()
            {
                auto timer =
                    timing.scoped("main_system.compose_full_saddle_matrix");
                auto conversion_timer =
                    timing.scoped(
                        "main_system.matrix_copy_or_conversion_wall");
                return std::move(blocks)
                    .make_full_system(false);
            }();
        detail::record_main_system_matrix_counters<Backend>(
            system,
            timing,
            1,
            1);

        if (initial_guess.has_value() &&
            initial_guess.full_vector->size() != system.solution.size())
        {
            throw std::runtime_error(
                "solve_main_system_blocks: initial guess size does not match the full saddle system.");
        }

        auto solve_once =
            [&](const la::concepts::SolverOptions& solve_options,
                bool use_initial_guess)
            {
                const auto setup_start = Clock::now();
                solver.compute(system.matrix, solve_options);
                const auto setup_end = Clock::now();

                const auto solve_start = Clock::now();
                if (use_initial_guess && initial_guess.has_value() &&
                    initial_guess.solve_correction_equation)
                {
                    auto correction_timer =
                        timing.scoped(
                            "main_system.correction_or_initial_guess_wall");
                    auto residual =
                        la::ops::subtract(
                            system.rhs,
                            la::ops::matvec(
                                system.matrix,
                                *initial_guess.full_vector));
                    typename Backend::Vector correction(system.solution.size());
                    solver.solve(residual, correction);
                    system.solution =
                        la::ops::add(*initial_guess.full_vector, correction);
                }
                else if (use_initial_guess && initial_guess.has_value())
                {
                    auto correction_timer =
                        timing.scoped(
                            "main_system.correction_or_initial_guess_wall");
                    solver.solve_with_initial_guess(
                        system.rhs,
                        *initial_guess.full_vector,
                        system.solution);
                }
                else
                {
                    solver.solve(system.rhs, system.solution);
                }
                const auto solve_end = Clock::now();

                return std::pair{
                    std::chrono::duration<double>(
                        setup_end - setup_start)
                        .count(),
                    std::chrono::duration<double>(
                        solve_end - solve_start)
                        .count()};
            };

        double solver_setup_seconds = 0.0;
        double solver_solve_seconds = 0.0;
        la::concepts::SolverDiagnostics solver_diagnostics{};

        auto apply_residual_corrections =
            [&]()
            {
                const auto correction_start = Clock::now();
                const int steps =
                    detail::apply_direct_residual_corrections<Backend>(
                        system.matrix,
                        system.rhs,
                        system.solution,
                        solver,
                        solver_diagnostics,
                        post_solve_diagnostics_mode,
                        options.direct_residual_retry_tolerance);
                const auto correction_end = Clock::now();
                if (steps > 0)
                {
                    solver_solve_seconds +=
                        std::chrono::duration<double>(
                            correction_end - correction_start)
                            .count();
                }
                return steps;
            };

        const auto solver_total_start = Clock::now();
        try
        {
            const auto [setup_seconds, solve_seconds] =
                solve_once(options, true);
            solver_setup_seconds += setup_seconds;
            solver_solve_seconds += solve_seconds;

            solver_diagnostics = solver.last_diagnostics();
            if (initial_guess.has_value())
            {
                detail::record_initial_guess_diagnostics<Backend>(
                    solver_diagnostics,
                    system.matrix,
                    *initial_guess.full_vector,
                    system.rhs,
                    timing);
            }
            detail::record_main_linear_system_diagnostics<Backend>(
                solver_diagnostics,
                system.matrix,
                system.solution,
                system.rhs,
                post_solve_diagnostics_mode,
                timing);
        }
        catch (const std::exception&)
        {
            const auto retry_options =
                detail::direct_failure_retry_options(options);
            if (!retry_options.has_value())
                throw;

            const auto [setup_seconds, solve_seconds] =
                solve_once(*retry_options, false);
            solver_setup_seconds += setup_seconds;
            solver_solve_seconds += solve_seconds;

            solver_diagnostics = solver.last_diagnostics();
            detail::record_main_linear_system_diagnostics<Backend>(
                solver_diagnostics,
                system.matrix,
                system.solution,
                system.rhs,
                post_solve_diagnostics_mode,
                timing);
            solver_diagnostics.requested_solver = options.solver;
            solver_diagnostics.residual_retry_attempted = true;
            solver_diagnostics.residual_retry_solver =
                retry_options->solver;
            solver_diagnostics.residual_after_retry =
                solver_diagnostics.linear_residual_relative.value_or(0.0);
            if (detail::should_retry_direct_residual(
                    options,
                    solver_diagnostics))
            {
                (void)apply_residual_corrections();
                solver_diagnostics.residual_after_retry =
                    solver_diagnostics.linear_residual_relative.value_or(0.0);
            }
        }

        if (!solver_diagnostics.residual_retry_attempted &&
            detail::should_retry_direct_residual(options, solver_diagnostics))
        {
            (void)apply_residual_corrections();
        }

        if (!solver_diagnostics.residual_retry_attempted &&
            detail::should_retry_direct_residual(options, solver_diagnostics))
        {
            if (auto retry_options =
                    detail::direct_residual_retry_options(
                        options,
                        solver_diagnostics);
                retry_options.has_value())
            {
                const double residual_before =
                    solver_diagnostics.linear_residual_relative.value_or(0.0);

                const auto retry_setup_start = Clock::now();
                solver.compute(system.matrix, *retry_options);
                const auto retry_setup_end = Clock::now();

                const auto retry_solve_start = Clock::now();
                solver.solve(system.rhs, system.solution);
                const auto retry_solve_end = Clock::now();

                auto retry_diagnostics = solver.last_diagnostics();
                detail::record_main_linear_system_diagnostics<Backend>(
                    retry_diagnostics,
                    system.matrix,
                    system.solution,
                    system.rhs,
                    post_solve_diagnostics_mode,
                    timing);
                retry_diagnostics.requested_solver = options.solver;
                retry_diagnostics.residual_retry_attempted = true;
                retry_diagnostics.residual_retry_solver =
                    retry_options->solver;
                retry_diagnostics.residual_before_retry = residual_before;
                retry_diagnostics.residual_after_retry =
                    retry_diagnostics.linear_residual_relative.value_or(0.0);
                retry_diagnostics.initial_guess_norm =
                    solver_diagnostics.initial_guess_norm;
                retry_diagnostics.initial_residual_absolute =
                    solver_diagnostics.initial_residual_absolute;
                retry_diagnostics.initial_residual_relative =
                    solver_diagnostics.initial_residual_relative;
                retry_diagnostics.direct_residual_correction_steps =
                    solver_diagnostics.direct_residual_correction_steps;
                retry_diagnostics.residual_before_correction =
                    solver_diagnostics.residual_before_correction;
                retry_diagnostics.residual_after_correction =
                    solver_diagnostics.residual_after_correction;

                solver_diagnostics = std::move(retry_diagnostics);
                solver_setup_seconds +=
                    std::chrono::duration<double>(
                        retry_setup_end - retry_setup_start)
                        .count();
                solver_solve_seconds +=
                    std::chrono::duration<double>(
                        retry_solve_end - retry_solve_start)
                        .count();
                if (detail::should_retry_direct_residual(
                        options,
                        solver_diagnostics))
                {
                    (void)apply_residual_corrections();
                    solver_diagnostics.residual_after_retry =
                        solver_diagnostics.linear_residual_relative.value_or(
                            0.0);
                }
            }
        }

        detail::throw_if_direct_residual_rejected(
            "solve_main_system_blocks",
            options,
            solver_diagnostics);
        timing.add(
            "main_system.solver_total_wall",
            std::chrono::duration<double>(
                Clock::now() - solver_total_start)
                .count());
        detail::record_solver_identity_counters(solver_diagnostics, timing);

        const auto postprocess_start = Clock::now();
        detail::record_main_solver_memory_diagnostics(
            solver_diagnostics,
            timing);
        detail::maybe_export_main_system<Backend>(
            system,
            solver_diagnostics,
            solver_setup_seconds,
            solver_solve_seconds,
            export_options);

        typename SaddleProblemSolveResult<
            Backend,
            YSpaceType,
            XSpaceType>::Vector lambda_true;
        typename SaddleProblemSolveResult<
            Backend,
            YSpaceType,
            XSpaceType>::Vector u_true;
        {
            auto timer = timing.scoped("main_system.solution_scatter_wall");
            auto split = la::saddle::split_saddle_point_solution<Backend>(
                system.solution,
                n_lambda,
                n_u);
            lambda_true = std::move(split.lambda);
            u_true = std::move(split.u);
        }

        SaddleProblemSolveResult<Backend, YSpaceType, XSpaceType> result;
        result.lambda_true = std::move(lambda_true);
        result.u_true = std::move(u_true);
        result.solver_diagnostics = std::move(solver_diagnostics);
        result.solver_setup_seconds = solver_setup_seconds;
        result.solver_solve_seconds = solver_solve_seconds;
        timing.add("main_system.solver_setup", result.solver_setup_seconds);
        timing.add("main_system.solver_solve", result.solver_solve_seconds);
        timing.add("solve.setup_seconds", result.solver_setup_seconds);
        timing.add(
            "solve.factorization_seconds",
            result.solver_setup_seconds);
        timing.add("solve.backsolve_seconds", result.solver_solve_seconds);
        timing.add(
            "main_system.postprocess_wall",
            std::chrono::duration<double>(
                Clock::now() - postprocess_start)
                .count());
        timing.add("main_system.cleanup_wall", 0.0);
        return result;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class YSpaceType,
        class XSpaceType,
        class ExampleType>
    SaddleProblemSolveResult<Backend, YSpaceType, XSpaceType>
    assemble_and_solve_into_functions(
        finite_element::Function<YSpaceType, typename Backend::Vector>& lambda_h,
        finite_element::Function<XSpaceType, typename Backend::Vector>& u_h,
        const YSpaceType& y_space,
        const XSpaceType& x_space,
        const ExampleType& example,
        typename Backend::Solver& solver,
        const la::concepts::SolverOptions& options = {},
        double zero_tol = 1e-15,
        double g_scale = -1.0,
        const MainSystemInitialGuess<Backend>& initial_guess = {},
        const MainSystemTwoLevelHierarchy<YSpaceType, XSpaceType>*
            two_level_hierarchy = nullptr,
        const finite_element::detail::TimingRecorder& timing = {},
        int main_assembly_max_threads = 0,
        double main_assembly_memory_budget_mb = 0.0,
        int main_two_pass_numeric_fill_max_threads = 1,
        double main_two_pass_numeric_fill_memory_budget_mb = 0.0,
        const MainSystemExportOptions& export_options = {},
        finite_element::assembly::TwoPassFullSaddleAssemblyCache2D<
            Backend,
            XSpaceType,
            YSpaceType>* two_pass_assembly_cache = nullptr)
    {
        constexpr bool can_use_two_pass_full_saddle =
            use_fixed_degree_main_kernel_2d_v<YSpaceType> &&
            use_fixed_degree_main_kernel_2d_v<XSpaceType> &&
            requires { typename Backend::SparsePatternBuilder; };

        if constexpr (can_use_two_pass_full_saddle)
        {
            if (!la::preconditioners::is_saddle_preconditioned_solve_requested(
                    options))
            {
                const int effective_main_assembly_max_threads =
                    YSpaceType::GT::dim_space_v == 2
                        ? main_assembly_max_threads
                        : 0;
                core::ScopedOpenMPMaxThreads thread_scope(
                    effective_main_assembly_max_threads);
                timing.add(
                    "main_system.thread_policy.configured_max_threads.count",
                    static_cast<double>(
                        effective_main_assembly_max_threads));
                timing.add(
                    "main_system.thread_policy.effective_max_threads.count",
                    static_cast<double>(
                        thread_scope.effective_max_threads()));
                timing.add(
                    "main_system.thread_policy.memory_budget_bytes",
                    main_assembly_memory_budget_mb * 1024.0 * 1024.0);

                finite_element::assembly::detail::AssemblySpaceCache<
                    YSpaceType> y_cache(y_space);
                finite_element::assembly::detail::AssemblySpaceCache<
                    XSpaceType> x_cache(x_space);
                finite_element::assembly::detail::ActiveAncestorCache<
                    XSpaceType> x_ancestor_cache(x_space);

                la::linear::LinearSystem<Backend> system;
                {
                    auto timer =
                        timing.scoped("main_system.direct_full_saddle_assembly");
                    auto assembly_total_timer =
                        timing.scoped("main_system.assembly_total_wall");
                    system =
                        finite_element::assembly::
                            assemble_main_full_saddle_two_pass_2d<
                                QSpace,
                                QTime,
                                Backend>(
                                x_space,
                                y_space,
                                x_cache,
                                y_cache,
                                x_ancestor_cache,
                                example.M,
                                example.ell,
                                example.u0,
                                zero_tol,
                                g_scale,
                                timing,
                                nullptr,
                                main_two_pass_numeric_fill_max_threads,
                                main_two_pass_numeric_fill_memory_budget_mb,
                                two_pass_assembly_cache);
                }

                auto result =
                    solve_main_linear_system<Backend, YSpaceType, XSpaceType>(
                        std::move(system),
                        y_space.dof_handler_ref().n_true_dofs(),
                        x_space.dof_handler_ref().n_true_dofs(),
                        solver,
                        options,
                        initial_guess,
                        timing,
                        export_options);

                {
                    auto timer =
                        timing.scoped("main_system.solution_update_wall");
                    auto function_timer =
                        timing.scoped("main_system.function_update_wall");
                    lambda_h.update_from_true_solution(result.lambda_true);
                    u_h.update_from_true_solution(result.u_true);
                }
                return result;
            }
        }

        auto blocks =
            [&]()
            {
                auto assembly_total_timer =
                    timing.scoped("main_system.assembly_total_wall");
                const int effective_main_assembly_max_threads =
                    YSpaceType::GT::dim_space_v == 2
                        ? main_assembly_max_threads
                        : 0;
                core::ScopedOpenMPMaxThreads thread_scope(
                    effective_main_assembly_max_threads);
                timing.add(
                    "main_system.thread_policy.configured_max_threads.count",
                    static_cast<double>(
                        effective_main_assembly_max_threads));
                timing.add(
                    "main_system.thread_policy.effective_max_threads.count",
                    static_cast<double>(
                        thread_scope.effective_max_threads()));
                timing.add(
                    "main_system.thread_policy.memory_budget_bytes",
                    main_assembly_memory_budget_mb * 1024.0 * 1024.0);

                return finite_element::system::assemble_main_system_blocks<
                    QSpace,
                    QTime,
                    Backend>(
                        y_space,
                        x_space,
                        example,
                        zero_tol,
                        g_scale,
                        timing);
            }();

        ParabolicGraphNormPreconditionerContext<Backend> graph_norm;
        MainPreconditionerContext<Backend>
            preconditioner_context;

        if (options.solver == la::concepts::SolverType::MINRES &&
            options.preconditioner ==
                la::concepts::PreconditionerType::ParabolicGraphNorm)
        {
            {
                auto timer =
                    timing.scoped("main_system.construct_graph_norm_context");
                graph_norm =
                    finite_element::assembly::assemble_parabolic_graph_norm_approximation<
                        QSpace,
                        QTime,
                        Backend>(
                            y_space,
                            x_space,
                            example.M,
                            blocks.A_y(),
                            blocks.C_signed(),
                            zero_tol);
            }
            preconditioner_context.graph_norm = &graph_norm;
        }
        static_cast<void>(two_level_hierarchy);

        auto result =
            solve_main_system_blocks<Backend, YSpaceType, XSpaceType>(
                std::move(blocks),
                solver,
                options,
                initial_guess,
                preconditioner_context,
                timing,
                export_options);

        {
            auto timer =
                timing.scoped("main_system.solution_update_wall");
            auto function_timer =
                timing.scoped("main_system.function_update_wall");
            lambda_h.update_from_true_solution(result.lambda_true);
            u_h.update_from_true_solution(result.u_true);
        }
        return result;
    }
}

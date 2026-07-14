#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <Eigen/SparseLU>

#ifndef ADAPPARABOLICFEM_HAVE_MKL_PARDISO
#define ADAPPARABOLICFEM_HAVE_MKL_PARDISO 0
#endif

#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
// Eigen's PardisoSupport stays fully optional and only participates in MKL-enabled builds.
#include <Eigen/PardisoSupport>
#endif

#include "linear_algebra/concepts/solver.hpp"
#include "linear_algebra/eigen_backend/operations.hpp"
#include "sparse_matrix.hpp"
#include "vector.hpp"

namespace la::eigen
{
    class Solver
    {
    public:
        using Matrix = SparseMatrix;
        using VectorType = Vector;
        using Options = la::concepts::SolverOptions;
        using Diagnostics = la::concepts::SolverDiagnostics;
        using Clock = std::chrono::steady_clock;

        void compute(const Matrix& A, const Options& options)
        {
            const bool preserve_sparse_lu =
                options.solver == la::concepts::SolverType::SparseLU &&
                options.reuse_symbolic_analysis_when_pattern_unchanged &&
                sparse_lu_ != nullptr &&
                sparse_lu_pattern_.matches(A.native());
            clear_(preserve_sparse_lu);
            computed_matrix_ = &A;
            options_ = options;
            initialize_diagnostics_(A, options);
            const auto validation =
                la::concepts::validate_solver_preconditioner(
                    options.solver,
                    options.preconditioner);
            la::concepts::apply_solver_preconditioner_validation_to_diagnostics(
                diagnostics_,
                validation);
            if (!validation.accepted)
                throw std::invalid_argument(validation.rejection_reason);

            if (A.rows() == 0 || A.cols() == 0)
            {
                throw std::invalid_argument(
                    "Solver::compute refuses to factor a linear system with zero rows or zero columns.");
            }

            switch (options.solver) {
                case la::concepts::SolverType::SparseLU:
                {
                    compute_sparse_lu_(A, options, preserve_sparse_lu);
                    break;
                }

                case la::concepts::SolverType::PardisoLU:
                {
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
                    compute_pardiso_lu_(A, options);
                    break;
#else
                    throw std::invalid_argument(
                        "PardisoLU was requested, but this build does not include MKL Pardiso support.");
#endif
                }

                case la::concepts::SolverType::PardisoLDLT:
                {
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
                    compute_pardiso_ldlt_(A, options);
                    break;
#else
                    throw std::invalid_argument(
                        "PardisoLDLT was requested, but this build does not include MKL Pardiso support.");
#endif
                }

                case la::concepts::SolverType::PardisoLDLTAuto:
                {
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
                    const auto symmetry =
                        la::ops::relative_symmetry_diagnostics(A);
                    if (symmetry.is_symmetric(options.symmetry_tolerance))
                    {
                        diagnostics_.effective_solver =
                            la::concepts::SolverType::PardisoLDLT;
                        compute_pardiso_ldlt_(A, options);
                    }
                    else
                    {
                        diagnostics_.effective_solver =
                            la::concepts::SolverType::PardisoLU;
                        diagnostics_.validation_warning_reason.reset();
                        std::cerr
                            << "Warning: PardisoLDLTAuto requested, but matrix relative asymmetry is "
                            << symmetry.relative_asymmetry
                            << " with tolerance "
                            << options.symmetry_tolerance
                            << ". Falling back to PardisoLU.\n";
                        compute_pardiso_lu_(A, options);
                    }
                    break;
#else
                    throw std::invalid_argument(
                        "PardisoLDLTAuto was requested, but this build does not include MKL Pardiso support.");
#endif
                }

                case la::concepts::SolverType::MINRES:
                    throw std::invalid_argument(
                        "The generic Eigen solver wrapper no longer constructs "
                        "MINRES. Use the main-system ParabolicGraphNorm saddle "
                        "preconditioner dispatch instead.");
            }
        }

        [[nodiscard]] const Diagnostics& last_diagnostics() const noexcept
        {
            return diagnostics_;
        }

        void solve(const VectorType& b, VectorType& x) const
        {
            if (sparse_lu_) {
                const auto solve_start = Clock::now();
                x.native() = sparse_lu_->solve(b.native());
                const auto solve_end = Clock::now();
                diagnostics_.direct_stats.backsolve_seconds =
                    std::chrono::duration<double>(solve_end - solve_start)
                        .count();
                if (sparse_lu_->info() != Eigen::Success) {
                    throw std::runtime_error("SparseLU solve failed.");
                }
                record_process_rss_after_solve_();
                record_linear_residual_(b, x);
                return;
            }

#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
            if (pardiso_lu_) {
                const auto solve_start = Clock::now();
                x.native() = pardiso_lu_->solve(b.native());
                const auto solve_end = Clock::now();
                diagnostics_.direct_stats.backsolve_seconds =
                    std::chrono::duration<double>(solve_end - solve_start)
                        .count();
                if (pardiso_lu_->info() != Eigen::Success) {
                    throw std::runtime_error("PardisoLU solve failed.");
                }
                record_process_rss_after_solve_();
                record_linear_residual_(b, x);
                return;
            }

            if (pardiso_ldlt_) {
                const auto solve_start = Clock::now();
                x.native() = pardiso_ldlt_->solve(b.native());
                const auto solve_end = Clock::now();
                diagnostics_.direct_stats.backsolve_seconds =
                    std::chrono::duration<double>(solve_end - solve_start)
                        .count();
                if (pardiso_ldlt_->info() != Eigen::Success) {
                    throw std::runtime_error("PardisoLDLT solve failed.");
                }
                record_process_rss_after_solve_();
                record_linear_residual_(b, x);
                return;
            }
#endif

            throw std::runtime_error("Solver::solve called before compute.");
        }

        void solve_with_initial_guess(
            const VectorType& b,
            const VectorType& initial_guess,
            VectorType& x) const
        {
            if (initial_guess.size() != b.size()) {
                throw std::runtime_error(
                    "Solver::solve_with_initial_guess initial guess size mismatch.");
            }

            // Direct solvers do not use an initial guess.
            solve(b, x);
        }

    private:
        [[nodiscard]] static bool is_direct_solver_(
            la::concepts::SolverType solver) noexcept
        {
            return la::concepts::solver_type_is_direct(solver);
        }

        void initialize_diagnostics_(const Matrix& A, const Options& options)
        {
            diagnostics_ = Diagnostics{};
            diagnostics_.requested_solver = options.solver;
            diagnostics_.effective_solver = options.solver;
            diagnostics_.preconditioner = options.preconditioner;
            diagnostics_.rows = A.rows();
            diagnostics_.cols = A.cols();
            diagnostics_.nnz_matrix =
                static_cast<std::size_t>(A.native().nonZeros());
            const auto symmetry =
                la::ops::relative_symmetry_diagnostics(A);
            diagnostics_.matrix_norm = symmetry.matrix_norm;
            diagnostics_.matrix_symmetry_difference_norm =
                symmetry.difference_norm;
            diagnostics_.matrix_relative_asymmetry =
                symmetry.relative_asymmetry;
            diagnostics_.direct = is_direct_solver_(options.solver);
            diagnostics_.iterative = !diagnostics_.direct;
            la::concepts::apply_preconditioner_safety_to_diagnostics(
                diagnostics_.preconditioner_stats,
                options.preconditioner);

            if (diagnostics_.direct) {
                diagnostics_.direct_stats.n = diagnostics_.rows;
                diagnostics_.direct_stats.nnz_matrix = diagnostics_.nnz_matrix;
            }

            if (auto warning =
                    la::concepts::pardiso_ldlt_memory_mode_warning(
                        options.solver,
                        options.pardiso_memory_mode);
                warning.has_value())
            {
                diagnostics_.validation_warning_reason = std::move(*warning);
            }
        }

        void record_iterative_diagnostics_(
            int iterations,
            double final_error,
            bool converged) const
        {
            diagnostics_.iterative_stats.iterations = iterations;
            diagnostics_.iterative_stats.final_error = final_error;
            diagnostics_.iterative_stats.converged = converged;
            diagnostics_.iterative_stats.backend_converged = converged;
            diagnostics_.iterative_stats.backend_reported_error =
                final_error;
            diagnostics_.iterative_stats.convergence_accepted_by_true_residual =
                false;
            diagnostics_.iterative_stats.residual_check_batches = 0;
            diagnostics_.iterative_stats.true_residual_stopping_used = false;
        }

        void record_linear_residual_(
            const VectorType& b,
            const VectorType& x) const
        {
            if (computed_matrix_ == nullptr)
                return;

            const auto residual =
                la::ops::subtract(la::ops::matvec(*computed_matrix_, x), b);
            const double residual_norm =
                std::sqrt(std::max(0.0, la::ops::dot(residual, residual)));
            const double rhs_norm =
                std::sqrt(std::max(0.0, la::ops::dot(b, b)));
            // With a zero RHS there is no meaningful relative scale.  Use 1
            // so the relative residual equals the absolute residual instead
            // of reporting an artificial infinity or NaN.
            const double denominator =
                rhs_norm > 0.0 ? rhs_norm : 1.0;

            diagnostics_.linear_residual_absolute = residual_norm;
            diagnostics_.linear_residual_relative =
                residual_norm / denominator;
        }

        [[nodiscard]] static double current_process_rss_kib_() noexcept
        {
            FILE* file = std::fopen("/proc/self/status", "r");
            if (file == nullptr)
                return 0.0;

            char line[256];
            unsigned long long rss_kib = 0;
            while (std::fgets(line, sizeof(line), file) != nullptr)
            {
                if (std::strncmp(line, "VmRSS:", 6) != 0)
                    continue;

                if (std::sscanf(line + 6, "%llu", &rss_kib) == 1)
                    break;
            }

            std::fclose(file);
            return static_cast<double>(rss_kib);
        }

        [[nodiscard]] static double available_system_memory_kib_() noexcept
        {
            FILE* file = std::fopen("/proc/meminfo", "r");
            if (file == nullptr)
                return 0.0;

            char key[64];
            char unit[32];
            unsigned long long value_kib = 0;
            unsigned long long mem_total_kib = 0;
            while (std::fscanf(file, "%63s %llu %31s", key, &value_kib, unit) == 3)
            {
                if (std::strcmp(key, "MemAvailable:") == 0)
                {
                    std::fclose(file);
                    return static_cast<double>(value_kib);
                }
                if (std::strcmp(key, "MemTotal:") == 0)
                    mem_total_kib = value_kib;
            }

            std::fclose(file);
            return static_cast<double>(mem_total_kib);
        }

        void record_process_rss_after_solve_() const
        {
            if (!diagnostics_.direct)
                return;

            diagnostics_.direct_stats.process_rss_after_solve =
                current_process_rss_kib_();
        }

        template<class IterativeSolver>
        void finish_iterative_solve_(
            const IterativeSolver& solver,
            const char* solver_name) const
        {
            const bool converged = solver.info() == Eigen::Success;
            record_iterative_diagnostics_(
                static_cast<int>(solver.iterations()),
                solver.error(),
                converged);
            if (!converged) {
                throw std::runtime_error(
                    std::string(solver_name) +
                    " solve failed. iterations=" +
                    std::to_string(solver.iterations()) +
                    ", error=" + std::to_string(solver.error()));
            }
        }

        struct SparseLUPatternSignature
        {
            using NativeMatrix = Matrix::native_type;
            using StorageIndex = typename NativeMatrix::StorageIndex;

            int rows = 0;
            int cols = 0;
            Eigen::Index nonzeros = 0;
            std::vector<StorageIndex> outer{};
            std::vector<StorageIndex> inner{};

            [[nodiscard]] bool matches(const NativeMatrix& A) const
            {
                if (outer.empty())
                    return false;
                if (rows != A.rows() || cols != A.cols() ||
                    nonzeros != A.nonZeros())
                    return false;

                const auto outer_size =
                    static_cast<std::size_t>(A.outerSize() + 1);
                const auto inner_size =
                    static_cast<std::size_t>(A.nonZeros());
                if (outer.size() != outer_size ||
                    inner.size() != inner_size)
                    return false;

                return std::equal(
                           outer.begin(),
                           outer.end(),
                           A.outerIndexPtr()) &&
                       std::equal(
                           inner.begin(),
                           inner.end(),
                           A.innerIndexPtr());
            }

            void assign(const NativeMatrix& A)
            {
                rows = static_cast<int>(A.rows());
                cols = static_cast<int>(A.cols());
                nonzeros = A.nonZeros();

                const auto outer_size =
                    static_cast<std::size_t>(A.outerSize() + 1);
                const auto inner_size =
                    static_cast<std::size_t>(A.nonZeros());
                outer.assign(
                    A.outerIndexPtr(),
                    A.outerIndexPtr() + outer_size);
                inner.assign(
                    A.innerIndexPtr(),
                    A.innerIndexPtr() + inner_size);
            }

            void clear()
            {
                rows = 0;
                cols = 0;
                nonzeros = 0;
                outer.clear();
                inner.clear();
            }
        };

        void compute_sparse_lu_(
            const Matrix& A,
            const Options& options,
            bool reuse_symbolic)
        {
            auto& direct = diagnostics_.direct_stats;
            direct.symbolic_pattern_cache_hits = sparse_lu_cache_hits_;
            direct.symbolic_pattern_cache_misses = sparse_lu_cache_misses_;

            if (!reuse_symbolic)
            {
                const auto construction_start = Clock::now();
                sparse_lu_ =
                    std::make_unique<Eigen::SparseLU<Matrix::native_type>>();
                const auto construction_end = Clock::now();
                direct.solver_object_construction_seconds =
                    std::chrono::duration<double>(
                        construction_end - construction_start)
                        .count();
                const auto symbolic_start = Clock::now();
                sparse_lu_->analyzePattern(A.native());
                const auto symbolic_end = Clock::now();
                direct.symbolic_analysis_seconds =
                    std::chrono::duration<double>(
                        symbolic_end - symbolic_start)
                        .count();
                direct.symbolic_analysis_reused = false;
                sparse_lu_pattern_.assign(A.native());
                if (options.reuse_symbolic_analysis_when_pattern_unchanged)
                    ++sparse_lu_cache_misses_;
            }
            else
            {
                direct.solver_object_construction_seconds = 0.0;
                direct.symbolic_analysis_seconds = 0.0;
                direct.symbolic_analysis_reused = true;
                ++sparse_lu_cache_hits_;
            }

            direct.symbolic_pattern_cache_hits = sparse_lu_cache_hits_;
            direct.symbolic_pattern_cache_misses = sparse_lu_cache_misses_;

            const auto numeric_start = Clock::now();
            sparse_lu_->factorize(A.native());
            const auto numeric_end = Clock::now();
            direct.numeric_factorization_seconds =
                std::chrono::duration<double>(
                    numeric_end - numeric_start)
                    .count();

            if (sparse_lu_->info() != Eigen::Success)
            {
                if (!reuse_symbolic)
                    sparse_lu_pattern_.clear();
                throw std::runtime_error("SparseLU factorization failed.");
            }
        }

        template<class IterativeSolver>
        void solve_iterative_system_(
            IterativeSolver& solver,
            const VectorType& b,
            VectorType& x,
            const char* solver_name) const
        {
            x.native() = solver.solve(b.native());
            record_linear_residual_(b, x);
            finish_iterative_solve_(solver, solver_name);
        }

        template<class IterativeSolver>
        void solve_iterative_system_with_guess_(
            IterativeSolver& solver,
            const VectorType& b,
            const VectorType& initial_guess,
            VectorType& x,
            const char* solver_name) const
        {
            if (initial_guess_satisfies_tolerance_(b, initial_guess, x))
            {
                record_linear_residual_(b, x);
                return;
            }

            x.native() =
                solver.solveWithGuess(b.native(), initial_guess.native());
            record_linear_residual_(b, x);
            finish_iterative_solve_(solver, solver_name);
        }

        [[nodiscard]] bool initial_guess_satisfies_tolerance_(
            const VectorType& b,
            const VectorType& initial_guess,
            VectorType& x) const
        {
            if (computed_matrix_ == nullptr)
                return false;

            VectorType::native_type residual =
                b.native() -
                computed_matrix_->native() * initial_guess.native();
            const double rhs_norm = b.native().norm();
            const double residual_norm = residual.norm();
            const double relative_error =
                rhs_norm > 0.0 ? residual_norm / rhs_norm : residual_norm;

            if (relative_error > options_.tolerance)
                return false;

            x.native() = initial_guess.native();
            record_iterative_diagnostics_(0, relative_error, true);
            return true;
        }

#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        void compute_pardiso_lu_(const Matrix& A, const Options& options)
        {
            compute_pardiso_impl_<
                Eigen::PardisoLU<Matrix::native_type>>(
                A,
                options,
                false,
                la::concepts::SolverType::PardisoLU);
        }

        void compute_pardiso_ldlt_(const Matrix& A, const Options& options)
        {
            compute_pardiso_impl_<
                Eigen::PardisoLDLT<Matrix::native_type>>(
                A,
                options,
                true,
                la::concepts::SolverType::PardisoLDLT);
        }

        template<class PardisoSolver>
        void compute_pardiso_impl_(
            const Matrix& A,
            const Options& options,
            bool enable_symmetric_weighted_matching,
            la::concepts::SolverType solver_type)
        {
            auto& direct = diagnostics_.direct_stats;
            const auto add_optional_seconds =
                [](std::optional<double>& target, double seconds)
                {
                    target = target.value_or(0.0) + seconds;
                };
            const auto build_solver =
                [&](la::concepts::PardisoMemoryMode memory_mode)
                {
                    auto solver = std::make_unique<PardisoSolver>();
                    configure_pardiso_parameters_(
                        *solver,
                        memory_mode,
                        enable_symmetric_weighted_matching,
                        options.pardiso_ldlt_robustness_profile);
                    return solver;
                };

            const auto construction_start = Clock::now();
            auto solver = build_solver(options.pardiso_memory_mode);
            const auto construction_end = Clock::now();
            direct.solver_object_construction_seconds =
                std::chrono::duration<double>(
                    construction_end - construction_start)
                    .count();

            const auto symbolic_start = Clock::now();
            solver->analyzePattern(A.native());
            const auto symbolic_end = Clock::now();
            direct.symbolic_analysis_seconds =
                std::chrono::duration<double>(
                    symbolic_end - symbolic_start)
                    .count();
            direct.symbolic_analysis_reused = false;
            if (solver->info() != Eigen::Success) {
                throw std::runtime_error(
                    std::string(
                        la::concepts::solver_type_name_for_validation(
                            solver_type)) +
                    " symbolic factorization failed.");
            }

            record_pardiso_analysis_memory_diagnostics_(*solver);
            if (options.pardiso_memory_mode ==
                la::concepts::PardisoMemoryMode::OutOfCore)
            {
                throw_if_pardiso_out_of_core_limit_exceeded_(
                    solver_type,
                    options,
                    diagnostics_.direct_stats);
            }

            const bool can_auto_switch =
                should_auto_switch_pardiso_out_of_core_(
                    options,
                    options.pardiso_memory_mode);
            if (options.pardiso_memory_mode ==
                    la::concepts::PardisoMemoryMode::InCore &&
                pardiso_in_core_estimate_exceeds_switch_threshold_(
                    options,
                    diagnostics_.direct_stats))
            {
                diagnostics_.direct_stats.memory_guard_triggered = true;

                if (can_auto_switch)
                {
                    const auto switch_solver =
                        selected_out_of_core_switch_solver_(
                            solver_type,
                            options);
                    diagnostics_.direct_stats
                        .pardiso_out_of_core_auto_switch_attempted = true;
                    diagnostics_.direct_stats
                        .pardiso_out_of_core_auto_switch_solver =
                        switch_solver;

                    if (switch_solver ==
                        la::concepts::SolverType::PardisoLU)
                    {
                        diagnostics_.effective_solver =
                            la::concepts::SolverType::PardisoLU;
                        compute_pardiso_lu_out_of_core_(A, options);
                        return;
                    }

                    const auto switch_construction_start = Clock::now();
                    solver = build_solver(
                        la::concepts::PardisoMemoryMode::OutOfCore);
                    const auto switch_construction_end = Clock::now();
                    add_optional_seconds(
                        direct.solver_object_construction_seconds,
                        std::chrono::duration<double>(
                            switch_construction_end -
                            switch_construction_start)
                            .count());
                    const auto switch_symbolic_start = Clock::now();
                    solver->analyzePattern(A.native());
                    const auto switch_symbolic_end = Clock::now();
                    add_optional_seconds(
                        direct.symbolic_analysis_seconds,
                        std::chrono::duration<double>(
                            switch_symbolic_end - switch_symbolic_start)
                            .count());
                    if (solver->info() != Eigen::Success) {
                        throw std::runtime_error(
                            std::string(
                                la::concepts::solver_type_name_for_validation(
                                    solver_type)) +
                            " out-of-core symbolic factorization failed.");
                    }
                    record_pardiso_analysis_memory_diagnostics_(*solver);
                    throw_if_pardiso_out_of_core_limit_exceeded_(
                        solver_type,
                        options,
                        diagnostics_.direct_stats);
                }
                else
                {
                    throw_pardiso_memory_limit_exceeded_(
                        solver_type,
                        options,
                        diagnostics_.direct_stats);
                }
            }

            const auto numeric_start = Clock::now();
            solver->factorize(A.native());
            const auto numeric_end = Clock::now();
            direct.numeric_factorization_seconds =
                std::chrono::duration<double>(
                    numeric_end - numeric_start)
                    .count();

            if (solver->info() != Eigen::Success)
            {
                if (should_auto_switch_pardiso_out_of_core_(
                        options,
                        options.pardiso_memory_mode))
                {
                    const auto switch_solver =
                        selected_out_of_core_switch_solver_(
                            solver_type,
                            options);
                    diagnostics_.direct_stats
                        .pardiso_out_of_core_auto_switch_attempted = true;
                    diagnostics_.direct_stats
                        .pardiso_out_of_core_auto_switch_solver =
                        switch_solver;

                    if (switch_solver ==
                        la::concepts::SolverType::PardisoLU)
                    {
                        diagnostics_.effective_solver =
                            la::concepts::SolverType::PardisoLU;
                        compute_pardiso_lu_out_of_core_(A, options);
                        return;
                    }

                    const auto switch_construction_start = Clock::now();
                    solver = build_solver(
                        la::concepts::PardisoMemoryMode::OutOfCore);
                    const auto switch_construction_end = Clock::now();
                    add_optional_seconds(
                        direct.solver_object_construction_seconds,
                        std::chrono::duration<double>(
                            switch_construction_end -
                            switch_construction_start)
                            .count());
                    const auto switch_symbolic_start = Clock::now();
                    solver->analyzePattern(A.native());
                    const auto switch_symbolic_end = Clock::now();
                    add_optional_seconds(
                        direct.symbolic_analysis_seconds,
                        std::chrono::duration<double>(
                            switch_symbolic_end - switch_symbolic_start)
                            .count());
                    if (solver->info() != Eigen::Success) {
                        throw std::runtime_error(
                            std::string(
                                la::concepts::solver_type_name_for_validation(
                                    solver_type)) +
                            " out-of-core symbolic factorization failed after in-core failure.");
                    }
                    record_pardiso_analysis_memory_diagnostics_(*solver);
                    throw_if_pardiso_out_of_core_limit_exceeded_(
                        solver_type,
                        options,
                        diagnostics_.direct_stats);
                    const auto retry_numeric_start = Clock::now();
                    solver->factorize(A.native());
                    const auto retry_numeric_end = Clock::now();
                    add_optional_seconds(
                        direct.numeric_factorization_seconds,
                        std::chrono::duration<double>(
                            retry_numeric_end - retry_numeric_start)
                            .count());
                }
            }

            if (solver->info() != Eigen::Success) {
                throw std::runtime_error(
                    std::string(
                        la::concepts::solver_type_name_for_validation(
                            solver_type)) +
                    " factorization failed.");
            }

            if constexpr (std::is_same_v<
                              PardisoSolver,
                              Eigen::PardisoLU<Matrix::native_type>>)
            {
                pardiso_lu_ = std::move(solver);
                record_pardiso_diagnostics_(*pardiso_lu_);
            }
            else
            {
                pardiso_ldlt_ = std::move(solver);
                record_pardiso_diagnostics_(*pardiso_ldlt_);
            }
        }

        void compute_pardiso_lu_out_of_core_(
            const Matrix& A,
            const Options& original_options)
        {
            Options out_of_core_options = original_options;
            out_of_core_options.solver = la::concepts::SolverType::PardisoLU;
            out_of_core_options.pardiso_memory_mode =
                la::concepts::PardisoMemoryMode::OutOfCore;
            out_of_core_options.pardiso_out_of_core_auto_switch = false;

            auto& direct = diagnostics_.direct_stats;
            const auto construction_start = Clock::now();
            pardiso_lu_ =
                std::make_unique<Eigen::PardisoLU<Matrix::native_type>>();
            configure_pardiso_parameters_(
                *pardiso_lu_,
                out_of_core_options.pardiso_memory_mode,
                false,
                out_of_core_options.pardiso_ldlt_robustness_profile);
            const auto construction_end = Clock::now();
            direct.solver_object_construction_seconds =
                direct.solver_object_construction_seconds.value_or(0.0) +
                std::chrono::duration<double>(
                    construction_end - construction_start)
                    .count();
            const auto symbolic_start = Clock::now();
            pardiso_lu_->analyzePattern(A.native());
            const auto symbolic_end = Clock::now();
            direct.symbolic_analysis_seconds =
                direct.symbolic_analysis_seconds.value_or(0.0) +
                std::chrono::duration<double>(
                    symbolic_end - symbolic_start)
                    .count();
            direct.symbolic_analysis_reused = false;
            if (pardiso_lu_->info() != Eigen::Success) {
                throw std::runtime_error(
                    "PardisoLU out-of-core symbolic factorization failed.");
            }
            record_pardiso_analysis_memory_diagnostics_(*pardiso_lu_);
            throw_if_pardiso_out_of_core_limit_exceeded_(
                la::concepts::SolverType::PardisoLU,
                out_of_core_options,
                diagnostics_.direct_stats);

            const auto numeric_start = Clock::now();
            pardiso_lu_->factorize(A.native());
            const auto numeric_end = Clock::now();
            direct.numeric_factorization_seconds =
                direct.numeric_factorization_seconds.value_or(0.0) +
                std::chrono::duration<double>(
                    numeric_end - numeric_start)
                    .count();
            if (pardiso_lu_->info() != Eigen::Success) {
                throw std::runtime_error(
                    "PardisoLU out-of-core factorization failed.");
            }
            record_pardiso_diagnostics_(*pardiso_lu_);
        }

        template<class PardisoSolver>
        static void configure_pardiso_parameters_(
            PardisoSolver& solver,
            la::concepts::PardisoMemoryMode memory_mode,
            bool enable_symmetric_weighted_matching,
            la::concepts::PardisoLdltRobustnessProfile ldlt_profile)
        {
            auto& iparm = solver.pardisoParameterArray();
            iparm[17] = -1;
            iparm[18] = -1;
            if (enable_symmetric_weighted_matching) {
                // Eigen's PARDISO wrapper disables weighted matching for
                // symmetric matrix types by default.  For the real
                // symmetric-indefinite saddle matrix (PARDISO mtype -2), MKL
                // recommends enabling matching when factorization accuracy is
                // poor.  The adaptive benchmark exposed LDLT residuals around
                // 1e-4 on tiny symmetric systems without this setting.
                iparm[12] = 1;
                apply_pardiso_ldlt_robustness_profile_(iparm, ldlt_profile);
            }
            iparm[59] =
                la::concepts::pardiso_memory_mode_iparm_59(memory_mode);
        }

        template<class PardisoParameterArray>
        static void apply_pardiso_ldlt_robustness_profile_(
            PardisoParameterArray& iparm,
            la::concepts::PardisoLdltRobustnessProfile profile)
        {
            using Profile = la::concepts::PardisoLdltRobustnessProfile;
            switch (profile)
            {
            case Profile::Production:
                break;
            case Profile::IterativeRefinement:
                iparm[7] = 2;
                break;
            case Profile::Scaling:
                iparm[10] = 1;
                break;
            case Profile::RefinementAndScaling:
                iparm[7] = 2;
                iparm[10] = 1;
                break;
            case Profile::PivotPerturbation1e13:
                iparm[9] = 13;
                break;
            case Profile::RefinementScalingPivotPerturbation1e13:
                iparm[7] = 2;
                iparm[9] = 13;
                iparm[10] = 1;
                break;
            }
        }

        template<class PardisoSolver>
        [[nodiscard]] static std::array<long long, 64>
        raw_pardiso_iparm_(PardisoSolver& solver)
        {
            auto& iparm = solver.pardisoParameterArray();
            std::array<long long, 64> raw_iparm{};
            for (std::size_t i = 0; i < raw_iparm.size(); ++i) {
                raw_iparm[i] = static_cast<long long>(iparm[i]);
            }
            return raw_iparm;
        }

        [[nodiscard]] static double memory_limit_kib_(
            const Options& options) noexcept
        {
            return options.direct_memory_limit_mb > 0.0
                ? options.direct_memory_limit_mb * 1024.0
                : 0.0;
        }

        [[nodiscard]] static double estimated_in_core_peak_kib_(
            const std::array<long long, 64>& iparm) noexcept
        {
            return static_cast<double>(
                std::max(iparm[14], iparm[15] + iparm[16]));
        }

        [[nodiscard]] static double estimated_incremental_pardiso_memory_kib_(
            const std::array<long long, 64>& iparm) noexcept
        {
            const double symbolic =
                std::max(0.0, static_cast<double>(iparm[14]));
            const double numerical =
                std::max(0.0, static_cast<double>(iparm[16]));
            const double peak = estimated_in_core_peak_kib_(iparm);
            return std::max(numerical, std::max(0.0, peak - symbolic));
        }

        [[nodiscard]] static double& pardiso_memory_estimate_multiplier_() noexcept
        {
            static double multiplier = 1.0;
            return multiplier;
        }

        [[nodiscard]] static double effective_memory_safety_factor_(
            const Options& options) noexcept
        {
            return std::max(
                options.direct_memory_safety_factor,
                pardiso_memory_estimate_multiplier_());
        }

        [[nodiscard]] static double calibrated_pardiso_memory_kib_(
            double raw_estimate_kib,
            const Options& options) noexcept
        {
            if (!(raw_estimate_kib > 0.0))
                return 0.0;

            return raw_estimate_kib *
                std::max(1.0, effective_memory_safety_factor_(options));
        }

        static void update_pardiso_memory_estimate_multiplier_(
            double raw_estimate_kib,
            double observed_delta_kib) noexcept
        {
            if (!(raw_estimate_kib > 0.0) || !(observed_delta_kib > 0.0))
                return;

            const double ratio = observed_delta_kib / raw_estimate_kib;
            if (!std::isfinite(ratio) || !(ratio > 0.0))
                return;

            auto& multiplier = pardiso_memory_estimate_multiplier_();
            const double blended = 0.75 * multiplier + 0.25 * ratio;
            multiplier = std::clamp(std::max(1.0, blended), 1.0, 3.0);
        }

        template<class PardisoSolver>
        void record_pardiso_analysis_memory_diagnostics_(
            PardisoSolver& solver)
        {
            const auto raw_iparm = raw_pardiso_iparm_(solver);
            auto& direct = diagnostics_.direct_stats;
            direct.pardiso_iparm = raw_iparm;
            direct.symbolic_memory = static_cast<double>(raw_iparm[14]);
            direct.numerical_factor_memory =
                static_cast<double>(raw_iparm[16]);
            direct.estimated_in_core_peak_memory =
                estimated_in_core_peak_kib_(raw_iparm);
            direct.memory_guard_estimated_extra_memory =
                calibrated_pardiso_memory_kib_(
                    estimated_incremental_pardiso_memory_kib_(raw_iparm),
                    options_);
            direct.process_rss_before_factorization =
                current_process_rss_kib_();
            direct.memory_limit = memory_limit_kib_(options_);
            direct.memory_guard_estimated_peak_memory =
                direct.process_rss_before_factorization.value_or(0.0) +
                direct.memory_guard_estimated_extra_memory.value_or(0.0);
            direct.effective_pardiso_memory_mode =
                static_cast<int>(raw_iparm[59]) == 2
                    ? la::concepts::PardisoMemoryMode::OutOfCore
                    : (static_cast<int>(raw_iparm[59]) == 1
                           ? la::concepts::PardisoMemoryMode::Auto
                           : la::concepts::PardisoMemoryMode::InCore);
            if (raw_iparm[62] > 0) {
                direct.out_of_core_minimum_memory =
                    static_cast<double>(raw_iparm[62]);
            }
        }

        [[nodiscard]] static bool
        should_auto_switch_pardiso_out_of_core_(
            const Options& options,
            la::concepts::PardisoMemoryMode current_memory_mode) noexcept
        {
            return options.pardiso_out_of_core_auto_switch &&
                current_memory_mode ==
                    la::concepts::PardisoMemoryMode::InCore;
        }

        [[nodiscard]] static bool
        pardiso_in_core_estimate_exceeds_switch_threshold_(
            const Options& options,
            const la::concepts::DirectSolverDiagnostics& direct) noexcept
        {
            const double limit_kib = memory_limit_kib_(options);
            if (!(limit_kib > 0.0) ||
                !direct.memory_guard_estimated_peak_memory.has_value())
            {
                return false;
            }

            const double threshold =
                options.pardiso_out_of_core_auto_switch
                    ? std::max(
                          0.0,
                          options.pardiso_out_of_core_switch_threshold)
                    : 1.0;
            const double effective_threshold =
                threshold > 0.0 ? threshold : 1.0;
            if (*direct.memory_guard_estimated_peak_memory >
                limit_kib * effective_threshold)
            {
                return true;
            }

            const double available_kib = available_system_memory_kib_();
            const double reserve_kib =
                std::max(0.0, options.direct_memory_reserve_mb) * 1024.0;
            if (available_kib > 0.0 &&
                direct.memory_guard_estimated_extra_memory.has_value())
            {
                return *direct.memory_guard_estimated_extra_memory >
                    std::max(0.0, available_kib - reserve_kib) *
                        effective_threshold;
            }

            return false;
        }

        [[nodiscard]] static bool
        pardiso_out_of_core_estimate_exceeds_limit_(
            const Options& options,
            const la::concepts::DirectSolverDiagnostics& direct) noexcept
        {
            const double limit_kib = memory_limit_kib_(options);
            if (!(limit_kib > 0.0) ||
                !direct.out_of_core_minimum_memory.has_value())
            {
                return false;
            }

            const double guarded_out_of_core_memory =
                calibrated_pardiso_memory_kib_(
                    *direct.out_of_core_minimum_memory,
                    options);

            if (guarded_out_of_core_memory > limit_kib)
                return true;

            const double available_kib = available_system_memory_kib_();
            const double reserve_kib =
                std::max(0.0, options.direct_memory_reserve_mb) * 1024.0;
            return available_kib > 0.0 &&
                guarded_out_of_core_memory >
                    std::max(0.0, available_kib - reserve_kib);
        }

        static void throw_if_pardiso_out_of_core_limit_exceeded_(
            la::concepts::SolverType solver_type,
            const Options& options,
            const la::concepts::DirectSolverDiagnostics& direct)
        {
            if (!pardiso_out_of_core_estimate_exceeds_limit_(
                    options,
                    direct))
            {
                return;
            }

            std::ostringstream message;
            message
                << std::string(
                       la::concepts::solver_type_name_for_validation(
                           solver_type))
                << " memory guard rejected out-of-core factorization";
            if (direct.out_of_core_minimum_memory.has_value())
            {
                message
                    << ": estimated out-of-core minimum memory "
                    << (*direct.out_of_core_minimum_memory / 1024.0)
                    << " MiB";
            }
            message
                << " exceeds memory limit "
                << options.direct_memory_limit_mb
                << " MiB.";
            throw la::concepts::DirectSolverMemoryLimitExceeded(message.str());
        }

        [[nodiscard]] static la::concepts::SolverType
        selected_out_of_core_switch_solver_(
            la::concepts::SolverType current_solver,
            const Options& options) noexcept
        {
            if (options.pardiso_out_of_core_switch_to_lu &&
                current_solver == la::concepts::SolverType::PardisoLDLT)
            {
                return la::concepts::SolverType::PardisoLU;
            }
            return current_solver;
        }

        [[noreturn]] static void throw_pardiso_memory_limit_exceeded_(
            la::concepts::SolverType solver_type,
            const Options& options,
            const la::concepts::DirectSolverDiagnostics& direct)
        {
            std::ostringstream message;
            message
                << std::string(
                       la::concepts::solver_type_name_for_validation(
                           solver_type))
                << " memory guard rejected numerical factorization";
            if (direct.estimated_in_core_peak_memory.has_value())
            {
                message
                    << ": PARDISO estimated in-core peak "
                    << (*direct.estimated_in_core_peak_memory / 1024.0)
                    << " MiB";
            }
            if (direct.memory_guard_estimated_extra_memory.has_value())
            {
                message
                    << ", estimated incremental PARDISO memory "
                    << (*direct.memory_guard_estimated_extra_memory / 1024.0)
                    << " MiB";
            }
            if (direct.process_rss_before_factorization.has_value())
            {
                message
                    << ", current process RSS "
                    << (*direct.process_rss_before_factorization / 1024.0)
                    << " MiB";
            }
            if (direct.memory_guard_estimated_peak_memory.has_value())
            {
                message
                    << ", estimated process peak "
                    << (*direct.memory_guard_estimated_peak_memory / 1024.0)
                    << " MiB";
            }
            message
                << " exceeds memory limit "
                << options.direct_memory_limit_mb
                << " MiB";
            if (options.pardiso_out_of_core_auto_switch)
            {
                message
                    << " after applying out-of-core switch threshold "
                    << options.pardiso_out_of_core_switch_threshold;
            }
            else
            {
                message
                    << "; enable PARDISO out-of-core auto-switch or raise "
                       "the main solver memory limit";
            }
            message << '.';
            throw la::concepts::DirectSolverMemoryLimitExceeded(message.str());
        }

        template<class PardisoSolver>
        void record_pardiso_diagnostics_(PardisoSolver& solver)
        {
            const auto raw_iparm = raw_pardiso_iparm_(solver);

            auto& direct = diagnostics_.direct_stats;
            direct.pardiso_iparm = raw_iparm;

            if (raw_iparm[17] >= 0) {
                direct.nnz_factors = static_cast<std::size_t>(raw_iparm[17]);
            }

            if (direct.nnz_factors.has_value() && diagnostics_.nnz_matrix > 0) {
                direct.fill_ratio =
                    static_cast<double>(*direct.nnz_factors) /
                    static_cast<double>(diagnostics_.nnz_matrix);
            }

            direct.symbolic_memory = static_cast<double>(raw_iparm[14]);
            direct.numerical_factor_memory = static_cast<double>(raw_iparm[16]);
            const auto estimated_in_core_peak =
                std::max(raw_iparm[14], raw_iparm[15] + raw_iparm[16]);
            direct.estimated_in_core_peak_memory =
                static_cast<double>(estimated_in_core_peak);
            const double raw_incremental_memory =
                estimated_incremental_pardiso_memory_kib_(raw_iparm);
            direct.memory_guard_estimated_extra_memory =
                calibrated_pardiso_memory_kib_(
                    raw_incremental_memory,
                    options_);
            direct.process_rss_after_factorization =
                current_process_rss_kib_();
            if (direct.process_rss_before_factorization.has_value() &&
                direct.process_rss_after_factorization.has_value())
            {
                update_pardiso_memory_estimate_multiplier_(
                    raw_incremental_memory,
                    *direct.process_rss_after_factorization -
                        *direct.process_rss_before_factorization);
            }

            if (raw_iparm[62] > 0) {
                direct.out_of_core_minimum_memory =
                    static_cast<double>(raw_iparm[62]);
            }

            direct.pardiso_perturbed_pivots = raw_iparm[13];
            direct.pardiso_positive_eigenvalues = raw_iparm[21];
            direct.pardiso_negative_eigenvalues = raw_iparm[22];
            direct.pardiso_iterative_refinement_steps = raw_iparm[7];
            direct.pardiso_pivot_perturbation = raw_iparm[9];
            direct.pardiso_scaling = raw_iparm[10];
            if (diagnostics_.effective_solver ==
                la::concepts::SolverType::PardisoLDLT)
            {
                direct.pardiso_ldlt_robustness_profile =
                    options_.pardiso_ldlt_robustness_profile;
            }
        }
#endif

        void clear_(bool preserve_sparse_lu = false)
        {
            if (!preserve_sparse_lu)
            {
                sparse_lu_.reset();
                sparse_lu_pattern_.clear();
            }
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
            pardiso_lu_.reset();
            pardiso_ldlt_.reset();
#endif
            computed_matrix_ = nullptr;
        }

        Options options_{};
        mutable Diagnostics diagnostics_{};
        const Matrix* computed_matrix_ = nullptr;

        std::unique_ptr<Eigen::SparseLU<Matrix::native_type>> sparse_lu_{};
        SparseLUPatternSignature sparse_lu_pattern_{};
        std::size_t sparse_lu_cache_hits_ = 0;
        std::size_t sparse_lu_cache_misses_ = 0;
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        std::unique_ptr<Eigen::PardisoLU<Matrix::native_type>> pardiso_lu_{};
        std::unique_ptr<Eigen::PardisoLDLT<Matrix::native_type>> pardiso_ldlt_{};
#endif
    };
}

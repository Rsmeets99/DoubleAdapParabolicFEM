#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "preconditioner.hpp"

#ifndef ADAPPARABOLICFEM_HAVE_MKL_PARDISO
#define ADAPPARABOLICFEM_HAVE_MKL_PARDISO 0
#endif

namespace la::concepts
{
    class DirectSolverMemoryLimitExceeded : public std::runtime_error
    {
    public:
        explicit DirectSolverMemoryLimitExceeded(const std::string& message)
            : std::runtime_error(message)
        {}
    };

    enum class SolverType
    {
        SparseLU,
        PardisoLU,
        PardisoLDLT,
        PardisoLDLTAuto,
        MINRES
    };

    [[nodiscard]] constexpr SolverType default_direct_solver_type() noexcept
    {
#if ADAPPARABOLICFEM_HAVE_MKL_PARDISO
        return SolverType::PardisoLDLTAuto;
#else
        return SolverType::SparseLU;
#endif
    }

    enum class PardisoMemoryMode
    {
        InCore,
        Auto,
        OutOfCore
    };

    enum class PardisoLdltRobustnessProfile
    {
        Production,
        IterativeRefinement,
        Scaling,
        RefinementAndScaling,
        PivotPerturbation1e13,
        RefinementScalingPivotPerturbation1e13
    };

    enum class DirectResidualRetryMode
    {
        Disabled,
        RetryWithSaferDirectSolver
    };

    enum class DenseFactorizationType
    {
        PartialPivotDenseLU,
        RankRevealingDenseLU
    };

    enum class SolverDiagnosticsMode
    {
        Off,
        Summary,
        Detailed
    };

    [[nodiscard]] constexpr std::string_view solver_diagnostics_mode_name(
        SolverDiagnosticsMode mode) noexcept
    {
        switch (mode)
        {
        case SolverDiagnosticsMode::Off:
            return "off";
        case SolverDiagnosticsMode::Summary:
            return "summary";
        case SolverDiagnosticsMode::Detailed:
            return "detailed";
        }

        return "unknown";
    }

    [[nodiscard]] constexpr std::string_view direct_residual_retry_mode_name(
        DirectResidualRetryMode mode) noexcept
    {
        switch (mode)
        {
        case DirectResidualRetryMode::Disabled:
            return "disabled";
        case DirectResidualRetryMode::RetryWithSaferDirectSolver:
            return "retry_with_safer_direct_solver";
        }

        return "unknown";
    }

    [[nodiscard]] constexpr int pardiso_memory_mode_iparm_59(
        PardisoMemoryMode mode) noexcept
    {
        switch (mode)
        {
        case PardisoMemoryMode::InCore:
            return 0;
        case PardisoMemoryMode::Auto:
            return 1;
        case PardisoMemoryMode::OutOfCore:
            return 2;
        }

        return 0;
    }

    [[nodiscard]] constexpr std::string_view pardiso_memory_mode_name(
        PardisoMemoryMode mode) noexcept
    {
        switch (mode)
        {
        case PardisoMemoryMode::InCore:
            return "in_core";
        case PardisoMemoryMode::Auto:
            return "auto";
        case PardisoMemoryMode::OutOfCore:
            return "out_of_core";
        }

        return "unknown";
    }

    [[nodiscard]] constexpr std::string_view
    pardiso_ldlt_robustness_profile_name(
        PardisoLdltRobustnessProfile profile) noexcept
    {
        switch (profile)
        {
        case PardisoLdltRobustnessProfile::Production:
            return "production";
        case PardisoLdltRobustnessProfile::IterativeRefinement:
            return "iterative_refinement";
        case PardisoLdltRobustnessProfile::Scaling:
            return "scaling";
        case PardisoLdltRobustnessProfile::RefinementAndScaling:
            return "refinement_scaling";
        case PardisoLdltRobustnessProfile::PivotPerturbation1e13:
            return "pivot_perturbation_1e13";
        case PardisoLdltRobustnessProfile::RefinementScalingPivotPerturbation1e13:
            return "refinement_scaling_pivot_perturbation_1e13";
        }

        return "unknown";
    }

    [[nodiscard]] constexpr bool pardiso_ldlt_memory_mode_is_experimental(
        SolverType solver,
        PardisoMemoryMode mode) noexcept
    {
        if (mode == PardisoMemoryMode::InCore)
            return false;

        return solver == SolverType::PardisoLDLT ||
            solver == SolverType::PardisoLDLTAuto;
    }

    [[nodiscard]] inline std::optional<std::string>
    pardiso_ldlt_memory_mode_warning(
        SolverType solver,
        PardisoMemoryMode mode)
    {
        if (!pardiso_ldlt_memory_mode_is_experimental(solver, mode))
            return std::nullopt;

        return
            "PARDISO LDLT with memory mode '" +
            std::string(pardiso_memory_mode_name(mode)) +
            "' is experimental for the saddle system: same-matrix diagnostics "
            "reproduced worse residuals for LDLT auto/out-of-core than for "
            "LDLT in-core. Prefer main_solver_pardiso_memory_mode=in_core for "
            "production LDLT runs.";
    }

    [[nodiscard]] constexpr bool solver_type_is_direct(
        SolverType solver) noexcept
    {
        switch (solver)
        {
        case SolverType::SparseLU:
        case SolverType::PardisoLU:
        case SolverType::PardisoLDLT:
        case SolverType::PardisoLDLTAuto:
            return true;
        case SolverType::MINRES:
            return false;
        }

        return false;
    }

    struct SolverOptions
    {
        SolverType solver = default_direct_solver_type();
        PreconditionerType preconditioner = PreconditionerType::None;
        // MKL PARDISO only: maps to iparm[59].
        PardisoMemoryMode pardiso_memory_mode = PardisoMemoryMode::InCore;
        // MKL PARDISO LDLT only.  Production preserves the current behavior:
        // weighted matching iparm[12] = 1 without extra robustness knobs.
        PardisoLdltRobustnessProfile pardiso_ldlt_robustness_profile =
            PardisoLdltRobustnessProfile::Production;

        int max_iterations = 1000;
        double tolerance = 1e-10;
        double symmetry_tolerance = 1.0e-12;

        bool use_true_residual_batching = true;
        int true_residual_check_interval = 100;
        DirectResidualRetryMode direct_residual_retry_mode =
            DirectResidualRetryMode::Disabled;
        double direct_residual_retry_tolerance = 1.0e-10;
        // Controls post-solve diagnostics around the main linear system.
        // Summary keeps the true residual available; detailed also performs
        // the expensive full sparse-matrix symmetry scan.
        SolverDiagnosticsMode diagnostics_mode =
            SolverDiagnosticsMode::Summary;

        // Global direct-solver memory guard.  A value <= 0 disables the guard.
        // PARDISO reports memory in KiB; the public option uses MiB.
        double direct_memory_limit_mb = 0.0;
        double direct_memory_reserve_mb = 0.0;
        double direct_memory_safety_factor = 1.15;
        // If enabled, an in-core PARDISO solve switches to out-of-core before
        // numerical factorization when the symbolic estimate reaches this
        // fraction of direct_memory_limit_mb.
        bool pardiso_out_of_core_auto_switch = false;
        double pardiso_out_of_core_switch_threshold = 0.85;
        // LDLT out-of-core has shown weaker residuals on this saddle system.
        // Prefer LU when the automatic memory fallback has to leave in-core.
        bool pardiso_out_of_core_switch_to_lu = true;

        // Reuse SparseLU symbolic analysis when the same Solver instance sees
        // an exactly identical compressed sparsity pattern. Numeric
        // factorization is still recomputed for every matrix.
        bool reuse_symbolic_analysis_when_pattern_unchanged = true;

        // Dense direct solves are used for small local systems.  The global
        // sparse solver backends ignore these fields.
        DenseFactorizationType dense_factorization =
            DenseFactorizationType::RankRevealingDenseLU;
        double dense_rank_revealing_threshold = 1.0e-12;
    };

    struct DirectSolverDiagnostics
    {
        std::optional<int> n{};
        std::optional<std::size_t> nnz_matrix{};
        std::optional<std::size_t> nnz_factors{};
        std::optional<double> fill_ratio{};
        std::optional<double> solver_object_construction_seconds{};
        std::optional<double> symbolic_analysis_seconds{};
        bool symbolic_analysis_reused = false;
        std::optional<std::size_t> symbolic_pattern_cache_hits{};
        std::optional<std::size_t> symbolic_pattern_cache_misses{};
        std::optional<double> numeric_factorization_seconds{};
        std::optional<double> backsolve_seconds{};
        // PARDISO memory diagnostics are stored in the units reported by iparm.
        std::optional<double> symbolic_memory{};
        std::optional<double> numerical_factor_memory{};
        std::optional<double> estimated_in_core_peak_memory{};
        std::optional<double> out_of_core_minimum_memory{};
        std::optional<double> process_rss_before_factorization{};
        std::optional<double> process_rss_after_factorization{};
        std::optional<double> process_rss_after_solve{};
        std::optional<double> memory_guard_estimated_extra_memory{};
        std::optional<double> memory_limit{};
        std::optional<double> memory_guard_estimated_peak_memory{};
        bool memory_guard_triggered = false;
        bool pardiso_out_of_core_auto_switch_attempted = false;
        std::optional<SolverType> pardiso_out_of_core_auto_switch_solver{};
        std::optional<PardisoMemoryMode> effective_pardiso_memory_mode{};
        // MKL PARDISO numerical diagnostics.  For symmetric-indefinite LDLT,
        // these correspond to iparm[13], iparm[21], and iparm[22].
        std::optional<PardisoLdltRobustnessProfile>
            pardiso_ldlt_robustness_profile{};
        std::optional<long long> pardiso_iterative_refinement_steps{};
        std::optional<long long> pardiso_pivot_perturbation{};
        std::optional<long long> pardiso_scaling{};
        std::optional<long long> pardiso_perturbed_pivots{};
        std::optional<long long> pardiso_positive_eigenvalues{};
        std::optional<long long> pardiso_negative_eigenvalues{};
        std::optional<std::array<long long, 64>> pardiso_iparm{};
    };

    struct IterativeSolverDiagnostics
    {
        std::optional<int> iterations{};
        std::optional<double> final_error{};
        std::optional<bool> converged{};
        std::optional<bool> backend_converged{};
        std::optional<double> backend_reported_error{};
        std::optional<bool> convergence_accepted_by_true_residual{};
        std::optional<int> residual_check_batches{};
        std::optional<double> final_true_residual{};
        std::optional<bool> true_residual_stopping_used{};
    };

    struct SolverDiagnostics
    {
        SolverType requested_solver = default_direct_solver_type();
        SolverType effective_solver = default_direct_solver_type();
        PreconditionerType preconditioner = PreconditionerType::None;

        int rows = 0;
        int cols = 0;
        std::size_t nnz_matrix = 0;

        bool direct = false;
        bool iterative = false;
        std::optional<double> linear_residual_absolute{};
        std::optional<double> linear_residual_relative{};
        std::optional<double> initial_guess_norm{};
        std::optional<double> initial_residual_absolute{};
        std::optional<double> initial_residual_relative{};
        std::optional<double> matrix_norm{};
        std::optional<double> matrix_symmetry_difference_norm{};
        std::optional<double> matrix_relative_asymmetry{};
        PreconditionerDiagnostics preconditioner_stats{};
        double preconditioner_setup_seconds = 0.0;

        DirectSolverDiagnostics direct_stats{};
        IterativeSolverDiagnostics iterative_stats{};
        bool residual_retry_attempted = false;
        std::optional<SolverType> residual_retry_solver{};
        std::optional<double> residual_before_retry{};
        std::optional<double> residual_after_retry{};
        int direct_residual_correction_steps = 0;
        std::optional<double> residual_before_correction{};
        std::optional<double> residual_after_correction{};
        std::optional<std::string> validation_warning_reason{};
        std::optional<std::string> validation_rejection_reason{};
    };

    struct SolverPreconditionerValidation
    {
        bool accepted = true;
        PreconditionerSafety preconditioner_safety{};
        std::string rejection_reason{};
    };

    [[nodiscard]] constexpr std::string_view solver_type_name_for_validation(
        SolverType solver) noexcept
    {
        switch (solver)
        {
        case SolverType::SparseLU:
            return "sparse_lu";
        case SolverType::PardisoLU:
            return "pardiso_lu";
        case SolverType::PardisoLDLT:
            return "pardiso_ldlt";
        case SolverType::PardisoLDLTAuto:
            return "pardiso_ldlt_auto";
        case SolverType::MINRES:
            return "minres_parabolic_graph_norm";
        }

        return "unknown";
    }

    [[nodiscard]] inline SolverPreconditionerValidation
    validate_solver_preconditioner(
        SolverType solver,
        PreconditionerType preconditioner)
    {
        SolverPreconditionerValidation validation;
        validation.preconditioner_safety =
            classify_preconditioner(preconditioner);

        const auto reject =
            [&](std::string reason)
            {
                validation.accepted = false;
                validation.rejection_reason = std::move(reason);
            };

        if (solver == SolverType::MINRES &&
            preconditioner != PreconditionerType::ParabolicGraphNorm)
        {
            reject(
                "MINRES is retained only for the ParabolicGraphNorm "
                "saddle preconditioner; requested preconditioner '" +
                std::string(preconditioner_type_name(preconditioner)) +
                "'.");
            return validation;
        }

        if (solver == SolverType::MINRES &&
            !validation.preconditioner_safety.is_spd_compatible())
        {
            reject(
                "MINRES requires a symmetric positive definite preconditioner; "
                "preconditioner '" +
                std::string(preconditioner_type_name(preconditioner)) +
                "' is " +
                std::string(validation.preconditioner_safety.description) +
                ".");
            return validation;
        }

        if (validation.preconditioner_safety.safety_class ==
            PreconditionerSafetyClass::UnknownUnsafe)
        {
            reject(
                "Preconditioner '" +
                std::string(preconditioner_type_name(preconditioner)) +
                "' has no safety classification for solver '" +
                std::string(solver_type_name_for_validation(solver)) + "'.");
            return validation;
        }

        return validation;
    }

    inline void apply_solver_preconditioner_validation_to_diagnostics(
        SolverDiagnostics& diagnostics,
        const SolverPreconditionerValidation& validation)
    {
        diagnostics.preconditioner_stats.assumes_spd =
            validation.preconditioner_safety.assumes_spd;
        diagnostics.preconditioner_stats.assumes_symmetric =
            validation.preconditioner_safety.assumes_symmetric;

        if (validation.preconditioner_safety.is_none())
        {
            diagnostics.preconditioner_stats.is_spd = true;
            diagnostics.preconditioner_stats.is_symmetric = true;
        }

        if (!validation.accepted)
        {
            diagnostics.validation_rejection_reason =
                validation.rejection_reason;
            diagnostics.preconditioner_stats.validation_rejection_reason =
                validation.rejection_reason;
        }
    }

    inline void sync_preconditioner_stats_from_legacy(
        SolverDiagnostics& diagnostics)
    {
        diagnostics.preconditioner_stats.setup_seconds =
            diagnostics.preconditioner_setup_seconds;
        diagnostics.preconditioner_stats.validation_rejection_reason =
            diagnostics.validation_rejection_reason;
    }

    inline void sync_preconditioner_legacy_from_stats(
        SolverDiagnostics& diagnostics)
    {
        diagnostics.preconditioner_setup_seconds =
            diagnostics.preconditioner_stats.setup_seconds;
        diagnostics.validation_rejection_reason =
            diagnostics.preconditioner_stats.validation_rejection_reason;
    }

    [[nodiscard]] inline SolverOptions make_sparse_lu_solver_options()
    {
        SolverOptions options;
        options.solver = SolverType::SparseLU;
        options.preconditioner = PreconditionerType::None;
        return options;
    }

    template<class Solver, class Matrix, class Vector>
    concept LinearSolverLike =
        requires(Solver s, const Matrix& A, const Vector& b, Vector& x, const SolverOptions& opts)
        {
            s.compute(A, opts);
            s.solve(b, x);
        };

    template<class Solver, class Matrix, class Vector>
    concept LinearSolverWithInitialGuessLike =
        LinearSolverLike<Solver, Matrix, Vector> &&
        requires(
            Solver s,
            const Matrix& A,
            const Vector& b,
            const Vector& x0,
            Vector& x,
            const SolverOptions& opts)
        {
            s.compute(A, opts);
            s.solve_with_initial_guess(b, x0, x);
        };
}

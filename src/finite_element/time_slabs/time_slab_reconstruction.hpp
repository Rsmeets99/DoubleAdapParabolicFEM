#pragma once

#include <concepts>
#include <chrono>
#include <cstddef>
#include <exception>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../../linear_algebra/concepts/solver.hpp"
#include "../../linear_algebra/concepts/vector.hpp"
#include "../../linear_algebra/operations/sparse_matrix_ops.hpp"
#include "../../core/openmp.hpp"
#include "../assembly/detail/active_cell_locator_time_slab.hpp"
#include "../assembly/detail/assembly_space_cache.hpp"
#include "../assembly/detail/openmp_assembly.hpp"
#include "../assembly/main_system/mat_A.hpp"
#include "../assembly/main_system/mat_B_time_slab.hpp"
#include "../assembly/main_system/vec_f.hpp"
#include "../coefficients/diffusion_coefficient.hpp"
#include "../detail/timing.hpp"
#include "time_slab_builder.hpp"
#include "time_slab_function.hpp"
#include "time_slab_reconstruction_operator.hpp"
#include "time_slab_space.hpp"

namespace finite_element::time_slabs
{
    template<class Backend,
             class XSpaceType,
             class YSpaceType>
    class TimeSlabReconstruction
    {
    public:
        using VectorType       = typename Backend::Vector;
        using SparseMatrixType = typename Backend::SparseMatrix;
        using SolverType       = typename Backend::Solver;

        using XSpace           = XSpaceType;
        using SourceYSpace     = YSpaceType;
        using GT               = typename SourceYSpace::GT;
        using FETraits         = typename SourceYSpace::FETraitsType;

        using SlabSpaceType      = TimeSlabSpace<GT, FETraits>;
        using SlabType           = typename SlabSpaceType::SlabType;
        using LocalSlabSpaceType = typename SlabType::SpaceType;
        using ReconstructedType  = TimeSlabFunction<SlabSpaceType, VectorType>;
        using ReconstructionOperator =
            TimeSlabReconstructionOperator<Backend, XSpace, SlabSpaceType>;
        using SlabDiagnostics =
            typename ReconstructionOperator::SlabDiagnostics;

        TimeSlabReconstruction(const SourceYSpace& y_space,
                               const XSpace& x_space)
            : source_y_space_(&y_space),
              x_space_(&x_space),
              slab_space_(y_space)
        {}

        void initialize(
            double time_tol = 0.0,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            TimeSlabBuilder<GT, FETraits>::initialize(
                slab_space_,
                time_tol,
                timing);

            {
                auto timer =
                    timing.scoped(
                        "time_slab.space_construction."
                        "reconstructed_function_init");
                reconstructed_.reset();
                reconstructed_.emplace(slab_space_);
                reconstructed_->set_zero();
            }
        }

        [[nodiscard]] const SourceYSpace& source_y_space() const noexcept
        {
            return *source_y_space_;
        }

        [[nodiscard]] const XSpace& x_space() const noexcept
        {
            return *x_space_;
        }

        [[nodiscard]] SlabSpaceType& slab_space_ref() noexcept
        {
            return slab_space_;
        }

        [[nodiscard]] const SlabSpaceType& slab_space_ref() const noexcept
        {
            return slab_space_;
        }

        [[nodiscard]] ReconstructedType& reconstructed_function()
        {
            ensure_initialized_();
            return *reconstructed_;
        }

        [[nodiscard]] const ReconstructedType& reconstructed_function() const
        {
            ensure_initialized_();
            return *reconstructed_;
        }

        [[nodiscard]] bool last_solve_all_slabs_used_openmp() const noexcept
        {
            return last_solve_all_slabs_used_openmp_;
        }

        [[nodiscard]] const std::vector<SlabDiagnostics>&
        last_slab_diagnostics() const noexcept
        {
            return last_slab_diagnostics_;
        }

        void set_slab_reconstruction_max_threads(int max_threads) noexcept
        {
            slab_reconstruction_max_threads_ = max_threads > 0 ? max_threads : 0;
        }

        void set_slab_reconstruction_memory_budget_mb(double budget_mb) noexcept
        {
            slab_reconstruction_memory_budget_mb_ =
                budget_mb > 0.0 ? budget_mb : 0.0;
        }

        void set_slab_reconstruction_operator_mode(std::string mode)
        {
            if (mode != "auto" &&
                mode != "identity_zero_load_fast_path" &&
                mode != "constant_diffusion_fast_path" &&
                mode != "generic_variable_path")
            {
                throw std::runtime_error(
                    "TimeSlabReconstruction: unsupported operator mode '" +
                    mode +
                    "'. Expected auto, identity_zero_load_fast_path, "
                    "constant_diffusion_fast_path, or generic_variable_path.");
            }
            slab_reconstruction_operator_mode_ = std::move(mode);
        }

        [[nodiscard]] const std::string&
        slab_reconstruction_operator_mode() const noexcept
        {
            return slab_reconstruction_operator_mode_;
        }

        [[nodiscard]] int last_configured_max_threads() const noexcept
        {
            return last_configured_max_threads_;
        }

        [[nodiscard]] int last_effective_max_threads() const noexcept
        {
            return last_effective_max_threads_;
        }

        [[nodiscard]] int last_selected_threads() const noexcept
        {
            return last_selected_threads_;
        }

        [[nodiscard]] double last_memory_budget_bytes() const noexcept
        {
            return last_memory_budget_bytes_;
        }

        [[nodiscard]] double last_estimated_per_thread_cache_bytes() const noexcept
        {
            return last_estimated_per_thread_cache_bytes_;
        }

        template<int QSpace, int QTime, class MFunction>
        void assemble_mat_A_on_slab(
            SparseMatrixType& A_slab,
            int slab_id,
            const MFunction& M,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& cache,
            double zero_tol = 1.0e-15) const
        {
            const auto& y_slab_space = slab_space_.slab(slab_id).fespace_ref();

            finite_element::assembly::assemble_mat_A<
                QSpace,
                QTime,
                Backend>(
                    A_slab,
                    y_slab_space,
                    M,
                    cache,
                    zero_tol);
        }

        template<int QSpace, int QTime, class MFunction>
        void assemble_mat_A_on_slab(
            SparseMatrixType& A_slab,
            int slab_id,
            const MFunction& M,
            double zero_tol = 1.0e-15) const
        {
            const auto& y_slab_space = slab_space_.slab(slab_id).fespace_ref();
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType> cache(y_slab_space);
            assemble_mat_A_on_slab<QSpace, QTime>(A_slab, slab_id, M, cache, zero_tol);
        }

        template<int QSpace, int QTime, class EllFunction>
        void assemble_vec_f_on_slab(
            VectorType& f_slab,
            int slab_id,
            const EllFunction& ell,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& cache,
            double zero_tol = 1.0e-15) const
        {
            const auto& y_slab_space = slab_space_.slab(slab_id).fespace_ref();

            finite_element::assembly::assemble_vec_f<
                QSpace,
                QTime>(
                    f_slab,
                    y_slab_space,
                    ell,
                    cache,
                    zero_tol);
        }

        template<int QSpace, int QTime, class EllFunction>
        void assemble_vec_f_on_slab(
            VectorType& f_slab,
            int slab_id,
            const EllFunction& ell,
            double zero_tol = 1.0e-15) const
        {
            const auto& y_slab_space = slab_space_.slab(slab_id).fespace_ref();
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType> cache(y_slab_space);
            assemble_vec_f_on_slab<QSpace, QTime>(f_slab, slab_id, ell, cache, zero_tol);
        }

        template<int QSpace, int QTime, class XFunctionType, class MFunction>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_vec_BT_u_on_slab(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& y_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            const MFunction& M,
            double zero_tol = 1.0e-15) const
        {
            ReconstructionOperator op(*x_space_, slab_space_);
            op.set_slab_reconstruction_operator_mode(
                slab_reconstruction_operator_mode_);
            op.template assemble_vec_BT_u_direct<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                x_cache,
                y_cache,
                ancestor_cache,
                M,
                zero_tol);
        }

        template<int QSpace, int QTime, class XFunctionType, class MFunction>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_vec_BT_u_on_slab_via_matrix_debug(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& y_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            const MFunction& M,
            double zero_tol = 1.0e-15) const
        {
            const auto& slab = slab_space_.slab(slab_id);
            SparseMatrixType B_slab;
            finite_element::assembly::assemble_mat_B_on_time_slab<
                QSpace,
                QTime,
                Backend>(
                    B_slab,
                    *x_space_,
                    slab,
                    x_cache,
                    y_cache,
                    ancestor_cache,
                    M,
                    zero_tol);

            bt_u_slab = la::ops::transpose_matvec(B_slab, u_delta.true_coefficients());
        }

        template<int QSpace, int QTime, class XFunctionType>
        void assemble_vec_BT_u_on_slab_via_matrix_debug(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& y_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            double zero_tol = 1.0e-15) const
        {
            assemble_vec_BT_u_on_slab_via_matrix_debug<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                x_cache,
                y_cache,
                ancestor_cache,
                coefficients::IdentityDiffusion<GT::dim_space_v>{},
                zero_tol);
        }

        template<int QSpace, int QTime, class XFunctionType, class MFunction>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_vec_BT_u_on_slab_via_matrix_debug(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            const MFunction& M,
            double zero_tol = 1.0e-15) const
        {
            const auto& slab = slab_space_.slab(slab_id);
            const auto& y_slab_space = slab.fespace_ref();

            finite_element::assembly::detail::AssemblySpaceCache<XSpace>
                x_cache(*x_space_);
            finite_element::assembly::detail::
                AssemblySpaceCache<LocalSlabSpaceType>
                    y_cache(y_slab_space);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>
                ancestor_cache(*x_space_);

            assemble_vec_BT_u_on_slab_via_matrix_debug<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                x_cache,
                y_cache,
                ancestor_cache,
                M,
                zero_tol);
        }

        template<int QSpace, int QTime, class XFunctionType>
        void assemble_vec_BT_u_on_slab_via_matrix_debug(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            double zero_tol = 1.0e-15) const
        {
            assemble_vec_BT_u_on_slab_via_matrix_debug<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                coefficients::IdentityDiffusion<GT::dim_space_v>{},
                zero_tol);
        }

        template<int QSpace, int QTime, class XFunctionType>
        void assemble_vec_BT_u_on_slab(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& y_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            double zero_tol = 1.0e-15) const
        {
            assemble_vec_BT_u_on_slab<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                x_cache,
                y_cache,
                ancestor_cache,
                coefficients::IdentityDiffusion<GT::dim_space_v>{},
                zero_tol);
        }

        template<int QSpace, int QTime, class XFunctionType, class MFunction>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_vec_BT_u_on_slab(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            const MFunction& M,
            double zero_tol = 1.0e-15) const
        {
            const auto& slab = slab_space_.slab(slab_id);
            const auto& y_slab_space = slab.fespace_ref();

            finite_element::assembly::detail::AssemblySpaceCache<XSpace> x_cache(*x_space_);
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType> y_cache(y_slab_space);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace> ancestor_cache(*x_space_);

            assemble_vec_BT_u_on_slab<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                x_cache,
                y_cache,
                ancestor_cache,
                M,
                zero_tol);
        }

        template<int QSpace, int QTime, class XFunctionType>
        void assemble_vec_BT_u_on_slab(
            VectorType& bt_u_slab,
            int slab_id,
            const XFunctionType& u_delta,
            double zero_tol = 1.0e-15) const
        {
            assemble_vec_BT_u_on_slab<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                coefficients::IdentityDiffusion<GT::dim_space_v>{},
                zero_tol);
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_rhs_f_minus_BT_u_on_slab(
            VectorType& rhs_slab,
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& y_cache,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            double zero_tol = 1.0e-15) const
        {
            VectorType f_slab;
            VectorType bt_u_slab;

            assemble_vec_f_on_slab<QSpace, QTime>(
                f_slab, slab_id, ell, y_cache, zero_tol);

            assemble_vec_BT_u_on_slab<QSpace, QTime>(
                bt_u_slab,
                slab_id,
                u_delta,
                x_cache,
                y_cache,
                ancestor_cache,
                M,
                zero_tol);

            if (f_slab.size() != bt_u_slab.size())
                throw std::runtime_error(
                    "TimeSlabReconstruction::assemble_rhs_f_minus_BT_u_on_slab: size mismatch.");

            rhs_slab.resize(f_slab.size());
            for (int i = 0; i < rhs_slab.size(); ++i)
                rhs_slab[i] = f_slab[i] - bt_u_slab[i];
        }

        template<int QSpace, int QTime, class EllFunction, class XFunctionType>
        void assemble_rhs_f_minus_BT_u_on_slab(
            VectorType& rhs_slab,
            int slab_id,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType>& y_cache,
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>& x_cache,
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>& ancestor_cache,
            double zero_tol = 1.0e-15) const
        {
            assemble_rhs_f_minus_BT_u_on_slab<
                QSpace,
                QTime>(
                    rhs_slab,
                    slab_id,
                    coefficients::IdentityDiffusion<GT::dim_space_v>{},
                    ell,
                    u_delta,
                    y_cache,
                    x_cache,
                    ancestor_cache,
                    zero_tol);
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType>
        requires (!std::convertible_to<MFunction, double>)
        void assemble_rhs_f_minus_BT_u_on_slab(
            VectorType& rhs_slab,
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            double zero_tol = 1.0e-15) const
        {
            const auto& y_slab_space = slab_space_.slab(slab_id).fespace_ref();

            finite_element::assembly::detail::AssemblySpaceCache<LocalSlabSpaceType> y_cache(y_slab_space);
            finite_element::assembly::detail::AssemblySpaceCache<XSpace> x_cache(*x_space_);
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace> ancestor_cache(*x_space_);

            assemble_rhs_f_minus_BT_u_on_slab<QSpace, QTime>(
                rhs_slab,
                slab_id,
                M,
                ell,
                u_delta,
                y_cache,
                x_cache,
                ancestor_cache,
                zero_tol);
        }

        template<int QSpace, int QTime, class EllFunction, class XFunctionType>
        void assemble_rhs_f_minus_BT_u_on_slab(
            VectorType& rhs_slab,
            int slab_id,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            double zero_tol = 1.0e-15) const
        {
            assemble_rhs_f_minus_BT_u_on_slab<
                QSpace,
                QTime>(
                    rhs_slab,
                    slab_id,
                    coefficients::IdentityDiffusion<GT::dim_space_v>{},
                    ell,
                    u_delta,
                    zero_tol);
        }

        template<int QSpace, int QTime, class MFunction, class EllFunction, class XFunctionType>
        void solve_slab(
            int slab_id,
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15)
        {
            ensure_initialized_();

            ReconstructionOperator op(*x_space_, slab_space_);
            op.set_slab_reconstruction_operator_mode(
                slab_reconstruction_operator_mode_);
            auto diagnostics =
                op.template solve_slab<QSpace, QTime>(
                    slab_id,
                    M,
                    ell,
                    u_delta,
                    solver,
                    options,
                    reconstructed_->slab_function(slab_id),
                    zero_tol);
            record_slab_diagnostics_(std::move(diagnostics));
        }

        template<int QSpace, int QTime, class MFunction, class EllFunction, class XFunctionType>
        void solve_all_slabs(
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            SolverType& solver,
            const la::concepts::SolverOptions& options = {},
            double zero_tol = 1.0e-15)
        {
            ensure_initialized_();

            const int n_slabs = slab_space_.n_slabs();
            last_solve_all_slabs_used_openmp_ = false;
            last_slab_diagnostics_.clear();
            const int effective_slab_reconstruction_max_threads =
                GT::dim_space_v == 2 ? slab_reconstruction_max_threads_ : 0;
            core::ScopedOpenMPMaxThreads thread_scope(
                effective_slab_reconstruction_max_threads);
            last_configured_max_threads_ =
                effective_slab_reconstruction_max_threads;
            last_effective_max_threads_ =
                thread_scope.effective_max_threads();
            last_selected_threads_ = 1;
            last_memory_budget_bytes_ =
                slab_reconstruction_memory_budget_mb_ * 1024.0 * 1024.0;
            last_estimated_per_thread_cache_bytes_ = 0.0;

            solve_all_slabs_with_shared_x_caches_<QSpace, QTime>(
                M,
                ell,
                u_delta,
                solver,
                options,
                zero_tol);
        }

    private:
        using Clock = std::chrono::steady_clock;

        [[nodiscard]] static double seconds_(
            const Clock::time_point& begin,
            const Clock::time_point& end)
        {
            return std::chrono::duration<double>(end - begin).count();
        }

        template<
            int QSpace,
            int QTime,
            class MFunction,
            class EllFunction,
            class XFunctionType>
        void solve_all_slabs_with_shared_x_caches_(
            const MFunction& M,
            const EllFunction& ell,
            const XFunctionType& u_delta,
            SolverType& solver,
            const la::concepts::SolverOptions& options,
            double zero_tol)
        {
            const int n_slabs = slab_space_.n_slabs();
            ReconstructionOperator op(*x_space_, slab_space_);
            op.set_slab_reconstruction_operator_mode(
                slab_reconstruction_operator_mode_);

            std::size_t estimated_per_thread_cache_bytes = 0u;
            if (last_memory_budget_bytes_ > 0.0)
            {
                finite_element::assembly::detail::AssemblySpaceCache<XSpace>
                    estimate_x_cache(*x_space_);
                finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>
                    estimate_ancestor_cache(*x_space_);
                estimated_per_thread_cache_bytes =
                    estimate_x_cache.estimated_live_bytes() +
                    estimate_ancestor_cache.estimated_live_bytes();
                last_estimated_per_thread_cache_bytes_ =
                    static_cast<double>(estimated_per_thread_cache_bytes);
            }

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
            if constexpr (std::default_initializable<SolverType>)
            {
                int n_threads =
                    finite_element::assembly::detail::
                        recommended_openmp_threads_for_slab_solves(n_slabs);
                if (last_memory_budget_bytes_ > 0.0 &&
                    estimated_per_thread_cache_bytes > 0u)
                {
                    const int budget_threads =
                        static_cast<int>(
                            last_memory_budget_bytes_ /
                            static_cast<double>(
                                estimated_per_thread_cache_bytes));
                    if (budget_threads < n_threads)
                        n_threads = budget_threads > 0 ? budget_threads : 1;
                    if (n_threads < 1)
                        n_threads = 1;
                }
                last_selected_threads_ = n_threads;
                if (n_threads > 1)
                {
                    last_solve_all_slabs_used_openmp_ = true;
                    std::exception_ptr error;
                    std::size_t observed_per_thread_cache_bytes =
                        estimated_per_thread_cache_bytes;

#pragma omp parallel num_threads(n_threads)
                    {
                        try
                        {
                            SolverType thread_solver;

                            const auto t_x0 = Clock::now();
                            finite_element::assembly::detail::
                                AssemblySpaceCache<XSpace>
                                    x_cache(*x_space_);
                            const auto t_x1 = Clock::now();

                            const auto t_ancestor0 = Clock::now();
                            finite_element::assembly::detail::
                                SourceActiveAncestorCache<XSpace>
                                    ancestor_cache(*x_space_);
                            const auto t_ancestor1 = Clock::now();

                            const double x_cache_seconds =
                                seconds_(t_x0, t_x1);
                            const double ancestor_cache_seconds =
                                seconds_(t_ancestor0, t_ancestor1);
                            const std::size_t shared_cache_bytes =
                                x_cache.estimated_live_bytes() +
                                ancestor_cache.estimated_live_bytes();
#pragma omp critical(adap_parabolic_fem_slab_reconstruction_cache_bytes)
                            {
                                if (observed_per_thread_cache_bytes == 0u)
                                {
                                    observed_per_thread_cache_bytes =
                                        shared_cache_bytes;
                                }
                            }
                            bool record_thread_cache = true;

#pragma omp for schedule(static)
                            for (int k = 0; k < n_slabs; ++k)
                            {
                                auto diagnostics =
                                    op.template
                                    solve_slab_with_shared_caches<
                                        QSpace,
                                        QTime>(
                                        k,
                                        M,
                                        ell,
                                        u_delta,
                                        thread_solver,
                                        options,
                                        reconstructed_->slab_function(k),
                                        x_cache,
                                        ancestor_cache,
                                        record_thread_cache
                                            ? x_cache_seconds
                                            : 0.0,
                                        record_thread_cache
                                            ? ancestor_cache_seconds
                                            : 0.0,
                                        shared_cache_bytes,
                                        zero_tol);
                                record_thread_cache = false;
                                record_slab_diagnostics_(std::move(diagnostics));
                            }
                        }
                        catch (...)
                        {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                            {
                                if (!error)
                                    error = std::current_exception();
                            }
                        }
                    }

                    finite_element::assembly::detail::
                        rethrow_parallel_exception(error);
                    if (last_estimated_per_thread_cache_bytes_ == 0.0)
                    {
                        last_estimated_per_thread_cache_bytes_ =
                            static_cast<double>(
                                observed_per_thread_cache_bytes);
                    }
                    return;
                }
            }
#endif

            const auto t_x0 = Clock::now();
            finite_element::assembly::detail::AssemblySpaceCache<XSpace>
                x_cache(*x_space_);
            const auto t_x1 = Clock::now();

            const auto t_ancestor0 = Clock::now();
            finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>
                ancestor_cache(*x_space_);
            const auto t_ancestor1 = Clock::now();

            const double x_cache_seconds = seconds_(t_x0, t_x1);
            const double ancestor_cache_seconds =
                seconds_(t_ancestor0, t_ancestor1);
            const std::size_t shared_cache_bytes =
                x_cache.estimated_live_bytes() +
                ancestor_cache.estimated_live_bytes();
            if (last_estimated_per_thread_cache_bytes_ == 0.0)
            {
                last_estimated_per_thread_cache_bytes_ =
                    static_cast<double>(shared_cache_bytes);
            }
            last_selected_threads_ = 1;
            bool record_shared_cache = true;

            for (int k = 0; k < n_slabs; ++k)
            {
                auto diagnostics =
                    op.template solve_slab_with_shared_caches<
                        QSpace,
                        QTime>(
                        k,
                        M,
                        ell,
                        u_delta,
                        solver,
                        options,
                        reconstructed_->slab_function(k),
                        x_cache,
                        ancestor_cache,
                        record_shared_cache ? x_cache_seconds : 0.0,
                        record_shared_cache ? ancestor_cache_seconds : 0.0,
                        shared_cache_bytes,
                        zero_tol);
                record_shared_cache = false;
                record_slab_diagnostics_(std::move(diagnostics));
            }
        }

        void ensure_initialized_() const
        {
            if (!reconstructed_.has_value())
                throw std::runtime_error(
                    "TimeSlabReconstruction: call initialize() first.");
        }

        const SourceYSpace* source_y_space_ = nullptr;
        const XSpace* x_space_              = nullptr;

        SlabSpaceType slab_space_;
        std::optional<ReconstructedType> reconstructed_{};
        int slab_reconstruction_max_threads_ = 0;
        double slab_reconstruction_memory_budget_mb_ = 0.0;
        std::string slab_reconstruction_operator_mode_ = "auto";
        bool last_solve_all_slabs_used_openmp_ = false;
        int last_configured_max_threads_ = 0;
        int last_effective_max_threads_ = 1;
        int last_selected_threads_ = 1;
        double last_memory_budget_bytes_ = 0.0;
        double last_estimated_per_thread_cache_bytes_ = 0.0;
        std::vector<SlabDiagnostics> last_slab_diagnostics_{};

        void record_slab_diagnostics_(SlabDiagnostics diagnostics)
        {
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
#pragma omp critical(adap_parabolic_fem_slab_reconstruction_diagnostics)
#endif
            {
                last_slab_diagnostics_.push_back(std::move(diagnostics));
            }
        }
    };
}

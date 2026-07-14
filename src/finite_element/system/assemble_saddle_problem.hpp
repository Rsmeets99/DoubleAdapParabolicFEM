#pragma once

#include <utility>

#include "../assembly/detail/active_cell_locator.hpp"
#include "../assembly/detail/assembly_diagnostics.hpp"
#include "../assembly/detail/assembly_space_cache.hpp"
#include "../assembly/detail/active_cell_locator_time_slab.hpp"
#include "../assembly/main_system/mat_A.hpp"
#include "../assembly/main_system/mat_B.hpp"
#include "../assembly/main_system/mat_C.hpp"
#include "../assembly/main_system/dense_local_main_blocks_2d.hpp"
#include "../assembly/main_system/main_assembly_quadrature_cache_2d.hpp"
#include "../assembly/main_system/vec_f.hpp"
#include "../assembly/main_system/vec_g.hpp"
#include "../detail/timing.hpp"

#include "../../linear_algebra/system/saddle_point_system.hpp"

#ifndef APF_FORCE_REFERENCE_MAIN_ASSEMBLY_2D
#define APF_FORCE_REFERENCE_MAIN_ASSEMBLY_2D 0
#endif

namespace finite_element::system
{
    template<class SpaceType>
    inline constexpr bool use_fixed_degree_main_kernel_2d_v =
        SpaceType::GT::dim_space_v == 2 &&
        SpaceType::FETraitsType::p_space_v >= 1 &&
        SpaceType::FETraitsType::p_space_v <= 4 &&
        SpaceType::FETraitsType::p_time_v >= 1 &&
        SpaceType::FETraitsType::p_time_v <= 4 &&
        APF_FORCE_REFERENCE_MAIN_ASSEMBLY_2D == 0;

    template<class Backend>
    class MainSystemBlocks
    {
    public:
        using SaddleBlocks = la::saddle::SaddlePointBlocks<Backend>;
        using SparseMatrix = typename Backend::SparseMatrix;
        using Vector = typename Backend::Vector;

        MainSystemBlocks() = default;

        explicit MainSystemBlocks(SaddleBlocks blocks)
            : saddle_blocks_(std::move(blocks))
        {}

        // Parabolic main-system block convention:
        //
        //   [ A_y  B^T      ] [lambda] = [f]
        //   [ B    C_signed ] [u     ]   [g]
        //
        // B is stored in its lower-left u-lambda orientation.  C_signed is the
        // already signed bottom-right block assembled from -gamma_0^T gamma_0;
        // future positive Schur-complement code should use C_pos = -C_signed.
        [[nodiscard]] const SparseMatrix& A_y() const noexcept
        {
            return saddle_blocks_.top_left_A();
        }

        [[nodiscard]] const SparseMatrix& lower_left_B() const noexcept
        {
            return saddle_blocks_.lower_left_B();
        }

        [[nodiscard]] const SparseMatrix& C_signed() const noexcept
        {
            return saddle_blocks_.signed_bottom_right_C();
        }

        [[nodiscard]] const Vector& f() const noexcept
        {
            return saddle_blocks_.lambda_rhs();
        }

        [[nodiscard]] const Vector& g() const noexcept
        {
            return saddle_blocks_.u_rhs();
        }

        [[nodiscard]] int n_lambda() const noexcept
        {
            return saddle_blocks_.n_lambda;
        }

        [[nodiscard]] int n_u() const noexcept
        {
            return saddle_blocks_.n_u;
        }

        [[nodiscard]] const SaddleBlocks& saddle_point_blocks() const noexcept
        {
            return saddle_blocks_;
        }

        [[nodiscard]] SaddleBlocks release_saddle_point_blocks() &&
        {
            return std::move(saddle_blocks_);
        }

        void validate() const
        {
            saddle_blocks_.validate();
        }

        [[nodiscard]] la::linear::LinearSystem<Backend> make_full_system(
            bool memory_bounded_composition = false) const &
        {
            return saddle_blocks_.make_full_system(memory_bounded_composition);
        }

        [[nodiscard]] la::linear::LinearSystem<Backend> make_full_system(
            bool memory_bounded_composition = false) &&
        {
            return std::move(saddle_blocks_)
                .make_full_system(memory_bounded_composition);
        }

    private:
        SaddleBlocks saddle_blocks_{};
    };

    template<
        int QSpace,
        int QTime,
        class Backend,
        class YSpaceType,
        class XSpaceType,
        class ExampleType>
    [[nodiscard]] MainSystemBlocks<Backend> assemble_main_system_blocks(
        const YSpaceType& y_space,
        const XSpaceType& x_space,
        const ExampleType& example,
        double zero_tol = 1e-15,
        double g_scale = -1.0,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        auto assembly_total_timer = timing.scoped("assembly.total_seconds");

        using SparseMatrix = typename Backend::SparseMatrix;
        using Vector       = typename Backend::Vector;

        SparseMatrix A_y;
        SparseMatrix lower_left_B;
        SparseMatrix C_signed;
        Vector f;
        Vector g;

        finite_element::assembly::detail::AssemblySpaceCache<YSpaceType> y_cache(y_space);
        finite_element::assembly::detail::AssemblySpaceCache<XSpaceType> x_cache(x_space);
        finite_element::assembly::detail::ActiveAncestorCache<XSpaceType> x_ancestor_cache(x_space);

        constexpr bool use_dense_y_blocks =
            use_fixed_degree_main_kernel_2d_v<YSpaceType>;
        if constexpr (use_dense_y_blocks)
        {
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                A_diagnostics;
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                B_diagnostics;
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                f_diagnostics;
            {
                auto timer =
                    timing.scoped("main_system.assemble_y_dense_local_blocks");
                auto numeric_timer =
                    timing.scoped("assembly.numeric_fill_seconds");
                finite_element::assembly::assemble_main_y_dense_local_blocks_2d<
                    QSpace,
                    QTime,
                    Backend>(
                        A_y,
                        lower_left_B,
                        f,
                        x_space,
                        y_space,
                        x_cache,
                        y_cache,
                        x_ancestor_cache,
                        example.M,
                        example.ell,
                        zero_tol,
                        &A_diagnostics,
                        &B_diagnostics,
                        &f_diagnostics,
                        timing);
                numeric_timer.stop();
            }
            timing.add("main_system.assemble_A_y", 0.0);
            finite_element::assembly::detail::record_assembly_diagnostics(
                timing,
                "main_system.assemble_A_y",
                A_diagnostics);
            timing.add("main_system.assemble_B", 0.0);
            finite_element::assembly::detail::record_assembly_diagnostics(
                timing,
                "main_system.assemble_B",
                B_diagnostics);
            timing.add("main_system.assemble_f", 0.0);
            finite_element::assembly::detail::record_assembly_diagnostics(
                timing,
                "main_system.assemble_f",
                f_diagnostics);
        }
        else
        {
            {
                finite_element::assembly::detail::AssemblyKernelDiagnostics
                    diagnostics;
                auto timer = timing.scoped("main_system.assemble_A_y");
                auto numeric_timer =
                    timing.scoped("assembly.numeric_fill_seconds");
                auto local_timer =
                    timing.scoped("assembly.local_kernel_seconds");
                finite_element::assembly::assemble_mat_A<QSpace, QTime, Backend>(
                    A_y,
                    y_space,
                    example.M,
                    y_cache,
                    zero_tol,
                    &diagnostics);
                local_timer.stop();
                numeric_timer.stop();
                timer.stop();
                finite_element::assembly::detail::record_assembly_diagnostics(
                    timing,
                    "main_system.assemble_A_y",
                    diagnostics);
            }

            {
                finite_element::assembly::detail::AssemblyKernelDiagnostics
                    diagnostics;
                auto timer = timing.scoped("main_system.assemble_B");
                auto numeric_timer =
                    timing.scoped("assembly.numeric_fill_seconds");
                auto local_timer =
                    timing.scoped("assembly.local_kernel_seconds");
                finite_element::assembly::assemble_mat_B<QSpace, QTime, Backend>(
                    lower_left_B,
                    x_space,
                    y_space,
                    x_cache,
                    y_cache,
                    x_ancestor_cache,
                    example.M,
                    zero_tol,
                    &diagnostics);
                local_timer.stop();
                numeric_timer.stop();
                timer.stop();
                finite_element::assembly::detail::record_assembly_diagnostics(
                    timing,
                    "main_system.assemble_B",
                    diagnostics);
            }

            {
                finite_element::assembly::detail::AssemblyKernelDiagnostics
                    diagnostics;
                auto timer = timing.scoped("main_system.assemble_f");
                auto rhs_timer = timing.scoped("assembly.rhs_seconds");
                finite_element::assembly::assemble_vec_f<QSpace, QTime>(
                    f,
                    y_space,
                    example.ell,
                    y_cache,
                    zero_tol,
                    &diagnostics);
                rhs_timer.stop();
                timer.stop();
                finite_element::assembly::detail::record_assembly_diagnostics(
                    timing,
                    "main_system.assemble_f",
                    diagnostics);
            }
        }

        constexpr bool use_dense_trace_blocks =
            use_fixed_degree_main_kernel_2d_v<XSpaceType>;
        if constexpr (use_dense_trace_blocks)
        {
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                C_diagnostics;
            finite_element::assembly::detail::AssemblyKernelDiagnostics
                g_diagnostics;
            {
                auto timer =
                    timing.scoped("main_system.assemble_trace_dense_local_blocks");
                auto numeric_timer =
                    timing.scoped("assembly.numeric_fill_seconds");
                finite_element::assembly::
                    assemble_main_trace_dense_local_blocks_2d<
                        QSpace,
                        QTime,
                        Backend>(
                        C_signed,
                        g,
                        x_space,
                        x_cache,
                        example.u0,
                        zero_tol,
                        g_scale,
                        &C_diagnostics,
                        &g_diagnostics,
                        timing);
                numeric_timer.stop();
            }
            timing.add("main_system.assemble_C_signed", 0.0);
            finite_element::assembly::detail::record_assembly_diagnostics(
                timing,
                "main_system.assemble_C_signed",
                C_diagnostics);
            timing.add("main_system.assemble_g", 0.0);
            finite_element::assembly::detail::record_assembly_diagnostics(
                timing,
                "main_system.assemble_g",
                g_diagnostics);
        }
        else
        {
            {
                auto timer = timing.scoped("main_system.assemble_C_signed");
                auto numeric_timer =
                    timing.scoped("assembly.numeric_fill_seconds");
                auto local_timer =
                    timing.scoped("assembly.local_kernel_seconds");
                finite_element::assembly::assemble_mat_C<QSpace, QTime, Backend>(
                    C_signed, x_space, x_cache, zero_tol);
                local_timer.stop();
                numeric_timer.stop();
            }

            {
                finite_element::assembly::detail::AssemblyKernelDiagnostics
                    diagnostics;
                auto timer = timing.scoped("main_system.assemble_g");
                auto rhs_timer = timing.scoped("assembly.rhs_seconds");
                finite_element::assembly::assemble_vec_g<QSpace, QTime>(
                    g,
                    x_space,
                    example.u0,
                    x_cache,
                    zero_tol,
                    g_scale,
                    &diagnostics);
                rhs_timer.stop();
                timer.stop();
                finite_element::assembly::detail::record_assembly_diagnostics(
                    timing,
                    "main_system.assemble_g",
                    diagnostics);
            }
        }

        return MainSystemBlocks<Backend>(
            la::saddle::make_saddle_point_blocks<Backend>(
                std::move(A_y),
                std::move(lower_left_B),
                std::move(C_signed),
                std::move(f),
                std::move(g)));
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class YSpaceType,
        class XSpaceType,
        class ExampleType>
    [[nodiscard]] la::saddle::SaddlePointBlocks<Backend> assemble_saddle_problem(
        const YSpaceType& y_space,
        const XSpaceType& x_space,
        const ExampleType& example,
        double zero_tol = 1e-15,
        double g_scale = -1.0,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        auto blocks = assemble_main_system_blocks<
            QSpace,
            QTime,
            Backend>(
                y_space,
                x_space,
                example,
                zero_tol,
                g_scale,
                timing);
        return std::move(blocks).release_saddle_point_blocks();
    }
}

#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <exception>
#include <stdexcept>
#include <vector>

#include "../constrained_dofs.hpp"
#include "../scatter.hpp"
#include "../detail/active_cell_locator.hpp"
#include "../detail/assembly_diagnostics.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../detail/openmp_assembly.hpp"
#include "../detail/space_time_basis_tables.hpp"
#include "../detail/trace_geometry_utils.hpp"
#include "../detail/zero_local.hpp"
#include "../../detail/timing.hpp"
#include "../../basis/space_time_basis_selector.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../geometry/cell_geometry.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"
#include "linear_algebra/concepts/vector.hpp"

#ifndef ADAPPARABOLICFEM_ENABLE_SAME_CELL_PRETABULATED_X_GRADS_2D
#define ADAPPARABOLICFEM_ENABLE_SAME_CELL_PRETABULATED_X_GRADS_2D 1
#endif

namespace finite_element::assembly
{
    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType,
        class MFunction,
        class EllFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             la::concepts::VectorLike<typename Backend::Vector>
    void assemble_main_y_dense_local_blocks_2d(
        typename Backend::SparseMatrix& A_y,
        typename Backend::SparseMatrix& lower_left_B,
        typename Backend::Vector& f,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MFunction& M,
        const EllFunction& ell,
        double zero_tol = 1.0e-15,
        detail::AssemblyKernelDiagnostics* A_diagnostics = nullptr,
        detail::AssemblyKernelDiagnostics* B_diagnostics = nullptr,
        detail::AssemblyKernelDiagnostics* f_diagnostics = nullptr,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        using Clock = std::chrono::steady_clock;
        using Vector = typename Backend::Vector;
        using SparseBuilder = typename Backend::SparseBuilder;
        using XBasis =
            finite_element::basis::SpaceTimeBasis<
                typename XFESpaceType::GT,
                typename XFESpaceType::FETraitsType>;
        using XTables =
            detail::SpaceTimeBasisTables<
                typename XFESpaceType::GT,
                typename XFESpaceType::FETraitsType,
                QSpace,
                QTime>;
        using YTables =
            detail::SpaceTimeBasisTables<
                typename YFESpaceType::GT,
                typename YFESpaceType::FETraitsType,
                QSpace,
                QTime>;
        using XGeometry =
            finite_element::geometry::CellGeometry<XFESpaceType, 2>;
        using YGeometry =
            finite_element::geometry::CellGeometry<YFESpaceType, 2>;

        static_assert(
            XFESpaceType::GT::dim_space_v == 2 &&
            YFESpaceType::GT::dim_space_v == 2,
            "assemble_main_y_dense_local_blocks_2d is only for 2+1D.");

        constexpr int x_dofs_per_cell = XFESpaceType::FETraitsType::dofs_per_cell;
        constexpr int y_dofs_per_cell = YFESpaceType::FETraitsType::dofs_per_cell;
        constexpr int n_q = YTables::n_cell_q;

        const auto& x_dof_handler = x_space.dof_handler_ref();
        const auto& y_dof_handler = y_space.dof_handler_ref();
        const auto& y_active_cells = y_space.active_cells();
        const int n_active_cells = static_cast<int>(y_active_cells.size());

        const std::size_t reserve_A_entries =
            static_cast<std::size_t>(n_active_cells) *
            static_cast<std::size_t>(y_dofs_per_cell) *
            static_cast<std::size_t>(y_dofs_per_cell) * 4u;
        const std::size_t reserve_B_entries =
            static_cast<std::size_t>(n_active_cells) *
            static_cast<std::size_t>(x_dofs_per_cell) *
            static_cast<std::size_t>(y_dofs_per_cell) * 4u;

        double geometry_cache_seconds = 0.0;
        double cell_restriction_seconds = 0.0;
        double local_kernel_seconds = 0.0;
        double scatter_seconds = 0.0;
        double rhs_seconds = 0.0;
        double pattern_finalize_seconds = 0.0;
        std::atomic<std::size_t> same_cell_pretabulated_x_gradient_cells{0u};

        const auto add_elapsed =
            [](double& accumulator, Clock::time_point start)
            {
                accumulator +=
                    std::chrono::duration<double>(Clock::now() - start)
                        .count();
            };

        std::vector<int> x_cell_ids(static_cast<std::size_t>(n_active_cells), -1);
        for (int item_index = 0; item_index < n_active_cells; ++item_index)
        {
            const int y_cell_id = y_active_cells[static_cast<std::size_t>(item_index)];
            const int x_cell_id =
                detail::find_active_ancestor_cell(
                    ancestor_cache,
                    x_space,
                    y_space,
                    y_cell_id);
            x_cell_ids[static_cast<std::size_t>(item_index)] = x_cell_id;

            auto cache_start = Clock::now();
            (void)x_cache.dof_expansion(x_cell_id);
            add_elapsed(cell_restriction_seconds, cache_start);
            cache_start = Clock::now();
            (void)x_cache.geometry(x_cell_id);
            add_elapsed(geometry_cache_seconds, cache_start);
            cache_start = Clock::now();
            (void)y_cache.dof_expansion(y_cell_id);
            add_elapsed(cell_restriction_seconds, cache_start);
            cache_start = Clock::now();
            (void)y_cache.geometry(y_cell_id);
            add_elapsed(geometry_cache_seconds, cache_start);
        }

        SparseBuilder A_builder;
        SparseBuilder B_builder;
        f.resize(y_dof_handler.n_true_dofs());
        f.set_zero();

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            auto A_builders =
                detail::make_thread_local_sparse_builders<SparseBuilder>(
                    detail::openmp_thread_count(),
                    reserve_A_entries);
            auto B_builders =
                detail::make_thread_local_sparse_builders<SparseBuilder>(
                    detail::openmp_thread_count(),
                    reserve_B_entries);
            auto f_vectors =
                detail::make_thread_local_vectors<Vector>(
                    detail::openmp_thread_count(),
                    f.size());
            std::exception_ptr error;

#pragma omp parallel
            {
                try
                {
                    const int thread_id = omp_get_thread_num();
                    auto& local_A_builder =
                        A_builders[static_cast<std::size_t>(thread_id)];
                    auto& local_B_builder =
                        B_builders[static_cast<std::size_t>(thread_id)];
                    auto& local_f_vector =
                        f_vectors[static_cast<std::size_t>(thread_id)];

                    la::local::FixedLocalMatrix<
                        y_dofs_per_cell,
                        y_dofs_per_cell> local_A;
                    la::local::FixedLocalMatrix<
                        x_dofs_per_cell,
                        y_dofs_per_cell> local_B;
                    la::local::FixedLocalVector<y_dofs_per_cell> local_f;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int y_cell_id =
                            y_active_cells[static_cast<std::size_t>(item_index)];
                        const int x_cell_id =
                            x_cell_ids[static_cast<std::size_t>(item_index)];

                        const auto& x_geom = x_cache.geometry(x_cell_id);
                        const auto& y_geom = y_cache.geometry(y_cell_id);
                        const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                        const auto& y_expanded = y_cache.dof_expansion(y_cell_id);
                        const bool same_cell_pretabulated_x_grads =
                            ADAPPARABOLICFEM_ENABLE_SAME_CELL_PRETABULATED_X_GRADS_2D &&
                            x_cell_id == y_cell_id;
                        if (same_cell_pretabulated_x_grads)
                        {
                            same_cell_pretabulated_x_gradient_cells.fetch_add(
                                1u,
                                std::memory_order_relaxed);
                        }

                        detail::zero_local_matrix(local_A);
                        detail::zero_local_matrix(local_B);
                        detail::zero_local_vector(local_f);

                        for (int q = 0; q < n_q; ++q)
                        {
                            const auto& xi_y = YTables::cell_rule.points[q];
                            const double dmu =
                                YGeometry::jacobian_measure(y_geom) *
                                YTables::cell_rule.weights[q];
                            const auto& phi_vals = YTables::values_on_cell_qp(q);
                            const auto& phi_grads_ref =
                                YTables::gradients_on_cell_qp(q);

                            const auto x_q =
                                YGeometry::map_to_physical(y_geom, xi_y);
                            const auto M_q =
                                coefficients::evaluate_diffusion_tensor<
                                    MFunction,
                                    2>(
                                    M,
                                    x_q);
                            const double ell_q = static_cast<double>(ell(x_q));

                            std::array<
                                typename YGeometry::SpatialGradient,
                                y_dofs_per_cell> gradx_phi_on_q{};
                            std::array<
                                typename YGeometry::SpatialGradient,
                                y_dofs_per_cell> M_gradx_phi_on_q{};
                            for (int j = 0; j < y_dofs_per_cell; ++j)
                            {
                                gradx_phi_on_q[static_cast<std::size_t>(j)] =
                                    YGeometry::spatial_gradient(
                                        y_geom,
                                        phi_grads_ref[j]);
                                M_gradx_phi_on_q[
                                    static_cast<std::size_t>(j)] =
                                    coefficients::apply_validated_M<2>(
                                        M_q,
                                        gradx_phi_on_q[
                                            static_cast<std::size_t>(j)]);
                            }

                            std::array<double, x_dofs_per_cell> dt_psi_on_q{};
                            std::array<
                                typename XGeometry::SpatialGradient,
                                x_dofs_per_cell> gradx_psi_on_q{};
                            const auto fill_x_derivatives =
                                [&](const auto& psi_grads_ref)
                                {
                                    for (int i = 0; i < x_dofs_per_cell; ++i)
                                    {
                                        dt_psi_on_q[
                                            static_cast<std::size_t>(i)] =
                                            XGeometry::time_derivative(
                                                x_geom,
                                                psi_grads_ref[i]);
                                        gradx_psi_on_q[
                                            static_cast<std::size_t>(i)] =
                                            XGeometry::spatial_gradient(
                                                x_geom,
                                                psi_grads_ref[i]);
                                    }
                                };
                            if (same_cell_pretabulated_x_grads)
                            {
                                fill_x_derivatives(
                                    XTables::gradients_on_cell_qp(q));
                            }
                            else
                            {
                                const auto xi_x =
                                    XGeometry::physical_to_reference(
                                        x_geom,
                                        x_q);
                                const auto psi_grads_ref =
                                    XBasis::grad_all(xi_x);
                                fill_x_derivatives(psi_grads_ref);
                            }

                            for (int i = 0; i < y_dofs_per_cell; ++i)
                            {
                                const auto& grad_i =
                                    gradx_phi_on_q[static_cast<std::size_t>(i)];
                                local_f[i] += ell_q * phi_vals[i] * dmu;
                                for (int j = 0; j < y_dofs_per_cell; ++j)
                                {
                                    local_A(i, j) +=
                                        coefficients::dot<2>(
                                            grad_i,
                                            M_gradx_phi_on_q[
                                                static_cast<std::size_t>(j)]) *
                                        dmu;
                                }
                            }

                            for (int i = 0; i < x_dofs_per_cell; ++i)
                            {
                                const double dt_psi_i =
                                    dt_psi_on_q[static_cast<std::size_t>(i)];
                                const auto& gradx_psi_i =
                                    gradx_psi_on_q[static_cast<std::size_t>(i)];
                                for (int j = 0; j < y_dofs_per_cell; ++j)
                                {
                                    local_B(i, j) +=
                                        (dt_psi_i * phi_vals[j] +
                                         coefficients::dot<2>(
                                             gradx_psi_i,
                                             M_gradx_phi_on_q[
                                                 static_cast<std::size_t>(j)])) *
                                        dmu;
                                }
                            }
                        }

                        scatter_matrix(
                            local_A_builder,
                            local_A,
                            y_expanded,
                            y_expanded,
                            zero_tol);
                        scatter_matrix(
                            local_B_builder,
                            local_B,
                            x_expanded,
                            y_expanded,
                            zero_tol);
                        scatter_vector(
                            local_f_vector,
                            local_f,
                            y_expanded,
                            zero_tol);
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

            detail::rethrow_parallel_exception(error);
            A_builder = detail::merge_sparse_builders(A_builders);
            B_builder = detail::merge_sparse_builders(B_builders);
            detail::reduce_thread_local_vectors(f, f_vectors);
        }
        else
#endif
        {
            A_builder.reserve(reserve_A_entries);
            B_builder.reserve(reserve_B_entries);

            la::local::FixedLocalMatrix<
                y_dofs_per_cell,
                y_dofs_per_cell> local_A;
            la::local::FixedLocalMatrix<
                x_dofs_per_cell,
                y_dofs_per_cell> local_B;
            la::local::FixedLocalVector<y_dofs_per_cell> local_f;

            for (int item_index = 0; item_index < n_active_cells; ++item_index)
            {
                const int y_cell_id =
                    y_active_cells[static_cast<std::size_t>(item_index)];
                const int x_cell_id =
                    x_cell_ids[static_cast<std::size_t>(item_index)];

                auto cache_start = Clock::now();
                const auto& x_geom = x_cache.geometry(x_cell_id);
                add_elapsed(geometry_cache_seconds, cache_start);
                cache_start = Clock::now();
                const auto& y_geom = y_cache.geometry(y_cell_id);
                add_elapsed(geometry_cache_seconds, cache_start);
                cache_start = Clock::now();
                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                add_elapsed(cell_restriction_seconds, cache_start);
                cache_start = Clock::now();
                const auto& y_expanded = y_cache.dof_expansion(y_cell_id);
                add_elapsed(cell_restriction_seconds, cache_start);
                const bool same_cell_pretabulated_x_grads =
                    ADAPPARABOLICFEM_ENABLE_SAME_CELL_PRETABULATED_X_GRADS_2D &&
                    x_cell_id == y_cell_id;
                if (same_cell_pretabulated_x_grads)
                {
                    same_cell_pretabulated_x_gradient_cells.fetch_add(
                        1u,
                        std::memory_order_relaxed);
                }

                const auto kernel_start = Clock::now();
                detail::zero_local_matrix(local_A);
                detail::zero_local_matrix(local_B);
                detail::zero_local_vector(local_f);

                const double jac_y = YGeometry::jacobian_measure(y_geom);
                for (int q = 0; q < n_q; ++q)
                {
                    const auto& xi_y = YTables::cell_rule.points[q];
                    const double dmu = jac_y * YTables::cell_rule.weights[q];
                    const auto& phi_vals = YTables::values_on_cell_qp(q);
                    const auto& phi_grads_ref =
                        YTables::gradients_on_cell_qp(q);

                    const auto x_q =
                        YGeometry::map_to_physical(y_geom, xi_y);
                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<
                            MFunction,
                            2>(
                            M,
                            x_q);
                    const double ell_q = static_cast<double>(ell(x_q));

                    std::array<
                        typename YGeometry::SpatialGradient,
                        y_dofs_per_cell> gradx_phi_on_q{};
                    std::array<
                        typename YGeometry::SpatialGradient,
                        y_dofs_per_cell> M_gradx_phi_on_q{};
                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        gradx_phi_on_q[static_cast<std::size_t>(j)] =
                            YGeometry::spatial_gradient(
                                y_geom,
                                phi_grads_ref[j]);
                        M_gradx_phi_on_q[static_cast<std::size_t>(j)] =
                            coefficients::apply_validated_M<2>(
                                M_q,
                                gradx_phi_on_q[
                                    static_cast<std::size_t>(j)]);
                    }

                    std::array<double, x_dofs_per_cell> dt_psi_on_q{};
                    std::array<
                        typename XGeometry::SpatialGradient,
                        x_dofs_per_cell> gradx_psi_on_q{};
                    const auto fill_x_derivatives =
                        [&](const auto& psi_grads_ref)
                        {
                            for (int i = 0; i < x_dofs_per_cell; ++i)
                            {
                                dt_psi_on_q[static_cast<std::size_t>(i)] =
                                    XGeometry::time_derivative(
                                        x_geom,
                                        psi_grads_ref[i]);
                                gradx_psi_on_q[static_cast<std::size_t>(i)] =
                                    XGeometry::spatial_gradient(
                                        x_geom,
                                        psi_grads_ref[i]);
                            }
                        };
                    if (same_cell_pretabulated_x_grads)
                    {
                        fill_x_derivatives(XTables::gradients_on_cell_qp(q));
                    }
                    else
                    {
                        const auto xi_x =
                            XGeometry::physical_to_reference(x_geom, x_q);
                        const auto psi_grads_ref = XBasis::grad_all(xi_x);
                        fill_x_derivatives(psi_grads_ref);
                    }

                    for (int i = 0; i < y_dofs_per_cell; ++i)
                    {
                        const auto& grad_i =
                            gradx_phi_on_q[static_cast<std::size_t>(i)];
                        local_f[i] += ell_q * phi_vals[i] * dmu;
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local_A(i, j) +=
                                coefficients::dot<2>(
                                    grad_i,
                                    M_gradx_phi_on_q[
                                        static_cast<std::size_t>(j)]) *
                                dmu;
                        }
                    }

                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        const double dt_psi_i =
                            dt_psi_on_q[static_cast<std::size_t>(i)];
                        const auto& gradx_psi_i =
                            gradx_psi_on_q[static_cast<std::size_t>(i)];
                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local_B(i, j) +=
                                (dt_psi_i * phi_vals[j] +
                                 coefficients::dot<2>(
                                     gradx_psi_i,
                                     M_gradx_phi_on_q[
                                         static_cast<std::size_t>(j)])) *
                                dmu;
                        }
                    }
                }
                add_elapsed(local_kernel_seconds, kernel_start);

                const auto scatter_start = Clock::now();
                scatter_matrix(
                    A_builder,
                    local_A,
                    y_expanded,
                    y_expanded,
                    zero_tol);
                scatter_matrix(
                    B_builder,
                    local_B,
                    x_expanded,
                    y_expanded,
                    zero_tol);
                add_elapsed(scatter_seconds, scatter_start);
                const auto rhs_start = Clock::now();
                scatter_vector(f, local_f, y_expanded, zero_tol);
                add_elapsed(rhs_seconds, rhs_start);
            }
        }

        if (A_diagnostics != nullptr)
        {
            A_diagnostics->active_cells =
                static_cast<std::size_t>(n_active_cells);
            A_diagnostics->quadrature_points =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(n_q);
            A_diagnostics->gradient_evaluations =
                A_diagnostics->quadrature_points *
                static_cast<std::size_t>(y_dofs_per_cell);
            A_diagnostics->diffusion_tensor_evaluations =
                A_diagnostics->quadrature_points;
            A_diagnostics->sparse_triplets_emitted = A_builder.size();
            A_diagnostics->peak_triplet_bytes =
                A_builder.size() *
                detail::estimated_triplet_bytes<SparseBuilder>();
        }

        if (B_diagnostics != nullptr)
        {
            B_diagnostics->active_cells =
                static_cast<std::size_t>(n_active_cells);
            B_diagnostics->quadrature_points =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(n_q);
            B_diagnostics->scalar_basis_evaluations =
                B_diagnostics->quadrature_points *
                static_cast<std::size_t>(y_dofs_per_cell);
            B_diagnostics->gradient_evaluations =
                B_diagnostics->quadrature_points *
                static_cast<std::size_t>(
                    x_dofs_per_cell + y_dofs_per_cell);
            B_diagnostics->diffusion_tensor_evaluations =
                B_diagnostics->quadrature_points;
            B_diagnostics->sparse_triplets_emitted = B_builder.size();
            B_diagnostics->peak_triplet_bytes =
                B_builder.size() *
                detail::estimated_triplet_bytes<SparseBuilder>();
        }

        if (f_diagnostics != nullptr)
        {
            f_diagnostics->active_cells =
                static_cast<std::size_t>(n_active_cells);
            f_diagnostics->quadrature_points =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(n_q);
            f_diagnostics->scalar_basis_evaluations =
                f_diagnostics->quadrature_points *
                static_cast<std::size_t>(y_dofs_per_cell);
            f_diagnostics->source_evaluations =
                f_diagnostics->quadrature_points;
        }

        {
            const auto finalize_start = Clock::now();
            A_y.resize(y_dof_handler.n_true_dofs(), y_dof_handler.n_true_dofs());
            A_y.set_from_builder(A_builder);
            lower_left_B.resize(
                x_dof_handler.n_true_dofs(),
                y_dof_handler.n_true_dofs());
            lower_left_B.set_from_builder(B_builder);
            add_elapsed(pattern_finalize_seconds, finalize_start);
        }

        if (A_diagnostics != nullptr)
        {
            A_diagnostics->final_matrix_nonzeros = detail::count_nonzeros(A_y);
            A_diagnostics->peak_sparse_matrix_bytes =
                detail::estimate_compressed_sparse_matrix_bytes(
                    A_y.rows(),
                    A_y.cols(),
                    A_diagnostics->final_matrix_nonzeros);
        }

        if (B_diagnostics != nullptr)
        {
            B_diagnostics->final_matrix_nonzeros =
                detail::count_nonzeros(lower_left_B);
            B_diagnostics->peak_sparse_matrix_bytes =
                detail::estimate_compressed_sparse_matrix_bytes(
                    lower_left_B.rows(),
                    lower_left_B.cols(),
                    B_diagnostics->final_matrix_nonzeros);
        }

        timing.add("assembly.geometry_cache_seconds", geometry_cache_seconds);
        timing.add(
            "assembly.cell_restriction_seconds",
            cell_restriction_seconds);
        timing.add("assembly.local_kernel_seconds", local_kernel_seconds);
        timing.add("assembly.scatter_seconds", scatter_seconds);
        timing.add("assembly.rhs_seconds", rhs_seconds);
        timing.add(
            "assembly.pattern_finalize_seconds",
            pattern_finalize_seconds);
        timing.add(
            "assembly.same_cell_pretabulated_x_gradient_cells.count",
            static_cast<double>(
                same_cell_pretabulated_x_gradient_cells.load(
                    std::memory_order_relaxed)));
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class InitialValueFunction>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder> &&
             la::concepts::VectorLike<typename Backend::Vector>
    void assemble_main_trace_dense_local_blocks_2d(
        typename Backend::SparseMatrix& C_signed,
        typename Backend::Vector& g,
        const XFESpaceType& x_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        const InitialValueFunction& u0,
        double zero_tol = 1.0e-15,
        double g_scale = -1.0,
        detail::AssemblyKernelDiagnostics* C_diagnostics = nullptr,
        detail::AssemblyKernelDiagnostics* g_diagnostics = nullptr,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        using Clock = std::chrono::steady_clock;
        using SparseBuilder = typename Backend::SparseBuilder;
        using XTables =
            detail::SpaceTimeBasisTables<
                typename XFESpaceType::GT,
                typename XFESpaceType::FETraitsType,
                QSpace,
                QTime>;
        using XGeometry =
            finite_element::geometry::CellGeometry<XFESpaceType, 2>;

        static_assert(
            XFESpaceType::GT::dim_space_v == 2,
            "assemble_main_trace_dense_local_blocks_2d is only for 2+1D.");

        constexpr int x_dofs_per_cell = XFESpaceType::FETraitsType::dofs_per_cell;
        constexpr int n_q = XTables::n_bottom_q;

        const auto& mesh = x_space.mesh_ref();
        const auto& dof_handler = x_space.dof_handler_ref();
        const auto& active_cells = x_space.active_cells();
        const int n_active_cells = static_cast<int>(active_cells.size());

        const std::size_t reserve_entries =
            static_cast<std::size_t>(n_active_cells) *
            static_cast<std::size_t>(x_dofs_per_cell) *
            static_cast<std::size_t>(x_dofs_per_cell) * 4u;

        SparseBuilder C_builder;
        g.resize(dof_handler.n_true_dofs());
        g.set_zero();

        double geometry_cache_seconds = 0.0;
        double cell_restriction_seconds = 0.0;
        double local_kernel_seconds = 0.0;
        double scatter_seconds = 0.0;
        double rhs_seconds = 0.0;
        double pattern_finalize_seconds = 0.0;

        const auto add_elapsed =
            [](double& accumulator, Clock::time_point start)
            {
                accumulator +=
                    std::chrono::duration<double>(Clock::now() - start)
                        .count();
            };

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            for (const int cell_id : active_cells)
            {
                auto cache_start = Clock::now();
                (void)x_cache.dof_expansion(cell_id);
                add_elapsed(cell_restriction_seconds, cache_start);
                cache_start = Clock::now();
                (void)x_cache.geometry(cell_id);
                add_elapsed(geometry_cache_seconds, cache_start);
            }

            auto C_builders =
                detail::make_thread_local_sparse_builders<SparseBuilder>(
                    detail::openmp_thread_count(),
                    reserve_entries);
            auto g_vectors =
                detail::make_thread_local_vectors<typename Backend::Vector>(
                    detail::openmp_thread_count(),
                    g.size());
            std::exception_ptr error;

#pragma omp parallel
            {
                try
                {
                    const int thread_id = omp_get_thread_num();
                    auto& local_C_builder =
                        C_builders[static_cast<std::size_t>(thread_id)];
                    auto& local_g_vector =
                        g_vectors[static_cast<std::size_t>(thread_id)];
                    la::local::FixedLocalMatrix<
                        x_dofs_per_cell,
                        x_dofs_per_cell> local_C;
                    la::local::FixedLocalVector<x_dofs_per_cell> local_g;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int cell_id =
                            active_cells[static_cast<std::size_t>(item_index)];
                        const auto& cell = mesh.cell(cell_id);
                        if (!cell.temporal_boundary[0])
                            continue;

                        const auto& expanded = x_cache.dof_expansion(cell_id);
                        const auto& geom = x_cache.geometry(cell_id);
                        const double trace_jac =
                            detail::initial_trace_measure<XGeometry>(geom);

                        detail::zero_local_matrix(local_C);
                        detail::zero_local_vector(local_g);

                        for (int q = 0; q < n_q; ++q)
                        {
                            const auto& xi_bottom =
                                XTables::bottom_rule.points[q];
                            const double w_q =
                                XTables::bottom_rule.weights[q];
                            const auto& values =
                                XTables::values_on_bottom_qp(q);
                            const double dgamma = trace_jac * w_q;
                            const auto x_q_st =
                                detail::map_bottom_qp_to_physical<XGeometry>(
                                    geom,
                                    xi_bottom);
                            const auto x_q =
                                detail::
                                    spatial_argument_from_space_time_point<
                                        XGeometry>(x_q_st);
                            const double u0_q =
                                static_cast<double>(u0(x_q));

                            for (int i = 0; i < x_dofs_per_cell; ++i)
                            {
                                local_g[i] +=
                                    g_scale * u0_q * values[i] * dgamma;
                                for (int j = 0; j < x_dofs_per_cell; ++j)
                                {
                                    local_C(i, j) -=
                                        values[i] * values[j] * dgamma;
                                }
                            }
                        }

                        scatter_matrix(
                            local_C_builder,
                            local_C,
                            expanded,
                            expanded,
                            zero_tol);
                        scatter_vector(
                            local_g_vector,
                            local_g,
                            expanded,
                            zero_tol);
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

            detail::rethrow_parallel_exception(error);
            C_builder = detail::merge_sparse_builders(C_builders);
            detail::reduce_thread_local_vectors(g, g_vectors);
        }
        else
#endif
        {
            C_builder.reserve(reserve_entries);
            la::local::FixedLocalMatrix<
                x_dofs_per_cell,
                x_dofs_per_cell> local_C;
            la::local::FixedLocalVector<x_dofs_per_cell> local_g;

            for (const int cell_id : active_cells)
            {
                const auto& cell = mesh.cell(cell_id);
                if (!cell.temporal_boundary[0])
                    continue;

                auto cache_start = Clock::now();
                const auto& expanded = x_cache.dof_expansion(cell_id);
                add_elapsed(cell_restriction_seconds, cache_start);
                cache_start = Clock::now();
                const auto& geom = x_cache.geometry(cell_id);
                add_elapsed(geometry_cache_seconds, cache_start);
                const double trace_jac =
                    detail::initial_trace_measure<XGeometry>(geom);

                const auto kernel_start = Clock::now();
                detail::zero_local_matrix(local_C);
                detail::zero_local_vector(local_g);

                for (int q = 0; q < n_q; ++q)
                {
                    const auto& xi_bottom = XTables::bottom_rule.points[q];
                    const double w_q = XTables::bottom_rule.weights[q];
                    const auto& values = XTables::values_on_bottom_qp(q);
                    const double dgamma = trace_jac * w_q;
                    const auto x_q_st =
                        detail::map_bottom_qp_to_physical<XGeometry>(
                            geom,
                            xi_bottom);
                    const auto x_q =
                        detail::spatial_argument_from_space_time_point<
                            XGeometry>(x_q_st);
                    const double u0_q =
                        static_cast<double>(u0(x_q));

                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        local_g[i] +=
                            g_scale * u0_q * values[i] * dgamma;
                        for (int j = 0; j < x_dofs_per_cell; ++j)
                        {
                            local_C(i, j) -=
                                values[i] * values[j] * dgamma;
                        }
                    }
                }
                add_elapsed(local_kernel_seconds, kernel_start);

                const auto scatter_start = Clock::now();
                scatter_matrix(
                    C_builder,
                    local_C,
                    expanded,
                    expanded,
                    zero_tol);
                add_elapsed(scatter_seconds, scatter_start);
                const auto rhs_start = Clock::now();
                scatter_vector(g, local_g, expanded, zero_tol);
                add_elapsed(rhs_seconds, rhs_start);
            }
        }

        {
            const auto finalize_start = Clock::now();
            C_signed.resize(
                dof_handler.n_true_dofs(),
                dof_handler.n_true_dofs());
            C_signed.set_from_builder(C_builder);
            add_elapsed(pattern_finalize_seconds, finalize_start);
        }

        int n_bottom_cells = 0;
        if (C_diagnostics != nullptr || g_diagnostics != nullptr)
        {
            for (const int cell_id : active_cells)
            {
                if (mesh.cell(cell_id).temporal_boundary[0])
                    ++n_bottom_cells;
            }
        }

        if (C_diagnostics != nullptr)
        {
            C_diagnostics->active_cells =
                static_cast<std::size_t>(n_bottom_cells);
            C_diagnostics->quadrature_points =
                static_cast<std::size_t>(n_bottom_cells) *
                static_cast<std::size_t>(n_q);
            C_diagnostics->scalar_basis_evaluations =
                C_diagnostics->quadrature_points *
                static_cast<std::size_t>(x_dofs_per_cell);
            C_diagnostics->sparse_triplets_emitted = C_builder.size();
            C_diagnostics->peak_triplet_bytes =
                C_builder.size() *
                detail::estimated_triplet_bytes<SparseBuilder>();
            C_diagnostics->final_matrix_nonzeros =
                detail::count_nonzeros(C_signed);
            C_diagnostics->peak_sparse_matrix_bytes =
                detail::estimate_compressed_sparse_matrix_bytes(
                    C_signed.rows(),
                    C_signed.cols(),
                    C_diagnostics->final_matrix_nonzeros);
        }

        if (g_diagnostics != nullptr)
        {
            g_diagnostics->active_cells =
                static_cast<std::size_t>(n_bottom_cells);
            g_diagnostics->quadrature_points =
                static_cast<std::size_t>(n_bottom_cells) *
                static_cast<std::size_t>(n_q);
            g_diagnostics->scalar_basis_evaluations =
                g_diagnostics->quadrature_points *
                static_cast<std::size_t>(x_dofs_per_cell);
            g_diagnostics->source_evaluations =
                g_diagnostics->quadrature_points;
        }

        timing.add("assembly.geometry_cache_seconds", geometry_cache_seconds);
        timing.add(
            "assembly.cell_restriction_seconds",
            cell_restriction_seconds);
        timing.add("assembly.local_kernel_seconds", local_kernel_seconds);
        timing.add("assembly.scatter_seconds", scatter_seconds);
        timing.add("assembly.rhs_seconds", rhs_seconds);
        timing.add(
            "assembly.pattern_finalize_seconds",
            pattern_finalize_seconds);
    }
}

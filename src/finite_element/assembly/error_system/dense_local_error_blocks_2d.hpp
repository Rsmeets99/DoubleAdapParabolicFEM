#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <vector>
#include <stdexcept>

#include "../detail/active_cell_locator_time_slab.hpp"
#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "local_ab_element_cache_2d.hpp"
#include "local_rhs_state_cache_2d.hpp"
#include "mat_A_2d.hpp"
#include "mat_B_2d.hpp"
#include "vec_f_2d.hpp"
#include "vec_g_2d.hpp"

#include "linear_algebra/eigen_backend/backend.hpp"
#include "linear_algebra/operations/linalg_ops.hpp"
#include "linear_algebra/system/saddle_point_system.hpp"

namespace finite_element::assembly::error_system
{
    using DenseBackend = la::eigen::Backend;
    using DenseMatrix = typename DenseBackend::DenseMatrix;
    using DenseVector = typename DenseBackend::Vector;

    [[nodiscard]] inline bool dense_vector_is_finite(
        const DenseVector& vector) noexcept
    {
        for (int i = 0; i < vector.size(); ++i)
        {
            if (!std::isfinite(vector[i]))
                return false;
        }
        return true;
    }

    [[nodiscard]] inline double dense_product_inf_norm(
        const DenseMatrix& A,
        const DenseMatrix& B)
    {
        if (A.cols() != B.rows())
        {
            throw std::runtime_error(
                "dense_product_inf_norm: dimension mismatch.");
        }

        double norm = 0.0;
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < B.cols(); ++j)
            {
                double value = 0.0;
                for (int k = 0; k < A.cols(); ++k)
                    value += A(i, k) * B(k, j);
                norm = std::max(norm, std::abs(value));
            }
        }
        return norm;
    }

    struct DenseLocalErrorBlocks
    {
        DenseMatrix A{};
        DenseMatrix B{};
        DenseMatrix C{};
        DenseVector f{};
        DenseVector g{};
        int n_lambda = 0;
        int n_u = 0;

        DenseLocalErrorBlocks() = default;

        DenseLocalErrorBlocks(int n_lambda_in, int n_u_in)
        {
            resize(n_lambda_in, n_u_in);
        }

        void resize(int n_lambda_in, int n_u_in)
        {
            if (n_lambda_in < 0 || n_u_in < 0)
            {
                throw std::runtime_error(
                    "DenseLocalErrorBlocks: negative dimension.");
            }

            n_lambda = n_lambda_in;
            n_u = n_u_in;
            A.set_zero(n_lambda, n_lambda);
            B.set_zero(n_u, n_lambda);
            C.set_zero(n_u, n_u);
            f.resize(n_lambda);
            f.set_zero();
            g.resize(n_u);
            g.set_zero();
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            const auto vector_bytes =
                [](const DenseVector& vector) noexcept
            {
                return static_cast<std::size_t>(vector.size()) *
                       sizeof(double);
            };

            return A.estimated_memory_bytes() +
                   B.estimated_memory_bytes() +
                   C.estimated_memory_bytes() +
                   vector_bytes(f) + vector_bytes(g);
        }
    };

    struct DenseLocalErrorConstraintMatrix2D
    {
        DenseMatrix matrix{};
        int n_lambda = 0;
        int n_u = 0;
        int n_constraints = 0;

        [[nodiscard]] int system_size() const noexcept
        {
            return n_lambda + n_u + n_constraints;
        }
    };

    template<class Backend>
    struct DenseLocalErrorExplicitSolveWorkspace2D
    {
        DenseLocalErrorConstraintMatrix2D system{};
        DenseVector rhs{};
        DenseVector solution{};
        typename Backend::DenseSolver solver{};

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return system.matrix.estimated_memory_bytes() +
                   static_cast<std::size_t>(rhs.size()) * sizeof(double) +
                   static_cast<std::size_t>(solution.size()) * sizeof(double);
        }
    };

    template<class LocalMatrixType, class RTFluxSpaceType>
    void scatter_rt_local_matrix_dense_2d(
        DenseMatrix& out,
        const LocalMatrixType& local,
        const RTFluxSpaceType& space,
        int patch_cell_index,
        double zero_tol = 1.0e-15)
    {
        if (local.rows != RTFluxSpaceType::local_dofs_v ||
            local.cols != RTFluxSpaceType::local_dofs_v)
        {
            throw std::runtime_error(
                "scatter_rt_local_matrix_dense_2d: local matrix has unexpected dimensions.");
        }

        const auto& map = space.cell_dof_map(patch_cell_index);
        for (int i = 0; i < RTFluxSpaceType::local_dofs_v; ++i)
        {
            const auto& I = map[static_cast<std::size_t>(i)];
            if (I.patch_dof_id < 0)
                continue;

            for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
            {
                const auto& J = map[static_cast<std::size_t>(j)];
                if (J.patch_dof_id < 0)
                    continue;

                const double value =
                    static_cast<double>(I.orientation_sign) *
                    static_cast<double>(J.orientation_sign) *
                    local(i, j);
                if (std::abs(value) > zero_tol)
                    out(I.patch_dof_id, J.patch_dof_id) += value;
            }
        }
    }

    template<class LocalMatrixType, class PatchScalarSpaceType, class RTFluxSpaceType>
    void scatter_divergence_local_matrix_dense_2d(
        DenseMatrix& out,
        const LocalMatrixType& local,
        const PatchScalarSpaceType& scalar_space,
        const RTFluxSpaceType& rt_space,
        int patch_cell_index,
        double zero_tol = 1.0e-15)
    {
        if (local.rows != PatchScalarSpaceType::local_dofs_v ||
            local.cols != RTFluxSpaceType::local_dofs_v)
        {
            throw std::runtime_error(
                "scatter_divergence_local_matrix_dense_2d: local matrix has unexpected dimensions.");
        }

        const auto& rt_map = rt_space.cell_dof_map(patch_cell_index);
        for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
        {
            const int I = scalar_space.local_to_patch_dof(patch_cell_index, i);
            for (int j = 0; j < RTFluxSpaceType::local_dofs_v; ++j)
            {
                const auto& J = rt_map[static_cast<std::size_t>(j)];
                if (J.patch_dof_id < 0)
                    continue;

                const double value =
                    static_cast<double>(J.orientation_sign) * local(i, j);
                if (std::abs(value) > zero_tol)
                    out(I, J.patch_dof_id) += value;
            }
        }
    }

    template<class LocalVectorType, class RTFluxSpaceType>
    void scatter_rt_local_vector_dense_2d(
        DenseVector& out,
        const LocalVectorType& local,
        const RTFluxSpaceType& space,
        int patch_cell_index)
    {
        if (local.size != RTFluxSpaceType::local_dofs_v)
        {
            throw std::runtime_error(
                "scatter_rt_local_vector_dense_2d: local vector has unexpected dimension.");
        }

        const auto& map = space.cell_dof_map(patch_cell_index);
        for (int local_dof_id = 0;
             local_dof_id < RTFluxSpaceType::local_dofs_v;
             ++local_dof_id)
        {
            const auto& entry =
                map[static_cast<std::size_t>(local_dof_id)];
            if (entry.patch_dof_id < 0)
                continue;

            out[entry.patch_dof_id] +=
                static_cast<double>(entry.orientation_sign) *
                local[local_dof_id];
        }
    }

    template<class LocalVectorType, class PatchScalarSpaceType>
    void scatter_scalar_local_vector_dense_2d(
        DenseVector& out,
        const LocalVectorType& local,
        const PatchScalarSpaceType& scalar_space,
        int patch_cell_index)
    {
        if (local.size != PatchScalarSpaceType::local_dofs_v)
        {
            throw std::runtime_error(
                "scatter_scalar_local_vector_dense_2d: local vector has unexpected dimension.");
        }

        for (int local_dof_id = 0;
             local_dof_id < PatchScalarSpaceType::local_dofs_v;
             ++local_dof_id)
        {
            const int patch_dof_id =
                scalar_space.local_to_patch_dof(
                    patch_cell_index,
                    local_dof_id);
            out[patch_dof_id] += local[local_dof_id];
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ABElementCacheType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_mat_A_time_2d_from_ab_cache(
        DenseMatrix& A,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const ABElementCacheType& ab_element_cache,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_dense_mat_A_time_2d<cached>: patch cell count mismatch.");
        }

        A.set_zero(flux_space.n_dofs(), flux_space.n_dofs());
        const int slab_id = flux_space.patch().slab_id;

        for (int patch_cell_index = 0;
             patch_cell_index < flux_space.n_patch_cells();
             ++patch_cell_index)
        {
            const int slab_cell_id =
                flux_space.patch().cell(patch_cell_index).slab_cell_id;
            const auto& local_A =
                ab_element_cache.cell(slab_id, slab_cell_id).A;
            scatter_rt_local_matrix_dense_2d(
                A,
                local_A,
                flux_space,
                patch_cell_index,
                zero_tol);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class MFunction>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_mat_A_time_2d(
        DenseMatrix& A,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const MFunction& M,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells() ||
            tables.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_dense_mat_A_time_2d: patch cell count mismatch.");
        }

        A.set_zero(flux_space.n_dofs(), flux_space.n_dofs());

        using MassData =
            detail::LocalRTMassQuadratureData2D<
                QSpace,
                QTime,
                PatchFluxSpaceType,
                PatchScalarSpaceType,
                MFunction>;
        const MassData mass_data(tables, M);

        for (int patch_cell_index = 0;
             patch_cell_index < flux_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local_A =
                assemble_local_rt_mass_matrix_time_2d<QSpace, QTime>(
                    tables,
                    mass_data,
                    patch_cell_index);
            scatter_rt_local_matrix_dense_2d(
                A,
                local_A,
                flux_space,
                patch_cell_index,
                zero_tol);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ABElementCacheType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_mat_B_time_2d_from_ab_cache(
        DenseMatrix& B,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const ABElementCacheType& ab_element_cache,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_dense_mat_B_time_2d<cached>: patch cell count mismatch.");
        }

        B.set_zero(scalar_space.n_dofs(), flux_space.n_dofs());
        const int slab_id = flux_space.patch().slab_id;

        for (int patch_cell_index = 0;
             patch_cell_index < scalar_space.n_patch_cells();
             ++patch_cell_index)
        {
            const int slab_cell_id =
                flux_space.patch().cell(patch_cell_index).slab_cell_id;
            const auto& local_B =
                ab_element_cache.cell(slab_id, slab_cell_id).B;
            scatter_divergence_local_matrix_dense_2d(
                B,
                local_B,
                scalar_space,
                flux_space,
                patch_cell_index,
                zero_tol);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_mat_B_time_2d(
        DenseMatrix& B,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        double zero_tol = 1.0e-15)
    {
        if (scalar_space.n_patch_cells() != flux_space.n_patch_cells() ||
            tables.n_patch_cells() != flux_space.n_patch_cells())
        {
            throw std::runtime_error(
                "assemble_dense_mat_B_time_2d: patch cell count mismatch.");
        }

        B.set_zero(scalar_space.n_dofs(), flux_space.n_dofs());

        for (int patch_cell_index = 0;
             patch_cell_index < scalar_space.n_patch_cells();
             ++patch_cell_index)
        {
            const auto local_B =
                assemble_local_divergence_coupling_matrix_time_2d<
                    QSpace,
                    QTime>(
                    scalar_space,
                    flux_space,
                    tables,
                    patch_cell_index);
            scatter_divergence_local_matrix_dense_2d(
                B,
                local_B,
                scalar_space,
                flux_space,
                patch_cell_index,
                zero_tol);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class XSpaceType,
        class SlabSpaceType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_vec_f_time_2d(
        DenseVector& f,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta)
    {
        static_cast<void>(scalar_space);
        f.resize(flux_space.n_dofs());
        f.set_zero();

        const auto& patch = flux_space.patch();
        for (int patch_cell_index = 0;
             patch_cell_index < patch.n_cells;
             ++patch_cell_index)
        {
            const int slab_id = patch.slab_id;
            const int slab_cell_id =
                patch.cell(patch_cell_index).slab_cell_id;
            const int source_cell_id =
                patch.cell(patch_cell_index).source_cell_id;
            const int x_cell_id =
                finite_element::assembly::detail::
                    find_active_ancestor_cell_from_source_cell(
                        *context.x_ancestor_cache,
                        *context.x_space,
                        source_cell_id);
            const auto& slab_geom =
                (*context.slab_geometry_caches)[
                    static_cast<std::size_t>(slab_id)]
                    .geometry(slab_cell_id);
            const auto& x_geom =
                context.x_geometry_cache->geometry(x_cell_id);

            la::local::FixedLocalVector<PatchFluxSpaceType::local_dofs_v>
                local_f;
            finite_element::assembly::detail::zero_local_vector(local_f);
            accumulate_patch_flux_rhs_on_cell_time_2d<QSpace, QTime>(
                local_f,
                flux_space,
                tables,
                patch_cell_index,
                slab_id,
                lambda_tilde,
                u_delta,
                x_cell_id,
                slab_geom,
                x_geom);
            scatter_rt_local_vector_dense_2d(
                f,
                local_f,
                flux_space,
                patch_cell_index);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class RHSStateCacheType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_vec_f_time_2d_from_rhs_cache(
        DenseVector& f,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const RHSStateCacheType& rhs_state_cache)
    {
        static_cast<void>(scalar_space);
        f.resize(flux_space.n_dofs());
        f.set_zero();

        const auto& patch = flux_space.patch();
        for (int patch_cell_index = 0;
             patch_cell_index < patch.n_cells;
             ++patch_cell_index)
        {
            la::local::FixedLocalVector<PatchFluxSpaceType::local_dofs_v>
                local_f;
            finite_element::assembly::detail::zero_local_vector(local_f);
            accumulate_patch_flux_rhs_on_cell_time_2d_from_rhs_cache<
                QSpace,
                QTime>(
                local_f,
                flux_space,
                tables,
                patch_cell_index,
                patch.slab_id,
                rhs_state_cache);
            scatter_rt_local_vector_dense_2d(
                f,
                local_f,
                flux_space,
                patch_cell_index);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_vec_g_time_2d(
        DenseVector& g,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        static_cast<void>(flux_space);
        g.resize(scalar_space.n_dofs());
        g.set_zero();

        const auto& patch = scalar_space.patch();
        for (int patch_cell_index = 0;
             patch_cell_index < patch.n_cells;
             ++patch_cell_index)
        {
            const int slab_id = patch.slab_id;
            const int slab_cell_id =
                patch.cell(patch_cell_index).slab_cell_id;
            const int source_cell_id =
                patch.cell(patch_cell_index).source_cell_id;
            const int x_cell_id =
                finite_element::assembly::detail::
                    find_active_ancestor_cell_from_source_cell(
                        *context.x_ancestor_cache,
                        *context.x_space,
                        source_cell_id);
            const auto& slab_geom =
                (*context.slab_geometry_caches)[
                    static_cast<std::size_t>(slab_id)]
                    .geometry(slab_cell_id);
            const auto& x_geom =
                context.x_geometry_cache->geometry(x_cell_id);

            la::local::FixedLocalVector<PatchScalarSpaceType::local_dofs_v>
                local_g;
            finite_element::assembly::detail::zero_local_vector(local_g);
            accumulate_patch_scalar_rhs_on_cell_time_2d<QSpace, QTime>(
                local_g,
                scalar_space,
                tables,
                patch_cell_index,
                slab_id,
                lambda_tilde,
                u_delta,
                x_cell_id,
                slab_geom,
                x_geom,
                ell,
                M);
            scatter_scalar_local_vector_dense_2d(
                g,
                local_g,
                scalar_space,
                patch_cell_index);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class RHSStateCacheType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_vec_g_time_2d_from_rhs_cache(
        DenseVector& g,
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const RHSStateCacheType& rhs_state_cache)
    {
        static_cast<void>(flux_space);
        g.resize(scalar_space.n_dofs());
        g.set_zero();

        const auto& patch = scalar_space.patch();
        for (int patch_cell_index = 0;
             patch_cell_index < patch.n_cells;
             ++patch_cell_index)
        {
            la::local::FixedLocalVector<PatchScalarSpaceType::local_dofs_v>
                local_g;
            finite_element::assembly::detail::zero_local_vector(local_g);
            accumulate_patch_scalar_rhs_on_cell_time_2d_from_rhs_cache<
                QSpace,
                QTime>(
                local_g,
                scalar_space,
                tables,
                patch_cell_index,
                patch.slab_id,
                rhs_state_cache);
            scatter_scalar_local_vector_dense_2d(
                g,
                local_g,
                scalar_space,
                patch_cell_index);
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    [[nodiscard]] DenseLocalErrorBlocks assemble_dense_local_error_problem_2d(
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType>& tables,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M,
        double zero_tol = 1.0e-15)
    {
        DenseLocalErrorBlocks blocks(flux_space.n_dofs(), scalar_space.n_dofs());
        assemble_dense_mat_A_time_2d<QSpace, QTime>(
            blocks.A,
            flux_space,
            scalar_space,
            tables,
            M,
            zero_tol);
        assemble_dense_mat_B_time_2d<QSpace, QTime>(
            blocks.B,
            flux_space,
            scalar_space,
            tables,
            zero_tol);
        blocks.C.set_zero(scalar_space.n_dofs(), scalar_space.n_dofs());
        assemble_dense_vec_f_time_2d<QSpace, QTime>(
            blocks.f,
            flux_space,
            scalar_space,
            tables,
            context,
            lambda_tilde,
            u_delta);
        assemble_dense_vec_g_time_2d<QSpace, QTime>(
            blocks.g,
            flux_space,
            scalar_space,
            tables,
            context,
            lambda_tilde,
            u_delta,
            ell,
            M);
        return blocks;
    }

    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction,
        class XSpaceType,
        class SlabSpaceType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    [[nodiscard]] DenseLocalErrorBlocks assemble_dense_local_error_problem_2d(
        const PatchFluxSpaceType& flux_space,
        const PatchScalarSpaceType& scalar_space,
        const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M,
        double zero_tol = 1.0e-15)
    {
        const finite_element::assembly::detail::LocalErrorQuadratureTables2D<
            QSpace,
            QTime,
            PatchFluxSpaceType,
            PatchScalarSpaceType> tables(flux_space, scalar_space);

        return assemble_dense_local_error_problem_2d<QSpace, QTime>(
            flux_space,
            scalar_space,
            tables,
            context,
            lambda_tilde,
            u_delta,
            ell,
            M,
            zero_tol);
    }

    template<class PatchScalarSpaceType>
    void fill_dense_local_error_scalar_constraint_system_2d(
        const DenseLocalErrorBlocks& blocks,
        const PatchScalarSpaceType& scalar_space,
        DenseLocalErrorConstraintMatrix2D& system,
        DenseVector& rhs,
        double zero_tol = 1.0e-15)
    {
        const int n_lambda = blocks.n_lambda;
        const int n_u = blocks.n_u;
        const int n_constraints =
            scalar_space.has_mean_zero_constraint()
                ? scalar_space.n_mean_zero_constraints()
                : 0;
        const int n_total = n_lambda + n_u + n_constraints;

        if (blocks.A.rows() != n_lambda || blocks.A.cols() != n_lambda ||
            blocks.B.rows() != n_u || blocks.B.cols() != n_lambda ||
            blocks.C.rows() != n_u || blocks.C.cols() != n_u ||
            blocks.f.size() != n_lambda || blocks.g.size() != n_u)
        {
            throw std::runtime_error(
                "fill_dense_local_error_scalar_constraint_system_2d: block dimension mismatch.");
        }

        system.n_lambda = n_lambda;
        system.n_u = n_u;
        system.n_constraints = n_constraints;
        system.matrix.set_zero(n_total, n_total);
        system.matrix.set_block(0, 0, blocks.A);
        system.matrix.set_transpose_block(0, n_lambda, blocks.B);
        system.matrix.set_block(n_lambda, 0, blocks.B);
        system.matrix.set_block(n_lambda, n_lambda, blocks.C);

        for (int constraint_id = 0;
             constraint_id < n_constraints;
             ++constraint_id)
        {
            const auto& row =
                scalar_space.mean_zero_constraint_row(constraint_id);
            if (row.size() != static_cast<std::size_t>(n_u))
            {
                throw std::runtime_error(
                    "fill_dense_local_error_scalar_constraint_system_2d: scalar constraint row size mismatch.");
            }

            const int constraint_col = n_lambda + n_u + constraint_id;
            const int constraint_row = n_lambda + n_u + constraint_id;
            for (int scalar_dof = 0; scalar_dof < n_u; ++scalar_dof)
            {
                const double value =
                    row[static_cast<std::size_t>(scalar_dof)];
                if (std::abs(value) <= zero_tol)
                    continue;

                system.matrix(n_lambda + scalar_dof, constraint_col) =
                    value;
                system.matrix(constraint_row, n_lambda + scalar_dof) =
                    value;
            }
        }

        rhs.resize(n_total);
        rhs.set_zero();
        for (int i = 0; i < system.n_lambda; ++i)
            rhs[i] = blocks.f[i];
        for (int i = 0; i < system.n_u; ++i)
            rhs[system.n_lambda + i] = blocks.g[i];
    }

    template<class PatchScalarSpaceType>
    [[nodiscard]] DenseLocalErrorConstraintMatrix2D
    build_dense_local_error_scalar_constraint_matrix_2d(
        const DenseLocalErrorBlocks& blocks,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15)
    {
        DenseLocalErrorConstraintMatrix2D system;
        DenseVector rhs;
        fill_dense_local_error_scalar_constraint_system_2d(
            blocks,
            scalar_space,
            system,
            rhs,
            zero_tol);

        return system;
    }

    [[nodiscard]] inline DenseVector
    build_dense_local_error_scalar_constraint_rhs_2d(
        const DenseLocalErrorBlocks& blocks,
        const DenseLocalErrorConstraintMatrix2D& system)
    {
        if (blocks.n_lambda != system.n_lambda ||
            blocks.n_u != system.n_u ||
            blocks.f.size() != system.n_lambda ||
            blocks.g.size() != system.n_u)
        {
            throw std::runtime_error(
                "build_dense_local_error_scalar_constraint_rhs_2d: block dimension mismatch.");
        }

        DenseVector rhs(system.system_size());
        rhs.set_zero();
        for (int i = 0; i < system.n_lambda; ++i)
            rhs[i] = blocks.f[i];
        for (int i = 0; i < system.n_u; ++i)
            rhs[system.n_lambda + i] = blocks.g[i];
        return rhs;
    }

    [[nodiscard]] inline bool
    dense_local_error_constraint_residual_is_acceptable_2d(
        const DenseMatrix& matrix,
        const DenseVector& rhs,
        const DenseVector& candidate)
    {
        if (candidate.size() != rhs.size() ||
            matrix.rows() != rhs.size() ||
            matrix.cols() != candidate.size() ||
            !dense_vector_is_finite(candidate))
        {
            return false;
        }

        double residual = 0.0;
        double rhs_norm = 0.0;
        for (int row = 0; row < matrix.rows(); ++row)
        {
            double value = -rhs[row];
            rhs_norm = std::max(rhs_norm, std::abs(rhs[row]));
            for (int col = 0; col < matrix.cols(); ++col)
                value += matrix(row, col) * candidate[col];
            residual = std::max(residual, std::abs(value));
        }
        const double scale = std::max(1.0, rhs_norm);
        return std::isfinite(residual) && residual <= 1.0e-10 * scale;
    }

    template<class Backend>
    [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
    split_dense_local_error_scalar_constraint_solution_2d(
        const DenseVector& solution,
        int n_lambda,
        int n_u)
    {
        if (solution.size() < n_lambda + n_u)
        {
            throw std::runtime_error(
                "split_dense_local_error_scalar_constraint_solution_2d: solution dimension mismatch.");
        }

        la::saddle::SaddlePointSolution<Backend> split;
        split.lambda.resize(n_lambda);
        split.u.resize(n_u);
        for (int i = 0; i < n_lambda; ++i)
            split.lambda[i] = solution[i];
        for (int i = 0; i < n_u; ++i)
            split.u[i] = solution[n_lambda + i];

        return split;
    }

    template<class Backend>
    [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
    solve_dense_local_error_scalar_constraint_system_2d(
        const DenseLocalErrorConstraintMatrix2D& system,
        const DenseVector& rhs,
        double* factorization_seconds = nullptr,
        double* solve_apply_seconds = nullptr)
    {
        if (system.matrix.rows() != system.system_size() ||
            system.matrix.cols() != system.system_size() ||
            rhs.size() != system.system_size())
        {
            throw std::runtime_error(
                "solve_dense_local_error_scalar_constraint_system_2d: system dimension mismatch.");
        }

        typename Backend::DenseSolver solver;
        la::concepts::SolverOptions options;
        options.dense_factorization =
            la::concepts::DenseFactorizationType::PartialPivotDenseLU;
        auto factor_start = std::chrono::steady_clock::now();
        solver.compute(system.matrix, options);
        auto factor_end = std::chrono::steady_clock::now();
        if (factorization_seconds != nullptr)
        {
            *factorization_seconds +=
                std::chrono::duration<double>(factor_end - factor_start)
                    .count();
        }

        auto solve_start = std::chrono::steady_clock::now();
        DenseVector solution = solver.solve(rhs);
        auto solve_end = std::chrono::steady_clock::now();
        if (solve_apply_seconds != nullptr)
        {
            *solve_apply_seconds +=
                std::chrono::duration<double>(solve_end - solve_start)
                    .count();
        }
        if (!dense_local_error_constraint_residual_is_acceptable_2d(
                system.matrix,
                rhs,
                solution))
        {
            options.dense_factorization =
                la::concepts::DenseFactorizationType::RankRevealingDenseLU;
            options.dense_rank_revealing_threshold = 1.0e-12;
            factor_start = std::chrono::steady_clock::now();
            solver.compute(system.matrix, options);
            factor_end = std::chrono::steady_clock::now();
            if (factorization_seconds != nullptr)
            {
                *factorization_seconds +=
                    std::chrono::duration<double>(factor_end - factor_start)
                        .count();
            }

            solve_start = std::chrono::steady_clock::now();
            solution = solver.solve(rhs);
            solve_end = std::chrono::steady_clock::now();
            if (solve_apply_seconds != nullptr)
            {
                *solve_apply_seconds +=
                    std::chrono::duration<double>(solve_end - solve_start)
                        .count();
            }
            if (!dense_local_error_constraint_residual_is_acceptable_2d(
                    system.matrix,
                    rhs,
                    solution))
            {
                throw std::runtime_error(
                    "solve_dense_local_error_scalar_constraint_system_2d: dense local solve residual is too large.");
            }
        }

        return split_dense_local_error_scalar_constraint_solution_2d<
            Backend>(
            solution,
            system.n_lambda,
            system.n_u);
    }

    template<class Backend, class PatchScalarSpaceType>
    [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
    solve_dense_local_error_blocks_with_scalar_constraints_workspace_2d(
        const DenseLocalErrorBlocks& blocks,
        const PatchScalarSpaceType& scalar_space,
        DenseLocalErrorExplicitSolveWorkspace2D<Backend>& workspace,
        double zero_tol = 1.0e-15,
        double* factorization_seconds = nullptr,
        double* solve_apply_seconds = nullptr)
    {
        fill_dense_local_error_scalar_constraint_system_2d(
            blocks,
            scalar_space,
            workspace.system,
            workspace.rhs,
            zero_tol);

        la::concepts::SolverOptions options;
        options.dense_factorization =
            la::concepts::DenseFactorizationType::PartialPivotDenseLU;

        auto factor_start = std::chrono::steady_clock::now();
        workspace.solver.compute(workspace.system.matrix, options);
        auto factor_end = std::chrono::steady_clock::now();
        if (factorization_seconds != nullptr)
        {
            *factorization_seconds +=
                std::chrono::duration<double>(factor_end - factor_start)
                    .count();
        }

        auto solve_start = std::chrono::steady_clock::now();
        workspace.solver.solve(workspace.rhs, workspace.solution);
        auto solve_end = std::chrono::steady_clock::now();
        if (solve_apply_seconds != nullptr)
        {
            *solve_apply_seconds +=
                std::chrono::duration<double>(solve_end - solve_start)
                    .count();
        }

        if (!dense_local_error_constraint_residual_is_acceptable_2d(
                workspace.system.matrix,
                workspace.rhs,
                workspace.solution))
        {
            options.dense_factorization =
                la::concepts::DenseFactorizationType::RankRevealingDenseLU;
            options.dense_rank_revealing_threshold = 1.0e-12;

            factor_start = std::chrono::steady_clock::now();
            workspace.solver.compute(workspace.system.matrix, options);
            factor_end = std::chrono::steady_clock::now();
            if (factorization_seconds != nullptr)
            {
                *factorization_seconds +=
                    std::chrono::duration<double>(factor_end - factor_start)
                        .count();
            }

            solve_start = std::chrono::steady_clock::now();
            workspace.solver.solve(workspace.rhs, workspace.solution);
            solve_end = std::chrono::steady_clock::now();
            if (solve_apply_seconds != nullptr)
            {
                *solve_apply_seconds +=
                    std::chrono::duration<double>(solve_end - solve_start)
                        .count();
            }

            if (!dense_local_error_constraint_residual_is_acceptable_2d(
                    workspace.system.matrix,
                    workspace.rhs,
                    workspace.solution))
            {
                throw std::runtime_error(
                    "solve_dense_local_error_blocks_with_scalar_constraints_workspace_2d: dense local solve residual is too large.");
            }
        }

        return split_dense_local_error_scalar_constraint_solution_2d<
            Backend>(
            workspace.solution,
            workspace.system.n_lambda,
            workspace.system.n_u);
    }

    template<class Backend, class PatchScalarSpaceType>
    [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
    solve_dense_local_error_blocks_with_scalar_constraints_2d(
        const DenseLocalErrorBlocks& blocks,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15,
        double* factorization_seconds = nullptr,
        double* solve_apply_seconds = nullptr)
    {
        const auto system =
            build_dense_local_error_scalar_constraint_matrix_2d(
                blocks,
                scalar_space,
                zero_tol);
        const auto rhs =
            build_dense_local_error_scalar_constraint_rhs_2d(blocks, system);

        return solve_dense_local_error_scalar_constraint_system_2d<Backend>(
            system,
            rhs,
            factorization_seconds,
            solve_apply_seconds);
    }

    template<class PatchScalarSpaceType>
    [[nodiscard]] DenseMatrix build_scalar_mean_zero_reduction_basis_2d(
        const PatchScalarSpaceType& scalar_space,
        int n_u,
        double zero_tol = 1.0e-15)
    {
        if (!scalar_space.has_mean_zero_constraint())
        {
            DenseMatrix identity;
            identity.set_identity(n_u, n_u);
            return identity;
        }

        const int n_constraints = scalar_space.n_mean_zero_constraints();
        if (n_constraints < 0 || n_constraints > n_u)
        {
            throw std::runtime_error(
                "build_scalar_mean_zero_reduction_basis_2d: invalid constraint count.");
        }

        DenseMatrix rows(n_constraints, n_u);
        rows.set_zero();
        for (int constraint_id = 0;
             constraint_id < n_constraints;
             ++constraint_id)
        {
            const auto& row =
                scalar_space.mean_zero_constraint_row(constraint_id);
            if (row.size() != static_cast<std::size_t>(n_u))
            {
                throw std::runtime_error(
                    "build_scalar_mean_zero_reduction_basis_2d: scalar constraint row size mismatch.");
            }

            for (int scalar_dof = 0; scalar_dof < n_u; ++scalar_dof)
                rows(constraint_id, scalar_dof) =
                    row[static_cast<std::size_t>(scalar_dof)];
        }

        DenseMatrix reduced_rows = rows;
        std::vector<int> pivot_columns;
        pivot_columns.reserve(static_cast<std::size_t>(n_constraints));

        int pivot_row = 0;
        for (int col = 0; col < n_u && pivot_row < n_constraints; ++col)
        {
            int best_row = pivot_row;
            double best_abs = std::abs(reduced_rows(pivot_row, col));
            for (int row = pivot_row + 1; row < n_constraints; ++row)
            {
                const double value = std::abs(reduced_rows(row, col));
                if (value > best_abs)
                {
                    best_abs = value;
                    best_row = row;
                }
            }

            if (best_abs <= zero_tol)
                continue;

            if (best_row != pivot_row)
            {
                for (int c = 0; c < n_u; ++c)
                    std::swap(
                        reduced_rows(best_row, c),
                        reduced_rows(pivot_row, c));
            }

            const double pivot = reduced_rows(pivot_row, col);
            for (int c = 0; c < n_u; ++c)
                reduced_rows(pivot_row, c) /= pivot;
            for (int row = 0; row < n_constraints; ++row)
            {
                if (row == pivot_row)
                    continue;

                const double factor = reduced_rows(row, col);
                if (std::abs(factor) <= zero_tol)
                    continue;

                for (int c = 0; c < n_u; ++c)
                    reduced_rows(row, c) -=
                        factor * reduced_rows(pivot_row, c);
            }

            pivot_columns.push_back(col);
            ++pivot_row;
        }

        const int rank = static_cast<int>(pivot_columns.size());
        if (rank != n_constraints)
        {
            throw std::runtime_error(
                "build_scalar_mean_zero_reduction_basis_2d: dependent scalar constraints.");
        }

        std::vector<bool> is_pivot(static_cast<std::size_t>(n_u), false);
        for (const int col : pivot_columns)
            is_pivot[static_cast<std::size_t>(col)] = true;

        std::vector<int> free_columns;
        free_columns.reserve(static_cast<std::size_t>(n_u - rank));
        for (int col = 0; col < n_u; ++col)
        {
            if (!is_pivot[static_cast<std::size_t>(col)])
                free_columns.push_back(col);
        }

        DenseMatrix Z(n_u, static_cast<int>(free_columns.size()));
        Z.set_zero();
        for (int reduced_col = 0;
             reduced_col < static_cast<int>(free_columns.size());
             ++reduced_col)
        {
            const int free_col =
                free_columns[static_cast<std::size_t>(reduced_col)];
            Z(free_col, reduced_col) = 1.0;
            for (int constraint_id = 0;
                 constraint_id < n_constraints;
                 ++constraint_id)
            {
                const int pivot_col =
                    pivot_columns[static_cast<std::size_t>(constraint_id)];
                Z(pivot_col, reduced_col) =
                    -reduced_rows(constraint_id, free_col);
            }
        }

        const double residual = dense_product_inf_norm(rows, Z);
        if (!std::isfinite(residual) || residual > 100.0 * zero_tol)
        {
            throw std::runtime_error(
                "build_scalar_mean_zero_reduction_basis_2d: reduction basis does not satisfy scalar constraints.");
        }

        return Z;
    }

    template<class PatchScalarSpaceType>
    [[nodiscard]] int reduced_scalar_dimension_2d(
        const PatchScalarSpaceType& scalar_space,
        int n_u)
    {
        return scalar_space.has_mean_zero_constraint()
                   ? n_u - scalar_space.n_mean_zero_constraints()
                   : n_u;
    }

    template<class Backend, class PatchScalarSpaceType>
    [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
    solve_dense_local_error_blocks_with_reduced_scalar_basis_2d(
        const DenseLocalErrorBlocks& blocks,
        const PatchScalarSpaceType& scalar_space,
        const DenseMatrix* reduction_basis,
        double zero_tol = 1.0e-15,
        double* transform_seconds = nullptr,
        double* factorization_seconds = nullptr,
        double* solve_apply_seconds = nullptr)
    {
        const int n_lambda = blocks.n_lambda;
        const int n_u = blocks.n_u;

        if (blocks.A.rows() != n_lambda || blocks.A.cols() != n_lambda ||
            blocks.B.rows() != n_u || blocks.B.cols() != n_lambda ||
            blocks.C.rows() != n_u || blocks.C.cols() != n_u ||
            blocks.f.size() != n_lambda || blocks.g.size() != n_u)
        {
            throw std::runtime_error(
                "solve_dense_local_error_blocks_with_reduced_scalar_basis_2d: block dimension mismatch.");
        }

        if (!scalar_space.has_mean_zero_constraint())
        {
            return solve_dense_local_error_blocks_with_scalar_constraints_2d<
                Backend>(
                blocks,
                scalar_space,
                zero_tol,
                factorization_seconds,
                solve_apply_seconds);
        }

        const auto transform_start = std::chrono::steady_clock::now();

        DenseMatrix owned_reduction_basis;
        if (reduction_basis == nullptr)
        {
            owned_reduction_basis =
                build_scalar_mean_zero_reduction_basis_2d(
                    scalar_space,
                    n_u,
                    zero_tol);
            reduction_basis = &owned_reduction_basis;
        }

        const DenseMatrix& Z = *reduction_basis;
        if (Z.rows() != n_u ||
            Z.cols() != n_u - scalar_space.n_mean_zero_constraints())
        {
            throw std::runtime_error(
                "solve_dense_local_error_blocks_with_reduced_scalar_basis_2d: reduction basis dimension mismatch.");
        }

        const int n_u_reduced = static_cast<int>(Z.cols());
        const int n_total = n_lambda + n_u_reduced;

        DenseMatrix B_reduced(n_u_reduced, n_lambda);
        B_reduced.set_zero();
        for (int reduced_i = 0; reduced_i < n_u_reduced; ++reduced_i)
        {
            for (int lambda_j = 0; lambda_j < n_lambda; ++lambda_j)
            {
                double value = 0.0;
                for (int scalar_i = 0; scalar_i < n_u; ++scalar_i)
                    value += Z(scalar_i, reduced_i) *
                             blocks.B(scalar_i, lambda_j);
                B_reduced(reduced_i, lambda_j) = value;
            }
        }

        DenseMatrix C_reduced(n_u_reduced, n_u_reduced);
        C_reduced.set_zero();
        if (blocks.C.max_abs_coeff() > zero_tol)
        {
            for (int reduced_i = 0; reduced_i < n_u_reduced; ++reduced_i)
            {
                for (int reduced_j = 0;
                     reduced_j < n_u_reduced;
                     ++reduced_j)
                {
                    double value = 0.0;
                    for (int scalar_i = 0; scalar_i < n_u; ++scalar_i)
                    {
                        for (int scalar_j = 0; scalar_j < n_u; ++scalar_j)
                        {
                            value += Z(scalar_i, reduced_i) *
                                     blocks.C(scalar_i, scalar_j) *
                                     Z(scalar_j, reduced_j);
                        }
                    }
                    C_reduced(reduced_i, reduced_j) = value;
                }
            }
        }

        DenseVector g_reduced(n_u_reduced);
        g_reduced.set_zero();
        for (int reduced_i = 0; reduced_i < n_u_reduced; ++reduced_i)
        {
            double value = 0.0;
            for (int scalar_i = 0; scalar_i < n_u; ++scalar_i)
                value += Z(scalar_i, reduced_i) * blocks.g[scalar_i];
            g_reduced[reduced_i] = value;
        }
        if (transform_seconds != nullptr)
        {
            const auto transform_end = std::chrono::steady_clock::now();
            *transform_seconds +=
                std::chrono::duration<double>(
                    transform_end - transform_start)
                    .count();
        }

        DenseMatrix matrix(n_total, n_total);
        matrix.set_zero();
        matrix.set_block(0, 0, blocks.A);
        matrix.set_transpose_block(0, n_lambda, B_reduced);
        matrix.set_block(n_lambda, 0, B_reduced);
        matrix.set_block(n_lambda, n_lambda, C_reduced);

        DenseVector rhs(n_total);
        rhs.set_zero();
        for (int i = 0; i < n_lambda; ++i)
            rhs[i] = blocks.f[i];
        for (int i = 0; i < n_u_reduced; ++i)
            rhs[n_lambda + i] = g_reduced[i];

        auto residual_is_acceptable =
            [&](const DenseVector& candidate)
            {
                if (candidate.size() != rhs.size() ||
                    !dense_vector_is_finite(candidate))
                {
                    return false;
                }

                const auto residual_vector =
                    la::ops::subtract(
                        la::ops::matvec(matrix, candidate),
                        rhs);
                const double residual =
                    la::ops::inf_norm(residual_vector);
                const double scale =
                    std::max(1.0, la::ops::inf_norm(rhs));
                return std::isfinite(residual) &&
                       residual <= 1.0e-10 * scale;
            };

        typename Backend::DenseSolver solver;
        la::concepts::SolverOptions solve_options;
        solve_options.dense_factorization =
            la::concepts::DenseFactorizationType::PartialPivotDenseLU;
        auto factor_start = std::chrono::steady_clock::now();
        solver.compute(matrix, solve_options);
        auto factor_end = std::chrono::steady_clock::now();
        if (factorization_seconds != nullptr)
        {
            *factorization_seconds +=
                std::chrono::duration<double>(factor_end - factor_start)
                    .count();
        }
        auto solve_start = std::chrono::steady_clock::now();
        DenseVector solution = solver.solve(rhs);
        auto solve_end = std::chrono::steady_clock::now();
        if (solve_apply_seconds != nullptr)
        {
            *solve_apply_seconds +=
                std::chrono::duration<double>(solve_end - solve_start)
                    .count();
        }
        if (!residual_is_acceptable(solution))
        {
            solve_options.dense_factorization =
                la::concepts::DenseFactorizationType::RankRevealingDenseLU;
            solve_options.dense_rank_revealing_threshold = 1.0e-12;
            factor_start = std::chrono::steady_clock::now();
            solver.compute(matrix, solve_options);
            factor_end = std::chrono::steady_clock::now();
            if (factorization_seconds != nullptr)
            {
                *factorization_seconds +=
                    std::chrono::duration<double>(factor_end - factor_start)
                        .count();
            }

            solve_start = std::chrono::steady_clock::now();
            solution = solver.solve(rhs);
            solve_end = std::chrono::steady_clock::now();
            if (solve_apply_seconds != nullptr)
            {
                *solve_apply_seconds +=
                    std::chrono::duration<double>(solve_end - solve_start)
                        .count();
            }
            if (!residual_is_acceptable(solution))
            {
                throw std::runtime_error(
                    "solve_dense_local_error_blocks_with_reduced_scalar_basis_2d: dense local solve residual is too large.");
            }
        }

        DenseVector u_reduced(n_u_reduced);
        for (int i = 0; i < n_u_reduced; ++i)
            u_reduced[i] = solution[n_lambda + i];
        const DenseVector u_full = Z.matvec(u_reduced);

        la::saddle::SaddlePointSolution<Backend> split;
        split.lambda.resize(n_lambda);
        split.u.resize(n_u);
        for (int i = 0; i < n_lambda; ++i)
            split.lambda[i] = solution[i];
        for (int i = 0; i < n_u; ++i)
            split.u[i] = u_full[i];

        return split;
    }

    template<class Backend, class PatchScalarSpaceType>
    [[nodiscard]] la::saddle::SaddlePointSolution<Backend>
    solve_dense_local_error_blocks_with_reduced_scalar_basis_2d(
        const DenseLocalErrorBlocks& blocks,
        const PatchScalarSpaceType& scalar_space,
        double zero_tol = 1.0e-15,
        double* transform_seconds = nullptr,
        double* factorization_seconds = nullptr,
        double* solve_apply_seconds = nullptr)
    {
        return solve_dense_local_error_blocks_with_reduced_scalar_basis_2d<
            Backend>(
            blocks,
            scalar_space,
            nullptr,
            zero_tol,
            transform_seconds,
            factorization_seconds,
            solve_apply_seconds);
    }
}

#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "assemble_local_error_problem.hpp"
#include "dense_local_error_blocks_2d.hpp"
#include "local_qpoint_state_cache_2d.hpp"

namespace finite_element::assembly::error_system
{
    template<class PatchFluxSpaceType, class QpointStateCellType>
    [[nodiscard]] const auto& local_rt_mass_matrix_from_qpoint_state_2d(
        const QpointStateCellType& state_cell)
    {
        static_cast<void>(PatchFluxSpaceType::local_dofs_v);
        return state_cell.local_A;
    }

    template<class PatchScalarSpaceType, class PatchFluxSpaceType, class QpointStateCellType>
    [[nodiscard]] const auto& local_divergence_matrix_from_qpoint_state_2d(
        const QpointStateCellType& state_cell)
    {
        static_cast<void>(PatchScalarSpaceType::local_dofs_v);
        static_cast<void>(PatchFluxSpaceType::local_dofs_v);
        return state_cell.local_B;
    }

    template<class LocalVectorType, class PatchFluxSpaceType, class QpointStateCellType>
    void accumulate_patch_flux_rhs_from_qpoint_state_2d(
        LocalVectorType& local,
        const PatchFluxSpaceType& flux_space,
        const QpointStateCellType& state_cell,
        int patch_cell_index)
    {
        const int local_vertex_index =
            flux_space.patch().cell(patch_cell_index).local_vertex_index;

        for (const auto& qp : state_cell.points)
        {
            const double psi_q =
                qp.partition_of_unity_values[
                    static_cast<std::size_t>(local_vertex_index)];
            for (int i = 0; i < PatchFluxSpaceType::local_dofs_v; ++i)
            {
                const auto& sigma_i =
                    qp.rt_basis_values[static_cast<std::size_t>(i)];
                local[i] +=
                    -psi_q *
                    (qp.grad_theta_tilde[0] * sigma_i[0] +
                     qp.grad_theta_tilde[1] * sigma_i[1]) *
                    qp.jacobian_weight;
            }
        }
    }

    template<class LocalVectorType, class PatchScalarSpaceType, class QpointStateCellType>
    void accumulate_patch_scalar_rhs_from_qpoint_state_2d(
        LocalVectorType& local,
        const PatchScalarSpaceType& scalar_space,
        const QpointStateCellType& state_cell,
        int patch_cell_index)
    {
        const int local_vertex_index =
            scalar_space.patch().cell(patch_cell_index).local_vertex_index;

        for (const auto& qp : state_cell.points)
        {
            const double psi_q =
                qp.partition_of_unity_values[
                    static_cast<std::size_t>(local_vertex_index)];
            const auto& grad_psi =
                qp.partition_of_unity_gradients[
                    static_cast<std::size_t>(local_vertex_index)];
            const double grad_psi_M_grad_theta =
                grad_psi[0] * qp.M_grad_theta_tilde[0] +
                grad_psi[1] * qp.M_grad_theta_tilde[1];
            const double rhs =
                psi_q * (qp.ell_value - qp.u_time_derivative) -
                grad_psi_M_grad_theta;
            const double rhs_dmu = rhs * qp.jacobian_weight;

            for (int i = 0; i < PatchScalarSpaceType::local_dofs_v; ++i)
            {
                local[i] +=
                    rhs_dmu *
                    qp.scalar_basis_values[static_cast<std::size_t>(i)];
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchSetType,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class QpointStateCacheType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_local_error_blocks_cell_first_time_2d_into_from_qpoint_state(
        const PatchSetType& patch_set,
        const std::vector<PatchFluxSpaceType>& flux_spaces,
        const std::vector<PatchScalarSpaceType>& scalar_spaces,
        const QpointStateCacheType& qpoint_state_cache,
        std::vector<DenseLocalErrorBlocks>& blocks,
        const std::vector<char>* operator_cache_hits = nullptr,
        LocalErrorProblemTimingStats* timing_stats = nullptr,
        double zero_tol = 1.0e-15)
    {
        static_cast<void>(QSpace);
        static_cast<void>(QTime);
        const int patch_count = patch_set.n_patches();
        if (static_cast<int>(flux_spaces.size()) != patch_count ||
            static_cast<int>(scalar_spaces.size()) != patch_count ||
            static_cast<int>(blocks.size()) != patch_count)
        {
            throw std::runtime_error(
                "assemble_dense_local_error_blocks_cell_first_time_2d_into_from_qpoint_state: size mismatch.");
        }
        if (operator_cache_hits != nullptr &&
            static_cast<int>(operator_cache_hits->size()) != patch_count)
        {
            throw std::runtime_error(
                "assemble_dense_local_error_blocks_cell_first_time_2d_into_from_qpoint_state: operator cache-hit count mismatch.");
        }

        const auto& slab_space = patch_set.slab_space();

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_A_seconds
                    : nullptr);
            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const auto& state_cell =
                        qpoint_state_cache.cell(slab_id, slab_cell_id);
                    const auto& local_A =
                        local_rt_mass_matrix_from_qpoint_state_2d<
                            PatchFluxSpaceType>(
                            state_cell);
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        if (operator_cache_hits != nullptr &&
                            (*operator_cache_hits)[
                                static_cast<std::size_t>(patch_id)] != 0)
                        {
                            continue;
                        }
                        scatter_rt_local_matrix_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].A,
                            local_A,
                            flux_spaces[static_cast<std::size_t>(patch_id)],
                            membership.patch_cell_index,
                            zero_tol);
                    }
                }
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_B_seconds
                    : nullptr);
            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const auto& state_cell =
                        qpoint_state_cache.cell(slab_id, slab_cell_id);
                    const auto& local_B =
                        local_divergence_matrix_from_qpoint_state_2d<
                            PatchScalarSpaceType,
                            PatchFluxSpaceType>(
                            state_cell);
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        if (operator_cache_hits != nullptr &&
                            (*operator_cache_hits)[
                                static_cast<std::size_t>(patch_id)] != 0)
                        {
                            continue;
                        }
                        scatter_divergence_local_matrix_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].B,
                            local_B,
                            scalar_spaces[static_cast<std::size_t>(patch_id)],
                            flux_spaces[static_cast<std::size_t>(patch_id)],
                            membership.patch_cell_index,
                            zero_tol);
                    }
                }
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_C_seconds
                    : nullptr);
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                if (operator_cache_hits != nullptr &&
                    (*operator_cache_hits)[
                        static_cast<std::size_t>(patch_id)] != 0)
                {
                    continue;
                }
                blocks[static_cast<std::size_t>(patch_id)].C.set_zero(
                    blocks[static_cast<std::size_t>(patch_id)].n_u,
                    blocks[static_cast<std::size_t>(patch_id)].n_u);
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_f_seconds
                    : nullptr);
            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const auto& state_cell =
                        qpoint_state_cache.cell(slab_id, slab_cell_id);
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        const auto& flux_space =
                            flux_spaces[static_cast<std::size_t>(patch_id)];
                        la::local::FixedLocalVector<
                            PatchFluxSpaceType::local_dofs_v> local_f;
                        finite_element::assembly::detail::zero_local_vector(
                            local_f);
                        accumulate_patch_flux_rhs_from_qpoint_state_2d(
                            local_f,
                            flux_space,
                            state_cell,
                            membership.patch_cell_index);
                        scatter_rt_local_vector_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].f,
                            local_f,
                            flux_space,
                            membership.patch_cell_index);
                    }
                }
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_g_seconds
                    : nullptr);
            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const auto& state_cell =
                        qpoint_state_cache.cell(slab_id, slab_cell_id);
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        const auto& scalar_space =
                            scalar_spaces[static_cast<std::size_t>(patch_id)];
                        la::local::FixedLocalVector<
                            PatchScalarSpaceType::local_dofs_v> local_g;
                        finite_element::assembly::detail::zero_local_vector(
                            local_g);
                        accumulate_patch_scalar_rhs_from_qpoint_state_2d(
                            local_g,
                            scalar_space,
                            state_cell,
                            membership.patch_cell_index);
                        scatter_scalar_local_vector_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].g,
                            local_g,
                            scalar_space,
                            membership.patch_cell_index);
                    }
                }
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class PatchSetType,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class RTCellCacheType,
        class ABElementCacheType,
        class RHSStateCacheType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    void assemble_dense_local_error_blocks_cell_first_time_2d_into(
        const PatchSetType& patch_set,
        const std::vector<PatchFluxSpaceType>& flux_spaces,
        const std::vector<PatchScalarSpaceType>& scalar_spaces,
        const RTCellCacheType& rt_cell_cache,
        const ABElementCacheType& ab_element_cache,
        const RHSStateCacheType& rhs_state_cache,
        std::vector<DenseLocalErrorBlocks>& blocks,
        const std::vector<char>* operator_cache_hits = nullptr,
        LocalErrorProblemTimingStats* timing_stats = nullptr,
        double zero_tol = 1.0e-15)
    {
        const int patch_count = patch_set.n_patches();
        if (static_cast<int>(flux_spaces.size()) != patch_count ||
            static_cast<int>(scalar_spaces.size()) != patch_count)
        {
            throw std::runtime_error(
                "assemble_dense_local_error_blocks_cell_first_time_2d: patch space count mismatch.");
        }
        if (static_cast<int>(blocks.size()) != patch_count)
        {
            throw std::runtime_error(
                "assemble_dense_local_error_blocks_cell_first_time_2d: block count mismatch.");
        }
        if (operator_cache_hits != nullptr &&
            static_cast<int>(operator_cache_hits->size()) != patch_count)
        {
            throw std::runtime_error(
                "assemble_dense_local_error_blocks_cell_first_time_2d: operator cache-hit count mismatch.");
        }

        using Tables =
            finite_element::assembly::detail::LocalErrorQuadratureTables2D<
                QSpace,
                QTime,
                PatchFluxSpaceType,
                PatchScalarSpaceType>;

        std::vector<Tables> tables;
        tables.reserve(static_cast<std::size_t>(patch_count));
        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->quadrature_table_construction_seconds
                    : nullptr);
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                tables.emplace_back(
                    flux_spaces[static_cast<std::size_t>(patch_id)],
                    scalar_spaces[static_cast<std::size_t>(patch_id)],
                    rt_cell_cache);
            }
        }
        if (timing_stats != nullptr)
        {
            timing_stats->local_table_construction_patches +=
                static_cast<double>(tables.size());
            for (const auto& table : tables)
            {
                timing_stats->local_table_construction_patch_cells +=
                    static_cast<double>(table.constructed_patch_cells());
                timing_stats->local_table_scalar_basis_qpoint_fills +=
                    static_cast<double>(table.scalar_basis_qpoint_fills());
                timing_stats
                    ->local_table_partition_of_unity_qpoint_fills +=
                    static_cast<double>(
                        table.partition_of_unity_qpoint_fills());
                timing_stats->local_table_owned_rt_basis_qpoint_fills +=
                    static_cast<double>(
                        table.owned_rt_basis_qpoint_fills());
            }
        }

        const auto& slab_space = patch_set.slab_space();

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_A_seconds
                    : nullptr);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const decltype(ab_element_cache.cell(slab_id, slab_cell_id).A)*
                        local_A = nullptr;
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        if (operator_cache_hits != nullptr &&
                            (*operator_cache_hits)[
                                static_cast<std::size_t>(patch_id)] != 0)
                        {
                            continue;
                        }
                        if (local_A == nullptr)
                        {
                            local_A =
                                &ab_element_cache.cell(slab_id, slab_cell_id).A;
                        }
                        scatter_rt_local_matrix_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].A,
                            *local_A,
                            flux_spaces[static_cast<std::size_t>(patch_id)],
                            membership.patch_cell_index,
                            zero_tol);
                    }
                }
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_B_seconds
                    : nullptr);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const decltype(ab_element_cache.cell(slab_id, slab_cell_id).B)*
                        local_B = nullptr;
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        if (operator_cache_hits != nullptr &&
                            (*operator_cache_hits)[
                                static_cast<std::size_t>(patch_id)] != 0)
                        {
                            continue;
                        }
                        if (local_B == nullptr)
                        {
                            local_B =
                                &ab_element_cache.cell(slab_id, slab_cell_id).B;
                        }
                        scatter_divergence_local_matrix_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].B,
                            *local_B,
                            scalar_spaces[static_cast<std::size_t>(patch_id)],
                            flux_spaces[static_cast<std::size_t>(patch_id)],
                            membership.patch_cell_index,
                            zero_tol);
                    }
                }
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_C_seconds
                    : nullptr);
            for (int patch_id = 0; patch_id < patch_count; ++patch_id)
            {
                if (operator_cache_hits != nullptr &&
                    (*operator_cache_hits)[
                        static_cast<std::size_t>(patch_id)] != 0)
                {
                    continue;
                }

                auto& block = blocks[static_cast<std::size_t>(patch_id)];
                block.C.set_zero(block.n_u, block.n_u);
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_f_seconds
                    : nullptr);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        const auto& flux_space =
                            flux_spaces[static_cast<std::size_t>(patch_id)];

                        la::local::FixedLocalVector<
                            PatchFluxSpaceType::local_dofs_v> local_f;
                        finite_element::assembly::detail::zero_local_vector(local_f);
                        accumulate_patch_flux_rhs_on_cell_time_2d_from_rhs_cache<
                            QSpace,
                            QTime>(
                            local_f,
                            flux_space,
                            tables[static_cast<std::size_t>(patch_id)],
                            membership.patch_cell_index,
                            slab_id,
                            rhs_state_cache);
                        scatter_rt_local_vector_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].f,
                            local_f,
                            flux_space,
                            membership.patch_cell_index);
                    }
                }
            }
        }

        {
            LocalErrorProblemScopedTiming timer(
                timing_stats != nullptr
                    ? &timing_stats->assemble_g_seconds
                    : nullptr);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                const auto& slab = slab_space.slab(slab_id);
                for (const int slab_cell_id : slab.active_cells())
                {
                    const int membership_count =
                        patch_set.cell_patch_count(slab_id, slab_cell_id);
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const int patch_id = membership.patch_id;
                        const auto& scalar_space =
                            scalar_spaces[static_cast<std::size_t>(patch_id)];

                        la::local::FixedLocalVector<
                            PatchScalarSpaceType::local_dofs_v> local_g;
                        finite_element::assembly::detail::zero_local_vector(local_g);
                        accumulate_patch_scalar_rhs_on_cell_time_2d_from_rhs_cache<
                            QSpace,
                            QTime>(
                            local_g,
                            scalar_space,
                            tables[static_cast<std::size_t>(patch_id)],
                            membership.patch_cell_index,
                            slab_id,
                            rhs_state_cache);
                        scatter_scalar_local_vector_dense_2d(
                            blocks[static_cast<std::size_t>(patch_id)].g,
                            local_g,
                            scalar_space,
                            membership.patch_cell_index);
                    }
                }
            }
        }

    }

    template<
        int QSpace,
        int QTime,
        class PatchSetType,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType,
        class RTCellCacheType,
        class ABElementCacheType,
        class RHSStateCacheType>
    requires (PatchFluxSpaceType::GT::dim_space_v == 2) &&
             requires { PatchFluxSpaceType::p_time_v; PatchScalarSpaceType::p_time_v; }
    [[nodiscard]] std::vector<DenseLocalErrorBlocks>
    assemble_dense_local_error_blocks_cell_first_time_2d(
        const PatchSetType& patch_set,
        const std::vector<PatchFluxSpaceType>& flux_spaces,
        const std::vector<PatchScalarSpaceType>& scalar_spaces,
        const RTCellCacheType& rt_cell_cache,
        const ABElementCacheType& ab_element_cache,
        const RHSStateCacheType& rhs_state_cache,
        LocalErrorProblemTimingStats* timing_stats = nullptr,
        double zero_tol = 1.0e-15)
    {
        const int patch_count = patch_set.n_patches();
        std::vector<DenseLocalErrorBlocks> blocks;
        blocks.reserve(static_cast<std::size_t>(patch_count));
        for (int patch_id = 0; patch_id < patch_count; ++patch_id)
        {
            blocks.emplace_back(
                flux_spaces[static_cast<std::size_t>(patch_id)].n_dofs(),
                scalar_spaces[static_cast<std::size_t>(patch_id)].n_dofs());
        }

        assemble_dense_local_error_blocks_cell_first_time_2d_into<
            QSpace,
            QTime>(
            patch_set,
            flux_spaces,
            scalar_spaces,
            rt_cell_cache,
            ab_element_cache,
            rhs_state_cache,
            blocks,
            nullptr,
            timing_stats,
            zero_tol);

        return blocks;
    }
}

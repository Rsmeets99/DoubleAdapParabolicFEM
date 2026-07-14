#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../detail/active_cell_locator_time_slab.hpp"
#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "mat_A_2d.hpp"
#include "shared_local_error_context_2d.hpp"
#include "vec_f_1d.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::error_system
{
    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType>
    class LocalErrorQpointStateCache2D
    {
    public:
        using FluxSpaceType = PatchFluxSpaceType;
        using ScalarSpaceType = PatchScalarSpaceType;
        using GT = typename FluxSpaceType::GT;
        using PatchType = typename FluxSpaceType::Patch;
        using Types = typename PatchType::Types;

        using SpatialReferencePoint =
            typename FluxSpaceType::SpatialReferencePoint;
        using SpaceTimeReferencePoint =
            typename FluxSpaceType::SpaceTimeReferencePoint;
        using SpaceTimePoint = typename Types::SpaceTimePoint;
        using SpatialGradient = coefficients::DiffusionVector<2>;
        using DiffusionTensor = coefficients::DiffusionTensor<2>;
        using VectorValue = typename FluxSpaceType::VectorValue;
        using RTBasisValues = typename FluxSpaceType::LocalValues;
        using RTBasisDivergences = typename FluxSpaceType::LocalDivergences;
        using ScalarBasisValues = typename ScalarSpaceType::LocalValues;
        using LocalAMatrix =
            la::local::FixedLocalMatrix<
                FluxSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v>;
        using LocalBMatrix =
            la::local::FixedLocalMatrix<
                ScalarSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v>;
        using LocalFVector =
            la::local::FixedLocalVector<FluxSpaceType::local_dofs_v>;
        using LocalGVector =
            la::local::FixedLocalVector<ScalarSpaceType::local_dofs_v>;

        static_assert(GT::dim_space_v == 2,
                      "LocalErrorQpointStateCache2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "LocalErrorQpointStateCache2D requires dim_time_v == 1.");

        using RTCellCache =
            finite_element::assembly::detail::
                LocalErrorRTCellQuadratureCache2D<
                    QSpace,
                    QTime,
                    FluxSpaceType>;

        static constexpr int n_spatial_quadrature_points_v =
            RTCellCache::n_spatial_quadrature_points_v;
        static constexpr int n_time_quadrature_points_v =
            RTCellCache::n_time_quadrature_points_v;
        static constexpr int n_quadrature_points_v =
            RTCellCache::n_quadrature_points_v;
        static constexpr auto time_rule = RTCellCache::time_rule;

        struct QuadraturePointData
        {
            SpatialReferencePoint spatial_reference_point{};
            double time_reference_point = 0.0;
            SpaceTimeReferencePoint reference_point{};
            SpaceTimePoint physical_point{};

            double jacobian_weight = 0.0;
            RTBasisValues rt_basis_values{};
            RTBasisDivergences rt_basis_divergences{};
            ScalarBasisValues scalar_basis_values{};

            DiffusionTensor diffusion_tensor{};
            DiffusionTensor inverse_diffusion_tensor{};

            SpatialGradient grad_lambda_tilde{};
            SpatialGradient grad_u_delta{};
            SpatialGradient grad_theta_tilde{};
            double u_time_derivative = 0.0;
            double ell_value = 0.0;
            SpatialGradient M_grad_theta_tilde{};

            std::array<double, 3> partition_of_unity_values{};
            std::array<SpatialGradient, 3> partition_of_unity_gradients{};
        };

        struct CellData
        {
            int slab_id = -1;
            int slab_cell_id = -1;
            int active_slab_cell_ordinal = -1;
            int source_cell_id = -1;
            int x_cell_id = -1;
            double spatial_jacobian_measure = 0.0;
            double time_jacobian_measure = 0.0;
            double jacobian_measure = 0.0;
            std::array<QuadraturePointData, n_quadrature_points_v> points{};
            LocalAMatrix local_A{};
            LocalBMatrix local_B{};
        };

        struct LocalErrorReferenceTables2D
        {
            using SpatialRTValues =
                typename FluxSpaceType::SpatialSpace::LocalValues;
            using SpatialRTDivergences =
                typename FluxSpaceType::SpatialSpace::LocalDivergences;
            using SpatialScalarValues =
                typename ScalarSpaceType::SpatialSpace::LocalValues;

            std::array<SpatialReferencePoint, n_spatial_quadrature_points_v>
                spatial_reference_points{};
            std::array<double, n_spatial_quadrature_points_v>
                spatial_reference_weights{};
            std::array<double, n_time_quadrature_points_v>
                time_reference_points{};
            std::array<double, n_time_quadrature_points_v>
                time_reference_weights{};
            std::array<int, n_quadrature_points_v> spatial_qpoint_index{};
            std::array<int, n_quadrature_points_v> time_qpoint_index{};
            std::array<SpatialRTValues, n_spatial_quadrature_points_v>
                rt_reference_values{};
            std::array<SpatialRTDivergences, n_spatial_quadrature_points_v>
                rt_reference_divergences{};
            std::array<SpatialScalarValues, n_spatial_quadrature_points_v>
                scalar_spatial_basis_values{};
            std::array<typename FluxSpaceType::TimeValues,
                       n_time_quadrature_points_v>
                flux_time_basis_values{};
            std::array<typename ScalarSpaceType::TimeValues,
                       n_time_quadrature_points_v>
                scalar_time_basis_values{};
            std::array<std::array<double, 3>,
                       n_spatial_quadrature_points_v>
                partition_values{};
            std::array<std::array<std::array<double, 2>, 3>,
                       n_spatial_quadrature_points_v>
                partition_reference_gradients{};
            std::array<ScalarBasisValues, n_quadrature_points_v>
                scalar_basis_values{};
            std::array<std::array<int, 2>, FluxSpaceType::local_dofs_v>
                flux_tensor_dof_map{};
            std::array<std::array<int, 2>, ScalarSpaceType::local_dofs_v>
                scalar_tensor_dof_map{};
            std::array<
                std::array<
                    std::array<std::array<double, 2>, 2>,
                    FluxSpaceType::spatial_local_dofs_v>,
                FluxSpaceType::spatial_local_dofs_v>
                rt_spatial_value_moments{};
            std::array<
                std::array<double, FluxSpaceType::n_time_dofs_v>,
                FluxSpaceType::n_time_dofs_v>
                flux_time_mass{};
            std::array<
                std::array<double, FluxSpaceType::spatial_local_dofs_v>,
                ScalarSpaceType::spatial_local_dofs_v>
                scalar_divergence_spatial_coupling{};
            std::array<
                std::array<double, FluxSpaceType::n_time_dofs_v>,
                ScalarSpaceType::n_time_dofs_v>
                scalar_flux_time_coupling{};
        };

        enum class OperatorDiffusionMode
        {
            identity = 0,
            constant = 1,
            variable = 2
        };

        struct OperatorCellState2D
        {
            int active_slab_cell_ordinal = -1;
            int slab_id = -1;
            int slab_cell_id = -1;
            int source_cell_id = -1;
            int x_cell_id = -1;
            double spatial_jacobian_measure = 0.0;
            double time_jacobian_measure = 0.0;
            double jacobian_measure = 0.0;
            OperatorDiffusionMode diffusion_mode =
                OperatorDiffusionMode::variable;
            DiffusionTensor diffusion_tensor{};
            DiffusionTensor inverse_diffusion_tensor{};
            LocalAMatrix local_A{};
            LocalBMatrix local_B{};
        };

        struct RHSQpointState2D
        {
            SpatialGradient grad_theta_tilde{};
            SpatialGradient M_grad_theta_tilde{};
            double u_time_derivative = 0.0;
            double ell_value = 0.0;
            double jacobian_weight = 0.0;
        };

        struct RHSCellState2D
        {
            int active_slab_cell_ordinal = -1;
            int slab_id = -1;
            int slab_cell_id = -1;
            double jacobian_measure = 0.0;
            std::array<RHSQpointState2D, n_quadrature_points_v> points{};
        };

        struct FluxDiagnosticCellState2D
        {
            int active_slab_cell_ordinal = -1;
            int slab_id = -1;
            int slab_cell_id = -1;
            int source_cell_id = -1;
            OperatorDiffusionMode diffusion_mode =
                OperatorDiffusionMode::variable;
            DiffusionTensor diffusion_tensor{};
            DiffusionTensor inverse_diffusion_tensor{};
            std::array<RHSQpointState2D, n_quadrature_points_v> points{};
        };

        struct CompactCellData
        {
            OperatorCellState2D operator_state{};
            RHSCellState2D rhs_state{};
        };

        struct AuditStats
        {
            double state_prepare_total_seconds = 0.0;
            double state_prepare_unique_count_seconds = 0.0;
            double state_prepare_set_allocation_seconds = 0.0;
            double state_prepare_map_index_build_seconds = 0.0;
            double state_prepare_ordinal_map_build_seconds = 0.0;
            double state_prepare_cell_vector_allocation_seconds = 0.0;
            double state_prepare_request_collection_seconds = 0.0;
            double state_prepare_debug_duplicate_request_count = 0.0;
            double state_prepare_memory_mb = 0.0;
            double state_index_mode = 0.0;
            double state_index_flat_lookup_count = 0.0;
            double state_index_map_lookup_count = 0.0;
            double state_index_fallback_hash_lookup_count = 0.0;
            double state_fill_total_seconds = 0.0;
            double state_fill_active_ancestor_lookup_seconds = 0.0;
            double state_fill_geometry_lookup_seconds = 0.0;
            double state_fill_affine_map_seconds = 0.0;
            double state_fill_time_basis_seconds = 0.0;
            double state_fill_spatial_rt_basis_seconds = 0.0;
            double state_fill_scalar_basis_seconds = 0.0;
            double state_fill_partition_of_unity_seconds = 0.0;
            double state_fill_lambda_gradient_seconds = 0.0;
            double state_fill_u_gradient_seconds = 0.0;
            double state_fill_load_evaluation_seconds = 0.0;
            double state_fill_diffusion_evaluation_seconds = 0.0;
            double state_fill_diffusion_inverse_seconds = 0.0;
            double state_fill_local_A_assembly_seconds = 0.0;
            double state_fill_local_B_assembly_seconds = 0.0;
            double state_fill_qpoints_processed = 0.0;
            double state_fill_sampled_qpoints = 0.0;
            double state_fill_state_count = 0.0;
            double coefficient_fast_path_enabled = 0.0;
            double coefficient_fast_path_identity_diffusion_cells = 0.0;
            double coefficient_fast_path_constant_diffusion_cells = 0.0;
            double coefficient_fast_path_zero_load_cells = 0.0;
            double coefficient_fast_path_generic_cells = 0.0;
            double operator_builder_mode = 0.0;
            double local_A_identity_reference_fast_path_count = 0.0;
            double local_A_constant_reference_fast_path_count = 0.0;
            double local_A_variable_generic_path_count = 0.0;
            double local_B_reference_fast_path_count = 0.0;
            double local_A_build_seconds = 0.0;
            double local_B_build_seconds = 0.0;
            double local_A_debug_max_abs_diff = 0.0;
            double local_A_debug_rel_frobenius_diff = 0.0;
            double local_B_debug_max_abs_diff = 0.0;
            double local_B_debug_rel_frobenius_diff = 0.0;
            double compact_state_shadow_enabled = 0.0;
            double compact_state_shadow_sample_count = 0.0;
            double compact_state_reference_rt_basis_max_abs_diff = 0.0;
            double compact_state_reference_scalar_basis_max_abs_diff = 0.0;
            double compact_state_reference_partition_value_max_abs_diff = 0.0;
            double compact_state_reference_partition_gradient_max_abs_diff = 0.0;
            double compact_state_local_A_max_abs_diff = 0.0;
            double compact_state_local_B_max_abs_diff = 0.0;
            double compact_state_rhs_f_max_abs_diff = 0.0;
            double compact_state_rhs_g_max_abs_diff = 0.0;
            double compact_state_grad_theta_max_abs_diff = 0.0;
            double compact_state_u_time_derivative_max_abs_diff = 0.0;
            double old_cell_data_bytes_per_cell =
                static_cast<double>(sizeof(CellData));
            double operator_state_bytes_per_cell =
                static_cast<double>(sizeof(OperatorCellState2D));
            double rhs_state_bytes_per_cell =
                static_cast<double>(sizeof(RHSCellState2D));
            double flux_diagnostic_state_bytes_per_cell =
                static_cast<double>(sizeof(FluxDiagnosticCellState2D));
            double reference_table_memory_mb =
                static_cast<double>(sizeof(LocalErrorReferenceTables2D)) /
                (1024.0 * 1024.0);
            double monolithic_cell_data_constructed_count = 0.0;
            double compact_operator_state_constructed_count = 0.0;
            double compact_rhs_state_constructed_count = 0.0;
            double monolithic_debug_path_used_count = 0.0;
            double thread_context_construction_count = 0.0;
            double thread_context_construction_seconds = 0.0;
            double geometry_cache_construction_seconds = 0.0;
            double ancestor_cache_construction_seconds = 0.0;
            double slab_geometry_cache_construction_seconds = 0.0;

            void add(const AuditStats& other) noexcept
            {
                state_prepare_total_seconds +=
                    other.state_prepare_total_seconds;
                state_prepare_unique_count_seconds +=
                    other.state_prepare_unique_count_seconds;
                state_prepare_set_allocation_seconds +=
                    other.state_prepare_set_allocation_seconds;
                state_prepare_map_index_build_seconds +=
                    other.state_prepare_map_index_build_seconds;
                state_prepare_ordinal_map_build_seconds +=
                    other.state_prepare_ordinal_map_build_seconds;
                state_prepare_cell_vector_allocation_seconds +=
                    other.state_prepare_cell_vector_allocation_seconds;
                state_prepare_request_collection_seconds +=
                    other.state_prepare_request_collection_seconds;
                state_prepare_debug_duplicate_request_count +=
                    other.state_prepare_debug_duplicate_request_count;
                state_prepare_memory_mb =
                    std::max(state_prepare_memory_mb,
                             other.state_prepare_memory_mb);
                state_index_mode =
                    std::max(state_index_mode, other.state_index_mode);
                state_index_flat_lookup_count +=
                    other.state_index_flat_lookup_count;
                state_index_map_lookup_count +=
                    other.state_index_map_lookup_count;
                state_index_fallback_hash_lookup_count +=
                    other.state_index_fallback_hash_lookup_count;
                state_fill_total_seconds += other.state_fill_total_seconds;
                state_fill_active_ancestor_lookup_seconds +=
                    other.state_fill_active_ancestor_lookup_seconds;
                state_fill_geometry_lookup_seconds +=
                    other.state_fill_geometry_lookup_seconds;
                state_fill_affine_map_seconds +=
                    other.state_fill_affine_map_seconds;
                state_fill_time_basis_seconds +=
                    other.state_fill_time_basis_seconds;
                state_fill_spatial_rt_basis_seconds +=
                    other.state_fill_spatial_rt_basis_seconds;
                state_fill_scalar_basis_seconds +=
                    other.state_fill_scalar_basis_seconds;
                state_fill_partition_of_unity_seconds +=
                    other.state_fill_partition_of_unity_seconds;
                state_fill_lambda_gradient_seconds +=
                    other.state_fill_lambda_gradient_seconds;
                state_fill_u_gradient_seconds +=
                    other.state_fill_u_gradient_seconds;
                state_fill_load_evaluation_seconds +=
                    other.state_fill_load_evaluation_seconds;
                state_fill_diffusion_evaluation_seconds +=
                    other.state_fill_diffusion_evaluation_seconds;
                state_fill_diffusion_inverse_seconds +=
                    other.state_fill_diffusion_inverse_seconds;
                state_fill_local_A_assembly_seconds +=
                    other.state_fill_local_A_assembly_seconds;
                state_fill_local_B_assembly_seconds +=
                    other.state_fill_local_B_assembly_seconds;
                state_fill_qpoints_processed +=
                    other.state_fill_qpoints_processed;
                state_fill_sampled_qpoints += other.state_fill_sampled_qpoints;
                state_fill_state_count += other.state_fill_state_count;
                coefficient_fast_path_enabled =
                    std::max(coefficient_fast_path_enabled,
                             other.coefficient_fast_path_enabled);
                coefficient_fast_path_identity_diffusion_cells +=
                    other.coefficient_fast_path_identity_diffusion_cells;
                coefficient_fast_path_constant_diffusion_cells +=
                    other.coefficient_fast_path_constant_diffusion_cells;
                coefficient_fast_path_zero_load_cells +=
                    other.coefficient_fast_path_zero_load_cells;
                coefficient_fast_path_generic_cells +=
                    other.coefficient_fast_path_generic_cells;
                operator_builder_mode =
                    std::max(operator_builder_mode,
                             other.operator_builder_mode);
                local_A_identity_reference_fast_path_count +=
                    other.local_A_identity_reference_fast_path_count;
                local_A_constant_reference_fast_path_count +=
                    other.local_A_constant_reference_fast_path_count;
                local_A_variable_generic_path_count +=
                    other.local_A_variable_generic_path_count;
                local_B_reference_fast_path_count +=
                    other.local_B_reference_fast_path_count;
                local_A_build_seconds += other.local_A_build_seconds;
                local_B_build_seconds += other.local_B_build_seconds;
                local_A_debug_max_abs_diff = std::max(
                    local_A_debug_max_abs_diff,
                    other.local_A_debug_max_abs_diff);
                local_A_debug_rel_frobenius_diff = std::max(
                    local_A_debug_rel_frobenius_diff,
                    other.local_A_debug_rel_frobenius_diff);
                local_B_debug_max_abs_diff = std::max(
                    local_B_debug_max_abs_diff,
                    other.local_B_debug_max_abs_diff);
                local_B_debug_rel_frobenius_diff = std::max(
                    local_B_debug_rel_frobenius_diff,
                    other.local_B_debug_rel_frobenius_diff);
                compact_state_shadow_enabled =
                    std::max(compact_state_shadow_enabled,
                             other.compact_state_shadow_enabled);
                compact_state_shadow_sample_count +=
                    other.compact_state_shadow_sample_count;
                compact_state_reference_rt_basis_max_abs_diff = std::max(
                    compact_state_reference_rt_basis_max_abs_diff,
                    other.compact_state_reference_rt_basis_max_abs_diff);
                compact_state_reference_scalar_basis_max_abs_diff = std::max(
                    compact_state_reference_scalar_basis_max_abs_diff,
                    other.compact_state_reference_scalar_basis_max_abs_diff);
                compact_state_reference_partition_value_max_abs_diff = std::max(
                    compact_state_reference_partition_value_max_abs_diff,
                    other.compact_state_reference_partition_value_max_abs_diff);
                compact_state_reference_partition_gradient_max_abs_diff = std::max(
                    compact_state_reference_partition_gradient_max_abs_diff,
                    other.compact_state_reference_partition_gradient_max_abs_diff);
                compact_state_local_A_max_abs_diff = std::max(
                    compact_state_local_A_max_abs_diff,
                    other.compact_state_local_A_max_abs_diff);
                compact_state_local_B_max_abs_diff = std::max(
                    compact_state_local_B_max_abs_diff,
                    other.compact_state_local_B_max_abs_diff);
                compact_state_rhs_f_max_abs_diff = std::max(
                    compact_state_rhs_f_max_abs_diff,
                    other.compact_state_rhs_f_max_abs_diff);
                compact_state_rhs_g_max_abs_diff = std::max(
                    compact_state_rhs_g_max_abs_diff,
                    other.compact_state_rhs_g_max_abs_diff);
                compact_state_grad_theta_max_abs_diff = std::max(
                    compact_state_grad_theta_max_abs_diff,
                    other.compact_state_grad_theta_max_abs_diff);
                compact_state_u_time_derivative_max_abs_diff = std::max(
                    compact_state_u_time_derivative_max_abs_diff,
                    other.compact_state_u_time_derivative_max_abs_diff);
                old_cell_data_bytes_per_cell = std::max(
                    old_cell_data_bytes_per_cell,
                    other.old_cell_data_bytes_per_cell);
                operator_state_bytes_per_cell = std::max(
                    operator_state_bytes_per_cell,
                    other.operator_state_bytes_per_cell);
                rhs_state_bytes_per_cell = std::max(
                    rhs_state_bytes_per_cell,
                    other.rhs_state_bytes_per_cell);
                flux_diagnostic_state_bytes_per_cell = std::max(
                    flux_diagnostic_state_bytes_per_cell,
                    other.flux_diagnostic_state_bytes_per_cell);
                reference_table_memory_mb = std::max(
                    reference_table_memory_mb,
                    other.reference_table_memory_mb);
                monolithic_cell_data_constructed_count +=
                    other.monolithic_cell_data_constructed_count;
                compact_operator_state_constructed_count +=
                    other.compact_operator_state_constructed_count;
                compact_rhs_state_constructed_count +=
                    other.compact_rhs_state_constructed_count;
                monolithic_debug_path_used_count +=
                    other.monolithic_debug_path_used_count;
                thread_context_construction_count +=
                    other.thread_context_construction_count;
                thread_context_construction_seconds +=
                    other.thread_context_construction_seconds;
                geometry_cache_construction_seconds +=
                    other.geometry_cache_construction_seconds;
                ancestor_cache_construction_seconds +=
                    other.ancestor_cache_construction_seconds;
                slab_geometry_cache_construction_seconds +=
                    other.slab_geometry_cache_construction_seconds;
            }

            [[nodiscard]] double qpoint_sample_scale() const noexcept
            {
                return state_fill_sampled_qpoints > 0.0
                           ? state_fill_qpoints_processed /
                                 state_fill_sampled_qpoints
                           : 0.0;
            }
        };

        LocalErrorQpointStateCache2D() = default;

        void set_cell_state_representation(std::string representation)
        {
            std::transform(
                representation.begin(),
                representation.end(),
                representation.begin(),
                [](unsigned char c) -> char
                {
                    if (c == '-' || c == ' ')
                        return '_';
                    return static_cast<char>(std::tolower(c));
                });
            if (representation != "monolithic_debug" &&
                representation != "compact_split")
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: invalid cell-state representation.");
            }
            compact_split_representation_ =
                representation == "compact_split";
        }

        [[nodiscard]] bool compact_split_representation() const noexcept
        {
            return compact_split_representation_;
        }

        void prepare_from_spaces(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces)
        {
            if (flux_spaces.size() != scalar_spaces.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: flux/scalar space count mismatch.");
            }

            clear_storage_();
            cell_index_by_key_.clear();
            cell_index_by_active_ordinal_.clear();
            build_requests_.clear();
            requested_patch_cells_ = 0;
            const auto unique_slab_cell_count =
                count_unique_slab_cells_from_flux_spaces_(flux_spaces);
            reserve_storage_(unique_slab_cell_count);
            build_requests_.reserve(unique_slab_cell_count);

            for (std::size_t patch_id = 0;
                 patch_id < flux_spaces.size();
                 ++patch_id)
            {
                collect_patch_cells_(
                    static_cast<int>(patch_id),
                    flux_spaces[patch_id],
                    scalar_spaces[patch_id]);
            }
        }

        void prepare_from_patch_cells(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces,
            const std::vector<
                finite_element::assembly::detail::
                    LocalErrorPatchCellBuildRequest2D>& requests,
            AuditStats* audit_stats = nullptr)
        {
            const auto prepare_begin = Clock::now();
            if (flux_spaces.size() != scalar_spaces.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: flux/scalar space count mismatch.");
            }

            clear_storage_();
            cell_index_by_key_.clear();
            cell_index_by_active_ordinal_.clear();
            build_requests_.clear();
            requested_patch_cells_ = 0;
            const auto unique_count_begin = Clock::now();
            const auto unique_slab_cell_count =
                count_unique_slab_cells_from_requests_(
                    flux_spaces,
                    requests);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_unique_count_seconds,
                unique_count_begin);
            if (audit_stats != nullptr)
            {
                audit_stats->state_prepare_set_allocation_seconds +=
                    std::chrono::duration<double>(
                        Clock::now() - unique_count_begin)
                        .count();
            }
            const auto allocation_begin = Clock::now();
            reserve_storage_(unique_slab_cell_count);
            build_requests_.reserve(unique_slab_cell_count);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_cell_vector_allocation_seconds,
                allocation_begin);

            const auto collection_begin = Clock::now();
            for (const auto& request : requests)
            {
                if (request.patch_id < 0 ||
                    request.patch_id >= static_cast<int>(flux_spaces.size()))
                {
                    throw std::runtime_error(
                        "LocalErrorQpointStateCache2D: patch id out of range.");
                }
                collect_patch_cell_(
                    request.patch_id,
                    flux_spaces[static_cast<std::size_t>(request.patch_id)],
                    scalar_spaces[static_cast<std::size_t>(request.patch_id)],
                    request.patch_cell_index,
                    audit_stats);
            }
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_request_collection_seconds,
                collection_begin);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_total_seconds,
                prepare_begin);
            if (audit_stats != nullptr)
            {
                audit_stats->state_index_mode = 0.0;
                audit_stats->state_prepare_memory_mb =
                    std::max(
                        audit_stats->state_prepare_memory_mb,
                        static_cast<double>(estimated_memory_bytes()) /
                            (1024.0 * 1024.0));
            }
        }

        template<class SharedContextType>
        void prepare_from_patch_cells_flat(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces,
            const std::vector<
                finite_element::assembly::detail::
                    LocalErrorPatchCellBuildRequest2D>& requests,
            const SharedContextType& shared_context,
            AuditStats* audit_stats = nullptr,
            bool debug_check_unique_ordinals = false)
        {
            const auto prepare_begin = Clock::now();
            if (flux_spaces.size() != scalar_spaces.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: flux/scalar space count mismatch.");
            }

            clear_storage_();
            cell_index_by_key_.clear();
            cell_index_by_active_ordinal_.clear();
            build_requests_.clear();
            requested_patch_cells_ = 0;
            if (audit_stats != nullptr)
                audit_stats->state_index_mode = 1.0;

            const auto ordinal_begin = Clock::now();
            cell_index_by_active_ordinal_.clear();
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_ordinal_map_build_seconds,
                ordinal_begin);

            const auto allocation_begin = Clock::now();
            reserve_storage_(requests.size());
            build_requests_.reserve(requests.size());
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_cell_vector_allocation_seconds,
                allocation_begin);

            if (debug_check_unique_ordinals && audit_stats != nullptr)
            {
                std::set<int> seen_ordinals;
                int duplicate_count = 0;
                for (const auto& request : requests)
                {
                    if (request.active_slab_cell_ordinal < 0)
                        continue;
                    if (!seen_ordinals.insert(
                            request.active_slab_cell_ordinal)
                             .second)
                    {
                        ++duplicate_count;
                    }
                }
                audit_stats->state_prepare_debug_duplicate_request_count +=
                    static_cast<double>(duplicate_count);
            }

            const auto collection_begin = Clock::now();
            for (const auto& request : requests)
            {
                if (request.patch_id < 0 ||
                    request.patch_id >= static_cast<int>(flux_spaces.size()))
                {
                    throw std::runtime_error(
                        "LocalErrorQpointStateCache2D: patch id out of range.");
                }
                collect_patch_cell_flat_(
                    request.patch_id,
                    flux_spaces[static_cast<std::size_t>(request.patch_id)],
                    scalar_spaces[static_cast<std::size_t>(request.patch_id)],
                    request.patch_cell_index,
                    request.active_slab_cell_ordinal,
                    shared_context,
                    audit_stats);
            }
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_request_collection_seconds,
                collection_begin);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_total_seconds,
                prepare_begin);
            if (audit_stats != nullptr)
            {
                audit_stats->state_prepare_memory_mb =
                    std::max(
                        audit_stats->state_prepare_memory_mb,
                        static_cast<double>(estimated_memory_bytes()) /
                            (1024.0 * 1024.0));
            }
        }

        [[nodiscard]] int n_build_requests() const noexcept
        {
            return static_cast<int>(build_requests_.size());
        }

        [[nodiscard]] int build_request_slab_id(int request_id) const
        {
            check_build_request_index_(request_id);
            return build_requests_[static_cast<std::size_t>(request_id)]
                .slab_id;
        }

        [[nodiscard]] int build_request_slab_cell_id(int request_id) const
        {
            check_build_request_index_(request_id);
            return build_requests_[static_cast<std::size_t>(request_id)]
                .slab_cell_id;
        }

        [[nodiscard]] int build_request_active_slab_cell_ordinal(
            int request_id) const
        {
            check_build_request_index_(request_id);
            return build_requests_[static_cast<std::size_t>(request_id)]
                .active_slab_cell_ordinal;
        }

        template<
            class XSpaceType,
            class SlabSpaceType,
            class ReconstructedFunctionType,
            class XFunctionType,
            class EllFunction,
            class MFunction>
        void fill_build_request(
            int request_id,
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces,
            const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M,
            AuditStats* audit_stats = nullptr,
            bool coefficient_fast_path = true,
            bool compact_state_shadow = false)
        {
            check_build_request_index_(request_id);
            const auto& request =
                build_requests_[static_cast<std::size_t>(request_id)];
            if (compact_split_representation_)
            {
                auto& compact_cell =
                    compact_cells_[static_cast<std::size_t>(
                        request.cache_id)];
                compact_cell.operator_state.active_slab_cell_ordinal =
                    request.active_slab_cell_ordinal;
                compact_cell.rhs_state.active_slab_cell_ordinal =
                    request.active_slab_cell_ordinal;
                fill_compact_cell_(
                    flux_spaces[static_cast<std::size_t>(request.patch_id)],
                    scalar_spaces[static_cast<std::size_t>(request.patch_id)],
                    request.patch_cell_index,
                    context,
                    lambda_tilde,
                    u_delta,
                    ell,
                    M,
                    compact_cell,
                    audit_stats,
                    coefficient_fast_path);
            }
            else
            {
                auto& cell_data =
                    cells_[static_cast<std::size_t>(request.cache_id)];
                cell_data.active_slab_cell_ordinal =
                    request.active_slab_cell_ordinal;
                fill_cell_(
                    flux_spaces[static_cast<std::size_t>(request.patch_id)],
                    scalar_spaces[static_cast<std::size_t>(request.patch_id)],
                    request.patch_cell_index,
                    context,
                    lambda_tilde,
                    u_delta,
                    ell,
                    M,
                    cell_data,
                    audit_stats,
                    coefficient_fast_path,
                    compact_state_shadow);
            }
        }

        [[nodiscard]] int requested_patch_cells() const noexcept
        {
            return requested_patch_cells_;
        }

        [[nodiscard]] int unique_slab_cells() const noexcept
        {
            return static_cast<int>(storage_size_());
        }

        [[nodiscard]] int duplicate_patch_cells() const noexcept
        {
            return requested_patch_cells_ - unique_slab_cells();
        }

        [[nodiscard]] double average_patch_cells_per_slab_cell() const noexcept
        {
            if (storage_size_() == 0)
                return 0.0;
            return static_cast<double>(requested_patch_cells_) /
                   static_cast<double>(storage_size_());
        }

        [[nodiscard]] std::size_t qpoint_fills() const noexcept
        {
            return storage_size_() *
                   static_cast<std::size_t>(n_quadrature_points_v);
        }

        [[nodiscard]] std::size_t rt_basis_qpoint_fills() const noexcept
        {
            return qpoint_fills();
        }

        [[nodiscard]] std::size_t scalar_basis_qpoint_fills() const noexcept
        {
            return qpoint_fills();
        }

        [[nodiscard]] std::size_t
        patch_equivalent_scalar_basis_qpoint_fills() const noexcept
        {
            return static_cast<std::size_t>(requested_patch_cells_) *
                   static_cast<std::size_t>(n_quadrature_points_v);
        }

        [[nodiscard]] std::size_t
        scalar_basis_qpoint_fills_avoided() const noexcept
        {
            const auto equivalent =
                patch_equivalent_scalar_basis_qpoint_fills();
            const auto shared = scalar_basis_qpoint_fills();
            return equivalent > shared ? equivalent - shared : 0u;
        }

        [[nodiscard]] std::size_t
        partition_of_unity_qpoint_fills() const noexcept
        {
            return 3u * qpoint_fills();
        }

        [[nodiscard]] std::size_t
        patch_equivalent_partition_of_unity_qpoint_fills() const noexcept
        {
            return static_cast<std::size_t>(requested_patch_cells_) *
                   static_cast<std::size_t>(n_quadrature_points_v);
        }

        [[nodiscard]] std::size_t
        partition_of_unity_qpoint_fills_avoided() const noexcept
        {
            const auto equivalent =
                patch_equivalent_partition_of_unity_qpoint_fills();
            const auto shared = partition_of_unity_qpoint_fills();
            return equivalent > shared ? equivalent - shared : 0u;
        }

        [[nodiscard]] std::size_t diffusion_tensor_evaluations() const noexcept
        {
            return qpoint_fills();
        }

        [[nodiscard]] std::size_t lambda_gradient_evaluations() const noexcept
        {
            return qpoint_fills();
        }

        [[nodiscard]] std::size_t u_gradient_evaluations() const noexcept
        {
            return qpoint_fills();
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return cells_.capacity() * sizeof(CellData) +
                   compact_cells_.capacity() * sizeof(CompactCellData) +
                   build_requests_.capacity() * (6 * sizeof(int)) +
                   cell_index_by_key_.size() *
                       (sizeof(std::pair<const std::pair<int, int>, int>) +
                        3 * sizeof(void*)) +
                   cell_index_by_active_ordinal_.capacity() * sizeof(int);
        }

        [[nodiscard]] const CellData& cell(
            int slab_id,
            int slab_cell_id) const
        {
            const auto it = cell_index_by_key_.find({slab_id, slab_cell_id});
            if (it == cell_index_by_key_.end())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D::cell: slab cell not cached.");
            }

            return cells_[static_cast<std::size_t>(it->second)];
        }

        [[nodiscard]] const CellData& cell_by_ordinal(
            int active_slab_cell_ordinal) const
        {
            const int cache_id =
                cache_id_for_active_ordinal_(active_slab_cell_ordinal);
            return cells_[static_cast<std::size_t>(cache_id)];
        }

        [[nodiscard]] const CompactCellData& compact_cell_by_ordinal(
            int active_slab_cell_ordinal) const
        {
            const int cache_id =
                cache_id_for_active_ordinal_(active_slab_cell_ordinal);
            return compact_cells_[static_cast<std::size_t>(cache_id)];
        }

        [[nodiscard]] const CompactCellData& compact_cell_by_index(
            int cache_id) const
        {
            if (cache_id < 0 ||
                static_cast<std::size_t>(cache_id) >= compact_cells_.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: compact cache index out of range.");
            }
            return compact_cells_[static_cast<std::size_t>(cache_id)];
        }

        [[nodiscard]] const CellData& cell_by_index(int cache_id) const
        {
            if (cache_id < 0 ||
                static_cast<std::size_t>(cache_id) >= cells_.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: cache index out of range.");
            }
            return cells_[static_cast<std::size_t>(cache_id)];
        }

        [[nodiscard]] CompactCellData& mutable_compact_cell_by_index(
            int cache_id)
        {
            if (cache_id < 0 ||
                static_cast<std::size_t>(cache_id) >= compact_cells_.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: compact cache index out of range.");
            }
            return compact_cells_[static_cast<std::size_t>(cache_id)];
        }

        [[nodiscard]] CellData& mutable_cell(
            int slab_id,
            int slab_cell_id)
        {
            const auto it = cell_index_by_key_.find({slab_id, slab_cell_id});
            if (it == cell_index_by_key_.end())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D::mutable_cell: slab cell not cached.");
            }

            return cells_[static_cast<std::size_t>(it->second)];
        }

        [[nodiscard]] CellData& mutable_cell_by_ordinal(
            int active_slab_cell_ordinal)
        {
            const int cache_id =
                cache_id_for_active_ordinal_(active_slab_cell_ordinal);
            return cells_[static_cast<std::size_t>(cache_id)];
        }

        [[nodiscard]] CellData& mutable_cell_by_index(int cache_id)
        {
            if (cache_id < 0 ||
                static_cast<std::size_t>(cache_id) >= cells_.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: cache index out of range.");
            }
            return cells_[static_cast<std::size_t>(cache_id)];
        }

        [[nodiscard]] bool has_cell(
            int slab_id,
            int slab_cell_id) const
        {
            return cell_index_by_key_.find({slab_id, slab_cell_id}) !=
                   cell_index_by_key_.end();
        }

        [[nodiscard]] bool has_cell_ordinal(
            int active_slab_cell_ordinal) const noexcept
        {
            return active_slab_cell_ordinal >= 0 &&
                   static_cast<std::size_t>(active_slab_cell_ordinal) <
                       cell_index_by_active_ordinal_.size() &&
                   cell_index_by_active_ordinal_[
                       static_cast<std::size_t>(
                           active_slab_cell_ordinal)] >= 0;
        }

        void store_cell(const CellData& cell_data)
        {
            const auto key =
                std::pair<int, int>{cell_data.slab_id, cell_data.slab_cell_id};
            const auto it = cell_index_by_key_.find(key);
            if (it != cell_index_by_key_.end())
            {
                cells_[static_cast<std::size_t>(it->second)] = cell_data;
                set_active_ordinal_index_(cell_data.active_slab_cell_ordinal,
                                          it->second);
                return;
            }

            const int cache_id = static_cast<int>(cells_.size());
            cell_index_by_key_.emplace(key, cache_id);
            cells_.emplace_back(cell_data);
            set_active_ordinal_index_(cell_data.active_slab_cell_ordinal,
                                      cache_id);
        }

        void store_cell(CellData&& cell_data)
        {
            const auto key =
                std::pair<int, int>{cell_data.slab_id, cell_data.slab_cell_id};
            const auto it = cell_index_by_key_.find(key);
            if (it != cell_index_by_key_.end())
            {
                const int active_slab_cell_ordinal =
                    cell_data.active_slab_cell_ordinal;
                cells_[static_cast<std::size_t>(it->second)] =
                    std::move(cell_data);
                set_active_ordinal_index_(active_slab_cell_ordinal,
                                          it->second);
                return;
            }

            const int cache_id = static_cast<int>(cells_.size());
            cell_index_by_key_.emplace(key, cache_id);
            cells_.emplace_back(std::move(cell_data));
            set_active_ordinal_index_(
                cells_[static_cast<std::size_t>(cache_id)]
                    .active_slab_cell_ordinal,
                cache_id);
        }

        void store_compact_cell(CompactCellData&& cell_data)
        {
            const int slab_id = cell_data.operator_state.slab_id;
            const int slab_cell_id = cell_data.operator_state.slab_cell_id;
            const int active_slab_cell_ordinal =
                cell_data.operator_state.active_slab_cell_ordinal;
            const auto key = std::pair<int, int>{slab_id, slab_cell_id};
            const auto it = cell_index_by_key_.find(key);
            if (it != cell_index_by_key_.end())
            {
                compact_cells_[static_cast<std::size_t>(it->second)] =
                    std::move(cell_data);
                set_active_ordinal_index_(
                    active_slab_cell_ordinal,
                    it->second);
                return;
            }

            const int cache_id = static_cast<int>(compact_cells_.size());
            cell_index_by_key_.emplace(key, cache_id);
            compact_cells_.emplace_back(std::move(cell_data));
            set_active_ordinal_index_(active_slab_cell_ordinal, cache_id);
        }

        [[nodiscard]] int stored_cell_count() const noexcept
        {
            return static_cast<int>(storage_size_());
        }

        [[nodiscard]] static std::size_t compact_cell_data_bytes()
        {
            return sizeof(CompactCellData);
        }

        [[nodiscard]] static const LocalErrorReferenceTables2D&
        reference_tables()
        {
            return reference_tables_();
        }

        template<class AffineMap>
        [[nodiscard]] static RTBasisValues
        reconstruct_rt_values_from_reference(
            const AffineMap& map,
            int qp_id)
        {
            return reconstruct_rt_values_(reference_tables_(), map, qp_id);
        }

        template<class AffineMap>
        [[nodiscard]] static RTBasisDivergences
        reconstruct_rt_divergences_from_reference(
            const AffineMap& map,
            int qp_id)
        {
            return reconstruct_rt_divergences_(reference_tables_(), map, qp_id);
        }

        [[nodiscard]] static FluxDiagnosticCellState2D
        flux_diagnostic_state_from_compact_state(
            const CompactCellData& state_cell)
        {
            FluxDiagnosticCellState2D diagnostic{};
            diagnostic.active_slab_cell_ordinal =
                state_cell.operator_state.active_slab_cell_ordinal;
            diagnostic.slab_id = state_cell.operator_state.slab_id;
            diagnostic.slab_cell_id = state_cell.operator_state.slab_cell_id;
            diagnostic.source_cell_id =
                state_cell.operator_state.source_cell_id;
            diagnostic.diffusion_mode =
                state_cell.operator_state.diffusion_mode;
            diagnostic.diffusion_tensor =
                state_cell.operator_state.diffusion_tensor;
            diagnostic.inverse_diffusion_tensor =
                state_cell.operator_state.inverse_diffusion_tensor;
            diagnostic.points = state_cell.rhs_state.points;
            return diagnostic;
        }

        template<class LocalVectorType>
        static void accumulate_patch_flux_rhs_from_compact_state(
            LocalVectorType& local,
            const FluxSpaceType& flux_space,
            const CompactCellData& state_cell,
            int patch_cell_index)
        {
            const int local_vertex_index =
                flux_space.patch().cell(patch_cell_index).local_vertex_index;
            const auto map =
                flux_space.physical_map_for_patch_cell(patch_cell_index);
            const auto& tables = reference_tables_();

            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const auto& qp =
                    state_cell.rhs_state.points[
                        static_cast<std::size_t>(qp_id)];
                const int spatial_qp_id =
                    tables.spatial_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const double psi_q =
                    tables.partition_values[
                        static_cast<std::size_t>(spatial_qp_id)][
                        static_cast<std::size_t>(local_vertex_index)];
                const auto rt_values =
                    reconstruct_rt_values_(tables, map, qp_id);
                for (int i = 0; i < FluxSpaceType::local_dofs_v; ++i)
                {
                    const auto& sigma_i =
                        rt_values[static_cast<std::size_t>(i)];
                    local[i] +=
                        -psi_q *
                        (qp.grad_theta_tilde[0] * sigma_i[0] +
                         qp.grad_theta_tilde[1] * sigma_i[1]) *
                        qp.jacobian_weight;
                }
            }
        }

        template<class LocalVectorType>
        static void accumulate_patch_scalar_rhs_from_compact_state(
            LocalVectorType& local,
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            const CompactCellData& state_cell,
            int patch_cell_index)
        {
            const int local_vertex_index =
                scalar_space.patch().cell(patch_cell_index).local_vertex_index;
            const auto map =
                flux_space.physical_map_for_patch_cell(patch_cell_index);
            const auto& tables = reference_tables_();

            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const auto& qp =
                    state_cell.rhs_state.points[
                        static_cast<std::size_t>(qp_id)];
                const int spatial_qp_id =
                    tables.spatial_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const double psi_q =
                    tables.partition_values[
                        static_cast<std::size_t>(spatial_qp_id)][
                        static_cast<std::size_t>(local_vertex_index)];
                const auto grad_psi =
                    map_reference_gradient_to_physical_(
                        map,
                        tables.partition_reference_gradients[
                            static_cast<std::size_t>(spatial_qp_id)][
                            static_cast<std::size_t>(local_vertex_index)]);
                const double grad_psi_M_grad_theta =
                    grad_psi[0] * qp.M_grad_theta_tilde[0] +
                    grad_psi[1] * qp.M_grad_theta_tilde[1];
                const double rhs =
                    psi_q * (qp.ell_value - qp.u_time_derivative) -
                    grad_psi_M_grad_theta;
                const double rhs_dmu = rhs * qp.jacobian_weight;
                const auto& scalar_values =
                    tables.scalar_basis_values[
                        static_cast<std::size_t>(qp_id)];
                for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
                {
                    local[i] +=
                        rhs_dmu *
                        scalar_values[static_cast<std::size_t>(i)];
                }
            }
        }

    private:
        using Clock = std::chrono::steady_clock;

        struct BuildRequest
        {
            int cache_id = -1;
            int patch_id = -1;
            int patch_cell_index = -1;
            int slab_id = -1;
            int slab_cell_id = -1;
            int active_slab_cell_ordinal = -1;
        };

        std::vector<CellData> cells_{};
        std::vector<CompactCellData> compact_cells_{};
        std::map<std::pair<int, int>, int> cell_index_by_key_{};
        std::vector<int> cell_index_by_active_ordinal_{};
        std::vector<BuildRequest> build_requests_{};
        int requested_patch_cells_ = 0;
        bool compact_split_representation_ = false;

        static void add_elapsed_(
            AuditStats* audit_stats,
            double AuditStats::*field,
            Clock::time_point begin)
        {
            if (audit_stats == nullptr)
                return;
            audit_stats->*field +=
                std::chrono::duration<double>(Clock::now() - begin).count();
        }

        void clear_storage_()
        {
            cells_.clear();
            compact_cells_.clear();
        }

        void reserve_storage_(std::size_t count)
        {
            if (compact_split_representation_)
                compact_cells_.reserve(count);
            else
                cells_.reserve(count);
        }

        [[nodiscard]] std::size_t storage_size_() const noexcept
        {
            return compact_split_representation_ ? compact_cells_.size()
                                                 : cells_.size();
        }

        [[nodiscard]] int append_storage_cell_()
        {
            const int cache_id = static_cast<int>(storage_size_());
            if (compact_split_representation_)
                compact_cells_.emplace_back();
            else
                cells_.emplace_back();
            return cache_id;
        }

        void set_storage_cell_metadata_(
            int cache_id,
            int slab_id,
            int slab_cell_id,
            int active_slab_cell_ordinal)
        {
            if (compact_split_representation_)
            {
                auto& compact_cell =
                    compact_cells_[static_cast<std::size_t>(cache_id)];
                compact_cell.operator_state.slab_id = slab_id;
                compact_cell.operator_state.slab_cell_id = slab_cell_id;
                compact_cell.operator_state.active_slab_cell_ordinal =
                    active_slab_cell_ordinal;
                compact_cell.rhs_state.slab_id = slab_id;
                compact_cell.rhs_state.slab_cell_id = slab_cell_id;
                compact_cell.rhs_state.active_slab_cell_ordinal =
                    active_slab_cell_ordinal;
            }
            else
            {
                auto& cell_data = cells_[static_cast<std::size_t>(cache_id)];
                cell_data.slab_id = slab_id;
                cell_data.slab_cell_id = slab_cell_id;
                cell_data.active_slab_cell_ordinal =
                    active_slab_cell_ordinal;
            }
        }

        [[nodiscard]] static std::size_t
        count_unique_slab_cells_from_flux_spaces_(
            const std::vector<FluxSpaceType>& flux_spaces)
        {
            std::set<std::pair<int, int>> keys;
            for (const auto& flux_space : flux_spaces)
            {
                const int slab_id = flux_space.patch().slab_id;
                for (int patch_cell_index = 0;
                     patch_cell_index < flux_space.n_patch_cells();
                     ++patch_cell_index)
                {
                    const int slab_cell_id =
                        flux_space.patch().cell(patch_cell_index)
                            .slab_cell_id;
                    keys.emplace(slab_id, slab_cell_id);
                }
            }
            return keys.size();
        }

        [[nodiscard]] static std::size_t
        count_unique_slab_cells_from_requests_(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<
                finite_element::assembly::detail::
                    LocalErrorPatchCellBuildRequest2D>& requests)
        {
            std::set<std::pair<int, int>> keys;
            for (const auto& request : requests)
            {
                if (request.patch_id < 0 ||
                    request.patch_id >= static_cast<int>(flux_spaces.size()))
                {
                    throw std::runtime_error(
                        "LocalErrorQpointStateCache2D: patch id out of range.");
                }
                const auto& flux_space =
                    flux_spaces[static_cast<std::size_t>(request.patch_id)];
                if (request.patch_cell_index < 0 ||
                    request.patch_cell_index >= flux_space.n_patch_cells())
                {
                    throw std::runtime_error(
                        "LocalErrorQpointStateCache2D: patch cell index out of range.");
                }
                const int slab_id = flux_space.patch().slab_id;
                const int slab_cell_id =
                    flux_space.patch().cell(request.patch_cell_index)
                        .slab_cell_id;
                keys.emplace(slab_id, slab_cell_id);
            }
            return keys.size();
        }

        void collect_patch_cells_(
            int patch_id,
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space)
        {
            if (flux_space.n_patch_cells() != scalar_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: patch cell count mismatch.");
            }

            requested_patch_cells_ += flux_space.n_patch_cells();
            const int slab_id = flux_space.patch().slab_id;

            for (int patch_cell_index = 0;
                 patch_cell_index < flux_space.n_patch_cells();
                 ++patch_cell_index)
            {
                const int slab_cell_id =
                    flux_space.patch().cell(patch_cell_index).slab_cell_id;
                const auto key = std::pair<int, int>{slab_id, slab_cell_id};
                if (cell_index_by_key_.find(key) != cell_index_by_key_.end())
                    continue;

                const int cache_id = append_storage_cell_();
                cell_index_by_key_.emplace(key, cache_id);
                set_storage_cell_metadata_(
                    cache_id,
                    slab_id,
                    slab_cell_id,
                    -1);
                build_requests_.push_back(
                    BuildRequest{
                        cache_id,
                        patch_id,
                        patch_cell_index,
                        slab_id,
                        slab_cell_id,
                        -1});
            }
        }

        void collect_patch_cell_(
            int patch_id,
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            AuditStats* audit_stats = nullptr)
        {
            if (flux_space.n_patch_cells() != scalar_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: patch cell count mismatch.");
            }
            if (patch_cell_index < 0 ||
                patch_cell_index >= flux_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: patch cell index out of range.");
            }

            ++requested_patch_cells_;
            const int slab_id = flux_space.patch().slab_id;
            const int slab_cell_id =
                flux_space.patch().cell(patch_cell_index).slab_cell_id;
            const auto key = std::pair<int, int>{slab_id, slab_cell_id};
            const auto map_begin = Clock::now();
            if (cell_index_by_key_.find(key) != cell_index_by_key_.end())
            {
                add_elapsed_(
                    audit_stats,
                    &AuditStats::state_prepare_map_index_build_seconds,
                    map_begin);
                return;
            }

            const int cache_id = append_storage_cell_();
            cell_index_by_key_.emplace(key, cache_id);
            set_storage_cell_metadata_(
                cache_id,
                slab_id,
                slab_cell_id,
                -1);
            build_requests_.push_back(
                BuildRequest{
                    cache_id,
                    patch_id,
                        patch_cell_index,
                        slab_id,
                        slab_cell_id,
                        -1});
            add_elapsed_(
                audit_stats,
                &AuditStats::state_prepare_map_index_build_seconds,
                map_begin);
        }

        template<class SharedContextType>
        void collect_patch_cell_flat_(
            int patch_id,
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            int active_slab_cell_ordinal,
            const SharedContextType& shared_context,
            AuditStats* audit_stats = nullptr)
        {
            if (flux_space.n_patch_cells() != scalar_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: patch cell count mismatch.");
            }
            if (patch_cell_index < 0 ||
                patch_cell_index >= flux_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: patch cell index out of range.");
            }

            ++requested_patch_cells_;
            const int slab_id = flux_space.patch().slab_id;
            const int slab_cell_id =
                flux_space.patch().cell(patch_cell_index).slab_cell_id;
            if (active_slab_cell_ordinal < 0)
            {
                active_slab_cell_ordinal =
                    shared_context.slab_cell_ordinal(slab_id, slab_cell_id);
                if (audit_stats != nullptr)
                    audit_stats->state_index_fallback_hash_lookup_count +=
                        1.0;
            }
            if (active_slab_cell_ordinal < 0 ||
                static_cast<std::size_t>(active_slab_cell_ordinal) >=
                    static_cast<std::size_t>(
                        shared_context.active_slab_cell_count()))
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: active slab-cell ordinal out of range.");
            }

            const int cache_id = append_storage_cell_();
            set_storage_cell_metadata_(
                cache_id,
                slab_id,
                slab_cell_id,
                active_slab_cell_ordinal);
            build_requests_.push_back(
                BuildRequest{
                    cache_id,
                    patch_id,
                    patch_cell_index,
                    slab_id,
                    slab_cell_id,
                    active_slab_cell_ordinal});
        }

        [[nodiscard]] int cache_id_for_active_ordinal_(
            int active_slab_cell_ordinal) const
        {
            if (active_slab_cell_ordinal < 0 ||
                static_cast<std::size_t>(active_slab_cell_ordinal) >=
                    cell_index_by_active_ordinal_.size())
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: active slab-cell ordinal not cached.");
            }
            const int cache_id =
                cell_index_by_active_ordinal_[
                    static_cast<std::size_t>(active_slab_cell_ordinal)];
            if (cache_id < 0)
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: active slab-cell ordinal not cached.");
            }
            return cache_id;
        }

        void set_active_ordinal_index_(
            int active_slab_cell_ordinal,
            int cache_id)
        {
            if (active_slab_cell_ordinal < 0)
                return;
            if (static_cast<std::size_t>(active_slab_cell_ordinal) >=
                cell_index_by_active_ordinal_.size())
            {
                cell_index_by_active_ordinal_.resize(
                    static_cast<std::size_t>(active_slab_cell_ordinal) + 1u,
                    -1);
            }
            cell_index_by_active_ordinal_[
                static_cast<std::size_t>(active_slab_cell_ordinal)] =
                cache_id;
        }

        void check_build_request_index_(int request_id) const
        {
            if (request_id < 0 ||
                request_id >= static_cast<int>(build_requests_.size()))
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: build request index out of range.");
            }
        }

        static double barycentric_value_(
            int local_vertex_index,
            const SpatialReferencePoint& x_ref)
        {
            switch (local_vertex_index)
            {
            case 0:
                return 1.0 - x_ref[0] - x_ref[1];
            case 1:
                return x_ref[0];
            case 2:
                return x_ref[1];
            default:
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: invalid local vertex index.");
            }
        }

        static std::array<double, 2> barycentric_reference_gradient_(
            int local_vertex_index)
        {
            switch (local_vertex_index)
            {
            case 0:
                return {-1.0, -1.0};
            case 1:
                return {1.0, 0.0};
            case 2:
                return {0.0, 1.0};
            default:
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: invalid local vertex index.");
            }
        }

        template<class AffineMap>
        static SpatialGradient map_reference_gradient_to_physical_(
            const AffineMap& map,
            const std::array<double, 2>& grad_ref)
        {
            return SpatialGradient{
                map.invJ00 * grad_ref[0] + map.invJ10 * grad_ref[1],
                map.invJ01 * grad_ref[0] + map.invJ11 * grad_ref[1]
            };
        }

        static const LocalErrorReferenceTables2D&
        reference_tables_()
        {
            static const LocalErrorReferenceTables2D tables =
                build_reference_tables_();
            return tables;
        }

        static LocalErrorReferenceTables2D
        build_reference_tables_()
        {
            LocalErrorReferenceTables2D tables{};
            for (int qt = 0; qt < QTime; ++qt)
            {
                const double t_ref = time_rule.points[qt][0];
                tables.time_reference_points[static_cast<std::size_t>(qt)] =
                    t_ref;
                tables.time_reference_weights[static_cast<std::size_t>(qt)] =
                    time_rule.weights[qt];
                FluxSpaceType::evaluate_time_basis(
                    t_ref,
                    tables.flux_time_basis_values[
                        static_cast<std::size_t>(qt)]);
                ScalarSpaceType::evaluate_time_basis(
                    t_ref,
                    tables.scalar_time_basis_values[
                        static_cast<std::size_t>(qt)]);
            }
            for (int ti = 0; ti < FluxSpaceType::n_time_dofs_v; ++ti)
            {
                for (int tj = 0; tj < FluxSpaceType::n_time_dofs_v; ++tj)
                {
                    double value = 0.0;
                    for (int qt = 0; qt < QTime; ++qt)
                    {
                        value +=
                            tables.time_reference_weights[
                                static_cast<std::size_t>(qt)] *
                            tables.flux_time_basis_values[
                                static_cast<std::size_t>(qt)][
                                static_cast<std::size_t>(ti)] *
                            tables.flux_time_basis_values[
                                static_cast<std::size_t>(qt)][
                                static_cast<std::size_t>(tj)];
                    }
                    tables.flux_time_mass[
                        static_cast<std::size_t>(ti)][
                        static_cast<std::size_t>(tj)] = value;
                }
            }
            for (int ti = 0; ti < ScalarSpaceType::n_time_dofs_v; ++ti)
            {
                for (int tj = 0; tj < FluxSpaceType::n_time_dofs_v; ++tj)
                {
                    double value = 0.0;
                    for (int qt = 0; qt < QTime; ++qt)
                    {
                        value +=
                            tables.time_reference_weights[
                                static_cast<std::size_t>(qt)] *
                            tables.scalar_time_basis_values[
                                static_cast<std::size_t>(qt)][
                                static_cast<std::size_t>(ti)] *
                            tables.flux_time_basis_values[
                                static_cast<std::size_t>(qt)][
                                static_cast<std::size_t>(tj)];
                    }
                    tables.scalar_flux_time_coupling[
                        static_cast<std::size_t>(ti)][
                        static_cast<std::size_t>(tj)] = value;
                }
            }

            int spatial_qp_id = 0;
            quadrature::reference::for_each_reference_triangle_duffy_point<
                QSpace>(
                [&](const double x,
                    const double y,
                    const double triangle_weight)
                {
                    const SpatialReferencePoint x_ref{x, y};
                    const std::size_t spatial_index =
                        static_cast<std::size_t>(spatial_qp_id);
                    tables.spatial_reference_points[spatial_index] = x_ref;
                    tables.spatial_reference_weights[spatial_index] =
                        triangle_weight;
                    tables.rt_reference_values[spatial_index] =
                        FluxSpaceType::ReferenceBasis::eval_all(x_ref);
                    tables.rt_reference_divergences[spatial_index] =
                        FluxSpaceType::ReferenceBasis::div_all(x_ref);
                    ScalarSpaceType::SpatialSpace::evaluate_local_basis(
                        x_ref,
                        tables.scalar_spatial_basis_values[spatial_index]);
                    for (int i = 0;
                         i < FluxSpaceType::spatial_local_dofs_v;
                         ++i)
                    {
                        const auto& value_i =
                            tables.rt_reference_values[spatial_index][
                                static_cast<std::size_t>(i)];
                        for (int j = 0;
                             j < FluxSpaceType::spatial_local_dofs_v;
                             ++j)
                        {
                            const auto& value_j =
                                tables.rt_reference_values[spatial_index][
                                    static_cast<std::size_t>(j)];
                            for (int a = 0; a < 2; ++a)
                            {
                                for (int b = 0; b < 2; ++b)
                                {
                                    tables.rt_spatial_value_moments[
                                        static_cast<std::size_t>(i)][
                                        static_cast<std::size_t>(j)][
                                        static_cast<std::size_t>(a)][
                                        static_cast<std::size_t>(b)] +=
                                        triangle_weight *
                                        value_i[
                                            static_cast<std::size_t>(a)] *
                                        value_j[
                                            static_cast<std::size_t>(b)];
                                }
                            }
                        }
                    }
                    for (int i = 0;
                         i < ScalarSpaceType::spatial_local_dofs_v;
                         ++i)
                    {
                        const double scalar_value =
                            tables.scalar_spatial_basis_values[
                                spatial_index][static_cast<std::size_t>(i)];
                        for (int j = 0;
                             j < FluxSpaceType::spatial_local_dofs_v;
                             ++j)
                        {
                            tables.scalar_divergence_spatial_coupling[
                                static_cast<std::size_t>(i)][
                                static_cast<std::size_t>(j)] +=
                                triangle_weight * scalar_value *
                                tables.rt_reference_divergences[
                                    spatial_index][
                                    static_cast<std::size_t>(j)];
                        }
                    }
                    for (int local_vertex = 0;
                         local_vertex < 3;
                         ++local_vertex)
                    {
                        tables.partition_values[spatial_index][
                            static_cast<std::size_t>(local_vertex)] =
                            barycentric_value_(local_vertex, x_ref);
                        tables.partition_reference_gradients[spatial_index][
                            static_cast<std::size_t>(local_vertex)] =
                            barycentric_reference_gradient_(local_vertex);
                    }

                    for (int qt = 0; qt < QTime; ++qt)
                    {
                        const int qp_id = spatial_qp_id * QTime + qt;
                        tables.spatial_qpoint_index[
                            static_cast<std::size_t>(qp_id)] =
                            spatial_qp_id;
                        tables.time_qpoint_index[
                            static_cast<std::size_t>(qp_id)] = qt;
                        for (int spatial_local_dof = 0;
                             spatial_local_dof <
                                 ScalarSpaceType::spatial_local_dofs_v;
                             ++spatial_local_dof)
                        {
                            const double spatial_value =
                                tables.scalar_spatial_basis_values[
                                    spatial_index][
                                    static_cast<std::size_t>(
                                        spatial_local_dof)];
                            for (int time_dof = 0;
                                 time_dof < ScalarSpaceType::n_time_dofs_v;
                                 ++time_dof)
                            {
                                const int local_id =
                                    spatial_local_dof *
                                        ScalarSpaceType::n_time_dofs_v +
                                    time_dof;
                                tables.scalar_basis_values[
                                    static_cast<std::size_t>(qp_id)][
                                    static_cast<std::size_t>(local_id)] =
                                    spatial_value *
                                    tables.scalar_time_basis_values[
                                        static_cast<std::size_t>(qt)][
                                        static_cast<std::size_t>(time_dof)];
                            }
                        }
                    }

                    ++spatial_qp_id;
                });

            for (int spatial_local_dof = 0;
                 spatial_local_dof < FluxSpaceType::spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                for (int time_dof = 0;
                     time_dof < FluxSpaceType::n_time_dofs_v;
                     ++time_dof)
                {
                    const int local_id =
                        spatial_local_dof * FluxSpaceType::n_time_dofs_v +
                        time_dof;
                    tables.flux_tensor_dof_map[
                        static_cast<std::size_t>(local_id)] =
                        {spatial_local_dof, time_dof};
                }
            }
            for (int spatial_local_dof = 0;
                 spatial_local_dof < ScalarSpaceType::spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                for (int time_dof = 0;
                     time_dof < ScalarSpaceType::n_time_dofs_v;
                     ++time_dof)
                {
                    const int local_id =
                        spatial_local_dof * ScalarSpaceType::n_time_dofs_v +
                        time_dof;
                    tables.scalar_tensor_dof_map[
                        static_cast<std::size_t>(local_id)] =
                        {spatial_local_dof, time_dof};
                }
            }

            return tables;
        }

        template<class AffineMap>
        static RTBasisValues reconstruct_rt_values_(
            const LocalErrorReferenceTables2D& tables,
            const AffineMap& map,
            int qp_id)
        {
            const int spatial_qp_id =
                tables.spatial_qpoint_index[static_cast<std::size_t>(qp_id)];
            const int time_qp_id =
                tables.time_qpoint_index[static_cast<std::size_t>(qp_id)];
            RTBasisValues values{};
            for (int spatial_local_dof = 0;
                 spatial_local_dof < FluxSpaceType::spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                const auto physical_spatial_value =
                    FluxSpaceType::PiolaBasis::push_forward_value(
                        map,
                        tables.rt_reference_values[
                            static_cast<std::size_t>(spatial_qp_id)][
                            static_cast<std::size_t>(spatial_local_dof)]);
                for (int time_dof = 0;
                     time_dof < FluxSpaceType::n_time_dofs_v;
                     ++time_dof)
                {
                    const int local_id =
                        spatial_local_dof * FluxSpaceType::n_time_dofs_v +
                        time_dof;
                    const double time_value =
                        tables.flux_time_basis_values[
                            static_cast<std::size_t>(time_qp_id)][
                            static_cast<std::size_t>(time_dof)];
                    values[static_cast<std::size_t>(local_id)] =
                        VectorValue{
                            physical_spatial_value[0] * time_value,
                            physical_spatial_value[1] * time_value};
                }
            }
            return values;
        }

        template<class AffineMap>
        static RTBasisDivergences reconstruct_rt_divergences_(
            const LocalErrorReferenceTables2D& tables,
            const AffineMap& map,
            int qp_id)
        {
            const int spatial_qp_id =
                tables.spatial_qpoint_index[static_cast<std::size_t>(qp_id)];
            const int time_qp_id =
                tables.time_qpoint_index[static_cast<std::size_t>(qp_id)];
            RTBasisDivergences divergences{};
            for (int spatial_local_dof = 0;
                 spatial_local_dof < FluxSpaceType::spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                const double physical_spatial_divergence =
                    FluxSpaceType::PiolaBasis::push_forward_divergence(
                        map,
                        tables.rt_reference_divergences[
                            static_cast<std::size_t>(spatial_qp_id)][
                            static_cast<std::size_t>(spatial_local_dof)]);
                for (int time_dof = 0;
                     time_dof < FluxSpaceType::n_time_dofs_v;
                     ++time_dof)
                {
                    const int local_id =
                        spatial_local_dof * FluxSpaceType::n_time_dofs_v +
                        time_dof;
                    divergences[static_cast<std::size_t>(local_id)] =
                        physical_spatial_divergence *
                        tables.flux_time_basis_values[
                            static_cast<std::size_t>(time_qp_id)][
                            static_cast<std::size_t>(time_dof)];
                }
            }
            return divergences;
        }

        static double max_abs_diff_(
            const SpatialGradient& a,
            const SpatialGradient& b)
        {
            return std::max(std::abs(a[0] - b[0]), std::abs(a[1] - b[1]));
        }

        template<class MatrixA, class MatrixB>
        static double matrix_max_abs_diff_(
            const MatrixA& a,
            const MatrixB& b)
        {
            double diff = 0.0;
            for (int i = 0; i < a.rows; ++i)
                for (int j = 0; j < a.cols; ++j)
                    diff = std::max(diff, std::abs(a(i, j) - b(i, j)));
            return diff;
        }

        template<class MatrixA, class MatrixB>
        static double matrix_relative_frobenius_diff_(
            const MatrixA& a,
            const MatrixB& b)
        {
            double diff_norm_sq = 0.0;
            double ref_norm_sq = 0.0;
            for (int i = 0; i < a.rows; ++i)
            {
                for (int j = 0; j < a.cols; ++j)
                {
                    const double diff = a(i, j) - b(i, j);
                    diff_norm_sq += diff * diff;
                    ref_norm_sq += b(i, j) * b(i, j);
                }
            }
            if (ref_norm_sq <= std::numeric_limits<double>::min())
                return std::sqrt(diff_norm_sq);
            return std::sqrt(diff_norm_sq / ref_norm_sq);
        }

        template<class VectorA, class VectorB>
        static double vector_max_abs_diff_(
            const VectorA& a,
            const VectorB& b)
        {
            double diff = 0.0;
            for (int i = 0; i < a.size; ++i)
                diff = std::max(diff, std::abs(a[i] - b[i]));
            return diff;
        }

        static bool diffusion_tensor_is_identity_(
            const DiffusionTensor& tensor)
        {
            constexpr double tol = 1.0e-14;
            return std::abs(tensor[0][0] - 1.0) <= tol &&
                   std::abs(tensor[1][1] - 1.0) <= tol &&
                   std::abs(tensor[0][1]) <= tol &&
                   std::abs(tensor[1][0]) <= tol;
        }

        static bool diffusion_tensor_close_(
            const DiffusionTensor& a,
            const DiffusionTensor& b)
        {
            constexpr double tol = 1.0e-14;
            return std::abs(a[0][0] - b[0][0]) <= tol &&
                   std::abs(a[0][1] - b[0][1]) <= tol &&
                   std::abs(a[1][0] - b[1][0]) <= tol &&
                   std::abs(a[1][1] - b[1][1]) <= tol;
        }

        template<class AffineMap>
        static void assemble_local_A_reference_fast_path_(
            const LocalErrorReferenceTables2D& tables,
            const AffineMap& map,
            double time_jacobian_measure,
            const DiffusionTensor& inverse_diffusion_tensor,
            LocalAMatrix& local_A)
        {
            const double det = map.detJ;
            const double spatial_scale =
                FluxSpaceType::PiolaBasis::jacobian_measure(map) /
                (det * det);

            std::array<std::array<double, 2>, 2> geometry_metric{};
            const double J[2][2] = {
                {map.J00, map.J01},
                {map.J10, map.J11}
            };
            for (int a = 0; a < 2; ++a)
            {
                for (int b = 0; b < 2; ++b)
                {
                    double value = 0.0;
                    for (int x = 0; x < 2; ++x)
                    {
                        for (int y = 0; y < 2; ++y)
                        {
                            value +=
                                J[x][a] *
                                inverse_diffusion_tensor[
                                    static_cast<std::size_t>(x)][
                                    static_cast<std::size_t>(y)] *
                                J[y][b];
                        }
                    }
                    geometry_metric[static_cast<std::size_t>(a)][
                        static_cast<std::size_t>(b)] =
                        spatial_scale * value;
                }
            }

            for (int i = 0; i < FluxSpaceType::local_dofs_v; ++i)
            {
                const auto i_map =
                    tables.flux_tensor_dof_map[static_cast<std::size_t>(i)];
                const int i_spatial = i_map[0];
                const int i_time = i_map[1];
                for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                {
                    const auto j_map =
                        tables.flux_tensor_dof_map[
                            static_cast<std::size_t>(j)];
                    const int j_spatial = j_map[0];
                    const int j_time = j_map[1];
                    double spatial_value = 0.0;
                    for (int a = 0; a < 2; ++a)
                    {
                        for (int b = 0; b < 2; ++b)
                        {
                            spatial_value +=
                                tables.rt_spatial_value_moments[
                                    static_cast<std::size_t>(i_spatial)][
                                    static_cast<std::size_t>(j_spatial)][
                                    static_cast<std::size_t>(a)][
                                    static_cast<std::size_t>(b)] *
                                geometry_metric[
                                    static_cast<std::size_t>(a)][
                                    static_cast<std::size_t>(b)];
                        }
                    }
                    local_A(i, j) +=
                        spatial_value * time_jacobian_measure *
                        tables.flux_time_mass[
                            static_cast<std::size_t>(i_time)][
                            static_cast<std::size_t>(j_time)];
                }
            }
        }

        template<class AffineMap>
        static void assemble_local_B_reference_fast_path_(
            const LocalErrorReferenceTables2D& tables,
            const AffineMap& map,
            double time_jacobian_measure,
            LocalBMatrix& local_B)
        {
            const double spatial_scale =
                FluxSpaceType::PiolaBasis::jacobian_measure(map) / map.detJ;
            for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
            {
                const auto i_map =
                    tables.scalar_tensor_dof_map[
                        static_cast<std::size_t>(i)];
                const int i_spatial = i_map[0];
                const int i_time = i_map[1];
                for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                {
                    const auto j_map =
                        tables.flux_tensor_dof_map[
                            static_cast<std::size_t>(j)];
                    const int j_spatial = j_map[0];
                    const int j_time = j_map[1];
                    local_B(i, j) +=
                        spatial_scale * time_jacobian_measure *
                        tables.scalar_divergence_spatial_coupling[
                            static_cast<std::size_t>(i_spatial)][
                            static_cast<std::size_t>(j_spatial)] *
                        tables.scalar_flux_time_coupling[
                            static_cast<std::size_t>(i_time)][
                            static_cast<std::size_t>(j_time)];
                }
            }
        }

        template<class AffineMap>
        static void assemble_local_operators_generic_constant_(
            const LocalErrorReferenceTables2D& tables,
            const AffineMap& map,
            double jacobian_measure,
            bool identity_diffusion,
            const DiffusionTensor& inverse_diffusion_tensor,
            LocalAMatrix& local_A,
            LocalBMatrix& local_B)
        {
            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const int spatial_qp_id =
                    tables.spatial_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const int time_qp_id =
                    tables.time_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const double jacobian_weight =
                    tables.spatial_reference_weights[
                        static_cast<std::size_t>(spatial_qp_id)] *
                    tables.time_reference_weights[
                        static_cast<std::size_t>(time_qp_id)] *
                    jacobian_measure;
                const auto rt_values =
                    reconstruct_rt_values_(tables, map, qp_id);
                const auto rt_divergences =
                    reconstruct_rt_divergences_(tables, map, qp_id);
                const auto& scalar_values =
                    tables.scalar_basis_values[
                        static_cast<std::size_t>(qp_id)];
                for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                {
                    const auto& sigma_j =
                        rt_values[static_cast<std::size_t>(j)];
                    const auto weighted_inverse_sigma_j =
                        identity_diffusion
                            ? VectorValue{
                                  jacobian_weight * sigma_j[0],
                                  jacobian_weight * sigma_j[1]}
                            : detail::apply_inverse_tensor_2d(
                                  inverse_diffusion_tensor,
                                  sigma_j,
                                  jacobian_weight);
                    for (int i = 0; i < FluxSpaceType::local_dofs_v; ++i)
                    {
                        const auto& sigma_i =
                            rt_values[static_cast<std::size_t>(i)];
                        local_A(i, j) +=
                            sigma_i[0] * weighted_inverse_sigma_j[0] +
                            sigma_i[1] * weighted_inverse_sigma_j[1];
                    }
                    for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
                    {
                        local_B(i, j) +=
                            scalar_values[static_cast<std::size_t>(i)] *
                            jacobian_weight *
                            rt_divergences[static_cast<std::size_t>(j)];
                    }
                }
            }
        }

        template<class AffineMap>
        static OperatorCellState2D build_operator_state_shadow_(
            const LocalErrorReferenceTables2D& tables,
            const AffineMap& map,
            const CellData& cell_data)
        {
            OperatorCellState2D state{};
            state.active_slab_cell_ordinal =
                cell_data.active_slab_cell_ordinal;
            state.slab_id = cell_data.slab_id;
            state.slab_cell_id = cell_data.slab_cell_id;
            state.source_cell_id = cell_data.source_cell_id;
            state.x_cell_id = cell_data.x_cell_id;
            state.spatial_jacobian_measure =
                cell_data.spatial_jacobian_measure;
            state.time_jacobian_measure = cell_data.time_jacobian_measure;
            state.jacobian_measure = cell_data.jacobian_measure;
            state.local_A.resize(
                FluxSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);
            state.local_B.resize(
                ScalarSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);

            state.diffusion_tensor =
                cell_data.points.front().diffusion_tensor;
            state.inverse_diffusion_tensor =
                cell_data.points.front().inverse_diffusion_tensor;
            state.diffusion_mode =
                diffusion_tensor_is_identity_(state.diffusion_tensor)
                    ? OperatorDiffusionMode::identity
                    : OperatorDiffusionMode::constant;
            for (const auto& qp : cell_data.points)
            {
                if (!diffusion_tensor_close_(
                        qp.diffusion_tensor,
                        state.diffusion_tensor))
                {
                    state.diffusion_mode = OperatorDiffusionMode::variable;
                    break;
                }
            }

            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const auto& old_qp =
                    cell_data.points[static_cast<std::size_t>(qp_id)];
                const auto rt_values =
                    reconstruct_rt_values_(tables, map, qp_id);
                const auto rt_divergences =
                    reconstruct_rt_divergences_(tables, map, qp_id);
                const auto& scalar_values =
                    tables.scalar_basis_values[
                        static_cast<std::size_t>(qp_id)];
                for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                {
                    const auto& sigma_j =
                        rt_values[static_cast<std::size_t>(j)];
                    const auto weighted_inverse_sigma_j =
                        state.diffusion_mode == OperatorDiffusionMode::identity
                            ? VectorValue{
                                  old_qp.jacobian_weight * sigma_j[0],
                                  old_qp.jacobian_weight * sigma_j[1]}
                            : detail::apply_inverse_tensor_2d(
                                  old_qp.inverse_diffusion_tensor,
                                  sigma_j,
                                  old_qp.jacobian_weight);
                    for (int i = 0; i < FluxSpaceType::local_dofs_v; ++i)
                    {
                        const auto& sigma_i =
                            rt_values[static_cast<std::size_t>(i)];
                        state.local_A(i, j) +=
                            sigma_i[0] * weighted_inverse_sigma_j[0] +
                            sigma_i[1] * weighted_inverse_sigma_j[1];
                    }
                    for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
                    {
                        state.local_B(i, j) +=
                            scalar_values[static_cast<std::size_t>(i)] *
                            old_qp.jacobian_weight *
                            rt_divergences[static_cast<std::size_t>(j)];
                    }
                }
            }

            return state;
        }

        static RHSCellState2D build_rhs_state_shadow_(
            const CellData& cell_data)
        {
            RHSCellState2D state{};
            state.active_slab_cell_ordinal =
                cell_data.active_slab_cell_ordinal;
            state.slab_id = cell_data.slab_id;
            state.slab_cell_id = cell_data.slab_cell_id;
            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const auto& old_qp =
                    cell_data.points[static_cast<std::size_t>(qp_id)];
                auto& shadow_qp =
                    state.points[static_cast<std::size_t>(qp_id)];
                shadow_qp.grad_theta_tilde = old_qp.grad_theta_tilde;
                shadow_qp.M_grad_theta_tilde = old_qp.M_grad_theta_tilde;
                shadow_qp.u_time_derivative = old_qp.u_time_derivative;
                shadow_qp.ell_value = old_qp.ell_value;
            }
            return state;
        }

        template<class AffineMap>
        static void compare_compact_split_shadow_(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            const AffineMap& map,
            const CellData& cell_data,
            AuditStats& audit_stats)
        {
            audit_stats.compact_state_shadow_enabled = 1.0;
            audit_stats.compact_state_shadow_sample_count += 1.0;
            audit_stats.old_cell_data_bytes_per_cell =
                static_cast<double>(sizeof(CellData));
            audit_stats.operator_state_bytes_per_cell =
                static_cast<double>(sizeof(OperatorCellState2D));
            audit_stats.rhs_state_bytes_per_cell =
                static_cast<double>(sizeof(RHSCellState2D));
            audit_stats.flux_diagnostic_state_bytes_per_cell =
                static_cast<double>(sizeof(FluxDiagnosticCellState2D));
            audit_stats.reference_table_memory_mb =
                static_cast<double>(sizeof(LocalErrorReferenceTables2D)) /
                (1024.0 * 1024.0);

            const auto& tables = reference_tables_();
            const auto operator_state =
                build_operator_state_shadow_(tables, map, cell_data);
            const auto rhs_state = build_rhs_state_shadow_(cell_data);

            audit_stats.compact_state_local_A_max_abs_diff = std::max(
                audit_stats.compact_state_local_A_max_abs_diff,
                matrix_max_abs_diff_(
                    operator_state.local_A,
                    cell_data.local_A));
            audit_stats.compact_state_local_B_max_abs_diff = std::max(
                audit_stats.compact_state_local_B_max_abs_diff,
                matrix_max_abs_diff_(
                    operator_state.local_B,
                    cell_data.local_B));

            LocalFVector old_f{};
            LocalFVector shadow_f{};
            LocalGVector old_g{};
            LocalGVector shadow_g{};
            old_f.resize(FluxSpaceType::local_dofs_v);
            shadow_f.resize(FluxSpaceType::local_dofs_v);
            old_g.resize(ScalarSpaceType::local_dofs_v);
            shadow_g.resize(ScalarSpaceType::local_dofs_v);

            const int local_vertex_index =
                flux_space.patch()
                    .cell(patch_cell_index)
                    .local_vertex_index;

            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const auto& old_qp =
                    cell_data.points[static_cast<std::size_t>(qp_id)];
                const auto& shadow_qp =
                    rhs_state.points[static_cast<std::size_t>(qp_id)];
                const auto rt_values =
                    reconstruct_rt_values_(tables, map, qp_id);
                const auto& scalar_values =
                    tables.scalar_basis_values[
                        static_cast<std::size_t>(qp_id)];
                const int spatial_qp_id =
                    tables.spatial_qpoint_index[
                        static_cast<std::size_t>(qp_id)];

                for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                {
                    audit_stats
                        .compact_state_reference_rt_basis_max_abs_diff =
                        std::max(
                            audit_stats
                                .compact_state_reference_rt_basis_max_abs_diff,
                            max_abs_diff_(
                                rt_values[static_cast<std::size_t>(j)],
                                old_qp.rt_basis_values[
                                    static_cast<std::size_t>(j)]));
                }
                for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
                {
                    audit_stats
                        .compact_state_reference_scalar_basis_max_abs_diff =
                        std::max(
                            audit_stats
                                .compact_state_reference_scalar_basis_max_abs_diff,
                            std::abs(
                                scalar_values[static_cast<std::size_t>(i)] -
                                old_qp.scalar_basis_values[
                                    static_cast<std::size_t>(i)]));
                }
                for (int local_vertex = 0;
                     local_vertex < 3;
                     ++local_vertex)
                {
                    const auto vertex_index =
                        static_cast<std::size_t>(local_vertex);
                    audit_stats
                        .compact_state_reference_partition_value_max_abs_diff =
                        std::max(
                            audit_stats
                                .compact_state_reference_partition_value_max_abs_diff,
                            std::abs(
                                tables.partition_values[
                                    static_cast<std::size_t>(
                                        spatial_qp_id)][vertex_index] -
                                old_qp.partition_of_unity_values[
                                    vertex_index]));
                    const auto physical_gradient =
                        map_reference_gradient_to_physical_(
                            map,
                            tables.partition_reference_gradients[
                                static_cast<std::size_t>(
                                    spatial_qp_id)][vertex_index]);
                    audit_stats
                        .compact_state_reference_partition_gradient_max_abs_diff =
                        std::max(
                            audit_stats
                                .compact_state_reference_partition_gradient_max_abs_diff,
                            max_abs_diff_(
                                physical_gradient,
                                old_qp.partition_of_unity_gradients[
                                    vertex_index]));
                }
                audit_stats.compact_state_grad_theta_max_abs_diff = std::max(
                    audit_stats.compact_state_grad_theta_max_abs_diff,
                    max_abs_diff_(
                        shadow_qp.grad_theta_tilde,
                        old_qp.grad_theta_tilde));
                audit_stats.compact_state_u_time_derivative_max_abs_diff =
                    std::max(
                        audit_stats
                            .compact_state_u_time_derivative_max_abs_diff,
                        std::abs(
                            shadow_qp.u_time_derivative -
                            old_qp.u_time_derivative));

                const double psi_q =
                    tables.partition_values[
                        static_cast<std::size_t>(spatial_qp_id)][
                        static_cast<std::size_t>(local_vertex_index)];
                const auto grad_psi =
                    map_reference_gradient_to_physical_(
                        map,
                        tables.partition_reference_gradients[
                            static_cast<std::size_t>(spatial_qp_id)][
                            static_cast<std::size_t>(local_vertex_index)]);
                const double old_psi_q =
                    old_qp.partition_of_unity_values[
                        static_cast<std::size_t>(local_vertex_index)];
                const auto& old_grad_psi =
                    old_qp.partition_of_unity_gradients[
                        static_cast<std::size_t>(local_vertex_index)];
                for (int i = 0; i < FluxSpaceType::local_dofs_v; ++i)
                {
                    const auto& old_sigma =
                        old_qp.rt_basis_values[
                            static_cast<std::size_t>(i)];
                    const auto& shadow_sigma =
                        rt_values[static_cast<std::size_t>(i)];
                    old_f[i] +=
                        -old_psi_q *
                        (old_qp.grad_theta_tilde[0] * old_sigma[0] +
                         old_qp.grad_theta_tilde[1] * old_sigma[1]) *
                        old_qp.jacobian_weight;
                    shadow_f[i] +=
                        -psi_q *
                        (shadow_qp.grad_theta_tilde[0] * shadow_sigma[0] +
                         shadow_qp.grad_theta_tilde[1] * shadow_sigma[1]) *
                        old_qp.jacobian_weight;
                }
                const double old_grad_psi_M_grad_theta =
                    old_grad_psi[0] * old_qp.M_grad_theta_tilde[0] +
                    old_grad_psi[1] * old_qp.M_grad_theta_tilde[1];
                const double shadow_grad_psi_M_grad_theta =
                    grad_psi[0] * shadow_qp.M_grad_theta_tilde[0] +
                    grad_psi[1] * shadow_qp.M_grad_theta_tilde[1];
                const double old_rhs =
                    old_psi_q *
                        (old_qp.ell_value - old_qp.u_time_derivative) -
                    old_grad_psi_M_grad_theta;
                const double shadow_rhs =
                    psi_q *
                        (shadow_qp.ell_value -
                         shadow_qp.u_time_derivative) -
                    shadow_grad_psi_M_grad_theta;
                for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
                {
                    old_g[i] +=
                        old_rhs * old_qp.jacobian_weight *
                        old_qp.scalar_basis_values[
                            static_cast<std::size_t>(i)];
                    shadow_g[i] +=
                        shadow_rhs * old_qp.jacobian_weight *
                        scalar_values[static_cast<std::size_t>(i)];
                }
            }

            audit_stats.compact_state_rhs_f_max_abs_diff = std::max(
                audit_stats.compact_state_rhs_f_max_abs_diff,
                vector_max_abs_diff_(old_f, shadow_f));
            audit_stats.compact_state_rhs_g_max_abs_diff = std::max(
                audit_stats.compact_state_rhs_g_max_abs_diff,
                vector_max_abs_diff_(old_g, shadow_g));
            static_cast<void>(scalar_space);
        }

        template<
            class XSpaceType,
            class SlabSpaceType,
            class ReconstructedFunctionType,
            class XFunctionType,
            class EllFunction,
            class MFunction>
        static void fill_compact_cell_(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M,
            CompactCellData& compact_cell,
            AuditStats* audit_stats = nullptr,
            bool coefficient_fast_path = true)
        {
            constexpr int audit_sample_stride = 128;
            const auto fill_begin = Clock::now();
            bool sample_qpoint_subkernels = false;
            if (audit_stats != nullptr)
            {
                const auto state_index =
                    static_cast<long long>(
                        audit_stats->state_fill_state_count);
                sample_qpoint_subkernels =
                    state_index % audit_sample_stride == 0;
                audit_stats->state_fill_state_count += 1.0;
                audit_stats->state_fill_qpoints_processed +=
                    static_cast<double>(n_quadrature_points_v);
                if (sample_qpoint_subkernels)
                    audit_stats->state_fill_sampled_qpoints +=
                        static_cast<double>(n_quadrature_points_v);
                audit_stats->compact_operator_state_constructed_count += 1.0;
                audit_stats->compact_rhs_state_constructed_count += 1.0;
            }

            const auto& patch_cell =
                flux_space.patch().cell(patch_cell_index);
            const int slab_id = flux_space.patch().slab_id;
            const int slab_cell_id = patch_cell.slab_cell_id;
            const int source_cell_id = patch_cell.source_cell_id;
            auto& operator_state = compact_cell.operator_state;
            auto& rhs_state = compact_cell.rhs_state;
            const int shared_ordinal =
                context.shared_context != nullptr
                    ? (operator_state.active_slab_cell_ordinal >= 0
                           ? operator_state.active_slab_cell_ordinal
                           : context.shared_context->slab_cell_ordinal(
                                 slab_id,
                                 slab_cell_id))
                    : -1;

            const auto ancestor_begin = Clock::now();
            const int x_cell_id =
                context.shared_context != nullptr
                    ? context.shared_context
                          ->slab_cell_metadata(shared_ordinal)
                          .active_x_cell_id
                    : finite_element::assembly::detail::
                          find_active_ancestor_cell_from_source_cell(
                              *context.x_ancestor_cache,
                              *context.x_space,
                              source_cell_id);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_active_ancestor_lookup_seconds,
                ancestor_begin);

            const auto geometry_begin = Clock::now();
            const auto& slab_geom =
                context.shared_context != nullptr
                    ? context.shared_context->slab_geometry(shared_ordinal)
                    : (*context.slab_geometry_caches)[
                          static_cast<std::size_t>(slab_id)]
                          .geometry(slab_cell_id);
            const auto& x_geom =
                context.shared_context != nullptr
                    ? context.shared_context->x_geometry(x_cell_id)
                    : context.x_geometry_cache->geometry(x_cell_id);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_geometry_lookup_seconds,
                geometry_begin);

            const auto map =
                flux_space.physical_map_for_patch_cell(patch_cell_index);
            operator_state.slab_id = slab_id;
            operator_state.slab_cell_id = slab_cell_id;
            operator_state.active_slab_cell_ordinal = shared_ordinal;
            operator_state.source_cell_id = source_cell_id;
            operator_state.x_cell_id = x_cell_id;
            operator_state.spatial_jacobian_measure =
                FluxSpaceType::PiolaBasis::jacobian_measure(map);
            operator_state.time_jacobian_measure =
                std::abs(flux_space.time_length());
            operator_state.jacobian_measure =
                operator_state.spatial_jacobian_measure *
                operator_state.time_jacobian_measure;
            operator_state.local_A.resize(
                FluxSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);
            operator_state.local_B.resize(
                ScalarSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);

            rhs_state.slab_id = slab_id;
            rhs_state.slab_cell_id = slab_cell_id;
            rhs_state.active_slab_cell_ordinal = shared_ordinal;
            rhs_state.jacobian_measure = operator_state.jacobian_measure;

            const bool zero_load_fast_path =
                coefficient_fast_path &&
                coefficients::is_zero_load_function(ell);
            const bool identity_diffusion_fast_path =
                coefficient_fast_path &&
                coefficients::is_identity_diffusion_function<
                    GT::dim_space_v>(M);
            const auto constant_diffusion_tensor =
                coefficient_fast_path
                    ? coefficients::constant_diffusion_tensor_if_available<
                          GT::dim_space_v>(M)
                    : std::optional<
                          coefficients::DiffusionTensor<
                              GT::dim_space_v>>{};
            const bool constant_diffusion_fast_path =
                coefficient_fast_path &&
                constant_diffusion_tensor.has_value() &&
                !identity_diffusion_fast_path;
            const DiffusionTensor identity_diffusion_tensor =
                coefficients::identity_diffusion_tensor<GT::dim_space_v>();
            const DiffusionTensor fast_diffusion_tensor =
                identity_diffusion_fast_path
                    ? identity_diffusion_tensor
                    : (constant_diffusion_fast_path
                           ? *constant_diffusion_tensor
                           : DiffusionTensor{});
            const DiffusionTensor fast_inverse_diffusion_tensor =
                identity_diffusion_fast_path
                    ? identity_diffusion_tensor
                    : (constant_diffusion_fast_path
                           ? detail::inverse_diffusion_tensor_2d(
                                 fast_diffusion_tensor)
                           : DiffusionTensor{});
            operator_state.diffusion_tensor = fast_diffusion_tensor;
            operator_state.inverse_diffusion_tensor =
                fast_inverse_diffusion_tensor;
            operator_state.diffusion_mode =
                identity_diffusion_fast_path
                    ? OperatorDiffusionMode::identity
                    : (constant_diffusion_fast_path
                           ? OperatorDiffusionMode::constant
                           : OperatorDiffusionMode::variable);
            if (audit_stats != nullptr)
            {
                audit_stats->coefficient_fast_path_enabled =
                    std::max(audit_stats->coefficient_fast_path_enabled,
                             coefficient_fast_path ? 1.0 : 0.0);
                if (identity_diffusion_fast_path)
                    audit_stats
                        ->coefficient_fast_path_identity_diffusion_cells +=
                        1.0;
                else if (constant_diffusion_fast_path)
                    audit_stats
                        ->coefficient_fast_path_constant_diffusion_cells +=
                        1.0;
                else
                    audit_stats->coefficient_fast_path_generic_cells += 1.0;
                if (zero_load_fast_path)
                    audit_stats->coefficient_fast_path_zero_load_cells +=
                        1.0;
            }

            const auto& tables = reference_tables_();
            std::array<double, QTime> time_physical_points{};
            const auto time_basis_begin = Clock::now();
            for (int qt = 0; qt < QTime; ++qt)
            {
                time_physical_points[static_cast<std::size_t>(qt)] =
                    flux_space.map_time_to_physical(
                        tables.time_reference_points[
                            static_cast<std::size_t>(qt)]);
            }
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_time_basis_seconds,
                time_basis_begin);

            const bool reference_operator_fast_path =
                operator_state.diffusion_mode ==
                    OperatorDiffusionMode::identity ||
                operator_state.diffusion_mode ==
                    OperatorDiffusionMode::constant;
            if (reference_operator_fast_path)
            {
                if (audit_stats != nullptr)
                {
                    audit_stats->operator_builder_mode = std::max(
                        audit_stats->operator_builder_mode,
                        1.0);
                    audit_stats->local_B_reference_fast_path_count += 1.0;
                    if (operator_state.diffusion_mode ==
                        OperatorDiffusionMode::identity)
                    {
                        audit_stats
                            ->local_A_identity_reference_fast_path_count +=
                            1.0;
                    }
                    else
                    {
                        audit_stats
                            ->local_A_constant_reference_fast_path_count +=
                            1.0;
                    }
                }
                const auto local_A_begin = Clock::now();
                assemble_local_A_reference_fast_path_(
                    tables,
                    map,
                    operator_state.time_jacobian_measure,
                    operator_state.inverse_diffusion_tensor,
                    operator_state.local_A);
                add_elapsed_(
                    audit_stats,
                    &AuditStats::local_A_build_seconds,
                    local_A_begin);
                const auto local_B_begin = Clock::now();
                assemble_local_B_reference_fast_path_(
                    tables,
                    map,
                    operator_state.time_jacobian_measure,
                    operator_state.local_B);
                add_elapsed_(
                    audit_stats,
                    &AuditStats::local_B_build_seconds,
                    local_B_begin);

                if (sample_qpoint_subkernels && audit_stats != nullptr)
                {
                    LocalAMatrix generic_A{};
                    LocalBMatrix generic_B{};
                    generic_A.resize(
                        FluxSpaceType::local_dofs_v,
                        FluxSpaceType::local_dofs_v);
                    generic_B.resize(
                        ScalarSpaceType::local_dofs_v,
                        FluxSpaceType::local_dofs_v);
                    assemble_local_operators_generic_constant_(
                        tables,
                        map,
                        operator_state.jacobian_measure,
                        operator_state.diffusion_mode ==
                            OperatorDiffusionMode::identity,
                        operator_state.inverse_diffusion_tensor,
                        generic_A,
                        generic_B);
                    audit_stats->local_A_debug_max_abs_diff = std::max(
                        audit_stats->local_A_debug_max_abs_diff,
                        matrix_max_abs_diff_(
                            operator_state.local_A,
                            generic_A));
                    audit_stats->local_A_debug_rel_frobenius_diff = std::max(
                        audit_stats->local_A_debug_rel_frobenius_diff,
                        matrix_relative_frobenius_diff_(
                            operator_state.local_A,
                            generic_A));
                    audit_stats->local_B_debug_max_abs_diff = std::max(
                        audit_stats->local_B_debug_max_abs_diff,
                        matrix_max_abs_diff_(
                            operator_state.local_B,
                            generic_B));
                    audit_stats->local_B_debug_rel_frobenius_diff = std::max(
                        audit_stats->local_B_debug_rel_frobenius_diff,
                        matrix_relative_frobenius_diff_(
                            operator_state.local_B,
                            generic_B));
                }
            }
            else if (audit_stats != nullptr)
            {
                audit_stats->local_A_variable_generic_path_count += 1.0;
            }

            constexpr int time_component = GT::dim_space_v;
            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const int spatial_qp_id =
                    tables.spatial_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const int time_qp_id =
                    tables.time_qpoint_index[
                        static_cast<std::size_t>(qp_id)];
                const auto& x_ref =
                    tables.spatial_reference_points[
                        static_cast<std::size_t>(spatial_qp_id)];
                const auto affine_qpoint_begin =
                    sample_qpoint_subkernels ? Clock::now()
                                             : Clock::time_point{};
                const auto xy =
                    FluxSpaceType::PiolaBasis::map_to_physical(map, x_ref);
                if (sample_qpoint_subkernels)
                    add_elapsed_(
                        audit_stats,
                        &AuditStats::state_fill_affine_map_seconds,
                        affine_qpoint_begin);
                auto& rhs_qp =
                    rhs_state.points[static_cast<std::size_t>(qp_id)];
                const double t_ref =
                    tables.time_reference_points[
                        static_cast<std::size_t>(time_qp_id)];
                const double t_phys =
                    time_physical_points[
                        static_cast<std::size_t>(time_qp_id)];
                const SpaceTimePoint physical_point{xy[0], xy[1], t_phys};
                rhs_qp.jacobian_weight =
                    tables.spatial_reference_weights[
                        static_cast<std::size_t>(spatial_qp_id)] *
                    tables.time_reference_weights[
                        static_cast<std::size_t>(time_qp_id)] *
                    operator_state.jacobian_measure;

                const auto lambda_begin =
                    sample_qpoint_subkernels ? Clock::now()
                                             : Clock::time_point{};
                const auto grad_lambda =
                    lambda_tilde.gradient_on_cell(
                        slab_id,
                        slab_cell_id,
                        physical_point,
                        slab_geom);
                if (sample_qpoint_subkernels)
                    add_elapsed_(
                        audit_stats,
                        &AuditStats::state_fill_lambda_gradient_seconds,
                        lambda_begin);

                const auto u_begin =
                    sample_qpoint_subkernels ? Clock::now()
                                             : Clock::time_point{};
                const auto grad_u =
                    u_delta.gradient_on_cell(
                        x_cell_id,
                        physical_point,
                        x_geom);
                if (sample_qpoint_subkernels)
                    add_elapsed_(
                        audit_stats,
                        &AuditStats::state_fill_u_gradient_seconds,
                        u_begin);

                const SpatialGradient grad_lambda_tilde{
                    grad_lambda[0],
                    grad_lambda[1]};
                const SpatialGradient grad_u_delta{grad_u[0], grad_u[1]};
                rhs_qp.grad_theta_tilde =
                    SpatialGradient{
                        grad_lambda_tilde[0] + grad_u_delta[0],
                        grad_lambda_tilde[1] + grad_u_delta[1]};
                rhs_qp.u_time_derivative = grad_u[time_component];

                const auto load_begin =
                    sample_qpoint_subkernels ? Clock::now()
                                             : Clock::time_point{};
                rhs_qp.ell_value =
                    zero_load_fast_path
                        ? 0.0
                        : static_cast<double>(ell(physical_point));
                if (sample_qpoint_subkernels)
                    add_elapsed_(
                        audit_stats,
                        &AuditStats::state_fill_load_evaluation_seconds,
                        load_begin);

                const auto diffusion_begin =
                    sample_qpoint_subkernels ? Clock::now()
                                             : Clock::time_point{};
                const DiffusionTensor diffusion_tensor =
                    identity_diffusion_fast_path ||
                            constant_diffusion_fast_path
                        ? fast_diffusion_tensor
                        : coefficients::evaluate_diffusion_tensor<
                              GT::dim_space_v>(
                              M,
                              physical_point);
                if (sample_qpoint_subkernels)
                    add_elapsed_(
                        audit_stats,
                        &AuditStats::state_fill_diffusion_evaluation_seconds,
                        diffusion_begin);

                const auto inverse_begin =
                    sample_qpoint_subkernels ? Clock::now()
                                             : Clock::time_point{};
                const DiffusionTensor inverse_diffusion_tensor =
                    identity_diffusion_fast_path ||
                            constant_diffusion_fast_path
                        ? fast_inverse_diffusion_tensor
                        : detail::inverse_diffusion_tensor_2d(
                              diffusion_tensor);
                if (sample_qpoint_subkernels)
                    add_elapsed_(
                        audit_stats,
                        &AuditStats::state_fill_diffusion_inverse_seconds,
                        inverse_begin);
                if (operator_state.diffusion_mode ==
                    OperatorDiffusionMode::variable)
                {
                    operator_state.diffusion_tensor = diffusion_tensor;
                    operator_state.inverse_diffusion_tensor =
                        inverse_diffusion_tensor;
                }

                rhs_qp.M_grad_theta_tilde =
                    identity_diffusion_fast_path
                        ? rhs_qp.grad_theta_tilde
                        : coefficients::apply_M<GT::dim_space_v>(
                              diffusion_tensor,
                              rhs_qp.grad_theta_tilde);

                if (!reference_operator_fast_path)
                {
                    const auto rt_basis_begin =
                        sample_qpoint_subkernels ? Clock::now()
                                                 : Clock::time_point{};
                    const auto rt_values =
                        reconstruct_rt_values_(tables, map, qp_id);
                    const auto rt_divergences =
                        reconstruct_rt_divergences_(tables, map, qp_id);
                    if (sample_qpoint_subkernels)
                        add_elapsed_(
                            audit_stats,
                            &AuditStats::state_fill_spatial_rt_basis_seconds,
                            rt_basis_begin);
                    const auto& scalar_values =
                        tables.scalar_basis_values[
                            static_cast<std::size_t>(qp_id)];

                    for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                    {
                        const auto& sigma_j =
                            rt_values[static_cast<std::size_t>(j)];
                        const auto weighted_inverse_sigma_j =
                            detail::apply_inverse_tensor_2d(
                                inverse_diffusion_tensor,
                                sigma_j,
                                rhs_qp.jacobian_weight);
                        const auto local_A_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        for (int i = 0; i < FluxSpaceType::local_dofs_v; ++i)
                        {
                            const auto& sigma_i =
                                rt_values[static_cast<std::size_t>(i)];
                            operator_state.local_A(i, j) +=
                                sigma_i[0] * weighted_inverse_sigma_j[0] +
                                sigma_i[1] * weighted_inverse_sigma_j[1];
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_local_A_assembly_seconds,
                                local_A_begin);

                        const auto local_B_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        for (int i = 0;
                             i < ScalarSpaceType::local_dofs_v;
                             ++i)
                        {
                            operator_state.local_B(i, j) +=
                                scalar_values[static_cast<std::size_t>(i)] *
                                rhs_qp.jacobian_weight *
                                rt_divergences[
                                    static_cast<std::size_t>(j)];
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_local_B_assembly_seconds,
                                local_B_begin);
                    }
                }
                static_cast<void>(t_ref);
            }

            static_cast<void>(scalar_space);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_total_seconds,
                fill_begin);
        }

        template<
            class XSpaceType,
            class SlabSpaceType,
            class ReconstructedFunctionType,
            class XFunctionType,
            class EllFunction,
            class MFunction>
        static void fill_cell_(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M,
            CellData& cell_data,
            AuditStats* audit_stats = nullptr,
            bool coefficient_fast_path = true,
            bool compact_state_shadow = false)
        {
            constexpr int audit_sample_stride = 128;
            const auto fill_begin = Clock::now();
            bool sample_qpoint_subkernels = false;
            bool sample_compact_shadow = false;
            if (audit_stats != nullptr)
            {
                audit_stats->monolithic_cell_data_constructed_count += 1.0;
                audit_stats->monolithic_debug_path_used_count += 1.0;
                const auto state_index =
                    static_cast<long long>(
                        audit_stats->state_fill_state_count);
                sample_qpoint_subkernels =
                    state_index % audit_sample_stride == 0;
                sample_compact_shadow =
                    compact_state_shadow && sample_qpoint_subkernels;
                audit_stats->state_fill_state_count += 1.0;
                audit_stats->state_fill_qpoints_processed +=
                    static_cast<double>(n_quadrature_points_v);
                if (sample_qpoint_subkernels)
                    audit_stats->state_fill_sampled_qpoints +=
                        static_cast<double>(n_quadrature_points_v);
            }

            const auto& patch_cell =
                flux_space.patch().cell(patch_cell_index);
            const int slab_id = flux_space.patch().slab_id;
            const int slab_cell_id = patch_cell.slab_cell_id;
            const int source_cell_id = patch_cell.source_cell_id;
            const int shared_ordinal =
                context.shared_context != nullptr
                    ? (cell_data.active_slab_cell_ordinal >= 0
                           ? cell_data.active_slab_cell_ordinal
                           : context.shared_context->slab_cell_ordinal(
                                 slab_id,
                                 slab_cell_id))
                    : -1;
            const auto ancestor_begin = Clock::now();
            const int x_cell_id =
                context.shared_context != nullptr
                    ? context.shared_context
                          ->slab_cell_metadata(shared_ordinal)
                          .active_x_cell_id
                    : finite_element::assembly::detail::
                          find_active_ancestor_cell_from_source_cell(
                              *context.x_ancestor_cache,
                              *context.x_space,
                              source_cell_id);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_active_ancestor_lookup_seconds,
                ancestor_begin);
            const auto geometry_begin = Clock::now();
            const auto& slab_geom =
                context.shared_context != nullptr
                    ? context.shared_context->slab_geometry(shared_ordinal)
                    : (*context.slab_geometry_caches)[
                          static_cast<std::size_t>(slab_id)]
                          .geometry(slab_cell_id);
            const auto& x_geom =
                context.shared_context != nullptr
                    ? context.shared_context->x_geometry(x_cell_id)
                    : context.x_geometry_cache->geometry(x_cell_id);
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_geometry_lookup_seconds,
                geometry_begin);
            const auto map =
                flux_space.physical_map_for_patch_cell(patch_cell_index);

            cell_data.slab_id = slab_id;
            cell_data.slab_cell_id = slab_cell_id;
            cell_data.active_slab_cell_ordinal = shared_ordinal;
            cell_data.source_cell_id = source_cell_id;
            cell_data.x_cell_id = x_cell_id;
            cell_data.spatial_jacobian_measure =
                FluxSpaceType::PiolaBasis::jacobian_measure(map);
            cell_data.time_jacobian_measure =
                std::abs(flux_space.time_length());
            cell_data.jacobian_measure =
                cell_data.spatial_jacobian_measure *
                cell_data.time_jacobian_measure;
            cell_data.local_A.resize(
                FluxSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);
            cell_data.local_B.resize(
                ScalarSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);

            const bool zero_load_fast_path =
                coefficient_fast_path &&
                coefficients::is_zero_load_function(ell);
            const bool identity_diffusion_fast_path =
                coefficient_fast_path &&
                coefficients::is_identity_diffusion_function<
                    GT::dim_space_v>(M);
            const auto constant_diffusion_tensor =
                coefficient_fast_path
                    ? coefficients::constant_diffusion_tensor_if_available<
                          GT::dim_space_v>(M)
                    : std::optional<
                          coefficients::DiffusionTensor<
                              GT::dim_space_v>>{};
            const bool constant_diffusion_fast_path =
                coefficient_fast_path &&
                constant_diffusion_tensor.has_value() &&
                !identity_diffusion_fast_path;
            const DiffusionTensor identity_diffusion_tensor =
                coefficients::identity_diffusion_tensor<GT::dim_space_v>();
            const DiffusionTensor fast_diffusion_tensor =
                identity_diffusion_fast_path
                    ? identity_diffusion_tensor
                    : (constant_diffusion_fast_path
                           ? *constant_diffusion_tensor
                           : DiffusionTensor{});
            const DiffusionTensor fast_inverse_diffusion_tensor =
                identity_diffusion_fast_path
                    ? identity_diffusion_tensor
                    : (constant_diffusion_fast_path
                           ? detail::inverse_diffusion_tensor_2d(
                                 fast_diffusion_tensor)
                           : DiffusionTensor{});
            if (audit_stats != nullptr)
            {
                audit_stats->coefficient_fast_path_enabled =
                    std::max(audit_stats->coefficient_fast_path_enabled,
                             coefficient_fast_path ? 1.0 : 0.0);
                if (identity_diffusion_fast_path)
                {
                    audit_stats
                        ->coefficient_fast_path_identity_diffusion_cells +=
                        1.0;
                }
                else if (constant_diffusion_fast_path)
                {
                    audit_stats
                        ->coefficient_fast_path_constant_diffusion_cells +=
                        1.0;
                }
                else
                {
                    audit_stats->coefficient_fast_path_generic_cells += 1.0;
                }
                if (zero_load_fast_path)
                    audit_stats->coefficient_fast_path_zero_load_cells +=
                        1.0;
            }

            std::array<double, QTime> time_reference_points{};
            std::array<double, QTime> time_reference_weights{};
            std::array<double, QTime> time_physical_points{};
            std::array<typename FluxSpaceType::TimeValues, QTime>
                flux_time_values{};
            std::array<typename ScalarSpaceType::TimeValues, QTime>
                scalar_time_values{};
            const auto time_basis_begin = Clock::now();
            for (int qt = 0; qt < QTime; ++qt)
            {
                const double t_ref = time_rule.points[qt][0];
                time_reference_points[static_cast<std::size_t>(qt)] = t_ref;
                time_reference_weights[static_cast<std::size_t>(qt)] =
                    time_rule.weights[qt];
                time_physical_points[static_cast<std::size_t>(qt)] =
                    flux_space.map_time_to_physical(t_ref);
                FluxSpaceType::evaluate_time_basis(
                    t_ref,
                    flux_time_values[static_cast<std::size_t>(qt)]);
                ScalarSpaceType::evaluate_time_basis(
                    t_ref,
                    scalar_time_values[static_cast<std::size_t>(qt)]);
            }
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_time_basis_seconds,
                time_basis_begin);

            std::array<SpatialGradient, 3> partition_gradients{};
            for (int local_vertex = 0; local_vertex < 3; ++local_vertex)
            {
                partition_gradients[static_cast<std::size_t>(local_vertex)] =
                    map_reference_gradient_to_physical_(
                        map,
                        barycentric_reference_gradient_(local_vertex));
            }
            constexpr int time_component = GT::dim_space_v;
            int spatial_qp_id = 0;
            quadrature::reference::for_each_reference_triangle_duffy_point<
                QSpace>(
                [&](const double x,
                    const double y,
                    const double triangle_weight)
                {
                    const SpatialReferencePoint x_ref{x, y};
                    const auto affine_qpoint_begin =
                        sample_qpoint_subkernels ? Clock::now()
                                                 : Clock::time_point{};
                    const auto xy =
                        FluxSpaceType::PiolaBasis::map_to_physical(
                            map,
                            x_ref);
                    if (sample_qpoint_subkernels)
                        add_elapsed_(
                            audit_stats,
                            &AuditStats::state_fill_affine_map_seconds,
                            affine_qpoint_begin);
                    const auto rt_basis_begin =
                        sample_qpoint_subkernels ? Clock::now()
                                                 : Clock::time_point{};
                    const auto spatial_rt_values =
                        FluxSpaceType::PiolaBasis::eval_all(map, x_ref);
                    const auto spatial_rt_divergences =
                        FluxSpaceType::PiolaBasis::div_all(map, x_ref);
                    if (sample_qpoint_subkernels)
                        add_elapsed_(
                            audit_stats,
                            &AuditStats::state_fill_spatial_rt_basis_seconds,
                            rt_basis_begin);

                    typename ScalarSpaceType::SpatialSpace::LocalValues
                        spatial_scalar_values{};
                    const auto spatial_scalar_begin =
                        sample_qpoint_subkernels ? Clock::now()
                                                 : Clock::time_point{};
                    ScalarSpaceType::SpatialSpace::evaluate_local_basis(
                        x_ref,
                        spatial_scalar_values);
                    if (sample_qpoint_subkernels)
                        add_elapsed_(
                            audit_stats,
                            &AuditStats::state_fill_scalar_basis_seconds,
                            spatial_scalar_begin);

                    for (int qt = 0; qt < QTime; ++qt)
                    {
                        const int qp_id = spatial_qp_id * QTime + qt;
                        auto& qp =
                            cell_data.points[static_cast<std::size_t>(qp_id)];

                        const double t_ref =
                            time_reference_points[
                                static_cast<std::size_t>(qt)];
                        const double time_weight =
                            time_reference_weights[
                                static_cast<std::size_t>(qt)];
                        const double t_phys =
                            time_physical_points[
                                static_cast<std::size_t>(qt)];
                        qp.spatial_reference_point = x_ref;
                        qp.time_reference_point = t_ref;
                        qp.reference_point =
                            SpaceTimeReferencePoint{x_ref[0], x_ref[1], t_ref};
                        qp.physical_point =
                            SpaceTimePoint{xy[0], xy[1], t_phys};
                        qp.jacobian_weight =
                            triangle_weight * time_weight *
                            cell_data.jacobian_measure;

                        const auto& flux_time =
                            flux_time_values[
                                static_cast<std::size_t>(qt)];
                        const auto rt_time_product_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        for (int spatial_local_dof = 0;
                             spatial_local_dof <
                                 FluxSpaceType::spatial_local_dofs_v;
                             ++spatial_local_dof)
                        {
                            const auto& spatial_value =
                                spatial_rt_values[
                                    static_cast<std::size_t>(
                                        spatial_local_dof)];
                            const double spatial_divergence =
                                spatial_rt_divergences[
                                    static_cast<std::size_t>(
                                        spatial_local_dof)];
                            for (int time_dof = 0;
                                 time_dof < FluxSpaceType::n_time_dofs_v;
                                 ++time_dof)
                            {
                                const int local_id =
                                    flux_space.local_dof_index(
                                        spatial_local_dof,
                                        time_dof);
                                const double time_value =
                                    flux_time[
                                        static_cast<std::size_t>(time_dof)];
                                qp.rt_basis_values[
                                    static_cast<std::size_t>(local_id)] =
                                    VectorValue{
                                        spatial_value[0] * time_value,
                                        spatial_value[1] * time_value
                                    };
                                qp.rt_basis_divergences[
                                    static_cast<std::size_t>(local_id)] =
                                    spatial_divergence * time_value;
                            }
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_spatial_rt_basis_seconds,
                                rt_time_product_begin);

                        const auto& scalar_time =
                            scalar_time_values[
                                static_cast<std::size_t>(qt)];
                        const auto scalar_time_product_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        for (int spatial_local_dof = 0;
                             spatial_local_dof <
                                 ScalarSpaceType::spatial_local_dofs_v;
                             ++spatial_local_dof)
                        {
                            const double spatial_value =
                                spatial_scalar_values[
                                    static_cast<std::size_t>(
                                        spatial_local_dof)];
                            for (int time_dof = 0;
                                 time_dof < ScalarSpaceType::n_time_dofs_v;
                                 ++time_dof)
                            {
                                const int local_id =
                                    scalar_space.local_dof_index(
                                        spatial_local_dof,
                                        time_dof);
                                qp.scalar_basis_values[
                                    static_cast<std::size_t>(local_id)] =
                                    spatial_value *
                                    scalar_time[
                                        static_cast<std::size_t>(time_dof)];
                            }
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_scalar_basis_seconds,
                                scalar_time_product_begin);

                        const auto partition_value_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        for (int local_vertex = 0;
                             local_vertex < 3;
                             ++local_vertex)
                        {
                            qp.partition_of_unity_values[
                                static_cast<std::size_t>(local_vertex)] =
                                barycentric_value_(local_vertex, x_ref);
                            qp.partition_of_unity_gradients[
                                static_cast<std::size_t>(local_vertex)] =
                                partition_gradients[
                                    static_cast<std::size_t>(local_vertex)];
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_partition_of_unity_seconds,
                                partition_value_begin);

                        const auto lambda_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        const auto grad_lambda =
                            lambda_tilde.gradient_on_cell(
                                slab_id,
                                slab_cell_id,
                                qp.physical_point,
                                slab_geom);
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_lambda_gradient_seconds,
                                lambda_begin);
                        const auto u_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        const auto grad_u =
                            u_delta.gradient_on_cell(
                                x_cell_id,
                                qp.physical_point,
                                x_geom);
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_u_gradient_seconds,
                                u_begin);
                        qp.grad_lambda_tilde =
                            SpatialGradient{grad_lambda[0], grad_lambda[1]};
                        qp.grad_u_delta =
                            SpatialGradient{grad_u[0], grad_u[1]};
                        qp.grad_theta_tilde =
                            SpatialGradient{
                                qp.grad_lambda_tilde[0] + qp.grad_u_delta[0],
                                qp.grad_lambda_tilde[1] + qp.grad_u_delta[1]};
                        qp.u_time_derivative = grad_u[time_component];
                        const auto load_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        if (zero_load_fast_path)
                        {
                            qp.ell_value = 0.0;
                        }
                        else
                        {
                            qp.ell_value =
                                static_cast<double>(ell(qp.physical_point));
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_load_evaluation_seconds,
                                load_begin);
                        const auto diffusion_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        if (identity_diffusion_fast_path ||
                            constant_diffusion_fast_path)
                        {
                            qp.diffusion_tensor = fast_diffusion_tensor;
                        }
                        else
                        {
                            qp.diffusion_tensor =
                                coefficients::evaluate_diffusion_tensor<
                                    GT::dim_space_v>(
                                    M,
                                    qp.physical_point);
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_diffusion_evaluation_seconds,
                                diffusion_begin);
                        const auto inverse_begin =
                            sample_qpoint_subkernels ? Clock::now()
                                                     : Clock::time_point{};
                        if (identity_diffusion_fast_path ||
                            constant_diffusion_fast_path)
                        {
                            qp.inverse_diffusion_tensor =
                                fast_inverse_diffusion_tensor;
                        }
                        else
                        {
                            qp.inverse_diffusion_tensor =
                                detail::inverse_diffusion_tensor_2d(
                                    qp.diffusion_tensor);
                        }
                        if (sample_qpoint_subkernels)
                            add_elapsed_(
                                audit_stats,
                                &AuditStats::state_fill_diffusion_inverse_seconds,
                                inverse_begin);
                        if (identity_diffusion_fast_path)
                        {
                            qp.M_grad_theta_tilde = qp.grad_theta_tilde;
                        }
                        else
                        {
                            qp.M_grad_theta_tilde =
                                coefficients::apply_M<GT::dim_space_v>(
                                    qp.diffusion_tensor,
                                    qp.grad_theta_tilde);
                        }
                        for (int j = 0;
                             j < FluxSpaceType::local_dofs_v;
                             ++j)
                        {
                            const auto& sigma_j =
                                qp.rt_basis_values[
                                    static_cast<std::size_t>(j)];
                            const auto weighted_inverse_sigma_j =
                                identity_diffusion_fast_path
                                    ? VectorValue{
                                          qp.jacobian_weight * sigma_j[0],
                                          qp.jacobian_weight * sigma_j[1]}
                                    : detail::apply_inverse_tensor_2d(
                                          qp.inverse_diffusion_tensor,
                                          sigma_j,
                                          qp.jacobian_weight);
                            const auto local_A_begin =
                                sample_qpoint_subkernels
                                    ? Clock::now()
                                    : Clock::time_point{};
                            for (int i = 0;
                                 i < FluxSpaceType::local_dofs_v;
                                 ++i)
                            {
                                const auto& sigma_i =
                                    qp.rt_basis_values[
                                        static_cast<std::size_t>(i)];
                                cell_data.local_A(i, j) +=
                                    sigma_i[0] * weighted_inverse_sigma_j[0] +
                                    sigma_i[1] * weighted_inverse_sigma_j[1];
                            }
                            if (sample_qpoint_subkernels)
                                add_elapsed_(
                                    audit_stats,
                                    &AuditStats::state_fill_local_A_assembly_seconds,
                                    local_A_begin);
                            const auto local_B_begin =
                                sample_qpoint_subkernels
                                    ? Clock::now()
                                    : Clock::time_point{};
                            for (int i = 0;
                                 i < ScalarSpaceType::local_dofs_v;
                                 ++i)
                            {
                                cell_data.local_B(i, j) +=
                                    qp.scalar_basis_values[
                                        static_cast<std::size_t>(i)] *
                                    qp.jacobian_weight *
                                    qp.rt_basis_divergences[
                                        static_cast<std::size_t>(j)];
                            }
                            if (sample_qpoint_subkernels)
                                add_elapsed_(
                                    audit_stats,
                                    &AuditStats::state_fill_local_B_assembly_seconds,
                                    local_B_begin);
                        }
                    }

                    ++spatial_qp_id;
                });

            if (spatial_qp_id != n_spatial_quadrature_points_v)
            {
                throw std::runtime_error(
                    "LocalErrorQpointStateCache2D: unexpected Duffy quadrature point count.");
            }
            if (sample_compact_shadow && audit_stats != nullptr)
            {
                compare_compact_split_shadow_(
                    flux_space,
                    scalar_space,
                    patch_cell_index,
                    map,
                    cell_data,
                    *audit_stats);
            }
            add_elapsed_(
                audit_stats,
                &AuditStats::state_fill_total_seconds,
                fill_begin);
        }
    };
}

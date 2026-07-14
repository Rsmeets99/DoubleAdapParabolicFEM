#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <unordered_set>

#include "../assembly/detail/active_cell_locator_time_slab.hpp"
#include "../assembly/detail/space_time_basis_tables.hpp"
#include "../coefficients/diffusion_coefficient.hpp"
#include "../detail/cell_geometry_cache.hpp"
#include "../detail/space_time_capabilities.hpp"
#include "../detail/timing.hpp"
#include "../geometry/cell_geometry.hpp"
#include "detail/time_slab_error_indicator_detail.hpp"
#include "time_slab_cellwise_errors.hpp"
#include "time_slab_equilibrated_flux_reconstruction.hpp"

namespace finite_element::time_slabs
{
    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] CellwiseEquilibratedFluxError<int>
    compute_equilibrated_flux_error_squared_by_source_cell_1plus1d(
        const TimeSlabEquilibratedFluxReconstruction1plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        using ReconstructionType = TimeSlabEquilibratedFluxReconstruction1plus1d<
            Backend,
            XSpaceType,
            SourceYSpaceType>;
        using SlabSpaceType      = typename ReconstructionType::SlabSpaceType;
        using SlabType           = typename SlabSpaceType::SlabType;
        using LocalSpaceType     = typename SlabType::SpaceType;
        using GT                 = typename SlabSpaceType::GT;
        using FETraits           = typename SlabSpaceType::FETraitsType;

        using Tables   =
            finite_element::assembly::detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
        using Geometry =
            finite_element::geometry::CellGeometry<LocalSpaceType, GT::dim_space_v>;

        finite_element::detail::require_1plus1d_time_slab_estimator_capability<GT>();

        CellwiseEquilibratedFluxError<int> result;

        const auto& slab_space   = reconstruction.slab_space_ref();
        const auto& x_space      = u_delta.fespace();

        result.by_source_cell_flux.reserve(slab_space.n_dofs());
        result.by_source_cell_residual.reserve(slab_space.n_dofs());
        finite_element::detail::CellGeometryCache<XSpaceType> x_geometry_cache(
            x_space);
        finite_element::assembly::detail::SourceActiveAncestorCache<XSpaceType> ancestor_cache(
            x_space);

        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
        {
            const auto& slab = slab_space.slab(slab_id);
            finite_element::detail::CellGeometryCache<LocalSpaceType> slab_geometry_cache(
                slab.fespace_ref());

            for (const int slab_cell_id : slab.active_cells())
            {
                const int source_cell_id = slab.source_cell_id(slab_cell_id);
                const int x_cell_id =
                    finite_element::assembly::detail::find_active_ancestor_cell_from_source_cell(
                        ancestor_cache,
                        x_space,
                        source_cell_id);

                const auto& x_geom      = x_geometry_cache.geometry(x_cell_id);
                const auto& slab_geom   = slab_geometry_cache.geometry(slab_cell_id);
                const double jac        = Geometry::jacobian_measure(slab_geom);

                double flux_contribution = 0.0;
                double residual_contribution = 0.0;

                for (int q = 0; q < Tables::n_cell_q; ++q)
                {
                    const auto& xi_q = Tables::cell_rule.points[q];
                    const double w_q = Tables::cell_rule.weights[q];
                    const auto p_q   = Geometry::map_to_physical(slab_geom, xi_q);
                    const double measure_factor =
                        detail::checked_positive_measure_factor(
                            jac,
                            w_q,
                            "compute_equilibrated_flux_error_squared_by_source_cell_1plus1d");

                    const auto grad_lambda_tilde =
                        lambda_tilde.gradient_on_cell(slab_id, slab_cell_id, p_q, slab_geom);
                    const auto grad_u =
                        u_delta.gradient_on_cell(x_cell_id, p_q, x_geom);
                    const double grad_theta_x = grad_lambda_tilde[0] + grad_u[0];

                    const auto flux_evaluation =
                        reconstruction.sigma_and_div_sigma_on_slab_cell(
                            slab_id,
                            slab_cell_id,
                            p_q);
                    const double sigma_q = flux_evaluation.sigma;
                    const double div_sigma_q = flux_evaluation.div_sigma;
                    const coefficients::DiffusionVector<1> sigma_vec{sigma_q};
                    const coefficients::DiffusionVector<1> grad_theta_vec{
                        grad_theta_x
                    };
                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<1>(M, p_q);

                    double flux_term =
                        detail::flux_mismatch_energy<1>(
                            M_q,
                            sigma_vec,
                            grad_theta_vec);

                    flux_term = detail::clamp_small_negative(flux_term);

                    if (!std::isfinite(flux_term) || flux_term < -1.0e-14)
                    {
                        std::ostringstream message;
                        message
                            << std::setprecision(17)
                            << "compute_equilibrated_flux_error_squared_by_source_cell_1plus1d flux: "
                            << "squared quantity is negative or non-finite"
                            << " flux_term=" << flux_term
                            << " slab_id=" << slab_id
                            << " slab_cell_id=" << slab_cell_id
                            << " source_cell_id=" << source_cell_id
                            << " x_cell_id=" << x_cell_id
                            << " q=" << q
                            << " x=" << p_q[0]
                            << " t=" << p_q[1]
                            << " sigma=" << sigma_q
                            << " div_sigma=" << div_sigma_q
                            << " grad_lambda_tilde_x=" << grad_lambda_tilde[0]
                            << " grad_u_x=" << grad_u[0]
                            << " grad_theta_x=" << grad_theta_x
                            << " M=" << M_q[0][0]
                            << ".";
                        throw std::runtime_error(message.str());
                    }

                    const double residual =
                        static_cast<double>(ell(p_q)) - grad_u[GT::dim_space_v] - div_sigma_q;

                    flux_contribution +=
                        detail::checked_nonnegative_contribution(
                            flux_term,
                            measure_factor,
                            "compute_equilibrated_flux_error_squared_by_source_cell_1plus1d flux");
                    residual_contribution +=
                        detail::checked_nonnegative_contribution(
                            residual * residual,
                            measure_factor,
                            "compute_equilibrated_flux_error_squared_by_source_cell_1plus1d residual");
                }

                detail::add_to_map(
                    result.by_source_cell_flux,
                    source_cell_id,
                    flux_contribution);
                detail::add_to_map(
                    result.by_source_cell_residual,
                    source_cell_id,
                    residual_contribution);
            }
        }

        detail::require_nonnegative_cellwise_map(
            result.by_source_cell_flux,
            "compute_equilibrated_flux_error_squared_by_source_cell_1plus1d flux aggregation");
        detail::require_nonnegative_cellwise_map(
            result.by_source_cell_residual,
            "compute_equilibrated_flux_error_squared_by_source_cell_1plus1d residual aggregation");

        return result;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] CellwiseEquilibratedFluxError<int>
    compute_equilibrated_flux_error_squared_by_source_cell_2plus1d(
        const TimeSlabEquilibratedFluxReconstruction2plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        using ReconstructionType = TimeSlabEquilibratedFluxReconstruction2plus1d<
            Backend,
            XSpaceType,
            SourceYSpaceType>;
        using SlabSpaceType      = typename ReconstructionType::SlabSpaceType;
        using SlabType           = typename SlabSpaceType::SlabType;
        using LocalSpaceType     = typename SlabType::SpaceType;
        using GT                 = typename SlabSpaceType::GT;
        using FETraits           = typename SlabSpaceType::FETraitsType;
        using FluxSpaceType      = typename ReconstructionType::FluxSpaceType;
        using FluxEvaluation     = typename ReconstructionType::FluxEvaluation;

        static_assert(
            GT::dim_space_v == 2 && GT::dim_time_v == 1,
            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d requires a 2+1D space-time discretization.");

        using Tables =
            finite_element::assembly::detail::SpaceTimeBasisTables<
                GT,
                FETraits,
                QSpace,
                QTime>;
        static_assert(Tables::n_cell_q > 0);

        CellwiseEquilibratedFluxError<int> result;

        const auto& slab_space = reconstruction.slab_space_ref();
        const auto& x_space    = u_delta.fespace();

        result.by_source_cell_flux.reserve(slab_space.n_dofs());
        result.by_source_cell_residual.reserve(slab_space.n_dofs());
        finite_element::detail::CellGeometryCache<XSpaceType> x_geometry_cache(
            x_space);
        finite_element::assembly::detail::SourceActiveAncestorCache<XSpaceType> ancestor_cache(
            x_space);

        using Clock = std::chrono::steady_clock;
        const bool collect_timing = timing.enabled();
        const auto elapsed_seconds = [](const Clock::time_point start)
        {
            return std::chrono::duration<double>(Clock::now() - start).count();
        };

        double gradient_evaluation_seconds = 0.0;
        double sigma_divergence_evaluation_seconds = 0.0;
        double diffusion_tensor_evaluation_seconds = 0.0;
        double map_accumulation_seconds = 0.0;
        std::size_t cells_visited = 0;
        std::size_t qpoints_visited = 0;
        std::size_t map_insertions = 0;
        const auto slab_cell_loop_start = Clock::now();

        const auto& patch_set = reconstruction.patch_set();
        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
        {
            const auto& slab = slab_space.slab(slab_id);
            finite_element::detail::CellGeometryCache<LocalSpaceType> slab_geometry_cache(
                slab.fespace_ref());

            for (const int slab_cell_id : slab.active_cells())
            {
                ++cells_visited;
                const int membership_count =
                    patch_set.cell_patch_count(slab_id, slab_cell_id);
                if (membership_count <= 0)
                {
                    throw std::runtime_error(
                        "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d: slab cell has no vertex-patch memberships.");
                }

                const auto& first_membership =
                    patch_set.cell_patch_membership(slab_id, slab_cell_id, 0);
                const auto& first_flux_space =
                    reconstruction.flux_space(first_membership.patch_id);
                const auto cell_map =
                    first_flux_space.physical_map_for_patch_cell(
                        first_membership.patch_cell_index);

                const int source_cell_id = slab.source_cell_id(slab_cell_id);
                const int x_cell_id =
                    finite_element::assembly::detail::find_active_ancestor_cell_from_source_cell(
                        ancestor_cache,
                        x_space,
                        source_cell_id);

                const auto& x_geom    = x_geometry_cache.geometry(x_cell_id);
                const auto& slab_geom = slab_geometry_cache.geometry(slab_cell_id);

                double flux_contribution = 0.0;
                double residual_contribution = 0.0;

                for (int q = 0; q < Tables::n_cell_q; ++q)
                {
                    ++qpoints_visited;
                    const auto& xi_q = Tables::cell_rule.points[q];
                    const double w_q = Tables::cell_rule.weights[q];
                    const auto p_q =
                        finite_element::geometry::CellGeometry<
                            LocalSpaceType,
                            GT::dim_space_v>::map_to_physical(
                                slab_geom,
                                xi_q);
                    const double measure_factor =
                        detail::checked_positive_measure_factor(
                            finite_element::geometry::CellGeometry<
                                LocalSpaceType,
                                GT::dim_space_v>::jacobian_measure(slab_geom),
                            w_q,
                            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d fast");

                    const auto gradient_start = Clock::now();
                    const auto grad_lambda_tilde =
                        lambda_tilde.gradient_on_cell(
                            slab_id,
                            slab_cell_id,
                            p_q,
                            slab_geom);
                    const auto grad_u =
                        u_delta.gradient_on_cell(x_cell_id, p_q, x_geom);
                    if (collect_timing)
                        gradient_evaluation_seconds +=
                            elapsed_seconds(gradient_start);

                    const double grad_theta_x =
                        grad_lambda_tilde[0] + grad_u[0];
                    const double grad_theta_y =
                        grad_lambda_tilde[1] + grad_u[1];

                    const auto sigma_start = Clock::now();
                    typename FluxSpaceType::LocalValues rt_basis_values{};
                    typename FluxSpaceType::LocalDivergences rt_basis_divergences{};
                    const typename FluxSpaceType::SpatialReferencePoint x_ref{
                        xi_q[0],
                        xi_q[1]
                    };
                    const double t_ref = xi_q[2];
                    const auto spatial_values =
                        FluxSpaceType::PiolaBasis::eval_all(cell_map, x_ref);
                    const auto spatial_divergences =
                        FluxSpaceType::PiolaBasis::div_all(cell_map, x_ref);
                    typename FluxSpaceType::TimeValues time_values{};
                    FluxSpaceType::evaluate_time_basis(t_ref, time_values);

                    for (int spatial_local_dof = 0;
                         spatial_local_dof < FluxSpaceType::spatial_local_dofs_v;
                         ++spatial_local_dof)
                    {
                        for (int time_dof = 0;
                             time_dof < FluxSpaceType::n_time_dofs_v;
                             ++time_dof)
                        {
                            const int local_dof_id =
                                spatial_local_dof *
                                    FluxSpaceType::n_time_dofs_v +
                                time_dof;
                            const double time_value =
                                time_values[static_cast<std::size_t>(time_dof)];
                            const auto& spatial_value =
                                spatial_values[
                                    static_cast<std::size_t>(spatial_local_dof)];
                            rt_basis_values[static_cast<std::size_t>(local_dof_id)] =
                                typename FluxSpaceType::VectorValue{
                                    spatial_value[0] * time_value,
                                    spatial_value[1] * time_value
                                };
                            rt_basis_divergences[
                                static_cast<std::size_t>(local_dof_id)] =
                                spatial_divergences[
                                    static_cast<std::size_t>(spatial_local_dof)] *
                                time_value;
                        }
                    }

                    FluxEvaluation flux_evaluation;
                    for (int membership_index = 0;
                         membership_index < membership_count;
                         ++membership_index)
                    {
                        const auto& membership =
                            patch_set.cell_patch_membership(
                                slab_id,
                                slab_cell_id,
                                membership_index);
                        const auto& flux_function =
                            reconstruction.flux_function(membership.patch_id);

                        for (int local_dof_id = 0;
                             local_dof_id < FluxSpaceType::local_dofs_v;
                             ++local_dof_id)
                        {
                            const double coefficient =
                                flux_function.local_coefficient(
                                    membership.patch_cell_index,
                                    local_dof_id);
                            if (coefficient == 0.0)
                                continue;

                            const auto& phi =
                                rt_basis_values[
                                    static_cast<std::size_t>(local_dof_id)];
                            flux_evaluation.sigma[0] += coefficient * phi[0];
                            flux_evaluation.sigma[1] += coefficient * phi[1];
                            flux_evaluation.div_sigma +=
                                coefficient *
                                rt_basis_divergences[
                                    static_cast<std::size_t>(local_dof_id)];
                        }
                    }
                    if (collect_timing)
                        sigma_divergence_evaluation_seconds +=
                            elapsed_seconds(sigma_start);

                    const auto& sigma_q = flux_evaluation.sigma;
                    const double div_sigma_q = flux_evaluation.div_sigma;
                    const coefficients::DiffusionVector<2> grad_theta_vec{
                        grad_theta_x,
                        grad_theta_y
                    };

                    const auto diffusion_start = Clock::now();
                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<2>(M, p_q);
                    if (collect_timing)
                        diffusion_tensor_evaluation_seconds +=
                            elapsed_seconds(diffusion_start);

                    double flux_term =
                        detail::flux_mismatch_energy<2>(
                            M_q,
                            sigma_q,
                            grad_theta_vec);

                    flux_term = detail::clamp_small_negative(flux_term);

                    const double residual =
                        static_cast<double>(ell(p_q)) -
                        grad_u[GT::dim_space_v] -
                        div_sigma_q;

                    flux_contribution +=
                        detail::checked_nonnegative_contribution(
                            flux_term,
                            measure_factor,
                            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d fast flux");
                    residual_contribution +=
                        detail::checked_nonnegative_contribution(
                            residual * residual,
                            measure_factor,
                            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d fast residual");
                }

                const auto map_start = Clock::now();
                detail::add_to_map(
                    result.by_source_cell_flux,
                    source_cell_id,
                    flux_contribution);
                detail::add_to_map(
                    result.by_source_cell_residual,
                    source_cell_id,
                    residual_contribution);
                map_insertions += 2u;
                if (collect_timing)
                    map_accumulation_seconds += elapsed_seconds(map_start);
            }
        }

        const auto finalize_start = Clock::now();

        if (collect_timing)
        {
            timing.add(
                "time_slab.flux_diagnostics.slab_cell_loop",
                elapsed_seconds(slab_cell_loop_start));
            timing.add(
                "time_slab.flux_diagnostics.gradient_evaluation",
                gradient_evaluation_seconds);
            timing.add(
                "time_slab.flux_diagnostics.sigma_divergence_evaluation",
                sigma_divergence_evaluation_seconds);
            timing.add(
                "time_slab.flux_diagnostics.diffusion_tensor_evaluation",
                diffusion_tensor_evaluation_seconds);
            timing.add(
                "time_slab.flux_diagnostics.map_accumulation",
                map_accumulation_seconds);
            timing.add("time_slab.flux_diagnostics.state_build_wall", 0.0);
            timing.add(
                "time_slab.flux_diagnostics.diagnostic_state_built.count",
                0.0);
            timing.add(
                "time_slab.flux_diagnostics.cells_visited.count",
                static_cast<double>(cells_visited));
            timing.add(
                "time_slab.flux_diagnostics.qpoints_visited.count",
                static_cast<double>(qpoints_visited));
            timing.add(
                "time_slab.flux_diagnostics.source_cells_touched.count",
                static_cast<double>(
                    std::max(
                        result.by_source_cell_flux.size(),
                        result.by_source_cell_residual.size())));
            timing.add(
                "time_slab.flux_diagnostics.map_insertions.count",
                static_cast<double>(map_insertions));
            timing.add(
                "time_slab.flux_diagnostics.allocations.count",
                2.0);
        }

        detail::require_nonnegative_cellwise_map(
            result.by_source_cell_flux,
            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d fast flux aggregation");
        detail::require_nonnegative_cellwise_map(
            result.by_source_cell_residual,
            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d fast residual aggregation");
        if (collect_timing)
        {
            timing.add(
                "time_slab.flux_diagnostics.source_finalize_sort",
                elapsed_seconds(finalize_start));
        }

        return result;
    }

    struct EquilibratedFluxDiagnostics2plus1dDebugComparison
    {
        double fast_total_flux = 0.0;
        double reference_total_flux = 0.0;
        double fast_total_residual = 0.0;
        double reference_total_residual = 0.0;
        double max_abs_flux_difference = 0.0;
        double max_abs_residual_difference = 0.0;

        [[nodiscard]] bool within(double tolerance) const noexcept
        {
            return std::abs(fast_total_flux - reference_total_flux) <= tolerance &&
                   std::abs(fast_total_residual - reference_total_residual) <= tolerance &&
                   max_abs_flux_difference <= tolerance &&
                   max_abs_residual_difference <= tolerance;
        }
    };

    template<class MapType>
    [[nodiscard]] inline double map_value_or_zero(
        const MapType& map,
        typename MapType::key_type key)
    {
        const auto it = map.find(key);
        return it == map.end() ? 0.0 : it->second;
    }

    template<class MapType>
    [[nodiscard]] inline double max_abs_map_difference(
        const MapType& a,
        const MapType& b)
    {
        using KeyType = typename MapType::key_type;

        std::unordered_set<KeyType> keys;
        keys.reserve(a.size() + b.size());
        for (const auto& [key, value] : a)
        {
            (void)value;
            keys.insert(key);
        }
        for (const auto& [key, value] : b)
        {
            (void)value;
            keys.insert(key);
        }

        double max_difference = 0.0;
        for (const auto key : keys)
        {
            max_difference =
                std::max(
                    max_difference,
                    std::abs(
                        map_value_or_zero(a, key) -
                        map_value_or_zero(b, key)));
        }

        return max_difference;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] EquilibratedFluxDiagnostics2plus1dDebugComparison
    debug_compare_equilibrated_flux_error_squared_by_source_cell_2plus1d(
        const TimeSlabEquilibratedFluxReconstruction2plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        const auto fast =
            compute_equilibrated_flux_error_squared_by_source_cell_2plus1d<
                QSpace,
                QTime>(
                reconstruction,
                lambda_tilde,
                u_delta,
                ell,
                M);
        const auto reference =
            compute_equilibrated_flux_error_squared_by_source_cell_2plus1d_reference<
                QSpace,
                QTime>(
                reconstruction,
                lambda_tilde,
                u_delta,
                ell,
                M);

        EquilibratedFluxDiagnostics2plus1dDebugComparison comparison;
        comparison.fast_total_flux = fast.total_flux();
        comparison.reference_total_flux = reference.total_flux();
        comparison.fast_total_residual = fast.total_residual();
        comparison.reference_total_residual = reference.total_residual();
        comparison.max_abs_flux_difference =
            max_abs_map_difference(
                fast.by_source_cell_flux,
                reference.by_source_cell_flux);
        comparison.max_abs_residual_difference =
            max_abs_map_difference(
                fast.by_source_cell_residual,
                reference.by_source_cell_residual);
        return comparison;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] CellwiseEquilibratedFluxError<int>
    compute_equilibrated_flux_error_squared_by_source_cell_2plus1d_reference(
        const TimeSlabEquilibratedFluxReconstruction2plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M,
        const finite_element::detail::TimingRecorder& timing = {})
    {
        using ReconstructionType = TimeSlabEquilibratedFluxReconstruction2plus1d<
            Backend,
            XSpaceType,
            SourceYSpaceType>;
        using SlabSpaceType      = typename ReconstructionType::SlabSpaceType;
        using SlabType           = typename SlabSpaceType::SlabType;
        using LocalSpaceType     = typename SlabType::SpaceType;
        using GT                 = typename SlabSpaceType::GT;
        using FETraits           = typename SlabSpaceType::FETraitsType;

        static_assert(
            GT::dim_space_v == 2 && GT::dim_time_v == 1,
            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d requires a 2+1D space-time discretization.");

        using Tables   =
            finite_element::assembly::detail::SpaceTimeBasisTables<GT, FETraits, QSpace, QTime>;
        using Geometry =
            finite_element::geometry::CellGeometry<LocalSpaceType, GT::dim_space_v>;

        CellwiseEquilibratedFluxError<int> result;

        const auto& slab_space = reconstruction.slab_space_ref();
        const auto& x_space    = u_delta.fespace();

        result.by_source_cell_flux.reserve(slab_space.n_dofs());
        result.by_source_cell_residual.reserve(slab_space.n_dofs());
        finite_element::detail::CellGeometryCache<XSpaceType> x_geometry_cache(
            x_space);
        finite_element::assembly::detail::SourceActiveAncestorCache<XSpaceType> ancestor_cache(
            x_space);

        using Clock = std::chrono::steady_clock;
        const bool collect_timing = timing.enabled();
        const auto elapsed_seconds = [](const Clock::time_point start)
        {
            return std::chrono::duration<double>(Clock::now() - start).count();
        };

        double gradient_evaluation_seconds = 0.0;
        double sigma_divergence_evaluation_seconds = 0.0;
        double diffusion_tensor_evaluation_seconds = 0.0;
        double map_accumulation_seconds = 0.0;
        std::size_t cells_visited = 0;
        std::size_t qpoints_visited = 0;
        std::size_t map_insertions = 0;
        const auto slab_cell_loop_start = Clock::now();

        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
        {
            const auto& slab = slab_space.slab(slab_id);
            finite_element::detail::CellGeometryCache<LocalSpaceType> slab_geometry_cache(
                slab.fespace_ref());

            for (const int slab_cell_id : slab.active_cells())
            {
                ++cells_visited;
                const int source_cell_id = slab.source_cell_id(slab_cell_id);
                const int x_cell_id =
                    finite_element::assembly::detail::find_active_ancestor_cell_from_source_cell(
                        ancestor_cache,
                        x_space,
                        source_cell_id);

                const auto& x_geom    = x_geometry_cache.geometry(x_cell_id);
                const auto& slab_geom = slab_geometry_cache.geometry(slab_cell_id);
                const double jac      = Geometry::jacobian_measure(slab_geom);

                double flux_contribution = 0.0;
                double residual_contribution = 0.0;

                for (int q = 0; q < Tables::n_cell_q; ++q)
                {
                    ++qpoints_visited;
                    const auto& xi_q = Tables::cell_rule.points[q];
                    const double w_q = Tables::cell_rule.weights[q];
                    const auto p_q   = Geometry::map_to_physical(slab_geom, xi_q);
                    const double measure_factor =
                        detail::checked_positive_measure_factor(
                            jac,
                            w_q,
                            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d");

                    const auto gradient_start = Clock::now();
                    const auto grad_lambda_tilde =
                        lambda_tilde.gradient_on_cell(slab_id, slab_cell_id, p_q, slab_geom);
                    const auto grad_u =
                        u_delta.gradient_on_cell(x_cell_id, p_q, x_geom);
                    if (collect_timing)
                        gradient_evaluation_seconds +=
                            elapsed_seconds(gradient_start);

                    const double grad_theta_x =
                        grad_lambda_tilde[0] + grad_u[0];
                    const double grad_theta_y =
                        grad_lambda_tilde[1] + grad_u[1];

                    const auto sigma_start = Clock::now();
                    const auto flux_evaluation =
                        reconstruction.sigma_and_div_sigma_on_slab_cell(
                            slab_id,
                            slab_cell_id,
                            p_q);
                    if (collect_timing)
                        sigma_divergence_evaluation_seconds +=
                            elapsed_seconds(sigma_start);
                    const auto& sigma_q = flux_evaluation.sigma;
                    const double div_sigma_q = flux_evaluation.div_sigma;
                    const coefficients::DiffusionVector<2> grad_theta_vec{
                        grad_theta_x,
                        grad_theta_y
                    };
                    const auto diffusion_start = Clock::now();
                    const auto M_q =
                        coefficients::evaluate_diffusion_tensor<2>(M, p_q);
                    if (collect_timing)
                        diffusion_tensor_evaluation_seconds +=
                            elapsed_seconds(diffusion_start);

                    double flux_term =
                        detail::flux_mismatch_energy<2>(
                            M_q,
                            sigma_q,
                            grad_theta_vec);

                    flux_term = detail::clamp_small_negative(flux_term);

                    const double residual =
                        static_cast<double>(ell(p_q)) -
                        grad_u[GT::dim_space_v] -
                        div_sigma_q;

                    flux_contribution +=
                        detail::checked_nonnegative_contribution(
                            flux_term,
                            measure_factor,
                            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d flux");
                    residual_contribution +=
                        detail::checked_nonnegative_contribution(
                            residual * residual,
                            measure_factor,
                            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d residual");
                }

                const auto map_start = Clock::now();
                detail::add_to_map(
                    result.by_source_cell_flux,
                    source_cell_id,
                    flux_contribution);
                detail::add_to_map(
                    result.by_source_cell_residual,
                    source_cell_id,
                    residual_contribution);
                map_insertions += 2u;
                if (collect_timing)
                    map_accumulation_seconds += elapsed_seconds(map_start);
            }
        }

        const auto finalize_start = Clock::now();

        if (collect_timing)
        {
            timing.add(
                "time_slab.flux_diagnostics.slab_cell_loop",
                elapsed_seconds(slab_cell_loop_start));
            timing.add(
                "time_slab.flux_diagnostics.gradient_evaluation",
                gradient_evaluation_seconds);
            timing.add(
                "time_slab.flux_diagnostics.sigma_divergence_evaluation",
                sigma_divergence_evaluation_seconds);
            timing.add(
                "time_slab.flux_diagnostics.diffusion_tensor_evaluation",
                diffusion_tensor_evaluation_seconds);
            timing.add(
                "time_slab.flux_diagnostics.map_accumulation",
                map_accumulation_seconds);
            timing.add("time_slab.flux_diagnostics.state_build_wall", 0.0);
            timing.add(
                "time_slab.flux_diagnostics.diagnostic_state_built.count",
                0.0);
            timing.add(
                "time_slab.flux_diagnostics.cells_visited.count",
                static_cast<double>(cells_visited));
            timing.add(
                "time_slab.flux_diagnostics.qpoints_visited.count",
                static_cast<double>(qpoints_visited));
            timing.add(
                "time_slab.flux_diagnostics.source_cells_touched.count",
                static_cast<double>(
                    std::max(
                        result.by_source_cell_flux.size(),
                        result.by_source_cell_residual.size())));
            timing.add(
                "time_slab.flux_diagnostics.map_insertions.count",
                static_cast<double>(map_insertions));
            timing.add(
                "time_slab.flux_diagnostics.allocations.count",
                2.0);
        }

        detail::require_nonnegative_cellwise_map(
            result.by_source_cell_flux,
            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d flux aggregation");
        detail::require_nonnegative_cellwise_map(
            result.by_source_cell_residual,
            "compute_equilibrated_flux_error_squared_by_source_cell_2plus1d residual aggregation");
        if (collect_timing)
        {
            timing.add(
                "time_slab.flux_diagnostics.source_finalize_sort",
                elapsed_seconds(finalize_start));
        }

        return result;
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_equilibrated_flux_squared_by_source_cell_1plus1d(
        const TimeSlabEquilibratedFluxReconstruction1plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        return compute_equilibrated_flux_error_squared_by_source_cell_1plus1d<
            QSpace,
            QTime>(
                reconstruction,
                lambda_tilde,
                u_delta,
                ell,
                M)
            .equilibrated_flux_part();
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_equilibrated_flux_squared_by_source_cell_2plus1d(
        const TimeSlabEquilibratedFluxReconstruction2plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        return compute_equilibrated_flux_error_squared_by_source_cell_2plus1d<
            QSpace,
            QTime>(
                reconstruction,
                lambda_tilde,
                u_delta,
                ell,
                M)
            .equilibrated_flux_part();
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_divergence_residual_squared_by_source_cell_1plus1d(
        const TimeSlabEquilibratedFluxReconstruction1plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        return compute_equilibrated_flux_error_squared_by_source_cell_1plus1d<
            QSpace,
            QTime>(
                reconstruction,
                lambda_tilde,
                u_delta,
                ell,
                M)
            .divergence_residual_part();
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XSpaceType,
        class SourceYSpaceType,
        class ReconstructedFunctionType,
        class XFunctionType,
        class EllFunction,
        class MFunction>
    [[nodiscard]] CellwiseSquaredError<int>
    compute_divergence_residual_squared_by_source_cell_2plus1d(
        const TimeSlabEquilibratedFluxReconstruction2plus1d<Backend, XSpaceType, SourceYSpaceType>& reconstruction,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        const EllFunction& ell,
        const MFunction& M)
    {
        return compute_equilibrated_flux_error_squared_by_source_cell_2plus1d<
            QSpace,
            QTime>(
                reconstruction,
                lambda_tilde,
                u_delta,
                ell,
                M)
            .divergence_residual_part();
    }
}

#pragma once

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../assembly/detail/active_cell_locator_time_slab.hpp"
#include "../assembly/detail/space_time_basis_tables.hpp"
#include "../detail/cell_geometry_cache.hpp"
#include "../geometry/cell_geometry.hpp"

namespace finite_element::time_slabs
{
    namespace detail
    {
        [[nodiscard]] inline bool estimator_input_trace_enabled()
        {
            const char* value = std::getenv("ADAPPFEM_ESTIMATOR_INPUT_TRACE");
            return value != nullptr && value[0] != '\0' && value[0] != '0';
        }

        [[nodiscard]] inline int estimator_input_trace_limit()
        {
            const char* value = std::getenv("ADAPPFEM_ESTIMATOR_INPUT_TRACE_LIMIT");
            if (value == nullptr || value[0] == '\0')
                return 64;

            try
            {
                const int parsed = std::stoi(value);
                return parsed > 0 ? parsed : 64;
            }
            catch (...)
            {
                return 64;
            }
        }

        template<std::size_t N>
        [[nodiscard]] inline std::array<double, N> nan_array()
        {
            std::array<double, N> out{};
            out.fill(std::numeric_limits<double>::quiet_NaN());
            return out;
        }

        template<class ArrayLike>
        [[nodiscard]] inline bool all_finite(const ArrayLike& values)
        {
            for (const double value : values)
            {
                if (!std::isfinite(value))
                    return false;
            }
            return true;
        }

        template<class ArrayLike>
        inline void append_array(
            std::ostream& out,
            const ArrayLike& values)
        {
            out << '[';
            for (std::size_t i = 0; i < values.size(); ++i)
            {
                if (i != 0)
                    out << ", ";
                out << values[i];
            }
            out << ']';
        }

        template<class FunctionType>
        inline void require_function_coefficients_consistent(
            const FunctionType& function,
            std::string_view label)
        {
            const auto& dof_handler = function.fespace().dof_handler_ref();
            const auto& true_coefficients = function.true_coefficients();
            const auto& full_coefficients = function.full_coefficients();

            if (true_coefficients.size() != dof_handler.n_true_dofs())
            {
                std::ostringstream message;
                message
                    << "validate_time_slab_estimator_inputs: "
                    << label
                    << " true coefficient vector has wrong size.";
                throw std::runtime_error(message.str());
            }

            if (full_coefficients.size() != dof_handler.n_dofs())
            {
                std::ostringstream message;
                message
                    << "validate_time_slab_estimator_inputs: "
                    << label
                    << " full coefficient vector has wrong size.";
                throw std::runtime_error(message.str());
            }

            for (int gid = 0; gid < dof_handler.n_dofs(); ++gid)
            {
                const double full_value = full_coefficients[gid];
                if (!std::isfinite(full_value))
                {
                    std::ostringstream message;
                    message
                        << "validate_time_slab_estimator_inputs: "
                        << label
                        << " full coefficient "
                        << gid
                        << " is not finite.";
                    throw std::runtime_error(message.str());
                }

                const auto& dof = dof_handler.dof(gid);
                double expected = 0.0;

                if (!dof.is_constrained)
                {
                    if (dof.true_dof_id < 0 ||
                        dof.true_dof_id >= dof_handler.n_true_dofs())
                    {
                        std::ostringstream message;
                        message
                            << "validate_time_slab_estimator_inputs: "
                            << label
                            << " unconstrained DoF has invalid true id.";
                        throw std::runtime_error(message.str());
                    }

                    expected = true_coefficients[dof.true_dof_id];
                }
                else
                {
                    if (dof.constraint_masters.size() !=
                        dof.constraint_weights.size())
                    {
                        std::ostringstream message;
                        message
                            << "validate_time_slab_estimator_inputs: "
                            << label
                            << " constrained DoF has inconsistent metadata.";
                        throw std::runtime_error(message.str());
                    }

                    for (std::size_t k = 0;
                         k < dof.constraint_masters.size();
                         ++k)
                    {
                        const int master_gid = dof.constraint_masters[k];
                        if (master_gid < 0 ||
                            master_gid >= dof_handler.n_dofs())
                        {
                            std::ostringstream message;
                            message
                                << "validate_time_slab_estimator_inputs: "
                                << label
                                << " constrained DoF references invalid master.";
                            throw std::runtime_error(message.str());
                        }

                        const auto& master_dof = dof_handler.dof(master_gid);
                        if (master_dof.is_constrained)
                        {
                            std::ostringstream message;
                            message
                                << "validate_time_slab_estimator_inputs: "
                                << label
                                << " constrained DoF references constrained master.";
                            throw std::runtime_error(message.str());
                        }

                        expected +=
                            dof.constraint_weights[k] *
                            full_coefficients[master_gid];
                    }
                }

                const double scale = 1.0 + std::abs(expected);
                if (std::abs(full_value - expected) > 1.0e-10 * scale)
                {
                    std::ostringstream message;
                    message
                        << "validate_time_slab_estimator_inputs: "
                        << label
                        << " coefficient scatter mismatch for gid "
                        << gid
                        << ": full="
                        << full_value
                        << " expected="
                        << expected
                        << '.';
                    throw std::runtime_error(message.str());
                }
            }
        }

        template<class SpaceType>
        inline void require_valid_active_cell_dofs(
            const SpaceType& space,
            std::string_view label)
        {
            const auto& dof_handler = space.dof_handler_ref();
            if (!dof_handler.validate())
            {
                std::ostringstream message;
                message
                    << "validate_time_slab_estimator_inputs: "
                    << label
                    << " DoF handler validation failed.";
                throw std::runtime_error(message.str());
            }

            for (const int cell_id : space.active_cells())
            {
                const auto& cell_dofs = dof_handler.cell_dofs(cell_id);
                for (std::size_t local = 0; local < cell_dofs.size(); ++local)
                {
                    const int gid = cell_dofs[local];
                    if (gid == -1)
                        continue;

                    if (gid < 0 || gid >= dof_handler.n_dofs())
                    {
                        std::ostringstream message;
                        message
                            << "validate_time_slab_estimator_inputs: "
                            << label
                            << " active cell "
                            << cell_id
                            << " local DoF "
                            << local
                            << " has invalid global id "
                            << gid
                            << '.';
                        throw std::runtime_error(message.str());
                    }
                }
            }
        }

        template<class FunctionType>
        inline void append_cell_dofs(
            std::ostream& out,
            std::string_view label,
            const FunctionType& function,
            int cell_id)
        {
            out << label << "_cell=" << cell_id << " dofs=[";

            const auto& space = function.fespace();
            if (!space.is_active_cell(cell_id))
            {
                out << "inactive]\n";
                return;
            }

            const auto& dof_handler = space.dof_handler_ref();
            const auto& coeffs = function.full_coefficients();
            const auto& dofs = dof_handler.cell_dofs(cell_id);

            for (std::size_t local = 0; local < dofs.size(); ++local)
            {
                if (local != 0)
                    out << ", ";

                const int gid = dofs[local];
                out << local << ':';
                if (gid < 0)
                {
                    out << "eliminated";
                    continue;
                }

                out << gid << ':';
                if (gid >= coeffs.size())
                    out << "invalid";
                else
                    out << coeffs[gid];
            }

            out << "]\n";
        }
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType,
        class XFunctionType>
    void validate_time_slab_estimator_inputs(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta,
        std::ostream* trace_stream = nullptr,
        int trace_limit = 0)
    {
        using SourceSpace = typename SourceFunctionType::SpaceType;
        using XSpace = typename XFunctionType::SpaceType;
        using SlabSpaceType = typename ReconstructedFunctionType::SlabSpaceType;
        using LocalSlabSpace = typename ReconstructedFunctionType::LocalSpaceType;
        using GT = typename SourceSpace::GT;
        using FETraits = typename SourceSpace::FETraitsType;

        static_assert(
            SourceSpace::GT::dim_space_v == 2 && SourceSpace::GT::dim_time_v == 1,
            "validate_time_slab_estimator_inputs currently targets 2+1D.");

        if (&lambda_delta.fespace().mesh_ref() != &u_delta.fespace().mesh_ref())
        {
            throw std::runtime_error(
                "validate_time_slab_estimator_inputs: X and Y spaces do not share the same mesh.");
        }

        const SlabSpaceType& slab_space = lambda_tilde.slab_space();
        if (&slab_space.source_space().mesh_ref() !=
            &lambda_delta.fespace().mesh_ref())
        {
            throw std::runtime_error(
                "validate_time_slab_estimator_inputs: slab source space does not share the Y mesh.");
        }

        detail::require_valid_active_cell_dofs(lambda_delta.fespace(), "Y source");
        detail::require_valid_active_cell_dofs(u_delta.fespace(), "X");
        detail::require_function_coefficients_consistent(lambda_delta, "lambda_delta");
        detail::require_function_coefficients_consistent(u_delta, "u_delta");

        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
        {
            const auto& slab_function = lambda_tilde.slab_function(slab_id);
            detail::require_valid_active_cell_dofs(
                slab_function.fespace(),
                "lambda_tilde slab");
            detail::require_function_coefficients_consistent(
                slab_function,
                "lambda_tilde slab");
        }

        using Tables =
            finite_element::assembly::detail::SpaceTimeBasisTables<
                GT,
                FETraits,
                QSpace,
                QTime>;
        using SourceGeometry =
            finite_element::geometry::CellGeometry<
                SourceSpace,
                GT::dim_space_v>;
        using XGeometry =
            finite_element::geometry::CellGeometry<
                XSpace,
                GT::dim_space_v>;
        using SlabGeometry =
            finite_element::geometry::CellGeometry<
                LocalSlabSpace,
                GT::dim_space_v>;

        ::finite_element::detail::CellGeometryCache<SourceSpace>
            source_geometry(lambda_delta.fespace());
        ::finite_element::detail::CellGeometryCache<XSpace>
            x_geometry(u_delta.fespace());
        finite_element::assembly::detail::SourceActiveAncestorCache<XSpace>
            x_ancestor_cache(u_delta.fespace());

        int traces_written = 0;

        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
        {
            const auto& slab = slab_space.slab(slab_id);
            const auto& slab_function = lambda_tilde.slab_function(slab_id);
            ::finite_element::detail::CellGeometryCache<LocalSlabSpace>
                slab_geometry(slab.fespace_ref());

            for (const int slab_cell_id : slab.active_cells())
            {
                const int source_cell_id = slab.source_cell_id(slab_cell_id);
                if (!lambda_delta.fespace().is_active_cell(source_cell_id))
                {
                    std::ostringstream message;
                    message
                        << "validate_time_slab_estimator_inputs: slab cell "
                        << slab_cell_id
                        << " in slab "
                        << slab_id
                        << " maps to inactive Y source cell "
                        << source_cell_id
                        << '.';
                    throw std::runtime_error(message.str());
                }

                const int x_cell_id = x_ancestor_cache.find(source_cell_id);
                if (!u_delta.fespace().is_active_cell(x_cell_id))
                {
                    std::ostringstream message;
                    message
                        << "validate_time_slab_estimator_inputs: source cell "
                        << source_cell_id
                        << " maps to inactive X ancestor "
                        << x_cell_id
                        << '.';
                    throw std::runtime_error(message.str());
                }

                const auto& source_geom =
                    source_geometry.geometry(source_cell_id);
                const auto& x_geom = x_geometry.geometry(x_cell_id);
                const auto& slab_geom =
                    slab_geometry.geometry(slab_cell_id);

                for (int q = 0; q < Tables::n_cell_q; ++q)
                {
                    const auto xi_slab =
                        Tables::cell_rule.points[static_cast<std::size_t>(q)];
                    const auto x_phys =
                        SlabGeometry::map_to_physical(slab_geom, xi_slab);
                    const auto xi_source =
                        SourceGeometry::physical_to_reference(
                            source_geom,
                            x_phys);
                    const auto xi_x =
                        XGeometry::physical_to_reference(x_geom, x_phys);

                    double lambda_value =
                        std::numeric_limits<double>::quiet_NaN();
                    double lambda_tilde_value =
                        std::numeric_limits<double>::quiet_NaN();
                    double u_value =
                        std::numeric_limits<double>::quiet_NaN();
                    auto lambda_gradient =
                        detail::nan_array<GT::dim_v>();
                    auto lambda_tilde_gradient =
                        detail::nan_array<GT::dim_v>();
                    auto u_gradient =
                        detail::nan_array<GT::dim_v>();

                    auto append_record =
                        [&](std::ostream& out, std::string_view reason)
                    {
                        out
                            << "2+1D estimator input trace: "
                            << reason
                            << '\n';
                        out << "slab=" << slab_id
                            << " slab_cell=" << slab_cell_id
                            << " source_cell=" << source_cell_id
                            << " x_ancestor_cell=" << x_cell_id
                            << " q=" << q << '\n';
                        out << "slab_reference=";
                        detail::append_array(out, xi_slab);
                        out << " source_reference=";
                        detail::append_array(out, xi_source);
                        out << " x_reference=";
                        detail::append_array(out, xi_x);
                        out << " physical=";
                        detail::append_array(out, x_phys);
                        out << '\n';
                        out << "lambda_delta_value=" << lambda_value
                            << " lambda_delta_gradient=";
                        detail::append_array(out, lambda_gradient);
                        out << '\n';
                        out << "lambda_tilde_value=" << lambda_tilde_value
                            << " lambda_tilde_gradient=";
                        detail::append_array(out, lambda_tilde_gradient);
                        out << '\n';
                        out << "u_delta_value=" << u_value
                            << " u_delta_gradient=";
                        detail::append_array(out, u_gradient);
                        out << '\n';
                        detail::append_cell_dofs(
                            out,
                            "Y_source",
                            lambda_delta,
                            source_cell_id);
                        detail::append_cell_dofs(
                            out,
                            "X",
                            u_delta,
                            x_cell_id);
                        detail::append_cell_dofs(
                            out,
                            "slab",
                            slab_function,
                            slab_cell_id);
                    };

                    if (!SourceGeometry::reference_point_inside(xi_source) ||
                        !XGeometry::reference_point_inside(xi_x))
                    {
                        std::ostringstream message;
                        append_record(
                            message,
                            "quadrature point is not inside source or X ancestor cell");
                        throw std::runtime_error(message.str());
                    }

                    try
                    {
                        lambda_value =
                            lambda_delta.value_on_cell(
                                source_cell_id,
                                x_phys,
                                source_geom);
                        lambda_gradient =
                            lambda_delta.gradient_on_cell(
                                source_cell_id,
                                x_phys,
                                source_geom);
                        lambda_tilde_value =
                            lambda_tilde.value_on_cell(
                                slab_id,
                                slab_cell_id,
                                x_phys,
                                slab_geom);
                        lambda_tilde_gradient =
                            lambda_tilde.gradient_on_cell(
                                slab_id,
                                slab_cell_id,
                                x_phys,
                                slab_geom);
                        u_value =
                            u_delta.value_on_cell(
                                x_cell_id,
                                x_phys,
                                x_geom);
                        u_gradient =
                            u_delta.gradient_on_cell(
                                x_cell_id,
                                x_phys,
                                x_geom);
                    }
                    catch (const std::exception& e)
                    {
                        std::ostringstream message;
                        append_record(message, "function evaluation threw");
                        message << "exception=" << e.what() << '\n';
                        throw std::runtime_error(message.str());
                    }

                    if (!std::isfinite(lambda_value) ||
                        !std::isfinite(lambda_tilde_value) ||
                        !std::isfinite(u_value) ||
                        !detail::all_finite(lambda_gradient) ||
                        !detail::all_finite(lambda_tilde_gradient) ||
                        !detail::all_finite(u_gradient))
                    {
                        std::ostringstream message;
                        append_record(
                            message,
                            "non-finite value or gradient at estimator quadrature point");
                        throw std::runtime_error(message.str());
                    }

                    if (trace_stream != nullptr &&
                        traces_written < trace_limit)
                    {
                        append_record(
                            *trace_stream,
                            "validated estimator quadrature point");
                        ++traces_written;
                    }
                }
            }
        }
    }

    template<
        int QSpace,
        int QTime,
        class SourceFunctionType,
        class ReconstructedFunctionType,
        class XFunctionType>
    void trace_time_slab_estimator_inputs_if_requested(
        const SourceFunctionType& lambda_delta,
        const ReconstructedFunctionType& lambda_tilde,
        const XFunctionType& u_delta)
    {
        using SourceSpace = typename SourceFunctionType::SpaceType;

        if constexpr (
            SourceSpace::GT::dim_space_v != 2 ||
            SourceSpace::GT::dim_time_v != 1)
        {
            static_cast<void>(lambda_delta);
            static_cast<void>(lambda_tilde);
            static_cast<void>(u_delta);
        }
        else
        {
            if (!detail::estimator_input_trace_enabled())
                return;

            validate_time_slab_estimator_inputs<QSpace, QTime>(
                lambda_delta,
                lambda_tilde,
                u_delta,
                &std::cerr,
                detail::estimator_input_trace_limit());
        }
    }
}

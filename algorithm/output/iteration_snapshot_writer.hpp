#pragma once

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "../adaptive_parameters.hpp"
#include "../adaptive_result.hpp"

#include "finite_element/io/io.hpp"
#include "finite_element/io/detail/binary_format_versions.hpp"

namespace adaptive_algorithm::output
{
    namespace detail
    {
        [[nodiscard]] inline std::string iteration_directory_name(
            const std::string& prefix,
            int iteration)
        {
            std::ostringstream name;
            name << prefix << '_' << std::setw(4) << std::setfill('0') << iteration;
            return name.str();
        }

        [[nodiscard]] inline std::filesystem::path snapshot_root_directory(
            const AdaptiveOutputSettings& settings)
        {
            return settings.output_directory / settings.snapshot_directory;
        }

        [[nodiscard]] inline const char* yaml_bool(bool value) noexcept
        {
            return value ? "true" : "false";
        }

        template<class SpaceType>
        void write_snapshot_root_metadata(
            const AdaptiveOutputSettings& settings)
        {
            const auto root_dir = snapshot_root_directory(settings);
            std::filesystem::create_directories(root_dir);

            std::ofstream out(root_dir / "metadata.yml");
            if (!out)
            {
                throw std::runtime_error(
                    "write_snapshot_root_metadata: failed to open metadata file.");
            }

            out << "layout: adaptive_iteration_snapshots\n";
            out << "layout_version: 1\n";
            out << "spatial_dimension: " << SpaceType::GT::dim_space_v << "\n";
            out << "dim_space: " << SpaceType::GT::dim_space_v << "\n";
            out << "dim_time: " << SpaceType::GT::dim_time_v << "\n";
            out << "mesh_filename: mesh.bin\n";
            out << "dofs_filename: dofs.bin\n";
            out << "dofs_written: "
                << yaml_bool(settings.save_snapshot_dofs) << "\n";
            out << "time_slab_estimator_diagnostics: "
                << yaml_bool(
                       SpaceType::GT::dim_space_v == 2 &&
                       SpaceType::GT::dim_time_v == 1)
                << "\n";
        }

        template<class SpaceType>
        void write_space_snapshot_metadata(
            const std::filesystem::path& output_dir,
            const std::string& space_role,
            bool write_dofs)
        {
            std::ofstream out(output_dir / "metadata.yml");
            if (!out)
            {
                throw std::runtime_error(
                    "write_space_snapshot_metadata: failed to open metadata file.");
            }

            out << "space_role: " << space_role << "\n";
            out << "spatial_dimension: " << SpaceType::GT::dim_space_v << "\n";
            out << "dim_space: " << SpaceType::GT::dim_space_v << "\n";
            out << "dim_time: " << SpaceType::GT::dim_time_v << "\n";
            out << "mesh_filename: mesh.bin\n";
            out << "dofs_filename: dofs.bin\n";
            out << "dofs_written: " << yaml_bool(write_dofs) << "\n";
        }

        inline void write_cell_id_list_csv(
            const std::vector<int>& cell_ids,
            const std::filesystem::path& filename)
        {
            std::filesystem::create_directories(filename.parent_path());

            std::ofstream out(filename);
            if (!out)
            {
                throw std::runtime_error(
                    "write_cell_id_list_csv: failed to open output file.");
            }

            out << "cell_id\n";
            for (const int cell_id : cell_ids)
                out << cell_id << '\n';
        }

        template<class SpaceType>
        void write_space_snapshot(
            const SpaceType& space,
            const std::filesystem::path& output_dir,
            const std::string& space_role,
            bool write_dofs)
        {
            std::filesystem::create_directories(output_dir);

            const auto mesh_file = output_dir / "mesh.bin";
            const auto dofs_file = output_dir / "dofs.bin";
            if (std::filesystem::exists(mesh_file) &&
                (!write_dofs || std::filesystem::exists(dofs_file)))
            {
                write_space_snapshot_metadata<SpaceType>(
                    output_dir,
                    space_role,
                    write_dofs);
                return;
            }

            space.write_mesh_binary(output_dir, "mesh.bin");

            if (write_dofs)
                space.write_dofs_binary(output_dir, "dofs.bin");

            write_space_snapshot_metadata<SpaceType>(
                output_dir,
                space_role,
                write_dofs);
        }

        template<class YIterationStateType>
        void write_time_slab_estimator_diagnostics_snapshot(
            const YIterationStateType& state,
            const std::filesystem::path& output_dir)
        {
            using FluxReconstructionType =
                typename YIterationStateType::FluxReconstructionType;
            using GT = typename FluxReconstructionType::GT;

            if constexpr (GT::dim_space_v == 2 && GT::dim_time_v == 1)
            {
                if (!state.flux_reconstruction.has_value())
                    return;

                std::filesystem::create_directories(output_dir);

                finite_element::io::write_time_slab_vertex_patches_binary(
                    state.flux_reconstruction->patch_set(),
                    output_dir,
                    "time_slab_vertex_patches.bin");
                finite_element::io::
                    write_time_slab_flux_reconstruction_samples_binary(
                        *state.flux_reconstruction,
                        output_dir,
                        "flux_reconstruction_samples.bin",
                        4,
                        4);
            }
        }

        template<typename CellIdType = int>
        void write_x_indicator_components_binary(
            const XMarkingIndicatorComponents<CellIdType>& indicators,
            const std::filesystem::path& output_dir,
            const std::string& filename = "x_indicator_components.bin")
        {
            using int_t = finite_element::io::detail::binary_int_t;

            std::filesystem::create_directories(output_dir);

            auto file =
                finite_element::io::detail::open_binary_output(
                    output_dir,
                    filename,
                    "write_x_indicator_components_binary");

            auto entries = indicators.eta_squared_by_x_cell.sorted_descending();
            const int_t n_entries = static_cast<int_t>(entries.size());
            const int_t n_components = 3;

            finite_element::io::detail::write_binary_block(
                file,
                finite_element::io::detail::make_versioned_header(
                    finite_element::io::detail::x_indicator_components_binary_format_version,
                    n_entries,
                    n_components),
                "write_x_indicator_components_binary");

            std::vector<int_t> cell_ids;
            std::vector<double> lambda_y;
            std::vector<double> initial_trace;
            std::vector<double> eta;

            cell_ids.reserve(entries.size());
            lambda_y.reserve(entries.size());
            initial_trace.reserve(entries.size());
            eta.reserve(entries.size());

            for (const auto& [cell_id, eta_value] : entries)
            {
                const auto find_or_zero =
                    [cell_id](const auto& map) -> double
                    {
                        const auto it = map.find(cell_id);
                        return it == map.end() ? 0.0 : it->second;
                    };

                cell_ids.push_back(static_cast<int_t>(cell_id));
                lambda_y.push_back(
                    find_or_zero(indicators.lambda_y_squared_by_x_cell.by_source_cell));
                initial_trace.push_back(
                    find_or_zero(indicators.initial_trace_squared_by_x_cell.by_source_cell));
                eta.push_back(eta_value);
            }

            finite_element::io::detail::write_binary_block(
                file, cell_ids, "write_x_indicator_components_binary");
            finite_element::io::detail::write_binary_block(
                file, lambda_y, "write_x_indicator_components_binary");
            finite_element::io::detail::write_binary_block(
                file, initial_trace, "write_x_indicator_components_binary");
            finite_element::io::detail::write_binary_block(
                file, eta, "write_x_indicator_components_binary");
        }
    }

    template<class XSpaceType>
    void write_outer_iteration_snapshot(
        int outer_iteration,
        const XSpaceType& x_space,
        const std::vector<int>& marked_x_cells,
        const AdaptiveOutputSettings& settings)
    {
        if (!settings.save_iteration_snapshots || settings.output_directory.empty())
            return;

        const auto outer_dir =
            detail::snapshot_root_directory(settings) /
            detail::iteration_directory_name("outer", outer_iteration);

        detail::write_snapshot_root_metadata<XSpaceType>(settings);
        detail::write_space_snapshot(
            x_space,
            outer_dir / "x_space",
            "x_space",
            settings.save_snapshot_dofs);
        detail::write_cell_id_list_csv(
            marked_x_cells,
            outer_dir / "marked_x_cells.csv");
    }

    template<class XSpaceType, class YSpaceType, class YIterationStateType>
    void write_inner_iteration_snapshot(
        int outer_iteration,
        int inner_iteration,
        const XSpaceType& x_space,
        const YSpaceType& y_space,
        const XMarkingIndicatorComponents<int>& x_indicators,
        const YIterationStateType& state,
        const AdaptiveOutputSettings& settings)
    {
        if (!settings.save_iteration_snapshots || settings.output_directory.empty())
            return;

        const auto outer_dir =
            detail::snapshot_root_directory(settings) /
            detail::iteration_directory_name("outer", outer_iteration);
        const auto inner_dir =
            outer_dir / detail::iteration_directory_name("inner", inner_iteration);

        detail::write_snapshot_root_metadata<XSpaceType>(settings);
        detail::write_space_snapshot(
            x_space,
            outer_dir / "x_space",
            "x_space",
            settings.save_snapshot_dofs);
        detail::write_space_snapshot(
            y_space,
            inner_dir / "y_space",
            "y_space",
            settings.save_snapshot_dofs);

        finite_element::io::write_cellwise_error_components_binary(
            state.estimator,
            inner_dir / "y_indicators",
            "estimator_components.bin");
        finite_element::io::write_cellwise_error_binary(
            x_indicators.lambda_y_squared_by_y_cell,
            inner_dir / "y_indicators",
            "lambda_y_by_y_cell.bin");

        detail::write_x_indicator_components_binary(
            x_indicators,
            inner_dir / "x_indicators");
        detail::write_time_slab_estimator_diagnostics_snapshot(
            state,
            inner_dir / "time_slab_estimator");

        detail::write_cell_id_list_csv(
            state.marked_source_cells,
            inner_dir / "marked_y_cells.csv");
    }
}

#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "detail/binary_format_versions.hpp"
#include "finite_element/fespace/io/detail/write_binary_block.hpp"

namespace finite_element::io
{
    template<class CellwiseEstimatorType>
    void write_cellwise_error_components_binary(
        const CellwiseEstimatorType& estimator,
        const std::filesystem::path& output_dir,
        const std::string& filename = "cellwise_error_components.bin",
        bool sort_descending = true)
    {
        using int_t = finite_element::io::detail::binary_int_t;

        std::filesystem::create_directories(output_dir);

        auto file =
            finite_element::io::detail::open_binary_output(
                output_dir,
                filename,
                "write_cellwise_error_components_binary");

        std::vector<std::pair<int, double>> entries;
        if (sort_descending)
            entries = estimator.estimator_squared.sorted_descending();
        else
            entries.assign(estimator.estimator_squared.by_source_cell.begin(),
                           estimator.estimator_squared.by_source_cell.end());

        const int_t n_entries = static_cast<int_t>(entries.size());
        const int_t n_components = 4;

        finite_element::io::detail::write_binary_block(
            file,
            finite_element::io::detail::make_versioned_header(
                finite_element::io::detail::cellwise_error_components_binary_format_version,
                n_entries,
                n_components),
            "write_cellwise_error_components_binary");

        std::vector<int_t> cell_ids;
        std::vector<double> equilibrated_flux_y;
        std::vector<double> reconstruction_y;
        std::vector<double> divergence_residual;
        std::vector<double> total;

        cell_ids.reserve(entries.size());
        equilibrated_flux_y.reserve(entries.size());
        reconstruction_y.reserve(entries.size());
        divergence_residual.reserve(entries.size());
        total.reserve(entries.size());

        for (const auto& [cell_id, total_value] : entries)
        {
            const auto find_or_zero =
                [cell_id](const auto& map) -> double
                {
                    const auto it = map.find(cell_id);
                    return it == map.end() ? 0.0 : it->second;
                };

            cell_ids.push_back(static_cast<int_t>(cell_id));
            equilibrated_flux_y.push_back(
                find_or_zero(estimator.equilibrated_flux_y_squared.by_source_cell));
            reconstruction_y.push_back(
                find_or_zero(estimator.reconstruction_y_squared.by_source_cell));
            divergence_residual.push_back(
                find_or_zero(estimator.divergence_residual_squared.by_source_cell));
            total.push_back(total_value);
        }

        finite_element::io::detail::write_binary_block(
            file, cell_ids, "write_cellwise_error_components_binary");
        finite_element::io::detail::write_binary_block(
            file, equilibrated_flux_y, "write_cellwise_error_components_binary");
        finite_element::io::detail::write_binary_block(
            file, reconstruction_y, "write_cellwise_error_components_binary");
        finite_element::io::detail::write_binary_block(
            file, divergence_residual, "write_cellwise_error_components_binary");
        finite_element::io::detail::write_binary_block(
            file, total, "write_cellwise_error_components_binary");
    }
}

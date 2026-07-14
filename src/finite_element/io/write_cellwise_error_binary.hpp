#pragma once

#include <filesystem>
#include <vector>
#include <utility>

#include "detail/binary_format_versions.hpp"
#include "finite_element/fespace/io/detail/write_binary_block.hpp"

namespace finite_element::io
{
    // ---------------------------------------------------------------------
    // Write CellwiseSquaredError to binary
    //
    // Layout:
    //   header (int32):
    //       [0] = n_entries
    //
    //   cell_ids (int32[n_entries])
    //   values   (float64[n_entries])
    //
    // Optional: sorted output
    // ---------------------------------------------------------------------
    template<class CellwiseErrorType>
    void write_cellwise_error_binary(
        const CellwiseErrorType& error,
        const std::filesystem::path& output_dir,
        const std::string& filename = "cellwise_error.bin",
        bool sort_descending = true)
    {
        using int_t = finite_element::io::detail::binary_int_t;

        std::filesystem::create_directories(output_dir);

        auto file =
            finite_element::io::detail::open_binary_output(
                output_dir,
                filename,
                "write_cellwise_error_binary");

        // Collect entries
        std::vector<std::pair<int, double>> entries;

        if (sort_descending)
            entries = error.sorted_descending();
        else
            entries.assign(error.by_source_cell.begin(),
                           error.by_source_cell.end());

        const int_t n_entries =
            static_cast<int_t>(entries.size());

        // --- header ---
        finite_element::io::detail::write_binary_block(
            file,
            finite_element::io::detail::make_versioned_header(
                finite_element::io::detail::cellwise_error_binary_format_version,
                n_entries),
            "write_cellwise_error_binary");

        // --- data ---
        std::vector<int_t> cell_ids;
        std::vector<double> values;

        cell_ids.reserve(entries.size());
        values.reserve(entries.size());

        for (const auto& [cell_id, value] : entries)
        {
            cell_ids.push_back(static_cast<int_t>(cell_id));
            values.push_back(value);
        }

        finite_element::io::detail::write_binary_block(
            file, cell_ids, "write_cellwise_error_binary");

        finite_element::io::detail::write_binary_block(
            file, values, "write_cellwise_error_binary");
    }
}

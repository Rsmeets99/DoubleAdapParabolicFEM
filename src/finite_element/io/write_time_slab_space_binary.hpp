#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "detail/time_slab_io_detail.hpp"
#include "finite_element/io/detail/binary_format_versions.hpp"
#include "write_fespace_bundle_binary.hpp"
#include "finite_element/fespace/io/detail/write_binary_block.hpp"

namespace finite_element::io
{
    template<class TimeSlabSpaceType>
    void write_time_slab_space_binary(
        const TimeSlabSpaceType& slab_space,
        const std::filesystem::path& output_dir,
        const std::string& metadata_filename = "time_slab_space.bin",
        const std::string& mesh_filename = "mesh.bin",
        const std::string& dofs_filename = "dofs.bin",
        const std::string& provenance_filename = "provenance.bin")
    {
        using int_t = finite_element::io::detail::binary_int_t;

        std::filesystem::create_directories(output_dir);

        {
            auto file =
                finite_element::io::detail::open_binary_output(
                    output_dir,
                    metadata_filename,
                    "write_time_slab_space_binary");

            const int_t n_slabs =
                static_cast<int_t>(slab_space.n_slabs());
            const int_t n_slab_times =
                static_cast<int_t>(slab_space.slab_times().size());

            const auto header = finite_element::io::detail::make_versioned_header(
                finite_element::io::detail::time_slab_space_metadata_binary_format_version,
                n_slabs,
                n_slab_times);

            finite_element::io::detail::write_binary_block(
                file, header, "write_time_slab_space_binary");

            std::vector<double> slab_times(
                slab_space.slab_times().begin(),
                slab_space.slab_times().end());

            finite_element::io::detail::write_binary_block(
                file, slab_times, "write_time_slab_space_binary");
        }

        for (int k = 0; k < slab_space.n_slabs(); ++k)
        {
            const auto& slab = slab_space.slab(k);
            const auto slab_dir =
                finite_element::io::detail::time_slab_directory(output_dir, k);

            std::filesystem::create_directories(slab_dir);

            write_fespace_bundle_binary(
                slab.fespace_ref(),
                slab_dir,
                mesh_filename,
                dofs_filename);

            auto file =
                finite_element::io::detail::open_binary_output(
                    slab_dir,
                    provenance_filename,
                    "write_time_slab_space_binary");

            const int_t slab_id   = static_cast<int_t>(slab.slab_id());
            const int_t n_cells   = static_cast<int_t>(slab.n_active_cells());
            const double t_begin  = slab.t_begin();
            const double t_end    = slab.t_end();

            const auto header = finite_element::io::detail::make_versioned_header(
                finite_element::io::detail::time_slab_provenance_binary_format_version,
                slab_id,
                n_cells);

            finite_element::io::detail::write_binary_block(
                file, header, "write_time_slab_space_binary");

            finite_element::io::detail::write_binary_block(
                file,
                std::vector<double>{t_begin, t_end},
                "write_time_slab_space_binary");

            std::vector<int_t> slab_local_cell_ids;
            std::vector<int_t> source_cell_ids;
            std::vector<double> slice_intervals;

            slab_local_cell_ids.reserve(static_cast<std::size_t>(n_cells));
            source_cell_ids.reserve(static_cast<std::size_t>(n_cells));
            slice_intervals.reserve(2 * static_cast<std::size_t>(n_cells));

            for (const auto& info : slab.sliced_cell_infos())
            {
                slab_local_cell_ids.push_back(
                    static_cast<int_t>(info.slab_local_cell_id));
                source_cell_ids.push_back(
                    static_cast<int_t>(info.source_cell_id));
                slice_intervals.push_back(info.t_begin);
                slice_intervals.push_back(info.t_end);
            }

            finite_element::io::detail::write_binary_block(
                file, slab_local_cell_ids, "write_time_slab_space_binary");
            finite_element::io::detail::write_binary_block(
                file, source_cell_ids, "write_time_slab_space_binary");
            finite_element::io::detail::write_binary_block(
                file, slice_intervals, "write_time_slab_space_binary");
        }
    }
}

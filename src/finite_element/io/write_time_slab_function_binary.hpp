#pragma once

#include <filesystem>
#include <string>

#include "detail/time_slab_io_detail.hpp"
#include "write_time_slab_space_binary.hpp"
#include "finite_element/fespace/io/write_function_binary.hpp"

namespace finite_element::io
{
    template<class TimeSlabFunctionType>
    void write_time_slab_function_binary(
        const TimeSlabFunctionType& function,
        const std::filesystem::path& output_dir,
        const std::string& metadata_filename = "time_slab_space.bin",
        const std::string& mesh_filename = "mesh.bin",
        const std::string& dofs_filename = "dofs.bin",
        const std::string& provenance_filename = "provenance.bin",
        const std::string& function_filename = "function.bin",
        const std::string& coefficients_filename = "true_coefficients.bin",
        int samples_x_per_cell = 32,
        int samples_t_per_cell = 32)
    {
        const auto& slab_space = function.slab_space();

        write_time_slab_space_binary(
            slab_space,
            output_dir,
            metadata_filename,
            mesh_filename,
            dofs_filename,
            provenance_filename);

        for (int k = 0; k < slab_space.n_slabs(); ++k)
        {
            const auto slab_dir =
                finite_element::io::detail::time_slab_directory(output_dir, k);

            const auto& slab_function = function.slab_function(k);

            finite_element::io::write_function_binary(
                slab_function,
                slab_dir,
                function_filename,
                samples_x_per_cell,
                samples_t_per_cell);

            finite_element::io::write_function_true_coefficients_binary(
                slab_function,
                slab_dir,
                coefficients_filename);
        }
    }
}
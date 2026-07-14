#pragma once

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "detail/write_binary_block.hpp"
#include "finite_element/io/detail/binary_format_versions.hpp"

namespace finite_element::io
{
    namespace detail
    {
        inline std::vector<double> make_interval_reference_samples(int n)
        {
            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(n));

            for (int i = 0; i < n; ++i)
                samples.push_back(static_cast<double>(i) / static_cast<double>(n - 1));

            return samples;
        }

        inline std::vector<double> make_triangle_barycentric_lattice_samples(int n)
        {
            std::vector<double> samples;
            samples.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n + 1));

            const double denominator = static_cast<double>(n - 1);
            for (int j = 0; j < n; ++j)
            {
                for (int i = 0; i < n - j; ++i)
                {
                    samples.push_back(static_cast<double>(i) / denominator);
                    samples.push_back(static_cast<double>(j) / denominator);
                }
            }

            return samples;
        }
    }

    template<class FunctionType>
    void write_function_binary(
        const FunctionType& function,
        const std::filesystem::path& output_dir,
        const std::string& filename,
        int samples_x_per_cell = 32,
        int samples_t_per_cell = 32)
    {
        using SpaceType = typename FunctionType::SpaceType;
        using GT        = typename SpaceType::GT;
        using int_t     = detail::binary_int_t;

        if (samples_x_per_cell < 2 || samples_t_per_cell < 2)
            throw std::runtime_error("write_function_binary: need at least 2 samples per direction.");

        const auto& space        = function.fespace();
        const auto& active_cells = space.active_cells();
        const int_t n_cells      = static_cast<int_t>(active_cells.size());

        if constexpr (GT::dim_space_v == 1 && GT::dim_time_v == 1)
        {
            const int_t nx           = static_cast<int_t>(samples_x_per_cell);
            const int_t nt           = static_cast<int_t>(samples_t_per_cell);
            const int_t values_per_cell = nx * nt;

            auto file = detail::open_binary_output(output_dir, filename, "write_function_binary");

            const auto header = detail::make_versioned_header(
                detail::function_binary_format_version,
                n_cells,
                static_cast<int_t>(GT::dim_space_v),
                static_cast<int_t>(GT::dim_time_v),
                nx,
                nt,
                values_per_cell);

            detail::write_binary_block(file, header, "write_function_binary");

            std::vector<int_t> cell_ids;
            cell_ids.reserve(static_cast<std::size_t>(n_cells));

            std::vector<double> values;
            values.reserve(static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(values_per_cell));

            for (const int cell_id : active_cells)
            {
                cell_ids.push_back(static_cast<int_t>(cell_id));

                for (int jt = 0; jt < nt; ++jt)
                {
                    const double eta = static_cast<double>(jt) / static_cast<double>(nt - 1);

                    for (int ix = 0; ix < nx; ++ix)
                    {
                        const double xi = static_cast<double>(ix) / static_cast<double>(nx - 1);
                        values.push_back(function.value_on_reference_cell(cell_id, {xi, eta}));
                    }
                }
            }

            detail::write_binary_block(file, cell_ids, "write_function_binary");
            detail::write_binary_block(file, values, "write_function_binary");
        }
        else if constexpr (GT::dim_space_v == 2 && GT::dim_time_v == 1)
        {
            const int_t ns_edge = static_cast<int_t>(samples_x_per_cell);
            const int_t nt = static_cast<int_t>(samples_t_per_cell);
            const int_t n_spatial_samples =
                static_cast<int_t>(samples_x_per_cell * (samples_x_per_cell + 1) / 2);
            const int_t values_per_cell = n_spatial_samples * nt;

            auto file = detail::open_binary_output(output_dir, filename, "write_function_binary");

            const auto reference_spatial_samples =
                detail::make_triangle_barycentric_lattice_samples(samples_x_per_cell);
            const auto reference_time_samples =
                detail::make_interval_reference_samples(samples_t_per_cell);

            const auto header = detail::make_versioned_header(
                detail::function_binary_format_version_2d,
                n_cells,
                static_cast<int_t>(GT::dim_space_v),
                static_cast<int_t>(GT::dim_time_v),
                ns_edge,
                nt,
                n_spatial_samples,
                values_per_cell,
                detail::function_sample_layout_triangle_barycentric_time,
                static_cast<int_t>(GT::dim_space_v),
                static_cast<int_t>(GT::dim_time_v));

            detail::write_binary_block(file, header, "write_function_binary");

            std::vector<int_t> cell_ids;
            cell_ids.reserve(static_cast<std::size_t>(n_cells));

            std::vector<double> values;
            values.reserve(static_cast<std::size_t>(n_cells) * static_cast<std::size_t>(values_per_cell));

            for (const int cell_id : active_cells)
            {
                cell_ids.push_back(static_cast<int_t>(cell_id));

                for (int jt = 0; jt < nt; ++jt)
                {
                    const double tau = reference_time_samples[static_cast<std::size_t>(jt)];

                    for (int is = 0; is < n_spatial_samples; ++is)
                    {
                        const std::size_t spatial_offset =
                            static_cast<std::size_t>(is) * static_cast<std::size_t>(GT::dim_space_v);
                        values.push_back(
                            function.value_on_reference_cell(
                                cell_id,
                                {
                                    reference_spatial_samples[spatial_offset],
                                    reference_spatial_samples[spatial_offset + 1u],
                                    tau
                                }));
                    }
                }
            }

            detail::write_binary_block(file, cell_ids, "write_function_binary");
            detail::write_binary_block(file, reference_spatial_samples, "write_function_binary");
            detail::write_binary_block(file, reference_time_samples, "write_function_binary");
            detail::write_binary_block(file, values, "write_function_binary");
        }
        else
        {
            throw std::runtime_error("write_function_binary: unsupported space-time dimension.");
        }
    }

    template<class FunctionType>
    void write_function_true_coefficients_binary(
        const FunctionType& function,
        const std::filesystem::path& output_dir,
        const std::string& filename)
    {
        using int_t = detail::binary_int_t;

        const auto& true_coefficients = function.true_coefficients();
        const int_t n_true_dofs = static_cast<int_t>(true_coefficients.size());

        auto file = detail::open_binary_output(output_dir, filename, "write_function_true_coefficients_binary");

        const auto header = detail::make_versioned_header(
            detail::function_true_coefficients_binary_format_version,
            n_true_dofs);
        detail::write_binary_block(file, header, "write_function_true_coefficients_binary");

        std::vector<double> values(static_cast<std::size_t>(n_true_dofs), 0.0);
        for (int i = 0; i < n_true_dofs; ++i)
            values[static_cast<std::size_t>(i)] = true_coefficients[i];

        detail::write_binary_block(file, values, "write_function_true_coefficients_binary");
    }
}

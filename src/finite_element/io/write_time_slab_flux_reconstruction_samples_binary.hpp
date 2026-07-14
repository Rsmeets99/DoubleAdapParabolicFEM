#pragma once

#include <cstddef>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "../geometry/cell_geometry.hpp"
#include "detail/binary_format_versions.hpp"
#include "finite_element/fespace/io/detail/write_binary_block.hpp"
#include "finite_element/fespace/io/write_function_binary.hpp"

namespace finite_element::io
{
    template<class FluxReconstructionType>
    void write_time_slab_flux_reconstruction_samples_binary(
        const FluxReconstructionType& reconstruction,
        const std::filesystem::path& output_dir,
        const std::string& filename = "flux_reconstruction_samples.bin",
        int samples_x_per_cell = 4,
        int samples_t_per_cell = 4)
    {
        using SlabSpaceType = typename FluxReconstructionType::SlabSpaceType;
        using GT            = typename SlabSpaceType::GT;
        using int_t         = finite_element::io::detail::binary_int_t;

        static_assert(GT::dim_space_v == 2 && GT::dim_time_v == 1,
                      "write_time_slab_flux_reconstruction_samples_binary requires a 2+1D reconstruction.");

        if (samples_x_per_cell < 2 || samples_t_per_cell < 2)
        {
            throw std::runtime_error(
                "write_time_slab_flux_reconstruction_samples_binary: need at least 2 samples per direction.");
        }

        const auto& slab_space = reconstruction.slab_space_ref();
        int n_slab_cells_raw = 0;
        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            n_slab_cells_raw += slab_space.slab(slab_id).n_active_cells();

        const int_t n_slabs = static_cast<int_t>(slab_space.n_slabs());
        const int_t n_slab_cells = static_cast<int_t>(n_slab_cells_raw);
        const int_t ns_edge = static_cast<int_t>(samples_x_per_cell);
        const int_t nt = static_cast<int_t>(samples_t_per_cell);
        const int_t n_spatial_samples =
            static_cast<int_t>(
                samples_x_per_cell * (samples_x_per_cell + 1) / 2);
        const int_t samples_per_cell = n_spatial_samples * nt;
        constexpr int_t value_components = 3;

        auto file =
            finite_element::io::detail::open_binary_output(
                output_dir,
                filename,
                "write_time_slab_flux_reconstruction_samples_binary");

        const auto header = finite_element::io::detail::make_versioned_header(
            finite_element::io::detail::
                time_slab_flux_reconstruction_samples_binary_format_version,
            n_slabs,
            n_slab_cells,
            static_cast<int_t>(GT::dim_space_v),
            static_cast<int_t>(GT::dim_time_v),
            ns_edge,
            nt,
            n_spatial_samples,
            samples_per_cell,
            value_components,
            finite_element::io::detail::
                function_sample_layout_triangle_barycentric_time);

        finite_element::io::detail::write_binary_block(
            file,
            header,
            "write_time_slab_flux_reconstruction_samples_binary");

        const auto reference_spatial_samples =
            finite_element::io::detail::make_triangle_barycentric_lattice_samples(
                samples_x_per_cell);
        const auto reference_time_samples =
            finite_element::io::detail::make_interval_reference_samples(
                samples_t_per_cell);

        std::vector<int_t> slab_ids;
        std::vector<int_t> slab_cell_ids;
        std::vector<int_t> source_cell_ids;
        std::vector<double> values;

        slab_ids.reserve(static_cast<std::size_t>(n_slab_cells));
        slab_cell_ids.reserve(static_cast<std::size_t>(n_slab_cells));
        source_cell_ids.reserve(static_cast<std::size_t>(n_slab_cells));
        values.reserve(
            static_cast<std::size_t>(n_slab_cells) *
            static_cast<std::size_t>(samples_per_cell) *
            static_cast<std::size_t>(value_components));

        for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
        {
            const auto& slab = slab_space.slab(slab_id);
            const auto& slab_fespace = slab.fespace_ref();
            using LocalSpaceType = std::remove_cvref_t<decltype(slab_fespace)>;
            using Geometry =
                finite_element::geometry::CellGeometry<
                    LocalSpaceType,
                    GT::dim_space_v>;

            for (const int slab_cell_id : slab.active_cells())
            {
                slab_ids.push_back(static_cast<int_t>(slab_id));
                slab_cell_ids.push_back(static_cast<int_t>(slab_cell_id));
                source_cell_ids.push_back(
                    static_cast<int_t>(
                        slab_space.source_cell_id(slab_id, slab_cell_id)));

                const auto geom = Geometry::make(slab_fespace, slab_cell_id);

                for (int jt = 0; jt < nt; ++jt)
                {
                    const double tau =
                        reference_time_samples[static_cast<std::size_t>(jt)];

                    for (int is = 0; is < n_spatial_samples; ++is)
                    {
                        const std::size_t spatial_offset =
                            static_cast<std::size_t>(is) *
                            static_cast<std::size_t>(GT::dim_space_v);
                        const auto p =
                            Geometry::map_to_physical(
                                geom,
                                {
                                    reference_spatial_samples[spatial_offset],
                                    reference_spatial_samples[spatial_offset + 1u],
                                    tau
                                });
                        const auto evaluation =
                            reconstruction.sigma_and_div_sigma_on_slab_cell(
                                slab_id,
                                slab_cell_id,
                                p);

                        values.push_back(evaluation.sigma[0]);
                        values.push_back(evaluation.sigma[1]);
                        values.push_back(evaluation.div_sigma);
                    }
                }
            }
        }

        finite_element::io::detail::write_binary_block(
            file,
            slab_ids,
            "write_time_slab_flux_reconstruction_samples_binary");
        finite_element::io::detail::write_binary_block(
            file,
            slab_cell_ids,
            "write_time_slab_flux_reconstruction_samples_binary");
        finite_element::io::detail::write_binary_block(
            file,
            source_cell_ids,
            "write_time_slab_flux_reconstruction_samples_binary");
        finite_element::io::detail::write_binary_block(
            file,
            reference_spatial_samples,
            "write_time_slab_flux_reconstruction_samples_binary");
        finite_element::io::detail::write_binary_block(
            file,
            reference_time_samples,
            "write_time_slab_flux_reconstruction_samples_binary");
        finite_element::io::detail::write_binary_block(
            file,
            values,
            "write_time_slab_flux_reconstruction_samples_binary");
    }
}

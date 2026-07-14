#pragma once

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "detail/time_slab_patch_io_detail.hpp"

namespace finite_element::io
{
    template<class PatchSetType>
    void write_time_slab_vertex_patches_binary(
        const PatchSetType& patch_set,
        const std::filesystem::path& output_dir,
        const std::string& filename = "time_slab_vertex_patches.bin")
    {
        using int_t = finite_element::io::detail::binary_int_t;
        using PatchType = typename PatchSetType::PatchType;
        constexpr const char* context = "write_time_slab_vertex_patches_binary";

        static_assert(PatchType::GT::dim_space_v == 2,
                      "write_time_slab_vertex_patches_binary requires 2D space.");
        static_assert(PatchType::GT::dim_time_v == 1,
                      "write_time_slab_vertex_patches_binary requires 1D time.");

        int max_cells_per_patch = 0;
        for (const auto& patch : patch_set.patches())
            max_cells_per_patch = std::max(max_cells_per_patch, patch.n_cells);

        const auto export_info =
            finite_element::io::detail::make_time_slab_patch_export_info(
                patch_set,
                static_cast<int_t>(max_cells_per_patch));

        auto file =
            finite_element::io::detail::open_binary_output(
                output_dir,
                filename,
                context);

        finite_element::io::detail::write_time_slab_patch_header(
            file,
            finite_element::io::detail::time_slab_vertex_patches_binary_format_version,
            export_info,
            context);

        const auto slab_index =
            finite_element::io::detail::make_time_slab_patch_slab_index(
                patch_set,
                export_info);
        finite_element::io::detail::write_time_slab_patch_slab_index(
            file,
            slab_index,
            context);

        const auto write_block =
            [&file, context](const auto& block)
            {
                finite_element::io::detail::write_time_slab_patch_block(
                    file,
                    block,
                    context);
            };

        std::vector<int_t> patch_ids;
        std::vector<int_t> slab_ids;
        std::vector<int_t> spatial_vertex_ids;
        std::vector<int_t> kinds;
        std::vector<int_t> n_cells;
        std::vector<double> spatial_vertices;
        std::vector<double> t_begins;
        std::vector<double> t_ends;

        patch_ids.reserve(static_cast<std::size_t>(export_info.n_patches));
        slab_ids.reserve(static_cast<std::size_t>(export_info.n_patches));
        spatial_vertex_ids.reserve(static_cast<std::size_t>(export_info.n_patches));
        kinds.reserve(static_cast<std::size_t>(export_info.n_patches));
        n_cells.reserve(static_cast<std::size_t>(export_info.n_patches));
        spatial_vertices.reserve(
            static_cast<std::size_t>(PatchType::GT::dim_space_v) *
            static_cast<std::size_t>(export_info.n_patches));
        t_begins.reserve(static_cast<std::size_t>(export_info.n_patches));
        t_ends.reserve(static_cast<std::size_t>(export_info.n_patches));

        const auto flat_size = export_info.flat_cell_size();
        constexpr int n_spatial_vertices_per_cell =
            PatchType::Types::n_spatial_vertices;
        const auto flat_vertex_id_size =
            export_info.flat_cell_component_size(n_spatial_vertices_per_cell);
        const auto flat_spatial_coordinate_size =
            export_info.flat_cell_component_size(
                n_spatial_vertices_per_cell * PatchType::GT::dim_space_v);
        const auto flat_time_interval_size =
            export_info.flat_cell_component_size(PatchType::GT::dim_time_v + 1);

        std::vector<int_t> cell_slab_cell_ids(flat_size, -1);
        std::vector<int_t> cell_source_cell_ids(flat_size, -1);
        std::vector<int_t> cell_local_vertex_indices(flat_size, -1);
        std::vector<int_t> cell_source_local_vertex_indices(flat_size, -1);
        std::vector<int_t> cell_source_spatial_vertex_ids(flat_size, -1);
        std::vector<int_t> cell_slab_spatial_vertex_ids(flat_vertex_id_size, -1);
        std::vector<int_t> cell_source_spatial_vertex_ids_per_cell(flat_vertex_id_size, -1);
        std::vector<double> cell_triangle_coordinates(flat_spatial_coordinate_size, 0.0);
        std::vector<double> cell_time_intervals(flat_time_interval_size, 0.0);

        const auto encode_kind =
            [](const auto& patch) -> int_t
            {
                return patch.is_boundary() ? 1 : 0;
            };

        for (int patch_id = 0; patch_id < patch_set.n_patches(); ++patch_id)
        {
            const auto& patch = patch_set.patch(patch_id);
            patch_ids.push_back(static_cast<int_t>(patch.patch_id));
            slab_ids.push_back(static_cast<int_t>(patch.slab_id));
            spatial_vertex_ids.push_back(static_cast<int_t>(patch.spatial_vertex_id));
            kinds.push_back(encode_kind(patch));
            n_cells.push_back(static_cast<int_t>(patch.n_cells));
            spatial_vertices.push_back(patch.spatial_vertex[0]);
            spatial_vertices.push_back(patch.spatial_vertex[1]);
            t_begins.push_back(patch.t_begin);
            t_ends.push_back(patch.t_end);

            const auto patch_offset =
                static_cast<std::size_t>(patch_id) *
                static_cast<std::size_t>(export_info.max_cells_per_patch);

            for (int patch_cell_index = 0; patch_cell_index < patch.n_cells; ++patch_cell_index)
            {
                const auto& cell = patch.cell(patch_cell_index);
                const auto flat_index =
                    patch_offset + static_cast<std::size_t>(patch_cell_index);
                const auto& slab_mesh =
                    patch_set.slab_space().slab(patch.slab_id).mesh_ref();
                const auto& slab_cell = slab_mesh.cell(cell.slab_cell_id);

                cell_slab_cell_ids[flat_index] =
                    static_cast<int_t>(cell.slab_cell_id);
                cell_source_cell_ids[flat_index] =
                    static_cast<int_t>(cell.source_cell_id);
                cell_local_vertex_indices[flat_index] =
                    static_cast<int_t>(cell.local_vertex_index);
                cell_source_local_vertex_indices[flat_index] =
                    static_cast<int_t>(cell.source_local_vertex_index);
                cell_source_spatial_vertex_ids[flat_index] =
                    static_cast<int_t>(cell.source_spatial_vertex_id);

                const auto vertex_flat_offset =
                    flat_index * static_cast<std::size_t>(n_spatial_vertices_per_cell);
                for (int local_vertex = 0;
                     local_vertex < n_spatial_vertices_per_cell;
                     ++local_vertex)
                {
                    const auto local_index = static_cast<std::size_t>(local_vertex);
                    cell_slab_spatial_vertex_ids[vertex_flat_offset + local_index] =
                        static_cast<int_t>(
                            cell.slab_spatial_vertex_ids[local_index]);
                    cell_source_spatial_vertex_ids_per_cell[vertex_flat_offset + local_index] =
                        static_cast<int_t>(
                            cell.source_spatial_vertex_ids[local_index]);

                    const auto& point =
                        slab_mesh.spatial_vertices()[
                            static_cast<std::size_t>(
                                slab_cell.spatial_vertex_ids[local_index])];
                    const auto coordinate_offset =
                        (flat_index *
                         static_cast<std::size_t>(n_spatial_vertices_per_cell) +
                         local_index) *
                        static_cast<std::size_t>(PatchType::GT::dim_space_v);
                    for (int d = 0; d < PatchType::GT::dim_space_v; ++d)
                    {
                        cell_triangle_coordinates[
                            coordinate_offset + static_cast<std::size_t>(d)] =
                            point[static_cast<std::size_t>(d)];
                    }
                }

                const auto time_offset =
                    flat_index *
                    static_cast<std::size_t>(PatchType::GT::dim_time_v + 1);
                for (int endpoint = 0;
                     endpoint < PatchType::GT::dim_time_v + 1;
                     ++endpoint)
                {
                    const int temporal_vertex_id =
                        slab_cell.temporal_vertex_ids[
                            static_cast<std::size_t>(endpoint)];
                    cell_time_intervals[
                        time_offset + static_cast<std::size_t>(endpoint)] =
                        slab_mesh.temporal_vertices()[
                            static_cast<std::size_t>(temporal_vertex_id)][0];
                }
            }
        }

        write_block(patch_ids);
        write_block(slab_ids);
        write_block(spatial_vertex_ids);
        write_block(kinds);
        write_block(n_cells);
        write_block(spatial_vertices);
        write_block(t_begins);
        write_block(t_ends);

        write_block(cell_slab_cell_ids);
        write_block(cell_source_cell_ids);
        write_block(cell_local_vertex_indices);
        write_block(cell_source_local_vertex_indices);
        write_block(cell_source_spatial_vertex_ids);
        write_block(cell_slab_spatial_vertex_ids);
        write_block(cell_source_spatial_vertex_ids_per_cell);
        write_block(cell_triangle_coordinates);
        write_block(cell_time_intervals);
    }
}

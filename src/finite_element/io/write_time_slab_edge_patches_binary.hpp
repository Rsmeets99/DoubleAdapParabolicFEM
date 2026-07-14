#pragma once

#include <filesystem>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "detail/time_slab_patch_io_detail.hpp"

namespace finite_element::io
{
    template<class PatchSetType>
    void write_time_slab_edge_patches_binary(
        const PatchSetType& patch_set,
        const std::filesystem::path& output_dir,
        const std::string& filename = "time_slab_edge_patches.bin")
    {
        using int_t    = finite_element::io::detail::binary_int_t;
        using PatchType = typename PatchSetType::PatchType;
        constexpr const char* context = "write_time_slab_edge_patches_binary";

        constexpr int_t max_cells_per_patch = static_cast<int_t>(
            std::tuple_size_v<decltype(std::declval<PatchType>().cells)>);
        const auto export_info =
            finite_element::io::detail::make_time_slab_patch_export_info(
                patch_set,
                max_cells_per_patch);

        auto file =
            finite_element::io::detail::open_binary_output(
                output_dir,
                filename,
                context);

        finite_element::io::detail::write_time_slab_patch_header(
            file,
            finite_element::io::detail::time_slab_edge_patches_binary_format_version,
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
        std::vector<double> x_vertices;
        std::vector<double> t_begins;
        std::vector<double> t_ends;

        const auto flat_size = export_info.flat_cell_size();
        std::vector<int_t> cell_slab_cell_ids(flat_size, -1);
        std::vector<int_t> cell_source_cell_ids(flat_size, -1);
        std::vector<int_t> cell_left_vertex_ids(flat_size, -1);
        std::vector<int_t> cell_right_vertex_ids(flat_size, -1);
        std::vector<int_t> cell_sides(flat_size, -1);
        std::vector<double> cell_x_begins(flat_size, 0.0);
        std::vector<double> cell_x_ends(flat_size, 0.0);

        patch_ids.reserve(static_cast<std::size_t>(export_info.n_patches));
        slab_ids.reserve(static_cast<std::size_t>(export_info.n_patches));
        spatial_vertex_ids.reserve(static_cast<std::size_t>(export_info.n_patches));
        kinds.reserve(static_cast<std::size_t>(export_info.n_patches));
        n_cells.reserve(static_cast<std::size_t>(export_info.n_patches));
        x_vertices.reserve(static_cast<std::size_t>(export_info.n_patches));
        t_begins.reserve(static_cast<std::size_t>(export_info.n_patches));
        t_ends.reserve(static_cast<std::size_t>(export_info.n_patches));

        const auto encode_kind =
            [](const auto& patch) -> int_t
            {
                return patch.is_boundary() ? 1 : 0;
            };

        const auto encode_side =
            [](const auto& side) -> int_t
            {
                using Side = std::decay_t<decltype(side)>;
                if (side == Side::left_of_vertex)
                    return 0;
                if (side == Side::right_of_vertex)
                    return 1;
                return -1;
            };

        for (int patch_id = 0; patch_id < patch_set.n_patches(); ++patch_id)
        {
            const auto& patch = patch_set.patch(patch_id);
            patch_ids.push_back(static_cast<int_t>(patch.patch_id));
            slab_ids.push_back(static_cast<int_t>(patch.slab_id));
            spatial_vertex_ids.push_back(static_cast<int_t>(patch.spatial_vertex_id));
            kinds.push_back(encode_kind(patch));
            n_cells.push_back(static_cast<int_t>(patch.n_cells));
            x_vertices.push_back(patch.x_vertex);
            t_begins.push_back(patch.t_begin);
            t_ends.push_back(patch.t_end);

            const auto patch_offset =
                static_cast<std::size_t>(patch_id) * static_cast<std::size_t>(max_cells_per_patch);

            for (int patch_cell_index = 0; patch_cell_index < patch.n_cells; ++patch_cell_index)
            {
                const auto& cell = patch.cell(patch_cell_index);
                const auto flat_index =
                    patch_offset + static_cast<std::size_t>(patch_cell_index);

                cell_slab_cell_ids[flat_index] =
                    static_cast<int_t>(cell.slab_cell_id);
                cell_source_cell_ids[flat_index] =
                    static_cast<int_t>(cell.source_cell_id);
                cell_left_vertex_ids[flat_index] =
                    static_cast<int_t>(cell.left_vertex_id);
                cell_right_vertex_ids[flat_index] =
                    static_cast<int_t>(cell.right_vertex_id);
                cell_sides[flat_index] = encode_side(cell.side);
                cell_x_begins[flat_index] = cell.x_begin;
                cell_x_ends[flat_index] = cell.x_end;
            }
        }

        write_block(patch_ids);
        write_block(slab_ids);
        write_block(spatial_vertex_ids);
        write_block(kinds);
        write_block(n_cells);
        write_block(x_vertices);
        write_block(t_begins);
        write_block(t_ends);

        write_block(cell_slab_cell_ids);
        write_block(cell_source_cell_ids);
        write_block(cell_left_vertex_ids);
        write_block(cell_right_vertex_ids);
        write_block(cell_sides);
        write_block(cell_x_begins);
        write_block(cell_x_ends);
    }
}

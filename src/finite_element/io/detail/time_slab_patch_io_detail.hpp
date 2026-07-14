#pragma once

#include <cstddef>
#include <fstream>
#include <vector>

#include "binary_format_versions.hpp"
#include "finite_element/fespace/io/detail/write_binary_block.hpp"

namespace finite_element::io::detail
{
    struct TimeSlabPatchExportInfo
    {
        binary_int_t n_slabs = 0;
        binary_int_t n_patches = 0;
        binary_int_t max_cells_per_patch = 0;

        [[nodiscard]] std::size_t flat_cell_size() const noexcept
        {
            return static_cast<std::size_t>(n_patches) *
                   static_cast<std::size_t>(max_cells_per_patch);
        }

        [[nodiscard]] std::size_t flat_cell_component_size(
            int components_per_cell) const noexcept
        {
            return flat_cell_size() * static_cast<std::size_t>(components_per_cell);
        }
    };

    struct TimeSlabPatchSlabIndex
    {
        std::vector<binary_int_t> slab_offsets{};
        std::vector<binary_int_t> slab_patch_ids{};
    };

    template<class PatchSetType>
    [[nodiscard]] TimeSlabPatchExportInfo make_time_slab_patch_export_info(
        const PatchSetType& patch_set,
        binary_int_t max_cells_per_patch)
    {
        return TimeSlabPatchExportInfo{
            static_cast<binary_int_t>(patch_set.slab_space().n_slabs()),
            static_cast<binary_int_t>(patch_set.n_patches()),
            max_cells_per_patch
        };
    }

    template<class PatchSetType>
    [[nodiscard]] TimeSlabPatchSlabIndex make_time_slab_patch_slab_index(
        const PatchSetType& patch_set,
        const TimeSlabPatchExportInfo& export_info)
    {
        TimeSlabPatchSlabIndex index;
        index.slab_offsets.reserve(static_cast<std::size_t>(export_info.n_slabs) + 1u);
        index.slab_patch_ids.reserve(static_cast<std::size_t>(export_info.n_patches));

        index.slab_offsets.push_back(0);
        for (int slab_id = 0; slab_id < patch_set.slab_space().n_slabs(); ++slab_id)
        {
            const auto& patch_ids = patch_set.slab_patch_ids(slab_id);
            for (const int patch_id : patch_ids)
                index.slab_patch_ids.push_back(static_cast<binary_int_t>(patch_id));
            index.slab_offsets.push_back(
                static_cast<binary_int_t>(index.slab_patch_ids.size()));
        }

        return index;
    }

    inline void write_time_slab_patch_header(
        std::ofstream& file,
        binary_int_t format_version,
        const TimeSlabPatchExportInfo& export_info,
        const char* context)
    {
        write_binary_block(
            file,
            make_versioned_header(
                format_version,
                export_info.n_slabs,
                export_info.n_patches,
                export_info.max_cells_per_patch),
            context);
    }

    template<class T>
    void write_time_slab_patch_block(
        std::ofstream& file,
        const std::vector<T>& block,
        const char* context)
    {
        write_binary_block(file, block, context);
    }

    inline void write_time_slab_patch_slab_index(
        std::ofstream& file,
        const TimeSlabPatchSlabIndex& index,
        const char* context)
    {
        write_time_slab_patch_block(file, index.slab_offsets, context);
        write_time_slab_patch_block(file, index.slab_patch_ids, context);
    }
}

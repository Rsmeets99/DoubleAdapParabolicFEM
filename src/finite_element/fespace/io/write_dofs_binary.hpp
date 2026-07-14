#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "finite_element/io/detail/binary_format_versions.hpp"
#include "detail/write_binary_block.hpp"
#include "detail/write_dofs_binary_detail.hpp"

namespace finite_element
{
    template<typename GeomTraits,
             typename FETraits,
             typename Policy>
    void
    FESpace<GeomTraits, FETraits, Policy>::write_dofs_binary(
        const std::filesystem::path& output_dir,
        const std::string& filename) const
    {
        using int_t = finite_element::io::detail::binary_int_t;

        constexpr int dim_v         = GT::dim_v;
        constexpr int dofs_per_cell = FETraits::dofs_per_cell;
        constexpr int p_space_v     = FETraits::p_space_v;
        constexpr int p_time_v      = FETraits::p_time_v;

        const int_t n_cells = static_cast<int_t>(active_cells().size());
        const int_t n_dofs  = static_cast<int_t>(dof_handler_.n_dofs());

        const auto dof_coords = finite_element::io::detail::build_dof_coordinates(*this);
        const auto cell_ids = finite_element::io::detail::build_cell_ids(*this);
        const auto cell_to_dofs_flat = finite_element::io::detail::build_cell_to_dofs_flat(*this);
        const auto local_reference_coords =
            finite_element::io::detail::build_local_reference_coords<ElemTables>(dofs_per_cell, dim_v);

        std::vector<std::uint8_t> is_constrained;
        std::vector<int_t> constraint_offsets;
        std::vector<int_t> constraint_masters;
        std::vector<double> constraint_weights;

        finite_element::io::detail::build_constraint_data(
            *this,
            is_constrained,
            constraint_offsets,
            constraint_masters,
            constraint_weights);

        const int_t n_constraint_entries = static_cast<int_t>(constraint_masters.size());

        auto file = finite_element::io::detail::open_binary_output(output_dir, filename, "write_dofs_binary");

        const auto header = finite_element::io::detail::make_versioned_header(
            finite_element::io::detail::dofs_binary_format_version,
            n_cells,
            static_cast<int_t>(dofs_per_cell),
            n_dofs,
            static_cast<int_t>(dim_v),
            static_cast<int_t>(p_space_v),
            static_cast<int_t>(p_time_v),
            n_constraint_entries);

        finite_element::io::detail::write_binary_block(file, header, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, dof_coords, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, cell_ids, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, cell_to_dofs_flat, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, local_reference_coords, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, is_constrained, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, constraint_offsets, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, constraint_masters, "write_dofs_binary");
        finite_element::io::detail::write_binary_block(file, constraint_weights, "write_dofs_binary");
    }
}

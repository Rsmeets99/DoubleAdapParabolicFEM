#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "finite_element/fespace/dofs/physical_dof_coords.hpp"
#include "finite_element/io/detail/binary_format_versions.hpp"

namespace finite_element::io::detail
{
    template<class SpaceType>
    std::vector<double> build_dof_coordinates(const SpaceType& space)
    {
        using GT = typename SpaceType::GT;

        constexpr int dim_v = GT::dim_v;
        const int n_dofs    = space.dof_handler_ref().n_dofs();

        std::vector<double> dof_coords(static_cast<std::size_t>(n_dofs) * dim_v, 0.0);
        std::vector<char> coord_written(static_cast<std::size_t>(n_dofs), 0);

        for (const int cell_id : space.active_cells())
        {
            const auto& cell_dofs = space.dof_handler_ref().cell_dofs(cell_id);

            for (int local_index = 0; local_index < SpaceType::FETraitsType::dofs_per_cell; ++local_index)
            {
                const int gid = cell_dofs[static_cast<std::size_t>(local_index)];
                if (gid < 0)
                    continue;

                if (!coord_written[static_cast<std::size_t>(gid)])
                {
                    const auto p = ::finite_element::fespace::physical_dof_coord(space, cell_id, local_index);

                    for (int d = 0; d < dim_v; ++d)
                        dof_coords[static_cast<std::size_t>(gid) * dim_v + d] = p[static_cast<std::size_t>(d)];

                    coord_written[static_cast<std::size_t>(gid)] = 1;
                }
            }
        }

        for (int gid = 0; gid < n_dofs; ++gid)
        {
            if (!coord_written[static_cast<std::size_t>(gid)])
            {
                throw std::runtime_error(
                    "write_dofs_binary: failed to determine coordinates for a global DoF.");
            }
        }

        return dof_coords;
    }

    template<class SpaceType>
    std::vector<binary_int_t> build_cell_ids(const SpaceType& space)
    {
        std::vector<binary_int_t> cell_ids;
        cell_ids.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
            cell_ids.push_back(static_cast<binary_int_t>(cell_id));

        return cell_ids;
    }

    template<class SpaceType>
    std::vector<binary_int_t> build_cell_to_dofs_flat(const SpaceType& space)
    {
        constexpr int dofs_per_cell = SpaceType::FETraitsType::dofs_per_cell;

        std::vector<binary_int_t> flat;
        flat.reserve(space.active_cells().size() * dofs_per_cell);

        for (const int cell_id : space.active_cells())
        {
            const auto& cell_dofs = space.dof_handler_ref().cell_dofs(cell_id);
            for (int local_index = 0; local_index < dofs_per_cell; ++local_index)
                flat.push_back(static_cast<binary_int_t>(cell_dofs[static_cast<std::size_t>(local_index)]));
        }

        return flat;
    }

    template<class ElemTables>
    std::vector<double> build_local_reference_coords(int dofs_per_cell, int dim_v)
    {
        std::vector<double> local_reference_coords(
            static_cast<std::size_t>(dofs_per_cell) * dim_v, 0.0);

        for (int local_index = 0; local_index < dofs_per_cell; ++local_index)
        {
            const auto& xi = ElemTables::coord(local_index);
            for (int d = 0; d < dim_v; ++d)
                local_reference_coords[static_cast<std::size_t>(local_index) * dim_v + d] =
                    xi[static_cast<std::size_t>(d)];
        }

        return local_reference_coords;
    }

    template<class SpaceType>
    void build_constraint_data(
        const SpaceType& space,
        std::vector<std::uint8_t>& is_constrained,
        std::vector<binary_int_t>& constraint_offsets,
        std::vector<binary_int_t>& constraint_masters,
        std::vector<double>& constraint_weights)
    {
        const int n_dofs = space.dof_handler_ref().n_dofs();

        is_constrained.assign(static_cast<std::size_t>(n_dofs), 0);
        constraint_offsets.assign(static_cast<std::size_t>(n_dofs) + 1, 0);
        constraint_masters.clear();
        constraint_weights.clear();

        binary_int_t running_offset = 0;

        for (int gid = 0; gid < n_dofs; ++gid)
        {
            const auto& dof = space.dof_handler_ref().dof(gid);

            is_constrained[static_cast<std::size_t>(gid)] =
                static_cast<std::uint8_t>(dof.is_constrained ? 1 : 0);

            constraint_offsets[static_cast<std::size_t>(gid)] = running_offset;

            if (!dof.is_constrained)
                continue;

            if (dof.constraint_masters.size() != dof.constraint_weights.size())
            {
                throw std::runtime_error(
                    "write_dofs_binary: inconsistent constraint master/weight sizes.");
            }

            for (std::size_t k = 0; k < dof.constraint_masters.size(); ++k)
            {
                constraint_masters.push_back(static_cast<binary_int_t>(dof.constraint_masters[k]));
                constraint_weights.push_back(dof.constraint_weights[k]);
                ++running_offset;
            }
        }

        constraint_offsets[static_cast<std::size_t>(n_dofs)] = running_offset;
    }
}

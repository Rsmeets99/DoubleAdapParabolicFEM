#pragma once

#include <vector>

#include "../cell.hpp"

namespace mesh::topology
{
    template<typename GeomTraits>
    [[nodiscard]] inline std::vector<core::CellId> leaf_cell_ids(
        const std::vector<Cell<GeomTraits>>& cells)
    {
        std::vector<core::CellId> result;
        result.reserve(cells.size());

        for (const auto& cell : cells)
        {
            if (cell.is_leaf)
                result.push_back(cell.cell_id);
        }

        return result;
    }
}
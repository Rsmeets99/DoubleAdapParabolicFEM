#pragma once

#include "../../../core/exceptions.hpp"
#include "../../mesh_traits.hpp"
#include "slice_cell_in_time_1d.hpp"
#include "slice_cell_in_time_2d.hpp"

namespace mesh::refinement::time_slicing
{
    template<typename GeomTraits>
    [[nodiscard]] inline typename mesh::MeshTypes<GeomTraits>::cell_id_type
    slice_cell_in_time(
        mesh::Mesh<GeomTraits>& target_mesh,
        const mesh::Mesh<GeomTraits>& source_mesh,
        typename mesh::MeshTypes<GeomTraits>::cell_id_type source_cell_id,
        double t_begin,
        double t_end,
        double tol = 1.0e-14)
    {
        if constexpr (GeomTraits::dim_space_v == 1)
        {
            return slice_cell_in_time_1d<GeomTraits>(
                target_mesh,
                source_mesh,
                source_cell_id,
                t_begin,
                t_end,
                tol);
        }
        else if constexpr (GeomTraits::dim_space_v == 2)
        {
            return slice_cell_in_time_2d<GeomTraits>(
                target_mesh,
                source_mesh,
                source_cell_id,
                t_begin,
                t_end,
                tol);
        }
        else
        {
            throw core::dimension_not_supported_error(
                "slice_cell_in_time is only implemented for 1+1D or 2+1D.");
        }
    }
}

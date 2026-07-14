#pragma once

#include <stdexcept>
#include <vector>

#include "../../../mesh/refinement/refinement_type.hpp"
#include "refine_1d.hpp"
#include "refine_2d.hpp"

namespace finite_element::fespace
{
    template<typename FESpaceType>
    inline void refine(
        FESpaceType& space,
        const std::vector<int>& marked,
        mesh::RefinementType requested_refinement_type =
            mesh::RefinementType::none)
    {
        // requested_refinement_type == none is the physical FE path. It is
        // resolved by mesh::refinement::next_split_type, so 2+1D spaces follow
        // the admissible generation-based spatial/spacetime sequence. Pure
        // temporal refinement remains available only for explicit auxiliary
        // callers.
        if constexpr (FESpaceType::GT::dim_space_v == 1)
        {
            detail::refinement_impl::refine_1d(
                space,
                marked,
                requested_refinement_type);
        }
        else if constexpr (FESpaceType::GT::dim_space_v == 2)
        {
            detail::refinement_impl::refine_2d(
                space,
                marked,
                requested_refinement_type);
        }
        else
        {
            throw std::runtime_error(
                "FESpace refinement is currently only implemented for dim_space_v == 1 or 2.");
        }
    }
}

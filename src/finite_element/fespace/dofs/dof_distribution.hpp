#pragma once

#include <stdexcept>

#include "dof_distribution_1d.hpp"
#include "dof_distribution_2d.hpp"

namespace finite_element::fespace
{
    template<typename FESpaceType>
    inline void distribute_dofs(FESpaceType& space)
    {
        if constexpr (FESpaceType::GT::dim_space_v == 1)
        {
            detail::dof_distribution_impl::distribute_dofs_1d(space);
        }
        else if constexpr (FESpaceType::GT::dim_space_v == 2)
        {
            detail::dof_distribution_impl::distribute_dofs_2d(space);
        }
        else
        {
            throw std::runtime_error("DoF distribution is currently only implemented for dim_space_v == 1 or 2.");
        }
    }
}

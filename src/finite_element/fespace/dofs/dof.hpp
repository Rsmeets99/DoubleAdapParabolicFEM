#pragma once

#include <array>
#include <vector>

namespace finite_element::fespace
{
    template<typename GeomTraits, typename FETraits>
    struct DoF
    {
        static constexpr int invalid_id = -1;

        bool is_constrained = false;

        // Valid iff !is_constrained
        int true_dof_id = invalid_id;

        // Valid iff is_constrained
        std::vector<int> constraint_masters;
        std::vector<double> constraint_weights;

        [[nodiscard]] bool is_true_dof() const noexcept
        {
            return !is_constrained;
        }

        [[nodiscard]] bool has_true_dof_id() const noexcept
        {
            return true_dof_id != invalid_id;
        }
    };

    struct DoFKeyValue
    {
        int cell_id = -1;
        int local_index = -1;
    };

    template<typename GeomTraits, typename FETraits>
    struct DoFTypes
    {
        using DoFType      = DoF<GeomTraits, FETraits>;
        using CellDoFArray = std::array<int, FETraits::dofs_per_cell>;
        using DoFValueType = DoFKeyValue;
    };
}

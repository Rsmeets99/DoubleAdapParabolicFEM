#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace finite_element::assembly
{
    struct WeightedTrueDof
    {
        int true_dof = -1;
        double weight = 0.0;
    };

    inline void add_weighted_true_dof(
        std::vector<WeightedTrueDof>& entries,
        int true_dof,
        double weight)
    {
        for (auto& entry : entries)
        {
            if (entry.true_dof == true_dof)
            {
                entry.weight += weight;
                return;
            }
        }

        entries.push_back({true_dof, weight});
    }

    // CellRestriction is the cell-local restriction matrix C_c in
    // u_cell = C_c u_true. Each row corresponds to one local basis DoF and
    // contains the true DoFs and weights needed to evaluate that local value.
    struct CellRestriction
    {
        std::vector<std::vector<WeightedTrueDof>> rows;

        [[nodiscard]] int size() const
        {
            return static_cast<int>(rows.size());
        }

        [[nodiscard]] const std::vector<WeightedTrueDof>& operator[](int i) const
        {
            return rows[static_cast<std::size_t>(i)];
        }

        [[nodiscard]] std::vector<WeightedTrueDof>& operator[](int i)
        {
            return rows[static_cast<std::size_t>(i)];
        }
    };

    using LocalDofExpansion = CellRestriction;

    template<class DoFHandlerType, class CellDofs>
    void validate_cell_restriction(
        const DoFHandlerType& dof_handler,
        const CellDofs& cell_dofs,
        const CellRestriction& restriction)
    {
        constexpr int dofs_per_cell = DoFHandlerType::ElemTables::dofs_per_cell;
        if (restriction.size() != dofs_per_cell)
        {
            throw std::runtime_error(
                "validate_cell_restriction: row count does not match local DoF count.");
        }

        for (int local_index = 0; local_index < dofs_per_cell; ++local_index)
        {
            const int gid = cell_dofs[local_index];
            const auto& row = restriction[local_index];

            if (gid < 0)
            {
                if (!row.empty())
                {
                    throw std::runtime_error(
                        "validate_cell_restriction: eliminated local DoF has a non-empty row.");
                }
                continue;
            }

            if (gid >= dof_handler.n_dofs())
            {
                throw std::runtime_error(
                    "validate_cell_restriction: local DoF references invalid global id.");
            }

            const auto& dof = dof_handler.dof(gid);
            if (!dof.is_constrained)
            {
                if (row.size() != 1U ||
                    row.front().true_dof != dof.true_dof_id ||
                    row.front().weight != 1.0)
                {
                    throw std::runtime_error(
                        "validate_cell_restriction: unconstrained local DoF does not have an identity row.");
                }
            }
            else
            {
                if (row.empty())
                {
                    throw std::runtime_error(
                        "validate_cell_restriction: constrained local DoF has an empty row.");
                }
            }

            int previous_true_dof = -1;
            for (const auto& entry : row)
            {
                if (entry.true_dof < 0 ||
                    entry.true_dof >= dof_handler.n_true_dofs())
                {
                    throw std::runtime_error(
                        "validate_cell_restriction: row references invalid true DoF.");
                }
                if (entry.true_dof <= previous_true_dof)
                {
                    throw std::runtime_error(
                        "validate_cell_restriction: row entries are not strictly ordered by true DoF.");
                }
                previous_true_dof = entry.true_dof;
            }
        }
    }

    template<class FESpaceType>
    CellRestriction build_cell_restriction(
        const FESpaceType& space,
        int cell_id)
    {
        using DoFHandlerType =
            std::remove_reference_t<decltype(space.dof_handler_ref())>;

        constexpr int dofs_per_cell = DoFHandlerType::ElemTables::dofs_per_cell;

        const auto& dof_handler = space.dof_handler_ref();
        const auto& cell_dofs   = dof_handler.cell_dofs(cell_id);

        CellRestriction out;
        out.rows.resize(dofs_per_cell);

        for (int local_index = 0; local_index < dofs_per_cell; ++local_index)
        {
            const int gid = cell_dofs[local_index];
            if (gid < 0)
                continue;

            const auto& dof = dof_handler.dof(gid);

            if (!dof.is_constrained)
            {
                const int true_id = dof.true_dof_id;
                if (true_id < 0 || true_id >= dof_handler.n_true_dofs())
                {
                    throw std::runtime_error(
                        "build_cell_restriction: unconstrained DoF has invalid true_dof_id.");
                }

                add_weighted_true_dof(out.rows[local_index], true_id, 1.0);
                continue;
            }

            if (dof.constraint_masters.size() != dof.constraint_weights.size())
            {
                throw std::runtime_error(
                    "build_cell_restriction: inconsistent constrained DoF metadata.");
            }

            out.rows[local_index].reserve(dof.constraint_masters.size());

            for (std::size_t k = 0; k < dof.constraint_masters.size(); ++k)
            {
                const int master_gid = dof.constraint_masters[k];
                const double weight  = dof.constraint_weights[k];

                if (master_gid < 0)
                    continue;

                if (master_gid >= dof_handler.n_dofs())
                {
                    throw std::runtime_error(
                        "build_cell_restriction: constrained DoF references invalid master global id.");
                }

                const auto& master_dof = dof_handler.dof(master_gid);

                if (master_dof.is_constrained)
                {
                    throw std::runtime_error(
                        "build_cell_restriction: constrained DoF references constrained master.");
                }

                const int master_true_id = master_dof.true_dof_id;
                if (master_true_id < 0 || master_true_id >= dof_handler.n_true_dofs())
                {
                    throw std::runtime_error(
                        "build_cell_restriction: master DoF has invalid true_dof_id.");
                }

                add_weighted_true_dof(
                    out.rows[local_index],
                    master_true_id,
                    weight);
            }
        }

        for (auto& row : out.rows)
        {
            std::sort(
                row.begin(),
                row.end(),
                [](const WeightedTrueDof& a, const WeightedTrueDof& b)
                {
                    return a.true_dof < b.true_dof;
                });
        }

        validate_cell_restriction(dof_handler, cell_dofs, out);
        return out;
    }

    template<class FESpaceType>
    LocalDofExpansion build_local_dof_expansion(
        const FESpaceType& space,
        int cell_id)
    {
        return build_cell_restriction(space, cell_id);
    }
}

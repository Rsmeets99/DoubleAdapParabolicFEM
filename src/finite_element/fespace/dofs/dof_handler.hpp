#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "dof.hpp"
#include "../../tables/element_tables.hpp"

namespace finite_element::fespace
{
    template<typename GeomTraits, typename FETraits>
    class DoFHandler : public DoFTypes<GeomTraits, FETraits>
    {
    public:
        using Types        = DoFTypes<GeomTraits, FETraits>;
        using DoFType      = typename Types::DoFType;
        using CellDoFArray = typename Types::CellDoFArray;
        using DoFValueType = typename Types::DoFValueType;

        using ElemTables = finite_element::tables::ElementDofTables<GeomTraits, FETraits>;
        static constexpr ElemTables elem_tables{};

        static constexpr int invalid_id = DoFType::invalid_id;

        void clear()
        {
            dofs_.clear();
            cell_to_dofs_.clear();
            cell_has_dofs_.clear();
            true_to_global_.clear();
            total_true_dofs_ = 0;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return static_cast<int>(dofs_.size());
        }

        [[nodiscard]] int n_true_dofs() const noexcept
        {
            return total_true_dofs_;
        }

        [[nodiscard]] std::size_t n_constrained_dofs() const noexcept
        {
            std::size_t count = 0;
            for (const auto& dof : dofs_)
            {
                if (dof.is_constrained)
                    ++count;
            }
            return count;
        }

        [[nodiscard]] std::size_t prolongation_nonzeros() const noexcept
        {
            std::size_t nnz = static_cast<std::size_t>(total_true_dofs_);
            for (const auto& dof : dofs_)
            {
                if (dof.is_constrained)
                    nnz += dof.constraint_masters.size();
            }
            return nnz;
        }

        void initialize_cell(int cell_id)
        {
            if (cell_id < 0)
                throw std::out_of_range("DoFHandler::initialize_cell: negative cell id.");

            ensure_cell_slot_(cell_id);
            cell_to_dofs_[static_cast<std::size_t>(cell_id)].fill(-1);
            cell_has_dofs_[static_cast<std::size_t>(cell_id)] = true;
        }

        void reserve_cell_slots(std::size_t n_cells)
        {
            cell_to_dofs_.reserve(n_cells);
            cell_has_dofs_.reserve(n_cells);
        }

        int add_dof(DoFType dof)
        {
            if (!dof.is_constrained)
            {
                if (dof.true_dof_id == invalid_id)
                    dof.true_dof_id = total_true_dofs_;

                if (dof.true_dof_id != total_true_dofs_)
                {
                    throw std::runtime_error(
                        "DoFHandler::add_dof: unconstrained DoF has unexpected true_dof_id.");
                }

                true_to_global_.push_back(static_cast<int>(dofs_.size()));
                ++total_true_dofs_;
            }
            else
            {
                if (dof.true_dof_id != invalid_id)
                {
                    throw std::runtime_error(
                        "DoFHandler::add_dof: constrained DoF must not have a true_dof_id.");
                }

                if (dof.constraint_masters.size() != dof.constraint_weights.size())
                {
                    throw std::runtime_error(
                        "DoFHandler::add_dof: constrained DoF has inconsistent master/weight sizes.");
                }
            }

            dofs_.push_back(std::move(dof));
            return static_cast<int>(dofs_.size()) - 1;
        }

        void set_cell_dof(int cell_id, int local_index, int global_id)
        {
            if (cell_id < 0)
                throw std::out_of_range("DoFHandler::set_cell_dof: negative cell id.");
            if (local_index < 0 ||
                static_cast<std::size_t>(local_index) >= cell_dofs_per_cell_())
            {
                throw std::out_of_range("DoFHandler::set_cell_dof: local index out of range.");
            }

            ensure_cell_slot_(cell_id);
            const auto index = static_cast<std::size_t>(cell_id);
            if (!cell_has_dofs_[index])
            {
                cell_to_dofs_[index].fill(-1);
                cell_has_dofs_[index] = true;
            }
            cell_to_dofs_[index][static_cast<std::size_t>(local_index)] = global_id;
        }

        [[nodiscard]] const DoFType& dof(int id) const
        {
            return dofs_.at(static_cast<std::size_t>(id));
        }

        [[nodiscard]] DoFType& dof(int id)
        {
            return dofs_.at(static_cast<std::size_t>(id));
        }

        [[nodiscard]] const CellDoFArray& cell_dofs(int cell_id) const
        {
            require_cell_dofs_(cell_id, "DoFHandler::cell_dofs");
            return cell_to_dofs_[static_cast<std::size_t>(cell_id)];
        }

        [[nodiscard]] CellDoFArray& cell_dofs(int cell_id)
        {
            require_cell_dofs_(cell_id, "DoFHandler::cell_dofs");
            return cell_to_dofs_[static_cast<std::size_t>(cell_id)];
        }

        [[nodiscard]] bool has_cell_dofs(int cell_id) const noexcept
        {
            return cell_id >= 0 &&
                   static_cast<std::size_t>(cell_id) < cell_has_dofs_.size() &&
                   cell_has_dofs_[static_cast<std::size_t>(cell_id)];
        }

        [[nodiscard]] std::size_t cell_slot_count() const noexcept
        {
            return cell_to_dofs_.size();
        }

        [[nodiscard]] std::size_t active_cell_slot_count() const noexcept
        {
            std::size_t count = 0;
            for (const bool has_dofs : cell_has_dofs_)
            {
                if (has_dofs)
                    ++count;
            }
            return count;
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            std::size_t bytes = 0;
            bytes += dofs_.capacity() * sizeof(DoFType);
            bytes += cell_to_dofs_.capacity() * sizeof(CellDoFArray);
            bytes += cell_has_dofs_.capacity() * sizeof(unsigned char);
            bytes += true_to_global_.capacity() * sizeof(int);
            for (const auto& dof : dofs_)
            {
                bytes += dof.constraint_masters.capacity() * sizeof(int);
                bytes += dof.constraint_weights.capacity() * sizeof(double);
            }
            return bytes;
        }

        [[nodiscard]] std::size_t
        constraints_estimated_memory_bytes() const noexcept
        {
            std::size_t bytes = 0;
            for (const auto& dof : dofs_)
            {
                if (!dof.is_constrained)
                    continue;
                bytes += sizeof(DoFType);
                bytes += dof.constraint_masters.capacity() * sizeof(int);
                bytes += dof.constraint_weights.capacity() * sizeof(double);
            }
            return bytes;
        }

        [[nodiscard]] int global_to_true(int global_id) const
        {
            const auto& d = dof(global_id);
            return d.true_dof_id;
        }

        [[nodiscard]] int true_to_global(int true_id) const
        {
            return true_to_global_.at(static_cast<std::size_t>(true_id));
        }

        [[nodiscard]] bool is_true_dof(int global_id) const
        {
            return !dof(global_id).is_constrained;
        }

        [[nodiscard]] bool validate() const
        {
            if (static_cast<int>(true_to_global_.size()) != total_true_dofs_)
                return false;

            for (int gid = 0; gid < n_dofs(); ++gid)
            {
                const auto& d = dofs_[gid];

                if (!d.is_constrained)
                {
                    if (d.true_dof_id < 0 || d.true_dof_id >= total_true_dofs_)
                        return false;

                    if (true_to_global_[d.true_dof_id] != gid)
                        return false;
                }
                else
                {
                    if (d.true_dof_id != invalid_id)
                        return false;

                    if (d.constraint_masters.size() != d.constraint_weights.size())
                        return false;

                    for (int master_gid : d.constraint_masters)
                    {
                        if (master_gid < 0 || master_gid >= n_dofs())
                            return false;

                        if (dofs_[master_gid].is_constrained)
                            return false;
                    }
                }
            }

            for (std::size_t cell_id = 0; cell_id < cell_to_dofs_.size(); ++cell_id)
            {
                if (!cell_has_dofs_[cell_id])
                    continue;

                for (const int gid : cell_to_dofs_[cell_id])
                {
                    if (gid == invalid_id)
                        continue;
                    if (gid < 0 || gid >= n_dofs())
                        return false;
                }
            }

            return true;
        }

    private:
        [[nodiscard]] static constexpr std::size_t cell_dofs_per_cell_() noexcept
        {
            return CellDoFArray{}.size();
        }

        void ensure_cell_slot_(int cell_id)
        {
            const auto required_size = static_cast<std::size_t>(cell_id) + 1U;
            if (required_size <= cell_to_dofs_.size())
                return;

            const auto old_size = cell_to_dofs_.size();
            cell_to_dofs_.resize(required_size);
            cell_has_dofs_.resize(required_size, false);
            for (std::size_t i = old_size; i < required_size; ++i)
                cell_to_dofs_[i].fill(-1);
        }

        void require_cell_dofs_(int cell_id, const char* context) const
        {
            if (!has_cell_dofs(cell_id))
                throw std::out_of_range(std::string(context) + ": cell id has no DoFs.");
        }

        std::vector<DoFType> dofs_{};
        std::vector<CellDoFArray> cell_to_dofs_{};
        std::vector<unsigned char> cell_has_dofs_{};

        // true_id -> global_id
        std::vector<int> true_to_global_{};

        int total_true_dofs_ = 0;
    };

    template<typename G, typename F>
    constexpr typename DoFHandler<G, F>::ElemTables DoFHandler<G, F>::elem_tables;
}

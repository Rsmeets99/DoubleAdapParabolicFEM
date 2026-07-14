#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>

#include "../basis/space_time_basis_selector.hpp"
#include "../geometry/cell_geometry.hpp"
#include "linear_algebra/concepts/vector.hpp"

namespace finite_element
{
    namespace detail
    {
        template<class VectorLikeA, class VectorLikeB>
        void copy_vector(VectorLikeA& dst, const VectorLikeB& src)
        {
            dst.resize(src.size());
            for (int i = 0; i < src.size(); ++i)
                dst[i] = src[i];
        }

        template<class VectorLike>
        void fill_vector(VectorLike& v, double value)
        {
            for (int i = 0; i < v.size(); ++i)
                v[i] = value;
        }
    }

    template<class FESpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class Function
    {
    public:
        using SpaceType       = FESpaceType;
        using Vector          = VectorType;
        using GT              = typename SpaceType::GT;
        using FETraits        = typename SpaceType::FETraitsType;
        using MeshType        = typename SpaceType::MeshType;
        using DoFHandlerType  = typename SpaceType::DoFHandlerType;

        using Basis    = finite_element::basis::SpaceTimeBasis<GT, FETraits>;
        using Geometry = finite_element::geometry::CellGeometry<SpaceType, GT::dim_space_v>;
        using GeometryData = typename Geometry::Data;

        using SpaceTimePoint = typename SpaceType::SpaceTimePoint;
        using SpatialPoint   = typename SpaceType::SpatialPoint;
        using TemporalPoint  = typename SpaceType::TemporalPoint;

        static constexpr int dim           = GT::dim_v;
        static constexpr int dofs_per_cell = FETraits::dofs_per_cell;

        using ValueType       = double;
        using GradientType    = std::array<double, dim>;
        using BasisValues     = std::array<double, dofs_per_cell>;
        using BasisGradients  = std::array<std::array<double, dim>, dofs_per_cell>;

        explicit Function(const SpaceType& space)
            : space_(&space),
              true_coefficients_(space.dof_handler_ref().n_true_dofs()),
              full_coefficients_(space.dof_handler_ref().n_dofs())
        {
            set_zero();
        }

        Function(const SpaceType& space, const Vector& true_coefficients)
            : space_(&space),
              true_coefficients_(space.dof_handler_ref().n_true_dofs()),
              full_coefficients_(space.dof_handler_ref().n_dofs())
        {
            update_from_true_solution(true_coefficients);
        }

        Function(const SpaceType& space, double initial_true_value)
            : space_(&space),
              true_coefficients_(space.dof_handler_ref().n_true_dofs()),
              full_coefficients_(space.dof_handler_ref().n_dofs())
        {
            fill_true_coefficients(initial_true_value);
        }

        [[nodiscard]] static Function ones(const SpaceType& space)
        {
            return Function(space, 1.0);
        }

        [[nodiscard]] const SpaceType& fespace() const noexcept
        {
            return *space_;
        }

        [[nodiscard]] Vector& true_coefficients() noexcept
        {
            return true_coefficients_;
        }

        [[nodiscard]] const Vector& true_coefficients() const noexcept
        {
            return true_coefficients_;
        }

        [[nodiscard]] const Vector& full_coefficients() const noexcept
        {
            return full_coefficients_;
        }

        void set_zero()
        {
            detail::fill_vector(true_coefficients_, 0.0);
            detail::fill_vector(full_coefficients_, 0.0);
        }

        void fill_true_coefficients(double value)
        {
            detail::fill_vector(true_coefficients_, value);
            scatter_forward();
        }

        void update_from_true_solution(const Vector& true_solution)
        {
            const auto& dof_handler = space_->dof_handler_ref();

            if (true_solution.size() != dof_handler.n_true_dofs())
            {
                throw std::runtime_error(
                    "Function::update_from_true_solution: size does not match n_true_dofs().");
            }

            detail::copy_vector(true_coefficients_, true_solution);
            scatter_forward();
        }

        void scatter_forward()
        {
            const auto& dof_handler = space_->dof_handler_ref();

            if (true_coefficients_.size() != dof_handler.n_true_dofs())
            {
                throw std::runtime_error(
                    "Function::scatter_forward: true coefficient vector has wrong size.");
            }

            full_coefficients_.resize(dof_handler.n_dofs());
            detail::fill_vector(full_coefficients_, 0.0);

            for (int gid = 0; gid < dof_handler.n_dofs(); ++gid)
            {
                const auto& dof = dof_handler.dof(gid);

                if (!dof.is_constrained)
                {
                    const int true_id = dof.true_dof_id;

                    if (true_id < 0 || true_id >= dof_handler.n_true_dofs())
                    {
                        throw std::runtime_error(
                            "Function::scatter_forward: unconstrained DoF has invalid true_dof_id.");
                    }

                    full_coefficients_[gid] = true_coefficients_[true_id];
                    continue;
                }

                if (dof.constraint_masters.size() != dof.constraint_weights.size())
                {
                    throw std::runtime_error(
                        "Function::scatter_forward: inconsistent constrained DoF metadata.");
                }

                double value = 0.0;
                for (std::size_t k = 0; k < dof.constraint_masters.size(); ++k)
                {
                    const int master_gid = dof.constraint_masters[k];
                    const double weight  = dof.constraint_weights[k];

                    if (master_gid < 0 || master_gid >= dof_handler.n_dofs())
                    {
                        throw std::runtime_error(
                            "Function::scatter_forward: constrained DoF references invalid master global id.");
                    }

                    const auto& master_dof = dof_handler.dof(master_gid);

                    if (master_dof.is_constrained)
                    {
                        throw std::runtime_error(
                            "Function::scatter_forward: constrained DoF references constrained master.");
                    }

                    const int master_true_id = master_dof.true_dof_id;

                    if (master_true_id < 0 || master_true_id >= dof_handler.n_true_dofs())
                    {
                        throw std::runtime_error(
                            "Function::scatter_forward: master DoF has invalid true_dof_id.");
                    }

                    value += weight * true_coefficients_[master_true_id];
                }

                full_coefficients_[gid] = value;
            }
        }

        [[nodiscard]] static BasisValues basis_values(const SpaceTimePoint& xi)
        {
            return Basis::eval_all(xi);
        }

        [[nodiscard]] static BasisGradients basis_reference_gradients(const SpaceTimePoint& xi)
        {
            return Basis::grad_all(xi);
        }

        [[nodiscard]] ValueType value_on_reference_cell(
            int cell_id,
            const SpaceTimePoint& xi) const
        {
            check_active_cell(cell_id);

            const auto phi = basis_values(xi);
            const auto& cell_dofs = space_->dof_handler_ref().cell_dofs(cell_id);

            double value = 0.0;
            for (int a = 0; a < dofs_per_cell; ++a)
            {
                const int gid = cell_dofs[a];
                if (gid < 0)
                    continue;

                value += full_coefficients_[gid] * phi[a];
            }

            return value;
        }

        [[nodiscard]] GradientType gradient_on_reference_cell(
            int cell_id,
            const SpaceTimePoint& xi) const
        {
            check_active_cell(cell_id);

            const auto dphi = basis_reference_gradients(xi);
            const auto& cell_dofs = space_->dof_handler_ref().cell_dofs(cell_id);

            GradientType grad{};
            for (int a = 0; a < dofs_per_cell; ++a)
            {
                const int gid = cell_dofs[a];
                if (gid < 0)
                    continue;

                const double ua = full_coefficients_[gid];
                for (int d = 0; d < dim; ++d)
                    grad[d] += ua * dphi[a][d];
            }

            return grad;
        }

        [[nodiscard]] ValueType value_on_cell(
            int cell_id,
            const SpaceTimePoint& x_phys) const
        {
            check_active_cell(cell_id);
            const auto geom = Geometry::make(*space_, cell_id);
            return value_on_cell(cell_id, x_phys, geom);
        }

        [[nodiscard]] ValueType value_on_cell(
            int cell_id,
            const SpaceTimePoint& x_phys,
            const GeometryData& geom) const
        {
            check_active_cell(cell_id);
            const auto xi   = Geometry::physical_to_reference(geom, x_phys);
            return value_on_reference_cell(cell_id, xi);
        }

        [[nodiscard]] GradientType gradient_on_cell(
            int cell_id,
            const SpaceTimePoint& x_phys) const
        {
            check_active_cell(cell_id);

            const auto geom     = Geometry::make(*space_, cell_id);
            return gradient_on_cell(cell_id, x_phys, geom);
        }

        [[nodiscard]] GradientType gradient_on_cell(
            int cell_id,
            const SpaceTimePoint& x_phys,
            const GeometryData& geom) const
        {
            check_active_cell(cell_id);

            const auto xi       = Geometry::physical_to_reference(geom, x_phys);
            const auto grad_ref = gradient_on_reference_cell(cell_id, xi);

            return Geometry::full_gradient(geom, grad_ref);
        }

    private:
        const SpaceType* space_ = nullptr;
        Vector true_coefficients_{};
        Vector full_coefficients_{};

        void check_active_cell(int cell_id) const
        {
            if (!space_->is_active_cell(cell_id))
                throw std::runtime_error("Function: cell is not active.");
        }
    };
}

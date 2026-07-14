#pragma once

#include <array>
#include <stdexcept>

#include "../basis/polynomials/legendre.hpp"

namespace finite_element::error_fespace
{
    class PatchDofMap
    {
    public:
        PatchDofMap() = default;

        PatchDofMap(int n_spatial_dofs, int n_time_dofs)
            : n_spatial_dofs_(n_spatial_dofs),
              n_time_dofs_(n_time_dofs)
        {
            if (n_spatial_dofs_ <= 0 || n_time_dofs_ <= 0)
            {
                throw std::runtime_error(
                    "PatchDofMap: expected strictly positive spatial and temporal dimensions.");
            }
        }

        [[nodiscard]] int n_spatial_dofs() const noexcept
        {
            return n_spatial_dofs_;
        }

        [[nodiscard]] int n_time_dofs() const noexcept
        {
            return n_time_dofs_;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return n_spatial_dofs_ * n_time_dofs_;
        }

        [[nodiscard]] int index(int spatial_basis_id, int time_basis_id) const
        {
            if (spatial_basis_id < 0 || spatial_basis_id >= n_spatial_dofs_)
            {
                throw std::runtime_error(
                    "PatchDofMap::index: spatial basis index out of range.");
            }

            if (time_basis_id < 0 || time_basis_id >= n_time_dofs_)
            {
                throw std::runtime_error(
                    "PatchDofMap::index: time basis index out of range.");
            }

            return spatial_basis_id * n_time_dofs_ + time_basis_id;
        }

    private:
        int n_spatial_dofs_ = 0;
        int n_time_dofs_    = 0;
    };

    namespace detail
    {
        template<std::size_t N>
        void zero_array_prefix(std::array<double, N>& values) noexcept
        {
            values.fill(0.0);
        }

        template<int MaxDegree>
        [[nodiscard]] std::array<double, MaxDegree + 1>
        shifted_legendre_family(double x_ref)
        {
            std::array<double, MaxDegree + 1> values{};
            basis::polynomials::LegendrePolynomials<MaxDegree>::family(
                2.0 * x_ref - 1.0,
                values);
            return values;
        }

        template<int MaxDegree>
        [[nodiscard]] std::array<double, MaxDegree + 1>
        shifted_legendre_derivative_family(double x_ref)
        {
            std::array<double, MaxDegree + 1> values{};
            basis::polynomials::LegendrePolynomials<MaxDegree>::derivative_family(
                2.0 * x_ref - 1.0,
                values);

            for (double& value : values)
                value *= 2.0;

            return values;
        }

        template<int MaxDegree>
        [[nodiscard]] inline double shifted_legendre(int degree, double x_ref)
        {
            return basis::polynomials::LegendrePolynomials<MaxDegree>::eval(
                degree,
                2.0 * x_ref - 1.0);
        }

        template<int MaxDegree>
        [[nodiscard]] inline double shifted_legendre_derivative(
            int degree,
            double x_ref)
        {
            return 2.0 *
                basis::polynomials::LegendrePolynomials<MaxDegree>::eval_derivative(
                    degree,
                    2.0 * x_ref - 1.0);
        }
    }
}

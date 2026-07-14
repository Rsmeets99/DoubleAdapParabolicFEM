#pragma once

#include <array>
#include <cmath>

namespace quadrature::map
{
    // =====================================================
    // Small determinant helpers
    // =====================================================
    constexpr double det2x2(const std::array<std::array<double, 2>, 2>& J) noexcept
    {
        return J[0][0] * J[1][1] - J[0][1] * J[1][0];
    }

    // =====================================================
    // 1D interval map: [0,1] -> [a,b]
    // =====================================================
    struct IntervalMap1D
    {
        double a{};
        double b{};

        constexpr double map(double s) const noexcept
        {
            return a + (b - a) * s;
        }

        constexpr double jacobian() const noexcept
        {
            return b - a;
        }
    };

    // =====================================================
    // Affine triangle map:
    // T_ref = conv{(0,0),(1,0),(0,1)} -> physical triangle
    // =====================================================
    struct TriangleMap2D
    {
        std::array<double, 2> v0{};
        std::array<double, 2> v1{};
        std::array<double, 2> v2{};

        constexpr std::array<double, 2> map(double x, double y) const noexcept
        {
            return {
                v0[0] + x * (v1[0] - v0[0]) + y * (v2[0] - v0[0]),
                v0[1] + x * (v1[1] - v0[1]) + y * (v2[1] - v0[1])
            };
        }

        constexpr std::array<std::array<double, 2>, 2> jacobian_matrix() const noexcept
        {
            return {{
                {{ v1[0] - v0[0], v2[0] - v0[0] }},
                {{ v1[1] - v0[1], v2[1] - v0[1] }}
            }};
        }

        constexpr double jacobian_determinant() const noexcept
        {
            return det2x2(jacobian_matrix());
        }

        constexpr double jacobian_measure() const noexcept
        {
            const double det = jacobian_determinant();
            return det >= 0.0 ? det : -det;
        }
    };

    // =====================================================
    // Product map for 1+1D quadrilateral = physical segment x physical time interval
    // Reference cell: [0,1] x [0,1], point = {x_ref, t_ref}
    // Physical point = {x_phys, t_phys}
    // Constant Jacobian = |dx/dx_ref| * |dt/dt_ref|
    // =====================================================
    struct SpaceTimeQuadMap1P1D
    {
        IntervalMap1D space{};
        IntervalMap1D time{};

        constexpr std::array<double, 2> map(double x_ref, double t_ref) const noexcept
        {
            return {
                space.map(x_ref),
                time.map(t_ref)
            };
        }

        constexpr double jacobian_measure() const noexcept
        {
            const double jx = space.jacobian();
            const double jt = time.jacobian();

            const double ax = jx >= 0.0 ? jx : -jx;
            const double at = jt >= 0.0 ? jt : -jt;
            return ax * at;
        }
    };

    // =====================================================
    // Product map for 2+1D triangular prism = physical triangle x physical time interval
    // Reference cell: T_ref x [0,1], point = {x_ref, y_ref, t_ref}
    // Physical point = {x_phys, y_phys, t_phys}
    // Constant Jacobian = |det J_triangle| * |dt/dt_ref|
    // =====================================================
    struct SpaceTimeTriPrismMap2P1D
    {
        TriangleMap2D space{};
        IntervalMap1D time{};

        constexpr std::array<double, 3> map(double x_ref, double y_ref, double t_ref) const noexcept
        {
            const auto xy = space.map(x_ref, y_ref);
            return {
                xy[0],
                xy[1],
                time.map(t_ref)
            };
        }

        constexpr double jacobian_measure() const noexcept
        {
            const double jt = time.jacobian();
            const double at = jt >= 0.0 ? jt : -jt;
            return space.jacobian_measure() * at;
        }
    };
}
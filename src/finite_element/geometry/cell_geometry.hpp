#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "quadrature/cell_mappings.hpp"

namespace finite_element::geometry
{
    template<class FESpaceType, int DimSpace>
    struct CellGeometry;

    // =========================================================================
    // 1+1D geometry
    // =========================================================================
    template<class FESpaceType>
    struct CellGeometry<FESpaceType, 1>
    {
        using SpaceType      = FESpaceType;
        using GT             = typename SpaceType::GT;
        using SpaceTimePoint = typename SpaceType::SpaceTimePoint;
        using RefPoint       = std::array<double, 2>;
        using RefGradient    = std::array<double, 2>;
        using FullGradient   = std::array<double, 2>;
        using SpatialGradient = std::array<double, 1>;

        static_assert(GT::dim_space_v == 1,
                      "CellGeometry<...,1> requires dim_space_v == 1.");

        struct Data
        {
            quadrature::map::SpaceTimeQuadMap1P1D map{};
            double x0 = 0.0;
            double hx = 0.0;
            double t0 = 0.0;
            double ht = 0.0;
            double inv_hx = 0.0;
            double inv_ht = 0.0;
        };

        static Data make(const SpaceType& space, int cell_id)
        {
            const auto& mesh = space.mesh_ref();
            const auto& cell = mesh.cell(cell_id);

            const double x0 = mesh.spatial_vertices()[cell.spatial_vertex_ids[0]][0];
            const double x1 = mesh.spatial_vertices()[cell.spatial_vertex_ids[1]][0];
            const double t0 = mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
            const double t1 = mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];

            const double hx = x1 - x0;
            const double ht = t1 - t0;

            if (std::abs(hx) < 1e-15 || std::abs(ht) < 1e-15)
                throw std::runtime_error("CellGeometry<1+1D>::make: degenerate cell.");

            Data data;
            data.map.space.a = x0;
            data.map.space.b = x1;
            data.map.time.a  = t0;
            data.map.time.b  = t1;
            data.x0 = x0;
            data.hx = hx;
            data.t0 = t0;
            data.ht = ht;
            data.inv_hx = 1.0 / hx;
            data.inv_ht = 1.0 / ht;
            return data;
        }

        static RefPoint interior_reference_point() noexcept
        {
            return {0.5, 0.5};
        }

        static SpaceTimePoint map_to_physical(const Data& data, const RefPoint& xi) noexcept
        {
            return data.map.map(xi[0], xi[1]);
        }

        static RefPoint physical_to_reference(const Data& data, const SpaceTimePoint& x) noexcept
        {
            return RefPoint{
                (x[0] - data.x0) * data.inv_hx,
                (x[1] - data.t0) * data.inv_ht
            };
        }

        static bool reference_point_inside(const RefPoint& xi, double tol = 1e-12) noexcept
        {
            return (-tol <= xi[0] && xi[0] <= 1.0 + tol) &&
                   (-tol <= xi[1] && xi[1] <= 1.0 + tol);
        }

        static bool contains_physical_point(
            const Data& data,
            const SpaceTimePoint& x,
            double tol = 1e-12) noexcept
        {
            return reference_point_inside(physical_to_reference(data, x), tol);
        }

        static double jacobian_measure(const Data& data) noexcept
        {
            return data.map.jacobian_measure();
        }

        static SpatialGradient spatial_gradient(
            const Data& data,
            const RefGradient& grad_ref) noexcept
        {
            return { grad_ref[0] * data.inv_hx };
        }

        static double time_derivative(
            const Data& data,
            const RefGradient& grad_ref) noexcept
        {
            return grad_ref[1] * data.inv_ht;
        }

        static FullGradient full_gradient(
            const Data& data,
            const RefGradient& grad_ref) noexcept
        {
            return {
                grad_ref[0] * data.inv_hx,
                grad_ref[1] * data.inv_ht
            };
        }

        static double spatial_grad_dot(
            const Data& data,
            const RefGradient& grad_i,
            const RefGradient& grad_j) noexcept
        {
            const auto gi = spatial_gradient(data, grad_i);
            const auto gj = spatial_gradient(data, grad_j);
            return gi[0] * gj[0];
        }
    };

    // =========================================================================
    // 2+1D geometry
    // =========================================================================
    template<class FESpaceType>
    struct CellGeometry<FESpaceType, 2>
    {
        using SpaceType      = FESpaceType;
        using GT             = typename SpaceType::GT;
        using SpaceTimePoint = typename SpaceType::SpaceTimePoint;
        using RefPoint       = std::array<double, 3>;
        using RefGradient    = std::array<double, 3>;
        using FullGradient   = std::array<double, 3>;
        using SpatialGradient = std::array<double, 2>;

        static_assert(GT::dim_space_v == 2,
                      "CellGeometry<...,2> requires dim_space_v == 2.");

        struct Data
        {
            quadrature::map::SpaceTimeTriPrismMap2P1D map{};

            double x0 = 0.0;
            double y0 = 0.0;

            double J00 = 0.0;
            double J01 = 0.0;
            double J10 = 0.0;
            double J11 = 0.0;

            double invJ00 = 0.0;
            double invJ01 = 0.0;
            double invJ10 = 0.0;
            double invJ11 = 0.0;

            double invJT00 = 0.0;
            double invJT01 = 0.0;
            double invJT10 = 0.0;
            double invJT11 = 0.0;

            double t0 = 0.0;
            double ht = 0.0;
            double inv_ht = 0.0;
        };

        static Data make(const SpaceType& space, int cell_id)
        {
            const auto& mesh = space.mesh_ref();
            const auto& cell = mesh.cell(cell_id);

            const auto& v0 = mesh.spatial_vertices()[cell.spatial_vertex_ids[0]];
            const auto& v1 = mesh.spatial_vertices()[cell.spatial_vertex_ids[1]];
            const auto& v2 = mesh.spatial_vertices()[cell.spatial_vertex_ids[2]];

            const double t0 = mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
            const double t1 = mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];
            const double ht = t1 - t0;

            const double J00 = v1[0] - v0[0];
            const double J01 = v2[0] - v0[0];
            const double J10 = v1[1] - v0[1];
            const double J11 = v2[1] - v0[1];

            const double detJ = J00 * J11 - J01 * J10;
            if (std::abs(detJ) < 1e-15 || std::abs(ht) < 1e-15)
                throw std::runtime_error("CellGeometry<2+1D>::make: degenerate cell.");

            Data data;
            data.map.space.v0 = v0;
            data.map.space.v1 = v1;
            data.map.space.v2 = v2;
            data.map.time.a   = t0;
            data.map.time.b   = t1;

            data.x0 = v0[0];
            data.y0 = v0[1];

            data.J00 = J00;
            data.J01 = J01;
            data.J10 = J10;
            data.J11 = J11;

            const double inv_detJ = 1.0 / detJ;

            data.invJ00 =  J11 * inv_detJ;
            data.invJ01 = -J01 * inv_detJ;
            data.invJ10 = -J10 * inv_detJ;
            data.invJ11 =  J00 * inv_detJ;

            data.invJT00 =  J11 * inv_detJ;
            data.invJT01 = -J10 * inv_detJ;
            data.invJT10 = -J01 * inv_detJ;
            data.invJT11 =  J00 * inv_detJ;

            data.t0 = t0;
            data.ht = ht;
            data.inv_ht = 1.0 / ht;

            return data;
        }

        static RefPoint interior_reference_point() noexcept
        {
            return {1.0 / 3.0, 1.0 / 3.0, 0.5};
        }

        static SpaceTimePoint map_to_physical(const Data& data, const RefPoint& xi) noexcept
        {
            return data.map.map(xi[0], xi[1], xi[2]);
        }

        static RefPoint physical_to_reference(const Data& data, const SpaceTimePoint& x) noexcept
        {
            const double dx = x[0] - data.x0;
            const double dy = x[1] - data.y0;

            const double xi  = data.invJ00 * dx + data.invJ01 * dy;
            const double eta = data.invJ10 * dx + data.invJ11 * dy;
            const double tau = (x[2] - data.t0) * data.inv_ht;

            return RefPoint{xi, eta, tau};
        }

        static bool reference_point_inside(const RefPoint& xi, double tol = 1e-12) noexcept
        {
            return (-tol <= xi[0]) &&
                   (-tol <= xi[1]) &&
                   (xi[0] + xi[1] <= 1.0 + tol) &&
                   (-tol <= xi[2] && xi[2] <= 1.0 + tol);
        }

        static bool contains_physical_point(
            const Data& data,
            const SpaceTimePoint& x,
            double tol = 1e-12) noexcept
        {
            return reference_point_inside(physical_to_reference(data, x), tol);
        }

        static double jacobian_measure(const Data& data) noexcept
        {
            return data.map.jacobian_measure();
        }

        static SpatialGradient spatial_gradient(
            const Data& data,
            const RefGradient& grad_ref) noexcept
        {
            return {
                data.invJT00 * grad_ref[0] + data.invJT01 * grad_ref[1],
                data.invJT10 * grad_ref[0] + data.invJT11 * grad_ref[1]
            };
        }

        static double time_derivative(
            const Data& data,
            const RefGradient& grad_ref) noexcept
        {
            return grad_ref[2] * data.inv_ht;
        }

        static FullGradient full_gradient(
            const Data& data,
            const RefGradient& grad_ref) noexcept
        {
            return {
                data.invJT00 * grad_ref[0] + data.invJT01 * grad_ref[1],
                data.invJT10 * grad_ref[0] + data.invJT11 * grad_ref[1],
                grad_ref[2] * data.inv_ht
            };
        }

        static double spatial_grad_dot(
            const Data& data,
            const RefGradient& grad_i,
            const RefGradient& grad_j) noexcept
        {
            const auto gi = spatial_gradient(data, grad_i);
            const auto gj = spatial_gradient(data, grad_j);
            return gi[0] * gj[0] + gi[1] * gj[1];
        }
    };
}
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>

#include "linear_algebra/dense/constexpr_linalg.hpp"

#include "../polynomials/dubiner_basis.hpp"
#include "../polynomials/legendre.hpp"
#include "quadrature/cell_mappings.hpp"
#include "quadrature/gauss_legendre_1d.hpp"
#include "quadrature/reference_quadrature.hpp"
#include "quadrature/reference_triangle_duffy.hpp"

namespace finite_element::basis::functions
{
    enum class RaviartThomasDofKind
    {
        edge_normal_moment,
        interior_vector_moment
    };

    struct RaviartThomasTriangleDof
    {
        RaviartThomasDofKind kind = RaviartThomasDofKind::edge_normal_moment;
        int face                 = -1;
        int edge_moment          = -1;
        int component            = -1;
        int scalar_moment        = -1;
    };

    template<int P>
    class RaviartThomasTriangleBasis
    {
        static_assert(P >= 1 && P <= 10,
                      "RaviartThomasTriangleBasis currently supports 1 <= p <= 10.");

    public:
        static constexpr int degree                  = P;
        static constexpr int scalar_dim              = (P + 1) * (P + 2) / 2;
        static constexpr int interior_scalar_dim     = P * (P + 1) / 2;
        static constexpr int edge_dofs               = 3 * (P + 1);
        static constexpr int interior_dofs           = P * (P + 1);
        static constexpr int N                       = (P + 1) * (P + 3);
        static constexpr int dimension               = N;
        static constexpr int raw_homogeneous_offset  = 2 * scalar_dim;
        static constexpr std::size_t n_values = static_cast<std::size_t>(N);

        using Point       = std::array<double, 2>;
        using VectorValue = std::array<double, 2>;
        using Values      = std::array<VectorValue, n_values>;
        using Divergences = std::array<double, n_values>;

        [[nodiscard]] static constexpr std::size_t array_index(const int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        static_assert(edge_dofs + interior_dofs == N,
                      "RT moment count must match RT dimension.");
        static_assert(raw_homogeneous_offset + P + 1 == N,
                      "RT spanning basis count must match RT dimension.");

        [[nodiscard]] static constexpr RaviartThomasTriangleDof dof(
            const int dof_id)
        {
            if (dof_id < edge_dofs)
            {
                return RaviartThomasTriangleDof{
                    RaviartThomasDofKind::edge_normal_moment,
                    dof_id / (P + 1),
                    dof_id % (P + 1),
                    -1,
                    -1
                };
            }

            const int local = dof_id - edge_dofs;
            return RaviartThomasTriangleDof{
                RaviartThomasDofKind::interior_vector_moment,
                -1,
                -1,
                local / interior_scalar_dim,
                local % interior_scalar_dim
            };
        }

        [[nodiscard]] static Values eval_all(const Point& point)
        {
            return eval_all(point[0], point[1]);
        }

        [[nodiscard]] static Values eval_all(const double x, const double y)
        {
            const auto raw   = raw_eval(x, y);
            const auto& coef = coefficients();

            Values values{};
            for (int basis_id = 0; basis_id < N; ++basis_id)
            {
                for (int raw_id = 0; raw_id < N; ++raw_id)
                {
                    values[static_cast<std::size_t>(basis_id)][0] +=
                        coef[static_cast<std::size_t>(raw_id)]
                            [static_cast<std::size_t>(basis_id)] *
                        raw.values[static_cast<std::size_t>(raw_id)][0];
                    values[static_cast<std::size_t>(basis_id)][1] +=
                        coef[static_cast<std::size_t>(raw_id)]
                            [static_cast<std::size_t>(basis_id)] *
                        raw.values[static_cast<std::size_t>(raw_id)][1];
                }
            }

            return values;
        }

        [[nodiscard]] static VectorValue value(
            const int basis_id,
            const Point& point)
        {
            const auto values = eval_all(point);
            return values[static_cast<std::size_t>(basis_id)];
        }

        [[nodiscard]] static Divergences div_all(const Point& point)
        {
            return div_all(point[0], point[1]);
        }

        [[nodiscard]] static Divergences div_all(
            const double x,
            const double y)
        {
            const auto raw   = raw_eval(x, y);
            const auto& coef = coefficients();

            Divergences divergences{};
            for (int basis_id = 0; basis_id < N; ++basis_id)
            {
                for (int raw_id = 0; raw_id < N; ++raw_id)
                {
                    divergences[static_cast<std::size_t>(basis_id)] +=
                        coef[static_cast<std::size_t>(raw_id)]
                            [static_cast<std::size_t>(basis_id)] *
                        raw.divergences[static_cast<std::size_t>(raw_id)];
                }
            }

            return divergences;
        }

        [[nodiscard]] static double divergence(
            const int basis_id,
            const Point& point)
        {
            const auto divergences = div_all(point);
            return divergences[static_cast<std::size_t>(basis_id)];
        }

        [[nodiscard]] static double moment(
            const int dof_id,
            const int basis_id)
        {
            const auto descriptor = dof(dof_id);
            if (descriptor.kind == RaviartThomasDofKind::edge_normal_moment)
            {
                return edge_moment(
                    descriptor.face,
                    descriptor.edge_moment,
                    basis_id);
            }

            return interior_moment(
                descriptor.component,
                descriptor.scalar_moment,
                basis_id);
        }

        [[nodiscard]] static double edge_moment(
            const int face,
            const int edge_moment_id,
            const int basis_id)
        {
            constexpr auto rule =
                quadrature::gauss_legendre::
                    gauss_legendre_rule_1d<P + 1>;

            const auto normal = outward_unit_normal(face);
            const double jac  = edge_jacobian(face);

            double result = 0.0;
            for (int q = 0; q < rule.size(); ++q)
            {
                const double s = rule.point(q)[0];
                const auto point = edge_point(face, s);
                const auto v = value(basis_id, point);
                const double mu =
                    polynomials::LegendrePolynomials<P>::eval(
                        edge_moment_id,
                        2.0 * s - 1.0);

                result += rule.weight(q) *
                          (v[0] * normal[0] + v[1] * normal[1]) *
                          mu *
                          jac;
            }

            return result;
        }

        [[nodiscard]] static double interior_moment(
            const int component,
            const int scalar_moment_id,
            const int basis_id)
        {
            return quadrature::reference::integrate_reference_triangle_duffy<
                2 * P>(
                [&](const double x, const double y)
                {
                    const auto v = value(basis_id, Point{x, y});
                    const auto phi =
                        polynomials::DubinerBasis<P - 1>::eval_all(
                            x,
                            y);

                    return v[static_cast<std::size_t>(component)] *
                           phi[static_cast<std::size_t>(scalar_moment_id)];
                });
        }

    private:
        struct RawEvaluation
        {
            Values values{};
            Divergences divergences{};
        };

        using CoefficientMatrix = cdla::Mat<
            static_cast<std::size_t>(N),
            static_cast<std::size_t>(N)>;

        [[nodiscard]] static constexpr double pow_int(
            const double x,
            const int exponent)
        {
            double result = 1.0;
            for (int i = 0; i < exponent; ++i)
                result *= x;
            return result;
        }

        [[nodiscard]] static RawEvaluation raw_eval(
            const double x,
            const double y)
        {
            RawEvaluation raw{};

            const auto phi =
                polynomials::DubinerBasis<P>::eval_all(x, y);
            const auto grad =
                polynomials::DubinerBasis<P>::grad_all(x, y);

            for (int i = 0; i < scalar_dim; ++i)
            {
                raw.values[array_index(i)][0U] = phi[array_index(i)];
                raw.divergences[array_index(i)] = grad[array_index(i)][0U];

                const int y_id = scalar_dim + i;
                raw.values[array_index(y_id)][1U] = phi[array_index(i)];
                raw.divergences[array_index(y_id)] = grad[array_index(i)][1U];
            }

            for (int k = 0; k <= P; ++k)
            {
                const int alpha = P - k;
                const int beta  = k;
                const double h =
                    pow_int(x, alpha) * pow_int(y, beta);
                const int id = raw_homogeneous_offset + k;

                raw.values[static_cast<std::size_t>(id)][0] = x * h;
                raw.values[static_cast<std::size_t>(id)][1] = y * h;
                raw.divergences[static_cast<std::size_t>(id)] =
                    static_cast<double>(P + 2) * h;
            }

            return raw;
        }

        [[nodiscard]] static constexpr Point edge_point(
            const int face,
            const double s)
        {
            switch (face)
            {
                case 0:
                    return Point{s, 0.0};
                case 1:
                    return Point{1.0 - s, s};
                default:
                    return Point{0.0, 1.0 - s};
            }
        }

        [[nodiscard]] static VectorValue outward_unit_normal(const int face)
        {
            constexpr double inv_sqrt2 =
                0.707106781186547524400844362104849039;

            switch (face)
            {
                case 0:
                    return VectorValue{0.0, -1.0};
                case 1:
                    return VectorValue{inv_sqrt2, inv_sqrt2};
                default:
                    return VectorValue{-1.0, 0.0};
            }
        }

        [[nodiscard]] static double edge_jacobian(const int face)
        {
            return face == 1
                ? 1.41421356237309504880168872420969808
                : 1.0;
        }

        [[nodiscard]] static double raw_moment(
            const int dof_id,
            const int raw_id)
        {
            const auto descriptor = dof(dof_id);
            if (descriptor.kind == RaviartThomasDofKind::edge_normal_moment)
            {
                return raw_edge_moment(
                    descriptor.face,
                    descriptor.edge_moment,
                    raw_id);
            }

            return raw_interior_moment(
                descriptor.component,
                descriptor.scalar_moment,
                raw_id);
        }

        [[nodiscard]] static double raw_edge_moment(
            const int face,
            const int edge_moment_id,
            const int raw_id)
        {
            constexpr auto rule =
                quadrature::gauss_legendre::
                    gauss_legendre_rule_1d<P + 1>;

            const auto normal = outward_unit_normal(face);
            const double jac  = edge_jacobian(face);

            double result = 0.0;
            for (int q = 0; q < rule.size(); ++q)
            {
                const double s = rule.point(q)[0];
                const auto point = edge_point(face, s);
                const auto raw = raw_eval(point[0], point[1]);
                const auto& v = raw.values[static_cast<std::size_t>(raw_id)];
                const double mu =
                    polynomials::LegendrePolynomials<P>::eval(
                        edge_moment_id,
                        2.0 * s - 1.0);

                result += rule.weight(q) *
                          (v[0] * normal[0] + v[1] * normal[1]) *
                          mu *
                          jac;
            }

            return result;
        }

        [[nodiscard]] static double raw_interior_moment(
            const int component,
            const int scalar_moment_id,
            const int raw_id)
        {
            return quadrature::reference::integrate_reference_triangle_duffy<
                2 * P>(
                [&](const double x, const double y)
                {
                    const auto raw = raw_eval(x, y);
                    const auto phi =
                        polynomials::DubinerBasis<P - 1>::eval_all(
                            x,
                            y);

                    return raw.values[static_cast<std::size_t>(raw_id)]
                               [static_cast<std::size_t>(component)] *
                           phi[static_cast<std::size_t>(scalar_moment_id)];
                });
        }

        [[nodiscard]] static CoefficientMatrix build_moment_matrix()
        {
            CoefficientMatrix matrix{};
            for (int dof_id = 0; dof_id < N; ++dof_id)
            {
                for (int raw_id = 0; raw_id < N; ++raw_id)
                {
                    matrix[static_cast<std::size_t>(dof_id)]
                          [static_cast<std::size_t>(raw_id)] =
                        raw_moment(dof_id, raw_id);
                }
            }

            return matrix;
        }

        [[nodiscard]] static CoefficientMatrix build_coefficients()
        {
            const auto moment_matrix = build_moment_matrix();

            const auto qr = cdla::qr_factorize(moment_matrix);
            CoefficientMatrix inverse{};
            if (!cdla::try_qr_inverse(qr, inverse, 1.0e-12))
                throw std::runtime_error(
                    "Reference Raviart-Thomas moment matrix is singular.");

            return inverse;
        }

        [[nodiscard]] static const CoefficientMatrix& coefficients()
        {
            static const CoefficientMatrix c = build_coefficients();
            return c;
        }
    };

    template<int P>
    class RaviartThomasTrianglePiolaBasis
    {
        static_assert(P >= 1 && P <= 10,
                      "RaviartThomasTrianglePiolaBasis currently supports 1 <= p <= 10.");

    public:
        using ReferenceBasis = RaviartThomasTriangleBasis<P>;

        static constexpr int degree    = P;
        static constexpr int N         = ReferenceBasis::N;
        static constexpr int dimension = N;

        using Point       = typename ReferenceBasis::Point;
        using VectorValue = typename ReferenceBasis::VectorValue;
        using Values      = typename ReferenceBasis::Values;
        using Divergences = typename ReferenceBasis::Divergences;

        struct AffineMap
        {
            Point v0{};

            double J00 = 0.0;
            double J01 = 0.0;
            double J10 = 0.0;
            double J11 = 0.0;

            double invJ00 = 0.0;
            double invJ01 = 0.0;
            double invJ10 = 0.0;
            double invJ11 = 0.0;

            double detJ = 0.0;
        };

        [[nodiscard]] static AffineMap make_affine_map(
            const Point& v0,
            const Point& v1,
            const Point& v2)
        {
            AffineMap map{};
            map.v0 = v0;
            map.J00 = v1[0] - v0[0];
            map.J01 = v2[0] - v0[0];
            map.J10 = v1[1] - v0[1];
            map.J11 = v2[1] - v0[1];

            finish_affine_map(map);
            return map;
        }

        [[nodiscard]] static AffineMap make_affine_map(
            const quadrature::map::TriangleMap2D& triangle_map)
        {
            return make_affine_map(
                triangle_map.v0,
                triangle_map.v1,
                triangle_map.v2);
        }

        template<class CellGeometryData>
        [[nodiscard]] static AffineMap make_affine_map_from_cell_geometry(
            const CellGeometryData& data)
        {
            AffineMap map{};
            map.v0 = Point{data.x0, data.y0};
            map.J00 = data.J00;
            map.J01 = data.J01;
            map.J10 = data.J10;
            map.J11 = data.J11;

            finish_affine_map(map);
            return map;
        }

        [[nodiscard]] static double det_jacobian(
            const AffineMap& map) noexcept
        {
            return map.detJ;
        }

        [[nodiscard]] static double jacobian_measure(
            const AffineMap& map) noexcept
        {
            return std::abs(map.detJ);
        }

        [[nodiscard]] static Point map_to_physical(
            const AffineMap& map,
            const Point& reference_point) noexcept
        {
            return Point{
                map.v0[0] +
                    map.J00 * reference_point[0] +
                    map.J01 * reference_point[1],
                map.v0[1] +
                    map.J10 * reference_point[0] +
                    map.J11 * reference_point[1]
            };
        }

        [[nodiscard]] static Point physical_to_reference(
            const AffineMap& map,
            const Point& physical_point) noexcept
        {
            const double dx = physical_point[0] - map.v0[0];
            const double dy = physical_point[1] - map.v0[1];

            return Point{
                map.invJ00 * dx + map.invJ01 * dy,
                map.invJ10 * dx + map.invJ11 * dy
            };
        }

        [[nodiscard]] static VectorValue push_forward_value(
            const AffineMap& map,
            const VectorValue& reference_value) noexcept
        {
            return VectorValue{
                (map.J00 * reference_value[0] +
                 map.J01 * reference_value[1]) / map.detJ,
                (map.J10 * reference_value[0] +
                 map.J11 * reference_value[1]) / map.detJ
            };
        }

        [[nodiscard]] static double push_forward_divergence(
            const AffineMap& map,
            const double reference_divergence) noexcept
        {
            return reference_divergence / map.detJ;
        }

        [[nodiscard]] static Values eval_all(
            const AffineMap& map,
            const Point& reference_point)
        {
            const auto reference_values =
                ReferenceBasis::eval_all(reference_point);

            Values physical_values{};
            for (int basis_id = 0; basis_id < N; ++basis_id)
            {
                physical_values[static_cast<std::size_t>(basis_id)] =
                    push_forward_value(
                        map,
                        reference_values[static_cast<std::size_t>(basis_id)]);
            }

            return physical_values;
        }

        [[nodiscard]] static VectorValue value(
            const AffineMap& map,
            const int basis_id,
            const Point& reference_point)
        {
            return eval_all(map, reference_point)
                [static_cast<std::size_t>(basis_id)];
        }

        [[nodiscard]] static Values eval_all_at_physical_point(
            const AffineMap& map,
            const Point& physical_point)
        {
            return eval_all(map, physical_to_reference(map, physical_point));
        }

        [[nodiscard]] static VectorValue value_at_physical_point(
            const AffineMap& map,
            const int basis_id,
            const Point& physical_point)
        {
            return eval_all_at_physical_point(map, physical_point)
                [static_cast<std::size_t>(basis_id)];
        }

        [[nodiscard]] static Divergences div_all(
            const AffineMap& map,
            const Point& reference_point)
        {
            const auto reference_divergences =
                ReferenceBasis::div_all(reference_point);

            Divergences physical_divergences{};
            for (int basis_id = 0; basis_id < N; ++basis_id)
            {
                physical_divergences[static_cast<std::size_t>(basis_id)] =
                    push_forward_divergence(
                        map,
                        reference_divergences[
                            static_cast<std::size_t>(basis_id)]);
            }

            return physical_divergences;
        }

        [[nodiscard]] static double divergence(
            const AffineMap& map,
            const int basis_id,
            const Point& reference_point)
        {
            return div_all(map, reference_point)
                [static_cast<std::size_t>(basis_id)];
        }

        [[nodiscard]] static Divergences div_all_at_physical_point(
            const AffineMap& map,
            const Point& physical_point)
        {
            return div_all(map, physical_to_reference(map, physical_point));
        }

        [[nodiscard]] static double divergence_at_physical_point(
            const AffineMap& map,
            const int basis_id,
            const Point& physical_point)
        {
            return div_all_at_physical_point(map, physical_point)
                [static_cast<std::size_t>(basis_id)];
        }

    private:
        static void finish_affine_map(AffineMap& map)
        {
            map.detJ = map.J00 * map.J11 - map.J01 * map.J10;

            if (std::abs(map.detJ) < 1.0e-15)
                throw std::runtime_error(
                    "RaviartThomasTrianglePiolaBasis: degenerate affine triangle.");

            const double inv_det = 1.0 / map.detJ;
            map.invJ00 =  map.J11 * inv_det;
            map.invJ01 = -map.J01 * inv_det;
            map.invJ10 = -map.J10 * inv_det;
            map.invJ11 =  map.J00 * inv_det;
        }
    };
}

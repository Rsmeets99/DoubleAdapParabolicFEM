#pragma once

#include <array>
#include <cstddef>

#include "linear_algebra/dense/constexpr_linalg.hpp"
#include "../basis/polynomials/jacobi.hpp"
#include "../basis/polynomials/legendre.hpp"

namespace finite_element::nodes
{
    template<int P>
    struct WarpBlendTriangleNodes
    {
        static_assert(P >= 1, "WarpBlendTriangleNodes requires P >= 1.");

        static constexpr int N = (P + 1) * (P + 2) / 2;

        using Point    = std::array<double, 2>;
        using Bary     = std::array<int, 3>;   // {a,b,c}, a+b+c=P
        using FaceList = std::array<int, 2>;

        struct NodeMeta
        {
            FaceList spatial_faces{-1, -1};
            int num_spatial_faces = 0;
            int vertex = -1; // 0,1,2 or -1
        };

        struct GeneratedData
        {
            std::array<Point, N> points{};
            std::array<Bary, N> barycentric_tuples{};
        };

        static constexpr double sqrt3 = 1.7320508075688772935274463415058723669;

        static constexpr std::array<double, P + 1> equidistant_edge_nodes()
        {
            std::array<double, P + 1> r{};
            for (int i = 0; i <= P; ++i)
                r[i] = -1.0 + 2.0 * static_cast<double>(i) / static_cast<double>(P);
            return r;
        }

        static constexpr std::array<double, P + 1> lobatto_edge_nodes()
        {
            return finite_element::basis::polynomials::LegendrePolynomials<P>::template gauss_lobatto_nodes<P + 1>();
        }

        static constexpr double alpha_opt()
        {
            if constexpr (P == 1)  return 0.0;
            if constexpr (P == 2)  return 0.0;
            if constexpr (P == 3)  return 1.4152;
            if constexpr (P == 4)  return 0.1001;
            if constexpr (P == 5)  return 0.2751;
            if constexpr (P == 6)  return 0.9800;
            if constexpr (P == 7)  return 1.0999;
            if constexpr (P == 8)  return 1.2832;
            if constexpr (P == 9)  return 1.3648;
            if constexpr (P == 10) return 1.4773;
            if constexpr (P == 11) return 1.4959;
            if constexpr (P == 12) return 1.5743;
            if constexpr (P == 13) return 1.5770;
            if constexpr (P == 14) return 1.6223;
            if constexpr (P == 15) return 1.6258;
            return 5.0 / 3.0;
        }

        static constexpr cdla::Vec<P + 1> build_warp_coefficients()
        {
            cdla::Vec<P + 1> rhs{};
            cdla::Mat<P + 1, P + 1> V{};

            const auto req = equidistant_edge_nodes();
            const auto rgl = lobatto_edge_nodes();

            for (int i = 0; i <= P; ++i)
            {
                rhs[i] = rgl[i] - req[i];

                typename finite_element::basis::polynomials::JacobiPolynomials<P>::Array J{};
                finite_element::basis::polynomials::JacobiPolynomials<P>::family(P, 0.0, 0.0, req[i], J);

                for (int n = 0; n <= P; ++n)
                    V[i][n] = J[n];
            }

            const auto qr = cdla::qr_factorize(V);
            return cdla::qr_solve(qr, rhs);
        }

        static constexpr cdla::Vec<P + 1> warp_coeff = build_warp_coefficients();

        static constexpr double eval_warp_poly(double r)
        {
            typename finite_element::basis::polynomials::JacobiPolynomials<P>::Array Jr{};
            finite_element::basis::polynomials::JacobiPolynomials<P>::family(P, 0.0, 0.0, r, Jr);

            double w = 0.0;
            for (int n = 0; n <= P; ++n)
                w += warp_coeff[n] * Jr[n];

            return w;
        }

        static constexpr double eval_warp_factor(double r)
        {
            const double denom = 1.0 - r * r;
            if (cdla::cabs(denom) < 1e-14)
                return 0.0;

            return eval_warp_poly(r) / denom;
        }

        static constexpr Point equilateral_to_reference(double xeq, double yeq)
        {
            // Equilateral vertices:
            // (-1, -1/sqrt3) -> (0,0)
            // ( 1, -1/sqrt3) -> (1,0)
            // ( 0,  2/sqrt3) -> (0,1)
            const double xr = 0.5 * (xeq + 1.0) - 0.5 * (yeq + 1.0 / sqrt3) / sqrt3;
            const double yr = (yeq + 1.0 / sqrt3) / sqrt3;
            return {xr, yr};
        }

        static constexpr GeneratedData generate_data()
        {
            GeneratedData data{};

            constexpr double alpha = alpha_opt();

            int id = 0;
            for (int c = 0; c <= P; ++c)
            {
                for (int b = 0; b <= P - c; ++b)
                {
                    const int a = P - b - c;

                    const double L1 = static_cast<double>(a) / static_cast<double>(P);
                    const double L2 = static_cast<double>(b) / static_cast<double>(P);
                    const double L3 = static_cast<double>(c) / static_cast<double>(P);

                    // Barycentric -> equilateral
                    double xeq = -L1 + L2;
                    double yeq = (-L1 - L2 + 2.0 * L3) / sqrt3;

                    // Edge blends
                    const double blend1 = 4.0 * L2 * L3; // opposite L1
                    const double blend2 = 4.0 * L3 * L1; // opposite L2
                    const double blend3 = 4.0 * L1 * L2; // opposite L3

                    const double warp1 =
                        blend1 * eval_warp_factor(L3 - L2) * (1.0 + alpha * alpha * L1 * L1);
                    const double warp2 =
                        blend2 * eval_warp_factor(L1 - L3) * (1.0 + alpha * alpha * L2 * L2);
                    const double warp3 =
                        blend3 * eval_warp_factor(L2 - L1) * (1.0 + alpha * alpha * L3 * L3);

                    // Correct tangent directions for this equilateral orientation:
                    // edge opposite L1: (-1/2, +sqrt3/2)
                    // edge opposite L2: (-1/2, -sqrt3/2)
                    // edge opposite L3: ( 1  , 0)
                    xeq += -0.5 * warp1 - 0.5 * warp2 + warp3;
                    yeq += (sqrt3 / 2.0) * warp1 - (sqrt3 / 2.0) * warp2;

                    data.points[id] = equilateral_to_reference(xeq, yeq);
                    data.barycentric_tuples[id] = {a, b, c};
                    ++id;
                }
            }

            return data;
        }

        static constexpr GeneratedData generated = generate_data();
        static constexpr auto points = generated.points;
        static constexpr auto barycentric_tuples = generated.barycentric_tuples;

        static constexpr NodeMeta make_node_meta(const Bary& n)
        {
            NodeMeta m{};

            // face 0: c == 0  => y = 0
            if (n[2] == 0)
                m.spatial_faces[m.num_spatial_faces++] = 0;

            // face 1: a == 0  => x + y = 1
            if (n[0] == 0)
                m.spatial_faces[m.num_spatial_faces++] = 1;

            // face 2: b == 0  => x = 0
            if (n[1] == 0)
                m.spatial_faces[m.num_spatial_faces++] = 2;

            if (n[0] == P && n[1] == 0 && n[2] == 0) m.vertex = 0;
            if (n[0] == 0 && n[1] == P && n[2] == 0) m.vertex = 1;
            if (n[0] == 0 && n[1] == 0 && n[2] == P) m.vertex = 2;

            return m;
        }

        static constexpr std::array<NodeMeta, N> generate_node_meta()
        {
            std::array<NodeMeta, N> meta{};
            for (int i = 0; i < N; ++i)
                meta[i] = make_node_meta(barycentric_tuples[i]);
            return meta;
        }

        static constexpr auto node_meta = generate_node_meta();

        static constexpr std::array<std::array<int, P + 1>, 3> generate_face_nodes()
        {
            std::array<std::array<int, P + 1>, 3> generated_face_nodes{};

            for (auto& face : generated_face_nodes)
                for (auto& x : face)
                    x = -1;

            for (int i = 0; i < N; ++i)
            {
                const Bary n = barycentric_tuples[i];
                const int a = n[0];
                const int b = n[1];
                const int c = n[2];

                // face 0: (0,0) -> (1,0), c == 0, order by b increasing
                if (c == 0)
                    generated_face_nodes[0][b] = i;

                // face 1: (1,0) -> (0,1), a == 0, order by c increasing
                if (a == 0)
                    generated_face_nodes[1][c] = i;

                // face 2: (0,1) -> (0,0), b == 0, order by a increasing
                if (b == 0)
                    generated_face_nodes[2][a] = i;
            }

            return generated_face_nodes;
        }

        static constexpr auto face_nodes = generate_face_nodes();

        static constexpr std::array<std::array<int, 3>, N> generate_face_ordinal()
        {
            std::array<std::array<int, 3>, N> ord{};

            for (auto& row : ord)
                row = {-1, -1, -1};

            for (int face = 0; face < 3; ++face)
            {
                for (int k = 0; k <= P; ++k)
                {
                    const int node = face_nodes[face][k];
                    ord[node][face] = k;
                }
            }

            return ord;
        }

        static constexpr auto face_ordinal = generate_face_ordinal();

        static constexpr std::array<int, 3> generate_vertex_nodes()
        {
            std::array<int, 3> verts{-1, -1, -1};

            for (int i = 0; i < N; ++i)
            {
                const int v = node_meta[i].vertex;
                if (v >= 0)
                    verts[v] = i;
            }

            return verts;
        }

        static constexpr auto vertex_nodes = generate_vertex_nodes();

        static constexpr bool node_on_face(int node_id, int face)
        {
            return face_ordinal[node_id][face] >= 0;
        }

        static constexpr bool node_is_vertex(int node_id)
        {
            return node_meta[node_id].vertex >= 0;
        }

        static constexpr int vertex_of_node(int node_id)
        {
            return node_meta[node_id].vertex;
        }

        // -------------------------------------------------------------------------
        // Validation
        // -------------------------------------------------------------------------

        static constexpr double validation_tol()
        {
            return 1e-12;
        }

        static constexpr bool approx_zero(double x, double tol = validation_tol())
        {
            return cdla::cabs(x) <= tol;
        }

        static constexpr bool approx_equal(double a, double b, double tol = validation_tol())
        {
            return cdla::cabs(a - b) <= tol;
        }

        static constexpr bool point_inside_reference_triangle(const Point& p,
                                                            double tol = validation_tol())
        {
            const double x = p[0];
            const double y = p[1];

            return x >= -tol &&
                y >= -tol &&
                x <= 1.0 + tol &&
                y <= 1.0 + tol &&
                (x + y) <= 1.0 + tol;
        }

        static constexpr bool point_on_face(const Point& p, int face,
                                            double tol = validation_tol())
        {
            const double x = p[0];
            const double y = p[1];

            if (face == 0) return approx_zero(y, tol);            // y = 0
            if (face == 1) return approx_equal(x + y, 1.0, tol);  // x + y = 1
            if (face == 2) return approx_zero(x, tol);            // x = 0

            return false;
        }

        static constexpr bool point_at_vertex(const Point& p, int vertex,
                                            double tol = validation_tol())
        {
            const double x = p[0];
            const double y = p[1];

            if (vertex == 0) return approx_equal(x, 0.0, tol) && approx_equal(y, 0.0, tol);
            if (vertex == 1) return approx_equal(x, 1.0, tol) && approx_equal(y, 0.0, tol);
            if (vertex == 2) return approx_equal(x, 0.0, tol) && approx_equal(y, 1.0, tol);

            return false;
        }

        static constexpr bool validate_inside()
        {
            for (int i = 0; i < N; ++i)
            {
                if (!point_inside_reference_triangle(points[i]))
                    return false;
            }
            return true;
        }

        static constexpr bool validate_faces()
        {
            for (int face = 0; face < 3; ++face)
            {
                for (int k = 0; k <= P; ++k)
                {
                    const int node = face_nodes[face][k];
                    if (node < 0 || node >= N)
                        return false;

                    if (!point_on_face(points[node], face))
                        return false;
                }
            }
            return true;
        }

        static constexpr bool validate_vertices()
        {
            for (int v = 0; v < 3; ++v)
            {
                const int node = vertex_nodes[v];
                if (node < 0 || node >= N)
                    return false;

                if (!point_at_vertex(points[node], v))
                    return false;
            }
            return true;
        }

        static constexpr bool validate_face_membership_consistency()
        {
            for (int i = 0; i < N; ++i)
            {
                for (int face = 0; face < 3; ++face)
                {
                    const bool by_meta =
                        (barycentric_tuples[i][2] == 0 && face == 0) ||
                        (barycentric_tuples[i][0] == 0 && face == 1) ||
                        (barycentric_tuples[i][1] == 0 && face == 2);

                    const bool by_ord = node_on_face(i, face);

                    if (by_meta != by_ord)
                        return false;
                }
            }
            return true;
        }

        static constexpr bool validate_all()
        {
            return validate_inside() &&
                validate_faces() &&
                validate_vertices() &&
                validate_face_membership_consistency();
        }

        static_assert(validate_all(),
                    "WarpBlendTriangleNodes validation failed: some nodes are outside "
                    "the reference triangle or not on their assigned faces/vertices.");
    };

    template<int P>
    constexpr cdla::Vec<P + 1> WarpBlendTriangleNodes<P>::warp_coeff;
}

#pragma once

#include <array>
#include <cstddef>

#include "linear_algebra/dense/constexpr_linalg.hpp" // for cdla
#include "dubiner_basis.hpp"

namespace finite_element::basis::polynomials
{
    template<int P, typename Nodes>
    struct TriangularLagrangeBasis
    {
        static_assert(P >= 0, "TriangularLagrangeBasis requires P >= 0");

        static constexpr int N = (P + 1) * (P + 2) / 2;
        static constexpr std::size_t n_values = static_cast<std::size_t>(N);

        using Point  = std::array<double, 2>;
        using Values = std::array<double, n_values>;
        using Grads  = std::array<std::array<double, 2>, n_values>;

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        static_assert(Nodes::N == N, "Node policy has wrong number of triangle nodes");

        // -------------------------------------------------------------------------
        // Build Vandermonde V[j][m] = psi_m( x_j, y_j )
        // where psi_m is the Dubiner modal basis and (x_j,y_j) are nodal points.
        // -------------------------------------------------------------------------
        static constexpr cdla::Mat<N, N> build_vandermonde()
        {
            cdla::Mat<N, N> V{};

            for (int j = 0; j < N; ++j)
            {
                const double x = Nodes::points[index(j)][0U];
                const double y = Nodes::points[index(j)][1U];
                const auto psi = DubinerBasis<P>::eval_all(x, y);

                for (int m = 0; m < N; ++m)
                    V[index(j)][index(m)] = psi[index(m)];
            }

            return V;
        }

        // -------------------------------------------------------------------------
        // We store nodal basis coefficients row-wise:
        //
        //   l_i(x,y) = sum_m coeff_by_node[i][m] * psi_m(x,y)
        //
        // If V[j][m] = psi_m(node_j), then nodal basis coefficients come from
        // transpose(V^{-1}).
        // -------------------------------------------------------------------------
        static constexpr cdla::Mat<N, N> build_coeff_by_node()
        {
            const auto V  = build_vandermonde();
            const auto qr = cdla::qr_factorize(V);
            const auto C  = cdla::qr_inverse(qr);
            return cdla::transpose(C);
        }

        static constexpr cdla::Mat<N, N> coeff_by_node = build_coeff_by_node();

        // -------------------------------------------------------------------------
        // Evaluate all nodal basis functions
        // -------------------------------------------------------------------------
        static constexpr Values eval_all(double x, double y)
        {
            const auto psi = DubinerBasis<P>::eval_all(x, y);

            Values out{};
            for (int i = 0; i < N; ++i)
            {
                double s = 0.0;
                for (int m = 0; m < N; ++m)
                    s += coeff_by_node[index(i)][index(m)] * psi[index(m)];
                out[index(i)] = s;
            }
            return out;
        }

        static constexpr Values eval_all(Point pt)
        {
            return eval_all(pt[0], pt[1]);
        }

        // -------------------------------------------------------------------------
        // Evaluate gradients of all nodal basis functions
        // -------------------------------------------------------------------------
        static constexpr Grads grad_all(double x, double y)
        {
            const auto dpsi = DubinerBasis<P>::grad_all(x, y);

            Grads out{};
            for (int i = 0; i < N; ++i)
            {
                double sx = 0.0;
                double sy = 0.0;

                for (int m = 0; m < N; ++m)
                {
                    sx += coeff_by_node[index(i)][index(m)] * dpsi[index(m)][0U];
                    sy += coeff_by_node[index(i)][index(m)] * dpsi[index(m)][1U];
                }

                out[index(i)][0U] = sx;
                out[index(i)][1U] = sy;
            }
            return out;
        }

        static constexpr Grads grad_all(Point pt)
        {
            return grad_all(pt[0], pt[1]);
        }

        // -------------------------------------------------------------------------
        // Pretabulation helpers
        // -------------------------------------------------------------------------
        template<std::size_t M>
        static constexpr std::array<Values, M>
        tabulate_values(const std::array<Point, M>& pts)
        {
            std::array<Values, M> table{};
            for (std::size_t k = 0; k < M; ++k)
                table[k] = eval_all(pts[k]);
            return table;
        }

        template<std::size_t M>
        static constexpr std::array<Grads, M>
        tabulate_gradients(const std::array<Point, M>& pts)
        {
            std::array<Grads, M> table{};
            for (std::size_t k = 0; k < M; ++k)
                table[k] = grad_all(pts[k]);
            return table;
        }

        // -------------------------------------------------------------------------
        // Forwarded node-topology helpers from the node policy
        // -------------------------------------------------------------------------

        // face 0: (0,0) -> (1,0)
        // face 1: (1,0) -> (0,1)
        // face 2: (0,1) -> (0,0)
        static constexpr bool node_on_face(int node_id, int face)
        {
            return Nodes::node_on_face(node_id, face);
        }

        static constexpr bool node_is_vertex(int node_id)
        {
            return Nodes::node_is_vertex(node_id);
        }

        // Returns:
        //   0 for (0,0)
        //   1 for (1,0)
        //   2 for (0,1)
        //  -1 if not a vertex
        static constexpr int vertex_of_node(int node_id)
        {
            return Nodes::vertex_of_node(node_id);
        }

        // Oriented ordinal of node on a face, or -1 if not on that face.
        static constexpr int face_ordinal(int node_id, int face)
        {
            return Nodes::face_ordinal[index(node_id)][index(face)];
        }

        // Node id of the k-th node on the given oriented face.
        static constexpr int face_node(int face, int k)
        {
            return Nodes::face_nodes[index(face)][index(k)];
        }

        // Node id of vertex 0,1,2.
        static constexpr int vertex_node(int vertex)
        {
            return Nodes::vertex_nodes[vertex];
        }
    };
}

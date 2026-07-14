#pragma once

#include <stdexcept>
#include <utility>

#include "../assembly/compose_block_system.hpp"
#include "linear_system.hpp"

namespace la::saddle
{
    template<class Backend>
    struct SaddlePointBlocks
    {
        using SparseMatrix = typename Backend::SparseMatrix;
        using Vector = typename Backend::Vector;

        // Stored block convention:
        //
        //   [ A  B^T ] [lambda] = [f]
        //   [ B   C  ] [u     ]   [g]
        //
        // B is stored in its lower-left u-lambda orientation.  The full
        // system composer inserts B.transpose() in the upper-right block so
        // the algebraic matrix is exactly symmetric when A and C are symmetric.
        //
        // For the parabolic main system, C is already the signed bottom-right
        // block assembled from -gamma_0^T gamma_0.  A positive Schur-complement
        // expression should therefore use C_pos = -C.
        SparseMatrix A;   // lambda-lambda
        SparseMatrix B;   // u-lambda, stored lower-left block
        SparseMatrix C;   // signed u-u block

        Vector f;         // lambda block rhs
        Vector g;         // u block rhs

        int n_lambda = 0;
        int n_u = 0;

        // Convention-preserving accessors for code that needs block context
        // without rediscovering it from the monolithic matrix.
        [[nodiscard]] const SparseMatrix& top_left_A() const noexcept
        {
            return A;
        }

        [[nodiscard]] const SparseMatrix& lower_left_B() const noexcept
        {
            return B;
        }

        [[nodiscard]] const SparseMatrix& signed_bottom_right_C() const noexcept
        {
            return C;
        }

        [[nodiscard]] const Vector& lambda_rhs() const noexcept
        {
            return f;
        }

        [[nodiscard]] const Vector& u_rhs() const noexcept
        {
            return g;
        }

        void validate() const
        {
            if (A.rows() != A.cols()) {
                throw std::runtime_error("SaddlePointBlocks: A must be square.");
            }
            if (C.rows() != C.cols()) {
                throw std::runtime_error("SaddlePointBlocks: C must be square.");
            }
            if (B.cols() != A.rows()) {
                throw std::runtime_error("SaddlePointBlocks: B.cols() must equal A.rows().");
            }
            if (B.rows() != C.rows()) {
                throw std::runtime_error("SaddlePointBlocks: B.rows() must equal C.rows().");
            }
            if (f.size() != A.rows()) {
                throw std::runtime_error("SaddlePointBlocks: f size mismatch.");
            }
            if (g.size() != C.rows()) {
                throw std::runtime_error("SaddlePointBlocks: g size mismatch.");
            }
            if (n_lambda != A.rows()) {
                throw std::runtime_error("SaddlePointBlocks: n_lambda mismatch.");
            }
            if (n_u != C.rows()) {
                throw std::runtime_error("SaddlePointBlocks: n_u mismatch.");
            }
        }

        void clear()
        {
            A = SparseMatrix{};
            B = SparseMatrix{};
            C = SparseMatrix{};
            f = Vector{};
            g = Vector{};
            n_lambda = 0;
            n_u = 0;
        }

        [[nodiscard]] la::linear::LinearSystem<Backend> make_full_system(
            bool memory_bounded_composition = false) const &
        {
            validate();

            la::linear::LinearSystem<Backend> out;
            out.matrix =
                memory_bounded_composition
                    ? la::block::compose_saddle_point_matrix_direct<Backend>(A, B, C)
                    : la::block::compose_saddle_point_matrix<Backend>(A, B, C);
            out.rhs = la::block::compose_2_block_vector<Backend>(f, g);
            out.solution = typename Backend::Vector(f.size() + g.size());
            out.solution.set_zero();
            return out;
        }

        [[nodiscard]] la::linear::LinearSystem<Backend> make_full_system(
            bool memory_bounded_composition = false) &&
        {
            validate();

            la::linear::LinearSystem<Backend> out;
            out.matrix =
                memory_bounded_composition
                    ? la::block::compose_saddle_point_matrix_direct<Backend>(A, B, C)
                    : la::block::compose_saddle_point_matrix<Backend>(A, B, C);
            out.rhs = la::block::compose_2_block_vector<Backend>(f, g);
            out.solution = typename Backend::Vector(f.size() + g.size());
            out.solution.set_zero();

            clear();
            return out;
        }
    };

    template<class Backend>
    SaddlePointBlocks<Backend> make_saddle_point_blocks(
        typename Backend::SparseMatrix A,
        typename Backend::SparseMatrix B,
        typename Backend::SparseMatrix C,
        typename Backend::Vector f,
        typename Backend::Vector g)
    {
        const int n_lambda = A.rows();
        const int n_u = C.rows();

        SaddlePointBlocks<Backend> out;
        out.A = std::move(A);
        out.B = std::move(B);
        out.C = std::move(C);
        out.f = std::move(f);
        out.g = std::move(g);
        out.n_lambda = n_lambda;
        out.n_u = n_u;
        out.validate();
        return out;
    }

    template<class Backend>
    struct SaddlePointSolution
    {
        using Vector = typename Backend::Vector;

        Vector lambda;
        Vector u;
    };

    template<class Backend>
    SaddlePointSolution<Backend> split_saddle_point_solution(
        const typename Backend::Vector& full_solution,
        int n_lambda,
        int n_u)
    {
        if (full_solution.size() != n_lambda + n_u) {
            throw std::runtime_error("split_saddle_point_solution: size mismatch.");
        }

        SaddlePointSolution<Backend> out;
        out.lambda.resize(n_lambda);
        out.u.resize(n_u);

        for (int i = 0; i < n_lambda; ++i) {
            out.lambda[i] = full_solution[i];
        }
        for (int i = 0; i < n_u; ++i) {
            out.u[i] = full_solution[n_lambda + i];
        }

        return out;
    }
}

#pragma once

#include <stdexcept>

namespace la::block
{
    template<class SparseMatrix>
    std::size_t estimate_nnz(const SparseMatrix& A)
    {
        std::size_t count = 0;
        A.for_each_nonzero(
            [&](int, int, double)
            {
                ++count;
            });
        return count;
    }

    template<class SparseMatrix, class SparseBuilder>
    void add_matrix_block(
        SparseBuilder& builder,
        const SparseMatrix& A,
        int row_offset,
        int col_offset)
    {
        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                builder.add(row_offset + i, col_offset + j, value);
            });
    }

    template<class SparseMatrix, class SparseBuilder>
    void add_matrix_block_transpose(
        SparseBuilder& builder,
        const SparseMatrix& A,
        int row_offset,
        int col_offset)
    {
        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                builder.add(row_offset + j, col_offset + i, value);
            });
    }

    template<class SparseMatrix>
    void add_matrix_block_direct(
        SparseMatrix& out,
        const SparseMatrix& A,
        int row_offset,
        int col_offset)
    {
        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                out.add(row_offset + i, col_offset + j, value);
            });
    }

    template<class SparseMatrix>
    void add_matrix_block_transpose_direct(
        SparseMatrix& out,
        const SparseMatrix& A,
        int row_offset,
        int col_offset)
    {
        A.for_each_nonzero(
            [&](int i, int j, double value)
            {
                out.add(row_offset + j, col_offset + i, value);
            });
    }

    template<class Backend>
    typename Backend::SparseMatrix compose_2x2_block_matrix(
        const typename Backend::SparseMatrix& A11,
        const typename Backend::SparseMatrix& A12,
        const typename Backend::SparseMatrix& A21,
        const typename Backend::SparseMatrix& A22)
    {
        using SparseBuilder = typename Backend::SparseBuilder;
        using SparseMatrix  = typename Backend::SparseMatrix;

        if (A11.rows() != A12.rows()) {
            throw std::runtime_error("compose_2x2_block_matrix: top block row mismatch.");
        }
        if (A21.rows() != A22.rows()) {
            throw std::runtime_error("compose_2x2_block_matrix: bottom block row mismatch.");
        }
        if (A11.cols() != A21.cols()) {
            throw std::runtime_error("compose_2x2_block_matrix: left block col mismatch.");
        }
        if (A12.cols() != A22.cols()) {
            throw std::runtime_error("compose_2x2_block_matrix: right block col mismatch.");
        }

        const int n_top = A11.rows();
        const int n_bottom = A21.rows();
        const int n_left = A11.cols();
        const int n_right = A12.cols();

        SparseBuilder builder;

        add_matrix_block(builder, A11, 0, 0);
        add_matrix_block(builder, A12, 0, n_left);
        add_matrix_block(builder, A21, n_top, 0);
        add_matrix_block(builder, A22, n_top, n_left);

        SparseMatrix out(n_top + n_bottom, n_left + n_right);
        out.set_from_builder(builder);
        return out;
    }

    template<class Backend>
    typename Backend::SparseMatrix compose_saddle_point_matrix(
        const typename Backend::SparseMatrix& A,
        const typename Backend::SparseMatrix& B,
        const typename Backend::SparseMatrix& C)
    {
        // Compose the stored saddle blocks using the convention
        //
        //   [ A  B^T ]
        //   [ B   C  ].
        //
        // B is supplied in lower-left u-lambda orientation.  C is the signed
        // lower-right block; for the parabolic main system it is already
        // -gamma_0^T gamma_0, not the positive trace mass matrix.
        using SparseBuilder = typename Backend::SparseBuilder;
        using SparseMatrix  = typename Backend::SparseMatrix;

        if (A.rows() != A.cols()) {
            throw std::runtime_error("compose_saddle_point_matrix: A must be square.");
        }
        if (C.rows() != C.cols()) {
            throw std::runtime_error("compose_saddle_point_matrix: C must be square.");
        }
        if (B.rows() != C.rows()) {
            throw std::runtime_error("compose_saddle_point_matrix: B.rows() must equal C.rows().");
        }
        if (B.cols() != A.rows()) {
            throw std::runtime_error("compose_saddle_point_matrix: B.cols() must equal A.rows().");
        }

        const int n_lambda = A.rows();
        const int n_u = C.rows();

        SparseBuilder builder;
        builder.reserve(
            estimate_nnz(A) +
            2 * estimate_nnz(B) +
            estimate_nnz(C));

        add_matrix_block(builder, A, 0, 0);
        add_matrix_block_transpose(builder, B, 0, n_lambda);
        add_matrix_block(builder, B, n_lambda, 0);
        add_matrix_block(builder, C, n_lambda, n_lambda);

        SparseMatrix out(n_lambda + n_u, n_lambda + n_u);
        out.set_from_builder(builder);
        return out;
    }

    template<class Backend>
    typename Backend::SparseMatrix compose_saddle_point_matrix_direct(
        const typename Backend::SparseMatrix& A,
        const typename Backend::SparseMatrix& B,
        const typename Backend::SparseMatrix& C)
    {
        using SparseMatrix = typename Backend::SparseMatrix;

        if (A.rows() != A.cols()) {
            throw std::runtime_error("compose_saddle_point_matrix_direct: A must be square.");
        }
        if (C.rows() != C.cols()) {
            throw std::runtime_error("compose_saddle_point_matrix_direct: C must be square.");
        }
        if (B.rows() != C.rows()) {
            throw std::runtime_error("compose_saddle_point_matrix_direct: B.rows() must equal C.rows().");
        }
        if (B.cols() != A.rows()) {
            throw std::runtime_error("compose_saddle_point_matrix_direct: B.cols() must equal A.rows().");
        }

        const int n_lambda = A.rows();
        const int n_u = C.rows();

        SparseMatrix out(n_lambda + n_u, n_lambda + n_u);
        add_matrix_block_direct(out, A, 0, 0);
        add_matrix_block_transpose_direct(out, B, 0, n_lambda);
        add_matrix_block_direct(out, B, n_lambda, 0);
        add_matrix_block_direct(out, C, n_lambda, n_lambda);
        out.compress();
        return out;
    }

    template<class Vector>
    void copy_vector_block(
        Vector& dst,
        int dst_offset,
        const Vector& src)
    {
        for (int i = 0; i < src.size(); ++i) {
            dst[dst_offset + i] = src[i];
        }
    }

    template<class Backend>
    typename Backend::Vector compose_2_block_vector(
        const typename Backend::Vector& v1,
        const typename Backend::Vector& v2)
    {
        using Vector = typename Backend::Vector;

        Vector out(v1.size() + v2.size());
        out.set_zero();

        copy_vector_block(out, 0, v1);
        copy_vector_block(out, v1.size(), v2);

        return out;
    }
}

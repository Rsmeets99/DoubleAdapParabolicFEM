#pragma once

#include <concepts>
#include <cstddef>

#include "vector.hpp"

namespace la::concepts
{
    template<class Matrix>
    concept DenseMatrixLike =
        VectorLike<typename Matrix::Vector> &&
        requires(
            Matrix A,
            const Matrix cA,
            const Matrix cB,
            const typename Matrix::Vector cx,
            int m,
            int n,
            int i,
            int j,
            double v)
        {
            typename Matrix::scalar_type;
            typename Matrix::index_type;
            typename Matrix::Vector;

            Matrix{};
            Matrix(m, n);

            A.resize(m, n);
            A.set_zero();
            A.set_identity(m, n);
            A.set(i, j, v);
            A.add(i, j, v);
            A.add_block(i, j, cB);
            A.add_block(i, j, cB, v);

            { cA.rows() } -> std::convertible_to<int>;
            { cA.cols() } -> std::convertible_to<int>;
            { cA.coeff(i, j) } -> std::convertible_to<double>;
            { cA(i, j) } -> std::convertible_to<double>;
            { A(i, j) } -> std::same_as<double&>;
            { cA.matvec(cx) } -> std::same_as<typename Matrix::Vector>;
            { cA.transpose_matvec(cx) } -> std::same_as<typename Matrix::Vector>;
            { cA.estimated_memory_bytes() } -> std::convertible_to<std::size_t>;
        };
}

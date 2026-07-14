#pragma once

#include <concepts>

namespace la::concepts
{
    template<class Matrix, class Builder>
    concept SparseMatrixLike =
        requires(Matrix A, const Matrix cA, const Builder& builder, int m, int n, int i, int j, double v)
        {
            Matrix{};
            Matrix(m, n);

            A.resize(m, n);
            A.set_from_builder(builder);
            A.set(i, j, v);
            A.add(i, j, v);
            A.compress();

            { cA.rows() } -> std::convertible_to<int>;
            { cA.cols() } -> std::convertible_to<int>;
            { cA.coeff(i, j) } -> std::convertible_to<double>;

            cA.for_each_nonzero(
                [](int, int, double) {});
        };
}
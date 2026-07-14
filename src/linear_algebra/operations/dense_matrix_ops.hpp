#pragma once

#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "../concepts/dense_matrix.hpp"
#include "../concepts/vector.hpp"

namespace la::ops
{
    template<class DenseMatrix>
    requires la::concepts::DenseMatrixLike<DenseMatrix>
    typename DenseMatrix::Vector matvec(
        const DenseMatrix& A,
        const typename DenseMatrix::Vector& x)
    {
        return A.matvec(x);
    }

    template<class DenseMatrix>
    requires la::concepts::DenseMatrixLike<DenseMatrix>
    typename DenseMatrix::Vector transpose_matvec(
        const DenseMatrix& A,
        const typename DenseMatrix::Vector& x)
    {
        return A.transpose_matvec(x);
    }

    template<class DenseMatrix>
    requires la::concepts::DenseMatrixLike<DenseMatrix>
    double quadratic_form(
        const DenseMatrix& A,
        const typename DenseMatrix::Vector& x)
    {
        if (A.rows() != A.cols())
            throw std::runtime_error(
                "la::ops::quadratic_form(dense): matrix must be square.");

        if (A.cols() != x.size())
            throw std::runtime_error(
                "la::ops::quadratic_form(dense): dimension mismatch.");

        const auto y = A.matvec(x);
        double value = 0.0;
        for (int i = 0; i < x.size(); ++i)
            value += x[i] * y[i];
        return value;
    }

    template<class DenseMatrix>
    requires la::concepts::DenseMatrixLike<DenseMatrix>
    void print_matrix(
        const DenseMatrix& A,
        std::ostream& os = std::cout,
        int width = 12)
    {
        os << "DenseMatrix (" << A.rows() << " x " << A.cols() << ")\n";
        for (int i = 0; i < A.rows(); ++i)
        {
            for (int j = 0; j < A.cols(); ++j)
                os << std::setw(width) << A.coeff(i, j) << ' ';
            os << '\n';
        }
    }
}

#pragma once

#include <cstddef>
#include <stdexcept>

#include <Eigen/Dense>

#include "vector.hpp"

namespace la::eigen
{
    class DenseMatrix
    {
    public:
        using scalar_type = double;
        using index_type = int;
        using native_type = Eigen::MatrixXd;
        using Vector = la::eigen::Vector;

        DenseMatrix() = default;

        DenseMatrix(index_type rows, index_type cols)
            : data_(rows, cols)
        {
            data_.setZero();
        }

        void resize(index_type rows, index_type cols)
        {
            data_.resize(rows, cols);
        }

        void set_zero()
        {
            data_.setZero();
        }

        void set_zero(index_type rows, index_type cols)
        {
            data_.setZero(rows, cols);
        }

        void set_identity(index_type rows, index_type cols)
        {
            data_.setIdentity(rows, cols);
        }

        void set(index_type i, index_type j, scalar_type value)
        {
            data_(i, j) = value;
        }

        void add(index_type i, index_type j, scalar_type value)
        {
            data_(i, j) += value;
        }

        void add_block(
            index_type row_offset,
            index_type col_offset,
            const DenseMatrix& block,
            scalar_type scale = 1.0)
        {
            if (row_offset < 0 || col_offset < 0 ||
                row_offset + block.rows() > rows() ||
                col_offset + block.cols() > cols())
            {
                throw std::runtime_error(
                    "DenseMatrix::add_block block does not fit.");
            }

            data_.block(row_offset, col_offset, block.rows(), block.cols()) +=
                scale * block.native();
        }

        void set_block(
            index_type row_offset,
            index_type col_offset,
            const DenseMatrix& block,
            scalar_type scale = 1.0)
        {
            if (row_offset < 0 || col_offset < 0 ||
                row_offset + block.rows() > rows() ||
                col_offset + block.cols() > cols())
            {
                throw std::runtime_error(
                    "DenseMatrix::set_block block does not fit.");
            }

            data_.block(row_offset, col_offset, block.rows(), block.cols()) =
                scale * block.native();
        }

        void set_transpose_block(
            index_type row_offset,
            index_type col_offset,
            const DenseMatrix& block,
            scalar_type scale = 1.0)
        {
            if (row_offset < 0 || col_offset < 0 ||
                row_offset + block.cols() > rows() ||
                col_offset + block.rows() > cols())
            {
                throw std::runtime_error(
                    "DenseMatrix::set_transpose_block block does not fit.");
            }

            data_.block(row_offset, col_offset, block.cols(), block.rows()) =
                scale * block.native().transpose();
        }

        [[nodiscard]] index_type rows() const noexcept
        {
            return static_cast<index_type>(data_.rows());
        }

        [[nodiscard]] index_type cols() const noexcept
        {
            return static_cast<index_type>(data_.cols());
        }

        [[nodiscard]] scalar_type coeff(index_type i, index_type j) const
        {
            return data_(i, j);
        }

        [[nodiscard]] scalar_type& operator()(index_type i, index_type j)
        {
            return data_(i, j);
        }

        [[nodiscard]] scalar_type operator()(index_type i, index_type j) const
        {
            return data_(i, j);
        }

        [[nodiscard]] Vector matvec(const Vector& x) const
        {
            if (cols() != x.size())
            {
                throw std::runtime_error(
                    "DenseMatrix::matvec dimension mismatch.");
            }

            Vector y(rows());
            y.native() = data_ * x.native();
            return y;
        }

        [[nodiscard]] Vector transpose_matvec(const Vector& x) const
        {
            if (rows() != x.size())
            {
                throw std::runtime_error(
                    "DenseMatrix::transpose_matvec dimension mismatch.");
            }

            Vector y(cols());
            y.native() = data_.transpose() * x.native();
            return y;
        }

        [[nodiscard]] scalar_type max_abs_coeff() const noexcept
        {
            return data_.size() == 0 ? 0.0 : data_.cwiseAbs().maxCoeff();
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return static_cast<std::size_t>(rows()) *
                   static_cast<std::size_t>(cols()) *
                   sizeof(scalar_type);
        }

        [[nodiscard]] native_type& native() noexcept
        {
            return data_;
        }

        [[nodiscard]] const native_type& native() const noexcept
        {
            return data_;
        }

    private:
        native_type data_{};
    };
}

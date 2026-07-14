#pragma once

#include <utility>
#include <Eigen/Sparse>

#include "sparse_builder.hpp"

namespace la::eigen
{
    class SparseMatrix
    {
    public:
        using scalar_type = double;
        using index_type  = int;
        using native_type = Eigen::SparseMatrix<scalar_type, Eigen::ColMajor, index_type>;

        SparseMatrix() = default;

        SparseMatrix(index_type rows, index_type cols)
            : data_(rows, cols)
        {
        }

        void resize(index_type rows, index_type cols)
        {
            data_.resize(rows, cols);
        }

        void set_from_builder(const SparseBuilder& builder)
        {
            data_.setFromTriplets(
                builder.triplets().begin(),
                builder.triplets().end(),
                [](scalar_type a, scalar_type b) { return a + b; }
            );
            data_.makeCompressed();
        }

        void set(index_type i, index_type j, scalar_type value)
        {
            data_.coeffRef(i, j) = value;
        }

        void add(index_type i, index_type j, scalar_type value)
        {
            data_.coeffRef(i, j) += value;
        }

        void compress()
        {
            data_.makeCompressed();
        }

        void prune(scalar_type reference, scalar_type epsilon)
        {
            data_.prune(reference, epsilon);
            data_.makeCompressed();
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
            return data_.coeff(i, j);
        }

        [[nodiscard]] native_type& native() noexcept
        {
            return data_;
        }

        [[nodiscard]] const native_type& native() const noexcept
        {
            return data_;
        }

        template<class Func>
        void for_each_nonzero(Func&& f) const
        {
            for (index_type k = 0; k < static_cast<index_type>(data_.outerSize()); ++k)
            {
                for (typename native_type::InnerIterator it(data_, k); it; ++it)
                {
                    std::forward<Func>(f)(
                        static_cast<index_type>(it.row()),
                        static_cast<index_type>(it.col()),
                        static_cast<scalar_type>(it.value()));
                }
            }
        }

    private:
        native_type data_{};
    };
}

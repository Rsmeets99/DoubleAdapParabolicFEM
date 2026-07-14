#pragma once

#include <Eigen/Core>

namespace la::eigen
{
    class Vector
    {
    public:
        using scalar_type = double;
        using index_type = int;
        using native_type = Eigen::VectorXd;

        Vector() = default;

        explicit Vector(index_type n)
            : data_(n)
        {
            data_.setZero();
        }

        void resize(index_type n)
        {
            data_.resize(n);
        }

        void set_zero()
        {
            data_.setZero();
        }

        void add(index_type i, scalar_type value)
        {
            data_[i] += value;
        }

        [[nodiscard]] index_type size() const noexcept
        {
            return static_cast<index_type>(data_.size());
        }

        [[nodiscard]] scalar_type& operator[](index_type i)
        {
            return data_[i];
        }

        [[nodiscard]] scalar_type operator[](index_type i) const
        {
            return data_[i];
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
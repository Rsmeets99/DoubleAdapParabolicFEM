#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace la::local
{
    struct LocalMatrix
    {
        int rows = 0;
        int cols = 0;
        std::vector<double> values;

        LocalMatrix() = default;

        LocalMatrix(int m, int n)
            : rows(m), cols(n), values(static_cast<std::size_t>(m * n), 0.0)
        {
        }

        void resize(int m, int n)
        {
            rows = m;
            cols = n;
            values.assign(static_cast<std::size_t>(m * n), 0.0);
        }

        double& operator()(int i, int j)
        {
            return values[static_cast<std::size_t>(i * cols + j)];
        }

        double operator()(int i, int j) const
        {
            return values[static_cast<std::size_t>(i * cols + j)];
        }
    };

    template<int Rows, int Cols>
    struct FixedLocalMatrix
    {
        static_assert(Rows >= 0 && Cols >= 0,
                      "FixedLocalMatrix dimensions must be non-negative.");

        static constexpr int rows_v = Rows;
        static constexpr int cols_v = Cols;

        int rows = Rows;
        int cols = Cols;
        std::array<double, static_cast<std::size_t>(Rows * Cols)> values{};

        FixedLocalMatrix() = default;

        FixedLocalMatrix(int m, int n)
        {
            resize(m, n);
        }

        void resize(int m, int n)
        {
            if (m != Rows || n != Cols)
            {
                throw std::runtime_error(
                    "FixedLocalMatrix: resize dimensions do not match compile-time extent.");
            }
            rows = Rows;
            cols = Cols;
            values.fill(0.0);
        }

        double& operator()(int i, int j)
        {
            return values[static_cast<std::size_t>(i * Cols + j)];
        }

        double operator()(int i, int j) const
        {
            return values[static_cast<std::size_t>(i * Cols + j)];
        }
    };

    struct LocalVector
    {
        int size = 0;
        std::vector<double> values;

        LocalVector() = default;

        explicit LocalVector(int n)
            : size(n), values(static_cast<std::size_t>(n), 0.0)
        {
        }

        void resize(int n)
        {
            size = n;
            values.assign(static_cast<std::size_t>(n), 0.0);
        }

        double& operator[](int i)
        {
            return values[static_cast<std::size_t>(i)];
        }

        double operator[](int i) const
        {
            return values[static_cast<std::size_t>(i)];
        }
    };

    template<int Size>
    struct FixedLocalVector
    {
        static_assert(Size >= 0,
                      "FixedLocalVector dimension must be non-negative.");

        static constexpr int size_v = Size;

        int size = Size;
        std::array<double, static_cast<std::size_t>(Size)> values{};

        FixedLocalVector() = default;

        explicit FixedLocalVector(int n)
        {
            resize(n);
        }

        void resize(int n)
        {
            if (n != Size)
            {
                throw std::runtime_error(
                    "FixedLocalVector: resize dimension does not match compile-time extent.");
            }
            size = Size;
            values.fill(0.0);
        }

        double& operator[](int i)
        {
            return values[static_cast<std::size_t>(i)];
        }

        double operator[](int i) const
        {
            return values[static_cast<std::size_t>(i)];
        }
    };
}

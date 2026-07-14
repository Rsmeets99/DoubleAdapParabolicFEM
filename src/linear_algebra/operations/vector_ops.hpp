#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "../concepts/vector.hpp"

namespace la::ops
{
    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    double inf_norm(const VectorLike& x)
    {
        double n = 0.0;
        for (int i = 0; i < x.size(); ++i)
            n = std::max(n, std::abs(x[i]));
        return n;
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    VectorLike subtract(const VectorLike& a, const VectorLike& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("la::ops::subtract: size mismatch.");

        VectorLike out(a.size());
        for (int i = 0; i < a.size(); ++i)
            out[i] = a[i] - b[i];
        return out;
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    VectorLike add(const VectorLike& a, const VectorLike& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("la::ops::add: size mismatch.");

        VectorLike out(a.size());
        for (int i = 0; i < a.size(); ++i)
            out[i] = a[i] + b[i];
        return out;
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    double dot(const VectorLike& a, const VectorLike& b)
    {
        if (a.size() != b.size())
            throw std::runtime_error("la::ops::dot: size mismatch.");

        double s = 0.0;
        for (int i = 0; i < a.size(); ++i)
            s += a[i] * b[i];
        return s;
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    void scale_in_place(VectorLike& x, double alpha)
    {
        for (int i = 0; i < x.size(); ++i)
            x[i] *= alpha;
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    VectorLike scaled(const VectorLike& x, double alpha)
    {
        VectorLike out(x.size());
        for (int i = 0; i < x.size(); ++i)
            out[i] = alpha * x[i];
        return out;
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    void axpy(double alpha, const VectorLike& x, VectorLike& y)
    {
        if (x.size() != y.size())
            throw std::runtime_error("la::ops::axpy: size mismatch.");

        for (int i = 0; i < x.size(); ++i)
            y[i] += alpha * x[i];
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    void print_vector(const VectorLike& x, std::ostream& os = std::cout, int width = 12)
    {
        os << "Vector (size = " << x.size() << ")\n";
        for (int i = 0; i < x.size(); ++i)
            os << std::setw(width) << x[i] << '\n';
    }

    template<class VectorLike>
    requires la::concepts::VectorLike<VectorLike>
    void print_vector_compact(const VectorLike& x, std::ostream& os = std::cout, int width = 12)
    {
        os << "Vector (size = " << x.size() << ") [";
        for (int i = 0; i < x.size(); ++i)
        {
            if (i > 0)
                os << ' ';
            os << std::setw(width) << x[i];
        }
        os << " ]\n";
    }
}
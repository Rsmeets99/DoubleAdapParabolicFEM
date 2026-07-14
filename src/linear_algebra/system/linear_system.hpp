#pragma once

namespace la::linear
{
    template<class Backend>
    struct LinearSystem
    {
        using SparseMatrix = typename Backend::SparseMatrix;
        using Vector = typename Backend::Vector;

        SparseMatrix matrix;
        Vector rhs;
        Vector solution;
    };
}
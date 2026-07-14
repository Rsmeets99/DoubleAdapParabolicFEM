#pragma once

#include "sparse_builder.hpp"
#include "sparse_pattern_builder.hpp"
#include "dense_matrix.hpp"
#include "dense_solver.hpp"
#include "vector.hpp"
#include "sparse_matrix.hpp"
#include "solver.hpp"

namespace la::eigen
{
    struct Backend
    {
        using scalar_type = double;
        using index_type = int;

        using SparseBuilder = la::eigen::SparseBuilder;
        using SparsePatternBuilder = la::eigen::SparsePatternBuilder;
        using Vector       = la::eigen::Vector;
        using DenseMatrix  = la::eigen::DenseMatrix;
        using DenseSolver  = la::eigen::DenseDirectSolver;
        using SparseMatrix = la::eigen::SparseMatrix;
        using Solver       = la::eigen::Solver;
    };
}

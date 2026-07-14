#pragma once

#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"

namespace finite_element::assembly::error_system
{
    template<class Backend, class PatchScalarSpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_C(
        typename Backend::SparseMatrix& C,
        const PatchScalarSpaceType& scalar_space)
    {
        typename Backend::SparseBuilder builder;
        builder.reserve(0);
        C.resize(scalar_space.n_dofs(), scalar_space.n_dofs());
        C.set_from_builder(builder);
    }
}

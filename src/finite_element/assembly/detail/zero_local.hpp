#pragma once

namespace finite_element::assembly::detail
{
    template<class LocalMatrixType>
    inline void zero_local_matrix(LocalMatrixType& local)
    {
        for (auto& v : local.values)
            v = 0.0;
    }

    template<class LocalVectorType>
    inline void zero_local_vector(LocalVectorType& local)
    {
        for (auto& v : local.values)
            v = 0.0;
    }
}
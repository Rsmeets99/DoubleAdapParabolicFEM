#include "detail/runner_case_execution.hpp"

namespace adaptive_algorithm::runners::detail
{
    int run_degree_2_dim_1(
        const RunnerOptions& options,
        const int quadrature_boost)
    {
        return run_degree_dimension_case<2, 1>(options, quadrature_boost);
    }
}

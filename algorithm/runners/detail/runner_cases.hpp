#pragma once

#include "runner_options.hpp"

namespace adaptive_algorithm::runners::detail
{
    int run_degree_1_dim_1(const RunnerOptions& options, int quadrature_boost);
    int run_degree_1_dim_2(const RunnerOptions& options, int quadrature_boost);
    int run_degree_2_dim_1(const RunnerOptions& options, int quadrature_boost);
    int run_degree_2_dim_2(const RunnerOptions& options, int quadrature_boost);
    int run_degree_3_dim_1(const RunnerOptions& options, int quadrature_boost);
    int run_degree_3_dim_2(const RunnerOptions& options, int quadrature_boost);
    int run_degree_4_dim_1(const RunnerOptions& options, int quadrature_boost);
    int run_degree_4_dim_2(const RunnerOptions& options, int quadrature_boost);
}

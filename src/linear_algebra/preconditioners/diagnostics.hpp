#pragma once

#include <cstddef>

namespace la::preconditioners
{
    struct ParabolicGraphNormApproximationDiagnostics
    {
        int hhat_rows = 0;
        int hhat_cols = 0;
        std::size_t hhat_nnz = 0;
        std::size_t c_signed_nnz = 0;
        std::size_t b_or_bdt_nnz_used = 0;
        std::size_t sparse_builder_entries = 0;
        double approximate_fill_ratio = 0.0;
        double setup_seconds = 0.0;
    };

    struct ParabolicGraphNormPreconditionerDiagnostics
    {
        bool is_spd = false;
        int n_lambda = 0;
        int n_u = 0;
        int hhat_rows = 0;
        int hhat_cols = 0;
        int nnz_hhat = 0;
        std::size_t c_signed_nnz = 0;
        std::size_t b_or_bdt_nnz_used = 0;
        double approximate_fill_ratio = 0.0;
        double setup_seconds = 0.0;
        ParabolicGraphNormApproximationDiagnostics approximation{};
    };
}

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "../../detail/timing.hpp"

namespace finite_element::assembly::detail
{
    struct AssemblyKernelDiagnostics
    {
        std::size_t active_cells = 0;
        std::size_t quadrature_points = 0;
        std::size_t scalar_basis_evaluations = 0;
        std::size_t gradient_evaluations = 0;
        std::size_t diffusion_tensor_evaluations = 0;
        std::size_t source_evaluations = 0;
        std::size_t sparse_triplets_emitted = 0;
        std::size_t final_matrix_nonzeros = 0;
        std::size_t peak_triplet_bytes = 0;
        std::size_t peak_sparse_matrix_bytes = 0;

        void add(const AssemblyKernelDiagnostics& other) noexcept
        {
            active_cells += other.active_cells;
            quadrature_points += other.quadrature_points;
            scalar_basis_evaluations += other.scalar_basis_evaluations;
            gradient_evaluations += other.gradient_evaluations;
            diffusion_tensor_evaluations += other.diffusion_tensor_evaluations;
            source_evaluations += other.source_evaluations;
            sparse_triplets_emitted += other.sparse_triplets_emitted;
            final_matrix_nonzeros += other.final_matrix_nonzeros;
            peak_triplet_bytes += other.peak_triplet_bytes;
            peak_sparse_matrix_bytes += other.peak_sparse_matrix_bytes;
        }
    };

    template<class SparseBuilder>
    [[nodiscard]] constexpr std::size_t estimated_triplet_bytes() noexcept
    {
        return sizeof(typename SparseBuilder::triplet_type);
    }

    template<class SparseMatrix>
    [[nodiscard]] std::size_t count_nonzeros(const SparseMatrix& A)
    {
        std::size_t nnz = 0;
        A.for_each_nonzero(
            [&](int, int, double)
            {
                ++nnz;
            });
        return nnz;
    }

    [[nodiscard]] inline std::size_t estimate_compressed_sparse_matrix_bytes(
        int rows,
        int cols,
        std::size_t nnz) noexcept
    {
        (void)rows;
        return
            nnz * (sizeof(double) + sizeof(int)) +
            static_cast<std::size_t>(cols + 1) * sizeof(int);
    }

    inline void record_assembly_diagnostics(
        const finite_element::detail::TimingRecorder& timing,
        std::string_view prefix,
        const AssemblyKernelDiagnostics& diagnostics)
    {
        if (!timing.enabled())
            return;

        const auto record =
            [&](const char* suffix, std::size_t value)
            {
                std::string phase(prefix);
                phase += suffix;
                timing.add(phase, static_cast<double>(value));
            };

        record(".active_cells.count", diagnostics.active_cells);
        record(".quadrature_points.count", diagnostics.quadrature_points);
        record(
            ".scalar_basis_evaluations.count",
            diagnostics.scalar_basis_evaluations);
        record(".gradient_evaluations.count", diagnostics.gradient_evaluations);
        record(
            ".diffusion_tensor_evaluations.count",
            diagnostics.diffusion_tensor_evaluations);
        record(".source_evaluations.count", diagnostics.source_evaluations);
        record(
            ".sparse_triplets_emitted.count",
            diagnostics.sparse_triplets_emitted);
        record(".final_matrix_nnz.count", diagnostics.final_matrix_nonzeros);
        record(".peak_triplet_bytes", diagnostics.peak_triplet_bytes);
        record(
            ".peak_sparse_matrix_bytes",
            diagnostics.peak_sparse_matrix_bytes);
    }
}

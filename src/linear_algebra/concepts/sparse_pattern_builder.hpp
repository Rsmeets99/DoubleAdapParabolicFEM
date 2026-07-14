#pragma once

#include <concepts>
#include <cstddef>

namespace la::concepts
{
    template<class Builder, class SparseMatrix>
    concept SparsePatternBuilderLike =
        requires(
            Builder b,
            int rows,
            int cols,
            int i,
            int j,
            double value,
            std::size_t n)
        {
            Builder{};

            b.resize(rows, cols);
            b.reserve_pattern(n);
            b.add_pattern(i, j);
            b.finalize_pattern();
            b.zero_values();
            b.add(i, j, value);
            b.add_to_slot(n, value);

            { b.rows() } -> std::convertible_to<int>;
            { b.cols() } -> std::convertible_to<int>;
            { b.slot(i, j) } -> std::convertible_to<std::size_t>;
            { b.pattern_entries() } -> std::convertible_to<std::size_t>;
            { b.numeric_nonzeros() } -> std::convertible_to<std::size_t>;
            { b.pattern_bytes() } -> std::convertible_to<std::size_t>;
            { b.numeric_matrix_bytes() } -> std::convertible_to<std::size_t>;
            { b.slot_map_bytes() } -> std::convertible_to<std::size_t>;

            { b.matrix() } -> std::same_as<const SparseMatrix&>;
            { b.release_matrix() } -> std::same_as<SparseMatrix>;
        };
}

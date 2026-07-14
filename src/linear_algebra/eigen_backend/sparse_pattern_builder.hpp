#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "sparse_matrix.hpp"

#ifndef APF_EIGEN_SPARSE_PATTERN_BUILDER_KEEP_DUPLICATES
#define APF_EIGEN_SPARSE_PATTERN_BUILDER_KEEP_DUPLICATES 0
#endif

namespace la::eigen
{
    class SparsePatternBuilder
    {
    public:
        using scalar_type = double;
        using index_type = int;
        using SparseMatrix = la::eigen::SparseMatrix;

        SparsePatternBuilder() = default;

        void resize(index_type rows, index_type cols)
        {
            rows_ = rows;
            cols_ = cols;
            rows_by_col_.clear();
            rows_by_col_.resize(static_cast<std::size_t>(cols_));
            pattern_entries_ = 0;
            pattern_candidate_count_ = 0;
            pattern_duplicate_count_ = 0;
            pattern_candidate_bytes_ = 0;
            peak_pattern_bytes_ = 0;
            matrix_ = SparseMatrix(rows, cols);
            finalized_ = false;
        }

        void reserve_pattern(std::size_t n)
        {
            if (cols_ <= 0)
                return;
#if APF_EIGEN_SPARSE_PATTERN_BUILDER_KEEP_DUPLICATES
            const std::size_t per_col =
                (n + static_cast<std::size_t>(cols_) - 1u) /
                static_cast<std::size_t>(cols_);
            for (auto& rows : rows_by_col_)
                rows.reserve(per_col);
#else
            const std::size_t per_col =
                (n + static_cast<std::size_t>(cols_) - 1u) /
                static_cast<std::size_t>(cols_);
            const std::size_t compact_hint = std::min<std::size_t>(per_col, 8u);
            if (compact_hint == 0u)
                return;
            for (auto& rows : rows_by_col_)
                rows.reserve(compact_hint);
#endif
        }

        void add_pattern(index_type row, index_type col)
        {
            validate_index_(row, col);
            ++pattern_candidate_count_;

            auto& rows = rows_by_col_[static_cast<std::size_t>(col)];
#if APF_EIGEN_SPARSE_PATTERN_BUILDER_KEEP_DUPLICATES
            rows.push_back(row);
#else
            if (rows.empty() || row > rows.back())
            {
                rows.push_back(row);
                return;
            }
            if (row == rows.back())
            {
                ++pattern_duplicate_count_;
                return;
            }

            const auto it = std::lower_bound(rows.begin(), rows.end(), row);
            if (it != rows.end() && *it == row)
            {
                ++pattern_duplicate_count_;
                return;
            }
            rows.insert(it, row);
#endif
        }

        void finalize_pattern()
        {
            pattern_entries_ = 0;
            if constexpr (APF_EIGEN_SPARSE_PATTERN_BUILDER_KEEP_DUPLICATES)
                pattern_duplicate_count_ = 0;
            pattern_candidate_bytes_ =
                rows_by_col_.capacity() * sizeof(std::vector<index_type>) +
                pattern_candidate_count_ * sizeof(index_type);
            peak_pattern_bytes_ =
                rows_by_col_.capacity() * sizeof(std::vector<index_type>);
            for (auto& rows : rows_by_col_)
            {
#if APF_EIGEN_SPARSE_PATTERN_BUILDER_KEEP_DUPLICATES
                const std::size_t before = rows.size();
                std::sort(rows.begin(), rows.end());
                rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
                pattern_duplicate_count_ += before - rows.size();
#else
                rows.shrink_to_fit();
#endif
                pattern_entries_ += rows.size();
                peak_pattern_bytes_ += rows.capacity() * sizeof(index_type);
            }

            typename SparseMatrix::native_type native(rows_, cols_);
            std::vector<index_type> entries_per_col(
                static_cast<std::size_t>(cols_),
                index_type{0});
            for (index_type col = 0; col < cols_; ++col)
                entries_per_col[static_cast<std::size_t>(col)] =
                    static_cast<index_type>(
                        rows_by_col_[static_cast<std::size_t>(col)].size());
            native.reserve(entries_per_col);

            for (index_type col = 0; col < cols_; ++col)
            {
                const auto& rows = rows_by_col_[static_cast<std::size_t>(col)];
                for (const index_type row : rows)
                    native.insert(row, col) = scalar_type{0.0};
            }
            native.makeCompressed();

            matrix_.native() = std::move(native);
            std::vector<std::vector<index_type>>().swap(rows_by_col_);
            finalized_ = true;
        }

        void zero_values()
        {
            ensure_finalized_();
            auto* values = matrix_.native().valuePtr();
            const std::size_t n_values =
                static_cast<std::size_t>(matrix_.native().nonZeros());
            std::fill(values, values + n_values, scalar_type{0.0});
        }

        void add(index_type row, index_type col, scalar_type value)
        {
            add_to_slot(slot(row, col), value);
        }

        [[nodiscard]] std::size_t slot(index_type row, index_type col) const
        {
            ensure_finalized_();
            validate_index_(row, col);
            const auto& native = matrix_.native();
            const auto outer_begin = native.outerIndexPtr()[col];
            const auto outer_end = native.outerIndexPtr()[col + 1];
            const auto* inner_begin = native.innerIndexPtr() + outer_begin;
            const auto* inner_end = native.innerIndexPtr() + outer_end;
            const auto* it = std::lower_bound(inner_begin, inner_end, row);
            if (it == inner_end || *it != row)
            {
                throw std::runtime_error(
                    "SparsePatternBuilder::add: entry is not in the finalized sparse pattern.");
            }
            return static_cast<std::size_t>(it - native.innerIndexPtr());
        }

        void add_to_slot(std::size_t slot, scalar_type value)
        {
            ensure_finalized_();
            if (slot >= numeric_nonzeros())
            {
                throw std::runtime_error(
                    "SparsePatternBuilder::add_to_slot: slot is out of bounds.");
            }
            matrix_.native().valuePtr()[slot] += value;
        }

        [[nodiscard]] index_type rows() const noexcept
        {
            return rows_;
        }

        [[nodiscard]] index_type cols() const noexcept
        {
            return cols_;
        }

        [[nodiscard]] std::size_t pattern_entries() const noexcept
        {
            return pattern_entries_;
        }

        [[nodiscard]] std::size_t pattern_candidate_count() const noexcept
        {
            return pattern_candidate_count_;
        }

        [[nodiscard]] std::size_t pattern_duplicate_count() const noexcept
        {
            return pattern_duplicate_count_;
        }

        [[nodiscard]] std::size_t pattern_candidate_bytes() const noexcept
        {
            return pattern_candidate_bytes_;
        }

        [[nodiscard]] std::size_t compact_pattern_bytes() const noexcept
        {
            return peak_pattern_bytes_;
        }

        [[nodiscard]] std::size_t pattern_bytes_saved() const noexcept
        {
            return pattern_candidate_bytes_ > peak_pattern_bytes_
                       ? pattern_candidate_bytes_ - peak_pattern_bytes_
                       : 0u;
        }

        [[nodiscard]] std::size_t numeric_nonzeros() const noexcept
        {
            return static_cast<std::size_t>(matrix_.native().nonZeros());
        }

        [[nodiscard]] std::size_t pattern_bytes() const noexcept
        {
            return peak_pattern_bytes_;
        }

        [[nodiscard]] std::size_t numeric_matrix_bytes() const noexcept
        {
            return
                numeric_nonzeros() * (sizeof(scalar_type) + sizeof(index_type)) +
                static_cast<std::size_t>(cols_ + 1) * sizeof(index_type);
        }

        [[nodiscard]] std::size_t slot_map_bytes() const noexcept
        {
            return 0;
        }

        [[nodiscard]] const SparseMatrix& matrix() const noexcept
        {
            return matrix_;
        }

        [[nodiscard]] SparseMatrix release_matrix()
        {
            ensure_finalized_();
            SparseMatrix out = std::move(matrix_);
            reset_();
            return out;
        }

    private:
        void validate_index_(index_type row, index_type col) const
        {
            if (row < 0 || col < 0 || row >= rows_ || col >= cols_)
            {
                throw std::runtime_error(
                    "SparsePatternBuilder: sparse index is out of bounds.");
            }
        }

        void ensure_finalized_() const
        {
            if (!finalized_)
            {
                throw std::runtime_error(
                    "SparsePatternBuilder: pattern has not been finalized.");
            }
        }

        void reset_()
        {
            rows_ = 0;
            cols_ = 0;
            rows_by_col_.clear();
            pattern_entries_ = 0;
            pattern_candidate_count_ = 0;
            pattern_duplicate_count_ = 0;
            pattern_candidate_bytes_ = 0;
            peak_pattern_bytes_ = 0;
            finalized_ = false;
        }

        index_type rows_ = 0;
        index_type cols_ = 0;
        std::vector<std::vector<index_type>> rows_by_col_{};
        std::size_t pattern_entries_ = 0;
        std::size_t pattern_candidate_count_ = 0;
        std::size_t pattern_duplicate_count_ = 0;
        std::size_t pattern_candidate_bytes_ = 0;
        std::size_t peak_pattern_bytes_ = 0;
        SparseMatrix matrix_{};
        bool finalized_ = false;
    };
}

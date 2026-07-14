#pragma once

#include <cstddef>
#include <iterator>
#include <vector>
#include <Eigen/Sparse>

namespace la::eigen
{
    class SparseBuilder
    {
    public:
        using scalar_type = double;
        using index_type = int;
        using triplet_type = Eigen::Triplet<scalar_type, index_type>;

        SparseBuilder() = default;

        void reserve(std::size_t n)
        {
            triplets_.reserve(n);
        }

        void add(index_type row, index_type col, scalar_type value)
        {
            triplets_.emplace_back(row, col, value);
        }

        void clear()
        {
            triplets_.clear();
        }

        void merge_from(const SparseBuilder& other)
        {
            triplets_.insert(
                triplets_.end(),
                other.triplets_.begin(),
                other.triplets_.end());
        }

        void merge_from(SparseBuilder&& other)
        {
            triplets_.insert(
                triplets_.end(),
                std::make_move_iterator(other.triplets_.begin()),
                std::make_move_iterator(other.triplets_.end()));
            std::vector<triplet_type>().swap(other.triplets_);
        }

        [[nodiscard]] const std::vector<triplet_type>& triplets() const noexcept
        {
            return triplets_;
        }

        [[nodiscard]] std::vector<triplet_type>& triplets() noexcept
        {
            return triplets_;
        }

        [[nodiscard]] std::size_t size() const noexcept
        {
            return triplets_.size();
        }

        [[nodiscard]] std::size_t capacity() const noexcept
        {
            return triplets_.capacity();
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return triplets_.capacity() * sizeof(triplet_type);
        }

    private:
        std::vector<triplet_type> triplets_{};
    };
}

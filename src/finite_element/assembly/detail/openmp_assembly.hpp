#pragma once

#include <algorithm>
#include <cstddef>
#include <exception>
#include <utility>
#include <vector>

#include "../../../core/openmp.hpp"

namespace finite_element::assembly::detail
{
    [[nodiscard]] inline bool should_use_openmp(
        int n_items,
        int min_items_per_thread,
        int min_total_items) noexcept
    {
        if (!core::has_openmp || n_items < min_total_items)
            return false;

        const int n_threads = core::max_openmp_threads();
        return n_threads > 1 && n_items >= n_threads * min_items_per_thread;
    }

    [[nodiscard]] inline int recommended_openmp_threads(
        int n_items,
        int min_items_per_thread,
        int min_total_items) noexcept
    {
        if (!core::has_openmp || n_items < min_total_items)
            return 1;

        const int max_threads = core::max_openmp_threads();
        if (max_threads <= 1)
            return 1;

        const int item_limited_threads =
            std::max(1, n_items / std::max(1, min_items_per_thread));
        return std::max(1, std::min(max_threads, item_limited_threads));
    }

    [[nodiscard]] inline bool should_use_openmp_for_cell_assembly(int n_items) noexcept
    {
        // Benchmark-driven defaults: small cell loops tend to lose to thread/setup and
        // merge overhead, while larger assembly batches benefit noticeably.
        return should_use_openmp(n_items, 32, 128);
    }

    [[nodiscard]] inline int recommended_openmp_threads_for_slab_solves(
        int n_items) noexcept
    {
        return recommended_openmp_threads(n_items, 1, 4);
    }

    [[nodiscard]] inline int recommended_openmp_threads_for_patch_solves(
        int n_items) noexcept
    {
        return recommended_openmp_threads(n_items, 1, 4);
    }

    [[nodiscard]] inline bool should_use_openmp_for_slab_solves(int n_items) noexcept
    {
        return recommended_openmp_threads_for_slab_solves(n_items) > 1;
    }

    [[nodiscard]] inline bool should_use_openmp_for_patch_solves(int n_items) noexcept
    {
        return recommended_openmp_threads_for_patch_solves(n_items) > 1;
    }

    [[nodiscard]] inline int openmp_thread_count() noexcept
    {
        return core::max_openmp_threads();
    }

    inline void rethrow_parallel_exception(const std::exception_ptr& error)
    {
        if (error)
            std::rethrow_exception(error);
    }

    template<class SparseBuilder>
    [[nodiscard]] std::vector<SparseBuilder> make_thread_local_sparse_builders(
        int n_threads,
        std::size_t total_reserve)
    {
        std::vector<SparseBuilder> builders;
        builders.reserve(static_cast<std::size_t>(n_threads));

        const std::size_t reserve_per_thread =
            n_threads > 0
                ? (total_reserve + static_cast<std::size_t>(n_threads) - 1u) /
                      static_cast<std::size_t>(n_threads)
                : total_reserve;

        for (int tid = 0; tid < n_threads; ++tid)
        {
            builders.emplace_back();
            builders.back().reserve(reserve_per_thread);
        }

        return builders;
    }

    template<class SparseBuilder>
    [[nodiscard]] SparseBuilder merge_sparse_builders(
        std::vector<SparseBuilder>& builders)
    {
        SparseBuilder merged;

        std::size_t total_size = 0;
        for (const auto& builder : builders)
            total_size += builder.size();

        merged.reserve(total_size);
        for (auto& builder : builders)
            merged.merge_from(std::move(builder));

        return merged;
    }

    template<class VectorLike>
    [[nodiscard]] std::vector<VectorLike> make_thread_local_vectors(
        int n_threads,
        int vector_size)
    {
        std::vector<VectorLike> vectors;
        vectors.reserve(static_cast<std::size_t>(n_threads));

        for (int tid = 0; tid < n_threads; ++tid)
        {
            vectors.emplace_back(vector_size);
            vectors.back().set_zero();
        }

        return vectors;
    }

    template<class VectorLike>
    void reduce_thread_local_vectors(
        VectorLike& out,
        const std::vector<VectorLike>& thread_local_vectors)
    {
        for (const auto& local : thread_local_vectors)
        {
            for (int i = 0; i < out.size(); ++i)
                out.add(i, local[i]);
        }
    }
}

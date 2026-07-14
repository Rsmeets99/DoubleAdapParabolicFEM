#pragma once

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
#include <omp.h>
#endif

namespace core
{
#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
    inline constexpr bool has_openmp = true;

    [[nodiscard]] inline int max_openmp_threads() noexcept
    {
        return omp_get_max_threads();
    }

    class ScopedOpenMPMaxThreads
    {
    public:
        explicit ScopedOpenMPMaxThreads(int requested_max_threads) noexcept
            : previous_max_threads_(omp_get_max_threads())
        {
            if (requested_max_threads > 0)
            {
                effective_max_threads_ = requested_max_threads;
                omp_set_num_threads(effective_max_threads_);
                active_ = true;
            }
            else
            {
                effective_max_threads_ = previous_max_threads_;
            }
        }

        ScopedOpenMPMaxThreads(const ScopedOpenMPMaxThreads&) = delete;
        ScopedOpenMPMaxThreads& operator=(const ScopedOpenMPMaxThreads&) = delete;

        ~ScopedOpenMPMaxThreads() noexcept
        {
            if (active_)
                omp_set_num_threads(previous_max_threads_);
        }

        [[nodiscard]] int previous_max_threads() const noexcept
        {
            return previous_max_threads_;
        }

        [[nodiscard]] int effective_max_threads() const noexcept
        {
            return effective_max_threads_;
        }

        [[nodiscard]] bool active() const noexcept
        {
            return active_;
        }

    private:
        int previous_max_threads_ = 1;
        int effective_max_threads_ = 1;
        bool active_ = false;
    };
#else
    inline constexpr bool has_openmp = false;

    [[nodiscard]] inline int max_openmp_threads() noexcept
    {
        return 1;
    }

    class ScopedOpenMPMaxThreads
    {
    public:
        explicit ScopedOpenMPMaxThreads(int) noexcept {}

        [[nodiscard]] int previous_max_threads() const noexcept
        {
            return 1;
        }

        [[nodiscard]] int effective_max_threads() const noexcept
        {
            return 1;
        }

        [[nodiscard]] bool active() const noexcept
        {
            return false;
        }
    };
#endif
}

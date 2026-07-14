#pragma once

#include <chrono>
#include <functional>
#include <string>
#include <string_view>
#include <utility>

namespace finite_element::detail
{
    using TimingCallback = std::function<void(std::string_view, double)>;

    class ScopedTiming
    {
    public:
        ScopedTiming() noexcept = default;

        ScopedTiming(const ScopedTiming&) = delete;
        ScopedTiming& operator=(const ScopedTiming&) = delete;

        ScopedTiming(ScopedTiming&& other) noexcept
            : callback_(std::move(other.callback_)),
              phase_(std::move(other.phase_)),
              start_(other.start_),
              active_(std::exchange(other.active_, false))
        {}

        ScopedTiming& operator=(ScopedTiming&& other) noexcept
        {
            if (this == &other)
                return *this;

            stop_noexcept_();
            callback_ = std::move(other.callback_);
            phase_ = std::move(other.phase_);
            start_ = other.start_;
            active_ = std::exchange(other.active_, false);
            return *this;
        }

        ~ScopedTiming() noexcept
        {
            stop_noexcept_();
        }

        [[nodiscard]] bool active() const noexcept
        {
            return active_;
        }

        void stop()
        {
            if (!active_ || !callback_)
                return;

            const auto end = Clock::now();
            const double seconds =
                std::chrono::duration<double>(end - start_).count();
            active_ = false;
            callback_(phase_, seconds);
        }

    private:
        using Clock = std::chrono::steady_clock;

        friend class TimingRecorder;

        ScopedTiming(TimingCallback callback, std::string_view phase)
            : callback_(std::move(callback)),
              phase_(phase),
              start_(Clock::now()),
              active_(static_cast<bool>(callback_))
        {}

        void stop_noexcept_() noexcept
        {
            try
            {
                stop();
            }
            catch (...)
            {
            }
        }

        TimingCallback callback_{};
        std::string phase_{};
        Clock::time_point start_{};
        bool active_ = false;
    };

    class TimingRecorder
    {
    public:
        TimingRecorder() = default;

        explicit TimingRecorder(TimingCallback callback)
            : callback_(std::move(callback))
        {}

        [[nodiscard]] bool enabled() const noexcept
        {
            return static_cast<bool>(callback_);
        }

        void set_callback(TimingCallback callback)
        {
            callback_ = std::move(callback);
        }

        void clear()
        {
            callback_ = {};
        }

        void add(std::string_view phase, double seconds) const
        {
            if (callback_)
                callback_(phase, seconds);
        }

        [[nodiscard]] ScopedTiming scoped(std::string_view phase) const
        {
            if (!callback_)
                return {};

            return ScopedTiming(callback_, phase);
        }

    private:
        TimingCallback callback_{};
    };
}

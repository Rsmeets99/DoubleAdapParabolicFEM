#pragma once

#include <algorithm>
#include <cstdint>
#include <cstddef>

#include "../../core/hash.hpp"

namespace mesh::topology
{
    struct TimePointIdKey
    {
        int vertex_id = -1;

        [[nodiscard]] bool operator==(const TimePointIdKey&) const noexcept =
            default;
        [[nodiscard]] bool operator<(const TimePointIdKey& other) const noexcept
        {
            return vertex_id < other.vertex_id;
        }
    };

    struct TimePointIdKeyHash
    {
        [[nodiscard]] std::size_t operator()(
            const TimePointIdKey& key) const noexcept
        {
            std::size_t seed = 0;
            core::hash_combine(seed, key.vertex_id);
            return seed;
        }
    };

    struct TimeIntervalIdKey
    {
        int v0 = -1;
        int v1 = -1;

        constexpr TimeIntervalIdKey() noexcept = default;

        constexpr TimeIntervalIdKey(int a, int b) noexcept
            : v0(a),
              v1(b)
        {
            if (v1 < v0)
                std::swap(v0, v1);
        }

        [[nodiscard]] bool operator==(const TimeIntervalIdKey&) const noexcept =
            default;
        [[nodiscard]] bool operator<(const TimeIntervalIdKey& other) const noexcept
        {
            if (v0 != other.v0)
                return v0 < other.v0;
            return v1 < other.v1;
        }
    };

    struct TimeIntervalIdKeyHash
    {
        [[nodiscard]] std::size_t operator()(
            const TimeIntervalIdKey& key) const noexcept
        {
            std::size_t seed = 0;
            core::hash_combine(seed, key.v0);
            core::hash_combine(seed, key.v1);
            return seed;
        }
    };

    struct DyadicTimePointKey
    {
        int root_interval_id = -1;
        int level = 0;
        std::uint64_t index = 0;

        [[nodiscard]] bool operator==(const DyadicTimePointKey&) const noexcept =
            default;
        [[nodiscard]] bool operator<(const DyadicTimePointKey& other) const noexcept
        {
            if (root_interval_id != other.root_interval_id)
                return root_interval_id < other.root_interval_id;
            if (level != other.level)
                return level < other.level;
            return index < other.index;
        }
    };

    struct DyadicTimePointKeyHash
    {
        [[nodiscard]] std::size_t operator()(
            const DyadicTimePointKey& key) const noexcept
        {
            std::size_t seed = 0;
            core::hash_combine(seed, key.root_interval_id);
            core::hash_combine(seed, key.level);
            core::hash_combine(seed, key.index);
            return seed;
        }
    };

    struct DyadicTimeIntervalKey
    {
        int root_interval_id = -1;
        int level = 0;
        std::uint64_t index = 0;

        [[nodiscard]] bool operator==(const DyadicTimeIntervalKey&) const noexcept =
            default;
        [[nodiscard]] bool operator<(const DyadicTimeIntervalKey& other) const noexcept
        {
            if (root_interval_id != other.root_interval_id)
                return root_interval_id < other.root_interval_id;
            if (level != other.level)
                return level < other.level;
            return index < other.index;
        }
    };

    struct DyadicTimeIntervalKeyHash
    {
        [[nodiscard]] std::size_t operator()(
            const DyadicTimeIntervalKey& key) const noexcept
        {
            std::size_t seed = 0;
            core::hash_combine(seed, key.root_interval_id);
            core::hash_combine(seed, key.level);
            core::hash_combine(seed, key.index);
            return seed;
        }
    };

    [[nodiscard]] constexpr TimePointIdKey make_time_point_id_key(
        const int vertex_id) noexcept
    {
        return TimePointIdKey{vertex_id};
    }

    [[nodiscard]] constexpr TimeIntervalIdKey make_time_interval_id_key(
        const int v0,
        const int v1) noexcept
    {
        return TimeIntervalIdKey{v0, v1};
    }

    [[nodiscard]] constexpr DyadicTimePointKey make_dyadic_time_point_key(
        const int root_interval_id,
        int level,
        std::uint64_t index) noexcept
    {
        while (level > 0 && (index % 2U) == 0U)
        {
            index /= 2U;
            --level;
        }
        return DyadicTimePointKey{root_interval_id, level, index};
    }

    [[nodiscard]] constexpr DyadicTimeIntervalKey make_dyadic_time_interval_key(
        const int root_interval_id,
        const int level,
        const std::uint64_t index) noexcept
    {
        return DyadicTimeIntervalKey{root_interval_id, level, index};
    }

    [[nodiscard]] constexpr DyadicTimePointKey dyadic_time_interval_begin_key(
        const DyadicTimeIntervalKey interval) noexcept
    {
        return make_dyadic_time_point_key(
            interval.root_interval_id,
            interval.level,
            interval.index);
    }

    [[nodiscard]] constexpr DyadicTimePointKey dyadic_time_interval_end_key(
        const DyadicTimeIntervalKey interval) noexcept
    {
        return make_dyadic_time_point_key(
            interval.root_interval_id,
            interval.level,
            interval.index + 1U);
    }
}

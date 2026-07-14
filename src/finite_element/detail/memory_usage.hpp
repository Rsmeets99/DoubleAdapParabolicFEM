#pragma once

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <optional>

namespace finite_element::detail
{
    [[nodiscard]] inline std::optional<std::size_t> proc_status_kib_(
        const char* field) noexcept
    {
        FILE* file = std::fopen("/proc/self/status", "r");
        if (file == nullptr)
            return std::nullopt;

        char line[256];
        std::size_t value_kib = 0;
        const auto field_len = std::strlen(field);
        while (std::fgets(line, sizeof(line), file) != nullptr)
        {
            if (std::strncmp(line, field, field_len) != 0)
                continue;

            if (std::sscanf(line + field_len, "%zu", &value_kib) == 1)
                break;
        }

        std::fclose(file);
        return value_kib > 0 ? std::optional<std::size_t>{value_kib}
                             : std::nullopt;
    }

    [[nodiscard]] inline std::size_t current_process_rss_bytes() noexcept
    {
        const auto rss_kib = proc_status_kib_("VmRSS:");
        return rss_kib.value_or(0) * std::size_t{1024};
    }

    [[nodiscard]] inline std::size_t peak_process_rss_bytes() noexcept
    {
        const auto rss_kib = proc_status_kib_("VmHWM:");
        return rss_kib.value_or(0) * std::size_t{1024};
    }

    [[nodiscard]] inline std::size_t available_system_memory_bytes() noexcept
    {
        FILE* file = std::fopen("/proc/meminfo", "r");
        if (file == nullptr)
            return 0;

        char key[64];
        char unit[32];
        std::size_t value_kib = 0;
        std::size_t mem_total_kib = 0;
        while (std::fscanf(file, "%63s %zu %31s", key, &value_kib, unit) == 3)
        {
            if (std::strcmp(key, "MemAvailable:") == 0)
            {
                std::fclose(file);
                return value_kib * std::size_t{1024};
            }
            if (std::strcmp(key, "MemTotal:") == 0)
                mem_total_kib = value_kib;
        }

        std::fclose(file);
        return mem_total_kib * std::size_t{1024};
    }
}

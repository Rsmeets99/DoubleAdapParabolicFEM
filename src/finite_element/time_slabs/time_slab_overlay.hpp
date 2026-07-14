#pragma once

#include <cctype>
#include <stdexcept>
#include <string>
#include <string_view>

namespace finite_element::time_slabs
{
    enum class TimeSlabBackend
    {
        CopiedMesh,
    };

    [[nodiscard]] inline constexpr const char* time_slab_backend_name(
        TimeSlabBackend) noexcept
    {
        return "copied_mesh";
    }

    [[nodiscard]] inline TimeSlabBackend parse_time_slab_backend(
        std::string_view text)
    {
        std::string normalized;
        normalized.reserve(text.size());
        for (const char c : text)
        {
            normalized.push_back(
                c == '-'
                    ? '_'
                    : static_cast<char>(
                          std::tolower(static_cast<unsigned char>(c))));
        }

        if (normalized == "copied_mesh" || normalized == "copied")
            return TimeSlabBackend::CopiedMesh;

        throw std::runtime_error(
            "Unsupported time slab backend '" + std::string(text) +
            "'. Supported value: copied_mesh.");
    }
}

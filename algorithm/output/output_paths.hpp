#pragma once

#include <cstdlib>
#include <filesystem>

namespace adaptive_algorithm::output
{
    [[nodiscard]] inline std::filesystem::path algorithm_repository_root()
    {
#ifdef ADAPPARABOLICFEM_SOURCE_DIR
        return std::filesystem::path(ADAPPARABOLICFEM_SOURCE_DIR);
#else
        return std::filesystem::current_path();
#endif
    }

    [[nodiscard]] inline std::filesystem::path algorithm_output_root()
    {
        if (const char* output_dir =
                std::getenv("ADAPPARABOLICFEM_ALGORITHM_OUTPUT_DIR"))
        {
            if (*output_dir != '\0')
                return std::filesystem::path(output_dir);
        }

        return algorithm_repository_root() / "algorithm_data";
    }

    [[nodiscard]] inline std::filesystem::path ensure_algorithm_output_dir(
        const std::filesystem::path& relative = {})
    {
        const auto output_dir =
            relative.empty()
                ? algorithm_output_root()
                : algorithm_output_root() / relative;

        std::filesystem::create_directories(output_dir);
        return output_dir;
    }
}

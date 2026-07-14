#pragma once

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace finite_element::io::detail
{
    inline std::string time_slab_directory_name(int slab_id)
    {
        std::ostringstream oss;
        oss << "slab_" << std::setw(4) << std::setfill('0') << slab_id;
        return oss.str();
    }

    inline std::filesystem::path time_slab_directory(
        const std::filesystem::path& output_dir,
        int slab_id)
    {
        return output_dir / time_slab_directory_name(slab_id);
    }
}
#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace finite_element::io::detail
{
    using binary_int_t = std::int32_t;

    inline std::ofstream open_binary_output(
        const std::filesystem::path& output_dir,
        const std::string& filename,
        const char* context)
    {
        std::filesystem::create_directories(output_dir);

        std::ofstream file(output_dir / filename, std::ios::binary | std::ios::trunc);
        if (!file.is_open())
        {
            throw std::runtime_error(
                std::string(context) + ": failed to open output file \""
                + (output_dir / filename).string() + "\".");
        }

        return file;
    }

    template<class T>
    void write_binary_block(
        std::ofstream& file,
        const std::vector<T>& data,
        const char* context)
    {
        if (data.empty())
            return;

        file.write(
            reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size() * sizeof(T)));

        if (!file)
            throw std::runtime_error(std::string(context) + ": failed while writing binary block.");
    }

    template<class Point>
    void append_point_coordinates(std::vector<double>& coords, const Point& p)
    {
        coords.insert(coords.end(), p.begin(), p.end());
    }

    template<class PointContainer>
    std::vector<double> flatten_point_coordinates(const PointContainer& points)
    {
        std::vector<double> coords;
        if constexpr (requires { points[0].size(); })
        {
            if (!points.empty())
                coords.reserve(points.size() * points[0].size());
        }

        for (const auto& p : points)
            append_point_coordinates(coords, p);

        return coords;
    }
}

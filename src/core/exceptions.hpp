#pragma once

#include <stdexcept>
#include <string>

namespace core
{
    class not_implemented_error : public std::logic_error
    {
    public:
        explicit not_implemented_error(const std::string& msg)
            : std::logic_error(msg) {}
    };

    class dimension_not_supported_error : public std::logic_error
    {
    public:
        explicit dimension_not_supported_error(const std::string& msg)
            : std::logic_error(msg) {}
    };

    class invalid_mesh_error : public std::runtime_error
    {
    public:
        explicit invalid_mesh_error(const std::string& msg)
            : std::runtime_error(msg) {}
    };

    class invalid_fespace_error : public std::runtime_error
    {
    public:
        explicit invalid_fespace_error(const std::string& msg)
            : std::runtime_error(msg) {}
    };
}
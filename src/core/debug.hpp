#pragma once
#include <stdexcept>
#include <string>
#include <iostream>

namespace core
{
    inline void require(bool cond, const std::string& msg)
    {
        if (!cond)
            throw std::runtime_error(msg);
    }

    inline void print_step(const std::string& msg)
    {
        std::cout << "[STEP] " << msg << '\n';
    }

    inline void print_ok(const std::string& msg)
    {
        std::cout << "  ok: " << msg << '\n';
    }
}

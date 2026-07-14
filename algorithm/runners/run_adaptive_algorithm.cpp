#include <exception>
#include <iostream>

#include "detail/runner_execution.hpp"

int main(int argc, char** argv)
{
    try
    {
        return adaptive_algorithm::runners::detail::run_from_command_line(argc, argv);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

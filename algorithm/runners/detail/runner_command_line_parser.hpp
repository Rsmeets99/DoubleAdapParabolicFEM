#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

#include "runner_option_specs.hpp"
#include "runner_usage_text.hpp"
#include "runner_value_parsing.hpp"

namespace adaptive_algorithm::runners::detail
{
    inline void apply_cli_flag_override(
        RunnerOptionOverrides& options,
        const CliOptionSpec& spec)
    {
        assign_bool_override(options, spec.id, spec.bool_flag_value);
    }

    [[nodiscard]] inline RunnerOptionOverrides parse_command_line_overrides(
        int argc,
        char** argv)
    {
        RunnerOptionOverrides options;

        for (int i = 1; i < argc; ++i)
        {
            const std::string_view arg(argv[i]);
            const auto* spec = find_cli_option_spec(arg);

            if (spec == nullptr)
            {
                throw std::runtime_error(
                    usage_text(argv[0]) +
                    "\nUnknown option '" + std::string(arg) +
                    "'. Run with --help to see the supported flags.");
            }

            if (spec->value_type == RunnerOptionValueType::bool_flag)
            {
                apply_cli_flag_override(options, *spec);
                continue;
            }

            const std::string value(require_next_argument(argc, argv, i));
            apply_typed_override(
                options,
                spec->id,
                spec->value_type,
                value,
                argv[0],
                std::string(spec->flag));
        }

        return options;
    }
}

#pragma once

#include <cctype>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "runner_option_specs.hpp"
#include "runner_usage_text.hpp"
#include "runner_value_parsing.hpp"

namespace adaptive_algorithm::runners::detail
{
    inline void require_yaml_config_file(
        const std::filesystem::path& filename,
        const char* argv0)
    {
        if (lowercase_copy(filename.extension().string()) != ".yml")
        {
            throw std::runtime_error(
                usage_text(argv0) +
                "\nUnsupported configuration file '" + filename.string() +
                "'. Only .yml files with flat keys and the optional one-level "
                "solver_diagnostics section are supported.");
        }
    }

    inline void apply_config_override(
        RunnerOptionOverrides& options,
        const ConfigOptionSpec& spec,
        const std::string& value,
        const char* argv0,
        const std::string& source)
    {
        if (spec.value_type == RunnerOptionValueType::bool_value && spec.invert_bool)
        {
            assign_bool_override(
                options,
                spec.id,
                !parse_bool_text(value, argv0, source + ":" + std::string(spec.key)));
            return;
        }

        apply_typed_override(
            options,
            spec.id,
            spec.value_type,
            value,
            argv0,
            source + ":" + std::string(spec.key));
    }

    [[nodiscard]] inline RunnerOptionOverrides parse_config_file(
        const std::filesystem::path& filename,
        const char* argv0)
    {
        require_yaml_config_file(filename, argv0);

        std::ifstream in(filename);
        if (!in)
        {
            throw std::runtime_error(
                usage_text(argv0) +
                "\nFailed to open configuration file '" +
                filename.string() + "'.");
        }

        RunnerOptionOverrides overrides;
        std::string line;
        std::string current_section;
        int line_number = 0;

        while (std::getline(in, line))
        {
            ++line_number;

            const bool indented =
                !line.empty() &&
                std::isspace(static_cast<unsigned char>(line.front())) != 0;
            std::string text = strip_inline_comment(line);
            if (text.empty())
                continue;

            const std::size_t colon_pos = text.find(':');
            if (colon_pos == std::string::npos)
            {
                throw std::runtime_error(
                    usage_text(argv0) +
                    "\nInvalid configuration line " +
                    std::to_string(line_number) +
                    " in '" + filename.string() +
                    "': expected YAML syntax 'key: value'.");
            }

            std::string key =
                normalize_key(trim_copy(text.substr(0, colon_pos)));
            const std::string value =
                strip_matching_quotes(
                    trim_copy(text.substr(colon_pos + 1)));

            if (!indented)
                current_section.clear();

            if (key.empty())
            {
                throw std::runtime_error(
                    usage_text(argv0) +
                    "\nInvalid configuration line " +
                    std::to_string(line_number) +
                    " in '" + filename.string() +
                    "': empty key.");
            }

            if (value.empty() && key == "solver_diagnostics")
            {
                current_section = key;
                continue;
            }

            if (indented && !current_section.empty())
                key = current_section + "." + key;

            const auto* spec = find_config_option_spec(key);
            if (spec == nullptr)
            {
                throw std::runtime_error(
                    usage_text(argv0) +
                    "\nUnknown configuration key '" + key + "' in " +
                    filename.string() + ". " + supported_config_keys_text());
            }

            apply_config_override(
                overrides,
                *spec,
                value,
                argv0,
                filename.string());
        }

        return overrides;
    }
}

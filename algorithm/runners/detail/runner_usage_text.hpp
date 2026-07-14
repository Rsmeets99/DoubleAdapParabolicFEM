#pragma once

#include <cstddef>
#include <string>

#include "runner_option_specs.hpp"

namespace adaptive_algorithm::runners::detail
{
    [[nodiscard]] inline std::string format_usage_label(const CliOptionSpec& spec)
    {
        constexpr std::size_t option_column_width = 28;

        std::string label = "  ";
        label += spec.flag;

        if (!spec.value_name.empty())
        {
            label += ' ';
            label += spec.value_name;
        }

        if (label.size() < option_column_width)
            label.append(option_column_width - label.size(), ' ');
        else
            label += ' ';

        return label;
    }

    [[nodiscard]] inline std::string usage_text(const char* argv0)
    {
        std::string text =
            "Usage: " + std::string(argv0) + " [options]\n"
            "\n"
            "Options:\n";

        for (const auto& spec : cli_option_specs)
        {
            if (!spec.visible_in_usage)
                continue;

            text += format_usage_label(spec);
            text += spec.description;
            text += '\n';

            if (!spec.continuation.empty())
            {
                text.append(28, ' ');
                text += spec.continuation;
                text += '\n';
            }
        }

        return text;
    }

    [[nodiscard]] inline std::string supported_config_keys_text()
    {
        std::string text = "Supported config keys: ";

        for (std::size_t i = 0; i < config_option_specs.size(); ++i)
        {
            if (!config_option_specs[i].visible_in_usage)
                continue;

            if (!text.ends_with(": "))
                text += ", ";

            text += config_option_specs[i].key;
        }

        text += '.';
        return text;
    }
}

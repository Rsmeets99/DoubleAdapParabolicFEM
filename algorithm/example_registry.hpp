#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "examples/space_time_1d/initial_data_1d_examples.hpp"
#include "examples/space_time_2d/initial_data_2d_examples.hpp"

namespace adaptive_algorithm::examples
{
    struct ExampleDescriptor
    {
        std::string_view name{};
        int dim_space = 1;
    };

    [[nodiscard]] inline std::string dimension_label(int spatial_dimension)
    {
        return std::to_string(spatial_dimension) + "+1D";
    }

    [[nodiscard]] inline std::vector<ExampleDescriptor> available_examples()
    {
        return {
            {"smooth_initial", 1},
            {"non_matching_initial", 1},
            {"boundary_singularity", 1},
            {"smooth_initial", 2},
            {"non_matching_initial", 2},
            {"boundary_singularity", 2}
        };
    }

    [[nodiscard]] inline std::vector<std::string> available_example_names()
    {
        const auto descriptors = available_examples();
        std::vector<std::string> names;
        names.reserve(descriptors.size());
        for (const auto& descriptor : descriptors)
            names.emplace_back(descriptor.name);
        return names;
    }

    [[nodiscard]] inline std::vector<std::string>
    available_example_names_for_dimension(int spatial_dimension)
    {
        const auto descriptors = available_examples();
        std::vector<std::string> names;
        for (const auto& descriptor : descriptors)
        {
            if (descriptor.dim_space == spatial_dimension)
                names.emplace_back(descriptor.name);
        }
        return names;
    }

    [[nodiscard]] inline std::vector<std::string> available_1d_example_names()
    {
        return available_example_names_for_dimension(1);
    }

    [[nodiscard]] inline std::vector<std::string> available_2d_example_names()
    {
        return available_example_names_for_dimension(2);
    }

    [[nodiscard]] inline std::string available_example_names_message()
    {
        const auto descriptors = available_examples();

        std::string text = "Valid examples: ";
        for (std::size_t i = 0; i < descriptors.size(); ++i)
        {
            if (i > 0)
                text += ", ";

            text += descriptors[i].name;
            text += " (";
            text += dimension_label(descriptors[i].dim_space);
            text += ")";
        }

        return text;
    }

    [[nodiscard]] inline bool example_is_registered(std::string_view name)
    {
        for (const auto& descriptor : available_examples())
            if (descriptor.name == name)
                return true;

        return false;
    }

    [[nodiscard]] inline bool example_is_registered_for_dimension(
        std::string_view name,
        int spatial_dimension)
    {
        for (const auto& descriptor : available_examples())
        {
            if (descriptor.name == name &&
                descriptor.dim_space == spatial_dimension)
            {
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] inline int example_spatial_dimension(std::string_view name)
    {
        std::optional<int> dimension;
        bool ambiguous = false;

        for (const auto& descriptor : available_examples())
        {
            if (descriptor.name == name)
            {
                if (!dimension.has_value())
                {
                    dimension = descriptor.dim_space;
                }
                else if (*dimension != descriptor.dim_space)
                {
                    ambiguous = true;
                }
            }
        }

        if (dimension.has_value())
        {
            if (ambiguous)
            {
                throw std::runtime_error(
                    "adaptive_algorithm examples: example '" +
                    std::string(name) +
                    "' is registered for multiple dimensions. Specify the "
                    "spatial dimension explicitly.");
            }

            return *dimension;
        }

        throw std::runtime_error(
            "adaptive_algorithm examples: unknown example '" +
            std::string(name) + "'. " + available_example_names_message());
    }

    [[nodiscard]] inline int example_spatial_dimension(
        std::string_view name,
        int spatial_dimension)
    {
        if (example_is_registered_for_dimension(name, spatial_dimension))
            return spatial_dimension;

        if (example_is_registered(name))
        {
            throw std::runtime_error(
                "adaptive_algorithm examples: example '" +
                std::string(name) +
                "' is not registered for dimension " +
                std::to_string(spatial_dimension) + ". " +
                available_example_names_message());
        }

        throw std::runtime_error(
            "adaptive_algorithm examples: unknown example '" +
            std::string(name) + "'. " + available_example_names_message());
    }

    [[nodiscard]] inline int example_spatial_dimension(
        std::string_view name,
        std::optional<int> spatial_dimension)
    {
        if (spatial_dimension.has_value())
            return example_spatial_dimension(name, *spatial_dimension);

        return example_spatial_dimension(name);
    }

    [[nodiscard]] inline bool is_1d_example(std::string_view name)
    {
        if (!example_is_registered(name))
            (void)example_spatial_dimension(name);

        return example_is_registered_for_dimension(name, 1);
    }

    [[nodiscard]] inline bool is_2d_example(std::string_view name)
    {
        if (!example_is_registered(name))
            (void)example_spatial_dimension(name);

        return example_is_registered_for_dimension(name, 2);
    }

    [[nodiscard]] inline SpaceTime1DExample make_1d_example(
        std::string_view name)
    {
        if (name == "smooth_initial")
            return smooth_initial_1d_example();

        if (name == "non_matching_initial")
            return non_matching_initial_1d_example();

        if (name == "boundary_singularity")
            return boundary_singularity_1d_example();

        throw std::runtime_error(
            "adaptive_algorithm examples: example '" + std::string(name) +
            "' is not a registered 1+1D example.");
    }

    [[nodiscard]] inline SpaceTime2DExample make_2d_example(
        std::string_view name)
    {
        if (name == "smooth_initial")
            return smooth_initial_2d_example();

        if (name == "non_matching_initial")
            return non_matching_initial_2d_example();

        if (name == "boundary_singularity")
            return boundary_singularity_2d_example();

        throw std::runtime_error(
            "adaptive_algorithm examples: example '" + std::string(name) +
            "' is not a registered 2+1D example.");
    }

    [[nodiscard]] inline SpaceTime1DExample make_example(std::string_view name)
    {
        return make_1d_example(name);
    }
}

#pragma once

#include <type_traits>
#include <vector>

#include "finite_element/fespace/io/detail/write_binary_block.hpp"

namespace finite_element::io::detail
{
    inline constexpr binary_int_t mesh_binary_format_version = 1;
    inline constexpr binary_int_t dofs_binary_format_version = 1;
    inline constexpr binary_int_t function_binary_format_version = 1;
    inline constexpr binary_int_t function_binary_format_version_2d = 2;
    inline constexpr binary_int_t function_true_coefficients_binary_format_version = 1;
    inline constexpr binary_int_t time_slab_space_metadata_binary_format_version = 1;
    inline constexpr binary_int_t time_slab_provenance_binary_format_version = 1;
    inline constexpr binary_int_t time_slab_edge_patches_binary_format_version = 1;
    inline constexpr binary_int_t time_slab_vertex_patches_binary_format_version = 2;
    inline constexpr binary_int_t time_slab_flux_reconstruction_samples_binary_format_version = 1;
    inline constexpr binary_int_t cellwise_error_binary_format_version = 1;
    inline constexpr binary_int_t cellwise_error_components_binary_format_version = 1;
    inline constexpr binary_int_t x_indicator_components_binary_format_version = 1;

    inline constexpr binary_int_t function_sample_layout_interval_tensor_time = 1;
    inline constexpr binary_int_t function_sample_layout_triangle_barycentric_time = 2;

    template<class... Ints>
    [[nodiscard]] inline std::vector<binary_int_t> make_versioned_header(
        binary_int_t version,
        Ints... values)
    {
        static_assert((std::is_integral_v<Ints> && ...),
                      "make_versioned_header expects integral payload entries.");

        std::vector<binary_int_t> header;
        header.reserve(sizeof...(Ints) + 1u);
        header.push_back(version);
        (header.push_back(static_cast<binary_int_t>(values)), ...);
        return header;
    }
}

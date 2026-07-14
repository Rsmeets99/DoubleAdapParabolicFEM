#pragma once

#include <filesystem>
#include <string>

#include "finite_element/io/detail/binary_format_versions.hpp"
#include "detail/write_binary_block.hpp"
#include "detail/write_mesh_binary_detail.hpp"

namespace finite_element
{
    template<typename GeomTraits,
             typename FETraits,
             typename Policy>
    void
    FESpace<GeomTraits, FETraits, Policy>::write_mesh_binary(
        const std::filesystem::path& output_dir,
        const std::string& filename) const
    {
        using int_t = finite_element::io::detail::binary_int_t;

        constexpr int dim_space_v = GT::dim_space_v;
        constexpr int dim_time_v  = GT::dim_time_v;

        constexpr int Tp_vertices = Types::n_spatial_vertices;
        constexpr int Ip_vertices = Types::n_temporal_vertices;
        constexpr int spatial_faces = Types::n_spatial_faces;
        constexpr int temporal_faces = Types::n_temporal_faces;
        constexpr int vertices_per_spatial_face = Types::n_spatial_face_vertices;
        constexpr int vertices_per_temporal_face = Types::n_spatial_vertices;

        const auto& mesh = mesh_ref();
        const auto& active_cells = this->active_cells();

        const int_t n_spatial_vertices = static_cast<int_t>(mesh.spatial_vertices().size());
        const int_t n_temporal_vertices = static_cast<int_t>(mesh.temporal_vertices().size());
        const int_t n_cells = static_cast<int_t>(active_cells.size());

        const auto spatial_coords =
            finite_element::io::detail::flatten_point_coordinates(mesh.spatial_vertices());
        const auto temporal_coords =
            finite_element::io::detail::flatten_point_coordinates(mesh.temporal_vertices());
        const auto cell_ids = finite_element::io::detail::build_mesh_cell_ids(*this);
        const auto spatial_vids = finite_element::io::detail::build_spatial_vertex_ids(*this);
        const auto temporal_vids = finite_element::io::detail::build_temporal_vertex_ids(*this);

        const auto spatial_neighbour_lists =
            finite_element::io::detail::build_spatial_neighbour_lists(*this);
        const auto temporal_neighbour_lists =
            finite_element::io::detail::build_temporal_neighbour_lists(*this);

        const int_t max_spatial_neigh =
            finite_element::io::detail::max_neighbours_per_face(active_cells, spatial_neighbour_lists);
        const int_t max_temporal_neigh =
            finite_element::io::detail::max_neighbours_per_face(active_cells, temporal_neighbour_lists);

        const auto spatial_neigh = finite_element::io::detail::flatten_neighbours(
            active_cells, spatial_neighbour_lists, spatial_faces, max_spatial_neigh);
        const auto temporal_neigh = finite_element::io::detail::flatten_neighbours(
            active_cells, temporal_neighbour_lists, temporal_faces, max_temporal_neigh);

        auto file = finite_element::io::detail::open_binary_output(output_dir, filename, "write_mesh_binary");

        const auto header = finite_element::io::detail::make_versioned_header(
            finite_element::io::detail::mesh_binary_format_version,
            n_spatial_vertices,
            n_temporal_vertices,
            static_cast<int_t>(dim_space_v),
            static_cast<int_t>(dim_time_v),
            n_cells,
            static_cast<int_t>(vertices_per_spatial_face),
            static_cast<int_t>(vertices_per_temporal_face),
            static_cast<int_t>(Tp_vertices),
            static_cast<int_t>(Ip_vertices),
            static_cast<int_t>(spatial_faces),
            max_spatial_neigh,
            static_cast<int_t>(temporal_faces),
            max_temporal_neigh);

        finite_element::io::detail::write_binary_block(file, header, "write_mesh_binary");
        finite_element::io::detail::write_binary_block(file, spatial_coords, "write_mesh_binary");
        finite_element::io::detail::write_binary_block(file, temporal_coords, "write_mesh_binary");
        finite_element::io::detail::write_binary_block(file, cell_ids, "write_mesh_binary");
        finite_element::io::detail::write_binary_block(file, spatial_vids, "write_mesh_binary");
        finite_element::io::detail::write_binary_block(file, temporal_vids, "write_mesh_binary");
        finite_element::io::detail::write_binary_block(file, spatial_neigh, "write_mesh_binary");
        finite_element::io::detail::write_binary_block(file, temporal_neigh, "write_mesh_binary");
    }
}

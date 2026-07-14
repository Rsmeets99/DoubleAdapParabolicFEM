#pragma once

#include <array>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "finite_element/io/detail/binary_format_versions.hpp"

namespace finite_element::io::detail
{
    template<class SpaceType>
    std::vector<binary_int_t> build_mesh_cell_ids(const SpaceType& space)
    {
        std::vector<binary_int_t> cell_ids;
        cell_ids.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
            cell_ids.push_back(static_cast<binary_int_t>(cell_id));

        return cell_ids;
    }

    template<class SpaceType>
    std::vector<binary_int_t> build_spatial_vertex_ids(const SpaceType& space)
    {
        constexpr int Tp_vertices = SpaceType::Types::n_spatial_vertices;

        std::vector<binary_int_t> spatial_vids;
        spatial_vids.reserve(space.active_cells().size() * Tp_vertices);

        for (const int cell_id : space.active_cells())
        {
            const auto& cell = space.mesh_ref().cell(cell_id);
            for (int v = 0; v < Tp_vertices; ++v)
                spatial_vids.push_back(static_cast<binary_int_t>(
                    cell.spatial_vertex_ids[static_cast<std::size_t>(v)]));
        }

        return spatial_vids;
    }

    template<class SpaceType>
    std::vector<binary_int_t> build_temporal_vertex_ids(const SpaceType& space)
    {
        constexpr int Ip_vertices = SpaceType::Types::n_temporal_vertices;

        std::vector<binary_int_t> temporal_vids;
        temporal_vids.reserve(space.active_cells().size() * Ip_vertices);

        for (const int cell_id : space.active_cells())
        {
            const auto& cell = space.mesh_ref().cell(cell_id);
            for (int v = 0; v < Ip_vertices; ++v)
                temporal_vids.push_back(static_cast<binary_int_t>(
                    cell.temporal_vertex_ids[static_cast<std::size_t>(v)]));
        }

        return temporal_vids;
    }

    template<class SpaceType>
    auto build_spatial_neighbour_lists(const SpaceType& space)
    {
        constexpr int spatial_faces = SpaceType::Types::n_spatial_faces;
        using SpatialNeighbourArray = std::array<std::vector<int>, spatial_faces>;

        std::unordered_map<int, SpatialNeighbourArray> neighbour_ids;
        neighbour_ids.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
            neighbour_ids.emplace(cell_id, SpatialNeighbourArray{});

        for (const auto& interface : space.adjacency_ref().spatial_interfaces)
        {
            if (interface.is_boundary)
                continue;

            const int c1 = interface.master_cell;
            const int f1 = interface.master_face;
            const int c2 = interface.slave_cell;
            const int f2 = interface.slave_face;

            if (!space.is_active_cell(c1) || c2 < 0 || !space.is_active_cell(c2))
                continue;

            neighbour_ids[c1][static_cast<std::size_t>(f1)].push_back(c2);
            neighbour_ids[c2][static_cast<std::size_t>(f2)].push_back(c1);
        }

        return neighbour_ids;
    }

    template<class SpaceType>
    auto build_temporal_neighbour_lists(const SpaceType& space)
    {
        constexpr int temporal_faces = SpaceType::Types::n_temporal_faces;
        using TemporalNeighbourArray = std::array<std::vector<int>, temporal_faces>;

        std::unordered_map<int, TemporalNeighbourArray> neighbour_ids;
        neighbour_ids.reserve(space.active_cells().size());

        for (const int cell_id : space.active_cells())
            neighbour_ids.emplace(cell_id, TemporalNeighbourArray{});

        for (const auto& interface : space.adjacency_ref().temporal_interfaces)
        {
            if (interface.is_boundary)
                continue;

            const int c1 = interface.master_cell;
            const int f1 = interface.master_face;
            const int c2 = interface.slave_cell;
            const int f2 = interface.slave_face;

            if (!space.is_active_cell(c1) || c2 < 0 || !space.is_active_cell(c2))
                continue;

            neighbour_ids[c1][static_cast<std::size_t>(f1)].push_back(c2);
            neighbour_ids[c2][static_cast<std::size_t>(f2)].push_back(c1);
        }

        return neighbour_ids;
    }

    template<class NeighbourMap>
    binary_int_t max_neighbours_per_face(const std::vector<int>& active_cells, const NeighbourMap& neighbour_map)
    {
        binary_int_t max_neigh = 0;

        for (const int cell_id : active_cells)
        {
            const auto it = neighbour_map.find(cell_id);
            if (it == neighbour_map.end())
                continue;

            for (const auto& face_neighbours : it->second)
            {
                const auto sz = static_cast<binary_int_t>(face_neighbours.size());
                if (sz > max_neigh)
                    max_neigh = sz;
            }
        }

        return max_neigh;
    }

    template<class NeighbourMap>
    std::vector<binary_int_t> flatten_neighbours(
        const std::vector<int>& active_cells,
        const NeighbourMap& neighbour_map,
        int n_faces,
        binary_int_t max_neigh_per_face)
    {
        std::vector<binary_int_t> flat;
        flat.reserve(
            active_cells.size()
            * static_cast<std::size_t>(n_faces)
            * static_cast<std::size_t>(max_neigh_per_face));

        for (const int cell_id : active_cells)
        {
            const auto it = neighbour_map.find(cell_id);

            for (int f = 0; f < n_faces; ++f)
            {
                if (it == neighbour_map.end())
                {
                    for (binary_int_t k = 0; k < max_neigh_per_face; ++k)
                        flat.push_back(static_cast<binary_int_t>(-1));
                    continue;
                }

                const auto& face_neighbours = it->second[static_cast<std::size_t>(f)];
                for (binary_int_t k = 0; k < max_neigh_per_face; ++k)
                {
                    const auto value =
                        (k < static_cast<binary_int_t>(face_neighbours.size()))
                            ? static_cast<binary_int_t>(face_neighbours[static_cast<std::size_t>(k)])
                            : static_cast<binary_int_t>(-1);
                    flat.push_back(value);
                }
            }
        }

        return flat;
    }
}

#pragma once

#include <array>
#include <functional>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "interface.hpp"

namespace mesh
{
    template<typename GeomTraits>
    class Mesh;
}

namespace finite_element::fespace
{
    template<typename GeomTraits, typename Policy>
    struct Adjacency
    {
        using MeshType = mesh::Mesh<GeomTraits>;
        using TimingCallback = std::function<void(std::string_view, double)>;

        std::vector<SpatialInterface<GeomTraits>> spatial_interfaces;
        std::vector<TemporalInterface<GeomTraits>> temporal_interfaces;

        std::unordered_map<int, std::array<std::vector<int>, GeomTraits::dim_space_v + 1>> cell_to_spatial;
        std::unordered_map<int, std::array<std::vector<int>, 2>> cell_to_temporal;

        void clear()
        {
            spatial_interfaces.clear();
            temporal_interfaces.clear();
            cell_to_spatial.clear();
            cell_to_temporal.clear();
        }

        void compute_adjacency(
            const std::vector<int>& active_cells,
            const MeshType& mesh,
            const TimingCallback& timing_callback = {});

        void compute_adjacency_incremental(
            const std::vector<int>& active_cells,
            const MeshType& mesh,
            const std::vector<int>& changed_cells,
            const TimingCallback& timing_callback = {});
    };
}

#include "adjacency_1d.hpp"
#include "adjacency_2d.hpp"

namespace finite_element::fespace
{
    template<typename GeomTraits, typename Policy>
    inline void Adjacency<GeomTraits, Policy>::compute_adjacency(
        const std::vector<int>& active_cells,
        const MeshType& mesh,
        const TimingCallback& timing_callback)
    {
        clear();

        if constexpr (GeomTraits::dim_space_v == 1)
        {
            static_cast<void>(timing_callback);
            compute_adjacency_1d<GeomTraits, Policy>(*this, active_cells, mesh);
        }
        else if constexpr (GeomTraits::dim_space_v == 2)
        {
            compute_adjacency_2d<GeomTraits, Policy>(
                *this,
                active_cells,
                mesh,
                timing_callback);
        }
        else
        {
            throw std::runtime_error("Adjacency is currently only implemented for dim_space_v == 1 or 2.");
        }
    }

    template<typename GeomTraits, typename Policy>
    inline void Adjacency<GeomTraits, Policy>::compute_adjacency_incremental(
        const std::vector<int>& active_cells,
        const MeshType& mesh,
        const std::vector<int>& changed_cells,
        const TimingCallback& timing_callback)
    {
        if constexpr (GeomTraits::dim_space_v == 2)
        {
            compute_adjacency_2d_incremental<GeomTraits, Policy>(
                *this,
                active_cells,
                mesh,
                changed_cells,
                timing_callback);
        }
        else
        {
            static_cast<void>(changed_cells);
            compute_adjacency(active_cells, mesh, timing_callback);
        }
    }
}

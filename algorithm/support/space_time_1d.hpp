#pragma once

#include <algorithm>
#include <string>
#include <vector>

#include "core/debug.hpp"
#include "mesh/mesh.hpp"
#include "mesh/mesh_traits.hpp"

namespace adaptive_algorithm::support
{
    template<class GeomTraits>
    [[nodiscard]] inline mesh::Mesh<GeomTraits> make_unit_mesh()
    {
        using MeshType = mesh::Mesh<GeomTraits>;
        using MeshTypes = mesh::MeshTypes<GeomTraits>;

        return MeshType(
            typename MeshTypes::SpatialPoint{0.0},
            typename MeshTypes::SpatialPoint{1.0},
            typename MeshTypes::TemporalPoint{0.0},
            typename MeshTypes::TemporalPoint{1.0});
    }

    template<class SpaceType>
    [[nodiscard]] inline SpaceType make_space(typename SpaceType::MeshType& mesh)
    {
        SpaceType space(mesh);
        space.initialize(mesh.leaf_cell_ids());
        return space;
    }

    template<class SpaceType>
    inline void refine_at_point(
        SpaceType& space,
        const typename SpaceType::SpaceTimePoint& p,
        const std::string& context = "refine_at_point")
    {
        const int cell_id = space.find_active_cell(p);
        core::require(cell_id >= 0, context + ": no active cell found for point.");
        space.refine(std::vector<int>{cell_id});
    }

    template<class SpaceType>
    [[nodiscard]] inline std::vector<int> unique_cells_containing_points(
        SpaceType& space,
        const std::vector<typename SpaceType::SpaceTimePoint>& points,
        const std::string& context = "unique_cells_containing_points")
    {
        std::vector<int> marked;
        marked.reserve(points.size());

        for (const auto& p : points)
        {
            const int cell_id = space.find_active_cell(p);
            core::require(cell_id >= 0, context + ": point not found in active cells.");
            marked.push_back(cell_id);
        }

        std::sort(marked.begin(), marked.end());
        marked.erase(std::unique(marked.begin(), marked.end()), marked.end());
        return marked;
    }

    template<class SpaceType>
    inline void refine_points_n_times(
        SpaceType& space,
        const std::vector<typename SpaceType::SpaceTimePoint>& points,
        int n_rounds,
        const std::string& context = "refine_points_n_times")
    {
        for (int round = 0; round < n_rounds; ++round)
        {
            for (const auto& p : points)
                refine_at_point(space, p, context);
        }
    }

    template<class SpaceType>
    inline void refine_uniform_levels(SpaceType& space, int n_levels)
    {
        for (int level = 0; level < n_levels; ++level)
        {
            const auto marked = space.active_cells();
            space.refine(marked);
        }
    }
}

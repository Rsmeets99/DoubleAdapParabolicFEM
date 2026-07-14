#pragma once

#include <unordered_map>
#include <vector>

#include "../../core/coord_key.hpp"
#include "../../core/ids.hpp"
#include "../types.hpp"

namespace mesh::detail
{
    template<typename GeomTraits>
    class VertexRegistry
    {
    public:
        using Types = MeshTypes<GeomTraits>;
        using SpatialPoint  = typename Types::SpatialPoint;
        using TemporalPoint = typename Types::TemporalPoint;
        using vertex_id_type = typename Types::vertex_id_type;

        void clear()
        {
            spatial_vertex_map_.clear();
            temporal_vertex_map_.clear();
        }

        [[nodiscard]] vertex_id_type get_or_create_spatial_vertex(
            std::vector<SpatialPoint>& spatial_vertices,
            const SpatialPoint& x)
        {
            const auto key = core::make_coord_key<GeomTraits::dim_space_v>(x);

            const auto it = spatial_vertex_map_.find(key);
            if (it != spatial_vertex_map_.end())
                return it->second;

            const auto new_id = static_cast<vertex_id_type>(spatial_vertices.size());
            spatial_vertices.push_back(x);
            spatial_vertex_map_.emplace(key, new_id);
            return new_id;
        }

        [[nodiscard]] vertex_id_type get_or_create_temporal_vertex(
            std::vector<TemporalPoint>& temporal_vertices,
            const TemporalPoint& t)
        {
            const auto key = core::make_coord_key<GeomTraits::dim_time_v>(t);

            const auto it = temporal_vertex_map_.find(key);
            if (it != temporal_vertex_map_.end())
                return it->second;

            const auto new_id = static_cast<vertex_id_type>(temporal_vertices.size());
            temporal_vertices.push_back(t);
            temporal_vertex_map_.emplace(key, new_id);
            return new_id;
        }

    private:
        std::unordered_map<
            core::CoordKey<GeomTraits::dim_space_v>,
            vertex_id_type> spatial_vertex_map_{};

        std::unordered_map<
            core::CoordKey<GeomTraits::dim_time_v>,
            vertex_id_type> temporal_vertex_map_{};
    };
}
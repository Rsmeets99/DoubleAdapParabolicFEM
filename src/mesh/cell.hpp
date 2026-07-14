#pragma once

#include <cmath>
#include <array>
#include <ostream>
#include <vector>

#include "refinement/refinement_type.hpp"
#include "types.hpp"

namespace mesh
{
    template<typename GeomTraits>
    struct SpatialFaceData
    {
        using Types = MeshTypes<GeomTraits>;

        using SpatialFaceVertexIds = typename Types::SpatialFaceVertexIds;
        using TemporalVertexIds    = typename Types::TemporalVertexIds;

        SpatialFaceVertexIds spatial_vertex_ids{};
        TemporalVertexIds temporal_vertex_ids{};

        bool operator==(const SpatialFaceData&) const noexcept = default;
    };

    template<typename GeomTraits>
    struct TemporalFaceData
    {
        using Types = MeshTypes<GeomTraits>;

        using SpatialVertexIds = typename Types::SpatialVertexIds;
        using vertex_id_type   = typename Types::vertex_id_type;

        SpatialVertexIds spatial_vertex_ids{};
        vertex_id_type temporal_vertex_id = -1;

        bool operator==(const TemporalFaceData&) const noexcept = default;
    };

    template<typename GeomTraits>
    struct Cell
    {
        using Types = MeshTypes<GeomTraits>;

        static constexpr int dim_space_v = Types::dim_space_v;
        static constexpr int dim_time_v  = Types::dim_time_v;

        static constexpr int n_spatial_vertices      = Types::n_spatial_vertices;
        static constexpr int n_temporal_vertices     = Types::n_temporal_vertices;
        static constexpr int n_spatial_faces         = Types::n_spatial_faces;
        static constexpr int n_temporal_faces        = Types::n_temporal_faces;

        using cell_id_type         = typename Types::cell_id_type;
        using vertex_id_type       = typename Types::vertex_id_type;
        using SpatialPoint         = typename Types::SpatialPoint;
        using TemporalPoint        = typename Types::TemporalPoint;
        using SpaceTimePoint       = typename Types::SpaceTimePoint;
        using SpatialVertexIds     = typename Types::SpatialVertexIds;
        using TemporalVertexIds    = typename Types::TemporalVertexIds;
        using LocalEdgeIndices     = typename Types::LocalEdgeIndices;
        using SpatialFaceType      = SpatialFaceData<GeomTraits>;
        using TemporalFaceType     = TemporalFaceData<GeomTraits>;

        cell_id_type cell_id   = -1;
        cell_id_type parent_id = -1;

        std::vector<cell_id_type> children{};

        // Storage-tree leaf marker only. A storage leaf is not the same as
        // being active in every FE space: FE spaces can use active antichains
        // that include non-leaf storage cells.
        bool is_leaf = true;

        // Global split count that determines the next split type via the policy.
        int generation = 0;

        // Separate counters for spatial and temporal refinement history.
        int spatial_level  = 0;
        int temporal_level = 0;

        // Ordered spatial simplex vertices. For NVB-style methods, this ordering matters.
        SpatialVertexIds spatial_vertex_ids{};

        // Temporal interval endpoints.
        TemporalVertexIds temporal_vertex_ids{};

        // Local vertex indices of the current spatial refinement edge inside spatial_vertex_ids.
        // For 1D this is always {0,1}. For d>=2 this is where NVB metadata can live.
        LocalEdgeIndices spatial_refinement_edge_local{ {0, 1} };

        RefinementType last_split_type = RefinementType::none;

        std::array<SpatialFaceType, n_spatial_faces> spatial_faces{};
        std::array<TemporalFaceType, n_temporal_faces> temporal_faces{};

        std::array<bool, n_spatial_faces> spatial_boundary{};
        std::array<bool, n_temporal_faces> temporal_boundary{};

        [[nodiscard]] bool has_parent() const noexcept
        {
            return parent_id >= 0;
        }
    };

    template<typename GeomTraits>
    [[nodiscard]] inline bool contains_coord_1d(
        const Cell<GeomTraits>& cell,
        const std::vector<typename Cell<GeomTraits>::SpatialPoint>& spatial_vertices,
        const std::vector<typename Cell<GeomTraits>::TemporalPoint>& temporal_vertices,
        const typename Cell<GeomTraits>::SpaceTimePoint& p)
    {
        static_assert(Cell<GeomTraits>::dim_space_v == 1,
                      "contains_coord_1d requires dim_space_v == 1.");
        static_assert(Cell<GeomTraits>::dim_time_v == 1,
                      "contains_coord_1d requires dim_time_v == 1.");

        const double xl = spatial_vertices[cell.spatial_vertex_ids[0]][0];
        const double xr = spatial_vertices[cell.spatial_vertex_ids[1]][0];

        const double t0 = temporal_vertices[cell.temporal_vertex_ids[0]][0];
        const double t1 = temporal_vertices[cell.temporal_vertex_ids[1]][0];

        return (xl <= p[0] && p[0] <= xr) && (t0 <= p[1] && p[1] <= t1);
    }

    template<typename GeomTraits>
    [[nodiscard]] inline bool contains_coord(
        const Cell<GeomTraits>& cell,
        const std::vector<typename Cell<GeomTraits>::SpatialPoint>& spatial_vertices,
        const std::vector<typename Cell<GeomTraits>::TemporalPoint>& temporal_vertices,
        const typename Cell<GeomTraits>::SpaceTimePoint& p)
    {
        if constexpr (Cell<GeomTraits>::dim_space_v == 1 &&
                      Cell<GeomTraits>::dim_time_v == 1)
        {
            return contains_coord_1d(cell, spatial_vertices, temporal_vertices, p);
        }
        else if constexpr (Cell<GeomTraits>::dim_space_v == 2 &&
                           Cell<GeomTraits>::dim_time_v == 1)
        {
            constexpr double tol = 1.0e-12;

            const auto& v0 = spatial_vertices[cell.spatial_vertex_ids[0]];
            const auto& v1 = spatial_vertices[cell.spatial_vertex_ids[1]];
            const auto& v2 = spatial_vertices[cell.spatial_vertex_ids[2]];

            const double J00 = v1[0] - v0[0];
            const double J01 = v2[0] - v0[0];
            const double J10 = v1[1] - v0[1];
            const double J11 = v2[1] - v0[1];
            const double det = J00 * J11 - J01 * J10;

            if (std::abs(det) < 1.0e-15)
                return false;

            const double dx = p[0] - v0[0];
            const double dy = p[1] - v0[1];
            const double inv_det = 1.0 / det;
            const double xi  = ( J11 * dx - J01 * dy) * inv_det;
            const double eta = (-J10 * dx + J00 * dy) * inv_det;

            const double t0 = temporal_vertices[cell.temporal_vertex_ids[0]][0];
            const double t1 = temporal_vertices[cell.temporal_vertex_ids[1]][0];

            return (-tol <= xi) &&
                   (-tol <= eta) &&
                   (xi + eta <= 1.0 + tol) &&
                   (t0 - tol <= p[2] && p[2] <= t1 + tol);
        }
        else
        {
            static_assert((Cell<GeomTraits>::dim_space_v == 1 ||
                           Cell<GeomTraits>::dim_space_v == 2) &&
                          Cell<GeomTraits>::dim_time_v == 1,
                          "contains_coord is currently only implemented for 1+1D or 2+1D.");
            return false;
        }
    }

    template<typename GeomTraits>
    inline std::ostream& operator<<(std::ostream& os, const Cell<GeomTraits>& cell)
    {
        os << "Cell(id=" << cell.cell_id
           << ", parent=" << cell.parent_id
           << ", is_leaf=" << cell.is_leaf
           << ", gen=" << cell.generation
           << ", slevel=" << cell.spatial_level
           << ", tlevel=" << cell.temporal_level
           << ", last_split_type=" << static_cast<int>(cell.last_split_type)
           << ", spatial_vertex_ids=[";

        for (std::size_t i = 0; i < cell.spatial_vertex_ids.size(); ++i)
        {
            os << cell.spatial_vertex_ids[i];
            if (i + 1 < cell.spatial_vertex_ids.size())
                os << ", ";
        }

        os << "], temporal_vertex_ids=[";

        for (std::size_t i = 0; i < cell.temporal_vertex_ids.size(); ++i)
        {
            os << cell.temporal_vertex_ids[i];
            if (i + 1 < cell.temporal_vertex_ids.size())
                os << ", ";
        }

        os << "], refinement_edge_local=["
           << cell.spatial_refinement_edge_local[0] << ", "
           << cell.spatial_refinement_edge_local[1] << "])";

        return os;
    }
}

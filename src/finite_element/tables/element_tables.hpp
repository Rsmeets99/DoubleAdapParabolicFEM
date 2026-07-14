#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#include "../basis/polynomials/segment_lagrange.hpp"    
#include "../basis/polynomials/triangular_lagrange.hpp"
#include "../basis/functions/quadrilateral_lagrange.hpp"

// -----------------------------------------------------------------------------
// Element DOF tables for tensor-product FE
// Supports:
//   - 1+1D quadrilaterals
//   - 2+1D triangular prisms
//
// Conventions:
//   dim == 1:
//     spatial faces : x = 0, x = 1
//     temporal faces: t = 0, t = 1
//
//   dim == 2:
//     reference triangle vertices:
//       v0 = (0,0)
//       v1 = (1,0)
//       v2 = (0,1)
//
//     spatial face numbering/orientation:
//       face 0: (0,0) -> (1,0)
//       face 1: (1,0) -> (0,1)
//       face 2: (0,1) -> (0,0)
//
//     temporal face 0: t-index == 0
//     temporal face 1: t-index == p_time
//
// Indexing matches basis-function ordering:
//   Quad:              local = i_space * (p_time+1) + j_time
//   Triangular prism:  local = i_tri   * (p_time+1) + j_time
//
// Stored identifiers:
//   - spatial_node_ids[local]  = spatial node number in the underlying spatial basis
//   - temporal_node_ids[local] = time node number in the underlying 1D basis
// -----------------------------------------------------------------------------

namespace finite_element::tables
{
    template<typename GeoTraits, typename FETraits>
    struct ElementDofTables
    {
        static constexpr int dim     = GeoTraits::dim_space_v;
        static constexpr int p_space = FETraits::p_space_v;
        static constexpr int p_time  = FETraits::p_time_v;

        using SpatialNodes  = typename FETraits::SpatialNodes;
        using TemporalNodes = typename FETraits::TemporalNodes;

        static_assert(dim == 1 || dim == 2,
                    "ElementDofTables only supports dim_space = 1 or 2.");
        static_assert(p_space >= 1,
                    "ElementDofTables requires p_space >= 1.");
        static_assert(p_time >= 1,
                    "ElementDofTables requires p_time >= 1.");

        static constexpr int spatial_dofs           = FETraits::total_spatial_dofs;
        static constexpr int temporal_dofs          = FETraits::total_temporal_dofs;
        static constexpr int dofs_per_cell          = FETraits::dofs_per_cell;
        static constexpr int dofs_per_spatial_face  = FETraits::dofs_per_spatial_face;
        static constexpr int dofs_per_temporal_face = FETraits::dofs_per_temporal_face;
        static constexpr int dofs_per_interior      = FETraits::dofs_per_interior;

        using Coord = std::array<double, dim + 1>;

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        enum class BoundaryType
        {
            Interior,
            Vertex,
            SpatialFace,
            TemporalFace,
            SpatialEdge,
            TemporalEdge
        };

        struct DofMeta
        {
            BoundaryType type = BoundaryType::Interior;

            // For dim==1 only spatial_faces[0] is used.
            // For dim==2, a DOF can lie on up to two spatial faces.
            std::array<int, 2> spatial_faces{-1, -1};
            int num_spatial_faces = 0;

            // -1 if not on temporal boundary, otherwise 0 or 1.
            int temporal_face = -1;

            bool is_vertex = false;
        };

        struct Tables
        {
            std::array<Coord, dofs_per_cell> coords{};
            std::array<int, dofs_per_cell> spatial_node_ids{};
            std::array<int, dofs_per_cell> temporal_node_ids{};
            std::array<DofMeta, dofs_per_cell> meta{};

            std::array<std::array<int, dofs_per_spatial_face>, (dim == 1 ? 2 : 3)> spatial_face_dofs{};
            std::array<std::array<int, dofs_per_temporal_face>, 2> temporal_face_dofs{};

            std::array<int, GeoTraits::vertices_per_cell> vertex_dofs{};
            int num_vertex_dofs = 0;

            std::array<int, dofs_per_interior> interior_dofs{};
            int num_interior_dofs = 0;
        };

    private:
        static constexpr void initialize_tables(Tables& t)
        {
            for (auto& c : t.coords)
                for (auto& x : c)
                    x = 0.0;

            for (auto& x : t.spatial_node_ids)
                x = -1;

            for (auto& x : t.temporal_node_ids)
                x = -1;

            for (auto& m : t.meta)
            {
                m.type = BoundaryType::Interior;
                m.spatial_faces = {-1, -1};
                m.num_spatial_faces = 0;
                m.temporal_face = -1;
                m.is_vertex = false;
            }

            for (auto& face : t.spatial_face_dofs)
                for (auto& x : face)
                    x = -1;

            for (auto& face : t.temporal_face_dofs)
                for (auto& x : face)
                    x = -1;

            for (auto& x : t.vertex_dofs)
                x = -1;

            for (auto& x : t.interior_dofs)
                x = -1;

            t.num_vertex_dofs = 0;
            t.num_interior_dofs = 0;
        }

        static constexpr Tables generate()
        {
            Tables t{};
            initialize_tables(t);

            int vertex_counter = 0;
            int interior_counter = 0;

            if constexpr (dim == 1)
            {
                // 1+1D quadrilateral
                //
                // local = i_space * (p_time+1) + j_time
                constexpr auto x_nodes = basis::polynomials::SegmentLagrangeBasis<p_space, SpatialNodes>::nodes;
                constexpr auto t_nodes = basis::polynomials::SegmentLagrangeBasis<p_time, TemporalNodes>::nodes;

                for (int i_space = 0; i_space <= p_space; ++i_space)
                {
                    for (int j_time = 0; j_time <= p_time; ++j_time)
                    {
                        const int local_index = i_space * (p_time + 1) + j_time;

                        t.coords[index(local_index)]            = {x_nodes[index(i_space)], t_nodes[index(j_time)]};
                        t.spatial_node_ids[index(local_index)]  = i_space;
                        t.temporal_node_ids[index(local_index)] = j_time;

                        DofMeta& m = t.meta[index(local_index)];

                        const bool on_left   = (i_space == 0);
                        const bool on_right  = (i_space == p_space);
                        const bool on_bottom = (j_time == 0);
                        const bool on_top    = (j_time == p_time);

                        if (on_left)
                        {
                            m.spatial_faces[0U] = 0;
                            m.num_spatial_faces = 1;
                            t.spatial_face_dofs[0U][index(j_time)] = local_index;
                        }
                        if (on_right)
                        {
                            m.spatial_faces[0U] = 1;
                            m.num_spatial_faces = 1;
                            t.spatial_face_dofs[1U][index(j_time)] = local_index;
                        }

                        if (on_bottom)
                        {
                            m.temporal_face = 0;
                            t.temporal_face_dofs[0U][index(i_space)] = local_index;
                        }
                        if (on_top)
                        {
                            m.temporal_face = 1;
                            t.temporal_face_dofs[1U][index(i_space)] = local_index;
                        }

                        m.is_vertex = (on_left || on_right) && (on_bottom || on_top);

                        if (m.is_vertex)
                            m.type = BoundaryType::Vertex;
                        else if (m.num_spatial_faces == 1)
                            m.type = BoundaryType::SpatialFace;
                        else if (m.temporal_face != -1)
                            m.type = BoundaryType::TemporalFace;
                        else
                            m.type = BoundaryType::Interior;

                        if (m.is_vertex)
                            t.vertex_dofs[index(vertex_counter++)] = local_index;

                        if (!on_left && !on_right && !on_bottom && !on_top)
                            t.interior_dofs[index(interior_counter++)] = local_index;
                    }
                }
            }
            else if constexpr (dim == 2)
            {
                static_assert(dofs_per_spatial_face == (p_space + 1) * (p_time + 1),
                            "Wrong dofs_per_spatial_face for triangular prism.");
                static_assert(dofs_per_temporal_face == SpatialNodes::N,
                            "Wrong dofs_per_temporal_face for triangular prism.");

                constexpr auto tri_pts   = SpatialNodes::points;
                constexpr auto tri_meta  = SpatialNodes::node_meta;
                constexpr auto face_ord  = SpatialNodes::face_ordinal;
                constexpr int  N_tri     = SpatialNodes::N;
                constexpr int  N_time    = p_time + 1;
                constexpr auto t_nodes   = basis::polynomials::SegmentLagrangeBasis<p_time, TemporalNodes>::nodes;

                for (int i_tri = 0; i_tri < N_tri; ++i_tri)
                {
                    const double x = tri_pts[index(i_tri)][0U];
                    const double y = tri_pts[index(i_tri)][1U];

                    const auto spatial_meta = tri_meta[index(i_tri)];
                    const int num_spatial_faces = spatial_meta.num_spatial_faces;
                    const auto face_list = spatial_meta.spatial_faces;

                    for (int j_time = 0; j_time < N_time; ++j_time)
                    {
                        const int local_index = i_tri * N_time + j_time;

                        t.coords[index(local_index)]            = {x, y, t_nodes[index(j_time)]};
                        t.spatial_node_ids[index(local_index)]  = i_tri;
                        t.temporal_node_ids[index(local_index)] = j_time;

                        DofMeta& m = t.meta[index(local_index)];
                        m.num_spatial_faces = num_spatial_faces;
                        m.spatial_faces = face_list;
                        m.temporal_face = (j_time == 0) ? 0 : ((j_time == p_time) ? 1 : -1);
                        m.is_vertex = (spatial_meta.vertex >= 0 && m.temporal_face != -1);

                        if (m.is_vertex)
                            m.type = BoundaryType::Vertex;
                        else if (num_spatial_faces == 2)
                            m.type = BoundaryType::TemporalEdge;
                        else if (num_spatial_faces == 1 && m.temporal_face != -1)
                            m.type = BoundaryType::SpatialEdge;
                        else if (num_spatial_faces == 1)
                            m.type = BoundaryType::SpatialFace;
                        else if (m.temporal_face != -1)
                            m.type = BoundaryType::TemporalFace;
                        else
                            m.type = BoundaryType::Interior;

                        for (int k = 0; k < num_spatial_faces; ++k)
                        {
                            const int face = face_list[index(k)];
                            const int edge_ord = face_ord[index(i_tri)][index(face)];

                            assert(edge_ord >= 0);
                            assert(edge_ord <= p_space);

                            const int face_local = edge_ord * N_time + j_time;
                            t.spatial_face_dofs[index(face)][index(face_local)] = local_index;
                        }

                        if (m.temporal_face != -1)
                            t.temporal_face_dofs[index(m.temporal_face)][index(i_tri)] = local_index;

                        if (m.is_vertex)
                            t.vertex_dofs[index(vertex_counter++)] = local_index;

                        if (num_spatial_faces == 0 && m.temporal_face == -1)
                            t.interior_dofs[index(interior_counter++)] = local_index;
                    }
                }
            }

            assert(vertex_counter <= GeoTraits::vertices_per_cell);
            assert(interior_counter <= dofs_per_interior);

            t.num_vertex_dofs = vertex_counter;
            t.num_interior_dofs = interior_counter;
            return t;
        }

    public:
        static constexpr Tables tables = generate();

        static constexpr const Coord& coord(int local_index)
        {
            return tables.coords[index(local_index)];
        }

        static constexpr int spatial_node_id(int local_index)
        {
            return tables.spatial_node_ids[index(local_index)];
        }

        static constexpr int temporal_node_id(int local_index)
        {
            return tables.temporal_node_ids[index(local_index)];
        }

        static constexpr const DofMeta& meta(int local_index)
        {
            return tables.meta[index(local_index)];
        }

        static constexpr const auto& spatial_face_dofs(int face_index)
        {
            return tables.spatial_face_dofs[index(face_index)];
        }

        static constexpr const auto& temporal_face_dofs(int face_index)
        {
            return tables.temporal_face_dofs[index(face_index)];
        }

        static constexpr const auto& vertex_dofs()
        {
            return tables.vertex_dofs;
        }

        static constexpr const auto& interior_dofs()
        {
            return tables.interior_dofs;
        }

        static constexpr int num_vertex_dofs()
        {
            return tables.num_vertex_dofs;
        }

        static constexpr int num_interior_dofs()
        {
            return tables.num_interior_dofs;
        }

        static constexpr bool on_spatial_face(int local_index, int face_index)
        {
            const auto& m = meta(local_index);
            for (int k = 0; k < m.num_spatial_faces; ++k)
                if (m.spatial_faces[index(k)] == face_index)
                    return true;
            return false;
        }

        static constexpr bool on_temporal_face(int local_index, int face_index)
        {
            return meta(local_index).temporal_face == face_index;
        }
    };

    template<typename G, typename F>
    constexpr typename ElementDofTables<G, F>::Tables
    ElementDofTables<G, F>::tables;
}

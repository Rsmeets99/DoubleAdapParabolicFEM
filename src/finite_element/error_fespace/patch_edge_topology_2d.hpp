#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../mesh/mesh.hpp"
#include "../../mesh/topology/boundary_2d.hpp"
#include "../time_slabs/time_slab_vertex_patch_builder.hpp"

namespace finite_element::error_fespace
{
    enum class PatchEdgeKind2D
    {
        internal,
        patch_boundary
    };

    template<typename GeomTraits, typename FETraits>
    struct PatchEdgeTopology2D
    {
        using GT            = GeomTraits;
        using FETraitsType  = FETraits;
        using Types         = mesh::MeshTypes<GeomTraits>;
        using EdgeVertexIds = typename Types::SpatialFaceVertexIds;
        using SpatialPoint  = typename Types::SpatialPoint;

        struct IncidentCell
        {
            int patch_cell_index = -1;
            int slab_cell_id     = -1;
            int local_face_id    = -1;
            int orientation_sign = 0;
        };

        struct Edge
        {
            int edge_id = -1;
            EdgeVertexIds canonical_vertex_ids{};
            SpatialPoint canonical_normal{};
            double length = 0.0;

            PatchEdgeKind2D kind = PatchEdgeKind2D::patch_boundary;
            bool on_physical_boundary = false;

            std::vector<IncidentCell> incident_cells{};

            [[nodiscard]] bool is_internal() const noexcept
            {
                return kind == PatchEdgeKind2D::internal;
            }

            [[nodiscard]] bool is_patch_boundary() const noexcept
            {
                return kind == PatchEdgeKind2D::patch_boundary;
            }

            [[nodiscard]] int n_incident_cells() const noexcept
            {
                return static_cast<int>(incident_cells.size());
            }
        };

        std::vector<Edge> edges{};
        std::vector<int> internal_edge_ids{};
        std::vector<int> patch_boundary_edge_ids{};
        std::vector<int> physical_boundary_edge_ids{};
        std::vector<int> artificial_boundary_edge_ids{};

        [[nodiscard]] int n_edges() const noexcept
        {
            return static_cast<int>(edges.size());
        }

        [[nodiscard]] const Edge& edge(int edge_id) const
        {
            if (edge_id < 0 || static_cast<std::size_t>(edge_id) >= edges.size())
                throw std::runtime_error("PatchEdgeTopology2D::edge: edge id out of range.");
            return edges[static_cast<std::size_t>(edge_id)];
        }

        [[nodiscard]] int edge_id_for_patch_cell_face(
            int patch_cell_index,
            int local_face_id) const noexcept
        {
            for (const auto& patch_edge : edges)
            {
                for (const auto& incident : patch_edge.incident_cells)
                {
                    if (incident.patch_cell_index == patch_cell_index &&
                        incident.local_face_id == local_face_id)
                    {
                        return patch_edge.edge_id;
                    }
                }
            }

            return -1;
        }

        [[nodiscard]] const Edge* edge_for_patch_cell_face(
            int patch_cell_index,
            int local_face_id) const noexcept
        {
            const int edge_id = edge_id_for_patch_cell_face(
                patch_cell_index,
                local_face_id);
            if (edge_id < 0)
                return nullptr;
            return &edges[static_cast<std::size_t>(edge_id)];
        }
    };

    template<typename GeomTraits, typename FETraits>
    class PatchEdgeTopologyBuilder2D
    {
    public:
        using Types         = mesh::MeshTypes<GeomTraits>;
        using MeshType      = mesh::Mesh<GeomTraits>;
        using SlabSpaceType = time_slabs::TimeSlabSpace<GeomTraits, FETraits>;
        using PatchType     = time_slabs::TimeSlabVertexPatch<GeomTraits, FETraits>;
        using TopologyType  = PatchEdgeTopology2D<GeomTraits, FETraits>;
        using EdgeType      = typename TopologyType::Edge;
        using IncidentCell  = typename TopologyType::IncidentCell;
        using EdgeVertexIds = typename TopologyType::EdgeVertexIds;
        using SpatialPoint  = typename TopologyType::SpatialPoint;

        [[nodiscard]] static TopologyType build(
            const PatchType& patch,
            const SlabSpaceType& slab_space)
        {
            static_assert(GeomTraits::dim_space_v == 2,
                          "PatchEdgeTopologyBuilder2D requires dim_space_v == 2.");
            static_assert(GeomTraits::dim_time_v == 1,
                          "PatchEdgeTopologyBuilder2D requires dim_time_v == 1.");

            if (patch.slab_id < 0 || patch.slab_id >= slab_space.n_slabs())
            {
                throw std::runtime_error(
                    "PatchEdgeTopologyBuilder2D::build: patch slab id out of range.");
            }

            TopologyType topology;
            const auto& slab_mesh = slab_space.slab(patch.slab_id).mesh_ref();

            for (int patch_cell_index = 0;
                 patch_cell_index < patch.n_cells;
                 ++patch_cell_index)
            {
                const auto& patch_cell = patch.cell(patch_cell_index);
                const auto& slab_cell = slab_mesh.cell(patch_cell.slab_cell_id);

                for (int local_face_id = 0;
                     local_face_id < Types::n_spatial_faces;
                     ++local_face_id)
                {
                    const auto edge_vertices =
                        slab_cell.spatial_faces[static_cast<std::size_t>(local_face_id)]
                            .spatial_vertex_ids;
                    const auto canonical_vertices = canonical_edge_vertices_(edge_vertices);

                    EdgeType* edge = find_edge_(topology, canonical_vertices);
                    if (edge == nullptr)
                    {
                        EdgeType new_edge;
                        new_edge.canonical_vertex_ids = canonical_vertices;
                        new_edge.canonical_normal =
                            canonical_normal_(slab_mesh, canonical_vertices);
                        new_edge.length = edge_length_(slab_mesh, canonical_vertices);
                        topology.edges.push_back(std::move(new_edge));
                        edge = &topology.edges.back();
                    }

                    IncidentCell incident;
                    incident.patch_cell_index = patch_cell_index;
                    incident.slab_cell_id = patch_cell.slab_cell_id;
                    incident.local_face_id = local_face_id;
                    incident.orientation_sign =
                        orientation_sign_(
                            slab_mesh,
                            patch_cell.slab_cell_id,
                            local_face_id,
                            edge->canonical_normal);
                    edge->incident_cells.push_back(incident);
                }
            }

            finalize_(topology, slab_mesh);
            return topology;
        }

    private:
        [[nodiscard]] static EdgeVertexIds canonical_edge_vertices_(
            EdgeVertexIds edge_vertices)
        {
            std::sort(edge_vertices.begin(), edge_vertices.end());
            return edge_vertices;
        }

        [[nodiscard]] static EdgeType* find_edge_(
            TopologyType& topology,
            const EdgeVertexIds& canonical_vertices) noexcept
        {
            for (auto& edge : topology.edges)
            {
                if (edge.canonical_vertex_ids == canonical_vertices)
                    return &edge;
            }

            return nullptr;
        }

        [[nodiscard]] static const EdgeType* find_edge_(
            const TopologyType& topology,
            const EdgeVertexIds& canonical_vertices) noexcept
        {
            for (const auto& edge : topology.edges)
            {
                if (edge.canonical_vertex_ids == canonical_vertices)
                    return &edge;
            }

            return nullptr;
        }

        [[nodiscard]] static std::array<double, 2> right_normal_(
            const SpatialPoint& a,
            const SpatialPoint& b)
        {
            const double dx = b[0] - a[0];
            const double dy = b[1] - a[1];
            const double length = std::sqrt(dx * dx + dy * dy);

            if (length <= 1.0e-15)
                throw std::runtime_error(
                    "PatchEdgeTopologyBuilder2D: degenerate spatial edge.");

            return {dy / length, -dx / length};
        }

        [[nodiscard]] static double edge_length_(
            const MeshType& mesh,
            const EdgeVertexIds& canonical_vertices)
        {
            const auto& a =
                mesh.spatial_vertices()[static_cast<std::size_t>(canonical_vertices[0])];
            const auto& b =
                mesh.spatial_vertices()[static_cast<std::size_t>(canonical_vertices[1])];

            const double dx = b[0] - a[0];
            const double dy = b[1] - a[1];
            return std::sqrt(dx * dx + dy * dy);
        }

        [[nodiscard]] static SpatialPoint canonical_normal_(
            const MeshType& mesh,
            const EdgeVertexIds& canonical_vertices)
        {
            const auto& a =
                mesh.spatial_vertices()[static_cast<std::size_t>(canonical_vertices[0])];
            const auto& b =
                mesh.spatial_vertices()[static_cast<std::size_t>(canonical_vertices[1])];

            const auto normal = right_normal_(a, b);
            return SpatialPoint{normal[0], normal[1]};
        }

        [[nodiscard]] static std::array<double, 2> local_outward_normal_(
            const MeshType& mesh,
            int slab_cell_id,
            int local_face_id)
        {
            const auto& cell = mesh.cell(slab_cell_id);
            const auto edge_vertices =
                cell.spatial_faces[static_cast<std::size_t>(local_face_id)]
                    .spatial_vertex_ids;

            const auto& a =
                mesh.spatial_vertices()[static_cast<std::size_t>(edge_vertices[0])];
            const auto& b =
                mesh.spatial_vertices()[static_cast<std::size_t>(edge_vertices[1])];

            return right_normal_(a, b);
        }

        [[nodiscard]] static int orientation_sign_(
            const MeshType& mesh,
            int slab_cell_id,
            int local_face_id,
            const SpatialPoint& canonical_normal)
        {
            const auto outward =
                local_outward_normal_(mesh, slab_cell_id, local_face_id);

            const double dot =
                outward[0] * canonical_normal[0] +
                outward[1] * canonical_normal[1];

            if (dot > 1.0e-10)
                return 1;
            if (dot < -1.0e-10)
                return -1;

            throw std::runtime_error(
                "PatchEdgeTopologyBuilder2D: local outward normal is not aligned with canonical normal.");
        }

        [[nodiscard]] static bool incident_face_lies_on_physical_boundary_(
            const MeshType& mesh,
            const IncidentCell& incident)
        {
            const auto& cell = mesh.cell(incident.slab_cell_id);
            if (cell.spatial_boundary[static_cast<std::size_t>(incident.local_face_id)])
                return true;

            const auto edge =
                canonical_edge_vertices_(
                    cell.spatial_faces[static_cast<std::size_t>(incident.local_face_id)]
                        .spatial_vertex_ids);

            return mesh::topology::spatial_edge_lies_on_mesh_boundary_2d<GeomTraits>(
                edge,
                mesh.spatial_boundary_face_vertex_ids(),
                mesh.spatial_vertices());
        }

        [[nodiscard]] static bool edge_lies_on_physical_boundary_(
            const MeshType& mesh,
            const EdgeType& edge)
        {
            for (const auto& incident : edge.incident_cells)
            {
                if (incident_face_lies_on_physical_boundary_(mesh, incident))
                    return true;
            }

            return false;
        }

        static void finalize_(TopologyType& topology, const MeshType& slab_mesh)
        {
            std::sort(
                topology.edges.begin(),
                topology.edges.end(),
                [](const EdgeType& a, const EdgeType& b)
                {
                    return a.canonical_vertex_ids < b.canonical_vertex_ids;
                });

            topology.internal_edge_ids.clear();
            topology.patch_boundary_edge_ids.clear();
            topology.physical_boundary_edge_ids.clear();
            topology.artificial_boundary_edge_ids.clear();

            for (int edge_id = 0;
                 edge_id < static_cast<int>(topology.edges.size());
                 ++edge_id)
            {
                auto& edge = topology.edges[static_cast<std::size_t>(edge_id)];
                edge.edge_id = edge_id;

                std::sort(
                    edge.incident_cells.begin(),
                    edge.incident_cells.end(),
                    [](const IncidentCell& a, const IncidentCell& b)
                    {
                        if (a.patch_cell_index != b.patch_cell_index)
                            return a.patch_cell_index < b.patch_cell_index;
                        return a.local_face_id < b.local_face_id;
                    });

                const int n_incident =
                    static_cast<int>(edge.incident_cells.size());
                if (n_incident == 1)
                {
                    edge.kind = PatchEdgeKind2D::patch_boundary;
                    edge.on_physical_boundary =
                        edge_lies_on_physical_boundary_(slab_mesh, edge);
                    topology.patch_boundary_edge_ids.push_back(edge_id);

                    if (edge.on_physical_boundary)
                        topology.physical_boundary_edge_ids.push_back(edge_id);
                    else
                        topology.artificial_boundary_edge_ids.push_back(edge_id);
                }
                else if (n_incident == 2)
                {
                    edge.kind = PatchEdgeKind2D::internal;
                    edge.on_physical_boundary = false;
                    topology.internal_edge_ids.push_back(edge_id);
                }
                else
                {
                    throw std::runtime_error(
                        "PatchEdgeTopologyBuilder2D: patch edge has invalid incident-cell count.");
                }
            }
        }
    };
}

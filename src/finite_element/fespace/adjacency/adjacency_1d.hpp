#pragma once

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

#include "interface.hpp"
#include "../../../mesh/mesh.hpp"
#include "../../../mesh/topology/interval_relations.hpp"

namespace finite_element::fespace
{
    template<typename GeomTraits, typename Policy>
    struct Adjacency;

    namespace detail::adjacency_impl
    {
        template<typename GeomTraits>
        struct FaceIntervalRecord1D
        {
            int cell_id   = -1;
            int face_id   = -1;
            int fixed_id  = -1; // x-vertex for spatial faces, t-vertex for temporal faces

            int left_id   = -1;
            int right_id  = -1;

            double left   = 0.0;
            double right  = 0.0;
        };

        template<typename GeomTraits>
        [[nodiscard]] inline bool same_interval(
            const FaceIntervalRecord1D<GeomTraits>& a,
            const FaceIntervalRecord1D<GeomTraits>& b) noexcept
        {
            return a.left_id == b.left_id && a.right_id == b.right_id;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline bool contains_interval(
            const FaceIntervalRecord1D<GeomTraits>& outer,
            const FaceIntervalRecord1D<GeomTraits>& inner) noexcept
        {
            return outer.left <= inner.left && inner.right <= outer.right;
        }

        template<typename GeomTraits>
        inline void sort_face_group(std::vector<FaceIntervalRecord1D<GeomTraits>>& group)
        {
            std::sort(group.begin(), group.end(),
                [](const auto& a, const auto& b)
                {
                    if (a.left != b.left)   return a.left < b.left;
                    if (a.right != b.right) return a.right > b.right; // longer first on same start
                    if (a.cell_id != b.cell_id) return a.cell_id < b.cell_id;
                    return a.face_id < b.face_id;
                });
        }

        template<typename GeomTraits>
        [[nodiscard]] inline std::unordered_map<int, std::vector<FaceIntervalRecord1D<GeomTraits>>>
        make_spatial_face_groups_1d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<mesh::topology::PrismRecord1D<GeomTraits>>& records)
        {
            std::unordered_map<int, std::vector<FaceIntervalRecord1D<GeomTraits>>> groups;
            groups.reserve(records.size() * 2);

            for (const auto& rec : records)
            {
                const auto& cell = mesh.cell(rec.cell_id);

                for (int face = 0; face < 2; ++face)
                {
                    FaceIntervalRecord1D<GeomTraits> f;
                    f.cell_id  = rec.cell_id;
                    f.face_id  = face;
                    f.fixed_id = cell.spatial_faces[face].spatial_vertex_ids[0];
                    f.left_id  = cell.spatial_faces[face].temporal_vertex_ids[0];
                    f.right_id = cell.spatial_faces[face].temporal_vertex_ids[1];
                    f.left     = rec.t0;
                    f.right    = rec.t1;
                    groups[f.fixed_id].push_back(f);
                }
            }

            for (auto& [fixed, group] : groups)
                sort_face_group(group);

            return groups;
        }

        template<typename GeomTraits>
        [[nodiscard]] inline std::unordered_map<int, std::vector<FaceIntervalRecord1D<GeomTraits>>>
        make_temporal_face_groups_1d(
            const mesh::Mesh<GeomTraits>& mesh,
            const std::vector<mesh::topology::PrismRecord1D<GeomTraits>>& records)
        {
            std::unordered_map<int, std::vector<FaceIntervalRecord1D<GeomTraits>>> groups;
            groups.reserve(records.size() * 2);

            for (const auto& rec : records)
            {
                const auto& cell = mesh.cell(rec.cell_id);

                for (int face = 0; face < 2; ++face)
                {
                    FaceIntervalRecord1D<GeomTraits> f;
                    f.cell_id  = rec.cell_id;
                    f.face_id  = face;
                    f.fixed_id = cell.temporal_faces[face].temporal_vertex_id;
                    f.left_id  = cell.temporal_faces[face].spatial_vertex_ids[0];
                    f.right_id = cell.temporal_faces[face].spatial_vertex_ids[1];
                    f.left     = rec.x0;
                    f.right    = rec.x1;
                    groups[f.fixed_id].push_back(f);
                }
            }

            for (auto& [fixed, group] : groups)
                sort_face_group(group);

            return groups;
        }
    }

    template<typename GeomTraits, typename Policy>
    inline void compute_adjacency_1d(
        Adjacency<GeomTraits, Policy>& adjacency,
        const std::vector<int>& active_cells,
        const mesh::Mesh<GeomTraits>& mesh)
    {
        static_assert(GeomTraits::dim_space_v == 1,
                      "compute_adjacency_1d requires dim_space_v == 1.");
        static_assert(GeomTraits::dim_time_v == 1,
                      "compute_adjacency_1d requires dim_time_v == 1.");

        adjacency.spatial_interfaces.reserve(active_cells.size() * 2);
        adjacency.temporal_interfaces.reserve(active_cells.size() * 2);
        adjacency.cell_to_spatial.reserve(active_cells.size());
        adjacency.cell_to_temporal.reserve(active_cells.size());

        for (const int cell_id : active_cells)
        {
            adjacency.cell_to_spatial[cell_id] = {};
            adjacency.cell_to_temporal[cell_id] = {};
        }

        const auto records = mesh::topology::make_active_prism_records_1d(active_cells, mesh);

        const auto push_spatial = [&](const SpatialInterface<GeomTraits>& iface)
        {
            adjacency.spatial_interfaces.push_back(iface);
            const int iface_id = static_cast<int>(adjacency.spatial_interfaces.size()) - 1;
            adjacency.cell_to_spatial[iface.master_cell][iface.master_face].push_back(iface_id);
            if (!iface.is_boundary && iface.slave_cell >= 0)
                adjacency.cell_to_spatial[iface.slave_cell][iface.slave_face].push_back(iface_id);
        };

        const auto push_temporal = [&](const TemporalInterface<GeomTraits>& iface)
        {
            adjacency.temporal_interfaces.push_back(iface);
            const int iface_id = static_cast<int>(adjacency.temporal_interfaces.size()) - 1;
            adjacency.cell_to_temporal[iface.master_cell][iface.master_face].push_back(iface_id);
            if (!iface.is_boundary && iface.slave_cell >= 0)
                adjacency.cell_to_temporal[iface.slave_cell][iface.slave_face].push_back(iface_id);
        };

        // ---------------------------------------------------------------------
        // Spatial interfaces: fixed x, varying temporal interval
        // ---------------------------------------------------------------------
        {
            const auto groups = detail::adjacency_impl::make_spatial_face_groups_1d(mesh, records);

            for (const auto& [fixed_id, group] : groups)
            {
                std::vector<char> matched(group.size(), 0);
                std::vector<int> open;

                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                {
                    const auto& cur = group[static_cast<std::size_t>(i)];

                    while (!open.empty() &&
                           group[static_cast<std::size_t>(open.back())].right <= cur.left)
                    {
                        open.pop_back();
                    }

                    bool local_match = false;

                    // Exact match first.
                    for (auto it = open.rbegin(); it != open.rend(); ++it)
                    {
                        const auto& cand = group[static_cast<std::size_t>(*it)];
                        if (!detail::adjacency_impl::same_interval(cand, cur))
                            continue;

                        if (cand.cell_id < cur.cell_id)
                        {
                            SpatialInterface<GeomTraits> iface{};
                            const auto& master = mesh.cell(cand.cell_id);
                            const auto& slave  = mesh.cell(cur.cell_id);

                            iface.master_cell = cand.cell_id;
                            iface.master_face = cand.face_id;
                            iface.slave_cell  = cur.cell_id;
                            iface.slave_face  = cur.face_id;

                            iface.master_spatial_vertex_ids  = master.spatial_faces[cand.face_id].spatial_vertex_ids;
                            iface.master_temporal_vertex_ids = master.spatial_faces[cand.face_id].temporal_vertex_ids;
                            iface.slave_spatial_vertex_ids   = slave.spatial_faces[cur.face_id].spatial_vertex_ids;
                            iface.slave_temporal_vertex_ids  = slave.spatial_faces[cur.face_id].temporal_vertex_ids;

                            push_spatial(iface);
                        }

                        matched[static_cast<std::size_t>(i)] = 1;
                        matched[static_cast<std::size_t>(*it)] = 1;
                        local_match = true;
                        break;
                    }

                    // Otherwise nearest containing interval gives hanging interface.
                    if (!local_match)
                    {
                        for (auto it = open.rbegin(); it != open.rend(); ++it)
                        {
                            const auto& cand = group[static_cast<std::size_t>(*it)];
                            if (!detail::adjacency_impl::contains_interval(cand, cur))
                                continue;

                            SpatialInterface<GeomTraits> iface{};
                            const auto& master = mesh.cell(cand.cell_id);
                            const auto& slave  = mesh.cell(cur.cell_id);

                            iface.master_cell = cand.cell_id;
                            iface.master_face = cand.face_id;
                            iface.slave_cell  = cur.cell_id;
                            iface.slave_face  = cur.face_id;
                            iface.is_hanging  = true;

                            iface.master_spatial_vertex_ids  = master.spatial_faces[cand.face_id].spatial_vertex_ids;
                            iface.master_temporal_vertex_ids = master.spatial_faces[cand.face_id].temporal_vertex_ids;
                            iface.slave_spatial_vertex_ids   = slave.spatial_faces[cur.face_id].spatial_vertex_ids;
                            iface.slave_temporal_vertex_ids  = slave.spatial_faces[cur.face_id].temporal_vertex_ids;

                            push_spatial(iface);

                            matched[static_cast<std::size_t>(i)] = 1;
                            matched[static_cast<std::size_t>(*it)] = 1;
                            local_match = true;
                            break;
                        }
                    }

                    open.push_back(i);
                }

                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                {
                    if (matched[static_cast<std::size_t>(i)])
                        continue;

                    const auto& rec = group[static_cast<std::size_t>(i)];
                    const auto& c = mesh.cell(rec.cell_id);

                    if (!c.spatial_boundary[rec.face_id])
                        continue;

                    SpatialInterface<GeomTraits> iface{};
                    iface.master_cell = rec.cell_id;
                    iface.master_face = rec.face_id;
                    iface.is_boundary = true;
                    iface.master_spatial_vertex_ids  = c.spatial_faces[rec.face_id].spatial_vertex_ids;
                    iface.master_temporal_vertex_ids = c.spatial_faces[rec.face_id].temporal_vertex_ids;
                    push_spatial(iface);
                }
            }
        }

        // ---------------------------------------------------------------------
        // Temporal interfaces: fixed t, varying spatial interval
        // ---------------------------------------------------------------------
        {
            const auto groups = detail::adjacency_impl::make_temporal_face_groups_1d(mesh, records);

            for (const auto& [fixed_id, group] : groups)
            {
                std::vector<char> matched(group.size(), 0);
                std::vector<int> open;

                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                {
                    const auto& cur = group[static_cast<std::size_t>(i)];

                    while (!open.empty() &&
                           group[static_cast<std::size_t>(open.back())].right <= cur.left)
                    {
                        open.pop_back();
                    }

                    bool local_match = false;

                    for (auto it = open.rbegin(); it != open.rend(); ++it)
                    {
                        const auto& cand = group[static_cast<std::size_t>(*it)];
                        if (!detail::adjacency_impl::same_interval(cand, cur))
                            continue;

                        if (cand.cell_id < cur.cell_id)
                        {
                            TemporalInterface<GeomTraits> iface{};
                            const auto& master = mesh.cell(cand.cell_id);
                            const auto& slave  = mesh.cell(cur.cell_id);

                            iface.master_cell = cand.cell_id;
                            iface.master_face = cand.face_id;
                            iface.slave_cell  = cur.cell_id;
                            iface.slave_face  = cur.face_id;

                            iface.master_spatial_vertex_ids = master.temporal_faces[cand.face_id].spatial_vertex_ids;
                            iface.master_temporal_vertex_id = master.temporal_faces[cand.face_id].temporal_vertex_id;
                            iface.slave_spatial_vertex_ids  = slave.temporal_faces[cur.face_id].spatial_vertex_ids;
                            iface.slave_temporal_vertex_id  = slave.temporal_faces[cur.face_id].temporal_vertex_id;

                            push_temporal(iface);
                        }

                        matched[static_cast<std::size_t>(i)] = 1;
                        matched[static_cast<std::size_t>(*it)] = 1;
                        local_match = true;
                        break;
                    }

                    if (!local_match)
                    {
                        for (auto it = open.rbegin(); it != open.rend(); ++it)
                        {
                            const auto& cand = group[static_cast<std::size_t>(*it)];
                            if (!detail::adjacency_impl::contains_interval(cand, cur))
                                continue;

                            TemporalInterface<GeomTraits> iface{};
                            const auto& master = mesh.cell(cand.cell_id);
                            const auto& slave  = mesh.cell(cur.cell_id);

                            iface.master_cell = cand.cell_id;
                            iface.master_face = cand.face_id;
                            iface.slave_cell  = cur.cell_id;
                            iface.slave_face  = cur.face_id;
                            iface.is_hanging  = true;

                            iface.master_spatial_vertex_ids = master.temporal_faces[cand.face_id].spatial_vertex_ids;
                            iface.master_temporal_vertex_id = master.temporal_faces[cand.face_id].temporal_vertex_id;
                            iface.slave_spatial_vertex_ids  = slave.temporal_faces[cur.face_id].spatial_vertex_ids;
                            iface.slave_temporal_vertex_id  = slave.temporal_faces[cur.face_id].temporal_vertex_id;

                            push_temporal(iface);

                            matched[static_cast<std::size_t>(i)] = 1;
                            matched[static_cast<std::size_t>(*it)] = 1;
                            local_match = true;
                            break;
                        }
                    }

                    open.push_back(i);
                }

                for (int i = 0; i < static_cast<int>(group.size()); ++i)
                {
                    if (matched[static_cast<std::size_t>(i)])
                        continue;

                    const auto& rec = group[static_cast<std::size_t>(i)];
                    const auto& c = mesh.cell(rec.cell_id);

                    if (!c.temporal_boundary[rec.face_id])
                        continue;

                    TemporalInterface<GeomTraits> iface{};
                    iface.master_cell = rec.cell_id;
                    iface.master_face = rec.face_id;
                    iface.is_boundary = true;
                    iface.master_spatial_vertex_ids = c.temporal_faces[rec.face_id].spatial_vertex_ids;
                    iface.master_temporal_vertex_id = c.temporal_faces[rec.face_id].temporal_vertex_id;
                    push_temporal(iface);
                }
            }
        }
    }
}

#pragma once

#include <algorithm>
#include <cmath>
#include <iterator>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "../detail/timing.hpp"
#include "../../mesh/topology/temporal_keys.hpp"
#include "time_slab_space.hpp"

namespace finite_element::time_slabs
{
    template<typename GeomTraits, typename FETraits>
    class TimeSlabBuilder
    {
    public:
        using GT              = GeomTraits;
        using FETraitsType    = FETraits;
        using PolicyType      = finite_element::SpaceOnlyPolicy;

        using SourceSpaceType = FESpace<GeomTraits, FETraits, PolicyType>;
        using SlabSpaceType   = TimeSlabSpace<GeomTraits, FETraits>;

        using TimePointRankMap = std::unordered_map<
            mesh::topology::TimePointIdKey,
            int,
            mesh::topology::TimePointIdKeyHash>;

        struct SlabEndpointData
        {
            std::vector<double> times{};
            TimePointRankMap rank_by_temporal_vertex_id{};
        };

        struct ActiveEndpointRecord
        {
            int temporal_vertex_id = -1;
            double time = 0.0;
        };

        [[nodiscard]] static SlabSpaceType build(const SourceSpaceType& source_space,
                                                 double time_tol = 0.0,
                                                 const finite_element::detail::TimingRecorder& timing = {})
        {
            SlabSpaceType out(source_space);
            initialize(out, time_tol, timing);
            return out;
        }

        static void initialize(
            SlabSpaceType& slab_space,
            double time_tol = 0.0,
            const finite_element::detail::TimingRecorder& timing = {})
        {
            auto total_timer =
                timing.scoped("time_slab.space_construction.builder");

            {
                auto timer =
                    timing.scoped("time_slab.space_construction.clear");
                slab_space.clear();
            }

            SlabEndpointData slab_endpoint_data;
            {
                auto timer =
                    timing.scoped(
                        "time_slab.space_construction.collect_active_times");
                slab_endpoint_data =
                    collect_active_slab_endpoint_data_(
                        slab_space.source_space());
            }
            if (slab_endpoint_data.times.size() < 2)
                return;

            auto endpoint_rank_by_temporal_vertex_id =
                std::move(slab_endpoint_data.rank_by_temporal_vertex_id);
            slab_space.set_slab_times_(std::move(slab_endpoint_data.times));
            {
                auto timer =
                    timing.scoped(
                        "time_slab.space_construction.slab_time_infos");
                slab_space.set_slab_time_infos_(
                    build_slab_time_infos_(
                        slab_space.source_space(),
                        slab_space.slab_times(),
                        endpoint_rank_by_temporal_vertex_id,
                        time_tol));
            }
            {
                auto timer =
                    timing.scoped(
                        "time_slab.space_construction."
                        "source_slab_provenance_map_init");
                slab_space.initialize_source_cell_provenance_(
                    slab_space.source_space().mesh_ref().n_cells());
            }

            const auto& source_space = slab_space.source_space();
            const auto& src_mesh = source_space.mesh_ref();

            const int n_slabs = static_cast<int>(slab_space.slab_times().size()) - 1;
            {
                auto timer =
                    timing.scoped("time_slab.space_construction.create_slabs");
                slab_space.reserve_slabs_(n_slabs);

                for (int k = 0; k < n_slabs; ++k)
                {
                    slab_space.add_empty_slab_(
                        k,
                        slab_space.slab_times()[static_cast<std::size_t>(k)],
                        slab_space.slab_times()[static_cast<std::size_t>(k + 1)]);
                    slab_space.slab(k).inherit_spatial_boundary_metadata_from_source(
                        source_space);
                }
            }

            int source_slab_slice_count = 0;
            {
                auto timer =
                    timing.scoped(
                        "time_slab.space_construction."
                        "source_slab_slicing_and_provenance");

                for (const int source_cell_id : source_space.active_cells())
                {
                    const auto& cell = src_mesh.cell(source_cell_id);

                    const double cell_t0 =
                        src_mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
                    const double cell_t1 =
                        src_mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];

                    const int k_begin =
                        slab_time_endpoint_index_(
                            endpoint_rank_by_temporal_vertex_id,
                            cell.temporal_vertex_ids[0],
                            slab_space.slab_times(),
                            cell_t0,
                            time_tol);
                    const int k_end_endpoint =
                        slab_time_endpoint_index_(
                            endpoint_rank_by_temporal_vertex_id,
                            cell.temporal_vertex_ids[1],
                            slab_space.slab_times(),
                            cell_t1,
                            time_tol);

                    if (k_begin < 0 ||
                        k_end_endpoint < 0 ||
                        k_begin >= k_end_endpoint)
                    {
                        throw std::runtime_error(
                            "TimeSlabBuilder::initialize: failed to locate slab range for source cell.");
                    }

                    for (int k = k_begin; k < k_end_endpoint; ++k)
                    {
                        const int slab_cell_id =
                            slab_space.slab(k).append_slice_from_source(
                                source_space,
                                source_cell_id,
                                k,
                                k + 1);

                        typename SlabSpaceType::SourceCellSlabLocation location;
                        location.source_cell_id = source_cell_id;
                        location.slab_id = k;
                        location.slab_local_cell_id = slab_cell_id;
                        location.slab_time_begin_id = k;
                        location.slab_time_end_id = k + 1;
                        location.slab_t_begin =
                            slab_space.slab_times()[static_cast<std::size_t>(k)];
                        location.slab_t_end =
                            slab_space.slab_times()[static_cast<std::size_t>(k + 1)];
                        location.source_t_begin = cell_t0;
                        location.source_t_end = cell_t1;
                        slab_space.register_source_cell_slab_location_(location);
                        ++source_slab_slice_count;
                    }
                }
            }

            {
                auto timer =
                    timing.scoped("time_slab.space_construction.slab_finalize");
                for (int k = 0; k < slab_space.n_slabs(); ++k)
                    slab_space.slab(k).finalize();
            }

            {
                auto timer =
                    timing.scoped(
                        "time_slab.space_construction."
                        "measure_partition_check");
                slab_space.assert_measure_partition();
            }

            timing.add(
                "time_slab.space_construction.slab_count.count",
                static_cast<double>(n_slabs));
            timing.add(
                "time_slab.slab_count.count",
                static_cast<double>(n_slabs));
            timing.add(
                "time_slab.space_construction.source_active_cells.count",
                static_cast<double>(source_space.active_cells().size()));
            timing.add(
                "time_slab.source_y_cells.count",
                static_cast<double>(source_space.active_cells().size()));
            timing.add(
                "time_slab.space_construction.source_slab_slices.count",
                static_cast<double>(source_slab_slice_count));
            timing.add(
                "time_slab.copied_slab_cells.count",
                static_cast<double>(source_slab_slice_count));
            timing.add(
                "time_slab.virtual_slab_cells.count",
                0.0);
        }

    private:
        [[nodiscard]] static SlabEndpointData collect_active_slab_endpoint_data_(
            const SourceSpaceType& source_space)
        {
            std::vector<ActiveEndpointRecord> endpoints;
            endpoints.reserve(2 * source_space.active_cells().size());

            const auto& mesh = source_space.mesh_ref();
            for (const int cell_id : source_space.active_cells())
            {
                const auto& cell = mesh.cell(cell_id);
                for (const int temporal_vertex_id : cell.temporal_vertex_ids)
                {
                    endpoints.push_back(
                        ActiveEndpointRecord{
                            temporal_vertex_id,
                            mesh.temporal_vertices()[
                                static_cast<std::size_t>(temporal_vertex_id)][0]});
                }
            }

            std::sort(
                endpoints.begin(),
                endpoints.end(),
                [](const ActiveEndpointRecord& a,
                   const ActiveEndpointRecord& b)
                {
                    if (a.time != b.time)
                        return a.time < b.time;
                    return a.temporal_vertex_id < b.temporal_vertex_id;
                });

            SlabEndpointData data;
            data.times.reserve(endpoints.size());
            data.rank_by_temporal_vertex_id.reserve(endpoints.size());

            for (const auto& endpoint : endpoints)
            {
                if (data.rank_by_temporal_vertex_id.find(
                        mesh::topology::make_time_point_id_key(
                            endpoint.temporal_vertex_id)) !=
                    data.rank_by_temporal_vertex_id.end())
                {
                    continue;
                }

                if (data.times.empty() || endpoint.time != data.times.back())
                    data.times.push_back(endpoint.time);

                data.rank_by_temporal_vertex_id.emplace(
                    mesh::topology::make_time_point_id_key(
                        endpoint.temporal_vertex_id),
                    static_cast<int>(data.times.size()) - 1);
            }

            return data;
        }

        [[nodiscard]] static std::vector<typename SlabSpaceType::SlabTimeInfo>
        build_slab_time_infos_(
            const SourceSpaceType& source_space,
            const std::vector<double>& slab_times,
            const TimePointRankMap& endpoint_rank_by_temporal_vertex_id,
            double time_tol)
        {
            std::vector<typename SlabSpaceType::SlabTimeInfo> infos;
            infos.reserve(slab_times.size());

            for (int time_id = 0;
                 time_id < static_cast<int>(slab_times.size());
                 ++time_id)
            {
                typename SlabSpaceType::SlabTimeInfo info;
                info.global_time_id = time_id;
                info.time = slab_times[static_cast<std::size_t>(time_id)];
                infos.push_back(std::move(info));
            }

            const auto& mesh = source_space.mesh_ref();
            for (const int cell_id : source_space.active_cells())
            {
                const auto& cell = mesh.cell(cell_id);
                for (const int temporal_vertex_id : cell.temporal_vertex_ids)
                {
                    const double t =
                        mesh.temporal_vertices()[temporal_vertex_id][0];

                    const int time_id =
                        slab_time_endpoint_index_(
                            endpoint_rank_by_temporal_vertex_id,
                            temporal_vertex_id,
                            slab_times,
                            t,
                            time_tol);
                    if (time_id < 0)
                        throw std::runtime_error(
                            "TimeSlabBuilder::build_slab_time_infos_: "
                            "failed to locate temporal endpoint.");

                    infos[static_cast<std::size_t>(time_id)]
                        .source_temporal_vertex_ids.push_back(
                            temporal_vertex_id);
                }
            }

            for (auto& info : infos)
            {
                std::sort(
                    info.source_temporal_vertex_ids.begin(),
                    info.source_temporal_vertex_ids.end());
                info.source_temporal_vertex_ids.erase(
                    std::unique(
                        info.source_temporal_vertex_ids.begin(),
                        info.source_temporal_vertex_ids.end()),
                    info.source_temporal_vertex_ids.end());
            }

            return infos;
        }

        [[nodiscard]] static int slab_time_endpoint_index_(
            const TimePointRankMap& endpoint_rank_by_temporal_vertex_id,
            int temporal_vertex_id,
            const std::vector<double>& slab_times,
            double t,
            double time_tol)
        {
            const auto it =
                endpoint_rank_by_temporal_vertex_id.find(
                    mesh::topology::make_time_point_id_key(temporal_vertex_id));
            if (it != endpoint_rank_by_temporal_vertex_id.end())
                return it->second;

            return slab_time_endpoint_index_(slab_times, t, time_tol);
        }

        [[nodiscard]] static int slab_time_endpoint_index_(
            const std::vector<double>& slab_times,
            double t,
            double time_tol)
        {
            if (slab_times.empty())
                return -1;

            const double tol = (time_tol > 0.0) ? time_tol : 0.0;
            auto best = slab_times.end();
            double best_distance = 0.0;

            const auto consider = [&](auto it)
            {
                if (it == slab_times.end())
                    return;
                const double distance = std::abs(*it - t);
                if (distance > tol)
                    return;
                if (best == slab_times.end() || distance < best_distance)
                {
                    best = it;
                    best_distance = distance;
                }
            };

            auto it = std::lower_bound(slab_times.begin(), slab_times.end(), t);
            consider(it);
            if (it != slab_times.begin())
                consider(std::prev(it));

            if (best == slab_times.end())
                return -1;

            return static_cast<int>(std::distance(slab_times.begin(), best));
        }
    };
}

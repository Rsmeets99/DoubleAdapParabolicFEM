#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../detail/timing.hpp"
#include "time_slab_space.hpp"

namespace finite_element::time_slabs
{
    enum class TimeSlabVertexPatchKind
    {
        interior,
        boundary
    };

    template<typename GeomTraits, typename FETraits>
    struct TimeSlabVertexPatch
    {
        using GT           = GeomTraits;
        using FETraitsType = FETraits;
        using Types        = mesh::MeshTypes<GeomTraits>;

        using SpatialPoint     = typename Types::SpatialPoint;
        using SpatialVertexIds = typename Types::SpatialVertexIds;

        struct CellData
        {
            int slab_cell_id     = -1;
            int source_cell_id   = -1;
            int local_vertex_index = -1;
            int source_local_vertex_index = -1;
            int source_spatial_vertex_id = -1;

            SpatialVertexIds slab_spatial_vertex_ids{};
            SpatialVertexIds source_spatial_vertex_ids{};
        };

        int patch_id          = -1;
        int slab_id           = -1;
        int spatial_vertex_id = -1;

        SpatialPoint spatial_vertex{};

        double t_begin = 0.0;
        double t_end   = 0.0;

        TimeSlabVertexPatchKind kind = TimeSlabVertexPatchKind::boundary;

        int n_cells = 0;
        std::vector<CellData> cells{};

        [[nodiscard]] bool is_boundary() const noexcept
        {
            return kind == TimeSlabVertexPatchKind::boundary;
        }

        [[nodiscard]] double time_length() const noexcept
        {
            return t_end - t_begin;
        }

        [[nodiscard]] const CellData& cell(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_cells)
            {
                throw std::runtime_error(
                    "TimeSlabVertexPatch::cell: patch cell index out of range.");
            }

            return cells[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] int find_patch_cell_index(int slab_cell_id) const noexcept
        {
            for (int i = 0; i < n_cells; ++i)
            {
                if (cells[static_cast<std::size_t>(i)].slab_cell_id == slab_cell_id)
                    return i;
            }

            return -1;
        }

        [[nodiscard]] bool contains_slab_cell(int slab_cell_id) const noexcept
        {
            return find_patch_cell_index(slab_cell_id) >= 0;
        }
    };

    template<class PatchType>
    inline constexpr bool is_time_slab_vertex_patch_v = false;

    template<typename GeomTraits, typename FETraits>
    inline constexpr bool is_time_slab_vertex_patch_v<
        TimeSlabVertexPatch<GeomTraits, FETraits>> = true;

    template<typename GeomTraits, typename FETraits>
    class TimeSlabVertexPatchSet
    {
    public:
        using GT            = GeomTraits;
        using FETraitsType  = FETraits;
        using Types         = mesh::MeshTypes<GeomTraits>;
        using SlabSpaceType = TimeSlabSpace<GeomTraits, FETraits>;
        using PatchType     = TimeSlabVertexPatch<GeomTraits, FETraits>;

        struct CellPatchMembership
        {
            int patch_id = -1;
            int patch_cell_index = -1;
            int local_vertex_index = -1;
            int spatial_vertex_id = -1;
        };

        explicit TimeSlabVertexPatchSet(const SlabSpaceType& slab_space)
            : slab_space_(&slab_space)
        {
            slab_data_.resize(static_cast<std::size_t>(slab_space.n_slabs()));

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                auto& slab_data = slab_data_[static_cast<std::size_t>(slab_id)];
                const auto n_cells = slab_space.slab(slab_id).mesh_ref().n_cells();
                slab_data.memberships.resize(n_cells);
                slab_data.membership_counts.assign(n_cells, 0);
            }
        }

        [[nodiscard]] const SlabSpaceType& slab_space() const noexcept
        {
            return *slab_space_;
        }

        [[nodiscard]] int n_patches() const noexcept
        {
            return static_cast<int>(patches_.size());
        }

        [[nodiscard]] const std::vector<PatchType>& patches() const noexcept
        {
            return patches_;
        }

        [[nodiscard]] const PatchType& patch(int patch_id) const
        {
            if (patch_id < 0 || static_cast<std::size_t>(patch_id) >= patches_.size())
                throw std::runtime_error("TimeSlabVertexPatchSet::patch: patch id out of range.");

            return patches_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] int cell_patch_count(int slab_id, int slab_cell_id) const
        {
            check_cell_index_(slab_id, slab_cell_id);
            return slab_data_[static_cast<std::size_t>(slab_id)]
                .membership_counts[static_cast<std::size_t>(slab_cell_id)];
        }

        [[nodiscard]] const CellPatchMembership& cell_patch_membership(
            int slab_id,
            int slab_cell_id,
            int membership_index) const
        {
            check_cell_index_(slab_id, slab_cell_id);

            const int count = cell_patch_count(slab_id, slab_cell_id);
            if (membership_index < 0 || membership_index >= count)
            {
                throw std::runtime_error(
                    "TimeSlabVertexPatchSet::cell_patch_membership: membership index out of range.");
            }

            return slab_data_[static_cast<std::size_t>(slab_id)]
                .memberships[static_cast<std::size_t>(slab_cell_id)]
                            [static_cast<std::size_t>(membership_index)];
        }

        [[nodiscard]] std::vector<CellPatchMembership> cell_patch_memberships(
            int slab_id,
            int slab_cell_id) const
        {
            check_cell_index_(slab_id, slab_cell_id);

            std::vector<CellPatchMembership> out;
            const int count = cell_patch_count(slab_id, slab_cell_id);
            out.reserve(static_cast<std::size_t>(count));

            for (int i = 0; i < count; ++i)
                out.push_back(cell_patch_membership(slab_id, slab_cell_id, i));

            return out;
        }

        [[nodiscard]] const std::vector<int>& slab_patch_ids(int slab_id) const
        {
            check_slab_index_(slab_id);
            return slab_data_[static_cast<std::size_t>(slab_id)].patch_ids;
        }

    private:
        template<typename, typename>
        friend class TimeSlabVertexPatchBuilder;

        struct SlabData
        {
            std::vector<std::array<CellPatchMembership, Types::n_spatial_vertices>> memberships{};
            std::vector<int> membership_counts{};
            std::vector<int> patch_ids{};
        };

        void add_patch_(PatchType patch)
        {
            const int patch_id = static_cast<int>(patches_.size());
            patch.patch_id = patch_id;
            patch.n_cells = static_cast<int>(patch.cells.size());

            if (patch.slab_id < 0 || patch.slab_id >= slab_space_->n_slabs())
            {
                throw std::runtime_error(
                    "TimeSlabVertexPatchSet::add_patch_: invalid slab id.");
            }

            auto& slab_data = slab_data_[static_cast<std::size_t>(patch.slab_id)];
            slab_data.patch_ids.push_back(patch_id);

            for (int patch_cell_index = 0; patch_cell_index < patch.n_cells; ++patch_cell_index)
            {
                const auto& patch_cell =
                    patch.cells[static_cast<std::size_t>(patch_cell_index)];
                const int slab_cell_id = patch_cell.slab_cell_id;

                if (slab_cell_id < 0 ||
                    static_cast<std::size_t>(slab_cell_id) >= slab_data.memberships.size())
                {
                    throw std::runtime_error(
                        "TimeSlabVertexPatchSet::add_patch_: slab cell id out of range.");
                }

                auto& count =
                    slab_data.membership_counts[static_cast<std::size_t>(slab_cell_id)];
                if (count >= Types::n_spatial_vertices)
                {
                    throw std::runtime_error(
                        "TimeSlabVertexPatchSet::add_patch_: slab cell has too many vertex-patch memberships.");
                }

                slab_data.memberships[static_cast<std::size_t>(slab_cell_id)]
                    [static_cast<std::size_t>(count)] =
                    CellPatchMembership{
                        patch_id,
                        patch_cell_index,
                        patch_cell.local_vertex_index,
                        patch.spatial_vertex_id};
                ++count;
            }

            patches_.push_back(std::move(patch));
        }

        void check_slab_index_(int slab_id) const
        {
            if (slab_id < 0 || slab_id >= slab_space_->n_slabs())
                throw std::runtime_error(
                    "TimeSlabVertexPatchSet: slab index out of range.");
        }

        void check_cell_index_(int slab_id, int slab_cell_id) const
        {
            check_slab_index_(slab_id);

            const auto& slab_data = slab_data_[static_cast<std::size_t>(slab_id)];
            if (slab_cell_id < 0 ||
                static_cast<std::size_t>(slab_cell_id) >= slab_data.memberships.size())
            {
                throw std::runtime_error(
                    "TimeSlabVertexPatchSet: slab cell index out of range.");
            }
        }

        const SlabSpaceType* slab_space_ = nullptr;
        std::vector<PatchType> patches_{};
        std::vector<SlabData> slab_data_{};
    };

    template<typename GeomTraits, typename FETraits>
    class TimeSlabVertexPatchBuilder
    {
    public:
        using GT            = GeomTraits;
        using FETraitsType  = FETraits;
        using Types         = mesh::MeshTypes<GeomTraits>;
        using SlabSpaceType = TimeSlabSpace<GeomTraits, FETraits>;
        using PatchType     = TimeSlabVertexPatch<GeomTraits, FETraits>;
        using PatchSetType  = TimeSlabVertexPatchSet<GeomTraits, FETraits>;

        struct BuildCounters
        {
            std::size_t cell_vertex_incidences_visited = 0;
            std::size_t coordinate_vertex_matches_performed = 0;
        };

        [[nodiscard]] static PatchSetType build(const SlabSpaceType& slab_space)
        {
            return build(slab_space, nullptr);
        }

        [[nodiscard]] static PatchSetType build(
            const SlabSpaceType& slab_space,
            const finite_element::detail::TimingRecorder& timing)
        {
            BuildCounters counters;
            auto patch_set = build(slab_space, &counters);
            record_performance_counters_(patch_set, counters, timing);
            return patch_set;
        }

        [[nodiscard]] static PatchSetType build(
            const SlabSpaceType& slab_space,
            BuildCounters* counters)
        {
            static_assert(GeomTraits::dim_space_v == 2,
                          "TimeSlabVertexPatchBuilder requires dim_space_v == 2.");
            static_assert(GeomTraits::dim_time_v == 1,
                          "TimeSlabVertexPatchBuilder requires dim_time_v == 1.");

            if (counters != nullptr)
                *counters = BuildCounters{};

            PatchSetType patch_set(slab_space);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
                build_for_slab_(patch_set, slab_id, counters);

            return patch_set;
        }

    private:
        using MeshType         = mesh::Mesh<GeomTraits>;
        using SpatialPoint     = typename Types::SpatialPoint;

        struct SlabIncidenceData
        {
            std::unordered_map<int, std::vector<typename PatchType::CellData>>
                vertex_to_slab_cells{};
            std::unordered_map<int, bool> vertex_is_boundary{};
            std::vector<int> vertex_ids{};
        };

        static constexpr double vertex_match_tol = 1.0e-12;

        static void record_performance_counters_(
            const PatchSetType& patch_set,
            const BuildCounters& counters,
            const finite_element::detail::TimingRecorder& timing)
        {
            timing.add(
                "patch.patch_count.count",
                static_cast<double>(patch_set.n_patches()));
            timing.add(
                "patch.cell_vertex_incidences.count",
                static_cast<double>(
                    counters.cell_vertex_incidences_visited));
            timing.add(
                "patch.coordinate_vertex_matches.count",
                static_cast<double>(
                    counters.coordinate_vertex_matches_performed));
        }

        [[nodiscard]] static bool points_equal_(
            const SpatialPoint& a,
            const SpatialPoint& b) noexcept
        {
            for (int d = 0; d < GeomTraits::dim_space_v; ++d)
            {
                if (std::abs(a[static_cast<std::size_t>(d)] -
                             b[static_cast<std::size_t>(d)]) > vertex_match_tol)
                {
                    return false;
                }
            }

            return true;
        }

        template<class VertexIdContainer>
        [[nodiscard]] static bool contains_vertex_(
            const VertexIdContainer& vertex_ids,
            int vertex_id) noexcept
        {
            return std::find(vertex_ids.begin(), vertex_ids.end(), vertex_id) !=
                   vertex_ids.end();
        }

        [[nodiscard]] static int source_local_vertex_index_for_point_(
            const MeshType& source_mesh,
            int source_cell_id,
            const SpatialPoint& spatial_vertex,
            BuildCounters* counters)
        {
            if (counters != nullptr)
                ++counters->coordinate_vertex_matches_performed;

            const auto& source_cell = source_mesh.cell(source_cell_id);

            for (int i = 0; i < Types::n_spatial_vertices; ++i)
            {
                const int source_vertex_id =
                    source_cell.spatial_vertex_ids[static_cast<std::size_t>(i)];
                const auto& source_point =
                    source_mesh.spatial_vertices()[static_cast<std::size_t>(source_vertex_id)];

                if (points_equal_(source_point, spatial_vertex))
                    return i;
            }

            throw std::runtime_error(
                "TimeSlabVertexPatchBuilder: failed to match slab vertex to source cell vertex.");
        }

        static void debug_assert_slab_vertex_matches_source_vertex_(
            const MeshType& slab_mesh,
            const MeshType& source_mesh,
            int slab_cell_id,
            int local_vertex_index,
            int source_cell_id,
            int source_local_vertex_index)
        {
            const auto& slab_cell = slab_mesh.cell(slab_cell_id);
            const auto& source_cell = source_mesh.cell(source_cell_id);
            const int slab_spatial_vertex_id =
                slab_cell.spatial_vertex_ids[
                    static_cast<std::size_t>(local_vertex_index)];
            const int source_spatial_vertex_id =
                source_cell.spatial_vertex_ids[
                    static_cast<std::size_t>(source_local_vertex_index)];
            const auto& slab_point =
                slab_mesh.spatial_vertices()[
                    static_cast<std::size_t>(slab_spatial_vertex_id)];
            const auto& source_point =
                source_mesh.spatial_vertices()[
                    static_cast<std::size_t>(source_spatial_vertex_id)];

            if (!points_equal_(slab_point, source_point))
            {
                throw std::runtime_error(
                    "TimeSlabVertexPatchBuilder: slab/source vertex provenance "
                    "does not match spatial coordinates.");
            }
        }

        template<typename SlabType>
        [[nodiscard]] static int source_local_vertex_index_for_slab_vertex_(
            const SlabType& slab,
            const MeshType& slab_mesh,
            const MeshType& source_mesh,
            int slab_cell_id,
            int local_vertex_index,
            BuildCounters* counters)
        {
            const auto& info = slab.sliced_cell_info(slab_cell_id);
            if (local_vertex_index >= 0 &&
                local_vertex_index < Types::n_spatial_vertices)
            {
                const int source_local_vertex_index =
                    info.source_local_vertex_indices[
                        static_cast<std::size_t>(local_vertex_index)];
                if (source_local_vertex_index >= 0 &&
                    source_local_vertex_index < Types::n_spatial_vertices)
                {
#ifndef NDEBUG
                    debug_assert_slab_vertex_matches_source_vertex_(
                        slab_mesh,
                        source_mesh,
                        slab_cell_id,
                        local_vertex_index,
                        info.source_cell_id,
                        source_local_vertex_index);
#endif
                    return source_local_vertex_index;
                }
            }

#ifndef NDEBUG
            const auto& slab_cell = slab_mesh.cell(slab_cell_id);
            const int slab_spatial_vertex_id =
                slab_cell.spatial_vertex_ids[
                    static_cast<std::size_t>(local_vertex_index)];
            const auto& slab_spatial_vertex =
                slab_mesh.spatial_vertices()[
                    static_cast<std::size_t>(slab_spatial_vertex_id)];
            return source_local_vertex_index_for_point_(
                source_mesh,
                info.source_cell_id,
                slab_spatial_vertex,
                counters);
#else
            static_cast<void>(slab);
            static_cast<void>(slab_mesh);
            static_cast<void>(source_mesh);
            static_cast<void>(slab_cell_id);
            static_cast<void>(local_vertex_index);
            static_cast<void>(counters);
            throw std::runtime_error(
                "TimeSlabVertexPatchBuilder: missing slab/source vertex "
                "provenance; coordinate matching is debug-only.");
#endif
        }

        template<typename CellType>
        [[nodiscard]] static bool local_vertex_touches_spatial_boundary_(
            const CellType& cell,
            int spatial_vertex_id) noexcept
        {
            for (int face_id = 0; face_id < Types::n_spatial_faces; ++face_id)
            {
                if (!cell.spatial_boundary[static_cast<std::size_t>(face_id)])
                    continue;

                const auto& face =
                    cell.spatial_faces[static_cast<std::size_t>(face_id)]
                        .spatial_vertex_ids;
                if (contains_vertex_(face, spatial_vertex_id))
                    return true;
            }

            return false;
        }

        template<typename SlabType>
        [[nodiscard]] static SlabIncidenceData build_slab_incidence_(
            const SlabType& slab,
            const MeshType& source_mesh,
            BuildCounters* counters)
        {
            SlabIncidenceData incidence;
            incidence.vertex_to_slab_cells.reserve(
                slab.active_cells().size() * Types::n_spatial_vertices);
            incidence.vertex_is_boundary.reserve(
                slab.active_cells().size() * Types::n_spatial_vertices);
            incidence.vertex_ids.reserve(
                slab.active_cells().size() * Types::n_spatial_vertices);

            const auto& slab_mesh = slab.mesh_ref();
            for (const int slab_cell_id : slab.active_cells())
            {
                const auto& slab_cell = slab_mesh.cell(slab_cell_id);
                const auto& info = slab.sliced_cell_info(slab_cell_id);
                const int source_cell_id = info.source_cell_id;
                const auto& source_cell = source_mesh.cell(source_cell_id);

                for (int local_vertex_index = 0;
                     local_vertex_index < Types::n_spatial_vertices;
                     ++local_vertex_index)
                {
                    if (counters != nullptr)
                        ++counters->cell_vertex_incidences_visited;

                    const int spatial_vertex_id =
                        slab_cell.spatial_vertex_ids[
                            static_cast<std::size_t>(local_vertex_index)];
                    const int source_local_vertex_index =
                        source_local_vertex_index_for_slab_vertex_(
                            slab,
                            slab_mesh,
                            source_mesh,
                            slab_cell_id,
                            local_vertex_index,
                            counters);

                    typename PatchType::CellData cell_data;
                    cell_data.slab_cell_id = slab_cell_id;
                    cell_data.source_cell_id = source_cell_id;
                    cell_data.local_vertex_index = local_vertex_index;
                    cell_data.source_local_vertex_index = source_local_vertex_index;
                    cell_data.source_spatial_vertex_id =
                        source_cell.spatial_vertex_ids[
                            static_cast<std::size_t>(source_local_vertex_index)];
                    cell_data.slab_spatial_vertex_ids = slab_cell.spatial_vertex_ids;
                    cell_data.source_spatial_vertex_ids =
                        source_cell.spatial_vertex_ids;

                    auto [cells_it, inserted] =
                        incidence.vertex_to_slab_cells.try_emplace(
                            spatial_vertex_id);
                    if (inserted)
                        incidence.vertex_ids.push_back(spatial_vertex_id);
                    cells_it->second.push_back(cell_data);

                    auto [boundary_it, boundary_inserted] =
                        incidence.vertex_is_boundary.try_emplace(
                            spatial_vertex_id,
                            false);
                    (void)boundary_inserted;
                    if (local_vertex_touches_spatial_boundary_(
                            slab_cell,
                            spatial_vertex_id))
                    {
                        boundary_it->second = true;
                    }
                }
            }

            std::sort(incidence.vertex_ids.begin(), incidence.vertex_ids.end());
            return incidence;
        }

        static void build_for_slab_(
            PatchSetType& patch_set,
            int slab_id,
            BuildCounters* counters)
        {
            const auto& slab_space = patch_set.slab_space();
            const auto& slab = slab_space.slab(slab_id);
            const auto& slab_mesh = slab.mesh_ref();
            const auto& source_mesh = slab_space.source_space().mesh_ref();

            const auto incidence =
                build_slab_incidence_(slab, source_mesh, counters);
            for (const int spatial_vertex_id : incidence.vertex_ids)
            {
                const auto cells_it =
                    incidence.vertex_to_slab_cells.find(spatial_vertex_id);
                if (cells_it == incidence.vertex_to_slab_cells.end() ||
                    cells_it->second.empty())
                {
                    continue;
                }

                PatchType patch;
                patch.slab_id = slab_id;
                patch.spatial_vertex_id = spatial_vertex_id;
                patch.spatial_vertex =
                    slab_mesh.spatial_vertices()[static_cast<std::size_t>(spatial_vertex_id)];
                patch.t_begin = slab.t_begin();
                patch.t_end   = slab.t_end();
                const auto boundary_it =
                    incidence.vertex_is_boundary.find(spatial_vertex_id);
                patch.kind =
                    (boundary_it != incidence.vertex_is_boundary.end() &&
                     boundary_it->second)
                    ? TimeSlabVertexPatchKind::boundary
                    : TimeSlabVertexPatchKind::interior;
                patch.cells = cells_it->second;

                std::sort(
                    patch.cells.begin(),
                    patch.cells.end(),
                    [](const auto& a, const auto& b)
                    {
                        if (a.slab_cell_id != b.slab_cell_id)
                            return a.slab_cell_id < b.slab_cell_id;
                        return a.source_cell_id < b.source_cell_id;
                    });

                patch.n_cells = static_cast<int>(patch.cells.size());
                if (patch.n_cells > 0)
                    patch_set.add_patch_(std::move(patch));
            }
        }
    };
}

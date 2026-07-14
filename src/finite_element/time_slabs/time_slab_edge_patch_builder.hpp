#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../detail/space_time_capabilities.hpp"

#include "time_slab_space.hpp"

namespace finite_element::time_slabs
{
    enum class TimeSlabEdgePatchKind
    {
        interior,
        boundary
    };

    enum class TimeSlabEdgePatchCellSide
    {
        left_of_vertex,
        right_of_vertex
    };

    template<typename GeomTraits, typename FETraits>
    struct TimeSlabEdgePatch
    {
        using GT             = GeomTraits;
        using FETraitsType   = FETraits;
        using Types          = mesh::MeshTypes<GeomTraits>;
        using SpaceTimePoint = typename Types::SpaceTimePoint;

        struct CellData
        {
            int slab_cell_id     = -1;
            int source_cell_id   = -1;
            int left_vertex_id   = -1;
            int right_vertex_id  = -1;
            double x_begin       = 0.0;
            double x_end         = 0.0;
            TimeSlabEdgePatchCellSide side =
                TimeSlabEdgePatchCellSide::left_of_vertex;

            [[nodiscard]] double length() const noexcept
            {
                return x_end - x_begin;
            }
        };

        int patch_id           = -1;
        int slab_id            = -1;
        int spatial_vertex_id  = -1;
        double x_vertex        = 0.0;
        double t_begin         = 0.0;
        double t_end           = 0.0;
        TimeSlabEdgePatchKind kind = TimeSlabEdgePatchKind::boundary;
        int n_cells            = 0;
        std::array<CellData, 2> cells{};

        [[nodiscard]] bool is_boundary() const noexcept
        {
            return kind == TimeSlabEdgePatchKind::boundary;
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
                    "TimeSlabEdgePatch::cell: patch cell index out of range.");
            }

            return cells[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] double spatial_reference(
            int patch_cell_index,
            double x) const
        {
            const auto& patch_cell = cell(patch_cell_index);
            return (x - patch_cell.x_begin) / patch_cell.length();
        }

        [[nodiscard]] double time_reference(double t) const noexcept
        {
            return (t - t_begin) / time_length();
        }

        [[nodiscard]] SpaceTimePoint map_to_physical(
            int patch_cell_index,
            double x_ref,
            double t_ref) const noexcept
        {
            const auto& patch_cell = cells[static_cast<std::size_t>(patch_cell_index)];

            SpaceTimePoint p{};
            p[0] = patch_cell.x_begin + patch_cell.length() * x_ref;
            p[GT::dim_space_v] = t_begin + time_length() * t_ref;
            return p;
        }

        [[nodiscard]] double cell_jacobian_measure(int patch_cell_index) const
        {
            return cell(patch_cell_index).length() * time_length();
        }

        [[nodiscard]] double partition_of_unity_value(
            int patch_cell_index,
            double x_ref) const
        {
            const auto side = cell(patch_cell_index).side;

            // In 1D the hat function centered at the patch vertex is affine on
            // each adjacent cell, so the local partition of unity is just x_ref
            // on the left cell and 1 - x_ref on the right cell.
            if (side == TimeSlabEdgePatchCellSide::left_of_vertex)
                return x_ref;

            return 1.0 - x_ref;
        }

        [[nodiscard]] double partition_of_unity_dx(int patch_cell_index) const
        {
            const auto& patch_cell = cell(patch_cell_index);
            const double inv_h = 1.0 / patch_cell.length();

            if (patch_cell.side == TimeSlabEdgePatchCellSide::left_of_vertex)
                return inv_h;

            return -inv_h;
        }

        [[nodiscard]] double weighted_haar_value(int patch_cell_index) const
        {
            if (is_boundary())
            {
                throw std::runtime_error(
                    "TimeSlabEdgePatch::weighted_haar_value: Haar mode only exists on interior patches.");
            }

            const double left_h  = cells[0].length();
            const double right_h = cells[1].length();
            const double denom   = left_h + right_h;

            if (patch_cell_index == 0)
                return right_h / denom;
            if (patch_cell_index == 1)
                return -left_h / denom;

            throw std::runtime_error(
                "TimeSlabEdgePatch::weighted_haar_value: patch cell index out of range.");
        }

        [[nodiscard]] int find_patch_cell_index(int slab_cell_id) const noexcept
        {
            for (int patch_cell_index = 0; patch_cell_index < n_cells; ++patch_cell_index)
            {
                if (cells[static_cast<std::size_t>(patch_cell_index)].slab_cell_id == slab_cell_id)
                    return patch_cell_index;
            }

            return -1;
        }
    };

    template<class PatchType>
    inline constexpr bool is_time_slab_edge_patch_v = false;

    template<typename GeomTraits, typename FETraits>
    inline constexpr bool is_time_slab_edge_patch_v<
        TimeSlabEdgePatch<GeomTraits, FETraits>> = true;

    template<typename GeomTraits, typename FETraits>
    class TimeSlabEdgePatchSet
    {
    public:
        using GT            = GeomTraits;
        using FETraitsType  = FETraits;
        using SlabSpaceType = TimeSlabSpace<GeomTraits, FETraits>;
        using PatchType     = TimeSlabEdgePatch<GeomTraits, FETraits>;

        struct CellPatchMembership
        {
            int patch_id         = -1;
            int patch_cell_index = -1;

            [[nodiscard]] bool is_valid() const noexcept
            {
                return patch_id >= 0 && patch_cell_index >= 0;
            }
        };

        explicit TimeSlabEdgePatchSet(const SlabSpaceType& slab_space)
            : slab_space_(&slab_space),
              slab_data_(static_cast<std::size_t>(slab_space.n_slabs()))
        {
            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
            {
                auto& slab_data = slab_data_[static_cast<std::size_t>(slab_id)];
                const auto n_cells = slab_space.slab(slab_id).mesh_ref().n_cells();

                slab_data.memberships.resize(n_cells);
                slab_data.membership_counts.resize(n_cells, 0);
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
            if (patch_id < 0 || patch_id >= n_patches())
                throw std::runtime_error(
                    "TimeSlabEdgePatchSet::patch: patch id out of range.");

            return patches_[static_cast<std::size_t>(patch_id)];
        }

        [[nodiscard]] int cell_patch_count(
            int slab_id,
            int slab_cell_id) const
        {
            check_cell_index_(slab_id, slab_cell_id);
            return slab_data_[static_cast<std::size_t>(slab_id)]
                .membership_counts[static_cast<std::size_t>(slab_cell_id)];
        }

        [[nodiscard]] CellPatchMembership cell_patch_membership(
            int slab_id,
            int slab_cell_id,
            int membership_index) const
        {
            check_cell_index_(slab_id, slab_cell_id);

            const auto count = cell_patch_count(slab_id, slab_cell_id);
            if (membership_index < 0 || membership_index >= count)
            {
                throw std::runtime_error(
                    "TimeSlabEdgePatchSet::cell_patch_membership: membership index out of range.");
            }

            return slab_data_[static_cast<std::size_t>(slab_id)]
                .memberships[static_cast<std::size_t>(slab_cell_id)]
                [static_cast<std::size_t>(membership_index)];
        }

        [[nodiscard]] const std::vector<int>& slab_patch_ids(int slab_id) const
        {
            check_slab_index_(slab_id);
            return slab_data_[static_cast<std::size_t>(slab_id)].patch_ids;
        }

    private:
        template<typename, typename>
        friend class TimeSlabEdgePatchBuilder;

        struct SlabData
        {
            std::vector<std::array<CellPatchMembership, 2>> memberships{};
            std::vector<int> membership_counts{};
            std::vector<int> patch_ids{};
        };

        void add_patch_(PatchType patch)
        {
            const int patch_id = static_cast<int>(patches_.size());
            patch.patch_id = patch_id;

            if (patch.slab_id < 0 || patch.slab_id >= slab_space_->n_slabs())
            {
                throw std::runtime_error(
                    "TimeSlabEdgePatchSet::add_patch_: invalid slab id.");
            }

            auto& slab_data = slab_data_[static_cast<std::size_t>(patch.slab_id)];
            slab_data.patch_ids.push_back(patch_id);

            for (int patch_cell_index = 0; patch_cell_index < patch.n_cells; ++patch_cell_index)
            {
                const auto slab_cell_id =
                    patch.cells[static_cast<std::size_t>(patch_cell_index)].slab_cell_id;

                if (slab_cell_id < 0 ||
                    static_cast<std::size_t>(slab_cell_id) >= slab_data.memberships.size())
                {
                    throw std::runtime_error(
                        "TimeSlabEdgePatchSet::add_patch_: slab cell id out of range.");
                }

                auto& count =
                    slab_data.membership_counts[static_cast<std::size_t>(slab_cell_id)];
                if (count >= 2)
                {
                    throw std::runtime_error(
                        "TimeSlabEdgePatchSet::add_patch_: 1D slab cell has more than two patch memberships.");
                }

                slab_data.memberships[static_cast<std::size_t>(slab_cell_id)]
                    [static_cast<std::size_t>(count)] =
                    CellPatchMembership{patch_id, patch_cell_index};
                ++count;
            }

            patches_.push_back(std::move(patch));
        }

        void check_slab_index_(int slab_id) const
        {
            if (slab_id < 0 || slab_id >= slab_space_->n_slabs())
                throw std::runtime_error(
                    "TimeSlabEdgePatchSet: slab index out of range.");
        }

        void check_cell_index_(int slab_id, int slab_cell_id) const
        {
            check_slab_index_(slab_id);

            const auto& slab_data = slab_data_[static_cast<std::size_t>(slab_id)];
            if (slab_cell_id < 0 ||
                static_cast<std::size_t>(slab_cell_id) >= slab_data.memberships.size())
            {
                throw std::runtime_error(
                    "TimeSlabEdgePatchSet: slab cell index out of range.");
            }
        }

        const SlabSpaceType* slab_space_ = nullptr;
        std::vector<PatchType> patches_{};
        std::vector<SlabData> slab_data_{};
    };

    template<typename GeomTraits, typename FETraits>
    class TimeSlabEdgePatchBuilder
    {
    public:
        using GT            = GeomTraits;
        using FETraitsType  = FETraits;
        using SlabSpaceType = TimeSlabSpace<GeomTraits, FETraits>;
        using PatchType     = TimeSlabEdgePatch<GeomTraits, FETraits>;
        using PatchSetType  = TimeSlabEdgePatchSet<GeomTraits, FETraits>;

        [[nodiscard]] static PatchSetType build(const SlabSpaceType& slab_space)
        {
            finite_element::detail::require_current_1plus1d_space_time_capability<GT>();

            PatchSetType patch_set(slab_space);

            for (int slab_id = 0; slab_id < slab_space.n_slabs(); ++slab_id)
                build_for_slab_(patch_set, slab_id);

            return patch_set;
        }

    private:
        struct OrderedSlabCell
        {
            int slab_cell_id    = -1;
            int source_cell_id  = -1;
            int left_vertex_id  = -1;
            int right_vertex_id = -1;
            double x_begin      = 0.0;
            double x_end        = 0.0;
        };

        static void build_for_slab_(PatchSetType& patch_set, int slab_id)
        {
            const auto& slab = patch_set.slab_space().slab(slab_id);
            const auto ordered_cells = ordered_cells_(slab);

            if (ordered_cells.empty())
                return;

            patch_set.add_patch_(make_boundary_patch_(
                slab_id,
                slab.t_begin(),
                slab.t_end(),
                ordered_cells.front(),
                true));

            for (int i = 0; i + 1 < static_cast<int>(ordered_cells.size()); ++i)
            {
                patch_set.add_patch_(make_interior_patch_(
                    slab_id,
                    slab.t_begin(),
                    slab.t_end(),
                    ordered_cells[static_cast<std::size_t>(i)],
                    ordered_cells[static_cast<std::size_t>(i + 1)]));
            }

            patch_set.add_patch_(make_boundary_patch_(
                slab_id,
                slab.t_begin(),
                slab.t_end(),
                ordered_cells.back(),
                false));
        }

        template<typename SlabType>
        [[nodiscard]] static std::vector<OrderedSlabCell> ordered_cells_(
            const SlabType& slab)
        {
            std::vector<OrderedSlabCell> ordered_cells;
            ordered_cells.reserve(slab.active_cells().size());

            const auto& mesh = slab.mesh_ref();
            for (const int slab_cell_id : slab.active_cells())
            {
                const auto& cell = mesh.cell(slab_cell_id);

                OrderedSlabCell ordered;
                ordered.slab_cell_id   = slab_cell_id;
                ordered.source_cell_id = slab.source_cell_id(slab_cell_id);
                ordered.left_vertex_id = cell.spatial_vertex_ids[0];
                ordered.right_vertex_id = cell.spatial_vertex_ids[1];
                ordered.x_begin = mesh.spatial_vertices()[cell.spatial_vertex_ids[0]][0];
                ordered.x_end   = mesh.spatial_vertices()[cell.spatial_vertex_ids[1]][0];
                ordered_cells.push_back(ordered);
            }

            std::sort(
                ordered_cells.begin(),
                ordered_cells.end(),
                [](const OrderedSlabCell& a, const OrderedSlabCell& b)
                {
                    if (a.x_begin != b.x_begin)
                        return a.x_begin < b.x_begin;
                    if (a.x_end != b.x_end)
                        return a.x_end < b.x_end;
                    return a.slab_cell_id < b.slab_cell_id;
                });

            return ordered_cells;
        }

        [[nodiscard]] static PatchType make_boundary_patch_(
            int slab_id,
            double t_begin,
            double t_end,
            const OrderedSlabCell& cell,
            bool left_boundary)
        {
            PatchType patch;
            patch.slab_id   = slab_id;
            patch.t_begin   = t_begin;
            patch.t_end     = t_end;
            patch.kind      = TimeSlabEdgePatchKind::boundary;
            patch.n_cells   = 1;

            patch.spatial_vertex_id = left_boundary ? cell.left_vertex_id : cell.right_vertex_id;
            patch.x_vertex          = left_boundary ? cell.x_begin : cell.x_end;

            auto& patch_cell = patch.cells[0];
            patch_cell.slab_cell_id   = cell.slab_cell_id;
            patch_cell.source_cell_id = cell.source_cell_id;
            patch_cell.left_vertex_id = cell.left_vertex_id;
            patch_cell.right_vertex_id = cell.right_vertex_id;
            patch_cell.x_begin        = cell.x_begin;
            patch_cell.x_end          = cell.x_end;
            patch_cell.side = left_boundary
                ? TimeSlabEdgePatchCellSide::right_of_vertex
                : TimeSlabEdgePatchCellSide::left_of_vertex;

            return patch;
        }

        [[nodiscard]] static PatchType make_interior_patch_(
            int slab_id,
            double t_begin,
            double t_end,
            const OrderedSlabCell& left_cell,
            const OrderedSlabCell& right_cell)
        {
            constexpr double continuity_tol = 1.0e-12;

            if (left_cell.right_vertex_id != right_cell.left_vertex_id)
            {
                throw std::runtime_error(
                    "TimeSlabEdgePatchBuilder::make_interior_patch_: adjacent slab cells do not share a spatial vertex.");
            }

            if (std::abs(left_cell.x_end - right_cell.x_begin) > continuity_tol)
            {
                throw std::runtime_error(
                    "TimeSlabEdgePatchBuilder::make_interior_patch_: adjacent slab cells are not spatially contiguous.");
            }

            PatchType patch;
            patch.slab_id          = slab_id;
            patch.t_begin          = t_begin;
            patch.t_end            = t_end;
            patch.kind             = TimeSlabEdgePatchKind::interior;
            patch.n_cells          = 2;
            patch.spatial_vertex_id = left_cell.right_vertex_id;
            patch.x_vertex         = left_cell.x_end;

            auto& patch_left = patch.cells[0];
            patch_left.slab_cell_id    = left_cell.slab_cell_id;
            patch_left.source_cell_id  = left_cell.source_cell_id;
            patch_left.left_vertex_id  = left_cell.left_vertex_id;
            patch_left.right_vertex_id = left_cell.right_vertex_id;
            patch_left.x_begin         = left_cell.x_begin;
            patch_left.x_end           = left_cell.x_end;
            patch_left.side            = TimeSlabEdgePatchCellSide::left_of_vertex;

            auto& patch_right = patch.cells[1];
            patch_right.slab_cell_id    = right_cell.slab_cell_id;
            patch_right.source_cell_id  = right_cell.source_cell_id;
            patch_right.left_vertex_id  = right_cell.left_vertex_id;
            patch_right.right_vertex_id = right_cell.right_vertex_id;
            patch_right.x_begin         = right_cell.x_begin;
            patch_right.x_end           = right_cell.x_end;
            patch_right.side            = TimeSlabEdgePatchCellSide::right_of_vertex;

            return patch;
        }
    };
}

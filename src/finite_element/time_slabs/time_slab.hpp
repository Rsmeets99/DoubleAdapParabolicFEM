#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../mesh/mesh.hpp"
#include "../../mesh/refinement/time_slicing/slice_cell_in_time.hpp"
#include "../fespace/fespace.hpp"
#include "../fespace/policies.hpp"

namespace finite_element::time_slabs
{
    template<typename GeomTraits, typename FETraits>
    class TimeSlab
    {
    public:
        using GT              = GeomTraits;
        using FETraitsType    = FETraits;
        using PolicyType      = finite_element::SpaceOnlyPolicy;

        using MeshType        = mesh::Mesh<GeomTraits>;
        using Types           = mesh::MeshTypes<GeomTraits>;
        using SpaceType       = FESpace<GeomTraits, FETraits, PolicyType>;
        using SourceSpaceType = FESpace<GeomTraits, FETraits, PolicyType>;

        using SpaceTimePoint  = typename Types::SpaceTimePoint;
        using SpatialPoint    = typename Types::SpatialPoint;
        using TemporalPoint   = typename Types::TemporalPoint;

        struct SlicedCellInfo
        {
            int slab_local_cell_id = -1;
            int source_cell_id     = -1;
            int slab_id            = -1;
            int slab_time_begin_id = -1;
            int slab_time_end_id   = -1;
            double t_begin         = 0.0;
            double t_end           = 0.0;
            double source_t_begin  = 0.0;
            double source_t_end    = 0.0;
            std::array<int, Types::n_spatial_vertices> source_local_vertex_indices =
                []()
                {
                    std::array<int, Types::n_spatial_vertices> indices{};
                    indices.fill(-1);
                    return indices;
                }();
        };

        TimeSlab() = default;

        TimeSlab(int slab_id, double t_begin, double t_end)
            : slab_id_(slab_id),
              t_begin_(t_begin),
              t_end_(t_end)
        {}

        TimeSlab(const TimeSlab&) = delete;
        TimeSlab& operator=(const TimeSlab&) = delete;

        TimeSlab(TimeSlab&& other) noexcept
            : mesh_(std::move(other.mesh_)),
              slab_id_(other.slab_id_),
              t_begin_(other.t_begin_),
              t_end_(other.t_end_),
              active_cells_(std::move(other.active_cells_)),
              sliced_cell_info_(std::move(other.sliced_cell_info_)),
              boundary_metadata_initialized_(other.boundary_metadata_initialized_)
        {
            rebuild_fespace_if_needed_(other.fespace_.has_value());
        }

        TimeSlab& operator=(TimeSlab&& other) noexcept
        {
            if (this == &other)
                return *this;

            mesh_             = std::move(other.mesh_);
            slab_id_          = other.slab_id_;
            t_begin_          = other.t_begin_;
            t_end_            = other.t_end_;
            active_cells_     = std::move(other.active_cells_);
            sliced_cell_info_ = std::move(other.sliced_cell_info_);
            boundary_metadata_initialized_ =
                other.boundary_metadata_initialized_;

            rebuild_fespace_if_needed_(other.fespace_.has_value());
            return *this;
        }

        void clear()
        {
            mesh_.clear();
            fespace_.reset();
            active_cells_.clear();
            sliced_cell_info_.clear();
            slab_id_ = -1;
            t_begin_ = 0.0;
            t_end_   = 0.0;
            boundary_metadata_initialized_ = false;
        }

        void set_interval(int slab_id, double t_begin, double t_end)
        {
            slab_id_ = slab_id;
            t_begin_ = t_begin;
            t_end_   = t_end;
        }

        [[nodiscard]] int slab_id() const noexcept
        {
            return slab_id_;
        }

        [[nodiscard]] double t_begin() const noexcept
        {
            return t_begin_;
        }

        [[nodiscard]] double t_end() const noexcept
        {
            return t_end_;
        }

        [[nodiscard]] bool contains_time(double t) const noexcept
        {
            return t_begin_ <= t && t <= t_end_;
        }

        [[nodiscard]] MeshType& mesh_ref() noexcept
        {
            return mesh_;
        }

        [[nodiscard]] const MeshType& mesh_ref() const noexcept
        {
            return mesh_;
        }

        void inherit_spatial_boundary_metadata_from_source(
            const SourceSpaceType& source_space)
        {
            mesh_.inherit_spatial_boundary_metadata_from(
                source_space.mesh_ref());
            boundary_metadata_initialized_ = true;
        }

        [[nodiscard]] SpaceType& fespace_ref()
        {
            if (!fespace_.has_value())
                throw std::runtime_error("TimeSlab::fespace_ref: slab FESpace not initialized.");
            return *fespace_;
        }

        [[nodiscard]] const SpaceType& fespace_ref() const
        {
            if (!fespace_.has_value())
                throw std::runtime_error("TimeSlab::fespace_ref: slab FESpace not initialized.");
            return *fespace_;
        }

        [[nodiscard]] const std::vector<int>& active_cells() const noexcept
        {
            return active_cells_;
        }

        [[nodiscard]] int n_active_cells() const noexcept
        {
            return static_cast<int>(active_cells_.size());
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return active_cells_.empty();
        }

        int append_slice_from_source(
            const SourceSpaceType& source_space,
            int source_cell_id)
        {
            return append_slice_from_source(
                source_space,
                source_cell_id,
                -1,
                -1);
        }

        int append_slice_from_source(
            const SourceSpaceType& source_space,
            int source_cell_id,
            int slab_time_begin_id,
            int slab_time_end_id)
        {
            if (!(t_begin_ < t_end_))
                throw std::runtime_error(
                    "TimeSlab::append_slice_from_source: invalid slab time interval.");

            ensure_spatial_boundary_metadata_from_source_(source_space);

            const int local_cell_id =
                mesh::refinement::time_slicing::slice_cell_in_time<GeomTraits>(
                    mesh_,
                    source_space.mesh_ref(),
                    source_cell_id,
                    t_begin_,
                    t_end_);

            active_cells_.push_back(local_cell_id);

            const auto& source_cell = source_space.mesh_ref().cell(source_cell_id);
            const double source_t_begin =
                source_space.mesh_ref().temporal_vertices()[
                    source_cell.temporal_vertex_ids[0]][0];
            const double source_t_end =
                source_space.mesh_ref().temporal_vertices()[
                    source_cell.temporal_vertex_ids[1]][0];

            SlicedCellInfo info;
            info.slab_local_cell_id = local_cell_id;
            info.source_cell_id     = source_cell_id;
            info.slab_id            = slab_id_;
            info.slab_time_begin_id = slab_time_begin_id;
            info.slab_time_end_id   = slab_time_end_id;
            info.t_begin            = t_begin_;
            info.t_end              = t_end_;
            info.source_t_begin     = source_t_begin;
            info.source_t_end       = source_t_end;
            for (int local_vertex = 0;
                 local_vertex < Types::n_spatial_vertices;
                 ++local_vertex)
            {
                info.source_local_vertex_indices[
                    static_cast<std::size_t>(local_vertex)] = local_vertex;
            }
            sliced_cell_info_.push_back(info);

            return local_cell_id;
        }

        void finalize()
        {
            fespace_.reset();
            fespace_.emplace(mesh_);
            fespace_->initialize(
                active_cells_,
                slab_fespace_initialization_options_());
        }

        [[nodiscard]] int source_cell_id(int slab_local_cell_id) const
        {
            if (slab_local_cell_id < 0 ||
                static_cast<std::size_t>(slab_local_cell_id) >= sliced_cell_info_.size())
            {
                throw std::runtime_error(
                    "TimeSlab::source_cell_id: slab-local cell id out of range.");
            }

            return sliced_cell_info_[static_cast<std::size_t>(slab_local_cell_id)].source_cell_id;
        }

        [[nodiscard]] const SlicedCellInfo& sliced_cell_info(int slab_local_cell_id) const
        {
            if (slab_local_cell_id < 0 ||
                static_cast<std::size_t>(slab_local_cell_id) >= sliced_cell_info_.size())
            {
                throw std::runtime_error(
                    "TimeSlab::sliced_cell_info: slab-local cell id out of range.");
            }

            return sliced_cell_info_[static_cast<std::size_t>(slab_local_cell_id)];
        }

        [[nodiscard]] const std::vector<SlicedCellInfo>& sliced_cell_infos() const noexcept
        {
            return sliced_cell_info_;
        }

    private:
        void ensure_spatial_boundary_metadata_from_source_(
            const SourceSpaceType& source_space)
        {
            if (!boundary_metadata_initialized_)
                inherit_spatial_boundary_metadata_from_source(source_space);
        }

        void rebuild_fespace_if_needed_(bool had_fespace)
        {
            fespace_.reset();
            if (had_fespace)
            {
                fespace_.emplace(mesh_);
                fespace_->initialize(
                    active_cells_,
                    slab_fespace_initialization_options_());
            }
        }

        [[nodiscard]] static finite_element::FESpaceInitializationOptions
        slab_fespace_initialization_options_()
        {
            finite_element::FESpaceInitializationOptions options;
            options.search_index =
                finite_element::SearchIndexBuildMode::Disabled;
            options.build_refinement_indices = false;
            options.build_edge_interval_index = false;
            return options;
        }

        MeshType mesh_{};
        std::optional<SpaceType> fespace_{};

        int slab_id_    = -1;
        double t_begin_ = 0.0;
        double t_end_   = 0.0;

        std::vector<int> active_cells_{};
        std::vector<SlicedCellInfo> sliced_cell_info_{};
        bool boundary_metadata_initialized_ = false;
    };
}

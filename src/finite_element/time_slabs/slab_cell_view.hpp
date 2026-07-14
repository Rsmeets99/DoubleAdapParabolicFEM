#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../../mesh/topology/temporal_keys.hpp"
#include "time_slab_overlay.hpp"
#include "time_slab_space.hpp"

namespace finite_element::time_slabs
{
    struct SlabCellKey
    {
        int slab_id = -1;
        int slab_local_ordinal = -1;
        int source_y_cell_id = -1;
        int slab_time_begin_id = -1;
        int slab_time_end_id = -1;

        [[nodiscard]] bool operator==(const SlabCellKey&) const noexcept =
            default;

        [[nodiscard]] bool operator<(const SlabCellKey& other) const noexcept
        {
            if (slab_id != other.slab_id)
                return slab_id < other.slab_id;
            if (slab_local_ordinal != other.slab_local_ordinal)
                return slab_local_ordinal < other.slab_local_ordinal;
            if (source_y_cell_id != other.source_y_cell_id)
                return source_y_cell_id < other.source_y_cell_id;
            if (slab_time_begin_id != other.slab_time_begin_id)
                return slab_time_begin_id < other.slab_time_begin_id;
            return slab_time_end_id < other.slab_time_end_id;
        }
    };

    struct SlabTimeIntervalView
    {
        int begin_id = -1;
        int end_id = -1;
        double t_begin = 0.0;
        double t_end = 0.0;
        mesh::topology::TimeIntervalIdKey endpoint_id_key{};
    };

    struct SourceTimeIntervalView
    {
        int begin_temporal_vertex_id = -1;
        int end_temporal_vertex_id = -1;
        double t_begin = 0.0;
        double t_end = 0.0;
        mesh::topology::TimeIntervalIdKey temporal_vertex_interval_key{};
        std::optional<mesh::topology::DyadicTimeIntervalKey> dyadic_key{};
    };

    template<typename GeomTraits>
    class SlabCellView
    {
    public:
        using GT = GeomTraits;
        using MeshType = mesh::Mesh<GeomTraits>;
        using CellType = typename MeshType::CellType;
        using Types = mesh::MeshTypes<GeomTraits>;
        using SpatialPoint = typename Types::SpatialPoint;
        using SpatialVertexIds = typename Types::SpatialVertexIds;
        using SpatialBoundaryFlags =
            std::array<bool, Types::n_spatial_faces>;
        using SourceLocalVertexIndices =
            std::array<int, Types::n_spatial_vertices>;

        struct Data
        {
            TimeSlabBackend backend = TimeSlabBackend::CopiedMesh;
            SlabCellKey key{};
            std::optional<int> copied_slab_local_cell_id{};
            SlabTimeIntervalView slab_time_interval{};
            SourceTimeIntervalView source_time_interval{};
            SpatialVertexIds source_spatial_vertex_ids{};
            SourceLocalVertexIndices source_local_vertex_indices{};
            SpatialBoundaryFlags spatial_boundary{};
            const MeshType* source_mesh = nullptr;
            const MeshType* copied_slab_mesh = nullptr;
        };

        SlabCellView() = default;

        explicit SlabCellView(Data data)
            : data_(data)
        {
            require_source_mesh_();
        }

        [[nodiscard]] TimeSlabBackend backend() const noexcept
        {
            return data_.backend;
        }

        [[nodiscard]] const char* backend_name() const noexcept
        {
            return time_slab_backend_name(data_.backend);
        }

        [[nodiscard]] bool is_copied_backend() const noexcept
        {
            return data_.backend == TimeSlabBackend::CopiedMesh;
        }

        [[nodiscard]] const SlabCellKey& key() const noexcept
        {
            return data_.key;
        }

        [[nodiscard]] int slab_id() const noexcept
        {
            return data_.key.slab_id;
        }

        [[nodiscard]] int slab_local_ordinal() const noexcept
        {
            return data_.key.slab_local_ordinal;
        }

        [[nodiscard]] int source_y_cell_id() const noexcept
        {
            return data_.key.source_y_cell_id;
        }

        [[nodiscard]] std::optional<int>
        copied_slab_local_cell_id() const noexcept
        {
            return data_.copied_slab_local_cell_id;
        }

        [[nodiscard]] const SlabTimeIntervalView&
        slab_time_interval() const noexcept
        {
            return data_.slab_time_interval;
        }

        [[nodiscard]] const SourceTimeIntervalView&
        source_time_interval() const noexcept
        {
            return data_.source_time_interval;
        }

        [[nodiscard]] const SpatialVertexIds&
        source_spatial_vertex_ids() const noexcept
        {
            return data_.source_spatial_vertex_ids;
        }

        [[nodiscard]] const SourceLocalVertexIndices&
        source_local_vertex_indices() const noexcept
        {
            return data_.source_local_vertex_indices;
        }

        [[nodiscard]] int source_local_vertex_index(
            const int slab_local_vertex_index) const
        {
            if (slab_local_vertex_index < 0 ||
                slab_local_vertex_index >= Types::n_spatial_vertices)
            {
                throw std::runtime_error(
                    "SlabCellView::source_local_vertex_index: local vertex "
                    "index out of range.");
            }
            return data_.source_local_vertex_indices[
                static_cast<std::size_t>(slab_local_vertex_index)];
        }

        [[nodiscard]] int source_spatial_vertex_id(
            const int slab_local_vertex_index) const
        {
            const int source_local =
                source_local_vertex_index(slab_local_vertex_index);
            if (source_local < 0 || source_local >= Types::n_spatial_vertices)
            {
                throw std::runtime_error(
                    "SlabCellView::source_spatial_vertex_id: mapped source "
                    "local vertex index out of range.");
            }
            return data_.source_spatial_vertex_ids[
                static_cast<std::size_t>(source_local)];
        }

        [[nodiscard]] const SpatialPoint& source_spatial_vertex(
            const int slab_local_vertex_index) const
        {
            return source_mesh().spatial_vertices()[
                static_cast<std::size_t>(
                    source_spatial_vertex_id(slab_local_vertex_index))];
        }

        [[nodiscard]] const SpatialBoundaryFlags&
        spatial_boundary() const noexcept
        {
            return data_.spatial_boundary;
        }

        [[nodiscard]] bool spatial_face_on_boundary(int face_id) const
        {
            if (face_id < 0 || face_id >= Types::n_spatial_faces)
            {
                throw std::runtime_error(
                    "SlabCellView::spatial_face_on_boundary: face id out of "
                    "range.");
            }
            return data_.spatial_boundary[static_cast<std::size_t>(face_id)];
        }

        [[nodiscard]] const MeshType& source_mesh() const
        {
            require_source_mesh_();
            return *data_.source_mesh;
        }

        [[nodiscard]] const CellType& source_cell() const
        {
            return source_mesh().cell(source_y_cell_id());
        }

        [[nodiscard]] bool has_copied_slab_cell() const noexcept
        {
            return data_.copied_slab_local_cell_id.has_value() &&
                   data_.copied_slab_mesh != nullptr;
        }

        [[nodiscard]] const MeshType& copied_slab_mesh() const
        {
            if (data_.copied_slab_mesh == nullptr)
            {
                throw std::runtime_error(
                    "SlabCellView::copied_slab_mesh: copied slab mesh not "
                    "available for this backend.");
            }
            return *data_.copied_slab_mesh;
        }

        [[nodiscard]] const CellType& copied_slab_cell() const
        {
            if (!has_copied_slab_cell())
            {
                throw std::runtime_error(
                    "SlabCellView::copied_slab_cell: copied slab cell not "
                    "available for this backend.");
            }
            return copied_slab_mesh().cell(*data_.copied_slab_local_cell_id);
        }

        [[nodiscard]] std::array<double, 2>
        slab_time_coordinates() const noexcept
        {
            return {
                data_.slab_time_interval.t_begin,
                data_.slab_time_interval.t_end
            };
        }

        [[nodiscard]] std::array<double, 2>
        source_time_coordinates() const noexcept
        {
            return {
                data_.source_time_interval.t_begin,
                data_.source_time_interval.t_end
            };
        }

    private:
        void require_source_mesh_() const
        {
            if (data_.source_mesh == nullptr)
            {
                throw std::runtime_error(
                    "SlabCellView: source mesh pointer is null.");
            }
        }

        Data data_{};
    };

    namespace detail
    {
        template<typename GeomTraits>
        [[nodiscard]] inline SourceTimeIntervalView source_time_interval_view(
            const mesh::Mesh<GeomTraits>& mesh,
            const int source_cell_id)
        {
            const auto& cell = mesh.cell(source_cell_id);

            SourceTimeIntervalView out;
            out.begin_temporal_vertex_id = cell.temporal_vertex_ids[0];
            out.end_temporal_vertex_id = cell.temporal_vertex_ids[1];
            out.t_begin =
                mesh.temporal_vertices()[
                    static_cast<std::size_t>(
                        out.begin_temporal_vertex_id)][0];
            out.t_end =
                mesh.temporal_vertices()[
                    static_cast<std::size_t>(
                        out.end_temporal_vertex_id)][0];
            out.temporal_vertex_interval_key =
                mesh::topology::make_time_interval_id_key(
                    out.begin_temporal_vertex_id,
                    out.end_temporal_vertex_id);
            out.dyadic_key = mesh.dyadic_temporal_interval_key(source_cell_id);
            return out;
        }

        [[nodiscard]] inline SlabTimeIntervalView slab_time_interval_view(
            const int begin_id,
            const int end_id,
            const double t_begin,
            const double t_end)
        {
            return SlabTimeIntervalView{
                begin_id,
                end_id,
                t_begin,
                t_end,
                mesh::topology::make_time_interval_id_key(begin_id, end_id)
            };
        }

        template<typename GeomTraits>
        [[nodiscard]] inline typename SlabCellView<GeomTraits>::SourceLocalVertexIndices
        identity_source_local_vertex_indices()
        {
            typename SlabCellView<GeomTraits>::SourceLocalVertexIndices out{};
            for (int local = 0;
                 local < mesh::MeshTypes<GeomTraits>::n_spatial_vertices;
                 ++local)
            {
                out[static_cast<std::size_t>(local)] = local;
            }
            return out;
        }
    }

    template<typename GeomTraits, typename FETraits>
    [[nodiscard]] inline SlabCellView<GeomTraits>
    make_copied_slab_cell_view(
        const TimeSlabSpace<GeomTraits, FETraits>& slab_space,
        const int slab_id,
        const int slab_local_ordinal)
    {
        const auto& slab = slab_space.slab(slab_id);
        if (slab_local_ordinal < 0 ||
            slab_local_ordinal >= slab.n_active_cells())
        {
            throw std::runtime_error(
                "make_copied_slab_cell_view: slab-local ordinal out of "
                "range.");
        }

        const auto& info = slab.sliced_cell_info(slab_local_ordinal);
        const auto& source_mesh = slab_space.source_space().mesh_ref();
        const auto& source_cell = source_mesh.cell(info.source_cell_id);
        const auto& copied_mesh = slab.mesh_ref();
        const auto& copied_cell = copied_mesh.cell(info.slab_local_cell_id);

        typename SlabCellView<GeomTraits>::Data data;
        data.backend = TimeSlabBackend::CopiedMesh;
        data.key = SlabCellKey{
            info.slab_id,
            slab_local_ordinal,
            info.source_cell_id,
            info.slab_time_begin_id,
            info.slab_time_end_id};
        data.copied_slab_local_cell_id = info.slab_local_cell_id;
        data.slab_time_interval =
            detail::slab_time_interval_view(
                info.slab_time_begin_id,
                info.slab_time_end_id,
                info.t_begin,
                info.t_end);
        data.source_time_interval =
            detail::source_time_interval_view(
                source_mesh,
                info.source_cell_id);
        data.source_spatial_vertex_ids = source_cell.spatial_vertex_ids;
        data.source_local_vertex_indices =
            info.source_local_vertex_indices;
        data.spatial_boundary = copied_cell.spatial_boundary;
        data.source_mesh = &source_mesh;
        data.copied_slab_mesh = &copied_mesh;
        return SlabCellView<GeomTraits>(data);
    }

    template<typename GeomTraits, typename FETraits>
    [[nodiscard]] inline std::vector<SlabCellView<GeomTraits>>
    copied_slab_cell_views_on_slab(
        const TimeSlabSpace<GeomTraits, FETraits>& slab_space,
        const int slab_id)
    {
        const auto& slab = slab_space.slab(slab_id);
        std::vector<SlabCellView<GeomTraits>> out;
        out.reserve(static_cast<std::size_t>(slab.n_active_cells()));
        for (int local = 0; local < slab.n_active_cells(); ++local)
            out.push_back(make_copied_slab_cell_view(slab_space, slab_id, local));
        return out;
    }

}

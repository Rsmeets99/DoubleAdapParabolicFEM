#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../../linear_algebra/concepts/vector.hpp"
#include "../../mesh/topology/boundary_2d.hpp"
#include "../basis/functions/raviart_thomas_triangle.hpp"
#include "../basis/polynomials/dubiner_basis.hpp"
#include "../time_slabs/slab_cell_view.hpp"
#include "../time_slabs/time_slab_vertex_patch_builder.hpp"
#include "patch_dof_map.hpp"
#include "patch_edge_topology_2d.hpp"
#include "quadrature/reference_quadrature.hpp"

namespace finite_element::error_fespace
{
    template<class PatchType>
    class CopiedPatchCellView2D
    {
    public:
        using Patch = PatchType;
        using GT = typename PatchType::GT;
        using FETraitsType = typename PatchType::FETraitsType;
        using Types = mesh::MeshTypes<GT>;
        using SlabSpaceType =
            time_slabs::TimeSlabSpace<GT, FETraitsType>;
        using SlabCellViewType = time_slabs::SlabCellView<GT>;
        using SpatialPoint = typename Types::SpatialPoint;
        using SpatialVertexIds = typename Types::SpatialVertexIds;
        using SpatialFaceVertexIds = typename Types::SpatialFaceVertexIds;

        CopiedPatchCellView2D(
            const PatchType& patch,
            const SlabSpaceType& slab_space)
            : patch_(&patch),
              slab_space_(&slab_space)
        {
            validate_patch_slab_();
            build_cell_views_();
        }

        [[nodiscard]] time_slabs::TimeSlabBackend backend() const noexcept
        {
            return time_slabs::TimeSlabBackend::CopiedMesh;
        }

        [[nodiscard]] const PatchType& patch() const noexcept
        {
            return *patch_;
        }

        [[nodiscard]] const SlabSpaceType& slab_space() const noexcept
        {
            return *slab_space_;
        }

        [[nodiscard]] int n_slabs() const noexcept
        {
            return slab_space_->n_slabs();
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return patch_->n_cells;
        }

        [[nodiscard]] const SlabCellViewType& slab_cell_view(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return slab_cell_views_[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] int slab_local_ordinal(int patch_cell_index) const
        {
            return slab_cell_view(patch_cell_index).slab_local_ordinal();
        }

        [[nodiscard]] time_slabs::SlabCellKey cell_key(
            int patch_cell_index) const
        {
            return slab_cell_view(patch_cell_index).key();
        }

        [[nodiscard]] SpatialVertexIds cell_spatial_vertex_ids(
            int patch_cell_index) const
        {
            return slab_cell_view(patch_cell_index).source_spatial_vertex_ids();
        }

        [[nodiscard]] SpatialFaceVertexIds cell_face_vertex_ids(
            int patch_cell_index,
            int local_face_id) const
        {
            check_local_face_id_(local_face_id);
            const auto& view = slab_cell_view(patch_cell_index);
            if (view.has_copied_slab_cell())
            {
                return view.copied_slab_cell()
                    .spatial_faces[static_cast<std::size_t>(local_face_id)]
                    .spatial_vertex_ids;
            }

            return view.source_cell()
                .spatial_faces[static_cast<std::size_t>(local_face_id)]
                .spatial_vertex_ids;
        }

        [[nodiscard]] const SpatialPoint& spatial_vertex(int vertex_id) const
        {
            return slab_cell_view(0).source_mesh().spatial_vertices()[
                static_cast<std::size_t>(vertex_id)];
        }

        [[nodiscard]] bool face_on_physical_boundary(
            int patch_cell_index,
            int local_face_id) const
        {
            check_local_face_id_(local_face_id);
            const auto& view = slab_cell_view(patch_cell_index);
            if (view.spatial_face_on_boundary(local_face_id))
                return true;

            auto edge = cell_face_vertex_ids(patch_cell_index, local_face_id);
            std::sort(edge.begin(), edge.end());
            return mesh::topology::
                spatial_edge_lies_on_mesh_boundary_2d<GT>(
                    edge,
                    view.source_mesh().spatial_boundary_face_vertex_ids(),
                    view.source_mesh().spatial_vertices());
        }

    private:
        void validate_patch_slab_() const
        {
            if (patch_->slab_id < 0 || patch_->slab_id >= slab_space_->n_slabs())
            {
                throw std::runtime_error(
                    "CopiedPatchCellView2D: patch slab id out of range.");
            }
        }

        void build_cell_views_()
        {
            slab_cell_views_.clear();
            slab_cell_views_.reserve(
                static_cast<std::size_t>(patch_->n_cells));
            for (int patch_cell_index = 0;
                 patch_cell_index < patch_->n_cells;
                 ++patch_cell_index)
            {
                const auto& patch_cell = patch_->cell(patch_cell_index);
                slab_cell_views_.push_back(
                    time_slabs::make_copied_slab_cell_view(
                        *slab_space_,
                        patch_->slab_id,
                        patch_cell.slab_cell_id));
            }
        }

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 ||
                patch_cell_index >= static_cast<int>(slab_cell_views_.size()))
            {
                throw std::runtime_error(
                    "CopiedPatchCellView2D: patch cell index out of range.");
            }
        }

        static void check_local_face_id_(int local_face_id)
        {
            if (local_face_id < 0 ||
                local_face_id >= Types::n_spatial_faces)
            {
                throw std::runtime_error(
                    "CopiedPatchCellView2D: local face id out of range.");
            }
        }

        const PatchType* patch_ = nullptr;
        const SlabSpaceType* slab_space_ = nullptr;
        std::vector<SlabCellViewType> slab_cell_views_{};
    };

    template<class PatchCellView>
    class PatchEdgeTopologyBuilder2DFromView
    {
    public:
        using GT = typename PatchCellView::GT;
        using FETraitsType = typename PatchCellView::FETraitsType;
        using Types = mesh::MeshTypes<GT>;
        using TopologyType = PatchEdgeTopology2D<GT, FETraitsType>;
        using EdgeType = typename TopologyType::Edge;
        using IncidentCell = typename TopologyType::IncidentCell;
        using EdgeVertexIds = typename TopologyType::EdgeVertexIds;
        using SpatialPoint = typename TopologyType::SpatialPoint;

        [[nodiscard]] static TopologyType build(const PatchCellView& view)
        {
            static_assert(GT::dim_space_v == 2);
            static_assert(GT::dim_time_v == 1);

            TopologyType topology;
            for (int patch_cell_index = 0;
                 patch_cell_index < view.n_patch_cells();
                 ++patch_cell_index)
            {
                for (int local_face_id = 0;
                     local_face_id < Types::n_spatial_faces;
                     ++local_face_id)
                {
                    const auto edge_vertices =
                        view.cell_face_vertex_ids(
                            patch_cell_index,
                            local_face_id);
                    const auto canonical_vertices =
                        canonical_edge_vertices_(edge_vertices);

                    EdgeType* edge = find_edge_(topology, canonical_vertices);
                    if (edge == nullptr)
                    {
                        EdgeType new_edge;
                        new_edge.canonical_vertex_ids = canonical_vertices;
                        new_edge.canonical_normal =
                            canonical_normal_(view, canonical_vertices);
                        new_edge.length =
                            edge_length_(view, canonical_vertices);
                        topology.edges.push_back(std::move(new_edge));
                        edge = &topology.edges.back();
                    }

                    IncidentCell incident;
                    incident.patch_cell_index = patch_cell_index;
                    incident.slab_cell_id =
                        view.slab_local_ordinal(patch_cell_index);
                    incident.local_face_id = local_face_id;
                    incident.orientation_sign =
                        orientation_sign_(
                            view,
                            patch_cell_index,
                            local_face_id,
                            edge->canonical_normal);
                    edge->incident_cells.push_back(incident);
                }
            }

            finalize_(topology, view);
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

        [[nodiscard]] static std::array<double, 2> right_normal_(
            const SpatialPoint& a,
            const SpatialPoint& b)
        {
            const double dx = b[0] - a[0];
            const double dy = b[1] - a[1];
            const double length = std::sqrt(dx * dx + dy * dy);
            if (length <= 1.0e-15)
            {
                throw std::runtime_error(
                    "PatchEdgeTopologyBuilder2DFromView: degenerate edge.");
            }
            return {dy / length, -dx / length};
        }

        [[nodiscard]] static double edge_length_(
            const PatchCellView& view,
            const EdgeVertexIds& canonical_vertices)
        {
            const auto& a = view.spatial_vertex(canonical_vertices[0]);
            const auto& b = view.spatial_vertex(canonical_vertices[1]);
            const double dx = b[0] - a[0];
            const double dy = b[1] - a[1];
            return std::sqrt(dx * dx + dy * dy);
        }

        [[nodiscard]] static SpatialPoint canonical_normal_(
            const PatchCellView& view,
            const EdgeVertexIds& canonical_vertices)
        {
            const auto& a = view.spatial_vertex(canonical_vertices[0]);
            const auto& b = view.spatial_vertex(canonical_vertices[1]);
            const auto normal = right_normal_(a, b);
            return SpatialPoint{normal[0], normal[1]};
        }

        [[nodiscard]] static std::array<double, 2> local_outward_normal_(
            const PatchCellView& view,
            int patch_cell_index,
            int local_face_id)
        {
            const auto edge_vertices =
                view.cell_face_vertex_ids(patch_cell_index, local_face_id);
            const auto& a = view.spatial_vertex(edge_vertices[0]);
            const auto& b = view.spatial_vertex(edge_vertices[1]);
            return right_normal_(a, b);
        }

        [[nodiscard]] static int orientation_sign_(
            const PatchCellView& view,
            int patch_cell_index,
            int local_face_id,
            const SpatialPoint& canonical_normal)
        {
            const auto outward =
                local_outward_normal_(view, patch_cell_index, local_face_id);
            const double dot =
                outward[0] * canonical_normal[0] +
                outward[1] * canonical_normal[1];
            if (dot > 1.0e-10)
                return 1;
            if (dot < -1.0e-10)
                return -1;

            throw std::runtime_error(
                "PatchEdgeTopologyBuilder2DFromView: local outward normal is "
                "not aligned with canonical normal.");
        }

        static void finalize_(
            TopologyType& topology,
            const PatchCellView& view)
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
                    const auto& incident = edge.incident_cells.front();
                    edge.on_physical_boundary =
                        view.face_on_physical_boundary(
                            incident.patch_cell_index,
                            incident.local_face_id);
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
                        "PatchEdgeTopologyBuilder2DFromView: patch edge has "
                        "invalid incident-cell count.");
                }
            }
        }
    };

    template<class PatchCellView, int PSpace>
    class PatchScalarSpace2DView
    {
    public:
        using PatchView = PatchCellView;
        using Patch = typename PatchCellView::Patch;
        using GT = typename PatchCellView::GT;
        using FETraitsType = typename PatchCellView::FETraitsType;
        using Types = mesh::MeshTypes<GT>;
        using ReferencePoint = std::array<double, 2>;

        static_assert(GT::dim_space_v == 2);
        static_assert(GT::dim_time_v == 1);
        static_assert(PSpace >= 1);
        static_assert(PSpace <= 10);

        static constexpr int p_space_v = PSpace;
        static constexpr int local_dofs_v = (PSpace + 1) * (PSpace + 2) / 2;
        static constexpr int constraint_quadrature_degree_v = PSpace;

        using LocalValues = std::array<double, local_dofs_v>;
        using LocalDofMap = std::array<int, local_dofs_v>;
        using Basis = basis::polynomials::DubinerBasis<PSpace>;

        explicit PatchScalarSpace2DView(PatchCellView patch_view)
            : patch_view_(std::move(patch_view))
        {
            build_cell_dof_maps_();
            build_mean_zero_constraint_row_();
        }

        [[nodiscard]] const PatchCellView& patch_view() const noexcept
        {
            return patch_view_;
        }

        [[nodiscard]] const Patch& patch() const noexcept
        {
            return patch_view_.patch();
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return patch_view_.n_patch_cells();
        }

        [[nodiscard]] int n_local_dofs_per_cell() const noexcept
        {
            return local_dofs_v;
        }

        [[nodiscard]] int n_broken_dofs() const noexcept
        {
            return n_patch_cells() * local_dofs_v;
        }

        [[nodiscard]] int n_effective_dofs() const noexcept
        {
            return n_broken_dofs() - (has_mean_zero_constraint() ? 1 : 0);
        }

        [[nodiscard]] int effective_dimension() const noexcept
        {
            return n_effective_dofs();
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return n_broken_dofs();
        }

        [[nodiscard]] bool has_mean_zero_constraint() const noexcept
        {
            return !patch().is_boundary();
        }

        [[nodiscard]] const std::vector<double>&
        mean_zero_constraint_row() const noexcept
        {
            return mean_zero_constraint_row_;
        }

        [[nodiscard]] const std::vector<LocalDofMap>&
        cell_dof_maps() const noexcept
        {
            return cell_dof_maps_;
        }

        [[nodiscard]] const LocalDofMap& cell_dof_map(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int local_dof_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_local_dof_index_(local_dof_id);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)]
                                 [static_cast<std::size_t>(local_dof_id)];
        }

        static void evaluate_local_basis(
            const ReferencePoint& x_ref,
            LocalValues& values)
        {
            values = Basis::eval_all(x_ref[0], x_ref[1]);
        }

        void evaluate_local_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            LocalValues& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            evaluate_local_basis(x_ref, values);
        }

        void evaluate_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            std::vector<double>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(static_cast<std::size_t>(n_broken_dofs()), 0.0);

            LocalValues local_values{};
            evaluate_local_basis(x_ref, local_values);
            for (int local_dof_id = 0;
                 local_dof_id < local_dofs_v;
                 ++local_dof_id)
            {
                values[static_cast<std::size_t>(
                    local_to_patch_dof(patch_cell_index, local_dof_id))] =
                    local_values[static_cast<std::size_t>(local_dof_id)];
            }
        }

        [[nodiscard]] double spatial_jacobian_measure(
            int patch_cell_index) const
        {
            const auto ids =
                patch_view_.cell_spatial_vertex_ids(patch_cell_index);
            const auto& v0 = patch_view_.spatial_vertex(ids[0]);
            const auto& v1 = patch_view_.spatial_vertex(ids[1]);
            const auto& v2 = patch_view_.spatial_vertex(ids[2]);
            const double det =
                (v1[0] - v0[0]) * (v2[1] - v0[1]) -
                (v1[1] - v0[1]) * (v2[0] - v0[0]);
            return std::abs(det);
        }

        [[nodiscard]] double spatial_cell_measure(
            int patch_cell_index) const
        {
            return 0.5 * spatial_jacobian_measure(patch_cell_index);
        }

        [[nodiscard]] double patch_measure() const
        {
            double measure = 0.0;
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                measure += spatial_cell_measure(patch_cell_index);
            }
            return measure;
        }

        [[nodiscard]] ReferencePoint physical_to_reference_on_patch_cell(
            int patch_cell_index,
            const typename Types::SpatialPoint& p) const
        {
            const auto ids =
                patch_view_.cell_spatial_vertex_ids(patch_cell_index);
            const auto& v0 = patch_view_.spatial_vertex(ids[0]);
            const auto& v1 = patch_view_.spatial_vertex(ids[1]);
            const auto& v2 = patch_view_.spatial_vertex(ids[2]);
            const double j00 = v1[0] - v0[0];
            const double j01 = v2[0] - v0[0];
            const double j10 = v1[1] - v0[1];
            const double j11 = v2[1] - v0[1];
            const double det = j00 * j11 - j01 * j10;
            if (std::abs(det) <= 1.0e-15)
            {
                throw std::runtime_error(
                    "PatchScalarSpace2DView: degenerate patch cell.");
            }

            const double dx = p[0] - v0[0];
            const double dy = p[1] - v0[1];
            return ReferencePoint{
                ( j11 * dx - j01 * dy) / det,
                (-j10 * dx + j00 * dy) / det};
        }

        [[nodiscard]] double mean_zero_constraint_value(
            const std::vector<double>& broken_coefficients) const
        {
            if (!has_mean_zero_constraint())
            {
                throw std::runtime_error(
                    "PatchScalarSpace2DView::mean_zero_constraint_value: "
                    "boundary patches have no mean-zero constraint.");
            }
            if (broken_coefficients.size() !=
                static_cast<std::size_t>(n_broken_dofs()))
            {
                throw std::runtime_error(
                    "PatchScalarSpace2DView::mean_zero_constraint_value: "
                    "coefficient size mismatch.");
            }

            double value = 0.0;
            for (int i = 0; i < n_broken_dofs(); ++i)
                value += mean_zero_constraint_row_[static_cast<std::size_t>(i)] *
                         broken_coefficients[static_cast<std::size_t>(i)];
            return value;
        }

    private:
        PatchCellView patch_view_;
        std::vector<LocalDofMap> cell_dof_maps_{};
        std::vector<double> mean_zero_constraint_row_{};

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchScalarSpace2DView: patch cell index out of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpace2DView: local scalar DoF index out of "
                    "range.");
            }
        }

        void build_cell_dof_maps_()
        {
            cell_dof_maps_.resize(static_cast<std::size_t>(n_patch_cells()));
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                auto& map =
                    cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
                for (int local_dof_id = 0;
                     local_dof_id < local_dofs_v;
                     ++local_dof_id)
                {
                    map[static_cast<std::size_t>(local_dof_id)] =
                        patch_cell_index * local_dofs_v + local_dof_id;
                }
            }
        }

        [[nodiscard]] double integrate_local_basis_(
            int patch_cell_index,
            int local_dof_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_local_dof_index_(local_dof_id);

            constexpr auto rule =
                quadrature::reference::make_reference_triangle_quadrature<
                    constraint_quadrature_degree_v>();
            const double jac = spatial_jacobian_measure(patch_cell_index);
            double integral = 0.0;
            for (int q = 0; q < rule.n_points; ++q)
            {
                const auto values =
                    Basis::eval_all(rule.points[static_cast<std::size_t>(q)][0],
                                    rule.points[static_cast<std::size_t>(q)][1]);
                integral +=
                    values[static_cast<std::size_t>(local_dof_id)] *
                    rule.weights[static_cast<std::size_t>(q)] *
                    jac;
            }
            return integral;
        }

        void build_mean_zero_constraint_row_()
        {
            mean_zero_constraint_row_.clear();
            if (!has_mean_zero_constraint())
                return;

            mean_zero_constraint_row_.assign(
                static_cast<std::size_t>(n_broken_dofs()),
                0.0);
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                for (int local_dof_id = 0;
                     local_dof_id < local_dofs_v;
                     ++local_dof_id)
                {
                    mean_zero_constraint_row_[static_cast<std::size_t>(
                        local_to_patch_dof(patch_cell_index, local_dof_id))] =
                        integrate_local_basis_(patch_cell_index, local_dof_id);
                }
            }
        }
    };

    template<class PatchCellView, int PSpace>
    class PatchRTFluxSpace2DView
    {
    public:
        using PatchView = PatchCellView;
        using Patch = typename PatchCellView::Patch;
        using GT = typename PatchCellView::GT;
        using FETraitsType = typename PatchCellView::FETraitsType;
        using Types = mesh::MeshTypes<GT>;
        using TopologyBuilder = PatchEdgeTopologyBuilder2DFromView<PatchCellView>;
        using TopologyType = typename TopologyBuilder::TopologyType;
        using ReferenceBasis =
            basis::functions::RaviartThomasTriangleBasis<PSpace>;
        using PiolaBasis =
            basis::functions::RaviartThomasTrianglePiolaBasis<PSpace>;
        using AffineMap = typename PiolaBasis::AffineMap;
        using ReferencePoint = typename ReferenceBasis::Point;
        using VectorValue = typename ReferenceBasis::VectorValue;
        using LocalValues = typename ReferenceBasis::Values;
        using LocalDivergences = typename ReferenceBasis::Divergences;

        static_assert(GT::dim_space_v == 2);
        static_assert(GT::dim_time_v == 1);
        static_assert(PSpace >= 1 && PSpace <= 10);

        static constexpr int p_space_v = PSpace;
        static constexpr int local_dofs_v = ReferenceBasis::N;
        static constexpr int local_edge_dofs_v = ReferenceBasis::edge_dofs;
        static constexpr int local_interior_dofs_v =
            ReferenceBasis::interior_dofs;
        static constexpr int edge_moments_per_edge_v = PSpace + 1;
        static constexpr bool uses_dense_nullspace_v = false;

        struct LocalDofMapEntry
        {
            int patch_dof_id = -1;
            int orientation_sign = 1;
            int edge_id = -1;
            int local_face_id = -1;
            bool is_edge_dof = false;
            bool is_constrained = false;
        };

        using LocalDofMap = std::array<LocalDofMapEntry, local_dofs_v>;

        struct EdgeDofBlock
        {
            int edge_id = -1;
            int first_patch_dof = -1;
            bool is_free = false;
            bool is_constrained = true;
        };

        explicit PatchRTFluxSpace2DView(PatchCellView patch_view)
            : patch_view_(std::move(patch_view)),
              topology_(TopologyBuilder::build(patch_view_))
        {
            build_physical_map_cache_();
            build_edge_dof_blocks_();
            build_cell_dof_maps_();
        }

        PatchRTFluxSpace2DView(
            PatchCellView patch_view,
            TopologyType topology)
            : patch_view_(std::move(patch_view)),
              topology_(std::move(topology))
        {
            build_physical_map_cache_();
            build_edge_dof_blocks_();
            build_cell_dof_maps_();
        }

        [[nodiscard]] const PatchCellView& patch_view() const noexcept
        {
            return patch_view_;
        }

        [[nodiscard]] const Patch& patch() const noexcept
        {
            return patch_view_.patch();
        }

        [[nodiscard]] const TopologyType& topology() const noexcept
        {
            return topology_;
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return patch_view_.n_patch_cells();
        }

        [[nodiscard]] int n_local_dofs_per_cell() const noexcept
        {
            return local_dofs_v;
        }

        [[nodiscard]] int n_local_edge_dofs_per_cell() const noexcept
        {
            return local_edge_dofs_v;
        }

        [[nodiscard]] int n_local_interior_dofs_per_cell() const noexcept
        {
            return local_interior_dofs_v;
        }

        [[nodiscard]] int n_free_patch_edges() const noexcept
        {
            return static_cast<int>(free_edge_ids_.size());
        }

        [[nodiscard]] int n_constrained_patch_edges() const noexcept
        {
            return static_cast<int>(constrained_edge_ids_.size());
        }

        [[nodiscard]] int n_free_edge_dofs() const noexcept
        {
            return n_free_patch_edges() * edge_moments_per_edge_v;
        }

        [[nodiscard]] int n_cell_interior_dofs() const noexcept
        {
            return n_patch_cells() * local_interior_dofs_v;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return n_free_edge_dofs() + n_cell_interior_dofs();
        }

        [[nodiscard]] int effective_dimension() const noexcept
        {
            return n_dofs();
        }

        [[nodiscard]] bool uses_dense_nullspace() const noexcept
        {
            return uses_dense_nullspace_v;
        }

        [[nodiscard]] const std::vector<int>& free_edge_ids() const noexcept
        {
            return free_edge_ids_;
        }

        [[nodiscard]] const std::vector<int>& constrained_edge_ids() const noexcept
        {
            return constrained_edge_ids_;
        }

        [[nodiscard]] const std::vector<EdgeDofBlock>&
        edge_dof_blocks() const noexcept
        {
            return edge_dof_blocks_;
        }

        [[nodiscard]] const EdgeDofBlock& edge_dof_block(int edge_id) const
        {
            check_edge_index_(edge_id);
            return edge_dof_blocks_[static_cast<std::size_t>(edge_id)];
        }

        [[nodiscard]] bool edge_is_free(int edge_id) const
        {
            return edge_dof_block(edge_id).is_free;
        }

        [[nodiscard]] bool edge_is_constrained(int edge_id) const
        {
            return edge_dof_block(edge_id).is_constrained;
        }

        [[nodiscard]] int edge_patch_dof(
            int edge_id,
            int edge_moment_id) const
        {
            check_edge_index_(edge_id);
            check_edge_moment_index_(edge_moment_id);
            const auto& block =
                edge_dof_blocks_[static_cast<std::size_t>(edge_id)];
            if (!block.is_free)
                return -1;
            return block.first_patch_dof + edge_moment_id;
        }

        [[nodiscard]] const std::vector<LocalDofMap>&
        cell_dof_maps() const noexcept
        {
            return cell_dof_maps_;
        }

        [[nodiscard]] const LocalDofMap& cell_dof_map(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] const LocalDofMapEntry& local_dof_entry(
            int patch_cell_index,
            int local_dof_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_local_dof_index_(local_dof_id);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)]
                                 [static_cast<std::size_t>(local_dof_id)];
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(patch_cell_index, local_dof_id).patch_dof_id;
        }

        [[nodiscard]] int local_orientation_sign(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(
                patch_cell_index,
                local_dof_id).orientation_sign;
        }

        [[nodiscard]] int local_edge_id(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(patch_cell_index, local_dof_id).edge_id;
        }

        [[nodiscard]] int local_edge_parameter_orientation_sign(
            int patch_cell_index,
            int local_face_id,
            int edge_moment_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            if (local_face_id < 0 || local_face_id >= Types::n_spatial_faces)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2DView: local face id out of range.");
            }
            check_edge_moment_index_(edge_moment_id);
            const int edge_id =
                topology_.edge_id_for_patch_cell_face(
                    patch_cell_index,
                    local_face_id);
            return edge_parameter_orientation_sign_(
                patch_cell_index,
                local_face_id,
                edge_id,
                edge_moment_id);
        }

        [[nodiscard]] bool local_dof_is_constrained(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(
                patch_cell_index,
                local_dof_id).is_constrained;
        }

        [[nodiscard]] AffineMap physical_map_for_patch_cell(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return physical_map_cache_[static_cast<std::size_t>(
                patch_cell_index)];
        }

        [[nodiscard]] ReferencePoint physical_to_reference_on_patch_cell(
            int patch_cell_index,
            const typename Types::SpatialPoint& p) const
        {
            return PiolaBasis::physical_to_reference(
                physical_map_for_patch_cell(patch_cell_index),
                p);
        }

        static void evaluate_reference_local_basis(
            const ReferencePoint& x_ref,
            LocalValues& values)
        {
            values = ReferenceBasis::eval_all(x_ref);
        }

        void evaluate_physical_local_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            LocalValues& values) const
        {
            values = PiolaBasis::eval_all(
                physical_map_for_patch_cell(patch_cell_index),
                x_ref);
        }

        void evaluate_physical_local_divergences_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            LocalDivergences& divergences) const
        {
            divergences = PiolaBasis::div_all(
                physical_map_for_patch_cell(patch_cell_index),
                x_ref);
        }

        void evaluate_reference_patch_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            std::vector<VectorValue>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(
                static_cast<std::size_t>(n_dofs()),
                VectorValue{0.0, 0.0});
            LocalValues local_values{};
            evaluate_reference_local_basis(x_ref, local_values);
            scatter_local_values_to_patch_(
                patch_cell_index,
                local_values,
                values);
        }

        void evaluate_physical_patch_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            std::vector<VectorValue>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(
                static_cast<std::size_t>(n_dofs()),
                VectorValue{0.0, 0.0});
            LocalValues local_values{};
            evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                local_values);
            scatter_local_values_to_patch_(
                patch_cell_index,
                local_values,
                values);
        }

    private:
        PatchCellView patch_view_;
        TopologyType topology_{};
        std::vector<EdgeDofBlock> edge_dof_blocks_{};
        std::vector<int> free_edge_ids_{};
        std::vector<int> constrained_edge_ids_{};
        std::vector<LocalDofMap> cell_dof_maps_{};
        std::vector<AffineMap> physical_map_cache_{};

        void build_physical_map_cache_()
        {
            physical_map_cache_.resize(
                static_cast<std::size_t>(n_patch_cells()));
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                const auto ids =
                    patch_view_.cell_spatial_vertex_ids(patch_cell_index);
                const auto& v0 = patch_view_.spatial_vertex(ids[0]);
                const auto& v1 = patch_view_.spatial_vertex(ids[1]);
                const auto& v2 = patch_view_.spatial_vertex(ids[2]);
                physical_map_cache_[static_cast<std::size_t>(
                    patch_cell_index)] =
                    PiolaBasis::make_affine_map(v0, v1, v2);
            }
        }

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2DView: patch cell index out of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2DView: local RT DoF index out of range.");
            }
        }

        void check_edge_index_(int edge_id) const
        {
            if (edge_id < 0 || edge_id >= topology_.n_edges())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2DView: patch edge id out of range.");
            }
        }

        static void check_edge_moment_index_(int edge_moment_id)
        {
            if (edge_moment_id < 0 ||
                edge_moment_id >= edge_moments_per_edge_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2DView: edge moment index out of range.");
            }
        }

        [[nodiscard]] bool edge_should_be_free_(
            const typename TopologyType::Edge& edge) const noexcept
        {
            if (edge.is_internal())
                return true;
            return patch().is_boundary() && edge.on_physical_boundary;
        }

        void build_edge_dof_blocks_()
        {
            edge_dof_blocks_.assign(
                static_cast<std::size_t>(topology_.n_edges()),
                EdgeDofBlock{});
            free_edge_ids_.clear();
            constrained_edge_ids_.clear();

            int next_patch_dof = 0;
            for (int edge_id = 0; edge_id < topology_.n_edges(); ++edge_id)
            {
                const auto& edge = topology_.edge(edge_id);
                auto& block =
                    edge_dof_blocks_[static_cast<std::size_t>(edge_id)];
                block.edge_id = edge_id;
                block.is_free = edge_should_be_free_(edge);
                block.is_constrained = !block.is_free;
                if (block.is_free)
                {
                    block.first_patch_dof = next_patch_dof;
                    next_patch_dof += edge_moments_per_edge_v;
                    free_edge_ids_.push_back(edge_id);
                }
                else
                {
                    block.first_patch_dof = -1;
                    constrained_edge_ids_.push_back(edge_id);
                }
            }
        }

        [[nodiscard]] const typename TopologyType::IncidentCell*
        incident_for_patch_cell_face_(
            int patch_cell_index,
            int local_face_id) const noexcept
        {
            const auto* edge =
                topology_.edge_for_patch_cell_face(
                    patch_cell_index,
                    local_face_id);
            if (edge == nullptr)
                return nullptr;
            for (const auto& incident : edge->incident_cells)
            {
                if (incident.patch_cell_index == patch_cell_index &&
                    incident.local_face_id == local_face_id)
                {
                    return &incident;
                }
            }
            return nullptr;
        }

        void build_cell_dof_maps_()
        {
            cell_dof_maps_.resize(static_cast<std::size_t>(n_patch_cells()));
            const int interior_offset = n_free_edge_dofs();
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                auto& map =
                    cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
                for (int local_face_id = 0;
                     local_face_id < Types::n_spatial_faces;
                     ++local_face_id)
                {
                    const auto* incident =
                        incident_for_patch_cell_face_(
                            patch_cell_index,
                            local_face_id);
                    if (incident == nullptr)
                    {
                        throw std::runtime_error(
                            "PatchRTFluxSpace2DView: missing patch edge "
                            "incident.");
                    }

                    const int edge_id =
                        topology_.edge_id_for_patch_cell_face(
                            patch_cell_index,
                            local_face_id);
                    const auto& block = edge_dof_block(edge_id);
                    for (int edge_moment_id = 0;
                         edge_moment_id < edge_moments_per_edge_v;
                         ++edge_moment_id)
                    {
                        const int local_dof_id =
                            local_face_id * edge_moments_per_edge_v +
                            edge_moment_id;
                        auto& entry =
                            map[static_cast<std::size_t>(local_dof_id)];
                        entry.patch_dof_id =
                            block.is_free
                                ? block.first_patch_dof + edge_moment_id
                                : -1;
                        entry.orientation_sign =
                            incident->orientation_sign *
                            edge_parameter_orientation_sign_(
                                patch_cell_index,
                                local_face_id,
                                edge_id,
                                edge_moment_id);
                        entry.edge_id = edge_id;
                        entry.local_face_id = local_face_id;
                        entry.is_edge_dof = true;
                        entry.is_constrained = block.is_constrained;
                    }
                }

                for (int local_interior_id = 0;
                     local_interior_id < local_interior_dofs_v;
                     ++local_interior_id)
                {
                    const int local_dof_id =
                        local_edge_dofs_v + local_interior_id;
                    auto& entry =
                        map[static_cast<std::size_t>(local_dof_id)];
                    entry.patch_dof_id =
                        interior_offset +
                        patch_cell_index * local_interior_dofs_v +
                        local_interior_id;
                    entry.orientation_sign = 1;
                    entry.edge_id = -1;
                    entry.local_face_id = -1;
                    entry.is_edge_dof = false;
                    entry.is_constrained = false;
                }
            }
        }

        [[nodiscard]] int edge_parameter_orientation_sign_(
            int patch_cell_index,
            int local_face_id,
            int edge_id,
            int edge_moment_id) const
        {
            const auto local_vertices =
                patch_view_.cell_face_vertex_ids(
                    patch_cell_index,
                    local_face_id);
            const auto canonical_vertices =
                topology_.edge(edge_id).canonical_vertex_ids;
            if (local_vertices[0] == canonical_vertices[0] &&
                local_vertices[1] == canonical_vertices[1])
            {
                return 1;
            }
            if (local_vertices[0] == canonical_vertices[1] &&
                local_vertices[1] == canonical_vertices[0])
            {
                return (edge_moment_id % 2 == 0) ? 1 : -1;
            }
            throw std::runtime_error(
                "PatchRTFluxSpace2DView: local edge vertices do not match "
                "canonical edge vertices.");
        }

        void scatter_local_values_to_patch_(
            int patch_cell_index,
            const LocalValues& local_values,
            std::vector<VectorValue>& patch_values) const
        {
            const auto& map = cell_dof_map(patch_cell_index);
            for (int local_dof_id = 0;
                 local_dof_id < local_dofs_v;
                 ++local_dof_id)
            {
                const auto& entry =
                    map[static_cast<std::size_t>(local_dof_id)];
                if (entry.patch_dof_id < 0)
                    continue;
                auto& value =
                    patch_values[static_cast<std::size_t>(entry.patch_dof_id)];
                const auto& local_value =
                    local_values[static_cast<std::size_t>(local_dof_id)];
                value[0] += static_cast<double>(entry.orientation_sign) *
                            local_value[0];
                value[1] += static_cast<double>(entry.orientation_sign) *
                            local_value[1];
            }
        }
    };

    template<class SpatialSpaceType, int PTime>
    class PatchScalarSpaceTime2DView
    {
    public:
        using SpatialSpace = SpatialSpaceType;
        using Patch = typename SpatialSpace::Patch;
        using GT = typename SpatialSpace::GT;
        using SpatialReferencePoint =
            typename SpatialSpace::ReferencePoint;
        using SpaceTimeReferencePoint = std::array<double, 3>;
        using TimeValues = std::array<double, PTime + 1>;

        static_assert(PTime >= 1);
        static constexpr int p_space_v = SpatialSpace::p_space_v;
        static constexpr int p_time_v = PTime;
        static constexpr int n_time_dofs_v = PTime + 1;
        static constexpr int spatial_local_dofs_v =
            SpatialSpace::local_dofs_v;
        static constexpr int local_dofs_v =
            spatial_local_dofs_v * n_time_dofs_v;

        using LocalValues = std::array<double, local_dofs_v>;
        using LocalDofMap = std::array<int, local_dofs_v>;

        explicit PatchScalarSpaceTime2DView(SpatialSpace spatial_space)
            : spatial_space_(std::move(spatial_space))
        {
            build_cell_dof_maps_();
            build_mean_zero_constraint_rows_();
        }

        [[nodiscard]] const SpatialSpace& spatial_space() const noexcept
        {
            return spatial_space_;
        }

        [[nodiscard]] SpatialSpace& spatial_space() noexcept
        {
            return spatial_space_;
        }

        [[nodiscard]] const Patch& patch() const noexcept
        {
            return spatial_space_.patch();
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return spatial_space_.n_patch_cells();
        }

        [[nodiscard]] int n_time_dofs() const noexcept
        {
            return n_time_dofs_v;
        }

        [[nodiscard]] int n_spatial_broken_dofs() const noexcept
        {
            return spatial_space_.n_broken_dofs();
        }

        [[nodiscard]] int n_spatial_effective_dofs() const noexcept
        {
            return spatial_space_.effective_dimension();
        }

        [[nodiscard]] int n_local_dofs_per_cell() const noexcept
        {
            return local_dofs_v;
        }

        [[nodiscard]] int n_broken_dofs() const noexcept
        {
            return spatial_space_.n_broken_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return n_broken_dofs();
        }

        [[nodiscard]] int n_effective_dofs() const noexcept
        {
            return spatial_space_.effective_dimension() * n_time_dofs_v;
        }

        [[nodiscard]] int effective_dimension() const noexcept
        {
            return n_effective_dofs();
        }

        [[nodiscard]] bool has_mean_zero_constraint() const noexcept
        {
            return spatial_space_.has_mean_zero_constraint();
        }

        [[nodiscard]] int n_mean_zero_constraints() const noexcept
        {
            return has_mean_zero_constraint() ? n_time_dofs_v : 0;
        }

        [[nodiscard]] const std::vector<std::vector<double>>&
        mean_zero_constraint_rows() const noexcept
        {
            return mean_zero_constraint_rows_;
        }

        [[nodiscard]] const std::vector<double>& mean_zero_constraint_row(
            int time_dof_id) const
        {
            check_constraint_time_dof_index_(time_dof_id);
            return mean_zero_constraint_rows_[static_cast<std::size_t>(
                time_dof_id)];
        }

        [[nodiscard]] double time_length() const noexcept
        {
            return patch().time_length();
        }

        [[nodiscard]] double map_time_to_physical(double t_ref) const noexcept
        {
            return patch().t_begin + t_ref * time_length();
        }

        [[nodiscard]] double map_time_to_reference(double t) const noexcept
        {
            return (t - patch().t_begin) / time_length();
        }

        static void evaluate_time_basis(double t_ref, TimeValues& values)
        {
            values = detail::shifted_legendre_family<PTime>(t_ref);
        }

        [[nodiscard]] int patch_dof_index(
            int spatial_patch_dof_id,
            int time_dof_id) const
        {
            check_spatial_patch_dof_index_(spatial_patch_dof_id);
            check_time_dof_index_(time_dof_id);
            return spatial_patch_dof_id * n_time_dofs_v + time_dof_id;
        }

        [[nodiscard]] int spatial_patch_dof_id(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return patch_dof_id / n_time_dofs_v;
        }

        [[nodiscard]] int time_dof_id(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return patch_dof_id % n_time_dofs_v;
        }

        [[nodiscard]] int local_dof_index(
            int spatial_local_dof_id,
            int time_dof_id) const
        {
            check_spatial_local_dof_index_(spatial_local_dof_id);
            check_time_dof_index_(time_dof_id);
            return spatial_local_dof_id * n_time_dofs_v + time_dof_id;
        }

        [[nodiscard]] int spatial_local_dof_id(int local_dof_id) const
        {
            check_local_dof_index_(local_dof_id);
            return local_dof_id / n_time_dofs_v;
        }

        [[nodiscard]] int local_time_dof_id(int local_dof_id) const
        {
            check_local_dof_index_(local_dof_id);
            return local_dof_id % n_time_dofs_v;
        }

        [[nodiscard]] const std::vector<LocalDofMap>&
        cell_dof_maps() const noexcept
        {
            return cell_dof_maps_;
        }

        [[nodiscard]] const LocalDofMap& cell_dof_map(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int local_dof_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_local_dof_index_(local_dof_id);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)]
                                 [static_cast<std::size_t>(local_dof_id)];
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int spatial_local_dof_id,
            int time_dof_id) const
        {
            return local_to_patch_dof(
                patch_cell_index,
                local_dof_index(spatial_local_dof_id, time_dof_id));
        }

        [[nodiscard]] double mean_zero_constraint_value(
            const std::vector<double>& coefficients,
            int time_dof_id) const
        {
            const auto& row = mean_zero_constraint_row(time_dof_id);
            if (coefficients.size() != static_cast<std::size_t>(n_dofs()))
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView::mean_zero_constraint_value: "
                    "coefficient size mismatch.");
            }

            double value = 0.0;
            for (int i = 0; i < n_dofs(); ++i)
            {
                value += row[static_cast<std::size_t>(i)] *
                         coefficients[static_cast<std::size_t>(i)];
            }
            return value;
        }

        [[nodiscard]] typename SpatialSpace::ReferencePoint
        physical_to_reference_on_patch_cell(
            int patch_cell_index,
            const typename mesh::MeshTypes<GT>::SpatialPoint& p) const
        {
            return spatial_space_.physical_to_reference_on_patch_cell(
                patch_cell_index,
                p);
        }

        void evaluate_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            LocalValues& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.fill(0.0);
            typename SpatialSpace::LocalValues spatial_values{};
            spatial_space_.evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                spatial_values);
            TimeValues time_values{};
            evaluate_time_basis(t_ref, time_values);
            for (int spatial_local_dof = 0;
                 spatial_local_dof < spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                for (int time_dof = 0;
                     time_dof < n_time_dofs_v;
                     ++time_dof)
                {
                    values[static_cast<std::size_t>(
                        local_dof_index(spatial_local_dof, time_dof))] =
                        spatial_values[static_cast<std::size_t>(
                            spatial_local_dof)] *
                        time_values[static_cast<std::size_t>(time_dof)];
                }
            }
        }

        void evaluate_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpaceTimeReferencePoint& x_ref,
            LocalValues& values) const
        {
            evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                SpatialReferencePoint{x_ref[0], x_ref[1]},
                x_ref[2],
                values);
        }

        void evaluate_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            std::vector<double>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(static_cast<std::size_t>(n_dofs()), 0.0);

            LocalValues local_values{};
            evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);

            for (int local_dof_id = 0;
                 local_dof_id < local_dofs_v;
                 ++local_dof_id)
            {
                values[static_cast<std::size_t>(
                    local_to_patch_dof(patch_cell_index, local_dof_id))] =
                    local_values[static_cast<std::size_t>(local_dof_id)];
            }
        }

    private:
        SpatialSpace spatial_space_;
        std::vector<LocalDofMap> cell_dof_maps_{};
        std::vector<std::vector<double>> mean_zero_constraint_rows_{};

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView: patch cell index out of "
                    "range.");
            }
        }

        static void check_time_dof_index_(int time_dof_id)
        {
            if (time_dof_id < 0 || time_dof_id >= n_time_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView: time DoF index out of range.");
            }
        }

        void check_constraint_time_dof_index_(int time_dof_id) const
        {
            if (!has_mean_zero_constraint())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView: boundary patches have no "
                    "mean-zero constraints.");
            }
            check_time_dof_index_(time_dof_id);
        }

        static void check_spatial_local_dof_index_(int spatial_local_dof_id)
        {
            if (spatial_local_dof_id < 0 ||
                spatial_local_dof_id >= spatial_local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView: spatial local DoF index out "
                    "of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView: local DoF index out of range.");
            }
        }

        void check_spatial_patch_dof_index_(int spatial_patch_dof_id) const
        {
            if (spatial_patch_dof_id < 0 ||
                spatial_patch_dof_id >= spatial_space_.n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView: spatial patch DoF index out "
                    "of range.");
            }
        }

        void check_patch_dof_index_(int patch_dof_id) const
        {
            if (patch_dof_id < 0 || patch_dof_id >= n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarSpaceTime2DView: patch DoF index out of "
                    "range.");
            }
        }

        void build_cell_dof_maps_()
        {
            cell_dof_maps_.resize(static_cast<std::size_t>(n_patch_cells()));
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                auto& map =
                    cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
                for (int spatial_local_dof = 0;
                     spatial_local_dof < spatial_local_dofs_v;
                     ++spatial_local_dof)
                {
                    const int spatial_patch_dof =
                        spatial_space_.local_to_patch_dof(
                            patch_cell_index,
                            spatial_local_dof);
                    for (int time_dof = 0;
                         time_dof < n_time_dofs_v;
                         ++time_dof)
                    {
                        map[static_cast<std::size_t>(
                            local_dof_index(spatial_local_dof, time_dof))] =
                            patch_dof_index(spatial_patch_dof, time_dof);
                    }
                }
            }
        }

        void build_mean_zero_constraint_rows_()
        {
            mean_zero_constraint_rows_.clear();
            if (!has_mean_zero_constraint())
                return;
            mean_zero_constraint_rows_.assign(
                static_cast<std::size_t>(n_time_dofs_v),
                std::vector<double>(static_cast<std::size_t>(n_dofs()), 0.0));
            const auto& spatial_row =
                spatial_space_.mean_zero_constraint_row();
            for (int spatial_patch_dof = 0;
                 spatial_patch_dof < spatial_space_.n_dofs();
                 ++spatial_patch_dof)
            {
                for (int time_dof = 0;
                     time_dof < n_time_dofs_v;
                     ++time_dof)
                {
                    mean_zero_constraint_rows_[static_cast<std::size_t>(
                        time_dof)][static_cast<std::size_t>(
                            patch_dof_index(spatial_patch_dof, time_dof))] =
                        spatial_row[static_cast<std::size_t>(
                            spatial_patch_dof)];
                }
            }
        }
    };

    template<class SpatialSpaceType, int PTime>
    class PatchRTFluxSpaceTime2DView
    {
    public:
        using SpatialSpace = SpatialSpaceType;
        using Patch = typename SpatialSpace::Patch;
        using GT = typename SpatialSpace::GT;
        using TopologyType = typename SpatialSpace::TopologyType;
        using ReferenceBasis = typename SpatialSpace::ReferenceBasis;
        using PiolaBasis = typename SpatialSpace::PiolaBasis;
        using SpatialReferencePoint =
            typename SpatialSpace::ReferencePoint;
        using SpaceTimeReferencePoint = std::array<double, 3>;
        using VectorValue = typename SpatialSpace::VectorValue;
        using TimeValues = std::array<double, PTime + 1>;

        static_assert(PTime >= 1);
        static constexpr int p_space_v = SpatialSpace::p_space_v;
        static constexpr int p_time_v = PTime;
        static constexpr int n_time_dofs_v = PTime + 1;
        static constexpr int spatial_local_dofs_v =
            SpatialSpace::local_dofs_v;
        static constexpr int local_dofs_v =
            spatial_local_dofs_v * n_time_dofs_v;
        static constexpr int spatial_local_edge_dofs_v =
            SpatialSpace::local_edge_dofs_v;
        static constexpr int spatial_local_interior_dofs_v =
            SpatialSpace::local_interior_dofs_v;
        static constexpr int local_edge_dofs_v =
            spatial_local_edge_dofs_v * n_time_dofs_v;
        static constexpr int local_interior_dofs_v =
            spatial_local_interior_dofs_v * n_time_dofs_v;
        static constexpr int edge_moments_per_edge_v =
            SpatialSpace::edge_moments_per_edge_v;
        static constexpr bool uses_dense_nullspace_v =
            SpatialSpace::uses_dense_nullspace_v;

        using LocalValues = std::array<VectorValue, local_dofs_v>;
        using LocalDivergences = std::array<double, local_dofs_v>;

        struct LocalDofMapEntry
        {
            int patch_dof_id = -1;
            int orientation_sign = 1;
            int spatial_patch_dof_id = -1;
            int spatial_local_dof_id = -1;
            int time_dof_id = -1;
            int edge_id = -1;
            int local_face_id = -1;
            bool is_edge_dof = false;
            bool is_constrained = false;
        };

        using LocalDofMap = std::array<LocalDofMapEntry, local_dofs_v>;

        explicit PatchRTFluxSpaceTime2DView(SpatialSpace spatial_space)
            : spatial_space_(std::move(spatial_space))
        {
            build_cell_dof_maps_();
        }

        [[nodiscard]] const SpatialSpace& spatial_space() const noexcept
        {
            return spatial_space_;
        }

        [[nodiscard]] SpatialSpace& spatial_space() noexcept
        {
            return spatial_space_;
        }

        [[nodiscard]] const Patch& patch() const noexcept
        {
            return spatial_space_.patch();
        }

        [[nodiscard]] const TopologyType& topology() const noexcept
        {
            return spatial_space_.topology();
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return spatial_space_.n_patch_cells();
        }

        [[nodiscard]] int n_time_dofs() const noexcept
        {
            return n_time_dofs_v;
        }

        [[nodiscard]] int n_spatial_dofs() const noexcept
        {
            return spatial_space_.n_dofs();
        }

        [[nodiscard]] int n_local_dofs_per_cell() const noexcept
        {
            return local_dofs_v;
        }

        [[nodiscard]] int n_local_edge_dofs_per_cell() const noexcept
        {
            return local_edge_dofs_v;
        }

        [[nodiscard]] int n_local_interior_dofs_per_cell() const noexcept
        {
            return local_interior_dofs_v;
        }

        [[nodiscard]] int n_free_patch_edges() const noexcept
        {
            return spatial_space_.n_free_patch_edges();
        }

        [[nodiscard]] int n_constrained_patch_edges() const noexcept
        {
            return spatial_space_.n_constrained_patch_edges();
        }

        [[nodiscard]] int n_free_edge_dofs() const noexcept
        {
            return spatial_space_.n_free_edge_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int n_cell_interior_dofs() const noexcept
        {
            return spatial_space_.n_cell_interior_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            return spatial_space_.n_dofs() * n_time_dofs_v;
        }

        [[nodiscard]] int effective_dimension() const noexcept
        {
            return n_dofs();
        }

        [[nodiscard]] bool uses_dense_nullspace() const noexcept
        {
            return spatial_space_.uses_dense_nullspace();
        }

        [[nodiscard]] double time_length() const noexcept
        {
            return patch().time_length();
        }

        [[nodiscard]] double map_time_to_physical(double t_ref) const noexcept
        {
            return patch().t_begin + t_ref * time_length();
        }

        [[nodiscard]] double map_time_to_reference(double t) const noexcept
        {
            return (t - patch().t_begin) / time_length();
        }

        static void evaluate_time_basis(double t_ref, TimeValues& values)
        {
            values = detail::shifted_legendre_family<PTime>(t_ref);
        }

        [[nodiscard]] int patch_dof_index(
            int spatial_patch_dof_id,
            int time_dof_id) const
        {
            check_spatial_patch_dof_index_(spatial_patch_dof_id);
            check_time_dof_index_(time_dof_id);
            return spatial_patch_dof_id * n_time_dofs_v + time_dof_id;
        }

        [[nodiscard]] int spatial_patch_dof_id(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return patch_dof_id / n_time_dofs_v;
        }

        [[nodiscard]] int time_dof_id(int patch_dof_id) const
        {
            check_patch_dof_index_(patch_dof_id);
            return patch_dof_id % n_time_dofs_v;
        }

        [[nodiscard]] int local_dof_index(
            int spatial_local_dof_id,
            int time_dof_id) const
        {
            check_spatial_local_dof_index_(spatial_local_dof_id);
            check_time_dof_index_(time_dof_id);
            return spatial_local_dof_id * n_time_dofs_v + time_dof_id;
        }

        [[nodiscard]] int spatial_local_dof_id(int local_dof_id) const
        {
            check_local_dof_index_(local_dof_id);
            return local_dof_id / n_time_dofs_v;
        }

        [[nodiscard]] int local_time_dof_id(int local_dof_id) const
        {
            check_local_dof_index_(local_dof_id);
            return local_dof_id % n_time_dofs_v;
        }

        [[nodiscard]] int edge_patch_dof(
            int edge_id,
            int edge_moment_id,
            int time_dof_id) const
        {
            check_time_dof_index_(time_dof_id);
            const int spatial_patch_dof =
                spatial_space_.edge_patch_dof(edge_id, edge_moment_id);
            return spatial_patch_dof < 0
                ? -1
                : patch_dof_index(spatial_patch_dof, time_dof_id);
        }

        [[nodiscard]] const std::vector<LocalDofMap>&
        cell_dof_maps() const noexcept
        {
            return cell_dof_maps_;
        }

        [[nodiscard]] const LocalDofMap& cell_dof_map(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
        }

        [[nodiscard]] const LocalDofMapEntry& local_dof_entry(
            int patch_cell_index,
            int local_dof_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_local_dof_index_(local_dof_id);
            return cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)]
                                 [static_cast<std::size_t>(local_dof_id)];
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(patch_cell_index, local_dof_id).patch_dof_id;
        }

        [[nodiscard]] int local_to_patch_dof(
            int patch_cell_index,
            int spatial_local_dof_id,
            int time_dof_id) const
        {
            return local_to_patch_dof(
                patch_cell_index,
                local_dof_index(spatial_local_dof_id, time_dof_id));
        }

        [[nodiscard]] int local_orientation_sign(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(
                patch_cell_index,
                local_dof_id).orientation_sign;
        }

        [[nodiscard]] bool local_dof_is_constrained(
            int patch_cell_index,
            int local_dof_id) const
        {
            return local_dof_entry(
                patch_cell_index,
                local_dof_id).is_constrained;
        }

        [[nodiscard]] int local_edge_parameter_orientation_sign(
            int patch_cell_index,
            int local_face_id,
            int edge_moment_id) const
        {
            return spatial_space_.local_edge_parameter_orientation_sign(
                patch_cell_index,
                local_face_id,
                edge_moment_id);
        }

        [[nodiscard]] typename SpatialSpace::PiolaBasis::AffineMap
        physical_map_for_patch_cell(int patch_cell_index) const
        {
            return spatial_space_.physical_map_for_patch_cell(patch_cell_index);
        }

        [[nodiscard]] SpatialReferencePoint
        physical_to_reference_on_patch_cell(
            int patch_cell_index,
            const typename mesh::MeshTypes<GT>::SpatialPoint& p) const
        {
            return spatial_space_.physical_to_reference_on_patch_cell(
                patch_cell_index,
                p);
        }

        void evaluate_physical_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            LocalValues& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.fill(VectorValue{0.0, 0.0});
            typename SpatialSpace::LocalValues spatial_values{};
            spatial_space_.evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                spatial_values);
            TimeValues time_values{};
            evaluate_time_basis(t_ref, time_values);
            for (int spatial_local_dof = 0;
                 spatial_local_dof < spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                for (int time_dof = 0;
                     time_dof < n_time_dofs_v;
                     ++time_dof)
                {
                    const int local_id =
                        local_dof_index(spatial_local_dof, time_dof);
                    const auto& spatial_value =
                        spatial_values[static_cast<std::size_t>(
                            spatial_local_dof)];
                    const double time_value =
                        time_values[static_cast<std::size_t>(time_dof)];
                    values[static_cast<std::size_t>(local_id)] =
                        VectorValue{
                            spatial_value[0] * time_value,
                            spatial_value[1] * time_value};
                }
            }
        }

        void evaluate_physical_local_divergences_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            LocalDivergences& divergences) const
        {
            check_patch_cell_index_(patch_cell_index);
            divergences.fill(0.0);
            typename SpatialSpace::LocalDivergences spatial_divergences{};
            spatial_space_.evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                x_ref,
                spatial_divergences);
            TimeValues time_values{};
            evaluate_time_basis(t_ref, time_values);
            for (int spatial_local_dof = 0;
                 spatial_local_dof < spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                for (int time_dof = 0;
                     time_dof < n_time_dofs_v;
                     ++time_dof)
                {
                    divergences[static_cast<std::size_t>(
                        local_dof_index(spatial_local_dof, time_dof))] =
                        spatial_divergences[static_cast<std::size_t>(
                            spatial_local_dof)] *
                        time_values[static_cast<std::size_t>(time_dof)];
                }
            }
        }

        void evaluate_physical_local_basis_on_patch_cell(
            int patch_cell_index,
            const SpaceTimeReferencePoint& x_ref,
            LocalValues& values) const
        {
            evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                SpatialReferencePoint{x_ref[0], x_ref[1]},
                x_ref[2],
                values);
        }

        void evaluate_physical_local_divergences_on_patch_cell(
            int patch_cell_index,
            const SpaceTimeReferencePoint& x_ref,
            LocalDivergences& divergences) const
        {
            evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                SpatialReferencePoint{x_ref[0], x_ref[1]},
                x_ref[2],
                divergences);
        }

        void evaluate_physical_patch_basis_on_patch_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref,
            std::vector<VectorValue>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(
                static_cast<std::size_t>(n_dofs()),
                VectorValue{0.0, 0.0});

            LocalValues local_values{};
            evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);

            const auto& map = cell_dof_map(patch_cell_index);
            for (int local_dof_id = 0;
                 local_dof_id < local_dofs_v;
                 ++local_dof_id)
            {
                const auto& entry =
                    map[static_cast<std::size_t>(local_dof_id)];
                if (entry.patch_dof_id < 0)
                    continue;

                auto& value =
                    values[static_cast<std::size_t>(entry.patch_dof_id)];
                const auto& local_value =
                    local_values[static_cast<std::size_t>(local_dof_id)];
                value[0] += static_cast<double>(entry.orientation_sign) *
                            local_value[0];
                value[1] += static_cast<double>(entry.orientation_sign) *
                            local_value[1];
            }
        }

    private:
        SpatialSpace spatial_space_;
        std::vector<LocalDofMap> cell_dof_maps_{};

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2DView: patch cell index out of "
                    "range.");
            }
        }

        static void check_time_dof_index_(int time_dof_id)
        {
            if (time_dof_id < 0 || time_dof_id >= n_time_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2DView: time DoF index out of range.");
            }
        }

        static void check_spatial_local_dof_index_(int spatial_local_dof_id)
        {
            if (spatial_local_dof_id < 0 ||
                spatial_local_dof_id >= spatial_local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2DView: spatial local DoF index out "
                    "of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2DView: local DoF index out of range.");
            }
        }

        void check_spatial_patch_dof_index_(int spatial_patch_dof_id) const
        {
            if (spatial_patch_dof_id < 0 ||
                spatial_patch_dof_id >= spatial_space_.n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2DView: spatial patch DoF index out "
                    "of range.");
            }
        }

        void check_patch_dof_index_(int patch_dof_id) const
        {
            if (patch_dof_id < 0 || patch_dof_id >= n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpaceTime2DView: patch DoF index out of "
                    "range.");
            }
        }

        void build_cell_dof_maps_()
        {
            cell_dof_maps_.resize(static_cast<std::size_t>(n_patch_cells()));
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                auto& map =
                    cell_dof_maps_[static_cast<std::size_t>(patch_cell_index)];
                const auto& spatial_map =
                    spatial_space_.cell_dof_map(patch_cell_index);
                for (int spatial_local_dof = 0;
                     spatial_local_dof < spatial_local_dofs_v;
                     ++spatial_local_dof)
                {
                    const auto& spatial_entry =
                        spatial_map[static_cast<std::size_t>(
                            spatial_local_dof)];
                    for (int time_dof = 0;
                         time_dof < n_time_dofs_v;
                         ++time_dof)
                    {
                        const int local_id =
                            local_dof_index(spatial_local_dof, time_dof);
                        auto& entry =
                            map[static_cast<std::size_t>(local_id)];
                        entry.spatial_patch_dof_id =
                            spatial_entry.patch_dof_id;
                        entry.patch_dof_id =
                            spatial_entry.patch_dof_id < 0
                                ? -1
                                : patch_dof_index(
                                    spatial_entry.patch_dof_id,
                                    time_dof);
                        entry.orientation_sign =
                            spatial_entry.orientation_sign;
                        entry.spatial_local_dof_id = spatial_local_dof;
                        entry.time_dof_id = time_dof;
                        entry.edge_id = spatial_entry.edge_id;
                        entry.local_face_id = spatial_entry.local_face_id;
                        entry.is_edge_dof = spatial_entry.is_edge_dof;
                        entry.is_constrained = spatial_entry.is_constrained;
                    }
                }
            }
        }
    };

    struct PatchFunctionDofKey2D
    {
        int slab_id = -1;
        int slab_local_ordinal = -1;
        int patch_id = -1;
        int local_patch_dof = -1;
    };

    namespace detail
    {
        template<class PatchCell>
        [[nodiscard]] int patch_cell_slab_local_ordinal(
            const PatchCell& patch_cell)
        {
            if constexpr (requires { patch_cell.slab_local_ordinal; })
                return patch_cell.slab_local_ordinal;
            else
                return patch_cell.slab_cell_id;
        }
    }

    template<class ScalarSpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class PatchScalarFunctionTime2DView
    {
    public:
        using SpaceType = ScalarSpaceType;
        using Vector = VectorType;
        using PatchType = typename SpaceType::Patch;
        using SpatialReferencePoint =
            typename SpaceType::SpatialReferencePoint;
        using SpaceTimePoint = typename PatchType::Types::SpaceTimePoint;

        explicit PatchScalarFunctionTime2DView(const SpaceType& space)
            : space_(&space),
              coefficients_(space.n_dofs())
        {
            set_zero();
        }

        [[nodiscard]] const SpaceType& space() const noexcept
        {
            return *space_;
        }

        [[nodiscard]] const Vector& coefficients() const noexcept
        {
            return coefficients_;
        }

        void set_zero()
        {
            coefficients_.resize(space_->n_dofs());
            coefficients_.set_zero();
        }

        void update_coefficients(const Vector& coefficients)
        {
            if (coefficients.size() != space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarFunctionTime2DView::update_coefficients: "
                    "size mismatch.");
            }
            coefficients_.resize(coefficients.size());
            for (int i = 0; i < coefficients.size(); ++i)
                coefficients_[i] = coefficients[i];
        }

        [[nodiscard]] PatchFunctionDofKey2D local_dof_key(
            int patch_cell_index,
            int local_dof_id) const
        {
            const auto& patch_cell = space_->patch().cell(patch_cell_index);
            return PatchFunctionDofKey2D{
                space_->patch().slab_id,
                detail::patch_cell_slab_local_ordinal(patch_cell),
                space_->patch().patch_id,
                space_->local_to_patch_dof(patch_cell_index, local_dof_id)};
        }

        [[nodiscard]] double local_coefficient(
            int patch_cell_index,
            int local_dof_id) const
        {
            const int patch_dof_id =
                space_->local_to_patch_dof(patch_cell_index, local_dof_id);
            if (patch_dof_id < 0 || patch_dof_id >= space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchScalarFunctionTime2DView::local_coefficient: patch "
                    "DoF id out of range.");
            }
            return coefficients_[patch_dof_id];
        }

        [[nodiscard]] double value_on_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref) const
        {
            typename SpaceType::LocalValues local_values{};
            space_->evaluate_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);
            double value = 0.0;
            for (int local_dof_id = 0;
                 local_dof_id < SpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                value += local_coefficient(patch_cell_index, local_dof_id) *
                         local_values[static_cast<std::size_t>(local_dof_id)];
            }
            return value;
        }

        [[nodiscard]] double value_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            const auto x_ref =
                space_->physical_to_reference_on_patch_cell(
                    patch_cell_index,
                    typename mesh::MeshTypes<
                        typename SpaceType::GT>::SpatialPoint{p[0], p[1]});
            return value_on_cell(
                patch_cell_index,
                x_ref,
                space_->map_time_to_reference(p[2]));
        }

    private:
        const SpaceType* space_ = nullptr;
        Vector coefficients_{};
    };

    template<class RTFluxSpaceType, class VectorType>
    requires la::concepts::VectorLike<VectorType>
    class PatchRTFluxFunctionTime2DView
    {
    public:
        struct CellEvaluation
        {
            typename RTFluxSpaceType::VectorValue value{0.0, 0.0};
            double divergence = 0.0;
        };

        using SpaceType = RTFluxSpaceType;
        using Vector = VectorType;
        using PatchType = typename SpaceType::Patch;
        using SpatialReferencePoint =
            typename SpaceType::SpatialReferencePoint;
        using SpaceTimePoint = typename PatchType::Types::SpaceTimePoint;
        using VectorValue = typename SpaceType::VectorValue;

        explicit PatchRTFluxFunctionTime2DView(const SpaceType& space)
            : space_(&space),
              coefficients_(space.n_dofs())
        {
            set_zero();
        }

        [[nodiscard]] const SpaceType& space() const noexcept
        {
            return *space_;
        }

        [[nodiscard]] const Vector& coefficients() const noexcept
        {
            return coefficients_;
        }

        void set_zero()
        {
            coefficients_.resize(space_->n_dofs());
            coefficients_.set_zero();
        }

        void update_coefficients(const Vector& coefficients)
        {
            if (coefficients.size() != space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxFunctionTime2DView::update_coefficients: "
                    "size mismatch.");
            }
            coefficients_.resize(coefficients.size());
            for (int i = 0; i < coefficients.size(); ++i)
                coefficients_[i] = coefficients[i];
        }

        [[nodiscard]] PatchFunctionDofKey2D local_dof_key(
            int patch_cell_index,
            int local_dof_id) const
        {
            const auto& patch_cell = space_->patch().cell(patch_cell_index);
            return PatchFunctionDofKey2D{
                space_->patch().slab_id,
                detail::patch_cell_slab_local_ordinal(patch_cell),
                space_->patch().patch_id,
                space_->local_to_patch_dof(patch_cell_index, local_dof_id)};
        }

        [[nodiscard]] double local_coefficient(
            int patch_cell_index,
            int local_dof_id) const
        {
            const auto& entry =
                space_->local_dof_entry(patch_cell_index, local_dof_id);
            if (entry.patch_dof_id < 0)
                return 0.0;
            if (entry.patch_dof_id >= space_->n_dofs())
            {
                throw std::runtime_error(
                    "PatchRTFluxFunctionTime2DView::local_coefficient: "
                    "patch DoF id out of range.");
            }
            return static_cast<double>(entry.orientation_sign) *
                   coefficients_[entry.patch_dof_id];
        }

        [[nodiscard]] CellEvaluation evaluate_on_cell(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref,
            double t_ref) const
        {
            typename SpaceType::LocalValues local_values{};
            typename SpaceType::LocalDivergences local_divergences{};
            space_->evaluate_physical_local_basis_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_values);
            space_->evaluate_physical_local_divergences_on_patch_cell(
                patch_cell_index,
                x_ref,
                t_ref,
                local_divergences);

            CellEvaluation evaluation;
            for (int local_dof_id = 0;
                 local_dof_id < SpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const double c =
                    local_coefficient(patch_cell_index, local_dof_id);
                const auto& phi =
                    local_values[static_cast<std::size_t>(local_dof_id)];
                evaluation.value[0] += c * phi[0];
                evaluation.value[1] += c * phi[1];
                evaluation.divergence +=
                    c * local_divergences[
                        static_cast<std::size_t>(local_dof_id)];
            }
            return evaluation;
        }

        [[nodiscard]] CellEvaluation evaluate_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            const auto x_ref =
                space_->physical_to_reference_on_patch_cell(
                    patch_cell_index,
                    typename mesh::MeshTypes<
                        typename SpaceType::GT>::SpatialPoint{p[0], p[1]});
            return evaluate_on_cell(
                patch_cell_index,
                x_ref,
                space_->map_time_to_reference(p[2]));
        }

        [[nodiscard]] VectorValue value_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            return evaluate_on_cell(patch_cell_index, p).value;
        }

        [[nodiscard]] double divergence_on_cell(
            int patch_cell_index,
            const SpaceTimePoint& p) const
        {
            return evaluate_on_cell(patch_cell_index, p).divergence;
        }

    private:
        const SpaceType* space_ = nullptr;
        Vector coefficients_{};
    };
}

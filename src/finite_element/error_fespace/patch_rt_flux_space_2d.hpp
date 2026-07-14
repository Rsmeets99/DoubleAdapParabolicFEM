#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../basis/functions/raviart_thomas_triangle.hpp"
#include "patch_edge_topology_2d.hpp"

namespace finite_element::error_fespace
{
    template<class PatchType, int PSpace>
    requires time_slabs::is_time_slab_vertex_patch_v<PatchType>
    class PatchRTFluxSpace2D
    {
    public:
        using Patch         = PatchType;
        using GT            = typename PatchType::GT;
        using FETraitsType  = typename PatchType::FETraitsType;
        using Types         = mesh::MeshTypes<GT>;
        using SlabSpaceType = time_slabs::TimeSlabSpace<GT, FETraitsType>;
        using MeshType      = mesh::Mesh<GT>;

        using TopologyBuilder =
            PatchEdgeTopologyBuilder2D<GT, FETraitsType>;
        using TopologyType = typename TopologyBuilder::TopologyType;
        using ReferenceBasis =
            basis::functions::RaviartThomasTriangleBasis<PSpace>;
        using PiolaBasis =
            basis::functions::RaviartThomasTrianglePiolaBasis<PSpace>;
        using AffineMap = typename PiolaBasis::AffineMap;

        using ReferencePoint = typename ReferenceBasis::Point;
        using VectorValue    = typename ReferenceBasis::VectorValue;
        using LocalValues    = typename ReferenceBasis::Values;
        using LocalDivergences = typename ReferenceBasis::Divergences;

        static_assert(GT::dim_space_v == 2,
                      "PatchRTFluxSpace2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "PatchRTFluxSpace2D requires dim_time_v == 1.");
        static_assert(PSpace >= 1 && PSpace <= 10,
                      "PatchRTFluxSpace2D currently supports 1 <= PSpace <= 10.");

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

        PatchRTFluxSpace2D(
            const PatchType& patch,
            const SlabSpaceType& slab_space)
            : patch_(&patch),
              slab_space_(&slab_space),
              topology_(TopologyBuilder::build(patch, slab_space))
        {
            validate_patch_slab_();
            build_physical_map_cache_();
            build_edge_dof_blocks_();
            build_cell_dof_maps_();
        }

        PatchRTFluxSpace2D(
            const PatchType& patch,
            const SlabSpaceType& slab_space,
            TopologyType topology)
            : patch_(&patch),
              slab_space_(&slab_space),
              topology_(std::move(topology))
        {
            validate_patch_slab_();
            build_physical_map_cache_();
            build_edge_dof_blocks_();
            build_cell_dof_maps_();
        }

        [[nodiscard]] const PatchType& patch() const noexcept
        {
            return *patch_;
        }

        [[nodiscard]] const SlabSpaceType& slab_space() const noexcept
        {
            return *slab_space_;
        }

        [[nodiscard]] const TopologyType& topology() const noexcept
        {
            return topology_;
        }

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return patch_->n_cells;
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

        [[nodiscard]] const std::vector<EdgeDofBlock>& edge_dof_blocks() const noexcept
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

        [[nodiscard]] const std::vector<LocalDofMap>& cell_dof_maps() const noexcept
        {
            return cell_dof_maps_;
        }

        [[nodiscard]] const LocalDofMap& cell_dof_map(int patch_cell_index) const
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
                    "PatchRTFluxSpace2D: local face id out of range.");
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

        [[nodiscard]] typename PiolaBasis::AffineMap physical_map_for_patch_cell(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return physical_map_ref_for_patch_cell_(patch_cell_index);
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
                physical_map_ref_for_patch_cell_(patch_cell_index),
                x_ref);
        }

        void evaluate_physical_local_divergences_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            LocalDivergences& divergences) const
        {
            divergences = PiolaBasis::div_all(
                physical_map_ref_for_patch_cell_(patch_cell_index),
                x_ref);
        }

        void evaluate_reference_patch_basis_on_patch_cell(
            int patch_cell_index,
            const ReferencePoint& x_ref,
            std::vector<VectorValue>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            values.assign(static_cast<std::size_t>(n_dofs()), VectorValue{0.0, 0.0});

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
            values.assign(static_cast<std::size_t>(n_dofs()), VectorValue{0.0, 0.0});

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
        const PatchType* patch_ = nullptr;
        const SlabSpaceType* slab_space_ = nullptr;
        TopologyType topology_{};

        std::vector<EdgeDofBlock> edge_dof_blocks_{};
        std::vector<int> free_edge_ids_{};
        std::vector<int> constrained_edge_ids_{};
        std::vector<LocalDofMap> cell_dof_maps_{};
        std::vector<AffineMap> physical_map_cache_{};

        void validate_patch_slab_() const
        {
            if (patch_->slab_id < 0 || patch_->slab_id >= slab_space_->n_slabs())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2D: patch slab id is out of range for slab space.");
            }
        }

        [[nodiscard]] const MeshType& slab_mesh_() const
        {
            return slab_space_->slab(patch_->slab_id).mesh_ref();
        }

        void build_physical_map_cache_()
        {
            physical_map_cache_.resize(static_cast<std::size_t>(n_patch_cells()));

            const auto& mesh = slab_mesh_();
            for (int patch_cell_index = 0;
                 patch_cell_index < n_patch_cells();
                 ++patch_cell_index)
            {
                const auto& patch_cell = patch_->cell(patch_cell_index);
                const auto& cell = mesh.cell(patch_cell.slab_cell_id);

                const auto& v0 =
                    mesh.spatial_vertices()[static_cast<std::size_t>(
                        cell.spatial_vertex_ids[0])];
                const auto& v1 =
                    mesh.spatial_vertices()[static_cast<std::size_t>(
                        cell.spatial_vertex_ids[1])];
                const auto& v2 =
                    mesh.spatial_vertices()[static_cast<std::size_t>(
                        cell.spatial_vertex_ids[2])];

                physical_map_cache_[static_cast<std::size_t>(patch_cell_index)] =
                    PiolaBasis::make_affine_map(v0, v1, v2);
            }
        }

        [[nodiscard]] const AffineMap& physical_map_ref_for_patch_cell_(
            int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            return physical_map_cache_[
                static_cast<std::size_t>(patch_cell_index)];
        }

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 || patch_cell_index >= n_patch_cells())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2D: patch cell index out of range.");
            }
        }

        static void check_local_dof_index_(int local_dof_id)
        {
            if (local_dof_id < 0 || local_dof_id >= local_dofs_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2D: local RT DoF index out of range.");
            }
        }

        void check_edge_index_(int edge_id) const
        {
            if (edge_id < 0 || edge_id >= topology_.n_edges())
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2D: patch edge id out of range.");
            }
        }

        static void check_edge_moment_index_(int edge_moment_id)
        {
            if (edge_moment_id < 0 ||
                edge_moment_id >= edge_moments_per_edge_v)
            {
                throw std::runtime_error(
                    "PatchRTFluxSpace2D: edge moment index out of range.");
            }
        }

        [[nodiscard]] bool edge_should_be_free_(
            const typename TopologyType::Edge& edge) const noexcept
        {
            if (edge.is_internal())
                return true;

            return patch_->is_boundary() && edge.on_physical_boundary;
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
            const auto* edge = topology_.edge_for_patch_cell_face(
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
                            "PatchRTFluxSpace2D: failed to find patch edge incident for local face.");
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
            const auto& patch_cell = patch_->cell(patch_cell_index);
            const auto& slab =
                slab_space_->slab(patch_->slab_id);
            const auto& slab_cell =
                slab.mesh_ref().cell(patch_cell.slab_cell_id);
            const auto local_vertices =
                slab_cell.spatial_faces[static_cast<std::size_t>(local_face_id)]
                    .spatial_vertex_ids;
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
                "PatchRTFluxSpace2D: local edge vertices do not match canonical edge vertices.");
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
}

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <iterator>
#include <string>
#include <string_view>
#include <vector>

#include "../core/exceptions.hpp"
#include "../core/ids.hpp"
#include "cell.hpp"
#include "detail/vertex_registry.hpp"
#include "initialization/root_cell.hpp"
#include "mesh_traits.hpp"
#include "refinement/refine.hpp"
#include "refinement/refinement_type.hpp"
#include "refinement/split_policy.hpp"
#include "topology/boundary.hpp"
#include "topology/faces.hpp"
#include "topology/interval_relations.hpp"
#include "topology/leaf_traversal.hpp"
#include "topology/orientation.hpp"
#include "topology/refinement_edge.hpp"
#include "topology/spatial_edge_adjacency_2d.hpp"
#include "topology/temporal_keys.hpp"
#include "types.hpp"

namespace mesh
{
    template<typename GeomTraits>
    class Mesh
    {
    public:
        using Types  = MeshTypes<GeomTraits>;

        using CellType        = Cell<GeomTraits>;
        using SpatialPoint    = typename Types::SpatialPoint;
        using TemporalPoint   = typename Types::TemporalPoint;
        using SpaceTimePoint  = typename Types::SpaceTimePoint;
        using SpatialEdge     = typename Types::SpatialEdgeVertexIds;
        using SpatialFace     = typename Types::SpatialFaceVertexIds;
        using SpatialSimplexPoints = typename Types::SpatialSimplexPoints;
        using cell_id_type    = typename Types::cell_id_type;
        using vertex_id_type  = typename Types::vertex_id_type;
        using TimingMetricCallback = std::function<void(std::string_view, double)>;

        Mesh() = default;

        Mesh(const SpatialPoint& x0,
             const SpatialPoint& x1,
             const TemporalPoint& t0,
             const TemporalPoint& t1)
        {
            static_assert(GeomTraits::dim_space_v == 1, "Currently, Mesh is only implemented for dim_space=1.");
            initialize_mesh(x0, x1, t0, t1);
        }

        void clear()
        {
            cells_.clear();
            spatial_vertices_.clear();
            temporal_vertices_.clear();
            spatial_boundary_vertex_ids_.clear();
            spatial_boundary_face_vertex_ids_.clear();
            temporal_boundary_vertex_ids_.clear();
            registry_.clear();
            bump_storage_version_();
        }

        void create_root_cell(
            const SpatialPoint& x0,
            const SpatialPoint& x1,
            const TemporalPoint& t0,
            const TemporalPoint& t1)
        {
            auto root = initialization::make_root_cell<GeomTraits>(
                spatial_vertices_,
                temporal_vertices_,
                registry_,
                spatial_boundary_vertex_ids_,
                temporal_boundary_vertex_ids_,
                x0,
                x1,
                t0,
                t1,
                0);

            cells_.push_back(root);
            rebuild_spatial_boundary_face_vertex_ids_from_root_faces_();
            bump_storage_version_();
        }

        [[nodiscard]] cell_id_type create_root_cell(
            const SpatialSimplexPoints& spatial_points,
            const TemporalPoint& t0,
            const TemporalPoint& t1)
        {
            const auto root_id = append_root_cell_(spatial_points, t0, t1);
            finalize_root_boundaries_();
            return root_id;
        }

        void initialize_mesh(
            const SpatialPoint& x0,
            const SpatialPoint& x1,
            const TemporalPoint& t0,
            const TemporalPoint& t1)
        {
            clear();
            create_root_cell(x0, x1, t0, t1);
        }

        void initialize_mesh(
            const std::vector<SpatialSimplexPoints>& root_spatial_cells,
            const TemporalPoint& t0,
            const TemporalPoint& t1)
        {
            if constexpr (GeomTraits::dim_space_v != 2)
            {
                throw core::dimension_not_supported_error(
                    "Mesh::initialize_mesh(root_spatial_cells, t0, t1) is only implemented for dim_space=2.");
            }
            else
            {
                if (root_spatial_cells.empty())
                    throw core::invalid_mesh_error("Mesh::initialize_mesh: no root cells provided.");

                clear();
                for (const auto& spatial_points : root_spatial_cells)
                    (void)append_root_cell_(spatial_points, t0, t1);

                finalize_root_boundaries_();
            }
        }

        [[nodiscard]] vertex_id_type get_or_create_spatial_vertex(const SpatialPoint& x)
        {
            const auto before = spatial_vertices_.size();
            const auto id =
                registry_.get_or_create_spatial_vertex(spatial_vertices_, x);
            if (spatial_vertices_.size() != before)
                bump_storage_version_();
            return id;
        }

        [[nodiscard]] vertex_id_type get_or_create_temporal_vertex(const TemporalPoint& t)
        {
            const auto before = temporal_vertices_.size();
            const auto id =
                registry_.get_or_create_temporal_vertex(temporal_vertices_, t);
            if (temporal_vertices_.size() != before)
                bump_storage_version_();
            return id;
        }

        void inherit_spatial_boundary_metadata_from(const Mesh& source)
        {
            if constexpr (GeomTraits::dim_space_v == 1)
            {
                for (const auto source_vertex_id :
                     source.spatial_boundary_vertex_ids_)
                {
                    const auto target_vertex_id =
                        get_or_create_spatial_vertex(
                            source.spatial_vertices_[static_cast<std::size_t>(
                                source_vertex_id)]);
                    if (std::find(
                            spatial_boundary_vertex_ids_.begin(),
                            spatial_boundary_vertex_ids_.end(),
                            target_vertex_id) ==
                        spatial_boundary_vertex_ids_.end())
                    {
                        spatial_boundary_vertex_ids_.push_back(target_vertex_id);
                    }
                }
            }
            else if constexpr (GeomTraits::dim_space_v == 2)
            {
                for (const auto& source_face :
                     source.spatial_boundary_face_vertex_ids_)
                {
                    SpatialFace target_face{};
                    for (std::size_t i = 0; i < target_face.size(); ++i)
                    {
                        const auto target_vertex_id =
                            get_or_create_spatial_vertex(
                                source.spatial_vertices_[static_cast<std::size_t>(
                                    source_face[i])]);
                        target_face[i] = target_vertex_id;

                        if (std::find(
                                spatial_boundary_vertex_ids_.begin(),
                                spatial_boundary_vertex_ids_.end(),
                                target_vertex_id) ==
                            spatial_boundary_vertex_ids_.end())
                        {
                            spatial_boundary_vertex_ids_.push_back(
                                target_vertex_id);
                        }
                    }

                    target_face = sorted_spatial_face_(target_face);
                    if (std::find(
                            spatial_boundary_face_vertex_ids_.begin(),
                            spatial_boundary_face_vertex_ids_.end(),
                            target_face) ==
                        spatial_boundary_face_vertex_ids_.end())
                    {
                        spatial_boundary_face_vertex_ids_.push_back(
                            target_face);
                    }
                }
            }
        }

        void ensure_spatial_vertex_orientation(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            topology::ensure_spatial_vertex_orientation(c, spatial_vertices_);
        }

        void ensure_temporal_vertex_orientation(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            topology::ensure_temporal_vertex_orientation(c, temporal_vertices_);
        }

        void compute_spatial_faces(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            topology::fill_spatial_faces(c);
        }

        void compute_temporal_faces(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            topology::fill_temporal_faces(c);
        }

        void compute_faces(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            topology::fill_faces(c);
        }

        void compute_spatial_boundary(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            if constexpr (GeomTraits::dim_space_v == 1)
            {
                topology::fill_spatial_boundary(c, spatial_boundary_vertex_ids_);
            }
            else if constexpr (GeomTraits::dim_space_v == 2)
            {
                topology::fill_spatial_boundary(
                    c,
                    spatial_boundary_face_vertex_ids_,
                    spatial_vertices_);
            }
        }

        void compute_temporal_boundary(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            topology::fill_temporal_boundary(c, temporal_boundary_vertex_ids_);
        }

        void compute_boundary(cell_id_type cell_id)
        {
            auto& c = cell(cell_id);
            if constexpr (GeomTraits::dim_space_v == 1)
            {
                topology::fill_boundary(
                    c,
                    spatial_boundary_vertex_ids_,
                    temporal_boundary_vertex_ids_);
            }
            else if constexpr (GeomTraits::dim_space_v == 2)
            {
                topology::fill_boundary(
                    c,
                    spatial_boundary_face_vertex_ids_,
                    spatial_vertices_,
                    temporal_boundary_vertex_ids_);
            }
        }

        void refine(cell_id_type cell_id,
                    RefinementType refinement_type = RefinementType::none,
                    TimingMetricCallback timing_callback = {})
        {
            const auto n_cells_before = cells_.size();
            if constexpr (GeomTraits::dim_space_v == 1)
            {
                refinement::refine(
                    cells_,
                    spatial_vertices_,
                    temporal_vertices_,
                    registry_,
                    spatial_boundary_vertex_ids_,
                    temporal_boundary_vertex_ids_,
                    cell_id,
                    refinement_type);
            }
            else if constexpr (GeomTraits::dim_space_v == 2)
            {
                refine_2d_with_closure_({cell_id}, refinement_type, timing_callback);
            }
            if (cells_.size() != n_cells_before)
                bump_storage_version_();
        }

        void refine(const std::vector<cell_id_type>& cell_ids,
                    RefinementType refinement_type = RefinementType::none,
                    TimingMetricCallback timing_callback = {})
        {
            const auto n_cells_before = cells_.size();
            if constexpr (GeomTraits::dim_space_v == 1)
            {
                refinement::refine(
                    cells_,
                    spatial_vertices_,
                    temporal_vertices_,
                    registry_,
                    spatial_boundary_vertex_ids_,
                    temporal_boundary_vertex_ids_,
                    cell_ids,
                    refinement_type);
            }
            else if constexpr (GeomTraits::dim_space_v == 2)
            {
                refine_2d_with_closure_(cell_ids, refinement_type, timing_callback);
            }
            if (cells_.size() != n_cells_before)
                bump_storage_version_();
        }

        [[nodiscard]] std::vector<cell_id_type> create_children_if_needed(
            cell_id_type cell_id,
            RefinementType refinement_type = RefinementType::none)
        {
            if (!valid_cell_id(cell_id))
                throw std::runtime_error(
                    "Mesh::create_children_if_needed: invalid cell id.");

            const auto& current = cell(cell_id);
            if (!current.is_leaf)
            {
                if (current.children.empty())
                    throw std::runtime_error(
                        "Mesh::create_children_if_needed: non-leaf cell has no children.");
                return current.children;
            }

            const auto n_cells_before = cells_.size();
            if constexpr (GeomTraits::dim_space_v == 1)
            {
                refinement::refine(
                    cells_,
                    spatial_vertices_,
                    temporal_vertices_,
                    registry_,
                    spatial_boundary_vertex_ids_,
                    temporal_boundary_vertex_ids_,
                    cell_id,
                    refinement_type);
            }
            else if constexpr (GeomTraits::dim_space_v == 2)
            {
                refinement::refine(
                    cells_,
                    spatial_vertices_,
                    temporal_vertices_,
                    registry_,
                    spatial_boundary_face_vertex_ids_,
                    temporal_boundary_vertex_ids_,
                    cell_id,
                    refinement_type);
            }
            if (cells_.size() != n_cells_before)
                bump_storage_version_();

            const auto& refined = cell(cell_id);
            if (refined.children.empty())
                throw std::runtime_error(
                    "Mesh::create_children_if_needed: child creation produced no children.");
            return refined.children;
        }

        void refine_storage_leaf_without_closure(
            cell_id_type cell_id,
            RefinementType refinement_type = RefinementType::none)
        {
            if (!valid_cell_id(cell_id))
                throw std::runtime_error(
                    "Mesh::refine_storage_leaf_without_closure: invalid cell id.");
            if (!cell(cell_id).is_leaf)
                throw std::runtime_error(
                    "Mesh::refine_storage_leaf_without_closure: cell is not a storage leaf.");
            (void)create_children_if_needed(cell_id, refinement_type);
        }

        [[nodiscard]] RefinementType next_refinement_type(cell_id_type cell_id) const
        {
            const auto& c = cell(cell_id);
            return refinement::next_split_type<GeomTraits>(c.generation);
        }

        [[nodiscard]] SpatialEdge spatial_refinement_edge(cell_id_type cell_id) const
        {
            return topology::spatial_refinement_edge_vertex_ids(cell(cell_id));
        }

        [[nodiscard]] bool temporal_intervals_overlap(cell_id_type a, cell_id_type b) const
        {
            return topology::temporal_intervals_overlap_1d(
                cell(a), cell(b), temporal_vertices_);
        }

        [[nodiscard]] bool temporal_interval_contains(cell_id_type outer, cell_id_type inner) const
        {
            return topology::temporal_interval_contains_1d(
                cell(outer), cell(inner), temporal_vertices_);
        }

        [[nodiscard]] bool spatial_intervals_overlap(cell_id_type a, cell_id_type b) const
        {
            return topology::spatial_intervals_overlap_1d(
                cell(a), cell(b), spatial_vertices_);
        }

        [[nodiscard]] bool spatial_interval_contains(cell_id_type outer, cell_id_type inner) const
        {
            return topology::spatial_interval_contains_1d(
                cell(outer), cell(inner), spatial_vertices_);
        }

        [[nodiscard]] bool contains_spatial_refinement_edge(cell_id_type container,
                                                    cell_id_type source) const
        {
            const auto edge = spatial_refinement_edge(source);
            return topology::cell_contains_spatial_edge(cell(container), edge, spatial_vertices_);
        }

        [[nodiscard]] std::vector<cell_id_type> leaf_cell_ids() const
        {
            return topology::leaf_cell_ids(cells_);
        }

        [[nodiscard]] std::vector<cell_id_type> root_cell_ids() const
        {
            std::vector<cell_id_type> roots;
            roots.reserve(cells_.size());

            for (const auto& c : cells_)
            {
                if (!c.has_parent())
                    roots.push_back(c.cell_id);
            }

            return roots;
        }

        [[nodiscard]] std::uint64_t storage_version() const noexcept
        {
            return storage_version_;
        }

        [[nodiscard]] bool valid_cell_id(cell_id_type id) const noexcept
        {
            return id >= 0 && static_cast<std::size_t>(id) < cells_.size();
        }

        [[nodiscard]] bool cell_is_storage_leaf(cell_id_type id) const noexcept
        {
            return valid_cell_id(id) &&
                   cells_[static_cast<std::size_t>(id)].is_leaf;
        }

        [[nodiscard]] bool is_ancestor(
            cell_id_type ancestor,
            cell_id_type descendant) const noexcept
        {
            if (!valid_cell_id(ancestor) || !valid_cell_id(descendant) ||
                ancestor == descendant)
            {
                return false;
            }

            cell_id_type current =
                cells_[static_cast<std::size_t>(descendant)].parent_id;
            std::size_t guard = 0;
            while (current >= 0)
            {
                if (!valid_cell_id(current))
                    return false;
                if (current == ancestor)
                    return true;

                current = cells_[static_cast<std::size_t>(current)].parent_id;
                ++guard;
                if (guard > cells_.size())
                    return false;
            }

            return false;
        }

        [[nodiscard]] std::vector<cell_id_type> ancestor_chain(
            cell_id_type cell_id) const
        {
            if (!valid_cell_id(cell_id))
                throw core::invalid_mesh_error(
                    "Mesh::ancestor_chain: id out of range.");

            std::vector<cell_id_type> chain;
            cell_id_type current = cell_id;

            while (current >= 0)
            {
                if (!valid_cell_id(current))
                    throw core::invalid_mesh_error(
                        "Mesh::ancestor_chain: id out of range.");

                chain.push_back(current);
                if (chain.size() > cells_.size())
                    throw core::invalid_mesh_error(
                        "Mesh::ancestor_chain: parent cycle detected.");

                const auto parent_id =
                    cells_[static_cast<std::size_t>(current)].parent_id;
                if (parent_id < 0)
                    break;

                current = parent_id;
            }

            return chain;
        }

        [[nodiscard]] topology::TimeIntervalIdKey root_temporal_interval_key(
            cell_id_type root_cell_id) const
        {
            if (!valid_cell_id(root_cell_id))
                throw core::invalid_mesh_error(
                    "Mesh::root_temporal_interval_key: id out of range.");
            const auto& root = cell(root_cell_id);
            if (root.has_parent())
                throw core::invalid_mesh_error(
                    "Mesh::root_temporal_interval_key: cell is not a root.");
            return topology::make_time_interval_id_key(
                root.temporal_vertex_ids[0],
                root.temporal_vertex_ids[1]);
        }

        [[nodiscard]] int root_temporal_interval_id(
            cell_id_type root_cell_id) const
        {
            const auto target = root_temporal_interval_key(root_cell_id);
            std::vector<topology::TimeIntervalIdKey> root_keys;
            for (const auto id : root_cell_ids())
                root_keys.push_back(root_temporal_interval_key(id));
            std::sort(root_keys.begin(), root_keys.end());
            root_keys.erase(
                std::unique(root_keys.begin(), root_keys.end()),
                root_keys.end());

            const auto it =
                std::lower_bound(root_keys.begin(), root_keys.end(), target);
            if (it == root_keys.end() || !(*it == target))
                throw core::invalid_mesh_error(
                    "Mesh::root_temporal_interval_id: root interval not found.");
            return static_cast<int>(std::distance(root_keys.begin(), it));
        }

        [[nodiscard]] topology::DyadicTimeIntervalKey
        dyadic_temporal_interval_key(cell_id_type cell_id) const
        {
            const auto chain = ancestor_chain(cell_id);
            if (chain.empty())
                throw core::invalid_mesh_error(
                    "Mesh::dyadic_temporal_interval_key: empty ancestor chain.");

            const auto root_id = chain.back();
            auto key = topology::make_dyadic_time_interval_key(
                root_temporal_interval_id(root_id),
                0,
                0U);

            for (auto it = chain.rbegin(); it + 1 != chain.rend(); ++it)
            {
                const auto parent_id = *it;
                const auto child_id = *(it + 1);
                const auto& parent = cell(parent_id);
                const auto& child = cell(child_id);

                const bool has_temporal_split =
                    parent.last_split_type == RefinementType::temporal ||
                    parent.last_split_type == RefinementType::spacetime;
                if (!has_temporal_split)
                    continue;

                if (key.level >= 63)
                    throw core::invalid_mesh_error(
                        "Mesh::dyadic_temporal_interval_key: temporal level too deep.");

                const bool lower_child =
                    child.temporal_vertex_ids[0] ==
                    parent.temporal_vertex_ids[0];
                const bool upper_child =
                    child.temporal_vertex_ids[1] ==
                    parent.temporal_vertex_ids[1];

                if (lower_child == upper_child)
                    throw core::invalid_mesh_error(
                        "Mesh::dyadic_temporal_interval_key: child is not a temporal half of parent.");

                key.index = 2U * key.index + (upper_child ? 1U : 0U);
                ++key.level;
            }

            return key;
        }

        [[nodiscard]] std::array<topology::DyadicTimePointKey, 2>
        dyadic_temporal_endpoint_keys(cell_id_type cell_id) const
        {
            const auto interval = dyadic_temporal_interval_key(cell_id);
            return {
                topology::dyadic_time_interval_begin_key(interval),
                topology::dyadic_time_interval_end_key(interval)
            };
        }

        [[nodiscard]] bool children_partition_parent_debug(
            cell_id_type cell_id) const noexcept
        {
            if (!valid_cell_id(cell_id))
                return false;

            const auto& parent = cells_[static_cast<std::size_t>(cell_id)];
            if (parent.cell_id != cell_id)
                return false;

            if (parent.is_leaf)
                return parent.children.empty();

            if (parent.children.empty() ||
                parent.last_split_type == RefinementType::none)
            {
                return false;
            }

            const auto expected_child_count =
                [](RefinementType split_type) -> std::size_t
                {
                    switch (split_type)
                    {
                        case RefinementType::spatial:
                        case RefinementType::temporal:
                            return 2;
                        case RefinementType::spacetime:
                            return 4;
                        case RefinementType::none:
                            return 0;
                    }
                    return 0;
                };

            if (parent.children.size() !=
                expected_child_count(parent.last_split_type))
            {
                return false;
            }

            const auto expected_spatial_level =
                parent.spatial_level +
                ((parent.last_split_type == RefinementType::spatial ||
                  parent.last_split_type == RefinementType::spacetime)
                     ? 1
                     : 0);
            const auto expected_temporal_level =
                parent.temporal_level +
                ((parent.last_split_type == RefinementType::temporal ||
                  parent.last_split_type == RefinementType::spacetime)
                     ? 1
                     : 0);

            for (std::size_t i = 0; i < parent.children.size(); ++i)
            {
                const auto child_id = parent.children[i];
                if (!valid_cell_id(child_id))
                    return false;

                for (std::size_t j = 0; j < i; ++j)
                {
                    if (parent.children[j] == child_id)
                        return false;
                }

                const auto& child =
                    cells_[static_cast<std::size_t>(child_id)];
                if (child.cell_id != child_id ||
                    child.parent_id != parent.cell_id ||
                    child.generation != parent.generation + 1 ||
                    child.spatial_level != expected_spatial_level ||
                    child.temporal_level != expected_temporal_level)
                {
                    return false;
                }

                if (parent.last_split_type == RefinementType::spatial &&
                    child.temporal_vertex_ids != parent.temporal_vertex_ids)
                {
                    return false;
                }

                if (parent.last_split_type == RefinementType::temporal &&
                    child.spatial_vertex_ids != parent.spatial_vertex_ids)
                {
                    return false;
                }
            }

            return true;
        }

        [[nodiscard]] bool contains_coord(cell_id_type cell_id,
                                          const SpaceTimePoint& p) const
        {
            return mesh::contains_coord(
                cell(cell_id),
                spatial_vertices_,
                temporal_vertices_,
                p);
        }

        [[nodiscard]] const CellType& cell(cell_id_type id) const
        {
            if (!valid_cell_id(id))
                throw core::invalid_mesh_error("Mesh::cell const: id out of range.");
            return cells_[static_cast<std::size_t>(id)];
        }

        [[nodiscard]] CellType& cell(cell_id_type id)
        {
            if (!valid_cell_id(id))
                throw core::invalid_mesh_error("Mesh::cell: id out of range.");
            return cells_[static_cast<std::size_t>(id)];
        }

        [[nodiscard]] const std::vector<CellType>& cells() const noexcept
        {
            return cells_;
        }

        [[nodiscard]] std::vector<CellType>& unsafe_cells_ref() noexcept
        {
            return cells_;
        }

        [[nodiscard]] const std::vector<SpatialPoint>& spatial_vertices() const noexcept
        {
            return spatial_vertices_;
        }

        [[nodiscard]] const std::vector<TemporalPoint>& temporal_vertices() const noexcept
        {
            return temporal_vertices_;
        }

        [[nodiscard]] const std::vector<vertex_id_type>& spatial_boundary_vertex_ids() const noexcept
        {
            return spatial_boundary_vertex_ids_;
        }

        [[nodiscard]] const std::vector<SpatialFace>& spatial_boundary_face_vertex_ids() const noexcept
        {
            return spatial_boundary_face_vertex_ids_;
        }

        [[nodiscard]] const std::vector<vertex_id_type>& temporal_boundary_vertex_ids() const noexcept
        {
            return temporal_boundary_vertex_ids_;
        }

        [[nodiscard]] std::size_t n_cells() const noexcept
        {
            return cells_.size();
        }

        [[nodiscard]] std::size_t n_spatial_vertices() const noexcept
        {
            return spatial_vertices_.size();
        }

        [[nodiscard]] std::size_t n_temporal_vertices() const noexcept
        {
            return temporal_vertices_.size();
        }

    private:
        [[nodiscard]] cell_id_type append_root_cell_(
            const SpatialSimplexPoints& spatial_points,
            const TemporalPoint& t0,
            const TemporalPoint& t1)
        {
            const auto root_id = static_cast<cell_id_type>(cells_.size());
            auto root = initialization::make_root_cell<GeomTraits>(
                spatial_vertices_,
                temporal_vertices_,
                registry_,
                spatial_points,
                t0,
                t1,
                root_id);

            cells_.push_back(root);
            bump_storage_version_();
            return root_id;
        }

        struct RefinementRequest2D
        {
            cell_id_type cell_id = -1;
            RefinementType refinement_type = RefinementType::none;
        };

        [[nodiscard]] RefinementType effective_refinement_type_2d_(
            cell_id_type cell_id,
            RefinementType requested_refinement_type) const
        {
            if (requested_refinement_type == RefinementType::none)
                return refinement::next_split_type<GeomTraits>(cell(cell_id).generation);
            return requested_refinement_type;
        }

        [[nodiscard]] static bool refinement_has_spatial_part_2d_(
            RefinementType refinement_type) noexcept
        {
            return refinement_type == RefinementType::spatial ||
                   refinement_type == RefinementType::spacetime;
        }

        static void record_timing_metric_(
            const TimingMetricCallback& callback,
            std::string_view phase,
            double value)
        {
            if (callback)
                callback(phase, value);
        }

        template<class SlicewiseAdjacency>
        static void record_adjacency_metrics_2d_(
            const TimingMetricCallback& callback,
            std::string_view phase_prefix,
            std::size_t active_cells_scanned,
            const SlicewiseAdjacency& adjacency)
        {
            if (!callback)
                return;

            const auto stats = topology::adjacency_stats_2d(adjacency);
            std::string phase(phase_prefix);

            record_timing_metric_(
                callback,
                phase + ".active_cells_scanned.count",
                static_cast<double>(active_cells_scanned));
            record_timing_metric_(
                callback,
                phase + ".time_slices_built.count",
                static_cast<double>(stats.time_slices_built));
            record_timing_metric_(
                callback,
                phase + ".edge_records_built.count",
                static_cast<double>(stats.edge_records_built));
        }

        static void record_local_closure_metrics_2d_(
            const TimingMetricCallback& callback,
            std::string_view phase_prefix,
            const topology::LocalSpatialClosureStats2D& stats)
        {
            if (!callback)
                return;

            std::string phase(phase_prefix);
            record_timing_metric_(
                callback,
                phase + ".active_cells_scanned.count",
                static_cast<double>(stats.active_cells_scanned));
            record_timing_metric_(
                callback,
                phase + ".time_slices_built.count",
                0.0);
            record_timing_metric_(
                callback,
                phase + ".edge_records_built.count",
                static_cast<double>(stats.edge_records_built));
            record_timing_metric_(
                callback,
                phase + ".seed_cells_scanned.count",
                static_cast<double>(stats.seed_cells_scanned));
            record_timing_metric_(
                callback,
                phase + ".seed_edge_records.count",
                static_cast<double>(stats.seed_edge_records));
            record_timing_metric_(
                callback,
                phase + ".candidate_edge_visits.count",
                static_cast<double>(stats.candidate_edge_visits));
            record_timing_metric_(
                callback,
                phase + ".edge_comparisons.count",
                static_cast<double>(stats.edge_comparisons));
            record_timing_metric_(
                callback,
                phase + ".time_overlap_tests.count",
                static_cast<double>(stats.time_overlap_tests));
            record_timing_metric_(
                callback,
                phase + ".same_spatial_overlap_scans.count",
                static_cast<double>(stats.same_spatial_overlap_scans));
        }

        static void record_local_verification_metrics_2d_(
            const TimingMetricCallback& callback,
            std::string_view phase_prefix,
            const topology::LocalSpatialConformityVerificationStats2D& stats)
        {
            if (!callback)
                return;

            std::string phase(phase_prefix);
            record_timing_metric_(
                callback,
                phase + ".active_cells_scanned.count",
                static_cast<double>(stats.active_cells_scanned));
            record_timing_metric_(
                callback,
                phase + ".time_slices_built.count",
                static_cast<double>(stats.time_slices_built));
            record_timing_metric_(
                callback,
                phase + ".edge_records_built.count",
                static_cast<double>(stats.edge_records_built));
            record_timing_metric_(
                callback,
                phase + ".local_edge_records_built.count",
                static_cast<double>(stats.local_edge_records_built));
            record_timing_metric_(
                callback,
                phase + ".fallback_to_full_check.count",
                static_cast<double>(stats.fallback_to_full_check));
            record_timing_metric_(
                callback,
                phase + ".seed_cells.count",
                static_cast<double>(stats.seed_cells));
            record_timing_metric_(
                callback,
                phase + ".seed_cells_scanned.count",
                static_cast<double>(stats.seed_cells_scanned));
            record_timing_metric_(
                callback,
                phase + ".active_edge_records_built.count",
                static_cast<double>(stats.active_edge_records_built));
            record_timing_metric_(
                callback,
                phase + ".seed_edge_records.count",
                static_cast<double>(stats.seed_edge_records));
            record_timing_metric_(
                callback,
                phase + ".candidate_edge_visits.count",
                static_cast<double>(stats.candidate_edge_visits));
            record_timing_metric_(
                callback,
                phase + ".candidate_cells.count",
                static_cast<double>(stats.candidate_cells));
            record_timing_metric_(
                callback,
                phase + ".candidate_cells_ratio",
                stats.candidate_cells_ratio);
            record_timing_metric_(
                callback,
                phase + ".seed_edge_records_checked.count",
                static_cast<double>(stats.seed_edge_records_checked));
            record_timing_metric_(
                callback,
                phase + ".singular_seed_edges.count",
                static_cast<double>(stats.singular_seed_edges));
            record_timing_metric_(
                callback,
                phase + ".nonconforming_seed_edges.count",
                static_cast<double>(stats.nonconforming_seed_edges));
            record_timing_metric_(
                callback,
                phase + ".active_edge_record_construction",
                stats.active_edge_record_construction_seconds);
            record_timing_metric_(
                callback,
                phase + ".active_cell_vertex_index_construction",
                stats.active_cell_vertex_index_construction_seconds);
            record_timing_metric_(
                callback,
                phase + ".seed_candidate_discovery",
                stats.seed_candidate_discovery_seconds);
            record_timing_metric_(
                callback,
                phase + ".local_slicewise_adjacency_rebuild",
                stats.local_slicewise_adjacency_rebuild_seconds);
            record_timing_metric_(
                callback,
                phase + ".seed_edge_conformity_check",
                stats.seed_edge_conformity_check_seconds);
            record_timing_metric_(
                callback,
                phase + ".local_check_time",
                stats.local_check_seconds);
            record_timing_metric_(
                callback,
                phase + ".full_check_time",
                stats.full_check_seconds);
        }

        [[nodiscard]] static RefinementType combine_refinement_types_2d_(
            RefinementType a,
            RefinementType b) noexcept
        {
            if (a == b)
                return a;
            if (a == RefinementType::none)
                return b;
            if (b == RefinementType::none)
                return a;
            if (a == RefinementType::spacetime || b == RefinementType::spacetime)
                return RefinementType::spacetime;
            if ((a == RefinementType::spatial && b == RefinementType::temporal) ||
                (a == RefinementType::temporal && b == RefinementType::spatial))
            {
                return RefinementType::spacetime;
            }
            return a;
        }

        void validate_refinement_request_2d_(
            cell_id_type cell_id,
            RefinementType refinement_type) const
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= cells_.size())
                throw core::invalid_mesh_error(
                    "Mesh::refine_2d_with_closure: cell_id out of range.");

            const auto& c = cells_[static_cast<std::size_t>(cell_id)];
            if (!c.is_leaf)
                throw core::invalid_mesh_error(
                    "Mesh::refine_2d_with_closure: can only refine active leaf cells.");

            if (refinement_type != RefinementType::spatial &&
                refinement_type != RefinementType::temporal &&
                refinement_type != RefinementType::spacetime)
            {
                throw core::invalid_mesh_error(
                    "Mesh::refine_2d_with_closure: unsupported 2D refinement type.");
            }
        }

        void enqueue_refinement_request_2d_(
            cell_id_type cell_id,
            RefinementType refinement_type,
            std::vector<RefinementRequest2D>& pending,
            std::vector<bool>& scheduled) const
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= cells_.size())
                throw core::invalid_mesh_error(
                    "Mesh::enqueue_refinement_request_2d: cell_id out of range.");

            const auto idx = static_cast<std::size_t>(cell_id);
            if (!cells_[idx].is_leaf)
                return;

            if (scheduled[idx])
            {
                for (auto& request : pending)
                {
                    if (request.cell_id == cell_id)
                    {
                        request.refinement_type =
                            combine_refinement_types_2d_(
                                request.refinement_type,
                                refinement_type);
                        return;
                    }
                }
            }

            scheduled[idx] = true;
            pending.push_back(RefinementRequest2D{cell_id, refinement_type});
        }

        void enqueue_same_spatial_leaf_requests_2d_(
            cell_id_type cell_id,
            RefinementType refinement_type,
            RefinementType propagated_refinement_type,
            std::vector<RefinementRequest2D>& pending,
            std::vector<bool>& scheduled) const
        {
            if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= cells_.size())
                throw core::invalid_mesh_error(
                    "Mesh::enqueue_same_spatial_leaf_requests_2d: cell_id out of range.");

            const auto target_refinement_type =
                effective_refinement_type_2d_(cell_id, refinement_type);

            if (!refinement_has_spatial_part_2d_(target_refinement_type))
            {
                enqueue_refinement_request_2d_(
                    cell_id,
                    target_refinement_type,
                    pending,
                    scheduled);
                return;
            }

            const auto& target_cell = cells_[static_cast<std::size_t>(cell_id)];
            for (const auto leaf_cell_id : leaf_cell_ids())
            {
                const auto& leaf_cell = cell(leaf_cell_id);
                if (!topology::same_spatial_cell_vertices_2d(leaf_cell, target_cell))
                    continue;

                if (!topology::temporal_intervals_overlap_positive_2d(
                        *this,
                        leaf_cell,
                        target_cell))
                {
                    continue;
                }

                const auto leaf_refinement_type =
                    leaf_cell_id == cell_id
                        ? target_refinement_type
                        : effective_refinement_type_2d_(
                              leaf_cell_id,
                              propagated_refinement_type);

                enqueue_refinement_request_2d_(
                    leaf_cell_id,
                    leaf_refinement_type,
                    pending,
                    scheduled);
            }
        }

        void enqueue_spatial_closure_forced_cells_2d_(
            std::vector<RefinementRequest2D>& pending,
            std::vector<bool>& scheduled,
            const std::vector<cell_id_type>& seed_cells,
            const TimingMetricCallback& timing_callback) const
        {
            const auto active_cells = leaf_cell_ids();
            const auto result =
                topology::collect_local_spatial_closure_forced_cells_2d(
                    *this,
                    active_cells,
                    seed_cells);
            record_local_closure_metrics_2d_(
                timing_callback,
                "fespace.refinement.mesh_refine.closure",
                result.stats);

            const auto pending_before = pending.size();
            for (const auto cell_id : result.forced_cell_ids)
            {
                enqueue_refinement_request_2d_(
                    cell_id,
                    effective_refinement_type_2d_(
                        cell_id,
                        RefinementType::none),
                    pending,
                    scheduled);
            }
            record_timing_metric_(
                timing_callback,
                "fespace.refinement.mesh_refine.closure.forced_refinements_added.count",
                static_cast<double>(pending.size() - pending_before));
        }

        void assert_leaf_spatial_conforming_full_2d_(
            const TimingMetricCallback& timing_callback) const
        {
            const auto active_cells = leaf_cell_ids();
            const auto final_adjacency =
                topology::build_slicewise_active_spatial_edge_adjacency_2d(
                    *this,
                    active_cells);
            record_adjacency_metrics_2d_(
                timing_callback,
                "fespace.refinement.mesh_refine.assert_leaf_spatial_conforming",
                active_cells.size(),
                final_adjacency);
            if (final_adjacency.has_singular_edges())
                throw core::invalid_mesh_error(
                    "Mesh::refine_2d_with_closure: closure left singular active spatial edges.");
            if (final_adjacency.has_nonconforming_edges())
                throw core::invalid_mesh_error(
                    "Mesh::refine_2d_with_closure: closure left nonconforming active edges.");
        }

        void assert_leaf_spatial_conforming_local_2d_(
            const std::vector<cell_id_type>& seed_cells,
            const TimingMetricCallback& timing_callback) const
        {
            const auto active_cells = leaf_cell_ids();
            const auto result =
                topology::verify_local_spatial_conforming_2d(
                    *this,
                    active_cells,
                    seed_cells);
            record_local_verification_metrics_2d_(
                timing_callback,
                "fespace.refinement.mesh_refine.assert_leaf_spatial_conforming",
                result.stats);

            if (!result.is_conforming)
                throw core::invalid_mesh_error(
                    "Mesh::refine_2d_with_closure: local closure check failed.");
        }

        void refine_2d_with_closure_(
            const std::vector<cell_id_type>& marked_cell_ids,
            RefinementType refinement_type,
            const TimingMetricCallback& timing_callback)
        {
            static_assert(GeomTraits::dim_space_v == 2,
                          "refine_2d_with_closure_ requires dim_space_v == 2.");

            if (marked_cell_ids.empty())
                return;

            std::vector<RefinementRequest2D> pending;
            pending.reserve(marked_cell_ids.size());

            std::vector<bool> scheduled(cells_.size(), false);
            for (const auto cell_id : marked_cell_ids)
            {
                const auto effective_type =
                    effective_refinement_type_2d_(cell_id, refinement_type);
                validate_refinement_request_2d_(cell_id, effective_type);
                enqueue_same_spatial_leaf_requests_2d_(
                    cell_id,
                    refinement_type,
                    RefinementType::none,
                    pending,
                    scheduled);
            }

            bool applied_spatial_part = false;
            std::vector<cell_id_type> verification_seed_cells;
            std::size_t cursor = 0;
            while (cursor < pending.size())
            {
                const auto wave_end = pending.size();
                bool wave_had_spatial_part = false;
                std::vector<cell_id_type> wave_seed_cells;
                while (cursor < wave_end)
                {
                    const auto request = pending[cursor++];
                    const auto cell_id = request.cell_id;
                    if (cell_id < 0 || static_cast<std::size_t>(cell_id) >= cells_.size())
                        throw core::invalid_mesh_error(
                            "Mesh::refine_2d_with_closure: pending cell_id out of range.");

                    if (!cells_[static_cast<std::size_t>(cell_id)].is_leaf)
                        continue;

                    refinement::refine(
                        cells_,
                        spatial_vertices_,
                        temporal_vertices_,
                        registry_,
                        spatial_boundary_face_vertex_ids_,
                        temporal_boundary_vertex_ids_,
                        cell_id,
                        request.refinement_type);

                    if (refinement_has_spatial_part_2d_(request.refinement_type))
                    {
                        wave_had_spatial_part = true;
                        for (const auto child_id :
                             cells_[static_cast<std::size_t>(cell_id)].children)
                        {
                            if (child_id >= 0 &&
                                static_cast<std::size_t>(child_id) <
                                    cells_.size() &&
                                cells_[static_cast<std::size_t>(child_id)]
                                    .is_leaf)
                            {
                                wave_seed_cells.push_back(child_id);
                                verification_seed_cells.push_back(child_id);
                            }
                        }
                    }
                }

                scheduled.resize(cells_.size(), false);
                if (wave_had_spatial_part)
                {
                    applied_spatial_part = true;
                    enqueue_spatial_closure_forced_cells_2d_(
                        pending,
                        scheduled,
                        wave_seed_cells,
                        timing_callback);
                }
            }

            if (applied_spatial_part)
            {
                std::sort(
                    verification_seed_cells.begin(),
                    verification_seed_cells.end());
                verification_seed_cells.erase(
                    std::unique(
                        verification_seed_cells.begin(),
                        verification_seed_cells.end()),
                    verification_seed_cells.end());
                assert_leaf_spatial_conforming_local_2d_(
                    verification_seed_cells,
                    timing_callback);
#ifndef NDEBUG
                assert_leaf_spatial_conforming_full_2d_(timing_callback);
#endif
            }
        }

        [[nodiscard]] static SpatialFace sorted_spatial_face_(SpatialFace face)
        {
            std::sort(face.begin(), face.end());
            return face;
        }

        void rebuild_spatial_boundary_face_vertex_ids_from_root_faces_()
        {
            struct FaceCount
            {
                SpatialFace face{};
                int count = 0;
            };

            std::vector<FaceCount> counts;

            for (const auto& c : cells_)
            {
                if (c.has_parent())
                    continue;

                for (const auto& face_data : c.spatial_faces)
                {
                    const auto face = sorted_spatial_face_(face_data.spatial_vertex_ids);
                    auto it = std::find_if(
                        counts.begin(),
                        counts.end(),
                        [&](const FaceCount& entry)
                        {
                            return entry.face == face;
                        });

                    if (it == counts.end())
                        counts.push_back(FaceCount{face, 1});
                    else
                        ++it->count;
                }
            }

            spatial_boundary_face_vertex_ids_.clear();
            spatial_boundary_vertex_ids_.clear();

            for (const auto& entry : counts)
            {
                if (entry.count != 1)
                    continue;

                spatial_boundary_face_vertex_ids_.push_back(entry.face);
                for (const auto vid : entry.face)
                {
                    if (std::find(
                            spatial_boundary_vertex_ids_.begin(),
                            spatial_boundary_vertex_ids_.end(),
                            vid) == spatial_boundary_vertex_ids_.end())
                    {
                        spatial_boundary_vertex_ids_.push_back(vid);
                    }
                }
            }
        }

        void rebuild_temporal_boundary_vertex_ids_from_roots_()
        {
            temporal_boundary_vertex_ids_.clear();

            for (const auto& c : cells_)
            {
                if (c.has_parent())
                    continue;

                for (const auto vid : c.temporal_vertex_ids)
                {
                    if (std::find(
                            temporal_boundary_vertex_ids_.begin(),
                            temporal_boundary_vertex_ids_.end(),
                            vid) == temporal_boundary_vertex_ids_.end())
                    {
                        temporal_boundary_vertex_ids_.push_back(vid);
                    }
                }
            }
        }

        void recompute_all_cell_boundaries_()
        {
            for (auto& c : cells_)
            {
                if constexpr (GeomTraits::dim_space_v == 1)
                {
                    topology::fill_boundary(
                        c,
                        spatial_boundary_vertex_ids_,
                        temporal_boundary_vertex_ids_);
                }
                else if constexpr (GeomTraits::dim_space_v == 2)
                {
                    topology::fill_boundary(
                        c,
                        spatial_boundary_face_vertex_ids_,
                        spatial_vertices_,
                        temporal_boundary_vertex_ids_);
                }
            }
        }

        void finalize_root_boundaries_()
        {
            rebuild_spatial_boundary_face_vertex_ids_from_root_faces_();
            rebuild_temporal_boundary_vertex_ids_from_roots_();
            recompute_all_cell_boundaries_();
        }

        void bump_storage_version_() noexcept
        {
            ++storage_version_;
        }

        std::vector<CellType> cells_{};
        std::vector<SpatialPoint> spatial_vertices_{};
        std::vector<TemporalPoint> temporal_vertices_{};

        std::vector<vertex_id_type> spatial_boundary_vertex_ids_{};
        std::vector<SpatialFace> spatial_boundary_face_vertex_ids_{};
        std::vector<vertex_id_type> temporal_boundary_vertex_ids_{};

        detail::VertexRegistry<GeomTraits> registry_{};
        std::uint64_t storage_version_ = 0;
    };
}

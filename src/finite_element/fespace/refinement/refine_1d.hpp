#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

#include "../../../mesh/refinement/refinement_type.hpp"

namespace finite_element::fespace::detail::refinement_impl
{
    struct ActiveRefinementMetrics1D
    {
        std::size_t initially_marked_active_cells = 0;
        std::size_t grading_forced_cells = 0;
        std::size_t actually_split_active_cells = 0;
        std::size_t storage_children_created = 0;
        std::size_t existing_children_reused = 0;
        std::size_t full_active_scans_in_normal_path = 0;
    };

    template<typename FESpaceType>
    inline void record_refinement_metric_count_1d(
        FESpaceType& space,
        std::string_view name,
        std::size_t value)
    {
        const auto count = static_cast<double>(value);
        space.record_timing_metric(name, count);
        space.record_timing_metric(std::string(name) + ".count", count);
    }

    template<typename FESpaceType>
    inline void record_refinement_metrics_1d(
        FESpaceType& space,
        const ActiveRefinementMetrics1D& metrics)
    {
        record_refinement_metric_count_1d(
            space,
            "refinement.initially_marked_active_cells",
            metrics.initially_marked_active_cells);
        record_refinement_metric_count_1d(
            space,
            "refinement.grading_forced_cells",
            metrics.grading_forced_cells);
        record_refinement_metric_count_1d(
            space,
            "refinement.actually_split_active_cells",
            metrics.actually_split_active_cells);
        record_refinement_metric_count_1d(
            space,
            "refinement.storage_children_created",
            metrics.storage_children_created);
        record_refinement_metric_count_1d(
            space,
            "refinement.existing_children_reused",
            metrics.existing_children_reused);
        record_refinement_metric_count_1d(
            space,
            "refinement.full_active_scans_in_normal_path",
            metrics.full_active_scans_in_normal_path);
        record_refinement_metric_count_1d(
            space,
            "refinement.full_active_scans",
            metrics.full_active_scans_in_normal_path);
        space.record_timing_metric("edge_index.remove_total", 0.0);
        space.record_timing_metric("edge_index.add_total", 0.0);
        space.record_timing_metric("edge_index.rebuild_total", 0.0);
        space.record_timing_metric("edge_index.query_total", 0.0);
        space.record_timing_metric(
            "refinement.local_conformity_check_total",
            0.0);
    }

    template<typename FESpaceType>
    [[nodiscard]] inline std::vector<int>
    refine_single_active_cell_1d(
        FESpaceType& space,
        int target_cell_id,
        mesh::RefinementType requested_refinement_type,
        ActiveRefinementMetrics1D& metrics)
    {
        auto& mesh = space.unsafe_mesh_ref();

        if (!space.is_active_cell(target_cell_id))
            throw std::runtime_error("FESpace::refine: marked cell is not active in this FESpace.");

        if (target_cell_id < 0 || static_cast<std::size_t>(target_cell_id) >= mesh.n_cells())
            throw std::runtime_error("FESpace::refine: marked cell id out of range.");

        // FE-space refinement operates on the active partition. If the active cell is
        // already a non-leaf in the mesh, we reuse its existing children instead of
        // asking the mesh layer to create them again.
        if (mesh.cell(target_cell_id).is_leaf)
        {
            const auto split_type =
                requested_refinement_type == mesh::RefinementType::none
                    ? mesh.next_refinement_type(target_cell_id)
                    : requested_refinement_type;
            mesh.refine(target_cell_id, split_type);
            ++metrics.storage_children_created;
        }
        else
        {
            ++metrics.existing_children_reused;
        }

        const auto& parent = mesh.cell(target_cell_id);
        if (parent.children.empty())
            throw std::runtime_error("FESpace::refine: refined active cell has no children.");

        std::vector<int> children;
        children.reserve(parent.children.size());

        for (const int child_id : parent.children)
        {
            children.push_back(child_id);
        }

        return children;
    }

    template<typename FESpaceType>
    inline void collect_temporal_grading_forced_refinements_1d(
        const FESpaceType& space,
        std::vector<int>& forced,
        std::unordered_set<int>& forced_seen)
    {
        if constexpr (!FESpaceType::PolicyType::enforce_temporal_grading)
        {
            return;
        }

        const auto& mesh = space.mesh_ref();
        const auto& adjacency = space.adjacency_ref();

        for (const auto& iface : adjacency.spatial_interfaces)
        {
            if (iface.is_boundary || !iface.is_hanging || iface.slave_cell < 0)
                continue;
            if (!space.is_active_cell(iface.master_cell) || !space.is_active_cell(iface.slave_cell))
                continue;

            const auto& master = mesh.cell(iface.master_cell);
            const auto& slave = mesh.cell(iface.slave_cell);

            if (slave.temporal_level > master.temporal_level + 1)
            {
                if (forced_seen.insert(iface.master_cell).second)
                    forced.push_back(iface.master_cell);
            }
        }
    }

    template<typename FESpaceType>
    inline void collect_spatial_grading_forced_refinements_1d(
        const FESpaceType& space,
        std::vector<int>& forced,
        std::unordered_set<int>& forced_seen)
    {
        if constexpr (!FESpaceType::PolicyType::enforce_spatial_grading_d1)
        {
            return;
        }

        const auto& mesh = space.mesh_ref();
        const auto& adjacency = space.adjacency_ref();

        for (const auto& iface : adjacency.temporal_interfaces)
        {
            if (iface.is_boundary || !iface.is_hanging || iface.slave_cell < 0)
                continue;
            if (!space.is_active_cell(iface.master_cell) || !space.is_active_cell(iface.slave_cell))
                continue;

            const auto& master = mesh.cell(iface.master_cell);
            const auto& slave = mesh.cell(iface.slave_cell);

            if (slave.spatial_level > master.spatial_level + 1)
            {
                if (forced_seen.insert(iface.master_cell).second)
                    forced.push_back(iface.master_cell);
            }
        }
    }

    template<typename FESpaceType>
    inline bool refine_closure_recursive_1d(
        FESpaceType& space,
        const std::vector<int>& marked,
        ActiveRefinementMetrics1D& metrics,
        mesh::RefinementType requested_refinement_type =
            mesh::RefinementType::none)
    {
        if (marked.empty())
            return false;

        // Step 1: refine the current wave.
        std::vector<int> children;
        std::unordered_set<int> add_ids;
        std::unordered_set<int> remove_ids;
        {
            std::unordered_set<int> child_seen;
            add_ids.reserve(marked.size() * 2);
            remove_ids.reserve(marked.size());

            space.time_phase(
                "refinement.child_creation_total",
                [&]()
                {
                    for (const int cell_id : marked)
                    {
                        if (!space.is_active_cell(cell_id))
                            continue;

                        const auto refined_children =
                            refine_single_active_cell_1d(
                                space,
                                cell_id,
                                requested_refinement_type,
                                metrics);
                        ++metrics.actually_split_active_cells;
                        remove_ids.insert(cell_id);

                        for (const int child_id : refined_children)
                        {
                            add_ids.insert(child_id);
                            if (child_seen.insert(child_id).second)
                                children.push_back(child_id);
                        }
                    }
                });
        }

        if (children.empty())
            return false;

        space.time_phase(
            "refinement.active_partition_update_total",
            [&]()
            {
                space.unsafe_update_active_cells(add_ids, remove_ids);
            });

        if constexpr (!(FESpaceType::PolicyType::enforce_spatial_grading_d1
                     || FESpaceType::PolicyType::enforce_temporal_grading))
        {
            return false;
        }

        // Step 2: rebuild adjacency on the new active partition so grading can be checked
        // directly on hanging interfaces.
        space.build_adjacency();

        // Step 3: recursively collect coarse masters whose hanging faces violate
        // the one-level grading rule in the relevant direction.
        std::vector<int> forced;
        std::unordered_set<int> forced_seen;

        collect_spatial_grading_forced_refinements_1d(space, forced, forced_seen);
        collect_temporal_grading_forced_refinements_1d(space, forced, forced_seen);
        metrics.grading_forced_cells += forced.size();

        // Step 4: recurse on the closure.
        [[maybe_unused]] const bool child_adjacency_is_current =
            refine_closure_recursive_1d(space, forced, metrics);
        return true;
    }

    template<typename FESpaceType>
    inline void refine_1d(
        FESpaceType& space,
        const std::vector<int>& marked,
        mesh::RefinementType requested_refinement_type =
            mesh::RefinementType::none)
    {
        using GeomTraits = typename FESpaceType::GT;
        static_assert(GeomTraits::dim_space_v == 1, "refine_1d requires dim_space_v == 1.");
        static_assert(GeomTraits::dim_time_v == 1, "refine_1d requires dim_time_v == 1.");

        std::vector<int> unique_marked;
        unique_marked.reserve(marked.size());

        {
            std::unordered_set<int> seen;
            for (const int cell_id : marked)
            {
                if (space.is_active_cell(cell_id) && seen.insert(cell_id).second)
                    unique_marked.push_back(cell_id);
            }
        }

        if (unique_marked.empty())
            return;

        ActiveRefinementMetrics1D metrics;
        metrics.initially_marked_active_cells = unique_marked.size();

        bool adjacency_is_current = false;
        space.time_phase(
            "refinement.queue_closure_total",
            [&]()
            {
                adjacency_is_current =
                    refine_closure_recursive_1d(
                        space,
                        unique_marked,
                        metrics,
                        requested_refinement_type);
            });

        if (adjacency_is_current)
            space.distribute_dofs();
        else
            space.rebuild();

        record_refinement_metrics_1d(space, metrics);
    }
}

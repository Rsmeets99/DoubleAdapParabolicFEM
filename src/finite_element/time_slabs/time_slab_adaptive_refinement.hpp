#pragma once

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "../assembly/detail/active_cell_locator.hpp"
#include "../../mesh/refinement/refinement_type.hpp"

namespace finite_element::time_slabs
{
    enum class LocalTimeSlabClosureMode
    {
        disabled,
        marked_split_cells
    };

    struct LocalTimeSlabClosureStats
    {
        int marked_split_cells = 0;
        int temporal_refinement_waves = 0;
        int temporally_refined_cells = 0;
    };

    template<class XSpaceType, class YSpaceType>
    [[nodiscard]] bool trial_space_embedded_in_test_space(
        const XSpaceType& x_space,
        const YSpaceType& y_space)
    {
        if (&x_space.mesh_ref() != &y_space.mesh_ref())
        {
            throw std::runtime_error(
                "trial_space_embedded_in_test_space: X and Y must share the same mesh.");
        }

        finite_element::assembly::detail::ActiveAncestorCache<XSpaceType> ancestor_cache(x_space);

        for (const int y_cell_id : y_space.active_cells())
        {
            try
            {
                (void)finite_element::assembly::detail::find_active_ancestor_cell(
                    ancestor_cache,
                    x_space,
                    y_space,
                    y_cell_id);
            }
            catch (...)
            {
                return false;
            }
        }

        return true;
    }

    template<class XSpaceType, class YSpaceType>
    void require_trial_space_embedded_in_test_space(
        const XSpaceType& x_space,
        const YSpaceType& y_space)
    {
        if (!trial_space_embedded_in_test_space(x_space, y_space))
        {
            throw std::runtime_error(
                "require_trial_space_embedded_in_test_space: X^delta is not embedded in Y^delta.");
        }
    }

    [[nodiscard]] inline double time_comparison_tolerance(
        double a,
        double b) noexcept
    {
        return 64.0 * std::numeric_limits<double>::epsilon() *
            std::max({1.0, std::abs(a), std::abs(b)});
    }

    template<class SpaceType>
    [[nodiscard]] std::vector<double> active_time_values(
        const SpaceType& space)
    {
        struct EndpointRecord
        {
            int temporal_vertex_id = -1;
            double time = 0.0;
        };

        std::vector<EndpointRecord> endpoints;
        endpoints.reserve(2U * space.active_cells().size());

        const auto& mesh = space.mesh_ref();
        const auto& temporal_vertices = mesh.temporal_vertices();
        for (const int cell_id : space.active_cells())
        {
            const auto& cell = mesh.cell(cell_id);
            for (const int temporal_vertex_id : cell.temporal_vertex_ids)
            {
                endpoints.push_back(
                    EndpointRecord{
                        temporal_vertex_id,
                        temporal_vertices[
                            static_cast<std::size_t>(temporal_vertex_id)][0]});
            }
        }

        std::sort(
            endpoints.begin(),
            endpoints.end(),
            [](const EndpointRecord& a, const EndpointRecord& b)
            {
                if (a.time != b.time)
                    return a.time < b.time;
                return a.temporal_vertex_id < b.temporal_vertex_id;
            });

        std::vector<double> values;
        values.reserve(endpoints.size());
        int last_vertex_id = -1;
        bool have_last_vertex = false;
        for (const auto& endpoint : endpoints)
        {
            if (have_last_vertex &&
                endpoint.temporal_vertex_id == last_vertex_id)
            {
                continue;
            }
            if (values.empty() || endpoint.time != values.back())
                values.push_back(endpoint.time);
            last_vertex_id = endpoint.temporal_vertex_id;
            have_last_vertex = true;
        }

        return values;
    }

    template<class SpaceType>
    [[nodiscard]] bool cell_is_crossed_by_time_value(
        const SpaceType& space,
        int cell_id,
        double t)
    {
        const auto& mesh = space.mesh_ref();
        const auto& cell = mesh.cell(cell_id);
        const auto& temporal_vertices = mesh.temporal_vertices();
        const double t0 =
            temporal_vertices[
                static_cast<std::size_t>(cell.temporal_vertex_ids[0])][0];
        const double t1 =
            temporal_vertices[
                static_cast<std::size_t>(cell.temporal_vertex_ids[1])][0];

        const double lower = std::min(t0, t1);
        const double upper = std::max(t0, t1);
        const double tol =
            std::max(time_comparison_tolerance(lower, t),
                     time_comparison_tolerance(upper, t));
        return t > lower + tol && t < upper - tol;
    }

    template<class SpaceType>
    [[nodiscard]] bool cell_is_crossed_by_time_values(
        const SpaceType& space,
        int cell_id,
        const std::vector<double>& time_values)
    {
        return std::any_of(
            time_values.begin(),
            time_values.end(),
            [&](double t)
            {
                return cell_is_crossed_by_time_value(space, cell_id, t);
            });
    }

    template<class CellwiseErrorType>
    [[nodiscard]] std::vector<int>
    doerfler_mark_test_space_entities(
        const CellwiseErrorType& error,
        double theta)
    {
        return error.doerfler_marking(theta);
    }

    template<class YSpaceType>
    [[nodiscard]] std::vector<int> active_children_of_cells(
        const YSpaceType& y_space,
        const std::vector<int>& parent_cells)
    {
        const auto& mesh = y_space.mesh_ref();
        std::vector<int> children;
        std::unordered_set<int> seen;

        for (const int parent_id : parent_cells)
        {
            const auto& parent = mesh.cell(parent_id);
            for (const int child_id : parent.children)
            {
                if (y_space.is_active_cell(child_id) &&
                    seen.insert(child_id).second)
                {
                    children.push_back(child_id);
                }
            }
        }

        return children;
    }

    template<class YSpaceType>
    void refine_temporally_for_time_slab_closure(
        YSpaceType& y_space,
        const std::vector<int>& crossed_cells)
    {
        // This is an auxiliary local time-slab closure operation. It
        // deliberately requests a pure temporal split so that cells crossed by
        // existing active time values are cut before later slab construction.
        // Normal X/Y physical refinement must use y_space.refine(marked),
        // which follows the generation-based split policy.
        y_space.refine(crossed_cells, mesh::RefinementType::temporal);
    }

    template<class XSpaceType, class YSpaceType>
    LocalTimeSlabClosureStats refine_test_space_from_marked_source_cells(
        YSpaceType& y_space,
        const XSpaceType& x_space,
        const std::vector<int>& marked_source_cells,
        LocalTimeSlabClosureMode closure_mode);

    template<class XSpaceType, class YSpaceType>
    void refine_test_space_from_marked_source_cells(
        YSpaceType& y_space,
        const XSpaceType& x_space,
        const std::vector<int>& marked_source_cells)
    {
        (void)refine_test_space_from_marked_source_cells(
            y_space,
            x_space,
            marked_source_cells,
            LocalTimeSlabClosureMode::disabled);
    }

    template<class XSpaceType, class YSpaceType>
    LocalTimeSlabClosureStats refine_test_space_from_marked_source_cells(
        YSpaceType& y_space,
        const XSpaceType& x_space,
        const std::vector<int>& marked_source_cells,
        LocalTimeSlabClosureMode closure_mode)
    {
        if (&x_space.mesh_ref() != &y_space.mesh_ref())
        {
            throw std::runtime_error(
                "refine_test_space_from_marked_source_cells: X and Y must share the same mesh.");
        }

        require_trial_space_embedded_in_test_space(x_space, y_space);

        std::vector<int> unique_marked;
        unique_marked.reserve(marked_source_cells.size());

        std::unordered_set<int> seen;
        for (const int cell_id : marked_source_cells)
        {
            if (!y_space.is_active_cell(cell_id))
            {
                throw std::runtime_error(
                    "refine_test_space_from_marked_source_cells: marked cell is not active in Y^delta.");
            }

            if (seen.insert(cell_id).second)
                unique_marked.push_back(cell_id);
        }

        if (unique_marked.empty())
            return {};

        LocalTimeSlabClosureStats stats;

        if (closure_mode == LocalTimeSlabClosureMode::disabled)
        {
            y_space.refine(unique_marked);
            require_trial_space_embedded_in_test_space(x_space, y_space);
            return stats;
        }

        const auto time_values = active_time_values(y_space);
        std::vector<int> normal_marked;
        std::vector<int> split_marked;
        normal_marked.reserve(unique_marked.size());
        split_marked.reserve(unique_marked.size());

        for (const int cell_id : unique_marked)
        {
            if (cell_is_crossed_by_time_values(y_space, cell_id, time_values))
                split_marked.push_back(cell_id);
            else
                normal_marked.push_back(cell_id);
        }

        stats.marked_split_cells = static_cast<int>(split_marked.size());

        if (!normal_marked.empty())
            y_space.refine(normal_marked);

        std::vector<int> closure_wave = std::move(split_marked);
        while (!closure_wave.empty())
        {
            std::vector<int> crossed_cells;
            crossed_cells.reserve(closure_wave.size());
            for (const int cell_id : closure_wave)
            {
                if (y_space.is_active_cell(cell_id) &&
                    cell_is_crossed_by_time_values(
                        y_space,
                        cell_id,
                        time_values))
                {
                    crossed_cells.push_back(cell_id);
                }
            }

            if (crossed_cells.empty())
                break;

            ++stats.temporal_refinement_waves;
            stats.temporally_refined_cells +=
                static_cast<int>(crossed_cells.size());

            refine_temporally_for_time_slab_closure(y_space, crossed_cells);
            closure_wave = active_children_of_cells(y_space, crossed_cells);
        }

        require_trial_space_embedded_in_test_space(x_space, y_space);
        return stats;
    }
}

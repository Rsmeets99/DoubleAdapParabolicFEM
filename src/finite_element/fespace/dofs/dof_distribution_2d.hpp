#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../../basis/polynomials/segment_lagrange.hpp"
#include "../../basis/polynomials/triangular_lagrange.hpp"
#include "../../../mesh/topology/boundary_2d.hpp"
#include "dof_entity_key_2d.hpp"
#include "dof_distribution_1d.hpp"

namespace finite_element::fespace::detail::dof_distribution_impl
{
    template<typename Context>
    [[nodiscard]] inline std::array<double, 2> reference_spatial_coord_2d(
        const Context& ctx,
        int cell_id,
        const std::array<double, Context::dim_v>& physical_point)
    {
        static_assert(Context::SpaceType::GT::dim_space_v == 2,
                      "reference_spatial_coord_2d requires dim_space_v == 2.");

        const auto& cell = ctx.mesh.cell(cell_id);
        const auto& v0 = ctx.mesh.spatial_vertices()[cell.spatial_vertex_ids[0]];
        const auto& v1 = ctx.mesh.spatial_vertices()[cell.spatial_vertex_ids[1]];
        const auto& v2 = ctx.mesh.spatial_vertices()[cell.spatial_vertex_ids[2]];

        const double J00 = v1[0] - v0[0];
        const double J01 = v2[0] - v0[0];
        const double J10 = v1[1] - v0[1];
        const double J11 = v2[1] - v0[1];
        const double det = J00 * J11 - J01 * J10;

        if (std::abs(det) < 1.0e-15)
        {
            throw std::runtime_error(
                "DoF distribution failed: degenerate master triangle in 2D constraint.");
        }

        const double dx = physical_point[0] - v0[0];
        const double dy = physical_point[1] - v0[1];
        const double inv_det = 1.0 / det;

        return {
            ( J11 * dx - J01 * dy) * inv_det,
            (-J10 * dx + J00 * dy) * inv_det
        };
    }

    template<typename Context>
    [[nodiscard]] inline bool spatial_point_on_boundary_2d(
        const Context& ctx,
        const std::array<double, Context::dim_v>& p)
    {
        typename Context::MeshType::SpatialPoint x{};
        x[0] = p[0];
        x[1] = p[1];

        const auto& spatial_vertices = ctx.mesh.spatial_vertices();
        for (const auto& boundary_edge : ctx.mesh.spatial_boundary_face_vertex_ids())
        {
            const auto& a =
                spatial_vertices[static_cast<std::size_t>(boundary_edge[0])];
            const auto& b =
                spatial_vertices[static_cast<std::size_t>(boundary_edge[1])];
            if (mesh::topology::point_on_segment_2d<typename Context::SpaceType::GT>(
                    x,
                    a,
                    b))
            {
                return true;
            }
        }

        return false;
    }

    template<typename Context>
    inline void mark_eliminated_occurrences_2d(Context& ctx)
    {
        for (int i = 0; i < static_cast<int>(ctx.local_occurrences.size()); ++i)
        {
            const auto& ref =
                ctx.local_occurrences[static_cast<std::size_t>(i)];
            const auto& cell = ctx.mesh.cell(ref.cell_id);
            const auto& meta = Context::ElemTables::meta(ref.local_index);

            bool on_spatial_boundary = false;
            for (int k = 0; k < meta.num_spatial_faces; ++k)
            {
                const int face = meta.spatial_faces[static_cast<std::size_t>(k)];
                if (face >= 0 && cell.spatial_boundary[static_cast<std::size_t>(face)])
                {
                    on_spatial_boundary = true;
                    break;
                }
            }

            if (!on_spatial_boundary)
            {
                const auto p =
                    finite_element::fespace::physical_dof_coord(
                        ctx.space,
                        ref.cell_id,
                        ref.local_index);
                on_spatial_boundary = spatial_point_on_boundary_2d(ctx, p);
            }

            if (on_spatial_boundary)
                ctx.is_eliminated[static_cast<std::size_t>(i)] = 1;
        }
    }

    template<typename ArrayType>
    inline void append_array_2d(std::ostringstream& out, const ArrayType& values)
    {
        out << '[';
        bool first = true;
        for (const auto& value : values)
        {
            if (!first)
                out << ',';
            first = false;
            out << value;
        }
        out << ']';
    }

    template<typename Context>
    inline void append_constraint_pairs_2d(
        std::ostringstream& out,
        const std::vector<std::pair<int, double>>& pairs)
    {
        out << '[';
        for (std::size_t i = 0; i < pairs.size(); ++i)
        {
            if (i > 0)
                out << ',';
            out << "(gid=" << pairs[i].first << ",w=" << pairs[i].second << ')';
        }
        out << ']';
    }

    struct ConstraintSource2D
    {
        bool is_spatial = false;
        int interface_id = -1;

        bool operator==(const ConstraintSource2D&) const noexcept = default;
    };

    struct CyclicCanonicalConstraintRoute2D : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    struct ScopedOccurrenceExpansionVisit2D
    {
        std::vector<char>& visiting;
        std::size_t id = 0;

        ScopedOccurrenceExpansionVisit2D(
            std::vector<char>& visiting_in,
            const int id_in)
            : visiting(visiting_in),
              id(static_cast<std::size_t>(id_in))
        {
            visiting[id] = 1;
        }

        ~ScopedOccurrenceExpansionVisit2D()
        {
            visiting[id] = 0;
        }
    };

    struct ScopedCanonicalKeyExpansionVisit2D
    {
        using Key = finite_element::fespace::DofEntityKey2D;
        using KeyHash = finite_element::fespace::DofEntityKey2DHash;

        std::unordered_set<Key, KeyHash>& visiting;
        Key key;

        ScopedCanonicalKeyExpansionVisit2D(
            std::unordered_set<Key, KeyHash>& visiting_in,
            const Key& key_in)
            : visiting(visiting_in),
              key(key_in)
        {
            visiting.insert(key);
        }

        ~ScopedCanonicalKeyExpansionVisit2D()
        {
            visiting.erase(key);
        }
    };

    template<typename Context>
    inline void append_constraint_source_summary_2d(
        std::ostringstream& out,
        Context& ctx,
        const ConstraintSource2D& source)
    {
        out << (source.is_spatial ? "spatial" : "temporal")
            << "#" << source.interface_id;
        if (source.interface_id < 0)
            return;

        if (source.is_spatial)
        {
            if (static_cast<std::size_t>(source.interface_id) >=
                ctx.adjacency.spatial_interfaces.size())
            {
                out << "<invalid>";
                return;
            }
            const auto& iface =
                ctx.adjacency.spatial_interfaces[
                    static_cast<std::size_t>(source.interface_id)];
            out << "{slave_cell=" << iface.slave_cell
                << ",slave_face=" << iface.slave_face
                << ",master_cell=" << iface.master_cell
                << ",master_face=" << iface.master_face
                << ",hanging=" << (iface.is_hanging ? 1 : 0)
                << '}';
        }
        else
        {
            if (static_cast<std::size_t>(source.interface_id) >=
                ctx.adjacency.temporal_interfaces.size())
            {
                out << "<invalid>";
                return;
            }
            const auto& iface =
                ctx.adjacency.temporal_interfaces[
                    static_cast<std::size_t>(source.interface_id)];
            out << "{slave_cell=" << iface.slave_cell
                << ",slave_face=" << iface.slave_face
                << ",master_cell=" << iface.master_cell
                << ",master_face=" << iface.master_face
                << ",hanging=" << (iface.is_hanging ? 1 : 0)
                << '}';
        }
    }

    template<typename Context>
    struct CanonicalDistributionState2D
    {
        using Key = finite_element::fespace::DofEntityKey2D;
        using KeyHash = finite_element::fespace::DofEntityKey2DHash;

        std::vector<finite_element::fespace::DofEntityKey2D> entity_keys{};
        std::vector<std::vector<ConstraintSource2D>> constraint_sources{};

        std::unordered_set<Key, KeyHash> constrained_keys{};
        std::unordered_map<Key, int, KeyHash> true_key_to_global{};
        std::unordered_map<Key, int, KeyHash> constrained_key_to_global{};
        std::unordered_map<Key, std::vector<std::pair<int, double>>, KeyHash>
            constrained_key_to_pairs{};
        std::unordered_map<Key, std::vector<int>, KeyHash>
            occurrence_ids_by_key{};
        std::unordered_set<Key, KeyHash> constrained_key_expansion_visiting{};

        std::vector<char> expansion_visiting{};
        std::vector<char> expansion_has_memo{};
        std::vector<std::vector<std::pair<int, double>>> expansion_memo{};
    };

    template<typename Context>
    [[nodiscard]] inline std::string canonical_key_summary_2d(
        const finite_element::fespace::DofEntityKey2D& key)
    {
        std::ostringstream out;
        out << "sk=" << static_cast<int>(key.spatial_kind)
            << " sids=[";
        append_array_2d(out, key.spatial_entity_ids);
        out << "] snode=[";
        append_array_2d(out, key.spatial_node_tuple);
        out << "] edge_ord=" << key.spatial_edge_node_ordinal
            << " tk=" << static_cast<int>(key.temporal_kind)
            << " tv=" << key.temporal_vertex_id
            << " tint=[";
        append_array_2d(out, key.temporal_interval_ids);
        out << "] tnode=" << key.temporal_node_ordinal
            << " disc=" << (key.discontinuous_time ? 1 : 0);
        return out.str();
    }

    template<typename Context>
    [[nodiscard]] inline std::string local_ref_summary_2d(
        Context& ctx,
        const typename Context::LocalDoFRef& ref)
    {
        const auto p =
            finite_element::fespace::physical_dof_coord(
                ctx.space,
                ref.cell_id,
                ref.local_index);
        std::ostringstream out;
        out << "cell=" << ref.cell_id
            << " local=" << ref.local_index
            << " coord=[";
        append_array_2d(out, p);
        out << ']';
        return out.str();
    }

    template<typename Context>
    inline void initialize_canonical_entity_keys_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state)
    {
        const auto n_occurrences = ctx.local_occurrences.size();
        state.entity_keys.resize(n_occurrences);
        state.constraint_sources.resize(n_occurrences);
        state.expansion_visiting.assign(n_occurrences, 0);
        state.expansion_has_memo.assign(n_occurrences, 0);
        state.expansion_memo.resize(n_occurrences);

        state.constrained_keys.reserve(n_occurrences);
        state.true_key_to_global.reserve(n_occurrences);
        state.constrained_key_to_global.reserve(n_occurrences);
        state.constrained_key_to_pairs.reserve(n_occurrences);
        state.occurrence_ids_by_key.reserve(n_occurrences);
        state.constrained_key_expansion_visiting.reserve(64U);

        std::size_t keys_reused = 0;
        std::size_t keys_rebuilt = 0;
        for (int id = 0; id < static_cast<int>(ctx.local_occurrences.size()); ++id)
        {
            const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(id)];
            const bool cache_hit =
                ctx.space.has_cached_dof_entity_key_2d(
                    ref.cell_id,
                    ref.local_index);
            const auto& debug =
                ctx.space.cached_dof_entity_key_2d(
                    ref.cell_id,
                    ref.local_index);
            if (cache_hit)
                ++keys_reused;
            else
                ++keys_rebuilt;
            state.entity_keys[static_cast<std::size_t>(id)] = debug.key;

            if (debug.spatial_boundary_eliminated &&
                !ctx.is_eliminated[static_cast<std::size_t>(id)])
            {
                throw std::runtime_error(
                    "2D DoF canonical key setup: boundary mismatch for " +
                    local_ref_summary_2d(ctx, ref));
            }
        }
        ctx.space.record_timing_metric(
            "fespace.rebuild.incremental.dof_keys_reused.count",
            static_cast<double>(keys_reused));
        ctx.space.record_timing_metric(
            "fespace.rebuild.incremental.dof_keys_rebuilt.count",
            static_cast<double>(keys_rebuilt));
        ctx.space.record_timing_metric(
            "dof_rebuild.entity_keys_reused",
            static_cast<double>(keys_reused));
        ctx.space.record_timing_metric(
            "dof_rebuild.entity_keys_rebuilt",
            static_cast<double>(keys_rebuilt));
    }

    template<typename Context>
    inline void add_constraint_source_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state,
        int cell_id,
        int local_index,
        ConstraintSource2D source)
    {
        const int id = linear_id(ctx, cell_id, local_index);
        if (ctx.is_eliminated[static_cast<std::size_t>(id)])
            return;

        auto& sources = state.constraint_sources[static_cast<std::size_t>(id)];
        if (std::find(sources.begin(), sources.end(), source) == sources.end())
            sources.push_back(source);
    }

    template<typename Context>
    inline void collect_constraint_sources_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state)
    {
        if constexpr (Context::SpaceType::PolicyType::continuous_in_space)
        {
            for (int interface_id = 0;
                 interface_id < static_cast<int>(ctx.adjacency.spatial_interfaces.size());
                 ++interface_id)
            {
                const auto& iface =
                    ctx.adjacency.spatial_interfaces[static_cast<std::size_t>(interface_id)];
                if (!iface.is_hanging)
                    continue;

                const auto& slave_face_dofs =
                    Context::ElemTables::spatial_face_dofs(iface.slave_face);
                for (int k = 0; k < Context::dofs_per_spatial_face; ++k)
                {
                    add_constraint_source_2d(
                        ctx,
                        state,
                        iface.slave_cell,
                        slave_face_dofs[static_cast<std::size_t>(k)],
                        ConstraintSource2D{true, interface_id});
                }
            }
        }

        if constexpr (Context::SpaceType::PolicyType::continuous_in_time)
        {
            for (int interface_id = 0;
                 interface_id < static_cast<int>(ctx.adjacency.temporal_interfaces.size());
                 ++interface_id)
            {
                const auto& iface =
                    ctx.adjacency.temporal_interfaces[static_cast<std::size_t>(interface_id)];
                if (!iface.is_hanging)
                    continue;

                const auto& slave_face_dofs =
                    Context::ElemTables::temporal_face_dofs(iface.slave_face);
                for (int k = 0; k < Context::dofs_per_temporal_face; ++k)
                {
                    add_constraint_source_2d(
                        ctx,
                        state,
                        iface.slave_cell,
                        slave_face_dofs[static_cast<std::size_t>(k)],
                        ConstraintSource2D{false, interface_id});
                }
            }
        }
    }

    template<typename Context>
    inline void assign_true_dofs_by_canonical_key_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state)
    {
        state.true_key_to_global.clear();
        state.true_key_to_global.reserve(ctx.local_occurrences.size());

        std::vector<finite_element::fespace::DofEntityKey2D> true_keys;
        true_keys.reserve(ctx.local_occurrences.size());

        for (int id = 0; id < static_cast<int>(ctx.local_occurrences.size()); ++id)
        {
            if (ctx.is_eliminated[static_cast<std::size_t>(id)])
                continue;
            if (!state.constraint_sources[static_cast<std::size_t>(id)].empty())
                continue;

            const auto& key = state.entity_keys[static_cast<std::size_t>(id)];
            if (state.constrained_keys.find(key) !=
                state.constrained_keys.end())
            {
                continue;
            }

            if (state.true_key_to_global.emplace(key, -1).second)
                true_keys.push_back(key);
        }

        std::sort(true_keys.begin(), true_keys.end());
        for (const auto& key : true_keys)
        {
            const int global_id =
                ctx.dof_handler.add_dof(typename Context::DoFType{});
            state.true_key_to_global[key] = global_id;
        }

        for (int id = 0; id < static_cast<int>(ctx.local_occurrences.size()); ++id)
        {
            if (ctx.is_eliminated[static_cast<std::size_t>(id)])
                continue;
            if (!state.constraint_sources[static_cast<std::size_t>(id)].empty())
                continue;

            const auto& key = state.entity_keys[static_cast<std::size_t>(id)];
            if (state.constrained_keys.find(key) !=
                state.constrained_keys.end())
            {
                continue;
            }

            const auto it = state.true_key_to_global.find(key);
            if (it == state.true_key_to_global.end())
            {
                const auto& ref =
                    ctx.local_occurrences[static_cast<std::size_t>(id)];
                throw std::runtime_error(
                    "DoF distribution failed: sorted true 2D canonical key "
                    "was not assigned for " +
                    local_ref_summary_2d(ctx, ref) +
                    " key={" + canonical_key_summary_2d<Context>(key) + "}");
            }

            const auto& ref =
                ctx.local_occurrences[static_cast<std::size_t>(id)];
            ctx.dof_handler.set_cell_dof(
                ref.cell_id,
                ref.local_index,
                it->second);
        }
    }

    template<typename Context>
    [[nodiscard]] inline std::vector<std::pair<int, double>>
    sorted_pairs_from_accumulated_vector_2d(
        std::vector<std::pair<int, double>> weights)
    {
        std::sort(
            weights.begin(),
            weights.end(),
            [](const auto& a, const auto& b)
            {
                return a.first < b.first;
            });

        std::vector<std::pair<int, double>> pairs;
        pairs.reserve(weights.size());
        for (const auto& [gid, weight] : weights)
        {
            if (pairs.empty() || pairs.back().first != gid)
            {
                pairs.emplace_back(gid, weight);
            }
            else
            {
                pairs.back().second += weight;
            }
        }

        pairs.erase(
            std::remove_if(
                pairs.begin(),
                pairs.end(),
                [](const auto& pair)
                {
                    return std::abs(pair.second) <= Context::constraint_tol;
                }),
            pairs.end());
        return pairs;
    }

    template<typename Context>
    [[nodiscard]] inline bool same_pair_list_2d(
        const std::vector<std::pair<int, double>>& a,
        const std::vector<std::pair<int, double>>& b)
    {
        if (a.size() != b.size())
            return false;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            if (a[i].first != b[i].first)
                return false;
            if (std::abs(a[i].second - b[i].second) >
                Context::constraint_tol)
            {
                return false;
            }
        }
        return true;
    }

    template<typename Context>
    [[nodiscard]] inline double cell_measure_2d(
        const Context& ctx,
        int cell_id)
    {
        const auto& cell = ctx.mesh.cell(cell_id);
        const double t0 =
            ctx.mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
        const double t1 =
            ctx.mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];

        const auto& a =
            ctx.mesh.spatial_vertices()[cell.spatial_vertex_ids[0]];
        const auto& b =
            ctx.mesh.spatial_vertices()[cell.spatial_vertex_ids[1]];
        const auto& c =
            ctx.mesh.spatial_vertices()[cell.spatial_vertex_ids[2]];
        const double area_twice =
            (b[0] - a[0]) * (c[1] - a[1]) -
            (b[1] - a[1]) * (c[0] - a[0]);
        return 0.5 * std::abs(area_twice) * std::abs(t1 - t0);
    }

    template<typename Context>
    [[nodiscard]] inline double constraint_source_master_measure_2d(
        const Context& ctx,
        const ConstraintSource2D& source)
    {
        if (source.is_spatial)
        {
            const auto& iface =
                ctx.adjacency.spatial_interfaces[
                    static_cast<std::size_t>(source.interface_id)];
            return cell_measure_2d(ctx, iface.master_cell);
        }

        const auto& iface =
            ctx.adjacency.temporal_interfaces[
                static_cast<std::size_t>(source.interface_id)];
        return cell_measure_2d(ctx, iface.master_cell);
    }

    template<typename Context>
    [[nodiscard]] inline double constraint_source_master_time_length_2d(
        const Context& ctx,
        const ConstraintSource2D& source)
    {
        const int master_cell =
            source.is_spatial
                ? ctx.adjacency.spatial_interfaces[
                      static_cast<std::size_t>(source.interface_id)]
                      .master_cell
                : ctx.adjacency.temporal_interfaces[
                      static_cast<std::size_t>(source.interface_id)]
                      .master_cell;
        const auto& cell = ctx.mesh.cell(master_cell);
        const double t0 =
            ctx.mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
        const double t1 =
            ctx.mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];
        return std::abs(t1 - t0);
    }

    template<typename Context>
    [[nodiscard]] inline std::vector<std::pair<int, double>>
    raw_master_occurrences_for_source_2d(
        Context& ctx,
        const typename Context::LocalDoFRef& ref,
        const ConstraintSource2D& source)
    {
        std::vector<std::pair<int, double>> out;
        const auto p_slave =
            finite_element::fespace::physical_dof_coord(
                ctx.space,
                ref.cell_id,
                ref.local_index);

        if (source.is_spatial)
        {
            const auto& iface =
                ctx.adjacency.spatial_interfaces[
                    static_cast<std::size_t>(source.interface_id)];
            const int master_cell = iface.master_cell;
            const int master_face = iface.master_face;

            const auto xi_eta =
                reference_spatial_coord_2d(ctx, master_cell, p_slave);
            const auto spatial_values =
                finite_element::basis::polynomials::TriangularLagrangeBasis<
                    Context::FETraits::p_space_v,
                    typename Context::SpatialNodes>::eval_all(xi_eta);

            const auto& master = ctx.mesh.cell(master_cell);
            const double tm0 =
                ctx.mesh.temporal_vertices()[static_cast<std::size_t>(
                    master.temporal_vertex_ids[0])][0];
            const double tm1 =
                ctx.mesh.temporal_vertices()[static_cast<std::size_t>(
                    master.temporal_vertex_ids[1])][0];
            const double denom = tm1 - tm0;
            if (std::abs(denom) < 1.0e-15)
            {
                throw std::runtime_error(
                    "DoF distribution failed: degenerate master temporal "
                    "interval in 2D spatial constraint.");
            }

            const double t_hat = (p_slave[2] - tm0) / denom;
            const auto temporal_values =
                finite_element::basis::polynomials::SegmentLagrangeBasis<
                    Context::FETraits::p_time_v,
                    typename Context::TemporalNodes>::eval_all(t_hat);
            const auto& master_face_dofs =
                Context::ElemTables::spatial_face_dofs(master_face);

            for (int k = 0; k < Context::dofs_per_spatial_face; ++k)
            {
                const int local = master_face_dofs[static_cast<std::size_t>(k)];
                const int spatial_node =
                    Context::ElemTables::spatial_node_id(local);
                const int temporal_node =
                    Context::ElemTables::temporal_node_id(local);
                const double weight =
                    spatial_values[static_cast<std::size_t>(spatial_node)] *
                    temporal_values[static_cast<std::size_t>(temporal_node)];
                if (std::abs(weight) <= Context::factor_tol)
                    continue;

                out.emplace_back(
                    linear_id(ctx, master_cell, local),
                    weight);
            }
        }
        else
        {
            const auto& iface =
                ctx.adjacency.temporal_interfaces[
                    static_cast<std::size_t>(source.interface_id)];
            const int master_cell = iface.master_cell;
            const int master_face = iface.master_face;

            const auto xi_eta =
                reference_spatial_coord_2d(ctx, master_cell, p_slave);
            const auto spatial_values =
                finite_element::basis::polynomials::TriangularLagrangeBasis<
                    Context::FETraits::p_space_v,
                    typename Context::SpatialNodes>::eval_all(xi_eta);
            const auto& master_face_dofs =
                Context::ElemTables::temporal_face_dofs(master_face);

            for (int k = 0; k < Context::dofs_per_temporal_face; ++k)
            {
                const int local = master_face_dofs[static_cast<std::size_t>(k)];
                const int spatial_node =
                    Context::ElemTables::spatial_node_id(local);
                const double weight =
                    spatial_values[static_cast<std::size_t>(spatial_node)];
                if (std::abs(weight) <= Context::factor_tol)
                    continue;

                out.emplace_back(
                    linear_id(ctx, master_cell, local),
                    weight);
            }
        }

        return out;
    }

    template<typename Context>
    inline void append_raw_route_summary_2d(
        std::ostringstream& out,
        Context& ctx,
        const CanonicalDistributionState2D<Context>& state,
        int id,
        const ConstraintSource2D& source)
    {
        const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(id)];
        const auto raw = raw_master_occurrences_for_source_2d(ctx, ref, source);
        out << '[';
        for (std::size_t i = 0; i < raw.size(); ++i)
        {
            if (i > 0)
                out << ',';
            const auto [master_id, weight] = raw[i];
            out << "{id=" << master_id << ",w=" << weight;
            if (master_id >= 0 &&
                static_cast<std::size_t>(master_id) <
                    ctx.local_occurrences.size())
            {
                const auto& master_ref =
                    ctx.local_occurrences[static_cast<std::size_t>(
                        master_id)];
                out << ',' << local_ref_summary_2d(ctx, master_ref)
                    << ",key={"
                    << canonical_key_summary_2d<Context>(
                        state.entity_keys[static_cast<std::size_t>(
                            master_id)])
                    << '}';
            }
            out << '}';
        }
        out << ']';
    }

    template<typename Context>
    [[nodiscard]] inline bool constraint_source_is_identity_on_key_2d(
        Context& ctx,
        const CanonicalDistributionState2D<Context>& state,
        int id,
        const ConstraintSource2D& source)
    {
        const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(id)];
        const auto raw = raw_master_occurrences_for_source_2d(ctx, ref, source);

        double same_key_weight = 0.0;
        bool saw_same_key = false;
        for (const auto& [master_id, weight] : raw)
        {
            if (std::abs(weight) <= Context::constraint_tol)
                continue;
            if (master_id < 0 ||
                static_cast<std::size_t>(master_id) >= state.entity_keys.size())
            {
                return false;
            }

            if (state.entity_keys[static_cast<std::size_t>(master_id)] !=
                state.entity_keys[static_cast<std::size_t>(id)])
            {
                return false;
            }

            same_key_weight += weight;
            saw_same_key = true;
        }

        return saw_same_key &&
               std::abs(same_key_weight - 1.0) <= Context::constraint_tol;
    }

    template<typename Context>
    inline void remove_identity_constraint_sources_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state)
    {
        for (int id = 0;
             id < static_cast<int>(state.constraint_sources.size());
             ++id)
        {
            auto& sources =
                state.constraint_sources[static_cast<std::size_t>(id)];
            sources.erase(
                std::remove_if(
                    sources.begin(),
                    sources.end(),
                    [&](const ConstraintSource2D& source)
                    {
                        return constraint_source_is_identity_on_key_2d(
                            ctx,
                            state,
                            id,
                            source);
                    }),
                sources.end());
        }
    }

    template<typename Context>
    inline void propagate_constraint_sources_by_canonical_key_2d(
        CanonicalDistributionState2D<Context>& state)
    {
        std::unordered_map<
            finite_element::fespace::DofEntityKey2D,
            std::vector<ConstraintSource2D>,
            finite_element::fespace::DofEntityKey2DHash>
            sources_by_key;
        sources_by_key.reserve(state.entity_keys.size());

        for (int id = 0;
             id < static_cast<int>(state.constraint_sources.size());
             ++id)
        {
            auto& key_sources =
                sources_by_key[
                    state.entity_keys[static_cast<std::size_t>(id)]];
            for (const auto& source :
                 state.constraint_sources[static_cast<std::size_t>(id)])
            {
                if (std::find(
                        key_sources.begin(),
                        key_sources.end(),
                        source) == key_sources.end())
                {
                    key_sources.push_back(source);
                }
            }
        }

        for (int id = 0;
             id < static_cast<int>(state.constraint_sources.size());
             ++id)
        {
            const auto it =
                sources_by_key.find(
                    state.entity_keys[static_cast<std::size_t>(id)]);
            if (it == sources_by_key.end())
                continue;
            state.constraint_sources[static_cast<std::size_t>(id)] =
                it->second;
        }
    }

    template<typename Context>
    inline void refresh_constrained_keys_and_occurrence_index_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state)
    {
        state.constrained_keys.clear();
        state.occurrence_ids_by_key.clear();
        state.constrained_keys.reserve(state.constraint_sources.size());
        state.occurrence_ids_by_key.reserve(state.constraint_sources.size());

        std::size_t indexed_occurrences = 0;
        std::size_t constrained_occurrences = 0;
        for (int id = 0;
             id < static_cast<int>(state.constraint_sources.size());
             ++id)
        {
            if (ctx.is_eliminated[static_cast<std::size_t>(id)])
                continue;

            state.occurrence_ids_by_key[
                state.entity_keys[static_cast<std::size_t>(id)]]
                .push_back(id);
            ++indexed_occurrences;

            if (!state.constraint_sources[static_cast<std::size_t>(id)].empty())
            {
                state.constrained_keys.insert(
                    state.entity_keys[static_cast<std::size_t>(id)]);
                ++constrained_occurrences;
            }
        }

        ctx.space.record_timing_metric(
            "dof_rebuild.occurrence_indexed",
            static_cast<double>(indexed_occurrences));
        ctx.space.record_timing_metric(
            "dof_rebuild.constrained_occurrences_indexed",
            static_cast<double>(constrained_occurrences));
        ctx.space.record_timing_metric(
            "dof_rebuild.constrained_keys_indexed",
            static_cast<double>(state.constrained_keys.size()));
        ctx.space.record_timing_metric(
            "dof_rebuild.canonical_occurrence_index_enabled",
            1.0);
    }

    template<typename Context>
    [[nodiscard]] inline std::vector<std::pair<int, double>>
    expand_occurrence_to_true_dofs_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state,
        int id);

    template<typename Context>
    [[nodiscard]] inline std::vector<std::pair<int, double>>
    expand_constrained_key_to_true_dofs_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state,
        const finite_element::fespace::DofEntityKey2D& key);

    template<typename Context>
    [[nodiscard]] inline std::vector<std::pair<int, double>>
    expand_through_constraint_source_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state,
        int id,
        const ConstraintSource2D& source)
    {
        const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(id)];
        const auto raw_masters =
            raw_master_occurrences_for_source_2d(ctx, ref, source);

        std::vector<std::pair<int, double>> weights;
        for (const auto& [master_id, weight] : raw_masters)
        {
            const auto master_expansion =
                expand_occurrence_to_true_dofs_2d(ctx, state, master_id);
            weights.reserve(weights.size() + master_expansion.size());
            for (const auto& [gid, master_weight] : master_expansion)
                weights.emplace_back(gid, weight * master_weight);
        }

        return sorted_pairs_from_accumulated_vector_2d<Context>(
            std::move(weights));
    }

    template<typename Context>
    [[nodiscard]] inline std::vector<std::pair<int, double>>
    expand_occurrence_to_true_dofs_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state,
        int id)
    {
        if (id < 0 || static_cast<std::size_t>(id) >= ctx.local_occurrences.size())
        {
            throw std::runtime_error(
                "DoF distribution failed: invalid local occurrence id in "
                "2D constraint expansion.");
        }

        if (ctx.is_eliminated[static_cast<std::size_t>(id)])
            return {};

        if (state.expansion_has_memo[static_cast<std::size_t>(id)])
            return state.expansion_memo[static_cast<std::size_t>(id)];

        const auto& key = state.entity_keys[static_cast<std::size_t>(id)];
        const auto& sources = state.constraint_sources[static_cast<std::size_t>(id)];

        if (state.constrained_keys.find(key) !=
                state.constrained_keys.end() &&
            state.constrained_key_expansion_visiting.find(key) ==
                state.constrained_key_expansion_visiting.end())
        {
            state.expansion_memo[static_cast<std::size_t>(id)] =
                expand_constrained_key_to_true_dofs_2d(ctx, state, key);
            state.expansion_has_memo[static_cast<std::size_t>(id)] = 1;
            return state.expansion_memo[static_cast<std::size_t>(id)];
        }

        if (sources.empty())
        {
            const auto it = state.true_key_to_global.find(key);
            if (it != state.true_key_to_global.end())
            {
                state.expansion_memo[static_cast<std::size_t>(id)] = {
                    {it->second, 1.0}};
                state.expansion_has_memo[static_cast<std::size_t>(id)] = 1;
                return state.expansion_memo[static_cast<std::size_t>(id)];
            }

            if (state.constrained_keys.find(key) !=
                state.constrained_keys.end())
            {
                state.expansion_memo[static_cast<std::size_t>(id)] =
                    expand_constrained_key_to_true_dofs_2d(ctx, state, key);
                state.expansion_has_memo[static_cast<std::size_t>(id)] = 1;
                return state.expansion_memo[static_cast<std::size_t>(id)];
            }

            {
                const auto& ref =
                    ctx.local_occurrences[static_cast<std::size_t>(id)];
                throw std::runtime_error(
                    "DoF distribution failed: unconstrained 2D canonical "
                    "entity has no true DoF for " +
                    local_ref_summary_2d(ctx, ref) +
                    " key={" + canonical_key_summary_2d<Context>(key) + "}");
            }
        }

        if (state.expansion_visiting[static_cast<std::size_t>(id)])
        {
            const auto it = state.true_key_to_global.find(key);
            if (it != state.true_key_to_global.end())
                return {{it->second, 1.0}};

            const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(id)];
            throw CyclicCanonicalConstraintRoute2D(
                "DoF distribution failed: cyclic 2D hanging constraint at " +
                local_ref_summary_2d(ctx, ref) +
                " key={" + canonical_key_summary_2d<Context>(key) + "}");
        }

        const ScopedOccurrenceExpansionVisit2D occurrence_visit(
            state.expansion_visiting,
            id);

        std::vector<std::pair<int, double>> reference_pairs;
        bool have_reference = false;
        int reference_source_index = -1;
        double reference_measure = 0.0;
        double reference_time_length = 0.0;
        for (const auto& source : sources)
        {
            const int source_index =
                static_cast<int>(&source - sources.data());
            const double source_measure =
                constraint_source_master_measure_2d(ctx, source);
            const double source_time_length =
                constraint_source_master_time_length_2d(ctx, source);
            auto candidate =
                expand_through_constraint_source_2d(ctx, state, id, source);
            if (!have_reference)
            {
                reference_pairs = std::move(candidate);
                have_reference = true;
                reference_source_index = source_index;
                reference_measure = source_measure;
                reference_time_length = source_time_length;
                continue;
            }

            constexpr double measure_tol = 1.0e-12;
            if (source_time_length >
                reference_time_length * (1.0 + measure_tol))
            {
                reference_pairs = std::move(candidate);
                reference_source_index = source_index;
                reference_measure = source_measure;
                reference_time_length = source_time_length;
                continue;
            }

            if (source_time_length <
                reference_time_length * (1.0 - measure_tol))
            {
                continue;
            }

            if (source_measure > reference_measure * (1.0 + measure_tol))
            {
                reference_pairs = std::move(candidate);
                reference_source_index = source_index;
                reference_measure = source_measure;
                reference_time_length = source_time_length;
                continue;
            }

            if (source_measure < reference_measure * (1.0 - measure_tol))
                continue;

            if (!same_pair_list_2d<Context>(reference_pairs, candidate))
            {
                const auto& ref =
                    ctx.local_occurrences[static_cast<std::size_t>(id)];
                std::ostringstream message;
                message
                    << "DoF distribution failed: multiple 2D hanging "
                    << "constraint routes disagree for "
                    << local_ref_summary_2d(ctx, ref)
                    << " key={" << canonical_key_summary_2d<Context>(key)
                    << "} reference_source=";
                append_constraint_source_summary_2d(
                    message,
                    ctx,
                    sources[static_cast<std::size_t>(
                        reference_source_index)]);
                message << " candidate_source=";
                append_constraint_source_summary_2d(message, ctx, source);
                message << " first=";
                append_constraint_pairs_2d<Context>(message, reference_pairs);
                message << " candidate=";
                append_constraint_pairs_2d<Context>(message, candidate);
                throw std::runtime_error(message.str());
            }
        }

        if (!have_reference)
        {
            const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(id)];
            throw std::runtime_error(
                "DoF distribution failed: all 2D hanging constraint routes "
                "were cyclic for " +
                local_ref_summary_2d(ctx, ref) +
                " key={" + canonical_key_summary_2d<Context>(key) + "}");
        }

        state.expansion_memo[static_cast<std::size_t>(id)] =
            std::move(reference_pairs);
        state.expansion_has_memo[static_cast<std::size_t>(id)] = 1;
        return state.expansion_memo[static_cast<std::size_t>(id)];
    }

    template<typename Context>
    [[nodiscard]] inline std::vector<std::pair<int, double>>
    expand_constrained_key_to_true_dofs_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state,
        const finite_element::fespace::DofEntityKey2D& key)
    {
        const auto memo_it = state.constrained_key_to_pairs.find(key);
        if (memo_it != state.constrained_key_to_pairs.end())
            return memo_it->second;

        if (state.constrained_key_expansion_visiting.find(key) !=
            state.constrained_key_expansion_visiting.end())
        {
            throw CyclicCanonicalConstraintRoute2D(
                "DoF distribution failed: cyclic 2D canonical hanging "
                "constraint key={" +
                canonical_key_summary_2d<Context>(key) + "}");
        }

        const ScopedCanonicalKeyExpansionVisit2D key_visit(
            state.constrained_key_expansion_visiting,
            key);

        std::vector<std::pair<int, double>> reference_pairs;
        bool have_reference = false;
        int reference_id = -1;
        int reference_source_index = -1;
        double reference_measure = 0.0;
        double reference_time_length = 0.0;

        const auto occurrence_it = state.occurrence_ids_by_key.find(key);
        if (occurrence_it == state.occurrence_ids_by_key.end())
        {
            throw std::runtime_error(
                "DoF distribution failed: constrained 2D canonical key has "
                "no indexed local occurrence key={" +
                canonical_key_summary_2d<Context>(key) + "}");
        }

        for (const int id : occurrence_it->second)
        {
            if (ctx.is_eliminated[static_cast<std::size_t>(id)])
                continue;

            const auto& sources =
                state.constraint_sources[static_cast<std::size_t>(id)];
            for (int source_index = 0;
                 source_index < static_cast<int>(sources.size());
                 ++source_index)
            {
                const double source_measure =
                    constraint_source_master_measure_2d(
                        ctx,
                        sources[static_cast<std::size_t>(source_index)]);
                const double source_time_length =
                    constraint_source_master_time_length_2d(
                        ctx,
                        sources[static_cast<std::size_t>(source_index)]);
                auto candidate =
                    expand_through_constraint_source_2d(
                        ctx,
                        state,
                        id,
                        sources[static_cast<std::size_t>(source_index)]);

                if (!have_reference)
                {
                    reference_pairs = std::move(candidate);
                    reference_id = id;
                    reference_source_index = source_index;
                    reference_measure = source_measure;
                    reference_time_length = source_time_length;
                    have_reference = true;
                    continue;
                }

                constexpr double measure_tol = 1.0e-12;
                if (source_time_length >
                    reference_time_length * (1.0 + measure_tol))
                {
                    reference_pairs = std::move(candidate);
                    reference_id = id;
                    reference_source_index = source_index;
                    reference_measure = source_measure;
                    reference_time_length = source_time_length;
                    continue;
                }

                if (source_time_length <
                    reference_time_length * (1.0 - measure_tol))
                {
                    continue;
                }

                if (source_measure > reference_measure * (1.0 + measure_tol))
                {
                    reference_pairs = std::move(candidate);
                    reference_id = id;
                    reference_source_index = source_index;
                    reference_measure = source_measure;
                    reference_time_length = source_time_length;
                    continue;
                }

                if (source_measure < reference_measure * (1.0 - measure_tol))
                    continue;

                if (!same_pair_list_2d<Context>(reference_pairs, candidate))
                {
                    const auto& ref =
                        ctx.local_occurrences[static_cast<std::size_t>(id)];
                    const auto& reference_ref =
                        ctx.local_occurrences[static_cast<std::size_t>(
                            reference_id)];
                    std::ostringstream message;
                    message
                        << "DoF distribution failed: same 2D canonical "
                        << "constrained entity has inconsistent direct "
                        << "hanging routes for "
                        << local_ref_summary_2d(ctx, ref)
                        << " key={" << canonical_key_summary_2d<Context>(key)
                        << "} reference="
                        << local_ref_summary_2d(ctx, reference_ref)
                        << " reference_source=";
                    append_constraint_source_summary_2d(
                        message,
                        ctx,
                        state.constraint_sources[
                            static_cast<std::size_t>(reference_id)]
                            [static_cast<std::size_t>(
                                reference_source_index)]);
                    message << " candidate_source=";
                    append_constraint_source_summary_2d(
                        message,
                        ctx,
                        sources[static_cast<std::size_t>(source_index)]);
                    message << " first=";
                    append_constraint_pairs_2d<Context>(
                        message,
                        reference_pairs);
                    message << " candidate=";
                    append_constraint_pairs_2d<Context>(message, candidate);
                    message << " reference_raw=";
                    append_raw_route_summary_2d(
                        message,
                        ctx,
                        state,
                        reference_id,
                        state.constraint_sources[
                            static_cast<std::size_t>(reference_id)]
                            [static_cast<std::size_t>(
                                reference_source_index)]);
                    message << " candidate_raw=";
                    append_raw_route_summary_2d(
                        message,
                        ctx,
                        state,
                        id,
                        sources[static_cast<std::size_t>(source_index)]);
                    throw std::runtime_error(message.str());
                }
            }
        }

        if (!have_reference)
        {
            throw std::runtime_error(
                "DoF distribution failed: constrained 2D canonical key has "
                "no direct hanging source key={" +
                canonical_key_summary_2d<Context>(key) + "}");
        }

        const auto [it, inserted] =
            state.constrained_key_to_pairs.emplace(key, reference_pairs);
        static_cast<void>(inserted);
        return it->second;
    }

    template<typename Context>
    inline void create_constrained_dofs_by_canonical_key_2d(
        Context& ctx,
        CanonicalDistributionState2D<Context>& state)
    {
        state.constrained_key_to_global.clear();
        state.constrained_key_to_global.reserve(state.constrained_keys.size());

        std::vector<finite_element::fespace::DofEntityKey2D> constrained_keys;
        constrained_keys.reserve(state.constrained_keys.size());
        for (const auto& key : state.constrained_keys)
            constrained_keys.push_back(key);
        std::sort(constrained_keys.begin(), constrained_keys.end());

        for (const auto& key : constrained_keys)
        {
            const auto pairs =
                expand_constrained_key_to_true_dofs_2d(ctx, state, key);

            if (pairs.empty())
                continue;

            typename Context::DoFType dof{};
            dof.is_constrained = true;
            dof.constraint_masters.reserve(pairs.size());
            dof.constraint_weights.reserve(pairs.size());
            for (const auto& [gid, weight] : pairs)
            {
                dof.constraint_masters.push_back(gid);
                dof.constraint_weights.push_back(weight);
            }

            const int constrained_gid = ctx.dof_handler.add_dof(dof);
            state.constrained_key_to_global.emplace(key, constrained_gid);
        }

        for (int id = 0;
             id < static_cast<int>(ctx.local_occurrences.size());
             ++id)
        {
            if (ctx.is_eliminated[static_cast<std::size_t>(id)])
                continue;

            const auto& ref =
                ctx.local_occurrences[static_cast<std::size_t>(id)];
            const auto& key = state.entity_keys[static_cast<std::size_t>(id)];
            if (state.constrained_keys.find(key) ==
                state.constrained_keys.end())
            {
                continue;
            }

            const auto pairs_it = state.constrained_key_to_pairs.find(key);
            if (pairs_it == state.constrained_key_to_pairs.end() ||
                pairs_it->second.empty())
            {
                ctx.is_eliminated[static_cast<std::size_t>(id)] = 1;
                ctx.dof_handler.set_cell_dof(ref.cell_id, ref.local_index, -1);
                continue;
            }

            const auto it_global = state.constrained_key_to_global.find(key);
            if (it_global == state.constrained_key_to_global.end())
            {
                throw std::runtime_error(
                    "DoF distribution failed: constrained 2D canonical key "
                    "has no global constrained DoF for " +
                    local_ref_summary_2d(ctx, ref) +
                    " key={" + canonical_key_summary_2d<Context>(key) + "}");
            }

            ctx.dof_handler.set_cell_dof(
                ref.cell_id,
                ref.local_index,
                it_global->second);
        }

        ctx.space.record_timing_metric(
            "dof_rebuild.constraints_reused",
            0.0);
        ctx.space.record_timing_metric(
            "dof_rebuild.constraints_rebuilt",
            static_cast<double>(state.constrained_key_to_global.size()));
    }

    template<typename Context>
    inline void record_canonical_dof_performance_counters_2d(
        Context& ctx,
        const CanonicalDistributionState2D<Context>& state)
    {
        using Key = finite_element::fespace::DofEntityKey2D;
        using KeyHash = finite_element::fespace::DofEntityKey2DHash;

        std::size_t boundary_eliminated = 0;
        for (const char eliminated : ctx.is_eliminated)
        {
            if (eliminated)
                ++boundary_eliminated;
        }

        std::size_t constraint_sources = 0;
        for (const auto& sources : state.constraint_sources)
            constraint_sources += sources.size();

        std::unordered_set<Key, KeyHash> unique_entity_keys;
        unique_entity_keys.reserve(state.entity_keys.size());
        for (const auto& key : state.entity_keys)
            unique_entity_keys.insert(key);

        ctx.space.record_timing_metric(
            "dof.local_occurrences.count",
            static_cast<double>(ctx.local_occurrences.size()));
        ctx.space.record_timing_metric(
            "dof.entity_keys.count",
            static_cast<double>(unique_entity_keys.size()));
        ctx.space.record_timing_metric(
            "dof.boundary_eliminated.count",
            static_cast<double>(boundary_eliminated));
        ctx.space.record_timing_metric(
            "dof.constraint_sources.count",
            static_cast<double>(constraint_sources));
        ctx.space.record_timing_metric(
            "dof.constrained_keys.count",
            static_cast<double>(state.constrained_keys.size()));
        ctx.space.record_timing_metric(
            "dof.true_dofs.count",
            static_cast<double>(ctx.dof_handler.n_true_dofs()));
        ctx.space.record_timing_metric(
            "dof.prolongation_nnz.count",
            static_cast<double>(ctx.dof_handler.prolongation_nonzeros()));
    }

    template<typename Context>
    inline void distribute_dofs_2d_canonical(Context& ctx)
    {
        CanonicalDistributionState2D<Context> state{};

        ctx.space.time_phase(
            "fespace.dof_distribution.canonical_key_construction",
            [&]()
            {
                initialize_canonical_entity_keys_2d(ctx, state);
            });
        ctx.space.time_phase(
            "fespace.dof_distribution.constraint_collection",
            [&]()
            {
                collect_constraint_sources_2d(ctx, state);
                remove_identity_constraint_sources_2d(ctx, state);
                refresh_constrained_keys_and_occurrence_index_2d(ctx, state);
            });
        ctx.space.time_phase(
            "fespace.dof_distribution.true_global_assignment",
            [&]()
            {
                assign_true_dofs_by_canonical_key_2d(ctx, state);
            });
        ctx.space.time_phase(
            "fespace.dof_distribution.recursive_constraint_expansion",
            [&]()
            {
                create_constrained_dofs_by_canonical_key_2d(ctx, state);
            });
        record_canonical_dof_performance_counters_2d(ctx, state);
    }

    template<typename FESpaceType>
    inline void distribute_dofs_2d(FESpaceType& space)
    {
        static_assert(FESpaceType::GT::dim_space_v == 2,
                      "distribute_dofs_2d requires dim_space_v == 2.");
        static_assert(FESpaceType::GT::dim_time_v == 1,
                      "distribute_dofs_2d requires dim_time_v == 1.");

        try
        {
            DistributionContext<FESpaceType> ctx(space);

            space.time_phase(
                "fespace.dof_distribution.cell_storage_init",
                [&]()
                {
                    initialize_cell_storage(ctx);
                });
            space.time_phase(
                "fespace.dof_distribution.local_occurrence_enumeration",
                [&]()
                {
                    enumerate_local_occurrences(ctx);
                });
            space.time_phase(
                "fespace.dof_distribution.boundary_elimination",
                [&]()
                {
                    mark_eliminated_occurrences_2d(ctx);
                });
            distribute_dofs_2d_canonical(ctx);
            space.time_phase(
                "fespace.dof_distribution.validation",
                [&]()
                {
                    validate_distribution(ctx);
                });
        }
        catch (const std::exception& e)
        {
            std::ostringstream message;
            message
                << e.what()
                << " [2D DoF distribution context:"
                << " p_space=" << FESpaceType::FETraitsType::p_space_v
                << " p_time=" << FESpaceType::FETraitsType::p_time_v
                << " policy="
                << (FESpaceType::PolicyType::continuous_in_time
                        ? "SpaceTimePolicy"
                        : "SpaceOnlyPolicy")
                << " active_cells=" << space.active_cells().size()
                << " mesh_cells=" << space.mesh_ref().n_cells()
                << " mesh_spatial_vertices="
                << space.mesh_ref().n_spatial_vertices()
                << " mesh_temporal_vertices="
                << space.mesh_ref().n_temporal_vertices();
            if (space.active_cells().size() <= 256U)
            {
                message << " active_cell_ids=[";
                for (std::size_t i = 0; i < space.active_cells().size(); ++i)
                {
                    if (i > 0)
                        message << ',';
                    message << space.active_cells()[i];
                }
                message << ']';
            }
            message << ']';
            throw std::runtime_error(message.str());
        }
    }
}

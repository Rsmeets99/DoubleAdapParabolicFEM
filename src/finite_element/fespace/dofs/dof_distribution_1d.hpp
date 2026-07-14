#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "../../basis/polynomials/segment_lagrange.hpp"
#include "../../../core/coord_key.hpp"
#include "physical_dof_coords.hpp"
#include "../detail/dsu.hpp"
#include "../detail/local_dof_ref.hpp"
#include "../detail/slave_constraint_source.hpp"

namespace finite_element::fespace::detail::dof_distribution_impl
{
    template<typename FESpaceType>
    struct DistributionContext
    {
        using SpaceType = FESpaceType;
        using DoFType = typename SpaceType::DoFType;
        using ElemTables = typename SpaceType::ElemTables;

        using MeshType = typename SpaceType::MeshType;
        using AdjacencyType = typename SpaceType::AdjacencyType;
        using DoFHandlerType = typename SpaceType::DoFHandlerType;

        using LocalDoFRef = finite_element::fespace::detail::LocalDoFRef;
        using LocalDoFRefHash = finite_element::fespace::detail::LocalDoFRefHash;
        using SlaveSource = finite_element::fespace::detail::SlaveConstraintSource;
        using DSU = finite_element::fespace::detail::DSU;

        using FETraits = typename SpaceType::FETraitsType;
        using SpatialNodes = typename FETraits::SpatialNodes;
        using TemporalNodes = typename FETraits::TemporalNodes;
        using CoordKeyType = core::CoordKey<SpaceType::GT::dim_v>;

        static constexpr int dofs_per_cell = ElemTables::dofs_per_cell;
        static constexpr int dofs_per_spatial_face = ElemTables::dofs_per_spatial_face;
        static constexpr int dofs_per_temporal_face = ElemTables::dofs_per_temporal_face;
        static constexpr int dim_v = SpaceType::GT::dim_v;
        static constexpr double factor_tol = 1e-14;
        static constexpr double constraint_tol = 1e-12;

        SpaceType& space;
        const MeshType& mesh;
        AdjacencyType& adjacency;
        DoFHandlerType& dof_handler;
        const std::vector<int>& active_cells;

        std::vector<LocalDoFRef> local_occurrences{};
        std::unordered_map<LocalDoFRef, int, LocalDoFRefHash> local_to_linear{};
        std::vector<char> is_eliminated{};
        std::vector<SlaveSource> slave_sources{};
        DSU dsu{0};
        std::vector<int> representatives{};
        std::unordered_map<int, int> root_to_global{};

        explicit DistributionContext(SpaceType& space_ref)
            : space(space_ref),
              mesh(space_ref.mesh_ref()),
              adjacency(space_ref.adjacency_ref()),
              dof_handler(space_ref.dof_handler_ref()),
              active_cells(space_ref.active_cells())
        {}
    };

    template<typename Context>
    [[nodiscard]] inline int linear_id(
        const Context& ctx,
        int cell_id,
        int local_index)
    {
        const auto it =
            ctx.local_to_linear.find(
                typename Context::LocalDoFRef{cell_id, local_index});
        if (it == ctx.local_to_linear.end())
        {
            std::string active;
            for (const int active_cell : ctx.active_cells)
            {
                if (!active.empty())
                    active += ',';
                active += std::to_string(active_cell);
            }
            throw std::runtime_error(
                "DoF distribution failed: local DoF reference is not active, cell=" +
                std::to_string(cell_id) +
                " local=" +
                std::to_string(local_index) +
                " active_cells=[" +
                active +
                "]");
        }
        return it->second;
    }

    template<typename Context>
    inline void initialize_cell_storage(Context& ctx)
    {
        ctx.dof_handler.clear();
        if (!ctx.active_cells.empty())
        {
            const auto max_cell_it =
                std::max_element(ctx.active_cells.begin(), ctx.active_cells.end());
            if (max_cell_it != ctx.active_cells.end() && *max_cell_it >= 0)
                ctx.dof_handler.reserve_cell_slots(
                    static_cast<std::size_t>(*max_cell_it) + 1U);
        }
        for (const int cell_id : ctx.active_cells)
            ctx.dof_handler.initialize_cell(cell_id);
    }

    template<typename Context>
    inline void enumerate_local_occurrences(Context& ctx)
    {
        ctx.local_occurrences.reserve(ctx.active_cells.size() * Context::dofs_per_cell);
        ctx.local_to_linear.reserve(ctx.active_cells.size() * Context::dofs_per_cell);

        for (const int cell_id : ctx.active_cells)
        {
            for (int local_index = 0; local_index < Context::dofs_per_cell; ++local_index)
            {
                const int id = static_cast<int>(ctx.local_occurrences.size());
                typename Context::LocalDoFRef ref{cell_id, local_index};
                ctx.local_occurrences.push_back(ref);
                ctx.local_to_linear.emplace(ref, id);
            }
        }

        ctx.is_eliminated.assign(ctx.local_occurrences.size(), 0);
        ctx.slave_sources.assign(ctx.local_occurrences.size(), typename Context::SlaveSource{});
        ctx.dsu = typename Context::DSU(static_cast<int>(ctx.local_occurrences.size()));
    }

    template<typename Context>
    inline void mark_eliminated_occurrences(Context& ctx)
    {
        for (int i = 0; i < static_cast<int>(ctx.local_occurrences.size()); ++i)
        {
            const auto& ref  = ctx.local_occurrences[static_cast<std::size_t>(i)];
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

            if (on_spatial_boundary)
                ctx.is_eliminated[static_cast<std::size_t>(i)] = 1;
        }
    }

    template<typename Context>
    inline void mark_slave_occurrence(
        Context& ctx,
        int cell_id,
        int local_index,
        bool is_spatial,
        int interface_id)
    {
        const int id = linear_id(ctx, cell_id, local_index);
        if (ctx.is_eliminated[static_cast<std::size_t>(id)])
            return;

        auto& src = ctx.slave_sources[static_cast<std::size_t>(id)];
        if (!src.is_slave)
        {
            src.is_slave = true;
            src.is_spatial = is_spatial;
            src.interface_id = interface_id;
        }
    }

    template<typename Context>
    inline void mark_slave_occurrences(Context& ctx)
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
                    mark_slave_occurrence(
                        ctx,
                        iface.slave_cell,
                        slave_face_dofs[static_cast<std::size_t>(k)],
                        true,
                        interface_id);
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
                    mark_slave_occurrence(
                        ctx,
                        iface.slave_cell,
                        slave_face_dofs[static_cast<std::size_t>(k)],
                        false,
                        interface_id);
                }
            }
        }
    }

    template<typename Context>
    inline void unite_conforming_pair(Context& ctx, int master_id, int slave_id)
    {
        if (ctx.is_eliminated[static_cast<std::size_t>(master_id)] ||
            ctx.is_eliminated[static_cast<std::size_t>(slave_id)])
        {
            return;
        }

        if (ctx.slave_sources[static_cast<std::size_t>(master_id)].is_slave ||
            ctx.slave_sources[static_cast<std::size_t>(slave_id)].is_slave)
        {
            return;
        }

        ctx.dsu.unite(master_id, slave_id);
    }

    template<typename Context>
    inline void build_conforming_unions(Context& ctx)
    {
        if constexpr (Context::SpaceType::PolicyType::continuous_in_space)
        {
            for (const auto& iface : ctx.adjacency.spatial_interfaces)
            {
                if (iface.is_boundary || iface.is_hanging)
                    continue;

                const auto& master_dofs =
                    Context::ElemTables::spatial_face_dofs(iface.master_face);
                const auto& slave_dofs =
                    Context::ElemTables::spatial_face_dofs(iface.slave_face);

                for (int k = 0; k < Context::dofs_per_spatial_face; ++k)
                {
                    const int lin_master =
                        linear_id(ctx, iface.master_cell, master_dofs[static_cast<std::size_t>(k)]);
                    const int lin_slave =
                        linear_id(ctx, iface.slave_cell, slave_dofs[static_cast<std::size_t>(k)]);

                    unite_conforming_pair(ctx, lin_master, lin_slave);
                }
            }
        }

        if constexpr (Context::SpaceType::PolicyType::continuous_in_time)
        {
            for (const auto& iface : ctx.adjacency.temporal_interfaces)
            {
                if (iface.is_boundary || iface.is_hanging)
                    continue;

                const auto& master_dofs =
                    Context::ElemTables::temporal_face_dofs(iface.master_face);
                const auto& slave_dofs =
                    Context::ElemTables::temporal_face_dofs(iface.slave_face);

                for (int k = 0; k < Context::dofs_per_temporal_face; ++k)
                {
                    const int lin_master =
                        linear_id(ctx, iface.master_cell, master_dofs[static_cast<std::size_t>(k)]);
                    const int lin_slave =
                        linear_id(ctx, iface.slave_cell, slave_dofs[static_cast<std::size_t>(k)]);

                    unite_conforming_pair(ctx, lin_master, lin_slave);
                }
            }
        }
    }

    template<typename Context>
    inline void unify_vertex_occurrences_by_coordinate(Context& ctx)
    {
        if constexpr (!Context::SpaceType::PolicyType::continuous_in_time)
        {
            return;
        }
        else
        {
            std::unordered_map<typename Context::CoordKeyType, int> vertex_owner;
            vertex_owner.reserve(ctx.local_occurrences.size());

            for (int i = 0; i < static_cast<int>(ctx.local_occurrences.size()); ++i)
            {
                if (ctx.is_eliminated[static_cast<std::size_t>(i)])
                    continue;
                if (ctx.slave_sources[static_cast<std::size_t>(i)].is_slave)
                    continue;

                const auto& ref  = ctx.local_occurrences[static_cast<std::size_t>(i)];
                const auto& meta = Context::ElemTables::meta(ref.local_index);
                if (!meta.is_vertex)
                    continue;

                const auto p = finite_element::fespace::physical_dof_coord(ctx.space, ref.cell_id, ref.local_index);
                const auto key = core::make_coord_key<Context::dim_v>(p);

                const auto it = vertex_owner.find(key);
                if (it == vertex_owner.end())
                    vertex_owner.emplace(key, i);
                else
                    ctx.dsu.unite(i, it->second);
            }
        }
    }

    template<typename Context>
    inline void compute_representatives(Context& ctx)
    {
        ctx.representatives.assign(ctx.local_occurrences.size(), -1);
        for (int i = 0; i < static_cast<int>(ctx.local_occurrences.size()); ++i)
            ctx.representatives[static_cast<std::size_t>(i)] = ctx.dsu.find(i);
    }

    template<typename Context>
    inline void assign_true_dofs(Context& ctx)
    {
        ctx.root_to_global.reserve(ctx.local_occurrences.size());

        for (int i = 0; i < static_cast<int>(ctx.local_occurrences.size()); ++i)
        {
            if (ctx.is_eliminated[static_cast<std::size_t>(i)])
                continue;
            if (ctx.slave_sources[static_cast<std::size_t>(i)].is_slave)
                continue;

            const int root = ctx.representatives[static_cast<std::size_t>(i)];
            auto it = ctx.root_to_global.find(root);

            int global_id = -1;
            if (it == ctx.root_to_global.end())
            {
                global_id = ctx.dof_handler.add_dof(typename Context::DoFType{});
                ctx.root_to_global.emplace(root, global_id);
            }
            else
            {
                global_id = it->second;
            }

            const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(i)];
            ctx.dof_handler.set_cell_dof(ref.cell_id, ref.local_index, global_id);
        }
    }

    template<typename Context>
    inline void accumulate_global_masters(
        Context& ctx,
        int cell_id,
        int local_index,
        double factor,
        std::unordered_map<int, double>& out_weights)
    {
        if (std::abs(factor) < Context::factor_tol)
            return;

        const int id = linear_id(ctx, cell_id, local_index);
        if (ctx.is_eliminated[static_cast<std::size_t>(id)])
            return;

        const auto& src = ctx.slave_sources[static_cast<std::size_t>(id)];
        if (!src.is_slave)
        {
            const int root = ctx.representatives[static_cast<std::size_t>(id)];
            const int gid = ctx.root_to_global.at(root);
            out_weights[gid] += factor;
            return;
        }

        if (src.is_spatial)
        {
            const auto& iface = ctx.adjacency.spatial_interfaces[static_cast<std::size_t>(src.interface_id)];
            const int master_cell = iface.master_cell;
            const int master_face = iface.master_face;

            const auto p_slave = finite_element::fespace::physical_dof_coord(ctx.space, cell_id, local_index);
            const double t_phys = p_slave[1];

            const auto& master = ctx.mesh.cell(master_cell);
            const double tm0 =
                ctx.mesh.temporal_vertices()[static_cast<std::size_t>(master.temporal_vertex_ids[0])][0];
            const double tm1 =
                ctx.mesh.temporal_vertices()[static_cast<std::size_t>(master.temporal_vertex_ids[1])][0];

            const double denom = tm1 - tm0;
            if (std::abs(denom) < 1e-15)
            {
                throw std::runtime_error(
                    "Degenerate master temporal interval in spatial hanging constraint.");
            }

            const double t_hat = (t_phys - tm0) / denom;
            const auto interp =
                finite_element::basis::polynomials::SegmentLagrangeBasis<
                    Context::FETraits::p_time_v,
                    typename Context::TemporalNodes
                >::eval_all(t_hat);
            const auto& master_face_dofs = Context::ElemTables::spatial_face_dofs(master_face);

            for (int j = 0; j < Context::dofs_per_spatial_face; ++j)
            {
                accumulate_global_masters(
                    ctx,
                    master_cell,
                    master_face_dofs[static_cast<std::size_t>(j)],
                    factor * interp[static_cast<std::size_t>(j)],
                    out_weights);
            }
        }
        else
        {
            const auto& iface = ctx.adjacency.temporal_interfaces[static_cast<std::size_t>(src.interface_id)];
            const int master_cell = iface.master_cell;
            const int master_face = iface.master_face;

            const auto p_slave = finite_element::fespace::physical_dof_coord(ctx.space, cell_id, local_index);
            const double x_phys = p_slave[0];

            const auto& master = ctx.mesh.cell(master_cell);
            const double xm0 =
                ctx.mesh.spatial_vertices()[static_cast<std::size_t>(master.spatial_vertex_ids[0])][0];
            const double xm1 =
                ctx.mesh.spatial_vertices()[static_cast<std::size_t>(master.spatial_vertex_ids[1])][0];

            const double denom = xm1 - xm0;
            if (std::abs(denom) < 1e-15)
            {
                throw std::runtime_error(
                    "Degenerate master spatial interval in temporal hanging constraint.");
            }

            const double x_hat = (x_phys - xm0) / denom;
            const auto interp =
                finite_element::basis::polynomials::SegmentLagrangeBasis<
                    Context::FETraits::p_space_v,
                    typename Context::SpatialNodes
                >::eval_all(x_hat);
            const auto& master_face_dofs = Context::ElemTables::temporal_face_dofs(master_face);

            for (int j = 0; j < Context::dofs_per_temporal_face; ++j)
            {
                accumulate_global_masters(
                    ctx,
                    master_cell,
                    master_face_dofs[static_cast<std::size_t>(j)],
                    factor * interp[static_cast<std::size_t>(j)],
                    out_weights);
            }
        }
    }

    template<typename Context>
    [[nodiscard]] inline std::vector<int> collect_sorted_slave_linear_ids(Context& ctx)
    {
        std::vector<int> slave_linear_ids;
        slave_linear_ids.reserve(ctx.local_occurrences.size());

        for (int i = 0; i < static_cast<int>(ctx.local_occurrences.size()); ++i)
        {
            if (!ctx.is_eliminated[static_cast<std::size_t>(i)] &&
                ctx.slave_sources[static_cast<std::size_t>(i)].is_slave)
            {
                slave_linear_ids.push_back(i);
            }
        }

        std::sort(
            slave_linear_ids.begin(),
            slave_linear_ids.end(),
            [&](int a, int b)
            {
                const auto& ra = ctx.local_occurrences[static_cast<std::size_t>(a)];
                const auto& rb = ctx.local_occurrences[static_cast<std::size_t>(b)];

                const int ga = ctx.mesh.cell(ra.cell_id).generation;
                const int gb = ctx.mesh.cell(rb.cell_id).generation;

                if (ga != gb)
                    return ga < gb;
                if (ra.cell_id != rb.cell_id)
                    return ra.cell_id < rb.cell_id;
                return ra.local_index < rb.local_index;
            });

        return slave_linear_ids;
    }

    template<typename Context>
    [[nodiscard]] inline bool same_constraint_data(
        const typename Context::DoFType& dof,
        const std::vector<std::pair<int, double>>& pairs)
    {
        if (!dof.is_constrained)
            return false;
        if (dof.constraint_masters.size() != pairs.size())
            return false;
        if (dof.constraint_weights.size() != pairs.size())
            return false;

        for (std::size_t i = 0; i < pairs.size(); ++i)
        {
            if (dof.constraint_masters[i] != pairs[i].first)
                return false;
            if (std::abs(dof.constraint_weights[i] - pairs[i].second) > Context::constraint_tol)
                return false;
        }

        return true;
    }

    template<typename Context>
    inline void create_constrained_dofs(Context& ctx)
    {
        const auto slave_linear_ids = collect_sorted_slave_linear_ids(ctx);

        std::unordered_map<typename Context::CoordKeyType, int> constrained_coord_to_gid;
        constrained_coord_to_gid.reserve(slave_linear_ids.size());

        for (const int id : slave_linear_ids)
        {
            const auto& ref = ctx.local_occurrences[static_cast<std::size_t>(id)];

            std::unordered_map<int, double> master_weight_map;
            accumulate_global_masters(ctx, ref.cell_id, ref.local_index, 1.0, master_weight_map);

            std::vector<std::pair<int, double>> pairs;
            pairs.reserve(master_weight_map.size());

            for (const auto& [gid, w] : master_weight_map)
            {
                if (std::abs(w) > Context::constraint_tol)
                    pairs.emplace_back(gid, w);
            }

            std::sort(
                pairs.begin(),
                pairs.end(),
                [](const auto& a, const auto& b)
                {
                    return a.first < b.first;
                });

            if (pairs.empty())
                continue;

            if (pairs.size() == 1 && std::abs(pairs[0].second - 1.0) < Context::constraint_tol)
            {
                ctx.dof_handler.set_cell_dof(ref.cell_id, ref.local_index, pairs[0].first);
                continue;
            }

            const auto p = finite_element::fespace::physical_dof_coord(ctx.space, ref.cell_id, ref.local_index);
            const auto key = core::make_coord_key<Context::dim_v>(p);

            const auto it_existing = constrained_coord_to_gid.find(key);
            if (it_existing != constrained_coord_to_gid.end())
            {
                const int existing_gid = it_existing->second;
                const auto& existing_dof = ctx.dof_handler.dof(existing_gid);

                if (!same_constraint_data<Context>(existing_dof, pairs))
                {
                    throw std::runtime_error(
                        "DoF distribution failed: same constrained coordinate has inconsistent masters/weights.");
                }

                ctx.dof_handler.set_cell_dof(ref.cell_id, ref.local_index, existing_gid);
                continue;
            }

            typename Context::DoFType dof{};
            dof.is_constrained = true;
            dof.constraint_masters.reserve(pairs.size());
            dof.constraint_weights.reserve(pairs.size());

            for (const auto& [gid, w] : pairs)
            {
                dof.constraint_masters.push_back(gid);
                dof.constraint_weights.push_back(w);
            }

            const int gid = ctx.dof_handler.add_dof(dof);
            constrained_coord_to_gid.emplace(key, gid);
            ctx.dof_handler.set_cell_dof(ref.cell_id, ref.local_index, gid);
        }
    }

    template<typename Context>
    inline void validate_distribution(const Context& ctx)
    {
        for (const int cell_id : ctx.active_cells)
        {
            const auto& cell_dofs = ctx.dof_handler.cell_dofs(cell_id);

            for (int local_index = 0; local_index < Context::dofs_per_cell; ++local_index)
            {
                const int id = linear_id(ctx, cell_id, local_index);

                if (ctx.is_eliminated[static_cast<std::size_t>(id)])
                {
                    if (cell_dofs[static_cast<std::size_t>(local_index)] != -1)
                    {
                        throw std::runtime_error(
                            "Dirichlet-eliminated DoF should remain -1.");
                    }
                }
                else if (cell_dofs[static_cast<std::size_t>(local_index)] < 0)
                {
                    throw std::runtime_error(
                        "DoF distribution failed: some non-eliminated local DoFs were left unassigned.");
                }
            }
        }
    }

    template<typename FESpaceType>
    inline void distribute_dofs_1d(FESpaceType& space)
    {
        DistributionContext<FESpaceType> ctx(space);

        initialize_cell_storage(ctx);
        enumerate_local_occurrences(ctx);
        mark_eliminated_occurrences(ctx);
        mark_slave_occurrences(ctx);
        build_conforming_unions(ctx);
        unify_vertex_occurrences_by_coordinate(ctx);
        compute_representatives(ctx);
        assign_true_dofs(ctx);
        create_constrained_dofs(ctx);
        validate_distribution(ctx);
    }
}

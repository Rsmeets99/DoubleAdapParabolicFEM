#pragma once

#include <array>

namespace finite_element::fespace
{
    template<typename FESpaceType>
    [[nodiscard]] inline auto physical_dof_coord_1d(
        const FESpaceType& space,
        int cell_id,
        int local_index)
    {
        using ElemTables = typename FESpaceType::ElemTables;
        using GT = typename FESpaceType::GT;

        static_assert(GT::dim_space_v == 1, "physical_dof_coord_1d requires dim_space_v == 1.");

        const auto& mesh = space.mesh_ref();
        const auto& cell = mesh.cell(cell_id);
        const auto& xi = ElemTables::coord(local_index);

        const double x0 = mesh.spatial_vertices()[cell.spatial_vertex_ids[0]][0];
        const double x1 = mesh.spatial_vertices()[cell.spatial_vertex_ids[1]][0];
        const double t0 = mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
        const double t1 = mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];

        return std::array<double, GT::dim_v>{
            x0 + (x1 - x0) * xi[0],
            t0 + (t1 - t0) * xi[1]
        };
    }

    template<typename FESpaceType>
    [[nodiscard]] inline auto physical_dof_coord_2d(
        const FESpaceType& space,
        int cell_id,
        int local_index)
    {
        using ElemTables = typename FESpaceType::ElemTables;
        using GT = typename FESpaceType::GT;

        static_assert(GT::dim_space_v == 2, "physical_dof_coord_2d requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1, "physical_dof_coord_2d requires dim_time_v == 1.");

        const auto& mesh = space.mesh_ref();
        const auto& cell = mesh.cell(cell_id);
        const auto& xi = ElemTables::coord(local_index);

        const auto& x0 = mesh.spatial_vertices()[cell.spatial_vertex_ids[0]];
        const auto& x1 = mesh.spatial_vertices()[cell.spatial_vertex_ids[1]];
        const auto& x2 = mesh.spatial_vertices()[cell.spatial_vertex_ids[2]];
        const double t0 = mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
        const double t1 = mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];

        return std::array<double, GT::dim_v>{
            x0[0] + (x1[0] - x0[0]) * xi[0] + (x2[0] - x0[0]) * xi[1],
            x0[1] + (x1[1] - x0[1]) * xi[0] + (x2[1] - x0[1]) * xi[1],
            t0 + (t1 - t0) * xi[2]
        };
    }

    template<typename FESpaceType>
    [[nodiscard]] inline auto physical_dof_coord(
        const FESpaceType& space,
        int cell_id,
        int local_index)
    {
        if constexpr (FESpaceType::GT::dim_space_v == 1)
        {
            return physical_dof_coord_1d(space, cell_id, local_index);
        }
        else if constexpr (FESpaceType::GT::dim_space_v == 2)
        {
            return physical_dof_coord_2d(space, cell_id, local_index);
        }
        else
        {
            static_assert(FESpaceType::GT::dim_space_v == 1 ||
                          FESpaceType::GT::dim_space_v == 2,
                          "physical_dof_coord is only implemented for 1+1D or 2+1D.");
        }
    }
}

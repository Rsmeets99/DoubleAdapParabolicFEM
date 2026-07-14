#pragma once

#include <array>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "mat_A_2d.hpp"

#include "linear_algebra/assembly/local_objects.hpp"

namespace finite_element::assembly::error_system
{
    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType>
    class LocalABElementCache2D
    {
    public:
        using FluxSpaceType = PatchFluxSpaceType;
        using ScalarSpaceType = PatchScalarSpaceType;
        using RTCellCache =
            finite_element::assembly::detail::
                LocalErrorRTCellQuadratureCache2D<
                    QSpace,
                    QTime,
                    FluxSpaceType>;

        struct CellData
        {
            int slab_id = -1;
            int slab_cell_id = -1;
            la::local::FixedLocalMatrix<
                FluxSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v> A{};
            la::local::FixedLocalMatrix<
                ScalarSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v> B{};

            CellData() = default;
        };

        LocalABElementCache2D() = default;

        template<class MFunction>
        LocalABElementCache2D(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces,
            const RTCellCache& rt_cell_cache,
            const MFunction& M)
        {
            prepare_from_spaces(flux_spaces, scalar_spaces);
            for (int request_id = 0;
                 request_id < n_build_requests();
                 ++request_id)
            {
                fill_build_request(
                    request_id,
                    flux_spaces,
                    scalar_spaces,
                    rt_cell_cache,
                    M);
            }
        }

        void prepare_from_spaces(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces)
        {
            if (flux_spaces.size() != scalar_spaces.size())
            {
                throw std::runtime_error(
                    "LocalABElementCache2D: flux/scalar space count mismatch.");
            }

            cells_.clear();
            cell_index_by_key_.clear();
            build_requests_.clear();
            requested_patch_cells_ = 0;
            const auto unique_slab_cell_count =
                count_unique_slab_cells_from_flux_spaces_(flux_spaces);
            cells_.reserve(unique_slab_cell_count);
            build_requests_.reserve(unique_slab_cell_count);

            for (std::size_t patch_id = 0;
                 patch_id < flux_spaces.size();
                 ++patch_id)
            {
                collect_patch_cells_(
                    static_cast<int>(patch_id),
                    flux_spaces[patch_id],
                    scalar_spaces[patch_id]);
            }
        }

        void prepare_from_patch_cells(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces,
            const std::vector<
                finite_element::assembly::detail::
                    LocalErrorPatchCellBuildRequest2D>& requests)
        {
            if (flux_spaces.size() != scalar_spaces.size())
            {
                throw std::runtime_error(
                    "LocalABElementCache2D: flux/scalar space count mismatch.");
            }

            cells_.clear();
            cell_index_by_key_.clear();
            build_requests_.clear();
            requested_patch_cells_ = 0;
            const auto unique_slab_cell_count =
                count_unique_slab_cells_from_requests_(
                    flux_spaces,
                    requests);
            cells_.reserve(unique_slab_cell_count);
            build_requests_.reserve(unique_slab_cell_count);

            for (const auto& request : requests)
            {
                if (request.patch_id < 0 ||
                    request.patch_id >= static_cast<int>(flux_spaces.size()))
                {
                    throw std::runtime_error(
                        "LocalABElementCache2D: patch id out of range.");
                }
                collect_patch_cell_(
                    request.patch_id,
                    flux_spaces[static_cast<std::size_t>(request.patch_id)],
                    scalar_spaces[static_cast<std::size_t>(request.patch_id)],
                    request.patch_cell_index);
            }
        }

        [[nodiscard]] int n_build_requests() const noexcept
        {
            return static_cast<int>(build_requests_.size());
        }

        [[nodiscard]] int build_request_slab_id(int request_id) const
        {
            check_build_request_index_(request_id);
            return build_requests_[static_cast<std::size_t>(request_id)].slab_id;
        }

        [[nodiscard]] int build_request_slab_cell_id(int request_id) const
        {
            check_build_request_index_(request_id);
            return build_requests_[static_cast<std::size_t>(request_id)].slab_cell_id;
        }

        template<class MFunction>
        void fill_build_request(
            int request_id,
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces,
            const RTCellCache& rt_cell_cache,
            const MFunction& M)
        {
            check_build_request_index_(request_id);
            const auto& request =
                build_requests_[static_cast<std::size_t>(request_id)];
            fill_cell_(
                flux_spaces[static_cast<std::size_t>(request.patch_id)],
                scalar_spaces[static_cast<std::size_t>(request.patch_id)],
                request.patch_cell_index,
                rt_cell_cache.cell(
                    request.slab_id,
                    request.slab_cell_id),
                M,
                cells_[static_cast<std::size_t>(request.cache_id)]);
        }

        [[nodiscard]] int requested_patch_cells() const noexcept
        {
            return requested_patch_cells_;
        }

        [[nodiscard]] int unique_slab_cells() const noexcept
        {
            return static_cast<int>(cells_.size());
        }

        [[nodiscard]] int duplicate_patch_cells() const noexcept
        {
            return requested_patch_cells_ - unique_slab_cells();
        }

        [[nodiscard]] const CellData& cell(
            int slab_id,
            int slab_cell_id) const
        {
            const auto it = cell_index_by_key_.find({slab_id, slab_cell_id});
            if (it == cell_index_by_key_.end())
            {
                throw std::runtime_error(
                    "LocalABElementCache2D: slab cell not cached.");
            }

            return cells_[static_cast<std::size_t>(it->second)];
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return cells_.capacity() * sizeof(CellData) +
                   build_requests_.capacity() * (5 * sizeof(int)) +
                   cell_index_by_key_.size() *
                       (sizeof(std::pair<const std::pair<int, int>, int>) +
                        3 * sizeof(void*));
        }

    private:
        struct BuildRequest
        {
            int cache_id = -1;
            int patch_id = -1;
            int patch_cell_index = -1;
            int slab_id = -1;
            int slab_cell_id = -1;
        };

        std::vector<CellData> cells_{};
        std::map<std::pair<int, int>, int> cell_index_by_key_{};
        std::vector<BuildRequest> build_requests_{};
        int requested_patch_cells_ = 0;

        [[nodiscard]] static std::size_t
        count_unique_slab_cells_from_flux_spaces_(
            const std::vector<FluxSpaceType>& flux_spaces)
        {
            std::set<std::pair<int, int>> keys;
            for (const auto& flux_space : flux_spaces)
            {
                const int slab_id = flux_space.patch().slab_id;
                for (int patch_cell_index = 0;
                     patch_cell_index < flux_space.n_patch_cells();
                     ++patch_cell_index)
                {
                    const int slab_cell_id =
                        finite_element::assembly::detail::
                            local_error_slab_cell_id_2d(
                                flux_space.patch().cell(patch_cell_index));
                    keys.emplace(slab_id, slab_cell_id);
                }
            }
            return keys.size();
        }

        [[nodiscard]] static std::size_t
        count_unique_slab_cells_from_requests_(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<
                finite_element::assembly::detail::
                    LocalErrorPatchCellBuildRequest2D>& requests)
        {
            std::set<std::pair<int, int>> keys;
            for (const auto& request : requests)
            {
                if (request.patch_id < 0 ||
                    request.patch_id >= static_cast<int>(flux_spaces.size()))
                {
                    throw std::runtime_error(
                        "LocalABElementCache2D: patch id out of range.");
                }
                const auto& flux_space =
                    flux_spaces[static_cast<std::size_t>(request.patch_id)];
                if (request.patch_cell_index < 0 ||
                    request.patch_cell_index >= flux_space.n_patch_cells())
                {
                    throw std::runtime_error(
                        "LocalABElementCache2D: patch cell index out of range.");
                }
                const int slab_id = flux_space.patch().slab_id;
                const int slab_cell_id =
                    finite_element::assembly::detail::
                        local_error_slab_cell_id_2d(
                            flux_space.patch().cell(
                                request.patch_cell_index));
                keys.emplace(slab_id, slab_cell_id);
            }
            return keys.size();
        }

        void collect_patch_cells_(
            int patch_id,
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space)
        {
            if (flux_space.n_patch_cells() != scalar_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalABElementCache2D: patch cell count mismatch.");
            }

            requested_patch_cells_ += flux_space.n_patch_cells();
            const int slab_id = flux_space.patch().slab_id;

            for (int patch_cell_index = 0;
                 patch_cell_index < flux_space.n_patch_cells();
                 ++patch_cell_index)
            {
                const int slab_cell_id =
                    finite_element::assembly::detail::
                        local_error_slab_cell_id_2d(
                            flux_space.patch().cell(patch_cell_index));
                const auto key = std::pair<int, int>{slab_id, slab_cell_id};
                if (cell_index_by_key_.find(key) != cell_index_by_key_.end())
                    continue;

                const int cache_id = static_cast<int>(cells_.size());
                cell_index_by_key_.emplace(key, cache_id);
                cells_.push_back(CellData{});
                build_requests_.push_back(
                    BuildRequest{
                        cache_id,
                        patch_id,
                        patch_cell_index,
                        slab_id,
                        slab_cell_id});
            }
        }

        void collect_patch_cell_(
            int patch_id,
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index)
        {
            if (flux_space.n_patch_cells() != scalar_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalABElementCache2D: patch cell count mismatch.");
            }
            if (patch_cell_index < 0 ||
                patch_cell_index >= flux_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalABElementCache2D: patch cell index out of range.");
            }

            ++requested_patch_cells_;
            const int slab_id = flux_space.patch().slab_id;
            const int slab_cell_id =
                finite_element::assembly::detail::
                    local_error_slab_cell_id_2d(
                        flux_space.patch().cell(patch_cell_index));
            const auto key = std::pair<int, int>{slab_id, slab_cell_id};
            if (cell_index_by_key_.find(key) != cell_index_by_key_.end())
                return;

            const int cache_id = static_cast<int>(cells_.size());
            cell_index_by_key_.emplace(key, cache_id);
            cells_.push_back(CellData{});
            build_requests_.push_back(
                BuildRequest{
                    cache_id,
                    patch_id,
                    patch_cell_index,
                    slab_id,
                    slab_cell_id});
        }

        void check_build_request_index_(int request_id) const
        {
            if (request_id < 0 ||
                request_id >= static_cast<int>(build_requests_.size()))
            {
                throw std::runtime_error(
                    "LocalABElementCache2D: build request index out of range.");
            }
        }

        template<class MFunction>
        static void fill_cell_(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            const typename RTCellCache::CellData& rt_cell,
            const MFunction& M,
            CellData& cell_data)
        {
            cell_data.slab_id = flux_space.patch().slab_id;
            cell_data.slab_cell_id =
                finite_element::assembly::detail::
                    local_error_slab_cell_id_2d(
                        flux_space.patch().cell(patch_cell_index));
            cell_data.A.resize(
                FluxSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);
            cell_data.B.resize(
                ScalarSpaceType::local_dofs_v,
                FluxSpaceType::local_dofs_v);

            for (const auto& qp : rt_cell.points)
            {
                accumulate_A_qpoint_(qp, M, cell_data.A);
                accumulate_B_qpoint_(
                    scalar_space,
                    patch_cell_index,
                    qp,
                    cell_data.B);
            }
        }

        template<class MFunction>
        static void accumulate_A_qpoint_(
            const typename RTCellCache::QuadraturePointData& qp,
            const MFunction& M,
            auto& local_A)
        {
            const auto M_q =
                coefficients::evaluate_diffusion_tensor<
                    FluxSpaceType::GT::dim_space_v>(
                        M,
                        qp.physical_point);
            const auto M_inverse = detail::inverse_diffusion_tensor_2d(M_q);

            typename FluxSpaceType::LocalValues
                weighted_inverse_rt_basis_values{};
            for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
            {
                weighted_inverse_rt_basis_values[
                    static_cast<std::size_t>(j)] =
                    detail::apply_inverse_tensor_2d(
                        M_inverse,
                        qp.rt_basis_values[static_cast<std::size_t>(j)],
                        qp.jacobian_weight);
            }

            for (int i = 0; i < FluxSpaceType::local_dofs_v; ++i)
            {
                const auto& sigma_i =
                    qp.rt_basis_values[static_cast<std::size_t>(i)];
                for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                {
                    const auto& M_inv_sigma_j_dmu =
                        weighted_inverse_rt_basis_values[
                            static_cast<std::size_t>(j)];
                    local_A(i, j) +=
                        sigma_i[0] * M_inv_sigma_j_dmu[0] +
                        sigma_i[1] * M_inv_sigma_j_dmu[1];
                }
            }
        }

        static void accumulate_B_qpoint_(
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            const typename RTCellCache::QuadraturePointData& qp,
            auto& local_B)
        {
            typename ScalarSpaceType::SpatialSpace::LocalValues
                spatial_scalar_values{};
            static_cast<void>(scalar_space);
            static_cast<void>(patch_cell_index);
            ScalarSpaceType::SpatialSpace::evaluate_local_basis(
                qp.spatial_reference_point,
                spatial_scalar_values);

            typename ScalarSpaceType::TimeValues time_values{};
            ScalarSpaceType::evaluate_time_basis(
                qp.time_reference_point,
                time_values);

            typename ScalarSpaceType::LocalValues scalar_values{};
            for (int spatial_local_dof = 0;
                 spatial_local_dof < ScalarSpaceType::spatial_local_dofs_v;
                 ++spatial_local_dof)
            {
                const double spatial_value =
                    spatial_scalar_values[
                        static_cast<std::size_t>(spatial_local_dof)];
                for (int time_dof = 0;
                     time_dof < ScalarSpaceType::n_time_dofs_v;
                     ++time_dof)
                {
                    const int local_id =
                        scalar_space.local_dof_index(
                            spatial_local_dof,
                            time_dof);
                    scalar_values[static_cast<std::size_t>(local_id)] =
                        spatial_value *
                        time_values[static_cast<std::size_t>(time_dof)];
                }
            }

            for (int i = 0; i < ScalarSpaceType::local_dofs_v; ++i)
            {
                const double weighted_q_i =
                    scalar_values[static_cast<std::size_t>(i)] *
                    qp.jacobian_weight;
                for (int j = 0; j < FluxSpaceType::local_dofs_v; ++j)
                {
                    local_B(i, j) +=
                        weighted_q_i *
                        qp.rt_basis_divergences[
                            static_cast<std::size_t>(j)];
                }
            }
        }
    };
}

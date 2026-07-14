#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "../detail/active_cell_locator_time_slab.hpp"
#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "shared_local_error_context_2d.hpp"
#include "vec_f_1d.hpp"

namespace finite_element::assembly::error_system
{
    template<int QSpace, int QTime, class PatchFluxSpaceType>
    class LocalRHSStateCache2D
    {
    public:
        using FluxSpaceType = PatchFluxSpaceType;
        using GT = typename FluxSpaceType::GT;
        using PatchType = typename FluxSpaceType::Patch;
        using Types = typename PatchType::Types;
        using SpaceTimePoint = typename Types::SpaceTimePoint;
        using SpatialGradient = coefficients::DiffusionVector<2>;
        using DiffusionTensor = coefficients::DiffusionTensor<2>;
        using RTCellCache =
            finite_element::assembly::detail::
                LocalErrorRTCellQuadratureCache2D<
                    QSpace,
                    QTime,
                    FluxSpaceType>;

        static_assert(GT::dim_space_v == 2,
                      "LocalRHSStateCache2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "LocalRHSStateCache2D requires dim_time_v == 1.");

        static constexpr int n_quadrature_points_v =
            RTCellCache::n_quadrature_points_v;

        struct QuadraturePointData
        {
            SpaceTimePoint physical_point{};
            SpatialGradient grad_lambda_tilde{};
            SpatialGradient grad_u_delta{};
            SpatialGradient grad_theta_tilde{};
            double u_time_derivative = 0.0;
            double ell_value = 0.0;
            DiffusionTensor diffusion_tensor{};
            SpatialGradient M_grad_theta_tilde{};
        };

        struct CellData
        {
            int slab_id = -1;
            int slab_cell_id = -1;
            int source_cell_id = -1;
            int x_cell_id = -1;
            std::array<QuadraturePointData, n_quadrature_points_v> points{};
        };

        LocalRHSStateCache2D() = default;

        template<
            class XSpaceType,
            class SlabSpaceType,
            class ReconstructedFunctionType,
            class XFunctionType,
            class EllFunction,
            class MFunction>
        LocalRHSStateCache2D(
            const std::vector<FluxSpaceType>& flux_spaces,
            const RTCellCache& rt_cell_cache,
            const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M)
        {
            prepare_from_flux_spaces(flux_spaces);
            for (int request_id = 0;
                 request_id < n_build_requests();
                 ++request_id)
            {
                fill_build_request(
                    request_id,
                    flux_spaces,
                    rt_cell_cache,
                    context,
                    lambda_tilde,
                    u_delta,
                    ell,
                    M);
            }
        }

        void prepare_from_flux_spaces(
            const std::vector<FluxSpaceType>& flux_spaces)
        {
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
                    flux_spaces[patch_id]);
            }
        }

        void prepare_from_patch_cells(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<
                finite_element::assembly::detail::
                    LocalErrorPatchCellBuildRequest2D>& requests)
        {
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
                        "LocalRHSStateCache2D: patch id out of range.");
                }
                collect_patch_cell_(
                    request.patch_id,
                    flux_spaces[static_cast<std::size_t>(request.patch_id)],
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
            return build_requests_[static_cast<std::size_t>(request_id)]
                .slab_id;
        }

        [[nodiscard]] int build_request_slab_cell_id(int request_id) const
        {
            check_build_request_index_(request_id);
            return build_requests_[static_cast<std::size_t>(request_id)]
                .slab_cell_id;
        }

        template<
            class XSpaceType,
            class SlabSpaceType,
            class ReconstructedFunctionType,
            class XFunctionType,
            class EllFunction,
            class MFunction>
        void fill_build_request(
            int request_id,
            const std::vector<FluxSpaceType>& flux_spaces,
            const RTCellCache& rt_cell_cache,
            const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M)
        {
            check_build_request_index_(request_id);
            const auto& request =
                build_requests_[static_cast<std::size_t>(request_id)];
            fill_cell_(
                flux_spaces[static_cast<std::size_t>(request.patch_id)],
                request.patch_cell_index,
                rt_cell_cache.cell(
                    request.slab_id,
                    request.slab_cell_id),
                context,
                lambda_tilde,
                u_delta,
                ell,
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
                    "LocalRHSStateCache2D: slab cell not cached.");
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
                        flux_space.patch().cell(patch_cell_index)
                            .slab_cell_id;
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
                        "LocalRHSStateCache2D: patch id out of range.");
                }
                const auto& flux_space =
                    flux_spaces[static_cast<std::size_t>(request.patch_id)];
                if (request.patch_cell_index < 0 ||
                    request.patch_cell_index >= flux_space.n_patch_cells())
                {
                    throw std::runtime_error(
                        "LocalRHSStateCache2D: patch cell index out of range.");
                }
                const int slab_id = flux_space.patch().slab_id;
                const int slab_cell_id =
                    flux_space.patch().cell(request.patch_cell_index)
                        .slab_cell_id;
                keys.emplace(slab_id, slab_cell_id);
            }
            return keys.size();
        }

        void collect_patch_cells_(
            int patch_id,
            const FluxSpaceType& flux_space)
        {
            requested_patch_cells_ += flux_space.n_patch_cells();
            const int slab_id = flux_space.patch().slab_id;

            for (int patch_cell_index = 0;
                 patch_cell_index < flux_space.n_patch_cells();
                 ++patch_cell_index)
            {
                const int slab_cell_id =
                    flux_space.patch().cell(patch_cell_index).slab_cell_id;
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
            int patch_cell_index)
        {
            if (patch_cell_index < 0 ||
                patch_cell_index >= flux_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalRHSStateCache2D: patch cell index out of range.");
            }

            ++requested_patch_cells_;
            const int slab_id = flux_space.patch().slab_id;
            const int slab_cell_id =
                flux_space.patch().cell(patch_cell_index).slab_cell_id;
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
                    "LocalRHSStateCache2D: build request index out of range.");
            }
        }

        template<
            class XSpaceType,
            class SlabSpaceType,
            class ReconstructedFunctionType,
            class XFunctionType,
            class EllFunction,
            class MFunction>
        static void fill_cell_(
            const FluxSpaceType& flux_space,
            int patch_cell_index,
            const typename RTCellCache::CellData& rt_cell,
            const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M,
            CellData& cell_data)
        {
            const auto& patch_cell =
                flux_space.patch().cell(patch_cell_index);
            const int slab_id = flux_space.patch().slab_id;
            const int slab_cell_id = patch_cell.slab_cell_id;
            const int source_cell_id = patch_cell.source_cell_id;
            const int x_cell_id =
                context.shared_context != nullptr
                    ? context.shared_context
                          ->active_x_cell_for_source_cell(source_cell_id)
                    : finite_element::assembly::detail::
                          find_active_ancestor_cell_from_source_cell(
                              *context.x_ancestor_cache,
                              *context.x_space,
                              source_cell_id);
            const auto& slab_geom =
                context.shared_context != nullptr
                    ? context.shared_context->slab_geometry(
                          context.shared_context->slab_cell_ordinal(
                              slab_id,
                              slab_cell_id))
                    : (*context.slab_geometry_caches)[
                          static_cast<std::size_t>(slab_id)]
                          .geometry(slab_cell_id);
            const auto& x_geom =
                context.shared_context != nullptr
                    ? context.shared_context->x_geometry(x_cell_id)
                    : context.x_geometry_cache->geometry(x_cell_id);

            cell_data.slab_id = slab_id;
            cell_data.slab_cell_id = slab_cell_id;
            cell_data.source_cell_id = source_cell_id;
            cell_data.x_cell_id = x_cell_id;

            constexpr int time_component = GT::dim_space_v;
            for (int qp_id = 0; qp_id < n_quadrature_points_v; ++qp_id)
            {
                const auto& rt_qp =
                    rt_cell.points[static_cast<std::size_t>(qp_id)];
                auto& qp =
                    cell_data.points[static_cast<std::size_t>(qp_id)];

                const auto grad_lambda =
                    lambda_tilde.gradient_on_cell(
                        slab_id,
                        slab_cell_id,
                        rt_qp.physical_point,
                        slab_geom);
                const auto grad_u =
                    u_delta.gradient_on_cell(
                        x_cell_id,
                        rt_qp.physical_point,
                        x_geom);

                qp.physical_point = rt_qp.physical_point;
                qp.grad_lambda_tilde =
                    SpatialGradient{grad_lambda[0], grad_lambda[1]};
                qp.grad_u_delta =
                    SpatialGradient{grad_u[0], grad_u[1]};
                qp.grad_theta_tilde =
                    SpatialGradient{
                        qp.grad_lambda_tilde[0] + qp.grad_u_delta[0],
                        qp.grad_lambda_tilde[1] + qp.grad_u_delta[1]};
                qp.u_time_derivative = grad_u[time_component];
                qp.ell_value = static_cast<double>(ell(qp.physical_point));
                qp.diffusion_tensor =
                    coefficients::evaluate_diffusion_tensor<
                        GT::dim_space_v>(
                        M,
                        qp.physical_point);
                qp.M_grad_theta_tilde =
                    coefficients::apply_validated_M<GT::dim_space_v>(
                        qp.diffusion_tensor,
                        qp.grad_theta_tilde);
            }
        }
    };
}

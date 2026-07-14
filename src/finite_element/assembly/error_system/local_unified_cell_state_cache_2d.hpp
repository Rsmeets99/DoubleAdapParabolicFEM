#pragma once

#include <stdexcept>
#include <vector>

#include "../detail/local_error_quadrature_tables_2d.hpp"
#include "local_ab_element_cache_2d.hpp"
#include "local_rhs_state_cache_2d.hpp"

namespace finite_element::assembly::error_system
{
    template<
        int QSpace,
        int QTime,
        class PatchFluxSpaceType,
        class PatchScalarSpaceType>
    class LocalUnifiedCellStateCache2D
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
        using ABElementCache =
            LocalABElementCache2D<
                QSpace,
                QTime,
                FluxSpaceType,
                ScalarSpaceType>;
        using RHSStateCache =
            LocalRHSStateCache2D<
                QSpace,
                QTime,
                FluxSpaceType>;

        LocalUnifiedCellStateCache2D() = default;

        void prepare_from_spaces(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<ScalarSpaceType>& scalar_spaces)
        {
            rt_cell_cache_.prepare_from_flux_spaces(flux_spaces);
            ab_element_cache_.prepare_from_spaces(
                flux_spaces,
                scalar_spaces);
            rhs_state_cache_.prepare_from_flux_spaces(flux_spaces);
            check_request_compatibility_();
        }

        [[nodiscard]] int n_build_requests() const noexcept
        {
            return rt_cell_cache_.n_build_requests();
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
            const std::vector<ScalarSpaceType>& scalar_spaces,
            const LocalErrorProblemContext<XSpaceType, SlabSpaceType>& context,
            const ReconstructedFunctionType& lambda_tilde,
            const XFunctionType& u_delta,
            const EllFunction& ell,
            const MFunction& M,
            bool fill_ab = true)
        {
            rt_cell_cache_.fill_build_request(
                request_id,
                flux_spaces);
            if (fill_ab)
            {
                ab_element_cache_.fill_build_request(
                    request_id,
                    flux_spaces,
                    scalar_spaces,
                    rt_cell_cache_,
                    M);
            }
            rhs_state_cache_.fill_build_request(
                request_id,
                flux_spaces,
                rt_cell_cache_,
                context,
                lambda_tilde,
                u_delta,
                ell,
                M);
        }

        [[nodiscard]] RTCellCache& rt_cell_cache() noexcept
        {
            return rt_cell_cache_;
        }

        [[nodiscard]] const RTCellCache& rt_cell_cache() const noexcept
        {
            return rt_cell_cache_;
        }

        [[nodiscard]] ABElementCache& ab_element_cache() noexcept
        {
            return ab_element_cache_;
        }

        [[nodiscard]] const ABElementCache& ab_element_cache() const noexcept
        {
            return ab_element_cache_;
        }

        [[nodiscard]] RHSStateCache& rhs_state_cache() noexcept
        {
            return rhs_state_cache_;
        }

        [[nodiscard]] const RHSStateCache& rhs_state_cache() const noexcept
        {
            return rhs_state_cache_;
        }

        [[nodiscard]] int requested_patch_cells() const noexcept
        {
            return rt_cell_cache_.requested_patch_cells();
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return rt_cell_cache_.estimated_memory_bytes() +
                   ab_element_cache_.estimated_memory_bytes() +
                   rhs_state_cache_.estimated_memory_bytes();
        }

        [[nodiscard]] int unique_slab_cells() const noexcept
        {
            return rt_cell_cache_.unique_slab_cells();
        }

        [[nodiscard]] int duplicate_patch_cells() const noexcept
        {
            return rt_cell_cache_.duplicate_patch_cells();
        }

    private:
        RTCellCache rt_cell_cache_{};
        ABElementCache ab_element_cache_{};
        RHSStateCache rhs_state_cache_{};

        void check_request_compatibility_() const
        {
            const int n_requests = rt_cell_cache_.n_build_requests();
            if (ab_element_cache_.n_build_requests() != n_requests ||
                rhs_state_cache_.n_build_requests() != n_requests)
            {
                throw std::runtime_error(
                    "LocalUnifiedCellStateCache2D: cache build-request count mismatch.");
            }

            for (int request_id = 0; request_id < n_requests; ++request_id)
            {
                const int slab_id =
                    rt_cell_cache_.build_request_slab_id(request_id);
                const int slab_cell_id =
                    rt_cell_cache_.build_request_slab_cell_id(request_id);
                if (ab_element_cache_.build_request_slab_id(request_id) !=
                        slab_id ||
                    ab_element_cache_.build_request_slab_cell_id(request_id) !=
                        slab_cell_id ||
                    rhs_state_cache_.build_request_slab_id(request_id) !=
                        slab_id ||
                    rhs_state_cache_.build_request_slab_cell_id(request_id) !=
                        slab_cell_id)
                {
                    throw std::runtime_error(
                        "LocalUnifiedCellStateCache2D: cache build-request ordering mismatch.");
                }
            }
        }
    };
}

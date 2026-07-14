#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../constrained_dofs.hpp"
#include "../scatter.hpp"
#include "../detail/active_cell_locator.hpp"
#include "../detail/assembly_diagnostics.hpp"
#include "../detail/assembly_space_cache.hpp"
#include "../detail/openmp_assembly.hpp"
#include "../detail/space_time_basis_tables.hpp"
#include "../detail/zero_local.hpp"
#include "../../basis/space_time_basis_selector.hpp"
#include "../../coefficients/diffusion_coefficient.hpp"
#include "../../geometry/cell_geometry.hpp"

#include "linear_algebra/assembly/local_objects.hpp"
#include "linear_algebra/concepts/sparse_builder.hpp"
#include "linear_algebra/concepts/sparse_matrix.hpp"
#include "linear_algebra/concepts/vector.hpp"

namespace finite_element::assembly
{
    template<int QSpace, int QTime, class YFESpaceType>
    class MainAssemblyQuadratureCache2D
    {
    public:
        using YSpaceType = YFESpaceType;
        using YGT        = typename YSpaceType::GT;
        using YFETraits  = typename YSpaceType::FETraitsType;
        using YGeometry  =
            finite_element::geometry::CellGeometry<YSpaceType, 2>;
        using Tables =
            detail::SpaceTimeBasisTables<YGT, YFETraits, QSpace, QTime>;
        using SpaceTimePoint = typename YSpaceType::SpaceTimePoint;
        using SpatialGradient = typename YGeometry::SpatialGradient;
        using DiffusionTensor =
            finite_element::coefficients::DiffusionTensor<2>;

        static_assert(
            YGT::dim_space_v == 2,
            "MainAssemblyQuadratureCache2D is only for 2+1D spaces.");

        static constexpr int dofs_per_cell = YFETraits::dofs_per_cell;
        static constexpr int n_q = Tables::n_cell_q;

        struct QPointData
        {
            SpaceTimePoint physical{};
            double dmu = 0.0;
            DiffusionTensor diffusion{};
            double ell = 0.0;
        };

        struct CellData
        {
            int cell_id = -1;
            std::array<QPointData, n_q> qpoints{};
        };

        explicit MainAssemblyQuadratureCache2D(const YSpaceType& y_space)
            : y_space_(&y_space),
              cells_(y_space.mesh_ref().n_cells())
        {}

        template<class MFunction, class EllFunction>
        void build_all(
            detail::AssemblySpaceCache<YSpaceType>& y_cache,
            const MFunction& M,
            const EllFunction& ell)
        {
            const auto& active_cells = y_space_->active_cells();
            const int n_active_cells = static_cast<int>(active_cells.size());

            for (const int cell_id : active_cells)
                (void)y_cache.geometry(cell_id);

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
            if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
            {
                std::exception_ptr error;

#pragma omp parallel for schedule(static)
                for (int item_index = 0; item_index < n_active_cells; ++item_index)
                {
                    try
                    {
                        const int cell_id =
                            active_cells[static_cast<std::size_t>(item_index)];
                        cells_[static_cast<std::size_t>(cell_id)].emplace(
                            build_cell_(cell_id, y_cache.geometry(cell_id), M, ell));
                    }
                    catch (...)
                    {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                        {
                            if (!error)
                                error = std::current_exception();
                        }
                    }
                }

                detail::rethrow_parallel_exception(error);
                count_built_entries_();
                return;
            }
#endif

            for (const int cell_id : active_cells)
            {
                cells_[static_cast<std::size_t>(cell_id)].emplace(
                    build_cell_(cell_id, y_cache.geometry(cell_id), M, ell));
            }
            count_built_entries_();
        }

        [[nodiscard]] const CellData& cell(int cell_id) const
        {
            if (cell_id < 0 ||
                static_cast<std::size_t>(cell_id) >= cells_.size() ||
                !cells_[static_cast<std::size_t>(cell_id)].has_value())
            {
                throw std::runtime_error(
                    "MainAssemblyQuadratureCache2D: missing cell cache entry.");
            }
            return *cells_[static_cast<std::size_t>(cell_id)];
        }

        [[nodiscard]] std::size_t estimated_live_bytes() const noexcept
        {
            return sizeof(*this) +
                   cells_.capacity() * sizeof(std::optional<CellData>) +
                   built_entries_ * sizeof(CellData);
        }

        [[nodiscard]] std::size_t built_entries() const noexcept
        {
            return built_entries_;
        }

    private:
        template<class MFunction, class EllFunction>
        [[nodiscard]] static CellData build_cell_(
            int cell_id,
            const typename YGeometry::Data& geom,
            const MFunction& M,
            const EllFunction& ell)
        {
            CellData cell;
            cell.cell_id = cell_id;

            const double jac = YGeometry::jacobian_measure(geom);
            for (int q = 0; q < n_q; ++q)
            {
                auto& qp = cell.qpoints[static_cast<std::size_t>(q)];
                const auto& xi_q = Tables::cell_rule.points[q];

                qp.physical = YGeometry::map_to_physical(geom, xi_q);
                qp.dmu = jac * Tables::cell_rule.weights[q];
                qp.diffusion =
                    finite_element::coefficients::evaluate_diffusion_tensor<
                        MFunction,
                        2>(
                            M,
                            qp.physical);
                qp.ell = static_cast<double>(ell(qp.physical));
            }

            return cell;
        }

        void count_built_entries_() noexcept
        {
            built_entries_ = 0;
            for (const auto& entry : cells_)
            {
                if (entry.has_value())
                    ++built_entries_;
            }
        }

        const YSpaceType* y_space_ = nullptr;
        std::vector<std::optional<CellData>> cells_{};
        std::size_t built_entries_ = 0;
    };

    template<
        int QSpace,
        int QTime,
        class Backend,
        class FESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_A_with_quadrature_cache_2d(
        typename Backend::SparseMatrix& A,
        const FESpaceType& space,
        const MainAssemblyQuadratureCache2D<QSpace, QTime, FESpaceType>&
            q_cache,
        detail::AssemblySpaceCache<FESpaceType>& fe_cache,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        using SparseBuilder = typename Backend::SparseBuilder;
        using QCache =
            MainAssemblyQuadratureCache2D<QSpace, QTime, FESpaceType>;
        using Tables = typename QCache::Tables;
        using Geometry = typename QCache::YGeometry;
        using SpatialGradient = typename QCache::SpatialGradient;

        constexpr int dofs_per_cell = QCache::dofs_per_cell;
        constexpr int n_q = QCache::n_q;

        const auto& dof_handler = space.dof_handler_ref();
        const auto& active_cells = space.active_cells();
        const int n_active_cells = static_cast<int>(active_cells.size());
        const std::size_t reserve_entries =
            static_cast<std::size_t>(n_active_cells) *
            static_cast<std::size_t>(dofs_per_cell) *
            static_cast<std::size_t>(dofs_per_cell) * 4u;

        SparseBuilder builder;

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            for (const int cell_id : active_cells)
                (void)fe_cache.dof_expansion(cell_id);

            auto builders =
                detail::make_thread_local_sparse_builders<SparseBuilder>(
                    detail::openmp_thread_count(),
                    reserve_entries);
            std::exception_ptr error;

#pragma omp parallel
            {
                try
                {
                    const int thread_id = omp_get_thread_num();
                    auto& local_builder =
                        builders[static_cast<std::size_t>(thread_id)];
                    la::local::FixedLocalMatrix<
                        dofs_per_cell,
                        dofs_per_cell> local;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int cell_id =
                            active_cells[static_cast<std::size_t>(item_index)];
                        const auto& expanded = fe_cache.dof_expansion(cell_id);
                        const auto& geom = fe_cache.geometry(cell_id);
                        const auto& cell = q_cache.cell(cell_id);

                        detail::zero_local_matrix(local);
                        std::array<SpatialGradient, dofs_per_cell>
                            spatial_grads_on_q{};
                        std::array<SpatialGradient, dofs_per_cell>
                            M_spatial_grads_on_q{};
                        for (int q = 0; q < n_q; ++q)
                        {
                            const auto& qp =
                                cell.qpoints[static_cast<std::size_t>(q)];
                            const auto& grads = Tables::gradients_on_cell_qp(q);
                            for (int i = 0; i < dofs_per_cell; ++i)
                            {
                                spatial_grads_on_q[static_cast<std::size_t>(i)] =
                                    Geometry::spatial_gradient(geom, grads[i]);
                                M_spatial_grads_on_q[
                                    static_cast<std::size_t>(i)] =
                                    coefficients::apply_validated_M<2>(
                                        qp.diffusion,
                                        spatial_grads_on_q[
                                            static_cast<std::size_t>(i)]);
                            }

                            for (int i = 0; i < dofs_per_cell; ++i)
                            {
                                const auto& grad_i =
                                    spatial_grads_on_q[static_cast<std::size_t>(i)];
                                for (int j = 0; j < dofs_per_cell; ++j)
                                {
                                    local(i, j) +=
                                        coefficients::dot<2>(
                                            grad_i,
                                            M_spatial_grads_on_q[
                                                static_cast<std::size_t>(j)]) *
                                        qp.dmu;
                                }
                            }
                        }

                        scatter_matrix(
                            local_builder,
                            local,
                            expanded,
                            expanded,
                            zero_tol);
                    }
                }
                catch (...)
                {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }

            detail::rethrow_parallel_exception(error);
            builder = detail::merge_sparse_builders(builders);
        }
        else
#endif
        {
            builder.reserve(reserve_entries);
            la::local::FixedLocalMatrix<
                dofs_per_cell,
                dofs_per_cell> local;

            for (const int cell_id : active_cells)
            {
                const auto& expanded = fe_cache.dof_expansion(cell_id);
                const auto& geom = fe_cache.geometry(cell_id);
                const auto& cell = q_cache.cell(cell_id);

                detail::zero_local_matrix(local);
                std::array<SpatialGradient, dofs_per_cell> spatial_grads_on_q{};
                std::array<SpatialGradient, dofs_per_cell>
                    M_spatial_grads_on_q{};
                for (int q = 0; q < n_q; ++q)
                {
                    const auto& qp = cell.qpoints[static_cast<std::size_t>(q)];
                    const auto& grads = Tables::gradients_on_cell_qp(q);
                    for (int i = 0; i < dofs_per_cell; ++i)
                    {
                        spatial_grads_on_q[static_cast<std::size_t>(i)] =
                            Geometry::spatial_gradient(geom, grads[i]);
                        M_spatial_grads_on_q[static_cast<std::size_t>(i)] =
                            coefficients::apply_validated_M<2>(
                                qp.diffusion,
                                spatial_grads_on_q[
                                    static_cast<std::size_t>(i)]);
                    }

                    for (int i = 0; i < dofs_per_cell; ++i)
                    {
                        const auto& grad_i =
                            spatial_grads_on_q[static_cast<std::size_t>(i)];
                        for (int j = 0; j < dofs_per_cell; ++j)
                        {
                            local(i, j) +=
                                coefficients::dot<2>(
                                    grad_i,
                                    M_spatial_grads_on_q[
                                        static_cast<std::size_t>(j)]) *
                                qp.dmu;
                        }
                    }
                }

                scatter_matrix(builder, local, expanded, expanded, zero_tol);
            }
        }

        if (diagnostics != nullptr)
        {
            diagnostics->active_cells =
                static_cast<std::size_t>(n_active_cells);
            diagnostics->quadrature_points =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(n_q);
            diagnostics->gradient_evaluations =
                diagnostics->quadrature_points *
                static_cast<std::size_t>(dofs_per_cell);
            diagnostics->diffusion_tensor_evaluations =
                diagnostics->quadrature_points;
            diagnostics->sparse_triplets_emitted = builder.size();
            diagnostics->peak_triplet_bytes =
                builder.size() *
                detail::estimated_triplet_bytes<SparseBuilder>();
        }

        A.resize(dof_handler.n_true_dofs(), dof_handler.n_true_dofs());
        A.set_from_builder(builder);

        if (diagnostics != nullptr)
        {
            diagnostics->final_matrix_nonzeros = detail::count_nonzeros(A);
            diagnostics->peak_sparse_matrix_bytes =
                detail::estimate_compressed_sparse_matrix_bytes(
                    A.rows(),
                    A.cols(),
                    diagnostics->final_matrix_nonzeros);
        }
    }

    template<
        int QSpace,
        int QTime,
        class Backend,
        class XFESpaceType,
        class YFESpaceType>
    requires la::concepts::SparseBuilderLike<typename Backend::SparseBuilder> &&
             la::concepts::SparseMatrixLike<typename Backend::SparseMatrix,
                                            typename Backend::SparseBuilder>
    void assemble_mat_B_with_quadrature_cache_2d(
        typename Backend::SparseMatrix& B,
        const XFESpaceType& x_space,
        const YFESpaceType& y_space,
        detail::AssemblySpaceCache<XFESpaceType>& x_cache,
        detail::AssemblySpaceCache<YFESpaceType>& y_fe_cache,
        detail::ActiveAncestorCache<XFESpaceType>& ancestor_cache,
        const MainAssemblyQuadratureCache2D<QSpace, QTime, YFESpaceType>&
            q_cache,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        using XBasis =
            finite_element::basis::SpaceTimeBasis<
                typename XFESpaceType::GT,
                typename XFESpaceType::FETraitsType>;
        using XGeometry =
            finite_element::geometry::CellGeometry<XFESpaceType, 2>;
        using QCache =
            MainAssemblyQuadratureCache2D<QSpace, QTime, YFESpaceType>;
        using YTables = typename QCache::Tables;
        using YGeometry = typename QCache::YGeometry;
        using YSpatialGradient = typename QCache::SpatialGradient;
        using SparseBuilder = typename Backend::SparseBuilder;

        constexpr int x_dofs_per_cell = XFESpaceType::FETraitsType::dofs_per_cell;
        constexpr int y_dofs_per_cell = YFESpaceType::FETraitsType::dofs_per_cell;
        constexpr int n_q = QCache::n_q;

        const auto& x_dof_handler = x_space.dof_handler_ref();
        const auto& y_dof_handler = y_space.dof_handler_ref();
        const auto& y_active_cells = y_space.active_cells();

        const int n_active_cells = static_cast<int>(y_active_cells.size());
        const std::size_t reserve_entries =
            static_cast<std::size_t>(n_active_cells) *
            static_cast<std::size_t>(x_dofs_per_cell) *
            static_cast<std::size_t>(y_dofs_per_cell) * 4u;

        std::vector<int> x_cell_ids(static_cast<std::size_t>(n_active_cells), -1);
        for (int item_index = 0; item_index < n_active_cells; ++item_index)
        {
            const int y_cell_id = y_active_cells[static_cast<std::size_t>(item_index)];
            const int x_cell_id =
                detail::find_active_ancestor_cell(
                    ancestor_cache,
                    x_space,
                    y_space,
                    y_cell_id);
            x_cell_ids[static_cast<std::size_t>(item_index)] = x_cell_id;

            (void)x_cache.dof_expansion(x_cell_id);
            (void)x_cache.geometry(x_cell_id);
            (void)y_fe_cache.dof_expansion(y_cell_id);
        }

        SparseBuilder builder;

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            auto builders =
                detail::make_thread_local_sparse_builders<SparseBuilder>(
                    detail::openmp_thread_count(),
                    reserve_entries);
            std::exception_ptr error;

#pragma omp parallel
            {
                try
                {
                    const int thread_id = omp_get_thread_num();
                    auto& local_builder =
                        builders[static_cast<std::size_t>(thread_id)];
                    la::local::FixedLocalMatrix<
                        x_dofs_per_cell,
                        y_dofs_per_cell> local;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int y_cell_id =
                            y_active_cells[static_cast<std::size_t>(item_index)];
                        const int x_cell_id =
                            x_cell_ids[static_cast<std::size_t>(item_index)];

                        const auto& x_geom = x_cache.geometry(x_cell_id);
                        const auto& y_geom = y_fe_cache.geometry(y_cell_id);
                        const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                        const auto& y_expanded =
                            y_fe_cache.dof_expansion(y_cell_id);
                        const auto& cell = q_cache.cell(y_cell_id);

                        detail::zero_local_matrix(local);
                        std::array<YSpatialGradient, y_dofs_per_cell>
                            gradx_phi_on_q{};
                        std::array<YSpatialGradient, y_dofs_per_cell>
                            M_gradx_phi_on_q{};
                        for (int q = 0; q < n_q; ++q)
                        {
                            const auto& qp =
                                cell.qpoints[static_cast<std::size_t>(q)];
                            const auto& phi_vals =
                                YTables::values_on_cell_qp(q);
                            const auto& phi_grads_ref =
                                YTables::gradients_on_cell_qp(q);
                            for (int j = 0; j < y_dofs_per_cell; ++j)
                            {
                                gradx_phi_on_q[static_cast<std::size_t>(j)] =
                                    YGeometry::spatial_gradient(
                                        y_geom,
                                        phi_grads_ref[j]);
                                M_gradx_phi_on_q[
                                    static_cast<std::size_t>(j)] =
                                    coefficients::apply_validated_M<2>(
                                        qp.diffusion,
                                        gradx_phi_on_q[
                                            static_cast<std::size_t>(j)]);
                            }

                            const auto xi_x =
                                XGeometry::physical_to_reference(
                                    x_geom,
                                    qp.physical);
                            const auto psi_grads_ref = XBasis::grad_all(xi_x);

                            std::array<double, x_dofs_per_cell> dt_psi_on_q{};
                            std::array<
                                typename XGeometry::SpatialGradient,
                                x_dofs_per_cell> gradx_psi_on_q{};

                            for (int i = 0; i < x_dofs_per_cell; ++i)
                            {
                                dt_psi_on_q[static_cast<std::size_t>(i)] =
                                    XGeometry::time_derivative(
                                        x_geom,
                                        psi_grads_ref[i]);
                                gradx_psi_on_q[static_cast<std::size_t>(i)] =
                                    XGeometry::spatial_gradient(
                                        x_geom,
                                        psi_grads_ref[i]);
                            }

                            for (int i = 0; i < x_dofs_per_cell; ++i)
                            {
                                const double dt_psi_i =
                                    dt_psi_on_q[static_cast<std::size_t>(i)];
                                const auto& gradx_psi_i =
                                    gradx_psi_on_q[static_cast<std::size_t>(i)];

                                for (int j = 0; j < y_dofs_per_cell; ++j)
                                {
                                    local(i, j) +=
                                        (dt_psi_i * phi_vals[j] +
                                         coefficients::dot<2>(
                                             gradx_psi_i,
                                             M_gradx_phi_on_q[
                                                 static_cast<std::size_t>(j)])) *
                                        qp.dmu;
                                }
                            }
                        }

                        scatter_matrix(
                            local_builder,
                            local,
                            x_expanded,
                            y_expanded,
                            zero_tol);
                    }
                }
                catch (...)
                {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }

            detail::rethrow_parallel_exception(error);
            builder = detail::merge_sparse_builders(builders);
        }
        else
#endif
        {
            builder.reserve(reserve_entries);
            la::local::FixedLocalMatrix<
                x_dofs_per_cell,
                y_dofs_per_cell> local;

            for (int item_index = 0; item_index < n_active_cells; ++item_index)
            {
                const int y_cell_id =
                    y_active_cells[static_cast<std::size_t>(item_index)];
                const int x_cell_id =
                    x_cell_ids[static_cast<std::size_t>(item_index)];

                const auto& x_geom = x_cache.geometry(x_cell_id);
                const auto& y_geom = y_fe_cache.geometry(y_cell_id);
                const auto& x_expanded = x_cache.dof_expansion(x_cell_id);
                const auto& y_expanded = y_fe_cache.dof_expansion(y_cell_id);
                const auto& cell = q_cache.cell(y_cell_id);

                detail::zero_local_matrix(local);
                std::array<YSpatialGradient, y_dofs_per_cell> gradx_phi_on_q{};
                std::array<YSpatialGradient, y_dofs_per_cell>
                    M_gradx_phi_on_q{};
                for (int q = 0; q < n_q; ++q)
                {
                    const auto& qp = cell.qpoints[static_cast<std::size_t>(q)];
                    const auto& phi_vals = YTables::values_on_cell_qp(q);
                    const auto& phi_grads_ref =
                        YTables::gradients_on_cell_qp(q);
                    for (int j = 0; j < y_dofs_per_cell; ++j)
                    {
                        gradx_phi_on_q[static_cast<std::size_t>(j)] =
                            YGeometry::spatial_gradient(y_geom, phi_grads_ref[j]);
                        M_gradx_phi_on_q[static_cast<std::size_t>(j)] =
                            coefficients::apply_validated_M<2>(
                                qp.diffusion,
                                gradx_phi_on_q[
                                    static_cast<std::size_t>(j)]);
                    }

                    const auto xi_x =
                        XGeometry::physical_to_reference(x_geom, qp.physical);
                    const auto psi_grads_ref = XBasis::grad_all(xi_x);

                    std::array<double, x_dofs_per_cell> dt_psi_on_q{};
                    std::array<
                        typename XGeometry::SpatialGradient,
                        x_dofs_per_cell> gradx_psi_on_q{};

                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        dt_psi_on_q[static_cast<std::size_t>(i)] =
                            XGeometry::time_derivative(
                                x_geom,
                                psi_grads_ref[i]);
                        gradx_psi_on_q[static_cast<std::size_t>(i)] =
                            XGeometry::spatial_gradient(
                                x_geom,
                                psi_grads_ref[i]);
                    }

                    for (int i = 0; i < x_dofs_per_cell; ++i)
                    {
                        const double dt_psi_i =
                            dt_psi_on_q[static_cast<std::size_t>(i)];
                        const auto& gradx_psi_i =
                            gradx_psi_on_q[static_cast<std::size_t>(i)];

                        for (int j = 0; j < y_dofs_per_cell; ++j)
                        {
                            local(i, j) +=
                                (dt_psi_i * phi_vals[j] +
                                 coefficients::dot<2>(
                                     gradx_psi_i,
                                     M_gradx_phi_on_q[
                                         static_cast<std::size_t>(j)])) *
                                qp.dmu;
                        }
                    }
                }

                scatter_matrix(
                    builder,
                    local,
                    x_expanded,
                    y_expanded,
                    zero_tol);
            }
        }

        if (diagnostics != nullptr)
        {
            diagnostics->active_cells =
                static_cast<std::size_t>(n_active_cells);
            diagnostics->quadrature_points =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(n_q);
            diagnostics->scalar_basis_evaluations =
                diagnostics->quadrature_points *
                static_cast<std::size_t>(y_dofs_per_cell);
            diagnostics->gradient_evaluations =
                diagnostics->quadrature_points *
                static_cast<std::size_t>(
                    x_dofs_per_cell + y_dofs_per_cell);
            diagnostics->diffusion_tensor_evaluations =
                diagnostics->quadrature_points;
            diagnostics->sparse_triplets_emitted = builder.size();
            diagnostics->peak_triplet_bytes =
                builder.size() *
                detail::estimated_triplet_bytes<SparseBuilder>();
        }

        B.resize(x_dof_handler.n_true_dofs(), y_dof_handler.n_true_dofs());
        B.set_from_builder(builder);

        if (diagnostics != nullptr)
        {
            diagnostics->final_matrix_nonzeros = detail::count_nonzeros(B);
            diagnostics->peak_sparse_matrix_bytes =
                detail::estimate_compressed_sparse_matrix_bytes(
                    B.rows(),
                    B.cols(),
                    diagnostics->final_matrix_nonzeros);
        }
    }

    template<
        int QSpace,
        int QTime,
        class VectorLike,
        class FESpaceType>
    requires la::concepts::VectorLike<VectorLike>
    void assemble_vec_f_with_quadrature_cache_2d(
        VectorLike& f,
        const FESpaceType& space,
        const MainAssemblyQuadratureCache2D<QSpace, QTime, FESpaceType>&
            q_cache,
        detail::AssemblySpaceCache<FESpaceType>& fe_cache,
        double zero_tol = 1e-15,
        detail::AssemblyKernelDiagnostics* diagnostics = nullptr)
    {
        using QCache =
            MainAssemblyQuadratureCache2D<QSpace, QTime, FESpaceType>;
        using Tables = typename QCache::Tables;
        constexpr int dofs_per_cell = QCache::dofs_per_cell;
        constexpr int n_q = QCache::n_q;

        const auto& dof_handler = space.dof_handler_ref();
        const auto& active_cells = space.active_cells();
        [[maybe_unused]] const int n_active_cells =
            static_cast<int>(active_cells.size());

        f.resize(dof_handler.n_true_dofs());
        f.set_zero();

#if defined(ADAPPARABOLICFEM_HAS_OPENMP) && ADAPPARABOLICFEM_HAS_OPENMP
        if (detail::should_use_openmp_for_cell_assembly(n_active_cells))
        {
            for (const int cell_id : active_cells)
                (void)fe_cache.dof_expansion(cell_id);

            auto thread_local_vectors =
                detail::make_thread_local_vectors<VectorLike>(
                    detail::openmp_thread_count(),
                    f.size());
            std::exception_ptr error;

#pragma omp parallel
            {
                try
                {
                    const int thread_id = omp_get_thread_num();
                    auto& local_rhs =
                        thread_local_vectors[static_cast<std::size_t>(thread_id)];
                    la::local::FixedLocalVector<dofs_per_cell> local;

#pragma omp for schedule(static)
                    for (int item_index = 0; item_index < n_active_cells; ++item_index)
                    {
                        const int cell_id =
                            active_cells[static_cast<std::size_t>(item_index)];
                        const auto& expanded = fe_cache.dof_expansion(cell_id);
                        const auto& cell = q_cache.cell(cell_id);

                        detail::zero_local_vector(local);
                        for (int q = 0; q < n_q; ++q)
                        {
                            const auto& qp =
                                cell.qpoints[static_cast<std::size_t>(q)];
                            const auto& values = Tables::values_on_cell_qp(q);
                            for (int i = 0; i < dofs_per_cell; ++i)
                                local[i] += qp.ell * values[i] * qp.dmu;
                        }

                        scatter_vector(local_rhs, local, expanded, zero_tol);
                    }
                }
                catch (...)
                {
#pragma omp critical(adap_parabolic_fem_parallel_error)
                    {
                        if (!error)
                            error = std::current_exception();
                    }
                }
            }

            detail::rethrow_parallel_exception(error);
            detail::reduce_thread_local_vectors(f, thread_local_vectors);
        }
        else
#endif
        {
            la::local::FixedLocalVector<dofs_per_cell> local;

            for (const int cell_id : active_cells)
            {
                const auto& expanded = fe_cache.dof_expansion(cell_id);
                const auto& cell = q_cache.cell(cell_id);

                detail::zero_local_vector(local);
                for (int q = 0; q < n_q; ++q)
                {
                    const auto& qp = cell.qpoints[static_cast<std::size_t>(q)];
                    const auto& values = Tables::values_on_cell_qp(q);
                    for (int i = 0; i < dofs_per_cell; ++i)
                        local[i] += qp.ell * values[i] * qp.dmu;
                }

                scatter_vector(f, local, expanded, zero_tol);
            }
        }

        if (diagnostics != nullptr)
        {
            diagnostics->active_cells =
                static_cast<std::size_t>(n_active_cells);
            diagnostics->quadrature_points =
                static_cast<std::size_t>(n_active_cells) *
                static_cast<std::size_t>(n_q);
            diagnostics->scalar_basis_evaluations =
                diagnostics->quadrature_points *
                static_cast<std::size_t>(dofs_per_cell);
            diagnostics->source_evaluations = diagnostics->quadrature_points;
        }
    }
}

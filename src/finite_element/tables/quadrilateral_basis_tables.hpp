#pragma once

#include <array>
#include <cstddef>

#include "../basis/functions/quadrilateral_lagrange.hpp"
#include "quadrature/reference_quadrature.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER [[gnu::noinline, gnu::cold]]
#else
#define ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
#endif

// -----------------------------------------------------------------------------
// Pretabulated basis-function values and gradients for 1+1D quadrilateral
// space-time elements on the reference cell [0,1] x [0,1].
//
// Cell quadrature:
//   tensor-product Gauss-Legendre(QSpace) x Gauss-Legendre(QTime)
//
// Bottom-face quadrature (for initial conditions):
//   Gauss-Legendre(QSpace) on x in [0,1], evaluated at t = 0
//
// Stored tables:
//   - cell_values[q][i]            = phi_i(x_q, t_q)
//   - cell_gradients[q][i]         = grad phi_i(x_q, t_q) in reference coords
//   - bottom_values[q][i]          = phi_i(x_q, 0)
//   - bottom_gradients[q][i]       = grad phi_i(x_q, 0) in reference coords
//
// Reduced bottom-trace tables:
//   - bottom_trace_values[q][i_s]      = phi_(i_s,0)(x_q, 0)
//   - bottom_trace_gradients[q][i_s]   = grad phi_(i_s,0)(x_q, 0)
// where i_s is the spatial node index and the local space-time basis index is
//   local = i_s * (Q+1) + 0
// -----------------------------------------------------------------------------
namespace finite_element::tables
{
    namespace detail
    {
        template<class CellValuesTable, class Values, bool UseRuntimeStorage>
        struct QuadrilateralCellValuesStorage;

        template<class CellValuesTable, class Values>
        struct QuadrilateralCellValuesStorage<CellValuesTable, Values, false>
        {
            CellValuesTable table;

            constexpr explicit QuadrilateralCellValuesStorage(
                const CellValuesTable& initial_table)
                : table(initial_table)
            {}

            constexpr const Values& operator[](std::size_t q) const
            {
                return table[q];
            }

            constexpr const Values& operator[](int q) const
            {
                return table[static_cast<std::size_t>(q)];
            }
        };

        template<class CellValuesTable, class Values>
        struct QuadrilateralCellValuesStorage<CellValuesTable, Values, true>
        {
            using RuntimeTable = const CellValuesTable& (*)();

            RuntimeTable runtime_table;

            constexpr explicit QuadrilateralCellValuesStorage(
                RuntimeTable runtime_table_in)
                : runtime_table(runtime_table_in)
            {}

            const Values& operator[](std::size_t q) const
            {
                return runtime_table()[q];
            }

            const Values& operator[](int q) const
            {
                return runtime_table()[static_cast<std::size_t>(q)];
            }
        };

        template<class CellGradientsTable, class Gradients, bool UseRuntimeStorage>
        struct QuadrilateralCellGradientsStorage;

        template<class CellGradientsTable, class Gradients>
        struct QuadrilateralCellGradientsStorage<CellGradientsTable, Gradients, false>
        {
            CellGradientsTable table;

            constexpr explicit QuadrilateralCellGradientsStorage(
                const CellGradientsTable& initial_table)
                : table(initial_table)
            {}

            constexpr const Gradients& operator[](std::size_t q) const
            {
                return table[q];
            }

            constexpr const Gradients& operator[](int q) const
            {
                return table[static_cast<std::size_t>(q)];
            }
        };

        template<class CellGradientsTable, class Gradients>
        struct QuadrilateralCellGradientsStorage<CellGradientsTable, Gradients, true>
        {
            using RuntimeTable = const CellGradientsTable& (*)();

            RuntimeTable runtime_table;

            constexpr explicit QuadrilateralCellGradientsStorage(
                RuntimeTable runtime_table_in)
                : runtime_table(runtime_table_in)
            {}

            const Gradients& operator[](std::size_t q) const
            {
                return runtime_table()[q];
            }

            const Gradients& operator[](int q) const
            {
                return runtime_table()[static_cast<std::size_t>(q)];
            }
        };
    }

    template<int P, int Q, int QSpace, int QTime, typename SpatialNodes, typename TemporalNodes>
    struct QuadrilateralBasisTables
    {
        static_assert(P >= 1, "QuadrilateralBasisTables requires P >= 1.");
        static_assert(Q >= 1, "QuadrilateralBasisTables requires Q >= 1.");
        static_assert(QSpace >= 1, "QuadrilateralBasisTables requires QSpace >= 1.");
        static_assert(QTime  >= 1, "QuadrilateralBasisTables requires QTime >= 1.");

        using Basis = basis::functions::QuadrilateralLagrangeBasis<P, Q, SpatialNodes, TemporalNodes>;

        static constexpr int dim         = 2;
        static constexpr int n_basis     = Basis::N;
        static constexpr int n_space_dof = Basis::N_space;
        static constexpr int n_time_dof  = Basis::N_time;

        using CellPoint   = std::array<double, 2>;
        using BottomPoint = std::array<double, 1>;
        
        static constexpr auto cell_rule =
            quadrature::reference::make_reference_quadrilateral_space_time_quadrature<QSpace, QTime>();

        static constexpr auto bottom_rule =
            quadrature::reference::make_reference_interval_quadrature<QSpace>();

        static constexpr int n_cell_q   = decltype(cell_rule)::n_points;
        static constexpr int n_bottom_q = decltype(bottom_rule)::n_points;
        static constexpr std::size_t n_basis_size = static_cast<std::size_t>(n_basis);
        static constexpr std::size_t n_space_dof_size = static_cast<std::size_t>(n_space_dof);
        static constexpr std::size_t n_cell_q_size = static_cast<std::size_t>(n_cell_q);
        static constexpr std::size_t n_bottom_q_size = static_cast<std::size_t>(n_bottom_q);

        using Values    = std::array<double, n_basis_size>;
        using Gradients = std::array<std::array<double, dim>, n_basis_size>;

        using TraceValues    = std::array<double, n_space_dof_size>;
        using TraceGradients = std::array<std::array<double, dim>, n_space_dof_size>;

        using CellValuesTable         = std::array<Values, n_cell_q_size>;
        using CellGradientsTable      = std::array<Gradients, n_cell_q_size>;
        using BottomValuesTable       = std::array<Values, n_bottom_q_size>;
        using BottomGradientsTable    = std::array<Gradients, n_bottom_q_size>;
        using BottomTraceValuesTable  = std::array<TraceValues, n_bottom_q_size>;
        using BottomTraceGradsTable   = std::array<TraceGradients, n_bottom_q_size>;

        static constexpr std::size_t index(int i) noexcept
        {
            return static_cast<std::size_t>(i);
        }

        static constexpr int trace_local_index(int i_space) noexcept
        {
            return i_space * n_time_dof;
        }

    private:
        static constexpr CellValuesTable build_cell_values()
        {
            CellValuesTable table{};
            for (int q = 0; q < n_cell_q; ++q)
                table[index(q)] = Basis::eval_all(cell_rule.points[index(q)]);
            return table;
        }

        static constexpr CellGradientsTable build_cell_gradients()
        {
            CellGradientsTable table{};
            for (int q = 0; q < n_cell_q; ++q)
                table[index(q)] = Basis::grad_all(cell_rule.points[index(q)]);
            return table;
        }

        static constexpr BottomValuesTable build_bottom_values()
        {
            BottomValuesTable table{};

            for (int q = 0; q < n_bottom_q; ++q)
            {
                const double x = bottom_rule.points[index(q)][0U];
                table[index(q)] = Basis::eval_all(CellPoint{x, 0.0});
            }

            return table;
        }

        static constexpr BottomGradientsTable build_bottom_gradients()
        {
            BottomGradientsTable table{};

            for (int q = 0; q < n_bottom_q; ++q)
            {
                const double x = bottom_rule.points[index(q)][0U];
                table[index(q)] = Basis::grad_all(CellPoint{x, 0.0});
            }

            return table;
        }

        static constexpr BottomTraceValuesTable build_bottom_trace_values()
        {
            BottomTraceValuesTable table{};

            for (int q = 0; q < n_bottom_q; ++q)
            {
                const double x = bottom_rule.points[index(q)][0U];
                const auto vals = Basis::eval_all(CellPoint{x, 0.0});

                for (int i_space = 0; i_space < n_space_dof; ++i_space)
                    table[index(q)][index(i_space)] = vals[index(trace_local_index(i_space))];
            }

            return table;
        }

        static constexpr BottomTraceGradsTable build_bottom_trace_gradients()
        {
            BottomTraceGradsTable table{};

            for (int q = 0; q < n_bottom_q; ++q)
            {
                const double x = bottom_rule.points[index(q)][0U];
                const auto grads = Basis::grad_all(CellPoint{x, 0.0});

                for (int i_space = 0; i_space < n_space_dof; ++i_space)
                    table[index(q)][index(i_space)] = grads[index(trace_local_index(i_space))];
            }

            return table;
        }

        static constexpr bool use_runtime_cell_values =
            (n_cell_q * n_basis > 512);
        static constexpr bool use_runtime_cell_gradients =
            (n_cell_q * n_basis * dim > 1024);

        ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
        static CellValuesTable build_cell_values_runtime()
        {
            return build_cell_values();
        }

        ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
        static CellGradientsTable build_cell_gradients_runtime()
        {
            return build_cell_gradients();
        }

        static const CellValuesTable& runtime_cell_values()
        {
            static const CellValuesTable table = build_cell_values_runtime();
            return table;
        }

        static const CellGradientsTable& runtime_cell_gradients()
        {
            static const CellGradientsTable table =
                build_cell_gradients_runtime();
            return table;
        }

        using CellValuesStorage = detail::QuadrilateralCellValuesStorage<
            CellValuesTable,
            Values,
            use_runtime_cell_values>;

        using CellGradientsStorage = detail::QuadrilateralCellGradientsStorage<
            CellGradientsTable,
            Gradients,
            use_runtime_cell_gradients>;

        static constexpr CellValuesStorage make_cell_values_storage()
        {
            if constexpr (use_runtime_cell_values)
                return CellValuesStorage{runtime_cell_values};
            else
                return CellValuesStorage{build_cell_values()};
        }

        static constexpr CellGradientsStorage make_cell_gradients_storage()
        {
            if constexpr (use_runtime_cell_gradients)
                return CellGradientsStorage{runtime_cell_gradients};
            else
                return CellGradientsStorage{build_cell_gradients()};
        }

    public:
        static constexpr CellValuesStorage      cell_values             = make_cell_values_storage();
        static constexpr CellGradientsStorage   cell_gradients          = make_cell_gradients_storage();
        static constexpr BottomValuesTable      bottom_values           = build_bottom_values();
        static constexpr BottomGradientsTable   bottom_gradients        = build_bottom_gradients();
        static constexpr BottomTraceValuesTable bottom_trace_values     = build_bottom_trace_values();
        static constexpr BottomTraceGradsTable  bottom_trace_gradients  = build_bottom_trace_gradients();

        static const Values& values_on_cell_qp(int q)
        {
            return cell_values[index(q)];
        }

        static const Gradients& gradients_on_cell_qp(int q)
        {
            return cell_gradients[index(q)];
        }

        static constexpr const Values& values_on_bottom_qp(int q)
        {
            return bottom_values[index(q)];
        }

        static constexpr const Gradients& gradients_on_bottom_qp(int q)
        {
            return bottom_gradients[index(q)];
        }

        static constexpr const TraceValues& trace_values_on_bottom_qp(int q)
        {
            return bottom_trace_values[index(q)];
        }

        static constexpr const TraceGradients& trace_gradients_on_bottom_qp(int q)
        {
            return bottom_trace_gradients[index(q)];
        }
    };

    template<int P, int Q, int QSpace, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::CellValuesStorage
    QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::cell_values;

    template<int P, int Q, int QSpace, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::CellGradientsStorage
    QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::cell_gradients;

    template<int P, int Q, int QSpace, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::BottomValuesTable
    QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::bottom_values;

    template<int P, int Q, int QSpace, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::BottomGradientsTable
    QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::bottom_gradients;

    template<int P, int Q, int QSpace, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::BottomTraceValuesTable
    QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::bottom_trace_values;

    template<int P, int Q, int QSpace, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::BottomTraceGradsTable
    QuadrilateralBasisTables<P, Q, QSpace, QTime, SpatialNodes, TemporalNodes>::bottom_trace_gradients;
}

#undef ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER

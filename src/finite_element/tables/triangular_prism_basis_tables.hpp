#pragma once

#include <array>
#include <cstddef>

#include "../basis/functions/triangular_prism_lagrange.hpp"
#include "quadrature/reference_quadrature.hpp"

#if defined(__GNUC__) || defined(__clang__)
#define ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER [[gnu::noinline, gnu::cold]]
#else
#define ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
#endif

// -----------------------------------------------------------------------------
// Pretabulated basis-function values and gradients for 2+1D triangular-prism
// space-time elements on the reference prism T_ref x [0,1], where
//
//   T_ref = conv{ (0,0), (1,0), (0,1) }.
//
// Cell quadrature:
//   positive Duffy triangle rule exact through QSpaceDegree on T_ref
//   tensor-product with Gauss-Legendre(QTime) in time
//
// Bottom-face quadrature (for initial conditions):
//   positive Duffy triangle rule exact through QSpaceDegree on T_ref,
//   evaluated at t = 0
//
// Stored tables:
//   - cell_values[q][i]            = phi_i(x_q, y_q, t_q)
//   - cell_gradients[q][i]         = grad phi_i(x_q, y_q, t_q) in reference coords
//   - bottom_values[q][i]          = phi_i(x_q, y_q, 0)
//   - bottom_gradients[q][i]       = grad phi_i(x_q, y_q, 0) in reference coords
//
// Reduced bottom-trace tables:
//   - bottom_trace_values[q][i_s]    = phi_(i_s,0)(x_q, y_q, 0)
//   - bottom_trace_gradients[q][i_s] = grad phi_(i_s,0)(x_q, y_q, 0)
// where i_s is the spatial triangle node index and the local space-time basis
// index is
//   local = i_s * (Q+1) + 0
// -----------------------------------------------------------------------------
namespace finite_element::tables
{
    namespace detail
    {
        template<class CellValuesTable, class Values, bool UseRuntimeStorage>
        struct TriangularPrismCellValuesStorage;

        template<class CellValuesTable, class Values>
        struct TriangularPrismCellValuesStorage<CellValuesTable, Values, false>
        {
            CellValuesTable table;

            constexpr explicit TriangularPrismCellValuesStorage(const CellValuesTable& initial_table)
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
        struct TriangularPrismCellValuesStorage<CellValuesTable, Values, true>
        {
            using RuntimeTable = const CellValuesTable& (*)();

            RuntimeTable runtime_table;

            constexpr explicit TriangularPrismCellValuesStorage(RuntimeTable runtime_table_in)
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
        struct TriangularPrismCellGradientsStorage;

        template<class CellGradientsTable, class Gradients>
        struct TriangularPrismCellGradientsStorage<CellGradientsTable, Gradients, false>
        {
            CellGradientsTable table;

            constexpr explicit TriangularPrismCellGradientsStorage(
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
        struct TriangularPrismCellGradientsStorage<CellGradientsTable, Gradients, true>
        {
            using RuntimeTable = const CellGradientsTable& (*)();

            RuntimeTable runtime_table;

            constexpr explicit TriangularPrismCellGradientsStorage(RuntimeTable runtime_table_in)
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

        template<class Table, class Entry, bool UseRuntimeStorage>
        struct TriangularPrismTableStorage;

        template<class Table, class Entry>
        struct TriangularPrismTableStorage<Table, Entry, false>
        {
            Table table;

            constexpr explicit TriangularPrismTableStorage(
                const Table& initial_table)
                : table(initial_table)
            {}

            constexpr const Entry& operator[](std::size_t q) const
            {
                return table[q];
            }

            constexpr const Entry& operator[](int q) const
            {
                return table[static_cast<std::size_t>(q)];
            }
        };

        template<class Table, class Entry>
        struct TriangularPrismTableStorage<Table, Entry, true>
        {
            using RuntimeTable = const Table& (*)();

            RuntimeTable runtime_table;

            constexpr explicit TriangularPrismTableStorage(
                RuntimeTable runtime_table_in)
                : runtime_table(runtime_table_in)
            {}

            const Entry& operator[](std::size_t q) const
            {
                return runtime_table()[q];
            }

            const Entry& operator[](int q) const
            {
                return runtime_table()[static_cast<std::size_t>(q)];
            }
        };
    }

    template<int P, int Q, int QSpaceDegree, int QTime, typename SpatialNodes, typename TemporalNodes>
    struct TriangularPrismBasisTables
    {
        static_assert(P >= 0, "TriangularPrismBasisTables requires P >= 0.");
        static_assert(Q >= 1, "TriangularPrismBasisTables requires Q >= 1.");
        static_assert(QSpaceDegree >= 1,
                    "TriangularPrismBasisTables requires QSpaceDegree >= 1.");
        static_assert(QTime >= 1, "TriangularPrismBasisTables requires QTime >= 1.");

        using Basis = basis::functions::TriangularPrismLagrangeBasis<P, Q, SpatialNodes, TemporalNodes>;

        static constexpr int dim         = 3;
        static constexpr int n_basis     = Basis::N;
        static constexpr int n_space_dof = Basis::N_tri;
        static constexpr int n_time_dof  = Basis::N_time;

        using CellPoint   = std::array<double, 3>;
        using BottomPoint = std::array<double, 2>;

        static constexpr auto cell_rule =
            quadrature::reference::make_reference_triangular_prism_space_time_quadrature<QSpaceDegree, QTime>();

        static constexpr auto bottom_rule =
            quadrature::reference::make_reference_triangle_quadrature<QSpaceDegree>();

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
                const double y = bottom_rule.points[index(q)][1U];
                table[index(q)] = Basis::eval_all(CellPoint{x, y, 0.0});
            }

            return table;
        }

        static constexpr BottomGradientsTable build_bottom_gradients()
        {
            BottomGradientsTable table{};

            for (int q = 0; q < n_bottom_q; ++q)
            {
                const double x = bottom_rule.points[index(q)][0U];
                const double y = bottom_rule.points[index(q)][1U];
                table[index(q)] = Basis::grad_all(CellPoint{x, y, 0.0});
            }

            return table;
        }

        static constexpr BottomTraceValuesTable build_bottom_trace_values()
        {
            BottomTraceValuesTable table{};

            for (int q = 0; q < n_bottom_q; ++q)
            {
                const double x = bottom_rule.points[index(q)][0U];
                const double y = bottom_rule.points[index(q)][1U];
                const auto vals = Basis::eval_all(CellPoint{x, y, 0.0});

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
                const double y = bottom_rule.points[index(q)][1U];
                const auto grads = Basis::grad_all(CellPoint{x, y, 0.0});

                for (int i_space = 0; i_space < n_space_dof; ++i_space)
                    table[index(q)][index(i_space)] = grads[index(trace_local_index(i_space))];
            }

            return table;
        }

        static constexpr bool use_runtime_cell_values =
            (n_cell_q * n_basis > 4096);
        static constexpr bool use_runtime_cell_gradients =
            (n_cell_q * n_basis * dim > 8192);
        static constexpr bool use_runtime_bottom_values =
            (n_bottom_q * n_basis > 2048);
        static constexpr bool use_runtime_bottom_gradients =
            (n_bottom_q * n_basis * dim > 4096);
        static constexpr bool use_runtime_bottom_trace_values =
            (n_bottom_q * n_space_dof > 512);
        static constexpr bool use_runtime_bottom_trace_gradients =
            (n_bottom_q * n_space_dof * dim > 1536);

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

        ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
        static BottomValuesTable build_bottom_values_runtime()
        {
            return build_bottom_values();
        }

        ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
        static BottomGradientsTable build_bottom_gradients_runtime()
        {
            return build_bottom_gradients();
        }

        ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
        static BottomTraceValuesTable build_bottom_trace_values_runtime()
        {
            return build_bottom_trace_values();
        }

        ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER
        static BottomTraceGradsTable build_bottom_trace_gradients_runtime()
        {
            return build_bottom_trace_gradients();
        }

        static const BottomValuesTable& runtime_bottom_values()
        {
            static const BottomValuesTable table = build_bottom_values_runtime();
            return table;
        }

        static const BottomGradientsTable& runtime_bottom_gradients()
        {
            static const BottomGradientsTable table =
                build_bottom_gradients_runtime();
            return table;
        }

        static const BottomTraceValuesTable& runtime_bottom_trace_values()
        {
            static const BottomTraceValuesTable table =
                build_bottom_trace_values_runtime();
            return table;
        }

        static const BottomTraceGradsTable& runtime_bottom_trace_gradients()
        {
            static const BottomTraceGradsTable table =
                build_bottom_trace_gradients_runtime();
            return table;
        }

        using CellValuesStorage = detail::TriangularPrismCellValuesStorage<
            CellValuesTable,
            Values,
            use_runtime_cell_values>;

        using CellGradientsStorage = detail::TriangularPrismCellGradientsStorage<
            CellGradientsTable,
            Gradients,
            use_runtime_cell_gradients>;

        using BottomValuesStorage = detail::TriangularPrismTableStorage<
            BottomValuesTable,
            Values,
            use_runtime_bottom_values>;

        using BottomGradientsStorage = detail::TriangularPrismTableStorage<
            BottomGradientsTable,
            Gradients,
            use_runtime_bottom_gradients>;

        using BottomTraceValuesStorage = detail::TriangularPrismTableStorage<
            BottomTraceValuesTable,
            TraceValues,
            use_runtime_bottom_trace_values>;

        using BottomTraceGradsStorage = detail::TriangularPrismTableStorage<
            BottomTraceGradsTable,
            TraceGradients,
            use_runtime_bottom_trace_gradients>;

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

        static constexpr BottomValuesStorage make_bottom_values_storage()
        {
            if constexpr (use_runtime_bottom_values)
                return BottomValuesStorage{runtime_bottom_values};
            else
                return BottomValuesStorage{build_bottom_values()};
        }

        static constexpr BottomGradientsStorage make_bottom_gradients_storage()
        {
            if constexpr (use_runtime_bottom_gradients)
                return BottomGradientsStorage{runtime_bottom_gradients};
            else
                return BottomGradientsStorage{build_bottom_gradients()};
        }

        static constexpr BottomTraceValuesStorage make_bottom_trace_values_storage()
        {
            if constexpr (use_runtime_bottom_trace_values)
                return BottomTraceValuesStorage{runtime_bottom_trace_values};
            else
                return BottomTraceValuesStorage{build_bottom_trace_values()};
        }

        static constexpr BottomTraceGradsStorage make_bottom_trace_gradients_storage()
        {
            if constexpr (use_runtime_bottom_trace_gradients)
                return BottomTraceGradsStorage{
                    runtime_bottom_trace_gradients};
            else
                return BottomTraceGradsStorage{
                    build_bottom_trace_gradients()};
        }

    public:
        static constexpr CellValuesStorage      cell_values             = make_cell_values_storage();
        static constexpr CellGradientsStorage   cell_gradients          = make_cell_gradients_storage();
        static constexpr BottomValuesStorage    bottom_values           = make_bottom_values_storage();
        static constexpr BottomGradientsStorage bottom_gradients        = make_bottom_gradients_storage();
        static constexpr BottomTraceValuesStorage bottom_trace_values   = make_bottom_trace_values_storage();
        static constexpr BottomTraceGradsStorage bottom_trace_gradients = make_bottom_trace_gradients_storage();

        static const Values& values_on_cell_qp(int q)
        {
            return cell_values[index(q)];
        }

        static const Gradients& gradients_on_cell_qp(int q)
        {
            return cell_gradients[index(q)];
        }

        static const Values& values_on_bottom_qp(int q)
        {
            return bottom_values[index(q)];
        }

        static const Gradients& gradients_on_bottom_qp(int q)
        {
            return bottom_gradients[index(q)];
        }

        static const TraceValues& trace_values_on_bottom_qp(int q)
        {
            return bottom_trace_values[index(q)];
        }

        static const TraceGradients& trace_gradients_on_bottom_qp(int q)
        {
            return bottom_trace_gradients[index(q)];
        }
    };

    template<int P, int Q, int QSpaceDegree, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::CellValuesStorage
    TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::cell_values;

    template<int P, int Q, int QSpaceDegree, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::CellGradientsStorage
    TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::cell_gradients;

    template<int P, int Q, int QSpaceDegree, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::BottomValuesStorage
    TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::bottom_values;

    template<int P, int Q, int QSpaceDegree, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::BottomGradientsStorage
    TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::bottom_gradients;

    template<int P, int Q, int QSpaceDegree, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::BottomTraceValuesStorage
    TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::bottom_trace_values;

    template<int P, int Q, int QSpaceDegree, int QTime, typename SpatialNodes, typename TemporalNodes>
    constexpr typename TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::BottomTraceGradsStorage
    TriangularPrismBasisTables<P, Q, QSpaceDegree, QTime, SpatialNodes, TemporalNodes>::bottom_trace_gradients;
}

#undef ADAPPARABOLICFEM_RUNTIME_TABLE_BUILDER

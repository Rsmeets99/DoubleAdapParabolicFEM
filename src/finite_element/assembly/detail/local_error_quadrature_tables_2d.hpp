#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#include "quadrature/gauss_legendre_1d.hpp"
#include "quadrature/reference_triangle_duffy.hpp"

namespace finite_element::assembly::detail
{
    struct LocalErrorPatchCellBuildRequest2D
    {
        int patch_id = -1;
        int patch_cell_index = -1;
        int active_slab_cell_ordinal = -1;
    };

    template<class PatchCellType>
    [[nodiscard]] inline int local_error_slab_cell_id_2d(
        const PatchCellType& patch_cell)
    {
        if constexpr (requires { patch_cell.slab_cell_id; })
            return patch_cell.slab_cell_id;
        else
            return patch_cell.slab_local_ordinal;
    }

    template<class PatchCellType>
    [[nodiscard]] inline int local_error_source_cell_id_2d(
        const PatchCellType& patch_cell)
    {
        if constexpr (requires { patch_cell.source_cell_id; })
            return patch_cell.source_cell_id;
        else
            return patch_cell.source_y_cell_id;
    }

    template<int QSpaceDegree, int QTime, class PatchFluxSpaceType>
    class LocalErrorRTCellQuadratureCache2D
    {
    public:
        using FluxSpaceType = PatchFluxSpaceType;
        using GT            = typename FluxSpaceType::GT;
        using PatchType     = typename FluxSpaceType::Patch;
        using Types         = typename PatchType::Types;

        using SpatialReferencePoint = typename FluxSpaceType::SpatialReferencePoint;
        using SpaceTimeReferencePoint = typename FluxSpaceType::SpaceTimeReferencePoint;
        using SpaceTimePoint = typename Types::SpaceTimePoint;
        using VectorValue = typename FluxSpaceType::VectorValue;
        using RTBasisValues = typename FluxSpaceType::LocalValues;
        using RTBasisDivergences = typename FluxSpaceType::LocalDivergences;

        static_assert(GT::dim_space_v == 2,
                      "LocalErrorRTCellQuadratureCache2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "LocalErrorRTCellQuadratureCache2D requires dim_time_v == 1.");
        static_assert(QSpaceDegree >= 0,
                      "LocalErrorRTCellQuadratureCache2D requires QSpaceDegree >= 0.");
        static_assert(QTime >= 1 && QTime <= 12,
                      "LocalErrorRTCellQuadratureCache2D supports 1 <= QTime <= 12.");

        static constexpr int space_degree_v = QSpaceDegree;
        static constexpr int time_order_v = QTime;
        static constexpr int duffy_order_v = (QSpaceDegree + 3) / 2;
        static_assert(duffy_order_v >= 1 && duffy_order_v <= 12,
                      "LocalErrorRTCellQuadratureCache2D supports spatial exactness degree <= 22.");

        static constexpr int n_spatial_quadrature_points_v =
            duffy_order_v * duffy_order_v;
        static constexpr int n_time_quadrature_points_v = QTime;
        static constexpr int n_quadrature_points_v =
            n_spatial_quadrature_points_v * n_time_quadrature_points_v;

        static constexpr auto time_rule =
            quadrature::gauss_legendre::gauss_legendre_rule_1d<QTime>;

        struct QuadraturePointData
        {
            SpatialReferencePoint spatial_reference_point{};
            double time_reference_point = 0.0;
            SpaceTimeReferencePoint reference_point{};
            SpaceTimePoint physical_point{};

            RTBasisValues rt_basis_values{};
            RTBasisDivergences rt_basis_divergences{};

            double triangle_reference_weight = 0.0;
            double time_reference_weight = 0.0;
            double reference_weight = 0.0;
            double spatial_jacobian_measure = 0.0;
            double time_jacobian_measure = 0.0;
            double jacobian_measure = 0.0;
            double jacobian_weight = 0.0;
        };

        struct CellData
        {
            int slab_id = -1;
            int slab_cell_id = -1;
            double spatial_jacobian_measure = 0.0;
            double time_jacobian_measure = 0.0;
            double jacobian_measure = 0.0;
            std::array<QuadraturePointData, n_quadrature_points_v> points{};
        };

        LocalErrorRTCellQuadratureCache2D() = default;

        explicit LocalErrorRTCellQuadratureCache2D(
            const FluxSpaceType& flux_space)
        {
            add_flux_space_(flux_space);
        }

        explicit LocalErrorRTCellQuadratureCache2D(
            const std::vector<FluxSpaceType>& flux_spaces)
        {
            prepare_from_flux_spaces(flux_spaces);
            for (int request_id = 0;
                 request_id < n_build_requests();
                 ++request_id)
            {
                fill_build_request(request_id, flux_spaces);
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
                collect_flux_space_(
                    static_cast<int>(patch_id),
                    flux_spaces[patch_id]);
            }
        }

        void prepare_from_patch_cells(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<LocalErrorPatchCellBuildRequest2D>& requests)
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
                        "LocalErrorRTCellQuadratureCache2D: patch id out of range.");
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

        void fill_build_request(
            int request_id,
            const std::vector<FluxSpaceType>& flux_spaces)
        {
            check_build_request_index_(request_id);
            const auto& request =
                build_requests_[static_cast<std::size_t>(request_id)];
            fill_cell_(
                flux_spaces[static_cast<std::size_t>(request.patch_id)],
                request.patch_cell_index,
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

        [[nodiscard]] double average_patch_cells_per_slab_cell() const noexcept
        {
            if (cells_.empty())
                return 0.0;
            return static_cast<double>(requested_patch_cells_) /
                   static_cast<double>(cells_.size());
        }

        [[nodiscard]] std::size_t rt_basis_qpoint_fills() const noexcept
        {
            return cells_.size() *
                   static_cast<std::size_t>(n_quadrature_points_v);
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return cells_.capacity() * sizeof(CellData) +
                   build_requests_.capacity() * (5 * sizeof(int)) +
                   cell_index_by_key_.size() *
                       (sizeof(std::pair<const std::pair<int, int>, int>) +
                        3 * sizeof(void*));
        }

        [[nodiscard]] const CellData& cell(
            int slab_id,
            int slab_cell_id) const
        {
            const auto it = cell_index_by_key_.find({slab_id, slab_cell_id});
            if (it == cell_index_by_key_.end())
            {
                throw std::runtime_error(
                    "LocalErrorRTCellQuadratureCache2D: slab cell not cached.");
            }

            return cells_[static_cast<std::size_t>(it->second)];
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
                    const auto& patch_cell =
                        flux_space.patch().cell(patch_cell_index);
                    keys.emplace(
                        slab_id,
                        local_error_slab_cell_id_2d(patch_cell));
                }
            }
            return keys.size();
        }

        [[nodiscard]] static std::size_t
        count_unique_slab_cells_from_requests_(
            const std::vector<FluxSpaceType>& flux_spaces,
            const std::vector<LocalErrorPatchCellBuildRequest2D>& requests)
        {
            std::set<std::pair<int, int>> keys;
            for (const auto& request : requests)
            {
                if (request.patch_id < 0 ||
                    request.patch_id >= static_cast<int>(flux_spaces.size()))
                {
                    throw std::runtime_error(
                        "LocalErrorRTCellQuadratureCache2D: patch id out of range.");
                }
                const auto& flux_space =
                    flux_spaces[static_cast<std::size_t>(request.patch_id)];
                if (request.patch_cell_index < 0 ||
                    request.patch_cell_index >= flux_space.n_patch_cells())
                {
                    throw std::runtime_error(
                        "LocalErrorRTCellQuadratureCache2D: patch cell index out of range.");
                }
                const int slab_id = flux_space.patch().slab_id;
                const auto& patch_cell =
                    flux_space.patch().cell(request.patch_cell_index);
                keys.emplace(
                    slab_id,
                    local_error_slab_cell_id_2d(patch_cell));
            }
            return keys.size();
        }

        void add_flux_space_(const FluxSpaceType& flux_space)
        {
            requested_patch_cells_ += flux_space.n_patch_cells();

            const int slab_id = flux_space.patch().slab_id;
            for (int patch_cell_index = 0;
                 patch_cell_index < flux_space.n_patch_cells();
                 ++patch_cell_index)
            {
                const auto& patch_cell =
                    flux_space.patch().cell(patch_cell_index);
                const int slab_cell_id =
                    local_error_slab_cell_id_2d(patch_cell);
                const auto key =
                    std::pair<int, int>{slab_id, slab_cell_id};
                if (cell_index_by_key_.find(key) != cell_index_by_key_.end())
                    continue;

                const int cache_id = static_cast<int>(cells_.size());
                cell_index_by_key_.emplace(key, cache_id);
                cells_.push_back(CellData{});
                fill_cell_(
                    flux_space,
                    patch_cell_index,
                    cells_.back());
            }
        }

        void collect_flux_space_(
            int patch_id,
            const FluxSpaceType& flux_space)
        {
            requested_patch_cells_ += flux_space.n_patch_cells();

            const int slab_id = flux_space.patch().slab_id;
            for (int patch_cell_index = 0;
                 patch_cell_index < flux_space.n_patch_cells();
                 ++patch_cell_index)
            {
                const auto& patch_cell =
                    flux_space.patch().cell(patch_cell_index);
                const int slab_cell_id =
                    local_error_slab_cell_id_2d(patch_cell);
                const auto key =
                    std::pair<int, int>{slab_id, slab_cell_id};
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
                    "LocalErrorRTCellQuadratureCache2D: patch cell index out of range.");
            }

            ++requested_patch_cells_;

            const int slab_id = flux_space.patch().slab_id;
            const auto& patch_cell =
                flux_space.patch().cell(patch_cell_index);
                const int slab_cell_id =
                    local_error_slab_cell_id_2d(patch_cell);
                const auto key =
                    std::pair<int, int>{slab_id, slab_cell_id};
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
                    "LocalErrorRTCellQuadratureCache2D: build request index out of range.");
            }
        }

        static void fill_cell_(
            const FluxSpaceType& flux_space,
            int patch_cell_index,
            CellData& cell_data)
        {
            const auto& patch_cell = flux_space.patch().cell(patch_cell_index);
            const auto map = flux_space.physical_map_for_patch_cell(patch_cell_index);

            cell_data.slab_id = flux_space.patch().slab_id;
            cell_data.slab_cell_id =
                local_error_slab_cell_id_2d(patch_cell);
            cell_data.spatial_jacobian_measure =
                FluxSpaceType::PiolaBasis::jacobian_measure(map);
            cell_data.time_jacobian_measure =
                std::abs(flux_space.time_length());
            cell_data.jacobian_measure =
                cell_data.spatial_jacobian_measure *
                cell_data.time_jacobian_measure;

            std::array<double, QTime> time_reference_points{};
            std::array<double, QTime> time_reference_weights{};
            std::array<double, QTime> time_physical_points{};
            std::array<typename FluxSpaceType::TimeValues, QTime>
                flux_time_values{};
            for (int qt = 0; qt < QTime; ++qt)
            {
                const double t_ref = time_rule.points[qt][0];
                time_reference_points[static_cast<std::size_t>(qt)] = t_ref;
                time_reference_weights[static_cast<std::size_t>(qt)] =
                    time_rule.weights[qt];
                time_physical_points[static_cast<std::size_t>(qt)] =
                    flux_space.map_time_to_physical(t_ref);
                FluxSpaceType::evaluate_time_basis(
                    t_ref,
                    flux_time_values[static_cast<std::size_t>(qt)]);
            }

            int spatial_qp_id = 0;
            quadrature::reference::for_each_reference_triangle_duffy_point<
                QSpaceDegree>(
                [&](const double x,
                    const double y,
                    const double triangle_weight)
                {
                    const SpatialReferencePoint x_ref{x, y};
                    const auto xy =
                        FluxSpaceType::PiolaBasis::map_to_physical(
                            map,
                            x_ref);
                    const auto spatial_rt_values =
                        FluxSpaceType::PiolaBasis::eval_all(map, x_ref);
                    const auto spatial_rt_divergences =
                        FluxSpaceType::PiolaBasis::div_all(map, x_ref);

                    for (int qt = 0; qt < QTime; ++qt)
                    {
                        const int qp_id =
                            spatial_qp_id * QTime + qt;
                        auto& qp =
                            cell_data.points[static_cast<std::size_t>(qp_id)];

                        const double t_ref =
                            time_reference_points[
                                static_cast<std::size_t>(qt)];
                        const double time_weight =
                            time_reference_weights[
                                static_cast<std::size_t>(qt)];
                        const double t_phys =
                            time_physical_points[
                                static_cast<std::size_t>(qt)];

                        qp.spatial_reference_point = x_ref;
                        qp.time_reference_point = t_ref;
                        qp.reference_point =
                            SpaceTimeReferencePoint{x_ref[0], x_ref[1], t_ref};
                        qp.physical_point =
                            SpaceTimePoint{xy[0], xy[1], t_phys};

                        const auto& time_values =
                            flux_time_values[
                                static_cast<std::size_t>(qt)];
                        for (int spatial_local_dof = 0;
                             spatial_local_dof <
                                 FluxSpaceType::spatial_local_dofs_v;
                             ++spatial_local_dof)
                        {
                            const auto& spatial_value =
                                spatial_rt_values[
                                    static_cast<std::size_t>(
                                        spatial_local_dof)];
                            const double spatial_divergence =
                                spatial_rt_divergences[
                                    static_cast<std::size_t>(
                                        spatial_local_dof)];
                            for (int time_dof = 0;
                                 time_dof < FluxSpaceType::n_time_dofs_v;
                                 ++time_dof)
                            {
                                const int local_id =
                                    flux_space.local_dof_index(
                                        spatial_local_dof,
                                        time_dof);
                                const double time_value =
                                    time_values[
                                        static_cast<std::size_t>(time_dof)];
                                qp.rt_basis_values[
                                    static_cast<std::size_t>(local_id)] =
                                    VectorValue{
                                        spatial_value[0] * time_value,
                                        spatial_value[1] * time_value
                                    };
                                qp.rt_basis_divergences[
                                    static_cast<std::size_t>(local_id)] =
                                    spatial_divergence * time_value;
                            }
                        }

                        qp.triangle_reference_weight = triangle_weight;
                        qp.time_reference_weight = time_weight;
                        qp.reference_weight = triangle_weight * time_weight;
                        qp.spatial_jacobian_measure =
                            cell_data.spatial_jacobian_measure;
                        qp.time_jacobian_measure =
                            cell_data.time_jacobian_measure;
                        qp.jacobian_measure = cell_data.jacobian_measure;
                        qp.jacobian_weight =
                            qp.reference_weight * qp.jacobian_measure;
                    }

                    ++spatial_qp_id;
                });

            if (spatial_qp_id != n_spatial_quadrature_points_v)
            {
                throw std::runtime_error(
                    "LocalErrorRTCellQuadratureCache2D: unexpected Duffy quadrature point count.");
            }
        }
    };

    template<int QSpaceDegree, int QTime, class PatchFluxSpaceType, class PatchScalarSpaceType>
    class LocalErrorQuadratureTables2D
    {
    public:
        using FluxSpaceType   = PatchFluxSpaceType;
        using ScalarSpaceType = PatchScalarSpaceType;
        using GT              = typename FluxSpaceType::GT;
        using PatchType       = typename FluxSpaceType::Patch;
        using Types           = typename PatchType::Types;

        using SpatialReferencePoint = typename FluxSpaceType::SpatialReferencePoint;
        using SpaceTimeReferencePoint = typename FluxSpaceType::SpaceTimeReferencePoint;
        using SpatialPoint = typename Types::SpatialPoint;
        using SpaceTimePoint = typename Types::SpaceTimePoint;
        using SpatialGradient = typename Types::SpatialPoint;
        using VectorValue = typename FluxSpaceType::VectorValue;

        using RTBasisValues = typename FluxSpaceType::LocalValues;
        using RTBasisDivergences = typename FluxSpaceType::LocalDivergences;
        using ScalarBasisValues = typename ScalarSpaceType::LocalValues;
        using RTCellCache =
            LocalErrorRTCellQuadratureCache2D<
                QSpaceDegree,
                QTime,
                FluxSpaceType>;

        static_assert(GT::dim_space_v == 2,
                      "LocalErrorQuadratureTables2D requires dim_space_v == 2.");
        static_assert(GT::dim_time_v == 1,
                      "LocalErrorQuadratureTables2D requires dim_time_v == 1.");
        static_assert(QSpaceDegree >= 0,
                      "LocalErrorQuadratureTables2D requires QSpaceDegree >= 0.");
        static_assert(QTime >= 1 && QTime <= 12,
                      "LocalErrorQuadratureTables2D supports 1 <= QTime <= 12.");

        static constexpr int space_degree_v = QSpaceDegree;
        static constexpr int time_order_v = QTime;
        static constexpr int duffy_order_v = (QSpaceDegree + 3) / 2;
        static_assert(duffy_order_v >= 1 && duffy_order_v <= 12,
                      "LocalErrorQuadratureTables2D supports spatial exactness degree <= 22.");

        static constexpr int n_spatial_quadrature_points_v =
            duffy_order_v * duffy_order_v;
        static constexpr int n_time_quadrature_points_v = QTime;
        static constexpr int n_quadrature_points_v =
            n_spatial_quadrature_points_v * n_time_quadrature_points_v;

        static constexpr auto time_rule =
            quadrature::gauss_legendre::gauss_legendre_rule_1d<QTime>;

        struct QuadraturePointData
        {
            SpatialReferencePoint spatial_reference_point{};
            double time_reference_point = 0.0;
            SpaceTimeReferencePoint reference_point{};
            SpaceTimePoint physical_point{};

            const RTBasisValues* rt_basis_values_ptr = nullptr;
            const RTBasisDivergences* rt_basis_divergences_ptr = nullptr;
            bool shared_rt_data = false;
            ScalarBasisValues scalar_basis_values{};

            double partition_of_unity_value = 0.0;
            SpatialGradient partition_of_unity_gradient{};

            double triangle_reference_weight = 0.0;
            double time_reference_weight = 0.0;
            double reference_weight = 0.0;
            double spatial_jacobian_measure = 0.0;
            double time_jacobian_measure = 0.0;
            double jacobian_measure = 0.0;
            double jacobian_weight = 0.0;

            [[nodiscard]] bool uses_shared_rt_data() const noexcept
            {
                return shared_rt_data;
            }

            [[nodiscard]] const RTBasisValues& rt_basis_values() const
            {
                if (rt_basis_values_ptr == nullptr)
                {
                    throw std::runtime_error(
                        "LocalErrorQuadratureTables2D: RT basis values not initialized.");
                }
                return *rt_basis_values_ptr;
            }

            [[nodiscard]] const RTBasisDivergences&
            rt_basis_divergences() const
            {
                if (rt_basis_divergences_ptr == nullptr)
                {
                    throw std::runtime_error(
                        "LocalErrorQuadratureTables2D: RT basis divergences not initialized.");
                }
                return *rt_basis_divergences_ptr;
            }
        };

        struct CellData
        {
            int patch_cell_index = -1;
            int slab_cell_id = -1;
            int local_vertex_index = -1;
            double spatial_jacobian_measure = 0.0;
            double time_jacobian_measure = 0.0;
            double jacobian_measure = 0.0;
            std::array<QuadraturePointData, n_quadrature_points_v> points{};
        };

        struct OwnedRTQuadraturePointData
        {
            RTBasisValues rt_basis_values{};
            RTBasisDivergences rt_basis_divergences{};
        };

        LocalErrorQuadratureTables2D(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space)
            : LocalErrorQuadratureTables2D(flux_space, scalar_space, nullptr)
        {}

        LocalErrorQuadratureTables2D(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            const RTCellCache& rt_cell_cache)
            : LocalErrorQuadratureTables2D(
                  flux_space,
                  scalar_space,
                  &rt_cell_cache,
                  nullptr)
        {}

        LocalErrorQuadratureTables2D(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            const RTCellCache& rt_cell_cache,
            const std::vector<int>& patch_cell_indices)
            : LocalErrorQuadratureTables2D(
                  flux_space,
                  scalar_space,
                  &rt_cell_cache,
                  &patch_cell_indices)
        {}

        LocalErrorQuadratureTables2D(const LocalErrorQuadratureTables2D& other)
            : cells_(other.cells_),
              owned_rt_points_(other.owned_rt_points_),
              cell_index_by_patch_cell_(other.cell_index_by_patch_cell_),
              patch_cell_count_(other.patch_cell_count_),
              scalar_basis_qpoint_fills_(
                  other.scalar_basis_qpoint_fills_),
              partition_of_unity_qpoint_fills_(
                  other.partition_of_unity_qpoint_fills_),
              owned_rt_basis_qpoint_fills_(
                  other.owned_rt_basis_qpoint_fills_)
        {
            rewire_owned_rt_pointers_();
        }

        LocalErrorQuadratureTables2D& operator=(
            const LocalErrorQuadratureTables2D& other)
        {
            if (this == &other)
                return *this;

            cells_ = other.cells_;
            owned_rt_points_ = other.owned_rt_points_;
            cell_index_by_patch_cell_ = other.cell_index_by_patch_cell_;
            patch_cell_count_ = other.patch_cell_count_;
            scalar_basis_qpoint_fills_ =
                other.scalar_basis_qpoint_fills_;
            partition_of_unity_qpoint_fills_ =
                other.partition_of_unity_qpoint_fills_;
            owned_rt_basis_qpoint_fills_ =
                other.owned_rt_basis_qpoint_fills_;
            rewire_owned_rt_pointers_();
            return *this;
        }

        LocalErrorQuadratureTables2D(LocalErrorQuadratureTables2D&& other) noexcept
            : cells_(std::move(other.cells_)),
              owned_rt_points_(std::move(other.owned_rt_points_)),
              cell_index_by_patch_cell_(
                  std::move(other.cell_index_by_patch_cell_)),
              patch_cell_count_(other.patch_cell_count_),
              scalar_basis_qpoint_fills_(
                  other.scalar_basis_qpoint_fills_),
              partition_of_unity_qpoint_fills_(
                  other.partition_of_unity_qpoint_fills_),
              owned_rt_basis_qpoint_fills_(
                  other.owned_rt_basis_qpoint_fills_)
        {
            rewire_owned_rt_pointers_();
        }

        LocalErrorQuadratureTables2D& operator=(
            LocalErrorQuadratureTables2D&& other) noexcept
        {
            if (this == &other)
                return *this;

            cells_ = std::move(other.cells_);
            owned_rt_points_ = std::move(other.owned_rt_points_);
            cell_index_by_patch_cell_ =
                std::move(other.cell_index_by_patch_cell_);
            patch_cell_count_ = other.patch_cell_count_;
            scalar_basis_qpoint_fills_ =
                other.scalar_basis_qpoint_fills_;
            partition_of_unity_qpoint_fills_ =
                other.partition_of_unity_qpoint_fills_;
            owned_rt_basis_qpoint_fills_ =
                other.owned_rt_basis_qpoint_fills_;
            rewire_owned_rt_pointers_();
            return *this;
        }

    private:
        std::vector<CellData> cells_{};
        std::vector<OwnedRTQuadraturePointData> owned_rt_points_{};
        std::vector<int> cell_index_by_patch_cell_{};
        int patch_cell_count_ = 0;
        std::size_t scalar_basis_qpoint_fills_ = 0;
        std::size_t partition_of_unity_qpoint_fills_ = 0;
        std::size_t owned_rt_basis_qpoint_fills_ = 0;

        LocalErrorQuadratureTables2D(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            const RTCellCache* rt_cell_cache)
            : LocalErrorQuadratureTables2D(
                  flux_space,
                  scalar_space,
                  rt_cell_cache,
                  nullptr)
        {}

        LocalErrorQuadratureTables2D(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            const RTCellCache* rt_cell_cache,
            const std::vector<int>* patch_cell_indices)
        {
            validate_spaces_(flux_space, scalar_space);

            patch_cell_count_ = flux_space.n_patch_cells();
            cell_index_by_patch_cell_.assign(
                static_cast<std::size_t>(patch_cell_count_),
                -1);

            if (rt_cell_cache == nullptr)
            {
                owned_rt_points_.resize(
                    static_cast<std::size_t>(flux_space.n_patch_cells()) *
                    static_cast<std::size_t>(n_quadrature_points_v));
            }

            const auto add_patch_cell =
                [&](int patch_cell_index)
                {
                    if (patch_cell_index < 0 ||
                        patch_cell_index >= patch_cell_count_)
                    {
                        throw std::runtime_error(
                            "LocalErrorQuadratureTables2D: requested patch cell index out of range.");
                    }
                    auto& mapped_index =
                        cell_index_by_patch_cell_[
                            static_cast<std::size_t>(patch_cell_index)];
                    if (mapped_index >= 0)
                        return;

                    mapped_index = static_cast<int>(cells_.size());
                    cells_.push_back(CellData{});
                    fill_cell_(
                        flux_space,
                        scalar_space,
                        patch_cell_index,
                        rt_cell_cache,
                        cells_.back());
                };

            if (patch_cell_indices == nullptr)
            {
                cells_.reserve(static_cast<std::size_t>(patch_cell_count_));
                for (int patch_cell_index = 0;
                     patch_cell_index < patch_cell_count_;
                     ++patch_cell_index)
                {
                    add_patch_cell(patch_cell_index);
                }
            }
            else
            {
                cells_.reserve(patch_cell_indices->size());
                for (const int patch_cell_index : *patch_cell_indices)
                    add_patch_cell(patch_cell_index);
            }
        }

        public:

        [[nodiscard]] int n_patch_cells() const noexcept
        {
            return patch_cell_count_;
        }

        [[nodiscard]] const std::vector<CellData>& cells() const noexcept
        {
            return cells_;
        }

        [[nodiscard]] const CellData& cell(int patch_cell_index) const
        {
            check_patch_cell_index_(patch_cell_index);
            const int cell_index =
                cell_index_by_patch_cell_[
                    static_cast<std::size_t>(patch_cell_index)];
            return cells_[static_cast<std::size_t>(cell_index)];
        }

        [[nodiscard]] const QuadraturePointData& quadrature_point(
            int patch_cell_index,
            int quadrature_point_id) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_quadrature_point_index_(quadrature_point_id);
            const int cell_index =
                cell_index_by_patch_cell_[
                    static_cast<std::size_t>(patch_cell_index)];
            return cells_[static_cast<std::size_t>(cell_index)]
                .points[static_cast<std::size_t>(quadrature_point_id)];
        }

        [[nodiscard]] std::size_t estimated_memory_bytes() const noexcept
        {
            return cells_.capacity() * sizeof(CellData) +
                   owned_rt_points_.capacity() *
                       sizeof(OwnedRTQuadraturePointData) +
                   cell_index_by_patch_cell_.capacity() * sizeof(int);
        }

        [[nodiscard]] std::size_t constructed_patch_cells() const noexcept
        {
            return cells_.size();
        }

        [[nodiscard]] std::size_t scalar_basis_qpoint_fills() const noexcept
        {
            return scalar_basis_qpoint_fills_;
        }

        [[nodiscard]] std::size_t
        partition_of_unity_qpoint_fills() const noexcept
        {
            return partition_of_unity_qpoint_fills_;
        }

        [[nodiscard]] std::size_t owned_rt_basis_qpoint_fills() const noexcept
        {
            return owned_rt_basis_qpoint_fills_;
        }

        [[nodiscard]] double partition_of_unity_value(
            int patch_cell_index,
            const SpatialReferencePoint& x_ref) const
        {
            check_patch_cell_index_(patch_cell_index);
            return barycentric_value_(
                cell(patch_cell_index).local_vertex_index,
                x_ref);
        }

        void fill_rt_patch_basis_values(
            const FluxSpaceType& flux_space,
            int patch_cell_index,
            int quadrature_point_id,
            std::vector<VectorValue>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_quadrature_point_index_(quadrature_point_id);

            values.assign(
                static_cast<std::size_t>(flux_space.n_dofs()),
                VectorValue{0.0, 0.0});

            const auto& qp =
                quadrature_point(patch_cell_index, quadrature_point_id);
            const auto& map = flux_space.cell_dof_map(patch_cell_index);
            const auto& rt_basis_values = qp.rt_basis_values();

            for (int local_dof_id = 0;
                 local_dof_id < FluxSpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const auto& entry =
                    map[static_cast<std::size_t>(local_dof_id)];
                if (entry.patch_dof_id < 0)
                    continue;

                auto& value = values[static_cast<std::size_t>(entry.patch_dof_id)];
                const auto& local_value =
                    rt_basis_values[static_cast<std::size_t>(local_dof_id)];
                value[0] += static_cast<double>(entry.orientation_sign) *
                            local_value[0];
                value[1] += static_cast<double>(entry.orientation_sign) *
                            local_value[1];
            }
        }

        void fill_rt_patch_basis_divergences(
            const FluxSpaceType& flux_space,
            int patch_cell_index,
            int quadrature_point_id,
            std::vector<double>& divergences) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_quadrature_point_index_(quadrature_point_id);

            divergences.assign(
                static_cast<std::size_t>(flux_space.n_dofs()),
                0.0);

            const auto& qp =
                quadrature_point(patch_cell_index, quadrature_point_id);
            const auto& map = flux_space.cell_dof_map(patch_cell_index);
            const auto& rt_basis_divergences = qp.rt_basis_divergences();

            for (int local_dof_id = 0;
                 local_dof_id < FluxSpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const auto& entry =
                    map[static_cast<std::size_t>(local_dof_id)];
                if (entry.patch_dof_id < 0)
                    continue;

                divergences[static_cast<std::size_t>(entry.patch_dof_id)] +=
                    static_cast<double>(entry.orientation_sign) *
                    rt_basis_divergences[
                        static_cast<std::size_t>(local_dof_id)];
            }
        }

        void fill_scalar_patch_basis_values(
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            int quadrature_point_id,
            std::vector<double>& values) const
        {
            check_patch_cell_index_(patch_cell_index);
            check_quadrature_point_index_(quadrature_point_id);

            values.assign(
                static_cast<std::size_t>(scalar_space.n_dofs()),
                0.0);

            const auto& qp =
                quadrature_point(patch_cell_index, quadrature_point_id);
            for (int local_dof_id = 0;
                 local_dof_id < ScalarSpaceType::local_dofs_v;
                 ++local_dof_id)
            {
                const int patch_dof_id =
                    scalar_space.local_to_patch_dof(
                        patch_cell_index,
                        local_dof_id);
                values[static_cast<std::size_t>(patch_dof_id)] +=
                    qp.scalar_basis_values[
                        static_cast<std::size_t>(local_dof_id)];
            }
        }

    private:
        static void validate_spaces_(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space)
        {
            if (flux_space.n_patch_cells() != scalar_space.n_patch_cells())
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: patch cell count mismatch.");
            }

            const auto& flux_patch = flux_space.patch();
            const auto& scalar_patch = scalar_space.patch();
            if (flux_patch.patch_id != scalar_patch.patch_id ||
                flux_patch.slab_id != scalar_patch.slab_id ||
                flux_patch.spatial_vertex_id != scalar_patch.spatial_vertex_id)
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: flux and scalar spaces are built on different patches.");
            }

            if (flux_space.time_length() <= 0.0 ||
                scalar_space.time_length() <= 0.0)
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: non-positive patch time length.");
            }
        }

        void check_patch_cell_index_(int patch_cell_index) const
        {
            if (patch_cell_index < 0 ||
                patch_cell_index >= patch_cell_count_)
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: patch cell index out of range.");
            }
            if (cell_index_by_patch_cell_[
                    static_cast<std::size_t>(patch_cell_index)] < 0)
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: patch cell not present in this partial table.");
            }
        }

        static void check_quadrature_point_index_(int quadrature_point_id)
        {
            if (quadrature_point_id < 0 ||
                quadrature_point_id >= n_quadrature_points_v)
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: quadrature point index out of range.");
            }
        }

        static double barycentric_value_(
            int local_vertex_index,
            const SpatialReferencePoint& x_ref)
        {
            switch (local_vertex_index)
            {
            case 0:
                return 1.0 - x_ref[0] - x_ref[1];
            case 1:
                return x_ref[0];
            case 2:
                return x_ref[1];
            default:
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: invalid local vertex index.");
            }
        }

        static std::array<double, 2> barycentric_reference_gradient_(
            int local_vertex_index)
        {
            switch (local_vertex_index)
            {
            case 0:
                return {-1.0, -1.0};
            case 1:
                return {1.0, 0.0};
            case 2:
                return {0.0, 1.0};
            default:
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: invalid local vertex index.");
            }
        }

        template<class AffineMap>
        static SpatialGradient map_reference_gradient_to_physical_(
            const AffineMap& map,
            const std::array<double, 2>& grad_ref)
        {
            return SpatialGradient{
                map.invJ00 * grad_ref[0] + map.invJ10 * grad_ref[1],
                map.invJ01 * grad_ref[0] + map.invJ11 * grad_ref[1]
            };
        }

        [[nodiscard]] static std::size_t owned_rt_point_index_(
            int patch_cell_index,
            int quadrature_point_id)
        {
            return static_cast<std::size_t>(patch_cell_index) *
                       static_cast<std::size_t>(n_quadrature_points_v) +
                   static_cast<std::size_t>(quadrature_point_id);
        }

        [[nodiscard]] OwnedRTQuadraturePointData& owned_rt_point_(
            int patch_cell_index,
            int quadrature_point_id)
        {
            const auto index =
                owned_rt_point_index_(patch_cell_index, quadrature_point_id);
            if (index >= owned_rt_points_.size())
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: owned RT qpoint index out of range.");
            }
            return owned_rt_points_[index];
        }

        void rewire_owned_rt_pointers_()
        {
            if (owned_rt_points_.empty())
                return;

            for (auto& cell : cells_)
            {
                if (cell.patch_cell_index < 0)
                    continue;

                for (int qp_id = 0;
                     qp_id < n_quadrature_points_v;
                     ++qp_id)
                {
                    auto& qp =
                        cell.points[static_cast<std::size_t>(qp_id)];
                    if (qp.uses_shared_rt_data())
                        continue;

                    auto& owned_rt_qp =
                        owned_rt_points_[owned_rt_point_index_(
                            cell.patch_cell_index,
                            qp_id)];
                    qp.rt_basis_values_ptr = &owned_rt_qp.rt_basis_values;
                    qp.rt_basis_divergences_ptr =
                        &owned_rt_qp.rt_basis_divergences;
                }
            }
        }

        void fill_cell_(
            const FluxSpaceType& flux_space,
            const ScalarSpaceType& scalar_space,
            int patch_cell_index,
            const RTCellCache* rt_cell_cache,
            CellData& cell_data)
        {
            const auto& patch_cell = flux_space.patch().cell(patch_cell_index);
            const auto map = flux_space.physical_map_for_patch_cell(patch_cell_index);
            const typename RTCellCache::CellData* rt_cell_data = nullptr;
            if (rt_cell_cache != nullptr)
            {
                rt_cell_data =
                    &rt_cell_cache->cell(
                        flux_space.patch().slab_id,
                        local_error_slab_cell_id_2d(patch_cell));
            }

            cell_data.patch_cell_index = patch_cell_index;
            cell_data.slab_cell_id =
                local_error_slab_cell_id_2d(patch_cell);
            cell_data.local_vertex_index = patch_cell.local_vertex_index;
            cell_data.spatial_jacobian_measure =
                FluxSpaceType::PiolaBasis::jacobian_measure(map);
            cell_data.time_jacobian_measure =
                std::abs(flux_space.time_length());
            cell_data.jacobian_measure =
                cell_data.spatial_jacobian_measure *
                cell_data.time_jacobian_measure;

            const auto grad_ref =
                barycentric_reference_gradient_(patch_cell.local_vertex_index);
            const auto grad_phys =
                map_reference_gradient_to_physical_(map, grad_ref);

            std::array<double, QTime> time_reference_points{};
            std::array<double, QTime> time_reference_weights{};
            std::array<double, QTime> time_physical_points{};
            std::array<typename FluxSpaceType::TimeValues, QTime>
                flux_time_values{};
            std::array<typename ScalarSpaceType::TimeValues, QTime>
                scalar_time_values{};
            for (int qt = 0; qt < QTime; ++qt)
            {
                const double t_ref = time_rule.points[qt][0];
                time_reference_points[static_cast<std::size_t>(qt)] = t_ref;
                time_reference_weights[static_cast<std::size_t>(qt)] =
                    time_rule.weights[qt];
                time_physical_points[static_cast<std::size_t>(qt)] =
                    flux_space.map_time_to_physical(t_ref);
                FluxSpaceType::evaluate_time_basis(
                    t_ref,
                    flux_time_values[static_cast<std::size_t>(qt)]);
                ScalarSpaceType::evaluate_time_basis(
                    t_ref,
                    scalar_time_values[static_cast<std::size_t>(qt)]);
            }

            int spatial_qp_id = 0;
            quadrature::reference::for_each_reference_triangle_duffy_point<
                QSpaceDegree>(
                [&](const double x,
                    const double y,
                    const double triangle_weight)
                {
                    const SpatialReferencePoint x_ref{x, y};
                    const double psi =
                        barycentric_value_(
                            patch_cell.local_vertex_index,
                            x_ref);
                    SpatialPoint xy{};
                    if (rt_cell_data == nullptr)
                    {
                        xy =
                            FluxSpaceType::PiolaBasis::map_to_physical(
                                map,
                                x_ref);
                    }
                    typename ScalarSpaceType::SpatialSpace::LocalValues
                        spatial_scalar_values{};
                    ScalarSpaceType::SpatialSpace::evaluate_local_basis(
                        x_ref,
                        spatial_scalar_values);

                    typename FluxSpaceType::SpatialSpace::LocalValues
                        spatial_rt_values{};
                    typename FluxSpaceType::SpatialSpace::LocalDivergences
                        spatial_rt_divergences{};
                    if (rt_cell_data == nullptr)
                    {
                        spatial_rt_values =
                            FluxSpaceType::PiolaBasis::eval_all(map, x_ref);
                        spatial_rt_divergences =
                            FluxSpaceType::PiolaBasis::div_all(map, x_ref);
                    }

                    for (int qt = 0; qt < QTime; ++qt)
                    {
                        const int qp_id =
                            spatial_qp_id * QTime + qt;
                        auto& qp =
                            cell_data.points[static_cast<std::size_t>(qp_id)];

                        if (rt_cell_data != nullptr)
                        {
                            const auto& rt_qp =
                                rt_cell_data->points[
                                    static_cast<std::size_t>(qp_id)];
                            qp.spatial_reference_point =
                                rt_qp.spatial_reference_point;
                            qp.time_reference_point =
                                rt_qp.time_reference_point;
                            qp.reference_point = rt_qp.reference_point;
                            qp.physical_point = rt_qp.physical_point;
                            qp.rt_basis_values_ptr = &rt_qp.rt_basis_values;
                            qp.rt_basis_divergences_ptr =
                                &rt_qp.rt_basis_divergences;
                            qp.shared_rt_data = true;
                            qp.triangle_reference_weight =
                                rt_qp.triangle_reference_weight;
                            qp.time_reference_weight =
                                rt_qp.time_reference_weight;
                            qp.reference_weight = rt_qp.reference_weight;
                            qp.spatial_jacobian_measure =
                                rt_qp.spatial_jacobian_measure;
                            qp.time_jacobian_measure =
                                rt_qp.time_jacobian_measure;
                            qp.jacobian_measure = rt_qp.jacobian_measure;
                            qp.jacobian_weight = rt_qp.jacobian_weight;
                        }
                        else
                        {
                            auto& owned_rt_qp =
                                owned_rt_point_(patch_cell_index, qp_id);
                            qp.rt_basis_values_ptr =
                                &owned_rt_qp.rt_basis_values;
                            qp.rt_basis_divergences_ptr =
                                &owned_rt_qp.rt_basis_divergences;
                            qp.shared_rt_data = false;

                            const double t_ref =
                                time_reference_points[
                                    static_cast<std::size_t>(qt)];
                            const double time_weight =
                                time_reference_weights[
                                    static_cast<std::size_t>(qt)];
                            const double t_phys =
                                time_physical_points[
                                    static_cast<std::size_t>(qt)];

                            qp.spatial_reference_point = x_ref;
                            qp.time_reference_point = t_ref;
                            qp.reference_point =
                                SpaceTimeReferencePoint{x_ref[0], x_ref[1], t_ref};
                            qp.physical_point =
                                SpaceTimePoint{xy[0], xy[1], t_phys};

                            const auto& time_values =
                                flux_time_values[
                                    static_cast<std::size_t>(qt)];
                            for (int spatial_local_dof = 0;
                                 spatial_local_dof <
                                     FluxSpaceType::spatial_local_dofs_v;
                                 ++spatial_local_dof)
                            {
                                const auto& spatial_value =
                                    spatial_rt_values[
                                        static_cast<std::size_t>(
                                            spatial_local_dof)];
                                const double spatial_divergence =
                                    spatial_rt_divergences[
                                        static_cast<std::size_t>(
                                            spatial_local_dof)];
                                for (int time_dof = 0;
                                     time_dof < FluxSpaceType::n_time_dofs_v;
                                     ++time_dof)
                                {
                                    const int local_id =
                                        flux_space.local_dof_index(
                                            spatial_local_dof,
                                            time_dof);
                                    const double time_value =
                                        time_values[
                                            static_cast<std::size_t>(time_dof)];
                                    owned_rt_qp.rt_basis_values[
                                        static_cast<std::size_t>(local_id)] =
                                        VectorValue{
                                            spatial_value[0] * time_value,
                                            spatial_value[1] * time_value
                                        };
                                    owned_rt_qp.rt_basis_divergences[
                                        static_cast<std::size_t>(local_id)] =
                                        spatial_divergence * time_value;
                                }
                            }

                            qp.triangle_reference_weight = triangle_weight;
                            qp.time_reference_weight = time_weight;
                            qp.reference_weight = triangle_weight * time_weight;
                            qp.spatial_jacobian_measure =
                                cell_data.spatial_jacobian_measure;
                            qp.time_jacobian_measure =
                                cell_data.time_jacobian_measure;
                            qp.jacobian_measure = cell_data.jacobian_measure;
                            qp.jacobian_weight =
                                qp.reference_weight * qp.jacobian_measure;
                            ++owned_rt_basis_qpoint_fills_;
                        }

                        const auto& time_values =
                            scalar_time_values[
                                static_cast<std::size_t>(qt)];
                        for (int spatial_local_dof = 0;
                             spatial_local_dof <
                                 ScalarSpaceType::spatial_local_dofs_v;
                             ++spatial_local_dof)
                        {
                            const double spatial_value =
                                spatial_scalar_values[
                                    static_cast<std::size_t>(
                                        spatial_local_dof)];
                            for (int time_dof = 0;
                                 time_dof < ScalarSpaceType::n_time_dofs_v;
                                 ++time_dof)
                            {
                                const int local_id =
                                    scalar_space.local_dof_index(
                                        spatial_local_dof,
                                        time_dof);
                                qp.scalar_basis_values[
                                    static_cast<std::size_t>(local_id)] =
                                    spatial_value *
                                    time_values[
                                        static_cast<std::size_t>(time_dof)];
                            }
                        }

                        qp.partition_of_unity_value = psi;
                        qp.partition_of_unity_gradient = grad_phys;
                        ++scalar_basis_qpoint_fills_;
                        ++partition_of_unity_qpoint_fills_;
                    }

                    ++spatial_qp_id;
                });

            if (spatial_qp_id != n_spatial_quadrature_points_v)
            {
                throw std::runtime_error(
                    "LocalErrorQuadratureTables2D: unexpected Duffy quadrature point count.");
            }
        }
    };
}

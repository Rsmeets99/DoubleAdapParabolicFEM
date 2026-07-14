#pragma once

#include <algorithm>
#include <cmath>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "detail/slab_lookup.hpp"
#include "time_slab.hpp"

namespace finite_element::time_slabs
{
    template<typename GeomTraits, typename FETraits>
    class TimeSlabBuilder;

    template<typename GeomTraits, typename FETraits>
    class TimeSlabSpace
    {
    public:
        using GT              = GeomTraits;
        using FETraitsType    = FETraits;
        using PolicyType      = finite_element::SpaceOnlyPolicy;

        using Types           = mesh::MeshTypes<GeomTraits>;
        using SlabType        = TimeSlab<GeomTraits, FETraits>;
        using SourceSpaceType = FESpace<GeomTraits, FETraits, PolicyType>;

        using SpaceTimePoint  = typename Types::SpaceTimePoint;
        using SpatialPoint    = typename Types::SpatialPoint;
        using TemporalPoint   = typename Types::TemporalPoint;

        struct ActiveCellLocation
        {
            int slab_id = -1;
            int cell_id = -1;

            [[nodiscard]] bool is_valid() const noexcept
            {
                return slab_id >= 0 && cell_id >= 0;
            }
        };

        struct ActiveCellHint
        {
            int slab_id = -1;
            int cell_id = -1;

            [[nodiscard]] bool is_valid() const noexcept
            {
                return slab_id >= 0 && cell_id >= 0;
            }

            void reset() noexcept
            {
                slab_id = -1;
                cell_id = -1;
            }
        };

        struct SlabTimeInfo
        {
            int global_time_id = -1;
            double time = 0.0;
            std::vector<int> source_temporal_vertex_ids{};
        };

        struct SourceCellSlabLocation
        {
            int source_cell_id = -1;
            int slab_id = -1;
            int slab_local_cell_id = -1;
            int slab_time_begin_id = -1;
            int slab_time_end_id = -1;
            double slab_t_begin = 0.0;
            double slab_t_end = 0.0;
            double source_t_begin = 0.0;
            double source_t_end = 0.0;
        };

        struct SourceMeasurePartition
        {
            int source_cell_id = -1;
            double source_measure = 0.0;
            double slab_measure_sum = 0.0;
            int slab_cell_count = 0;

            [[nodiscard]] double difference() const noexcept
            {
                return slab_measure_sum - source_measure;
            }
        };

        struct AlreadySlabbedTrueDofPermutation
        {
            std::vector<int> slab_true_offsets{};
            std::vector<int> source_true_to_slab_true{};
            std::vector<int> slab_true_to_source_true{};

            [[nodiscard]] int n_source_true_dofs() const noexcept
            {
                return static_cast<int>(source_true_to_slab_true.size());
            }

            [[nodiscard]] int n_flat_slab_true_dofs() const noexcept
            {
                return static_cast<int>(slab_true_to_source_true.size());
            }

            [[nodiscard]] bool is_bijection() const noexcept
            {
                for (const int id : source_true_to_slab_true)
                {
                    if (id < 0 ||
                        static_cast<std::size_t>(id) >=
                            slab_true_to_source_true.size())
                    {
                        return false;
                    }
                }

                for (const int id : slab_true_to_source_true)
                {
                    if (id < 0 ||
                        static_cast<std::size_t>(id) >=
                            source_true_to_slab_true.size())
                    {
                        return false;
                    }
                }

                return true;
            }
        };

        explicit TimeSlabSpace(const SourceSpaceType& source_space)
            : source_space_(&source_space)
        {}

        [[nodiscard]] const SourceSpaceType& source_space() const noexcept
        {
            return *source_space_;
        }

        [[nodiscard]] int n_slabs() const noexcept
        {
            return static_cast<int>(slabs_.size());
        }

        [[nodiscard]] int n_slab_cells() const noexcept
        {
            int total = 0;
            for (const auto& slab : slabs_)
                total += slab.n_active_cells();
            return total;
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return slabs_.empty();
        }

        [[nodiscard]] const std::vector<double>& slab_times() const noexcept
        {
            return slab_times_;
        }

        [[nodiscard]] const std::vector<SlabTimeInfo>& slab_time_infos() const noexcept
        {
            return slab_time_infos_;
        }

        [[nodiscard]] const SlabTimeInfo& slab_time_info(int global_time_id) const
        {
            if (global_time_id < 0 ||
                static_cast<std::size_t>(global_time_id) >=
                    slab_time_infos_.size())
            {
                throw std::runtime_error(
                    "TimeSlabSpace::slab_time_info: global time id out of range.");
            }
            return slab_time_infos_[static_cast<std::size_t>(global_time_id)];
        }

        [[nodiscard]] SlabType& slab(int k)
        {
            check_slab_index_(k);
            return slabs_[static_cast<std::size_t>(k)];
        }

        [[nodiscard]] const SlabType& slab(int k) const
        {
            check_slab_index_(k);
            return slabs_[static_cast<std::size_t>(k)];
        }

        void clear()
        {
            slabs_.clear();
            slab_times_.clear();
            slab_time_infos_.clear();
            source_cell_to_slab_locations_.clear();
        }

        [[nodiscard]] int find_slab(double t) const
        {
            return detail::slab_index_from_time(
                slab_times_,
                t,
                detail::SlabEndpointMode::containing_point);
        }

        [[nodiscard]] ActiveCellLocation find_active_cell(const SpaceTimePoint& p) const
        {
            const int slab_id = find_slab(p[GT::dim_space_v]);
            if (slab_id < 0)
                return {};

            const int cell_id = find_active_cell_on_slab_(slab_id, p);
            if (cell_id < 0)
                return {};

            return {slab_id, cell_id};
        }

        [[nodiscard]] ActiveCellLocation find_active_cell(
            int hint_slab_id,
            int hint_cell_id,
            const SpaceTimePoint& p) const
        {
            if (hint_slab_id >= 0 && hint_slab_id < n_slabs())
            {
                const int cell_id = find_active_cell_on_slab_(
                    hint_slab_id,
                    hint_cell_id,
                    p);

                if (cell_id >= 0)
                    return {hint_slab_id, cell_id};
            }

            return find_active_cell(p);
        }

        [[nodiscard]] ActiveCellLocation find_active_cell(
            const SpaceTimePoint& p,
            ActiveCellHint& hint) const
        {
            if (hint.is_valid() && hint.slab_id < n_slabs())
            {
                const int cell_id = find_active_cell_on_slab_(
                    hint.slab_id,
                    hint.cell_id,
                    p);

                if (cell_id >= 0)
                {
                    hint.cell_id = cell_id;
                    return {hint.slab_id, cell_id};
                }
            }

            const auto loc = find_active_cell(p);
            if (loc.is_valid())
            {
                hint.slab_id = loc.slab_id;
                hint.cell_id = loc.cell_id;
            }
            else
            {
                hint.reset();
            }

            return loc;
        }

        [[nodiscard]] int source_cell_id(int slab_id, int slab_local_cell_id) const
        {
            return slab(slab_id).source_cell_id(slab_local_cell_id);
        }

        [[nodiscard]] const std::vector<SourceCellSlabLocation>&
        slab_locations_for_source_cell(int source_cell_id) const
        {
            if (source_cell_id < 0 ||
                static_cast<std::size_t>(source_cell_id) >=
                    source_cell_to_slab_locations_.size())
            {
                throw std::runtime_error(
                    "TimeSlabSpace::slab_locations_for_source_cell: "
                    "source cell id out of range.");
            }

            return source_cell_to_slab_locations_[
                static_cast<std::size_t>(source_cell_id)];
        }

        [[nodiscard]] const std::vector<std::vector<SourceCellSlabLocation>>&
        source_cell_slab_locations() const noexcept
        {
            return source_cell_to_slab_locations_;
        }

        template<typename Callable>
        void for_each_slab(Callable&& callable)
        {
            for (int k = 0; k < n_slabs(); ++k)
                callable(k, slab(k));
        }

        template<typename Callable>
        void for_each_slab(Callable&& callable) const
        {
            for (int k = 0; k < n_slabs(); ++k)
                callable(k, slab(k));
        }

        [[nodiscard]] int n_true_dofs() const noexcept
        {
            int total = 0;

            for (const auto& slab : slabs_)
                total += slab.fespace_ref().dof_handler_ref().n_true_dofs();

            return total;
        }

        [[nodiscard]] int n_dofs() const noexcept
        {
            int total = 0;

            for (const auto& slab : slabs_)
                total += slab.fespace_ref().dof_handler_ref().n_dofs();

            return total;
        }

        [[nodiscard]] std::vector<SourceMeasurePartition>
        source_measure_partitions() const
        {
            std::vector<SourceMeasurePartition> out;
            out.reserve(source_space().active_cells().size());

            for (const int source_cell_id : source_space().active_cells())
            {
                SourceMeasurePartition partition;
                partition.source_cell_id = source_cell_id;
                partition.source_measure =
                    cell_measure_(source_space().mesh_ref(), source_cell_id);

                if (source_cell_id >= 0 &&
                    static_cast<std::size_t>(source_cell_id) <
                        source_cell_to_slab_locations_.size())
                {
                    const auto& locations =
                        source_cell_to_slab_locations_[
                            static_cast<std::size_t>(source_cell_id)];
                    partition.slab_cell_count =
                        static_cast<int>(locations.size());
                    for (const auto& location : locations)
                    {
                        partition.slab_measure_sum +=
                            cell_measure_(
                                slab(location.slab_id).mesh_ref(),
                                location.slab_local_cell_id);
                    }
                }

                out.push_back(partition);
            }

            return out;
        }

        void assert_measure_partition(double tolerance = 1.0e-12) const
        {
            for (const auto& partition : source_measure_partitions())
            {
                const double scale =
                    std::max(1.0, std::abs(partition.source_measure));
                if (std::abs(partition.difference()) > tolerance * scale)
                {
                    std::ostringstream message;
                    message
                        << "TimeSlabSpace::assert_measure_partition: "
                        << "source cell " << partition.source_cell_id
                        << " measure mismatch source="
                        << partition.source_measure
                        << " slab_sum=" << partition.slab_measure_sum
                        << " slab_cell_count="
                        << partition.slab_cell_count;
                    throw std::runtime_error(message.str());
                }
            }
        }

        [[nodiscard]] AlreadySlabbedTrueDofPermutation
        already_slabbed_true_dof_permutation(double tolerance = 1.0e-12) const
        {
            const auto& source_dofs = source_space().dof_handler_ref();
            AlreadySlabbedTrueDofPermutation out;
            out.slab_true_offsets.resize(static_cast<std::size_t>(n_slabs()) + 1U, 0);

            for (int slab_id = 0; slab_id < n_slabs(); ++slab_id)
            {
                out.slab_true_offsets[static_cast<std::size_t>(slab_id + 1)] =
                    out.slab_true_offsets[static_cast<std::size_t>(slab_id)] +
                    slab(slab_id).fespace_ref().dof_handler_ref().n_true_dofs();
            }

            const int n_source_true = source_dofs.n_true_dofs();
            const int n_flat_slab_true = out.slab_true_offsets.back();
            out.source_true_to_slab_true.assign(
                static_cast<std::size_t>(n_source_true),
                -1);
            out.slab_true_to_source_true.assign(
                static_cast<std::size_t>(n_flat_slab_true),
                -1);

            for (const int source_cell_id : source_space().active_cells())
            {
                const auto& locations =
                    slab_locations_for_source_cell(source_cell_id);
                if (locations.size() != 1U)
                {
                    std::ostringstream message;
                    message
                        << "TimeSlabSpace::already_slabbed_true_dof_permutation: "
                        << "source cell " << source_cell_id
                        << " has " << locations.size()
                        << " slab pieces, expected one.";
                    throw std::runtime_error(message.str());
                }

                const auto& location = locations.front();
                if (std::abs(location.slab_t_begin -
                             location.source_t_begin) > tolerance ||
                    std::abs(location.slab_t_end -
                             location.source_t_end) > tolerance)
                {
                    std::ostringstream message;
                    message
                        << "TimeSlabSpace::already_slabbed_true_dof_permutation: "
                        << "source cell " << source_cell_id
                        << " is sliced in time.";
                    throw std::runtime_error(message.str());
                }

                const auto& slab_dofs =
                    slab(location.slab_id).fespace_ref().dof_handler_ref();
                const auto& source_cell_dofs =
                    source_dofs.cell_dofs(source_cell_id);
                const auto& slab_cell_dofs =
                    slab_dofs.cell_dofs(location.slab_local_cell_id);

                for (int local = 0; local < FETraits::dofs_per_cell; ++local)
                {
                    const int source_gid =
                        source_cell_dofs[static_cast<std::size_t>(local)];
                    const int slab_gid =
                        slab_cell_dofs[static_cast<std::size_t>(local)];

                    if (source_gid < 0 || slab_gid < 0)
                    {
                        if (source_gid != slab_gid)
                        {
                            throw std::runtime_error(
                                "TimeSlabSpace::already_slabbed_true_dof_permutation: "
                                "source/slab boundary elimination mismatch.");
                        }
                        continue;
                    }

                    const auto& source_dof = source_dofs.dof(source_gid);
                    const auto& slab_dof = slab_dofs.dof(slab_gid);
                    if (source_dof.is_constrained || slab_dof.is_constrained)
                        continue;

                    const int source_true = source_dof.true_dof_id;
                    const int slab_true =
                        out.slab_true_offsets[
                            static_cast<std::size_t>(location.slab_id)] +
                        slab_dof.true_dof_id;

                    set_permutation_entry_(
                        out.source_true_to_slab_true,
                        source_true,
                        slab_true,
                        "source_true_to_slab_true");
                    set_permutation_entry_(
                        out.slab_true_to_source_true,
                        slab_true,
                        source_true,
                        "slab_true_to_source_true");
                }
            }

            require_no_unmapped_(
                out.source_true_to_slab_true,
                "source true DoF");
            require_no_unmapped_(
                out.slab_true_to_source_true,
                "flat slab true DoF");

            verify_already_slabbed_local_maps_(out, tolerance);
            return out;
        }

    private:
        friend class TimeSlabBuilder<GeomTraits, FETraits>;

        void check_slab_index_(int k) const
        {
            if (k < 0 || k >= n_slabs())
                throw std::runtime_error("TimeSlabSpace::slab: slab index out of range.");
        }

        [[nodiscard]] int find_active_cell_on_slab_(
            const int slab_id,
            const SpaceTimePoint& p) const
        {
            const auto& space = slab(slab_id).fespace_ref();
            if (space.partition_view().active_cell_search_index_is_disabled())
                return space.find_active_cell_by_scan(p);
            return space.find_active_cell(p);
        }

        [[nodiscard]] int find_active_cell_on_slab_(
            const int slab_id,
            const int hint_cell_id,
            const SpaceTimePoint& p) const
        {
            const auto& space = slab(slab_id).fespace_ref();
            if (!space.partition_view().active_cell_search_index_is_disabled())
                return space.find_active_cell(hint_cell_id, p);

            if (hint_cell_id >= 0 &&
                space.mesh_ref().valid_cell_id(hint_cell_id) &&
                space.is_active_cell(hint_cell_id) &&
                space.mesh_ref().contains_coord(hint_cell_id, p))
            {
                return hint_cell_id;
            }

            return space.find_active_cell_by_scan(p);
        }

        void set_slab_times_(std::vector<double>&& slab_times)
        {
            slab_times_ = std::move(slab_times);
        }

        void set_slab_time_infos_(std::vector<SlabTimeInfo>&& slab_time_infos)
        {
            slab_time_infos_ = std::move(slab_time_infos);
        }

        void initialize_source_cell_provenance_(std::size_t n_source_cells)
        {
            source_cell_to_slab_locations_.clear();
            source_cell_to_slab_locations_.resize(n_source_cells);
        }

        void register_source_cell_slab_location_(
            const SourceCellSlabLocation& location)
        {
            if (location.source_cell_id < 0 ||
                static_cast<std::size_t>(location.source_cell_id) >=
                    source_cell_to_slab_locations_.size())
            {
                throw std::runtime_error(
                    "TimeSlabSpace::register_source_cell_slab_location_: "
                    "source cell id out of range.");
            }

            source_cell_to_slab_locations_[
                static_cast<std::size_t>(location.source_cell_id)]
                .push_back(location);
        }

        void reserve_slabs_(int n)
        {
            if (n < 0)
                throw std::runtime_error("TimeSlabSpace::reserve_slabs_: negative size.");
            slabs_.reserve(static_cast<std::size_t>(n));
        }

        void add_empty_slab_(int slab_id, double t_begin, double t_end)
        {
            slabs_.emplace_back(slab_id, t_begin, t_end);
        }

        [[nodiscard]] static double cell_measure_(
            const typename SourceSpaceType::MeshType& mesh,
            int cell_id)
        {
            const auto& cell = mesh.cell(cell_id);
            const double t0 =
                mesh.temporal_vertices()[cell.temporal_vertex_ids[0]][0];
            const double t1 =
                mesh.temporal_vertices()[cell.temporal_vertex_ids[1]][0];
            const double dt = std::abs(t1 - t0);

            if constexpr (GT::dim_space_v == 1)
            {
                const double x0 =
                    mesh.spatial_vertices()[cell.spatial_vertex_ids[0]][0];
                const double x1 =
                    mesh.spatial_vertices()[cell.spatial_vertex_ids[1]][0];
                return std::abs(x1 - x0) * dt;
            }
            else
            {
                const auto& a =
                    mesh.spatial_vertices()[cell.spatial_vertex_ids[0]];
                const auto& b =
                    mesh.spatial_vertices()[cell.spatial_vertex_ids[1]];
                const auto& c =
                    mesh.spatial_vertices()[cell.spatial_vertex_ids[2]];
                const double area_twice =
                    (b[0] - a[0]) * (c[1] - a[1]) -
                    (b[1] - a[1]) * (c[0] - a[0]);
                return 0.5 * std::abs(area_twice) * dt;
            }
        }

        static void set_permutation_entry_(
            std::vector<int>& map,
            int index,
            int value,
            const char* label)
        {
            if (index < 0 ||
                static_cast<std::size_t>(index) >= map.size())
            {
                throw std::runtime_error(
                    "TimeSlabSpace::already_slabbed_true_dof_permutation: "
                    "permutation index out of range.");
            }

            int& slot = map[static_cast<std::size_t>(index)];
            if (slot < 0)
            {
                slot = value;
                return;
            }

            if (slot != value)
            {
                std::ostringstream message;
                message
                    << "TimeSlabSpace::already_slabbed_true_dof_permutation: "
                    << label << " disagreement at index " << index
                    << " existing=" << slot
                    << " candidate=" << value;
                throw std::runtime_error(message.str());
            }
        }

        static void require_no_unmapped_(
            const std::vector<int>& map,
            const char* label)
        {
            for (int i = 0; i < static_cast<int>(map.size()); ++i)
            {
                if (map[static_cast<std::size_t>(i)] < 0)
                {
                    std::ostringstream message;
                    message
                        << "TimeSlabSpace::already_slabbed_true_dof_permutation: "
                        << label << " " << i << " has no correspondence.";
                    throw std::runtime_error(message.str());
                }
            }
        }

        template<class DoFHandlerType>
        [[nodiscard]] static std::vector<std::pair<int, double>>
        expand_to_true_(
            const DoFHandlerType& dofs,
            int global_id)
        {
            std::map<int, double> weights;
            if (global_id < 0)
                return {};

            const auto& dof = dofs.dof(global_id);
            if (!dof.is_constrained)
            {
                weights[dof.true_dof_id] += 1.0;
            }
            else
            {
                for (std::size_t k = 0;
                     k < dof.constraint_masters.size();
                     ++k)
                {
                    const int master_gid = dof.constraint_masters[k];
                    const double weight = dof.constraint_weights[k];
                    const auto& master = dofs.dof(master_gid);
                    if (master.is_constrained)
                    {
                        throw std::runtime_error(
                            "TimeSlabSpace::expand_to_true_: constrained "
                            "DoF references constrained master.");
                    }
                    weights[master.true_dof_id] += weight;
                }
            }

            std::vector<std::pair<int, double>> out;
            out.reserve(weights.size());
            for (const auto& [true_id, weight] : weights)
            {
                if (std::abs(weight) > 1.0e-13)
                    out.emplace_back(true_id, weight);
            }
            return out;
        }

        [[nodiscard]] static bool same_true_expansion_(
            const std::vector<std::pair<int, double>>& a,
            const std::vector<std::pair<int, double>>& b,
            double tolerance)
        {
            if (a.size() != b.size())
                return false;
            for (std::size_t i = 0; i < a.size(); ++i)
            {
                if (a[i].first != b[i].first)
                    return false;
                if (std::abs(a[i].second - b[i].second) > tolerance)
                    return false;
            }
            return true;
        }

        void verify_already_slabbed_local_maps_(
            const AlreadySlabbedTrueDofPermutation& permutation,
            double tolerance) const
        {
            const auto& source_dofs = source_space().dof_handler_ref();

            for (const int source_cell_id : source_space().active_cells())
            {
                const auto& locations =
                    slab_locations_for_source_cell(source_cell_id);
                const auto& location = locations.front();
                const auto& slab_dofs =
                    slab(location.slab_id).fespace_ref().dof_handler_ref();

                const auto& source_cell_dofs =
                    source_dofs.cell_dofs(source_cell_id);
                const auto& slab_cell_dofs =
                    slab_dofs.cell_dofs(location.slab_local_cell_id);

                for (int local = 0; local < FETraits::dofs_per_cell; ++local)
                {
                    const int source_gid =
                        source_cell_dofs[static_cast<std::size_t>(local)];
                    const int slab_gid =
                        slab_cell_dofs[static_cast<std::size_t>(local)];

                    if (source_gid < 0 || slab_gid < 0)
                    {
                        if (source_gid != slab_gid)
                        {
                            throw std::runtime_error(
                                "TimeSlabSpace::verify_already_slabbed_local_maps_: "
                                "boundary elimination mismatch.");
                        }
                        continue;
                    }

                    const auto source_pairs =
                        expand_to_true_(source_dofs, source_gid);
                    auto slab_pairs = expand_to_true_(slab_dofs, slab_gid);
                    for (auto& [slab_true, weight] : slab_pairs)
                    {
                        const int flat_slab_true =
                            permutation.slab_true_offsets[
                                static_cast<std::size_t>(location.slab_id)] +
                            slab_true;
                        slab_true =
                            permutation.slab_true_to_source_true[
                                static_cast<std::size_t>(flat_slab_true)];
                    }
                    std::sort(slab_pairs.begin(), slab_pairs.end());

                    if (!same_true_expansion_(
                            source_pairs,
                            slab_pairs,
                            tolerance))
                    {
                        std::ostringstream message;
                        message
                            << "TimeSlabSpace::verify_already_slabbed_local_maps_: "
                            << "local map mismatch source_cell="
                            << source_cell_id
                            << " slab=" << location.slab_id
                            << " slab_cell="
                            << location.slab_local_cell_id
                            << " local=" << local;
                        throw std::runtime_error(message.str());
                    }
                }
            }
        }

        const SourceSpaceType* source_space_ = nullptr;
        std::vector<double> slab_times_{};
        std::vector<SlabTimeInfo> slab_time_infos_{};
        std::vector<SlabType> slabs_{};
        std::vector<std::vector<SourceCellSlabLocation>>
            source_cell_to_slab_locations_{};
    };
}

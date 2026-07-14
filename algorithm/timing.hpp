#pragma once

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace adaptive_algorithm
{
    struct TimingOptions
    {
        bool enabled = false;
        std::string detail_level = "summary";
    };

    enum class MetricKind
    {
        duration_seconds,
        counter,
        bytes,
        megabytes,
        ratio,
        status,
        text
    };

    [[nodiscard]] inline std::string_view
    metric_kind_name(const MetricKind kind) noexcept
    {
        switch (kind)
        {
        case MetricKind::duration_seconds:
            return "duration_seconds";
        case MetricKind::counter:
            return "counter";
        case MetricKind::bytes:
            return "bytes";
        case MetricKind::megabytes:
            return "megabytes";
        case MetricKind::ratio:
            return "ratio";
        case MetricKind::status:
            return "status";
        case MetricKind::text:
            return "text";
        }

        return "duration_seconds";
    }

    [[nodiscard]] inline bool metric_phase_contains_(
        const std::string_view phase,
        const std::string_view token) noexcept
    {
        return phase.find(token) != std::string_view::npos;
    }

    [[nodiscard]] inline bool metric_phase_ends_with_(
        const std::string_view phase,
        const std::string_view suffix) noexcept
    {
        return phase.size() >= suffix.size() &&
            phase.substr(phase.size() - suffix.size()) == suffix;
    }

    [[nodiscard]] inline MetricKind
    infer_metric_kind(std::string_view phase) noexcept
    {
        if (metric_phase_contains_(phase, "_bytes") ||
            metric_phase_contains_(phase, ".bytes") ||
            metric_phase_ends_with_(phase, "bytes"))
        {
            return MetricKind::bytes;
        }

        if (metric_phase_contains_(phase, "_mb") ||
            metric_phase_contains_(phase, ".mb"))
        {
            return MetricKind::megabytes;
        }

        if (metric_phase_contains_(phase, "_seconds") ||
            metric_phase_contains_(phase, ".seconds") ||
            metric_phase_ends_with_(phase, "seconds") ||
            metric_phase_contains_(phase, "_wall") ||
            metric_phase_contains_(phase, ".wall") ||
            metric_phase_ends_with_(phase, "wall"))
        {
            return MetricKind::duration_seconds;
        }

        if (metric_phase_contains_(phase, "_ratio") ||
            metric_phase_contains_(phase, ".ratio") ||
            metric_phase_ends_with_(phase, "ratio") ||
            metric_phase_contains_(phase, "_fraction") ||
            metric_phase_ends_with_(phase, "fraction") ||
            metric_phase_contains_(phase, "residual") ||
            metric_phase_contains_(phase, "relative_residual") ||
            metric_phase_contains_(phase, "lambda_difference") ||
            metric_phase_contains_(phase, "hit_rate") ||
            metric_phase_contains_(
                phase,
                "average_patch_memberships_per_slab_cell") ||
            metric_phase_contains_(phase, "mean_candidates_per_query") ||
            metric_phase_contains_(phase, "mean_true_records_per_query") ||
            metric_phase_contains_(phase, "queries_per_") ||
            metric_phase_contains_(phase, ".queries_per_"))
        {
            return MetricKind::ratio;
        }

        if (phase == "reconstruction_y_norm.accumulator_type")
        {
            return MetricKind::status;
        }

        if (metric_phase_contains_(phase, "_enabled") ||
            metric_phase_contains_(phase, ".enabled") ||
            metric_phase_ends_with_(phase, "enabled") ||
            metric_phase_contains_(phase, "fallback_reason") ||
            metric_phase_contains_(phase, "fallback_full") ||
            metric_phase_contains_(phase, "conflict_strategy") ||
            metric_phase_contains_(phase, "_used") ||
            metric_phase_ends_with_(phase, "used") ||
            metric_phase_contains_(phase, "streaming_used") ||
            metric_phase_contains_(phase, "slot_maps_enabled") ||
            metric_phase_contains_(phase, "two_pass_direct_full_saddle"))
        {
            return MetricKind::status;
        }

        if (metric_phase_ends_with_(phase, ".count") ||
            metric_phase_contains_(phase, "_count") ||
            metric_phase_contains_(phase, ".count.") ||
            metric_phase_contains_(phase, "checksum") ||
            metric_phase_contains_(phase, "hash") ||
            metric_phase_contains_(phase, "low32") ||
            metric_phase_contains_(phase, "high32") ||
            metric_phase_contains_(phase, "cutoff_index") ||
            metric_phase_contains_(phase, "mode_size") ||
            metric_phase_contains_(phase, "vector_size") ||
            metric_phase_contains_(phase, "active_slab_cells") ||
            metric_phase_contains_(phase, "source_cells") ||
            metric_phase_contains_(phase, "memberships") ||
            metric_phase_contains_(phase, "membership_scans") ||
            metric_phase_contains_(phase, "state_build_requests") ||
            metric_phase_contains_(phase, "state_constructions") ||
            metric_phase_contains_(phase, "evictions") ||
            metric_phase_contains_(phase, "faces") ||
            metric_phase_contains_(phase, "edges") ||
            metric_phase_contains_(phase, "tiles") ||
            metric_phase_contains_(phase, "sampled_qpoints") ||
            metric_phase_contains_(phase, "qpoints_processed") ||
            metric_phase_contains_(phase, "processed_qpoints") ||
            metric_phase_contains_(phase, "_slabs") ||
            metric_phase_contains_(phase, ".slabs") ||
            metric_phase_contains_(phase, "records_visited") ||
            metric_phase_contains_(
                phase,
                "reconstruction_y_norm.slab_cell_views_allocated") ||
            metric_phase_contains_(
                phase,
                "reconstruction_y_norm.qpoints_visited") ||
            metric_phase_contains_(
                phase,
                "reconstruction_y_norm.source_cells_touched") ||
            metric_phase_contains_(
                phase,
                "reconstruction_y_norm.slab_cells_touched") ||
            metric_phase_contains_(phase, "qpoints_visited") ||
            metric_phase_contains_(phase, "slab_cells_visited") ||
            metric_phase_contains_(phase, "active_records") ||
            metric_phase_contains_(phase, "inactive_records") ||
            metric_phase_contains_(phase, "candidate_records") ||
            metric_phase_contains_(phase, "containment_candidates") ||
            metric_phase_contains_(phase, "query_candidates") ||
            metric_phase_contains_(phase, "true_records_returned") ||
            metric_phase_contains_(phase, "rejects") ||
            metric_phase_contains_(phase, "rejected") ||
            metric_phase_contains_(phase, "discarded") ||
            metric_phase_contains_(phase, "max_candidates") ||
            metric_phase_contains_(phase, "mean_candidates") ||
            metric_phase_contains_(phase, "mean_true_records") ||
            metric_phase_contains_(phase, "query_keys") ||
            metric_phase_contains_(phase, "cache_entries") ||
            metric_phase_contains_(phase, "visited_set_size") ||
            metric_phase_contains_(phase, "support_line_group") ||
            metric_phase_contains_(phase, "records_active") ||
            metric_phase_contains_(phase, "records_inactive") ||
            metric_phase_contains_(phase, "remove_cell_records_touched") ||
            metric_phase_contains_(phase, "queries") ||
            metric_phase_contains_(phase, "full_active_scans") ||
            metric_phase_contains_(phase, "partition_update_calls") ||
            metric_phase_contains_(phase, "batched_split_cells") ||
            metric_phase_contains_(phase, "cell_slot_count") ||
            metric_phase_contains_(phase, "children_created") ||
            metric_phase_contains_(phase, "actually_split_active_cells") ||
            metric_phase_contains_(phase, "actually_split_cells") ||
            metric_phase_contains_(phase, "grading_forced_cells") ||
            metric_phase_contains_(phase, "blocker_cells") ||
            metric_phase_contains_(phase, "closure_cells") ||
            metric_phase_contains_(phase, "forced_cells") ||
            metric_phase_contains_(phase, "refined_cells") ||
            metric_phase_contains_(phase, "marked_cells") ||
            metric_phase_contains_(phase, "active_cells") ||
            metric_phase_contains_(phase, "true_dofs") ||
            metric_phase_contains_(phase, "cut_dofs") ||
            metric_phase_contains_(phase, "constrained_dofs") ||
            metric_phase_contains_(phase, "constraint_rows") ||
            metric_phase_contains_(phase, "prolongation_nnz") ||
            metric_phase_contains_(phase, "local_occurrences") ||
            metric_phase_contains_(phase, "occurrence_indexed") ||
            metric_phase_contains_(phase, "occurrences_indexed") ||
            metric_phase_contains_(phase, "entity_keys") ||
            metric_phase_contains_(phase, "boundary_eliminated") ||
            metric_phase_contains_(phase, "constraint_sources") ||
            metric_phase_contains_(phase, "constrained_keys") ||
            metric_phase_contains_(phase, "constraints_rebuilt") ||
            metric_phase_contains_(phase, "source_y_cells") ||
            metric_phase_contains_(phase, "slab_count") ||
            metric_phase_contains_(phase, "virtual_slab_cells") ||
            metric_phase_contains_(phase, "copied_slab_cells") ||
            metric_phase_contains_(phase, "distinct_spatial_meshes") ||
            metric_phase_contains_(phase, "reused_spatial_layouts") ||
            metric_phase_contains_(phase, "patch_count") ||
            metric_phase_contains_(phase, "n_patch") ||
            metric_phase_contains_(phase, "max_patch_cells") ||
            metric_phase_contains_(phase, "mean_patch_cells") ||
            metric_phase_contains_(phase, "patch_dofs") ||
            metric_phase_contains_(phase, "patch_solves") ||
            metric_phase_contains_(phase, "dense_patch_solves") ||
            metric_phase_contains_(phase, "sparse_patch_solves") ||
            metric_phase_contains_(phase, "touched_cells") ||
            metric_phase_contains_(phase, "changed_cells") ||
            metric_phase_contains_(phase, "queue_pops") ||
            metric_phase_contains_(phase, "pending_cells_seen") ||
            metric_phase_contains_(phase, "repeated_pending_cell_pops") ||
            metric_phase_contains_(phase, "requeued_due_to_blockers") ||
            metric_phase_contains_(phase, "blockers_found") ||
            metric_phase_contains_(phase, "blockers_already_seen") ||
            metric_phase_contains_(
                phase,
                "closure_decision_cache_possible") ||
            metric_phase_contains_(phase, "max_batch_size") ||
            metric_phase_contains_(phase, "average_batch_size") ||
            metric_phase_contains_(phase, "group_max_size") ||
            metric_phase_contains_(phase, "group_mean_size") ||
            metric_phase_contains_(phase, "cell_restrictions_reused") ||
            metric_phase_contains_(phase, "geometry_cache_hits") ||
            metric_phase_contains_(phase, "geometry_cache_misses") ||
            metric_phase_contains_(
                phase,
                "reconstruction_y_norm.virtual_geometry_constructed") ||
            metric_phase_contains_(
                phase,
                "reconstruction_y_norm.copied_geometry_cache_lookups") ||
            metric_phase_contains_(phase, "cache_hits") ||
            metric_phase_contains_(phase, "cache_misses") ||
            metric_phase_contains_(phase, "interval_index_hits") ||
            metric_phase_contains_(phase, "interval_index_misses") ||
            metric_phase_contains_(phase, "cell_states_constructed") ||
            metric_phase_contains_(phase, "total_cell_state_rebuilds") ||
            metric_phase_contains_(phase, "cell_state_rebuilds") ||
            metric_phase_contains_(phase, "coordinate_vertex_matches") ||
            metric_phase_contains_(phase, "fused_used") ||
            metric_phase_contains_(phase, "near_cap_") ||
            metric_phase_contains_(phase, "nnz") ||
            metric_phase_contains_(phase, "waves") ||
            metric_phase_contains_(phase, "_calls") ||
            metric_phase_contains_(phase, ".calls") ||
            metric_phase_ends_with_(phase, "calls") ||
            metric_phase_contains_(phase, "_steps") ||
            metric_phase_contains_(phase, ".steps") ||
            metric_phase_ends_with_(phase, "steps"))
        {
            return MetricKind::counter;
        }

        return MetricKind::duration_seconds;
    }

    struct TimingRecord
    {
        std::string phase{};
        MetricKind metric_kind = MetricKind::duration_seconds;
        double total_seconds = 0.0;
        double last_seconds = 0.0;
        std::size_t call_count = 0;
    };

    class TimingCollector
    {
    public:
        using Clock = std::chrono::steady_clock;

        class ScopedTimer
        {
        public:
            ScopedTimer() noexcept = default;

            ScopedTimer(const ScopedTimer&) = delete;
            ScopedTimer& operator=(const ScopedTimer&) = delete;

            ScopedTimer(ScopedTimer&& other) noexcept
                : collector_(std::exchange(other.collector_, nullptr)),
                  phase_(std::move(other.phase_)),
                  start_(other.start_),
                  active_(std::exchange(other.active_, false))
            {}

            ScopedTimer& operator=(ScopedTimer&& other) noexcept
            {
                if (this == &other)
                    return *this;

                stop_noexcept_();
                collector_ = std::exchange(other.collector_, nullptr);
                phase_ = std::move(other.phase_);
                start_ = other.start_;
                active_ = std::exchange(other.active_, false);
                return *this;
            }

            ~ScopedTimer() noexcept
            {
                stop_noexcept_();
            }

            void stop()
            {
                if (!active_ || collector_ == nullptr)
                    return;

                const auto end = Clock::now();
                const double seconds =
                    std::chrono::duration<double>(end - start_).count();
                active_ = false;
                collector_->add(phase_, seconds);
            }

            [[nodiscard]] bool active() const noexcept
            {
                return active_;
            }

        private:
            using Clock = std::chrono::steady_clock;

            friend class TimingCollector;

            ScopedTimer(TimingCollector& collector, std::string_view phase)
                : collector_(&collector),
                  phase_(phase),
                  start_(Clock::now()),
                  active_(true)
            {}

            void stop_noexcept_() noexcept
            {
                try
                {
                    stop();
                }
                catch (...)
                {
                }
            }

            TimingCollector* collector_ = nullptr;
            std::string phase_{};
            Clock::time_point start_{};
            bool active_ = false;
        };

        TimingCollector() = default;

        explicit TimingCollector(bool enabled) noexcept
            : enabled_(enabled)
        {}

        explicit TimingCollector(TimingOptions options) noexcept
            : enabled_(options.enabled)
        {}

        [[nodiscard]] bool enabled() const noexcept
        {
            return enabled_;
        }

        void set_enabled(bool enabled) noexcept
        {
            enabled_ = enabled;
        }

        void reset()
        {
            records_.clear();
            add_call_count_ = 0;
            linear_search_steps_ = 0;
            record_count_max_ = 0;
            add_seconds_total_ = 0.0;
        }

        void add(std::string_view phase, double seconds)
        {
            if (!enabled_)
                return;

            const auto add_start = Clock::now();
            ++add_call_count_;

            std::size_t linear_search_steps = 0;
            auto* record = find_record_(phase, &linear_search_steps);
            linear_search_steps_ += linear_search_steps;
            if (record == nullptr)
            {
                TimingRecord new_record{};
                new_record.phase = std::string(phase);
                new_record.metric_kind = infer_metric_kind(phase);
                records_.push_back(std::move(new_record));
                record = &records_.back();
            }
            record_count_max_ =
                std::max(record_count_max_, records_.size());

            record->last_seconds = seconds;
            record->total_seconds += seconds;
            ++record->call_count;

            add_seconds_total_ +=
                std::chrono::duration<double>(Clock::now() - add_start)
                    .count();
        }

        [[nodiscard]] ScopedTimer scoped(std::string_view phase)
        {
            if (!enabled_)
                return {};

            return ScopedTimer(*this, phase);
        }

        [[nodiscard]] const std::vector<TimingRecord>& records() const noexcept
        {
            return records_;
        }

        [[nodiscard]] std::vector<TimingRecord> records_snapshot() const
        {
            auto snapshot = records_;
            append_snapshot_metric_(
                snapshot,
                "timing_collector.add_calls",
                MetricKind::counter,
                static_cast<double>(add_call_count_));
            append_snapshot_metric_(
                snapshot,
                "timing_collector.linear_search_steps",
                MetricKind::counter,
                static_cast<double>(linear_search_steps_));
            append_snapshot_metric_(
                snapshot,
                "timing_collector.record_count_max",
                MetricKind::counter,
                static_cast<double>(record_count_max_));
            append_snapshot_metric_(
                snapshot,
                "timing_collector.add_seconds",
                MetricKind::duration_seconds,
                add_seconds_total_);
            return snapshot;
        }

        [[nodiscard]] std::optional<TimingRecord> find(std::string_view phase) const
        {
            for (const auto& record : records_)
            {
                if (std::string_view(record.phase) == phase)
                    return record;
            }

            return std::nullopt;
        }

    private:
        [[nodiscard]] TimingRecord* find_record_(
            std::string_view phase,
            std::size_t* linear_search_steps = nullptr) noexcept
        {
            for (auto& record : records_)
            {
                if (linear_search_steps != nullptr)
                    ++*linear_search_steps;
                if (std::string_view(record.phase) == phase)
                    return &record;
            }

            return nullptr;
        }

        static void append_snapshot_metric_(
            std::vector<TimingRecord>& snapshot,
            std::string phase,
            MetricKind kind,
            double value)
        {
            TimingRecord record;
            record.phase = std::move(phase);
            record.metric_kind = kind;
            record.total_seconds = value;
            record.last_seconds = value;
            record.call_count = 1;
            snapshot.push_back(std::move(record));
        }

        bool enabled_ = false;
        std::vector<TimingRecord> records_{};
        std::size_t add_call_count_ = 0;
        std::size_t linear_search_steps_ = 0;
        std::size_t record_count_max_ = 0;
        double add_seconds_total_ = 0.0;
    };
}

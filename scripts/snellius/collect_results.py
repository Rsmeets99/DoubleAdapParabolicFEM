#!/usr/bin/env python3
"""Collect Snellius run_adaptive_algorithm outputs into CSV and archive folders.

The collector is intentionally tolerant of interrupted jobs.  It records which
expected files exist, uses partial histories when full histories are unavailable,
and copies available run artifacts into a stable collection directory.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
from pathlib import Path
from typing import Any


EXPECTED_FILES = (
    "adaptive_summary.txt",
    "outer_history.csv",
    "inner_history.csv",
    "timing_history.csv",
    "run_parameters.yml",
    "snellius_job_info.txt",
    "partial_outer_history.csv",
    "partial_inner_history.csv",
    "partial_timing_history.csv",
)

SUMMARY_KEYS = (
    "problem_name",
    "spatial_dimension",
    "polynomial_degree",
    "outer_iterations",
    "converged",
    "terminated_early",
    "termination_reason",
    "final_x_active_cells",
    "final_x_true_dofs",
    "final_y_active_cells",
    "final_y_true_dofs",
    "final_eta_squared",
    "final_eta",
    "final_y_estimator_squared",
    "final_y_estimator",
    "final_configured_rho",
    "final_effective_rho",
    "final_effective_rho_available",
    "final_posteriori_estimator_configured_rho_squared",
    "final_posteriori_estimator_effective_rho_squared",
    "final_g_estimator_enabled",
    "final_g_estimator_computed",
    "final_g_true_dofs",
    "final_g_lambda_difference_squared",
    "final_g_lambda_difference",
)

LAST_ROW_KEYS = (
    "outer_iteration",
    "inner_iteration",
    "history_status",
    "elapsed_seconds",
    "iteration_seconds",
    "x_active_cells_before",
    "x_true_dofs_before",
    "x_active_cells_after",
    "x_true_dofs_after",
    "y_active_cells_before",
    "y_true_dofs_before",
    "y_active_cells_after",
    "y_true_dofs_after",
    "eta_squared",
    "eta",
    "y_estimator_squared",
    "y_estimator",
    "effective_rho",
    "effective_rho_available",
    "posteriori_estimator_configured_rho_squared",
    "posteriori_estimator_effective_rho_squared",
    "g_estimator_computed",
    "g_true_dofs",
    "g_lambda_difference_squared",
    "g_lambda_difference",
    "main_solve_selected_solver",
    "main_solve_effective_solver",
    "main_solve_solver_status",
    "main_solve_matrix_rows",
    "main_solve_matrix_cols",
    "main_solve_matrix_nnz",
    "main_solve_n",
    "main_solve_nnz_matrix",
    "main_solve_nnz_factors",
    "main_solve_factor_nnz",
    "main_solve_fill_ratio",
    "main_solve_symbolic_analysis_seconds",
    "main_solve_symbolic_analysis_reused",
    "main_solve_symbolic_pattern_cache_hits",
    "main_solve_symbolic_pattern_cache_misses",
    "main_solve_numeric_factorization_seconds",
    "main_solve_backsolve_seconds",
    "main_solve_estimated_factor_memory_bytes",
    "main_solve_symbolic_memory",
    "main_solve_numerical_factor_memory",
    "main_solve_estimated_in_core_peak_memory",
    "main_solve_out_of_core_minimum_memory",
    "main_solve_process_rss_before_factorization",
    "main_solve_process_rss_after_factorization",
    "main_solve_process_rss_after_solve",
    "main_solve_memory_guard_estimated_extra_memory",
    "main_solve_direct_memory_limit",
    "main_solve_memory_guard_estimated_peak_memory",
    "main_solve_memory_guard_triggered",
    "main_solve_ooc_auto_switch_attempted",
    "main_solve_ooc_auto_switch_solver",
    "main_solve_effective_pardiso_memory_mode",
    "main_solve_iteration_count",
    "main_solve_final_residual",
    "linear_residual_absolute",
    "linear_residual_relative",
    "main_solve_initial_guess_norm",
    "main_solve_initial_residual_absolute",
    "main_solve_initial_residual_relative",
    "main_solve_backend_converged",
    "main_solve_backend_reported_error",
    "main_solve_accepted_by_true_residual",
    "main_solve_residual_check_batches",
    "main_solve_final_true_residual",
    "main_solve_true_residual_stopping_used",
    "main_solve_matrix_norm",
    "main_solve_symmetry_difference_norm",
    "main_solve_relative_asymmetry",
    "main_solve_residual_retry_attempted",
    "main_solve_residual_retry_solver",
    "main_solve_residual_before_retry",
    "main_solve_residual_after_retry",
    "main_solve_residual_correction_steps",
    "main_solve_residual_before_correction",
    "main_solve_residual_after_correction",
    "main_solve_preconditioner_setup_seconds",
    "main_solve_setup_seconds",
    "main_solve_solve_seconds",
    "time_slab_backend",
    "time_slab_backend_effective",
)

PARAM_KEYS = (
    "example",
    "dimension",
    "p",
    "max_dofs_target",
    "max_x_true_dofs",
    "max_y_true_dofs",
    "output_profile",
    "main_solver",
    "main_solver_pardiso_memory_mode",
    "main_solver_diagnostics",
    "main_solver_memory_limit_mb",
    "memory_limit_mb",
    "memory_reserve_mb",
    "time_slab_backend",
    "compute_g_estimator",
    "compute_g_estimator_every_inner_iteration",
    "force_accept_inner_with_effective_rho",
    "uniform_x_refinement",
    "uniform_y_refinement",
    "main_assembly_max_threads",
    "main_two_pass_numeric_fill_max_threads",
    "slab_reconstruction_max_threads",
    "local_error_max_threads",
)

JOB_INFO_KEYS = (
    "job_id",
    "job_name",
    "partition",
    "time_limit",
    "memory_mb",
    "threads",
    "omp_threads",
    "mkl_threads",
    "run_id",
    "base_output",
    "use_scratch",
    "flat_output",
    "load_modules",
    "module_set",
    "config",
    "executable",
    "run_output",
    "final_output",
    "started_at",
    "finished_at",
    "failed_at",
    "terminated_by",
    "terminated_at",
    "run_exit_code",
    "copy_back_started_at",
    "copy_back_completed_at",
)


def read_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if len(row) >= 2:
                data[row[0]] = row[1]
    return data


def read_flat_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.strip().strip('"')
        data[key.strip()] = value
    return data


def read_equals_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip().strip('"')
    return data


def is_missing(value: Any) -> bool:
    return value is None or str(value) in ("", "MISSING")


def is_duplicate_header_row(row: dict[str, str]) -> bool:
    comparable = [
        (key, value)
        for key, value in row.items()
        if key is not None and value is not None and key != ""
    ]
    if not comparable:
        return False
    matches = sum(1 for key, value in comparable if value.strip() == key)
    return matches >= max(2, len(comparable) // 2)


def read_last_csv_row(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {
                key: "" if value is None else value
                for key, value in dict(row).items()
                if key is not None
            }
            if is_duplicate_header_row(cleaned):
                continue
            rows.append(cleaned)

        if not rows:
            return {}

        preferred_statuses = {"completed", "interrupted"}
        for row in reversed(rows):
            if row.get("history_status", "") in preferred_statuses:
                return row

        return rows[-1]


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def output_base_dir_for_manifest_row(project_root: Path, row: dict[str, str]) -> Path:
    output = row.get("output", "")
    if not output:
        return project_root / "MISSING_OUTPUT"
    path = Path(output)
    return path if path.is_absolute() else project_root / path


def has_run_artifacts(path: Path) -> bool:
    return any((path / name).exists() for name in EXPECTED_FILES)


def discovered_run_dirs(output_base: Path) -> list[Path]:
    runs_dir = output_base / "runs"
    if not runs_dir.exists() or not runs_dir.is_dir():
        return []
    candidates = [
        path
        for path in runs_dir.iterdir()
        if path.is_dir() and has_run_artifacts(path)
    ]
    return sorted(candidates, key=lambda path: path.name)


def output_dirs_for_manifest_row(
    project_root: Path,
    row: dict[str, str],
    run_selection: str,
) -> list[tuple[Path, str, Path]]:
    output_base = output_base_dir_for_manifest_row(project_root, row)
    run_dirs = discovered_run_dirs(output_base)

    if run_selection == "legacy":
        return [(output_base, "legacy", output_base)]

    if run_selection == "all":
        selected = [(path, path.name, output_base) for path in run_dirs]
        if has_run_artifacts(output_base):
            selected.insert(0, (output_base, "legacy", output_base))
        return selected or [(output_base, "missing", output_base)]

    if run_dirs:
        latest = run_dirs[-1]
        return [(latest, latest.name, output_base)]
    return [(output_base, "legacy", output_base)]


def safe_case_id(row: dict[str, str]) -> str:
    value = row.get("case_id") or row.get("config") or "missing_case_id"
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def safe_path_component(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
    return safe or "missing"


def copy_available_files(output_dir: Path, archive_case_dir: Path | None) -> list[str]:
    copied: list[str] = []
    if archive_case_dir is None:
        return copied
    archive_case_dir.mkdir(parents=True, exist_ok=True)
    for name in EXPECTED_FILES:
        src = output_dir / name
        if not src.exists() or not src.is_file():
            continue
        dst = archive_case_dir / name
        shutil.copy2(src, dst)
        copied.append(name)
    return copied


def run_status_for(
    output_dir: Path,
    summary: dict[str, str],
    outer: dict[str, str],
    inner: dict[str, str],
    partial_outer: dict[str, str],
    partial_inner: dict[str, str],
    job_info: dict[str, str],
) -> str:
    if summary and (outer or inner):
        return "completed_or_terminated_with_full_history"
    if summary:
        return "summary_only"
    if (partial_outer or partial_inner) and job_info.get("terminated_by"):
        return "partial_history_terminated"
    if (partial_outer or partial_inner) and job_info.get("failed_at"):
        return "partial_history_failed"
    if partial_outer or partial_inner:
        return "partial_history_only"
    if job_info.get("terminated_by"):
        return "terminated_without_required_history"
    if job_info.get("failed_at"):
        return "failed_without_required_history"
    if output_dir.exists():
        return "output_dir_without_required_history"
    return "missing_output"


def collect_row(
    project_root: Path,
    manifest_row: dict[str, str],
    output_dir: Path,
    run_id: str,
    output_base: Path,
    archive_root: Path | None,
) -> dict[str, Any]:
    summary = read_key_value_file(output_dir / "adaptive_summary.txt")
    params = read_flat_yaml(output_dir / "run_parameters.yml")
    job_info = read_equals_file(output_dir / "snellius_job_info.txt")
    outer = read_last_csv_row(output_dir / "outer_history.csv")
    inner = read_last_csv_row(output_dir / "inner_history.csv")
    timing = read_last_csv_row(output_dir / "timing_history.csv")
    partial_outer = read_last_csv_row(output_dir / "partial_outer_history.csv")
    partial_inner = read_last_csv_row(output_dir / "partial_inner_history.csv")
    partial_timing = read_last_csv_row(output_dir / "partial_timing_history.csv")

    outer_row = outer or partial_outer
    inner_row = inner or partial_inner
    timing_row = timing or partial_timing
    effective_run_id = job_info.get("run_id", run_id)
    archive_case_dir = None
    if archive_root is not None:
        archive_case_dir = archive_root / safe_case_id(manifest_row) / safe_path_component(effective_run_id)
    copied_files = copy_available_files(output_dir, archive_case_dir)

    result: dict[str, Any] = {
        "config": manifest_row.get("config", "MISSING"),
        "case_id": manifest_row.get("case_id", "MISSING"),
        "dimension": manifest_row.get("dimension", "MISSING"),
        "example": manifest_row.get("example", "MISSING"),
        "p": manifest_row.get("p", "MISSING"),
        "mode": manifest_row.get("mode", manifest_row.get("variant", "MISSING")),
        "variant": manifest_row.get("variant", manifest_row.get("mode", "MISSING")),
        "target_label": manifest_row.get("target_label", manifest_row.get("stage_label", "MISSING")),
        "max_dofs_target": manifest_row.get("max_dofs_target", params.get("max_dofs_target", "MISSING")),
        "max_x_true_dofs": manifest_row.get("max_x_true_dofs", params.get("max_x_true_dofs", "MISSING")),
        "max_y_true_dofs": manifest_row.get("max_y_true_dofs", params.get("max_y_true_dofs", "MISSING")),
        "run_id": effective_run_id,
        "output_base": str(output_base),
        "output": str(output_dir),
        "archive_dir": "MISSING" if archive_case_dir is None else str(archive_case_dir),
        "archived_files": ";".join(copied_files),
        "run_status": run_status_for(output_dir, summary, outer, inner, partial_outer, partial_inner, job_info),
        "summary_present": int(bool(summary)),
        "outer_history_present": int((output_dir / "outer_history.csv").exists()),
        "inner_history_present": int((output_dir / "inner_history.csv").exists()),
        "timing_history_present": int((output_dir / "timing_history.csv").exists()),
        "run_parameters_present": int((output_dir / "run_parameters.yml").exists()),
        "snellius_job_info_present": int((output_dir / "snellius_job_info.txt").exists()),
        "partial_outer_history_present": int((output_dir / "partial_outer_history.csv").exists()),
        "partial_inner_history_present": int((output_dir / "partial_inner_history.csv").exists()),
        "partial_timing_history_present": int((output_dir / "partial_timing_history.csv").exists()),
    }

    for key in SUMMARY_KEYS:
        result[key] = summary.get(key, "MISSING")
    for key in PARAM_KEYS:
        result[f"param_{key}"] = params.get(key, "MISSING")
    for key in JOB_INFO_KEYS:
        result[f"job_{key}"] = job_info.get(key, "MISSING")
    for key in LAST_ROW_KEYS:
        result[f"outer_last_{key}"] = outer_row.get(key, "MISSING")
        result[f"inner_last_{key}"] = inner_row.get(key, "MISSING")
    for key in ("phase", "metric_kind", "total_seconds", "last_seconds", "call_count"):
        result[f"timing_last_{key}"] = timing_row.get(key, "MISSING")

    # Fallback final values from last history rows make interrupted runs useful.
    fallback_candidates = {
        "outer_iterations": ("outer_last_outer_iteration", "inner_last_outer_iteration"),
        "final_x_active_cells": (
            "outer_last_x_active_cells_after",
            "outer_last_x_active_cells_before",
        ),
        "final_x_true_dofs": (
            "outer_last_x_true_dofs_after",
            "outer_last_x_true_dofs_before",
        ),
        "final_y_active_cells": (
            "outer_last_y_active_cells_after",
            "inner_last_y_active_cells_after",
            "outer_last_y_active_cells_before",
            "inner_last_y_active_cells_before",
        ),
        "final_y_true_dofs": (
            "outer_last_y_true_dofs_after",
            "inner_last_y_true_dofs_after",
            "outer_last_y_true_dofs_before",
            "inner_last_y_true_dofs_before",
        ),
        "final_eta_squared": ("outer_last_eta_squared", "inner_last_eta_squared"),
        "final_eta": ("outer_last_eta", "inner_last_eta"),
        "final_y_estimator_squared": (
            "outer_last_y_estimator_squared",
            "inner_last_y_estimator_squared",
        ),
        "final_y_estimator": ("outer_last_y_estimator", "inner_last_y_estimator"),
        "final_effective_rho": ("outer_last_effective_rho", "inner_last_effective_rho"),
        "final_effective_rho_available": (
            "outer_last_effective_rho_available",
            "inner_last_effective_rho_available",
        ),
        "final_posteriori_estimator_configured_rho_squared": (
            "outer_last_posteriori_estimator_configured_rho_squared",
            "inner_last_posteriori_estimator_configured_rho_squared",
        ),
        "final_posteriori_estimator_effective_rho_squared": (
            "outer_last_posteriori_estimator_effective_rho_squared",
            "inner_last_posteriori_estimator_effective_rho_squared",
        ),
        "final_g_estimator_computed": (
            "outer_last_g_estimator_computed",
            "inner_last_g_estimator_computed",
        ),
        "final_g_true_dofs": ("outer_last_g_true_dofs", "inner_last_g_true_dofs"),
        "final_g_lambda_difference_squared": (
            "outer_last_g_lambda_difference_squared",
            "inner_last_g_lambda_difference_squared",
        ),
        "final_g_lambda_difference": (
            "outer_last_g_lambda_difference",
            "inner_last_g_lambda_difference",
        ),
    }
    for final_key, fallback_keys in fallback_candidates.items():
        if not is_missing(result.get(final_key)):
            continue
        for fallback_key in fallback_keys:
            fallback_value = result.get(fallback_key)
            if not is_missing(fallback_value):
                result[final_key] = fallback_value
                break

    sqrt_fallback_pairs = {
        "final_eta": "final_eta_squared",
        "final_y_estimator": "final_y_estimator_squared",
        "final_g_lambda_difference": "final_g_lambda_difference_squared",
    }
    for value_key, squared_key in sqrt_fallback_pairs.items():
        if not is_missing(result.get(value_key)):
            continue
        try:
            squared_value = float(str(result.get(squared_key, "")))
        except ValueError:
            continue
        if math.isfinite(squared_value) and squared_value >= 0.0:
            result[value_key] = f"{math.sqrt(squared_value):.17g}"

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="production_configs/snellius/manifest.csv")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--out", default="production_configs/snellius/collected_results.csv")
    parser.add_argument("--archive-dir", default="production_configs/snellius/collected_outputs")
    parser.add_argument("--no-archive", action="store_true")
    parser.add_argument(
        "--run-selection",
        choices=("latest", "all", "legacy"),
        default="latest",
        help="Which output directories to collect for each manifest row. Default: latest timestamped run, falling back to legacy flat output.",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = project_root / out_path
    archive_root = None
    if not args.no_archive:
        archive_root = Path(args.archive_dir)
        if not archive_root.is_absolute():
            archive_root = project_root / archive_root

    rows: list[dict[str, Any]] = []
    for manifest_row in read_manifest(manifest_path):
        for output_dir, run_id, output_base in output_dirs_for_manifest_row(
            project_root,
            manifest_row,
            args.run_selection,
        ):
            rows.append(
                collect_row(
                    project_root,
                    manifest_row,
                    output_dir,
                    run_id,
                    output_base,
                    archive_root,
                )
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("run_status", "MISSING"))
        status_counts[status] = status_counts.get(status, 0) + 1
    print(f"Wrote {len(rows)} rows to {out_path}")
    if archive_root is not None:
        print(f"Archived available files under {archive_root}")
    for status, count in sorted(status_counts.items()):
        print(f"{status}={count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

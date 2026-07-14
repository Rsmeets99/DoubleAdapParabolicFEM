#!/usr/bin/env python3
"""Generate Snellius production configs for run_adaptive_algorithm.

The runner consumes flat YAML-style key/value files, so this script intentionally
avoids a YAML dependency and writes only scalar values understood by the runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DIMENSIONS = (1, 2)
EXAMPLES = ("smooth_initial", "non_matching_initial", "boundary_singularity")
DEGREES = (1, 2, 3, 4)
DEFAULT_STAGE_CAPS = (250_000, 500_000, 1_000_000, 4_000_000, 6_000_000)


@dataclass(frozen=True)
class Variant:
    name: str
    refinement_mode: str
    uniform_x_refinement: bool
    uniform_y_refinement: bool
    compute_g_estimator: bool
    compute_g_estimator_every_inner_iteration: bool
    force_accept_inner_with_effective_rho: bool
    max_inner_override: int | None = None


@dataclass(frozen=True)
class ConfigCase:
    dimension: int
    example: str
    degree: int
    stage_cap: int
    variant: Variant

    @property
    def target_label(self) -> str:
        return target_label_for_dofs(self.stage_cap)

    @property
    def dimension_label(self) -> str:
        return f"{self.dimension}d"

    @property
    def case_id(self) -> str:
        return (
            f"{self.dimension_label}_{self.example}_p{self.degree}_"
            f"{self.variant.name}_{self.target_label}"
        )

    @property
    def relative_path(self) -> Path:
        return (
            Path(self.dimension_label)
            / self.example
            / f"p{self.degree}"
            / self.variant.name
            / f"{self.target_label}.yml"
        )


@dataclass(frozen=True)
class ThreadPolicy:
    cpus: int
    omp_threads: int
    mkl_threads: int
    main_assembly_threads: int
    main_numeric_fill_threads: int
    slab_reconstruction_threads: int
    local_error_threads: int


def target_label_for_dofs(dofs: int) -> str:
    known = {
        250_000: "pilot_250k",
        500_000: "pilot_500k",
        1_000_000: "pilot_1M",
        4_000_000: "production_4M",
        6_000_000: "production_6M",
    }
    if dofs in known:
        return known[dofs]
    if dofs >= 1_000_000 and dofs % 1_000_000 == 0:
        return f"production_{dofs // 1_000_000}M"
    if dofs % 1_000 == 0:
        return f"pilot_{dofs // 1_000}k"
    return f"target_{dofs}"


def parse_stage_caps(text: str) -> tuple[int, ...]:
    caps: list[int] = []
    for raw in text.split(","):
        item = raw.strip().replace("_", "")
        if not item:
            continue
        multiplier = 1
        lowered = item.lower()
        if lowered.endswith("k"):
            multiplier = 1_000
            lowered = lowered[:-1]
        elif lowered.endswith("m"):
            multiplier = 1_000_000
            lowered = lowered[:-1]
        value = int(float(lowered) * multiplier)
        if value <= 0:
            raise argparse.ArgumentTypeError("stage caps must be positive")
        caps.append(value)
    if not caps:
        raise argparse.ArgumentTypeError("at least one stage cap is required")
    return tuple(caps)


def yaml_scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "MISSING"
    text = str(value)
    if not text:
        return '""'
    if any(ch in text for ch in (":", "#", "\n", "\r")):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def format_config(entries: Iterable[tuple[str, object]], header_lines: Iterable[str]) -> str:
    lines: list[str] = []
    for line in header_lines:
        lines.append(f"# {line}")
    if lines:
        lines.append("")
    lines.extend(f"{key}: {yaml_scalar(value)}" for key, value in entries)
    lines.append("")
    return "\n".join(lines)


def build_variants(args: argparse.Namespace) -> tuple[Variant, ...]:
    return (
        Variant(
            name="adaptive_standard",
            refinement_mode="adaptive",
            uniform_x_refinement=False,
            uniform_y_refinement=False,
            compute_g_estimator=False,
            compute_g_estimator_every_inner_iteration=False,
            force_accept_inner_with_effective_rho=False,
        ),
        Variant(
            name="adaptive_g_effective_rho",
            refinement_mode="adaptive",
            uniform_x_refinement=False,
            uniform_y_refinement=False,
            compute_g_estimator=True,
            compute_g_estimator_every_inner_iteration=True,
            force_accept_inner_with_effective_rho=False,
        ),
        Variant(
            name="adaptive_force_effective_rho_no_inner_refinement",
            refinement_mode="adaptive",
            uniform_x_refinement=False,
            uniform_y_refinement=False,
            compute_g_estimator=not args.no_g_estimator_in_force_variant,
            compute_g_estimator_every_inner_iteration=(
                not args.no_g_estimator_in_force_variant
            ),
            force_accept_inner_with_effective_rho=True,
            max_inner_override=1,
        ),
        Variant(
            name="uniform_standard",
            refinement_mode="uniform_xy",
            uniform_x_refinement=True,
            uniform_y_refinement=True,
            compute_g_estimator=args.compute_g_estimator_in_uniform,
            compute_g_estimator_every_inner_iteration=(
                args.compute_g_estimator_in_uniform
            ),
            force_accept_inner_with_effective_rho=False,
        ),
    )


def thread_policy_for_case(case: ConfigCase, args: argparse.Namespace) -> ThreadPolicy:
    """Return the F3 production thread policy for generated Snellius configs."""
    if case.dimension == 2 and case.degree >= 3:
        cpus = 16
        omp_threads = 8
        mkl_threads = 8
        slab_threads = 16
        local_error_threads = 8
    elif case.dimension == 2 and case.degree == 2:
        cpus = 8
        omp_threads = 8
        mkl_threads = 8
        slab_threads = 4
        local_error_threads = 8
    else:
        cpus = 4
        omp_threads = 4
        mkl_threads = 4
        slab_threads = 4
        local_error_threads = 4

    return ThreadPolicy(
        cpus=args.snellius_cpus if args.snellius_cpus is not None else cpus,
        omp_threads=(
            args.snellius_omp_threads
            if args.snellius_omp_threads is not None
            else omp_threads
        ),
        mkl_threads=(
            args.snellius_mkl_threads
            if args.snellius_mkl_threads is not None
            else mkl_threads
        ),
        main_assembly_threads=args.main_assembly_max_threads,
        main_numeric_fill_threads=args.main_two_pass_numeric_fill_max_threads,
        slab_reconstruction_threads=(
            args.slab_reconstruction_max_threads
            if args.slab_reconstruction_max_threads is not None
            else slab_threads
        ),
        local_error_threads=(
            args.local_error_max_threads
            if args.local_error_max_threads is not None
            else local_error_threads
        ),
    )


def config_entries(case: ConfigCase, args: argparse.Namespace) -> list[tuple[str, object]]:
    variant = case.variant
    max_inner = variant.max_inner_override or args.max_inner
    thread_policy = thread_policy_for_case(case, args)
    output_dir = (
        Path(args.algorithm_output_root)
        / case.dimension_label
        / case.example
        / f"p{case.degree}"
        / variant.name
        / case.target_label
    )

    entries: list[tuple[str, object]] = [
        ("example", case.example),
        ("dimension", case.dimension),
        ("rho", args.rho),
        ("max_outer", args.max_outer),
        ("max_inner", max_inner),
        ("theta_x", args.theta_x),
        ("theta_y", args.theta_y),
        ("uniform_refinement_mode", variant.refinement_mode),
        ("uniform_x_refinement", variant.uniform_x_refinement),
        ("uniform_y_refinement", variant.uniform_y_refinement),
        (
            "force_accept_inner_with_effective_rho",
            variant.force_accept_inner_with_effective_rho,
        ),
        ("compute_g_estimator", variant.compute_g_estimator),
        ("compute_g_estimator_on_empty_y_marking_stop", False),
        (
            "compute_g_estimator_every_inner_iteration",
            variant.compute_g_estimator_every_inner_iteration,
        ),
        ("g_solver", "same_as_main"),
        ("g_solver_tolerance", 0.0),
        ("g_solver_memory_limit_mb", 0.0),
        ("p", case.degree),
        ("zero_tol", args.zero_tol),
        ("divergence_residual_l2_tolerance", args.divergence_tol),
        ("eta_squared_stop", 0.0),
        ("inner_estimator_squared_stop", 0.0),
        ("max_wall_time_seconds", args.max_wall_time_seconds),
        ("max_dofs_target", "x"),
        ("max_x_true_dofs", case.stage_cap),
        ("max_y_true_dofs", 10 * case.stage_cap),
        ("main_solver", "pardiso_ldlt"),
        ("main_solver_pardiso_memory_mode", args.pardiso_memory_mode),
        ("main_solver_max_iterations", args.main_solver_max_iterations),
        ("main_solver_tolerance", args.main_solver_tolerance),
        ("main_solver_symmetry_tolerance", args.main_solver_symmetry_tolerance),
        ("main_solver_direct_residual_retry", True),
        ("main_solver_direct_residual_retry_tolerance", 1.0e-10),
        ("main_solver_ooc_auto_switch", args.main_solver_ooc_auto_switch),
        ("main_solver_ooc_switch_threshold", args.main_solver_ooc_switch_threshold),
        ("main_solver_ooc_switch_to_lu", True),
        ("main_solver_reuse_symbolic_analysis", True),
        ("time_slab_backend", "copied_mesh"),
        ("allow_copied_time_slab_estimator_fallback", True),
        ("output_profile", "production"),
        ("save_heavy_diagnostics", False),
        ("export_history", True),
        ("save_mesh_statistics", True),
        ("save_estimator_components", False),
        ("save_refinement_history", False),
        ("save_iteration_snapshots", False),
        ("save_snapshot_dofs", False),
        ("enable_timing_breakdown", True),
        ("timing_history_filename", "timing_history.csv"),
        ("timing_detail_level", "summary"),
        ("solver_diagnostics.enabled", args.solver_diagnostics),
        ("solver_diagnostics.export_matrix_market", False),
        ("solver_diagnostics.export_rhs", False),
        ("solver_diagnostics.export_solution", False),
        ("solver_diagnostics.max_export_dofs", args.solver_diagnostics_max_export_dofs),
        ("quiet", True),
        ("check_divergence_residual", False),
        ("stop_on_empty_y_marking", True),
        ("use_adaptive_initial_guess", False),
        ("solve_main_system_correction", False),
        ("fused_error_and_flux_diagnostics", True),
        ("local_error_reuse_patch_solve_workspace", True),
        ("deterministic_estimator_reductions", True),
        ("doerfler_near_tie_tolerance", 0.0),
        ("local_error_context_storage", "shared_immutable"),
        ("local_error_state_index_mode", "flat"),
        ("local_error_cell_state_representation", "compact_split"),
        ("local_error_coefficient_fast_path", True),
        ("local_error_cell_state_cache_mode", "off"),
        ("local_error_cell_state_cache_budget_mb", 1024.0),
        ("local_error_flux_diagnostics_mode", "auto"),
        ("local_error_patch_solver", "current_dense"),
        ("slab_reconstruction_operator_mode", "auto"),
        ("shared_context_validation", "off"),
        ("main_assembly_max_threads", thread_policy.main_assembly_threads),
        (
            "main_two_pass_numeric_fill_max_threads",
            thread_policy.main_numeric_fill_threads,
        ),
        ("slab_reconstruction_max_threads", thread_policy.slab_reconstruction_threads),
        ("local_error_max_threads", thread_policy.local_error_threads),
        ("output", output_dir.as_posix()),
    ]

    if args.memory_limit_mb is not None:
        entries.insert(22, ("memory_limit_mb", args.memory_limit_mb))
    if args.memory_reserve_mb is not None:
        entries.insert(23, ("memory_reserve_mb", args.memory_reserve_mb))
    if args.main_solver_memory_limit_mb is not None:
        entries.insert(34, ("main_solver_memory_limit_mb", args.main_solver_memory_limit_mb))

    return entries


def iter_cases(args: argparse.Namespace) -> Iterable[ConfigCase]:
    variants = build_variants(args)
    for dimension in DIMENSIONS:
        for example in EXAMPLES:
            for degree in DEGREES:
                for variant in variants:
                    for stage_cap in args.stage_dof_caps:
                        yield ConfigCase(
                            dimension=dimension,
                            example=example,
                            degree=degree,
                            stage_cap=stage_cap,
                            variant=variant,
                        )


def recommend_resources(case: ConfigCase) -> dict[str, object]:
    """Return conservative first-run resource recommendations.

    These are scheduling defaults for staged production pilots, not guarantees.
    They intentionally bias high for G-estimator and high-p 2D runs.
    """
    dofs = case.stage_cap
    dim = case.dimension
    p = case.degree
    g = case.variant.compute_g_estimator
    force = case.variant.force_accept_inner_with_effective_rho
    uniform = case.variant.uniform_x_refinement and case.variant.uniform_y_refinement

    if dim == 1:
        if dofs <= 1_000_000:
            partition, memory, wall, risk = "rome", 64_000, 24, "low"
        elif dofs <= 4_000_000:
            partition, memory, wall, risk = "rome", 128_000, 72, "medium"
        else:
            partition, memory, wall, risk = "fat_rome", 256_000, 120, "medium"
        if g:
            memory = int(memory * 1.5)
            wall = min(120, int(wall * 1.5))
            risk = "medium" if risk == "low" else "high"
        if p >= 4 and dofs >= 4_000_000:
            risk = "high"
        return {
            "partition": partition,
            "memory_mb": memory,
            "wall_time_hours": wall,
            "risk": risk,
        }

    if dofs <= 500_000:
        partition, memory, wall, risk = "fat_rome", 256_000, 48, "medium"
    elif dofs <= 1_000_000:
        partition, memory, wall, risk = "fat_genoa", 512_000, 72, "high"
    elif dofs <= 4_000_000:
        partition, memory, wall, risk = "himem_4tb", 1_500_000, 120, "very_high"
    else:
        partition, memory, wall, risk = "himem_8tb", 3_000_000, 120, "very_high"

    if g:
        memory = int(memory * (2.0 if p <= 2 else 1.6))
        wall = 120
        risk = "very_high" if dofs >= 1_000_000 or p >= 3 else "high"
    if p >= 4:
        wall = 120
        risk = "very_high"
    elif p >= 3 and dofs >= 1_000_000:
        risk = "very_high"
    if uniform and dofs >= 4_000_000:
        risk = "very_high"
    if force and risk == "medium":
        risk = "medium_high"

    return {
        "partition": partition,
        "memory_mb": memory,
        "wall_time_hours": wall,
        "risk": risk,
    }


def write_index_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    by_target: dict[str, int] = {}
    by_mode: dict[str, int] = {}
    for row in rows:
        by_target[str(row["target_label"])] = by_target.get(str(row["target_label"]), 0) + 1
        by_mode[str(row["mode"])] = by_mode.get(str(row["mode"]), 0) + 1

    lines = [
        "# Snellius Production Run Matrix Index",
        "",
        "Generated by `scripts/snellius/generate_production_configs.py`.",
        "",
        f"- total configs: `{len(rows)}`",
        "- dimensions: `1D`, `2D`",
        "- examples: `smooth_initial`, `non_matching_initial`, `boundary_singularity`",
        "- polynomial degrees: `1`, `2`, `3`, `4`",
        "- modes: `adaptive_standard`, `adaptive_g_effective_rho`, `adaptive_force_effective_rho_no_inner_refinement`, `uniform_standard`",
        "- targets: `pilot_250k`, `pilot_500k`, `pilot_1M`, `production_4M`, `production_6M`",
        "- target semantics: each target is a maximum `X^delta` true-DoF cap; generated configs set `max_y_true_dofs` to 10x the X cap.",
        "",
        "Resource recommendations are conservative scheduling estimates for staged pilots and production runs. They are not guarantees; high-p 2D and G-estimator cases should still be advanced through pilots first.",
        "",
        "## Counts By Target",
        "",
        "| Target | Configs |",
        "|---|---:|",
    ]
    for target in ("pilot_250k", "pilot_500k", "pilot_1M", "production_4M", "production_6M"):
        lines.append(f"| {target} | {by_target.get(target, 0)} |")
    lines.extend(["", "## Counts By Mode", "", "| Mode | Configs |", "|---|---:|"])
    for mode in (
        "adaptive_standard",
        "adaptive_g_effective_rho",
        "adaptive_force_effective_rho_no_inner_refinement",
        "uniform_standard",
    ):
        lines.append(f"| {mode} | {by_mode.get(mode, 0)} |")
    lines.extend([
        "",
        "## Full Index",
        "",
        "| Config | Dim | Example | p | Mode | Target X DoFs | Max Y DoFs | Partition | Memory MB | Wall h | CPUs | OMP | MKL | Risk | G | Force rho |",
        "|---|---:|---|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ])
    for row in rows:
        lines.append(
            "| {config_filename} | {dimension} | {example} | {p} | {mode} | {target_dofs} | {max_y_true_dofs} | {recommended_partition} | {recommended_memory_mb} | {recommended_wall_time_hours} | {recommended_cpus} | {recommended_omp_threads} | {recommended_mkl_threads} | {expected_risk_level} | {g_estimator_enabled} | {force_effective_rho_enabled} |".format(**row)
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_configs(args: argparse.Namespace) -> int:
    output_root = Path(args.output_root)
    cases = list(iter_cases(args))

    if args.dry_run:
        for case in cases[: min(10, len(cases))]:
            print((output_root / case.relative_path).as_posix())
        if len(cases) > 10:
            print(f"... {len(cases) - 10} more")
        print(f"Would generate {len(cases)} configs under {output_root.as_posix()}")
        return len(cases)

    manifest_path = output_root / "manifest.csv"
    index_csv_path = output_root / "index.csv"
    index_md_path = output_root / "index.md"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_fields = [
        "config",
        "case_id",
        "dimension",
        "example",
        "p",
        "mode",
        "variant",
        "refinement_mode",
        "target_label",
        "max_x_true_dofs",
        "max_y_true_dofs",
        "max_dofs_target",
        "max_outer",
        "max_inner",
        "compute_g_estimator",
        "compute_g_estimator_every_inner_iteration",
        "force_accept_inner_with_effective_rho",
        "uniform_x_refinement",
        "uniform_y_refinement",
        "output",
        "memory_limit_mb",
        "main_solver_memory_limit_mb",
        "recommended_cpus",
        "recommended_omp_threads",
        "recommended_mkl_threads",
        "main_assembly_max_threads",
        "main_two_pass_numeric_fill_max_threads",
        "slab_reconstruction_max_threads",
        "local_error_max_threads",
    ]
    index_fields = [
        "config_filename",
        "dimension",
        "example",
        "p",
        "mode",
        "target_dofs",
        "target_label",
        "max_x_true_dofs",
        "max_y_true_dofs",
        "recommended_partition",
        "recommended_memory_mb",
        "recommended_wall_time_hours",
        "recommended_cpus",
        "recommended_omp_threads",
        "recommended_mkl_threads",
        "main_assembly_max_threads",
        "main_two_pass_numeric_fill_max_threads",
        "slab_reconstruction_max_threads",
        "local_error_max_threads",
        "expected_risk_level",
        "g_estimator_enabled",
        "g_estimator_every_inner",
        "force_effective_rho_enabled",
    ]

    index_rows: list[dict[str, object]] = []
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file, index_csv_path.open("w", newline="", encoding="utf-8") as index_file:
        manifest_writer = csv.DictWriter(
            manifest_file,
            lineterminator="\n",
            fieldnames=manifest_fields,
        )
        index_writer = csv.DictWriter(
            index_file,
            lineterminator="\n",
            fieldnames=index_fields,
        )
        manifest_writer.writeheader()
        index_writer.writeheader()

        for case in cases:
            path = output_root / case.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            entries = config_entries(case, args)
            entry_map = dict(entries)
            recommendation = recommend_resources(case)
            thread_policy = thread_policy_for_case(case, args)
            header = [
                "Generated by scripts/snellius/generate_production_configs.py.",
                f"case_id: {case.case_id}",
                f"target_label: {case.target_label}",
                "Target semantics: target label is max_x_true_dofs; max_y_true_dofs is 10x.",
                "Snellius build preset: release-snellius-generic-mkl-pardiso.",
                "Backend: copied_mesh. Output profile: production.",
            ]
            path.write_text(format_config(entries, header), encoding="utf-8")
            rel_config = path.relative_to(output_root).as_posix()
            manifest_writer.writerow(
                {
                    "config": rel_config,
                    "case_id": case.case_id,
                    "dimension": case.dimension,
                    "example": case.example,
                    "p": case.degree,
                    "mode": case.variant.name,
                    "variant": case.variant.name,
                    "refinement_mode": case.variant.refinement_mode,
                    "target_label": case.target_label,
                    "max_x_true_dofs": case.stage_cap,
                    "max_y_true_dofs": 10 * case.stage_cap,
                    "max_dofs_target": "x",
                    "max_outer": args.max_outer,
                    "max_inner": entry_map["max_inner"],
                    "compute_g_estimator": int(case.variant.compute_g_estimator),
                    "compute_g_estimator_every_inner_iteration": int(
                        case.variant.compute_g_estimator_every_inner_iteration
                    ),
                    "force_accept_inner_with_effective_rho": int(
                        case.variant.force_accept_inner_with_effective_rho
                    ),
                    "uniform_x_refinement": int(case.variant.uniform_x_refinement),
                    "uniform_y_refinement": int(case.variant.uniform_y_refinement),
                    "output": entry_map["output"],
                    "memory_limit_mb": (
                        "runner_default"
                        if args.memory_limit_mb is None
                        else args.memory_limit_mb
                    ),
                    "main_solver_memory_limit_mb": (
                        "runner_default"
                        if args.main_solver_memory_limit_mb is None
                        else args.main_solver_memory_limit_mb
                    ),
                    "recommended_cpus": thread_policy.cpus,
                    "recommended_omp_threads": thread_policy.omp_threads,
                    "recommended_mkl_threads": thread_policy.mkl_threads,
                    "main_assembly_max_threads": thread_policy.main_assembly_threads,
                    "main_two_pass_numeric_fill_max_threads": (
                        thread_policy.main_numeric_fill_threads
                    ),
                    "slab_reconstruction_max_threads": (
                        thread_policy.slab_reconstruction_threads
                    ),
                    "local_error_max_threads": thread_policy.local_error_threads,
                }
            )
            index_row = {
                "config_filename": rel_config,
                "dimension": f"{case.dimension}D",
                "example": case.example,
                "p": case.degree,
                "mode": case.variant.name,
                "target_dofs": case.stage_cap,
                "target_label": case.target_label,
                "max_x_true_dofs": case.stage_cap,
                "max_y_true_dofs": 10 * case.stage_cap,
                "recommended_partition": recommendation["partition"],
                "recommended_memory_mb": recommendation["memory_mb"],
                "recommended_wall_time_hours": recommendation["wall_time_hours"],
                "recommended_cpus": thread_policy.cpus,
                "recommended_omp_threads": thread_policy.omp_threads,
                "recommended_mkl_threads": thread_policy.mkl_threads,
                "main_assembly_max_threads": thread_policy.main_assembly_threads,
                "main_two_pass_numeric_fill_max_threads": (
                    thread_policy.main_numeric_fill_threads
                ),
                "slab_reconstruction_max_threads": (
                    thread_policy.slab_reconstruction_threads
                ),
                "local_error_max_threads": thread_policy.local_error_threads,
                "expected_risk_level": recommendation["risk"],
                "g_estimator_enabled": int(case.variant.compute_g_estimator),
                "g_estimator_every_inner": int(
                    case.variant.compute_g_estimator_every_inner_iteration
                ),
                "force_effective_rho_enabled": int(
                    case.variant.force_accept_inner_with_effective_rho
                ),
            }
            index_writer.writerow(index_row)
            index_rows.append(index_row)

    write_index_markdown(index_md_path, index_rows)

    readme_path = output_root / "README.md"
    readme_path.write_text(
        "\n".join(
            [
                "# Snellius Production Configs",
                "",
                "Generated by `scripts/snellius/generate_production_configs.py`.",
                "",
                f"- Config files: `{len(cases)}`",
                f"- Manifest: `{manifest_path.name}`",
                f"- Final index CSV: `{index_csv_path.name}`",
                f"- Final index Markdown: `{index_md_path.name}`",
                "- Snellius default build preset: `release-snellius-generic-mkl-pardiso`",
                "- Time-slab backend: `copied_mesh`",
                "- Output profile: `production`",
                "- Main solver: `pardiso_ldlt`",
                "- F3 thread policy: p=1 uses 4 OMP/MKL threads; 2D p=2/p>=3 use 8 OMP/MKL threads; 2D p>=3 reserves 16 CPUs for slab reconstruction",
                "- Local-error patch solver: `current_dense`",
                "- Flux diagnostics: `auto` streaming reuse",
                "- G variants: `compute_g_estimator_every_inner_iteration=true`",
                "- Target semantics: target labels are maximum `X^delta` true-DoF caps",
                "- Generated Y cap: `max_y_true_dofs = 10 * max_x_true_dofs`",
                "",
                "Run a case with:",
                "",
                "```bash",
                "out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm --config <config.yml>",
                "```",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Generated {len(cases)} configs under {output_root.as_posix()}")
    print(f"Wrote manifest {manifest_path.as_posix()}")
    print(f"Wrote index {index_csv_path.as_posix()}")
    print(f"Wrote index {index_md_path.as_posix()}")
    return len(cases)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Snellius production YAML configs."
    )
    parser.add_argument(
        "--output-root",
        default="production_configs/snellius",
        help="Directory where config files and manifest/index files are written.",
    )
    parser.add_argument(
        "--algorithm-output-root",
        default="algorithm_data/snellius",
        help="Output root written into generated configs.",
    )
    parser.add_argument(
        "--stage-dof-caps",
        type=parse_stage_caps,
        default=DEFAULT_STAGE_CAPS,
        help="Comma-separated X^delta true-DoF caps. Default: 250k,500k,1M,4M,6M. Supports k/m suffixes.",
    )
    parser.add_argument("--rho", type=float, default=1.0)
    parser.add_argument("--theta-x", type=float, default=0.4)
    parser.add_argument("--theta-y", type=float, default=0.4)
    parser.add_argument("--max-outer", type=int, default=200)
    parser.add_argument("--max-inner", type=int, default=50)
    parser.add_argument("--max-wall-time-seconds", type=float, default=0.0)
    parser.add_argument("--zero-tol", type=float, default=1.0e-15)
    parser.add_argument("--divergence-tol", type=float, default=1.0e-4)
    parser.add_argument(
        "--memory-limit-mb",
        type=float,
        default=None,
        help="Optional whole-process memory cap to write into every config.",
    )
    parser.add_argument(
        "--memory-reserve-mb",
        type=float,
        default=None,
        help="Optional MemAvailable reserve to write into every config.",
    )
    parser.add_argument(
        "--main-solver-memory-limit-mb",
        type=float,
        default=None,
        help="Optional direct main-solver memory cap to write into every config.",
    )
    parser.add_argument(
        "--pardiso-memory-mode",
        default="in_core",
        choices=("in_core", "auto", "out_of_core"),
    )
    parser.add_argument("--main-solver-max-iterations", type=int, default=20_000)
    parser.add_argument("--main-solver-tolerance", type=float, default=1.0e-8)
    parser.add_argument("--main-solver-symmetry-tolerance", type=float, default=1.0e-12)
    parser.add_argument(
        "--main-solver-ooc-auto-switch",
        action="store_true",
        help="Write main_solver_ooc_auto_switch: true.",
    )
    parser.add_argument("--main-solver-ooc-switch-threshold", type=float, default=0.85)
    parser.add_argument(
        "--no-g-estimator-in-force-variant",
        action="store_true",
        help="Disable G estimator in the force-effective-rho variant.",
    )
    parser.add_argument(
        "--compute-g-estimator-in-uniform",
        action="store_true",
        help="Enable G estimator in uniform configs. Default: disabled.",
    )
    parser.add_argument(
        "--solver-diagnostics",
        action="store_true",
        help="Enable solver diagnostic metadata in generated configs.",
    )
    parser.add_argument("--solver-diagnostics-max-export-dofs", type=int, default=20_000)
    parser.add_argument("--main-assembly-max-threads", type=int, default=4)
    parser.add_argument("--main-two-pass-numeric-fill-max-threads", type=int, default=4)
    parser.add_argument("--slab-reconstruction-max-threads", type=int, default=None)
    parser.add_argument("--local-error-max-threads", type=int, default=None)
    parser.add_argument("--snellius-cpus", type=int, default=None)
    parser.add_argument("--snellius-omp-threads", type=int, default=None)
    parser.add_argument("--snellius-mkl-threads", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    write_configs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

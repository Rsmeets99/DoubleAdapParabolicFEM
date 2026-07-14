#!/usr/bin/env python3
"""Prepare clean TikZ-ready CSVs from ``algorithm_data_ordered``.

The ordered data tree keeps one directory per scientific case:

    algorithm_data_ordered/<dim>/<example>/<p>/<target>/<mode>/

This script selects, for each ``dim/example/p/mode``, the largest available
X-delta target and writes one long-form CSV per ``dim/example`` pair to
``algorithm_data_plot_cleaned``.  Rows are tagged by ``record_type``:

``outer``
    One row per outer iteration.  Use these rows for estimator-vs-X-dofs plots.

``inner``
    One row per inner iteration.  Use these for raw inner-loop diagnostics.

``inner_summary``
    One row per outer iteration summarizing inner-loop ratios with min/max,
    average, and final values for band plots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


P_RE = re.compile(r"^p(?P<order>\d+)$")
DIM_RE = re.compile(r"^\d+d$")
TARGET_RE = re.compile(r"(\d+(?:\.\d+)?)([kKmM])")

MODE_LABELS = {
    "adaptive_force_effective_rho_no_inner_refinement": "force_eff_rho_no_inner",
    "adaptive_g_effective_rho": "g_eff_rho",
    "adaptive_standard": "adaptive",
    "uniform_standard": "uniform",
}

LATEX_MODE_LABELS = {
    "adaptive_force_effective_rho_no_inner_refinement":
        r"Adaptive (forced $\rho_{\mathrm{eff}}$)",
    "adaptive_g_effective_rho": r"Adaptive ($G$, $\rho_{\mathrm{eff}}$)",
    "adaptive_standard": r"Adaptive",
    "uniform_standard": r"Uniform",
}

LATEX_EXAMPLE_LABELS = {
    "boundary_singularity": "Boundary singularity",
    "non_matching_initial": "Non-matching initial",
    "smooth_initial": "Smooth initial",
}

OUTER_HISTORY_NAMES = ("outer_history.csv", "partial_outer_history.csv")
INNER_HISTORY_NAMES = ("inner_history.csv", "partial_inner_history.csv")

OUTPUT_FIELDS = (
    "record_type",
    "dimension",
    "example",
    "p",
    "p_int",
    "mode",
    "mode_short",
    "target",
    "target_x_dofs",
    "run_status",
    "job_id",
    "run_id",
    "source_group",
    "source_case_relpath",
    "history_source",
    "history_status",
    "outer_iteration",
    "inner_iteration",
    "inner_count",
    "x_true_dofs",
    "x_true_dofs_before",
    "x_true_dofs_after",
    "y_true_dofs_before",
    "y_true_dofs_after",
    "g_true_dofs",
    "iteration_seconds",
    "elapsed_seconds",
    "inner_iterations",
    "y_converged",
    "stopped_on_empty_y_marking",
    "refined_x",
    "uniform_x_refinement",
    "uniform_y_refinement",
    "x_refinement_mode",
    "configured_rho",
    "effective_rho",
    "effective_rho_available",
    "effective_rho_reason",
    "posteriori_factor_configured_rho",
    "posteriori_factor_effective_rho",
    "posteriori_improvement_factor",
    "eta_squared",
    "eta",
    "posteriori_estimator_configured_rho_squared",
    "posteriori_estimator_configured_rho",
    "posteriori_estimator_effective_rho_squared",
    "posteriori_estimator_effective_rho",
    "posteriori_estimator_configured_rho_final10_loglog_slope",
    "posteriori_estimator_configured_rho_final10_convergence_rate",
    "posteriori_estimator_configured_rho_final10_point_count",
    "posteriori_estimator_effective_rho_final10_loglog_slope",
    "posteriori_estimator_effective_rho_final10_convergence_rate",
    "posteriori_estimator_effective_rho_final10_point_count",
    "posteriori_estimator_effective_rho_squared_final10_loglog_slope",
    "posteriori_estimator_effective_rho_squared_final10_convergence_rate",
    "posteriori_estimator_effective_rho_squared_final10_point_count",
    "lambda_y_squared",
    "lambda_y",
    "y_estimator_squared",
    "y_estimator",
    "y_estimator_threshold_squared",
    "y_estimator_threshold",
    "y_flux_squared",
    "y_flux",
    "y_reconstruction_squared",
    "y_reconstruction",
    "divergence_residual_squared",
    "divergence_residual",
    "divergence_residual_l2",
    "flux_over_reconstruction",
    "flux_squared_over_reconstruction_squared",
    "reconstruction_over_flux",
    "reconstruction_squared_over_flux_squared",
    "flux_fraction_of_y_estimator",
    "reconstruction_fraction_of_y_estimator",
    "divergence_over_y_estimator",
    "divergence_squared_over_y_estimator_squared",
    "g_estimator_enabled",
    "g_estimator_computed",
    "g_estimator_skipped_reason",
    "g_solver_status",
    "g_lambda_difference_available",
    "g_lambda_difference_squared",
    "g_lambda_difference",
    "g_over_y_estimator",
    "g_squared_over_y_estimator_squared",
    "flux_over_reconstruction_min",
    "flux_over_reconstruction_max",
    "flux_over_reconstruction_avg",
    "flux_over_reconstruction_final",
    "reconstruction_over_flux_min",
    "reconstruction_over_flux_max",
    "reconstruction_over_flux_avg",
    "reconstruction_over_flux_final",
    "g_over_y_estimator_min",
    "g_over_y_estimator_max",
    "g_over_y_estimator_avg",
    "g_over_y_estimator_final",
    "divergence_over_y_estimator_min",
    "divergence_over_y_estimator_max",
    "divergence_over_y_estimator_avg",
    "divergence_over_y_estimator_final",
    "flux_fraction_min",
    "flux_fraction_max",
    "flux_fraction_avg",
    "flux_fraction_final",
    "g_ratio_value_count",
    "flux_ratio_value_count",
    "reconstruction_ratio_value_count",
    "divergence_ratio_value_count",
)

INDEX_FIELDS = (
    "dimension",
    "example",
    "p",
    "mode",
    "target",
    "target_x_dofs",
    "status",
    "job_id",
    "csv_path",
    "outer_history_source",
    "inner_history_source",
    "outer_rows",
    "inner_rows",
    "inner_summary_rows",
    "outer_effective_posteriori_rows",
    "inner_g_difference_rows",
    "inner_flux_reconstruction_ratio_rows",
    "inner_reconstruction_flux_ratio_rows",
    "inner_divergence_ratio_rows",
    "configured_posteriori_final10_rate",
    "configured_posteriori_final10_points",
    "effective_posteriori_final10_rate",
    "effective_posteriori_final10_points",
    "effective_posteriori_squared_final10_rate",
    "effective_posteriori_squared_final10_points",
    "source_case_relpath",
)


@dataclass(frozen=True)
class Case:
    dim: str
    example: str
    p: str
    mode: str
    target: str
    target_x_dofs: float
    case_dir: Path
    run_dir: Path
    source_manifest: dict[str, object]
    status: str
    job_id: str
    run_id: str
    source_group: str
    sort_epoch: float

    @property
    def p_int(self) -> int:
        match = P_RE.match(self.p)
        return int(match.group("order")) if match else 9999

    @property
    def mode_short(self) -> str:
        return MODE_LABELS.get(self.mode, self.mode)


def target_dofs(target: str) -> float:
    matches = TARGET_RE.findall(target)
    if not matches:
        return -1.0
    value, suffix = matches[-1]
    multiplier = 1_000_000.0 if suffix.lower() == "m" else 1_000.0
    return float(value) * multiplier


def status_score(status: str) -> int:
    if status == "completed":
        return 4
    if status in {"interrupted", "history_present"}:
        return 3
    if status == "partial":
        return 2
    if status == "failed":
        return 1
    return 0


def p_sort_key(p: str) -> tuple[int, str]:
    match = P_RE.match(p)
    return (int(match.group("order")), p) if match else (9999, p)


def dim_sort_key(dim: str) -> tuple[int, str]:
    number = dim[:-1]
    return (int(number), dim) if dim.endswith("d") and number.isdigit() else (9999, dim)


def finite_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"MISSING", "NA", "N/A", "NONE", "NULL"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def safe_sqrt(value: object) -> float | None:
    number = finite_float(value)
    if number is None or number < 0.0:
        return None
    return math.sqrt(number)


def divide(numerator: object, denominator: object) -> float | None:
    top = finite_float(numerator)
    bottom = finite_float(denominator)
    if top is None or bottom is None or bottom == 0.0:
        return None
    return top / bottom


def csv_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.17g}"
    return str(value)


def markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def markdown_table(headers: tuple[str, ...], rows: Iterable[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def latex_rate(value: object) -> str:
    number = finite_float(value)
    if number is None:
        return r"\multicolumn{1}{c}{--}"
    return f"{number:.3f}"


def latex_fit_points(configured_points: object, effective_points: object) -> str:
    configured = str(configured_points)
    effective = str(effective_points)
    if configured == effective:
        return latex_escape(configured)
    return latex_escape(f"{configured}/{effective}")


def latex_target_label(target: object) -> str:
    text = str(target)
    for prefix in ("production_", "pilot_"):
        if text.startswith(prefix):
            text = text[len(prefix):]
            break
    return latex_escape(text)


def convergence_rates_latex_table(index_rows: list[dict[str, object]]) -> str:
    sorted_rows = sorted(
        index_rows,
        key=lambda item: (
            dim_sort_key(str(item["dimension"])),
            str(item["example"]),
            p_sort_key(str(item["p"])),
            str(item["mode"]),
            target_dofs(str(item["target"])),
        ),
    )

    lines = [
        r"\begin{longtable}{lll l l c S[table-format=1.3] S[table-format=1.3]}",
        r"\caption{Final-ten-point non-squared posteriori estimator convergence rates. "
        r"The rates are obtained by least-squares fits of "
        r"$\log(\eta)$ against $\log(\dim X^\delta)$ on the final available "
        r"outer iterations of each run.}\label{tab:apf-nonsquared-convergence-rates}\\",
        r"\toprule",
        r"Dimension & Example & Order & Mode & Target & Fit pts. "
        r"& {$r(\eta_{\rho})$} & {$r(\eta_{\rho_{\mathrm{eff}}})$} \\",
        r"\midrule",
        r"\endfirsthead",
        r"\caption[]{Final-ten-point non-squared posteriori estimator convergence rates (continued).}\\",
        r"\toprule",
        r"Dimension & Example & Order & Mode & Target & Fit pts. "
        r"& {$r(\eta_{\rho})$} & {$r(\eta_{\rho_{\mathrm{eff}}})$} \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{8}{r}{Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]

    previous_group: tuple[str, str] | None = None
    for row in sorted_rows:
        group = (str(row["dimension"]), str(row["example"]))
        if previous_group is not None and group != previous_group:
            lines.append(r"\addlinespace")
        previous_group = group

        dim = latex_escape(str(row["dimension"]).upper())
        example = LATEX_EXAMPLE_LABELS.get(str(row["example"]), str(row["example"]))
        p_label = f"$p={latex_escape(str(row['p']).lstrip('p'))}$"
        mode = LATEX_MODE_LABELS.get(str(row["mode"]), latex_escape(row["mode"]))
        target = latex_target_label(row["target"])
        points = latex_fit_points(
            row["configured_posteriori_final10_points"],
            row["effective_posteriori_final10_points"],
        )
        configured_rate = latex_rate(row["configured_posteriori_final10_rate"])
        effective_rate = latex_rate(row["effective_posteriori_final10_rate"])
        lines.append(
            f"{dim} & {latex_escape(example)} & {p_label} & {mode} & "
            f"{target} & {points} & {configured_rate} & {effective_rate} \\\\"
        )

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def read_json(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def read_key_value_file(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def discover_cases(ordered_root: Path) -> list[Case]:
    cases: list[Case] = []
    for manifest_path in ordered_root.rglob("source_manifest.json"):
        try:
            rel = manifest_path.relative_to(ordered_root)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 6:
            continue
        dim, example, p, target, mode, _ = parts
        if not DIM_RE.match(dim) or not P_RE.match(p):
            continue

        case_dir = manifest_path.parent
        run_dir = case_dir / "run"
        if not run_dir.is_dir():
            continue

        manifest = read_json(manifest_path)
        source = manifest.get("source", {})
        source_dict = source if isinstance(source, dict) else {}
        job_info = manifest.get("job_info", {})
        job_info_dict = job_info if isinstance(job_info, dict) else {}
        fallback_info = read_key_value_file(run_dir / "snellius_job_info.txt")

        status = str(source_dict.get("status") or fallback_info.get("run_status") or "unknown")
        job_id = str(source_dict.get("job_id") or job_info_dict.get("job_id") or fallback_info.get("job_id") or "")
        run_id = str(source_dict.get("run_id") or run_dir.name)
        source_group = str(source_dict.get("algorithm_group") or "")
        sort_epoch_raw = source_dict.get("sort_epoch")
        sort_epoch = finite_float(sort_epoch_raw)
        if sort_epoch is None:
            sort_epoch = manifest_path.stat().st_mtime

        cases.append(
            Case(
                dim=dim,
                example=example,
                p=p,
                mode=mode,
                target=target,
                target_x_dofs=target_dofs(target),
                case_dir=case_dir,
                run_dir=run_dir,
                source_manifest=manifest,
                status=status,
                job_id=job_id,
                run_id=run_id,
                source_group=source_group,
                sort_epoch=sort_epoch,
            )
        )
    return cases


def history_file(run_dir: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = run_dir / name
        if path.exists() and path.stat().st_size > 0:
            return path
    return None


def read_history(path: Path, key_fields: tuple[str, ...]) -> list[dict[str, str]]:
    rows_by_key: dict[tuple[str, ...], dict[str, str]] = {}
    order: list[tuple[str, ...]] = []

    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle)
        header: list[str] | None = None
        for raw_row in reader:
            if not raw_row or not any(cell.strip() for cell in raw_row):
                continue
            if header is None:
                header = [cell.strip().lstrip("\ufeff") for cell in raw_row]
                continue
            if raw_row[: len(header)] == header:
                continue
            padded = raw_row + [""] * max(0, len(header) - len(raw_row))
            row = {field: padded[index].strip() for index, field in enumerate(header)}
            key = tuple(row.get(field, "") for field in key_fields)
            if any(key):
                if key not in rows_by_key:
                    order.append(key)
                rows_by_key[key] = row

    def sort_key(row: dict[str, str]) -> tuple[int, int]:
        outer = int(finite_float(row.get("outer_iteration")) or 0)
        inner = int(finite_float(row.get("inner_iteration")) or 0)
        return (outer, inner)

    return sorted((rows_by_key[key] for key in order), key=sort_key)


def select_highest_cases(cases: Iterable[Case]) -> list[Case]:
    selected: dict[tuple[str, str, str, str], Case] = {}
    for case in cases:
        key = (case.dim, case.example, case.p, case.mode)
        old = selected.get(key)
        if old is None:
            selected[key] = case
            continue
        new_score = (case.target_x_dofs, status_score(case.status), case.sort_epoch)
        old_score = (old.target_x_dofs, status_score(old.status), old.sort_epoch)
        if new_score > old_score:
            selected[key] = case
    return sorted(
        selected.values(),
        key=lambda item: (
            dim_sort_key(item.dim),
            item.example,
            p_sort_key(item.p),
            item.mode,
        ),
    )


def common_row(case: Case, repo_root: Path, record_type: str, history_path: Path | None) -> dict[str, str]:
    try:
        rel_case = case.case_dir.relative_to(repo_root)
    except ValueError:
        rel_case = case.case_dir
    return {
        "record_type": record_type,
        "dimension": case.dim,
        "example": case.example,
        "p": case.p,
        "p_int": str(case.p_int),
        "mode": case.mode,
        "mode_short": case.mode_short,
        "target": case.target,
        "target_x_dofs": csv_value(case.target_x_dofs),
        "run_status": case.status,
        "job_id": case.job_id,
        "run_id": case.run_id,
        "source_group": case.source_group,
        "source_case_relpath": str(rel_case),
        "history_source": history_path.name if history_path else "",
    }


def add_norm_pair(row: dict[str, str], source: dict[str, str], squared_name: str, norm_name: str) -> None:
    squared = source.get(squared_name, "")
    norm = source.get(norm_name, "")
    if not norm:
        norm = csv_value(safe_sqrt(squared))
    row[squared_name] = csv_value(finite_float(squared)) if finite_float(squared) is not None else ""
    row[norm_name] = csv_value(finite_float(norm)) if finite_float(norm) is not None else ""


def add_shared_quantities(row: dict[str, str], source: dict[str, str]) -> None:
    passthrough = (
        "history_status",
        "outer_iteration",
        "inner_iteration",
        "iteration_seconds",
        "elapsed_seconds",
        "y_true_dofs_before",
        "y_true_dofs_after",
        "g_true_dofs",
        "inner_iterations",
        "y_converged",
        "stopped_on_empty_y_marking",
        "refined_x",
        "uniform_x_refinement",
        "uniform_y_refinement",
        "x_refinement_mode",
        "configured_rho",
        "effective_rho",
        "effective_rho_available",
        "effective_rho_reason",
        "posteriori_factor_configured_rho",
        "posteriori_factor_effective_rho",
        "posteriori_improvement_factor",
        "divergence_residual_l2",
        "g_estimator_enabled",
        "g_estimator_computed",
        "g_estimator_skipped_reason",
        "g_solver_status",
        "g_lambda_difference_available",
    )
    for field in passthrough:
        if field in source:
            row[field] = source[field]

    for squared_name, norm_name in (
        ("eta_squared", "eta"),
        ("posteriori_estimator_configured_rho_squared", "posteriori_estimator_configured_rho"),
        ("posteriori_estimator_effective_rho_squared", "posteriori_estimator_effective_rho"),
        ("lambda_y_squared", "lambda_y"),
        ("y_estimator_squared", "y_estimator"),
        ("y_estimator_threshold_squared", "y_estimator_threshold"),
        ("y_flux_squared", "y_flux"),
        ("y_reconstruction_squared", "y_reconstruction"),
        ("divergence_residual_squared", "divergence_residual"),
        ("g_lambda_difference_squared", "g_lambda_difference"),
    ):
        add_norm_pair(row, source, squared_name, norm_name)

    row["flux_over_reconstruction"] = csv_value(
        divide(row.get("y_flux"), row.get("y_reconstruction"))
    )
    row["flux_squared_over_reconstruction_squared"] = csv_value(
        divide(row.get("y_flux_squared"), row.get("y_reconstruction_squared"))
    )
    row["reconstruction_over_flux"] = csv_value(
        divide(row.get("y_reconstruction"), row.get("y_flux"))
    )
    row["reconstruction_squared_over_flux_squared"] = csv_value(
        divide(row.get("y_reconstruction_squared"), row.get("y_flux_squared"))
    )
    row["flux_fraction_of_y_estimator"] = csv_value(
        divide(row.get("y_flux"), row.get("y_estimator"))
    )
    row["reconstruction_fraction_of_y_estimator"] = csv_value(
        divide(row.get("y_reconstruction"), row.get("y_estimator"))
    )
    row["divergence_over_y_estimator"] = csv_value(
        divide(row.get("divergence_residual"), row.get("y_estimator"))
    )
    row["divergence_squared_over_y_estimator_squared"] = csv_value(
        divide(row.get("divergence_residual_squared"), row.get("y_estimator_squared"))
    )
    row["g_over_y_estimator"] = csv_value(
        divide(row.get("g_lambda_difference"), row.get("y_estimator"))
    )
    row["g_squared_over_y_estimator_squared"] = csv_value(
        divide(row.get("g_lambda_difference_squared"), row.get("y_estimator_squared"))
    )


def build_outer_row(
    case: Case,
    repo_root: Path,
    source: dict[str, str],
    history_path: Path | None,
) -> dict[str, str]:
    row = common_row(case, repo_root, "outer", history_path)
    add_shared_quantities(row, source)
    before = source.get("x_true_dofs_before", "")
    after = source.get("x_true_dofs_after", "")
    row["x_true_dofs_before"] = before
    row["x_true_dofs_after"] = after
    row["x_true_dofs"] = before or after
    return normalize_output_row(row)


def build_inner_row(
    case: Case,
    repo_root: Path,
    source: dict[str, str],
    history_path: Path | None,
    x_by_outer: dict[str, tuple[str, str]],
) -> dict[str, str]:
    row = common_row(case, repo_root, "inner", history_path)
    add_shared_quantities(row, source)
    outer = source.get("outer_iteration", "")
    before, after = x_by_outer.get(outer, ("", ""))
    row["x_true_dofs_before"] = before
    row["x_true_dofs_after"] = after
    row["x_true_dofs"] = before or after
    return normalize_output_row(row)


def finite_values(rows: list[dict[str, str]], field: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = finite_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def final_value(rows: list[dict[str, str]], field: str) -> float | None:
    for row in reversed(rows):
        value = finite_float(row.get(field))
        if value is not None:
            return value
    return None


def final_loglog_fit(
    rows: list[dict[str, str]],
    y_field: str,
    max_points: int = 10,
) -> tuple[float | None, float | None, int]:
    points: list[tuple[float, float]] = []
    for row in rows:
        x_value = finite_float(row.get("x_true_dofs"))
        y_value = finite_float(row.get(y_field))
        if x_value is None or y_value is None:
            continue
        if x_value <= 0.0 or y_value <= 0.0:
            continue
        points.append((x_value, y_value))

    points = points[-max_points:]
    if len(points) < 2:
        return None, None, len(points)

    log_x = [math.log(point[0]) for point in points]
    log_y = [math.log(point[1]) for point in points]
    mean_x = sum(log_x) / len(log_x)
    mean_y = sum(log_y) / len(log_y)
    denominator = sum((value - mean_x) ** 2 for value in log_x)
    if denominator <= 0.0:
        return None, None, len(points)

    slope = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(log_x, log_y)
    ) / denominator
    return slope, -slope, len(points)


def add_case_convergence_rates(rows: list[dict[str, str]]) -> dict[str, str]:
    outer_rows = [row for row in rows if row.get("record_type") == "outer"]
    configured_slope, configured_rate, configured_count = final_loglog_fit(
        outer_rows,
        "posteriori_estimator_configured_rho",
    )
    effective_slope, effective_rate, effective_count = final_loglog_fit(
        outer_rows,
        "posteriori_estimator_effective_rho",
    )
    effective_squared_slope, effective_squared_rate, effective_squared_count = final_loglog_fit(
        outer_rows,
        "posteriori_estimator_effective_rho_squared",
    )

    values = {
        "posteriori_estimator_configured_rho_final10_loglog_slope": csv_value(
            configured_slope
        ),
        "posteriori_estimator_configured_rho_final10_convergence_rate": csv_value(
            configured_rate
        ),
        "posteriori_estimator_configured_rho_final10_point_count": str(
            configured_count
        ),
        "posteriori_estimator_effective_rho_final10_loglog_slope": csv_value(
            effective_slope
        ),
        "posteriori_estimator_effective_rho_final10_convergence_rate": csv_value(
            effective_rate
        ),
        "posteriori_estimator_effective_rho_final10_point_count": str(
            effective_count
        ),
        "posteriori_estimator_effective_rho_squared_final10_loglog_slope": csv_value(
            effective_squared_slope
        ),
        "posteriori_estimator_effective_rho_squared_final10_convergence_rate": csv_value(
            effective_squared_rate
        ),
        "posteriori_estimator_effective_rho_squared_final10_point_count": str(
            effective_squared_count
        ),
    }
    for row in rows:
        row.update(values)
    return values


def add_summary_stats(
    row: dict[str, str],
    rows: list[dict[str, str]],
    source_field: str,
    output_prefix: str,
) -> None:
    values = finite_values(rows, source_field)
    row[f"{output_prefix}_min"] = csv_value(min(values) if values else None)
    row[f"{output_prefix}_max"] = csv_value(max(values) if values else None)
    row[f"{output_prefix}_avg"] = csv_value(sum(values) / len(values) if values else None)
    row[f"{output_prefix}_final"] = csv_value(final_value(rows, source_field))


def build_inner_summary_rows(
    case: Case,
    repo_root: Path,
    inner_rows: list[dict[str, str]],
    inner_history_path: Path | None,
    x_by_outer: dict[str, tuple[str, str]],
) -> list[dict[str, str]]:
    by_outer: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inner_rows:
        by_outer[row.get("outer_iteration", "")].append(row)

    summaries: list[dict[str, str]] = []
    for outer, rows in sorted(by_outer.items(), key=lambda item: int(finite_float(item[0]) or 0)):
        summary = common_row(case, repo_root, "inner_summary", inner_history_path)
        before, after = x_by_outer.get(outer, ("", ""))
        summary["outer_iteration"] = outer
        summary["inner_count"] = str(len(rows))
        summary["x_true_dofs_before"] = before
        summary["x_true_dofs_after"] = after
        summary["x_true_dofs"] = before or after
        summary["g_ratio_value_count"] = str(len(finite_values(rows, "g_over_y_estimator")))
        summary["flux_ratio_value_count"] = str(len(finite_values(rows, "flux_over_reconstruction")))
        summary["reconstruction_ratio_value_count"] = str(
            len(finite_values(rows, "reconstruction_over_flux"))
        )
        summary["divergence_ratio_value_count"] = str(
            len(finite_values(rows, "divergence_over_y_estimator"))
        )
        add_summary_stats(
            summary,
            rows,
            "flux_over_reconstruction",
            "flux_over_reconstruction",
        )
        add_summary_stats(
            summary,
            rows,
            "reconstruction_over_flux",
            "reconstruction_over_flux",
        )
        add_summary_stats(summary, rows, "g_over_y_estimator", "g_over_y_estimator")
        add_summary_stats(
            summary,
            rows,
            "divergence_over_y_estimator",
            "divergence_over_y_estimator",
        )
        add_summary_stats(summary, rows, "flux_fraction_of_y_estimator", "flux_fraction")
        summaries.append(normalize_output_row(summary))
    return summaries


def normalize_output_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: "" for field in OUTPUT_FIELDS}
    for key, value in row.items():
        if key in normalized:
            normalized[key] = csv_value(value)
    return normalized


def process_case(
    case: Case,
    repo_root: Path,
) -> tuple[list[dict[str, str]], dict[str, object]]:
    outer_path = history_file(case.run_dir, OUTER_HISTORY_NAMES)
    inner_path = history_file(case.run_dir, INNER_HISTORY_NAMES)

    outer_source_rows = (
        read_history(outer_path, ("outer_iteration",)) if outer_path is not None else []
    )
    outer_rows = [
        build_outer_row(case, repo_root, source_row, outer_path)
        for source_row in outer_source_rows
    ]
    x_by_outer = {
        row.get("outer_iteration", ""): (
            row.get("x_true_dofs_before", ""),
            row.get("x_true_dofs_after", ""),
        )
        for row in outer_rows
    }

    inner_source_rows = (
        read_history(inner_path, ("outer_iteration", "inner_iteration"))
        if inner_path is not None
        else []
    )
    inner_rows = [
        build_inner_row(case, repo_root, source_row, inner_path, x_by_outer)
        for source_row in inner_source_rows
    ]
    summary_rows = build_inner_summary_rows(
        case,
        repo_root,
        inner_rows,
        inner_path,
        x_by_outer,
    )

    all_rows = outer_rows + inner_rows + summary_rows
    rate_values = add_case_convergence_rates(all_rows)
    index_row = {
        "dimension": case.dim,
        "example": case.example,
        "p": case.p,
        "mode": case.mode,
        "target": case.target,
        "target_x_dofs": csv_value(case.target_x_dofs),
        "status": case.status,
        "job_id": case.job_id,
        "outer_history_source": outer_path.name if outer_path else "",
        "inner_history_source": inner_path.name if inner_path else "",
        "outer_rows": len(outer_rows),
        "inner_rows": len(inner_rows),
        "inner_summary_rows": len(summary_rows),
        "outer_effective_posteriori_rows": len(
            finite_values(outer_rows, "posteriori_estimator_effective_rho")
        ),
        "inner_g_difference_rows": len(finite_values(inner_rows, "g_lambda_difference")),
        "inner_flux_reconstruction_ratio_rows": len(
            finite_values(inner_rows, "flux_over_reconstruction")
        ),
        "inner_reconstruction_flux_ratio_rows": len(
            finite_values(inner_rows, "reconstruction_over_flux")
        ),
        "inner_divergence_ratio_rows": len(
            finite_values(inner_rows, "divergence_over_y_estimator")
        ),
        "configured_posteriori_final10_rate": rate_values[
            "posteriori_estimator_configured_rho_final10_convergence_rate"
        ],
        "configured_posteriori_final10_points": rate_values[
            "posteriori_estimator_configured_rho_final10_point_count"
        ],
        "effective_posteriori_final10_rate": rate_values[
            "posteriori_estimator_effective_rho_final10_convergence_rate"
        ],
        "effective_posteriori_final10_points": rate_values[
            "posteriori_estimator_effective_rho_final10_point_count"
        ],
        "effective_posteriori_squared_final10_rate": rate_values[
            "posteriori_estimator_effective_rho_squared_final10_convergence_rate"
        ],
        "effective_posteriori_squared_final10_points": rate_values[
            "posteriori_estimator_effective_rho_squared_final10_point_count"
        ],
    }
    try:
        index_row["source_case_relpath"] = str(case.case_dir.relative_to(repo_root))
    except ValueError:
        index_row["source_case_relpath"] = str(case.case_dir)
    return all_rows, index_row


def write_csv(path: Path, rows: list[dict[str, str]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_readme(
    output_root: Path,
    index_rows: list[dict[str, object]],
    file_rows: list[dict[str, object]],
    ordered_root: Path,
) -> None:
    status_counts: dict[str, int] = defaultdict(int)
    for row in index_rows:
        status_counts[str(row["status"])] += 1
    status_text = ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items()))

    file_table = [
        (
            row["dimension"],
            row["example"],
            row["csv_path"],
            row["case_count"],
            row["outer_rows"],
            row["inner_rows"],
            row["inner_summary_rows"],
        )
        for row in sorted(
            file_rows,
            key=lambda item: (dim_sort_key(str(item["dimension"])), str(item["example"])),
        )
    ]

    selected_table = [
        (
            row["dimension"],
            row["example"],
            row["p"],
            MODE_LABELS.get(str(row["mode"]), str(row["mode"])),
            row["target"],
            row["status"],
            row["job_id"] or "-",
            row["outer_rows"],
            row["inner_rows"],
            row["inner_g_difference_rows"],
            row["inner_flux_reconstruction_ratio_rows"],
            row["inner_reconstruction_flux_ratio_rows"],
            row["inner_divergence_ratio_rows"],
        )
        for row in sorted(
            index_rows,
            key=lambda item: (
                dim_sort_key(str(item["dimension"])),
                str(item["example"]),
                p_sort_key(str(item["p"])),
                str(item["mode"]),
            ),
        )
    ]

    latex_convergence_table = convergence_rates_latex_table(index_rows)

    text = f"""# Cleaned Plot Data

This directory is managed by `scripts/snellius/prepare_plot_data.py`.

Source directory:

`{ordered_root}`

The script selects the largest available X-delta target for every
`dim/example/p/mode` case in `algorithm_data_ordered`, then writes one
TikZ-friendly long-form CSV per `dim/example` pair.

Last update: `{datetime.now(timezone.utc).isoformat(timespec="seconds")}`

Selected cases: `{len(index_rows)}`. Status counts: `{status_text}`.

## CSV Layout

Use `record_type=outer` for estimator-vs-X-dofs convergence plots.
Use `record_type=inner` for raw inner-iteration diagnostics.
Use `record_type=inner_summary` for one row per outer iteration containing
min/max/average/final ratio values for band plots.

Important plotting columns:

- `x_true_dofs`: the X-delta DoFs of the space on which the estimator was computed.
- `posteriori_estimator_configured_rho`: square root of the configured-rho posteriori estimator squared.
- `posteriori_estimator_effective_rho`: square root of the effective-rho posteriori estimator squared, blank when no effective rho is available.
- `g_lambda_difference`: the available G-delta difference norm.
- `y_estimator`: the inner-loop estimator norm.
- `y_flux` and `y_reconstruction`: the two inner-estimator component norms.
- `divergence_residual`: the inner-loop divergence/PDE-balance residual norm.
- `flux_over_reconstruction`: `y_flux / y_reconstruction`, i.e. theta_flux / theta_rec.
- `reconstruction_over_flux`: `y_reconstruction / y_flux`, i.e. theta_rec / theta_flux.
- `divergence_over_y_estimator`: `divergence_residual / y_estimator`.
- `g_over_y_estimator`: `g_lambda_difference / y_estimator`.
- `posteriori_estimator_configured_rho_final10_convergence_rate`: `-slope`
  from a log-log least-squares fit of `posteriori_estimator_configured_rho`
  against `x_true_dofs` on the final up-to-10 positive outer rows.
- `posteriori_estimator_effective_rho_final10_convergence_rate`: `-slope`
  from the same fit using the non-squared `posteriori_estimator_effective_rho`.
- `posteriori_estimator_effective_rho_squared_final10_convergence_rate`: `-slope`
  from the same fit using `posteriori_estimator_effective_rho_squared`.

The squared quantities are kept next to the square-rooted plotting quantities
for checking and for log-log plots based on squared norms if desired.

## Appendix LaTeX Convergence-Rate Table

The table below uses the non-squared posteriori estimators and is intended for
paper appendices. It assumes the LaTeX packages `booktabs`, `longtable`, and
`siunitx`.

```latex
{latex_convergence_table}
```

## Generated Files

{markdown_table(("dim", "example", "csv", "cases", "outer", "inner", "summary"), file_table)}

## Selected Highest-Target Cases

{markdown_table(("dim", "example", "p", "mode", "target", "status", "job", "outer", "inner", "G rows", "flux/recon rows", "recon/flux rows", "div/y rows"), selected_table)}
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create TikZ-ready plot CSVs from algorithm_data_ordered.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to the parent of scripts/snellius.",
    )
    parser.add_argument(
        "--ordered-root",
        type=Path,
        default=None,
        help="Input ordered data root. Defaults to <repo-root>/algorithm_data_ordered.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Output root. Defaults to <repo-root>/algorithm_data_plot_cleaned.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned files without writing output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    ordered_root = (args.ordered_root or repo_root / "algorithm_data_ordered").resolve()
    output_root = (args.output_root or repo_root / "algorithm_data_plot_cleaned").resolve()

    if not ordered_root.exists():
        print(f"error: ordered root does not exist: {ordered_root}", file=sys.stderr)
        return 2

    cases = discover_cases(ordered_root)
    selected = select_highest_cases(cases)
    if not selected:
        print(f"error: no ordered cases found below {ordered_root}", file=sys.stderr)
        return 2

    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    index_rows: list[dict[str, object]] = []

    for case in selected:
        rows, index_row = process_case(case, repo_root)
        key = (case.dim, case.example)
        grouped_rows[key].extend(rows)
        index_rows.append(index_row)

    file_rows: list[dict[str, object]] = []
    for (dim, example), rows in sorted(
        grouped_rows.items(), key=lambda item: (dim_sort_key(item[0][0]), item[0][1])
    ):
        rows.sort(
            key=lambda row: (
                p_sort_key(row["p"]),
                row["mode"],
                {"outer": 0, "inner": 1, "inner_summary": 2}.get(row["record_type"], 9),
                int(finite_float(row.get("outer_iteration")) or -1),
                int(finite_float(row.get("inner_iteration")) or -1),
            )
        )
        csv_path = output_root / dim / f"{example}.csv"
        rel_csv = csv_path.relative_to(output_root)
        case_count = len({(row["p"], row["mode"], row["target"]) for row in rows})
        file_rows.append(
            {
                "dimension": dim,
                "example": example,
                "csv_path": str(rel_csv),
                "case_count": case_count,
                "outer_rows": sum(1 for row in rows if row["record_type"] == "outer"),
                "inner_rows": sum(1 for row in rows if row["record_type"] == "inner"),
                "inner_summary_rows": sum(
                    1 for row in rows if row["record_type"] == "inner_summary"
                ),
            }
        )
        if args.dry_run:
            print(f"DRY-RUN write {csv_path} rows={len(rows)}")
        else:
            write_csv(csv_path, rows, OUTPUT_FIELDS)

    for row in index_rows:
        csv_rel = Path(str(row["dimension"])) / f"{row['example']}.csv"
        row["csv_path"] = str(csv_rel)

    if args.dry_run:
        print(f"DRY-RUN write {output_root / 'index.csv'} rows={len(index_rows)}")
        print(f"DRY-RUN write {output_root / 'README.md'}")
    else:
        write_csv(
            output_root / "index.csv",
            [{field: csv_value(row.get(field, "")) for field in INDEX_FIELDS} for row in index_rows],
            INDEX_FIELDS,
        )
        write_csv(
            output_root / "files.csv",
            [
                {
                    "dimension": csv_value(row["dimension"]),
                    "example": csv_value(row["example"]),
                    "csv_path": csv_value(row["csv_path"]),
                    "case_count": csv_value(row["case_count"]),
                    "outer_rows": csv_value(row["outer_rows"]),
                    "inner_rows": csv_value(row["inner_rows"]),
                    "inner_summary_rows": csv_value(row["inner_summary_rows"]),
                }
                for row in file_rows
            ],
            (
                "dimension",
                "example",
                "csv_path",
                "case_count",
                "outer_rows",
                "inner_rows",
                "inner_summary_rows",
            ),
        )
        write_readme(output_root, index_rows, file_rows, ordered_root)

    print(
        "summary: "
        f"discovered={len(cases)} selected={len(selected)} "
        f"dim_example_files={len(grouped_rows)} output_root={output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

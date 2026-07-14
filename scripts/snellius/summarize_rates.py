#!/usr/bin/env python3
"""Summarize staged Snellius convergence sequences and rates."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


METRIC_CANDIDATES = (
    ("exact_error", ("final_exact_error", "final_error", "final_l2_error", "final_h10_error"), False),
    ("exact_error_squared", ("final_exact_error_squared", "final_error_squared", "final_l2_error_squared"), True),
    ("effective_rho_estimator", ("final_posteriori_estimator_effective_rho_squared",), True),
    ("configured_rho_estimator", ("final_posteriori_estimator_configured_rho_squared",), True),
    ("g_lambda_difference", ("final_g_lambda_difference",), False),
    ("g_lambda_difference_squared", ("final_g_lambda_difference_squared",), True),
    ("eta", ("final_eta",), False),
    ("eta_squared", ("final_eta_squared",), True),
    ("y_estimator", ("final_y_estimator",), False),
    ("y_estimator_squared", ("final_y_estimator_squared",), True),
)


def parse_float(text: str | None) -> float | None:
    if text is None or text in ("", "MISSING", "NA", "nan"):
        return None
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= 0.0:
        return None
    return value


def metric_value(row: dict[str, str], columns: Iterable[str], squared: bool) -> tuple[float | None, str]:
    for column in columns:
        value = parse_float(row.get(column))
        if value is None:
            continue
        return (math.sqrt(value) if squared else value), column
    return None, "MISSING"


def linear_slope(points: list[tuple[float, float]]) -> float:
    n = len(points)
    mean_x = sum(x for x, _ in points) / n
    mean_y = sum(y for _, y in points) / n
    denom = sum((x - mean_x) ** 2 for x, _ in points)
    if denom == 0.0:
        return float("nan")
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denom


def target_order(row: dict[str, str]) -> int:
    for key in ("max_x_true_dofs", "final_x_true_dofs", "max_y_true_dofs"):
        value = parse_float(row.get(key))
        if value is not None:
            return int(value)
    return 0


def is_usable_row(row: dict[str, str], include_partial: bool) -> bool:
    status = row.get("run_status", "")
    if status in ("completed_or_terminated_with_full_history", "summary_only"):
        return True
    return include_partial and "partial" in status


def build_sequence_rows(rows: list[dict[str, str]], group_columns: list[str], dof_column: str, include_partial: bool) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not is_usable_row(row, include_partial):
            continue
        key = tuple(row.get(column, "MISSING") for column in group_columns)
        groups[key].append(row)

    sequence_rows: list[dict[str, str]] = []
    summary_rows: list[dict[str, str]] = []

    for key, group_rows in sorted(groups.items()):
        group_rows.sort(key=target_order)
        for metric_name, columns, squared in METRIC_CANDIDATES:
            metric_points: list[tuple[float, float]] = []
            used_targets: list[str] = []
            previous_dofs: float | None = None
            previous_value: float | None = None
            previous_target = "MISSING"
            for row in group_rows:
                dofs = parse_float(row.get(dof_column))
                value, source_column = metric_value(row, columns, squared)
                if dofs is None or value is None:
                    continue
                if previous_dofs is not None and previous_value is not None and dofs != previous_dofs:
                    pair_rate = -math.log(value / previous_value) / math.log(dofs / previous_dofs)
                else:
                    pair_rate = float("nan")
                seq = {column: value_ for column, value_ in zip(group_columns, key)}
                seq.update(
                    {
                        "target_label": row.get("target_label", "MISSING"),
                        "run_status": row.get("run_status", "MISSING"),
                        "dof_column": dof_column,
                        "dofs": f"{dofs:.12g}",
                        "metric": metric_name,
                        "metric_source_column": source_column,
                        "metric_value": f"{value:.12g}",
                        "previous_target_label": previous_target,
                        "rate_from_previous": "MISSING" if math.isnan(pair_rate) else f"{pair_rate:.12g}",
                    }
                )
                sequence_rows.append(seq)
                metric_points.append((math.log(dofs), math.log(value)))
                used_targets.append(row.get("target_label", "MISSING"))
                previous_dofs = dofs
                previous_value = value
                previous_target = row.get("target_label", "MISSING")

            if len(metric_points) >= 2:
                slope = linear_slope(metric_points)
                rate = -slope
                status = "ok"
            else:
                slope = float("nan")
                rate = float("nan")
                status = "insufficient_points"
            summary = {column: value_ for column, value_ in zip(group_columns, key)}
            summary.update(
                {
                    "metric": metric_name,
                    "status": status,
                    "n_points": str(len(metric_points)),
                    "targets_used": ";".join(used_targets),
                    "dof_column": dof_column,
                    "slope_log_metric_vs_log_dofs": "MISSING" if math.isnan(slope) else f"{slope:.12g}",
                    "estimated_rate": "MISSING" if math.isnan(rate) else f"{rate:.12g}",
                }
            )
            summary_rows.append(summary)

    return sequence_rows, summary_rows


def write_csv(path: Path, rows: list[dict[str, str]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summary_rows: list[dict[str, str]], sequence_rows: list[dict[str, str]]) -> None:
    ok = [row for row in summary_rows if row.get("status") == "ok"]
    lines = [
        "# Snellius Rate Summary",
        "",
        f"- summary rows: `{len(summary_rows)}`",
        f"- sequence rows: `{len(sequence_rows)}`",
        f"- fitted rates: `{len(ok)}`",
        "- default DoF axis: `final_x_true_dofs`",
        "",
        "Rates are fitted as `-slope(log(metric), log(DoFs))`.  Squared estimator columns are square-rooted before fitting.",
        "",
        "## Fitted Rates",
        "",
        "| Dimension | Example | p | Mode | Metric | Points | Targets | Rate | Status |",
        "|---|---|---:|---|---|---:|---|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {dimension} | {example} | {p} | {mode} | {metric} | {n_points} | {targets_used} | {estimated_rate} | {status} |".format(**row)
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", default="production_configs/snellius/collected_results.csv")
    parser.add_argument("--out", default="production_configs/snellius/rate_summary.csv", help="Summary CSV output.")
    parser.add_argument("--sequence-out", default="production_configs/snellius/rate_sequences.csv")
    parser.add_argument("--markdown-out", default="production_configs/snellius/rate_summary.md")
    parser.add_argument("--dof-column", default="final_x_true_dofs")
    parser.add_argument("--group-columns", default="dimension,example,p,mode")
    parser.add_argument("--include-partial", action="store_true", help="Use partial-history rows when they have usable DoFs/metrics.")
    args = parser.parse_args()

    input_path = Path(args.results_csv)
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    group_columns = [item.strip() for item in args.group_columns.split(",") if item.strip()]
    sequence_rows, summary_rows = build_sequence_rows(rows, group_columns, args.dof_column, args.include_partial)

    preferred_summary = group_columns + ["metric", "status", "n_points", "targets_used", "dof_column", "slope_log_metric_vs_log_dofs", "estimated_rate"]
    preferred_sequence = group_columns + ["target_label", "run_status", "dof_column", "dofs", "metric", "metric_source_column", "metric_value", "previous_target_label", "rate_from_previous"]
    write_csv(Path(args.out), summary_rows, preferred_summary)
    write_csv(Path(args.sequence_out), sequence_rows, preferred_sequence)
    write_markdown(Path(args.markdown_out), summary_rows, sequence_rows)
    print(f"Wrote {len(summary_rows)} summary rows to {args.out}")
    print(f"Wrote {len(sequence_rows)} sequence rows to {args.sequence_out}")
    print(f"Wrote Markdown summary to {args.markdown_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

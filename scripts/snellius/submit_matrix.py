#!/usr/bin/env python3
"""Submit selected Snellius production configs from the generated manifest."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DEFAULT_MANIFEST = "production_configs/snellius/manifest.csv"
DEFAULT_STAGE_SUBMITTER = "scripts/snellius/submit_stage.py"


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def split_filter(text: str | None) -> set[str]:
    if not text:
        return set()
    return {item.strip() for item in text.split(",") if item.strip()}


def normalize_dimension(text: str) -> str:
    lowered = text.lower().strip()
    if lowered.endswith("d"):
        lowered = lowered[:-1]
    return lowered


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_project_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def row_matches(row: dict[str, str], args: argparse.Namespace) -> bool:
    filters = {
        "dimension": split_filter(args.dimension),
        "example": split_filter(args.example),
        "p": split_filter(args.p),
        "mode": split_filter(args.mode),
        "target_label": split_filter(args.target),
    }
    if filters["dimension"]:
        allowed = {normalize_dimension(item) for item in filters["dimension"]}
        if normalize_dimension(row.get("dimension", "")) not in allowed:
            return False
    if filters["example"] and row.get("example", "") not in filters["example"]:
        return False
    if filters["p"] and row.get("p", "") not in filters["p"]:
        return False
    if filters["mode"] and row.get("mode", row.get("variant", "")) not in filters["mode"]:
        return False
    if filters["target_label"] and row.get("target_label", "") not in filters["target_label"]:
        return False
    return True


def target_order(row: dict[str, str]) -> int:
    for key in ("max_x_true_dofs", "target_dofs", "max_y_true_dofs"):
        value = row.get(key)
        if value:
            try:
                return int(float(value))
            except ValueError:
                pass
    label = row.get("target_label", "")
    match = re.search(r"(\d+)([kKmM]?)", label)
    if not match:
        return 0
    value = int(match.group(1))
    suffix = match.group(2).lower()
    if suffix == "k":
        return value * 1000
    if suffix == "m":
        return value * 1000000
    return value


def profile_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row.get("dimension", ""),
        row.get("example", ""),
        row.get("p", ""),
        row.get("mode", row.get("variant", "")),
    )


def parse_job_id(output: str) -> str | None:
    for line in output.splitlines():
        match = re.search(r"Submitted batch job\s+(\S+)", line)
        if match:
            return match.group(1)
    return None


def submit_row(row: dict[str, str], args: argparse.Namespace, project_root: Path, dependency: str | None) -> tuple[int, str, str | None]:
    stage_submitter = resolve_project_path(project_root, args.stage_submitter)
    config_path = resolve_project_path(resolve_project_path(project_root, args.config_root), row["config"])
    command = [
        sys.executable,
        str(stage_submitter),
        "--config",
        str(config_path),
        "--project-root",
        str(project_root),
        "--index",
        args.index,
        "--config-root",
        args.config_root,
        "--cpus",
        str(args.cpus),
    ]
    passthrough = [
        ("--partition", args.partition),
        ("--time", args.time),
        ("--memory-mb", args.memory_mb),
        ("--account", args.account),
        ("--qos", args.qos),
        ("--executable", args.executable),
        ("--logs-dir", args.logs_dir),
        ("--sbatch-dir", args.sbatch_dir),
        ("--run-id", args.run_id),
    ]
    for flag, value in passthrough:
        if value:
            command.extend([flag, str(value)])
    if dependency:
        command.extend(["--dependency", f"afterok:{dependency}"])
    if args.load_modules:
        command.append("--load-modules")
        command.extend(["--module-set", args.module_set])
    if args.use_scratch:
        command.append("--use-scratch")
    if args.dry_run:
        command.append("--dry-run")
    if args.print_commands:
        command.append("--print-command")

    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    return completed.returncode, completed.stdout, parse_job_id(completed.stdout)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root_from_script()))
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--index", default="production_configs/snellius/index.csv")
    parser.add_argument("--config-root", default="production_configs/snellius")
    parser.add_argument("--stage-submitter", default=DEFAULT_STAGE_SUBMITTER)
    parser.add_argument("--dimension", help="Comma-separated dimensions, e.g. 1,2.")
    parser.add_argument("--example", help="Comma-separated examples.")
    parser.add_argument("--p", help="Comma-separated polynomial degrees.")
    parser.add_argument("--mode", help="Comma-separated modes.")
    parser.add_argument("--target", help="Comma-separated target labels, e.g. pilot_250k,production_6M.")
    parser.add_argument("--limit", type=int, default=0, help="Submit at most N selected configs.")
    parser.add_argument("--chain-stages", action="store_true", help="Chain targets per dimension/example/p/mode with afterok.")
    parser.add_argument("--partition")
    parser.add_argument("--time")
    parser.add_argument("--memory-mb")
    parser.add_argument("--cpus", type=int, default=16)
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--executable")
    parser.add_argument("--logs-dir")
    parser.add_argument("--sbatch-dir")
    parser.add_argument("--run-id", help="Explicit run directory id under output/runs/. Defaults to submit timestamp plus SLURM job id.")
    parser.add_argument("--load-modules", action="store_true", help="Load the default Snellius runtime module stack inside each generated sbatch job.")
    parser.add_argument("--module-set", default="2025a", help="Module stack used with --load-modules. Supported: 2025a.")
    parser.add_argument("--use-scratch", action="store_true", help="Run jobs on local scratch and copy back on exit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-commands", action="store_true")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    project_root = resolve_project_path(Path.cwd(), args.project_root).resolve()
    manifest = resolve_project_path(project_root, args.manifest)
    rows = [row for row in read_manifest(manifest) if row_matches(row, args)]
    rows.sort(key=lambda row: (*profile_key(row), target_order(row)))
    if args.limit > 0:
        rows = rows[: args.limit]

    last_job_by_profile: dict[tuple[str, str, str, str], str] = {}
    submitted = 0
    failed = 0
    for row in rows:
        key = profile_key(row)
        dependency = last_job_by_profile.get(key) if args.chain_stages else None
        code, output, job_id = submit_row(row, args, project_root, dependency)
        print(output, end="" if output.endswith("\n") else "\n")
        submitted += 1
        if code != 0:
            failed += 1
        elif args.chain_stages:
            last_job_by_profile[key] = job_id or f"DRYRUN{submitted}"

    print(f"selected_configs={len(rows)}")
    print(f"submitted_attempts={submitted}")
    print(f"failed_attempts={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

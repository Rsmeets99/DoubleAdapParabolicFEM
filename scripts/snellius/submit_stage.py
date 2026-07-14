#!/usr/bin/env python3
"""Submit one Snellius production config as a staged SLURM job.

This Python wrapper reads the generated production index when available and
passes the recommended partition, memory, and wall time to submit_single_run.sh.
It is intentionally thin: the shell script still owns sbatch-file generation and
scratch-copy behavior.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_INDEX = "production_configs/snellius/index.csv"
DEFAULT_SUBMITTER = "scripts/snellius/submit_single_run.sh"


def project_root_from_script() -> Path:
    return Path(__file__).resolve().parents[2]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def resolve_project_path(project_root: Path, path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else project_root / path


def config_key(project_root: Path, config_root: Path, config: Path) -> str:
    resolved = config.resolve()
    for base in (config_root.resolve(), project_root.resolve()):
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            pass
    return config.as_posix()


def find_index_row(project_root: Path, index_path: Path, config_root: Path, config: Path) -> dict[str, str]:
    rows = read_csv_rows(index_path)
    if not rows:
        return {}
    keys = {
        config_key(project_root, config_root, config),
        config.as_posix(),
        config.name,
    }
    try:
        keys.add(config.resolve().relative_to(project_root.resolve()).as_posix())
    except ValueError:
        pass
    for row in rows:
        value = row.get("config_filename", "")
        if value in keys:
            return row
        candidate = (config_root / value).resolve()
        if candidate == config.resolve():
            return row
    return {}


def slurm_time_from_hours(hours_text: str, fallback: str = "120:00:00") -> str:
    if not hours_text:
        return fallback
    if ":" in hours_text:
        return hours_text
    try:
        hours = float(hours_text)
    except ValueError:
        return fallback
    if hours <= 0:
        return fallback
    whole_hours = int(hours)
    minutes = int(round((hours - whole_hours) * 60.0))
    if minutes >= 60:
        whole_hours += 1
        minutes -= 60
    return f"{whole_hours}:{minutes:02d}:00"


def positive_int_text(text: str | None) -> str | None:
    if text is None or text == "":
        return None
    try:
        value = int(float(text))
    except ValueError:
        return None
    return str(value) if value > 0 else None


def build_submit_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    project_root = resolve_project_path(Path.cwd(), args.project_root).resolve()
    config = resolve_project_path(project_root, args.config).resolve()
    index_path = resolve_project_path(project_root, args.index)
    config_root = resolve_project_path(project_root, args.config_root)
    submitter = resolve_project_path(project_root, args.submitter)

    if not config.exists():
        raise FileNotFoundError(f"config not found: {config}")
    if not submitter.exists():
        raise FileNotFoundError(f"submitter not found: {submitter}")

    row = find_index_row(project_root, index_path, config_root, config)

    partition = args.partition or row.get("recommended_partition") or "rome"
    memory_mb = args.memory_mb or positive_int_text(row.get("recommended_memory_mb")) or "auto"
    time_limit = args.time or slurm_time_from_hours(row.get("recommended_wall_time_hours", ""))

    executable = args.executable or str(project_root / "out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm")
    cpus = args.cpus or row.get("recommended_cpus") or "16"
    omp_threads = args.omp_threads or row.get("recommended_omp_threads") or cpus
    mkl_threads = args.mkl_threads or row.get("recommended_mkl_threads") or omp_threads

    command = [
        str(submitter),
        "--config",
        str(config),
        "--partition",
        partition,
        "--time",
        time_limit,
        "--memory-mb",
        memory_mb,
        "--cpus",
        str(cpus),
        "--omp-threads",
        str(omp_threads),
        "--mkl-threads",
        str(mkl_threads),
        "--project-root",
        str(project_root),
        "--executable",
        executable,
    ]
    passthrough = [
        ("--account", args.account),
        ("--qos", args.qos),
        ("--dependency", args.dependency),
        ("--logs-dir", args.logs_dir),
        ("--sbatch-dir", args.sbatch_dir),
        ("--job-name", args.job_name),
        ("--run-id", args.run_id),
    ]
    for flag, value in passthrough:
        if value:
            command.extend([flag, str(value)])
    if args.load_modules:
        command.append("--load-modules")
        command.extend(["--module-set", args.module_set])
    if args.use_scratch:
        command.append("--use-scratch")
    if args.dry_run:
        command.append("--dry-run")

    metadata = {
        "config": str(config),
        "partition": partition,
        "memory_mb": memory_mb,
        "time": time_limit,
        "index_row_found": "1" if row else "0",
        "target_label": row.get("target_label", "MISSING"),
        "max_x_true_dofs": row.get("max_x_true_dofs", "MISSING"),
        "max_y_true_dofs": row.get("max_y_true_dofs", "MISSING"),
        "cpus": str(cpus),
        "omp_threads": str(omp_threads),
        "mkl_threads": str(mkl_threads),
        "run_id": args.run_id or "auto",
        "load_modules": "1" if args.load_modules else "0",
        "module_set": args.module_set,
    }
    return command, metadata


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Config YAML to submit.")
    parser.add_argument("--project-root", default=str(project_root_from_script()))
    parser.add_argument("--index", default=DEFAULT_INDEX, help="Generated Snellius index CSV.")
    parser.add_argument("--config-root", default="production_configs/snellius")
    parser.add_argument("--submitter", default=DEFAULT_SUBMITTER)
    parser.add_argument("--partition", help="Override recommended partition.")
    parser.add_argument("--time", help="Override recommended SLURM time limit.")
    parser.add_argument("--memory-mb", help="Override recommended memory request.")
    parser.add_argument("--cpus", type=int)
    parser.add_argument("--omp-threads", type=int)
    parser.add_argument("--mkl-threads", type=int)
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--dependency")
    parser.add_argument("--executable")
    parser.add_argument("--logs-dir")
    parser.add_argument("--sbatch-dir")
    parser.add_argument("--job-name")
    parser.add_argument("--run-id", help="Explicit run directory id under output/runs/. Defaults to submit timestamp plus SLURM job id.")
    parser.add_argument("--load-modules", action="store_true", help="Load the default Snellius runtime module stack inside the generated sbatch job.")
    parser.add_argument("--module-set", default="2025a", help="Module stack used with --load-modules. Supported: 2025a.")
    parser.add_argument("--use-scratch", action="store_true", help="Run on local scratch and copy back on exit.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--print-command", action="store_true")
    return parser


def main() -> int:
    parser = make_parser()
    args = parser.parse_args()
    command, metadata = build_submit_command(args)
    if args.print_command:
        print(" ".join(command))
    for key, value in metadata.items():
        print(f"{key}={value}")
    completed = subprocess.run(command, text=True, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())

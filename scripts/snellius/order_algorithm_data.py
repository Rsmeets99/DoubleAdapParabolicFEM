#!/usr/bin/env python3
"""Order Snellius algorithm outputs into a stable analysis tree.

The production runs intentionally used several output roots such as
``algorithm_data/snellius_mem56`` and ``algorithm_data/snellius_next_mem176``.
This utility scans all of those roots, chooses the newest run for each
scientific case, and copies it into ``algorithm_data_ordered``.

The ordered layout is:

    algorithm_data_ordered/<dim>/<example>/<p>/<target>/<mode>/
        run/
        slurm_logs/
        source_manifest.json

``mode`` is kept in the path so adaptive, G-estimator, and uniform runs do not
overwrite each other.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


RUN_STAMP_RE = re.compile(r"(?P<date>\d{8})_(?P<time>\d{6})(?:_(?P<micro>\d{6}))?")
P_RE = re.compile(r"^p\d+$")
DIM_RE = re.compile(r"^\d+d$")
SLURM_LOG_RE = re.compile(r"-(?P<job_id>\d+)\.(?:out|err|accounting\.txt)$")

MANAGED_DIRS = ("run", "slurm_logs")
MANAGED_FILES = ("source_manifest.json",)
LOG_INFO_KEYS = (
    "slurm_stdout",
    "slurm_stderr",
    "slurm_accounting_log",
    "slurm_accounting_snapshot",
)

MODE_LABELS = {
    "adaptive_force_effective_rho_no_inner_refinement": "force_eff_rho_no_inner",
    "adaptive_g_effective_rho": "g_eff_rho",
    "adaptive_standard": "adaptive",
    "uniform_standard": "uniform",
}


@dataclass(frozen=True)
class Candidate:
    source_group: str
    dim: str
    example: str
    p: str
    mode: str
    target: str
    run_id: str
    run_dir: Path
    job_id: str
    status: str
    sort_epoch: float
    sort_stamp: str
    job_info: dict[str, str]

    @property
    def key(self) -> tuple[str, str, str, str, str]:
        return (self.dim, self.example, self.p, self.target, self.mode)


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


def parse_run_stamp(run_id: str) -> tuple[float | None, str]:
    match = RUN_STAMP_RE.search(run_id)
    if not match:
        return None, ""
    micro = int(match.group("micro") or "0")
    try:
        stamp = datetime.strptime(
            f"{match.group('date')}_{match.group('time')}", "%Y%m%d_%H%M%S"
        ).replace(microsecond=micro)
    except ValueError:
        return None, ""
    return stamp.timestamp(), stamp.isoformat(timespec="microseconds")


def parse_iso_epoch(value: str) -> float | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def run_status(run_dir: Path, job_info: dict[str, str]) -> str:
    exit_code = job_info.get("run_exit_code", "").strip()
    if exit_code == "0":
        return "completed"
    if job_info.get("terminated_by") or job_info.get("terminated_at"):
        return "interrupted"
    if job_info.get("failed_at") or (exit_code and exit_code != "0"):
        return "failed"
    if (run_dir / "outer_history.csv").exists():
        return "completed"
    if any((run_dir / name).exists() for name in ("partial_outer_history.csv", "adaptive_summary.txt")):
        return "partial"
    return "unknown"


def parse_candidate_from_run_dir(
    algorithm_root: Path,
    run_dir: Path,
    job_info: dict[str, str] | None = None,
) -> Candidate | None:
    try:
        rel = run_dir.relative_to(algorithm_root)
    except ValueError:
        return None

    parts = rel.parts
    if len(parts) < 8 or parts[-2] != "runs":
        return None

    source_group, dim, example, p, mode, target = parts[:6]
    run_id = parts[-1]
    if not DIM_RE.match(dim) or not P_RE.match(p):
        return None

    info = job_info if job_info is not None else read_key_value_file(run_dir / "snellius_job_info.txt")
    job_id = info.get("job_id", "")
    if not job_id:
        job_match = re.search(r"job(\d+)", run_id)
        job_id = job_match.group(1) if job_match else ""

    stamp_epoch, stamp_text = parse_run_stamp(run_id)
    info_epochs = [
        parse_iso_epoch(info.get(key, ""))
        for key in ("finished_at", "failed_at", "terminated_at", "started_at")
    ]
    epoch = stamp_epoch or next((value for value in info_epochs if value is not None), None)
    if epoch is None:
        epoch = run_dir.stat().st_mtime
    if not stamp_text:
        stamp_text = datetime.fromtimestamp(epoch).isoformat(timespec="seconds")

    return Candidate(
        source_group=source_group,
        dim=dim,
        example=example,
        p=p,
        mode=mode,
        target=target,
        run_id=run_id,
        run_dir=run_dir,
        job_id=job_id,
        status=run_status(run_dir, info),
        sort_epoch=epoch,
        sort_stamp=stamp_text,
        job_info=info,
    )


def discover_candidates(algorithm_root: Path, include_legacy: bool) -> list[Candidate]:
    candidates: list[Candidate] = []
    seen_dirs: set[Path] = set()

    for info_file in algorithm_root.rglob("snellius_job_info.txt"):
        run_dir = info_file.parent
        candidate = parse_candidate_from_run_dir(
            algorithm_root, run_dir, read_key_value_file(info_file)
        )
        if candidate is None:
            continue
        candidates.append(candidate)
        seen_dirs.add(run_dir.resolve())

    if not include_legacy:
        return candidates

    for params_file in algorithm_root.rglob("run_parameters.yml"):
        run_dir = params_file.parent
        if run_dir.resolve() in seen_dirs:
            continue
        try:
            rel = run_dir.relative_to(algorithm_root)
        except ValueError:
            continue
        parts = rel.parts
        if "runs" in parts or len(parts) != 6:
            continue
        source_group, dim, example, p, mode, target = parts
        if not DIM_RE.match(dim) or not P_RE.match(p):
            continue
        epoch = run_dir.stat().st_mtime
        candidates.append(
            Candidate(
                source_group=source_group,
                dim=dim,
                example=example,
                p=p,
                mode=mode,
                target=target,
                run_id="legacy_flat",
                run_dir=run_dir,
                job_id="",
                status=run_status(run_dir, {}),
                sort_epoch=epoch,
                sort_stamp=datetime.fromtimestamp(epoch).isoformat(timespec="seconds"),
                job_info={},
            )
        )

    return candidates


def status_score(status: str) -> int:
    if status == "completed":
        return 4
    if status in {"history_present", "interrupted"}:
        return 3
    if status == "partial":
        return 2
    if status == "failed":
        return 1
    return 0


def select_latest(
    candidates: Iterable[Candidate],
    prefer_complete: bool,
) -> dict[tuple[str, str, str, str, str], Candidate]:
    selected: dict[tuple[str, str, str, str, str], Candidate] = {}
    for candidate in candidates:
        old = selected.get(candidate.key)
        if old is None:
            selected[candidate.key] = candidate
            continue
        if prefer_complete:
            new_key = (status_score(candidate.status), candidate.sort_epoch)
            old_key = (status_score(old.status), old.sort_epoch)
        else:
            new_key = (candidate.sort_epoch, status_score(candidate.status))
            old_key = (old.sort_epoch, status_score(old.status))
        if new_key > old_key:
            selected[candidate.key] = candidate
    return selected


def build_log_index(logs_root: Path) -> dict[str, list[Path]]:
    by_job: dict[str, list[Path]] = {}
    if not logs_root.exists():
        return by_job
    for path in logs_root.rglob("*"):
        if not path.is_file():
            continue
        match = SLURM_LOG_RE.search(path.name)
        if not match:
            continue
        by_job.setdefault(match.group("job_id"), []).append(path)
    for paths in by_job.values():
        paths.sort()
    return by_job


def resolve_existing_path(raw_value: str, repo_root: Path) -> Path | None:
    if not raw_value:
        return None
    path = Path(raw_value)
    candidates = [path] if path.is_absolute() else [repo_root / path, path]

    if path.is_absolute() and repo_root.name in path.parts:
        index = path.parts.index(repo_root.name)
        candidates.append(repo_root.joinpath(*path.parts[index + 1 :]))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def log_files_for_candidate(
    candidate: Candidate,
    repo_root: Path,
    log_index: dict[str, list[Path]],
) -> list[Path]:
    paths: list[Path] = []
    seen: set[Path] = set()

    for key in LOG_INFO_KEYS:
        resolved = resolve_existing_path(candidate.job_info.get(key, ""), repo_root)
        if resolved is None or not resolved.is_file():
            continue
        real = resolved.resolve()
        if real not in seen:
            paths.append(resolved)
            seen.add(real)

    for path in log_index.get(candidate.job_id, []):
        real = path.resolve()
        if real not in seen:
            paths.append(path)
            seen.add(real)

    return paths


def case_dir(output_root: Path, candidate: Candidate) -> Path:
    return output_root / candidate.dim / candidate.example / candidate.p / candidate.target / candidate.mode


def read_existing_manifest(path: Path) -> dict[str, object]:
    manifest_path = path / "source_manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def existing_epoch(manifest: dict[str, object]) -> float | None:
    source = manifest.get("source")
    if not isinstance(source, dict):
        return None
    value = source.get("sort_epoch")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def should_update(destination: Path, candidate: Candidate, force: bool) -> tuple[bool, str]:
    if force:
        return True, "force"
    if not destination.exists():
        return True, "new"
    previous_epoch = existing_epoch(read_existing_manifest(destination))
    if previous_epoch is None:
        return True, "existing_without_manifest"
    if candidate.sort_epoch > previous_epoch:
        return True, "newer"
    return False, "up_to_date"


def remove_managed_outputs(destination: Path) -> None:
    for name in MANAGED_DIRS:
        path = destination / name
        if path.exists():
            shutil.rmtree(path)
    for name in MANAGED_FILES:
        path = destination / name
        if path.exists():
            path.unlink()


def copy_case(
    candidate: Candidate,
    destination: Path,
    repo_root: Path,
    log_index: dict[str, list[Path]],
    dry_run: bool,
) -> list[Path]:
    logs = log_files_for_candidate(candidate, repo_root, log_index)
    if dry_run:
        return logs

    destination.mkdir(parents=True, exist_ok=True)
    remove_managed_outputs(destination)

    shutil.copytree(candidate.run_dir, destination / "run", symlinks=True)

    logs_dir = destination / "slurm_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for log_path in logs:
        target = logs_dir / log_path.name
        if target.exists():
            target.unlink()
        shutil.copy2(log_path, target)

    manifest = {
        "ordered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "key": {
            "dimension": candidate.dim,
            "example": candidate.example,
            "p": candidate.p,
            "target": candidate.target,
            "mode": candidate.mode,
        },
        "source": {
            "algorithm_group": candidate.source_group,
            "path": str(candidate.run_dir),
            "run_id": candidate.run_id,
            "job_id": candidate.job_id,
            "status": candidate.status,
            "sort_epoch": candidate.sort_epoch,
            "sort_stamp": candidate.sort_stamp,
        },
        "job_info": candidate.job_info,
        "copied_logs": [str(path) for path in logs],
    }
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return logs


def write_index(output_root: Path, rows: list[dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    output_root.mkdir(parents=True, exist_ok=True)
    fields = (
        "dimension",
        "example",
        "p",
        "target",
        "mode",
        "status",
        "job_id",
        "run_id",
        "sort_stamp",
        "source_group",
        "source_path",
        "ordered_path",
        "log_count",
        "action",
        "reason",
    )
    with (output_root / "index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def target_dofs(target: object) -> float:
    text = str(target)
    matches = re.findall(r"(\d+(?:\.\d+)?)([kKmM])", text)
    if not matches:
        return -1.0
    value, suffix = matches[-1]
    multiplier = 1_000_000.0 if suffix.lower() == "m" else 1_000.0
    return float(value) * multiplier


def display_dofs(value: float) -> str:
    if value < 0:
        return ""
    if value >= 1_000_000:
        return f"{value / 1_000_000:g}M"
    if value >= 1_000:
        return f"{value / 1_000:g}k"
    return f"{value:g}"


def target_sort_tuple(target: object) -> tuple[float, str]:
    return (target_dofs(target), str(target))


def p_sort_key(p: object) -> tuple[int, str]:
    text = str(p)
    if text.startswith("p") and text[1:].isdigit():
        return (int(text[1:]), text)
    return (9999, text)


def dim_sort_key(dim: object) -> tuple[int, str]:
    text = str(dim)
    if text.endswith("d") and text[:-1].isdigit():
        return (int(text[:-1]), text)
    return (9999, text)


def short_mode(mode: object) -> str:
    return MODE_LABELS.get(str(mode), str(mode))


def markdown_cell(value: object) -> str:
    text = str(value)
    return text.replace("|", "\\|").replace("\n", "<br>")


def markdown_table(headers: tuple[str, ...], body_rows: list[tuple[object, ...]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in body_rows:
        lines.append("| " + " | ".join(markdown_cell(value) for value in row) + " |")
    return "\n".join(lines)


def all_data_table_rows(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            str(row["dimension"]),
            str(row["example"]),
            str(row["p"]),
            str(row["target"]),
        )
        grouped.setdefault(key, []).append(row)

    table_rows: list[tuple[object, ...]] = []
    for dim, example, p, target in sorted(
        grouped,
        key=lambda key: (
            dim_sort_key(key[0]),
            key[1],
            p_sort_key(key[2]),
            target_sort_tuple(key[3]),
        ),
    ):
        entries = sorted(grouped[(dim, example, p, target)], key=lambda item: str(item["mode"]))
        modes = ", ".join(
            f"{short_mode(item['mode'])}={item['status']}:{item['job_id'] or '-'}"
            for item in entries
        )
        table_rows.append((dim, example, p, target, display_dofs(target_dofs(target)), modes))
    return table_rows


def highest_data_table_rows(rows: list[dict[str, object]]) -> list[tuple[object, ...]]:
    grouped: dict[tuple[str, str, str, str], dict[str, object]] = {}
    for row in rows:
        key = (
            str(row["dimension"]),
            str(row["example"]),
            str(row["p"]),
            str(row["mode"]),
        )
        old = grouped.get(key)
        if old is None or target_sort_tuple(row["target"]) > target_sort_tuple(old["target"]):
            grouped[key] = row

    table_rows: list[tuple[object, ...]] = []
    for key in sorted(
        grouped,
        key=lambda item: (dim_sort_key(item[0]), item[1], p_sort_key(item[2]), short_mode(item[3])),
    ):
        row = grouped[key]
        table_rows.append(
            (
                row["dimension"],
                row["example"],
                row["p"],
                short_mode(row["mode"]),
                row["target"],
                display_dofs(target_dofs(row["target"])),
                row["status"],
                row["job_id"] or "-",
                row["source_group"],
            )
        )
    return table_rows


def write_readme(output_root: Path, rows: list[dict[str, object]], dry_run: bool) -> None:
    if dry_run:
        return
    completed = sum(1 for row in rows if row["status"] == "completed")
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        status_counts[status] = status_counts.get(status, 0) + 1

    all_rows = all_data_table_rows(rows)
    highest_rows = highest_data_table_rows(rows)

    status_text = ", ".join(
        f"{status}={count}" for status, count in sorted(status_counts.items())
    )

    text = f"""# Ordered Snellius Algorithm Data

This directory is managed by `scripts/snellius/order_algorithm_data.py`.

Layout:

`<dim>/<example>/<p>/<target>/<mode>/run/`

The sibling `slurm_logs/` directory contains the matching `.out`, `.err`, and
accounting files when they were discoverable from `snellius_job_info.txt` or
`snellius_logs/`.

`source_manifest.json` records the raw source directory and job metadata used
for the selected copy. Re-running the organizer updates only cases for which a
newer matching raw run exists, unless `--force` is used.

Last inventory update: `{datetime.now(timezone.utc).isoformat(timespec="seconds")}`

Selected run count: `{len(rows)}` total, `{completed}` completed. Status counts:
`{status_text}`.

## All Available Ordered Data

Each mode entry is written as `mode=status:job_id`.

{markdown_table(("dim", "example", "p", "X target", "X dofs", "modes"), all_rows)}

## Highest Available X-Delta Runs

This table keeps one row per `dim/example/p/mode`, choosing the largest available
X-dof target.

{markdown_table(("dim", "example", "p", "mode", "highest target", "X dofs", "status", "job", "source group"), highest_rows)}
"""
    (output_root / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy the newest Snellius algorithm_data run for each case into algorithm_data_ordered.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root. Defaults to the parent of scripts/snellius.",
    )
    parser.add_argument(
        "--algorithm-root",
        type=Path,
        default=None,
        help="Raw algorithm_data root. Defaults to <repo-root>/algorithm_data.",
    )
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=None,
        help="Raw snellius_logs root. Defaults to <repo-root>/snellius_logs.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Ordered output root. Defaults to <repo-root>/algorithm_data_ordered.",
    )
    parser.add_argument(
        "--include-legacy-flat",
        action="store_true",
        help="Also include pre-runs-layout flat output directories.",
    )
    parser.add_argument(
        "--prefer-complete",
        action="store_true",
        help="Prefer an older completed run over a newer failed/partial duplicate.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace managed ordered files even when the selected source is not newer.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned updates without copying files.",
    )
    parser.add_argument(
        "--no-readme",
        action="store_true",
        help="Do not update algorithm_data_ordered/README.md.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of selected cases processed. Useful with --dry-run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    algorithm_root = (args.algorithm_root or repo_root / "algorithm_data").resolve()
    logs_root = (args.logs_root or repo_root / "snellius_logs").resolve()
    output_root = (args.output_root or repo_root / "algorithm_data_ordered").resolve()

    if not algorithm_root.exists():
        print(f"error: algorithm root does not exist: {algorithm_root}", file=sys.stderr)
        return 2

    candidates = discover_candidates(algorithm_root, include_legacy=args.include_legacy_flat)
    selected = select_latest(candidates, prefer_complete=args.prefer_complete)
    ordered = sorted(selected.values(), key=lambda item: item.key)
    if args.limit > 0:
        ordered = ordered[: args.limit]

    log_index = build_log_index(logs_root)

    rows: list[dict[str, object]] = []
    action_counts: dict[str, int] = {}

    for candidate in ordered:
        destination = case_dir(output_root, candidate)
        update, reason = should_update(destination, candidate, force=args.force)
        action = "copy" if update else "skip"
        if update:
            logs = copy_case(candidate, destination, repo_root, log_index, dry_run=args.dry_run)
        else:
            logs = log_files_for_candidate(candidate, repo_root, log_index)

        action_counts[action] = action_counts.get(action, 0) + 1
        rows.append(
            {
                "dimension": candidate.dim,
                "example": candidate.example,
                "p": candidate.p,
                "target": candidate.target,
                "mode": candidate.mode,
                "status": candidate.status,
                "job_id": candidate.job_id,
                "run_id": candidate.run_id,
                "sort_stamp": candidate.sort_stamp,
                "source_group": candidate.source_group,
                "source_path": str(candidate.run_dir),
                "ordered_path": str(destination),
                "log_count": len(logs),
                "action": action,
                "reason": reason,
            }
        )

        prefix = "DRY-RUN " if args.dry_run else ""
        print(
            f"{prefix}{action:4s} {reason:25s} "
            f"{candidate.dim}/{candidate.example}/{candidate.p}/{candidate.target}/{candidate.mode} "
            f"job={candidate.job_id or '-'} logs={len(logs)}"
        )

    write_index(output_root, rows, dry_run=args.dry_run)
    if not args.no_readme:
        write_readme(output_root, rows, dry_run=args.dry_run)

    print(
        "summary: "
        f"discovered={len(candidates)} selected={len(selected)} processed={len(ordered)} "
        + " ".join(f"{key}={value}" for key, value in sorted(action_counts.items()))
    )
    if args.dry_run:
        print(f"dry-run: no files were copied to {output_root}")
    else:
        print(f"ordered data root: {output_root}")
        print(f"index: {output_root / 'index.csv'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

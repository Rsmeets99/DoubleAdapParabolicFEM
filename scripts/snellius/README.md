# Snellius Scripts

This directory contains the scripts used to build the production executable,
generate the Snellius configuration matrix, submit jobs, collect raw outputs,
and prepare plotting CSVs.

For the end-to-end workflow, see the top-level `README_SNELLIUS.md`. This file
is a command reference for the scripts in this directory.

## Build

```bash
scripts/snellius/build_snellius.sh --load-modules --jobs 4
```

Important flags:

- `--preset NAME`: CMake configure preset. Default:
  `release-snellius-generic-mkl-pardiso`.
- `--build-preset NAME`: CMake build preset. Default:
  `build-release-snellius-generic-mkl-pardiso`.
- `--target NAME`: CMake target. Default: `run_adaptive_algorithm`.
- `--jobs N`: parallel build jobs. Default: `SLURM_CPUS_PER_TASK` or `nproc`.
- `--load-modules`: load the documented Snellius module stack.
- `--module-set NAME`: module stack used with `--load-modules`. Default:
  `2025a`.

## Generate Production Configs

```bash
python3 scripts/snellius/generate_production_configs.py
```

Outputs are written under `production_configs/snellius/` by default.

Important flags:

- `--output-root PATH`: directory for YAML configs, manifest, and index files.
- `--algorithm-output-root PATH`: output root written into generated configs.
- `--stage-dof-caps LIST`: comma-separated X-space true-DoF caps. Supports
  `k` and `m` suffixes.
- `--rho VALUE`, `--theta-x VALUE`, `--theta-y VALUE`: estimator and marking
  parameters.
- `--max-outer N`, `--max-inner N`: adaptive iteration limits.
- `--max-wall-time-seconds VALUE`: wall-clock stop condition written into each
  config. `0` disables the cap.
- `--zero-tol VALUE`, `--divergence-tol VALUE`: numerical tolerances.
- `--memory-limit-mb MB`, `--memory-reserve-mb MB`: process memory controls
  written into each config.
- `--main-solver-memory-limit-mb MB`: direct main-solver memory cap.
- `--pardiso-memory-mode in_core|auto|out_of_core`: Pardiso memory mode.
- `--main-solver-max-iterations N`, `--main-solver-tolerance VALUE`,
  `--main-solver-symmetry-tolerance VALUE`: main-solver controls.
- `--main-solver-ooc-auto-switch`: enable automatic out-of-core switching.
- `--main-solver-ooc-switch-threshold VALUE`: memory threshold for automatic
  out-of-core switching.
- `--no-g-estimator-in-force-variant`: disable the G estimator in the
  force-effective-rho variant.
- `--compute-g-estimator-in-uniform`: enable the G estimator for uniform runs.
- `--solver-diagnostics`: enable solver diagnostic metadata in generated
  configs.
- `--solver-diagnostics-max-export-dofs N`: DoF cap for diagnostic exports.
- `--main-assembly-max-threads N`,
  `--main-two-pass-numeric-fill-max-threads N`,
  `--slab-reconstruction-max-threads N`,
  `--local-error-max-threads N`: algorithm thread caps.
- `--snellius-cpus N`, `--snellius-omp-threads N`,
  `--snellius-mkl-threads N`: scheduler/runtime thread recommendations.
- `--dry-run`: print the planned generation without writing files.

## Submit One Config

The usual entry point is `submit_stage.py`, which reads recommendations from
the generated `index.csv` and delegates to `submit_single_run.sh`.

```bash
python3 scripts/snellius/submit_stage.py \
  --config production_configs/snellius/1d/smooth_initial/p1/adaptive_standard/pilot_250k.yml \
  --load-modules \
  --dry-run
```

Important `submit_stage.py` flags:

- `--config PATH`: generated YAML config to submit. Required.
- `--project-root PATH`: repository root.
- `--index PATH`: generated Snellius index CSV.
- `--config-root PATH`: generated config root.
- `--submitter PATH`: lower-level shell submitter.
- `--partition NAME`, `--time LIMIT`, `--memory-mb MB`, `--cpus N`:
  scheduler overrides.
- `--omp-threads N`, `--mkl-threads N`: runtime thread overrides.
- `--account NAME`, `--qos NAME`: optional scheduler account and QoS.
- `--dependency SPEC`: optional Slurm dependency such as `afterok:12345`.
- `--executable PATH`: executable to run.
- `--logs-dir PATH`: Slurm log base directory.
- `--sbatch-dir PATH`: generated sbatch base directory.
- `--job-name NAME`: Slurm job name.
- `--run-id ID`: explicit run directory id under `output/runs/`.
- `--load-modules`, `--module-set NAME`: load the runtime module stack.
- `--use-scratch`: run on node scratch and copy back on exit.
- `--dry-run`: write the sbatch file but do not call `sbatch`.
- `--print-command`: print the delegated shell command.

`submit_single_run.sh` accepts the same operational settings and writes the
actual sbatch script. It also records job metadata and accounting information
when available.

## Submit A Matrix

```bash
python3 scripts/snellius/submit_matrix.py \
  --dimension 1 \
  --example smooth_initial \
  --p 1 \
  --mode adaptive_standard \
  --target pilot_250k \
  --load-modules \
  --dry-run
```

Important matrix-selection flags:

- `--manifest PATH`: generated manifest CSV.
- `--index PATH`: generated index CSV.
- `--config-root PATH`: generated config root.
- `--stage-submitter PATH`: stage-submission script.
- `--dimension LIST`: comma-separated dimensions, for example `1,2`.
- `--example LIST`: comma-separated example names.
- `--p LIST`: comma-separated polynomial degrees.
- `--mode LIST`: comma-separated run modes.
- `--target LIST`: comma-separated target labels.
- `--limit N`: submit at most `N` selected configs.
- `--chain-stages`: chain increasing targets per dimension/example/p/mode with
  `afterok` dependencies.
- `--print-commands`: print delegated stage commands.

Scheduler, runtime, scratch, module, and dry-run flags mirror
`submit_stage.py`.

## Collect And Summarize Results

Collect manifest rows into a compact results CSV:

```bash
python3 scripts/snellius/collect_results.py \
  --manifest production_configs/snellius/manifest.csv \
  --run-selection latest \
  --out production_configs/snellius/collected_results.csv
```

Important flags:

- `--manifest PATH`: generated manifest CSV.
- `--project-root PATH`: repository root.
- `--out PATH`: collected results CSV.
- `--archive-dir PATH`: directory for archived output snippets.
- `--no-archive`: skip archived output snippets.
- `--run-selection latest|all|legacy`: choose which run directories to collect.

Summarize convergence rates:

```bash
python3 scripts/snellius/summarize_rates.py \
  --results-csv production_configs/snellius/collected_results.csv \
  --out production_configs/snellius/rate_summary.csv \
  --sequence-out production_configs/snellius/rate_sequences.csv \
  --markdown-out production_configs/snellius/rate_summary.md
```

Important flags:

- `--results-csv PATH`: collected results CSV.
- `--out PATH`: summary CSV.
- `--sequence-out PATH`: per-sequence CSV.
- `--markdown-out PATH`: Markdown summary.
- `--dof-column NAME`: DoF column used as the independent variable.
- `--group-columns LIST`: comma-separated grouping columns.
- `--include-partial`: include partial-history rows when usable.

## Prepare Plot Data

Order raw run directories into a stable analysis tree:

```bash
python3 scripts/snellius/order_algorithm_data.py
```

Important flags:

- `--repo-root PATH`: repository root.
- `--algorithm-root PATH`: raw algorithm output root.
- `--logs-root PATH`: raw Slurm logs root.
- `--output-root PATH`: ordered output root.
- `--include-legacy-flat`: include older flat output layouts.
- `--prefer-complete`: prefer an older completed duplicate over a newer failed
  or partial duplicate.
- `--force`: replace managed ordered files even when the selected source is not
  newer.
- `--dry-run`: print planned updates without copying files.
- `--no-readme`: do not update `algorithm_data_ordered/README.md`.
- `--limit N`: process at most `N` selected cases.

Create cleaned plot CSVs from the ordered tree:

```bash
python3 scripts/snellius/prepare_plot_data.py \
  --ordered-root algorithm_data_ordered \
  --output-root algorithm_data_plot_cleaned
```

Important flags:

- `--repo-root PATH`: repository root.
- `--ordered-root PATH`: input ordered data root.
- `--output-root PATH`: cleaned plot-data output root.
- `--dry-run`: print planned files without writing output.

## Validation

Check shell syntax:

```bash
bash -n scripts/snellius/build_snellius.sh scripts/snellius/submit_single_run.sh
```

Check Python syntax:

```bash
python3 -m py_compile \
  scripts/snellius/generate_production_configs.py \
  scripts/snellius/submit_stage.py \
  scripts/snellius/submit_matrix.py \
  scripts/snellius/collect_results.py \
  scripts/snellius/summarize_rates.py \
  scripts/snellius/order_algorithm_data.py \
  scripts/snellius/prepare_plot_data.py
```

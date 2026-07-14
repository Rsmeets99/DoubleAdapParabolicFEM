# Snellius Usage

This guide documents how to build, generate configs, submit jobs, and collect
results on Snellius for the publication snapshot.

For a per-script command reference, see `scripts/snellius/README.md`.

## Build

Recommended Snellius build:

```bash
scripts/snellius/build_snellius.sh --load-modules --jobs 4
```

The helper loads the `2025a` module set and builds
`out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm`.

Equivalent direct CMake commands:

```bash
cmake --preset release-snellius-generic-mkl-pardiso
cmake --build --preset build-release-snellius-generic-mkl-pardiso --target run_adaptive_algorithm
```

Architecture-specific presets are also available when the job is pinned to a
matching CPU family:

```bash
cmake --preset release-snellius-rome-mkl-pardiso
cmake --preset release-snellius-genoa-mkl-pardiso
```

## Generate Configs

Production configs are generated from the checked-in script:

```bash
python3 scripts/snellius/generate_production_configs.py
```

Generated files are written under `production_configs/snellius/`, including:

- `manifest.csv`
- `index.csv`
- `index.md`
- YAML configs under `1d/` and `2d/`

The generated YAML owns algorithm thread caps such as
`main_assembly_max_threads`, `main_two_pass_numeric_fill_max_threads`,
`slab_reconstruction_max_threads`, and `local_error_max_threads`.

## Thread Policy

The generated index encodes the default Slurm and algorithm thread policy:

| Case family | Slurm CPUs | OMP/MKL threads | Main assembly | Numeric fill | Slab reconstruction | Local error |
|---|---:|---:|---:|---:|---:|---:|
| 1D and 2D p=1 | 4 | 4 | 4 | 4 | 4 | 4 |
| 2D p=2 | 8 | 8 | 4 | 4 | 4 | 8 |
| 2D p>=3 | 16 | 8 | 4 | 4 | 16 | 8 |

`submit_stage.py` reads `recommended_cpus`, `recommended_omp_threads`, and
`recommended_mkl_threads` from `index.csv` unless explicitly overridden.

## Submit A Single Config

Dry-run first:

```bash
python3 scripts/snellius/submit_stage.py \
  --config production_configs/snellius/1d/smooth_initial/p1/adaptive_standard/pilot_250k.yml \
  --partition rome \
  --time 01:00:00 \
  --memory-mb 28000 \
  --cpus 16 \
  --load-modules \
  --dry-run
```

Remove `--dry-run` to submit the generated sbatch file with `sbatch`.

The lower-level shell submitter accepts the same operational settings:

```bash
scripts/snellius/submit_single_run.sh \
  --config production_configs/snellius/1d/smooth_initial/p1/adaptive_standard/pilot_250k.yml \
  --partition rome \
  --time 01:00:00 \
  --memory-mb 28000 \
  --cpus 16 \
  --load-modules \
  --dry-run
```

## Submit A Matrix

Dry-run a filtered subset:

```bash
python3 scripts/snellius/submit_matrix.py \
  --dimension 1 \
  --example smooth_initial \
  --p 1 \
  --mode adaptive_standard,uniform_standard,adaptive_g_effective_rho,adaptive_force_effective_rho_no_inner_refinement \
  --target pilot_250k \
  --partition rome \
  --time 01:00:00 \
  --memory-mb 28000 \
  --cpus 16 \
  --load-modules \
  --dry-run
```

Use `--chain-stages` to submit increasing target caps with
`afterok:<previous_job_id>` dependencies per dimension/example/order/mode
profile.

## Slurm Job Behavior

Generated jobs request one node and one task:

```text
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=<cpus>
```

The executable is launched as:

```bash
srun --ntasks=1 --cpus-per-task="${THREADS}" \
  "${EXECUTABLE}" \
  --config "${CONFIG}" \
  --output "${RUN_OUTPUT}"
```

Jobs request a wall-time warning:

```text
#SBATCH --signal=B:TERM@120
```

The generated script traps `TERM` and `INT`, forwards termination to `srun`,
copies partial output back when scratch is used, and records run metadata in
`snellius_job_info.txt`.

## Output Layout

By default, each submission writes algorithm output to a unique run directory:

```text
algorithm_data/.../<target>/runs/<submit_timestamp>_job<slurm_job_id>/
```

Slurm logs and generated sbatch files are also grouped by submission:

```text
snellius_logs/runs/<submit_timestamp>_<case_id>/%x-%j.out
snellius_logs/runs/<submit_timestamp>_<case_id>/%x-%j.err
.snellius_jobs/runs/<submit_timestamp>_<case_id>/<job_name>.sbatch
```

Use `--run-id <id>` for a reproducible run directory. Jobs write directly to
the final output directory by default; `--use-scratch` opts into node scratch
plus copy-back.

## Collect And Summarize

Collect the latest run per generated config:

```bash
python3 scripts/snellius/collect_results.py \
  --manifest production_configs/snellius/manifest.csv \
  --run-selection latest \
  --out production_configs/snellius/collected_results.csv
```

Build rate summaries:

```bash
python3 scripts/snellius/summarize_rates.py \
  --results-csv production_configs/snellius/collected_results.csv \
  --out production_configs/snellius/rate_summary.csv \
  --sequence-out production_configs/snellius/rate_sequences.csv \
  --markdown-out production_configs/snellius/rate_summary.md
```

## Validation

Shell syntax:

```bash
bash -n scripts/snellius/build_snellius.sh scripts/snellius/submit_single_run.sh
```

Python syntax:

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

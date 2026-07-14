# Algorithm Output Data

This directory is the default location for generated algorithm output.

The publication snapshot does not include raw production outputs. The checked-in
paper data lives in `algorithm_data_plot/`, which contains cleaned CSV files
prepared from completed production runs.

## Local Runs

Example YAML files under `algorithm/examples/` write to subdirectories of this
directory. For example:

```bash
./path/to/run_adaptive_algorithm \
  --config algorithm/examples/space_time_2d/smooth_initial.yml
```

writes to:

```text
algorithm_data/space_time_2d/smooth_initial_config_run
```

Use an explicit output directory to keep local runs separate:

```bash
./path/to/run_adaptive_algorithm \
  --example smooth_initial \
  --dimension 1 \
  --p 1 \
  --output algorithm_data/local/smooth_initial_p1
```

For smoke checks that should not write run artifacts, use:

```bash
./path/to/run_adaptive_algorithm \
  --example smooth_initial \
  --dimension 1 \
  --p 1 \
  --max-outer 0 \
  --quiet \
  --output-profile minimal \
  --no-export
```

## Typical Files

Depending on the output flags, a run directory may contain:

- `outer_history.csv`: one row per adaptive outer iteration.
- `inner_history.csv`: one row per inner iteration.
- `timing_history.csv`: timing breakdowns when timing output is enabled.
- estimator-component CSV files when component export is enabled.
- refinement-history and mesh-statistics files when requested.
- mesh and DoF snapshots when snapshot export is enabled.

## Repository Hygiene

Generated raw outputs can be large and may include machine-specific run
metadata. Keep this directory out of publication archives unless a specific raw
run is intentionally being shared.

The cleaned, paper-facing CSV layout is documented in
`algorithm_data_plot/README.md`.

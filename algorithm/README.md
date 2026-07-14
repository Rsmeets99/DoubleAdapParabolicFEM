# Algorithm Runner

This directory contains the paper-level adaptive algorithm layer built on top
of the numerical code in `src/`.

## Layout

- `adaptive_driver.hpp`: adaptive solve loop and high-level orchestration.
- `adaptive_parameters.hpp`: parameters shared by command-line and config-file
  execution.
- `adaptive_result.hpp`: result records used by output writers.
- `example_registry.hpp`: registry of named example problems.
- `examples/`: checked-in YAML configurations and example definitions.
- `output/`: history, snapshot, logging, and output-path helpers.
- `runners/`: `run_adaptive_algorithm` entry point and degree/dimension runner
  translation units.
- `support/`: dimension-specific problem construction helpers.

## Executable

The CMake target is `run_adaptive_algorithm`. The executable supports direct
command-line options and YAML config files. To inspect the complete CLI for the
current build, run:

```bash
./path/to/run_adaptive_algorithm --help
```

List registered examples:

```bash
./path/to/run_adaptive_algorithm --list-examples
```

Run a checked-in 1+1D config:

```bash
./path/to/run_adaptive_algorithm \
  --config algorithm/examples/space_time_1d/smooth_initial.yml
```

Run a checked-in 2+1D config:

```bash
./path/to/run_adaptive_algorithm \
  --config algorithm/examples/space_time_2d/smooth_initial.yml
```

For a quick smoke run without exporting output data:

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

## Example Configurations

The checked-in YAML files under `examples/space_time_1d/` and
`examples/space_time_2d/` are complete runner configurations for the named
examples:

- `smooth_initial`
- `boundary_singularity`
- `non_matching_initial`

Important config fields include:

- `example`: registered example name.
- `dimension`: `1` for 1+1D or `2` for 2+1D.
- `p`: polynomial degree used by the selected runner.
- `rho`, `theta_x`, `theta_y`: estimator and marking parameters.
- `max_outer`, `max_inner`: adaptive outer-loop and inner-loop limits.
- `max_y_true_dofs`: cap for the auxiliary Y-space size.
- `main_solver`: main linear solver choice.
- `output`: output directory for generated run artifacts.
- `quiet`: suppress console progress output.
- `export_history`, `save_estimator_components`, `save_refinement_history`,
  `save_mesh_statistics`, `save_snapshots`, `save_snapshot_dofs`: output
  controls.
- `enable_timing_breakdown`, `timing_history_filename`,
  `timing_detail_level`: timing output controls.

Command-line options override values read from `--config`.

## Output

The runner writes output to the directory selected by the config `output` key
or by the `--output` command-line option. By convention, local generated output
goes under `algorithm_data/`; see `algorithm_data/README.md`.

Use `--output-profile minimal --no-export` for smoke checks that should not
leave generated algorithm data behind.

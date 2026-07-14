# Algorithm Runner

This directory contains the paper-level adaptive algorithm layer built on top
of the numerical code in `src/`. It defines the registered model problems,
drives the outer X-space and inner Y-space adaptive loops, writes run history,
and exposes the `run_adaptive_algorithm` executable.

## Layout

| Path | Purpose |
|---|---|
| `adaptive_driver.hpp` | Adaptive solve loop and high-level orchestration. |
| `adaptive_parameters.hpp` | Runner parameters shared by config-file and command-line execution. |
| `adaptive_result.hpp` | Result records used by output writers. |
| `example_registry.hpp` | Registry of named example problems and their supported dimensions. |
| `examples/space_time_1d/` | Checked-in 1+1D example definitions and YAML configs. |
| `examples/space_time_2d/` | Checked-in 2+1D example definitions and YAML configs. |
| `output/` | Output-path, history, snapshot, and logging helpers. |
| `runners/` | Executable entry point, option parsing, and degree/dimension dispatch. |
| `support/` | Dimension-specific problem construction helpers. |

## Executable

The CMake target is `run_adaptive_algorithm`. The executable supports direct
command-line options and flat YAML-style config files. Command-line options
override values loaded from `--config`.

Build locally without MKL, then run:

```bash
out/build/local-release/run_adaptive_algorithm --help
out/build/local-release/run_adaptive_algorithm --list-examples
```

A config file is usually the most reproducible way to run the code:

```bash
out/build/local-release/run_adaptive_algorithm \
  --config algorithm/examples/space_time_1d/smooth_initial.yml
```

Equivalent direct command-line run:

```bash
out/build/local-release/run_adaptive_algorithm \
  --example smooth_initial \
  --dimension 1 \
  --p 1 \
  --rho 1.0 \
  --theta-x 0.4 \
  --theta-y 0.4 \
  --max-outer 20 \
  --max-inner 20 \
  --max-y-dofs 500000 \
  --output algorithm_data/local/smooth_initial_1d_p1
```

For a quick smoke run without exported output:

```bash
out/build/local-release/run_adaptive_algorithm \
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

| Example | 1+1D config | 2+1D config |
|---|---|---|
| `smooth_initial` | `examples/space_time_1d/smooth_initial.yml` | `examples/space_time_2d/smooth_initial.yml` |
| `boundary_singularity` | `examples/space_time_1d/boundary_singularity.yml` | `examples/space_time_2d/boundary_singularity.yml` |
| `non_matching_initial` | `examples/space_time_1d/non_matching_initial.yml` | `examples/space_time_2d/non_matching_initial.yml` |

Important config fields include:

| Key | Meaning |
|---|---|
| `example` | Registered example name. |
| `dimension` | `1` for 1+1D or `2` for 2+1D. |
| `p` | Polynomial degree, dispatched to the matching compiled runner. |
| `rho` | Inner-loop estimator acceptance parameter. |
| `theta_x`, `theta_y` | Doerfler bulk parameters for X and Y marking. |
| `uniform_refinement_mode` | `adaptive`, `uniform_x`, `uniform_y`, or `uniform_xy`. |
| `max_outer`, `max_inner` | Adaptive outer-loop and inner-loop iteration caps. |
| `max_x_true_dofs`, `max_y_true_dofs` | Hard true-DoF caps for stopping before the next solve. |
| `max_dofs_target` | Which DoF cap is considered the primary requested target. |
| `main_solver` | Main linear solver choice. |
| `output` | Output directory for generated run artifacts. |
| `output_profile` | Grouped output policy: `minimal`, `production`, `benchmark`, or `debug`. |
| `quiet` | Suppress console iteration tables. |
| `export_history`, `save_estimator_components`, `save_refinement_history` | CSV/text output controls. |
| `save_mesh_statistics`, `save_snapshots`, `save_snapshot_dofs` | Mesh and snapshot output controls. |
| `enable_timing_breakdown`, `timing_history_filename`, `timing_detail_level` | Timing output controls. |

Config files are flat key/value files, not full nested YAML documents. The
parser accepts both the primary keys above and several compatibility aliases;
the executable prints the accepted key list when config parsing fails.

## Production-Style Examples

The Snellius scripts generate the production matrix used for large runs. By
default:

```bash
python3 scripts/snellius/generate_production_configs.py
```

creates configs below `production_configs/snellius/` for:

| Axis | Values |
|---|---|
| dimensions | `1`, `2` |
| examples | `smooth_initial`, `non_matching_initial`, `boundary_singularity` |
| degrees | `1`, `2`, `3`, `4` |
| variants | `adaptive_standard`, `adaptive_g_effective_rho`, `adaptive_force_effective_rho_no_inner_refinement`, `uniform_standard` |
| default X-DoF targets | `pilot_250k`, `pilot_500k`, `pilot_1M`, `production_4M`, `production_6M` |

That default command generates 480 configs. Generated production configs set
`max_dofs_target: x` and `max_y_true_dofs` to ten times the X-space target.

Run one generated adaptive G-estimator case:

```bash
out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm \
  --config production_configs/snellius/1d/smooth_initial/p1/adaptive_g_effective_rho/production_4M.yml
```

Run a force-effective-rho/no-inner-refinement case:

```bash
out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm \
  --config production_configs/snellius/2d/boundary_singularity/p2/adaptive_force_effective_rho_no_inner_refinement/production_4M.yml
```

Run a uniform X/Y refinement comparison:

```bash
out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm \
  --config production_configs/snellius/1d/non_matching_initial/p3/uniform_standard/production_4M.yml
```

The checked-in `algorithm_data_plot/` files were prepared from a selected
completed subset of production and pilot runs. See
`algorithm_data_plot/README.md` for the exact retained run table.

## Output

The runner writes output to the directory selected by the config `output` key
or by the `--output` command-line option. By convention, local generated output
goes under `algorithm_data/`; see `algorithm_data/README.md`.

Typical exported files include:

| File | Meaning |
|---|---|
| `outer_history.csv` | One row per accepted outer X iteration. |
| `inner_history.csv` | One row per inner Y iteration. |
| `timing_history.csv` | Timing breakdown when enabled. |
| estimator-component CSV files | Detailed local estimator components when requested. |
| refinement-history files | Refinement decisions and mesh statistics when requested. |
| snapshot binary files | Mesh, error, and DoF snapshots when snapshot export is enabled. |

Use `--output-profile minimal --no-export` for smoke checks that should not
leave generated algorithm data behind.

## All Command-Line Flags

This table mirrors `algorithm/runners/detail/runner_option_specs.hpp`.
Run `./path/to/run_adaptive_algorithm --help` after rebuilding to inspect the
exact help text for the current executable.

| Flag | Value | Purpose |
|---|---|---|
| `--config` | `PATH` | Load parameters from a flat `.yml` config file. |
| `--example` | `NAME` | Predefined example name. |
| `--dimension` | `N` | Expected spatial dimension of the selected example. Supported: `1`, `2`. If omitted, the dimension is inferred from the registered example. |
| `--dim-space` | `N` | Alias for `--dimension`. |
| `--rho` | `VALUE` | Rho parameter for the inner Y loop. |
| `--max-outer` | `N` | Maximum number of outer X iterations. |
| `--max-inner` | `N` | Maximum number of inner Y iterations. |
| `--force-accept-inner-with-effective-rho` |  | Accept the last inner iteration at `max-inner` by computing the required effective rho. Configured rho is ignored as an inner stopping criterion while this mode is active. |
| `--no-force-accept-inner-with-effective-rho` |  | Use the configured rho criterion for normal inner-loop acceptance. |
| `--effective-rho-only-inner-acceptance` |  | Alias for `--force-accept-inner-with-effective-rho`. |
| `--compute-g-estimator` |  | Assemble and solve the accepted-iteration `G^delta` enriched reconstruction. `G^delta` uses `Q_{p+1,p+1}` on the accepted Y mesh by default. |
| `--no-compute-g-estimator` |  | Do not construct or solve the `G^delta` enriched reconstruction. |
| `--compute-g-estimator-on-empty-y-marking-stop` |  | Also compute `G^delta` when `stop_on_empty_y_marking` ends the inner loop. Rejected Y iterations compute `G^delta` only with `--compute-g-estimator-every-inner-iteration`. |
| `--no-compute-g-estimator-on-empty-y-marking-stop` |  | Do not compute `G^delta` for an empty-Y-marking stop. |
| `--compute-g-estimator-every-inner-iteration` |  | Compute and save `G^delta` diagnostics for every inner Y iteration. Requires `--compute-g-estimator` and can be much more expensive than accepted-only `G^delta`. |
| `--no-compute-g-estimator-every-inner-iteration` |  | Compute `G^delta` only for accepted inner iterations. |
| `--g-solver` | `NAME` | Solver used for accepted-iteration `G^delta` reconstruction solves. Default: `same_as_main`. Supported: `same_as_main` plus the main-solver choices. |
| `--g-solver-tolerance` | `VALUE` | Tolerance for the `G^delta` reconstruction solver. Default: `0`, meaning inherit the effective main-solver tolerance. |
| `--g-solver-memory-limit-mb` | `MB` | Direct-solver memory limit for the `G^delta` reconstruction solver. Default: `0`, meaning inherit the effective main-solver memory limit. |
| `--theta-x` | `VALUE` | Doerfler bulk parameter for X marking. |
| `--theta-y` | `VALUE` | Doerfler bulk parameter for Y marking. |
| `--uniform-x-refinement` |  | Refine every active `X^delta` cell after each accepted outer solve instead of Doerfler-marking X. The inner Y loop remains adaptive. |
| `--adaptive-x-refinement` |  | Use the default Doerfler X-marking instead of uniform X refinement. |
| `--uniform-y-refinement` |  | Refine every active `Y^delta` cell after a non-accepted inner solve instead of Doerfler-marking Y. |
| `--adaptive-y-refinement` |  | Use the default Doerfler Y-marking instead of uniform Y refinement. |
| `--uniform-refinement-mode` | `MODE` | Set the effective refinement mode. Supported: `adaptive`, `uniform_x`, `uniform_y`, `uniform_xy`. Explicit `--uniform-x-refinement` or `--uniform-y-refinement` flags override this mode. |
| `--p` | `N` | Polynomial degree. Supported: `1`, `2`, `3`, `4`. |
| `--zero-tol` | `VALUE` | Zero tolerance used in assembly and filtering. |
| `--divergence-tol` | `VALUE` | Allowed L2 norm for the divergence-residual sanity check. |
| `--eta-stop` | `VALUE` | Stop once `eta_squared` is below this threshold. |
| `--inner-estimator-stop` | `VALUE` | Stop the inner loop once the estimator squared is below this threshold. |
| `--max-wall-time-seconds` | `VALUE` | Soft wall-time budget for the adaptive run. `0` disables. The active solve is not interrupted; the driver stops before the next adaptive iteration. |
| `--max-y-dofs` | `N` | Stop before the next inner solve if `Y^delta` true DoFs exceed this limit. Default: `500000`. Use `0` to disable. |
| `--max-x-dofs` | `N` | Stop before the next outer iteration if `X^delta` true DoFs exceed this limit. Default: `0` disables. Production configs use this as the primary DoF target. |
| `--max-dofs-target` | `TARGET` | Select which DoF cap is the primary requested maximum. Supported: `y`, `x`. Existing configs default to `y`; Snellius production configs set `x`. |
| `--memory-limit-mb` | `MB` | Whole-process adaptive-run memory cap. Default: inherits `--main-solver-memory-limit-mb`. Use `0` together with `--main-solver-memory-limit-mb 0` to disable. |
| `--memory-reserve-mb` | `MB` | Minimum Linux `MemAvailable` reserve kept by the adaptive memory guard. Default: `2048` MiB. |
| `--memory-guard-safety-factor` | `VALUE` | Multiplier applied to adaptive and PARDISO memory estimates. Default: `1.15`. |
| `--memory-guard-near-cap-fraction` | `VALUE` | Fraction of the memory cap at which thread counts are reduced and PARDISO out-of-core switching is enabled earlier. Default: `0.85`. |
| `--increased-accuracy` |  | Use higher runner accuracy: `QSpace += 2`, `QTime += 2`, and solver residual tolerances are tightened by a factor 10. |
| `--high-accuracy` |  | Alias for `--increased-accuracy`. |
| `--normal-accuracy` |  | Disable the increased-accuracy quadrature and solver-tolerance boost. |
| `--main-solver` | `NAME` | Main-system solver choice. Default: `pardiso_ldlt` in MKL builds, `sparse_lu` otherwise. Supported: `sparse_lu`, `pardiso_lu`, `pardiso_ldlt`, `pardiso_ldlt_auto`, `minres_parabolic_graph_norm`. Without MKL, PARDISO choices fall back to `sparse_lu`. |
| `--main-solver-pardiso-memory-mode` | `MODE` | MKL PARDISO memory mode for the main solver. Supported: `in_core`, `auto`, `out_of_core`. Used only with PARDISO solvers. |
| `--main-solver-max-iterations` | `N` | Maximum number of iterations for the main-system iterative solver. Ignored by direct solvers. |
| `--main-solver-tolerance` | `VALUE` | Relative tolerance for the main-system iterative solver. Ignored by direct solvers. |
| `--main-solver-symmetry-tolerance` | `VALUE` | Relative symmetry tolerance for `pardiso_ldlt_auto`. Used only when selecting between PARDISO LDLT and PARDISO LU automatically. |
| `--main-solver-direct-residual-retry` |  | Enable direct-solver true-residual validation and retry with a safer direct solver. PARDISO LDLT retries with PARDISO LU; unavailable PARDISO requests use SparseLU in non-MKL builds. |
| `--no-main-solver-direct-residual-retry` |  | Disable direct-solver residual retry. |
| `--main-solver-direct-residual-retry-tolerance` | `VALUE` | True relative residual threshold for direct-solver retry. Default: `1e-10`. |
| `--main-solver-diagnostics` | `MODE` | Post-solve main-system diagnostic mode. Supported: `off`, `summary`, `detailed`. |
| `--main-solver-memory-limit-mb` | `MB` | Direct main-solver memory limit. Default: `10000` MiB. Use `0` to disable. |
| `--main-solver-ooc-auto-switch` |  | Switch MKL PARDISO from in-core to out-of-core before factorization when the memory estimate approaches the limit. |
| `--no-main-solver-ooc-auto-switch` |  | Disable automatic PARDISO out-of-core fallback. |
| `--main-solver-ooc-switch-threshold` | `VALUE` | Fraction of `--main-solver-memory-limit-mb` at which in-core PARDISO switches to out-of-core. Default: `0.85`. |
| `--main-solver-ooc-switch-to-lu` |  | Use PARDISO LU for automatic out-of-core fallback from LDLT. |
| `--main-solver-ooc-keep-ldlt` |  | Keep PARDISO LDLT when automatic out-of-core fallback is triggered. |
| `--main-solver-reuse-symbolic-analysis` |  | Reuse SparseLU symbolic analysis when the same solver sees an identical sparsity pattern. |
| `--no-main-solver-reuse-symbolic-analysis` |  | Disable SparseLU symbolic-analysis reuse. |
| `--output` | `PATH` | Output directory. Defaults to `algorithm_data/<example>/`. |
| `--print-iterations` |  | Enable iteration-table printing. |
| `--quiet` |  | Disable iteration-table printing. |
| `--check-divergence` |  | Enforce the divergence-residual sanity check. |
| `--no-check-divergence` |  | Disable the divergence-residual sanity check. |
| `--stop-on-empty-y-marking` |  | Stop if Y-marking becomes empty before the threshold is met. |
| `--allow-empty-y-marking` |  | Continue even if Y-marking becomes empty. |
| `--local-time-slab-closure` |  | Temporally close marked Y cells crossed by existing time interfaces. Only marked split cells are closed; non-split marked cells use the normal refinement rule. |
| `--no-local-time-slab-closure` |  | Disable local time-slab closure for marked Y cells. |
| `--use-adaptive-initial-guess` |  | Use the previous adaptive solution prolongated to the refined space as the main-solve initial guess. |
| `--no-adaptive-initial-guess` |  | Do not pass a prolongated adaptive initial guess to the main solve. |
| `--solve-main-system-correction` |  | Solve `K e = b - K x0` and set `x = x0 + e` when an adaptive initial guess is available. |
| `--no-solve-main-system-correction` |  | Pass the adaptive initial guess directly instead of solving the correction equation. |
| `--fused-error-and-flux-diagnostics` |  | Use fused 2+1D local-error/flux-diagnostic data when the reconstruction path provides it. |
| `--no-fused-error-and-flux-diagnostics` |  | Disable fused local-error/flux diagnostics and use the standalone diagnostic fallback. |
| `--local-error-reuse-patch-solve-workspace` |  | Reuse per-thread dense local patch solve workspaces in 2+1D local-error solves. |
| `--no-local-error-reuse-patch-solve-workspace` |  | Disable per-thread dense local patch solve workspace reuse in 2+1D local-error solves. |
| `--deterministic-estimator-reductions` |  | Use deterministic source-cell estimator reductions and Doerfler marking. |
| `--no-deterministic-estimator-reductions` |  | Use the legacy unordered-map estimator reduction path. |
| `--doerfler-near-tie-tolerance` | `VALUE` | Diagnostic tolerance for counting near ties at the Doerfler cutoff. Default: `0.0`. This records near ties but never changes marking. |
| `--refinement-edge-query-cache` |  | Enable the per-refinement-call 2D edge-query cache. |
| `--no-refinement-edge-query-cache` |  | Disable the per-refinement-call 2D edge-query cache. |
| `--refinement-batch-target-split-cells` | `N` | Target number of split active cells before flushing a 2D indexed-refinement batch. Default: `32`. |
| `--post-flush-closure-mode` | `MODE` | Choose the 2D post-flush forced-closure query set. Supported: `off_debug`, `split_edges_only_debug`, `split_and_inherited_edges`, `affected_edges`, `presplit_neighbour`, `all_faces_debug`. |
| `--post-flush-affected-containment-only` |  | Use the experimental containment-only query path for affected-edge post-flush closure. |
| `--no-post-flush-affected-containment-only` |  | Disable the experimental affected-edge containment-only post-flush query path. |
| `--refinement-full-conformity-check` |  | Run the full 2D active spatial conformity verifier after every refinement, even in release builds. |
| `--no-refinement-full-conformity-check` |  | Disable the full 2D active spatial conformity verifier in release builds. |
| `--refinement-main-closure-query-mode` | `MODE` | Choose the 2D main-queue closure edge-query mode. Supported: `exact_and_ancestor`, `exact_ancestor_plus_containment`, `old_bidirectional_debug`. |
| `--local-error-patch-tile-size` | `N` | Use tiled 2+1D local-error patch assembly/solves with at most N patches per tile. `0` uses the degree-aware default. |
| `--local-error-cell-chunk-size` | `N` | When tiled 2+1D local-error assembly is active, stream shared cell-state caches in chunks of at most N slab cells. `0` uses the degree-aware default. |
| `--local-error-max-threads` | `N` | Maximum OpenMP threads for 2+1D local-error work. `0` uses the automatic cap. |
| `--local-error-memory-budget-mb` | `MB` | Approximate memory budget for 2+1D local-error thread selection. `0` disables budget limiting. |
| `--local-error-worker-context-mode` | `MODE` | Deprecated debug alias for local-error context storage. Supported: `persistent`, `per_chunk_debug`, `persistent_all_p_debug`. |
| `--local-error-context-storage` | `MODE` | Choose the local-error context storage/shadow mode. Supported: `shared_immutable`, `per_chunk_debug`, `persistent_per_thread_debug`, `shared_immutable_shadow`. |
| `--local-error-state-index-mode` | `MODE` | Choose the 2+1D local-error slab-cell state lookup mode. Supported: `flat`, `map_debug`. |
| `--local-error-cell-state-cache-mode` | `MODE` | Choose the 2+1D cross-tile local-error cell-state cache mode. Supported: `off`, `tile`, `bounded_lru`, `lifetime_window`, `full_if_fits`. |
| `--local-error-cell-state-cache-budget-mb` | `MB` | Memory budget for bounded LRU/full-if-fits local-error cell-state reuse. `0` disables bounded storage. |
| `--local-error-cell-state-representation` | `MODE` | Choose the 2+1D local-error cell-state representation. Supported: `compact_split`, `monolithic_debug`. |
| `--local-error-flux-diagnostics-mode` | `MODE` | Choose the 2+1D flux-diagnostics execution mode. Supported: `auto`, `streaming_reuse`, `standalone`. |
| `--local-error-patch-solver` | `MODE` | Choose the 2+1D dense local patch solver. Supported: `current_dense`, `reduced_scalar_dense`, `auto`. |
| `--local-error-coefficient-fast-path` |  | Use coefficient-aware local-error fast paths for identity/constant diffusion and zero load. |
| `--no-local-error-coefficient-fast-path` |  | Disable coefficient-aware local-error fast paths and force the generic q-point coefficient path. |
| `--local-error-compact-state-shadow` |  | Build compact/split local-error shadow states on sampled cells and compare against monolithic cell data. |
| `--no-local-error-compact-state-shadow` |  | Disable compact/split local-error shadow-state validation. |
| `--shared-context-validation` | `MODE` | Choose shared immutable context validation sampling. Supported: `off`, `sample`, `full_debug`. |
| `--slab-reconstruction-operator-mode` | `MODE` | Choose coefficient handling for copied-slab reconstruction assembly. Supported: `auto`, `identity_zero_load_fast_path`, `constant_diffusion_fast_path`, `generic_variable_path`. |
| `--main-assembly-max-threads` | `N` | Maximum OpenMP threads for main-system assembly. Default: `4`; `0` uses the automatic OpenMP maximum. |
| `--slab-reconstruction-max-threads` | `N` | Maximum OpenMP threads for 2+1D slab reconstruction assembly/solves. Default: `4`; `0` uses the automatic OpenMP maximum. |
| `--main-assembly-memory-budget-mb` | `MB` | Recorded memory budget for main-system assembly thread-local buffers. `0` disables budget accounting. |
| `--slab-reconstruction-memory-budget-mb` | `MB` | Approximate memory budget for 2+1D slab reconstruction thread-local caches. `0` disables budget limiting. |
| `--main-two-pass-numeric-fill-max-threads` | `N` | Maximum OpenMP threads for memory-bounded two-pass main-system numeric fill. `0` uses the default of `4`. |
| `--main-two-pass-numeric-fill-memory-budget-mb` | `MB` | Approximate memory budget for two-pass numeric-fill thread-local buffers. `0` disables budget limiting. |
| `--time-slab-backend` | `NAME` | Time-slab backend interface to construct. Supported: `copied_mesh`. |
| `--allow-copied-time-slab-estimator-fallback` |  | Compatibility flag accepted by older configs; `copied_mesh` is always used. |
| `--strict-virtual-time-slab-estimator` |  | Compatibility flag accepted by older configs; `copied_mesh` is always used. |
| `--virtual-backend-diagnostics` |  | Compatibility flag accepted by older configs; virtual diagnostics are disabled. |
| `--no-virtual-backend-diagnostics` |  | Compatibility flag accepted by older configs; virtual diagnostics are disabled. |
| `--solver-diagnostics` |  | Write per-main-solve solver diagnostics metadata files. Matrix/RHS/solution export stays disabled unless requested separately. |
| `--no-solver-diagnostics` |  | Disable per-main-solve solver diagnostics metadata files. |
| `--export-main-matrix-market` |  | When solver diagnostics are enabled, export the full main-system matrix in MatrixMarket format if below the DoF cap. |
| `--export-main-rhs` |  | When solver diagnostics are enabled, export the full main-system RHS vector if below the DoF cap. |
| `--export-main-solution` |  | When solver diagnostics are enabled, export the solved full main-system vector if below the DoF cap. |
| `--solver-diagnostics-max-export-dofs` | `N` | Maximum main-system rows for MatrixMarket/RHS/solution export. Default: `20000`. |
| `--output-profile` | `NAME` | Grouped output policy. Supported: `minimal`, `production`, `benchmark`, `debug`. Individual output flags still override the profile. |
| `--save-heavy-diagnostics` |  | Enable detailed estimator components, refinement history, snapshots, and detailed timing. Snapshot DoF binaries remain opt-in via `--save-snapshot-dofs`. |
| `--export` |  | Enable CSV/text history export. |
| `--no-export` |  | Disable CSV/text history export. |
| `--estimator-components` |  | Enable detailed estimator-component output. |
| `--no-estimator-components` |  | Skip detailed estimator-component output. |
| `--refinement-history` |  | Enable `refinement_history.txt` export. |
| `--no-refinement-history` |  | Skip `refinement_history.txt`. |
| `--mesh-statistics` |  | Enable final mesh-statistics lines in the summary. |
| `--no-mesh-statistics` |  | Skip mesh-statistics lines in the summary. |
| `--no-snapshots` |  | Disable per-iteration mesh/error snapshots. |
| `--save-snapshots` |  | Save per-iteration mesh/error snapshots. |
| `--no-snapshot-dofs` |  | Disable DoF binaries inside the snapshots. |
| `--save-snapshot-dofs` |  | Also save DoF binaries inside the snapshots. |
| `--enable-timing-breakdown` |  | Enable timing breakdown collection and CSV export. |
| `--no-timing-breakdown` |  | Disable timing breakdown collection and CSV export. |
| `--timing-history-filename` | `NAME` | Timing breakdown CSV filename. Default: `timing_history.csv`. |
| `--timing-detail-level` | `LEVEL` | Timing breakdown detail level. Supported: `summary`, `detailed`. Currently both write run-level aggregate rows. |
| `--list-examples` |  | Print the available example names and dimensions and exit. |
| `--help` |  | Print this help text and exit. |

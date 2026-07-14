# Cleaned Plot Data

This directory contains cleaned CSV data prepared from completed production
runs. The files are intended for convergence and diagnostic plots associated
with the accompanying paper.

Regenerate this layout from an ordered result directory with:

```bash
python3 scripts/snellius/prepare_plot_data.py \
  --ordered-root algorithm_data_ordered \
  --output-root algorithm_data_plot
```

## Files

- `files.csv`: index of exported CSV files.
- `index.csv`: case-level index of selected runs.
- `1d/*.csv`: long-form 1D plot data by example.
- `2d/*.csv`: long-form 2D plot data by example.

## Included Run Sizes

The checked-in plot data is a selected completed subset of the larger Snellius
production matrix. The exact retained runs are listed in `index.csv`; the table
below records the dimension, example, polynomial degree, run mode, DoF target,
Slurm job id, and row counts used in the cleaned CSVs.

| Plot CSV | Case count | Outer rows | Inner rows | Inner summary rows |
|---|---:|---:|---:|---:|
| `1d/boundary_singularity.csv` | 12 | 383 | 448 | 383 |
| `1d/non_matching_initial.csv` | 12 | 394 | 402 | 394 |
| `1d/smooth_initial.csv` | 15 | 490 | 732 | 490 |
| `2d/boundary_singularity.csv` | 9 | 238 | 240 | 238 |
| `2d/non_matching_initial.csv` | 9 | 220 | 220 | 220 |
| `2d/smooth_initial.csv` | 12 | 360 | 367 | 360 |

| Dim | Example | p | Mode | Target | X DoFs | Job | Outer rows | Inner rows |
|---|---|---:|---|---|---:|---:|---:|---:|
| 1d | boundary_singularity | 1 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332815 | 43 | 43 |
| 1d | boundary_singularity | 1 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24339160 | 41 | 68 |
| 1d | boundary_singularity | 1 | `uniform_standard` | `production_4M` | 4,000,000 | 24332817 | 10 | 10 |
| 1d | boundary_singularity | 2 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332818 | 45 | 45 |
| 1d | boundary_singularity | 2 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24332819 | 43 | 53 |
| 1d | boundary_singularity | 2 | `uniform_standard` | `production_4M` | 4,000,000 | 24332820 | 9 | 9 |
| 1d | boundary_singularity | 3 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332821 | 45 | 45 |
| 1d | boundary_singularity | 3 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24332822 | 45 | 59 |
| 1d | boundary_singularity | 3 | `uniform_standard` | `production_4M` | 4,000,000 | 24332823 | 9 | 9 |
| 1d | boundary_singularity | 4 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339267 | 43 | 43 |
| 1d | boundary_singularity | 4 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339268 | 42 | 56 |
| 1d | boundary_singularity | 4 | `uniform_standard` | `production_2M` | 2,000,000 | 24339269 | 8 | 8 |
| 1d | non_matching_initial | 1 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332824 | 44 | 44 |
| 1d | non_matching_initial | 1 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24332825 | 44 | 48 |
| 1d | non_matching_initial | 1 | `uniform_standard` | `production_4M` | 4,000,000 | 24332826 | 10 | 10 |
| 1d | non_matching_initial | 2 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332827 | 47 | 47 |
| 1d | non_matching_initial | 2 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24332828 | 47 | 47 |
| 1d | non_matching_initial | 2 | `uniform_standard` | `production_4M` | 4,000,000 | 24332829 | 9 | 9 |
| 1d | non_matching_initial | 3 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332830 | 46 | 46 |
| 1d | non_matching_initial | 3 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24332831 | 46 | 49 |
| 1d | non_matching_initial | 3 | `uniform_standard` | `production_4M` | 4,000,000 | 24332832 | 9 | 9 |
| 1d | non_matching_initial | 4 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339270 | 42 | 42 |
| 1d | non_matching_initial | 4 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339271 | 42 | 43 |
| 1d | non_matching_initial | 4 | `uniform_standard` | `production_2M` | 2,000,000 | 24339272 | 8 | 8 |
| 1d | smooth_initial | 1 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332787 | 46 | 46 |
| 1d | smooth_initial | 1 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24339161 | 42 | 73 |
| 1d | smooth_initial | 1 | `adaptive_standard` | `production_2M` | 2,000,000 | 24330389 | 39 | 64 |
| 1d | smooth_initial | 1 | `uniform_standard` | `production_4M` | 4,000,000 | 24332789 | 10 | 10 |
| 1d | smooth_initial | 2 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332790 | 42 | 42 |
| 1d | smooth_initial | 2 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24332791 | 40 | 78 |
| 1d | smooth_initial | 2 | `adaptive_standard` | `production_2M` | 2,000,000 | 24330393 | 37 | 64 |
| 1d | smooth_initial | 2 | `uniform_standard` | `production_4M` | 4,000,000 | 24332792 | 9 | 9 |
| 1d | smooth_initial | 3 | `adaptive_force_effective_rho_no_inner_refinement` | `production_4M` | 4,000,000 | 24332793 | 44 | 44 |
| 1d | smooth_initial | 3 | `adaptive_g_effective_rho` | `production_4M` | 4,000,000 | 24332794 | 42 | 94 |
| 1d | smooth_initial | 3 | `adaptive_standard` | `production_2M` | 2,000,000 | 24330396 | 40 | 82 |
| 1d | smooth_initial | 3 | `uniform_standard` | `production_4M` | 4,000,000 | 24332795 | 9 | 9 |
| 1d | smooth_initial | 4 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339273 | 44 | 44 |
| 1d | smooth_initial | 4 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339274 | 38 | 65 |
| 1d | smooth_initial | 4 | `uniform_standard` | `production_2M` | 2,000,000 | 24339275 | 8 | 8 |
| 2d | boundary_singularity | 1 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339162 | 43 | 43 |
| 2d | boundary_singularity | 1 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24357997 | 43 | 45 |
| 2d | boundary_singularity | 1 | `uniform_standard` | `production_2M` | 2,000,000 | 24339164 | 7 | 7 |
| 2d | boundary_singularity | 2 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339165 | 36 | 36 |
| 2d | boundary_singularity | 2 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339166 | 36 | 36 |
| 2d | boundary_singularity | 2 | `uniform_standard` | `production_2M` | 2,000,000 | 24339167 | 6 | 6 |
| 2d | boundary_singularity | 3 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339168 | 31 | 31 |
| 2d | boundary_singularity | 3 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339169 | 31 | 31 |
| 2d | boundary_singularity | 3 | `uniform_standard` | `production_2M` | 2,000,000 | 24339170 | 5 | 5 |
| 2d | non_matching_initial | 1 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339171 | 39 | 39 |
| 2d | non_matching_initial | 1 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339172 | 39 | 39 |
| 2d | non_matching_initial | 1 | `uniform_standard` | `production_2M` | 2,000,000 | 24339173 | 7 | 7 |
| 2d | non_matching_initial | 2 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339174 | 32 | 32 |
| 2d | non_matching_initial | 2 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339175 | 32 | 32 |
| 2d | non_matching_initial | 2 | `uniform_standard` | `production_2M` | 2,000,000 | 24339177 | 6 | 6 |
| 2d | non_matching_initial | 3 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24339178 | 30 | 30 |
| 2d | non_matching_initial | 3 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24339179 | 30 | 30 |
| 2d | non_matching_initial | 3 | `uniform_standard` | `production_2M` | 2,000,000 | 24339180 | 5 | 5 |
| 2d | smooth_initial | 1 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24352661 | 45 | 45 |
| 2d | smooth_initial | 1 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24352662 | 44 | 50 |
| 2d | smooth_initial | 1 | `adaptive_standard` | `pilot_500k` | 500,000 | 24330376 | 38 | 39 |
| 2d | smooth_initial | 1 | `uniform_standard` | `production_2M` | 2,000,000 | 24332864 | 7 | 7 |
| 2d | smooth_initial | 2 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24332865 | 38 | 38 |
| 2d | smooth_initial | 2 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24332866 | 38 | 38 |
| 2d | smooth_initial | 2 | `adaptive_standard` | `pilot_500k` | 500,000 | 24330370 | 32 | 32 |
| 2d | smooth_initial | 2 | `uniform_standard` | `production_2M` | 2,000,000 | 24332867 | 6 | 6 |
| 2d | smooth_initial | 3 | `adaptive_force_effective_rho_no_inner_refinement` | `production_2M` | 2,000,000 | 24332868 | 38 | 38 |
| 2d | smooth_initial | 3 | `adaptive_g_effective_rho` | `production_2M` | 2,000,000 | 24332869 | 38 | 38 |
| 2d | smooth_initial | 3 | `adaptive_standard` | `pilot_500k` | 500,000 | 24330373 | 31 | 31 |
| 2d | smooth_initial | 3 | `uniform_standard` | `production_2M` | 2,000,000 | 24332870 | 5 | 5 |

## Record Types

- `record_type=outer`: estimator-versus-X-DoFs convergence rows.
- `record_type=inner`: raw inner-iteration diagnostic rows.
- `record_type=inner_summary`: one row per outer iteration with summary
  statistics for inner-iteration ratios.

## Common Plot Columns

- `x_true_dofs`: X-space true DoFs used for the estimator.
- `posteriori_estimator_configured_rho`: square root of the configured-rho
  posteriori estimator.
- `posteriori_estimator_effective_rho`: square root of the effective-rho
  posteriori estimator when available.
- `g_lambda_difference`: available G-delta difference norm.
- `y_estimator`: inner-loop estimator norm.
- `y_flux` and `y_reconstruction`: inner-estimator component norms.
- `divergence_residual`: inner-loop PDE-balance residual norm.
- `flux_over_reconstruction`: `y_flux / y_reconstruction`.
- `reconstruction_over_flux`: `y_reconstruction / y_flux`.
- `divergence_over_y_estimator`: `divergence_residual / y_estimator`.
- `g_over_y_estimator`: `g_lambda_difference / y_estimator`.

Squared estimator quantities are kept next to square-rooted plotting quantities
for consistency checks and squared-norm log-log plots.

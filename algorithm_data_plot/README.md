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

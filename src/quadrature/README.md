# Quadrature

`quadrature/` contains reference integration rules and cell mappings used by
finite-element assembly and error estimation.

## Files

| File | Purpose |
|---|---|
| `quadrature_rule.hpp` | Generic quadrature rule container and point/weight representation. |
| `gauss_legendre_1d.hpp` | 1D Gauss-Legendre rules. |
| `tensor_product_quadrature.hpp` | Tensor-product construction for space-time product cells. |
| `dunavant_triangle.hpp` | Dunavant-style reference triangle rules. |
| `reference_triangle_duffy.hpp` | Duffy-transformed triangle rules for singular or tensor-derived integrations. |
| `reference_quadrature.hpp` | Dimension/reference-cell dispatch for standard rules. |
| `cell_mappings.hpp` | Reference-to-physical mapping helpers and Jacobian-related data. |

## Usage Notes

- Main-system and local-error assembly choose quadrature orders through the
  finite-element assembly layer, then consume the rules from this directory.
- The algorithm runner's `--increased-accuracy` flag raises the effective
  quadrature orders used by the degree/dimension runners.
- Keep quadrature rules deterministic and immutable after construction; many
  assembly caches assume reference rules can be reused safely.

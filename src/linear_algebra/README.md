# Linear Algebra

`linear_algebra/` provides the backend-neutral interfaces and the Eigen-backed
implementation used by the adaptive runner. It also contains saddle-system
wrappers, sparse/dense operations, block assembly helpers, and preconditioners.

## Layout

| Path | Purpose |
|---|---|
| `concepts/` | Compile-time concepts for vectors, sparse matrices, dense matrices, solvers, builders, and preconditioners. |
| `eigen_backend/` | Eigen-backed concrete vector, matrix, builder, solver, and dense-solver types. |
| `eigen_backend/preconditioners/` | Eigen-backed preconditioner and MINRES support for the parabolic graph norm path. |
| `operations/` | Backend-neutral scalar, vector, sparse, dense, and general linear-algebra operations. |
| `system/` | Linear-system and saddle-point-system containers plus solve dispatch. |
| `preconditioners/` | Backend-neutral preconditioner diagnostics, Schur approximations, solve failure types, and saddle-system solve helpers. |
| `assembly/` | Helpers for composing block systems and local objects. |
| `dense/constexpr_linalg.hpp` | Small fixed-size compile-time dense linear algebra helpers. |

## Solver Paths

The runner exposes solver choices through `--main-solver` and `--g-solver`.
The available paths depend on build configuration:

| Solver family | Build requirement | Notes |
|---|---|---|
| `sparse_lu` | Eigen only | Default fallback when MKL/PARDISO is not enabled. |
| `pardiso_lu`, `pardiso_ldlt`, `pardiso_ldlt_auto` | `ENABLE_MKL_PARDISO=ON` and oneMKL found by CMake | Production Snellius configs use `pardiso_ldlt`. |
| `minres_parabolic_graph_norm` | Eigen backend and graph-norm preconditioner support | Iterative path controlled by the main-solver iteration/tolerance flags. |

The CMake definition `ADAPPARABOLICFEM_HAVE_MKL_PARDISO` records whether the
PARDISO-backed paths are available. Without MKL, PARDISO requests fall back to
the Eigen SparseLU path in the runner.

## Development Notes

- Keep algorithm-level code against the concepts and system abstractions where
  possible. Concrete Eigen types should stay in `eigen_backend/`.
- Sparse assembly should use the sparse pattern/builder abstractions rather
  than filling Eigen matrices ad hoc.
- Direct-solver diagnostics, residual retries, memory limits, and optional
  MatrixMarket/RHS/solution exports are controlled at the runner level but
  implemented through this backend layer.

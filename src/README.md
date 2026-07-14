# Source Tree

This directory contains the numerical C++ implementation used by the adaptive
algorithm in `../algorithm/`. The code is organized as a header-only library
exposed through the CMake interface target `adap_parabolic_fem`. The production
executable includes these headers directly and is compiled from the runner
translation units in `../algorithm/runners/`.

## Module Map

| Directory | Responsibility |
|---|---|
| `core/` | Small shared utilities: strongly typed ids, coordinate keys, hashing, OpenMP helpers, debug checks, and exceptions. |
| `mesh/` | Adaptive space-time mesh data structures, topology queries, boundary detection, refinement, and time-slicing. |
| `quadrature/` | Reference quadrature rules, tensor-product rules, triangle rules, Duffy transforms, and cell mappings. |
| `finite_element/` | Basis functions, nodes, finite-element spaces, DoF distributions, assembly, time-slab spaces, estimators, reconstruction, and binary IO. |
| `linear_algebra/` | Backend-neutral linear algebra concepts, Eigen-backed matrices/vectors/solvers, block assembly helpers, saddle-system wrappers, and preconditioners. |

Each major subsystem has its own README with a more focused file map:

- `core/README.md`
- `mesh/README.md`
- `quadrature/README.md`
- `finite_element/README.md`
- `linear_algebra/README.md`

## High-Level Data Flow

The adaptive runner builds and evolves a problem through the following layers:

1. `mesh/` creates an initial space-time mesh and refines active cells.
2. `finite_element/fespace/` builds finite-element spaces and DoF layouts on
   the mesh.
3. `quadrature/`, `finite_element/basis/`, `finite_element/nodes/`, and
   `finite_element/tables/` provide local shape-function and integration data.
4. `finite_element/assembly/main_system/` assembles the primal saddle system.
5. `linear_algebra/` solves the assembled system using the configured backend
   and solver.
6. `finite_element/time_slabs/` constructs copied time-slab views, local patch
   problems, equilibrated flux reconstructions, and cellwise indicators.
7. `algorithm/` consumes the indicators, marks/refines X and Y meshes, and
   writes histories and optional binary artifacts through `finite_element/io/`.

## Build Model

The library target is declared in the top-level `CMakeLists.txt`:

```cmake
add_library(adap_parabolic_fem INTERFACE)
target_include_directories(adap_parabolic_fem INTERFACE src .)
target_link_libraries(adap_parabolic_fem INTERFACE Eigen3::Eigen)
```

Important build definitions supplied by CMake:

| Definition | Meaning |
|---|---|
| `ADAPPARABOLICFEM_SOURCE_DIR` | Absolute source directory used by examples and diagnostics. |
| `ADAPPARABOLICFEM_BINARY_DIR` | Active CMake binary directory. |
| `ADAPPARABOLICFEM_HAS_OPENMP` | `1` when OpenMP was found and linked, otherwise `0`. |
| `ADAPPARABOLICFEM_HAVE_MKL_PARDISO` | `1` when oneMKL/PARDISO support is enabled, otherwise `0`. |

The public include paths are `src/` and the repository root, so includes use
repository-relative paths such as:

```cpp
#include "mesh/mesh.hpp"
#include "finite_element/time_slabs/time_slab_space.hpp"
#include "linear_algebra/eigen_backend/backend.hpp"
```

## Design Conventions

- The code is template-heavy and dimension-aware. Many routines are specialized
  for the supported 1+1D and 2+1D cases.
- Data structures use explicit ids and keys rather than raw integer positions
  when identity must survive refinement or reordering.
- `linear_algebra/concepts/` defines the backend-facing contracts; the Eigen
  implementation lives under `linear_algebra/eigen_backend/`.
- OpenMP-specific code is guarded through the CMake-provided OpenMP definition
  and helpers in `core/openmp.hpp`.
- Binary output is versioned in `finite_element/io/detail/` and read by the
  plotting helpers in `../python/`.
- Production performance options are exposed at the algorithm runner level,
  but most of the implementation hooks live in `finite_element/time_slabs/`,
  `finite_element/assembly/`, and `linear_algebra/`.

## Where To Start

For mesh and refinement behavior, start with:

- `mesh/mesh.hpp`
- `mesh/cell.hpp`
- `mesh/refinement/refine.hpp`
- `mesh/topology/leaf_traversal.hpp`

For the main solve path, start with:

- `finite_element/system/assemble_saddle_problem.hpp`
- `finite_element/system/solve_main_system.hpp`
- `finite_element/assembly/main_system/`
- `linear_algebra/system/saddle_point_system.hpp`

For estimator and time-slab behavior, start with:

- `finite_element/time_slabs/time_slab_adaptive_loop.hpp`
- `finite_element/time_slabs/time_slab_error_indicators.hpp`
- `finite_element/time_slabs/time_slab_equilibrated_flux_reconstruction.hpp`
- `finite_element/time_slabs/time_slab_estimator_combination.hpp`

For binary outputs consumed by the Python plotters, start with:

- `finite_element/fespace/io/write_mesh_binary.hpp`
- `finite_element/fespace/io/write_dofs_binary.hpp`
- `finite_element/io/write_time_slab_space_binary.hpp`
- `finite_element/io/write_time_slab_function_binary.hpp`

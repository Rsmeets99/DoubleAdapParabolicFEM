# Finite Element

`finite_element/` is the largest subsystem. It builds finite-element spaces on
adaptive meshes, constructs DoF distributions, assembles main and local error
systems, manages time-slab views, computes estimators, and writes binary
artifacts for the Python plotting tools.

## Layout

| Path | Purpose |
|---|---|
| `basis/` | Polynomial and finite-element basis functions for segments, triangles, quadrilaterals, and triangular prisms. |
| `nodes/` | Node-set definitions used by finite-element bases. |
| `tables/` | Precomputed or cached element tables for basis evaluation and quadrature. |
| `geometry/` | Cell geometry wrappers used by assembly and mapping code. |
| `coefficients/` | Coefficient abstractions such as diffusion coefficients. |
| `fespace/` | Finite-element space construction, DoF distribution, adjacency, constraints, prolongation, refinement, and basic binary IO. |
| `assembly/main_system/` | Local and global assembly of the primal saddle system. |
| `assembly/error_system/` | Local patch problem assembly for error estimation and flux reconstruction. |
| `assembly/detail/` | Shared local form, trace, cache, zeroing, OpenMP, and diagnostic helpers. |
| `system/` | High-level assembly and solve helpers for the main system. |
| `error_fespace/` | Patch scalar/flux spaces and patch functions used by equilibrated estimators. |
| `time_slabs/` | Time-slab construction, copied-slab backend, adaptive Y-loop, flux reconstruction, estimator combination, and slab-specific binary outputs. |
| `io/` | Binary writers for time-slab spaces, functions, cellwise errors, patches, and reconstruction samples. |
| `detail/` | Shared memory/timing/capability helpers. |

## Main Solve Path

The main solve path uses:

| Header or directory | Role |
|---|---|
| `fespace/fespace.hpp` | Finite-element space container. |
| `fespace/dofs/` | DoF entities, handlers, distributions, and physical coordinates. |
| `assembly/main_system/` | Local matrices/vectors and global sparse saddle-system assembly. |
| `assembly/scatter.hpp` | Scatter local contributions into global vectors/matrices. |
| `system/assemble_saddle_problem.hpp` | High-level saddle problem assembly entry point. |
| `system/solve_main_system.hpp` | Main-system solve entry point. |
| `system/main_preconditioner_context.hpp` | Context for preconditioned iterative solves. |

## Error And Estimator Path

The estimator path is centered around time slabs and local patch problems:

| Header or directory | Role |
|---|---|
| `time_slabs/make_time_slab_space.hpp` | Construct a time-slab space from mesh/FE data. |
| `time_slabs/time_slab_space.hpp` | Slab-space representation and source-cell relationships. |
| `time_slabs/time_slab_adaptive_loop.hpp` | Inner Y-space adaptive loop. |
| `time_slabs/time_slab_error_indicators.hpp` | Cellwise error indicator computation. |
| `time_slabs/time_slab_equilibrated_flux_reconstruction.hpp` | Equilibrated flux reconstruction. |
| `time_slabs/time_slab_equilibrated_flux_indicators.hpp` | Flux-based indicator components. |
| `time_slabs/time_slab_estimator_combination.hpp` | Combination of estimator components into runner-facing quantities. |
| `error_fespace/` | Patch spaces and patch functions used by local estimator solves. |
| `assembly/error_system/` | Local patch matrix/vector assembly and performance-oriented 2+1D caches. |

## IO And Plotting Artifacts

Binary outputs used by `../python/` are written through:

| Header | Artifact type |
|---|---|
| `fespace/io/write_mesh_binary.hpp` | Mesh geometry and cell topology. |
| `fespace/io/write_dofs_binary.hpp` | DoF coordinates and associations. |
| `fespace/io/write_function_binary.hpp` | FE function values. |
| `io/write_time_slab_space_binary.hpp` | Time-slab mesh/space state. |
| `io/write_time_slab_function_binary.hpp` | Time-slab function values. |
| `io/write_cellwise_error_binary.hpp` | Cellwise estimator/error values. |
| `io/write_cellwise_error_components_binary.hpp` | Split estimator components. |
| `io/write_time_slab_edge_patches_binary.hpp` | Edge patch geometry/data. |
| `io/write_time_slab_vertex_patches_binary.hpp` | Vertex patch geometry/data. |
| `io/write_time_slab_flux_reconstruction_samples_binary.hpp` | Flux reconstruction samples. |

Format-version constants live in `io/detail/` and should be updated whenever a
binary layout changes.

## Development Notes

- The 1+1D and 2+1D paths share high-level names but often use specialized
  local assembly and topology code. Check the `_1d` and `_2d` headers before
  generalizing behavior.
- Time-slab code keeps links back to source mesh cells so estimator reductions
  can mark the correct X/Y cells.
- Several runner performance flags map directly to implementation choices in
  `time_slabs/` and `assembly/error_system/`, including local-error cache
  modes, patch solvers, slab reconstruction threading, and fused diagnostics.

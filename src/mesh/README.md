# Mesh

`mesh/` implements the adaptive space-time mesh layer. It stores cells,
tracks parent/child refinement structure, exposes topology queries, and applies
dimension-specific refinement rules used by the adaptive algorithm.

## Files And Directories

| Path | Purpose |
|---|---|
| `cell.hpp` | Cell record and geometric/refinement metadata. |
| `mesh.hpp` | Main mesh container, active-cell access, and refinement ownership. |
| `mesh_traits.hpp` | Dimension-specific mesh traits. |
| `types.hpp` | Shared mesh type aliases and basic enums. |
| `detail/vertex_registry.hpp` | Internal vertex registry used to reuse and identify vertices during mesh construction/refinement. |
| `initialization/root_cell.hpp` | Root-cell construction helpers. |
| `refinement/refine.hpp` | Dimension-dispatch entry point for refinement. |
| `refinement/refine_1d.hpp` | 1+1D refinement implementation. |
| `refinement/refine_2d.hpp` | 2+1D refinement implementation and conformity handling. |
| `refinement/refinement_type.hpp` | Refinement type definitions. |
| `refinement/split_policy.hpp` | Policy decisions for split selection. |
| `refinement/time_slicing/` | Helpers for slicing cells at time interfaces. |
| `topology/` | Boundary, face, orientation, adjacency, interval, and active-leaf traversal utilities. |

## Refinement And Topology

The adaptive algorithm marks source cells through error indicators. The mesh
layer then converts those marks into dimension-specific refinements while
preserving the conformity assumptions required by the finite-element spaces.

Important topology helpers:

| Header | Responsibility |
|---|---|
| `topology/leaf_traversal.hpp` | Traverse active leaves of the refinement tree. |
| `topology/faces.hpp` | Dimension-dispatched face queries. |
| `topology/boundary.hpp` | Dimension-dispatched boundary queries. |
| `topology/orientation.hpp` | Dimension-dispatched orientation helpers. |
| `topology/spatial_edge_adjacency_2d.hpp` | 2D spatial edge adjacency used by conformity/refinement logic. |
| `topology/temporal_keys.hpp` | Keys for time intervals and temporal interfaces. |
| `topology/interval_relations.hpp` | Interval overlap and containment utilities. |

## Development Notes

- Refinement must preserve stable cell identity and parent/child relations
  because DoF distribution, estimator reduction, and output all refer back to
  mesh cells.
- The 2+1D path has extra closure logic for active spatial conformity. Runner
  flags such as `--refinement-edge-query-cache`,
  `--post-flush-closure-mode`, and `--refinement-main-closure-query-mode`
  expose some of those implementation choices for production/debug runs.
- Time-slicing helpers are used when temporal interfaces need to cut through
  spatial cells during time-slab construction or closure.

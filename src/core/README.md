# Core Utilities

`core/` contains small shared building blocks used throughout the mesh,
finite-element, and linear-algebra layers. These headers should stay lightweight
and dependency-poor.

## Files

| File | Purpose |
|---|---|
| `ids.hpp` | Strongly typed ids for cells, vertices, DoFs, and related entities. |
| `coord_key.hpp` | Hashable coordinate keys used when identifying or deduplicating geometric points. |
| `hash.hpp` | Hash-combine helpers and hashing support for composite keys. |
| `exceptions.hpp` | Project-specific exception helpers. |
| `debug.hpp` | Debug assertions and diagnostic checks used by internal algorithms. |
| `openmp.hpp` | Small OpenMP abstraction helpers used when OpenMP is available or disabled. |

## Usage Notes

- Prefer the typed ids from `ids.hpp` over raw integers when an object has
  stable identity across refinement, assembly, or output.
- Keep hashing and key definitions deterministic. The adaptive estimator and
  refinement paths rely on reproducible reductions and stable lookup behavior.
- Put only cross-cutting utilities here. Mesh, finite-element, and linear
  algebra concepts belong in their respective subsystem directories.

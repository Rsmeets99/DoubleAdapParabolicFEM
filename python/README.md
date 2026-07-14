# Python Mesh Plotting Helpers

This directory contains interactive and headless plotting helpers for mesh and
DoF artifacts written by the C++ code. The user-facing entry points are:

- `mesh_plot.py`: plots 1+1D space-time meshes.
- `mesh_2d_plot.py`: plots 2+1D triangular-prism space-time meshes.

The remaining Python files are local helper modules for binary parsing,
geometry construction, display controls, and shared command-line behavior.

## Setup

Run the setup commands from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -r requirements-dev.txt
```

The plotting scripts use Matplotlib. For non-interactive environments, pass
`--no-show` and optionally `--save-png`.

## 1+1D Mesh Plotter

Plot explicit mesh and DoF files:

```bash
.venv/bin/python3 python/mesh_plot.py \
  --mesh path/to/mesh.bin \
  --dofs path/to/dofs.bin
```

If `--mesh` is omitted, the script looks in the repository data directory for
default generator artifacts:

- `data/mesh_test_spacetime.bin` and `data/dofs_test_spacetime.bin` when
  `--policy spacetime` is used.
- `data/mesh_test_spaceonly.bin` and `data/dofs_test_spaceonly.bin` when
  `--policy spaceonly` is used.

Common 1+1D flags:

- `--mesh PATH`: mesh binary to read.
- `--dofs PATH`: matching DoF binary to read.
- `--policy spacetime|spaceonly`: default artifact pair to use when `--mesh`
  is omitted. Default: `spacetime`.
- `--no-dofs`: read and plot only the mesh.
- `--dof-labels`: label DoFs on the selected cell.
- `--mesh-only` or `--static-mesh`: draw a static mesh without selection,
  neighbour navigation, or DoF overlays.
- `--hide-axes` or `--no-axes`: hide axis decorations and titles.
- `--tight-frame`: remove subplot padding around the mesh.
- `--uniform-color` or `--no-cell-colors`: draw cells with one mesh color.
- `--cell-colors` or `--viridis-cell-colors`: color cells by row index.
- `--no-show`: construct figures and exit without opening a GUI window.
- `--save-png PATH`: save open figures as PNG files.

Example headless export:

```bash
.venv/bin/python3 python/mesh_plot.py \
  --mesh path/to/mesh.bin \
  --dofs path/to/dofs.bin \
  --mesh-only \
  --no-show \
  --save-png figures/mesh_1d.png
```

## 2+1D Mesh Plotter

Plot explicit mesh and DoF files:

```bash
.venv/bin/python3 python/mesh_2d_plot.py \
  --mesh path/to/mesh_2d.bin \
  --dofs path/to/dofs_2d.bin
```

If `--mesh` is omitted, the script looks in the repository data directory for
default generator artifacts:

- `data/mesh_2d_spaceonly.bin` and `data/dofs_2d_spaceonly.bin` when
  `--policy spaceonly` is used.
- `data/mesh_2d_spacetime.bin` and `data/dofs_2d_spacetime.bin` when
  `--policy spacetime` is used.

Common 2+1D flags:

- `--mesh PATH`: mesh binary to read.
- `--dofs PATH`: matching DoF binary to read.
- `--policy spaceonly|spacetime`: default artifact pair to use when `--mesh`
  is omitted. Default: `spaceonly`.
- `--cell N`: initial selected cell id or row.
- `--no-dofs`: do not plot DoFs.
- `--dof-labels`: label DoFs on the selected cell.
- `--left-only` or `--overview-only`: show only the full-mesh overview panel.
- `--hide-axes` or `--no-axes`: hide axis decorations and titles.
- `--uniform-color` or `--no-cell-colors`: draw cells with one mesh color.
- `--view-preset default|top|bottom`: choose the initial 3D view. The `bottom`
  preset views the mesh from the `t=0` side.
- `--view-elev VALUE`: override the Matplotlib 3D elevation angle in degrees.
- `--view-azim VALUE`: override the Matplotlib 3D azimuth angle in degrees.
- `--view-roll VALUE`: override the Matplotlib 3D roll angle in degrees.
- `--self-test`: construct, navigate, redraw, and close the viewer without
  showing it.
- `--no-show`: construct figures and exit without opening a GUI window.
- `--save-png PATH`: save open figures as PNG files.

Example headless export:

```bash
.venv/bin/python3 python/mesh_2d_plot.py \
  --mesh path/to/mesh_2d.bin \
  --dofs path/to/dofs_2d.bin \
  --view-preset bottom \
  --overview-only \
  --no-show \
  --save-png figures/mesh_2d.png
```

## PNG Output Rules

`--save-png` expects a `.png` file path or a directory-like path without a
suffix. If one figure is open and a `.png` path is given, that exact path is
used. If multiple figures are open, a `.png` path is treated as a filename
prefix and the script writes numbered files such as `mesh_00.png` and
`mesh_01.png`. If the path has no suffix, figures are written as
`figure_00.png`, `figure_01.png`, and so on inside that directory.

## Troubleshooting

- Missing default artifacts mean the relevant C++ generator output is not
  present under `data/`. Pass explicit `--mesh` and `--dofs` paths, or generate
  the binary artifacts first.
- Use `--no-show --save-png ...` on headless systems.
- Use `--self-test` with `mesh_2d_plot.py` to verify that the 2+1D viewer can
  construct and redraw a figure without opening a window.

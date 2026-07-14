# AdapParabolicFEM

This repository is a publication-oriented snapshot of the adaptive parabolic
finite-element code used for the accompanying paper. It contains the numerical
source code, runnable example configurations, plotting helpers, and cleaned
plot data needed to inspect and reproduce the reported computations.

The snapshot is intentionally focused on the production executable and the
artifacts needed for the paper. Tests, benchmarks, raw production outputs,
generated production configs, scheduler logs, and internal profiling notes are
not included.

## Development Environment

This code was developed and written on Windows using WSL2 with an Ubuntu
distribution. The local setup commands below therefore assume an Ubuntu/Debian
shell. Other Linux distributions should work with equivalent packages, but the
package names and compiler/OpenMP runtime packages may differ.

## Contents

- `src/`: mesh, finite-element, quadrature, assembly, and linear-algebra code.
  See `src/README.md`.
- `algorithm/`: adaptive driver, example configurations, output writers, and
  the `run_adaptive_algorithm` executable entry point. See
  `algorithm/README.md`.
- `algorithm_data/`: default location for generated run output. See
  `algorithm_data/README.md`.
- `algorithm_data_plot/`: cleaned CSV files prepared from completed production
  runs for plotting.
- `python/`: 1+1D and 2+1D mesh plotting helpers. See `python/README.md`.
- `scripts/`: operational helper scripts.
- `CMakeLists.txt`, `CMakePresets.json`, and `cmake/`: build configuration.

## Requirements And Installation

The C++ code is header-only apart from the production executable
`run_adaptive_algorithm`. A local non-MKL build is enough for development,
small checks, and reading the examples. The Snellius production presets use
Intel oneMKL/PARDISO and are documented separately in `README_SNELLIUS.md`.

### System Packages

On Ubuntu or Debian, install the compiler, CMake, Eigen, and Python virtualenv
support with:

```bash
sudo apt update
sudo apt install \
  build-essential \
  clang \
  cmake \
  libomp-dev \
  libeigen3-dev \
  python3 \
  python3-venv \
  python3-pip \
  python3-tk
```

Official installation references:

| Component | Reference |
|---|---|
| CMake | [CMake downloads](https://cmake.org/download/) |
| Eigen | [Eigen getting started](https://eigen.tuxfamily.org/dox/GettingStarted.html) |
| GCC | [GCC installation](https://gcc.gnu.org/install/) |
| Clang | [Clang getting started](https://clang.llvm.org/get_started.html) |
| OpenMP | [OpenMP compilers and tools](https://www.openmp.org/resources/openmp-compilers-tools/) |
| Intel oneMKL | [Get started with Intel oneAPI Math Kernel Library](https://www.intel.com/content/www/us/en/docs/onemkl/get-started-guide/current/overview.html) |
| Python 3 | [Python downloads](https://www.python.org/downloads/) |
| WSL2 with Ubuntu | [Install Ubuntu on WSL 2](https://documentation.ubuntu.com/wsl/latest/howto/install-ubuntu-wsl2/) |

The project requires:

| Requirement | Purpose | Notes |
|---|---|---|
| CMake 3.16 or newer | Configure and build the executable | Direct `cmake -S ... -B ...` commands use the project minimum. The checked-in presets require a CMake version with `CMakePresets.json` version 3 support. |
| C++20 compiler | Compile `run_adaptive_algorithm` | GCC and Clang builds are supported through standard CMake compiler discovery. |
| Eigen 3.4 | Dense/sparse linear algebra backend | CMake first tries `find_package(Eigen3)`, then common include paths, then downloads Eigen 3.4.0 when `ADAPPARABOLICFEM_FETCH_EIGEN=ON`. |
| OpenMP | Optional shared-memory parallelism | Enabled when `ADAPPARABOLICFEM_USE_OPENMP=ON` and CMake finds OpenMP. The code also builds without OpenMP. |
| Intel oneMKL | Optional PARDISO direct solvers | Required only when configuring with `ENABLE_MKL_PARDISO=ON` or using the Snellius production presets. |
| Python 3 | Plotting and Snellius data scripts | Python packages are listed in `requirements-dev.txt`. |

If you do not have system Eigen, omit `libeigen3-dev` and keep
`-DADAPPARABOLICFEM_FETCH_EIGEN=ON` in the configure command below. That
fallback downloads Eigen during CMake configure, so it needs network access.

### Build Locally Without MKL

Run these commands from the repository root:

```bash
cmake -S . -B out/build/local-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MKL_PARDISO=OFF \
  -DADAPPARABOLICFEM_FETCH_EIGEN=ON

cmake --build out/build/local-release \
  --target run_adaptive_algorithm \
  -j "$(nproc)"
```

The executable is then:

```bash
out/build/local-release/run_adaptive_algorithm
```

Check that the runner starts and sees the registered examples:

```bash
out/build/local-release/run_adaptive_algorithm --list-examples
```

Run a small checked-in configuration:

```bash
out/build/local-release/run_adaptive_algorithm \
  --config algorithm/examples/space_time_1d/smooth_initial.yml
```

For a smoke check that does not export run artifacts:

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

### Build With oneMKL/PARDISO

Use MKL only when you need the PARDISO solver path. Install Intel oneAPI oneMKL
or load it through your cluster module system, then point CMake at the MKL
installation if package discovery does not find it:

```bash
cmake -S . -B out/build/local-mkl-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_MKL_PARDISO=ON \
  -DMKL_ROOT=/opt/intel/oneapi/mkl/latest

cmake --build out/build/local-mkl-release \
  --target run_adaptive_algorithm \
  -j "$(nproc)"
```

The Snellius production presets are MKL/PARDISO builds:

```bash
cmake --preset release-snellius-generic-mkl-pardiso
cmake --build --preset build-release-snellius-generic-mkl-pardiso \
  --target run_adaptive_algorithm
```

For a full cluster workflow, including module loading, production config
generation, Slurm submission, collection, and summary scripts, use
`README_SNELLIUS.md`.

### Python Virtual Environment

The Python helpers are used for mesh plotting and for preparing/ordering
Snellius result data. Create the environment from the repository root:

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install --upgrade pip
.venv/bin/python3 -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` currently installs:

| Package | Version range | Used for |
|---|---|---|
| `numpy` | `>=1.26,<3` | Binary readers and numerical plot preparation. |
| `matplotlib` | `>=3.8,<4` | Interactive and headless mesh plots. |

For interactive Matplotlib windows on Ubuntu or WSL, install `python3-tk` as
shown in the system-package command. On headless systems, pass `--no-show` and
optionally `--save-png` to the plotting scripts.

## Repository Guides

- `src/README.md`: source-tree architecture and module responsibilities.
- `algorithm/README.md`: how the executable, examples, and output files are
  organized.
- `algorithm_data/README.md`: where generated run output is written and what
  should be kept out of the repository.
- `algorithm_data_plot/README.md`: format of the checked-in cleaned plot CSVs.
- `python/README.md`: mesh plotting setup, commands, and flags.
- `README_SNELLIUS.md`: Snellius build, submission, collection, and summary
  workflow.

## Cluster Runs

For Snellius or a similar Slurm cluster, do not start from the local build
commands above. Use `README_SNELLIUS.md`, which documents the production module
stack, MKL/PARDISO presets, generated configuration matrix, thread policy,
submission helpers, and collection workflow.

## Minimal Local Check

After building the executable, list the registered examples:

```bash
./path/to/run_adaptive_algorithm --list-examples
```

Run one checked-in configuration:

```bash
./path/to/run_adaptive_algorithm \
  --config algorithm/examples/space_time_2d/smooth_initial.yml
```

The output directory is read from the config file. The example above writes to
`algorithm_data/space_time_2d/smooth_initial_config_run`.

## Plotting

The plotting helpers read binary mesh and DoF artifacts generated by the code:

```bash
python3 python/mesh_plot.py --mesh path/to/mesh.bin --dofs path/to/dofs.bin
python3 python/mesh_2d_plot.py --mesh path/to/mesh_2d.bin --dofs path/to/dofs_2d.bin
```

See `python/README.md` for setup instructions and plotting flags.

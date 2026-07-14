#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/snellius/build_snellius.sh [options]

Configure and build the Snellius production executable.

Options:
  --preset NAME         CMake configure preset
                        (default: release-snellius-generic-mkl-pardiso)
  --build-preset NAME   CMake build preset
                        (default: build-release-snellius-generic-mkl-pardiso)
  --target NAME         Build target (default: run_adaptive_algorithm)
  --jobs N              Parallel build jobs (default: SLURM_CPUS_PER_TASK or nproc)
  --load-modules        Purge/load the default Snellius build module stack
                        before configuring
  --module-set NAME     Module stack for --load-modules (default: 2025a)
                        Supported: 2025a
  --help                Show this help text

By default this script uses the modules already loaded in the shell.  Use
--load-modules for the verified Snellius 2025a GCC/oneMKL/Eigen/CMake stack.
USAGE
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/../.." && pwd)"

preset="release-snellius-generic-mkl-pardiso"
build_preset="build-release-snellius-generic-mkl-pardiso"
target="run_adaptive_algorithm"
load_modules="0"
module_set="2025a"
jobs="${SLURM_CPUS_PER_TASK:-}"
if [[ -z "${jobs}" ]]; then
    jobs="$(nproc 2>/dev/null || echo 8)"
fi

ensure_module_command() {
    if type module >/dev/null 2>&1; then
        return 0
    fi

    for init_script in \
        /etc/profile.d/lmod.sh \
        /usr/share/lmod/lmod/init/bash \
        /usr/share/Modules/init/bash; do
        if [[ -r "${init_script}" ]]; then
            # shellcheck source=/dev/null
            source "${init_script}"
            break
        fi
    done

    type module >/dev/null 2>&1
}

load_snellius_modules() {
    ensure_module_command || {
        echo "Could not initialize the environment modules command." >&2
        echo "Load modules manually, or run without --load-modules." >&2
        exit 1
    }

    local modules=()
    case "${module_set}" in
        2025a)
            modules=(
                2025
                intel/2025a
                CMake/3.31.3-GCCcore-14.2.0
                imkl/2025.1.0
                Eigen/3.4.0-GCCcore-14.2.0
            )
            ;;
        *)
            echo "Unknown module set: ${module_set}" >&2
            echo "Supported module sets: 2025a" >&2
            exit 2
            ;;
    esac

    echo "Purging modules and loading Snellius module set: ${module_set}"
    module purge
    module load "${modules[@]}"
    module list
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --preset)
            preset="${2:?missing value for --preset}"
            shift 2
            ;;
        --build-preset)
            build_preset="${2:?missing value for --build-preset}"
            shift 2
            ;;
        --target)
            target="${2:?missing value for --target}"
            shift 2
            ;;
        --jobs)
            jobs="${2:?missing value for --jobs}"
            shift 2
            ;;
        --load-modules)
            load_modules="1"
            shift
            ;;
        --module-set)
            module_set="${2:?missing value for --module-set}"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "${project_root}"

if [[ "${load_modules}" == "1" ]]; then
    load_snellius_modules
fi

echo "Configuring with CMake preset: ${preset}"
cmake --preset "${preset}"

echo "Building target ${target} with build preset: ${build_preset}"
cmake --build --preset "${build_preset}" --target "${target}" --parallel "${jobs}"

executable="out/build/${preset}/${target}"
if [[ "${target}" == "run_adaptive_algorithm" ]]; then
    if [[ ! -x "${executable}" ]]; then
        echo "Expected executable not found: ${executable}" >&2
        exit 1
    fi
    echo "Executable ready: ${executable}"
else
    echo "Build completed for target ${target}."
fi

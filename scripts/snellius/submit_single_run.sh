#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: scripts/snellius/submit_single_run.sh --config CONFIG.yml [options]

Submit one run_adaptive_algorithm config as a Snellius SLURM job.

Options:
  --config PATH          YAML config to run (required)
  --partition NAME       rome|genoa|fat_rome|fat_genoa|himem_4tb|himem_8tb
                         (default: rome)
  --time LIMIT           SLURM time limit (default: 120:00:00)
  --memory-mb MB         SLURM memory request in MiB. Use auto to read config.
                         (default: auto, fallback: 12000)
  --cpus N               SLURM cpus-per-task (default: 16)
  --omp-threads N        OMP_NUM_THREADS for the job (default: cpus)
  --mkl-threads N        MKL_NUM_THREADS for the job (default: OMP threads)
  --job-name NAME        SLURM job name (default: derived from config path)
  --account NAME         Optional SLURM account
  --qos NAME             Optional SLURM QoS
  --dependency SPEC      Optional SLURM dependency, e.g. afterok:12345
  --executable PATH      run_adaptive_algorithm executable
                         (default: out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm)
  --project-root PATH    Project root (default: repository root)
  --logs-dir PATH        SLURM log base directory (default: snellius_logs)
  --sbatch-dir PATH      Generated sbatch base directory (default: .snellius_jobs)
  --run-id ID            Explicit run directory id under output/runs/
                         (default: submit timestamp plus SLURM job id)
  --use-scratch          Run on local scratch and copy back on exit
                         (default: write directly to final output)
  --load-modules         Load the default Snellius runtime module stack inside
                         the generated sbatch job before running the executable
  --module-set NAME      Module stack for --load-modules (default: 2025a)
                         Supported: 2025a
  --dry-run              Write the sbatch file but do not call sbatch
  --help                 Show this help text

By default the job writes directly to a unique output/runs/<run-id>/ directory
to avoid overwriting previous data. Use --use-scratch when local scratch is
needed; the job then copies partial/final outputs back on exit. SLURM logs and
generated sbatch files are likewise grouped under unique per-submission
subdirectories of --logs-dir and --sbatch-dir.
USAGE
}

die() {
    echo "error: $*" >&2
    exit 2
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
project_root="$(cd "${script_dir}/../.." && pwd -P)"

config=""
partition="rome"
time_limit="120:00:00"
memory_mb="auto"
cpus="16"
omp_threads=""
mkl_threads=""
job_name=""
account=""
qos=""
dependency=""
executable="out/build/release-snellius-generic-mkl-pardiso/run_adaptive_algorithm"
logs_dir="snellius_logs"
sbatch_dir=".snellius_jobs"
run_id=""
use_scratch="0"
dry_run="0"
load_modules="0"
module_set="2025a"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config="${2:?missing value for --config}"
            shift 2
            ;;
        --partition)
            partition="${2:?missing value for --partition}"
            shift 2
            ;;
        --time)
            time_limit="${2:?missing value for --time}"
            shift 2
            ;;
        --memory-mb)
            memory_mb="${2:?missing value for --memory-mb}"
            shift 2
            ;;
        --cpus)
            cpus="${2:?missing value for --cpus}"
            shift 2
            ;;
        --omp-threads)
            omp_threads="${2:?missing value for --omp-threads}"
            shift 2
            ;;
        --mkl-threads)
            mkl_threads="${2:?missing value for --mkl-threads}"
            shift 2
            ;;
        --job-name)
            job_name="${2:?missing value for --job-name}"
            shift 2
            ;;
        --account)
            account="${2:?missing value for --account}"
            shift 2
            ;;
        --qos)
            qos="${2:?missing value for --qos}"
            shift 2
            ;;
        --dependency)
            dependency="${2:?missing value for --dependency}"
            shift 2
            ;;
        --executable)
            executable="${2:?missing value for --executable}"
            shift 2
            ;;
        --project-root)
            project_root="${2:?missing value for --project-root}"
            shift 2
            ;;
        --logs-dir)
            logs_dir="${2:?missing value for --logs-dir}"
            shift 2
            ;;
        --sbatch-dir)
            sbatch_dir="${2:?missing value for --sbatch-dir}"
            shift 2
            ;;
        --run-id)
            run_id="${2:?missing value for --run-id}"
            shift 2
            ;;
        --use-scratch)
            use_scratch="1"
            shift
            ;;
        --load-modules)
            load_modules="1"
            shift
            ;;
        --module-set)
            module_set="${2:?missing value for --module-set}"
            shift 2
            ;;
        --dry-run)
            dry_run="1"
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            die "unknown option: $1"
            ;;
    esac
done

[[ -n "${config}" ]] || die "--config is required"

case "${partition}" in
    rome|genoa|fat_rome|fat_genoa|himem_4tb|himem_8tb) ;;
    *) die "unsupported partition: ${partition}" ;;
esac

case "${module_set}" in
    2025a) ;;
    *) die "unsupported module set: ${module_set}" ;;
esac

if [[ -n "${run_id}" && ! "${run_id}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    die "--run-id may only contain letters, digits, '.', '_', and '-'"
fi

abs_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        local dir
        dir="$(dirname "${path}")"
        local base
        base="$(basename "${path}")"
        printf '%s/%s\n' "$(cd "${dir}" && pwd -P)" "${base}"
    fi
}

read_config_value() {
    local file="$1"
    local key="$2"
    awk -F: -v key="${key}" '
        $1 == key {
            sub(/^[ \t]+/, "", $2)
            sub(/[ \t\r]+$/, "", $2)
            gsub(/^"|"$/, "", $2)
            print $2
            exit
        }
    ' "${file}"
}

project_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s/%s\n' "${project_root}" "${path}"
    fi
}

is_positive_number() {
    awk -v value="$1" 'BEGIN { exit !(value + 0 > 0) }'
}

project_root="$(cd "${project_root}" && pwd -P)"
config="$(abs_path "${config}")"
executable="$(abs_path "${executable}")"
logs_dir="$(project_path "${logs_dir}")"
sbatch_dir="$(project_path "${sbatch_dir}")"

[[ -f "${config}" ]] || die "config not found: ${config}"
[[ -x "${executable}" ]] || die "executable not found or not executable: ${executable}"

config_output="$(read_config_value "${config}" output || true)"
[[ -n "${config_output}" ]] || die "config has no output: key"
if [[ "${config_output}" = /* ]]; then
    final_output="${config_output}"
else
    final_output="${project_root}/${config_output}"
fi

if [[ "${memory_mb}" == "auto" ]]; then
    memory_mb=""
    for key in memory_limit_mb main_solver_memory_limit_mb; do
        value="$(read_config_value "${config}" "${key}" || true)"
        if [[ -n "${value}" ]] && is_positive_number "${value}"; then
            value_int="$(awk -v value="${value}" 'BEGIN { printf "%d", value }')"
            if [[ -z "${memory_mb}" || "${value_int}" -gt "${memory_mb}" ]]; then
                memory_mb="${value_int}"
            fi
        fi
    done
    if [[ -z "${memory_mb}" ]]; then
        memory_mb="12000"
    fi
fi

case_id="$(python3 - "${config}" "${project_root}" <<'PY'
from pathlib import Path
import sys
config = Path(sys.argv[1])
root = Path(sys.argv[2])
try:
    rel = config.relative_to(root)
except ValueError:
    rel = config
stem = str(rel.with_suffix(""))
print(stem.replace("/", "_").replace(" ", "_"))
PY
)"

if [[ -z "${job_name}" ]]; then
    job_name="apf_${case_id}"
fi
job_name="${job_name:0:128}"

timestamp="$(python3 - <<'PY'
from datetime import datetime

print(datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
PY
)"
submission_timestamp="${timestamp}"
submission_id="${submission_timestamp}_${case_id}"
run_logs_dir="${logs_dir}/runs/${submission_id}"
run_sbatch_dir="${sbatch_dir}/runs/${submission_id}"
mkdir -p "${run_logs_dir}" "${run_sbatch_dir}"
sbatch_file="${run_sbatch_dir}/${job_name}.sbatch"

quote() {
    printf '%q' "$1"
}

{
    echo "#!/usr/bin/env bash"
    echo "#SBATCH --job-name=${job_name}"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --partition=${partition}"
    echo "#SBATCH --time=${time_limit}"
    echo "#SBATCH --cpus-per-task=${cpus}"
    echo "#SBATCH --mem=${memory_mb}M"
    echo "#SBATCH --output=${run_logs_dir}/%x-%j.out"
    echo "#SBATCH --error=${run_logs_dir}/%x-%j.err"
    echo "#SBATCH --signal=B:TERM@120"
    [[ -n "${account}" ]] && echo "#SBATCH --account=${account}"
    [[ -n "${qos}" ]] && echo "#SBATCH --qos=${qos}"
    [[ -n "${dependency}" ]] && echo "#SBATCH --dependency=${dependency}"
    cat <<EOF
set -euo pipefail

PROJECT_ROOT=$(quote "${project_root}")
CONFIG=$(quote "${config}")
EXECUTABLE=$(quote "${executable}")
BASE_OUTPUT=$(quote "${final_output}")
LOGS_RUN_DIR=$(quote "${run_logs_dir}")
CASE_ID=$(quote "${case_id}")
SUBMISSION_TIMESTAMP=$(quote "${submission_timestamp}")
REQUESTED_RUN_ID=$(quote "${run_id}")
USE_SCRATCH=${use_scratch}
LOAD_MODULES=${load_modules}
MODULE_SET=$(quote "${module_set}")
THREADS="\${SLURM_CPUS_PER_TASK:-${cpus}}"
OMP_THREADS_CONFIG=$(quote "${omp_threads}")
MKL_THREADS_CONFIG=$(quote "${mkl_threads}")
OMP_THREADS="\${OMP_THREADS_CONFIG:-\${THREADS}}"
MKL_THREADS="\${MKL_THREADS_CONFIG:-\${OMP_THREADS}}"
SLURM_JOB_ID_VALUE="\${SLURM_JOB_ID:-manual}"
SLURM_JOB_NAME_VALUE="\${SLURM_JOB_NAME:-${job_name}}"
SLURM_STDOUT="\${LOGS_RUN_DIR}/\${SLURM_JOB_NAME_VALUE}-\${SLURM_JOB_ID_VALUE}.out"
SLURM_STDERR="\${LOGS_RUN_DIR}/\${SLURM_JOB_NAME_VALUE}-\${SLURM_JOB_ID_VALUE}.err"
SLURM_ACCOUNTING_LOG="\${LOGS_RUN_DIR}/\${SLURM_JOB_NAME_VALUE}-\${SLURM_JOB_ID_VALUE}.accounting.txt"

ensure_module_command() {
    if type module >/dev/null 2>&1; then
        return 0
    fi

    for init_script in \\
        /etc/profile.d/lmod.sh \\
        /usr/share/lmod/lmod/init/bash \\
        /usr/share/Modules/init/bash; do
        if [[ -r "\${init_script}" ]]; then
            # shellcheck source=/dev/null
            source "\${init_script}"
            break
        fi
    done

    type module >/dev/null 2>&1
}

load_snellius_modules() {
    ensure_module_command || {
        echo "Could not initialize the environment modules command." >&2
        exit 1
    }

    local modules=()
    case "\${MODULE_SET}" in
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
            echo "Unknown module set: \${MODULE_SET}" >&2
            exit 2
            ;;
    esac

    echo "Purging modules and loading Snellius module set: \${MODULE_SET}"
    module purge
    module load "\${modules[@]}"
}

cd "\${PROJECT_ROOT}"

if [[ -n "\${REQUESTED_RUN_ID}" ]]; then
    RUN_ID="\${REQUESTED_RUN_ID}"
else
    RUN_ID="\${SUBMISSION_TIMESTAMP}_job\${SLURM_JOB_ID:-manual}"
fi
FINAL_OUTPUT="\${BASE_OUTPUT}/runs/\${RUN_ID}"

if [[ "\${USE_SCRATCH}" == "1" ]]; then
    SCRATCH_BASE="\${TMPDIR:-}"
    if [[ -z "\${SCRATCH_BASE}" ]]; then
        SCRATCH_BASE="\${PROJECT_ROOT}/.snellius_scratch"
    fi
    SCRATCH_ROOT="\${SCRATCH_BASE}/adapparabolicfem_\${SLURM_JOB_ID:-manual}_\${CASE_ID}"
    RUN_OUTPUT="\${SCRATCH_ROOT}/output"
else
    SCRATCH_ROOT=""
    RUN_OUTPUT="\${FINAL_OUTPUT}"
fi

mkdir -p "\${RUN_OUTPUT}" "\${FINAL_OUTPUT}"

SRUN_PID=""
COPY_BACK_DONE=0

write_loaded_modules() {
    if type module >/dev/null 2>&1; then
        module list > "\${RUN_OUTPUT}/loaded_modules.txt" 2>&1 || true
        echo "loaded_modules_file=\${FINAL_OUTPUT}/loaded_modules.txt" >> "\${RUN_OUTPUT}/snellius_job_info.txt" || true
    fi
}

write_slurm_accounting_snapshot() {
    local snapshot="\${RUN_OUTPUT}/snellius_accounting_snapshot.txt"
    {
        echo "captured_at=\$(date -Is)"
        echo "note=best-effort snapshot written before the batch script exits"
        echo "job_id=\${SLURM_JOB_ID_VALUE}"
        echo "job_name=\${SLURM_JOB_NAME_VALUE}"
        echo "partition=${partition}"
        echo "requested_memory_mb=${memory_mb}"
        echo "time_limit=${time_limit}"
        echo "stdout=\${SLURM_STDOUT}"
        echo "stderr=\${SLURM_STDERR}"
        echo "work_dir=\${PROJECT_ROOT}"
        echo
        echo "[scontrol show job]"
        if command -v scontrol >/dev/null 2>&1 && [[ "\${SLURM_JOB_ID_VALUE}" != "manual" ]]; then
            scontrol show job "\${SLURM_JOB_ID_VALUE}" || true
        else
            echo "scontrol unavailable or not running under Slurm"
        fi
        echo
        echo "[sacct]"
        if command -v sacct >/dev/null 2>&1 && [[ "\${SLURM_JOB_ID_VALUE}" != "manual" ]]; then
            sacct -j "\${SLURM_JOB_ID_VALUE}" \\
                --format=JobID,JobName%120,Partition,State,ExitCode,Elapsed,CPUTime,TotalCPU,MaxRSS,ReqMem,AllocCPUS,NNodes,NodeList%80 \\
                --parsable2 || true
        else
            echo "sacct unavailable or not running under Slurm"
        fi
    } > "\${snapshot}" 2>&1 || true

    cp "\${snapshot}" "\${SLURM_ACCOUNTING_LOG}" 2>/dev/null || true
    echo "slurm_accounting_snapshot=\${FINAL_OUTPUT}/snellius_accounting_snapshot.txt" >> "\${RUN_OUTPUT}/snellius_job_info.txt" || true
    echo "slurm_accounting_log=\${SLURM_ACCOUNTING_LOG}" >> "\${RUN_OUTPUT}/snellius_job_info.txt" || true
}

copy_back() {
    if [[ "\${COPY_BACK_DONE}" == "1" ]]; then
        return 0
    fi

    echo "copy_back_started_at=\$(date -Is)" >> "\${RUN_OUTPUT}/snellius_job_info.txt" || true

    if [[ "\${USE_SCRATCH}" == "1" && -d "\${RUN_OUTPUT}" ]]; then
        mkdir -p "\${FINAL_OUTPUT}"
        if command -v rsync >/dev/null 2>&1; then
            rsync -a "\${RUN_OUTPUT}/" "\${FINAL_OUTPUT}/"
        else
            cp -a "\${RUN_OUTPUT}/." "\${FINAL_OUTPUT}/"
        fi
        echo "copy_back_completed_at=\$(date -Is)" >> "\${FINAL_OUTPUT}/snellius_job_info.txt" || true
    else
        echo "copy_back_completed_at=\$(date -Is)" >> "\${RUN_OUTPUT}/snellius_job_info.txt" || true
    fi
    COPY_BACK_DONE=1
}

on_exit() {
    local status="\$?"
    copy_back || true
    exit "\${status}"
}

handle_termination() {
    local signal_name="\${1:-TERM}"
    echo "terminated_by=\${signal_name}" >> "\${RUN_OUTPUT}/snellius_job_info.txt" || true
    echo "terminated_at=\$(date -Is)" >> "\${RUN_OUTPUT}/snellius_job_info.txt" || true

    if [[ -n "\${SRUN_PID}" ]]; then
        kill -TERM "\${SRUN_PID}" 2>/dev/null || true
        wait "\${SRUN_PID}" 2>/dev/null || true
        SRUN_PID=""
    fi

    copy_back || true
    case "\${signal_name}" in
        INT) exit 130 ;;
        *) exit 143 ;;
    esac
}

trap on_exit EXIT
trap 'handle_termination TERM' TERM
trap 'handle_termination INT' INT

{
    echo "job_id=\${SLURM_JOB_ID:-manual}"
    echo "job_name=\${SLURM_JOB_NAME:-${job_name}}"
    echo "partition=${partition}"
    echo "time_limit=${time_limit}"
    echo "memory_mb=${memory_mb}"
    echo "threads=\${THREADS}"
    echo "omp_threads=\${OMP_THREADS}"
    echo "mkl_threads=\${MKL_THREADS}"
    echo "run_id=\${RUN_ID}"
    echo "base_output=\${BASE_OUTPUT}"
    echo "use_scratch=\${USE_SCRATCH}"
    echo "load_modules=\${LOAD_MODULES}"
    echo "module_set=\${MODULE_SET}"
    echo "config=\${CONFIG}"
    echo "executable=\${EXECUTABLE}"
    echo "run_output=\${RUN_OUTPUT}"
    echo "final_output=\${FINAL_OUTPUT}"
    echo "slurm_stdout=\${SLURM_STDOUT}"
    echo "slurm_stderr=\${SLURM_STDERR}"
    echo "started_at=\$(date -Is)"
} > "\${RUN_OUTPUT}/snellius_job_info.txt"

if [[ "\${LOAD_MODULES}" == "1" ]]; then
    load_snellius_modules
    write_loaded_modules
fi

export OMP_NUM_THREADS="\${OMP_THREADS}"
export MKL_NUM_THREADS="\${MKL_THREADS}"
export OPENBLAS_NUM_THREADS="\${OMP_THREADS}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export MKL_DYNAMIC=FALSE

run_status=0
srun --ntasks=1 --cpus-per-task="\${THREADS}" \\
    "\${EXECUTABLE}" \\
    --config "\${CONFIG}" \\
    --output "\${RUN_OUTPUT}" &
SRUN_PID="\$!"

wait "\${SRUN_PID}" || run_status="\$?"
SRUN_PID=""

echo "run_exit_code=\${run_status}" >> "\${RUN_OUTPUT}/snellius_job_info.txt"
if [[ "\${run_status}" -eq 0 ]]; then
    echo "finished_at=\$(date -Is)" >> "\${RUN_OUTPUT}/snellius_job_info.txt"
else
    echo "failed_at=\$(date -Is)" >> "\${RUN_OUTPUT}/snellius_job_info.txt"
fi

write_slurm_accounting_snapshot

exit "\${run_status}"
EOF
} > "${sbatch_file}"

chmod +x "${sbatch_file}"

if [[ "${dry_run}" == "1" ]]; then
    echo "Dry run. Wrote ${sbatch_file}"
    echo "Would run: sbatch ${sbatch_file}"
    echo "logs_dir=${run_logs_dir}"
    echo "sbatch_file=${sbatch_file}"
    exit 0
fi

sbatch_output="$(sbatch "${sbatch_file}")"
echo "${sbatch_output}"
echo "logs_dir=${run_logs_dir}"
echo "sbatch_file=${sbatch_file}"

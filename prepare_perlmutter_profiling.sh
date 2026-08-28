#!/usr/bin/env bash

set -euo pipefail

readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SHARDING_DIR="${PROJECT_DIR}/sharding"
readonly PREPARE_LOG="${PWD}/prepare_perlmutter_profiling.log"

exec > >(tee "${PREPARE_LOG}") 2>&1
echo "Preparation log: ${PREPARE_LOG}"
export PS4='+ ${BASH_SOURCE[0]}:${LINENO}: '
set -x

if (( $# != 0 )); then
    echo "Usage: $0" >&2
    exit 2
fi

: "${SCRATCH:?SCRATCH is not set; run this script on Perlmutter}"

readonly DEPOT_DIR="${PERLMUTTER_DEPOT:-${SCRATCH}/GB25/perlmutter-main-depot}"
export JULIA_DEPOT_PATH="${DEPOT_DIR}:"
export XDG_CACHE_HOME="${DEPOT_DIR}/cache"
export JULIA_CUDA_USE_COMPAT=false
export JULIA_CUDA_MEMORY_POOL=none

if [[ "${RESET_JULIA_DEPOT:-false}" == "true" ]]; then
    rm -rf -- "${DEPOT_DIR}"
fi
mkdir -p "${DEPOT_DIR}" "${XDG_CACHE_HOME}"

[[ -f "${PROJECT_DIR}/Project.toml" ]] || {
    echo "Missing GB-25 project: ${PROJECT_DIR}" >&2
    exit 1
}
[[ -f "${SHARDING_DIR}/perlmutter_scaling_test.jl" ]] || {
    echo "Missing Perlmutter generator in ${SHARDING_DIR}" >&2
    exit 1
}

module load cudatoolkit/13.2
module load nccl/2.29.2-cu13

JULIA_BIN="${JULIA_BIN:-$(command -v julia || true)}"
readonly JULIA_BIN
[[ -x "${JULIA_BIN}" ]] || {
    echo "julia not found; load Julia or set JULIA_BIN" >&2
    exit 1
}

NSYS_BIN="${NSYS_BIN:-$(command -v nsys || true)}"
if [[ -z "${NSYS_BIN}" ]]; then
    for nsys_candidate in /opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin/nsys; do
        [[ -x "${nsys_candidate}" ]] && NSYS_BIN="${nsys_candidate}"
    done
fi
readonly NSYS_BIN
[[ -x "${NSYS_BIN}" ]] || {
    echo "nsys not found; set NSYS_BIN to an Nsight Systems binary" >&2
    exit 1
}

echo "Julia: ${JULIA_BIN}"
"${JULIA_BIN}" --version
echo "Nsight Systems: ${NSYS_BIN}"
"${NSYS_BIN}" --version
echo "Loaded modules:"
module -t list 2>&1
echo "CUDA compiler:"
nvcc --version | tail -n 1

nsys_help_file="$(mktemp)"
"${NSYS_BIN}" profile --help > "${nsys_help_file}" 2>&1 || true
for nsys_flag in --cuda-graph-trace --capture-range-end; do
    grep -q -- "${nsys_flag}" "${nsys_help_file}" || {
        echo "nsys at ${NSYS_BIN} lacks ${nsys_flag}" >&2
        rm -f -- "${nsys_help_file}"
        exit 1
    }
done
rm -f -- "${nsys_help_file}"

"${JULIA_BIN}" --startup-file=no --project="${PROJECT_DIR}" -O0 \
    -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false)'
"${JULIA_BIN}" --startup-file=no --project="${PROJECT_DIR}" -O0 \
    -e 'using CUDA; CUDA.set_runtime_version!(v"13.2"; local_toolkit=true)'
"${JULIA_BIN}" --startup-file=no --project="${PROJECT_DIR}" -O0 \
    -e 'using Pkg; Pkg.precompile()'
"${JULIA_BIN}" --startup-file=no --project="${PROJECT_DIR}" \
    --compiled-modules=strict -O0 -e 'using GordonBell25'

OPENSSL_LIB="${GB25_OPENSSL_LIB:-$(find "${DEPOT_DIR}/artifacts" -maxdepth 3 \
    -name 'libcrypto.so.3' -print -quit 2>/dev/null || true)}"
readonly OPENSSL_LIB
[[ -f "${OPENSSL_LIB}" ]] || {
    echo "Could not locate the depot's libcrypto.so.3; set GB25_OPENSSL_LIB" >&2
    exit 1
}

cd "${SHARDING_DIR}"
generator_output=$("${JULIA_BIN}" --startup-file=no --project="${PROJECT_DIR}" -O0 \
    perlmutter_scaling_test.jl sharded_baroclinic_instability_simulation_run.jl 2>&1)
printf '%s\n' "${generator_output}"

run_dir=$(printf '%s\n' "${generator_output}" |
    sed -n 's/^.*Writing all output to: //p' | tail -n 1)
[[ -n "${run_dir}" ]] || {
    echo "Could not determine generated run directory" >&2
    exit 1
}

write_profile_submit() {
    local baseline="$1"
    local mode="$2"
    local destination="${baseline%.sh}_${mode}.sh"
    local temporary="${destination}.tmp"
    local output_dir="${baseline%/*}"

    awk -v mode="${mode}" -v output_dir="${output_dir}" \
        -v nsys_bin="${NSYS_BIN}" -v openssl_lib="${OPENSSL_LIB}" '
        function emit_profiler() {
            if (mode == "nsys") {
                print "    " nsys_bin " profile --trace=cuda,nvtx --sample=none \\"
                print "        --capture-range=cudaProfilerApi --capture-range-end=stop \\"
                print "        --cuda-graph-trace=node \\"
                print "        --output=" output_dir "/nsys-job-%q{SLURM_JOB_ID}-rank-%q{SLURM_PROCID} \\"
                print "    env GB25_NSYS=true JULIA_CUDA_NSYS=" nsys_bin " \\"
                print "        LD_PRELOAD=" openssl_lib ":${LD_PRELOAD} \\"
            } else if (mode == "xprof") {
                print "    env GB25_XPROF=true \\"
            }
        }
        !injected && /^[[:space:]]*srun/ { in_srun = 1 }
        !injected && in_srun && /launcher\.sh/ {
            end = index($0, "launcher.sh") + length("launcher.sh") - 1
            tail = substr($0, end + 1)
            if (tail ~ /^[[:space:]]*\\[[:space:]]*$/) {
                print
                emit_profiler()
            } else {
                sub(/^[[:space:]]+/, "", tail)
                print substr($0, 1, end) " \\"
                emit_profiler()
                print "    " tail
            }
            injected = 1
            next
        }
        { print }
        END { if (!injected) exit 1 }
    ' "${baseline}" > "${temporary}"
    chmod --reference="${baseline}" "${temporary}"
    mv -f -- "${temporary}" "${destination}"
}

for baseline in "${run_dir}"/ngpu=*/submit.sh; do
    write_profile_submit "${baseline}" nsys
    write_profile_submit "${baseline}" xprof
done

for generated_submit in "${run_dir}"/ngpu=*/submit*.sh; do
    bash -n "${generated_submit}"
done

echo "Generated profiling scripts:"
for profile_submit in "${run_dir}"/ngpu=*/submit_nsys.sh \
                      "${run_dir}"/ngpu=*/submit_xprof.sh; do
    printf '%s\n' "${profile_submit}"
done

echo "No jobs were submitted. Submit each profile separately when ready:"
for profile_submit in "${run_dir}"/ngpu=*/submit_nsys.sh \
                      "${run_dir}"/ngpu=*/submit_xprof.sh; do
    printf 'sbatch %q\n' "${profile_submit}"
done

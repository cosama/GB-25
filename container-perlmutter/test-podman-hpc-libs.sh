#!/usr/bin/env bash
set -euo pipefail

image="docker.io/nersc/base_gpu:26.06"
account="m5176"
script_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "No Slurm allocation detected; requesting one GPU node."
  exec salloc -A "$account" -C gpu -q interactive -N 1 -t 00:10:00 --gpus-per-node=1 bash "$script_path"
fi

log_file="podman-hpc-libs-$(date -u +%Y%m%dT%H%M%SZ).log"

exec > >(tee "$log_file") 2>&1

echo "Writing log to $log_file"
echo "Image: $image"
echo "Slurm job: $SLURM_JOB_ID"
echo

if ! command -v podman-hpc >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
  module load podman-hpc || true
fi

probe="$(cat <<'IN_CONTAINER'
set -uo pipefail

echo "== OS =="
cat /etc/os-release 2>/dev/null || true

echo
echo "== Devices =="
ls -l /dev/nvidia* 2>/dev/null || echo "no /dev/nvidia* devices"

echo
echo "== Relevant Environment =="
env | sort | grep -E '^(CUDA|NCCL|MPI|MPICH|PMI|PMIX|CRAY|FI_|LD_LIBRARY_PATH|LIBRARY_PATH|LD_PRELOAD|PATH)=' || true

echo
echo "== Libraries =="
ld_paths="${LD_LIBRARY_PATH:-}"
for lib in \
  'libcuda.so*' \
  'libcudart.so*' \
  'libnvidia-ml.so*' \
  'libnccl.so*' \
  'libnccl-net.so*' \
  'libnccl-net-ofi.so*' \
  'libmpi.so*' \
  'libmpich.so*' \
  'libmpi_cray.so*' \
  'libmpifort.so*' \
  'libmpicxx.so*' \
  'libpmi*.so*' \
  'libpmix*.so*' \
  'libfabric.so*' \
  'libcxi.so*'
do
  echo "-- $lib"
  find -L ${ld_paths//:/ } /usr/lib64 /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/cuda \
    -name "$lib" 2>/dev/null | sort -u | sed -n '1,40p'
done

echo
echo "== Linker Cache =="
ldconfig -p 2>/dev/null | grep -E 'lib(cuda|cudart|nvidia-ml|nccl|mpi|mpich|pmi|pmix|fabric|cxi|mpifort|mpicxx)\.so|libmpi_cray\.so' || true

echo
echo "== Injected MPI Dependency Checks =="
for lib in \
  /opt/udiImage/modules/mpich/libmpi.so* \
  /usr/lib/libmpi.so*
do
  [ -e "$lib" ] || continue
  echo "-- $lib"
  ldd "$lib" 2>&1 | sed -n '1,80p'
done

echo
echo "== Commands =="
for exe in nvidia-smi mpiexec mpirun mpichversion ompi_info; do
  if command -v "$exe" >/dev/null 2>&1; then
    echo "-- $exe: $(command -v "$exe")"
    "$exe" --version 2>&1 | sed -n '1,8p' || true
  fi
done
IN_CONTAINER
)"

run_probe() {
  local label="$1"
  local nccl_flag="$2"

  echo
  echo "################################################################################"
  echo "Probe: $label"
  echo "Command: podman-hpc run --rm --gpu --mpi $nccl_flag $image bash -lc <probe>"
  echo "################################################################################"

  srun -N 1 -n 1 -G 1 podman-hpc run --rm --gpu --mpi "$nccl_flag" \
    "$image" \
    bash -lc "$probe"
}

run_probe "CUDA 12 NCCL plus regular MPI injection" "--nccl-cu12"

echo
echo "Done. Log written to $log_file"

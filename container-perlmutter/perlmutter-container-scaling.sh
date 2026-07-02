#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

account="${ACCOUNT:-m5176}"
queue="${QUEUE:-regular}"
time_limit="${TIME:-01:00:00}"
image="${IMAGE:-localhost/gb25-perlmutter:cuda12}"
submit="${SUBMIT:-0}"
gpus_per_node="${GPUS_PER_NODE:-4}"
cpus_per_task="${CPUS_PER_TASK:-32}"
ngpus_list="${NGPUS_LIST:-4}"
grid_x="${GRID_X:-256}"
grid_y="${GRID_Y:-256}"
grid_z="${GRID_Z:-128}"
run_prefix="${RUN_PREFIX:-runs}"
run_postfix="${RUN_POSTFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
run_source="${RUN_SOURCE:-$repo_root/sharding/sharded_baroclinic_instability_simulation_run.jl}"
podman_hpc="${PODMAN_HPC:-podman-hpc}"
podman_run_extra="${PODMAN_RUN_EXTRA:-}"
srun_extra="${SRUN_EXTRA:-}"

nccl_flag="${NCCL_FLAG:---nccl-cu12}"
preferences_file="$script_dir/LocalPreferences.toml"

if [ -z "${OUT_DIR:-}" ] && [ -z "${SCRATCH:-}" ]; then
  echo "Set OUT_DIR or SCRATCH before generating Perlmutter jobs." >&2
  exit 2
fi

out_dir="${OUT_DIR:-$SCRATCH/GB25}"

if [ ! -f "$run_source" ]; then
  echo "RUN_SOURCE does not exist: $run_source" >&2
  exit 2
fi
if [ ! -f "$script_dir/Manifest.toml" ]; then
  echo "Missing $script_dir/Manifest.toml. Run container-perlmutter/update-manifest.sh first." >&2
  exit 2
fi

if [ "$submit" != "0" ] && [ "$submit" != "1" ]; then
  echo "SUBMIT must be 0 or 1." >&2
  exit 2
fi

timestamp="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
run_root="$out_dir/$run_prefix/${timestamp}_${run_postfix}"
mkdir -p "$run_root"

cp "$repo_root/Project.toml" "$run_root/Project.toml"
cp "$script_dir/Manifest.toml" "$run_root/Manifest.toml"
cp "$preferences_file" "$run_root/LocalPreferences.toml"
cp "$script_dir/JuliaProject.toml" "$run_root/JuliaProject.toml"

{
  printf 'repo_root = "%s"\n' "$repo_root"
  printf 'image = "%s"\n' "$image"
  printf 'cuda_family = "12"\n'
  printf 'ngpus_list = "%s"\n' "$ngpus_list"
  printf 'grid_x = "%s"\n' "$grid_x"
  printf 'grid_y = "%s"\n' "$grid_y"
  printf 'grid_z = "%s"\n' "$grid_z"
  if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    printf 'git_branch = "%s"\n' "$(git -C "$repo_root" rev-parse --abbrev-ref HEAD)"
    printf 'git_describe = "%s"\n' "$(git -C "$repo_root" describe --tags --always --dirty)"
  fi
} > "$run_root/run-info.toml"

if git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git -C "$repo_root" diff --no-ext-diff HEAD > "$run_root/git.diff" || true
fi

for ngpu in $ngpus_list; do
  case "$ngpu" in
    ''|*[!0-9]*)
      echo "NGPUS_LIST entries must be positive integers, got '$ngpu'." >&2
      exit 2
      ;;
  esac

  if [ "$ngpu" -eq 0 ]; then
    echo "NGPUS_LIST entries must be positive integers." >&2
    exit 2
  fi

  if [ "$ngpu" -eq 1 ]; then
    nnodes=1
    local_gpus=1
  else
    if [ $((ngpu % gpus_per_node)) -ne 0 ]; then
      echo "Ngpu=$ngpu must be divisible by GPUS_PER_NODE=$gpus_per_node." >&2
      exit 2
    fi
    nnodes=$((ngpu / gpus_per_node))
    local_gpus="$gpus_per_node"
  fi

  visible_devices="$(seq -s, 0 $((local_gpus - 1)))"
  resolution_fraction=$((4 * ngpu))
  ngpu_string="$(printf "%05d" "$ngpu")"
  job_dir="$run_root/ngpu=$ngpu_string"
  mkdir -p "$job_dir/julia_depot"

  run_file="$job_dir/$(basename "$run_source")"
  cp "$run_source" "$run_file"

  submit_file="$job_dir/submit.sh"
  cat > "$submit_file" <<EOF
#!/bin/bash -l
#SBATCH -C gpu&hbm40g
#SBATCH -q $queue
#SBATCH --gpu-bind=none
#SBATCH --job-name=GB25_${run_prefix}_${run_postfix}
#SBATCH --time=$time_limit
#SBATCH --nodes=$nnodes
#SBATCH --ntasks-per-node=1
#SBATCH --account=$account
#SBATCH --output=$job_dir/%j.out
#SBATCH --error=$job_dir/%j.err

set -euo pipefail

module load podman-hpc || true

export SBATCH_ACCOUNT=$account
export SALLOC_ACCOUNT=$account
export CUDA_VISIBLE_DEVICES=$visible_devices
export Ngpu=$ngpu
export resolution_fraction=$resolution_fraction
export TZ=UTC
export JULIA_DEBUG=Reactant,Reactant_jll
export JULIA_DEPOT_PATH=$job_dir/julia_depot:/usr/local/julia_depot
export JULIA_PROJECT=/opt/GB-25
export JULIA_LOAD_PATH=@:/opt/GB-25/container-perlmutter:@v#.#:@stdlib
export JULIA_CUDA_MEMORY_POOL=none
export JULIA_CUDA_USE_COMPAT=false
export JULIA_CUDA_USE_BINARYBUILDER=false
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD=16777216
export NCCL_BUFFSIZE=33554432
export XLA_REACTANT_GPU_MEM_FRACTION=0.9
export XLA_FLAGS="--xla_gpu_first_collective_call_warn_stuck_timeout_seconds=40 --xla_gpu_first_collective_call_terminate_timeout_seconds=80 \${XLA_FLAGS:-}"
export XLA_FLAGS="--xla_disable_hlo_passes=host-offload-legalize,hlo_constant_splitter,multi_output_fusion \${XLA_FLAGS}"

unset no_proxy http_proxy https_proxy NO_PROXY HTTP_PROXY HTTPS_PROXY

srun_extra_args=($srun_extra)
podman_extra_args=($podman_run_extra)
container_cmd=(
  "$podman_hpc"
  run
  --rm
  --gpu
  --mpi
  "$nccl_flag"
  "\${podman_extra_args[@]}"
  -v
  "$job_dir:$job_dir"
  -v
  "$job_dir/julia_depot:/job_julia_depot"
  -v
  "\${SCRATCH}:\${SCRATCH}"
  --env
  CUDA_VISIBLE_DEVICES
  --env
  Ngpu
  --env
  resolution_fraction
  --env
  TZ
  --env
  JULIA_DEBUG
  --env
  JULIA_DEPOT_PATH=/job_julia_depot:/usr/local/julia_depot
  --env
  JULIA_PROJECT
  --env
  JULIA_LOAD_PATH
  --env
  JULIA_CUDA_MEMORY_POOL
  --env
  JULIA_CUDA_USE_COMPAT
  --env
  JULIA_CUDA_USE_BINARYBUILDER
  --env
  FI_CXI_RDZV_GET_MIN
  --env
  FI_CXI_SAFE_DEVMEM_COPY_THRESHOLD
  --env
  NCCL_BUFFSIZE
  --env
  XLA_REACTANT_GPU_MEM_FRACTION
  --env
  XLA_FLAGS
  --env
  SLURM_JOB_ID
  --env
  SLURM_STEP_NODELIST
  --env
  SLURM_NTASKS
  --env
  SLURM_PROCID
  --env
  SLURM_LOCALID
  --env
  SLURM_STEP_NUM_NODES
)
julia_cmd=(
  julia
  --startup-file=no
  --project=/opt/GB-25
  --compiled-modules=strict
  -O0
  "$run_file"
  --grid-x
  "$grid_x"
  --grid-y
  "$grid_y"
  --grid-z
  "$grid_z"
)

srun_cmd=(
  srun
  -n
  $nnodes
  -c
  $cpus_per_task
  -G
  $ngpu
  --cpu-bind=verbose,cores
  "\${srun_extra_args[@]}"
  "\${container_cmd[@]}"
  "$image"
  "\${julia_cmd[@]}"
)

"\${srun_cmd[@]}"
EOF
  chmod 0755 "$submit_file"

  if [ "$submit" = "1" ]; then
    sbatch "$submit_file"
  else
    printf 'Wrote %s\n' "$submit_file"
  fi
done

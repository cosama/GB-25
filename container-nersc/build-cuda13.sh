#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
image="${IMAGE:-localhost/gb25-nersc:cuda13}"
podman="${PODMAN:-podman}"

exec "$podman" build   --build-arg CUDA_IMAGE=docker.io/nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04   --build-arg NERSC_LOCAL_PREFERENCES=container-nersc/LocalPreferences.cuda13.toml   -f "$script_dir/Containerfile"   -t "$image"   "$repo_root"

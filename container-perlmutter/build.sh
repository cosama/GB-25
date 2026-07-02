#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
image="${IMAGE:-localhost/gb25-perlmutter:cuda12}"
base_image="${BASE_IMAGE:-docker.io/nersc/base_gpu:26.06}"
stage_parent="${BUILD_TMPDIR:-/tmp}"
stage_root="$(mktemp -d "$stage_parent/gb25-perlmutter-build.XXXXXX")"
stage_repo="$stage_root/repo"

if command -v podman-hpc >/dev/null 2>&1; then
  podman=podman-hpc
  migrate=1
else
  podman=podman
  migrate=0
fi

if ! command -v "$podman" >/dev/null 2>&1; then
  echo "Missing container builder: $podman" >&2
  exit 2
fi

cleanup() {
  rm -rf "$stage_root"
}
trap cleanup EXIT

mkdir -p "$stage_repo"
tar -C "$repo_root" \
  --exclude='.git' \
  --exclude='podman-hpc-libs-*.log' \
  --exclude='container-perlmutter/podman-hpc-libs-*.log' \
  -cf - . | tar -C "$stage_repo" -xf -

"$podman" build \
  --build-arg BASE_IMAGE="$base_image" \
  --build-arg PERLMUTTER_LOCAL_PREFERENCES=container-perlmutter/LocalPreferences.toml \
  -f "$stage_repo/container-perlmutter/Containerfile" \
  -t "$image" \
  "$stage_repo"

if [ "$migrate" = "1" ]; then
  "$podman" migrate "$image"
fi

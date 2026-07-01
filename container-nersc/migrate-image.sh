#!/usr/bin/env bash
set -euo pipefail

image="${IMAGE:-localhost/gb25-nersc:cuda13}"
podman_hpc="${PODMAN_HPC:-podman-hpc}"

exec "$podman_hpc" migrate "$image"

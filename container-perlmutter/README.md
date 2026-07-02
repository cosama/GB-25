# Perlmutter Container

This directory contains the Perlmutter-specific GB-25 container workflow. The
root `Project.toml` remains the package definition. The lockfile here,
`container-perlmutter/Manifest.toml`, is intentionally image-specific because it is
resolved for the container Julia/CUDA/Reactant stack.

## Lock The Julia Environment

Run this whenever `Project.toml` changes or local package changes need to be
reflected in the Perlmutter image:

```bash
container-perlmutter/update-manifest.sh
```

The script resolves the root `Project.toml` in a temporary project and
writes the resulting lockfile back to `container-perlmutter/Manifest.toml`.

## Build Images

```bash
container-perlmutter/build.sh
```

Default:

```bash
IMAGE=localhost/gb25-perlmutter:cuda12
```

The build uses the newest NERSC GPU base image found on Docker Hub,
`docker.io/nersc/base_gpu:26.06` (CUDA 12.9.1, NCCL 2.27.3, MPICH 5.0.1).

The build script stages the context under `/tmp` before invoking the container
builder to avoid filesystem metadata issues from mounted project paths. It uses
`podman-hpc` and migrates the image when `podman-hpc` is available; otherwise it
falls back to `podman`. The image build instantiates the locked environment but
deliberately avoids Julia imports and precompilation. Compilation is
target-architecture-sensitive and should happen on the Perlmutter GPU runtime,
not during image build.

## Perlmutter Use

Generate Slurm scripts without submitting:

```bash
SUBMIT=0 NGPUS_LIST="4 8 32" IMAGE=localhost/gb25-perlmutter:cuda12   container-perlmutter/perlmutter-container-scaling.sh
```

Submit generated jobs:

```bash
SUBMIT=1 NGPUS_LIST="4" IMAGE=localhost/gb25-perlmutter:cuda12   container-perlmutter/perlmutter-container-scaling.sh
```

The launch path is bash-only. It assumes Julia exists inside the container and
uses `podman-hpc run` under `srun` so Perlmutter supplies the CUDA, regular
Cray MPI, and NCCL runtime injections.

## Tests

```bash
container-perlmutter/tests/test-local-preferences.sh
```

To verify the NERSC base image and `podman-hpc` GPU/MPI/NCCL injection on
Perlmutter:

```bash
container-perlmutter/test-podman-hpc-libs.sh
```

The probe requests a small GPU allocation automatically when run from a login
node and writes a timestamped `podman-hpc-libs-*.log` file.

## MPI Caveat

The runtime uses `podman-hpc --mpi`, not `--cuda-mpi`. The CUDA-aware MPI module
currently injects a CUDA 11 GTL preload on Perlmutter, which is incompatible
with the CUDA 12.9 NERSC base image. GB-25's Reactant/XLA sharded path should use
NCCL for GPU collectives, with regular Cray MPI available for process/runtime
coordination.

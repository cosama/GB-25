# NERSC Container

This directory contains the NERSC-specific GB-25 container workflow. The fork
root `Project.toml` remains the package definition. The lockfile here,
`container-nersc/Manifest.toml`, is intentionally image-specific because it is
resolved for the container Julia/CUDA/Reactant stack.

## Lock The Julia Environment

Run this whenever `Project.toml` changes or local package changes need to be
reflected in the NERSC image:

```bash
container-nersc/update-manifest.sh
```

The script resolves the fork root `Project.toml` in a temporary project and
writes the resulting lockfile back to `container-nersc/Manifest.toml`. It does
not create or update a root `Manifest.toml`.

## Build Images

```bash
container-nersc/build-cuda12.sh
container-nersc/build-cuda13.sh
```

Defaults:

```bash
IMAGE=localhost/gb25-nersc:cuda12
IMAGE=localhost/gb25-nersc:cuda13
```

The image build instantiates the locked environment but deliberately avoids
Julia imports and precompilation. Compilation is target-architecture-sensitive
and should happen on the Perlmutter GPU runtime, not during image build.

## Perlmutter Use

After the image is available on Perlmutter, migrate it once:

```bash
IMAGE=localhost/gb25-nersc:cuda13 container-nersc/migrate-image.sh
```

Generate Slurm scripts without submitting:

```bash
SUBMIT=0 NGPUS_LIST="4 8 32" IMAGE=localhost/gb25-nersc:cuda13 CUDA_FAMILY=13   container-nersc/perlmutter-container-scaling.sh
```

Submit generated jobs:

```bash
SUBMIT=1 NGPUS_LIST="4" IMAGE=localhost/gb25-nersc:cuda13 CUDA_FAMILY=13   container-nersc/perlmutter-container-scaling.sh
```

The launch path is bash-only. It assumes Julia exists inside the container and
uses `podman-hpc run` under `srun` so Perlmutter supplies the CUDA, MPI, and
NCCL runtime injections.

## Tests

```bash
container-nersc/tests/test-local-preferences.sh
```

## MPI Caveat

The container includes MPICH only as a local/build fallback. Production runs
should use the optimized NERSC injection from `podman-hpc`. If MPI.jl does not
select that injected MPI stack at runtime, the next non-hacky step is to add the
official `MPIPreferences` configuration path to the package environment rather
than patching library paths manually.

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
container_dir="$(cd "$script_dir/.." && pwd)"
julia_image="${JULIA_IMAGE:-docker.io/library/julia:1.11.7}"
podman="${PODMAN:-podman}"

exec "$podman" run --rm   -v "$container_dir:/container-perlmutter:ro"   -w /container-perlmutter   "$julia_image"   julia --startup-file=no --compiled-modules=no -e '
using TOML

project = TOML.parsefile("JuliaProject.toml")
extras = project["extras"]
extras["CUDA_Runtime_jll"] == "76a88914-d11a-5bdc-97e0-2f5a05c973a2" || error("JuliaProject.toml: missing CUDA_Runtime_jll extra")
extras["MPIPreferences"] == "3da0fdf6-3ccc-4f1b-acd9-58baa6c99267" || error("JuliaProject.toml: missing MPIPreferences extra")
extras["Reactant_jll"] == "0192cb87-2b54-54ad-80e0-3be72ad8a3c0" || error("JuliaProject.toml: missing Reactant_jll extra")

expected = Dict(
    "LocalPreferences.toml" => "12.9",
)

for (file, version) in expected
    prefs = TOML.parsefile(file)
    cuda_runtime = prefs["CUDA_Runtime_jll"]
    reactant = prefs["Reactant_jll"]
    mpi = prefs["MPIPreferences"]

    string(cuda_runtime["local"]) == "true" || error("$(file): CUDA_Runtime_jll.local must be true")
    cuda_runtime["version"] == version || error("$(file): CUDA_Runtime_jll.version must be $(version)")
    reactant["gpu"] == "cuda" || error("$(file): Reactant_jll.gpu must be cuda")
    reactant["gpu_version"] == version || error("$(file): Reactant_jll.gpu_version must be $(version)")
    reactant["mode"] == "opt" || error("$(file): Reactant_jll.mode must be opt")
    haskey(reactant, "cuda_version") && error("$(file): use gpu_version, not cuda_version, for this Reactant_jll lock")
    mpi["_format"] == "1.0" || error("$(file): MPIPreferences._format must be 1.0")
    mpi["abi"] == "MPICH" || error("$(file): MPIPreferences.abi must be MPICH")
    mpi["binary"] == "system" || error("$(file): MPIPreferences.binary must be system")
    mpi["libmpi"] == "/opt/udiImage/modules/mpich/libmpi.so" || error("$(file): MPIPreferences.libmpi must point at injected MPICH")
    mpi["mpiexec"] == "srun" || error("$(file): MPIPreferences.mpiexec must be srun")
end
'

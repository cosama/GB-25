#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
container_dir="$(cd "$script_dir/.." && pwd)"
julia_image="${JULIA_IMAGE:-docker.io/library/julia:1.12.6}"
podman="${PODMAN:-podman}"

exec "$podman" run --rm   -v "$container_dir:/container-nersc:ro"   -w /container-nersc   "$julia_image"   julia --startup-file=no --compiled-modules=no -e '
using TOML

expected = Dict(
    "LocalPreferences.cuda12.toml" => "12.9",
    "LocalPreferences.cuda13.toml" => "13.1",
)

for (file, version) in expected
    prefs = TOML.parsefile(file)
    cuda_runtime = prefs["CUDA_Runtime_jll"]
    reactant = prefs["Reactant_jll"]

    string(cuda_runtime["local"]) == "true" || error("$(file): CUDA_Runtime_jll.local must be true")
    cuda_runtime["version"] == version || error("$(file): CUDA_Runtime_jll.version must be $(version)")
    reactant["gpu"] == "cuda" || error("$(file): Reactant_jll.gpu must be cuda")
    reactant["gpu_version"] == version || error("$(file): Reactant_jll.gpu_version must be $(version)")
    reactant["mode"] == "opt" || error("$(file): Reactant_jll.mode must be opt")
    haskey(reactant, "cuda_version") && error("$(file): use gpu_version, not cuda_version, for this Reactant_jll lock")
end
'

#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
julia_image="${JULIA_IMAGE:-docker.io/library/julia:1.11.7}"
julia_depot="${JULIA_DEPOT:-/tmp/gb25-perlmutter-julia-depot}"
preferences_file="${PREFERENCES_FILE:-$script_dir/LocalPreferences.toml}"
podman="${PODMAN:-podman}"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/gb25-perlmutter-manifest.XXXXXX")"
tmp_project="$tmp_root/project"
tmp_preferences_project="$tmp_project/container-perlmutter"

cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT

mkdir -p "$tmp_project" "$tmp_preferences_project" "$julia_depot"
cp "$repo_root/Project.toml" "$tmp_project/Project.toml"
cp "$preferences_file" "$tmp_preferences_project/LocalPreferences.toml"
cp "$script_dir/JuliaProject.toml" "$tmp_preferences_project/JuliaProject.toml"

"$podman" run --rm   -e JULIA_DEPOT_PATH=/julia_depot   -e JULIA_LOAD_PATH=@:/project/container-perlmutter:@v#.#:@stdlib   -e JULIA_CUDA_USE_BINARYBUILDER=false   -e JULIA_PKG_PRECOMPILE_AUTO=0   -v "$julia_depot:/julia_depot"   -v "$tmp_project:/project"   -w /project   "$julia_image"   julia --startup-file=no --compiled-modules=no -e '
using Pkg

Pkg.activate("/project")
if isempty(Pkg.Registry.reachable_registries())
    Pkg.Registry.add(Pkg.RegistrySpec(name="General"))
end
Pkg.resolve()
Pkg.instantiate()
'

cp "$tmp_project/Manifest.toml" "$script_dir/Manifest.toml"

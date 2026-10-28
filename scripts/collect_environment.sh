#!/usr/bin/env bash
# Lightweight cluster inventory. Follow local policy for where this may run.
# This script does not submit jobs, request allocations, compile, or launch MPI.
set -euo pipefail
umask 077

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
output_dir="$project_root/private/environment"
mkdir -p -- "$output_dir"
output_file="$(mktemp "$output_dir/environment-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"

record_command() {
    printf '\n##'
    printf ' %q' "$@"
    printf '\n'
    if command -v "$1" >/dev/null 2>&1; then
        "$@" 2>&1 || printf '[command failed; verify manually]\n'
    else
        printf '[unavailable in current shell]\n'
    fi
}

{
    printf 'Parallel Sparse MoE environment inventory\n'
    printf 'UTC: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'Run inside an allocation separately to record compute-node hardware.\n'
    record_command hostname
    record_command uname -srm
    record_command lscpu
    record_command free -h
    record_command cmake --version
    record_command git --version
    record_command c++ --version
    record_command mpicxx --version
    record_command mpirun --version
    printf '\n## Loaded modules\n'
    if type module >/dev/null 2>&1; then
        module list 2>&1 || true
    else
        printf 'module function unavailable in this shell\n'
    fi
    printf '\n## Scheduler commands visible in this shell\n'
    for tool in srun sbatch salloc qsub qstat bsub bjobs; do
        if command -v "$tool" >/dev/null 2>&1; then
            command -v "$tool"
        fi
    done
    printf '\n## OpenMP settings\n'
    printf 'OMP_NUM_THREADS=%s\n' "${OMP_NUM_THREADS:-UNSET}"
    printf 'OMP_PLACES=%s\n' "${OMP_PLACES:-UNSET}"
    printf 'OMP_PROC_BIND=%s\n' "${OMP_PROC_BIND:-UNSET}"
    printf '\nAccount, quota, scheduler policy, and storage rules require manual confirmation.\n'
} > "$output_file" 2>&1

printf 'Saved private inventory: %s\n' "$output_file"

#!/usr/bin/env bash
# Wrapper for a launcher command supplied by the selected cluster's documentation.
# Example (only if the local guide prescribes it):
#   OMP_NUM_THREADS=2 scripts/run_launch_probe.sh -- mpirun -np 2 \
#     build/moe_launch_probe --expect-threads 2
set -euo pipefail

if [[ "${1:-}" != "--" || "$#" -lt 2 ]]; then
  cat >&2 <<'USAGE'
Usage: scripts/run_launch_probe.sh -- <launcher and program command>

This wrapper deliberately does not choose a scheduler, partition, account, or
MPI launcher. Use the exact launcher prescribed by the selected Atlas/Vanda
documentation. Set OMP_NUM_THREADS and affinity settings before invocation.
USAGE
  exit 2
fi

shift
exec "$@"

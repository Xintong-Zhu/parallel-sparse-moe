#!/bin/bash
# Run on the login node: compute nodes need no Git installation.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
commit=$(git rev-parse --verify HEAD)
changes=$(git status --porcelain --untracked-files=no)
if [[ -n "$changes" ]]; then
  echo 'Commit tracked source changes before submitting.' >&2
  exit 1
fi
command -v qsub >/dev/null
mkdir -p "$repo/private"
snapshot=$(mktemp -d "$repo/private/cpu-source-${commit:0:12}-XXXXXXXX")
# PBS -v separates variables with commas; reject ambiguous path values.
if [[ "$snapshot" == *','* || "$snapshot" == *$'\n'* || "$snapshot" == *' '* ]]; then
  echo 'Submit from a repository path without commas, newlines, or spaces.' >&2
  exit 1
fi
git archive "$commit" | tar -x -C "$snapshot"
printf '%s\n' "$commit" > "$snapshot/.moe-source-commit"
(
  cd "$snapshot"
  find . -type f ! -name .moe-source.sha256 -print0 | sort -z |
    xargs -0 sha256sum > .moe-source.sha256
)
printf 'Frozen source: %s\nGit commit: %s\n' "$snapshot" "$commit"
qsub -v "MOE_SOURCE_DIR=$snapshot" "$snapshot/scripts/vanda_cpu_formal.pbs"

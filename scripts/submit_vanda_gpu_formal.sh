#!/bin/bash
# Run on the Vanda login node from a committed checkout.
set -euo pipefail
repo=$(git rev-parse --show-toplevel)
cd "$repo"
commit=$(git rev-parse --verify HEAD)
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo 'Commit tracked source changes before submitting.' >&2
  exit 1
fi
command -v qsub >/dev/null
mkdir -p "$repo/private"
snapshot=$(mktemp -d "$repo/private/gpu-source-${commit:0:12}-XXXXXXXX")
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
qsub -v "MOE_SOURCE_DIR=$snapshot" "$snapshot/scripts/vanda_gpu_formal.pbs"

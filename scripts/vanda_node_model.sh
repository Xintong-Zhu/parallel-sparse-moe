#!/bin/bash
set -euo pipefail

model=$(lscpu | awk -F: '/^Model name:/ {
  sub(/^[[:space:]]+/, "", $2)
  print $2
  exit
}')
if [[ -z "$model" ]]; then
  echo "CPU model unavailable on $(hostname)" >&2
  exit 1
fi
printf '%s,%s\n' "$(hostname)" "$model"

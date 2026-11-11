#!/bin/bash
set -euo pipefail
node=$(hostname -s)
[[ "$node" =~ ^[a-zA-Z0-9_-]+$ ]]
{ hostname; id; lscpu; } > "${1:?output directory required}/topology-${node}.txt"

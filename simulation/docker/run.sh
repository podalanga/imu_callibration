#!/usr/bin/env bash
# Start the container shell (GUI via X11, NVIDIA GPU, host network, workspace mounted at /ws).
#   ./run.sh            -> service "sim"  (image drone_sim:px4)
#   ./run.sh base       -> service "base" (image drone_sim:base)
set -euo pipefail
cd "$(dirname "$0")"
xhost +local:docker >/dev/null 2>&1 || true
exec docker compose run --rm "${1:-sim}" bash

#!/usr/bin/env bash
# Build a drone_sim image. Long: ~30-60 min the first time (PX4 SITL build).
#   ./build.sh            -> drone_sim:px4   (target px4: PX4 SITL + XRCE agent + px4_msgs underlay)
#   ./build.sh base       -> drone_sim:base  (ROS + gz + apriltag only, no PX4)
# Env: PX4_BUILD_JOBS (default 6; the PX4 build needs ~2.5 GB RAM per job)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

TARGET="${1:-px4}"
case "$TARGET" in
    px4|base) ;;
    *) echo "usage: $0 [px4|base]" >&2; exit 2 ;;
esac

docker build --target "$TARGET" -t "drone_sim:$TARGET" \
    --build-arg PX4_BUILD_JOBS="${PX4_BUILD_JOBS:-6}" .

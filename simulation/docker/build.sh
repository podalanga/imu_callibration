#!/usr/bin/env bash
# Build the full image (PX4 SITL + XRCE agent + px4_msgs underlay). Long: ~30-60 min the first time.
#   ./build.sh            -> drone_sim:px4   (target px4)
#   ./build.sh base       -> drone_sim:base  (ROS + gz + apriltag only, no PX4)
set -euo pipefail
cd "$(dirname "$0")"
TARGET="${1:-px4}"
docker build --target "$TARGET" -t "drone_sim:$TARGET" \
  --build-arg PX4_BUILD_JOBS="${PX4_BUILD_JOBS:-6}" .

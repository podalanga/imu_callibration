#!/usr/bin/env bash
# Container entrypoint: source ROS 2, the px4_msgs underlay and (if built) the /ws overlay, then run the command.
# Also used by run.sh for every `docker exec`, so each command sees the same environment.
set -e

source /opt/ros/jazzy/setup.bash
[ -f /opt/px4_ws/install/setup.bash ] && source /opt/px4_ws/install/setup.bash
[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash

exec "$@"

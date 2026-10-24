#!/usr/bin/env bash
# drone_sim container helper. Run `./run.sh help` for the command list.
#
#   ./run.sh                    start the container, build the workspace and launch the full sim (Gazebo GUI)
#   ./run.sh bash               open a shell in the running container
#   ./run.sh help               everything else
#
# Environment: DRONE_SIM_SERVICE=base selects the PX4-less image; ROS_DOMAIN_ID (default 42) is passed through.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
SERVICE="${DRONE_SIM_SERVICE:-sim}"
case "$SERVICE" in
    sim)  CONTAINER=drone_sim ;;
    base) CONTAINER=drone_sim_base ;;
    *)    echo "error: DRONE_SIM_SERVICE must be 'sim' or 'base' (got '$SERVICE')" >&2; exit 2 ;;
esac

log()  { printf '\033[1;34m[run.sh]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[run.sh] error:\033[0m %s\n' "$*" >&2; exit 1; }

compose() { docker compose -f "$COMPOSE_FILE" --profile "$SERVICE" "$@"; }

is_running() {
    [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null)" = "true" ]
}


# Run a command inside the container with the ROS environment sourced (via the entrypoint).
in_container() {
    is_running || die "container '$CONTAINER' is not running (start it with: ./run.sh up)"
    # -t only when attached to a terminal, so the script also works from pipes / CI.
    # (Tested here, not in a $(...) helper: inside command substitution stdout is a pipe, never a tty.)
    local flags=(-i)
    if [ -t 0 ] && [ -t 1 ]; then flags=(-it); fi
    docker exec "${flags[@]}" -e DISPLAY="${DISPLAY:-:0}" "$CONTAINER" /entrypoint.sh "$@"
}

cmd_up() {
    if is_running; then
        log "container '$CONTAINER' already running"
        return
    fi
    docker image inspect "drone_sim:$([ "$SERVICE" = sim ] && echo px4 || echo base)" >/dev/null 2>&1 \
        || die "image missing; build it first: ./run.sh image"
    xhost +local:docker >/dev/null 2>&1 || log "xhost not available: GUI windows may not open"
    log "starting container '$CONTAINER'"
    compose up -d "$SERVICE"
}

cmd_down() {
    log "stopping and removing '$CONTAINER'"
    compose down --remove-orphans
}

cmd_build_ws() {
    cmd_up
    log "building /ws (colcon)"
    in_container bash -c 'cd /ws && colcon build --symlink-install "$@"' _ "$@"
}

sim_running() {
    is_running && docker exec "$CONTAINER" pgrep -f "[r]os2 launch drone_sim" >/dev/null
}

cmd_launch() {
    cmd_up
    # A second launch would start another gz + PX4 on the same ports/topics and corrupt both runs.
    sim_running && die "a sim is already running in '$CONTAINER' (stop it with: ./run.sh kill)"
    cmd_build_ws
    log "ros2 launch drone_sim sim.launch.py $*  (Ctrl+C to stop)"
    in_container ros2 launch drone_sim sim.launch.py "$@"
}

# The [x]-style patterns in cmd_kill_sim / cmd_status stop pkill/pgrep from matching their own bash -c line.
cmd_kill_sim() {
    is_running || { log "container not running"; return; }
    log "stopping running launch / gz / PX4 processes"
    docker exec "$CONTAINER" bash -c \
        'pkill -INT -f "[r]os2 launch drone_sim" || true; sleep 3; pkill -9 -f "[g]z sim|[b]in/px4|[M]icroXRCEAgent|[d]rone_sim/lib" || true'
}

cmd_status() {
    if is_running; then
        log "container '$CONTAINER' is running"
        docker ps --filter "name=^${CONTAINER}$" --format '  {{.Names}}  {{.Image}}  {{.Status}}'
        docker exec "$CONTAINER" bash -c 'pgrep -af "[r]os2 launch|[g]z sim |[b]in/px4" | sed "s/^/  /" | grep . || echo "  (sim not launched)"'
    else
        log "container '$CONTAINER' is not running"
    fi
}

cmd_camera() {
    case "${1:-}" in
        "")  in_container ros2 run drone_sim camera_view ;;
        raw) in_container ros2 run rqt_image_view rqt_image_view /camera/image_raw ;;
        *)   die "usage: ./run.sh camera [raw]" ;;
    esac
}

cmd_snapshot() {
    local name="${1:-camera_$(date +%Y%m%d_%H%M%S).png}"
    in_container bash -c 'mkdir -p /ws/snapshots && ros2 run drone_sim camera_view --save "/ws/snapshots/$1"' _ "$name"
    log "saved $(cd "$SCRIPT_DIR/../ros2_ws" && pwd)/snapshots/$name"
}

usage() {
    cat <<EOF
Usage: ./run.sh [command] [args...]

Simulation
  (none) | sim [launch args]   start container, colcon build, launch the full sim with the Gazebo GUI
  headless [launch args]       same, without GUI (headless:=true)
  launch [launch args]         launch without rebuilding the workspace
  kill                         stop a running launch (gz, PX4, nodes) but keep the container
  gui                          attach a Gazebo GUI to a running (headless) sim

Container
  up                           start the named container '$CONTAINER' in the background
  bash | shell                 interactive shell in the container (ROS sourced)
  exec <cmd...>                run a command in the container, e.g. ./run.sh exec ros2 node list
  build [colcon args]          colcon build --symlink-install in /ws (re-run after editing config/*.yaml)
  status                       container + sim process status
  logs                         container logs (follow)
  restart                      down + up
  down | stop                  stop and remove the container

Sensors and debugging (sim must be running)
  camera                       live camera window with AprilTag outlines + camera height (q / Esc closes)
  camera raw                   plain rqt_image_view on /camera/image_raw
  snapshot [name.png]          save one annotated camera frame to simulation/ros2_ws/snapshots/
  tags                         echo AprilTag detections (id + pixel centre)
  imu [ideal|noisy]            echo /imu/noisy (default) or /imu/ideal
  uwb [ranges]                 echo /uwb/position (multilateration fix) or the raw /uwb/ranges
  odom                         echo ground-truth pose /ground_truth/odom
  ctl                          echo /controller/status (state, traversals done, errors)
  topics                       ros2 topic list
  hz <topic>                   publish rate of a topic, e.g. ./run.sh hz /camera/image_raw
  echo <topic> [args]          ros2 topic echo, e.g. ./run.sh echo /imu/noisy --once
  graph                        rqt_graph (nodes and topics)

Image
  image [px4|base]             build the Docker image (./build.sh)

Launch args: headless:=true gui:=false detector:=false px4:=false reset_params:=true config_dir:=<dir>
Env: DRONE_SIM_SERVICE=base (PX4-less image), ROS_DOMAIN_ID (default 42)
EOF
}

main() {
    local cmd="${1:-sim}"
    [ $# -gt 0 ] && shift
    case "$cmd" in
        sim|start)       cmd_launch "$@" ;;
        headless)        cmd_launch headless:=true "$@" ;;
        launch)          cmd_up; sim_running && die "a sim is already running (./run.sh kill first)"
                         in_container ros2 launch drone_sim sim.launch.py "$@" ;;
        kill)            cmd_kill_sim ;;
        gui)             in_container gz sim -g ;;
        up)              cmd_up ;;
        bash|shell)      cmd_up; in_container bash ;;
        exec)            [ $# -gt 0 ] || die "usage: ./run.sh exec <cmd...>"; cmd_up; in_container "$@" ;;
        build)           cmd_build_ws "$@" ;;
        status|ps)       cmd_status ;;
        logs)            docker logs -f "$CONTAINER" ;;
        restart)         cmd_down; cmd_up ;;
        down|stop)       cmd_down ;;
        camera)          cmd_camera "$@" ;;
        snapshot)        cmd_snapshot "$@" ;;
        tags)            in_container ros2 topic echo /detections --field detections ;;
        imu)             in_container ros2 topic echo "/imu/${1:-noisy}" ;;
        uwb)             in_container ros2 topic echo "/uwb/${1:-position}" ;;
        odom)            in_container ros2 topic echo /ground_truth/odom --field pose.pose ;;
        ctl)             in_container ros2 topic echo /controller/status ;;
        topics)          in_container ros2 topic list ;;
        hz)              [ $# -gt 0 ] || die "usage: ./run.sh hz <topic>"; in_container ros2 topic hz "$@" ;;
        echo)            [ $# -gt 0 ] || die "usage: ./run.sh echo <topic> [args]"; in_container ros2 topic echo "$@" ;;
        graph)           in_container rqt_graph ;;
        image)           "$SCRIPT_DIR/build.sh" "$@" ;;
        help|-h|--help)  usage ;;
        *)               usage >&2; die "unknown command '$cmd'" ;;
    esac
}

main "$@"

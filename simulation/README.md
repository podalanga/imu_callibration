# drone_sim: PX4 SITL + Gazebo Harmonic + ROS 2 Jazzy, AprilTag-track drone

**Tested 2026-10-25** (GUI launch in `drone_sim:px4`): arms, takes off, flies all 5 traversals; see `progress.md`.

A PX4 x500 flies 2.5 back-and-forth rounds = 5 one-way traversals (x = -5.75 m <-> 5.75 m, y = 0, yaw = 0) at a constant
0.3 m/s (short raised-cosine stop/reverse at each end, no dwell) over a 22-tag AprilTag track (tag36h11) at a camera
optical-centre height of 1.10 m. Sensors: downward camera matching the OnePlus Nord 4 main camera (1920x1080, hfov 1.2112 rad),
ideal and noisy IMUs and a UWB tag, all at the CoM, plus ranges to 8 anchors on the corners of a 12 x 12 x 12 m cube.
World origin = track centre = centre of the cube's floor face; the 20 x 20 m floor matches the Gazebo GUI grid.
Everything runs in Docker (Ubuntu 24.04, ROS 2 Jazzy, Gazebo Harmonic). No parameters live in code: see `config/*.yaml`.

## Control architecture ("idealised controller")
```
              gz sim  (Gazebo Harmonic)
  world: floor, 22 tags, 8 UWB anchors, x500_tagcam_0
        |                            |
  x500 motors +             our sensors:
  PX4 sim IMU/baro/mag      camera, /imu/ideal,
        |                   /imu/noisy, GT odometry
   PX4 gz_bridge                     |
        |                      ros_gz_bridge
        v                            v
 +--------------+            +----------------+
 |   PX4 SITL   |            |     ROS 2      |
 | EKF2 + pos / |  XRCE-DDS  | mission, EV,   |
 | att / rate   |<---------->| UWB, tf,       |
 | loops        | udp 8888   | apriltag nodes |
 +--------------+            +----------------+
```

| ROS 2 node | Subscribes | Publishes |
|---|---|---|
| `px4_mission_node` | `/ground_truth/odom`, `/fmu/out/vehicle_status_v1`, `/fmu/out/vehicle_local_position` | `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, `/fmu/in/vehicle_command`, `/controller/status`, `/trajectory/reference` |
| `gt_to_px4_ev_node` | `/ground_truth/odom` (ENU/FLU) | `/fmu/in/vehicle_visual_odometry` (NED/FRD) |
| `uwb_node` | `/ground_truth/odom` (simulated ranges from GT) | `/uwb/ranges` |
| `uwb_multilateration_node` | `/uwb/ranges` | `/uwb/position` |
| `tf_node` | `/ground_truth/odom` | TF tree, `/ground_truth/pose` |
| `apriltag` (apriltag_ros) | `/camera/image_raw`, `/camera/camera_info` | `/detections` |
| `gz_bridge` (ros_gz_bridge) | gz topics | `/clock`, `/ground_truth/odom`, camera, IMUs |

* PX4 EKF2 fuses **ground truth as external vision** (position, velocity, yaw; `EKF2_EV_CTRL=15`, height reference =
  vision, GPS/baro/mag fusion off, tiny noise). Physics are real: PX4 commands motors, the gz motor model produces thrust, nothing is teleported.
* PX4 uses its own simulated IMU (`x500_base` `imu_sensor`) for its estimator and attitude loops. Our `/imu/ideal` and
  `/imu/noisy` sensors are independent copies at the CoM, for data collection only; they are not fed to PX4.
* The mission node (`px4_mission_node`) plans in world ENU (camera-height path from `trajectory.py`, raised-cosine velocity
  profile), commands the **CoM** setpoint = camera path minus the rotated camera offset, and converts to NED at the PX4 boundary:
  `x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu, yaw_ned = pi/2 - yaw_enu`. Setpoints carry position + velocity + acceleration feedforward + yaw at 50 Hz (>= 10 Hz is required before and during offboard).
* PX4 local origin = world origin (the EV is the GT world pose), so the NED local frame is the world frame rotated; the x500 starts at local NED (y = -5.75).
* Traversals are counted from ground truth: a traversal is done when the reference profile finished, along-track error < 5 cm and speed < 0.05 m/s.

## Exact commands
Everything goes through `docker/run.sh`. It keeps one named, long-lived container (`drone_sim`; `drone_sim_base` with
`DRONE_SIM_SERVICE=base`) and runs each command in it with `docker exec`, with ROS 2, px4_msgs and `/ws` sourced by
`docker/entrypoint.sh`.
```bash
cd simulation/docker
./run.sh image                 # build drone_sim:px4 (PX4 + XRCE agent + px4_msgs; long, PX4 uses 6 jobs ~ 15 GB RAM)
./run.sh                       # start container, colcon build, launch the full sim with the Gazebo GUI (Ctrl+C stops it)
./run.sh headless              # same without GUI; extra launch args pass through: ./run.sh headless detector:=false
./run.sh bash                  # shell in the running container
./run.sh exec ros2 topic echo /uwb/position
./run.sh ctl                   # echo /controller/status      | ./run.sh topics | ./run.sh camera (rqt_image_view)
./run.sh gui                   # attach a Gazebo GUI to a headless sim
./run.sh build                 # colcon build only (needed after editing config/*.yaml)
./run.sh kill                  # stop the sim, keep the container  | ./run.sh status | ./run.sh logs
./run.sh down                  # stop and remove the container
./run.sh help                  # full list
```
Networking: host network (X11, `gz sim -g` and host ROS 2 tools work directly), but discovery is limited to this machine
(`ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`, `GZ_IP=127.0.0.1`, `ROS_DOMAIN_ID=42`; override with the same env vars).
Only one sim can run per machine: two host-network sims would mix their gz/PX4 topics.
Launch args: `headless` (false), `gui` (true), `detector` (true), `px4` (true), `reset_params` (true: deletes PX4
`parameters.bson` so airframe changes apply), `px4_dir` (default `$PX4_DIR`), `config_dir`.
Camera view without X: run headless and use `ros2 run image_view image_saver` or `ros2 bag record /camera/image_raw`.
Without a GPU, drop `runtime: nvidia` in `docker/docker-compose.yml` and use `headless:=true` (software EGL, slow).

## Looking at the sensors
From the host, while a sim is running (`./run.sh help` lists everything):
```bash
./run.sh camera                # live camera window: AprilTag outlines + ids + camera height (q / Esc closes)
./run.sh camera raw            # plain rqt_image_view
./run.sh snapshot [name.png]   # one annotated frame -> simulation/ros2_ws/snapshots/
./run.sh tags                  # AprilTag detections (id, pixel centre, corners)
./run.sh imu [ideal|noisy]     # IMU messages (default noisy)
./run.sh uwb [ranges]          # UWB position fix, or the 8 raw ranges
./run.sh odom                  # ground-truth pose
./run.sh ctl                   # mission state, traversals done, tracking error
./run.sh hz /camera/image_raw  # rate of any topic   |   ./run.sh echo <topic> [--once]   |   ./run.sh graph
```
Inside the container (`./run.sh bash`, ROS already sourced):
```bash
ros2 topic list                                        # every topic
ros2 topic echo /imu/noisy --once                      # one message
ros2 topic echo /detections --field detections         # only the detections
ros2 topic echo /uwb/position --field pose.pose.position
ros2 topic hz /imu/noisy                               # rate (200 Hz expected)
ros2 run drone_sim camera_view                         # camera window with detections
ros2 run drone_sim camera_view --save /ws/snapshots/frame.png
ros2 run rqt_image_view rqt_image_view /camera/image_raw
ros2 run tf2_tools view_frames                         # TF tree -> frames.pdf
ros2 bag record /imu/noisy /imu/ideal /camera/image_raw /camera/camera_info /detections /uwb/ranges /ground_truth/odom
gz topic -l                                            # Gazebo-side topics
```

## Sources and pinned revisions
| Component | Repo | Revision |
|---|---|---|
| Base image | `osrf/ros:jazzy-desktop` | Ubuntu 24.04, ROS 2 Jazzy |
| Gazebo Harmonic + bridge | `ros-jazzy-ros-gz` (github.com/gazebosim/ros_gz, `jazzy`) | apt |
| AprilTag | `ros-jazzy-apriltag-ros` (github.com/christianrauch/apriltag_ros) | apt (source read at tag `3.2.2`) |
| PX4 | github.com/PX4/PX4-Autopilot | tag `v1.16.2`, shallow, submodules shallow |
| Drone models | github.com/PX4/PX4-gazebo-models (submodule `Tools/simulation/gz`) | as pinned by PX4 v1.16.2 (`models/x500`, `x500_base`) |
| DDS bridge | github.com/eProsima/Micro-XRCE-DDS-Agent | tag `v2.4.3` |
| PX4 messages | github.com/PX4/px4_msgs | branch `release/1.16` (cloned and built in the image at `/opt/px4_ws`; not vendored) |
Without Docker: `git clone --depth 1 -b release/1.16 https://github.com/PX4/px4_msgs` into your workspace `src/`.
Tag artwork: `apriltags_imu_track_70mm/` (copied to `ros2_ws/src/drone_sim/models/apriltags/textures`).

## Topics
| Topic | Type | Notes |
|---|---|---|
| `/clock` | rosgraph_msgs/Clock | from gz (all nodes use sim time) |
| `/ground_truth/odom` | nav_msgs/Odometry | world ENU, point = CoM (OdometryPublisher `xyz_offset`), twist in body (FLU) frame |
| `/ground_truth/pose` | geometry_msgs/PoseStamped | from tf_node |
| `/camera/image_raw`, `/camera/camera_info` | sensor_msgs/Image, CameraInfo | 1920x1080, 30 Hz, frame `camera_optical_frame` |
| `/detections` | apriltag_msgs/AprilTagDetectionArray | apriltag_ros; TF `apriltag_det_<id>` |
| `/imu/ideal`, `/imu/noisy` | sensor_msgs/Imu | 200 Hz, `imu_ideal_link`, `imu_noisy_link`; ignore `orientation` |
| `/uwb/ranges` | drone_sim_interfaces/UwbRangeArray | 10 Hz, 8 ranges |
| `/uwb/position` | geometry_msgs/PoseWithCovarianceStamped | multilateration fix, frame `world`, cov = sigma^2 (J^T J)^-1 |
| `/controller/status`, `/trajectory/reference` | ControllerStatus, PoseStamped | CoM reference, world |
| `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, `/fmu/in/vehicle_command` | px4_msgs | mission node -> PX4 |
| `/fmu/in/vehicle_visual_odometry` | px4_msgs/VehicleOdometry | GT as EV, NED/FRD, 50 Hz |
| `/fmu/out/vehicle_status_v1`, `/fmu/out/vehicle_local_position` | px4_msgs | v1.16 appends `_vN` when MESSAGE_VERSION > 0 (VehicleStatus = 1); QoS best effort + transient local |

## TF tree
```
world -> base_link (dynamic, GT odom)         base_link = CoM
base_link -> com_link, camera_link, imu_ideal_link, imu_noisy_link, uwb_tag_link
camera_link -> camera_optical_frame
world -> uwb_anchor_1..8, apriltag_gt_<id>    (static; GT, world-aligned axes)
camera_optical_frame -> apriltag_det_<id>     (apriltag_ros, estimated)
```

## Frames
World ENU (X along the track, Y left, Z up); body FLU. Camera mount in `base_link`: xyz (0,0,-0.06), rpy (0, pi/2, pi/2):
looks along -Z, image right = world +X, image up = world +Y. All mounts are relative to the actual x500 CoM
(`base_link`, identity inertial pose, mass 2.0 kg). Tag k: x = k-6 (= track-frame x of `tag_layout.csv` - 5), left y = +0.20 (id 2(k-1)), right y = -0.20 (id 2(k-1)+1);
70 mm black square, 87.5 mm including the white border; artwork up along world +X (see `texture_yaw_offset_rad` and `progress.md`).
UWB: 8 anchors at (+/-6, +/-6, 0) and (+/-6, +/-6, 12) (cube corners, floor face centred on the track); Gauss-Newton solver `uwb_solver.py`.

## Configs (`ros2_ws/src/drone_sim/config/`)
`world.yaml` floor, physics, lighting, magnetic field | `apriltags.yaml` family, sizes, texture yaw offset, detector params, 22 tags |
`uwb.yaml` anchors, noise, `multilateration` section | `drone.yaml` x500 facts, CoM, mounts, spawn, odometry |
`camera.yaml` | `imu.yaml` | `controller.yaml` mission rate, `px4` topics/QoS/commands, `ev` bridge | `trajectory.yaml`.
PX4 parameters: `docker/airframes/4099_gz_x500_tagcam` (x500 + EV fusion, no GPS/baro/mag, no RC).
The YAMLs are copied into `install/` even with `--symlink-install`: re-run `colcon build` after editing them.

## Limitations
- 1920x1080 camera renders at ~19 Hz headless (~11 Hz with GUI) on an RTX 3060 Laptop, below the configured 30 Hz.
- Control is idealised: PX4's estimator is fed ground truth (no UWB/tag/IMU estimation), so the sensors are for data collection and downstream estimators.
- The OdometryPublisher velocity is finite-differenced and rolling-mean smoothed (small lag); EKF2_EV_DELAY is set to 10 ms.
- x500 geometry (legs, battery) may be partly visible at the image edges; it was not rendered or checked.
- Carrier links for the camera/IMUs add 3 x 1 g at the CoM region (negligible, not retuned).
- If PX4 refuses to arm, check `/fmu/out/vehicle_status_v1` / `/fmu/out/failsafe_flags` flags and `ros2 topic echo /fmu/out/estimator_status_flags`.
- Delete `build/px4_sitl_default/rootfs/parameters.bson` (launch does it with `reset_params:=true`) after changing the airframe.
- No lens distortion, rolling shutter, motion blur, RF or multipath effects.

**Tested 2026-10-25** (see `progress.md`, v3).

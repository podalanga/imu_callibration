# Progress notes (v2: PX4 SITL + Gazebo Harmonic + ROS 2 Jazzy, Docker)

Status: v3 tested end to end (see the v3 section at the bottom). Earlier notes below are kept for reference.
built, nothing launched). Only the base-image tag render check was run (below). Pure-Python pieces (`uwb_solver.py`, `frames.py`,
`world_gen.py` output well-formedness) were exercised inside the base container.

## Done
- `docker/`: multi-stage Dockerfile (`base`, `px4`), docker-compose.yml (services `sim`, `base`), build.sh, run.sh,
  `airframes/4099_gz_x500_tagcam`.
- `drone_sim`: world_gen (x500_tagcam model dir + world + bridge + apriltag params), `px4_mission_node`, `gt_to_px4_ev_node`,
  `uwb_multilateration_node` + `uwb_solver.py`, `frames.py` (ENU/NED), launch with `px4:=true|false`. `trajectory_controller_node.py` removed.
- px4_msgs not vendored: cloned (`release/1.16`) and built into `/opt/px4_ws` in the `px4` image stage.

## Research findings (read from upstream source)
A. **sdformat 1.9 `camera.sdf`**: `<camera_info_topic>` is a child of `<camera>` (not of `<sensor>`). gz-sensors `Sensor.cc` reads `<gz_frame_id>`
   as a child of `<sensor>`; the camera header frame_id comes from it. Used as in v1.
B. **gz-sim8 OdometryPublisher.cc**: pose is the *model* world pose (`worldPose(model)`), absolute world, times `<xyz_offset>/<rpy_offset>`
   (so the CoM is obtained with `xyz_offset = (0,0,0.24)`); twist is in the body frame (3D: displacement rotated by `RotateVectorReverse`, angular in body),
   finite-difference + rolling mean. Params: `odom_frame`, `robot_base_frame` (header frame_id / child_frame_id strings only), `dimensions`, `xyz_offset`,
   `rpy_offset`, `gaussian_noise`, `odom_publish_frequency`, `odom_topic`, `odom_covariance_topic`, `tf_topic`.
C. **apriltag_ros 3.2.2 (christianrauch, `AprilTagNode.cpp`)**: params `family` ("36h11"), `size`, `max_hamming`, `detector.{threads,decimate,blur,refine,sharpening,debug}`,
   `pose_estimation_method` ("pnp"/"homography"), `tag.ids`, `tag.frames`, `tag.sizes`, `profile`, `image_transport`. Subscribes `image_rect` and `camera_info`
   (image_transport camera subscriber), publishes `detections`; unknown ids are dropped when `tag.frames` is set. Generated params now follow this. Verified live in the render check.
D. **PX4 v1.16.2** (latest 1.16.x tag):
   - `dds_topics.yaml`: unversioned names `/fmu/in/offboard_control_mode`, `/fmu/in/trajectory_setpoint`, `/fmu/in/vehicle_command`, `/fmu/in/vehicle_visual_odometry`,
     `/fmu/out/vehicle_status`, `/fmu/out/vehicle_local_position`, `/fmu/out/vehicle_odometry` (no `_v1` suffix in 1.16).
   - Offboard example QoS: best effort (reliability), transient local durability (subscriptions), keep last 1.
   - `px4-rc.gzsim`: `PX4_GZ_STANDALONE` skips starting gz and waits for `/world/$PX4_GZ_WORLD/scene/info`; the world name is discovered from the `/world/*/clock` topic;
     `PX4_GZ_MODEL_NAME` set -> `gz_bridge start -w <world> -n <model>` attaches to the existing model (no spawn); `PX4_GZ_MODELS/WORLDS` only used when PX4 spawns/starts gz.
     `PX4_SYS_AUTOSTART` overrides the airframe id; else `PX4_SIM_MODEL` is matched against `^[0-9]+_<model>$` in the airframes dir.
   - `4001_gz_x500`: sources `rc.mc_defaults`, sets `SIM_GZ_EN`, CA_ROTOR0..3 geometry, `SIM_GZ_EC_FUNC/MIN/MAX`, `MPC_THR_HOVER 0.60`, `NAV_DLL_ACT 2`. 4099 sources it and overrides.
   - Airframes are **not** globbed: `ROMFS/.../init.d-posix/airframes/CMakeLists.txt` lists each file explicitly, so the Dockerfile copies 4099 into the source tree and `sed`-inserts it after `4001_gz_x500` before building.
   - EKF2 params (v1.16 yaml): `EKF2_EV_CTRL` bitmask (0 hpos, 1 vpos, 2 3D vel, 3 yaw) -> 15; `EKF2_HGT_REF` 3 = vision; `EKF2_GPS_CTRL` bitmask -> 0; `EKF2_BARO_CTRL` is a boolean -> 0;
     `EKF2_MAG_TYPE` 5 = none; `EKF2_EV_DELAY` ms; `EKF2_EVP_NOISE`, `EKF2_EVV_NOISE`, `EKF2_EVA_NOISE`, `EKF2_EV_NOISE_MD`. No-RC: `COM_RC_IN_MODE` 4 (ignore sticks), `NAV_RCL_ACT`, `COM_RCL_EXCEPT` bit 2 = offboard,
     `COM_ARM_WO_GPS` (default 1), `NAV_DLL_ACT 0` (4001 sets 2), `COM_OF_LOSS_T`.
   - `Tools/setup/ubuntu.sh` flags: `--no-nuttx`, `--no-sim-tools` (the latter would add OSRF `gz-harmonic`; skipped since ros_gz provides Gazebo). It calls `sudo` and `lsb_release`.
   - `gz_bridge` CMake: `find_package(gz-transport NAMES gz-transport14 gz-transport13)`; the ROS vendor packages provide gz-transport13/gz-msgs10 under `/opt/ros/jazzy/opt/*_vendor` (checked in the base image), so the Dockerfile passes them via `CMAKE_PREFIX_PATH`.
   - SITL launch convention (`sitl_multiple_run.sh`): `px4 -i <n> -d <build>/etc`, run from an instance working dir. The launch uses `build/px4_sitl_default/rootfs`.
E. **PX4-gazebo-models main**: `x500` = `<include merge=true>model://x500_base` + 4 `MulticopterMotorModel` plugins (`command/motor_speed`, motors 0..3, `MotorFailurePlugin`);
   `x500_base`: model `<pose>0 0 0.24</pose>`, link `base_link` mass 2.0, inertia 0.0217/0.0217/0.04, **no inertial pose** (CoM = base_link origin), rotors `rotor_0..3`,
   sensors `air_pressure_sensor`, `magnetometer_sensor`, `imu_sensor` (250 Hz, `gz_frame_id base_link`), `navsat_sensor`. `x500_mono_cam_down` = `<include merge=true>x500` + `mono_cam` include + fixed `CameraJoint`
   (parent `base_link`). `x500_tagcam` follows that pattern (merge include, carrier links `relative_to="base_link"`, fixed joints). The PX4 default world also needs the AirPressure/Magnetometer/NavSat/Imu systems and `<magnetic_field>` + `<spherical_coordinates>`, now generated.
F. **px4_msgs release/1.16**: `VehicleOdometry`: timestamp, timestamp_sample (us), `pose_frame` (UNKNOWN 0, NED 1, FRD 2), `position[3]`, `q[4]` (FRD body -> reference, w first), `velocity_frame` (NED 1, FRD 2, BODY_FRD 3), `velocity[3]`,
   `angular_velocity[3]`, `position/orientation/velocity_variance[3]`, `reset_counter`, `quality`. `TrajectorySetpoint`: timestamp, `position`, `velocity`, `acceleration`, `jerk`, `yaw`, `yawspeed`, NED, NaN = not controlled.

## Render check (the only runtime used)
- Image: `docker build --target base -t drone_sim:base simulation/docker` (built OK; Gazebo is `gz-sim 8.15.0` from the ROS vendor packages).
- Test world generated by the same `world_gen._tags/_floor/_world_header` (tags id0 at (0,+0.25), id1 at (0,-0.25)), static camera at z = 1.10 with the v1 mount rpy (0, pi/2, pi/2), hfov 1.1906, 1280x720.
  `gz sim -s -r --headless-rendering` + `ros_gz_bridge` (image, camera_info) + `apriltag_node` with the generated params; GPU via `--gpus all`.
- Result: **both ids 0 and 1 detected** (hamming 0, decision margin ~200), tags at image y = 145 (id0, +Y side, top) and 575 (id1), x = 640, ~59.4 px side (expected 60 px). Image right = world +X, image up = world +Y.
  Not mirrored (homography determinant > 0).
- With `texture_yaw_offset_rad = 0` the artwork-up (apriltag corners p0,p1 = bottom edge, p2,p3 = top edge) pointed to image left = world **-X**. With `texture_yaw_offset_rad = pi` the corner order flips and the artwork-up points to image right = world **+X**; the setting is now `pi` in `apriltags.yaml`. No OBJ/DAE quad needed (no mirroring).
- Containers were `--rm` (none left). The test files are in the session scratchpad only.

## Open risks (check first when running)
1. PX4 build inside Docker is untested: `gz_bridge` must find gz-transport13/gz-msgs10 from the ROS vendor prefixes; at run time the `px4` binary needs `LD_LIBRARY_PATH` from `source /opt/ros/jazzy/setup.bash` (done by the launch environment).
   If the build fails, fallback: add the OSRF apt repo and install `gz-harmonic` (what `ubuntu.sh` does without `--no-sim-tools`) before the PX4 build.
2. `px4` command line in `sim.launch.py` (`-i 0 -d <build>/etc`, cwd `rootfs`) follows `sitl_multiple_run.sh`; confirm `etc/init.d-posix/rcS` is found, otherwise add `-s etc/init.d-posix/rcS`.
3. OffboardControlMode with position+velocity+acceleration all true: PX4 accepts it as feedforward; if it rejects the combination, drop `acceleration`.
4. EKF2 must reach `xy_valid/z_valid` on EV before arming; the mission node waits for it (`require_local_position`). If PX4 timestamps differ from sim time, EV messages are dropped (`EKF2_EV_DELAY`, timesync).
5. x500 `MotorFailurePlugin` is loaded from `GZ_SIM_SYSTEM_PLUGIN_PATH` (PX4 build tree).
6. Merged-include frame names (`base_link`) and `relative_to="base_link"` for the added links follow `x500_mono_cam_down`, not run.
7. Camera 6 cm below the CoM: x500 legs may be in view at the edges (not rendered).
8. OdometryPublisher velocity is smoothed finite difference; yaw-rate/accel feedforward timing matches PX4's loops only approximately.
9. Spawn: model origin z = 0 puts the CoM at 0.24 m; skid contact not checked.

## v3 (2026-10-25): recentred world, UWB cube, slow 2.5-round flight, Nord 4 camera — tested
- World origin = track centre = centre of the UWB cube's floor face. Floor 20 x 20 m at the origin = Gazebo GUI default grid
  (was 30 x 12 at x = 5, misaligned with the grid). Tags x = k-6 (track-frame x - 5), y = +/-0.20 (matches `tag_layout.csv`; was 0.25).
- UWB: 8 anchors on the corners of a 12 x 12 x 12 m cube, (+/-6, +/-6, {0, 12}); ceiling markers hang below their point.
- Trajectory: x = -5.75 <-> 5.75, 5 traversals (2.5 rounds), v = 0.3 m/s, a_max = 0.3, dwell 0, 5 s settle hover after takeoff.
- Camera: OnePlus Nord 4 main camera (25 mm equiv., 4:3 sensor, 16:9 video) -> 1920x1080, hfov 1.2112 rad.
- Bug fix: PX4 v1.16 publishes VehicleStatus (MESSAGE_VERSION 1) on `/fmu/out/vehicle_status_v1`; the mission node listened on
  `/fmu/out/vehicle_status` and never armed. Configs are copied into install/ even with --symlink-install: rebuild after YAML edits.
- Test run (headless, own Docker network so no other gz/PX4 instance interferes): arms, takes off, completes 5/5 traversals in ~200 s;
  during traversals camera height 1.095-1.100 m, median speed 0.300 m/s (90 % of samples within 0.29-0.31), |y| <= 2 mm,
  reference error max 4.4 cm; AprilTags detected; UWB fix within a few cm of GT. GUI screenshot: grid coincides with the floor.
- Limitation: 1920x1080 rendering reaches ~19 Hz headless / ~11 Hz with the GUI on an RTX 3060 Laptop (configured 30 Hz).
- Running two sims with host networking (docker-compose `network_mode: host`) at once mixes their gz/PX4 topics.

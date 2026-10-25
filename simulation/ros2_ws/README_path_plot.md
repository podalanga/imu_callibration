# Live ESKF Path Plotting in Gazebo and RViz


# TLDR
 - run "./run.sh"
 - in another terminal run "/run.sh exec python3 /ws/eskf_all_in_one.py"
 (This is all you basically need to run the path plotter in gazebo)

## 1. Overview

This project extends the existing PX4 SITL, Gazebo and ROS 2 drone simulation with live position estimation and trajectory visualization.

The setup runs the live Error-State Kalman Filter (ESKF), displays the estimated and ground-truth trajectories in Gazebo, and visualizes the corresponding paths in RViz.

The trajectory plotting is an additional visualization feature. It does not replace or modify the ESKF's estimation mathematics.

## 2. Demonstration Recordings

Screen recordings demonstrating the trajectory visualization are included with this project.

- **Gazebo:** Shows the drone simulation with the estimated and ground-truth trajectories.
- **RViz:** Shows the corresponding trajectory visualization using ROS 2 data.

Keep both recordings in the repository when pushing your changes to GitHub.

For example, if your recordings are stored in a `recordings/` directory, add links like these after replacing the filenames with your actual recording names:

- [Watch Gazebo path plotting](recordings/gazebo_path_plotting.mp4)
- [Watch RViz path plotting](recordings/rviz_path_plotting.mp4)

These filenames are examples. Update the links to match the actual recording files in the repository.

## 3. Additional Files

The following files support the live path plotting workflow.

| File | Purpose |
|---|---|
| `eskf_live_node.py` | Runs the live ESKF estimator and publishes estimated position. |
| `eskf_live_rviz.rviz` | Stores the RViz visualization configuration. |
| `eskf_gz_overlay.py` | Provides the Gazebo trajectory overlay as a standalone script. |
| `eskf_all_in_one.py` | Launches the estimator, Gazebo overlay and RViz together. |
| `README_path_plotting.md` | Documents the setup, commands and visualization workflow. |

The files must be committed to GitHub to make them available after a fresh clone.

The combined launcher expects the ESKF script and RViz configuration at:

```text
/ws/eskf_live_node.py
/ws/eskf_live_rviz.rviz
```

The combined launcher is expected at:

```text
/ws/eskf_all_in_one.py
```

The standalone Gazebo overlay can be kept at `/ws/eskf_gz_overlay.py`.

If you change the filenames or directory structure, update the paths in the Python launcher accordingly.

## 4. Environment

The documented setup assumes:

- Ubuntu
- Docker
- ROS 2 Jazzy
- PX4 SITL
- Gazebo Harmonic / Gazebo Sim 8
- The existing `drone_sim` container
- The repository's Docker wrapper at `simulation/docker/run.sh`

The repository path used in the commands below is:


## 5. ROS 2 Topics

The visualization uses the following topics.

| Topic | Message type | Purpose |
|---|---|---|
| `/ground_truth/odom` | `nav_msgs/msg/Odometry` | Gazebo ground-truth position |
| `/eskf/odometry` | `nav_msgs/msg/Odometry` | ESKF estimated position |
| `/imu/ideal` | `sensor_msgs/msg/Imu` | Ideal IMU input, if used by the estimator |
| `/imu/noisy` | `sensor_msgs/msg/Imu` | Noisy IMU input, if used by the estimator |
| `/uwb/position` | `geometry_msgs/msg/PoseWithCovarianceStamped` | UWB position input |
| `/uwb/ranges` | `drone_sim_interfaces/msg/UwbRangeArray` | UWB range input |

The exact sensor inputs depend on the version of `eskf_live_node.py` being used.

## 6. Setup After Cloning the Repository

### 6.1 Add the additional files

Place the plotting scripts and RViz configuration in the repository, keeping the filenames consistent with the launcher.

An example layout is:

```text
imu_callibration/
├── eskf_live_node.py
├── eskf_live_rviz.rviz
├── eskf_gz_overlay.py
├── eskf_all_in_one.py
├── README_path_plotting.md
├── recordings/
│   ├── gazebo_path_plotting.mp4
│   └── rviz_path_plotting.mp4
└── simulation/
    └── docker/
        └── run.sh
```

The recording filenames are examples. Use the actual names of your screen recordings.

If the ESKF node already exists elsewhere in the repository, use the working version rather than creating a duplicate implementation.

### 6.2 Make the Docker wrapper executable

From the repository root:

```bash
chmod +x simulation/docker/run.sh
```

### 6.3 Copy files into the container if necessary

If the repository is not already mounted into `/ws`, copy the files into the running container.

Run these commands from the repository root:

```bash
docker cp eskf_live_node.py drone_sim:/ws/eskf_live_node.py
docker cp eskf_live_rviz.rviz drone_sim:/ws/eskf_live_rviz.rviz
docker cp eskf_gz_overlay.py drone_sim:/ws/eskf_gz_overlay.py
docker cp eskf_all_in_one.py drone_sim:/ws/eskf_all_in_one.py
```

If the files are already available inside `/ws` through a Docker mount, these copy commands are unnecessary.

## 7. Running the Path Plotting

### Step 1: Start the simulation

Start the existing PX4 and Gazebo simulation using the project's normal startup procedure.

Wait until Gazebo is running and the required ROS 2 topics are publishing data.

### Step 2: Launch the combined visualization

Open a terminal on the host and run:

(in the simulation/docker directory)
```bash
./run.sh exec python3 /ws/eskf_all_in_one.py
```

This launches the ESKF node, Gazebo trajectory overlay and RViz together, assuming the simulation container and required topics are already running.

Use `Ctrl+C` to stop the combined launcher. Stop any leftover processes before launching it again to avoid duplicate estimators or marker publishers.

## 8. Understanding the Visualization

- **Blue sphere and line:** ESKF estimated position and trajectory.
- **Green sphere and line:** Gazebo ground-truth position and trajectory.
- **Spheres:** Show the latest positions.
- **Lines:** Show the accumulated position histories.

The frame-safe Gazebo overlay anchors each line marker at its first recorded point and expresses the remaining vertices relative to that anchor. This is intended to avoid ambiguity between the marker pose and line-point coordinates.

The sphere size and line scale can be adjusted near the top of `eskf_all_in_one.py`.

## 9. Troubleshooting

### RViz works, but Gazebo paths do not appear

Check whether both odometry topics publish data:

```bash
./run.sh exec ros2 topic list
./run.sh exec ros2 topic echo /eskf/odometry --once
./run.sh exec ros2 topic echo /ground_truth/odom --once
```

If RViz shows both trajectories but Gazebo does not, investigate the Gazebo marker overlay and its coordinate handling before modifying the ESKF.

### Only spheres appear

Check that the launcher includes both line markers and spheres in its marker-array updates, and that at least two trajectory points have been collected.

### The trajectory is offset or appears in the wrong plane

Check that the ESKF and ground-truth positions use the same world coordinate convention. Ensure the overlay does not mix body-frame coordinates with world-frame coordinates.

### Inspect the Gazebo marker service

```bash
./run.sh exec gz service -l
./run.sh exec gz service -i -s /marker_array
```

The tested overlay used Gazebo's `/marker_array` service with `gz.msgs.Marker_V` requests and `gz.msgs.Boolean` responses. Service availability can depend on the Gazebo version and running simulation.

### The launcher cannot find a file

Check the file locations referenced by `ESKF_SCRIPT` and `RVIZ_CONFIG` near the top of the launcher.


## 10. Notes

- Do not run multiple copies of the ESKF node.
- Do not run multiple copies of the Gazebo overlay.
- Confirm that all required scripts and RViz configuration files are committed to GitHub.
- Keep the recording links consistent with the actual filenames and paths.
- Verify the topic names, Docker mounts and Gazebo marker service against the version of the project being used.

This document describes the path plotting workflow and the commands used to run and publish the additional files.

LIVE ESKF + ROS 2 / GAZEBO DRONE SIM
====================================

Files
-----
eskf_live_node.py     Live 15-state ESKF node
eskf_live_rviz.rviz   RViz2 view with estimated and ground-truth trajectories

Copy both files into the host folder:
  simulation/ros2_ws/

That folder is mounted inside the running container as /ws according to the setup guide.

Run from simulation/docker in terminal 1 (keep the simulation running):
  ./run.sh

If it is already running, do not start a second simulation. In terminal 2:
  ./run.sh exec python3 /ws/eskf_live_node.py

The node subscribes to:
  /imu/noisy
  /uwb/ranges
  /ground_truth/odom (evaluation/visualization only; never used to correct the filter)

It publishes:
  /eskf/odometry
  /eskf/estimated_path
  /eskf/ground_truth_path
  /eskf/estimated_line
  /eskf/ground_truth_line

RViz2
-----
Open a third terminal and try:
  ./run.sh exec rviz2 -d /ws/eskf_live_rviz.rviz

This depends on the container image having rviz2 installed and GUI forwarding enabled. If that command
cannot open a GUI, the ESKF node and topics can still run; use the project's GUI/display setup or a
host RViz2 connected to the same ROS_DOMAIN_ID/network.

Quick checks:
  ./run.sh exec ros2 topic echo /eskf/odometry --once
  ./run.sh exec ros2 topic info /eskf/estimated_path

Important notes
---------------
- Anchor coordinates and IDs are copied from the supplied uwb.yaml: x/y = +/-6 m, z = 0 or 12 m.
- The filter initializes from the first usable UWB range array, using the configured 1.16 m low-height
  seed to resolve the cube's vertical mirror ambiguity.
- IMU units are expected to be rad/s and m/s^2, as provided by sensor_msgs/Imu.
- The filter does not use the IMU message's orientation field or ground truth as estimator inputs.
- This is a research/simulation implementation, not flight-certified navigation software.
- The filter has no magnetometer topic in this simulation input set, so yaw is initialized to zero and
  has no direct absolute heading correction. This can affect position during turns.
- If imports fail, check that the container has numpy installed and that drone_sim_interfaces was built
  in the ROS workspace (it already appears in the supplied topic message type).

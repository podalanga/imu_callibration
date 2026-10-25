#!/usr/bin/env python3
"""One-command live ESKF + frame-safe Gazebo trajectory overlay + RViz launcher."""

import math
import signal
import subprocess
import sys
import time
from collections import deque

import rclpy
from rclpy.node import Node as RosNode
from nav_msgs.msg import Odometry
from gz.msgs10.marker_pb2 import Marker
from gz.msgs10.marker_v_pb2 import Marker_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node as GzNode

# ------------------------- USER-EDITABLE SETTINGS -------------------------
SPHERE_DIAMETER = 0.04        # metres
LINE_THICKNESS = 1.0          # Gazebo marker scale; 1.0 restores the known-visible setting
MAX_PATH_POINTS = 800
MARKER_UPDATE_SECONDS = 0.25
ESKF_SCRIPT = "/ws/eskf_live_node.py"
RVIZ_CONFIG = "/ws/eskf_live_rviz.rviz"
# -------------------------------------------------------------------------


class GazeboOverlay(RosNode):
    def __init__(self):
        super().__init__("eskf_gazebo_overlay")
        self.eskf_path = deque(maxlen=MAX_PATH_POINTS)
        self.truth_path = deque(maxlen=MAX_PATH_POINTS)
        self.eskf_pos = None
        self.truth_pos = None
        self.last_eskf_sample = 0.0
        self.last_truth_sample = 0.0
        self.last_warning = 0.0
        self.gz_node = GzNode()
        self.create_subscription(Odometry, "/eskf/odometry", self.eskf_callback, 10)
        self.create_subscription(Odometry, "/ground_truth/odom", self.truth_callback, 10)
        self.create_timer(MARKER_UPDATE_SECONDS, self.update_markers)
        self.get_logger().info("BLUE = ESKF; GREEN = ground truth.")
        self.get_logger().info(
            f"Sphere diameter={SPHERE_DIAMETER} m; line width={LINE_THICKNESS} m."
        )

    @staticmethod
    def get_position(msg):
        p = msg.pose.pose.position
        xyz = (float(p.x), float(p.y), float(p.z))
        return xyz if all(math.isfinite(v) for v in xyz) else None

    @staticmethod
    def record(path, position, last_time, now):
        if position is None or now - last_time < 0.10:
            return last_time
        if not path:
            path.append(position)
            return now
        prev = path[-1]
        distance2 = sum((position[i] - prev[i]) ** 2 for i in range(3))
        if distance2 >= 0.0025 or now - last_time >= 0.5:
            path.append(position)
            return now
        return last_time

    def eskf_callback(self, msg):
        now = time.monotonic()
        p = self.get_position(msg)
        if p is not None:
            self.eskf_pos = p
            self.last_eskf_sample = self.record(self.eskf_path, p, self.last_eskf_sample, now)

    def truth_callback(self, msg):
        now = time.monotonic()
        p = self.get_position(msg)
        if p is not None:
            self.truth_pos = p
            self.last_truth_sample = self.record(self.truth_path, p, self.last_truth_sample, now)

    @staticmethod
    def set_material(marker, rgb):
        for name in ("diffuse", "ambient", "emissive"):
            color = getattr(marker.material, name)
            factor = 0.25 if name == "emissive" else 1.0
            color.r = rgb[0] * factor
            color.g = rgb[1] * factor
            color.b = rgb[2] * factor
            color.a = 1.0

    @classmethod
    def make_line(cls, marker_id, points, rgb):
        marker = Marker()
        marker.ns = "eskf_comparison"
        marker.id = marker_id
        marker.action = Marker.ADD_MODIFY
        marker.type = Marker.LINE_STRIP
        # Gazebo applies the marker pose to LINE_STRIP points. Anchor the
        # marker at the first world point and express every vertex relative to
        # that anchor. This avoids relying on implicit world/local-frame
        # behaviour in Gazebo's marker renderer.
        anchor_x, anchor_y, anchor_z = points[0]
        marker.pose.position.x = anchor_x
        marker.pose.position.y = anchor_y
        marker.pose.position.z = anchor_z
        marker.pose.orientation.w = 1.0

        # Keep Gazebo's known-visible LINE_STRIP scale. A tiny scale can
        # collapse/shrink the rendered path in some Gazebo marker renderers.
        marker.scale.x = LINE_THICKNESS
        marker.scale.y = 1.0
        marker.scale.z = 1.0
        cls.set_material(marker, rgb)
        for x, y, z in points:
            point = marker.point.add()
            point.x = x - anchor_x
            point.y = y - anchor_y
            point.z = z - anchor_z
        return marker

    @classmethod
    def make_sphere(cls, marker_id, position, rgb):
        marker = Marker()
        marker.ns = "eskf_comparison"
        marker.id = marker_id
        marker.action = Marker.ADD_MODIFY
        marker.type = Marker.SPHERE
        marker.pose.position.x = position[0]
        marker.pose.position.y = position[1]
        marker.pose.position.z = position[2]
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = SPHERE_DIAMETER
        cls.set_material(marker, rgb)
        return marker

    def warn_throttled(self, message):
        now = time.monotonic()
        if now - self.last_warning > 5.0:
            self.get_logger().warning(message)
            self.last_warning = now

    def update_markers(self):
        # Send paths and spheres together every time. This avoids paths being
        # cleared by marker-array implementations when a sphere-only batch arrives.
        markers = []
        if len(self.eskf_path) >= 2:
            markers.append(self.make_line(1, list(self.eskf_path), (0.10, 0.35, 1.0)))
        if len(self.truth_path) >= 2:
            markers.append(self.make_line(2, list(self.truth_path), (0.10, 1.0, 0.20)))
        if self.eskf_pos is not None:
            markers.append(self.make_sphere(3, self.eskf_pos, (0.10, 0.35, 1.0)))
        if self.truth_pos is not None:
            markers.append(self.make_sphere(4, self.truth_pos, (0.10, 1.0, 0.20)))
        if not markers:
            return

        batch = Marker_V()
        for marker in markers:
            batch.marker.add().CopyFrom(marker)
        try:
            ok, response = self.gz_node.request(
                "/marker_array", batch, Marker_V, Boolean, 500
            )
            if not ok or not response.data:
                self.warn_throttled("Gazebo did not confirm the marker-array request.")
        except Exception as exc:
            self.warn_throttled(f"Gazebo marker update failed: {exc}")


def main():
    children = []

    def stop_children(*_):
        for proc in children:
            if proc.poll() is None:
                proc.terminate()

    signal.signal(signal.SIGTERM, stop_children)
    try:
        print("[Launcher] Starting live ESKF estimator...", flush=True)
        estimator = subprocess.Popen([sys.executable, ESKF_SCRIPT])
        children.append(estimator)
        time.sleep(3.0)
        if estimator.poll() is not None:
            raise RuntimeError(
                f"ESKF process exited with code {estimator.returncode}; see output above."
            )

        print("[Launcher] Starting RViz...", flush=True)
        children.append(subprocess.Popen(["rviz2", "-d", RVIZ_CONFIG]))

        print("[Launcher] Starting Gazebo trajectory overlay...", flush=True)
        rclpy.init()
        overlay = GazeboOverlay()
        try:
            rclpy.spin(overlay)
        finally:
            overlay.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[Launcher] ERROR: {exc}", file=sys.stderr, flush=True)
    finally:
        for proc in children:
            if proc.poll() is None:
                proc.terminate()
        deadline = time.monotonic() + 3.0
        for proc in children:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=max(0.1, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()

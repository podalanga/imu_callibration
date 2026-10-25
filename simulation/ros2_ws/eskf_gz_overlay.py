#!/usr/bin/env python3

from collections import deque
import math
import time

import rclpy
from rclpy.node import Node as RosNode
from nav_msgs.msg import Odometry

from gz.msgs10.marker_pb2 import Marker
from gz.msgs10.marker_v_pb2 import Marker_V
from gz.msgs10.boolean_pb2 import Boolean
from gz.transport13 import Node as GzNode


class ESKFOverlay(RosNode):
    def __init__(self):
        super().__init__("eskf_gazebo_overlay")

        self.eskf_path = deque(maxlen=1500)
        self.truth_path = deque(maxlen=1500)

        self.eskf_pos = None
        self.truth_pos = None
        self.last_eskf_sample = 0.0
        self.last_truth_sample = 0.0
        self.last_path_update = 0.0
        self.last_warning = 0.0

        self.gz_node = GzNode()

        self.create_subscription(
            Odometry, "/eskf/odometry", self.eskf_callback, 10
        )
        self.create_subscription(
            Odometry, "/ground_truth/odom", self.truth_callback, 10
        )

        # Update moving position markers at 5 Hz.
        self.create_timer(0.2, self.update_markers)

        self.get_logger().info(
            "Gazebo overlay active: blue = ESKF, green = ground truth."
        )

    @staticmethod
    def position(msg):
        p = msg.pose.pose.position
        result = (float(p.x), float(p.y), float(p.z))
        if not all(math.isfinite(v) for v in result):
            return None
        return result

    @staticmethod
    def add_path_sample(path, position, last_sample, now):
        # Avoid filling the path with nearly identical high-rate samples.
        if position is None or now - last_sample < 0.10:
            return last_sample

        if not path:
            path.append(position)
            return now

        previous = path[-1]
        distance_sq = sum(
            (position[i] - previous[i]) ** 2 for i in range(3)
        )

        # Add a point after meaningful movement or a short interval.
        if distance_sq >= 0.0025 or now - last_sample >= 0.5:
            path.append(position)
            return now

        return last_sample

    def eskf_callback(self, msg):
        now = time.monotonic()
        position = self.position(msg)
        if position is None:
            return

        self.eskf_pos = position
        self.last_eskf_sample = self.add_path_sample(
            self.eskf_path, position, self.last_eskf_sample, now
        )

    def truth_callback(self, msg):
        now = time.monotonic()
        position = self.position(msg)
        if position is None:
            return

        self.truth_pos = position
        self.last_truth_sample = self.add_path_sample(
            self.truth_path, position, self.last_truth_sample, now
        )

    @staticmethod
    def set_material(marker, rgb):
        material = marker.material
        material.diffuse.r = rgb[0]
        material.diffuse.g = rgb[1]
        material.diffuse.b = rgb[2]
        material.diffuse.a = 1.0

        material.ambient.r = rgb[0]
        material.ambient.g = rgb[1]
        material.ambient.b = rgb[2]
        material.ambient.a = 1.0

        material.emissive.r = rgb[0] * 0.25
        material.emissive.g = rgb[1] * 0.25
        material.emissive.b = rgb[2] * 0.25
        material.emissive.a = 1.0

    @classmethod
    def make_line(cls, marker_id, points, rgb):
        marker = Marker()
        marker.ns = "eskf_comparison"
        marker.id = marker_id
        marker.action = Marker.ADD_MODIFY
        marker.type = Marker.LINE_STRIP

        # CRITICAL: keep path coordinates in their original world scale.
        marker.pose.orientation.w = 1.0
        marker.scale.x = 1.0
        marker.scale.y = 1.0
        marker.scale.z = 1.0

        cls.set_material(marker, rgb)

        for x, y, z in points:
            point = marker.point.add()
            point.x = x
            point.y = y
            point.z = z

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

        marker.scale.x = 0.22
        marker.scale.y = 0.22
        marker.scale.z = 0.22

        cls.set_material(marker, rgb)
        return marker

    def send_markers(self, markers):
        if not markers:
            return

        batch = Marker_V()
        for marker in markers:
            batch.marker.add().CopyFrom(marker)

        try:
            success, response = self.gz_node.request(
                "/marker_array", batch, Marker_V, Boolean, 250
            )

            if not success or not response.data:
                now = time.monotonic()
                if now - self.last_warning > 5.0:
                    self.get_logger().warning(
                        "Gazebo did not accept the marker batch. "
                        "Check the overlay service error if this continues."
                    )
                    self.last_warning = now

        except Exception as exc:
            now = time.monotonic()
            if now - self.last_warning > 5.0:
                self.get_logger().error(f"Gazebo marker update failed: {exc}")
                self.last_warning = now

    def update_markers(self):
        markers = []
        now = time.monotonic()

        # Update paths once per second instead of resending thousands of
        # points on every moving-marker update.
        if now - self.last_path_update >= 1.0:
            if len(self.eskf_path) >= 2:
                markers.append(
                    self.make_line(
                        1, list(self.eskf_path), (0.10, 0.35, 1.0)
                    )
                )

            if len(self.truth_path) >= 2:
                markers.append(
                    self.make_line(
                        2, list(self.truth_path), (0.10, 1.0, 0.20)
                    )
                )

            self.last_path_update = now

        # Send both current-position markers together.
        if self.eskf_pos is not None:
            markers.append(
                self.make_sphere(3, self.eskf_pos, (0.10, 0.35, 1.0))
            )

        if self.truth_pos is not None:
            markers.append(
                self.make_sphere(4, self.truth_pos, (0.10, 1.0, 0.20))
            )

        self.send_markers(markers)


def main():
    rclpy.init()
    node = ESKFOverlay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

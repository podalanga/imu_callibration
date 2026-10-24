"""Ground-truth odometry -> PX4 external vision (VehicleOdometry on /fmu/in/vehicle_visual_odometry).

Input : nav_msgs/Odometry, world ENU, pose of the CoM, twist in the FLU body frame (gz OdometryPublisher).
Output: px4_msgs/VehicleOdometry, pose_frame NED, q = FRD body in NED, velocity in NED.
PX4's local origin therefore equals the world origin (NED local frame = world frame rotated).
Timestamps are simulation-time microseconds (PX4 SITL with gz runs on the gz clock).
"""
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from px4_msgs.msg import VehicleOdometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from drone_sim.config_loader import load_all
from drone_sim.frames import enu_to_ned, flu_to_frd, quat_enu_flu_to_ned_frd, quat_xyzw_to_matrix


class GtToPx4Ev(Node):
    def __init__(self):
        super().__init__('gt_to_px4_ev')
        self.declare_parameter('config_dir', '')
        cfg = load_all(self.get_parameter('config_dir').value)
        c = cfg['controller']
        self.ev = c['ev']
        self.twist_in_body = bool(c['odom_twist_in_body'])
        self.period_ns = int(1e9 / float(self.ev['rate_hz']))
        self.last_pub_ns = -10 ** 18
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self.pub = self.create_publisher(VehicleOdometry, c['px4']['visual_odometry_topic'], qos)
        self.create_subscription(Odometry, c['odom_topic'], self.on_odom, 20)
        self.get_logger().info('GT -> PX4 EV bridge up (%s Hz)' % self.ev['rate_hz'])

    def on_odom(self, m):
        t_sample_ns = m.header.stamp.sec * 10 ** 9 + m.header.stamp.nanosec
        if t_sample_ns - self.last_pub_ns < self.period_ns:
            return
        self.last_pub_ns = t_sample_ns
        p = m.pose.pose.position
        q = m.pose.pose.orientation
        R = quat_xyzw_to_matrix(q.x, q.y, q.z, q.w)
        vl = np.array([m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z])
        v_enu = R @ vl if self.twist_in_body else vl
        w_flu = np.array([m.twist.twist.angular.x, m.twist.twist.angular.y, m.twist.twist.angular.z])

        o = VehicleOdometry()
        o.timestamp = int(self.get_clock().now().nanoseconds // 1000)
        o.timestamp_sample = int(t_sample_ns // 1000)
        o.pose_frame = VehicleOdometry.POSE_FRAME_NED
        o.position = [float(x) for x in enu_to_ned([p.x, p.y, p.z])]
        o.q = [float(x) for x in quat_enu_flu_to_ned_frd(q.x, q.y, q.z, q.w)]
        o.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        o.velocity = [float(x) for x in enu_to_ned(v_enu)]
        o.angular_velocity = [float(x) for x in flu_to_frd(w_flu)]
        o.position_variance = [float(self.ev['position_variance'])] * 3
        o.orientation_variance = [float(self.ev['orientation_variance'])] * 3
        o.velocity_variance = [float(self.ev['velocity_variance'])] * 3
        o.quality = int(self.ev['quality'])
        self.pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    node = GtToPx4Ev()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()

"""UWB multilateration: /uwb/ranges -> /uwb/position (PoseWithCovarianceStamped, world frame)."""
import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node

from drone_sim.config_loader import load_all
from drone_sim.uwb_solver import solve
from drone_sim_interfaces.msg import UwbRangeArray


class UwbMultilateration(Node):
    def __init__(self):
        super().__init__('uwb_multilateration')
        self.declare_parameter('config_dir', '')
        cfg = load_all(self.get_parameter('config_dir').value)
        u = cfg['uwb']
        self.m = u['multilateration']
        self.anchors = {int(a['id']): np.array([a['x'], a['y'], a['z']], dtype=float) for a in u['anchors']}
        self.last = np.array([self.m['init_xy'][0], self.m['init_xy'][1], self.m['z_prior_m']], dtype=float)
        self.pub = self.create_publisher(PoseWithCovarianceStamped, self.m['position_topic'], 10)
        self.create_subscription(UwbRangeArray, self.m['ranges_topic'], self.on_ranges, 10)

    def on_ranges(self, msg):
        ids = [r.anchor_id for r in msg.ranges if r.anchor_id in self.anchors]
        if len(ids) < int(self.m['min_anchors']):
            return
        A = np.array([self.anchors[i] for i in ids])
        rng = np.array([r.range_m for r in msg.ranges if r.anchor_id in self.anchors])
        sig = [r.stddev_m for r in msg.ranges if r.stddev_m > 0.0]
        sigma = float(np.mean(sig)) if sig else float(self.m['sigma_m'])
        x0 = self.last if self.m['use_last_fix_init'] else np.array(
            [self.m['init_xy'][0], self.m['init_xy'][1], self.m['z_prior_m']])
        out = solve(A, rng, x0, z_prior=self.m['z_prior_m'], z_min=self.m['z_min_m'],
                    max_iter=int(self.m['max_iterations']), tol=float(self.m['tolerance_m']),
                    sigma=sigma, huber=float(self.m['huber_threshold_m']))
        if out is None:
            self.get_logger().warn('multilateration failed', throttle_duration_sec=5.0)
            return
        x, cov, _, _ = out
        self.last = x
        o = PoseWithCovarianceStamped()
        o.header.stamp = msg.header.stamp
        o.header.frame_id = self.m['frame_id']
        o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = (float(v) for v in x)
        o.pose.pose.orientation.w = 1.0
        c = np.full(36, 0.0)
        for i in range(3):
            for j in range(3):
                c[6 * i + j] = float(cov[i, j])
        for k in (21, 28, 35):
            c[k] = 1e6      # orientation not observed
        o.pose.covariance = [float(v) for v in c]
        self.pub.publish(o)


def main(args=None):
    rclpy.init(args=args)
    node = UwbMultilateration()
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

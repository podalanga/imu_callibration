"""Geometry-based UWB range simulator (no RF/multipath model)."""
from collections import deque

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time

from drone_sim.config_loader import load_all, mount, quat_to_matrix
from drone_sim_interfaces.msg import UwbRange, UwbRangeArray


class UwbNode(Node):
    def __init__(self):
        super().__init__('uwb_node')
        self.declare_parameter('config_dir', '')
        cfg = load_all(self.get_parameter('config_dir').value)
        u = cfg['uwb']
        self.u = u
        self.anchors = u['anchors']
        self.apos = np.array([[a['x'], a['y'], a['z']] for a in self.anchors], dtype=float)
        self.abias = np.array([a.get('bias_m', 0.0) for a in self.anchors], dtype=float)
        self.tag_off = mount(cfg, 'uwb_tag')[0]
        self.delay_ns = int(u['delay_s'] * 1e9)
        self.rng = np.random.default_rng(int(u['seed']))
        self.buf = deque()  # (t_ns, base position, rotation matrix)
        self.create_subscription(Odometry, cfg['controller']['odom_topic'], self.on_odom, 50)
        self.pub = self.create_publisher(UwbRangeArray, '/uwb/ranges', 10)
        self.create_timer(1.0 / u['rate_hz'], self.tick)

    def on_odom(self, m):
        t = m.header.stamp.sec * 10**9 + m.header.stamp.nanosec
        pp = m.pose.pose.position
        q = m.pose.pose.orientation
        self.buf.append((t, np.array([pp.x, pp.y, pp.z]), quat_to_matrix(q.x, q.y, q.z, q.w)))
        horizon = t - self.delay_ns - 2 * 10**9
        while len(self.buf) > 2 and self.buf[0][0] < horizon:
            self.buf.popleft()

    def tick(self):
        if not self.buf:
            return
        t_meas = self.get_clock().now().nanoseconds - self.delay_ns
        sel = None
        for e in reversed(self.buf):
            if e[0] <= t_meas:
                sel = e
                break
        if sel is None:
            return
        _, p, R = sel
        tag = p + R @ self.tag_off
        rng_true = np.linalg.norm(self.apos - tag, axis=1)
        ideal = bool(self.u['ideal_mode'])
        sigma = 0.0 if ideal else float(self.u['noise_stddev_m'])
        if ideal:
            meas = rng_true
        else:
            meas = rng_true + self.u['global_bias_m'] + self.abias + self.rng.normal(0.0, sigma, len(rng_true))
        stamp = Time(nanoseconds=int(t_meas)).to_msg()
        out = UwbRangeArray()
        out.header.stamp = stamp
        out.header.frame_id = self.u['tag_frame']
        out.tag_frame = self.u['tag_frame']
        for a, r in zip(self.anchors, meas):
            m = UwbRange()
            m.header = out.header
            m.anchor_id = int(a['id'])
            m.anchor_name = a['name']
            m.range_m = float(r)
            m.stddev_m = sigma
            m.ideal = ideal
            out.ranges.append(m)
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = UwbNode()
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

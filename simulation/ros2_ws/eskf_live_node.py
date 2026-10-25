#!/usr/bin/env python3
"""Live 15-state error-state Kalman filter for the ROS 2 drone_sim workspace.

Subscribes to /imu/noisy, /uwb/ranges, and /ground_truth/odom (evaluation only).
Publishes /eskf/odometry, /eskf/estimated_path, and /eskf/ground_truth_path.

The filter does NOT consume ground truth or the IMU message's orientation field.
Requires ROS 2 Jazzy Python packages, numpy, and drone_sim_interfaces in the container.
"""
import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from drone_sim_interfaces.msg import UwbRangeArray

G = 9.81
# Exact coordinates from simulation/ros2_ws/src/drone_sim/config/uwb.yaml.
ANCHORS = np.array([
    [-6., -6., 0.], [6., -6., 0.], [6., 6., 0.], [-6., 6., 0.],
    [-6., -6., 12.], [6., -6., 12.], [6., 6., 12.], [-6., 6., 12.]
], dtype=float)


def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1., 0., 0., 0.])


def qmul(a, b):
    aw, ax, ay, az = a; bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw])


def qrotvec(phi):
    angle = float(np.linalg.norm(phi))
    if angle < 1e-10:
        return qnorm(np.r_[1., 0.5*np.asarray(phi, dtype=float)])
    return np.r_[math.cos(angle/2), np.asarray(phi)*math.sin(angle/2)/angle]


def qmatrix(q):
    w, x, y, z = qnorm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def euler_q(roll, pitch, yaw):
    cr, sr = math.cos(roll/2), math.sin(roll/2)
    cp, sp = math.cos(pitch/2), math.sin(pitch/2)
    cy, sy = math.cos(yaw/2), math.sin(yaw/2)
    return qnorm(np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                           cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]))


def stamp_seconds(stamp):
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def trilaterate(ranges, start=None):
    """Gauss-Newton range solution; start z low to select the floor-side mirror solution."""
    valid = [(int(i), float(r)) for i, r in ranges.items()
             if 1 <= int(i) <= 8 and math.isfinite(float(r)) and float(r) > 0.05]
    if len(valid) < 4:
        return None
    ids = np.array([i-1 for i, _ in valid], dtype=int)
    rr = np.array([r for _, r in valid], dtype=float)
    aa = ANCHORS[ids]
    p = np.array(start if start is not None else [0., 0., 1.16], dtype=float)
    for _ in range(40):
        dv = p[None, :] - aa
        dd = np.maximum(np.linalg.norm(dv, axis=1), 1e-8)
        H = dv / dd[:, None]
        residual = rr - dd
        try:
            step = np.linalg.lstsq(H, residual, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None
        p += step
        # The vehicle operates close to the floor; prevent convergence to the mirrored top solution.
        p[2] = float(np.clip(p[2], 0.05, 11.95))
        if np.linalg.norm(step) < 1e-5:
            break
    return p


class LiveESKF(Node):
    def __init__(self):
        super().__init__('eskf_live_node')
        self.declare_parameter('imu_topic', '/imu/noisy')
        self.declare_parameter('uwb_topic', '/uwb/ranges')
        self.declare_parameter('ground_truth_topic', '/ground_truth/odom')
        self.declare_parameter('uwb_sigma_m', 0.05)
        self.declare_parameter('gyro_noise', 0.015)
        self.declare_parameter('accel_noise', 0.15)
        self.declare_parameter('initial_height_m', 1.16)
        self.declare_parameter('use_gravity_update', True)
        self.declare_parameter('gravity_update_sigma', 0.12)
        self.declare_parameter('max_path_points', 20000)
        self.uwb_sigma = float(self.get_parameter('uwb_sigma_m').value)
        self.gyro_noise = float(self.get_parameter('gyro_noise').value)
        self.accel_noise = float(self.get_parameter('accel_noise').value)
        self.initial_height = float(self.get_parameter('initial_height_m').value)
        self.use_gravity_update = bool(self.get_parameter('use_gravity_update').value)
        self.gravity_sigma = float(self.get_parameter('gravity_update_sigma').value)
        self.max_points = int(self.get_parameter('max_path_points').value)

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Imu, self.get_parameter('imu_topic').value, self.imu_cb, qos)
        self.create_subscription(UwbRangeArray, self.get_parameter('uwb_topic').value, self.uwb_cb, qos)
        self.create_subscription(Odometry, self.get_parameter('ground_truth_topic').value, self.gt_cb, qos)
        self.odom_pub = self.create_publisher(Odometry, '/eskf/odometry', 10)
        self.path_pub = self.create_publisher(Path, '/eskf/estimated_path', 10)
        self.gt_path_pub = self.create_publisher(Path, '/eskf/ground_truth_path', 10)
        self.est_marker_pub = self.create_publisher(Marker, '/eskf/estimated_line', 10)
        self.gt_marker_pub = self.create_publisher(Marker, '/eskf/ground_truth_line', 10)

        self.initialized = False
        self.last_imu_t = None
        self.last_range_t = -math.inf
        self.pending_ranges = deque(maxlen=100)
        self.last_range_map = {}
        self.p = np.zeros(3); self.v = np.zeros(3); self.q = np.array([1.,0.,0.,0.])
        self.bg = np.zeros(3); self.ba = np.zeros(3)
        self.P = np.diag([1.0]*3 + [1.0]*3 + [0.25]*3 + [0.05**2]*3 + [0.2**2]*3)
        self.est_path = Path(); self.est_path.header.frame_id = 'world'
        self.gt_path = Path(); self.gt_path.header.frame_id = 'world'
        self.est_points = []; self.gt_points = []
        self.err_sq_sum = 0.0; self.err_count = 0
        self.last_log_t = -math.inf
        self.get_logger().info('Live 15-state ESKF started; anchors loaded from the 12 m cube configuration.')
        self.get_logger().info('Ground truth is used only for evaluation and a separate visualization path.')

    def uwb_cb(self, msg):
        t = stamp_seconds(msg.header.stamp)
        ranges = {}
        for r in msg.ranges:
            if 1 <= int(r.anchor_id) <= 8 and math.isfinite(float(r.range_m)) and r.range_m > 0:
                ranges[int(r.anchor_id)] = float(r.range_m)
        if len(ranges) >= 4:
            self.last_range_map = ranges
            self.pending_ranges.append((t, ranges))
            if not self.initialized:
                p0 = trilaterate(ranges, [0., 0., self.initial_height])
                if p0 is not None:
                    self.p = p0
                    self.initialized = True
                    self.get_logger().info(f'Initialized position from UWB ranges at x={p0[0]:.2f}, y={p0[1]:.2f}, z={p0[2]:.2f} m')

    def imu_cb(self, msg):
        t = stamp_seconds(msg.header.stamp)
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float)
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z], dtype=float)
        if not np.all(np.isfinite(gyro)) or not np.all(np.isfinite(accel)):
            return
        if not self.initialized:
            return
        if self.last_imu_t is None:
            # Start with roll/pitch from the gravity direction; yaw is unobservable without a magnetometer.
            roll = math.atan2(accel[1], accel[2])
            pitch = math.atan2(-accel[0], math.sqrt(accel[1]**2 + accel[2]**2) + 1e-12)
            self.q = euler_q(roll, pitch, 0.0)
            self.last_imu_t = t
            self._apply_pending_ranges(t)
            self._publish(t)
            return
        dt_raw = t - self.last_imu_t
        self.last_imu_t = t
        if dt_raw <= 0 or dt_raw > 0.5:
            # Avoid catastrophic propagation over paused/reset simulation time.
            if dt_raw > 0.5:
                self.get_logger().warn(f'Large IMU timestamp gap ({dt_raw:.3f}s); skipping propagation step.')
            self._apply_pending_ranges(t)
            self._publish(t)
            return
        dt = float(np.clip(dt_raw, 1e-4, 0.05))
        omega = gyro - self.bg
        f = accel - self.ba
        R = qmatrix(self.q)
        a_world = R @ f + np.array([0., 0., -G])
        self.p = self.p + self.v*dt + 0.5*a_world*dt*dt
        self.v = self.v + a_world*dt
        self.q = qnorm(qmul(self.q, qrotvec(omega*dt)))
        R = qmatrix(self.q)

        F = np.eye(15)
        F[0:3,3:6] = np.eye(3)*dt
        F[0:3,6:9] = -0.5*R@skew(f)*dt*dt
        F[0:3,12:15] = -0.5*R*dt*dt
        F[3:6,6:9] = -R@skew(f)*dt
        F[3:6,12:15] = -R*dt
        F[6:9,6:9] = np.eye(3)-skew(omega)*dt
        F[6:9,9:12] = -np.eye(3)*dt
        Gm = np.zeros((15,12))
        Gm[0:3,0:3] = -0.5*R*dt*dt
        Gm[3:6,0:3] = -R*dt
        Gm[6:9,3:6] = -np.eye(3)*dt
        Gm[9:12,6:9] = np.eye(3)*math.sqrt(dt)
        Gm[12:15,9:12] = np.eye(3)*math.sqrt(dt)
        Sigma = np.diag([self.accel_noise**2]*3 + [self.gyro_noise**2]*3 + [0.0002**2]*3 + [0.01**2]*3)
        self.P = F @ self.P @ F.T + Gm @ Sigma @ Gm.T
        self.P = 0.5*(self.P + self.P.T)

        if self.use_gravity_update:
            fn = float(np.linalg.norm(f))
            if fn > 1e-6 and abs(fn-G) < 2.0:
                h = R.T @ np.array([0.,0.,1.])
                H = np.zeros((3,15)); H[:,6:9] = skew(h)
                residual = f/fn - h
                self._update(residual, H, np.eye(3)*self.gravity_sigma**2, gate=12.0)
        self._apply_pending_ranges(t)
        self._publish(t)

    def _update(self, residual, H, Rm, gate=None):
        S = H @ self.P @ H.T + Rm
        try:
            nis = float(residual.T @ np.linalg.solve(S, residual))
            K = np.linalg.solve(S, H @ self.P).T
        except np.linalg.LinAlgError:
            return False
        if gate is not None and nis > gate:
            return False
        dx = K @ residual
        I = np.eye(15)
        Ptmp = (I-K@H) @ self.P @ (I-K@H).T + K @ Rm @ K.T
        self.p += dx[0:3]; self.v += dx[3:6]
        dtheta = dx[6:9]
        self.q = qnorm(qmul(self.q, qrotvec(dtheta)))
        self.bg += dx[9:12]; self.ba += dx[12:15]
        J = np.eye(15); J[6:9,6:9] = np.eye(3)-0.5*skew(dtheta)
        self.P = J @ Ptmp @ J.T
        self.P = 0.5*(self.P + self.P.T)
        return True

    def _apply_pending_ranges(self, now):
        # Process each UWB array once. Its own stamp is retained for diagnostics; the update is applied
        # at the latest available IMU state (up to one IMU period of latency in normal operation).
        while self.pending_ranges and self.pending_ranges[0][0] <= now + 0.01:
            tr, ranges = self.pending_ranges.popleft()
            if tr <= self.last_range_t:
                continue
            self.last_range_t = tr
            for aid, z in ranges.items():
                a = ANCHORS[aid-1]
                dv = self.p - a
                pred = float(np.linalg.norm(dv))
                if pred < 1e-6:
                    continue
                H = np.zeros((1,15)); H[0,0:3] = dv/pred
                residual = np.array([z-pred])
                self._update(residual, H, np.array([[self.uwb_sigma**2]]), gate=16.0)

    def _pose_stamped(self, t):
        ps = PoseStamped()
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.header.frame_id = 'world'
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, self.p)
        ps.pose.orientation.w, ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z = map(float, self.q)
        return ps

    def _publish(self, t):
        now = self.get_clock().now().to_msg()
        od = Odometry(); od.header.stamp = now; od.header.frame_id = 'world'; od.child_frame_id = 'eskf_body'
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = map(float, self.p)
        od.pose.pose.orientation.w, od.pose.pose.orientation.x, od.pose.pose.orientation.y, od.pose.pose.orientation.z = map(float, self.q)
        od.twist.twist.linear.x, od.twist.twist.linear.y, od.twist.twist.linear.z = map(float, self.v)
        od.pose.covariance[0] = float(self.P[0,0]); od.pose.covariance[7] = float(self.P[1,1]); od.pose.covariance[14] = float(self.P[2,2])
        self.odom_pub.publish(od)
        ps = self._pose_stamped(t)
        self.est_points.append(ps)
        if len(self.est_points) > self.max_points: self.est_points = self.est_points[-self.max_points:]
        path = Path(); path.header.stamp = now; path.header.frame_id = 'world'; path.poses = list(self.est_points)
        self.path_pub.publish(path)
        self._publish_line(self.est_marker_pub, self.est_points, 0, 0.1, 0.85, 0.2)

        if self.err_count and t - self.last_log_t > 2.0:
            rmse = math.sqrt(self.err_sq_sum / self.err_count)
            self.get_logger().info(f'Live ESKF position RMSE vs ground truth: {rmse:.3f} m ({self.err_count} matched samples)')
            self.last_log_t = t

    def _publish_line(self, pub, points, marker_id, r, g, b):
        if not points: return
        m = Marker(); m.header.stamp = self.get_clock().now().to_msg(); m.header.frame_id = 'world'
        m.ns = 'eskf_trajectory'; m.id = marker_id; m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.035; m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
        m.points = [p.pose.position for p in points]
        pub.publish(m)

    def gt_cb(self, msg):
        # Evaluation/visualization only: never changes the ESKF state.
        gt = msg.pose.pose.position
        ps = PoseStamped(); ps.header = msg.header; ps.header.frame_id = 'world'; ps.pose = msg.pose.pose
        self.gt_points.append(ps)
        if len(self.gt_points) > self.max_points: self.gt_points = self.gt_points[-self.max_points:]
        path = Path(); path.header.stamp = self.get_clock().now().to_msg(); path.header.frame_id = 'world'; path.poses = list(self.gt_points)
        self.gt_path_pub.publish(path)
        self._publish_line(self.gt_marker_pub, self.gt_points, 1, 0.95, 0.15, 0.1)
        if self.initialized:
            e = np.array([self.p[0]-gt.x, self.p[1]-gt.y, self.p[2]-gt.z], dtype=float)
            if np.all(np.isfinite(e)) and np.linalg.norm(e) < 100:
                self.err_sq_sum += float(e @ e); self.err_count += 1


def main(args=None):
    rclpy.init(args=args)
    node = LiveESKF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

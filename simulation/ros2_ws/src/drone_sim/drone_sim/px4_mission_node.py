"""PX4 offboard mission: arm, switch to OFFBOARD, stream pos/vel/acc setpoints, 5 traversals.

Same state machine and reference profiles as the v1 controller (drone_sim.trajectory); the inner loops are
PX4's. Everything is planned in the world ENU frame and converted to NED only when published.
Progress (phases, traversal counting) is judged from the ground-truth odometry, not from PX4's estimate.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, Vector3
from nav_msgs.msg import Odometry
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition,
                          VehicleStatus)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from drone_sim.config_loader import camera_offset_in_base, load_all, quat_to_matrix
from drone_sim.frames import enu_to_ned, yaw_enu_to_ned
from drone_sim.trajectory import build_landing, build_plan
from drone_sim_interfaces.msg import ControllerStatus

NAN3 = [float('nan')] * 3


class Px4Mission(Node):
    def __init__(self):
        super().__init__('px4_mission')
        self.declare_parameter('config_dir', '')
        cfg = load_all(self.get_parameter('config_dir').value)
        self.c = cfg['controller']
        self.px = self.c['px4']
        self.t = cfg['trajectory']
        self.hover_cam_z = cfg['camera']['optical_centre_height_m']
        self.r_cam = camera_offset_in_base(cfg)
        self.n_total = int(self.t['n_traversals'])

        self.odom = None
        self.t_first_odom = None
        self.status = None
        self.local_pos = None
        self.state = 'WAIT_SIM'
        self.t_state = 0.0
        self.takeoff = None
        self.traversals = []
        self.phase = None
        self.phase_t0 = 0.0
        self.hold_cam = None
        self.hold_com = None
        self.completed = 0
        self.current = 0
        self.direction = ''
        self.finished = False
        self.last_cmd_t = -1e9
        self.last_warn_t = -1e9
        self.step_count = 0

        px = self.px
        pub_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE,
                             history=HistoryPolicy.KEEP_LAST, depth=1)
        q = px['qos_out']
        sub_qos = QoSProfile(reliability=ReliabilityPolicy[q['reliability'].upper()],
                             durability=DurabilityPolicy[q['durability'].upper()],
                             history=HistoryPolicy.KEEP_LAST, depth=int(q['depth']))
        self.create_subscription(Odometry, self.c['odom_topic'], self.on_odom, 10)
        self.create_subscription(VehicleStatus, px['vehicle_status_topic'], self.on_status, sub_qos)
        self.create_subscription(VehicleLocalPosition, px['local_position_topic'], self.on_local, sub_qos)
        self.pub_ocm = self.create_publisher(OffboardControlMode, px['offboard_control_mode_topic'], pub_qos)
        self.pub_sp = self.create_publisher(TrajectorySetpoint, px['trajectory_setpoint_topic'], pub_qos)
        self.pub_vc = self.create_publisher(VehicleCommand, px['vehicle_command_topic'], pub_qos)
        self.pub_status = self.create_publisher(ControllerStatus, self.c['status_topic'], 10)
        self.pub_ref = self.create_publisher(PoseStamped, self.c['reference_topic'], 10)
        self.create_timer(1.0 / float(self.c['rate_hz']), self.step)
        self.get_logger().info('PX4 mission node up; waiting for ground-truth odometry')

    def on_odom(self, msg):
        self.odom = msg

    def on_status(self, msg):
        self.status = msg

    def on_local(self, msg):
        self.local_pos = msg

    # ------------------------------------------------------------------
    def now_s(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def now_us(self):
        return int(self.get_clock().now().nanoseconds // 1000)

    def goto(self, state, now):
        self.get_logger().info('state %s -> %s (t=%.2f)' % (self.state, state, now))
        self.state = state
        self.t_state = now

    def start_phase(self, phase, now):
        self.phase = phase
        self.phase_t0 = now
        self.direction = phase.direction

    def send_command(self, command, p1=0.0, p2=0.0):
        m = VehicleCommand()
        m.timestamp = self.now_us()
        m.command = int(command)
        m.param1 = float(p1)
        m.param2 = float(p2)
        m.target_system = int(self.px['target_system'])
        m.target_component = int(self.px['target_component'])
        m.source_system = int(self.px['source_system'])
        m.source_component = int(self.px['source_component'])
        m.from_external = True
        self.pub_vc.publish(m)

    def offboard_active(self):
        s = self.status
        return s is not None and s.nav_state == int(self.px['nav_state_offboard'])

    def armed(self):
        s = self.status
        return s is not None and s.arming_state == int(self.px['arming_state_armed'])

    def px4_ready(self, now):
        if not self.px['require_status']:
            return now - self.t_first_odom >= self.px['ev_ready_s']
        if self.status is None:
            return False
        if self.px['require_local_position']:
            lp = self.local_pos
            if lp is None or not (lp.xy_valid and lp.z_valid):
                return False
        return now - self.t_first_odom >= self.px['ev_ready_s']

    # ------------------------------------------------------------------
    def step(self):
        now = self.now_s()
        o = self.odom
        if o is None or now <= 0.0:
            return
        if self.t_first_odom is None:
            self.t_first_odom = now
        pq = o.pose.pose
        p = np.array([pq.position.x, pq.position.y, pq.position.z])
        q = pq.orientation
        R = quat_to_matrix(q.x, q.y, q.z, q.w)
        vl = np.array([o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z])
        v_w = R @ vl if self.c['odom_twist_in_body'] else vl
        dz = float((R @ self.r_cam)[2])
        cam = p + R @ self.r_cam
        if self.hold_com is None:
            self.hold_com = p.copy()

        ref_com = self.hold_com.copy()
        v_ref = np.zeros(3)
        a_ref = np.zeros(3)

        if self.state == 'WAIT_SIM':
            self.goto('WARMUP', now)
        if self.state == 'WARMUP':
            if now - self.t_state >= self.px['offboard_warmup_s'] and self.px4_ready(now):
                self.goto('ARMING', now)
            elif now - self.t_state > 15.0 and now - self.last_warn_t > 5.0:
                self.last_warn_t = now
                self.get_logger().warn('waiting for PX4 (status=%s, local_position valid=%s)' % (
                    self.status is not None,
                    None if self.local_pos is None else (bool(self.local_pos.xy_valid and self.local_pos.z_valid))))
        if self.state == 'ARMING':
            if now - self.last_cmd_t >= self.px['command_retry_s']:
                self.last_cmd_t = now
                self.send_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)   # custom mode, offboard
                self.send_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)
            done = (self.offboard_active() and self.armed()) if self.px['require_status'] \
                else (now - self.t_state >= 2.0)
            if done:
                self.hold_cam = cam.copy()
                self.takeoff, self.traversals = build_plan(self.t, cam, self.hover_cam_z)
                self.start_phase(self.takeoff, now)
                self.goto('TAKEOFF', now)

        # ---- reference & transitions
        if self.state in ('TAKEOFF', 'TRAVERSE', 'LAND'):
            el = now - self.phase_t0
            ref_cam, v_ref, a_ref = self.phase.reference(el)
            ref_com = np.array(ref_cam, dtype=float)
            ref_com[2] -= dz
            if el >= self.phase.duration:
                along = abs(self.phase.along_track_error(cam))
                if along < self.t['pos_tol_m'] and np.linalg.norm(v_w) < self.t['vel_tol_mps']:
                    self.on_phase_done(now)
        elif self.state == 'DWELL':
            ref_com = np.array(self.hold_cam, dtype=float)
            ref_com[2] -= dz
            # hover after takeoff so the climb settles before the first pass; dwell_s between passes
            wait = self.t['takeoff_settle_s'] if self.completed == 0 else self.t['dwell_s']
            if now - self.t_state >= wait:
                self.current = self.completed + 1
                self.start_phase(self.traversals[self.completed], now)
                self.goto('TRAVERSE', now)
        elif self.state in ('HOVER_FINISH', 'LANDED'):
            ref_com = np.array(self.hold_cam, dtype=float)
            ref_com[2] -= dz

        err = ref_com - p
        # ---- PX4 messages (published every step, i.e. at rate_hz >= 10 Hz, before and during offboard)
        v_cmd = v_ref + np.array([self.px['extra_kp_xy'], self.px['extra_kp_xy'], self.px['extra_kp_z']]) * err
        yaw_ned = yaw_enu_to_ned(float(self.px['yaw_enu_rad']))
        ts = self.now_us()
        ocm = OffboardControlMode()
        ocm.timestamp = ts
        ocm.position = True
        ocm.velocity = True
        ocm.acceleration = True
        ocm.attitude = False
        ocm.body_rate = False
        self.pub_ocm.publish(ocm)
        sp = TrajectorySetpoint()
        sp.timestamp = ts
        sp.position = [float(x) for x in enu_to_ned(ref_com)]
        sp.velocity = [float(x) for x in enu_to_ned(v_cmd)]
        sp.acceleration = [float(x) for x in enu_to_ned(a_ref)]
        sp.jerk = NAN3
        sp.yaw = float(yaw_ned)
        sp.yawspeed = float('nan')
        self.pub_sp.publish(sp)

        self.step_count += 1
        if self.step_count % int(self.c['status_decimation']) == 0:
            self.publish_status(ref_com, v_ref, p, err, float(cam[2]))

    def on_phase_done(self, now):
        self.hold_cam = self.phase.p1.copy()
        if self.state == 'TAKEOFF':
            self.goto('DWELL', now)
        elif self.state == 'TRAVERSE':
            self.completed += 1
            self.get_logger().info('traversal %d/%d complete' % (self.completed, self.n_total))
            if self.completed >= self.n_total:
                self.finished = True
                if self.t['land_at_finish']:
                    ground_z = float(self.takeoff.p0[2] + self.t['land_height_margin_m'])
                    self.start_phase(build_landing(self.t, self.hold_cam, ground_z), now)
                    self.goto('LAND', now)
                else:
                    self.goto('HOVER_FINISH', now)
            else:
                self.goto('DWELL', now)
        elif self.state == 'LAND':
            self.goto('LANDED', now)

    def publish_status(self, ref_com, v_ref, p, err, cam_z):
        stamp = self.get_clock().now().to_msg()
        m = ControllerStatus()
        m.header.stamp = stamp
        m.header.frame_id = 'world'
        m.state = self.state
        m.traversals_completed = int(self.completed)
        m.traversals_total = int(self.n_total)
        m.current_traversal = int(self.current)
        m.direction = self.direction
        m.reference_position = Point(x=float(ref_com[0]), y=float(ref_com[1]), z=float(ref_com[2]))
        m.reference_velocity = Vector3(x=float(v_ref[0]), y=float(v_ref[1]), z=float(v_ref[2]))
        m.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
        m.position_error_norm = float(np.linalg.norm(err))
        m.camera_height_m = cam_z
        m.finished = bool(self.finished)
        self.pub_status.publish(m)
        r = PoseStamped()
        r.header.stamp = stamp
        r.header.frame_id = 'world'
        r.pose.position = m.reference_position
        r.pose.orientation.w = 1.0
        self.pub_ref.publish(r)


def main(args=None):
    rclpy.init(args=args)
    node = Px4Mission()
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

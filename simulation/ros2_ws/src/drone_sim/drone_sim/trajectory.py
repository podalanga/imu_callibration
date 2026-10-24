"""Pure-python reference trajectory: raised-cosine velocity ramps, closed form pos/vel/acc."""
import math

import numpy as np


class Profile1D:
    """Rest-to-rest 1D move of length `distance`.

    Velocity ramps 0 -> v with a raised cosine over T = pi*v/(2*a_max) (peak accel = a_max),
    cruises, and ramps back. If the distance is too short for full speed, v is reduced so that
    the two ramps just fit (v = sqrt(2*a_max*D/pi)).
    """

    def __init__(self, distance, v_max, a_max):
        self.D = max(0.0, float(distance))
        if self.D < 1e-9:
            self.v = self.T = self.tc = self.duration = 0.0
            return
        v = float(v_max)
        if self.D < math.pi * v * v / (2.0 * a_max):
            v = math.sqrt(2.0 * a_max * self.D / math.pi)
        self.v = v
        self.T = math.pi * v / (2.0 * a_max)
        self.tc = max(0.0, self.D / v - self.T)
        self.duration = 2.0 * self.T + self.tc

    def at(self, t):
        """Return (s, v, a) at time t (clamped to [0, duration])."""
        if self.duration <= 0.0:
            return self.D, 0.0, 0.0
        t = min(max(t, 0.0), self.duration)
        v, T, tc = self.v, self.T, self.tc
        if t < T:
            w = math.pi / T
            return (0.5 * v * (t - math.sin(w * t) / w),
                    0.5 * v * (1.0 - math.cos(w * t)),
                    0.5 * v * w * math.sin(w * t))
        if t < T + tc:
            return 0.5 * v * T + v * (t - T), v, 0.0
        tau = t - T - tc
        w = math.pi / T
        s0 = 0.5 * v * T + v * tc
        return (s0 + 0.5 * v * (tau + math.sin(w * tau) / w),
                0.5 * v * (1.0 + math.cos(w * tau)),
                -0.5 * v * w * math.sin(w * tau))


class Phase:
    """Straight-line move p0 -> p1 following a Profile1D."""

    def __init__(self, name, p0, p1, v_max, a_max, direction=''):
        self.name = name
        self.direction = direction
        self.p0 = np.asarray(p0, dtype=float)
        self.p1 = np.asarray(p1, dtype=float)
        d = self.p1 - self.p0
        self.length = float(np.linalg.norm(d))
        self.u = d / self.length if self.length > 1e-9 else np.zeros(3)
        self.profile = Profile1D(self.length, v_max, a_max)
        self.duration = self.profile.duration

    def reference(self, t):
        s, v, a = self.profile.at(t)
        return self.p0 + self.u * s, self.u * v, self.u * a

    def along_track_error(self, p):
        """Signed distance of p past the end point along the segment direction."""
        return float(np.dot(np.asarray(p) - self.p1, self.u))


def build_plan(traj_cfg, start_cam_pos, hover_z):
    """Takeoff phase + the fixed list of traversal phases (positions are camera-optical-centre)."""
    xl, xr, y = traj_cfg['x_left'], traj_cfg['x_right'], traj_cfg['y']
    takeoff = Phase('takeoff', start_cam_pos, [xl, y, hover_z],
                    traj_cfg['takeoff_v'], traj_cfg['takeoff_a'], 'up')
    trav = []
    for i in range(int(traj_cfg['n_traversals'])):
        l2r = (i % 2 == 0)
        a, b = (xl, xr) if l2r else (xr, xl)
        trav.append(Phase('traverse_%d' % (i + 1), [a, y, hover_z], [b, y, hover_z],
                          traj_cfg['v_cruise'], traj_cfg['a_max'],
                          'left_to_right' if l2r else 'right_to_left'))
    return takeoff, trav


def build_landing(traj_cfg, from_pos, ground_z):
    return Phase('land', from_pos, [from_pos[0], from_pos[1], ground_z],
                 traj_cfg['land_v'], traj_cfg['land_a'], 'down')

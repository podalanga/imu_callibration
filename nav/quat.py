"""Rotation utilities. Convention used everywhere in this repository:

* Hamilton quaternions, scalar first: q = [w, x, y, z].
* q maps body to world: v_world = R(q) @ v_body.
* Attitude error is a local (right) perturbation: q_true = q (x) Exp(dtheta).
"""
import numpy as np


def skew(a):
    return np.array([[0.0, -a[2], a[1]],
                     [a[2], 0.0, -a[0]],
                     [-a[1], a[0], 0.0]])


def qmul(p, q):
    w1, x1, y1, z1 = p
    w2, x2, y2, z2 = q
    return np.array([w1*w2 - x1*x2 - y1*y2 - z1*z2,
                     w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2,
                     w1*z2 + x1*y2 - y1*x2 + z1*w2])


def qconj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qnorm(q):
    return q / np.linalg.norm(q)


def qexp(phi):
    """Rotation vector -> unit quaternion."""
    a = np.linalg.norm(phi)
    if a < 1e-9:
        return qnorm(np.r_[1.0, 0.5 * phi])
    return np.r_[np.cos(a / 2), np.sin(a / 2) * phi / a]


def qlog(q):
    """Unit quaternion -> rotation vector (shortest)."""
    q = q if q[0] >= 0 else -q
    v = q[1:]
    n = np.linalg.norm(v)
    if n < 1e-12:
        return 2.0 * v
    return 2.0 * np.arctan2(n, q[0]) * v / n


def q2R(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])


def euler2q(roll, pitch, yaw):
    """ZYX: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return qmul(qmul(qexp(np.array([0, 0, yaw])), qexp(np.array([0, pitch, 0]))),
                qexp(np.array([roll, 0, 0])))


def q2euler(q):
    R = q2R(q)
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw])


def tilt_from_accel(f):
    """Roll and pitch from specific force at rest (body z up => reads +g)."""
    roll = np.arctan2(f[1], f[2])
    pitch = np.arctan2(-f[0], np.hypot(f[1], f[2]))
    return roll, pitch

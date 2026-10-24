"""ENU/FLU (ROS, Gazebo world) <-> NED/FRD (PX4) conversions. Pure numpy, no ROS.

Position/velocity/acceleration: x_ned = y_enu, y_ned = x_enu, z_ned = -z_enu.
Yaw: yaw_ned = pi/2 - yaw_enu (heading measured clockwise from north vs counter-clockwise from east).
Orientation: R_ned_frd = M R_enu_flu D, with M the ENU->NED axis swap and D = diag(1,-1,-1) (FLU->FRD).
Quaternions are [w, x, y, z] unless a function name says xyzw.
"""
import math

import numpy as np

M_ENU_TO_NED = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
D_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])


def wrap_pi(a):
    return math.atan2(math.sin(a), math.cos(a))


def enu_to_ned(v):
    v = np.asarray(v, dtype=float)
    return np.array([v[1], v[0], -v[2]])


def ned_to_enu(v):
    return enu_to_ned(v)  # the swap is its own inverse


def yaw_enu_to_ned(yaw_enu):
    return wrap_pi(math.pi / 2.0 - yaw_enu)


def yaw_ned_to_enu(yaw_ned):
    return wrap_pi(math.pi / 2.0 - yaw_ned)


def flu_to_frd(v):
    v = np.asarray(v, dtype=float)
    return np.array([v[0], -v[1], -v[2]])


def matrix_to_quat_wxyz(R):
    R = np.asarray(R, dtype=float)
    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0.0:
        s = 2.0 * math.sqrt(t + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return q


def quat_xyzw_to_matrix(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_enu_flu_to_R_ned_frd(R_enu_flu):
    return M_ENU_TO_NED @ np.asarray(R_enu_flu) @ D_FLU_TO_FRD


def quat_enu_flu_to_ned_frd(x, y, z, w):
    """ROS quaternion (x,y,z,w) of FLU body in ENU -> PX4 quaternion [w,x,y,z] of FRD body in NED."""
    return matrix_to_quat_wxyz(R_enu_flu_to_R_ned_frd(quat_xyzw_to_matrix(x, y, z, w)))

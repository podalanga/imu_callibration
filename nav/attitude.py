"""Attitude baselines (Chapter 10): gyro integration, complementary, Mahony, Madgwick."""
import numpy as np

from .quat import qmul, qexp, q2R, euler2q, tilt_from_accel


def gyro_only(q0, gyro, t):
    q = np.array(q0, float); out = [q]
    for k in range(1, len(t)):
        q = qmul(q, qexp(gyro[k] * (t[k] - t[k - 1]))); q /= np.linalg.norm(q); out.append(q)
    return np.array(out)


def mahony(q0, gyro, acc, t, kp=1.0, ki=0.02, g=9.80665, gate=0.05):
    q = np.array(q0, float); b = np.zeros(3); out = [q]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        e = np.zeros(3)
        n = np.linalg.norm(acc[k])
        if abs(n - g) < gate * g:
            v_hat = q2R(q).T @ np.array([0, 0, 1.0])
            e = np.cross(acc[k] / n, v_hat)
        b = b + ki * e * dt              # b = -bias estimate: d(bias)/dt = -ki e
        w = gyro[k] + b + kp * e         # b here is minus the bias estimate
        q = qmul(q, qexp(w * dt)); q /= np.linalg.norm(q); out.append(q)
    return np.array(out)


def madgwick(q0, gyro, acc, t, beta=0.033):
    q = np.array(q0, float); out = [q]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        a = acc[k] / np.linalg.norm(acc[k])
        w0, x, y, z = q
        fv = np.array([2 * (x * z - w0 * y) - a[0], 2 * (w0 * x + y * z) - a[1], 1 - 2 * x * x - 2 * y * y - a[2]])
        J = np.array([[-2 * y, 2 * z, -2 * w0, 2 * x], [2 * x, 2 * w0, 2 * z, 2 * y], [0, -4 * x, -4 * y, 0]])
        grad = J.T @ fv
        nrm = np.linalg.norm(grad)
        qdot = 0.5 * qmul(q, np.r_[0, gyro[k]]) - (beta * grad / nrm if nrm > 0 else 0)
        q = q + qdot * dt; q /= np.linalg.norm(q); out.append(q)
    return np.array(out)


def initial_q(acc, n):
    r, p = tilt_from_accel(acc[:n].mean(axis=0))
    return euler2q(r, p, 0.0)

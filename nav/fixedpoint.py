"""Float64 mirror of firmware/main/mahony_q.c: same algorithm, no quantisation.

Comparing the two on the same data isolates the error caused by fixed-point arithmetic (test X12).
"""
import numpy as np


class MahonyStrapdown:
    def __init__(self, q0, kp=1.0, ki=0.02, g=9.80665, w_still=0.004, a_still=0.04, still_need=60):
        self.q = np.array(q0, float); self.b = np.zeros(3); self.v = np.zeros(3); self.p = np.zeros(3)
        self.kp, self.ki, self.g = kp, ki, g
        self.w_still, self.a_still, self.need = w_still, a_still, still_need
        self.count = 0; self.still = False; self.ew = 0.0; self.a_ref = None; self.da = np.zeros(3)

    def step(self, a, w, dt):
        q = self.q
        an = np.linalg.norm(a)
        dev = abs(an - self.g)
        e = np.zeros(3)
        if self.still and an > 0 and dev < self.g / 20:      # correct tilt only while stopped
            u = a / an
            vh = np.array([2 * (q[1]*q[3] - q[0]*q[2]), 2 * (q[0]*q[1] + q[2]*q[3]),
                           q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2])
            e = np.cross(u, vh)
        self.b += self.ki * e * dt                    # b = -bias estimate, so db/dt = +ki e
        wc = w + self.b + self.kp * e
        dth = wc * dt
        w0, x, y, z = q
        R = np.array([[1 - 2*(y*y + z*z), 2*(x*y - w0*z), 2*(x*z + w0*y)],
                      [2*(x*y + w0*z), 1 - 2*(x*x + z*z), 2*(y*z - w0*x)],
                      [2*(x*z - w0*y), 2*(y*z + w0*x), 1 - 2*(x*x + y*y)]])
        aw = R @ a - np.array([0, 0, self.g])
        self.p = self.p + self.v * dt
        self.v = self.v + aw * dt
        dq = np.array([-(x*dth[0] + y*dth[1] + z*dth[2]), w0*dth[0] + y*dth[2] - z*dth[1],
                       w0*dth[1] - x*dth[2] + z*dth[0], w0*dth[2] + x*dth[1] - y*dth[0]])
        self.q = q + 0.5 * dq
        self.q /= np.linalg.norm(self.q)
        wb = w + self.b
        if self.a_ref is None:
            self.a_ref = np.array(a, float)
        self.da += (a - self.a_ref - self.da) / 16
        self.ew += (wb @ wb - self.ew) / 16
        self.count = self.count + 1 if (self.ew < self.w_still**2 and self.da @ self.da < self.a_still**2) else 0
        self.still = self.count >= self.need
        if self.still:
            self.v[:] = 0
            self.b += (-w - self.b) / 1024
            self.a_ref += (a - self.a_ref) / 64


def run(imu, q0, n_skip=0, **kw):
    f = MahonyStrapdown(q0, **kw)
    t = imu.t
    out = []
    for k in range(n_skip + 1, len(t)):
        f.step(imu.acc[k], imu.gyro[k], t[k] - t[k - 1])
        out.append(np.r_[imu.t_us[k], f.p, f.v, f.q, f.still])
    return np.array(out)

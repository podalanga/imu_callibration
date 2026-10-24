"""15-state error-state Kalman filter (Chapter 12). Float64 reference implementation.

Error state order: dp(0:3) dv(3:6) dtheta(6:9) dba(9:12) dbg(12:15).
The C port in firmware/main/eskf.c mirrors this file line by line.
"""
import numpy as np

from .quat import skew, qmul, qexp, q2R

G0 = 9.80665
CHI2_99 = {1: 6.63, 2: 9.21, 3: 11.34}


class ESKF:
    def __init__(self, q0, bg0=None, ba0=None, P0_sigmas=None, Na=3.92e-3, Ng=8.73e-5,
                 Kba=1e-5, Kbg=1e-6, g=G0):
        self.p = np.zeros(3)
        self.v = np.zeros(3)
        self.q = np.array(q0, float)
        self.ba = np.zeros(3) if ba0 is None else np.array(ba0, float)
        self.bg = np.zeros(3) if bg0 is None else np.array(bg0, float)
        s = P0_sigmas or dict(p=0.01, v=0.01, att=np.radians([0.2, 0.2, 0.5]), ba=0.02, bg=np.radians(0.01))
        self.P = np.diag(np.r_[np.full(3, s["p"]), np.full(3, s["v"]), np.broadcast_to(s["att"], 3),
                               np.full(3, s["ba"]), np.full(3, s["bg"])].astype(float) ** 2)
        self.Na, self.Ng, self.Kba, self.Kbg = Na, Ng, Kba, Kbg
        self.gw = np.array([0.0, 0.0, -g])

    @property
    def R(self):
        return q2R(self.q)

    def predict(self, f_m, w_m, dt):
        R = q2R(self.q)
        f = f_m - self.ba
        w = w_m - self.bg
        a = R @ f + self.gw
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        dq = qexp(w * dt)
        self.q = qmul(self.q, dq)
        self.q /= np.linalg.norm(self.q)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ skew(f) * dt
        F[3:6, 9:12] = -R * dt
        F[6:9, 6:9] = q2R(dq).T
        F[6:9, 12:15] = -np.eye(3) * dt
        Qd = np.diag(np.r_[np.zeros(3), np.full(3, self.Na**2 * dt), np.full(3, self.Ng**2 * dt),
                           np.full(3, self.Kba**2 * dt), np.full(3, self.Kbg**2 * dt)])
        self.P = F @ self.P @ F.T + Qd

    def update(self, nu, H, Rz, gate=True):
        nu = np.atleast_1d(nu)
        H = np.atleast_2d(H)
        Rz = np.atleast_2d(Rz)
        S = H @ self.P @ H.T + Rz
        nis = float(nu @ np.linalg.solve(S, nu))
        if gate and nis > CHI2_99.get(len(nu), 16.0):
            return False, nis
        K = self.P @ H.T @ np.linalg.inv(S)
        dx = K @ nu
        IKH = np.eye(15) - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ Rz @ K.T
        self.P = 0.5 * (self.P + self.P.T)
        self.inject(dx)
        return True, nis

    def inject(self, dx):
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = qmul(self.q, qexp(dx[6:9]))
        self.q /= np.linalg.norm(self.q)
        self.ba = self.ba + dx[9:12]
        self.bg = self.bg + dx[12:15]
        Gm = np.eye(15)
        Gm[6:9, 6:9] -= skew(0.5 * dx[6:9])
        self.P = Gm @ self.P @ Gm.T

    # ---------------- measurement models (Chapters 12 and 13)
    def zupt(self, sigma):
        H = np.zeros((3, 15)); H[:, 3:6] = np.eye(3)
        return self.update(-self.v, H, sigma**2 * np.eye(3))

    def zaru(self, w_m, sigma):
        H = np.zeros((3, 15)); H[:, 12:15] = np.eye(3)
        return self.update(w_m - self.bg, H, sigma**2 * np.eye(3))

    def nhc(self, sigma):
        R = q2R(self.q)
        vb = R.T @ self.v
        S = np.array([[0, 1.0, 0], [0, 0, 1.0]])
        H = np.zeros((2, 15))
        H[:, 3:6] = S @ R.T
        H[:, 6:9] = S @ skew(vb)
        return self.update(-(S @ vb), H, sigma**2 * np.eye(2))

    def line(self, sigma, y0=0.0):
        H = np.zeros((1, 15)); H[0, 1] = 1.0
        return self.update(np.array([y0 - self.p[1]]), H, np.array([[sigma**2]]))

    def body_velocity(self, v_meas_b, sigma, rows=(0, 1)):
        """Optical flow (rows 0,1) or wheel speed + NHC (rows 0,1,2)."""
        R = q2R(self.q)
        vb = R.T @ self.v
        S = np.eye(3)[list(rows)]
        H = np.zeros((len(rows), 15))
        H[:, 3:6] = S @ R.T
        H[:, 6:9] = S @ skew(vb)
        return self.update(np.asarray(v_meas_b) - S @ vb, H, sigma**2 * np.eye(len(rows)))

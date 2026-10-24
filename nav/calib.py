"""Multi-position IMU calibration (Tedaldi, Pretto, Menegatti, ICRA 2014), Chapter 07.

Model (raw sensor reading S -> orthogonal calibrated frame O):
    a_O = Ta Ka (a_S + ba)      Ta upper triangular, 3 angles
    w_O = Tg Kg (w_S + bg)      Tg full, 6 angles; bg fixed from the initial static period
"""
from dataclasses import dataclass, field
import numpy as np
import yaml
from scipy.optimize import least_squares

from .quat import qnorm


def local_gravity(lat_deg, h_m=0.0):
    """WGS84 normal gravity (Somigliana) with free-air correction, m/s^2."""
    s2 = np.sin(np.radians(lat_deg)) ** 2
    g0 = 9.7803253359 * (1 + 0.00193185265241 * s2) / np.sqrt(1 - 0.00669437999013 * s2)
    return g0 - 3.086e-6 * h_m


def Ta_of(p):
    ayz, azy, azx = p
    return np.array([[1, -ayz, azy], [0, 1, -azx], [0, 0, 1.0]])


def Tg_of(p):
    gyz, gzy, gxz, gzx, gxy, gyx = p
    return np.array([[1, -gyz, gzy], [gxz, 1, -gzx], [-gxy, gyx, 1.0]])


@dataclass
class CalibParams:
    Ta: np.ndarray = field(default_factory=lambda: np.eye(3))
    Ka: np.ndarray = field(default_factory=lambda: np.eye(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Tg: np.ndarray = field(default_factory=lambda: np.eye(3))
    Kg: np.ndarray = field(default_factory=lambda: np.eye(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    info: dict = field(default_factory=dict)

    def apply_acc(self, a):
        return (self.Ta @ self.Ka @ (np.atleast_2d(a) + self.ba).T).T

    def apply_gyro(self, w):
        return (self.Tg @ self.Kg @ (np.atleast_2d(w) + self.bg).T).T

    def to_yaml(self, path):
        d = {k: np.asarray(getattr(self, k)).tolist() for k in ["Ta", "Ka", "ba", "Tg", "Kg", "bg"]}
        d["convention"] = "a_O = Ta Ka (a_S + ba); w_O = Tg Kg (w_S + bg); units m/s^2, rad/s"
        d["info"] = self.info
        with open(path, "w") as f:
            yaml.safe_dump(d, f, sort_keys=False)

    @staticmethod
    def from_yaml(path):
        with open(path) as f:
            d = yaml.safe_load(f)
        return CalibParams(**{k: np.array(d[k], float) for k in ["Ta", "Ka", "ba", "Tg", "Kg", "bg"]},
                           info=d.get("info", {}))


# ---------------------------------------------------------------- static detector
def variance_magnitude(acc, w):
    """Centred sliding-window variance per axis, combined: sqrt(sum var_i^2)."""
    c1 = np.cumsum(np.vstack([np.zeros(3), acc]), axis=0)
    c2 = np.cumsum(np.vstack([np.zeros(3), acc ** 2]), axis=0)
    n = len(acc)
    h = w // 2
    lo = np.clip(np.arange(n) - h, 0, n)
    hi = np.clip(np.arange(n) + h + 1, 0, n)
    cnt = (hi - lo)[:, None]
    mean = (c1[hi] - c1[lo]) / cnt
    var = (c2[hi] - c2[lo]) / cnt - mean ** 2
    return np.sqrt(np.sum(np.maximum(var, 0) ** 2, axis=1))


def static_intervals(acc, fs, t_init=30.0, k=3.0, win_s=1.0, min_s=1.0, trim=0.15):
    """Return list of (start, end) sample indices of static intervals."""
    w = max(3, int(win_s * fs))
    n_init = int(t_init * fs)
    var_init = acc[:n_init].var(axis=0)
    s_init = np.sqrt(np.sum(var_init ** 2))
    s = variance_magnitude(acc, w)
    mask = s ** 2 < k * s_init ** 2
    mask[:n_init] = True                      # initial period is static by protocol
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    out = []
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a >= min_s * fs:
            m = int((b - a) * trim) if a > 0 else 0
            out.append((a + m, b - m))
    return out


# ---------------------------------------------------------------- accelerometer
def _acc_residual(p, means, g):
    a = (Ta_of(p[0:3]) @ np.diag(p[3:6]) @ (means + p[6:9]).T).T
    return g ** 2 - np.sum(a ** 2, axis=1)


def fit_accel(means, g):
    p0 = np.r_[0, 0, 0, 1, 1, 1, 0, 0, 0.0]
    r = least_squares(_acc_residual, p0, args=(means, g), method="lm", x_scale="jac")
    J = r.jac
    dof = max(1, len(means) - 9)
    s2 = np.sum(r.fun ** 2) / dof
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        std = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        std = np.full(9, np.nan)
    return r.x, r.fun, std


def calibrate_accel(acc, fs, g, t_init=30.0, ks=range(1, 11), win_s=1.0, min_intervals=12):
    """Parameterless: sweep detector threshold k, keep the fit with least mean residual."""
    best = None
    for k in ks:
        iv = static_intervals(acc, fs, t_init, k=k, win_s=win_s)
        if len(iv) < min_intervals:
            continue
        means = np.array([acc[a:b].mean(axis=0) for a, b in iv])
        p, res, std = fit_accel(means, g)
        score = np.mean(res ** 2)
        if best is None or score < best["score"]:
            best = dict(k=k, intervals=iv, params=p, residuals=res, std=std, score=score)
    if best is None:
        raise RuntimeError("Too few static intervals; collect more poses or a longer initial period")
    return best


# ---------------------------------------------------------------- gyroscope
def _qmul_v(p, q):
    w1, x1, y1, z1 = p.T
    w2, x2, y2, z2 = q.T
    return np.stack([w1*w2 - x1*x2 - y1*y2 - z1*z2, w1*x2 + x1*w2 + y1*z2 - z1*y2,
                     w1*y2 - x1*z2 + y1*w2 + z1*x2, w1*z2 + x1*y2 - y1*x2 + z1*w2], axis=1)


def _qdot(q, w):
    return 0.5 * _qmul_v(q, np.column_stack([np.zeros(len(w)), w]))


def _q2R_v(q):
    w, x, y, z = q.T
    return np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
                     np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
                     np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], 1)


def integrate_rk4(W, DT):
    """W: (M, L, 3) rates padded with zeros; DT: (M, L-1). Returns final quaternions (M,4)."""
    M, L, _ = W.shape
    q = np.tile([1.0, 0, 0, 0], (M, 1))
    for k in range(L - 1):
        dt = DT[:, k:k + 1]
        w0, w1 = W[:, k], W[:, k + 1]
        wm = 0.5 * (w0 + w1)
        k1 = _qdot(q, w0)
        k2 = _qdot(q + 0.5 * dt * k1, wm)
        k3 = _qdot(q + 0.5 * dt * k2, wm)
        k4 = _qdot(q + dt * k3, w1)
        q = q + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def _motions(gyro, t, intervals):
    segs = [(intervals[j - 1][1] - 1, intervals[j][0] + 1) for j in range(1, len(intervals))]
    L = max(b - a for a, b in segs)
    W = np.zeros((len(segs), L, 3))
    DT = np.zeros((len(segs), L - 1))
    for i, (a, b) in enumerate(segs):
        W[i, :b - a] = gyro[a:b]
        DT[i, :b - a - 1] = np.diff(t[a:b])
    lens = np.array([b - a for a, b in segs])
    return W, DT, lens


def _gyro_residual(p, W, DT, lens, U, bg):
    T = Tg_of(p[0:6]) @ np.diag(p[6:9])
    Wc = np.einsum("ij,mlj->mli", T, W + bg)
    mask = (np.arange(W.shape[1])[None, :] < lens[:, None])[..., None]
    Wc = Wc * mask
    q = integrate_rk4(Wc, DT)
    R = _q2R_v(q)
    pred = np.einsum("mji,mj->mi", R, U[:-1])        # R^T u_{j-1}
    return (U[1:] - pred).ravel()


def calibrate_gyro(gyro, t, intervals, acc_means_cal, t_init_samples):
    bg = -gyro[:t_init_samples].mean(axis=0)
    U = acc_means_cal / np.linalg.norm(acc_means_cal, axis=1, keepdims=True)
    W, DT, lens = _motions(gyro, t, intervals)
    p0 = np.r_[np.zeros(6), 1, 1, 1.0]
    r = least_squares(_gyro_residual, p0, args=(W, DT, lens, U, bg), method="lm", x_scale="jac")
    J = r.jac
    dof = max(1, len(r.fun) - 9)
    s2 = np.sum(r.fun ** 2) / dof
    try:
        std = np.sqrt(np.diag(s2 * np.linalg.inv(J.T @ J)))
    except np.linalg.LinAlgError:
        std = np.full(9, np.nan)
    return r.x, bg, r.fun, std


def calibrate(imu, g, t_init=30.0):
    """Full pipeline on one calibration session. imu: nav.io.Imu."""
    t = imu.t
    fs = 1.0 / np.median(np.diff(t))
    A = calibrate_accel(imu.acc, fs, g, t_init)
    pa = A["params"]
    cp = CalibParams(Ta=Ta_of(pa[0:3]), Ka=np.diag(pa[3:6]), ba=pa[6:9])
    means_cal = cp.apply_acc(np.array([imu.acc[a:b].mean(axis=0) for a, b in A["intervals"]]))
    pg, bg, res_g, std_g = calibrate_gyro(imu.gyro, t, A["intervals"], means_cal, int(t_init * fs))
    cp.Tg, cp.Kg, cp.bg = Tg_of(pg[0:6]), np.diag(pg[6:9]), bg
    cp.info = dict(k=int(A["k"]), n_poses=len(A["intervals"]), g=float(g),
                   acc_rms_residual_mg=float(np.sqrt(np.mean((A["residuals"] / (2 * g)) ** 2)) / g * 1e3),
                   gyro_rms_residual_deg=float(np.degrees(np.sqrt(np.mean(res_g ** 2)))),
                   acc_param_std=A["std"].tolist(), gyro_param_std=std_g.tolist())
    return cp, A["intervals"]


# ---------------------------------------------------------------- baseline and validation
def six_position(imu, g, t_init=30.0):
    """Baseline: per-axis bias and scale from the most axis-aligned static poses; gyro bias only."""
    fs = 1.0 / np.median(np.diff(imu.t))
    iv = static_intervals(imu.acc, fs, t_init, k=3)
    means = np.array([imu.acc[a:b].mean(axis=0) for a, b in iv])
    ba, ka = np.zeros(3), np.ones(3)
    for i in range(3):
        rp, rm = means[:, i].max(), means[:, i].min()
        if rp < 0.8 * g or rm > -0.8 * g:
            raise RuntimeError(f"Axis {i}: need both up and down poses")
        ba[i] = -0.5 * (rp + rm)
        ka[i] = 2 * g / (rp - rm)
    bg = -imu.gyro[:int(t_init * fs)].mean(axis=0)
    return CalibParams(Ka=np.diag(ka), ba=ba, bg=bg, info=dict(method="six-position"))


def gravity_norm_errors(cp, imu, g, t_init=10.0):
    """Held-out check C1: |a_O| - g on every static interval, in mg."""
    fs = 1.0 / np.median(np.diff(imu.t))
    iv = static_intervals(imu.acc, fs, t_init, k=3)
    means = np.array([imu.acc[a:b].mean(axis=0) for a, b in iv])
    a = cp.apply_acc(means) if cp is not None else means
    return (np.linalg.norm(a, axis=1) - g) / g * 1e3


def rotation_angle(cp, imu, i0, i1):
    """Held-out check C2: total rotation angle (deg) integrated between samples i0 and i1."""
    w = cp.apply_gyro(imu.gyro[i0:i1]) if cp is not None else imu.gyro[i0:i1]
    q = integrate_rk4(w[None], np.diff(imu.t[i0:i1])[None])[0]
    return np.degrees(2 * np.arccos(np.clip(abs(q[0]), 0, 1)))

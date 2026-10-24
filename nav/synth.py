"""Synthetic data with exact ground truth (Chapter 15).

Two generators:
  calibration_session()  static poses joined by smooth rotations, for testing nav.calib
  track_run()            the 25 m out-and-back trolley run with stops, for testing the filter

Both produce *ideal* calibrated signals first, then corrupt them with an ErrorModel.
"""
from dataclasses import dataclass, field
import numpy as np

from .quat import qmul, qexp, qlog, q2R, qconj, euler2q
from .io import Imu, Dataset

G0 = 9.80665


def min_jerk(tau):
    tau = np.clip(tau, 0, 1)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    ds = 30 * tau**2 - 60 * tau**3 + 30 * tau**4
    dds = 60 * tau - 180 * tau**2 + 120 * tau**3
    return s, ds, dds


@dataclass
class ErrorModel:
    """Raw = (T K)^-1 * ideal - b  + noise, then quantised. Defaults ~ MPU6050 datasheet."""
    Ta: np.ndarray = field(default_factory=lambda: np.eye(3))
    Ka: np.ndarray = field(default_factory=lambda: np.eye(3))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Tg: np.ndarray = field(default_factory=lambda: np.eye(3))
    Kg: np.ndarray = field(default_factory=lambda: np.eye(3))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3))
    Na: float = 400e-6 * G0                 # m/s^2/sqrt(Hz)
    Ng: float = np.radians(0.005)           # rad/s/sqrt(Hz)
    Ka_rw: float = 0.0                      # accel bias random walk, m/s^3/sqrt(Hz)
    Kg_rw: float = 0.0                      # gyro bias random walk, rad/s^2/sqrt(Hz)
    temp_slope_bg: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s per degC
    temp_rate: float = 0.0                  # degC per s (warm-up drift during the run)
    lsb_acc: float = G0 / 8192              # +-4 g
    lsb_gyro: float = np.radians(1 / 65.5)  # +-500 dps
    jitter_us: float = 0.0
    seed: int = 0

    @staticmethod
    def random_mpu6050(seed=0, scale=1.0):
        """Plausible *uncalibrated* MPU6050 errors, for calibration round-trip tests."""
        r = np.random.default_rng(seed)
        ang = lambda n, s: r.normal(0, s, n)
        from .calib import Ta_of, Tg_of
        return ErrorModel(Ta=Ta_of(ang(3, 0.01 * scale)), Ka=np.diag(1 + r.normal(0, 0.02 * scale, 3)),
                          ba=r.normal(0, 0.3 * scale, 3),
                          Tg=Tg_of(ang(6, 0.01 * scale)), Kg=np.diag(1 + r.normal(0, 0.02 * scale, 3)),
                          bg=r.normal(0, np.radians(2) * scale, 3), seed=seed)

    def apply(self, t, gyro_ideal, acc_ideal):
        r = np.random.default_rng(self.seed + 1)
        n = len(t)
        dt = np.median(np.diff(t))
        rw = lambda K: np.cumsum(r.normal(0, K * np.sqrt(dt), (n, 3)), axis=0)
        temp = 25 + self.temp_rate * (t - t[0])
        bg = self.bg + rw(self.Kg_rw) + np.outer(temp - 25, self.temp_slope_bg)
        ba = self.ba + rw(self.Ka_rw)
        acc = (np.linalg.inv(self.Ta @ self.Ka) @ acc_ideal.T).T - ba
        gyr = (np.linalg.inv(self.Tg @ self.Kg) @ gyro_ideal.T).T - bg
        acc = acc + r.normal(0, self.Na / np.sqrt(dt), (n, 3))
        gyr = gyr + r.normal(0, self.Ng / np.sqrt(dt), (n, 3))
        if self.lsb_acc:
            acc = np.round(acc / self.lsb_acc) * self.lsb_acc
        if self.lsb_gyro:
            gyr = np.round(gyr / self.lsb_gyro) * self.lsb_gyro
        t_us = np.round(t * 1e6 + (r.normal(0, self.jitter_us, n) if self.jitter_us else 0)).astype(np.int64)
        return Imu(t_us=t_us, gyro=gyr, acc=acc, temp=temp, seq=np.arange(n))

    def as_config(self):
        return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


# ---------------------------------------------------------------- calibration session
def calibration_session(n_poses=40, fs=200.0, t_init=40.0, hold=3.0, move=1.5, g=G0, seed=0):
    """Ideal signals: initial static period, then n_poses random orientations held still."""
    r = np.random.default_rng(seed)
    faces = [euler2q(0, 0, 0), euler2q(np.pi, 0, 0), euler2q(np.pi / 2, 0, 0), euler2q(-np.pi / 2, 0, 0),
             euler2q(0, np.pi / 2, 0), euler2q(0, -np.pi / 2, 0)]
    poses = list(faces)
    while len(poses) < n_poses:
        v = r.normal(size=4)
        poses.append(v / np.linalg.norm(v))
    q = euler2q(0, 0, 0)
    segs = [("hold", q, None, t_init)]
    for p in poses[1:]:
        segs.append(("move", q, p, move))
        segs.append(("hold", p, None, hold))
        q = p
    dt = 1.0 / fs
    T, Q, W = [], [], []
    t0 = 0.0
    for kind, qa, qb, dur in segs:
        ts = np.arange(0, dur, dt)
        if kind == "hold":
            for tt in ts:
                T.append(t0 + tt); Q.append(qa); W.append(np.zeros(3))
        else:
            phi = qlog(qmul(qconj(qa), qb))         # body-frame rotation vector
            s, ds, _ = min_jerk(ts / dur)
            for tt, si, dsi in zip(ts, s, ds):
                T.append(t0 + tt); Q.append(qmul(qa, qexp(si * phi))); W.append(dsi / dur * phi)
        t0 += dur
    T, Q, W = np.array(T), np.array(Q), np.array(W)
    F = np.array([q2R(qq).T @ np.array([0, 0, g]) for qq in Q])   # rotation about the sensor: a = 0
    return T, W, F, Q


# ---------------------------------------------------------------- track run
def track_run(fs=200.0, length=25.0, spacing=5.0, seg_time=10.0, stop=3.0, t_static=10.0,
              turn_time=4.0, bump_deg=0.0, bump_hz=3.0, g=G0):
    """Out-and-back along x: stops at every marker, 180 deg turn at the far end.

    Returns t, gyro_ideal, acc_ideal, gt (N,11), markers (K,3).
    """
    dt = 1.0 / fs
    n_seg = int(round(length / spacing))
    plan = [("static", 0.0, 0.0, 0.0, t_static)]
    for k in range(n_seg):
        plan += [("move", k * spacing, spacing, 0.0, seg_time), ("static", (k + 1) * spacing, 0, 0.0, stop)]
    plan += [("turn", length, 0, 0.0, turn_time), ("static", length, 0, np.pi, stop)]
    for k in range(n_seg):
        s0 = length - k * spacing
        plan += [("move", s0, -spacing, np.pi, seg_time), ("static", s0 - spacing, 0, np.pi, stop)]
    plan[-1] = ("static", 0.0, 0, np.pi, t_static)

    T, S, V, A, YAW, YR, YA = [], [], [], [], [], [], []
    markers = []
    t0 = 0.0
    for i, (kind, s0, ds, yaw, dur) in enumerate(plan):
        ts = np.arange(0, dur, dt)
        tau = ts / dur
        if kind == "static":
            markers.append((t0 + 0.5 * dur, s0))
            S += [s0] * len(ts); V += [0] * len(ts); A += [0] * len(ts)
            YAW += [yaw] * len(ts); YR += [0] * len(ts); YA += [0] * len(ts)
        elif kind == "move":
            m, dm, ddm = min_jerk(tau)
            S += list(s0 + ds * m); V += list(ds * dm / dur); A += list(ds * ddm / dur**2)
            YAW += [yaw] * len(ts); YR += [0] * len(ts); YA += [0] * len(ts)
        else:
            m, dm, ddm = min_jerk(tau)
            S += [s0] * len(ts); V += [0] * len(ts); A += [0] * len(ts)
            YAW += list(np.pi * m); YR += list(np.pi * dm / dur); YA += list(np.pi * ddm / dur**2)
        T += list(t0 + ts)
        t0 += dur
    T, S, V, A = map(np.array, (T, S, V, A))
    YAW, YR = np.array(YAW), np.array(YR)
    env = np.clip(np.abs(V) / 0.2, 0, 1)          # vibration fades in and out with speed (continuous)
    amp = np.radians(bump_deg)
    roll = amp * np.sin(2 * np.pi * bump_hz * T) * env
    pitch = amp * np.sin(2 * np.pi * bump_hz * 1.3 * T + 1.0) * env
    droll = np.gradient(roll, T)
    dpitch = np.gradient(pitch, T)
    # ZYX Euler rates -> body rates
    wx = droll - YR * np.sin(pitch)
    wy = dpitch * np.cos(roll) + YR * np.cos(pitch) * np.sin(roll)
    wz = -dpitch * np.sin(roll) + YR * np.cos(pitch) * np.cos(roll)
    gyro = np.column_stack([wx, wy, wz])
    Q = np.array([euler2q(a, b, c) for a, b, c in zip(roll, pitch, YAW)])
    acc_w = np.column_stack([A, np.zeros_like(A), np.zeros_like(A)])
    acc = np.array([q2R(q).T @ (a + np.array([0, 0, g])) for q, a in zip(Q, acc_w)])
    gt = np.column_stack([np.round(T * 1e6), S, np.zeros_like(S), np.zeros_like(S), Q,
                          V, np.zeros_like(V), np.zeros_like(V)])
    mk = np.array([(round(t * 1e6), i, s) for i, (t, s) in enumerate(markers)])
    return T, gyro, acc, gt, mk


def make_track_dataset(err: ErrorModel, **kw):
    t, w, f, gt, mk = track_run(**kw)
    imu = err.apply(t, w, f)
    cfg = dict(source="synthetic track_run", track_kw={k: v for k, v in kw.items()}, error_model=err.as_config())
    return Dataset(imu=imu, markers=mk, gt=gt, config=cfg)


def make_calibration_dataset(err: ErrorModel, **kw):
    t, w, f, q = calibration_session(**kw)
    imu = err.apply(t, w, f)
    return Dataset(imu=imu, config=dict(source="synthetic calibration_session", error_model=err.as_config()))

"""Synthetic data with exact ground truth (Chapter 15).

calibration_session()  static poses joined by smooth rotations, for testing nav.calib
static_session()       zero motion, pure sensor-error simulation, for testing noise/drift analysis

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


# ---------------------------------------------------------------- static session (drift ground truth)
def static_session(duration_s=600.0, fs=200.0, g=G0):
    """Phone at rest, level, for the whole run: ideal signals are exactly zero rotation / zero acceleration.

    Returns t, gyro_ideal (all zero), acc_ideal (all [0,0,g]), gt (N,11): p=0, v=0, q=identity throughout.
    Any nonzero position/velocity/attitude a filter reports on this data is pure drift from sensor error.
    """
    t = np.arange(0, duration_s, 1.0 / fs)
    n = len(t)
    gyro = np.zeros((n, 3))
    acc = np.tile([0.0, 0.0, g], (n, 1))
    q = np.tile([1.0, 0, 0, 0], (n, 1))
    gt = np.column_stack([np.round(t * 1e6), np.zeros((n, 3)), q, np.zeros((n, 3))])
    return t, gyro, acc, gt


def make_static_dataset(err: ErrorModel, **kw):
    t, w, f, gt = static_session(**kw)
    imu = err.apply(t, w, f)
    return Dataset(imu=imu, gt=gt, config=dict(source="synthetic static_session", error_model=err.as_config()))


def make_calibration_dataset(err: ErrorModel, **kw):
    t, w, f, q = calibration_session(**kw)
    imu = err.apply(t, w, f)
    return Dataset(imu=imu, config=dict(source="synthetic calibration_session", error_model=err.as_config()))

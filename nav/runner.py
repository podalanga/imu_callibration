"""Causal navigation pipeline: calibration -> ESKF predict -> stop detector -> aiding updates.

Same logic as firmware/main/nav.c. Run on any dataset in the common format.
"""
from dataclasses import dataclass, asdict
import numpy as np

from .eskf import ESKF, G0
from .quat import euler2q, tilt_from_accel
from .calib import CalibParams


@dataclass
class NavConfig:
    # noise model (from Allan deviation; defaults = MPU6050 datasheet)
    Na: float = 400e-6 * G0
    Ng: float = np.radians(0.005)
    Kba: float = 1e-4
    Kbg: float = 2e-5
    g: float = G0
    # initialisation
    t_init: float = 8.0            # s of the initial static period used for alignment
    # stop detector (SHOE, causal window)
    win: int = 20                  # samples
    sigma_a: float = 0.05          # m/s^2
    sigma_w: float = 0.005         # rad/s
    gamma: float = 5.0             # tune with scripts/tune_detector.py
    min_still: float = 0.3         # s the detector must stay below gamma before ZUPT starts
    # updates
    use_zupt: bool = True
    zupt_sigma: float = 0.01       # m/s
    zaru_sigma: float = 0.002      # rad/s
    zupt_every: int = 4            # apply every n-th stationary sample
    use_nhc: bool = True           # trolley mode only
    nhc_sigma: float = 0.05
    nhc_every: int = 10
    nhc_max_wz: float = 0.3        # rad/s, NHC off while turning
    use_line: bool = False         # straight-line map constraint (report on and off)
    line_sigma: float = 0.05
    line_every: int = 20
    gate: bool = True

    def to_dict(self):
        return asdict(self)


class StopDetector:
    def __init__(self, cfg: NavConfig, fs):
        self.c = cfg
        self.buf_f = np.zeros((cfg.win, 3))
        self.buf_w = np.zeros((cfg.win, 3))
        self.n = 0
        self.still_count = 0
        self.need = int(round(cfg.min_still * fs))

    def step(self, f, w):
        i = self.n % self.c.win
        self.buf_f[i], self.buf_w[i] = f, w
        self.n += 1
        if self.n < self.c.win:
            return False, np.inf
        fm = self.buf_f.mean(axis=0)
        d = self.buf_f - self.c.g * fm / np.linalg.norm(fm)
        T = np.mean(np.sum(d**2, axis=1) / self.c.sigma_a**2 + np.sum(self.buf_w**2, axis=1) / self.c.sigma_w**2)
        self.still_count = self.still_count + 1 if T < self.c.gamma else 0
        return self.still_count >= self.need, T


def run(imu, cfg: NavConfig = None, calib: CalibParams = None, oracle_still=None, log_nis=True):
    """Run the filter causally over an Imu. Returns dict with est rows and diagnostics.

    oracle_still: optional bool array (from ground truth) to test the filter apart from the detector.
    """
    cfg = cfg or NavConfig()
    gyro = calib.apply_gyro(imu.gyro) if calib else imu.gyro
    acc = calib.apply_acc(imu.acc) if calib else imu.acc
    t = imu.t
    fs = 1.0 / np.median(np.diff(t))
    n0 = int(round(cfg.t_init * fs))
    roll, pitch = tilt_from_accel(acc[:n0].mean(axis=0))
    f = ESKF(euler2q(roll, pitch, 0.0), bg0=gyro[:n0].mean(axis=0),
             Na=cfg.Na, Ng=cfg.Ng, Kba=cfg.Kba, Kbg=cfg.Kbg, g=cfg.g)
    det = StopDetector(cfg, fs)
    rows, nis = [], {"zupt": [], "zaru": [], "nhc": [], "line": []}
    rejected = {k: 0 for k in nis}
    still_flags = np.zeros(len(t), bool)
    k_still = 0
    for k in range(n0, len(t)):
        dt = t[k] - t[k - 1]
        f.predict(acc[k], gyro[k], dt)
        still, _ = det.step(acc[k], gyro[k])
        if oracle_still is not None:
            still = bool(oracle_still[k])
        still_flags[k] = still
        if still and cfg.use_zupt:
            k_still += 1
            if k_still % cfg.zupt_every == 0:
                for name, res in (("zupt", f.zupt(cfg.zupt_sigma)), ("zaru", f.zaru(gyro[k], cfg.zaru_sigma))):
                    ok, e = res
                    nis[name].append(e); rejected[name] += (not ok)
        else:
            k_still = 0
            if cfg.use_nhc and k % cfg.nhc_every == 0 and abs(gyro[k][2] - f.bg[2]) < cfg.nhc_max_wz:
                ok, e = f.nhc(cfg.nhc_sigma); nis["nhc"].append(e); rejected["nhc"] += (not ok)
        if cfg.use_line and k % cfg.line_every == 0:
            ok, e = f.line(cfg.line_sigma); nis["line"].append(e); rejected["line"] += (not ok)
        rows.append(np.r_[imu.t_us[k], f.p, f.v, f.q, f.ba, f.bg, still])
    return dict(est=np.array(rows), nis={k: np.array(v) for k, v in nis.items()},
                rejected=rejected, still=still_flags, filter=f)


def oracle_from_gt(gt, n_imu, speed_thresh=1e-4, yaw_rate=None):
    """Truly stationary samples from synthetic ground truth (|v| ~ 0 and not turning)."""
    v = np.linalg.norm(gt[:, 8:11], axis=1)
    still = v < speed_thresh
    q = gt[:, 4:8]
    dq = np.r_[0, np.linalg.norm(np.diff(q, axis=0), axis=1)]
    still &= dq < 1e-7
    return still[:n_imu]

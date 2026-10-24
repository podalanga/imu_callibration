"""Common dataset format (Chapter 06 of the guide).

dataset_dir/
  config.yaml      free-form metadata (source, ranges, rate, seed, ...)
  imu.csv          t_us,seq,gx,gy,gz,ax,ay,az,temp_c   rad/s, m/s^2, raw sensor frame
  markers.csv      t_us,marker_id,s_m                  track runs only
  groundtruth.csv  t_us,px,py,pz,qw,qx,qy,qz,vx,vy,vz   synthetic / EuRoC only
  est.csv          t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,stationary
"""
import os
from dataclasses import dataclass, field
import numpy as np
import yaml

IMU_COLS = "t_us,seq,gx,gy,gz,ax,ay,az,temp_c"
MARKER_COLS = "t_us,marker_id,s_m"
GT_COLS = "t_us,px,py,pz,qw,qx,qy,qz,vx,vy,vz"
EST_COLS = "t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,stationary"


@dataclass
class Imu:
    t_us: np.ndarray            # int64
    gyro: np.ndarray            # (N,3) rad/s
    acc: np.ndarray             # (N,3) m/s^2
    temp: np.ndarray = None     # (N,) degC
    seq: np.ndarray = None      # (N,) int

    def __len__(self):
        return len(self.t_us)

    @property
    def t(self):
        return (self.t_us - self.t_us[0]) * 1e-6


@dataclass
class Dataset:
    imu: Imu
    markers: np.ndarray = None      # (K,3): t_us, id, s_m
    gt: np.ndarray = None           # (M,11)
    config: dict = field(default_factory=dict)


def csv_path(path):
    """path, or path + ".gz" when only the compressed copy exists (large recordings are stored gzipped so
    they fit GitHub's 100 MB file limit; np.loadtxt reads .gz directly)."""
    if not os.path.exists(path) and os.path.exists(path + ".gz"):
        return path + ".gz"
    return path


def _load(path, cols):
    path = csv_path(path)
    if not os.path.exists(path):
        return None
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    assert a.shape[1] == len(cols.split(",")), f"{path}: expected columns {cols}"
    return a


def load(d):
    a = _load(os.path.join(d, "imu.csv"), IMU_COLS)
    imu = Imu(t_us=a[:, 0].astype(np.int64), seq=a[:, 1].astype(np.int64),
              gyro=a[:, 2:5], acc=a[:, 5:8], temp=a[:, 8])
    cfg = {}
    if os.path.exists(os.path.join(d, "config.yaml")):
        with open(os.path.join(d, "config.yaml")) as f:
            cfg = yaml.safe_load(f) or {}
    return Dataset(imu=imu, markers=_load(os.path.join(d, "markers.csv"), MARKER_COLS),
                   gt=_load(os.path.join(d, "groundtruth.csv"), GT_COLS), config=cfg)


def _plain(x):
    """Convert numpy scalars/arrays inside nested config dicts to YAML-safe Python types."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    return x


def save(d, ds):
    os.makedirs(d, exist_ok=True)
    i = ds.imu
    temp = i.temp if i.temp is not None else np.full(len(i), 25.0)
    seq = i.seq if i.seq is not None else np.arange(len(i))
    a = np.column_stack([i.t_us, seq, i.gyro, i.acc, temp])
    np.savetxt(os.path.join(d, "imu.csv"), a, delimiter=",", header=IMU_COLS, comments="",
               fmt=["%d", "%d"] + ["%.9g"] * 7)
    if ds.markers is not None:
        np.savetxt(os.path.join(d, "markers.csv"), ds.markers, delimiter=",",
                   header=MARKER_COLS, comments="", fmt=["%d", "%d", "%.4f"])
    if ds.gt is not None:
        np.savetxt(os.path.join(d, "groundtruth.csv"), ds.gt, delimiter=",",
                   header=GT_COLS, comments="", fmt=["%d"] + ["%.9g"] * 10)
    with open(os.path.join(d, "config.yaml"), "w") as f:
        yaml.safe_dump(_plain(ds.config), f, sort_keys=False)


def save_est(path, rows):
    np.savetxt(path, np.asarray(rows), delimiter=",", header=EST_COLS, comments="",
               fmt=["%d"] + ["%.9g"] * 16 + ["%d"])


def load_est(path):
    return np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)

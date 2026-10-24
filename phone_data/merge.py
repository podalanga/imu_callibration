"""Merge a phone run's raw streams into the common dataset format (nav/io.py), and report acquisition health.

  python phone_data/merge.py runs/static_01

Inputs written by phone_logger.py: raw_acc.csv, raw_gyro.csv (t_ns + 6 values), battery.csv, meta.yaml.
The gyro timestamps are the master clock; the accelerometer is interpolated onto them (on this phone both
come from one FIFO with identical timestamps, so the interpolation is exact). `seq` is the sample index
implied by the timestamp, so a dropped sample shows up as a sequence gap in scripts/check_acq.py.
temp_c is the battery temperature (Android does not expose the IMU's own), NaN if it was not polled.
"""
import os, sys
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io
from phone_data.stream import load_raw


def clean(t, v):
    """Sort by timestamp and drop repeated timestamps. Returns t, v, n_dropped, n_out_of_order."""
    back = int(np.sum(np.diff(t) < 0))
    o = np.argsort(t, kind="stable")
    t, v = t[o], v[o]
    keep = np.r_[True, np.diff(t) > 0]
    return t[keep], v[keep], int(np.sum(~keep)), back


def timing(t_ns):
    """Sampling-interval statistics from hardware timestamps."""
    dt = np.diff(t_ns) * 1e-6                               # ms
    med = float(np.median(dt))
    gaps = dt > 1.5 * med
    return dict(n=int(len(t_ns)), duration_s=float((t_ns[-1] - t_ns[0]) * 1e-9), rate_hz=1e3 / med,
                dt_median_ms=med, dt_std_us=float(dt.std() * 1e3), dt_min_ms=float(dt.min()),
                dt_max_ms=float(dt.max()), gaps=int(gaps.sum()),
                missing_samples=int(np.sum(np.round(dt[gaps] / med) - 1)))


def timing_from_imu(imu):
    """Acquisition report for a dataset that was not recorded by phone_logger (synthetic, EuRoC)."""
    s = timing(imu.t_us.astype(np.int64) * 1000)
    return dict(acc=s, gyro=s)


def build(run_dir):
    ta, va = load_raw(os.path.join(run_dir, "raw_acc.csv"))
    tg, vg = load_raw(os.path.join(run_dir, "raw_gyro.csv"))
    ta, va, dup_a, back_a = clean(ta, va)
    tg, vg, dup_g, back_g = clean(tg, vg)
    m = (tg >= ta[0]) & (tg <= ta[-1])
    t, gyro = tg[m], vg[m, :3]
    acc = np.column_stack([np.interp(t, ta, va[:, i]) for i in range(3)])
    temp = np.full(len(t), np.nan)
    bp = os.path.join(run_dir, "battery.csv")
    if os.path.exists(bp):
        b = np.loadtxt(bp, delimiter=",", skiprows=1, ndmin=2)
        if len(b):
            temp = np.interp(t, b[:, 1], b[:, 2])
    dt_med = np.median(np.diff(t))
    seq = np.round((t - t[0]) / dt_med).astype(np.int64)
    meta = {}
    if os.path.exists(os.path.join(run_dir, "meta.yaml")):
        with open(os.path.join(run_dir, "meta.yaml")) as f:
            meta = yaml.safe_load(f) or {}
    report = dict(acc=timing(ta), gyro=timing(tg),
                  dropped_duplicates=dict(acc=dup_a, gyro=dup_g), out_of_order=dict(acc=back_a, gyro=back_g),
                  same_clock=bool(len(ta) == len(tg) and np.array_equal(ta, tg)))
    cfg = dict(meta, source="phone (SensorServer, uncalibrated accel + gyro)",
               frame="Android device frame: x right, y toward top edge, z out of the screen",
               units="gyro rad/s, acc m/s^2, temp_c = battery temperature",
               rate_hz=round(1e9 / dt_med, 2), acquisition=report,
               android_bias_estimate=dict(acc=va[:, 3:6].mean(axis=0).tolist(), gyro=vg[:, 3:6].mean(axis=0).tolist()))
    imu = io.Imu(t_us=(t // 1000).astype(np.int64), gyro=gyro, acc=acc, temp=temp, seq=seq)
    io.save(run_dir, io.Dataset(imu=imu, config=cfg))
    return report


def print_report(r):
    for k in ("acc", "gyro"):
        s = r[k]
        print(f"{k:4s} {s['n']:8d} samples  {s['duration_s']:8.1f} s  {s['rate_hz']:6.1f} Hz  "
              f"dt median {s['dt_median_ms']:.3f} ms  std {s['dt_std_us']:6.1f} us  max {s['dt_max_ms']:7.2f} ms  "
              f"gaps {s['gaps']} ({s['missing_samples']} samples missing)")
    print(f"duplicates dropped {r['dropped_duplicates']}, out of order {r['out_of_order']}, "
          f"accel and gyro share timestamps: {r['same_clock']}")


if __name__ == "__main__":
    print_report(build(sys.argv[1]))

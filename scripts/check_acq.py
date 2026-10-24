"""Acquisition acceptance tests A1, A2, A3 on a recorded log (gate M1).

  python scripts/check_acq.py runs/static_10min               # A1 timing + A2 noise (box still)
  python scripts/check_acq.py runs/faces --faces              # A3: box placed on its 6 faces, ~10 s each
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, calib

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--faces", action="store_true")
ap.add_argument("--rate", type=float, default=200.0)
a = ap.parse_args()
imu = io.load(a.dataset).imu
dt = np.diff(imu.t_us) * 1e-3
gaps = int(np.sum(np.diff(imu.seq) != 1))
print(f"samples {len(imu)}, duration {imu.t[-1]:.1f} s")
print(f"A1 interval: median {np.median(dt):.3f} ms, min {dt.min():.3f}, max {dt.max():.3f}, "
      f"std {dt.std()*1e3:.1f} us, sequence gaps {gaps}  ->  {'PASS' if gaps == 0 and abs(np.median(dt) - 1e3/a.rate) < 0.05 else 'CHECK'}")
if not a.faces:
    G = 9.80665
    sa, sg = imu.acc.std(axis=0), np.degrees(imu.gyro.std(axis=0))
    # datasheet: 400 ug/rtHz and 0.005 dps/rtHz over ~44 Hz DLPF bandwidth (equivalent noise bw ~ 1.57*44 for 1st order)
    exp_a, exp_g = 400e-6 * G * np.sqrt(1.57 * 44), 0.005 * np.sqrt(1.57 * 44)
    print(f"A2 accel std [mg]  {np.round(sa / G * 1e3, 2)}  (datasheet-level ~{exp_a / G * 1e3:.1f})")
    print(f"A2 gyro  std [dps] {np.round(sg, 4)}  (datasheet-level ~{exp_g:.3f})")
    print(f"   accel mean [m/s^2] {np.round(imu.acc.mean(axis=0), 3)}, |mean| {np.linalg.norm(imu.acc.mean(axis=0)):.3f}")
    print(f"   gyro mean [dps] {np.round(np.degrees(imu.gyro.mean(axis=0)), 3)} (raw bias, removed by calibration)")
    print(f"   temperature {imu.temp.min():.1f} to {imu.temp.max():.1f} C")
    rep = np.mean(np.all(np.diff(imu.acc, axis=0) == 0, axis=1))
    print(f"   repeated identical samples: {rep*100:.2f} %  (more than ~0 means stale reads)")
else:
    iv = calib.static_intervals(imu.acc, a.rate, t_init=min(5.0, imu.t[-1] / 4), k=3, min_s=2.0)
    print("A3 faces: axis that reads about +g must be the one pointing up")
    for s0, s1 in iv:
        m = imu.acc[s0:s1].mean(axis=0)
        i = int(np.argmax(np.abs(m)))
        print(f"   t={imu.t[s0]:6.1f}s  mean {np.round(m, 2)}  ->  {'-' if m[i] < 0 else '+'}{'xyz'[i]} up")

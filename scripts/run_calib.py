"""Calibrate from a multi-position session and validate on held-out sessions (tests C1, C3, C6).

  python scripts/run_calib.py runs/calib_s1 --holdout runs/calib_s2 runs/calib_s3 \
         --lat 46.52 --alt 400 --out calib.yaml --plots results/calib

Use the venue's latitude and altitude for local gravity.
"""
import argparse, os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, calib

ap = argparse.ArgumentParser()
ap.add_argument("session")
ap.add_argument("--holdout", nargs="*", default=[])
ap.add_argument("--lat", type=float, required=True)
ap.add_argument("--alt", type=float, default=0.0)
ap.add_argument("--t_init", type=float, default=40.0)
ap.add_argument("--out", default="calib.yaml")
ap.add_argument("--plots", default=None)
a = ap.parse_args()

g = calib.local_gravity(a.lat, a.alt)
ds = io.load(a.session)
cp, iv = calib.calibrate(ds.imu, g, a.t_init)
cp.info["session"] = a.session
cp.to_yaml(a.out)
print(f"local g = {g:.5f} m/s^2; {len(iv)} poses; detector k = {cp.info['k']}")
print(json.dumps({k: v for k, v in cp.info.items() if "std" not in k}, indent=1))
print("Ka", np.diag(cp.Ka).round(5), "ba", cp.ba.round(4))
print("Kg", np.diag(cp.Kg).round(5), "bg [deg/s]", np.degrees(cp.bg).round(4))

baseline = calib.six_position(ds.imu, g, a.t_init)
rows = []
for h in a.holdout:
    hi = io.load(h).imu
    for name, p in (("raw", None), ("six-position", baseline), ("multi-position", cp)):
        e = calib.gravity_norm_errors(p, hi, g)
        rows.append((h, name, float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e)))))
        print(f"held-out {h:30s} {name:15s} |a|-g RMS {rows[-1][2]:7.3f} mg   max {rows[-1][3]:7.3f} mg")

if a.plots:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(a.plots, exist_ok=True)
    t = ds.imu.t
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax[0].plot(t, ds.imu.acc, lw=0.6); ax[0].set_ylabel("Accel [m/s²]")
    for s0, s1 in iv:
        ax[0].axvspan(t[s0], t[s1 - 1], color="g", alpha=0.15)
    ax[1].plot(t, np.degrees(ds.imu.gyro), lw=0.6); ax[1].set_ylabel("Gyro [°/s]"); ax[1].set_xlabel("Time [s]")
    ax[0].set_title(f"Calibration session: {len(iv)} detected static intervals (green)")
    fig.tight_layout(); fig.savefig(os.path.join(a.plots, "static_intervals.png"), dpi=110)
    if a.holdout:
        hi = io.load(a.holdout[0]).imu
        fig, ax = plt.subplots(figsize=(7, 4))
        for name, p in (("raw", None), ("six-position", baseline), ("multi-position", cp)):
            ax.hist(calib.gravity_norm_errors(p, hi, g), bins=30, alpha=0.6, label=name)
        ax.set_xlabel("|a| − g on held-out static poses [mg]"); ax.set_ylabel("Poses"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(a.plots, "heldout_gravity_norm.png"), dpi=110)
    print("plots in", a.plots)

"""Write synthetic datasets in the common format.

  python scripts/make_synth.py calib data/synth_calib --seed 3
  python scripts/make_synth.py track data/synth_track --seed 5 --bump 0.3 [--stress]
  python scripts/make_synth.py static data/synth_static --minutes 60      (Allan test input)
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import synth, io, calib

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["calib", "track", "static"])
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0, help="session seed (poses, noise)")
ap.add_argument("--sensor", type=int, default=3, help="sensor seed: same value = same simulated MPU6050")
ap.add_argument("--bump", type=float, default=0.3)
ap.add_argument("--spacing", type=float, default=5.0)
ap.add_argument("--minutes", type=float, default=60)
ap.add_argument("--stress", action="store_true", help="add effects the filter does not model")
a = ap.parse_args()
G = synth.G0

if a.kind == "calib":
    err = synth.ErrorModel.random_mpu6050(seed=a.sensor)
    err.seed = a.seed
    ds = synth.make_calibration_dataset(err, seed=a.seed)
elif a.kind == "track":
    err = synth.ErrorModel(ba=np.array([2e-3, -1.5e-3, 1e-3]) * G, bg=np.radians([0.02, -0.02, 0.02]),
                           Ka_rw=1e-4, Kg_rw=2e-5, seed=a.seed)
    if a.stress:
        err.Ta = calib.Ta_of([0.002, -0.002, 0.002]); err.Ka = np.diag([1.003, 0.997, 1.002])
        err.Tg = calib.Tg_of([0.003, -0.002, 0.002, 0.003, -0.003, 0.002]); err.Kg = np.diag([1.01, 0.99, 1.01])
        err.temp_slope_bg = np.radians([0.003] * 3); err.temp_rate = 0.02; err.jitter_us = 20
    ds = synth.make_track_dataset(err, bump_deg=a.bump, spacing=a.spacing)
else:
    fs = 200.0
    t = np.arange(0, a.minutes * 60, 1 / fs)
    err = synth.ErrorModel(Ka_rw=1e-4, Kg_rw=2e-5, seed=a.seed)
    imu = err.apply(t, np.zeros((len(t), 3)), np.tile([0, 0, G], (len(t), 1)))
    ds = io.Dataset(imu=imu, config=dict(source="synthetic static", error_model=err.as_config()))
io.save(a.out, ds)
print(f"wrote {a.out}: {len(ds.imu)} samples")

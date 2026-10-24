"""Write synthetic datasets in the common format.

  python scripts/make_synth.py calib data/synth_calib --seed 3
  python scripts/make_synth.py static data/synth_static --minutes 60      (Allan test / drift input)
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import synth, io

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["calib", "static"])
ap.add_argument("out")
ap.add_argument("--seed", type=int, default=0, help="session seed (poses, noise)")
ap.add_argument("--sensor", type=int, default=3, help="sensor seed: same value = same simulated IMU")
ap.add_argument("--minutes", type=float, default=60)
a = ap.parse_args()

if a.kind == "calib":
    err = synth.ErrorModel.random_mpu6050(seed=a.sensor)
    err.seed = a.seed
    ds = synth.make_calibration_dataset(err, seed=a.seed)
else:
    err = synth.ErrorModel(Ka_rw=1e-4, Kg_rw=2e-5, seed=a.seed)
    ds = synth.make_static_dataset(err, duration_s=a.minutes * 60)
io.save(a.out, ds)
print(f"wrote {a.out}: {len(ds.imu)} samples")

"""Write synthetic datasets in the common format.

  python scripts/make_synth.py calib data/synth_calib --seed 3
  python scripts/make_synth.py static data/synth_static --minutes 60      (Allan test / drift input)
  python scripts/make_synth.py static data/phone_sim --minutes 30 --noise-model phone_noise.yaml
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
ap.add_argument("--noise-model", default=None,
                help="phone_noise.yaml from characterise_noise.py: simulate that real sensor's noise and rate")
ap.add_argument("--turn-on", choices=["measured", "random"], default="measured")
a = ap.parse_args()

fs = 200.0
if a.noise_model:
    err = synth.ErrorModel.from_noise_yaml(a.noise_model, seed=a.seed, turn_on=a.turn_on)
    import yaml
    with open(a.noise_model) as f:
        fs = float(yaml.safe_load(f)["error_model"]["fs"])
if a.kind == "calib":
    if not a.noise_model:
        err = synth.ErrorModel.random_mpu6050(seed=a.sensor)
    err.seed = a.seed
    ds = synth.make_calibration_dataset(err, seed=a.seed, fs=fs)
else:
    if not a.noise_model:
        err = synth.ErrorModel(Ka_rw=1e-4, Kg_rw=2e-5, seed=a.seed)
    ds = synth.make_static_dataset(err, duration_s=a.minutes * 60, fs=fs)
io.save(a.out, ds)
print(f"wrote {a.out}: {len(ds.imu)} samples")

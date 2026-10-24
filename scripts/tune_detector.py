"""Choose the stop-detector threshold on TRAINING runs (task B4).

Truth for "stopped" comes from the marker presses: each press is made during a stop, so the
samples within +-hold/2 of a press are labelled stationary; samples further than `margin` from
any press while the run is in progress are labelled moving.

  python scripts/tune_detector.py runs/track_01 runs/track_02 --calib calib.yaml
Pick the largest gamma with zero false positives (moving labelled as still): a false ZUPT
injects a wrong velocity with high confidence, a missed one only wastes a stop.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, runner
from nav.calib import CalibParams

ap = argparse.ArgumentParser()
ap.add_argument("datasets", nargs="+")
ap.add_argument("--calib", default=None)
ap.add_argument("--hold", type=float, default=2.0, help="seconds certainly still around a press")
ap.add_argument("--margin", type=float, default=4.0, help="seconds from a press beyond which it is moving")
a = ap.parse_args()
cp = CalibParams.from_yaml(a.calib) if a.calib else None

print(f"{'gamma':>6} {'sigma_w':>8} {'false still':>12} {'missed still':>13}")
for gamma in [3, 4, 5, 6, 8, 10, 15]:
    for sw in [0.003, 0.005, 0.01]:
        fp = fn = 0
        for d in a.datasets:
            ds = io.load(d)
            t = ds.imu.t_us * 1e-6
            press = ds.markers[:, 0] * 1e-6
            dist = np.min(np.abs(t[:, None] - press[None, :]), axis=1)
            still_true = dist < a.hold / 2
            moving_true = (dist > a.margin) & (t > press[0]) & (t < press[-1])
            r = runner.run(ds.imu, runner.NavConfig(gamma=gamma, sigma_w=sw), calib=cp)
            det = r["still"]
            fp += int(np.sum(det & moving_true))
            fn += int(np.sum(~det & still_true))
        print(f"{gamma:6.1f} {sw:8.3f} {fp:12d} {fn:13d}")

"""Allan deviation of a static recording; writes noise.yaml for the filter (task A3).

  python scripts/run_allan.py runs/static_3h --calib calib.yaml --out noise.yaml --plot results/allan.png
"""
import argparse, os, sys
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, allan
from nav.calib import CalibParams

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--calib", default=None)
ap.add_argument("--out", default="noise.yaml")
ap.add_argument("--plot", default=None)
a = ap.parse_args()

imu = io.load(a.dataset).imu
if a.calib:
    cp = CalibParams.from_yaml(a.calib)
    imu.gyro, imu.acc = cp.apply_gyro(imu.gyro), cp.apply_acc(imu.acc)
res = allan.analyse(imu)
mean = lambda key, ax: float(np.mean([res[f"{key}_{c}"].get(ax, np.nan) for c in "xyz"]))
noise = dict(Na=mean("acc", "N"), Ng=mean("gyro", "N"), B_acc=mean("acc", "B"), B_gyro=mean("gyro", "B"),
             K_acc=mean("acc", "K"), K_gyro=mean("gyro", "K"), duration_s=float(imu.t[-1]))
for k, v in res.items():
    print(f"{k:7s} N={v.get('N', np.nan):.3e}  B={v['B']:.3e} at tau={v['tau_B']:.0f}s  K={v.get('K', np.nan):.3e}")
print("gyro N =", np.degrees(noise["Ng"]), "deg/s/sqrt(Hz);  accel N =", noise["Na"] / 9.80665 * 1e6, "ug/sqrt(Hz)")
with open(a.out, "w") as f:
    yaml.safe_dump(noise, f)
if a.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, (key, unit, sc) in enumerate((("gyro", "°/s", np.degrees(1)), ("acc", "m/s²", 1.0))):
        for c in "xyz":
            r = res[f"{key}_{c}"]
            ax[i].errorbar(r["taus"], r["adev"] * sc, yerr=r["err"] * sc, fmt="-", lw=1, label=f"{c}")
        ax[i].set_xscale("log"); ax[i].set_yscale("log"); ax[i].grid(True, which="both", alpha=0.3)
        ax[i].set_xlabel("Averaging time τ [s]"); ax[i].set_ylabel(f"Allan deviation [{unit}]")
        ax[i].set_title(f"{key}: N at τ=1 s, B = min/0.664"); ax[i].legend()
    fig.tight_layout(); fig.savefig(a.plot, dpi=110)

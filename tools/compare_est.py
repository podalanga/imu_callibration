"""Compare an embedded estimate (HIL replay or host_replay) with the Python reference on the same data.

  python tools/compare_est.py results/hil_est.csv data/synth_track [--plot results/c_vs_py.png]
Reports the position difference over time and the CPU time per step if the file has a cycles column.
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, runner

ap = argparse.ArgumentParser()
ap.add_argument("est")
ap.add_argument("dataset")
ap.add_argument("--mhz", type=float, default=240.0)
ap.add_argument("--plot", default=None)
a = ap.parse_args()
with open(a.est) as fh:
    cols = fh.readline().strip().split(",")
e = np.loadtxt(a.est, delimiter=",", skiprows=1, ndmin=2)
p = runner.run(io.load(a.dataset).imu, runner.NavConfig())["est"]
common, ie, ip = np.intersect1d(e[:, 0], p[:, 0], return_indices=True)
d = np.linalg.norm(e[ie, 1:4] - p[ip, 1:4], axis=1)
print(f"{len(common)} common samples; position difference max {d.max()*1e3:.3f} mm, final {d[-1]*1e3:.3f} mm")
print(f"stop-flag disagreement: {np.mean(e[ie, cols.index('stationary')] != p[ip, 17])*100:.3f} %")
if "cycles" in cols:
    us = e[:, cols.index("cycles")] / a.mhz
    print(f"step time: mean {us.mean():.0f} us, p99 {np.percentile(us, 99):.0f} us, max {us.max():.0f} us "
          f"(budget 5000 us at 200 Hz -> p99 load {np.percentile(us, 99) / 50:.1f} %)")
if a.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2 if "cycles" in cols else 1, figsize=(12, 4), squeeze=False)
    ax[0, 0].plot((common - common[0]) * 1e-6, d * 1e3); ax[0, 0].set_xlabel("Time [s]")
    ax[0, 0].set_ylabel("|p_embedded − p_python| [mm]"); ax[0, 0].set_title("Embedded vs float64 reference")
    if "cycles" in cols:
        ax[0, 1].hist(us, bins=60); ax[0, 1].axvline(5000, color="r", label="5 ms budget")
        ax[0, 1].set_xlabel("Step time [µs]"); ax[0, 1].set_ylabel("Samples"); ax[0, 1].legend()
    fig.tight_layout(); fig.savefig(a.plot, dpi=110)

"""Experiment X2: roll/pitch accuracy of gyro integration, Mahony and Madgwick against ground truth.

  python scripts/compare_attitude.py data/euroc_V101 [--calib calib.yaml] [--plot results/x2.png]
Works on any dataset with groundtruth.csv (EuRoC Vicon rooms, synthetic).
"""
import argparse, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, attitude
from nav.quat import q2euler, euler2q
from nav.calib import CalibParams

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("--calib"); ap.add_argument("--plot")
a = ap.parse_args()
ds = io.load(a.dataset); imu = ds.imu
if a.calib:
    cp = CalibParams.from_yaml(a.calib); imu.gyro, imu.acc = cp.apply_gyro(imu.gyro), cp.apply_acc(imu.acc)
gt = ds.gt
t_gt = gt[:, 0]
k0 = int(np.searchsorted(imu.t_us, t_gt[0]))                 # start where truth starts
imu_t = imu.t[k0:]; g = imu.gyro[k0:]; f = imu.acc[k0:]; tus = imu.t_us[k0:]
e0 = q2euler(gt[0, 4:8]); q0 = euler2q(e0[0], e0[1], e0[2])  # initialise from truth: compare drift, not alignment
res = {"gyro only": attitude.gyro_only(q0, g, imu_t), "Mahony": attitude.mahony(q0, g, f, imu_t),
       "Madgwick": attitude.madgwick(q0, g, f, imu_t)}
idx = np.searchsorted(t_gt, tus).clip(0, len(gt) - 1)
truth = np.degrees(np.array([q2euler(q)[:2] for q in gt[idx, 4:8]]))
out = {}
for name, Q in res.items():
    est = np.degrees(np.array([q2euler(q)[:2] for q in Q]))
    err = (est - truth + 180) % 360 - 180
    out[name] = err
    print(f"{name:10s} roll RMS {np.sqrt(np.mean(err[:,0]**2)):6.2f} deg  pitch RMS {np.sqrt(np.mean(err[:,1]**2)):6.2f} deg  "
          f"max {np.abs(err).max():6.2f} deg")
if a.plot:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for name, err in out.items():
        ax[0].plot(imu_t - imu_t[0], err[:, 0], lw=0.8, label=name); ax[1].plot(imu_t - imu_t[0], err[:, 1], lw=0.8)
    ax[0].set_ylabel("Roll error [°]"); ax[1].set_ylabel("Pitch error [°]"); ax[1].set_xlabel("Time [s]"); ax[0].legend()
    fig.tight_layout(); fig.savefig(a.plot, dpi=110)

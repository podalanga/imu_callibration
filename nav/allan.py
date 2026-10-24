"""Overlapping Allan deviation and noise-parameter extraction (Chapter 08)."""
import numpy as np


def adev(x, fs, n_taus=60):
    """x: (N,) rate samples. Returns taus [s], adev, approx 1-sigma error bars."""
    x = np.asarray(x, float)
    N = len(x)
    theta = np.r_[0.0, np.cumsum(x)] / fs
    ms = np.unique(np.logspace(0, np.log10(max(1, (N - 1) // 2)), n_taus).astype(int))
    taus, devs, errs = [], [], []
    for m in ms:
        tau = m / fs
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        if len(d) < 2:
            continue
        sig2 = np.sum(d ** 2) / (2 * tau ** 2 * len(d))
        taus.append(tau); devs.append(np.sqrt(sig2))
        errs.append(np.sqrt(sig2) / np.sqrt(2 * max(1, N / m - 1)))
    return np.array(taus), np.array(devs), np.array(errs)


def noise_params(taus, devs):
    """N from a slope -1/2 fit at short tau, B from the minimum, K from a +1/2 fit if present."""
    lt, ld = np.log10(taus), np.log10(devs)
    slope = np.gradient(ld, lt)
    out = {}
    i = np.where(np.abs(slope + 0.5) < 0.15)[0]
    if len(i):
        c = np.median(ld[i] + 0.5 * lt[i])          # log sigma = c - 0.5 log tau
        out["N"] = 10 ** c                          # sigma at tau = 1 s
    imin = int(np.argmin(devs))
    out["B"] = devs[imin] / 0.664
    out["tau_B"] = taus[imin]
    j = np.where((np.abs(slope - 0.5) < 0.15) & (np.arange(len(taus)) > imin))[0]
    if len(j):
        c = np.median(ld[j] - 0.5 * lt[j])          # log sigma = c + 0.5 log tau
        out["K"] = 10 ** (c + 0.5 * np.log10(3.0))  # sigma at tau = 3 s on that line
    return out


def analyse(imu):
    """Allan parameters per axis for gyro (rad/s) and accel (m/s^2)."""
    fs = 1.0 / np.median(np.diff(imu.t))
    res = {}
    for name, data in (("gyro", imu.gyro), ("acc", imu.acc)):
        for i, ax in enumerate("xyz"):
            taus, d, e = adev(data[:, i], fs)
            res[f"{name}_{ax}"] = dict(taus=taus, adev=d, err=e, **noise_params(taus, d))
    return res

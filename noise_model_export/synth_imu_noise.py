"""Add the measured CPH2661 phone IMU noise to ideal (true) gyro/accel signals.

Standalone: needs numpy and pyyaml (scipy only speeds up the Gauss-Markov filter, matplotlib only for --validate).

    from synth_imu_noise import load_model, add_noise
    m = load_model("phone_imu_noise_model.yaml")
    gyro_meas, acc_meas, t_meas = add_noise(t, gyro_true, acc_true, m, seed=1)

    python synth_imu_noise.py --validate        # simulate 1 h static, compare Allan curves with the real phone

Per axis: reading = truth + b_on + b_gm(t) + b_rw(t) + w(t) (+ temp_slope*dT), then rounded to one LSB.
Units: gyro rad/s, accel m/s^2, t in seconds. Arrays are (n, 3).
"""
import argparse
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))


def load_model(path=os.path.join(HERE, "phone_imu_noise_model.yaml")):
    with open(path) as f:
        m = yaml.safe_load(f)
    for s in ("gyro", "acc"):
        for k, v in m[s].items():
            m[s][k] = np.asarray(v, float)
    return m


def gauss_markov(rng, n, dt, sigma, tau):
    """(n,) first-order Gauss-Markov process, stationary std sigma, correlation time tau, started stationary."""
    if sigma <= 0 or tau <= 0:
        return np.zeros(n)
    phi = np.exp(-dt / tau)
    w = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
    w[0] = rng.normal(0, sigma)
    try:
        from scipy.signal import lfilter
        return lfilter([1.0], [1.0, -phi], w)
    except ImportError:
        out = np.empty(n)
        acc = 0.0
        for k in range(n):
            acc = phi * acc + w[k]
            out[k] = acc
        return out


def sensor_errors(rng, n, dt, p, turn_on="measured", dtemp=None, quantise=True):
    """(n,3) additive error for one sensor (p = model["gyro"] or model["acc"])."""
    lsb = float(p["lsb"]) if quantise else 0.0
    # the measured N already contains rounding noise (lsb^2/12 per sample); take it out so it is not counted twice
    white_std = np.sqrt(np.maximum(p["N"] ** 2 / dt - lsb ** 2 / 12, 0))
    if turn_on == "measured":
        b_on = p["turn_on_bias"]
    elif turn_on == "zero":
        b_on = np.zeros(3)
    else:                                  # a number: draw a fresh turn-on bias with this std per axis
        b_on = rng.normal(0, float(turn_on), 3)
    e = np.empty((n, 3))
    for i in range(3):
        e[:, i] = (b_on[i] + rng.normal(0, white_std[i], n)
                   + gauss_markov(rng, n, dt, p["sigma_gm"][i], p["tau_gm"][i])
                   + np.cumsum(rng.normal(0, p["K"][i] * np.sqrt(dt), n)))
    if dtemp is not None:
        e += np.outer(dtemp, p["temp_slope"])
    return e


def add_noise(t, gyro_true, acc_true, model, seed=0, gyro_turn_on="measured", acc_turn_on="zero",
              dtemp=None, quantise=True, jitter=True):
    """Noisy readings of ideal signals sampled at times t (ideally uniform at model['fs_hz']).

    gyro_turn_on / acc_turn_on: "measured" (this phone's power-up), "zero", or a std to draw a new one.
    dtemp: optional (n,) temperature change from the start in degC (adds temp_slope * dtemp).
    Returns gyro, acc, and the timestamps with the phone's ~3 us jitter added.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, float)
    n, dt = len(t), float(np.median(np.diff(t)))
    gyro = np.asarray(gyro_true, float) + sensor_errors(rng, n, dt, model["gyro"], gyro_turn_on, dtemp, quantise)
    acc = np.asarray(acc_true, float) + sensor_errors(rng, n, dt, model["acc"], acc_turn_on, dtemp, quantise)
    if quantise:
        gyro = np.round(gyro / model["gyro"]["lsb"]) * model["gyro"]["lsb"]
        acc = np.round(acc / model["acc"]["lsb"]) * model["acc"]["lsb"]
    t_out = t + (rng.normal(0, model["timestamp_jitter_us"] * 1e-6, n) if jitter else 0.0)
    return gyro, acc, t_out


def adev(x, fs, taus):
    """Overlapping Allan deviation of rate samples x at the given averaging times."""
    theta = np.r_[0.0, np.cumsum(x)] / fs
    out = []
    for tau in taus:
        m = max(1, int(round(tau * fs)))
        d = theta[2 * m:] - 2 * theta[m:-m] + theta[:-2 * m]
        out.append(np.sqrt(np.sum(d ** 2) / (2 * (m / fs) ** 2 * len(d))) if len(d) > 1 else np.nan)
    return np.array(out)


def validate(model, out_png, seed=0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meas = np.genfromtxt(os.path.join(HERE, "measured_allan.csv"), delimiter=",", names=True)
    fs = model["fs_hz"]
    t = np.arange(int(model["record_s"] * fs)) / fs
    g_true = np.zeros((len(t), 3))
    a_true = np.tile([0.0, 0.0, 9.78], (len(t), 1))
    g, a, _ = add_noise(t, g_true, a_true, model, seed=seed)
    taus = meas["tau_s"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (name, X, cols, unit) in zip(axs, (("Gyroscope", g, ("gx", "gy", "gz"), "rad/s"),
                                               ("Accelerometer", a, ("ax", "ay", "az"), "m/s$^2$"))):
        for i, (c, col) in enumerate(zip(cols, ("C0", "C1", "C2"))):
            sim = adev(X[:, i], fs, taus)
            ax.loglog(taus, meas[c], col, lw=2, label=f"{c} measured")
            ax.loglog(taus, sim, col, ls="--", lw=1.2, label=f"{c} synthetic")
            ok = taus <= model["record_s"] / 6
            err = np.median(np.abs(sim[ok] / meas[c][ok] - 1)) * 100
            print(f"{c}: median |synthetic/measured - 1| = {err:.1f}%  (tau <= {model['record_s'] / 6:.0f} s)")
        ax.set(title=name, xlabel="averaging time tau [s]", ylabel=f"Allan deviation [{unit}]")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    fig.suptitle("Real phone (1 h static) vs synth_imu_noise.py (1 h static, same model)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print("wrote", out_png)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=os.path.join(HERE, "phone_imu_noise_model.yaml"))
    ap.add_argument("--validate", action="store_true", help="simulate 1 h static and compare Allan curves")
    ap.add_argument("--out", default=os.path.join(HERE, "allan_validation.png"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.validate:
        validate(load_model(a.model), a.out, a.seed)
    else:
        ap.print_help()

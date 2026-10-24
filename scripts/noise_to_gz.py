"""Write the phone's measured noise model into the drone sim's /imu/noisy settings (gz-sensors gaussian noise).

  python scripts/noise_to_gz.py runs/One_hr_drift_test/phone_noise.yaml
  python scripts/noise_to_gz.py phone_noise.yaml --imu-yaml simulation/ros2_ws/src/drone_sim/config/imu.yaml --dry-run

gz-sensors (GaussianNoiseModel.cc) per axis and per sample, at the sim's rate_hz:
  output = truth + bias + N(0, stddev)
  bias   = turn-on N(bias_mean, bias_stddev) with a random sign, then each step
           bias <- exp(-dt/tau) bias + N(0, sigma_b * sqrt(tau/2 * (1 - exp(-2 dt/tau))))
so the mapping from the noise model is
  stddev                        = N * sqrt(rate_hz)              (N is a density, the sim rate differs from the phone's)
  dynamic_bias_correlation_time = tau_c                          (Gauss-Markov time constant, error_model tau_*)
  dynamic_bias_stddev           = sigma_GM * sqrt(2 / tau_c)     (gz's stationary std is sigma_b * sqrt(tau/2))
  mean (gyro)                   = RMS of the measured gyro means  (the phone's turn-on bias size, as a constant offset)
  bias_mean = bias_stddev       = 0
gz starts the correlated bias at the turn-on sample and lets it decay to zero within tau_c, so a turn-on bias put in
bias_stddev would fade away in about a minute (0.24 deg/s here, far more than the phone's real wander). `mean` is added
to every sample and never decays, so the offset goes there instead; gz uses the same value, and sign, on x, y and z.
gz applies one set to x, y and z, so each value is the worst axis. The accelerometer offset stays 0: the static run's
accel means include the table's tilt, so they only become a bias after calibration (calib.yaml).
"""
import argparse, os
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMU_YAML = os.path.join(ROOT, "simulation", "ros2_ws", "src", "drone_sim", "config", "imu.yaml")


def gz_params(noise, rate_hz):
    """noise: phone_noise.yaml contents -> {"gyro": {...}, "accel": {...}} in gz <noise> terms."""
    em = noise["error_model"]
    out = {}
    for name, N, B, tau, turn_on in (("gyro", em["Ng"], em["B_gyro"], em["tau_gyro"],
                                      float(np.sqrt(np.mean(np.square(em["bg"]))))),
                                     ("accel", em["Na"], em["B_acc"], em["tau_acc"], 0.0)):
        i = int(np.argmax(B))                                    # worst axis keeps sigma and tau together
        out[name] = dict(mean=float(turn_on), stddev=float(np.max(N)) * np.sqrt(rate_hz), bias_mean=0.0, bias_stddev=0.0,
                         dynamic_bias_stddev=float(B[i]) * np.sqrt(2.0 / tau[i]), dynamic_bias_correlation_time=float(tau[i]))
    return out


def gz_sample(p, n, rate_hz, rng):
    """n samples of one gz gaussian-noise axis on a zero signal (the same recursion gz-sensors runs)."""
    dt, tau, sb = 1.0 / rate_hz, p["dynamic_bias_correlation_time"], p["dynamic_bias_stddev"]
    phi, sd = np.exp(-dt / tau), sb * np.sqrt(tau / 2 * -np.expm1(-2 * dt / tau))
    b0 = rng.normal(p["bias_mean"], p["bias_stddev"]) * rng.choice([-1, 1]) if p["bias_stddev"] > 0 else p["bias_mean"]
    bias = np.empty(n)
    b = b0
    kick = rng.normal(0.0, sd, n)
    for k in range(n):
        b = phi * b + kick[k]
        bias[k] = b
    return bias + rng.normal(p["mean"], p["stddev"], n)


def _block(name, unit, p, src):
    lines = [f"  {name}:".ljust(33) + f"# {unit}; {src}"]
    lines += [f"    {k}: {v:.6g}" for k, v in p.items()]
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("noise", help="phone_noise.yaml from characterise_noise.py or the console analysis")
    ap.add_argument("--imu-yaml", default=IMU_YAML)
    ap.add_argument("--dry-run", action="store_true", help="print the new file instead of writing it")
    a = ap.parse_args()
    with open(a.noise) as f:
        noise = yaml.safe_load(f)
    with open(a.imu_yaml) as f:
        text = f.read()
    cfg = yaml.safe_load(text)
    rate = float(cfg["rate_hz"])
    p = gz_params(noise, rate)
    head = text[:text.index("\nnoisy:")].splitlines()
    head = [l for l in head if not l.startswith("# noisy:") and not l.startswith("# from ")]
    src = os.path.relpath(os.path.abspath(a.noise), ROOT).replace(os.sep, "/")
    new = head + [f"# noisy: gz-sensors gaussian noise, same parameters applied to x, y, z of each channel.",
                  f"# from {src} by scripts/noise_to_gz.py: stddev = N*sqrt(rate_hz), dynamic bias = the measured",
                  "# Gauss-Markov bias instability (worst axis), gyro mean = measured turn-on bias size (constant: gz",
                  "# lets a bias_stddev offset decay to zero within the correlation time). accel offset: not measured yet.",
                  "noisy:"] + _block("gyro", "rad/s", p["gyro"], "phone") + _block("accel", "m/s^2", p["accel"], "phone") + [""]
    if a.dry_run:
        print("\n".join(new))
        return
    with open(a.imu_yaml, "w", newline="\n") as f:
        f.write("\n".join(new))
    print(f"wrote {a.imu_yaml}; rebuild the sim workspace (./run.sh build) so world_gen picks it up")


if __name__ == "__main__":
    main()

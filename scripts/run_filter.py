"""Run the causal drift pipeline on a static session (phone at rest) and report drift growth.

  python scripts/run_filter.py runs/phone_static_10min --calib calib.yaml --noise noise.yaml \
         --ablation --plots results/drift
  python scripts/run_filter.py runs/phone_static_10min --config '{"zupt_every": 1}' --est runs/phone_static_10min/est.csv

Assumes the IMU sat still for the whole run: true position/velocity are zero and attitude is
constant throughout, so the filter's own estimate moving away from that IS the drift error,
no ground truth needed.
"""
import argparse, os, sys, json
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, runner
from nav.calib import CalibParams
from nav.quat import qconj, qlog, qmul

ABLATION = [("free inertial", dict(use_zupt=False)), ("ZUPT + ZARU", dict(use_zupt=True))]


def drift_metrics(est, q0):
    """est rows: t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,stationary."""
    t_s = (est[:, 0] - est[0, 0]) * 1e-6
    p_norm = np.linalg.norm(est[:, 1:4], axis=1)
    v_norm = np.linalg.norm(est[:, 4:7], axis=1)
    tilt_deg = np.degrees([2 * np.linalg.norm(qlog(qmul(qconj(q0), q))) for q in est[:, 7:11]])
    return dict(t_s=t_s, p_norm=p_norm, v_norm=v_norm, tilt_deg=tilt_deg,
                final_p_m=float(p_norm[-1]), max_p_m=float(p_norm.max()),
                final_v_mps=float(v_norm[-1]), final_tilt_deg=float(tilt_deg[-1]))


def summary_line(name, m):
    return (f"{name:16s} final drift {m['final_p_m']*100:7.2f} cm  (max {m['max_p_m']*100:7.2f} cm)  "
            f"final |v| {m['final_v_mps']*1000:7.2f} mm/s  final tilt {m['final_tilt_deg']:6.3f} deg")


def plot_drift(results, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for name, m in results.items():
        ax[0].semilogy(m["t_s"], np.maximum(m["p_norm"], 1e-6), lw=1, label=name)
        ax[1].semilogy(m["t_s"], np.maximum(m["v_norm"], 1e-6), lw=1, label=name)
        ax[2].plot(m["t_s"], m["tilt_deg"], lw=1, label=name)
    ax[0].set_xlabel("Time [s]"); ax[0].set_ylabel("|p - 0| [m]"); ax[0].set_title("Position drift"); ax[0].legend()
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("|v - 0| [m/s]"); ax[1].set_title("Velocity drift"); ax[1].legend()
    ax[2].set_xlabel("Time [s]"); ax[2].set_ylabel("Tilt error [deg]"); ax[2].set_title("Attitude drift"); ax[2].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def run_ablation(imu, calib=None, base=None, runs=ABLATION, est_path=None):
    """Run each configuration over imu; returns {name: metrics + "rejected" + "nis"}."""
    results = {}
    for name, kw in runs:
        cfg = runner.NavConfig(**{**(base or {}), **kw})
        r = runner.run(imu, cfg, calib=calib)
        m = drift_metrics(r["est"], r["est"][0, 7:11])      # reference = attitude right after alignment
        m["rejected"] = r["rejected"]
        m["nis"] = {k: float(v.mean()) for k, v in r["nis"].items() if len(v)}
        results[name] = m
        print(summary_line(name, m), "| gated:", r["rejected"], "| mean NIS:", {k: round(v, 2) for k, v in m["nis"].items()})
        if est_path:
            io.save_est(est_path, r["est"])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--calib", default=None)
    ap.add_argument("--noise", default=None, help="noise.yaml from run_allan.py")
    ap.add_argument("--config", default="{}", help="JSON overrides of NavConfig")
    ap.add_argument("--ablation", action="store_true")
    ap.add_argument("--est", default=None)
    ap.add_argument("--plots", default=None)
    a = ap.parse_args()

    ds = io.load(a.dataset)
    cp = CalibParams.from_yaml(a.calib) if a.calib else None
    base = json.loads(a.config)
    if a.noise:
        with open(a.noise) as f:
            n = yaml.safe_load(f)
        base.update({k: n[k] for k in ("Na", "Ng", "Kba", "Kbg") if k in n})   # Kba/Kbg only when K was resolved
    results = run_ablation(ds.imu, cp, base, ABLATION if a.ablation else [("configured", {})],
                           None if a.ablation else a.est)
    if a.plots and results:
        os.makedirs(a.plots, exist_ok=True)
        plot_drift(results, os.path.join(a.plots, "drift.png"))
        with open(os.path.join(a.plots, "metrics.json"), "w") as f:
            json.dump({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in results.items()},
                      f, indent=1, default=float)
        print("plots in", a.plots)


if __name__ == "__main__":
    main()

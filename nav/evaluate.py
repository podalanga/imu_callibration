"""Scoring (Chapter 05): checkpoint errors against marker presses. No alignment is fitted."""
import numpy as np


def checkpoint_errors(est, markers, motion_start_us=None):
    """est: rows t_us,px,py,pz,...  markers: rows t_us,id,s_m (true position = [s,0,0]).

    Returns dict with per-checkpoint arrays and summary metrics.
    """
    t_est = est[:, 0]
    tk = markers[:, 0]
    keep = (tk >= t_est[0]) & (tk <= t_est[-1])
    tk, s = tk[keep], markers[keep, 2]
    p = np.column_stack([np.interp(tk, t_est, est[:, i]) for i in (1, 2, 3)])
    truth = np.column_stack([s, np.zeros_like(s), np.zeros_like(s)])
    e_vec = p - truth
    e = np.linalg.norm(e_vec[:, :2], axis=1)                 # horizontal error, 2D task
    d = np.r_[0, np.cumsum(np.abs(np.diff(s)))]               # true distance travelled
    t0 = motion_start_us if motion_start_us is not None else tk[0]
    in30 = (tk - t0) <= 30e6
    out = dict(t_s=(tk - tk[0]) * 1e-6, s=s, dist=d, err=e, along=e_vec[:, 0], cross=e_vec[:, 1],
               max_err=float(e.max()), rms_err=float(np.sqrt(np.mean(e**2))),
               return_to_start=float(e[-1]), return_along=float(e_vec[-1, 0]), return_cross=float(e_vec[-1, 1]),
               max_err_first_30s=float(e[in30].max()) if in30.any() else float("nan"),
               max_rel_err=float(np.max(e[d > 0] / d[d > 0])) if (d > 0).any() else float("nan"),
               total_distance=float(d[-1]))
    out["tier1"] = out["max_err_first_30s"] < 1.0
    out["tier2"] = out["max_err"] <= 0.03 * out["total_distance"]          # 3% of the 50 m path
    out["tier2_strict"] = bool(np.all(e[d > 0] <= 0.03 * d[d > 0]))     # 3% at every checkpoint
    return out


def summary_line(name, m):
    return (f"{name:28s} max {m['max_err']:6.2f} m | RMS {m['rms_err']:6.2f} m | "
            f"return {m['return_to_start']:6.2f} m ({100*m['return_to_start']/m['total_distance']:4.1f}%) | "
            f"T1 {'pass' if m['tier1'] else 'fail'} | T2 {'pass' if m['tier2'] else 'fail'}")


def plot_run(m, est, markers, title, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(est[:, 1], est[:, 2], lw=1, label="estimate")
    ax[0].plot(markers[:, 2], 0 * markers[:, 2], "k|", ms=12, label="markers (truth)")
    ax[0].set_xlabel("x [m]"); ax[0].set_ylabel("y [m]"); ax[0].set_title("Trajectory"); ax[0].legend()
    ax[1].semilogy(m["t_s"], np.maximum(m["err"], 1e-3), "o-")
    ax[1].axhline(1.0, color="g", ls="--", label="Tier 1: 1 m"); ax[1].axvline(30, color="g", ls=":")
    ax[1].set_xlabel("Time since first checkpoint [s]"); ax[1].set_ylabel("Error [m]"); ax[1].legend()
    ax[1].set_title("Error vs time")
    ax[2].plot(m["dist"], m["err"], "o-", label="error")
    ax[2].plot(m["dist"], 0.03 * m["dist"], "g--", label="3% of distance")
    ax[2].set_xlabel("Distance travelled [m]"); ax[2].set_ylabel("Error [m]"); ax[2].legend()
    ax[2].set_title("Error vs distance")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)

"""Automatic analysis of one recorded test, run as its own process so it never disturbs a recording.

  python phone_data/analysis.py runs/static_2026-10-24_1630 --kind static [--extra runs/turnon_1 ...] [--calib calib.yaml]

Stages (progress in <run>/progress.json, results in <run>/results.json, report in <run>/report.html):
  1. merge raw streams -> imu.csv (if not done yet)
  2. data quality: rate, lost samples, timing jitter, disturbances (phone touched / moved), temperature
  3. noise model on the longest undisturbed stretch (nav/noise_model.py), plus repeatability from --extra
  4. measured free drift on undisturbed segments vs the prediction from the noise model
  5. long-run free inertial drift: position, velocity and tilt from the ESKF with no aiding (static tests; at most
     LONG_MAX_S of data). ZUPT/ZARU is not part of the report; scripts/run_filter.py --ablation still runs it by hand.
  6. report.html

kind: "static" (full), "quick" (no long-run drift), "turnon" (quality + bias only; combine several with --extra).
"""
import argparse, json, os, sys, time, traceback
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from nav import io, noise_model as nm
from nav.calib import CalibParams
from phone_data import merge, report

G = 9.80665
LONG_MAX_S = 20 * 60          # the Python ESKF runs ~5 s per minute of data
T_ALIGN = 10.0


# ---------------------------------------------------------------- plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_allan(res, curves, path):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    for k, (s, unit, sc) in enumerate((("gyro", "°/s", np.degrees(1)), ("acc", "m/s²", 1.0))):
        for ax in "xyz":
            taus, dev, err = curves[s][ax]
            l, = axs[k].loglog(taus, dev * sc, lw=1.2, label=ax)
            r = res[s][ax]
            axs[k].loglog(taus, r["N"] / np.sqrt(taus) * sc, ":", color=l.get_color(), lw=0.8)
            axs[k].plot(r["tau_B"], r["B"] * 0.664 * sc, "o", color=l.get_color())
            if np.isfinite(r["K"]):
                axs[k].loglog(taus, r["K"] * np.sqrt(taus / 3) * sc, "--", color=l.get_color(), lw=0.8)
        axs[k].grid(True, which="both", alpha=0.3); axs[k].legend(title="axis")
        axs[k].set_xlabel("Averaging time τ [s]"); axs[k].set_ylabel(f"Allan deviation [{unit}]")
        axs[k].set_title({"gyro": "Gyroscope", "acc": "Accelerometer"}[s])
    fig.suptitle("Allan deviation. Dotted: white noise N. Dot: bias instability B. Dashed: rate random walk K.")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_bias_vs_time(imu, path):
    plt = _plt()
    tm, gm = nm.window_means(imu.t, imu.gyro, 60.0)
    _, am = nm.window_means(imu.t, imu.acc, 60.0)
    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(tm / 60, np.degrees(gm - gm[0]) * 1e3, label=list("xyz")); axs[0].set_ylabel("Gyro bias change [m°/s]")
    axs[1].plot(tm / 60, (am - am[0]) / G * 1e3, label=list("xyz")); axs[1].set_ylabel("Accel change [mg]")
    axs[1].set_xlabel("Time [min]")
    for ax in axs:
        ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    if imu.temp is not None and not np.all(np.isnan(imu.temp)):
        _, Tm = nm.window_means(imu.t, imu.temp[:, None], 60.0)
        for ax in axs:
            tw = ax.twinx(); tw.plot(tm / 60, Tm[:, 0], "k--", lw=1); tw.set_ylabel("Battery temp [°C]")
    axs[0].set_title("60 s mean relative to the first minute (dashed: battery temperature)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_dt_hist(imu, path):
    plt = _plt()
    dt = np.diff(imu.t_us) * 1e-3
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dt, bins=100, log=True); ax.set_xlabel("Sample interval [ms] (hardware timestamps)"); ax.set_ylabel("Count")
    ax.set_title(f"median {np.median(dt):.3f} ms, std {dt.std() * 1e3:.1f} µs, max {dt.max():.2f} ms")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_disturbances(imu, dist, path):
    plt = _plt()
    tm, gm = nm.window_means(imu.t, imu.gyro, 1.0)
    _, g2 = nm.window_means(imu.t, imu.gyro ** 2, 1.0)
    spread = np.degrees(np.linalg.norm(np.sqrt(np.maximum(g2 - gm ** 2, 0)), axis=1))
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(tm / 60, spread, lw=0.8, color="0.3")
    for s, e, _, _ in dist["events"]:
        ax.axvspan(s / 60, e / 60, color="#d1495b", alpha=0.3, lw=0)
    ax.set_yscale("log"); ax.set_xlabel("Time [min]"); ax.set_ylabel("Gyro spread per 1 s [°/s]")
    ax.set_title(f"Stillness: {dist['validity']} ({100 * dist['fraction']:.2f} % of the time disturbed, red)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def plot_free_drift(tt, head, pos, p, path):
    plt = _plt()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    axs[0].plot(tt, np.degrees(head.T), color="0.75", lw=0.6)
    axs[0].plot(tt, np.degrees(np.sqrt(np.mean(head ** 2, axis=0))), "b", lw=2, label="measured RMS")
    axs[0].plot(tt, np.degrees(p["heading"]), "r--", lw=2, label="predicted 1σ")
    axs[0].set_ylabel("Heading error [°]"); axs[0].set_title(f"Gyro z free drift, {len(head)} segments")
    axs[1].loglog(tt[1:], pos.T[1:], color="0.75", lw=0.6)
    axs[1].loglog(tt[1:], np.sqrt(np.mean(pos ** 2, axis=0))[1:], "b", lw=2, label="measured RMS")
    for key, style, lab in (("position", "r--", "predicted 1σ (total)"), ("pos_acc_white", "g:", "accel white noise"),
                            ("pos_acc_bias", "m:", "accel bias"), ("pos_tilt", "c:", "tilt from gyro bias")):
        axs[1].loglog(tt[1:], np.sqrt(2) * p[key][1:] if key != "position" else p[key][1:], style,
                      lw=1.5 if key == "position" else 1, label=lab)
    axs[1].axhline(1.0, color="k", lw=0.8); axs[1].axvline(30, color="k", lw=0.8, ls=":")
    axs[1].set_ylabel("Horizontal position error [m]"); axs[1].set_title("Free inertial position drift (Tier 1: 1 m at 30 s)")
    for ax in axs:
        ax.set_xlabel("Time since alignment [s]"); ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------- helpers
def sub_imu(imu, i0, i1):
    sl = slice(i0, i1)
    return io.Imu(t_us=imu.t_us[sl], gyro=imu.gyro[sl], acc=imu.acc[sl],
                  temp=None if imu.temp is None else imu.temp[sl], seq=None if imu.seq is None else imu.seq[sl])


def verdict(level, label, value, note=""):
    return dict(level=level, label=label, value=value, note=note)


def quality_checks(acq, dist, imu, kind):
    out = []
    rate = min(acq["acc"]["rate_hz"], acq["gyro"]["rate_hz"])
    out.append(verdict("pass" if rate >= 390 else ("warn" if rate >= 190 else "fail"), "Sample rate", f"{rate:.1f} Hz",
                       "" if rate >= 390 else "Set the Sensor Server sampling rate to its fastest value."))
    lost = acq["acc"]["missing_samples"] + acq["gyro"]["missing_samples"]
    n = acq["acc"]["n"] + acq["gyro"]["n"]
    out.append(verdict("pass" if lost == 0 else ("warn" if lost < 1e-3 * n else "fail"), "Lost samples",
                       f"{lost} ({100 * lost / max(n, 1):.3f} %)",
                       "" if lost == 0 else "Keep the app in the foreground, battery Unrestricted, USB instead of Wi-Fi."))
    jit = max(acq["acc"]["dt_std_us"], acq["gyro"]["dt_std_us"])
    out.append(verdict("pass" if jit < 50 else ("warn" if jit < 500 else "fail"), "Timing jitter", f"{jit:.1f} µs std"))
    lvl = {"valid": "pass", "partly valid": "warn", "invalid": "fail"}[dist["validity"]]
    out.append(verdict(lvl, "Stillness", f"{dist['validity']}, {100 * dist['fraction']:.2f} % disturbed",
                       "" if lvl == "pass" else f"{len(dist['events'])} disturbance(s); the analysis uses undisturbed data only."))
    mins = imu.t[-1] / 60
    need = {"static": (60, 10), "quick": (1, 0.5), "turnon": (3, 1)}[kind]
    out.append(verdict("pass" if mins >= need[0] else ("warn" if mins >= need[1] else "fail"), "Duration", f"{mins:.1f} min",
                       "" if mins >= need[0] else f"Bias instability and random walk need at least {need[0]} min."))
    if imu.temp is not None and not np.all(np.isnan(imu.temp)):
        out.append(verdict("info", "Battery temperature", f"{np.nanmin(imu.temp):.1f} to {np.nanmax(imu.temp):.1f} °C"))
    return out


class Progress:
    def __init__(self, run_dir, stages):
        self.path, self.stages, self.t0 = os.path.join(run_dir, "progress.json"), stages, time.time()

    def set(self, i, msg=None, done=False, error=None):
        d = dict(stage=i, n=len(self.stages), name=self.stages[min(i, len(self.stages) - 1)], msg=msg or "",
                 done=done, error=error, elapsed_s=time.time() - self.t0)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(d, f)
        os.replace(tmp, self.path)
        print(f"[{i + 1}/{len(self.stages)}] {d['name']} {msg or ''}", flush=True)


# ---------------------------------------------------------------- pipeline
def analyse(run_dir, kind="static", extras=(), calib_path=None, out_dir=None):
    """out_dir: where progress, plots, results and the report go (default: run_dir)."""
    out = out_dir or run_dir
    os.makedirs(out, exist_ok=True)
    stages = ["Merging raw streams", "Checking data quality", "Characterising noise", "Measuring drift",
              "Long-run free inertial drift", "Writing the report"]
    pr = Progress(out, stages)
    plots = os.path.join(out, "plots")
    os.makedirs(plots, exist_ok=True)
    meta = {}
    if os.path.exists(os.path.join(run_dir, "meta.yaml")):
        with open(os.path.join(run_dir, "meta.yaml")) as f:
            meta = yaml.safe_load(f) or {}
    R = dict(run=os.path.basename(os.path.normpath(out)), source=os.path.basename(os.path.normpath(run_dir)), kind=kind, meta=meta, extras=list(extras),
             calibrated=bool(calib_path), calib=calib_path, plots={}, notes=[])

    pr.set(0)
    if os.path.exists(os.path.join(run_dir, "raw_gyro.csv")) and not os.path.exists(os.path.join(run_dir, "imu.csv")):
        merge.build(run_dir)
    ds = io.load(run_dir)
    imu = ds.imu
    acq = ds.config.get("acquisition") or merge.timing_from_imu(imu)
    cp = CalibParams.from_yaml(calib_path) if calib_path else None
    if cp:
        imu.gyro, imu.acc = cp.apply_gyro(imu.gyro), cp.apply_acc(imu.acc)

    pr.set(1)
    dist = nm.disturbances(imu)
    R["acquisition"] = acq
    R["disturbances"] = dict(validity=dist["validity"], fraction=dist["fraction"], events=dist["events"])
    R["checks"] = quality_checks(acq, dist, imu, kind)
    R["duration_s"] = float(imu.t[-1])
    R["jitter_heading_error_deg"] = np.degrees(nm.jitter_effect(imu)).tolist()
    plot_dt_hist(imu, os.path.join(plots, "dt_hist.png")); R["plots"]["dt_hist"] = "plots/dt_hist.png"
    plot_disturbances(imu, dist, os.path.join(plots, "stillness.png")); R["plots"]["stillness"] = "plots/stillness.png"
    i0, i1 = nm.longest_clean(imu.t, dist["mask"])
    clean = sub_imu(imu, i0, i1)
    R["clean_stretch_s"] = [float(imu.t[i0]), float(imu.t[i1 - 1])] if i1 > i0 else None
    clean_s = clean.t[-1] if i1 - i0 > 1 else 0.0

    pr.set(2)
    if clean_s >= 30:
        extra_imus = []
        for e in extras:
            x = io.load(e).imu
            if cp:
                x.gyro, x.acc = cp.apply_gyro(x.gyro), cp.apply_acc(x.acc)
            extra_imus.append(x)
        res, curves = nm.characterise([clean] + extra_imus)
        res["timing"] = nm.timing_stats(clean.t)
        R["noise"] = dict(res, filter=nm.filter_params(res), error_model=nm.error_model_params(res))
        R["extra_means"] = {os.path.basename(os.path.normpath(e)): dict(gyro=x.gyro.mean(axis=0).tolist(), acc=x.acc.mean(axis=0).tolist())
                            for e, x in zip(extras, extra_imus)}
        with open(os.path.join(out, "phone_noise.yaml"), "w") as f:
            yaml.safe_dump(io._plain(dict(R["noise"]["filter"], source=[run_dir] + list(extras), calibrated=bool(cp),
                                          gyro=res["gyro"], acc=res["acc"], timing=res["timing"],
                                          repeatability=res.get("repeatability"), error_model=R["noise"]["error_model"])),
                           f, sort_keys=False, default_flow_style=None)
        plot_allan(res, curves, os.path.join(plots, "allan.png")); R["plots"]["allan"] = "plots/allan.png"
        if clean_s >= 180:
            plot_bias_vs_time(clean, os.path.join(plots, "bias_vs_time.png")); R["plots"]["bias_vs_time"] = "plots/bias_vs_time.png"
    else:
        R["notes"].append("Less than 30 s of undisturbed data: no noise model.")

    pr.set(3)
    if "noise" in R:
        tc = nm.temp_compensate(imu, R["noise"])         # in place: clean is a view of imu, so it is compensated too
        R["temp_compensation"] = tc
        if tc:
            R["notes"].append(f"Drift uses temperature-compensated readings (reference {tc['T_ref']:.1f} °C).")
    if "noise" in R and kind != "turnon":
        undisturbed_s = imu.t[-1] * (1 - dist["fraction"])
        seg = 60.0 if undisturbed_s >= 6 * (60.0 + T_ALIGN) else 20.0          # 60 s segments once >= 6 of them fit
        tt, head, pos = nm.drift_segments(imu, T_ALIGN, seg, disturbed=dist["mask"])
        if tt is not None:
            p = nm.predicted_drift_axes(res, tt, t_align=T_ALIGN)
            checks = [c for c in (10.0, 20.0, 30.0, 60.0) if c <= tt[-1]]
            idx = [int(np.searchsorted(tt, c)) for c in checks]
            R["drift"] = dict(segment_s=seg, segments=int(len(head)), checkpoints_s=checks,
                              measured_heading_deg=[float(np.degrees(np.sqrt(np.mean(head[:, i] ** 2)))) for i in idx],
                              predicted_heading_deg=[float(np.degrees(p["heading"][i])) for i in idx],
                              measured_position_m=[float(np.sqrt(np.mean(pos[:, i] ** 2))) for i in idx],
                              predicted_position_m=[float(p["position"][i]) for i in idx],
                              terms_at_last_m=dict(accel_white=float(np.sqrt(2) * p["pos_acc_white"][idx[-1]]),
                                                   accel_bias=float(np.sqrt(2) * p["pos_acc_bias"][idx[-1]]),
                                                   tilt_from_gyro=float(np.sqrt(2) * p["pos_tilt"][idx[-1]])))
            plot_free_drift(tt, head, pos, p, os.path.join(plots, "free_drift.png")); R["plots"]["free_drift"] = "plots/free_drift.png"
        else:
            R["notes"].append(f"Not enough undisturbed data for a {seg + T_ALIGN:.0f} s drift segment.")

    pr.set(4)
    if kind == "static" and clean_s >= 60:
        import run_filter
        part = sub_imu(clean, 0, int(np.searchsorted(clean.t, LONG_MAX_S)))
        if clean_s > LONG_MAX_S:
            R["notes"].append(f"Long-run drift uses the first {LONG_MAX_S / 60:.0f} min of the undisturbed data.")
        base = {}
        if "noise" in R:
            fp = R["noise"]["filter"]
            base = dict(Na=fp["Na"], Ng=fp["Ng"])
        fr = run_filter.run_ablation(part, None, base, runs=run_filter.ABLATION[:1])     # free inertial only
        run_filter.plot_drift(fr, os.path.join(plots, "filter_drift.png")); R["plots"]["filter_drift"] = "plots/filter_drift.png"
        R["filter"] = {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, np.ndarray)} for k, v in fr.items()}
        R["filter_duration_s"] = float(part.t[-1])

    pr.set(5)
    R["conclusions"] = report.conclusions(R)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(io._plain(R), f, indent=1, default=float)
    with open(os.path.join(out, "report.html"), "w", encoding="utf-8") as f:
        f.write(report.render(R, out))
    pr.set(5, "done", done=True)
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--kind", choices=["static", "quick", "turnon"], default="static")
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--calib", default=None)
    ap.add_argument("--out", default=None, help="write outputs here instead of run_dir (combined reports)")
    a = ap.parse_args()
    try:
        analyse(a.run_dir, a.kind, a.extra, a.calib, a.out)
    except Exception as e:
        traceback.print_exc()
        Progress(a.out or a.run_dir, ["failed"]).set(0, done=True, error=f"{e.__class__.__name__}: {e}")
        sys.exit(1)

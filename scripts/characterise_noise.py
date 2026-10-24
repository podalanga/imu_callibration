"""Characterise each sensor's noise from static recordings; write a noise model for the filter and simulator.

  python scripts/characterise_noise.py runs/static_2h --extra runs/static_s1 runs/static_s2 \
         --out phone_noise.yaml --plots results/noise

The first dataset (longest, >= 1 h recommended) sets N, B, K, temperature slope and quantisation; the
--extra short sessions (each started after a phone/app restart) add turn-on bias repeatability.
Output keys Na/Ng at the top level are what scripts/run_filter.py --noise reads; `error_model` is what
scripts/make_synth.py --noise-model reads; `drift` compares predicted and measured free-inertial drift.
"""
import argparse, os, sys
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, noise_model as nm
from nav.calib import CalibParams

ap = argparse.ArgumentParser()
ap.add_argument("dataset")
ap.add_argument("--extra", nargs="*", default=[])
ap.add_argument("--calib", default=None, help="apply calib.yaml first (characterise the calibrated output)")
ap.add_argument("--segment", type=float, default=60.0, help="s per free-drift segment")
ap.add_argument("--out", default="phone_noise.yaml")
ap.add_argument("--plots", default=None)
a = ap.parse_args()

dsets = [io.load(d) for d in [a.dataset] + a.extra]
imus = [d.imu for d in dsets]
if a.calib:
    cp = CalibParams.from_yaml(a.calib)
    for imu in imus:
        imu.gyro, imu.acc = cp.apply_gyro(imu.gyro), cp.apply_acc(imu.acc)
res, curves = nm.characterise(imus)
main = max(imus, key=lambda i: i.t[-1])
G = 9.80665

print(f"{res['duration_s'] / 60:.1f} min at {res['timing']['fs']:.1f} Hz, timing std {res['timing']['dt_std_us']:.1f} us, "
      f"gaps {res['timing']['gaps']}, sessions {res['sessions']}")
for s, unit, sc in (("gyro", "deg/s", np.degrees(1)), ("acc", "mg", 1e3 / G)):
    for ax in "xyz":
        r = res[s][ax]
        print(f"  {s:4s} {ax}: mean {r['mean'] * sc:9.4f} {unit}  std {r['std'] * sc:8.4f}  "
              f"N {r['N'] * sc:.3e} {unit}/rtHz  B {r['B'] * sc:.3e} {unit} at {r['tau_B']:.0f} s  "
              f"K {r['K'] * sc:.2e}  lsb {r['lsb'] * sc:.2e}" + ("" if r["temp_slope"] is None else
                                                                  f"  temp {r['temp_slope'] * sc:+.2e} {unit}/C"))

fp = nm.filter_params(res)
em = nm.error_model_params(res)
checks = [10.0, 30.0, 60.0]
pred = nm.predicted_drift_axes(res, np.array(checks), t_align=10.0)
tt, head, pos = nm.drift_segments(main, segment_s=a.segment)
drift = dict(checkpoints_s=checks, predicted_heading_deg=np.degrees(pred["heading"]).tolist(),
             predicted_position_m=pred["position"].tolist())
if tt is not None:
    idx = [int(np.searchsorted(tt, c)) for c in checks if c <= tt[-1]]
    drift.update(segments=int(len(head)),
                 measured_heading_rms_deg=[float(np.degrees(np.sqrt(np.mean(head[:, i] ** 2)))) for i in idx],
                 measured_position_rms_m=[float(np.sqrt(np.mean(pos[:, i] ** 2))) for i in idx])
jit = nm.jitter_effect(main)
print("drift at", checks, "s  predicted heading [deg]", np.round(drift["predicted_heading_deg"], 3),
      " position [m]", np.round(drift["predicted_position_m"], 3))
if "measured_position_rms_m" in drift:
    print(f"   measured over {drift['segments']} segments: heading RMS [deg]",
          np.round(drift["measured_heading_rms_deg"], 3), " position RMS [m]", np.round(drift["measured_position_rms_m"], 3))
print("heading error from assuming a fixed sample interval over the whole run [deg]:", np.round(np.degrees(jit), 4))

out = dict(fp, source=[a.dataset] + a.extra, calibrated=bool(a.calib), timing=res["timing"],
           gyro=res["gyro"], acc=res["acc"], repeatability=res.get("repeatability"), temp_c=res.get("temp_c"),
           error_model=em, drift=drift, jitter_heading_error_rad=jit)
with open(a.out, "w") as f:
    yaml.safe_dump(io._plain(out), f, sort_keys=False, default_flow_style=None)
print("wrote", a.out)

if a.plots:
    from phone_data import analysis as an
    os.makedirs(a.plots, exist_ok=True)
    an.plot_allan(res, curves, os.path.join(a.plots, "allan.png"))
    an.plot_bias_vs_time(main, os.path.join(a.plots, "bias_vs_time.png"))
    an.plot_dt_hist(main, os.path.join(a.plots, "dt_hist.png"))
    if tt is not None:
        p = nm.predicted_drift_axes(res, tt, t_align=10.0)
        an.plot_free_drift(tt, head, pos, p, os.path.join(a.plots, "free_drift.png"))
    print("plots in", a.plots)

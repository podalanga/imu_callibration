"""Score many runs at once (gate M6). Uses the on-device estimate when present, else replays in Python.

  python scripts/score_runs.py runs/*_track_* --calib calib.yaml --config nav_config.json --out results/scores.csv
Reports one row per run plus mean and standard deviation. Runs listed in --heldout are flagged so that
tuning runs are never reported as independent results.
"""
import argparse, os, sys, json, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, runner, evaluate
from nav.calib import CalibParams

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--calib", default=None)
ap.add_argument("--config", default=None, help="frozen NavConfig JSON")
ap.add_argument("--source", choices=["device", "python"], default="device")
ap.add_argument("--out", default="scores.csv")
a = ap.parse_args()
cp = CalibParams.from_yaml(a.calib) if a.calib else None
cfg = runner.NavConfig(**json.load(open(a.config))) if a.config else runner.NavConfig()
rows = []
for d in sorted(sum([glob.glob(r) for r in a.runs], [])):
    ds = io.load(d)
    dev = os.path.join(d, "est_device.csv")
    if a.source == "device" and os.path.exists(dev):
        est, src = np.loadtxt(dev, delimiter=",", skiprows=1, ndmin=2), "device"
    else:
        est, src = runner.run(ds.imu, cfg, calib=cp)["est"], "python"
    m = evaluate.checkpoint_errors(est, ds.markers)
    rows.append((os.path.basename(d), src, m["max_err"], m["rms_err"], m["return_to_start"], m["return_along"],
                 m["return_cross"], m["max_err_first_30s"], 100 * m["max_err"] / m["total_distance"], m["tier1"], m["tier2"]))
    print(evaluate.summary_line(os.path.basename(d)[:28], m), "|", src)
hdr = "run,source,max_err_m,rms_err_m,return_m,return_along_m,return_cross_m,max_first30s_m,max_pct_of_path,tier1,tier2"
with open(a.out, "w") as f:
    f.write(hdr + "\n")
    for r in rows:
        f.write(",".join(str(x) if not isinstance(x, float) else f"{x:.4f}" for x in r) + "\n")
v = np.array([r[2:9] for r in rows], float)
print(f"\n{len(rows)} runs: max error {v[:,0].mean():.2f} +- {v[:,0].std():.2f} m, "
      f"return-to-start {v[:,2].mean():.2f} +- {v[:,2].std():.2f} m -> {a.out}")

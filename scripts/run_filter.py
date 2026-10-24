"""Run the causal filter on one dataset and score it, or run the full ablation (X3 to X7).

  python scripts/run_filter.py runs/track_07 --calib calib.yaml --noise noise.yaml --ablation --plots results/track_07
  python scripts/run_filter.py runs/track_07 --config '{"use_line": true}' --est runs/track_07/est.csv
"""
import argparse, os, sys, json
import numpy as np, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import io, runner, evaluate
from nav.calib import CalibParams

ABLATION = [("X3 free inertial", dict(use_zupt=False, use_nhc=False)),
            ("X5 ZUPT + ZARU", dict(use_nhc=False)),
            ("X6 + NHC", dict()),
            ("X7 + line constraint", dict(use_line=True))]

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
    base.update(Na=n["Na"], Ng=n["Ng"])
runs = ABLATION if a.ablation else [("configured", {})]
results = {}
for name, kw in runs:
    cfg = runner.NavConfig(**{**base, **kw})
    r = runner.run(ds.imu, cfg, calib=cp)
    if ds.markers is not None:
        m = evaluate.checkpoint_errors(r["est"], ds.markers)
        print(evaluate.summary_line(name, m), "| gated:", r["rejected"],
              "| mean NIS:", {k: round(float(v.mean()), 2) for k, v in r["nis"].items() if len(v)})
        results[name] = {k: v for k, v in m.items() if not isinstance(v, np.ndarray)}
        if a.plots:
            os.makedirs(a.plots, exist_ok=True)
            evaluate.plot_run(m, r["est"], ds.markers, name, os.path.join(a.plots, name.split()[0] + ".png"))
    if a.est and not a.ablation:
        io.save_est(a.est, r["est"])
if a.plots and results:
    with open(os.path.join(a.plots, "metrics.json"), "w") as f:
        json.dump(results, f, indent=1, default=float)

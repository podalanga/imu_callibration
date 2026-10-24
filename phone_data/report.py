"""Self-contained HTML report (plots embedded) from phone_data/analysis.py results. Print it to PDF from the browser."""
import base64, html, os
import numpy as np

G = 9.80665
D = np.degrees(1)


def _f(v, d=3):
    try:
        return "–" if v is None or not np.isfinite(v) else f"{v:.{d}f}"
    except TypeError:
        return str(v)


def _e(v):
    try:
        return "–" if v is None or not np.isfinite(v) else f"{v:.2e}"
    except TypeError:
        return str(v)


def _lim(r, key, scale, fmt):
    """Value of B or K, or "< limit" when the recording was too short to resolve it."""
    v, ok = r[key], r.get(key + "_resolved", True)
    if ok and v is not None and np.isfinite(v):
        return fmt(v * scale)
    lim = r.get(key + "_limit")
    return f"&lt; {fmt(lim * scale)}" if lim is not None and np.isfinite(lim) else "–"


def conclusions(R):
    """Plain-language findings, most important first."""
    out = []
    bad = [c for c in R["checks"] if c["level"] in ("warn", "fail")]
    out.append("Data quality: all checks passed." if not bad else
               "Data quality: " + "; ".join(f"{c['label'].lower()} {c['level']} ({c['value']})" for c in bad) + ".")
    n = R.get("noise")
    if n:
        g = n["gyro"]
        bval = lambda r: _f(r["B"] * D * 3600, 1) if r.get("B_resolved", True) else f"< {_f(r['B_limit'] * D * 3600, 0)}"
        out.append("Gyro white noise N = " + ", ".join(_f(g[a]["N"] * D * 1e3, 2) for a in "xyz")
                   + " m°/s/√Hz (x, y, z); bias instability B = " + ", ".join(bval(g[a]) for a in "xyz") + " °/h.")
        if not all(g[a].get("B_resolved", True) for a in "xyz"):
            out.append("Bias instability is not resolved on every axis: the Allan curve has not flattened within the "
                       "recording, so only an upper limit is known. Record for at least an hour to measure it.")
        a_ = n["acc"]
        out.append("Accelerometer white noise N = " + ", ".join(_f(a_[a]["N"] / G * 1e6, 0) for a in "xyz") + " µg/√Hz.")
        if n.get("repeatability"):
            rg = np.array(n["repeatability"]["gyro"]) * D
            out.append(f"Gyro turn-on bias changes by {_f(rg.max(), 3)} °/s (1σ, worst axis) between starts, "
                       "so the bias must be re-estimated at every start.")
    d = R.get("drift")
    if d:
        i = len(d["checkpoints_s"]) - 1 if 30.0 not in d["checkpoints_s"] else d["checkpoints_s"].index(30.0)
        t = d["checkpoints_s"][i]
        mh, ph, mp, pp = d["measured_heading_deg"][i], d["predicted_heading_deg"][i], d["measured_position_m"][i], d["predicted_position_m"][i]
        out.append(f"Without aiding, after {t:.0f} s the heading drifts {_f(mh, 3)}° (predicted {_f(ph, 3)}°) and the position "
                   f"{_f(mp, 2)} m (predicted {_f(pp, 2)} m), over {d['segments']} undisturbed segments.")
        terms = d["terms_at_last_m"]
        big = max(terms, key=terms.get)
        out.append(f"The largest predicted position term is {big.replace('_', ' ')} ({_f(terms[big], 2)} m at "
                   f"{d['checkpoints_s'][-1]:.0f} s): " + ("better gyro bias estimation (longer still periods) is the most "
                   "effective fix." if big == "tilt_from_gyro" else "accelerometer bias estimation matters most here."))
        if t == 30.0:
            out.append(f"Free inertial integration {'meets' if mp <= 1.0 else 'misses'} the Tier 1 limit of 1 m at 30 s "
                       "on this phone; aiding is required for Tier 2.")
        ratio = mh / ph if ph else np.nan
        if np.isfinite(ratio):
            out.append(f"The noise model {'explains' if 0.5 <= ratio <= 2 else 'does not fully explain'} the measured heading "
                       f"drift (measured / predicted = {_f(ratio, 2)}).")
    fi = (R.get("filter") or {}).get("free inertial")
    if fi:
        out.append(f"Left to integrate freely for {R['filter_duration_s'] / 60:.1f} min, the IMU alone reaches "
                   f"{_f(fi['final_p_m'], 1)} m of position error, {_f(fi['final_v_mps'], 2)} m/s of velocity error and "
                   f"{_f(fi['final_tilt_deg'], 2)}° of tilt: without aiding the solution is unusable after seconds.")
    jit = np.max(np.abs(R.get("jitter_heading_error_deg", [0])))
    out.append(f"Assuming a fixed sample interval instead of hardware timestamps would add only {_f(jit, 4)}° of heading "
               "error over the run: timing jitter is negligible.")
    return out + R.get("notes", [])


def _img(run_dir, rel, alt):
    p = os.path.join(run_dir, rel)
    if not rel or not os.path.isfile(p):
        return ""
    with open(p, "rb") as f:
        b = base64.b64encode(f.read()).decode()
    return f'<figure><img alt="{html.escape(alt)}" src="data:image/png;base64,{b}"><figcaption>{html.escape(alt)}</figcaption></figure>'


def _table(head, rows, cls=""):
    h = "".join(f"<th>{html.escape(str(c))}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tw"><table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


def render(R, run_dir):
    m = R.get("meta", {})
    P = R.get("plots", {})
    badge = lambda lvl: f'<span class="b {lvl}">{ {"pass": "PASS", "warn": "WARN", "fail": "FAIL", "info": "INFO"}[lvl] }</span>'
    s = []
    s.append(f"""<header><h1>Drift analysis report</h1>
<p class="sub">{html.escape(R['run'])} · {html.escape({'static': 'Static drift test', 'quick': 'Quick check', 'turnon': 'Turn-on repeatability'}[R['kind']])}
 · {html.escape(str(m.get('started', '')))}</p>
<dl><dt>Phone</dt><dd>{html.escape(str(m.get('model', '–')))} (Android {html.escape(str(m.get('android', '–')))})</dd>
<dt>Duration</dt><dd>{R['duration_s'] / 60:.1f} min</dd>
<dt>Calibration</dt><dd>{'applied: ' + html.escape(str(R['calib'])) if R['calibrated'] else 'not applied (raw readings)'}</dd>
<dt>Note</dt><dd>{html.escape(str(m.get('note', '')) or '–')}</dd>
{'<dt>Combined with</dt><dd>' + html.escape(', '.join(os.path.basename(os.path.normpath(e)) for e in R['extras'])) + '</dd>' if R['extras'] else ''}</dl></header>""")

    s.append("<section><h2>Findings</h2><ol>" + "".join(f"<li>{html.escape(c)}</li>" for c in R["conclusions"]) + "</ol></section>")

    s.append("<section><h2>1. Data quality</h2>")
    s.append(_table(["Check", "Result", "Value", "Note"], [[html.escape(c["label"]), badge(c["level"]), html.escape(c["value"]),
                                                           html.escape(c["note"])] for c in R["checks"]]))
    acq = R["acquisition"]
    s.append(_table(["Sensor", "Samples", "Rate [Hz]", "dt median [ms]", "dt std [µs]", "dt max [ms]", "Gaps", "Lost samples"],
                    [[k, f"{v['n']:,}", _f(v["rate_hz"], 1), _f(v["dt_median_ms"], 3), _f(v["dt_std_us"], 1), _f(v["dt_max_ms"], 2),
                      v["gaps"], v["missing_samples"]] for k, v in acq.items() if isinstance(v, dict) and "n" in v]))
    ev = R["disturbances"]["events"]
    if ev:
        s.append("<h3>Disturbances</h3>" + _table(["Start [s]", "End [s]", "Peak gyro spread [°/s]", "Tilt change [°]"],
                                                   [[_f(a, 0), _f(b, 0), _f(c, 2), _f(d, 2)] for a, b, c, d in ev[:25]]))
        if len(ev) > 25:
            s.append(f"<p>…and {len(ev) - 25} more.</p>")
    if R.get("clean_stretch_s"):
        a, b = R["clean_stretch_s"]
        s.append(f"<p>Noise model and long-run drift use the longest undisturbed stretch: {a / 60:.1f} to {b / 60:.1f} min. "
                 "Drift segments skip any disturbed time.</p>")
    s.append('<div class="grid">' + _img(run_dir, P.get("stillness", ""), "Stillness over the run: 1 s gyro spread, disturbed periods in red")
             + _img(run_dir, P.get("dt_hist", ""), "Sample interval distribution from hardware timestamps") + "</div></section>")

    n = R.get("noise")
    if n:
        s.append("<section><h2>2. Noise model</h2><p>Per axis, from a least-squares fit of the Allan variance "
                 "σ²(τ) = N²/τ + (0.664 B)² + K² τ/3 over τ ≤ T/10, each point weighted by its uncertainty. "
                 "\"&lt; x\" means the term did not show up within the recording: x is an upper limit, not a value. Readings are raw (Android device frame: x right, y to the top edge, "
                 "z out of the screen)" + (", after calibration." if R["calibrated"] else ".") + "</p>")
        g = n["gyro"]
        s.append("<h3>Gyroscope</h3>" + _table(["Axis", "Mean [°/s]", "Std [°/s]", "N [m°/s/√Hz]", "B [°/h]", "τ at B [s]", "K [°/s/√s]", "Temp slope [°/s/°C]"],
                 [[a, _f(g[a]["mean"] * D, 4), _f(g[a]["std"] * D, 4), _f(g[a]["N"] * D * 1e3, 2), _lim(g[a], "B", D * 3600, lambda v: _f(v, 2)),
                   _f(g[a]["tau_B"], 0), _lim(g[a], "K", D, _e), _e(g[a]["temp_slope"] * D if g[a]["temp_slope"] is not None else None)] for a in "xyz"]))
        ac = n["acc"]
        s.append("<h3>Accelerometer</h3>" + _table(["Axis", "Mean [mg]", "Std [mg]", "N [µg/√Hz]", "B [µg]", "τ at B [s]", "K [µg/√s]", "Temp slope [mg/°C]"],
                 [[a, _f(ac[a]["mean"] / G * 1e3, 2), _f(ac[a]["std"] / G * 1e3, 3), _f(ac[a]["N"] / G * 1e6, 1), _lim(ac[a], "B", 1e6 / G, lambda v: _f(v, 1)),
                   _f(ac[a]["tau_B"], 0), _lim(ac[a], "K", 1e6 / G, _e), _e(ac[a]["temp_slope"] / G * 1e3 if ac[a]["temp_slope"] is not None else None)] for a in "xyz"]))
        if n.get("repeatability"):
            r = n["repeatability"]
            s.append("<h3>Turn-on repeatability</h3><p>Standard deviation of the session means across "
                     f"{n['sessions']} sessions.</p>" + _table(["", "x", "y", "z"],
                     [["Gyro [°/s]"] + [_f(v * D, 4) for v in r["gyro"]], ["Accel [mg]"] + [_f(v / G * 1e3, 2) for v in r["acc"]]]))
            if R.get("extra_means"):
                s.append(_table(["Session", "Gyro mean x, y, z [°/s]", "Accel mean x, y, z [m/s²]"],
                                [[html.escape(k), ", ".join(_f(v * D, 4) for v in x["gyro"]), ", ".join(_f(v, 4) for v in x["acc"])]
                                 for k, x in R["extra_means"].items()]))
        s.append(_img(run_dir, P.get("allan", ""), "Allan deviation with the fitted noise terms")
                 + _img(run_dir, P.get("bias_vs_time", ""), "Bias change over time against battery temperature") + "</section>")

    d = R.get("drift")
    if d:
        s.append("<section><h2>3. Drift: measured against the noise model</h2><p>Each segment aligns on "
                 f"10 s (bias and gravity from the mean), then integrates freely for {d['segment_s']:.0f} s: gyro → attitude error θ, "
                 "accelerometer rotated through θ (a + θ × a, which is where tilt leaks gravity into position) → velocity → position. "
                 f"Measured = RMS over {d['segments']} undisturbed segments. Predicted 1σ: heading √((N√t)² + (b t)²) and position "
                 "√2 · √((N_a t^1.5/√3)² + (b_a t²/2)² + (g b_g t³/6)²), where each bias b = √(B² + N²/t_align) includes the error of "
                 "the 10 s alignment average.</p>")
        s.append(_table(["Time [s]", "Heading, measured [°]", "Heading, predicted [°]", "Position, measured [m]", "Position, predicted [m]"],
                        [[_f(t, 0), _f(a, 4), _f(b, 4), _f(c, 3), _f(e, 3)] for t, a, b, c, e in
                         zip(d["checkpoints_s"], d["measured_heading_deg"], d["predicted_heading_deg"], d["measured_position_m"], d["predicted_position_m"])]))
        tm = d["terms_at_last_m"]
        s.append(f"<p>Predicted position terms at {d['checkpoints_s'][-1]:.0f} s: accelerometer white noise {_f(tm['accel_white'], 3)} m, "
                 f"accelerometer bias {_f(tm['accel_bias'], 3)} m, tilt from gyro bias {_f(tm['tilt_from_gyro'], 3)} m.</p>")
        s.append(_img(run_dir, P.get("free_drift", ""), "Free inertial drift: grey = segments, blue = measured RMS, red dashed = prediction") + "</section>")

    fi = (R.get("filter") or {}).get("free inertial")
    if fi:
        s.append(f"<section><h2>4. Long-run free inertial drift</h2><p>15-state error-state Kalman filter with no aiding over "
                 f"{R['filter_duration_s'] / 60:.1f} min of undisturbed data, starting from a 10 s alignment. Truth is zero motion "
                 "and constant attitude, so any estimated motion is drift.</p>")
        s.append(_table(["Final position error", "Max position error", "Final |v| [m/s]", "Final tilt [°]"],
                        [[_f(fi["final_p_m"], 3) + " m", _f(fi["max_p_m"], 3) + " m", _f(fi["final_v_mps"], 3), _f(fi["final_tilt_deg"], 3)]]))
        s.append(_img(run_dir, P.get("filter_drift", ""), "Position, velocity and tilt drift with no aiding") + "</section>")

    s.append("<section><h2>Files</h2><p>In this run folder: raw_acc.csv, raw_gyro.csv (every event with its hardware timestamp), "
             "imu.csv (merged, raw sensor frame, SI units), battery.csv, phone_noise.yaml (noise model for the "
             "simulator), results.json (every number in this report).</p></section>")

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Drift report: {html.escape(R['run'])}</title><style>
:root{{--bg:#fff;--ink:#1d2330;--muted:#5f6b7f;--line:#dde1e8;--pass:#1b7f4b;--warn:#b54708;--fail:#b42318;--info:#2e6da4}}
@media (prefers-color-scheme: dark){{:root{{--bg:#14181f;--ink:#e6e9ef;--muted:#9aa3b5;--line:#2c3340}} img{{background:#fff}}}}
body{{margin:0 auto;max-width:1100px;padding:24px 16px;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
h1{{margin:0;font-size:26px}} h2{{margin-top:36px;border-bottom:2px solid var(--line);padding-bottom:4px;font-size:20px}} h3{{font-size:15px;margin:18px 0 6px}}
.sub{{color:var(--muted);margin:4px 0 12px}} dl{{display:grid;grid-template-columns:max-content 1fr;gap:2px 16px;margin:0}} dt{{color:var(--muted)}} dd{{margin:0}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;width:100%;margin:8px 0 12px;font-variant-numeric:tabular-nums;font-size:14px}}
th,td{{text-align:left;padding:5px 8px;border-bottom:1px solid var(--line);vertical-align:top}} th{{color:var(--muted);font-weight:600}}
.b{{display:inline-block;min-width:44px;text-align:center;border-radius:4px;padding:1px 6px;font-size:12px;font-weight:700;color:#fff}}
.b.pass{{background:var(--pass)}} .b.warn{{background:var(--warn)}} .b.fail{{background:var(--fail)}} .b.info{{background:var(--info)}}
figure{{margin:12px 0}} img{{max-width:100%;border:1px solid var(--line);border-radius:6px}} figcaption{{color:var(--muted);font-size:13px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,420px),1fr));gap:12px}}
ol li{{margin:4px 0}} @media print{{h2{{break-before:auto}} figure,table{{break-inside:avoid}}}}
</style></head><body>{''.join(s)}</body></html>"""

"""Per-sensor, per-axis noise characterisation from static recordings, and drift predicted from it.

The model is the standard one used by nav/synth.py (ErrorModel) and the ESKF:
  reading = truth + turn-on bias + bias instability (Gauss-Markov) + bias random walk (K) + white noise (N)
            + temperature slope * (T - T0), then quantised to one LSB.
N, B, K come from the Allan deviation (nav/allan.py). Units: gyro rad/s, accel m/s^2.
"""
import numpy as np

from . import allan

# A first-order Gauss-Markov process with std s and correlation time tc has its Allan minimum of about
# 0.618 s at tau ~ 1.89 tc. Matching that minimum to the flicker-floor estimate B = min/0.664 gives:
GM_SIGMA_PER_B = 0.664 / 0.618
GM_TAU_PER_TAUB = 1.0 / 1.89


def lsb_estimate(x):
    """Smallest step between distinct readings: the quantisation step if the output is not rescaled."""
    u = np.unique(np.round(x, 12))
    d = np.diff(u)
    d = d[d > 1e-12]
    return float(np.percentile(d, 5)) if len(d) else float("nan")


def window_means(t, x, win_s):
    """Non-overlapping window means of x (N,k) over time t (N,)."""
    k = np.floor((t - t[0]) / win_s).astype(int)
    n = np.bincount(k)
    ok = n > 0
    tm = np.bincount(k, t)[ok] / n[ok]
    xm = np.column_stack([np.bincount(k, x[:, i])[ok] / n[ok] for i in range(x.shape[1])])
    return tm, xm


def temp_slope(t, x, temp, win_s=60.0, min_range_c=0.5):
    """Linear slope of the window-mean reading against temperature, per axis; None if T barely changed."""
    if temp is None or np.all(np.isnan(temp)) or np.nanmax(temp) - np.nanmin(temp) < min_range_c:
        return None
    tm, xm = window_means(t, np.column_stack([x, temp]), win_s)
    T = xm[:, -1]
    return [float(np.polyfit(T, xm[:, i], 1)[0]) for i in range(x.shape[1])]


def fit_allan(taus, dev, T, err=None):
    """Fit sigma^2(tau) = N^2/tau + (0.664 B)^2 + K^2 tau/3 on tau <= T/10 (N, B, K >= 0).

    Reading B off the curve minimum fails when the curve is still falling at the longest tau (the minimum
    then sits in the noisy tail); a fit of the three standard terms over the reliable range does not.
    Each point is weighted by its own relative uncertainty (err = adev's 1-sigma error bars): long-tau points,
    averaged over few clusters, count less. tau_B is where the fitted curve is lowest.

    A term is "resolved" when it carries at least half of the fitted variance somewhere in the range. When B or
    K is not resolved the recording is too short to see it: B_limit / K_limit are then upper limits from the
    lowest / last point of the measured curve, and B or K should be reported as "< limit", not as a value.
    """
    from scipy.optimize import nnls
    m = taus <= max(T / 10, taus[min(len(taus) - 1, 5)])
    tt, v = taus[m], dev[m] ** 2
    rel = np.ones_like(tt) if err is None else 2 * np.maximum(err[m] / dev[m], 1e-3)   # rel. error of sigma^2
    w = 1.0 / (v * rel)
    A = np.column_stack([1 / tt, np.full_like(tt, 0.664 ** 2), tt / 3]) * w[:, None]
    (N2, B2, K2), _ = nnls(A, v * w)
    N, B, K = np.sqrt([N2, B2, K2])
    model = N2 / tt + B2 * 0.664 ** 2 + K2 * tt / 3
    share = lambda term: float(np.max(term / model)) if np.all(model > 0) else 0.0
    B_res = share(np.full_like(tt, B2 * 0.664 ** 2)) >= 0.5
    K_res = share(K2 * tt / 3) >= 0.5
    tau_B = float(np.clip(np.sqrt(3) * N / K, tt[0], tt[-1])) if K > 0 else float(tt[-1])
    return dict(N=float(N), B=float(B), K=float(K) if K > 0 else float("nan"), tau_B=tau_B,
                B_resolved=bool(B_res), K_resolved=bool(K_res),
                B_limit=float(np.min(dev[m]) / 0.664), K_limit=float(dev[m][-1] * np.sqrt(3 / tt[-1])))


def characterise_sensor(t, x, temp=None):
    """x: (N,3) static readings. Returns per-axis dict and the Allan curves."""
    fs = 1.0 / np.median(np.diff(t))
    axes, curves = {}, {}
    slope = temp_slope(t, x, temp)
    for i, a in enumerate("xyz"):
        taus, dev, err = allan.adev(x[:, i], fs)
        p = fit_allan(taus, dev, t[-1] - t[0], err)
        axes[a] = dict(mean=float(x[:, i].mean()), std=float(x[:, i].std()), **p,
                       lsb=lsb_estimate(x[:, i]), temp_slope=None if slope is None else slope[i])
        curves[a] = (taus, dev, err)
    return axes, curves


def timing_stats(t):
    dt = np.diff(t)
    med = np.median(dt)
    return dict(fs=float(1.0 / med), dt_std_us=float(dt.std() * 1e6), gaps=int(np.sum(dt > 1.5 * med)))


def characterise(imus):
    """imus: list of nav.io.Imu static sessions. The longest one sets N, B, K; all of them set repeatability."""
    main = max(imus, key=lambda i: i.t[-1])
    res = dict(sessions=len(imus), duration_s=float(main.t[-1]), timing=timing_stats(main.t))
    curves = {}
    for name, data in (("gyro", main.gyro), ("acc", main.acc)):
        res[name], curves[name] = characterise_sensor(main.t, data, main.temp)
    if len(imus) > 1:                          # turn-on to turn-on bias change
        res["repeatability"] = dict(gyro=np.std([i.gyro.mean(axis=0) for i in imus], axis=0, ddof=1).tolist(),
                                    acc=np.std([i.acc.mean(axis=0) for i in imus], axis=0, ddof=1).tolist())
    if main.temp is not None and not np.all(np.isnan(main.temp)):
        res["temp_c"] = [float(np.nanmin(main.temp)), float(np.nanmax(main.temp))]
    return res, curves


def axis_array(res, sensor, key):
    return np.array([res[sensor][a][key] for a in "xyz"], float)


def error_model_params(res):
    """Parameters for nav.synth.ErrorModel (per axis), from characterise()."""
    nz = lambda v: np.nan_to_num(v).tolist()
    rep = res.get("repeatability", {})
    out = dict(fs=res["timing"]["fs"], jitter_us=res["timing"]["dt_std_us"],
               Na=nz(axis_array(res, "acc", "N")), Ng=nz(axis_array(res, "gyro", "N")),
               Ka_rw=nz(axis_array(res, "acc", "K")), Kg_rw=nz(axis_array(res, "gyro", "K")),
               B_acc=nz(axis_array(res, "acc", "B") * GM_SIGMA_PER_B),
               tau_acc=nz(axis_array(res, "acc", "tau_B") * GM_TAU_PER_TAUB),
               B_gyro=nz(axis_array(res, "gyro", "B") * GM_SIGMA_PER_B),
               tau_gyro=nz(axis_array(res, "gyro", "tau_B") * GM_TAU_PER_TAUB),
               bg=nz(axis_array(res, "gyro", "mean")),
               bias_repeat_acc=rep.get("acc", [0.0] * 3), bias_repeat_gyro=rep.get("gyro", [0.0] * 3),
               lsb_acc=float(np.nanmedian(axis_array(res, "acc", "lsb"))),
               lsb_gyro=float(np.nanmedian(axis_array(res, "gyro", "lsb"))))
    ts = res["gyro"]["x"]["temp_slope"]
    if ts is not None:                    # apply() makes reading = truth - bg(T): negate the measured reading slope
        out["temp_slope_bg"] = nz(-axis_array(res, "gyro", "temp_slope"))
    return out


def temp_compensate(imu, res):
    """Remove the measured linear temperature dependence (characterise() slopes) from gyro and accel, in place.

    reading_corrected = reading - slope * (T - T_ref), T_ref = median temperature of the run. Returns the
    applied slopes per sensor, or None when no slope was measured (temperature changed < 0.5 C) or there is
    no temperature channel.
    """
    if imu.temp is None or np.all(np.isnan(imu.temp)) or res["gyro"]["x"]["temp_slope"] is None:
        return None
    T = np.nan_to_num(imu.temp - np.nanmedian(imu.temp))
    out = {}
    for name in ("gyro", "acc"):
        s = axis_array(res, name, "temp_slope")
        x = imu.gyro if name == "gyro" else imu.acc
        x -= np.outer(T, s)
        out[name] = s.tolist()
    out["T_ref"] = float(np.nanmedian(imu.temp))
    return out


def filter_params(res):
    """Scalar noise values for nav.runner.NavConfig / scripts/run_filter.py --noise (same keys as run_allan.py)."""
    def m(s, k):
        v = axis_array(res, s, k)
        return float(np.mean(v[np.isfinite(v)])) if np.any(np.isfinite(v)) else float("nan")
    out = dict(Na=m("acc", "N"), Ng=m("gyro", "N"), B_acc=m("acc", "B"), B_gyro=m("gyro", "B"),
               K_acc=m("acc", "K"), K_gyro=m("gyro", "K"), duration_s=res["duration_s"])
    # bias random walk for the ESKF (NavConfig.Kba / Kbg), only from axes where K was actually resolved;
    # otherwise the filter keeps its defaults rather than a K the recording could not see
    for sensor, key in (("acc", "Kba"), ("gyro", "Kbg")):
        k = [res[sensor][a]["K"] for a in "xyz" if res[sensor][a].get("K_resolved") and np.isfinite(res[sensor][a]["K"])]
        if k:
            out[key] = float(np.mean(k))
    return out


def predicted_drift(t, Ng, Bg, Na, Ba, t_align=None, g=9.80665):
    """1-sigma drift of a free inertial solution on a static, level sensor.

    The initial bias comes from averaging a still period of t_align seconds; that average itself is off by
    N/sqrt(t_align), a constant bias error over the run (None = bias known perfectly). Bias terms per sensor
    combine instability B and that alignment error: b = sqrt(B^2 + N^2/t_align).
    heading:  white gyro noise N*sqrt(t), bias b_g*t
    position: white accel noise N*t^1.5/sqrt(3), accel bias b_a*t^2/2, tilt from gyro bias g*b_g*t^3/6
    """
    t = np.asarray(t, float)
    bg = np.sqrt(Bg ** 2 + (Ng ** 2 / t_align if t_align else 0.0))
    ba = np.sqrt(Ba ** 2 + (Na ** 2 / t_align if t_align else 0.0))
    p = dict(heading=np.sqrt((Ng * np.sqrt(t)) ** 2 + (bg * t) ** 2),
             pos_acc_white=Na * t ** 1.5 / np.sqrt(3), pos_acc_bias=ba * t ** 2 / 2, pos_tilt=g * bg * t ** 3 / 6)
    # horizontal error = 2 axes, so the total is sqrt(2) times the per-axis value
    p["position"] = np.sqrt(2) * np.sqrt(p["pos_acc_white"] ** 2 + p["pos_acc_bias"] ** 2 + p["pos_tilt"] ** 2)
    return p


def predicted_drift_axes(res, t, t_align=None, g=9.80665):
    """predicted_drift() with the axes that matter for a level sensor: gyro z for heading, gyro x/y (tilt) and
    accel x/y for horizontal position. res: characterise() output."""
    a = lambda s, k, axes: float(np.sqrt(np.mean(np.nan_to_num(axis_array(res, s, k)[list(axes)]) ** 2)))
    head = predicted_drift(t, a("gyro", "N", [2]), a("gyro", "B", [2]), 0.0, 0.0, t_align, g)["heading"]
    p = predicted_drift(t, a("gyro", "N", [0, 1]), a("gyro", "B", [0, 1]), a("acc", "N", [0, 1]), a("acc", "B", [0, 1]), t_align, g)
    p["heading"] = head
    return p


def free_drift(imu, t_start=0.0, t_align=10.0, segment_s=60.0):
    """Free strapdown solution on static data: the measured counterpart of predicted_drift().

    The window [t_start, t_start + t_align] gives the gyro bias and the gravity vector (accel bias included),
    so the solution starts with a perfectly known bias. Afterwards the gyro residual is integrated to an
    attitude error theta, the accelerometer is rotated through it (small-angle: a + theta x a, which is
    where tilt couples gravity into position) and integrated twice.
    Returns t (N,), theta (N,3) rad, velocity (N,3) m/s, position (N,3) m, in the body frame at the start
    (level when the phone lies flat, which is the static-test protocol).
    """
    t = imu.t
    na, n0 = int(np.searchsorted(t, t_start)), int(np.searchsorted(t, t_start + t_align))
    n1 = int(np.searchsorted(t, t_start + t_align + segment_s))
    w0, a0 = imu.gyro[na:n0].mean(axis=0), imu.acc[na:n0].mean(axis=0)
    tt = t[n0:n1] - t[n0]
    dt = np.diff(t[n0:n1], prepend=t[n0])[:, None]
    theta = np.cumsum((imu.gyro[n0:n1] - w0) * dt, axis=0)
    a = imu.acc[n0:n1]
    vel = np.cumsum((a + np.cross(theta, a) - a0) * dt, axis=0)
    pos = np.cumsum(vel * dt, axis=0)
    return tt, theta, vel, pos


def drift_segments(imu, t_align=10.0, segment_s=60.0, disturbed=None):
    """Run free_drift() on back-to-back segments of a long static recording.

    Returns t and the per-segment heading error (rad, about z) and horizontal position error (m), shape
    (segments, N): their RMS across segments is the measured 1-sigma drift to compare with the prediction.
    disturbed: optional bool mask per sample (from disturbances()); segments touching it are skipped.
    """
    out_t, head, pos = None, [], []
    s = 0.0
    while s + t_align + segment_s <= imu.t[-1]:
        if disturbed is not None:
            i0, i1 = np.searchsorted(imu.t, [s, s + t_align + segment_s])
            if disturbed[i0:i1].any():
                s += t_align + segment_s
                continue
        tt, th, _, p = free_drift(imu, s, t_align, segment_s)
        if out_t is None:
            out_t = tt
        n = min(len(out_t), len(tt))
        head.append(th[:n, 2]); pos.append(np.linalg.norm(p[:n, :2], axis=1))
        out_t = out_t[:n]
        s += t_align + segment_s
    if not head:
        return None, None, None
    n = min(len(h) for h in head)
    return out_t[:n], np.array([h[:n] for h in head]), np.array([p[:n] for p in pos])


# A static test is only valid while nothing touches the phone. A 1 s window is "disturbed" when its spread
# is far above the run's typical window (vibration, taps), or when the gravity direction has moved
# (the phone was nudged to a new resting tilt, which does not show up as vibration).
DIST_WIN_S = 1.0
DIST_K = 4.0                          # x the median window spread
DIST_FLOOR_GYRO = np.radians(0.5)     # rad/s: never flag below this spread
DIST_FLOOR_ACC = 0.05                 # m/s^2
DIST_TILT = np.radians(0.5)           # rad: gravity direction change that counts as "moved"


def disturbance_rule(g_std, a_std, tilt, base_g, base_a):
    """The flagging rule shared by the offline and live detectors (all arguments per window)."""
    return ((g_std > max(DIST_K * base_g, DIST_FLOOR_GYRO)) | (a_std > max(DIST_K * base_a, DIST_FLOOR_ACC))
            | (tilt > DIST_TILT))


def disturbances(imu):
    """Find disturbed periods in a static recording.

    Returns dict: mask (bool per sample), events [(start_s, end_s, peak gyro spread deg/s, peak tilt deg)],
    fraction of time disturbed, and validity: "valid" (none), "partly valid" (< 5 %), "invalid".
    """
    t, g, a = imu.t, imu.gyro, imu.acc
    k = np.floor((t - t[0]) / DIST_WIN_S).astype(int)
    n = np.bincount(k).astype(float)
    ok = n > 1
    mean = lambda x: np.column_stack([np.bincount(k, x[:, i]) for i in range(3)]) / np.maximum(n, 1)[:, None]
    spread = lambda x: np.linalg.norm(np.sqrt(np.maximum(mean(x ** 2) - mean(x) ** 2, 0)), axis=1)
    g_std, a_std = spread(g), spread(a)
    am = mean(a)
    ref = np.median(am[ok], axis=0)
    cosang = am @ ref / (np.linalg.norm(am, axis=1) * np.linalg.norm(ref) + 1e-12)
    tilt = np.arccos(np.clip(cosang, -1, 1))
    flag = disturbance_rule(g_std, a_std, tilt, np.median(g_std[ok]), np.median(a_std[ok])) & ok
    events = []
    for w in np.flatnonzero(flag):
        if events and w == events[-1][1]:
            e = events[-1]
            events[-1] = (e[0], w + 1, max(e[2], g_std[w]), max(e[3], tilt[w]))
        else:
            events.append((w, w + 1, g_std[w], tilt[w]))
    mask = flag[k]
    frac = float(mask.mean())
    return dict(mask=mask, fraction=frac,
                events=[(float(s * DIST_WIN_S), float(e * DIST_WIN_S), float(np.degrees(p)), float(np.degrees(tl)))
                        for s, e, p, tl in events],
                validity="valid" if frac == 0 else ("partly valid" if frac < 0.05 else "invalid"))


def longest_clean(t, mask):
    """Sample range [i0, i1) of the longest undisturbed stretch."""
    edges = np.flatnonzero(np.diff(np.r_[1, mask.astype(int), 1]))
    runs = list(zip(edges[0::2], edges[1::2]))
    if not runs:
        return 0, 0
    return max(runs, key=lambda r: t[r[1] - 1] - t[r[0]])


def jitter_effect(imu):
    """Heading error caused by assuming a fixed sample interval instead of the hardware timestamps (rad)."""
    t, w = imu.t, imu.gyro
    dt = np.diff(t)
    nominal = np.median(dt)
    true_ang = np.sum(w[1:] * dt[:, None], axis=0)
    fixed_ang = np.sum(w[1:] * nominal, axis=0)
    return (fixed_ang - true_ang).tolist()

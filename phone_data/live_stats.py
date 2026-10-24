"""Running acquisition health and static-drift metrics for the live dashboard.

Everything is updated per event in O(1); numpy work happens only in snapshot(), called a few times a second.

Static drift experiment (phone left untouched): the first `align_s` seconds after start or reset give the
initial bias (gyro) and the gravity + bias vector (accel). After that, the residual is integrated:
  gyro:  angle error   = integral of (w - w0)                grows ~ N*sqrt(t) + B*t
  accel: velocity error = integral of (a - a0), position = integral of velocity    grows ~ t^1.5 and t^2
What remains is noise, bias change, and the error of the initial average itself (N/sqrt(align_s)), which
is what the Allan parameters predict; the theory envelopes include all three (nav/noise_model.py).
"""
from collections import deque
import time
import numpy as np

from nav import allan, noise_model as nm
from nav.noise_model import fit_allan

BLOCK_HZ = 20          # block averages kept for the live Allan deviation (tau >= 50 ms)
PLOT_HZ, PLOT_S = 50, 10
DRIFT_HZ = 2
SLOW_S = 10            # window of the bias-vs-temperature means


def thin(n, k):
    return np.arange(n) if n <= k else np.linspace(0, n - 1, k).astype(int)


class Channel:
    def __init__(self, name, align_s):
        self.name, self.align_s = name, align_s
        self.n = 0
        self.t0 = self.t_last = None
        self.nominal = None
        self.first_dts = []
        self.recent_dt = deque(maxlen=4000)
        self.gaps = self.missing = self.dups = 0
        self.max_gap = 0.0
        self.mean, self.m2 = np.zeros(3), np.zeros(3)
        self.plot = deque(maxlen=PLOT_HZ * PLOT_S)
        self.t_plot = None
        self.blk_sum, self.blk_n, self.blk_end, self.blocks = np.zeros(3), 0, None, []
        self.slow_sum, self.slow_n, self.slow_end, self.slow = np.zeros(3), 0, None, []
        self.android_bias = None
        self.allan, self.t_allan = {}, 0.0
        self.reset()

    def reset(self):
        self.t_align = None
        self.align_sum, self.align_n = np.zeros(3), 0
        self.bias0 = None
        self.i1, self.i2 = np.zeros(3), np.zeros(3)
        self.drift = []
        self.t_drift = None

    def update(self, t_ns, v):
        x = np.asarray(v[:3], float)
        if len(v) >= 6:
            self.android_bias = v[3:6]
        t = t_ns * 1e-9
        if self.t_last is not None:
            dt = t - self.t_last
            if dt <= 0:
                self.dups += 1
                return
            self.recent_dt.append(dt)
            if self.nominal is None:
                self.first_dts.append(dt)
                if len(self.first_dts) == 200:
                    self.nominal = float(np.median(self.first_dts))
            elif dt > 1.5 * self.nominal:
                self.gaps += 1
                self.missing += int(round(dt / self.nominal)) - 1
                self.max_gap = max(self.max_gap, dt)
        else:
            self.t0, dt = t, 0.0
        self.t_last = t
        self.n += 1
        d = x - self.mean                                   # Welford running mean / variance
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)
        if self.t_plot is None or t >= self.t_plot:
            self.plot.append((t, *x))
            self.t_plot = t + 1.0 / PLOT_HZ
        self.blk_sum += x; self.blk_n += 1
        if self.blk_end is None:
            self.blk_end = t + 1.0 / BLOCK_HZ
        elif t >= self.blk_end:
            self.blocks.append(self.blk_sum / self.blk_n)
            self.blk_sum, self.blk_n = np.zeros(3), 0
            self.blk_end += 1.0 / BLOCK_HZ
        self.slow_sum += x; self.slow_n += 1
        if self.slow_end is None:
            self.slow_end = t + SLOW_S
        elif t >= self.slow_end:
            self.slow.append((t - SLOW_S / 2, *(self.slow_sum / self.slow_n)))
            self.slow_sum, self.slow_n = np.zeros(3), 0
            self.slow_end += SLOW_S
        if self.bias0 is None:
            if self.t_align is None:
                self.t_align = t
            self.align_sum += x; self.align_n += 1
            if t - self.t_align >= self.align_s:
                self.bias0 = self.align_sum / self.align_n
                self.t_start = t
        elif dt > 0:
            e = x - self.bias0
            self.i1 += e * dt
            self.i2 += self.i1 * dt
            if self.t_drift is None or t >= self.t_drift:
                self.drift.append((t - self.t_start, *self.i1, *self.i2))
                self.t_drift = t + 1.0 / DRIFT_HZ

    def noise(self):
        """N and B per axis: from the live Allan deviation once available, else N from the sample std."""
        N = np.array([self.allan.get(a, {}).get("N", np.nan) for a in "xyz"])
        B = np.array([self.allan.get(a, {}).get("B", np.nan) for a in "xyz"])
        if np.any(np.isnan(N)) and self.n > 1 and self.nominal:
            N = np.where(np.isnan(N), np.sqrt(self.m2 / (self.n - 1)) * np.sqrt(self.nominal), N)
        return N, np.nan_to_num(B)

    def update_allan(self, every_s=30.0):
        if len(self.blocks) < 20 * BLOCK_HZ or time.monotonic() - self.t_allan < every_s:
            return
        self.t_allan = time.monotonic()
        b = np.array(self.blocks)
        out = {}
        for i, a in enumerate("xyz"):
            taus, dev, err = allan.adev(b[:, i], BLOCK_HZ, n_taus=40)
            out[a] = dict(taus=taus.tolist(), adev=dev.tolist(), **fit_allan(taus, dev, len(b) / BLOCK_HZ, err))
        self.allan = out

    def snapshot(self):
        rd = np.array(self.recent_dt) if self.recent_dt else np.array([np.nan])
        s = dict(n=self.n, duration_s=(self.t_last - self.t0) if self.n else 0.0,
                 rate_hz=float(1.0 / np.median(rd)) if self.recent_dt else 0.0,
                 dt_median_ms=float(np.median(rd) * 1e3), dt_std_us=float(np.std(rd) * 1e6),
                 dt_max_ms=float(np.max(rd) * 1e3), gaps=self.gaps, missing=self.missing,
                 max_gap_ms=self.max_gap * 1e3, dups=self.dups,
                 mean=self.mean.tolist(), std=(np.sqrt(self.m2 / max(1, self.n - 1))).tolist(),
                 android_bias=list(self.android_bias) if self.android_bias is not None else None,
                 aligning=self.bias0 is None)
        if self.plot:
            p = np.array(self.plot)
            s["plot"] = np.round(np.column_stack([p[:, 0] - p[-1, 0], p[:, 1:]]), 6).T.tolist()
        if self.drift:
            d = np.array(self.drift)[thin(len(self.drift), 400)]
            N, B = self.noise()
            B = np.sqrt(B ** 2 + N ** 2 / self.align_s)          # bias left after averaging align_s seconds
            tt = d[:, 0:1]
            s["drift"] = np.round(d, 6).T.tolist()
            # 1-sigma envelopes, per axis: white noise alone, and white noise + bias instability
            if self.name == "gyro":
                env = np.sqrt((N * np.sqrt(tt)) ** 2 + (B * tt) ** 2)
                s["envelope"] = dict(i1_white=(N * np.sqrt(tt)).T.tolist(), i1=env.T.tolist())
            else:
                env_v = np.sqrt((N * np.sqrt(tt)) ** 2 + (B * tt) ** 2)
                env_p = np.sqrt((N * tt ** 1.5 / np.sqrt(3)) ** 2 + (B * tt ** 2 / 2) ** 2)
                s["envelope"] = dict(i1=env_v.T.tolist(), i2_white=(N * tt ** 1.5 / np.sqrt(3)).T.tolist(),
                                     i2=env_p.T.tolist())
        if self.slow:
            sl = np.array(self.slow)
            s["slow"] = np.round(np.column_stack([sl[:, 0] - self.t0, sl[:, 1:]]), 7).T.tolist()
        s["allan"] = self.allan
        return s


class LiveDisturbance:
    """Online version of nav.noise_model.disturbances(): same 1 s windows, same rule, updated per sample."""

    def __init__(self):
        self.t_end = None
        self.sum = {k: np.zeros(3) for k in ("g", "g2", "a", "a2")}
        self.n = 0
        self.g_hist, self.a_hist, self.ref, self.ref_list = [], [], None, []
        self.events, self.disturbed_s, self.windows = [], 0, 0
        self.last_acc = None

    def update_acc(self, a):
        self.last_acc = np.asarray(a[:3], float)

    def update_gyro(self, t, g):
        if self.last_acc is None:
            return
        if self.t_end is None:
            self.t0, self.t_end = t, t + nm.DIST_WIN_S
        if t >= self.t_end:
            self._close()
            self.t_end += nm.DIST_WIN_S
        g = np.asarray(g[:3], float)
        for k, v in (("g", g), ("g2", g * g), ("a", self.last_acc), ("a2", self.last_acc ** 2)):
            self.sum[k] += v
        self.n += 1

    def _close(self):
        if self.n < 2:
            return
        m = {k: v / self.n for k, v in self.sum.items()}
        g_std = float(np.linalg.norm(np.sqrt(np.maximum(m["g2"] - m["g"] ** 2, 0))))
        a_std = float(np.linalg.norm(np.sqrt(np.maximum(m["a2"] - m["a"] ** 2, 0))))
        if self.ref is None:
            self.ref_list.append(m["a"])
            if len(self.ref_list) == 10:
                self.ref = np.median(self.ref_list, axis=0)
        tilt = 0.0 if self.ref is None else float(np.arccos(np.clip(m["a"] @ self.ref / (np.linalg.norm(m["a"]) * np.linalg.norm(self.ref)), -1, 1)))
        if len(self.g_hist) < 10:          # learn the resting spread first; the offline pass re-checks this period
            self.g_hist.append(g_std); self.a_hist.append(a_std)
            self.windows += 1
            self.sum = {k: np.zeros(3) for k in self.sum}
            self.n = 0
            return
        base_g, base_a = float(np.median(self.g_hist)), float(np.median(self.a_hist))
        flag = bool(nm.disturbance_rule(g_std, a_std, tilt, base_g, base_a))
        if not flag:
            self.g_hist.append(g_std); self.a_hist.append(a_std)
        w = self.windows
        self.windows += 1
        if flag:
            self.disturbed_s += 1
            s = (self.t_end - nm.DIST_WIN_S) - self.t0
            if self.events and self.events[-1]["end_w"] == w:
                e = self.events[-1]
                e.update(end_w=w + 1, end_s=s + nm.DIST_WIN_S, peak_dps=max(e["peak_dps"], np.degrees(g_std)),
                         tilt_deg=max(e["tilt_deg"], np.degrees(tilt)))
            else:
                self.events.append(dict(end_w=w + 1, start_s=s, end_s=s + nm.DIST_WIN_S, peak_dps=float(np.degrees(g_std)),
                                        tilt_deg=float(np.degrees(tilt))))
        self.sum = {k: np.zeros(3) for k in self.sum}
        self.n = 0

    def snapshot(self):
        frac = self.disturbed_s / self.windows if self.windows else 0.0
        return dict(fraction=frac, disturbed_s=self.disturbed_s, events=self.events[-20:], n_events=len(self.events),
                    validity="valid" if frac == 0 else ("partly valid" if frac < 0.05 else "invalid"))


class LiveStats:
    def __init__(self, names=("acc", "gyro"), align_s=10.0):
        self.ch = {n: Channel(n, align_s) for n in names}
        self.dist = LiveDisturbance()
        self.battery = []          # (sensor-clock s, temp_c)
        self.clock_offset = None   # min(host_ns - sensor_ns): maps host time onto the sensor clock
        self.status = "starting"

    def update(self, name, t_ns, values, host_ns):
        off = host_ns - t_ns
        if self.clock_offset is None or off < self.clock_offset:
            self.clock_offset = off
        if name in self.ch:
            self.ch[name].update(t_ns, values)
        if name == "acc":
            self.dist.update_acc(values)
        elif name == "gyro":
            self.dist.update_gyro(t_ns * 1e-9, values)

    def sensor_ns(self, host_ns):
        return host_ns - self.clock_offset if self.clock_offset is not None else None

    def add_battery(self, sensor_ns, temp_c):
        self.battery.append((sensor_ns * 1e-9, temp_c))

    def reset(self):
        for c in self.ch.values():
            c.reset()

    def snapshot(self):
        for c in self.ch.values():
            c.update_allan()
        t0 = min((c.t0 for c in self.ch.values() if c.t0 is not None), default=0.0)
        b = [(t - t0, v) for t, v in self.battery]
        return dict(status=self.status, channels={n: c.snapshot() for n, c in self.ch.items()}, disturbance=self.dist.snapshot(),
                    battery=np.array(b).T.tolist() if b else [[], []],
                    temp_c=self.battery[-1][1] if self.battery else None)

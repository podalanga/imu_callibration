"""IMU test console: live demo, guided drift tests with automatic analysis and reports, past sessions.

  python phone_data/app.py                 # then open http://localhost:8050
  python phone_data/app.py --runs runs --http 8050

One process holds one WebSocket to the phone (stream.py) and fans every event out to: the monitor (rates,
plots, stillness), the demo (Mahony attitude for the 3D view, drift race) and, while a test runs, the
recorder (raw files + LiveStats). The app sets up `adb forward` itself and checks the phone over adb.
Analysis runs as a separate process (analysis.py) so it can never slow a recording down.
"""
import argparse, asyncio, json, math, os, re, shutil, subprocess, sys, time
from collections import deque
import numpy as np
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phone_data import stream
from phone_data.live_stats import LiveStats
from phone_data.phone_logger import RawWriter, adb, phone_meta

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = "github.umer0586.sensorserver"
PLOT_HZ, PLOT_S = 50, 10


def jdump(x):
    return re.sub(r"-?\bInfinity\b|\bNaN\b", "null", json.dumps(x, default=float))


# ---------------------------------------------------------------- quaternion helpers (pure Python: per sample)
def qmul(a, b):
    w0, x0, y0, z0 = a; w1, x1, y1, z1 = b
    return (w0*w1 - x0*x1 - y0*y1 - z0*z1, w0*x1 + x0*w1 + y0*z1 - z0*y1,
            w0*y1 - x0*z1 + y0*w1 + z0*x1, w0*z1 + x0*y1 - y0*x1 + z0*w1)


def qexp(wx, wy, wz, dt):
    ang = math.sqrt(wx*wx + wy*wy + wz*wz) * dt
    if ang < 1e-12:
        return (1.0, 0.5*wx*dt, 0.5*wy*dt, 0.5*wz*dt)
    s = math.sin(ang / 2) / (ang / dt)
    return (math.cos(ang / 2), wx*s, wy*s, wz*s)


def qnorm(q):
    n = math.sqrt(sum(c*c for c in q))
    return tuple(c / n for c in q)


def up_in_body(q):
    """World z (up) expressed in the body frame: third row of R(q)."""
    w, x, y, z = q
    return (2*(x*z - w*y), 2*(y*z + w*x), w*w - x*x - y*y + z*z)


def yaw(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))


def euler_deg(q):
    w, x, y, z = q
    return [math.degrees(math.atan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))),
            math.degrees(math.asin(max(-1.0, min(1.0, 2*(w*y - z*x))))), math.degrees(yaw(q))]


def tilt_between(q, q0):
    a, b = up_in_body(q), up_in_body(q0)
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(i*j for i, j in zip(a, b))))))


# ---------------------------------------------------------------- monitor: rates, plots, stillness
class SensorMon:
    def __init__(self):
        self.plot = deque(maxlen=PLOT_HZ * PLOT_S)
        self.t_plot = None
        self.dts = deque(maxlen=2000)
        self.recent = deque(maxlen=800)          # ~2 s of samples at 395 Hz, for stillness
        self.t_last = None
        self.n = self.gaps = self.lost = 0
        self.nominal = None

    def update(self, t_ns, v):
        t = t_ns * 1e-9
        if self.t_last is not None:
            dt = t - self.t_last
            if dt <= 0:
                return
            self.dts.append(dt)
            if self.nominal is None and len(self.dts) == 400:
                self.nominal = float(np.median(self.dts))
            elif self.nominal and dt > 1.5 * self.nominal:
                self.gaps += 1
                self.lost += int(round(dt / self.nominal)) - 1
        self.t_last = t
        self.n += 1
        x = v[:3]
        self.recent.append(x)
        if self.t_plot is None or t >= self.t_plot:
            self.plot.append((t, *x))
            self.t_plot = t + 1.0 / PLOT_HZ

    def stats(self):
        if len(self.dts) < 10:
            return dict(rate_hz=0.0, n=self.n)
        d = np.array(self.dts)
        return dict(rate_hz=float(1 / np.median(d)), dt_std_us=float(d.std() * 1e6), dt_max_ms=float(d.max() * 1e3),
                    gaps=self.gaps, lost=self.lost, n=self.n)

    def plot_data(self, scale=1.0):
        if not self.plot:
            return None
        p = np.array(self.plot)
        return np.round(np.column_stack([p[:, 0] - p[-1, 0], p[:, 1:] * scale]), 5).T.tolist()


class Demo:
    """Mahony attitude for the 3D phone, plus a drift race: raw gyro vs bias-corrected gyro vs Mahony."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.q, self.e_int = (1.0, 0.0, 0.0, 0.0), [0.0, 0.0, 0.0]
        self.acc, self.t_last, self.t0 = None, None, None
        self.race = None
        self.g_recent = deque(maxlen=2000)

    def update_acc(self, v):
        self.acc = v[:3]
        if self.t0 is None:                       # start level with the measured tilt, yaw 0
            ax, ay, az = self.acc
            r, p = math.atan2(ay, az), math.atan2(-ax, math.hypot(ay, az))
            cr, sr, cp, sp = math.cos(r / 2), math.sin(r / 2), math.cos(p / 2), math.sin(p / 2)
            self.q = (cr*cp, sr*cp, cr*sp, -sr*sp)
            self.t0 = time.monotonic()

    def update_gyro(self, t_ns, v):
        t = t_ns * 1e-9
        wx, wy, wz = v[:3]
        self.g_recent.append((wx, wy, wz))
        if self.t_last is None or self.acc is None:
            self.t_last = t
            return
        dt = t - self.t_last
        self.t_last = t
        if dt <= 0 or dt > 0.1:
            return
        kp = 5.0 if time.monotonic() - self.t0 < 2.0 else 1.0
        ki = 0.02
        ax, ay, az = self.acc
        n = math.sqrt(ax*ax + ay*ay + az*az)
        ex = ey = ez = 0.0
        if n > 0 and abs(n / 9.80665 - 1) < 0.15:
            ax, ay, az = ax / n, ay / n, az / n
            vx, vy, vz = up_in_body(self.q)
            ex, ey, ez = ay*vz - az*vy, az*vx - ax*vz, ax*vy - ay*vx
            self.e_int = [self.e_int[0] + ex*dt, self.e_int[1] + ey*dt, self.e_int[2] + ez*dt]
        cx = wx + kp*ex + ki*self.e_int[0]
        cy = wy + kp*ey + ki*self.e_int[1]
        cz = wz + kp*ez + ki*self.e_int[2]
        self.q = qnorm(qmul(self.q, qexp(cx, cy, cz, dt)))
        r = self.race
        if r:
            r["q_raw"] = qnorm(qmul(r["q_raw"], qexp(wx, wy, wz, dt)))
            b = r["bias"]
            r["q_corr"] = qnorm(qmul(r["q_corr"], qexp(wx - b[0], wy - b[1], wz - b[2], dt)))
            el = time.monotonic() - r["t_wall"]
            if el >= r["t_next"]:
                r["t_next"] += 0.1
                d = lambda q, q0: (math.degrees(yaw(q) - yaw(q0) + math.pi) % 360.0) - 180.0
                r["t"].append(round(el, 2))
                for k, q in (("raw", r["q_raw"]), ("corr", r["q_corr"]), ("mahony", self.q)):
                    r["tilt"][k].append(round(tilt_between(q, r["q0"]), 4))
                    r["head"][k].append(round(d(q, r["q0"]), 4))

    def start_race(self):
        g = np.array(self.g_recent) if self.g_recent else np.zeros((1, 3))
        self.race = dict(q0=self.q, q_raw=self.q, q_corr=self.q, bias=g.mean(axis=0).tolist(), t_wall=time.monotonic(),
                         t_next=0.0, t=[], tilt=dict(raw=[], corr=[], mahony=[]), head=dict(raw=[], corr=[], mahony=[]))

    def race_data(self):
        r = self.race
        if not r:
            return None
        n = len(r["t"])
        idx = range(n) if n <= 600 else np.linspace(0, n - 1, 600).astype(int)
        pick = lambda a: [a[i] for i in idx]
        return dict(t=pick(r["t"]), bias_dps=[math.degrees(b) for b in r["bias"]],
                    tilt={k: pick(v) for k, v in r["tilt"].items()}, head={k: pick(v) for k, v in r["head"].items()})


# ---------------------------------------------------------------- the console
class Console:
    def __init__(self, runs_dir, port, serial):
        self.runs = os.path.abspath(runs_dir)
        os.makedirs(self.runs, exist_ok=True)
        self.port, self.serial = port, serial
        self.mon = {k: SensorMon() for k in ("acc", "gyro", "mag")}
        self.demo = Demo()
        self.conn = dict(adb="checking", forward=False, stream="connecting", model=None, plugged=None, stay_awake=None,
                         foreground=None, temp_c=None)
        self.rec = None
        self.jobs = {}
        self.clients = {}
        self.clock_offset = None

    # ----- phone side
    async def phone_loop(self):
        types = (stream.ACC, stream.GYRO, stream.MAG)
        while True:
            try:
                self.conn["stream"] = "connecting"
                async for name, t_ns, v, host_ns in stream.phone_events("localhost", self.port, types):
                    self.conn["stream"] = "live"
                    off = host_ns - t_ns
                    if self.clock_offset is None or off < self.clock_offset:
                        self.clock_offset = off
                    self.mon[name].update(t_ns, v)
                    if name == "acc":
                        self.demo.update_acc(v)
                    elif name == "gyro":
                        self.demo.update_gyro(t_ns, v)
                    r = self.rec
                    if r and r["active"]:
                        r["writer"].write(name, t_ns, v)
                        r["stats"].update(name, t_ns, v, host_ns)
            except (OSError, asyncio.TimeoutError, Exception) as e:   # keep the console alive whatever the phone does
                self.conn["stream"] = f"down ({e.__class__.__name__})"
            self.demo.reset()
            for m in self.mon.values():
                m.t_last = None
            await asyncio.sleep(2.0)

    async def adb_loop(self):
        n = 0
        while True:
            if not shutil.which("adb"):
                self.conn["adb"] = "adb not installed"
            else:
                out = await asyncio.to_thread(adb, ["devices"])
                lines = [l.split() for l in (out or "").splitlines()[1:] if l.strip()]
                devs = [l for l in lines if not self.serial or l[0] == self.serial]
                self.conn["adb"] = devs[0][1] if devs else "no phone"
                if devs and devs[0][1] == "device":
                    self.serial = self.serial or devs[0][0]
                    fwd = await asyncio.to_thread(adb, ["forward", "--list"], self.serial) or ""
                    if f"tcp:{self.port} tcp:{self.port}" not in fwd:
                        await asyncio.to_thread(adb, ["forward", f"tcp:{self.port}", f"tcp:{self.port}"], self.serial)
                        fwd = await asyncio.to_thread(adb, ["forward", "--list"], self.serial) or ""
                    self.conn["forward"] = f"tcp:{self.port}" in fwd
                    if self.conn["model"] is None:
                        self.conn.update(await asyncio.to_thread(phone_meta, self.serial))
                    if n % 2 == 0:
                        await self.phone_checks()
            n += 1
            await asyncio.sleep(3.0)

    async def phone_checks(self):
        bat = await asyncio.to_thread(adb, ["shell", "dumpsys", "battery"], self.serial) or ""
        kv = dict(l.strip().split(": ", 1) for l in bat.splitlines() if ": " in l)
        if "temperature" in kv:
            self.conn["temp_c"] = int(kv["temperature"]) / 10.0
            self.conn["plugged"] = any(kv.get(k) == "true" for k in ("AC powered", "USB powered", "Wireless powered"))
            r = self.rec
            if r and r["active"] and self.clock_offset is not None:
                sns = time.time_ns() - self.clock_offset
                r["stats"].add_battery(sns, self.conn["temp_c"])
                r["battery"].write(f"{time.time_ns()},{sns},{self.conn['temp_c']},{kv.get('level', '')},{int(self.conn['plugged'])}\n")
        stay = (await asyncio.to_thread(adb, ["shell", "settings", "get", "global", "stay_on_while_plugged_in"], self.serial) or "").strip()
        self.conn["stay_awake"] = stay not in ("", "0", "null")
        act = await asyncio.to_thread(adb, ["shell", "dumpsys activity activities | grep -E 'topResumedActivity|mResumedActivity'"], self.serial) or ""
        self.conn["foreground"] = PKG in act

    def stillness(self):
        g, a = self.mon["gyro"].recent, self.mon["acc"].recent
        if len(g) < 200 or len(a) < 200:
            return dict(still=False, flat=False, gyro_dps=None, acc_std=None, tilt_deg=None)
        g, a = np.array(g), np.array(a)
        g_dps = float(np.degrees(np.linalg.norm(g.std(axis=0))))
        a_std = float(np.linalg.norm(a.std(axis=0)))
        m = a.mean(axis=0)
        tilt = float(np.degrees(np.arccos(np.clip(abs(m[2]) / np.linalg.norm(m), -1, 1))))
        # thresholds sit above this phone's resting noise (accel ~0.10 m/s^2, gyro ~0.2 deg/s, 3-axis norm of the std)
        return dict(still=g_dps < 1.0 and a_std < 0.15, flat=tilt < 5.0, gyro_dps=g_dps, acc_std=a_std, tilt_deg=tilt)

    # ----- recording
    def start(self, kind, minutes, name, note, calib):
        if self.rec and self.rec["active"]:
            raise ValueError("a test is already running")
        if self.conn["stream"] != "live":
            raise ValueError("the phone is not streaming")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip()) or time.strftime(f"{kind}_%Y-%m-%d_%H%M%S")
        out = os.path.join(self.runs, name)
        if os.path.exists(out):
            raise ValueError(f"{name} already exists")
        os.makedirs(out)
        meta = dict(phone_meta(self.serial), kind=kind, planned_minutes=minutes, note=note,
                    started=time.strftime("%Y-%m-%d %H:%M:%S"), checks=dict(conn=dict(self.conn), still=self.stillness()),
                    url=stream.ws_url("localhost", self.port, (stream.ACC, stream.GYRO, stream.MAG)))
        with open(os.path.join(out, "meta.yaml"), "w") as f:
            yaml.safe_dump(json.loads(jdump(meta)), f, sort_keys=False)
        bf = open(os.path.join(out, "battery.csv"), "w", buffering=1)
        bf.write("host_ns,sensor_ns,temp_c,level,plugged\n")
        self.rec = dict(active=True, name=name, dir=out, kind=kind, minutes=minutes, calib=calib, t0=time.monotonic(),
                        writer=RawWriter(out), battery=bf, stats=LiveStats(("acc", "gyro"), align_s=10.0), meta=meta)
        return name

    def stop(self, reason="stopped"):
        r = self.rec
        if not r or not r["active"]:
            return None
        r["active"] = False
        r["writer"].close(); r["battery"].close()
        snap = r["stats"].dist.snapshot()
        r["meta"].update(ended=time.strftime("%Y-%m-%d %H:%M:%S"), stop_reason=reason,
                         recorded_s=time.monotonic() - r["t0"], live_disturbance=snap)
        with open(os.path.join(r["dir"], "meta.yaml"), "w") as f:
            yaml.safe_dump(json.loads(jdump(r["meta"])), f, sort_keys=False)
        self.analyse(r["name"], r["kind"], calib=r["calib"])
        return r["name"]

    def analyse(self, name, kind, extras=(), out_name=None, calib=False):
        cmd = [sys.executable, os.path.join(HERE, "analysis.py"), os.path.join(self.runs, name), "--kind", kind]
        if extras:
            cmd += ["--extra"] + [os.path.join(self.runs, e) for e in extras]
        if out_name:
            cmd += ["--out", os.path.join(self.runs, out_name)]
        if calib and os.path.exists(os.path.join(ROOT, "calib.yaml")):
            cmd += ["--calib", os.path.join(ROOT, "calib.yaml")]
        job = out_name or name
        log = open(os.path.join(self.runs, job, "analysis.log"), "w")
        self.jobs[job] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT,
                                          env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        return job

    async def timer_loop(self):
        while True:
            r = self.rec
            if r and r["active"] and r["minutes"] and time.monotonic() - r["t0"] >= r["minutes"] * 60:
                self.stop("duration reached")
            for k, p in list(self.jobs.items()):
                if p.poll() is not None:
                    del self.jobs[k]
            await asyncio.sleep(0.5)

    # ----- snapshots
    def progress(self, name):
        p = os.path.join(self.runs, name, "progress.json")
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def status(self):
        r = self.rec
        rec = None
        if r:
            el = time.monotonic() - r["t0"] if r["active"] else r["meta"].get("recorded_s", 0)
            rec = dict(active=r["active"], name=r["name"], kind=r["kind"], minutes=r["minutes"], elapsed_s=el,
                       remaining_s=max(0.0, r["minutes"] * 60 - el) if r["minutes"] else None,
                       stats=r["stats"].snapshot() if r["active"] else None, progress=self.progress(r["name"]))
        return dict(type="status", conn=self.conn, still=self.stillness(), rec=rec,
                    rates={k: m.stats() for k, m in self.mon.items()}, jobs=list(self.jobs),
                    calib_available=os.path.exists(os.path.join(ROOT, "calib.yaml")))

    def demo_msg(self):
        return dict(type="demo", q=self.demo.q, euler=euler_deg(self.demo.q),
                    plots=dict(acc=self.mon["acc"].plot_data(), gyro=self.mon["gyro"].plot_data(np.degrees(1)),
                               mag=self.mon["mag"].plot_data()),
                    rates={k: m.stats() for k, m in self.mon.items()}, race=self.demo.race_data())

    def sessions(self):
        out = []
        names = os.listdir(self.runs)
        for name in sorted(names, key=lambda n: os.path.getctime(os.path.join(self.runs, n)), reverse=True):   # newest first
            d = os.path.join(self.runs, name)
            if not os.path.isdir(d):
                continue
            meta, res = {}, {}
            try:
                with open(os.path.join(d, "meta.yaml")) as f:
                    meta = yaml.safe_load(f) or {}
            except OSError:
                pass
            try:
                with open(os.path.join(d, "results.json"), encoding="utf-8") as f:
                    res = json.load(f)
            except (OSError, ValueError):
                pass
            checks = res.get("checks", [])
            kind = "combined" if meta.get("kind") == "combined" else (res.get("kind") or meta.get("kind", "?"))
            out.append(dict(name=name, kind=kind, started=str(meta.get("started", "")),
                            note=meta.get("note", ""), duration_min=round(res.get("duration_s", meta.get("recorded_s", 0) or 0) / 60, 1),
                            validity=res.get("disturbances", {}).get("validity"),
                            worst=("fail" if any(c["level"] == "fail" for c in checks) else
                                   "warn" if any(c["level"] == "warn" for c in checks) else "pass" if checks else None),
                            report=os.path.exists(os.path.join(d, "report.html")), progress=self.progress(name),
                            running=name in self.jobs, recording=bool(self.rec and self.rec["active"] and self.rec["name"] == name),
                            extras=res.get("extras", [])))
        return out

    # ----- web
    async def broadcast_loop(self):
        k = 0
        while True:
            await asyncio.sleep(0.1)
            k += 1
            demo = [ws for ws, view in self.clients.items() if view == "demo"]
            if demo:
                msg = jdump(self.demo_msg())
                for ws in demo:
                    await self._send(ws, msg)
            if k % 5 == 0 and self.clients:
                msg = jdump(self.status())
                for ws in list(self.clients):
                    await self._send(ws, msg)

    async def _send(self, ws, msg):
        try:
            await ws.send_str(msg)
        except (ConnectionResetError, RuntimeError):
            self.clients.pop(ws, None)

    def web_app(self):
        from aiohttp import web
        routes = web.RouteTableDef()

        @routes.get("/")
        async def index(_):
            return web.FileResponse(os.path.join(HERE, "web", "app.html"))

        @routes.get("/ws")
        async def ws_handler(req):
            ws = web.WebSocketResponse(heartbeat=10)
            await ws.prepare(req)
            self.clients[ws] = "home"
            try:
                async for m in ws:
                    try:
                        self.clients[ws] = json.loads(m.data).get("view", "home")
                    except (ValueError, AttributeError):
                        pass
            finally:
                self.clients.pop(ws, None)
            return ws

        def ok(**kw):
            return web.json_response(dict(ok=True, **kw))

        def err(e):
            return web.json_response(dict(ok=False, error=str(e)), status=400)

        @routes.post("/api/record/start")
        async def rec_start(req):
            b = await req.json()
            try:
                return ok(name=self.start(b.get("kind", "static"), float(b.get("minutes") or 0), b.get("name", ""),
                                          b.get("note", ""), bool(b.get("calib"))))
            except (ValueError, OSError) as e:
                return err(e)

        @routes.post("/api/record/stop")
        async def rec_stop(_):
            return ok(name=self.stop("stopped by user"))

        @routes.post("/api/record/reset_drift")
        async def rec_reset(_):
            if self.rec and self.rec["active"]:
                self.rec["stats"].reset()
            return ok()

        @routes.post("/api/demo/{what}")
        async def demo_ctl(req):
            w = req.match_info["what"]
            if w == "reset":
                self.demo.reset()
            elif w == "race_start":
                self.demo.start_race()
            elif w == "race_stop":
                self.demo.race = None
            return ok()

        @routes.get("/api/sessions")
        async def sessions(_):
            return web.json_response(self.sessions())

        @routes.post("/api/analyse")
        async def reanalyse(req):
            b = await req.json()
            if not os.path.isdir(os.path.join(self.runs, b.get("name", ""))):
                return err("no such session")
            return ok(job=self.analyse(b["name"], b.get("kind", "static"), calib=bool(b.get("calib"))))

        @routes.post("/api/combined")
        async def combined(req):
            b = await req.json()
            main, extras = b.get("main"), b.get("extras", [])
            if not main or not extras:
                return err("choose one main session and at least one more")
            name = time.strftime("combined_%Y-%m-%d_%H%M%S")
            os.makedirs(os.path.join(self.runs, name))
            with open(os.path.join(self.runs, name, "meta.yaml"), "w") as f:
                yaml.safe_dump(dict(kind="combined", main=main, extras=extras, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                                    note=f"{main} + {', '.join(extras)}"), f, sort_keys=False)
            kind = "static" if b.get("kind", "static") == "static" else "turnon"
            return ok(job=self.analyse(main, kind, extras, out_name=name, calib=bool(b.get("calib"))))

        app = web.Application()
        app.add_routes(routes)
        app.router.add_static("/runs/", self.runs, show_index=False)
        app.router.add_static("/web/", os.path.join(HERE, "web"))
        return app

    async def main(self, http):
        from aiohttp import web
        runner = web.AppRunner(self.web_app())
        await runner.setup()
        await web.TCPSite(runner, "localhost", http).start()
        print(f"IMU test console: http://localhost:{http}   (runs in {self.runs})", flush=True)
        await asyncio.gather(self.phone_loop(), self.adb_loop(), self.timer_loop(), self.broadcast_loop())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--http", type=int, default=8050)
    ap.add_argument("--port", type=int, default=8080, help="SensorServer port on the phone (forwarded over adb)")
    ap.add_argument("--serial", default=None)
    a = ap.parse_args()
    c = Console(a.runs, a.port, a.serial)
    try:
        asyncio.run(c.main(a.http))
    except KeyboardInterrupt:
        pass
    finally:
        if c.rec and c.rec["active"]:
            c.stop("console closed")
            print(f"stopped {c.rec['name']}; its analysis continues in the background")

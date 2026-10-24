"""Record every phone IMU sample with its hardware timestamp, with an optional live drift dashboard.

  adb forward tcp:8080 tcp:8080                                   # once per USB connection
  python phone_data/phone_logger.py runs/static_01 --minutes 120 --dashboard --note "floor, untouched"
  python phone_data/phone_logger.py --dashboard                   # watch only, nothing saved
  python phone_data/phone_logger.py --replay runs/static_01 --speed 20 --dashboard

Writes into the run folder: raw_acc.csv, raw_gyro.csv (every event, t_ns + 6 values), battery.csv (battery
temperature polled over adb every 10 s), meta.yaml, then on stop merges them into imu.csv + config.yaml
(phone_data/merge.py) so every nav/ and scripts/ tool runs on it. Dashboard: http://localhost:8050

The file writer never waits for the dashboard: the dashboard reads a snapshot twice a second, and a slow
browser only skips frames.
"""
import argparse, asyncio, json, os, re, shutil, subprocess, sys, time
import yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phone_data import stream, merge
from phone_data.live_stats import LiveStats

HERE = os.path.dirname(os.path.abspath(__file__))


class RawWriter:
    def __init__(self, out):
        self.out, self.files = out, {}

    def write(self, name, t_ns, values):
        f = self.files.get(name)
        if f is None:
            f = self.files[name] = open(os.path.join(self.out, f"raw_{name}.csv"), "w", buffering=1 << 16)
            f.write(stream.RAW_COLS + "\n")
        f.write(f"{t_ns}," + ",".join(f"{x:.9g}" for x in values[:6]) + "\n")

    def flush(self):
        for f in self.files.values():
            f.flush()

    def close(self):
        for f in self.files.values():
            f.close()


def adb(args, serial=None, timeout=5):
    exe = shutil.which("adb")
    if exe is None:
        return None
    cmd = [exe] + (["-s", serial] if serial else []) + args
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (subprocess.TimeoutExpired, OSError):
        return None


def phone_meta(serial):
    g = lambda p: (adb(["shell", "getprop", p], serial) or "").strip()
    return dict(model=g("ro.product.model"), android=g("ro.build.version.release"), serial=serial or g("ro.serialno"))


async def poll_battery(stats, out, serial, every_s=10.0):
    f = None
    if out:
        f = open(os.path.join(out, "battery.csv"), "w", buffering=1)
        f.write("host_ns,sensor_ns,temp_c,level,plugged\n")
    try:
        while True:
            txt = await asyncio.to_thread(adb, ["shell", "dumpsys", "battery"], serial)
            host_ns = time.time_ns()
            if txt and stats.clock_offset is not None:
                kv = dict(l.strip().split(": ", 1) for l in txt.splitlines() if ": " in l)
                temp = int(kv.get("temperature", "0")) / 10.0
                plugged = int(kv.get("AC powered") == "true" or kv.get("USB powered") == "true")
                sns = stats.sensor_ns(host_ns)
                stats.add_battery(sns, temp)
                if f:
                    f.write(f"{host_ns},{sns},{temp},{kv.get('level', '')},{plugged}\n")
            await asyncio.sleep(every_s)
    finally:
        if f:
            f.close()


async def serve_dashboard(stats, port):
    from aiohttp import web
    clients = set()

    async def index(_):
        return web.FileResponse(os.path.join(HERE, "dashboard.html"))

    async def ws_handler(req):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(req)
        clients.add(ws)
        try:
            async for _ in ws:
                pass
        finally:
            clients.discard(ws)
        return ws

    async def reset(_):
        stats.reset()
        return web.json_response({"ok": True})

    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ws", ws_handler), web.post("/reset", reset)])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "localhost", port).start()
    print(f"dashboard: http://localhost:{port}")
    while True:
        await asyncio.sleep(0.5)
        if clients:
            msg = re.sub(r"-?\bInfinity\b|\bNaN\b", "null", json.dumps(stats.snapshot(), default=float))
            for ws in list(clients):
                try:
                    await ws.send_str(msg)
                except (ConnectionResetError, RuntimeError):
                    clients.discard(ws)


async def acquire(a, stats, writer):
    names = ("acc", "gyro", "mag") if a.mag else ("acc", "gyro")
    types = [stream.ACC, stream.GYRO] + ([stream.MAG] if a.mag else [])
    t_end = time.monotonic() + a.minutes * 60 if a.minutes else None
    t_status = time.monotonic()
    while True:
        src = stream.replay_events(a.replay, names, a.speed) if a.replay else stream.phone_events(a.host, a.port, types)
        stats.status = "replay" if a.replay else "connecting"
        try:
            async for name, t_ns, values, host_ns in src:
                stats.status = "replay" if a.replay else "live"
                if writer:
                    writer.write(name, t_ns, values)
                stats.update(name, t_ns, values, host_ns)
                now = time.monotonic()
                if now - t_status > 1.0:
                    t_status = now
                    if writer:
                        writer.flush()
                    c = stats.ch
                    print(f"\r{c['gyro'].t_last - c['gyro'].t0 if c['gyro'].n else 0:8.1f} s | "
                          + " | ".join(f"{n} {s.n} @ {1 / s.nominal if s.nominal else 0:5.1f} Hz gaps {s.gaps}"
                                       for n, s in c.items())
                          + (f" | {stats.battery[-1][1]:.1f} C" if stats.battery else ""), end="", flush=True)
                if t_end and now > t_end:
                    return
            if a.replay:
                stats.status = "replay finished"
                await asyncio.Event().wait()     # keep the dashboard up
        except (OSError, asyncio.TimeoutError) as e:
            if a.replay:
                raise
            stats.status = f"disconnected: {e.__class__.__name__}, retrying"
            print(f"\nphone stream not reachable ({e}); is SensorServer running and `adb forward tcp:{a.port} tcp:{a.port}` set? retrying in 2 s")
        await asyncio.sleep(2.0)


async def main(a, stats, writer):
    tasks = [asyncio.create_task(acquire(a, stats, writer))]
    if a.dashboard:
        tasks.append(asyncio.create_task(serve_dashboard(stats, a.http)))
    if not a.replay and shutil.which("adb"):
        tasks.append(asyncio.create_task(poll_battery(stats, a.out, a.serial)))
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()
    for t in done:
        t.result()


def cli(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?", help="run folder to record into; omit to only watch")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--dashboard", action="store_true")
    ap.add_argument("--http", type=int, default=8050)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--serial", default=None, help="adb device serial if more than one is connected")
    ap.add_argument("--mag", action="store_true", help="also record the uncalibrated magnetometer")
    ap.add_argument("--align", type=float, default=10.0, help="s of initial still period for the drift panels")
    ap.add_argument("--replay", default=None, help="replay a recorded run folder instead of the phone")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)

    stats = LiveStats(("acc", "gyro", "mag") if a.mag else ("acc", "gyro"), align_s=a.align)
    if a.replay and os.path.exists(os.path.join(a.replay, "battery.csv")):
        import numpy as np
        for row in np.loadtxt(os.path.join(a.replay, "battery.csv"), delimiter=",", skiprows=1, ndmin=2):
            stats.add_battery(row[1], row[2])
    writer = None
    if a.out and not a.replay:
        os.makedirs(a.out, exist_ok=True)
        meta = dict(phone_meta(a.serial), note=a.note, started=time.strftime("%Y-%m-%d %H:%M:%S"),
                    url=stream.ws_url(a.host, a.port))
        with open(os.path.join(a.out, "meta.yaml"), "w") as f:
            yaml.safe_dump(meta, f, sort_keys=False)
        writer = RawWriter(a.out)
    try:
        asyncio.run(main(a, stats, writer))
    except KeyboardInterrupt:
        pass
    finally:
        if writer:
            writer.close()
            print(f"\nmerging {a.out} ...")
            merge.print_report(merge.build(a.out))
            print(f"wrote {os.path.join(a.out, 'imu.csv')}; next: python scripts/check_acq.py {a.out} --rate 395")


if __name__ == "__main__":
    cli()

"""Sensor events from the Android SensorServer app (package github.umer0586.sensorserver).

The app pushes one JSON message per sensor event on a single WebSocket:
  {"type": "android.sensor.gyroscope_uncalibrated", "values": [x, y, z, bx, by, bz], "timestamp": ns, "accuracy": 3}

`timestamp` is SensorEvent.timestamp: the hardware sample time in ns on the phone's boot clock. Always use it,
never the arrival time on the laptop. USB/Wi-Fi and Android scheduling add milliseconds of jitter to arrival,
not to this timestamp. For *_uncalibrated sensors, values[3:6] is Android's own bias estimate (not applied).

Phone setup: start the server in SensorServer, then on the laptop `adb forward tcp:8080 tcp:8080`.
Set the app's sampling rate to its fastest value; this phone (BMI26x) delivers about 395 Hz.
"""
import asyncio, json, time, urllib.parse
import numpy as np

ACC = "android.sensor.accelerometer_uncalibrated"
GYRO = "android.sensor.gyroscope_uncalibrated"
MAG = "android.sensor.magnetic_field_uncalibrated"
SHORT = {ACC: "acc", GYRO: "gyro", MAG: "mag"}
RAW_COLS = "t_ns,x,y,z,bx,by,bz"


def ws_url(host="localhost", port=8080, types=(ACC, GYRO)):
    return f"ws://{host}:{port}/sensors/connect?types=" + urllib.parse.quote(json.dumps(list(types)))


async def phone_events(host="localhost", port=8080, types=(ACC, GYRO)):
    """Yield (name, t_ns, values, host_ns) for every event, in arrival order."""
    import aiohttp
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(ws_url(host, port, types), heartbeat=10, max_msg_size=0) as ws:
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    d = json.loads(msg.data)
                    yield SHORT.get(d["type"], d["type"]), int(d["timestamp"]), d["values"], time.time_ns()
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break


def load_raw(path):
    """raw_<sensor>.csv -> (t_ns int64 (N,), values (N,6)). Boot-clock ns stay < 2^53, so float64 is exact."""
    from nav.io import csv_path
    a = np.loadtxt(csv_path(path), delimiter=",", skiprows=1, ndmin=2, dtype=np.float64)
    if a.size == 0:
        return np.zeros(0, np.int64), np.zeros((0, 6))
    return a[:, 0].astype(np.int64), a[:, 1:]


async def replay_events(run_dir, names=("acc", "gyro"), speed=1.0):
    """Feed a recorded run's raw_*.csv back in timestamp order, paced at `speed` x real time."""
    import os
    rows = []
    for n in names:
        p = os.path.join(run_dir, f"raw_{n}.csv")
        if os.path.exists(p) or os.path.exists(p + ".gz"):
            t, v = load_raw(p)
            rows += [(int(ti), n, vi.tolist()) for ti, vi in zip(t, v)]
    rows.sort(key=lambda r: r[0])
    if not rows:
        return
    t_first, wall0 = rows[0][0], time.monotonic()
    for t, n, v in rows:
        ahead = (t - t_first) * 1e-9 / speed - (time.monotonic() - wall0)
        if ahead > 0.02:
            await asyncio.sleep(ahead)
        yield n, t, v, time.time_ns()

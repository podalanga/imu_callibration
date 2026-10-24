"""Record a phone IMU session into the common dataset format (nav/io.py), using the same
multi-threaded websocket pattern as websocket_sensor.py, but writing to disk instead of just
printing — so recordings can feed scripts/run_calib.py, scripts/run_allan.py, scripts/run_filter.py.

  adb forward tcp:8080 tcp:8080                      # USB, once per (re)connect
  python phone_data/record_phone.py runs/phone_calib_s1
  python phone_data/record_phone.py runs/phone_static_10min --minutes 10

Ctrl+C to stop (or pass --minutes to stop automatically). Writes imu.csv
(t_us,seq,gx,gy,gz,ax,ay,az,temp_c) and config.yaml in `out`.

accel/gyro are the *_uncalibrated Android sensors, already SI (m/s^2, rad/s) — calibration (bias,
scale, non-orthogonality) is applied afterwards by nav.calib, not here. temp_c isn't exposed by
phone sensors, so it's written as a constant placeholder. Magnetometer is captured too but isn't
part of the common IMU schema, so it goes to a separate mag.csv.
"""
import argparse, csv, os, time
import websocket
import ujson as json
import yaml
import threading

latest = {"accel": None, "gyro": None, "mag": None}


def _stream(host, sensor_type, key):
    def on_message(ws, message):
        latest[key] = message

    def on_error(ws, error):
        print(f"Error [{key.upper()}]: {error}")

    def on_close(ws, code, msg):
        print(f"--- {key.upper()} stream disconnected ---")

    url = f"ws://{host}/sensor/connect?type={sensor_type}"
    ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever(ping_interval=10, skip_utf8_validation=True)


def start_streams(host):
    sensors = {"accel": "android.sensor.accelerometer_uncalibrated",
               "gyro": "android.sensor.gyroscope_uncalibrated",
               "mag": "android.sensor.magnetic_field_uncalibrated"}
    for key, s_type in sensors.items():
        threading.Thread(target=_stream, args=(host, s_type, key), daemon=True).start()


ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--host", default="localhost:8080",
                help="phone's websocket host:port (via adb forward, or the phone's Wi-Fi IP:8080)")
ap.add_argument("--rate", type=float, default=200.0, help="sample loop rate, Hz")
ap.add_argument("--minutes", type=float, default=None, help="stop automatically after this many minutes")
ap.add_argument("--note", default="")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
print(f"Connecting to {a.host} ...")
start_streams(a.host)
time.sleep(1.0)
print("Streaming! Press Ctrl+C to stop." if a.minutes is None else f"Streaming for {a.minutes:.1f} min...")

fi = open(os.path.join(a.out, "imu.csv"), "w", newline="")
fm = open(os.path.join(a.out, "mag.csv"), "w", newline="")
wi, wm = csv.writer(fi), csv.writer(fm)
wi.writerow("t_us,seq,gx,gy,gz,ax,ay,az,temp_c".split(","))
wm.writerow("t_us,seq,mx,my,mz".split(","))

seq = 0
t0 = time.perf_counter()
period = 1.0 / a.rate
next_t = t0
try:
    while a.minutes is None or (time.perf_counter() - t0) < a.minutes * 60:
        raw_a, raw_g, raw_m = latest["accel"], latest["gyro"], latest["mag"]
        if raw_a and raw_g:
            av = json.loads(raw_a)["values"][0:3]
            gv = json.loads(raw_g)["values"][0:3]
            t_us = int((time.perf_counter() - t0) * 1e6)
            wi.writerow([t_us, seq, *[f"{v:.9g}" for v in gv], *[f"{v:.9g}" for v in av], "25.00"])
            if raw_m:
                mv = json.loads(raw_m)["values"][0:3]
                wm.writerow([t_us, seq, *[f"{v:.9g}" for v in mv]])
            seq += 1
            if seq % int(a.rate * 5) == 0:
                print(f"\r{seq} samples ({seq / a.rate:.0f} s)", end="")
        next_t += period
        sleep = next_t - time.perf_counter()
        if sleep > 0:
            time.sleep(sleep)
except KeyboardInterrupt:
    pass
fi.close()
fm.close()
with open(os.path.join(a.out, "config.yaml"), "w") as f:
    yaml.safe_dump(dict(source="phone websocket (accelerometer_uncalibrated, gyroscope_uncalibrated)",
                        host=a.host, recorded=time.strftime("%Y-%m-%d %H:%M:%S"), rate_hz=a.rate,
                        n_samples=seq, note=a.note), f, sort_keys=False)
print(f"\nsaved {seq} samples to {a.out}")

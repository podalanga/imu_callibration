"""Hardware-in-the-loop replay (Chapter 15): stream a dataset into the ESP32 filter and collect its output.

  python tools/hil_replay.py /dev/ttyUSB0 data/synth_track --out results/hil_est.csv
Then compare with Python:  python tools/compare_est.py results/hil_est.csv data/synth_track
The ESP32 processes every sample with the same firmware as live mode and returns one EST frame
per sample, including the CPU cycles of that step.
"""
import argparse, os, sys, time, csv
import numpy as np
import serial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import protocol as P
from nav import io, runner

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("dataset")
ap.add_argument("--out", default="hil_est.csv")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--window", type=int, default=16, help="max samples in flight (flow control)")
a = ap.parse_args()

imu = io.load(a.dataset).imu
ser = serial.Serial(a.port, a.baud, timeout=0.01)
ser.reset_input_buffer()
ser.write(P.encode_ctrl(P.CMD_REPLAY)); time.sleep(0.2)
par = P.Parser()
sent, received_t = 0, 0
n = len(imu)
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow("t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,stationary,cycles".split(","))
    last_t = None
    t_start = time.time()
    fs = 1.0 / np.median(np.diff(imu.t))
    n_align = int(round(runner.NavConfig().t_init * fs))   # samples consumed before the first estimate
    while True:
        # flow control: never more than `window` samples beyond what the ESP32 has answered
        limit = min(n, n_align + received_t + a.window)
        while sent < limit:
            ser.write(P.encode_replay(imu.t_us[sent], imu.acc[sent], imu.gyro[sent]))
            sent += 1
            if sent < n_align and sent % 10 == 0:          # ~1000 samples/s during alignment
                time.sleep(0.01)
        for ftype, f in par.feed(ser.read(4096)):
            if ftype == P.T_EST:
                w.writerow([f[0], *[f"{v:.7g}" for v in f[1:17]], f[18], f[17]])
                received_t += 1
                last_t = time.time()
        if sent >= n and last_t and time.time() - last_t > 1.0:
            break
        if time.time() - t_start > 3600:
            break
print(f"sent {sent}, received {received_t} estimates, crc errors {par.crc_errors} -> {a.out}")
ser.write(P.encode_ctrl(P.CMD_LIVE))

"""Record a run from the ESP32 into a dataset folder (common format).

  python tools/logger.py /dev/ttyUSB0 runs/track_01 --note "trolley, 5 m stops, calib v2"
Ctrl+C to stop. Writes imu.csv (raw, SI units, sensor frame), markers.csv (button presses),
est_device.csv (on-board filter output), stats.csv, config.yaml.
Press the BOOT button while stopped over each marker; use --positions to map presses to metres.
"""
import argparse, os, sys, time, csv
import yaml
import serial                                     # pip install pyserial
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("out")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--positions", default="0,5,10,15,20,25,20,15,10,5,0",
                help="surveyed s [m] of each successive press (out-and-back default)")
ap.add_argument("--note", default="")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
pos = [float(x) for x in a.positions.split(",")]

ser = serial.Serial(a.port, a.baud, timeout=0.05)
ser.reset_input_buffer()
ser.write(P.encode_ctrl(P.CMD_LIVE))
par = P.Parser()
fi = open(os.path.join(a.out, "imu.csv"), "w", newline="")
fm = open(os.path.join(a.out, "markers.csv"), "w", newline="")
fe = open(os.path.join(a.out, "est_device.csv"), "w", newline="")
fs = open(os.path.join(a.out, "stats.csv"), "w", newline="")
wi, wm, we, ws = csv.writer(fi), csv.writer(fm), csv.writer(fe), csv.writer(fs)
wi.writerow("t_us,seq,gx,gy,gz,ax,ay,az,temp_c".split(","))
wm.writerow(["t_us", "marker_id", "s_m"])
we.writerow("t_us,px,py,pz,vx,vy,vz,qw,qx,qy,qz,bax,bay,baz,bgx,bgy,bgz,cycles,stationary".split(","))
ws.writerow("host_time,samples,fifo_overflows,queue_drops,seq_gaps,max_cycles,gated,nan_resets,cpu_mhz".split(","))
n_raw = n_mark = 0
t0 = time.time()
try:
    while True:
        for ftype, f in par.feed(ser.read(4096)):
            if ftype == P.T_RAW:
                seq, t_us, g, acc, temp = P.raw_to_si(f)
                wi.writerow([t_us, seq, *[f"{v:.9g}" for v in g], *[f"{v:.9g}" for v in acc], f"{temp:.2f}"])
                n_raw += 1
            elif ftype == P.T_MARK:
                s = pos[n_mark] if n_mark < len(pos) else float("nan")
                wm.writerow([f[0], n_mark, s]); fm.flush()
                print(f"\nmarker {n_mark} at s = {s} m"); n_mark += 1
            elif ftype == P.T_EST:
                we.writerow([f[0], *[f"{v:.7g}" for v in f[1:17]], f[17], f[18]])
            elif ftype == P.T_STAT:
                ws.writerow([f"{time.time():.3f}", *f])
                print(f"\r{n_raw} samples | overflows {f[1]} drops {f[2]} gaps {f[3]} | "
                      f"max step {f[4] / max(1, f[7]):.0f} us | crc errors {par.crc_errors}   ", end="")
except KeyboardInterrupt:
    pass
for fh in (fi, fm, fe, fs):
    fh.close()
with open(os.path.join(a.out, "config.yaml"), "w") as f:
    yaml.safe_dump(dict(source="MPU6050 via ESP32", port=a.port, recorded=time.strftime("%Y-%m-%d %H:%M:%S"),
                        duration_s=round(time.time() - t0, 1), accel_range_g=4, gyro_range_dps=500, rate_hz=200,
                        dlpf_cfg=3, marker_positions=pos, n_markers=n_mark, note=a.note), f, sort_keys=False)
print(f"\nsaved {n_raw} samples, {n_mark} markers to {a.out}")

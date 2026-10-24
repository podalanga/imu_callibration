"""Live demo display (task E2): trajectory and stop flags from the ESP32's EST frames.

  python tools/viewer.py /dev/ttyUSB0 --track 25
Shows the on-board estimate in real time, the taped line and marker positions, and a status line.
Run logger.py instead when you also want to record; both cannot open the port at once.
"""
import argparse, os, sys, collections
import serial
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import protocol as P

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("--baud", type=int, default=921600)
ap.add_argument("--track", type=float, default=25.0)
a = ap.parse_args()
ser = serial.Serial(a.port, a.baud, timeout=0.01)
ser.write(P.encode_ctrl(P.CMD_RESET))
par = P.Parser()
xs, ys = collections.deque(maxlen=20000), collections.deque(maxlen=20000)
stops_x, stops_y = [], []
was_still = False
stat = None

plt.ion()
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot([0, a.track], [0, 0], color="0.7", lw=6, solid_capstyle="butt", label="taped line")
for s in range(0, int(a.track) + 1, 5):
    ax.plot([s, s], [-0.15, 0.15], color="0.4")
line, = ax.plot([], [], lw=2, label="ESP32 estimate")
pts, = ax.plot([], [], "o", ms=6, label="detected stops")
head, = ax.plot([], [], "s", ms=9)
ax.set_xlim(-1, a.track + 1); ax.set_ylim(-1.5, 1.5); ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("x along track [m]"); ax.set_ylabel("y [m]"); ax.legend(loc="upper right")
txt = ax.text(0.01, 0.95, "", transform=ax.transAxes, va="top", family="monospace")
while plt.fignum_exists(fig.number):
    for ftype, f in par.feed(ser.read(8192)):
        if ftype == P.T_EST:
            x, y, still = f[1], f[2], f[18]
            xs.append(x); ys.append(y)
            if still and not was_still:
                stops_x.append(x); stops_y.append(y)
            was_still = bool(still)
        elif ftype == P.T_STAT:
            stat = f
    if xs:
        line.set_data(xs, ys); pts.set_data(stops_x, stops_y); head.set_data([xs[-1]], [ys[-1]])
        msg = f"x {xs[-1]:6.2f} m  y {ys[-1]:5.2f} m  {'STOPPED' if was_still else 'moving '}"
        if stat:
            msg += f" | step max {stat[4] / max(1, stat[7]):.0f} us | gaps {stat[3]} | gated {stat[5]}"
        txt.set_text(msg)
    plt.pause(0.05)

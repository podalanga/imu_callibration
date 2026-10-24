"""Record a phone IMU session into the common dataset format (nav/io.py). Kept for its command line;
the work is done by phone_logger.py.

  adb forward tcp:8080 tcp:8080                      # USB, once per (re)connect
  python phone_data/record_phone.py runs/phone_calib_s1
  python phone_data/record_phone.py runs/phone_static_10min --minutes 10

The earlier version sampled the latest value of each stream at a fixed loop rate and stamped it with
laptop time, which dropped about half the phone's samples and replaced hardware timestamps with arrival
jitter. Now every event is kept with its SensorEvent timestamp; the rate is whatever the phone delivers
(set the Sensor Server app to its fastest rate: ~395 Hz on the CPH2661), so --rate is ignored.
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phone_data import phone_logger

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--host", default="localhost:8080",
                help="phone's websocket host:port (via adb forward, or the phone's Wi-Fi IP:8080)")
ap.add_argument("--rate", type=float, default=None, help="ignored: every phone sample is recorded")
ap.add_argument("--minutes", type=float, default=0)
ap.add_argument("--note", default="")
ap.add_argument("--dashboard", action="store_true")
a = ap.parse_args()
if a.rate:
    print("--rate is ignored: every sample the phone delivers is recorded with its hardware timestamp")
host, _, port = a.host.partition(":")
phone_logger.cli([a.out, "--host", host, "--port", port or "8080", "--minutes", str(a.minutes), "--note", a.note, "--mag"]
                 + (["--dashboard"] if a.dashboard else []))

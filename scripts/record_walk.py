"""One command to record a walk: phone IMU (every sample, hardware timestamps) + phone camera video, into one folder.

  python scripts/record_walk.py runs/walk_01 --note "tag track, 5 lengths"
  python scripts/record_walk.py runs/a4_test --a4-page 1          # camera track against one printed A4 page
  python scripts/record_walk.py runs/test --minutes 0.5          # stop by itself after 30 s

Needs: the phone on USB with adb authorised, the SensorServer app open with its server started, and scrcpy
(winget install Genymobile.scrcpy). It sets up `adb forward`, starts scrcpy recording the back camera (a preview
window opens) and records the IMU in this process. Ctrl+C here (or closing the preview window) stops both cleanly.
Afterwards it merges the IMU streams and, unless --no-track, runs scripts/camera_track.py on the video.

The run folder then holds everything for post-processing:
  video.mkv                     camera, 1920x1080 at 30 fps (camera 2: main back camera, no lens switching)
  raw_acc.csv, raw_gyro.csv     every IMU event: t_ns (phone sensor clock) + values
  imu.csv, config.yaml          merged IMU on the gyro clock (phone_data/merge.py), what nav/ and scripts/ read
  battery.csv, meta.yaml        battery temperature over adb, phone model
  clock.csv                     laptop time <-> phone sensor time, once a second (maps video times to IMU times)
  scrcpy.log                    scrcpy output with laptop timestamps (first frame = video start)
  session.yaml                  settings, start / first frame / stop times, sample and frame counts
  track.csv, track.png          camera position from the tags (camera_track.py)

Sync: the video's t = 0 is its first frame; session.yaml gives that frame's laptop time and clock.csv turns it into
phone sensor time, which aligns video and IMU to within the preview latency (~0.1 s). For better, do 3 quick twists
of the phone at the start and the end of the walk: the gyro and the camera both see them.
"""
import argparse, asyncio, csv, os, shutil, signal, subprocess, sys, threading, time
from types import SimpleNamespace
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from phone_data import phone_logger, merge
from phone_data.live_stats import LiveStats


def find_scrcpy():
    exe = shutil.which("scrcpy")
    if exe:
        return exe
    base = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Microsoft", "WinGet", "Packages")
    for root, _, files in os.walk(base):
        if "scrcpy.exe" in files:
            return os.path.join(root, "scrcpy.exe")
    raise SystemExit("scrcpy not found: winget install Genymobile.scrcpy")


class Scrcpy:
    """scrcpy recording in its own process group, so Ctrl+C here does not reach it and Ctrl+Break stops it cleanly
    (the video file is finalised)."""

    def __init__(self, exe, out, a):
        self.cmd = [exe, "--video-source=camera", f"--camera-id={a.camera_id}", f"--camera-size={a.size}",
                    f"--camera-fps={a.fps}", "--no-audio", f"--record={os.path.join(out, 'video.mkv')}",
                    f"--window-title={os.path.basename(os.path.normpath(out))} - recording (Ctrl+C in the terminal to stop)"]
        if a.serial:
            self.cmd.append(f"--serial={a.serial}")
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        self.p = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  creationflags=flags, start_new_session=os.name != "nt")
        self.start_ns, self.first_frame_ns = time.time_ns(), None
        self.log = open(os.path.join(out, "scrcpy.log"), "w")
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.p.stdout:
            ns = time.time_ns()
            if self.first_frame_ns is None and "Texture" in line:       # first decoded frame
                self.first_frame_ns = ns
            self.log.write(f"{ns} {line}")
            self.log.flush()

    def stop(self):
        if self.p.poll() is None:
            self.p.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
            try:
                self.p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.p.kill()
        time.sleep(0.2)
        self.log.close()
        return self.p.returncode


async def clock_log(stats, out):
    with open(os.path.join(out, "clock.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["host_ns", "sensor_ns"])
        while True:
            if stats.clock_offset is not None:
                h = time.time_ns()
                w.writerow([h, int(stats.sensor_ns(h))])
                f.flush()
            await asyncio.sleep(1.0)


async def watch(cam):
    while cam.p.poll() is None:
        await asyncio.sleep(0.3)
    print("\npreview window closed: stopping")


async def record(la, stats, writer, cam, out):
    tasks = [asyncio.create_task(phone_logger.main(la, stats, writer)),
             asyncio.create_task(clock_log(stats, out)), asyncio.create_task(watch(cam))]
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
    for t in pending:
        t.cancel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", help="run folder, e.g. runs/walk_01")
    ap.add_argument("--note", default="")
    ap.add_argument("--minutes", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--camera-id", type=int, default=2)
    ap.add_argument("--size", default="1920x1080")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--mag", action="store_true", help="also record the magnetometer")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--a4-page", type=int, default=None, help="camera track against this printed A4 page")
    ap.add_argument("--layout", default=None, help="tag_layout.csv for the camera track (default: the floor track)")
    ap.add_argument("--no-track", action="store_true", help="skip camera_track.py after recording")
    a = ap.parse_args()
    out = a.out
    if os.path.exists(os.path.join(out, "video.mkv")):
        raise SystemExit(f"{out} already has a recording; pick a new folder")
    os.makedirs(out, exist_ok=True)

    if (phone_logger.adb(["get-state"], a.serial) or "").strip() != "device":
        raise SystemExit("no authorised phone on adb (adb devices)")
    phone_logger.adb(["forward", f"tcp:{a.port}", f"tcp:{a.port}"], a.serial)
    top = phone_logger.adb(["shell", "dumpsys", "activity", "activities"], a.serial) or ""
    if "sensorserver" not in next((l for l in top.splitlines() if "ResumedActivity" in l), "").lower():
        print("warning: SensorServer is not in the foreground; open it and start the server for full-rate IMU")

    meta = dict(phone_logger.phone_meta(a.serial), note=a.note, started=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "meta.yaml"), "w") as f:
        yaml.safe_dump(meta, f, sort_keys=False)
    names = ("acc", "gyro", "mag") if a.mag else ("acc", "gyro")
    stats = LiveStats(names)
    writer = phone_logger.RawWriter(out)
    la = SimpleNamespace(out=out, minutes=a.minutes, dashboard=False, http=8050, host="localhost", port=a.port,
                         serial=a.serial, mag=a.mag, replay=None, speed=1.0)

    cam = Scrcpy(find_scrcpy(), out, a)
    print(f"recording into {out}  (camera {a.camera_id} {a.size} @ {a.fps} fps + IMU)\n"
          "stand still 10 s, 3 quick twists, walk, 3 twists, stand still; Ctrl+C to stop")
    t_start = time.time_ns()
    try:
        asyncio.run(record(la, stats, writer, cam, out))
    except KeyboardInterrupt:
        pass
    t_stop = time.time_ns()
    rc = cam.stop()
    writer.close()
    print(f"\nscrcpy exit {rc}; merging the IMU ...")
    rep = merge.build(out)
    merge.print_report(rep)

    import cv2
    vc = cv2.VideoCapture(os.path.join(out, "video.mkv"))
    n_frames = 0
    while vc.grab():
        n_frames += 1
    first = cam.first_frame_ns
    sess = dict(note=a.note, phone=meta, camera=dict(id=a.camera_id, size=a.size, fps=a.fps, frames=n_frames,
                                                     scrcpy=" ".join(cam.cmd)),
                laptop_ns=dict(start=t_start, scrcpy_start=cam.start_ns, first_frame=first, stop=t_stop),
                video_first_frame_sensor_ns=int(stats.sensor_ns(first)) if first and stats.clock_offset is not None else None,
                imu=dict(acc=stats.ch["acc"].n, gyro=stats.ch["gyro"].n),
                sync_note="video t=0 is the first frame; first_frame laptop time -> sensor time via clock.csv; "
                          "preview latency ~0.1 s, refine with the twist gestures")
    with open(os.path.join(out, "session.yaml"), "w") as f:
        yaml.safe_dump(sess, f, sort_keys=False)
    print(f"video: {n_frames} frames; IMU: gyro {stats.ch['gyro'].n} / acc {stats.ch['acc'].n} samples")

    if not a.no_track and n_frames:
        cmd = [sys.executable, os.path.join(ROOT, "scripts", "camera_track.py"), os.path.join(out, "video.mkv"), "--out", out]
        cmd += ["--a4-page", str(a.a4_page)] if a.a4_page else (["--layout", a.layout] if a.layout else [])
        subprocess.run(cmd)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
